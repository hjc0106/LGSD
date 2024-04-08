# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn import ConvModule

from mmdet.core import multi_apply

from mmdet3d.core import bbox3d2result, merge_aug_bboxes_3d
from ..builder import DETECTORS, build_cl_head
from .mvx_two_stage import MVXTwoStageDetector
from ..utils.pdistiller_utils import index_feature, knn_feature, pool_features1d, pool_features
from ..utils.sg_utils import index_points, knn_query


@DETECTORS.register_module()
class CenterPoint(MVXTwoStageDetector):
    """Base class of Multi-modality VoxelNet."""

    def __init__(self,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(CenterPoint,
              self).__init__(pts_voxel_layer, pts_voxel_encoder,
                             pts_middle_encoder, pts_fusion_layer,
                             img_backbone, pts_backbone, img_neck, pts_neck,
                             pts_bbox_head, img_roi_head, img_rpn_head,
                             train_cfg, test_cfg, pretrained, init_cfg)

    def extract_feat(self, points, img, img_metas, return_pts_middle=False):
        """Extract features from images and points."""
        img_feats = self.extract_img_feat(img, img_metas)
        if return_pts_middle:
            pts_feats, pts_infos = self.extract_pts_feat(points, img_feats, img_metas, return_middle=True)
            return (img_feats, pts_feats, pts_infos)
        else:
            pts_feats = self.extract_pts_feat(points, img_feats, img_metas, return_middle=False)
            return (img_feats, pts_feats)
                    
    def extract_pts_feat(self, pts, img_feats, img_metas, return_middle=False):
        """Extract features of points."""
        if not self.with_pts_bbox:
            return None
        pts_middle_infos = dict()
        voxels, num_points, coors = self.voxelize(pts)
        pts_middle_infos["num_points"] = num_points
        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors)
        pts_middle_infos["voxel_features"] = voxel_features  # [num_voxel, 5]
        batch_size = coors[-1, 0] + 1
        x = self.pts_middle_encoder(voxel_features, coors, batch_size)  # [B, C*D, H, W] [B, 128*2, 128, 128]
        pts_middle_infos["middle_encoder_features"] = x
        x = self.pts_backbone(x)  # [B, 128, 128, 128] [B, 256, 64, 64]
        pts_middle_infos["backbone_features"] = x
        if self.with_pts_neck:
            x = self.pts_neck(x)
            pts_middle_infos["neck_features"] = x
        if return_middle:
            return x, pts_middle_infos
        else:
            return x  # x[0]: tensor, shape=[B, 512, 128, 128]

    def forward_pts_train(self,
                          pts_feats,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          img_metas,
                          gt_bboxes_ignore=None):
        """Forward function for point cloud branch.

        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.

        Returns:
            dict: Losses of each branch.
        """
        outs = self.pts_bbox_head(pts_feats)
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
        losses = self.pts_bbox_head.loss(*loss_inputs)
        return losses

    def simple_test_pts(self, x, img_metas, rescale=False):
        """Test function of point cloud branch."""
        outs = self.pts_bbox_head(x)
        bbox_list = self.pts_bbox_head.get_bboxes(
            outs, img_metas, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return bbox_results

    def aug_test_pts(self, feats, img_metas, rescale=False):
        """Test function of point cloud branch with augmentaiton.

        The function implementation process is as follows:

            - step 1: map features back for double-flip augmentation.
            - step 2: merge all features and generate boxes.
            - step 3: map boxes back for scale augmentation.
            - step 4: merge results.

        Args:
            feats (list[torch.Tensor]): Feature of point cloud.
            img_metas (list[dict]): Meta information of samples.
            rescale (bool, optional): Whether to rescale bboxes.
                Default: False.

        Returns:
            dict: Returned bboxes consists of the following keys:

                - boxes_3d (:obj:`LiDARInstance3DBoxes`): Predicted bboxes.
                - scores_3d (torch.Tensor): Scores of predicted boxes.
                - labels_3d (torch.Tensor): Labels of predicted boxes.
        """
        # only support aug_test for one sample
        outs_list = []
        for x, img_meta in zip(feats, img_metas):
            outs = self.pts_bbox_head(x)
            # merge augmented outputs before decoding bboxes
            for task_id, out in enumerate(outs):
                for key in out[0].keys():
                    if img_meta[0]['pcd_horizontal_flip']:
                        outs[task_id][0][key] = torch.flip(
                            outs[task_id][0][key], dims=[2])
                        if key == 'reg':
                            outs[task_id][0][key][:, 1, ...] = 1 - outs[
                                task_id][0][key][:, 1, ...]
                        elif key == 'rot':
                            outs[task_id][0][
                                key][:, 0,
                                     ...] = -outs[task_id][0][key][:, 0, ...]
                        elif key == 'vel':
                            outs[task_id][0][
                                key][:, 1,
                                     ...] = -outs[task_id][0][key][:, 1, ...]
                    if img_meta[0]['pcd_vertical_flip']:
                        outs[task_id][0][key] = torch.flip(
                            outs[task_id][0][key], dims=[3])
                        if key == 'reg':
                            outs[task_id][0][key][:, 0, ...] = 1 - outs[
                                task_id][0][key][:, 0, ...]
                        elif key == 'rot':
                            outs[task_id][0][
                                key][:, 1,
                                     ...] = -outs[task_id][0][key][:, 1, ...]
                        elif key == 'vel':
                            outs[task_id][0][
                                key][:, 0,
                                     ...] = -outs[task_id][0][key][:, 0, ...]

            outs_list.append(outs)

        preds_dicts = dict()
        scale_img_metas = []

        # concat outputs sharing the same pcd_scale_factor
        for i, (img_meta, outs) in enumerate(zip(img_metas, outs_list)):
            pcd_scale_factor = img_meta[0]['pcd_scale_factor']
            if pcd_scale_factor not in preds_dicts.keys():
                preds_dicts[pcd_scale_factor] = outs
                scale_img_metas.append(img_meta)
            else:
                for task_id, out in enumerate(outs):
                    for key in out[0].keys():
                        preds_dicts[pcd_scale_factor][task_id][0][key] += out[
                            0][key]

        aug_bboxes = []

        for pcd_scale_factor, preds_dict in preds_dicts.items():
            for task_id, pred_dict in enumerate(preds_dict):
                # merge outputs with different flips before decoding bboxes
                for key in pred_dict[0].keys():
                    preds_dict[task_id][0][key] /= len(outs_list) / len(
                        preds_dicts.keys())
            bbox_list = self.pts_bbox_head.get_bboxes(
                preds_dict, img_metas[0], rescale=rescale)
            bbox_list = [
                dict(boxes_3d=bboxes, scores_3d=scores, labels_3d=labels)
                for bboxes, scores, labels in bbox_list
            ]
            aug_bboxes.append(bbox_list[0])

        if len(preds_dicts.keys()) > 1:
            # merge outputs with different scales after decoding bboxes
            merged_bboxes = merge_aug_bboxes_3d(aug_bboxes, scale_img_metas,
                                                self.pts_bbox_head.test_cfg)
            return merged_bboxes
        else:
            for key in bbox_list[0].keys():
                bbox_list[0][key] = bbox_list[0][key].to('cpu')
            return bbox_list[0]

    def aug_test(self, points, img_metas, imgs=None, rescale=False):
        """Test function with augmentaiton."""
        img_feats, pts_feats = self.extract_feats(points, img_metas, imgs)
        bbox_list = dict()
        if pts_feats and self.with_pts_bbox:
            pts_bbox = self.aug_test_pts(pts_feats, img_metas, rescale)
            bbox_list.update(pts_bbox=pts_bbox)
        return [bbox_list]

    # For Getting Flops
    def forward_dummy(self, points):
        """
            Args:
                points: tensor, shape=[1, N, 4]
        """
        pts_feats = self.extract_pts_feat(points, None, None)
        outs = self.pts_bbox_head(pts_feats)

        return outs

# Baseline
@DETECTORS.register_module()
class FeatKDCenterPoint(CenterPoint):
    def __init__(self, 
                 distillation,
                 **kwargs):
        super(FeatKDCenterPoint, self).__init__(**kwargs)
        self.distillation = distillation
        self.loss_balance = distillation["loss_balance"]
        # To align feature dim between student and teacher 32, 64, 128
        self.stu_adapt_list = nn.ModuleList()
        for i in range(distillation["num_levels"]):
            self.stu_adapt_list.append(
                ConvModule(
                    in_channels=distillation["in_channels"][i],
                    out_channels=distillation["feat_channels"][i],
                    kernel_size=1,
                    conv_cfg=distillation["conv_cfg"],
                    norm_cfg=distillation["norm_cfg"],
                    act_cfg=distillation["act_cfg"]
                )     
            )

    def feat_distillation(self, stu_feats, tea_feats, level_idx):
        """
            Args:
                stu_feats: tensor, shape=[B, C, H, W]
                tea_feats: tensor, shape=[B, D, H, W]
                level_idx: int
            Return:
                loss_kd: tensor

        """
        loss_kd = self.loss_balance * F.mse_loss(self.stu_adapt_list[level_idx](stu_feats), tea_feats, reduction="mean")

        return loss_kd, 

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img=None,
                      proposals=None,
                      teacher=None,
                      gt_bboxes_ignore=None):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor, optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """
        with torch.no_grad():
            teacher = teacher.to(points[0].device)
            _, __, tea_pts_infos = teacher.extract_feat(points, img=img, img_metas=img_metas, return_pts_middle=True)
        img_feats, pts_feats, stu_pts_infos = self.extract_feat(
            points, img=img, img_metas=img_metas, return_pts_middle=True)
        losses = dict()
        if pts_feats:
            losses_pts = self.forward_pts_train(pts_feats, gt_bboxes_3d,
                                                gt_labels_3d, img_metas,
                                                gt_bboxes_ignore)
            losses.update(losses_pts)
        if img_feats:
            losses_img = self.forward_img_train(
                img_feats,
                img_metas=img_metas,
                gt_bboxes=gt_bboxes,
                gt_labels=gt_labels,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposals=proposals)
            losses.update(losses_img)
        
        # FitNets
        level_list = list(range(self.distillation["num_levels"]))
        loss_kd = multi_apply(
            self.feat_distillation,
            stu_pts_infos["backbone_features"],
            tea_pts_infos["backbone_features"],
            level_list, 
        )[0]
        losses["loss_kd"] = loss_kd           
            
        return losses

    def train_step(self, data, optimizer, teacher):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``, \
                ``num_samples``.

                - ``loss`` is a tensor for back propagation, which can be a
                  weighted sum of multiple losses.
                - ``log_vars`` contains all the variables to be sent to the
                  logger.
                - ``num_samples`` indicates the batch size (when the model is
                  DDP, it means the batch size on each GPU), which is used for
                  averaging the logs.
        """
        data["teacher"] = teacher
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img_metas']))

        return outputs

@DETECTORS.register_module()
class PAKDCenterPoint(CenterPoint):
    def __init__(self, 
                 distillation,
                 **kwargs):
        super(PAKDCenterPoint, self).__init__(**kwargs)
        self.distillation = distillation
        self.loss_balance = distillation["loss_balance"]

    def payattention_distillation(self, stu_feat, tea_feat):
        """
            Args:
                stu_feat: type=Tensor, shape=[B, C_S, H, W]
                tea_feat: type=Tensor, shape=[B, C_T, H, W]
            Return:
                loss
        """
        B = stu_feat.shape[0]
        stu_atten = F.normalize(stu_feat.pow(2).mean(1).view(B, -1))
        tea_atten = F.normalize(tea_feat.pow(2).mean(1).view(B, -1))

        loss = torch.pow(stu_atten - tea_atten, 2).sum() * self.loss_balance
        
        return loss,

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img=None,
                      proposals=None,
                      teacher=None,
                      gt_bboxes_ignore=None):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor, optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """
        with torch.no_grad():
            teacher = teacher.to(points[0].device)
            _, __, tea_pts_infos = teacher.extract_feat(points, img=img, img_metas=img_metas, return_pts_middle=True)
        img_feats, pts_feats, stu_pts_infos = self.extract_feat(
            points, img=img, img_metas=img_metas, return_pts_middle=True)
        losses = dict()
        if pts_feats:
            losses_pts = self.forward_pts_train(pts_feats, gt_bboxes_3d,
                                                gt_labels_3d, img_metas,
                                                gt_bboxes_ignore)
            losses.update(losses_pts)
        if img_feats:
            losses_img = self.forward_img_train(
                img_feats,
                img_metas=img_metas,
                gt_bboxes=gt_bboxes,
                gt_labels=gt_labels,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposals=proposals)
            losses.update(losses_img)
        
        # PAKD
        loss_kd = multi_apply(
            self.payattention_distillation,
            stu_pts_infos["backbone_features"],
            tea_pts_infos["backbone_features"],
        )[0]
        losses["loss_pakd"] = loss_kd
        
        return losses

    def train_step(self, data, optimizer, teacher):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``, \
                ``num_samples``.

                - ``loss`` is a tensor for back propagation, which can be a
                  weighted sum of multiple losses.
                - ``log_vars`` contains all the variables to be sent to the
                  logger.
                - ``num_samples`` indicates the batch size (when the model is
                  DDP, it means the batch size on each GPU), which is used for
                  averaging the logs.
        """
        data["teacher"] = teacher
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img_metas']))

        return outputs

@DETECTORS.register_module()
class SPKDCenterPoint(CenterPoint):
    def __init__(self, 
                 distillation,
                 **kwargs):
        super(SPKDCenterPoint, self).__init__(**kwargs)
        self.distillation = distillation
        self.loss_balance = distillation["loss_balance"]
        # To align feature dim between student and teacher 32, 64, 128
        self.stu_adapt_list = nn.ModuleList()
        for i in range(distillation["num_levels"]):
            self.stu_adapt_list.append(
                ConvModule(
                    in_channels=distillation["in_channels"][i],
                    out_channels=distillation["feat_channels"][i],
                    kernel_size=1,
                    conv_cfg=distillation["conv_cfg"],
                    norm_cfg=distillation["norm_cfg"],
                    act_cfg=distillation["act_cfg"]
                )     
            )

    def batch_similarity(self, feats):
        """
            Args:
                feats: shape=[B, C, H, W]
            Returns:
                batch_sim: shape=[B, B]
        """
        feats = feats.view(feats.size(0), -1)  # [B, C*H*W]
        Q = torch.mm(feats, feats.transpose(0, 1))
        norm_Q = Q / torch.norm(Q, p=2, dim=1).unsqueeze(1).expand(Q.shape)
        
        return norm_Q    

    def spatial_similarity(self, feats):
        """
            Args:
                feats: shape=[B, C, H, W]
            Returns:
                spatial_sim: shape=[B, 1, H*W, H*W]
        """
        feats = feats.view(feats.size(0), feats.size(1), -1)  # [B, C, H*W]
        norm_feats = feats / (torch.sqrt(torch.sum(torch.pow(feats, 2), 1)).unsqueeze(1).expand(feats.shape) + 1e-7)
        s = norm_feats.transpose(1, 2).bmm(norm_feats)  # [B, H*W, C] @ [B, C, H*w] = [B, H*W, H*W]
        s = s.unsqueeze(1)
        return s

    def channel_similarity(self, feats):
        """
            Args:
                feats: shape=[B, C, H, W]
            Returns:
                spatial_sim: shape=[B, 1, C, C]
        """
        feats = feats.view(feats.size(0), feats.size(1), -1)  # [B, C, H*W]
        norm_fm = feats / (torch.sqrt(torch.sum(torch.pow(feats, 2), 2)).unsqueeze(2).expand(feats.shape) + 1e-7)
        s = norm_fm.bmm(norm_fm.transpose(1,2))  # [B, C, C]
        s = s.unsqueeze(1)  # [B, 1, C, C]
        return s 
        
    def sp_distillation(self, stu_feat, tea_feat, level_idx):
        """
            Args:
                stu_feat: type=Tensor, shape=[B, C_S, H, W]
                tea_feat: type=Tensor, shape=[B, C_T, H, W]
            Returns:
                loss
        """
        # tea_feat = self.tea_adapt(tea_feat)
        # batch similarity
        L2 = nn.MSELoss(reduction="mean")
        stu_bs, tea_bs = self.batch_similarity(stu_feat), self.batch_similarity(tea_feat)
        stu_ss, tea_ss = self.spatial_similarity(stu_feat), self.spatial_similarity(tea_feat)
        tea_cs = self.channel_similarity(tea_feat)
        stu_cs = self.channel_similarity(self.stu_adapt_list[level_idx](stu_feat))
        loss_bs = self.loss_balance[0] * L2(stu_bs, tea_bs)
        loss_ss = self.loss_balance[1] * L2(stu_ss, tea_ss)
        loss_cs = self.loss_balance[2] * L2(stu_cs, tea_cs)
        
        loss = loss_bs + loss_ss + loss_cs

        return loss,  

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img=None,
                      proposals=None,
                      teacher=None,
                      gt_bboxes_ignore=None):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor, optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """
        with torch.no_grad():
            teacher = teacher.to(points[0].device)
            _, __, tea_pts_infos = teacher.extract_feat(points, img=img, img_metas=img_metas, return_pts_middle=True)
        img_feats, pts_feats, stu_pts_infos = self.extract_feat(
            points, img=img, img_metas=img_metas, return_pts_middle=True)
        losses = dict()
        if pts_feats:
            losses_pts = self.forward_pts_train(pts_feats, gt_bboxes_3d,
                                                gt_labels_3d, img_metas,
                                                gt_bboxes_ignore)
            losses.update(losses_pts)
        if img_feats:
            losses_img = self.forward_img_train(
                img_feats,
                img_metas=img_metas,
                gt_bboxes=gt_bboxes,
                gt_labels=gt_labels,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposals=proposals)
            losses.update(losses_img)
        
        # FitNets
        level_list = list(range(self.distillation["num_levels"]))
        loss_kd = multi_apply(
            self.sp_distillation,
            stu_pts_infos["backbone_features"],
            tea_pts_infos["backbone_features"],
            level_list, 
        )[0]
        losses["loss_kd"] = loss_kd           
            
        return losses

    def train_step(self, data, optimizer, teacher):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``, \
                ``num_samples``.

                - ``loss`` is a tensor for back propagation, which can be a
                  weighted sum of multiple losses.
                - ``log_vars`` contains all the variables to be sent to the
                  logger.
                - ``num_samples`` indicates the batch size (when the model is
                  DDP, it means the batch size on each GPU), which is used for
                  averaging the logs.
        """
        data["teacher"] = teacher
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img_metas']))

        return outputs
 
@DETECTORS.register_module()
class PPKDCenterPoint(CenterPoint):
    def forward_pts_train(self,
                          pts_feats,
                          tea_pred_infos,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          img_metas,
                          gt_bboxes_ignore=None):
        """Forward function for point cloud branch.

        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            tea_pred_infos: dict
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.

        Returns:
            dict: Losses of each branch.
        """
        outs = self.pts_bbox_head(pts_feats)
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs, tea_pred_infos]
        losses = self.pts_bbox_head.loss(*loss_inputs)
        return losses

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img=None,
                      proposals=None,
                      teacher=None,
                      gt_bboxes_ignore=None):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor, optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """
        with torch.no_grad():
            tea_pred_infos = dict()
            tea_pred_heatmap_list = []
            tea_pred_reg_list = []
            teacher = teacher.to(points[0].device)
            _, tea_pts_feats, tea_pts_infos = teacher.extract_feat(points, img=img, img_metas=img_metas, return_pts_middle=True)
            tea_preds = teacher.pts_bbox_head.forward(tea_pts_feats)
            for tea_pred in tea_preds:
                if isinstance(tea_pred, list) and len(tea_pred) == 1:
                    tea_pred_heatmap_list.append(tea_pred[0]["heatmap"])
                    tea_pred_reg_list.append(torch.cat([tea_pred[0][head_name] for head_name in ["reg", "height", "dim", "rot"]], dim=1))
                else:
                    raise NotImplementedError
            tea_pred_infos["heatmap"] = tea_pred_heatmap_list
            tea_pred_infos["reg"] = tea_pred_reg_list

        img_feats, pts_feats, stu_pts_infos = self.extract_feat(
            points, img=img, img_metas=img_metas, return_pts_middle=True)
        losses = dict()
        if pts_feats:
            losses_pts = self.forward_pts_train(pts_feats, tea_pred_infos, gt_bboxes_3d,
                                                gt_labels_3d, img_metas,
                                                gt_bboxes_ignore)
            losses.update(losses_pts)
        if img_feats:
            losses_img = self.forward_img_train(
                img_feats,
                img_metas=img_metas,
                gt_bboxes=gt_bboxes,
                gt_labels=gt_labels,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposals=proposals)
            losses.update(losses_img)
                
        return losses    

    def train_step(self, data, optimizer, teacher):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``, \
                ``num_samples``.

                - ``loss`` is a tensor for back propagation, which can be a
                  weighted sum of multiple losses.
                - ``log_vars`` contains all the variables to be sent to the
                  logger.
                - ``num_samples`` indicates the batch size (when the model is
                  DDP, it means the batch size on each GPU), which is used for
                  averaging the logs.
        """
        data["teacher"] = teacher
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img_metas']))

        return outputs

@DETECTORS.register_module()
class PDCenterPoint(CenterPoint):
    def __init__(self, 
                 distillation,
                 **kwargs):
        super(PDCenterPoint, self).__init__(**kwargs)
        self.distillation = distillation
        self.kd_type = distillation["kd_type"]
        self.num_layers = distillation["num_layers"]
        self.num_voxels = distillation["num_voxels"]
        self.kneighbours = distillation["kneighbours"]
        self.temperature = distillation["temperature"]
        self.gcn_layers_init() 
    
    def gcn_layers_init(self):
        tea_gcn_cfg, stu_gcn_cfg = self.distillation["tea_gcn_cfg"], self.distillation["stu_gcn_cfg"]
        if self.kd_type == "voxel":
            self.tea_gcn_adapt = ConvModule(
                in_channels=tea_gcn_cfg["in_channels"]*2,
                out_channels=tea_gcn_cfg["feat_channels"],
                kernel_size=(1, 1),
                stride=(1, 1),
                conv_cfg=tea_gcn_cfg["conv_cfg"],
                norm_cfg=tea_gcn_cfg["norm_cfg"],
                act_cfg=tea_gcn_cfg["act_cfg"],
                bias=tea_gcn_cfg["bias"],
            )
            self.stu_gcn_adapt = ConvModule(
                in_channels=stu_gcn_cfg["in_channels"]*2,
                out_channels=stu_gcn_cfg["feat_channels"],
                kernel_size=(1, 1),
                stride=(1, 1),
                conv_cfg=stu_gcn_cfg["conv_cfg"],
                norm_cfg=stu_gcn_cfg["norm_cfg"],
                act_cfg=stu_gcn_cfg["act_cfg"],
                bias=stu_gcn_cfg["bias"],
            )
        elif self.kd_type == "backbone":
            self.tea_gcn_adapt_list = nn.ModuleList()
            self.stu_gcn_adapt_list = nn.ModuleList()
            for i in range(self.num_layers):
                self.tea_gcn_adapt_list.append(
                    ConvModule(
                        in_channels=tea_gcn_cfg["in_channels"][i] * 2,
                        out_channels=tea_gcn_cfg["feat_channels"],
                        kernel_size=(1, 1),
                        stride=(1, 1),
                        conv_cfg=tea_gcn_cfg["conv_cfg"],
                        norm_cfg=tea_gcn_cfg["norm_cfg"],
                        act_cfg=tea_gcn_cfg["act_cfg"],
                        bias=tea_gcn_cfg["bias"],
                    )
                )    
                self.stu_gcn_adapt_list.append(
                    ConvModule(
                        in_channels=stu_gcn_cfg["in_channels"][i] * 2,
                        out_channels=stu_gcn_cfg["feat_channels"],
                        kernel_size=(1, 1),
                        stride=(1, 1),
                        conv_cfg=stu_gcn_cfg["conv_cfg"],
                        norm_cfg=stu_gcn_cfg["norm_cfg"],
                        act_cfg=stu_gcn_cfg["act_cfg"],
                        bias=stu_gcn_cfg["bias"],
                    )
                )

    def get_voxel_knn(self, teacher_feature, student_feature, num_voxel_points, 
                        num_voxels=6000, kneighbours=128, reweight=False,
                        sample_mode='top', fuse_mode='idendity'):
        """Perform voxel Local KNN graph knowledge distillation with reweighting strategy.

        Args:
            teacher_feature (torch.Tensor): (nvoxels, C)
                Teacher features of locally grouped points/pillars before pooling.
            student_feature (torch.Tensor): (nvoxels, C)
                Student features of locally grouped points/pillars before pooling.
            num_voxel_points (torch.Tensor): (nvoxels, npoints)
                Number of points in the voxel.
            num_voxels (int):
                Number of voxels after sampling.
            kneighbours (int):
                Value of number of knn neighbours.
            reweight (bool, optional):
                Whether to use reweight to further filter voxel features for KD.
                Defaults to False.
            sample_mode (str, optional):
                Type of sampling method. Defaults to 'top'.
            fuse_mode (str, optional): Defaults to 'idendity'.
                Type of fusing the knn graph centering features.
        Returns:
            torch.Tensor:
                Feature distance between the student and teacher.
        """
        # query_voxel_idx: num_voxels
        if sample_mode == 'top':
            _, query_voxel_idx = torch.topk(num_voxel_points, num_voxels, dim=-1, largest=True, sorted=False)
        elif sample_mode == 'bottom':
            _, query_voxel_idx = torch.topk(num_voxel_points, num_voxels, dim=-1, largest=False, sorted=False)
        elif sample_mode == 'rand':
            query_voxel_idx = torch.randperm(teacher_feature.shape[0], device=teacher_feature.device)
            query_voxel_idx = query_voxel_idx[int(min(num_voxels, query_voxel_idx.shape[0])):]
        elif sample_mode == 'mixed':
            _, query_voxel_idx_pos = torch.topk(num_voxel_points, num_voxels//2, dim=-1, largest=True, sorted=False)
            _, query_voxel_idx_neg = torch.topk(num_voxel_points, num_voxels//2, dim=-1, largest=False, sorted=False)
            query_voxel_idx = torch.cat([query_voxel_idx_pos, query_voxel_idx_neg])
        else:
            raise NotImplementedError
        # query_feature_xxx: num_voxels x channel
        query_feature_tea, query_feature_stu = teacher_feature[query_voxel_idx], student_feature[query_voxel_idx]  # [num_samples, D_tea] [num_samples, D_stu]
        # cluster_idx: 1 x num_voxels x kneighbours
        # cluster_idx = query_ball_feature(radius, kneighbours, teacher_feature, query_feature_tea) # ball query
        cluster_idx = knn_feature(kneighbours, teacher_feature.unsqueeze(0), query_feature_tea.unsqueeze(0)) # knn
        # grouped_voxels_xxx: 1 x num_voxels x kneighbours x channel -> 1 x channel x num_voxels x kneighbours          # K-neighbours features
        grouped_voxels_tea = index_feature(teacher_feature.unsqueeze(0), cluster_idx).permute(0, 3, 1, 2).contiguous()  # [1, D_tea, num_samples, K]
        grouped_voxels_stu = index_feature(student_feature.unsqueeze(0), cluster_idx).permute(0, 3, 1, 2).contiguous()  # [1, D_stu, num_samples, K]
        
        # num_voxels x channel -> 1 x channels x num_voxels x 1 -> 1 x num_voxels x channels x kneighbours   # samples features
        new_query_feature_tea = query_feature_tea.transpose(1, 0).unsqueeze(0).unsqueeze(-1).repeat(         # [1, D_tea, num_samples, K]
                                    1, 1, 1, kneighbours).contiguous()
        new_query_feature_stu = query_feature_stu.transpose(1, 0).unsqueeze(0).unsqueeze(-1).repeat(         # [1, D_stu, num_samples, K]
                                    1, 1, 1, kneighbours).contiguous()

        # KNN graph center feature fusion
        # 1 x channel x num_voxels x kneighbours -> 1 x (2xchannel) x num_voxels x kneighbours 
        if fuse_mode == 'idendity':
            # XXX We can concat the center feature
            grouped_voxels_tea = torch.cat([grouped_voxels_tea, new_query_feature_tea], dim=1)  # [1, 2*D_tea, num_samples, K]
            grouped_voxels_stu = torch.cat([grouped_voxels_stu, new_query_feature_stu], dim=1)  # [1, 2*D_stu, num_samples, K]
        elif fuse_mode == 'sub':
            # XXX Or, we can also use residual graph feature
            grouped_voxels_tea = torch.cat([new_query_feature_tea-grouped_voxels_tea, new_query_feature_tea], dim=1)
            grouped_voxels_stu = torch.cat([new_query_feature_stu-grouped_voxels_stu, new_query_feature_stu], dim=1)
        # 特征维度对齐
        grouped_voxels_tea = self.tea_gcn_adapt(grouped_voxels_tea)
        grouped_voxels_stu = self.stu_gcn_adapt(grouped_voxels_stu)
        
        # 1 x channel' x num_voxels x kneighbours -> num_voxels x channel' x kneighbours
        grouped_voxels_tea = grouped_voxels_tea.squeeze(0).transpose(1, 0)  # [num_samples, F_D, K]
        grouped_voxels_stu = grouped_voxels_stu.squeeze(0).transpose(1, 0)  # [num_samples, F_D, K]

        # calculate teacher and student knowledge gap
        dist = torch.FloatTensor([0.0]).to(grouped_voxels_stu.device)
        if reweight is False:
            dist += torch.dist(grouped_voxels_tea, grouped_voxels_stu) * 5e-4
        else:
            reweight = num_voxel_points[query_voxel_idx]
            reweight = F.softmax(reweight.float() / self.temperature, dim=-1) * reweight.shape[0]
            reweight = reweight.view(reweight.shape[0], 1, 1)
            _dist = F.mse_loss(grouped_voxels_tea, grouped_voxels_stu, reduce=False) * reweight
            dist += (_dist.mean() * 2)
        return dist

    def get_knn_voxel_fp(self, teacher_feature, student_feature, num_voxel_points=None, 
                            num_voxels=256, kneighbours=128, reweight=False, relation=False,
                            pool_mode='none', sample_mode='top', fuse_mode='idendity'):
        batch_size = teacher_feature.shape[0]
        # B C H W -> B (H*W) C: batch_size x nvoxels x channel_tea
        teacher_feature = teacher_feature.view(batch_size, teacher_feature.shape[1], -1).transpose(2, 1).contiguous()
        student_feature = student_feature.view(batch_size, student_feature.shape[1], -1).transpose(2, 1).contiguous()

        if num_voxel_points is None:
            num_voxel_points = pool_features1d(teacher_feature).squeeze(-1)
            # num_voxel_points = teacher_feature.abs().mean(-1)
        # query_voxel_idx: batch x num_voxels
        if sample_mode == 'top':
            _, query_voxel_idx = torch.topk(num_voxel_points, num_voxels, dim=-1, largest=True, sorted=False)
        elif sample_mode == 'bottom':
            _, query_voxel_idx = torch.topk(num_voxel_points, num_voxels, dim=-1, largest=False, sorted=False)
        elif sample_mode == 'rand':
            query_voxel_idx = torch.randperm(teacher_feature.shape[0], device=teacher_feature.device)
            query_voxel_idx = query_voxel_idx[int(min(num_voxels, query_voxel_idx.shape[0])):]
        elif sample_mode == 'mixed':
            _, query_voxel_idx_pos = torch.topk(num_voxel_points, num_voxels//2, dim=-1, largest=True, sorted=False)
            _, query_voxel_idx_neg = torch.topk(num_voxel_points, num_voxels//2, dim=-1, largest=False, sorted=False)
            query_voxel_idx = torch.cat([query_voxel_idx_pos, query_voxel_idx_neg])
        else:
            raise NotImplementedError
        # query_feature_xxx: B x num_voxels x channel
        query_feature_tea, query_feature_stu = index_feature(teacher_feature, query_voxel_idx), index_feature(student_feature, query_voxel_idx)
        # cluster_idx: B x num_voxels x kneighbours
        # cluster_idx = query_ball_feature(radius, kneighbours, teacher_feature, query_feature_tea) # ball query
        cluster_idx = knn_feature(kneighbours, teacher_feature, query_feature_tea) # knn
        # grouped_voxels_xxx: B x num_voxels x kneighbours x channel -> B x channel x num_voxels x kneighbours
        grouped_voxels_tea = index_feature(teacher_feature, cluster_idx).permute(0, 3, 1, 2).contiguous()
        grouped_voxels_stu = index_feature(student_feature, cluster_idx).permute(0, 3, 1, 2).contiguous()

        # B x num_voxels x channel -> B x channels x num_voxels x 1 -> B x num_voxels x channels x kneighbours
        new_query_feature_tea = query_feature_tea.transpose(2, 1).unsqueeze(-1).repeat(
                                    1, 1, 1, kneighbours).contiguous()
        new_query_feature_stu = query_feature_stu.transpose(2, 1).unsqueeze(-1).repeat(
                                    1, 1, 1, kneighbours).contiguous()

        # KNN graph center feature fusion
        # 1 x channel x num_voxels x kneighbours -> 1 x (2xchannel) x num_voxels x kneighbours 
        if fuse_mode == 'idendity':
            # XXX We can concat the center feature
            grouped_voxels_tea = torch.cat([grouped_voxels_tea, new_query_feature_tea], dim=1)
            grouped_voxels_stu = torch.cat([grouped_voxels_stu, new_query_feature_stu], dim=1)
        elif fuse_mode == 'sub':
            # XXX Or, we can also use residual graph feature
            grouped_voxels_tea = torch.cat([new_query_feature_tea-grouped_voxels_tea, new_query_feature_tea], dim=1)
            grouped_voxels_stu = torch.cat([new_query_feature_stu-grouped_voxels_stu, new_query_feature_stu], dim=1)

        grouped_voxels_tea = self.tea_gcn_adapt(grouped_voxels_tea)
        grouped_voxels_stu = self.stu_gcn_adapt(grouped_voxels_stu)

        if pool_mode != 'none':
            # global feature extraction via local feature aggreation
            # B C N K -> B C N
            grouped_points_tea = pool_features(grouped_points_tea, pool_mode)
            grouped_points_stu = pool_features(grouped_points_stu, pool_mode)

        # batch x channel' x num_voxels x kneighbours -> batch x num_voxels x channel' x kneighbours
        grouped_voxels_tea = grouped_voxels_tea.transpose(2, 1)
        grouped_voxels_stu = grouped_voxels_stu.transpose(2, 1)

        # calculate teacher and student knowledge gap
        dist = torch.FloatTensor([0.0]).to(grouped_voxels_stu.device)
        if reweight is False:
            dist += torch.dist(grouped_voxels_tea, grouped_voxels_stu) * 5e-4
        else:
            # reweight: B x num_voxel x 1
            reweight = index_feature(num_voxel_points.unsqueeze(-1), query_voxel_idx)
            reweight = F.softmax(reweight.float() / self.kd_temperature, dim=-1) * reweight.shape[0]
            if grouped_voxels_tea.ndim == 4:
                reweight = reweight.unsqueeze(-1)
            _dist = F.mse_loss(grouped_voxels_tea, grouped_voxels_stu, reduce=False) * reweight
            dist += (_dist.mean() * 2)
        return dist

    def get_knn_voxel_backbone(self, teacher_feature, student_feature, layer_idx, num_voxel_points=None, 
                                num_voxels=6000, kneighbours=128, reweight=False,
                                sample_mode='top', fuse_mode='idendity'):
        """Perform voxel Local KNN graph knowledge distillation with reweighting strategy.

        Args:
            teacher_feature (torch.Tensor): (nvoxels, C)
                Teacher features of locally grouped points/pillars before pooling.
            student_feature (torch.Tensor): (nvoxels, C)
                Student features of locally grouped points/pillars before pooling.
            num_voxel_points (torch.Tensor): (nvoxels, npoints)
                Number of points in the voxel.
            num_voxels (int):
                Number of voxels after sampling.
            kneighbours (int):
                Value of number of knn neighbours.
            reweight (bool, optional):
                Whether to use reweight to further filter voxel features for KD.
                Defaults to False.
            sample_mode (str, optional):
                Type of sampling method. Defaults to 'top'.
            fuse_mode (str, optional): Defaults to 'idendity'.
                Type of fusing the knn graph centering features.
        Returns:
            torch.Tensor:
                Feature distance between the student and teacher.
        """
        teacher_feature, student_feature = teacher_feature[layer_idx], student_feature[layer_idx]
        batch_size = teacher_feature.shape[0]
        # B C H W -> B (H*W) C: batch_size x nvoxels x channel_tea
        teacher_feature = teacher_feature.view(batch_size, teacher_feature.shape[1], -1).transpose(2, 1).contiguous()
        student_feature = student_feature.view(batch_size, student_feature.shape[1], -1).transpose(2, 1).contiguous()

        if num_voxel_points is None:
            num_voxel_points = pool_features1d(teacher_feature).squeeze(-1)
        # query_voxel_idx: batch x num_voxels
        if sample_mode == 'top':
            _, query_voxel_idx = torch.topk(num_voxel_points, num_voxels, dim=-1, largest=True, sorted=False)
        elif sample_mode == 'bottom':
            _, query_voxel_idx = torch.topk(num_voxel_points, num_voxels, dim=-1, largest=False, sorted=False)
        elif sample_mode == 'rand':
            query_voxel_idx = torch.randperm(teacher_feature.shape[0], device=teacher_feature.device)
            query_voxel_idx = query_voxel_idx[int(min(num_voxels, query_voxel_idx.shape[0])):]
        elif sample_mode == 'mixed':
            _, query_voxel_idx_pos = torch.topk(num_voxel_points, num_voxels//2, dim=-1, largest=True, sorted=False)
            _, query_voxel_idx_neg = torch.topk(num_voxel_points, num_voxels//2, dim=-1, largest=False, sorted=False)
            query_voxel_idx = torch.cat([query_voxel_idx_pos, query_voxel_idx_neg])
        else:
            raise NotImplementedError

        # query_feature_xxx: B x num_voxels x channel
        query_feature_tea, query_feature_stu = index_feature(teacher_feature, query_voxel_idx), index_feature(student_feature, query_voxel_idx)
        # cluster_idx: B x num_voxels x kneighbours
        # cluster_idx = query_ball_feature(radius, kneighbours, teacher_feature, query_feature_tea) # ball query
        cluster_idx = knn_feature(kneighbours, teacher_feature, query_feature_tea) # knn

        # grouped_voxels_xxx: B x num_voxels x kneighbours x channel -> B x channel x num_voxels x kneighbours
        grouped_voxels_tea = index_feature(teacher_feature, cluster_idx).permute(0, 3, 1, 2).contiguous()
        grouped_voxels_stu = index_feature(student_feature, cluster_idx).permute(0, 3, 1, 2).contiguous()
        
        # B x num_voxels x channel -> B x channels x num_voxels x 1 -> B x num_voxels x channels x kneighbours
        new_query_feature_tea = query_feature_tea.transpose(2, 1).unsqueeze(-1).repeat(
                                    1, 1, 1, kneighbours).contiguous()
        new_query_feature_stu = query_feature_stu.transpose(2, 1).unsqueeze(-1).repeat(
                                    1, 1, 1, kneighbours).contiguous()

        # KNN graph center feature fusion
        # 1 x channel x num_voxels x kneighbours -> 1 x (2xchannel) x num_voxels x kneighbours 
        if fuse_mode == 'idendity':
            # XXX We can concat the center feature
            grouped_voxels_tea = torch.cat([grouped_voxels_tea, new_query_feature_tea], dim=1)
            grouped_voxels_stu = torch.cat([grouped_voxels_stu, new_query_feature_stu], dim=1)
        elif fuse_mode == 'sub':
            # XXX Or, we can also use residual graph feature
            grouped_voxels_tea = torch.cat([new_query_feature_tea-grouped_voxels_tea, new_query_feature_tea], dim=1)
            grouped_voxels_stu = torch.cat([new_query_feature_stu-grouped_voxels_stu, new_query_feature_stu], dim=1)

        grouped_voxels_tea = self.tea_gcn_adapt_list[layer_idx](grouped_voxels_tea)
        grouped_voxels_stu = self.stu_gcn_adapt_list[layer_idx](grouped_voxels_stu)
        
        # batch x channel' x num_voxels x kneighbours -> batch x num_voxels x channel' x kneighbours
        grouped_voxels_tea = grouped_voxels_tea.transpose(2, 1)
        grouped_voxels_stu = grouped_voxels_stu.transpose(2, 1)

        # calculate teacher and student knowledge gap
        dist = torch.FloatTensor([0.0]).to(grouped_voxels_stu.device)
        if reweight is False:
            dist += torch.dist(grouped_voxels_tea, grouped_voxels_stu) * 5e-4
        else:
            # reweight: B x num_voxel x 1
            reweight = index_feature(num_voxel_points.unsqueeze(-1), query_voxel_idx)
            reweight = F.softmax(reweight.float() / 1.e-4, dim=-1) * reweight.shape[0]
            if grouped_voxels_tea.ndim == 4:
                reweight = reweight.unsqueeze(-1)
            _dist = F.mse_loss(grouped_voxels_tea, grouped_voxels_stu, reduce=False) * reweight
            dist += (_dist.mean() * 2)
        return dist

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img=None,
                      proposals=None,
                      teacher=None,
                      gt_bboxes_ignore=None):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor, optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """
        with torch.no_grad():
            teacher = teacher.to(points[0].device)
            _, __, tea_pts_infos = teacher.extract_feat(points, img=img, img_metas=img_metas, return_pts_middle=True)
        img_feats, pts_feats, stu_pts_infos = self.extract_feat(
            points, img=img, img_metas=img_metas, return_pts_middle=True)
        losses = dict()
        if pts_feats:
            losses_pts = self.forward_pts_train(pts_feats, gt_bboxes_3d,
                                                gt_labels_3d, img_metas,
                                                gt_bboxes_ignore)
            losses.update(losses_pts)
        if img_feats:
            losses_img = self.forward_img_train(
                img_feats,
                img_metas=img_metas,
                gt_bboxes=gt_bboxes,
                gt_labels=gt_labels,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposals=proposals)
            losses.update(losses_img)
        
        # PDistiller
        if tea_pts_infos is not None:
            if self.kd_type == "voxel":
                loss_kd = torch.FloatTensor([0.0]).to(points[0].device)
                loss_kd += self.get_voxel_knn(
                    teacher_feature=tea_pts_infos['voxel_features'], student_feature=stu_pts_infos['voxel_features'],
                    num_voxel_points=tea_pts_infos['num_points'], num_voxels=self.num_voxels, kneighbours=self.kneighbours, 
                    reweight=True, sample_mode='top', fuse_mode='idendity'
                )
            elif self.kd_type == "backbone":
                loss_kd = torch.FloatTensor([0.0]).to(points[0].device)
                for i in range(self.num_layers):
                    loss_kd += self.get_knn_voxel_backbone(
                        teacher_feature=tea_pts_infos["backbone_features"], student_feature=stu_pts_infos["backbone_features"],
                        layer_idx=i, num_voxel_points=None, num_voxels=self.num_voxels, kneighbours=self.kneighbours,
                        reweight=True, sample_mode="top", fuse_mode="idendity"

                    )  
            
            else:
                raise NotImplementedError
        losses["loss_kd"] = loss_kd           
            
        return losses

    def train_step(self, data, optimizer, teacher):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``, \
                ``num_samples``.

                - ``loss`` is a tensor for back propagation, which can be a
                  weighted sum of multiple losses.
                - ``log_vars`` contains all the variables to be sent to the
                  logger.
                - ``num_samples`` indicates the batch size (when the model is
                  DDP, it means the batch size on each GPU), which is used for
                  averaging the logs.
        """
        data["teacher"] = teacher
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img_metas']))

        return outputs

# LGSD
@DETECTORS.register_module()
class LGSDCenterPoint(CenterPoint):
    def __init__(self,
                 distillation,
                 **kwargs):
        super(LGSDCenterPoint, self).__init__(**kwargs)
        self.distillation = distillation
        self.loss_balance = distillation["loss_balance"]
        self.local_cfg = distillation["local_cfg"]
        self.kd_type = distillation["kd_type"]
        self.HierLocalAndGlobal()
    
    def HierLocalAndGlobal(self):
        if self.kd_type == "voxel":
            self.SG_layers = self.local_cfg["SG_layers"]
            tea_in_channels, tea_out_channels = self.local_cfg["tea_in_channels"], self.local_cfg["tea_out_channels"]
            stu_in_channels, stu_out_channels = self.local_cfg["stu_in_channels"], self.local_cfg["stu_out_channels"]
            conv_cfg, norm_cfg, act_cfg = self.local_cfg["conv_cfg"], self.local_cfg["norm_cfg"], self.local_cfg["act_cfg"]
            self.tea_localconv = ConvModule(
                tea_in_channels * 2,
                tea_out_channels,
                kernel_size=1,
                stride=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
            )
            self.tea_adptconv = ConvModule(
                tea_out_channels,
                stu_out_channels,
                kernel_size=1,
                stride=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
            )
            self.stu_localconv = ConvModule(
                stu_in_channels * 2,
                stu_out_channels,
                kernel_size=1,
                stride=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,            
            )
            self.cl_head = build_cl_head(self.local_cfg["cl_head_cfg"])
        elif self.kd_type == "backbone":
            tea_in_channels, tea_out_channels = self.local_cfg["tea_in_channels"], self.local_cfg["tea_out_channels"]
            stu_in_channels, stu_out_channels = self.local_cfg["stu_in_channels"], self.local_cfg["stu_out_channels"]
            conv_cfg, norm_cfg, act_cfg = self.local_cfg["conv_cfg"], self.local_cfg["norm_cfg"], self.local_cfg["act_cfg"]
            self.tea_localconv_list = nn.ModuleList()
            self.tea_adptconv_list = nn.ModuleList()
            self.stu_localconv_list = nn.ModuleList()
            self.channel_atten_list = nn.ModuleList()
            self.cl_head_list = nn.ModuleList()
            cl_head_cfg = self.local_cfg["cl_head_cfg"]
            for i in range(len(tea_in_channels)):
                self.tea_localconv_list.append(
                    ConvModule(
                        tea_in_channels[i] * 2,
                        tea_out_channels[i],
                        kernel_size=1,
                        stride=1,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg,
                    )
                )
                self.tea_adptconv_list.append(
                    ConvModule(
                        tea_out_channels[i],
                        stu_out_channels[i],
                        kernel_size=1,
                        stride=1,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg,
                    )
                )                
                self.stu_localconv_list.append(
                    ConvModule(
                        stu_in_channels[i] * 2,
                        stu_out_channels[i],
                        kernel_size=1,
                        stride=1,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg,
                    )
                )
                # self.channel_atten_list.append(
                #     nn.Sequential(
                #         nn.Linear(tea_in_channels[i], tea_in_channels[i] // 4, bias=False),
                #         nn.ReLU(inplace=True),
                #         nn.Linear(tea_in_channels[i] // 4, tea_in_channels[i], bias=False),
                #         nn.Sigmoid(),
                #     )
                # )
                self.cl_head_list.append(build_cl_head(cl_head_cfg))
                if i != len(tea_in_channels) - 1:
                    cl_head_cfg["tea_channels"] = tea_out_channels[i + 1]
                    cl_head_cfg["stu_channels"] = stu_out_channels[i + 1]
                    cl_head_cfg["mid_channels"] = tea_out_channels[i + 1]

    def KNN_Grouping(self, query_voxel_id, voxel_feats, K):
        """
        Args:
            query_voxel_id: tensor, shape=[num_voxels]
            voxel_feats: tensor, shape=[num_voxels, D]
            num_sampling: int
            K: int
        Returns:

        """
        # sampling voxel according the number of points in voxel
        # 
        # query feats
        if len(voxel_feats.shape) == 3:
            # voxel_feats.shape = [B, N, D]
            assert len(query_voxel_id.shape) == 2
            query_voxel_feats = index_points(voxel_feats, query_voxel_id)  # [B, num_sampling, D]
            neighbor_idx = knn_query(K, voxel_feats, query_voxel_feats)  # [B, num_sampling, K]
            neighbor_voxel_feats = index_points(voxel_feats, neighbor_idx).permute(0, 3, 1, 2).contiguous()  # [B, D, num_sampling, K]
            query_voxel_feats = query_voxel_feats.transpose(2, 1).unsqueeze(-1).repeat(1, 1, 1, K).contiguous()  # [B, D, num_sampling, K]
        elif len(voxel_feats.shape) == 2:
            # voxel_feats.shape = [num_sampling, D]
            query_voxel_feats = voxel_feats[query_voxel_id]
            # get K nearest neighbors
            neighbor_idx = knn_query(K, voxel_feats.unsqueeze(0), query_voxel_feats.unsqueeze(0)) # [1, num_sampling, K]
            # get K-th neighbors feats
            neighbor_voxel_feats = index_points(voxel_feats.unsqueeze(0), neighbor_idx).permute(0, 3, 1, 2).contiguous()  # [1, D, num_sampling, K]
            # aligned feature dim
            query_voxel_feats = query_voxel_feats.transpose(1, 0).unsqueeze(0).unsqueeze(-1).repeat(1, 1, 1, K).contiguous()
        else:
            raise NotImplementedError
        
        return query_voxel_feats, neighbor_voxel_feats

    def backbone_local_global_distillation(self, voxel_num_points, tea_feats, stu_feats, num_samplings, num_neighbors, layer_idx):
        """
        Args:
            voxel_num_points: [B, N]
            tea_feats: [B, tea_feat_dim, H, W]
            stu_feats: [B, stu_feat_dim, H, W]
            num_samplings: int
            num_neighbors: int
            layer_idx: int
        Returns:

        """
        B, C, H, W = tea_feats.shape
        # [B, C, H, W] -> [B, C, H*W] -> [B, H*W, C]
        tea_flatten_feats = tea_feats.view(B, tea_feats.shape[1], -1).transpose(2, 1).contiguous()
        stu_flatten_feats = stu_feats.view(B, stu_feats.shape[1], -1).transpose(2, 1).contiguous()
        # aggegrate channel 
        if voxel_num_points is None:  
            # max pool
            # voxel_num_points = F.adaptive_max_pool1d(tea_flatten_feats, 1).squeeze(-1)  # [B, H*W]
            # SENet
            channel_agg_tea = tea_feats * self.channel_atten_list[layer_idx](F.adaptive_avg_pool2d(tea_feats, 1).reshape(B, C))[:, :, None, None]
            voxel_num_points = F.adaptive_max_pool1d(channel_agg_tea.view(B, C, -1).transpose(2, 1).contiguous(), 1).squeeze(-1)
            
        # sampling
        _, query_voxel_id = torch.topk(voxel_num_points, num_samplings, dim=-1, largest=True, sorted=False)  # [B, num_samplings]
        # KNN Grouping
        tea_query_feats, tea_neighbor_feats = self.KNN_Grouping(query_voxel_id, tea_flatten_feats, num_neighbors)  # [B, D, num_samplings, K]
        stu_query_feats, stu_neighbor_feats = self.KNN_Grouping(query_voxel_id, stu_flatten_feats, num_neighbors)
        
        # localKD
        tea_local_feats = torch.cat([tea_query_feats - tea_neighbor_feats, tea_query_feats], dim=1)  # [B, 2*tea_dim, num_samplings, K]
        stu_local_feats = torch.cat([stu_query_feats - stu_neighbor_feats, stu_query_feats], dim=1)   # [B, 2*stu_dim, num_samplings, K]
        tea_local_feats = self.tea_localconv_list[layer_idx](tea_local_feats)  # [B, tea_dim, num_sampling, K]
        tea_local_apt_feats = self.tea_adptconv_list[layer_idx](tea_local_feats)  # [B, stu_dim, num_sampling, K]
        stu_local_feats = self.stu_localconv_list[layer_idx](stu_local_feats)  # [B, stu_dim, num_sampling, K]
       
        # feat
        local_feat_loss = F.mse_loss(stu_local_feats, tea_local_apt_feats, reduction="mean")
        # relation
        stu_feat_dim, tea_feat_dim = stu_local_feats.shape[1], tea_local_feats.shape[1]
        local_relation_loss = self.cl_head_list[layer_idx](stu_local_feats.permute(0, 2, 3, 1).reshape(-1, stu_feat_dim), tea_local_feats.permute(0, 2, 3, 1).reshape(-1, tea_feat_dim))
        local_loss = self.loss_balance[0] * local_feat_loss + self.loss_balance[0] * local_relation_loss

        # aggregate   [B, stu_dim, num_sampling, K] -> [B, feat_dim, num_sampling] -> [B*num_sampling, feat_dim]
        tea_global_feats = torch.mean(tea_local_feats, dim=-1).permute(0, 2, 1).reshape(-1, tea_feat_dim)  # [num_sampling, tea_vf] [num_sampling, stu_vf]
        stu_global_feats = torch.mean(stu_local_feats, dim=-1).permute(0, 2, 1).reshape(-1, stu_feat_dim)
        # relation matrix by computing cos  # [B*num_sampling, stu_feat_dim] [B*num_sampling, tea_feat_dim]
        tea_norm_global_feats = tea_global_feats / ((torch.norm(tea_global_feats, p=2, dim=-1, keepdim=True) + 1e-5))
        tea_relation_mat = torch.matmul(tea_norm_global_feats, tea_norm_global_feats.transpose(1, 0))

        stu_norm_global_feats = stu_global_feats / ((torch.norm(stu_global_feats, p=2, dim=-1, keepdim=True) + 1e-5))
        stu_relation_mat = torch.matmul(stu_norm_global_feats, stu_norm_global_feats.transpose(1, 0))

        global_loss = self.loss_balance[1] * F.l1_loss(stu_relation_mat, tea_relation_mat, reduction="mean")

        return tea_global_feats, stu_global_feats, local_loss, global_loss

    def backbone_local_global_distillation_plus(self, voxel_num_points, tea_feats, stu_feats, num_samplings, num_neighbors, backbone_layer_idx, sg_layer_idx):
        """
        Args:
            voxel_num_points: [B, N]
            tea_feats: [B, tea_feat_dim, H, W]
            stu_feats: [B, stu_feat_dim, H, W]
            num_samplings: int
            num_neighbors: int
            backbone_layer_idx: int
            sg_layer_idx: int
        Returns:

        """
        if len(tea_feats.shape) == 4:
            B, C, H, W = tea_feats.shape
            # [B, C, H, W] -> [B, C, H*W] -> [B, H*W, C]
            tea_flatten_feats = tea_feats.view(B, tea_feats.shape[1], -1).transpose(2, 1).contiguous()
            stu_flatten_feats = stu_feats.view(B, stu_feats.shape[1], -1).transpose(2, 1).contiguous()
        elif len(tea_feats.shape) == 3:
            tea_flatten_feats, stu_flatten_feats = tea_feats, stu_feats
        else:
            raise NotImplementedError
        # aggegrate channel 
        if voxel_num_points is None:  
            # max pool
            voxel_num_points = F.adaptive_max_pool1d(tea_flatten_feats, 1).squeeze(-1)  # [B, H*W]
            # SENet
            # channel_agg_tea = tea_feats * self.channel_atten_list[backbone_layer_idx](F.adaptive_avg_pool2d(tea_feats, 1).reshape(B, C))[:, :, None, None]
            # voxel_num_points = F.adaptive_max_pool1d(channel_agg_tea.view(B, C, -1).transpose(2, 1).contiguous(), 1).squeeze(-1)
            
        # sampling
        _, query_voxel_id = torch.topk(voxel_num_points, num_samplings, dim=-1, largest=True, sorted=False)  # [B, num_samplings]
        sampling_voxel_info = torch.gather(voxel_num_points, 1, query_voxel_id)
        # KNN Grouping [B, D, num_samplings, K]
        tea_query_feats, tea_neighbor_feats = self.KNN_Grouping(query_voxel_id, tea_flatten_feats, num_neighbors)
        stu_query_feats, stu_neighbor_feats = self.KNN_Grouping(query_voxel_id, stu_flatten_feats, num_neighbors)
        
        # localKD
        tea_local_feats = torch.cat([tea_query_feats - tea_neighbor_feats, tea_query_feats], dim=1)  # [B, 2*tea_dim, num_samplings, K]
        stu_local_feats = torch.cat([stu_query_feats - stu_neighbor_feats, stu_query_feats], dim=1)   # [B, 2*stu_dim, num_samplings, K]
        tea_local_feats = self.tea_localconv_list[backbone_layer_idx](tea_local_feats)  # [B, tea_dim, num_sampling, K]
        tea_local_apt_feats = self.tea_adptconv_list[backbone_layer_idx](tea_local_feats)  # [B, stu_dim, num_sampling, K]
        stu_local_feats = self.stu_localconv_list[backbone_layer_idx](stu_local_feats)  # [B, stu_dim, num_sampling, K]
       
        # feat
        local_feat_loss = F.mse_loss(stu_local_feats, tea_local_apt_feats, reduction="mean")
        # relation
        stu_feat_dim, tea_feat_dim = stu_local_feats.shape[1], tea_local_feats.shape[1]
        local_relation_loss = self.cl_head_list[backbone_layer_idx](stu_local_feats.permute(0, 2, 3, 1).reshape(-1, stu_feat_dim), tea_local_feats.permute(0, 2, 3, 1).reshape(-1, tea_feat_dim))
        local_loss = self.loss_balance[0] * local_feat_loss + self.loss_balance[0] * local_relation_loss

        # aggregate   [B, stu_dim, num_sampling, K] -> [B, num_sampling, feat_dim] -> [B*num_sampling, feat_dim]
        tea_global_feats_batch = torch.mean(tea_local_feats, dim=-1).permute(0, 2, 1).contiguous()
        stu_global_feats_batch = torch.mean(stu_local_feats, dim=-1).permute(0, 2, 1).contiguous()
        # [B, num_sampling, feat_dim] -> [B*num_sampling, feat_dim]
        tea_global_feats = tea_global_feats_batch.reshape(-1, tea_feat_dim)  # [num_sampling, tea_vf] [num_sampling, stu_vf]
        stu_global_feats = stu_global_feats_batch.reshape(-1, stu_feat_dim)
        # relation matrix by computing cos  # [B*num_sampling, stu_feat_dim] [B*num_sampling, tea_feat_dim]
        tea_norm_global_feats = tea_global_feats / ((torch.norm(tea_global_feats, p=2, dim=-1, keepdim=True) + 1e-5))
        tea_relation_mat = torch.matmul(tea_norm_global_feats, tea_norm_global_feats.transpose(1, 0))

        stu_norm_global_feats = stu_global_feats / ((torch.norm(stu_global_feats, p=2, dim=-1, keepdim=True) + 1e-5))
        stu_relation_mat = torch.matmul(stu_norm_global_feats, stu_norm_global_feats.transpose(1, 0))

        global_loss = self.loss_balance[1] * F.l1_loss(stu_relation_mat, tea_relation_mat, reduction="mean")

        return sampling_voxel_info, tea_global_feats_batch, stu_global_feats_batch, local_loss, global_loss

    def voxel_local_global_distillation(self, voxel_num_points, tea_feats, stu_feats, num_samplings, num_neighbors, layer_idx):
        """
        Args:
            voxel_num_points:
            tea_feats: [num_sampling, tea_feat_dim]
            stu-feats: [num_sampling, stu_feat_dim]
            num_samplings:
            num_neighbors:
            layer_idx: 
        Returns:

        """
        # sampling
        _, query_voxel_id = torch.topk(voxel_num_points, num_samplings, dim=-1, largest=True, sorted=False)
        sampling_voxel_info = voxel_num_points[query_voxel_id]
        # KNN Grouping
        tea_query_feats, tea_neighbor_feats = self.KNN_Grouping(query_voxel_id, tea_feats, num_neighbors)
        stu_query_feats, stu_neighbor_feats = self.KNN_Grouping(query_voxel_id, stu_feats, num_neighbors)
        
        # localKD
        tea_local_feats = torch.cat([tea_query_feats - tea_neighbor_feats, tea_query_feats], dim=1)  # [1, 2*tea_vf, num_sampling, K]
        stu_local_feats = torch.cat([stu_query_feats - stu_neighbor_feats, stu_query_feats], dim=1)   # [1, 2*stu_vf, num_sampling, K]
        tea_local_feats = self.tea_localconv(tea_local_feats)  # [1, tea_vf, num_sampling, K]
        tea_local_apt_feats = self.tea_adptconv(tea_local_feats).squeeze(0).permute(1, 2, 0).contiguous()  # [num_sampling, K, stu_vf]
        stu_local_feats = self.stu_localconv(stu_local_feats).squeeze(0).permute(1, 2, 0).contiguous()  # [num_sampling, K, stu_vf]
        # feat
        local_feat_loss = F.mse_loss(stu_local_feats, tea_local_apt_feats, reduction="mean")
        # relation
        if layer_idx == (self.SG_layers - 1):
            local_relation_loss = self.cl_head(stu_local_feats.reshape(-1, stu_local_feats.shape[-1]), tea_local_feats.squeeze(0).permute(1, 2, 0).reshape(-1, tea_local_feats.shape[1]))
            local_loss = self.loss_balance[0] * local_feat_loss + self.loss_balance[0] * local_relation_loss
        else:
            local_loss = local_feat_loss
        
        # aggregate
        tea_global_feats, stu_global_feats = torch.mean(tea_local_feats.squeeze(0).permute(1, 2, 0), dim=1), torch.mean(stu_local_feats, dim=1)  # [num_sampling, tea_vf] [num_sampling, stu_vf]
        
        # relation matrix by computing cos
        tea_norm_global_feats = tea_global_feats / ((torch.norm(tea_global_feats, p=2, dim=-1, keepdim=True) + 1e-5))
        tea_relation_mat = torch.matmul(tea_norm_global_feats, tea_norm_global_feats.transpose(1, 0))

        stu_norm_global_feats = stu_global_feats / ((torch.norm(stu_global_feats, p=2, dim=-1, keepdim=True) + 1e-5))
        stu_relation_mat = torch.matmul(stu_norm_global_feats, stu_norm_global_feats.transpose(1, 0))

        global_loss = self.loss_balance[1] * F.l1_loss(stu_relation_mat, tea_relation_mat, reduction="mean")

        
        return sampling_voxel_info, tea_global_feats, stu_global_feats, local_loss, global_loss

    def HierRelationDistillation(self, tea_info_dict, stu_info_dict, num_samplings, num_neighbors, type):
        """
        Args:
            tea_info_dict: type=dict
            stu_info_dict: type=dict
            num_voxels: num voxels by sampling, type=list
            num_neighbors: K, type=list
            type: str, ["voxel", "backbone", "fpn"]
        Note:
            {}_info_dict:
                num_points: [B*num_voxels]
                voxel_features: [B*num_voxels, {}_vf]
                middle_encoder_features: 
                backbone_features: List[Tensor]
                neck_features: [B, feat_dim, H, W]
        """
        total_local_loss = torch.FloatTensor([0.0]).to(tea_info_dict["voxel_features"].device)
        total_global_loss = torch.FloatTensor([0.0]).to(tea_info_dict["voxel_features"].device)
        if type == "voxel": 
            # sampling voxel according the number of points in voxel
            voxel_num_points = tea_info_dict["num_points"]
            tea_voxel_feats, stu_voxel_feats = tea_info_dict["voxel_features"], stu_info_dict["voxel_features"]
            for i in range(self.SG_layers):
                voxel_num_points, tea_voxel_feats, stu_voxel_feats, local_loss, global_loss = self.voxel_local_global_distillation(voxel_num_points, 
                                                                                            tea_voxel_feats, 
                                                                                            stu_voxel_feats, 
                                                                                            num_samplings[i], 
                                                                                            num_neighbors[i],
                                                                                            i)
                total_local_loss += local_loss
                total_global_loss += global_loss
        elif type == "backbone":
            tea_backbone_feats = tea_info_dict["backbone_features"]
            stu_backbone_feats = stu_info_dict["backbone_features"]
            if self.local_cfg["sg_layers"]:
                for i in range(len(tea_backbone_feats)):
                    per_tea_backbone_feats = tea_backbone_feats[i]
                    per_stu_backbone_feats = stu_backbone_feats[i]
                    voxel_num_points = None
                    for j in range(self.local_cfg["sg_layers"]):
                        voxel_num_points, per_tea_backbone_feats, per_stu_backbone_feats, local_loss, global_loss = self.backbone_local_global_distillation_plus(voxel_num_points, 
                                                                                                                                    per_tea_backbone_feats, 
                                                                                                                                    per_stu_backbone_feats,
                                                                                                                                    num_samplings[i] // (2**j),
                                                                                                                                    num_neighbors[i] // (2**j),
                                                                                                                                    backbone_layer_idx=i,
                                                                                                                                    sg_layer_idx=j,
                                                                                                                                    )
                        total_local_loss += local_loss
                        total_global_loss += global_loss
                # avg
                total_local_loss /= ((i+1) * (j+1))
                total_global_loss /= ((i+1) * (j+1))
            else:
                for i in range(len(tea_backbone_feats)):
                    tea_voxel_feats, stu_voxel_feats, local_loss, global_loss = self.backbone_local_global_distillation(None,
                                                                                                tea_backbone_feats[i], 
                                                                                                stu_backbone_feats[i], 
                                                                                                num_samplings[i],
                                                                                                num_neighbors[i],
                                                                                                layer_idx=i,
                                                                                                )
                    total_local_loss += local_loss
                    total_global_loss += global_loss
        else:
            raise NotImplementedError
        
        return total_local_loss, total_global_loss

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img=None,
                      proposals=None,
                      teacher=None,
                      gt_bboxes_ignore=None):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor, optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """
        with torch.no_grad():
            teacher = teacher.to(points[0].device)
            _, __, tea_pts_infos = teacher.extract_feat(points, img=img, img_metas=img_metas, return_pts_middle=True)
        img_feats, pts_feats, stu_pts_infos = self.extract_feat(
            points, img=img, img_metas=img_metas, return_pts_middle=True)
        losses = dict()
        if pts_feats:
            losses_pts = self.forward_pts_train(pts_feats, gt_bboxes_3d,
                                                gt_labels_3d, img_metas,
                                                gt_bboxes_ignore)
            losses.update(losses_pts)
        if img_feats:
            losses_img = self.forward_img_train(
                img_feats,
                img_metas=img_metas,
                gt_bboxes=gt_bboxes,
                gt_labels=gt_labels,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposals=proposals)
            losses.update(losses_img)
        
        # HRKD
        loss_local, loss_global = self.HierRelationDistillation(tea_pts_infos, 
                                                                stu_pts_infos, 
                                                                num_samplings=self.local_cfg["num_samplings"], 
                                                                num_neighbors=self.local_cfg["num_neighbors"],
                                                                type=self.kd_type)
        losses["loss_local"] = loss_local
        losses["loss_global"] = loss_global
        
        return losses    

    def train_step(self, data, optimizer, teacher):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``, \
                ``num_samples``.

                - ``loss`` is a tensor for back propagation, which can be a
                  weighted sum of multiple losses.
                - ``log_vars`` contains all the variables to be sent to the
                  logger.
                - ``num_samples`` indicates the batch size (when the model is
                  DDP, it means the batch size on each GPU), which is used for
                  averaging the logs.
        """
        data["teacher"] = teacher
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img_metas']))

        return outputs


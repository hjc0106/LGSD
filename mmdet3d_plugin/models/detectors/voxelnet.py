# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from torch.nn import functional as F

from mmcv.ops import Voxelization
from mmcv.runner import force_fp32
from mmcv.cnn import ConvModule

from mmdet.core import multi_apply

from mmdet3d.core import bbox3d2result, merge_aug_bboxes_3d
from ..builder import DETECTORS, build_voxel_encoder, build_middle_encoder, build_cl_head
from .single_stage import SingleStage3DDetector
from mmdet3d.models.utils import knn_feature, index_feature, pool_features, pool_features1d
from mmdet3d.models.utils import knn_query, index_points

@DETECTORS.register_module()
class VoxelNet(SingleStage3DDetector):
    r"""`VoxelNet <https://arxiv.org/abs/1711.06396>`_ for 3D detection."""

    def __init__(self,
                 voxel_layer,
                 voxel_encoder,
                 middle_encoder,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 pretrained=None):
        super(VoxelNet, self).__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg,
            pretrained=pretrained)
        self.voxel_layer = Voxelization(**voxel_layer)
        self.voxel_encoder = build_voxel_encoder(voxel_encoder)
        self.middle_encoder = build_middle_encoder(middle_encoder)
    
    # overwrite
    def extract_feat(self, points, img_metas=None, return_middle=False):
        """Extract features from points."""
        if return_middle:
            info_dict = dict()
            voxels, num_points, coors = self.voxelize(points)
            info_dict["num_points"] = num_points
            voxel_features = self.voxel_encoder(voxels, num_points, coors)
            info_dict["voxel_features"] = voxel_features
            batch_size = coors[-1, 0].item() + 1
            x = self.middle_encoder(voxel_features, coors, batch_size)
            info_dict["middle_encoder_features"] = x
            x = self.backbone(x)
            info_dict["backbone_features"] = x
            if self.with_neck:
                x = self.neck(x)
                info_dict["neck_features"] = x
            return x, info_dict
        else:
            voxels, num_points, coors = self.voxelize(points)
            voxel_features = self.voxel_encoder(voxels, num_points, coors)
            batch_size = coors[-1, 0].item() + 1
            x = self.middle_encoder(voxel_features, coors, batch_size)
            x = self.backbone(x)
            if self.with_neck:
                x = self.neck(x)
            return x           

    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points):
        """Apply hard voxelization to points."""
        voxels, coors, num_points = [], [], []
        for res in points:
            res_voxels, res_coors, res_num_points = self.voxel_layer(res)
            voxels.append(res_voxels)
            coors.append(res_coors)
            num_points.append(res_num_points)
        voxels = torch.cat(voxels, dim=0)
        num_points = torch.cat(num_points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        return voxels, num_points, coors_batch

    def forward_train(self,
                      points,
                      img_metas,
                      gt_bboxes_3d,
                      gt_labels_3d,
                      gt_bboxes_ignore=None):
        """Training forward function.

        Args:
            points (list[torch.Tensor]): Point cloud of each sample.
            img_metas (list[dict]): Meta information of each sample
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.

        Returns:
            dict: Losses of each branch.
        """
        x = self.extract_feat(points, img_metas)
        outs = self.bbox_head(x)
        loss_inputs = outs + (gt_bboxes_3d, gt_labels_3d, img_metas)
        losses = self.bbox_head.loss(
            *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        return losses

    def simple_test(self, points, img_metas, imgs=None, rescale=False):
        """Test function without augmentaiton."""
        x = self.extract_feat(points, img_metas)
        outs = self.bbox_head(x)
        bbox_list = self.bbox_head.get_bboxes(
            *outs, img_metas, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return bbox_results

    def aug_test(self, points, img_metas, imgs=None, rescale=False):
        """Test function with augmentaiton."""
        feats = self.extract_feats(points, img_metas)

        # only support aug_test for one sample
        aug_bboxes = []
        for x, img_meta in zip(feats, img_metas):
            outs = self.bbox_head(x)
            bbox_list = self.bbox_head.get_bboxes(
                *outs, img_meta, rescale=rescale)
            bbox_list = [
                dict(boxes_3d=bboxes, scores_3d=scores, labels_3d=labels)
                for bboxes, scores, labels in bbox_list
            ]
            aug_bboxes.append(bbox_list[0])

        # after merging, bboxes will be rescaled to the original image size
        merged_bboxes = merge_aug_bboxes_3d(aug_bboxes, img_metas,
                                            self.bbox_head.test_cfg)

        return [merged_bboxes]

# Baselines
@DETECTORS.register_module()
class LogitKDVoxelNet(VoxelNet):
    def __init__(self,
                 distillation, 
                 voxel_layer, 
                 voxel_encoder, 
                 middle_encoder, 
                 backbone, 
                 neck=None, 
                 bbox_head=None, 
                 train_cfg=None, 
                 test_cfg=None, 
                 init_cfg=None, 
                 pretrained=None):
        bbox_head["loss_balance"] = distillation["loss_balance"]
        bbox_head["temperature"] = distillation["temperature"]
        super(LogitKDVoxelNet, self).__init__(voxel_layer, voxel_encoder, middle_encoder, backbone, neck, bbox_head, train_cfg, test_cfg, init_cfg, pretrained)

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

    def forward_train(self,
                      points,
                      img_metas,
                      gt_bboxes_3d,
                      gt_labels_3d,
                      teacher=None,
                      gt_bboxes_ignore=None):
        """Training forward function.

        Args:
            points (list[torch.Tensor]): Point cloud of each sample.
            img_metas (list[dict]): Meta information of each sample
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.

        Returns:
            dict: Losses of each branch.
        """
        with torch.no_grad():
            teacher = teacher.to(points[0].device)
            tea_neck_feats = teacher.extract_feat(points, img_metas, return_middle=False)
            tea_cls_scores = teacher.bbox_head(tea_neck_feats)[0]
        x, stu_info = self.extract_feat(points, img_metas, return_middle=True)
        outs = self.bbox_head(x)
        loss_inputs = outs + (tea_cls_scores, gt_bboxes_3d, gt_labels_3d, img_metas)
        losses = self.bbox_head.loss(
            *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        
        return losses

@DETECTORS.register_module()
class FeatKDVoxelNet(VoxelNet):
    def __init__(self,
                 distillation, 
                 voxel_layer, 
                 voxel_encoder, 
                 middle_encoder, 
                 backbone, 
                 neck=None, 
                 bbox_head=None, 
                 train_cfg=None, 
                 test_cfg=None, 
                 init_cfg=None, 
                 pretrained=None):
        super(FeatKDVoxelNet, self).__init__(voxel_layer, voxel_encoder, middle_encoder, backbone, neck, bbox_head, train_cfg, test_cfg, init_cfg, pretrained)
        self.distillation = distillation
        self.loss_balance = distillation["loss_balance"]
        self.num_levels = distillation["num_levels"]
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

    def forward_train(self,
                      points,
                      img_metas,
                      gt_bboxes_3d,
                      gt_labels_3d,
                      teacher=None,
                      gt_bboxes_ignore=None):
        """Training forward function.

        Args:
            points (list[torch.Tensor]): Point cloud of each sample.
            img_metas (list[dict]): Meta information of each sample
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.

        Returns:
            dict: Losses of each branch.
        """
        with torch.no_grad():
            teacher = teacher.to(points[0].device)
            tea_neck_feats, tea_info = teacher.extract_feat(points, img_metas, return_middle=True)
        x, stu_info = self.extract_feat(points, img_metas, return_middle=True)
        outs = self.bbox_head(x)
        loss_inputs = outs + (gt_bboxes_3d, gt_labels_3d, img_metas)
        losses = self.bbox_head.loss(
            *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        
        # Feat KD
        level_list=list(range(self.num_levels))
        loss_kd = multi_apply(
            self.feat_distillation,
            stu_info["backbone_features"],
            tea_info["backbone_features"],
            level_list,
        )[0]
        losses["loss_kd"] = loss_kd
        
        return losses

@DETECTORS.register_module()
class PAKDVoxelNet(VoxelNet):
    def __init__(self,
                 distillation, 
                 voxel_layer, 
                 voxel_encoder, 
                 middle_encoder, 
                 backbone, 
                 neck=None, 
                 bbox_head=None, 
                 train_cfg=None, 
                 test_cfg=None, 
                 init_cfg=None, 
                 pretrained=None):
        bbox_head["loss_balance"] = distillation["loss_balance"][1]
        bbox_head["temperature"] = distillation["temperature"]
        super(PAKDVoxelNet, self).__init__(voxel_layer, voxel_encoder, middle_encoder, backbone, neck, bbox_head, train_cfg, test_cfg, init_cfg, pretrained)
        self.loss_balance = distillation["loss_balance"][0]

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

    def forward_train(self,
                      points,
                      img_metas,
                      gt_bboxes_3d,
                      gt_labels_3d,
                      teacher=None,
                      gt_bboxes_ignore=None):
        """Training forward function.

        Args:
            points (list[torch.Tensor]): Point cloud of each sample.
            img_metas (list[dict]): Meta information of each sample
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.

        Returns:
            dict: Losses of each branch.
        """
        with torch.no_grad():
            teacher = teacher.to(points[0].device)
            tea_neck_feats, tea_info = teacher.extract_feat(points, img_metas, return_middle=True)
            tea_cls_scores = teacher.bbox_head(tea_neck_feats)[0]
        x, stu_info = self.extract_feat(points, img_metas, return_middle=True)
        outs = self.bbox_head(x)
        loss_inputs = outs + (tea_cls_scores, gt_bboxes_3d, gt_labels_3d, img_metas)
        losses = self.bbox_head.loss(
            *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        
        # PAKD
        loss_kd = multi_apply(
            self.payattention_distillation,
            stu_info["backbone_features"],
            tea_info["backbone_features"],
        )[0]
        losses["loss_pakd"] = loss_kd
        
        return losses

@DETECTORS.register_module()
class SPKDVoxelNet(VoxelNet):
    def __init__(self,
                 distillation, 
                 voxel_layer, 
                 voxel_encoder, 
                 middle_encoder, 
                 backbone, 
                 neck=None, 
                 bbox_head=None, 
                 train_cfg=None, 
                 test_cfg=None, 
                 init_cfg=None, 
                 pretrained=None):
        super(SPKDVoxelNet, self).__init__(voxel_layer, voxel_encoder, middle_encoder, backbone, neck, bbox_head, train_cfg, test_cfg, init_cfg, pretrained)
        self.loss_balance = distillation["loss_balance"]
        self.num_levels = distillation["num_levels"]
        # To align feature dim between student and teacher 32, 64, 128
        self.stu_adapt_list = nn.ModuleList()
        for i in range(self.num_levels):
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

    def forward_train(self,
                      points,
                      img_metas,
                      gt_bboxes_3d,
                      gt_labels_3d,
                      teacher=None,
                      gt_bboxes_ignore=None):
        """Training forward function.

        Args:
            points (list[torch.Tensor]): Point cloud of each sample.
            img_metas (list[dict]): Meta information of each sample
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.

        Returns:
            dict: Losses of each branch.
        """
        with torch.no_grad():
            teacher = teacher.to(points[0].device)
            tea_neck_feats, tea_info = teacher.extract_feat(points, img_metas, return_middle=True)
        x, stu_info = self.extract_feat(points, img_metas, return_middle=True)
        outs = self.bbox_head(x)
        loss_inputs = outs + (gt_bboxes_3d, gt_labels_3d, img_metas)
        losses = self.bbox_head.loss(
            *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        
        # SPKD
        level_list=list(range(self.num_levels))
        loss_kd = multi_apply(
            self.sp_distillation,
            stu_info["backbone_features"][1:],
            tea_info["backbone_features"][1:],
            level_list[1:],
        )[0]
        losses["loss_kd"] = loss_kd
        
        return losses

class NonLocalBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=2, sub_sample=True, bn_layer=True, downsample_stride=2):
        super(NonLocalBlockND, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(downsample_stride, downsample_stride))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x):
        '''
        :param x: (b, c, t, h, w)
        :
        :
        '''

        batch_size = x.size(0)  #   2 , 256 , 300 , 300

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)   #   2 , 128 , 150 x 150
        g_x = g_x.permute(0, 2, 1)                                  #   2 , 150 x 150, 128

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)   #   2 , 128 , 300 x 300
        theta_x = theta_x.permute(0, 2, 1)                                  #   2 , 300 x 300 , 128
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)       #   2 , 128 , 150 x 150
        f = torch.matmul(theta_x, phi_x)    #   2 , 300x300 , 150x150
        N = f.size(-1)  #   150 x 150
        f_div_C = f / N #   2 , 300x300, 150x150

        y = torch.matmul(f_div_C, g_x)  #   2, 300x300, 128
        y = y.permute(0, 2, 1).contiguous() #   2, 128, 300x300
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z

@DETECTORS.register_module()
class TAEDVoxelNet(VoxelNet):
    def __init__(self,
                 distillation,
                 **kwargs):
        super(TAEDVoxelNet, self).__init__(**kwargs)
        self.distillation = distillation
        self.kd_type = distillation["kd_type"]
        self.adaptation_type = distillation["adaptation_type"]
        self.build_layers()

    def build_layers(self):
        tea_feat_dim, stu_feat_dim = self.distillation["tea_feat_dim"], self.distillation["stu_feat_dim"]
        num_layers = len(tea_feat_dim)
        #   self.roi_adaptation_layer = nn.Conv2d(256, 256, kernel_size=1)
        if self.adaptation_type == '3x3conv':
            #   3x3 conv
            self.adaptation_layers = nn.ModuleList([
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            ])
        if self.adaptation_type == '1x1conv':
            #  1x1 conv
            self.adaptation_layers = nn.ModuleList()
            self.c_attention_adaptation = nn.ModuleList()
            self.channel_wise_adaptation = nn.ModuleList()
            self.spatial_wise_adaptation = nn.ModuleList()
            self.student_non_local = nn.ModuleList()
            self.teacher_non_local = nn.ModuleList()
            self.non_local_adaptation = nn.ModuleList()
            for idx, (stu_dim, tea_dim) in enumerate(zip(stu_feat_dim, tea_feat_dim)):
                self.c_attention_adaptation.append(
                    nn.Sequential(
                        nn.Conv2d(
                            stu_dim,
                            tea_dim,
                            kernel_size=1,
                            stride=1,
                            padding=0
                        ),
                        nn.ReLU(),
                    )
                )
                self.channel_wise_adaptation.append(
                    nn.Linear(stu_dim, tea_dim)
                )
                self.spatial_wise_adaptation.append(
                    nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1),
                )
                self.adaptation_layers.append(
                    nn.Conv2d(
                        stu_dim,
                        tea_dim,
                        kernel_size=1,
                        stride=1,
                        padding=0
                    )
                )
                self.non_local_adaptation.append(
                    nn.Conv2d(stu_dim, 
                              tea_dim, 
                              kernel_size=1,
                              stride=1,
                              padding=0)
                )
                if idx == num_layers - 1:
                    self.student_non_local.append(
                        NonLocalBlockND(in_channels=stu_dim)
                    )
                    self.teacher_non_local.append(
                        NonLocalBlockND(in_channels=tea_dim)
                    )
                else:
                    self.student_non_local.append(
                        NonLocalBlockND(in_channels=stu_dim, inter_channels=64, downsample_stride=8//(2**idx))
                    )
                    self.teacher_non_local.append(
                        NonLocalBlockND(in_channels=tea_dim, inter_channels=64, downsample_stride=8//(2**idx))
                    )

        if self.adaptation_type == '3x3conv+bn':
            #   3x3 conv + bn
            self.adaptation_layers = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(256, affine=True)
                ),
                nn.Sequential(
                    nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(256, affine=True)
                ),
                nn.Sequential(
                    nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(256, affine=True)
                ),
                nn.Sequential(
                    nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(256, affine=True)
                ),
                nn.Sequential(
                    nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(256, affine=True)
                )
            ])
        if self.adaptation_type == '1x1conv+bn':
            #   1x1 conv + bn
            self.adaptation_layers = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
                    nn.BatchNorm2d(256, affine=True)
                ),
                nn.Sequential(
                    nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
                    nn.BatchNorm2d(256, affine=True)
                ),
                nn.Sequential(
                    nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
                    nn.BatchNorm2d(256, affine=True)
                ),
                nn.Sequential(
                    nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
                    nn.BatchNorm2d(256, affine=True)
                ),
                nn.Sequential(
                    nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
                    nn.BatchNorm2d(256, affine=True)
                )
            ])
    

    def dist2(self, tensor_a, tensor_b, attention_mask=None, channel_attention_mask=None):
        diff = (tensor_a - tensor_b) ** 2
        #   print(diff.size())      batchsize x 1 x W x H,
        #   print(attention_mask.size()) batchsize x 1 x W x H
        diff = diff * attention_mask
        diff = diff * channel_attention_mask
        diff = torch.sum(diff) ** 0.5
        return diff   
         
    def forward_train(self,
                      points,
                      img_metas,
                      gt_bboxes_3d,
                      gt_labels_3d,
                      teacher=None,
                      gt_bboxes_ignore=None):
        """Training forward function.

        Args:
            points (list[torch.Tensor]): Point cloud of each sample.
            img_metas (list[dict]): Meta information of each sample
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.

        Returns:
            dict: Losses of each branch.
        """
        with torch.no_grad():
            teacher = teacher.to(points[0].device)
            tea_neck_feats, tea_info = teacher.extract_feat(points, img_metas, return_middle=True)
        x, stu_info = self.extract_feat(points, img_metas, return_middle=True)
        outs = self.bbox_head(x)
        loss_inputs = outs + (gt_bboxes_3d, gt_labels_3d, img_metas)
        losses = self.bbox_head.loss(
            *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)

        # TAED
        t, s_ratio, c_t, c_s_ratio = list(self.distillation["hyper_cfg"].values())
        
        kd_feat_loss = torch.FloatTensor([0.0]).to(losses["loss_cls"][0].device)
        kd_channel_loss = torch.FloatTensor([0.0]).to(losses["loss_cls"][0].device)
        kd_spatial_loss = torch.FloatTensor([0.0]).to(losses["loss_cls"][0].device)
        kd_nonlocal_loss = torch.FloatTensor([0.0]).to(losses["loss_cls"][0].device)

        if tea_info is not None:
            if self.kd_type == "backbone":
                tea_feats = tea_info["backbone_features"]
                stu_feats = stu_info["backbone_features"]
                for _i in range(len(tea_feats)):
                    t_attention_mask = torch.mean(torch.abs(tea_feats[_i]), [1], keepdim=True)
                    size = t_attention_mask.size()
                    t_attention_mask = t_attention_mask.view(stu_feats[0].size(0), -1)
                    t_attention_mask = torch.softmax(t_attention_mask / t, dim=1) * size[-1] * size[-2]
                    t_attention_mask = t_attention_mask.view(size)

                    s_attention_mask = torch.mean(torch.abs(stu_feats[_i]), [1], keepdim=True)
                    size = s_attention_mask.size()
                    s_attention_mask = s_attention_mask.view(stu_feats[0].size(0), -1)
                    s_attention_mask = torch.softmax(s_attention_mask / t, dim=1) * size[-1] * size[-2]
                    s_attention_mask = s_attention_mask.view(size)

                    c_t_attention_mask = torch.mean(torch.abs(tea_feats[_i]), [2, 3], keepdim=True)  # 2 x 256 x 1 x1
                    c_size = c_t_attention_mask.size()
                    c_t_attention_mask = c_t_attention_mask.view(stu_feats[0].size(0), -1)  # 2 x 256
                    c_t_attention_mask = torch.softmax(c_t_attention_mask / c_t, dim=1) * c_size[1]
                    c_t_attention_mask = c_t_attention_mask.view(c_size)  # 2 x 256 -> 2 x 256 x 1 x 1

                    c_s_attention_mask = torch.mean(torch.abs(stu_feats[_i]), [2, 3], keepdim=True)  # 2 x 256 x 1 x1
                    c_size = c_s_attention_mask.size()
                    c_s_attention_mask = c_s_attention_mask.view(stu_feats[0].size(0), -1)  # 2 x 256
                    c_s_attention_mask = torch.softmax(c_s_attention_mask / c_t, dim=1) * c_size[1]
                    c_s_attention_mask = c_s_attention_mask.view(c_size)  # 2 x 256 -> 2 x 256 x 1 x 1

                    sum_attention_mask = (t_attention_mask + s_attention_mask * s_ratio) / (1 + s_ratio)
                    sum_attention_mask = sum_attention_mask.detach()

                    c_sum_attention_mask = (c_t_attention_mask + self.c_attention_adaptation[_i](c_s_attention_mask) * c_s_ratio) / (1 + c_s_ratio)
                    # c_sum_attention_mask = c_sum_attention_mask.detach()

                    kd_feat_loss += self.dist2(tea_feats[_i], self.adaptation_layers[_i](stu_feats[_i]), attention_mask=sum_attention_mask,
                                        channel_attention_mask=c_sum_attention_mask) * 7e-5 * 6
                    kd_channel_loss += torch.dist(torch.mean(tea_feats[_i], [2, 3]),
                                                self.channel_wise_adaptation[_i](torch.mean(stu_feats[_i], [2, 3]))) * 4e-3 * 6
                    t_spatial_pool = torch.mean(tea_feats[_i], [1]).view(tea_feats[_i].size(0), 1, tea_feats[_i].size(2),
                                                                    tea_feats[_i].size(3))
                    s_spatial_pool = torch.mean(stu_feats[_i], [1]).view(stu_feats[_i].size(0), 1, stu_feats[_i].size(2),
                                                                stu_feats[_i].size(3))
                    kd_spatial_loss += torch.dist(t_spatial_pool, self.spatial_wise_adaptation[_i](s_spatial_pool)) * 4e-3 * 6                  

                    # nonlocal 
                    s_relation = self.student_non_local[_i](stu_feats[_i])
                    t_relation = self.teacher_non_local[_i](tea_feats[_i])
                    kd_nonlocal_loss += torch.dist(self.non_local_adaptation[_i](s_relation), t_relation, p=2)
            else:
                raise NotImplementedError
        
        losses.update({'kd_feat_loss': kd_feat_loss})
        losses.update({'kd_channel_loss': kd_channel_loss})
        losses.update({'kd_spatial_loss': kd_spatial_loss}) 
        losses.update({'kd_nonlocal_loss': kd_nonlocal_loss * 7e-5 * 6})          

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
class PDistillerVoxelNet(VoxelNet):
    def __init__(self,
                 distillation, 
                 voxel_layer, 
                 voxel_encoder, 
                 middle_encoder, 
                 backbone, 
                 neck=None, 
                 bbox_head=None, 
                 train_cfg=None, 
                 test_cfg=None, 
                 init_cfg=None, 
                 pretrained=None):
        super(PDistillerVoxelNet, self).__init__(voxel_layer, voxel_encoder, middle_encoder, backbone, neck, bbox_head, train_cfg, test_cfg, init_cfg, pretrained)
        self.distillation = distillation
        self.num_voxels = distillation["num_voxels"]
        self.kneighbours = distillation["kneighbours"]
        self.temperature = distillation["temperature"]
        self.gcn_layers_init()

    def gcn_layers_init(self):
        if self.distillation["type"] == "PointPillars":
            tea_gcn_cfg, stu_gcn_cfg = self.distillation["tea_gcn_cfg"], self.distillation["stu_gcn_cfg"]
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
        elif self.distillation["type"] == "SECOND":
            tea_gcn_cfg, stu_gcn_cfg = self.distillation["tea_gcn_cfg"], self.distillation["stu_gcn_cfg"]
            self.tea_gcn_adapt_list = nn.ModuleList()
            self.stu_gcn_adapt_list = nn.ModuleList()
            for i in range(self.distillation["num_layers"]):
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
        else:
            raise NotImplementedError       
            
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
        dist = torch.FloatTensor([0.0]).to(grouped_voxels_tea.device)
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
        teacher_feature, student_feature = teacher_feature[layer_idx], student_feature[layer_idx]  # [B, C, H, W]
        batch_size = teacher_feature.shape[0]
        # B C H W -> B (H*W) C: batch_size x nvoxels x channel_tea [B, H*W, C]
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
        dist = torch.FloatTensor([0.0]).to(grouped_voxels_tea.device)
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

    def forward_train(self,
                      points,
                      img_metas,
                      gt_bboxes_3d,
                      gt_labels_3d,
                      teacher=None,
                      gt_bboxes_ignore=None):
        """Training forward function.

        Args:
            points (list[torch.Tensor]): Point cloud of each sample.
            img_metas (list[dict]): Meta information of each sample
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.

        Returns:
            dict: Losses of each branch.
        """
        with torch.no_grad():
            teacher = teacher.to(points[0].device)
            _, tea_info = teacher.extract_feat(points, img_metas, return_middle=True)
        x, stu_info = self.extract_feat(points, img_metas, return_middle=True)
        outs = self.bbox_head(x)
        loss_inputs = outs + (gt_bboxes_3d, gt_labels_3d, img_metas)
        losses = self.bbox_head.loss(
            *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        
        # PointDistiller KD
        if tea_info is not None:
            if self.distillation["type"] == "PointPillars":
                loss_kd = torch.FloatTensor([0.0]).to(losses["loss_cls"][0].device)
                loss_kd += self.get_voxel_knn(
                    teacher_feature=tea_info['voxel_features'], student_feature=stu_info['voxel_features'],
                    num_voxel_points=tea_info['num_points'], num_voxels=self.num_voxels, kneighbours=self.kneighbours, 
                    reweight=True, sample_mode='top', fuse_mode='idendity'
                )
            elif self.distillation["type"] == "SECOND":
                loss_kd = torch.FloatTensor([0.0]).to(losses["loss_cls"][0].device)
                for i in range(self.distillation["num_layers"]):
                    loss_kd += self.get_knn_voxel_backbone(
                        teacher_feature=tea_info['backbone_features'], student_feature=stu_info['backbone_features'],
                        layer_idx=i, num_voxel_points=None, num_voxels=self.num_voxels, kneighbours=self.kneighbours, 
                        reweight=True, sample_mode='top', fuse_mode='idendity'
                    )                
         
            else:
                raise NotImplementedError
        losses["loss_kd"] = loss_kd
        
        return losses

# LGSD
@DETECTORS.register_module()
class LGSDVoxelNet(VoxelNet):
    def __init__(self,
                 distillation, 
                 voxel_layer, 
                 voxel_encoder, 
                 middle_encoder, 
                 backbone, 
                 neck=None, 
                 bbox_head=None, 
                 train_cfg=None, 
                 test_cfg=None, 
                 init_cfg=None, 
                 pretrained=None):
        super(LGSDVoxelNet, self).__init__(voxel_layer, voxel_encoder, middle_encoder, backbone, neck, bbox_head, train_cfg, test_cfg, init_cfg, pretrained)
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
                self.channel_atten_list.append(
                    nn.Sequential(
                        nn.Linear(tea_in_channels[i], tea_in_channels[i] // 4, bias=False),
                        nn.ReLU(inplace=True),
                        nn.Linear(tea_in_channels[i] // 4, tea_in_channels[i], bias=False),
                        nn.Sigmoid(),
                    )
                )
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
            # voxel_num_points = F.adaptive_max_pool1d(tea_flatten_feats, 1).squeeze(-1)  # [B, H*W]
            # SENet
            channel_agg_tea = tea_feats * self.channel_atten_list[backbone_layer_idx](F.adaptive_avg_pool2d(tea_feats, 1).reshape(B, C))[:, :, None, None]
            voxel_num_points = F.adaptive_max_pool1d(channel_agg_tea.view(B, C, -1).transpose(2, 1).contiguous(), 1).squeeze(-1)
            
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
        # local_loss = self.loss_balance[0] * local_feat_loss
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
                      points,
                      img_metas,
                      gt_bboxes_3d,
                      gt_labels_3d,
                      teacher=None,
                      gt_bboxes_ignore=None):
        """Training forward function.

        Args:
            points (list[torch.Tensor]): Point cloud of each sample.
            img_metas (list[dict]): Meta information of each sample
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.

        Returns:
            dict: Losses of each branch.
        """
        with torch.no_grad():
            teacher = teacher.to(points[0].device)
            _, tea_info = teacher.extract_feat(points, img_metas, return_middle=True)
        x, stu_info = self.extract_feat(points, img_metas, return_middle=True)
        outs = self.bbox_head(x)
        loss_inputs = outs + (gt_bboxes_3d, gt_labels_3d, img_metas)
        losses = self.bbox_head.loss(
            *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        
        # LGSD
        loss_local, loss_global = self.HierRelationDistillation(tea_info, 
                                                                stu_info, 
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

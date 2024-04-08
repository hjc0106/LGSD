import torch
from mmcv.cnn import ConvModule, xavier_init
from torch import nn as nn
from torch.nn import functional as F
from mmcv.runner import BaseModule
from ..builder import CL_HEADS
from mmdet3d.models.builder import build_loss

@CL_HEADS.register_module()
class MoCoCLHead(BaseModule):
    def __init__(self,
                 tea_channels,
                 stu_channels,
                 mid_channels=128,
                 stu_proj_num=1,
                 tea_proj_num=1,
                 T=0.07,
                 loss_cl=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            ):
        super().__init__()
        stu_projs = []; tea_projs = []
        tea_input_channels = tea_channels
        stu_input_channels = stu_channels
        for ii in range(stu_proj_num):
            stu_proj = nn.Sequential(
                nn.Linear(stu_input_channels, mid_channels),
                # nn.BatchNorm1d(mid_channels),
                # nn.ReLU(inplace=True)
            )
            stu_input_channels = mid_channels
            stu_projs.append(stu_proj)
        for ii in range(tea_proj_num):
            tea_proj = nn.Sequential(
                nn.Linear(tea_input_channels, mid_channels),
                # nn.BatchNorm1d(mid_channels),
                # nn.ReLU(inplace=True)
            )
            tea_input_channels = mid_channels
            tea_projs.append(tea_proj)
        self.stu_projs = nn.ModuleList(stu_projs)
        self.tea_projs = nn.ModuleList(tea_projs)
        # 2 layer mlp encoder
        self.encoder_stu = nn.Sequential(
            nn.Linear(mid_channels, mid_channels), 
            nn.BatchNorm1d(mid_channels), 
            nn.ReLU(inplace=True),
            nn.Linear(mid_channels, mid_channels), 
        )
        self.encoder_tea = nn.Sequential(
            nn.Linear(mid_channels, mid_channels), 
            nn.BatchNorm1d(mid_channels), 
            nn.ReLU(inplace=True),
            nn.Linear(mid_channels, mid_channels),
        ) 
        self.mid_channels = mid_channels
        self.T = T
        self.loss_cl = build_loss(loss_cl)

    # @force_fp32(apply_to=('logits', 'labels'))
    # def loss(self, logits, labels):
    #     loss_cl = self.loss_cl(logits, labels)
    #     return loss_cl

    def forward(self, stu_feats, tea_feats):
        """
            Args:
                stu_feats: tensor, shape=[N, Stu_C]
                tea_feats: tensor, shape=[N, Tea_C]
        """
        for stu_proj in self.stu_projs:
            stu_feats = stu_proj(stu_feats)
        for tea_proj in self.tea_projs:
            tea_feats = tea_proj(tea_feats)

        stu_feats = self.encoder_stu(stu_feats)
        stu_feats = torch.nn.functional.normalize(stu_feats, dim=1)

        tea_feats = self.encoder_tea(tea_feats)
        tea_feats = torch.nn.functional.normalize(tea_feats, dim=1)

        # compute logits
        # Einstein sum is more intuitive
        logits = torch.einsum('nc,ck->nk', [tea_feats, stu_feats.T])
        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.arange(logits.shape[0]).cuda(logits.device)
        loss_cl = self.loss_cl(logits, labels)
        return loss_cl
# LGSD based on MMDetection3d

## 1: Install mmdetection3d

Install mmdetection3d based [instruction](https://mmdetection3d.readthedocs.io/en/latest/get_started.html).

- torch==1.13.0+cu117
- torchvision==0.14.0+cu117
- mmcv-full==1.6.0
- mmdet==2.26.0
- mmsegmentation==0.29.1
- mmdet3d==1.00rc3
- numpy==1.23.5
- yapf==0.40.1

## 2: Add KD training to mmdetection3d

- Replace their codes with our codes, including mmcv/runner/epoch_based_runner.py, mmdet3d/apis, mmdet3d/models
- Add our configs/distillation to configs from mmdetection3d
- Add our tools to tools from mmdetection3d

## 3: Teacher Preparation

Download the official teacher models from Openmmlab or train the teacher models.

```python
# Download PointPillars
wget https://download.openmmlab.com/mmdetection3d/v1.0.0_models/pointpillars/hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d/hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d_20210826_104936-fca299c1.pth
# Train CenterPoints
python tools/train.py configs/distillation/centerpoint_01voxel_second_secfpn_4x8_cyclic_20e_nus.py
```

Put it in teacher_checkpoints. You need to create this folder by yourself.

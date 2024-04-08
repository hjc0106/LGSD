# Local and Global Structural Knowledge Distillation
![framework.png](./framework.png)

# Environment

This codebase was tested with the following environment configurations. It may work with other versions.

- CUDA 11.3/CUDA 11.7
- Python 3.8+
- PyTorch 1.9.0/PyTorch 1.13.0
- MMCV 1.6.0
- MMDetection 2.26.0
- MMSegmentation 0.29.1
- MMRotate 0.3.4
- MMDetection3d 1.0.0rc3

# Installation
Please refer to [getting_started.md](./getting_started.md) for installation.

# Datasets

We use [KITTI](https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) and [nuScenes](https://www.nuscenes.org/) datasets, please follow the official instructions for set up ([KITTI instruction](https://mmdetection3d.readthedocs.io/en/latest/advanced_guides/datasets/kitti.html), [nuScenes instruction](https://mmdetection3d.readthedocs.io/en/latest/advanced_guides/datasets/nuscenes.html)).

# Run

Please make sure you have set up the environments and you can start knowledge distillation by running

```python
DEVICE_ID = {gpu_id}
# for single gpu
CUDA_VISIBLE_DEVICES=$DEVICE_ID python tools/train_kd.py {distillation_cfg}
# for multiple gpus
bash ./tools/dist_train_kd.sh <distillation_cfg> 8 
```

# Acknowledgements

Many thanks to following codes that help us a lot in building this codebase:

- [PointDistiller](https://github.com/RunpeiDong/PointDistiller)
- [Towards Efficient 3D Object Detection with Knowledge Distillation](https://github.com/CVMI-Lab/SparseKD)
- [mmdetection](https://github.com/open-mmlab/mmdetection)
- [mmdetection3d](https://github.com/open-mmlab/mmdetection3d)

# Citation

If you find our work useful in your research, please consider citing:

```python
To be continued
```

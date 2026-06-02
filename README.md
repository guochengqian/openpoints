# OpenPoints

OpenPoints is a library built for fairly benchmarking and easily reproducing point-based methods for point cloud understanding. It is born in the course of [PointNeXt](https://github.com/guochengqian/PointNeXt) project and is used as an engine therein.

**For any question related to OpenPoints, please open an issue in [PointNeXt](https://github.com/guochengqian/PointNeXt) repo.**

OpenPoints currently supports reproducing the following models:
- PointNet
- DGCNN
- DeepGCN
- PointNet++
- ASSANet
- PointMLP
- PointNeXt
- Pix4Point
- PointVector



## Features

1. **Extensibility**: supports many representative networks for point cloud understanding, such as *PointNet, DGCNN, DeepGCN, PointNet++, ASSANet, PointMLP*, and our ***PointNeXt***. More networks can be built easily based on our framework since **OpenPoints support a wide range of basic operations including graph convolutions, self-attention, farthest point sampling, ball query, *e.t.c***.

2. **Ease of Use**: *Build* model, optimizer, scheduler, loss function,  and data loader *easily from cfg*. Train and validate different models on various tasks by simply changing the `cfg\*\*.yaml` file. 

      ```
   model = build_model_from_cfg(cfg.model)
   criterion = build_criterion_from_cfg(cfg.criterion_args)
   ```



## Installation

OpenPoints can be installed as the `openpoints` Python package:

```bash
pip install openpoints
```

The PyPI package installs the Python library (`import openpoints`) and the files needed by the datasets and configs. CUDA/C++ operators such as `pointnet2_batch_cuda`, `pointops_cuda`, `chamfer`, and `emd_cuda` are still built from a source checkout or source distribution for now because PyTorch/CUDA wheels must match the user's Python, PyTorch, CUDA, and platform versions.

For full training/evaluation with CUDA ops, install from source after installing PyTorch:

```bash
git clone --recursive https://github.com/guochengqian/openpoints.git
cd openpoints
pip install -e .[data,viz,wandb]
cd cpp/pointnet2_batch && python setup.py install && cd ../..
cd cpp/pointops && python setup.py install && cd ../..
cd cpp/chamfer_dist && python setup.py install && cd ../..
cd cpp/emd && python setup.py install && cd ../..
```

If the `openpoints` name is unavailable on a package index mirror, the same code can be published as `openpoints-torch`; the import name remains `openpoints`.

## Usage

OpenPoints serves as the engine for PointNeXt. Please refer to [PointNeXt](https://github.com/guochengqian/PointNeXt) for complete training, evaluation, and model-zoo examples.



## Citation

If you use this library, please kindly acknowledge our work:
```tex
@Article{qian2022pointnext,
  author  = {Qian, Guocheng and Li, Yuchen and Peng, Houwen and Mai, Jinjie and Hammoud, Hasan and Elhoseiny, Mohamed and Ghanem, Bernard},
  title   = {PointNeXt: Revisiting PointNet++ with Improved Training and Scaling Strategies},
  journal = {arXiv:2206.04670},
  year    = {2022},
}
```


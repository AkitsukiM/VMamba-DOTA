# VMamba+MMRotate+DOTA-v1.0旋转目标检测

## 写在前面

本项目仅为中文区的CVers提供快速实装VMamba视觉表征模型的参考例。

本项目以代码差分方式提交，请务必参考：

* MMRotate: [code](https://github.com/open-mmlab/mmrotate)
* DOTA_devkit: [code](https://github.com/CAPTAIN-WHU/DOTA_devkit)
* VMamba: [paper](https://arxiv.org/abs/2401.10166) [code](https://github.com/MzeroMiko/VMamba)
* Mamba: [paper](https://arxiv.org/abs/2312.00752) [code](https://github.com/state-spaces/mamba)

本项目组织方式：

* 把VMamba/classification/models/文件夹放到主目录下
* 把VMamba/detection/model.py文件放到./mmrotate/models/backbones/下改个名字和__init__.py
* 基于swin的config文件做一个vmamba的config文件

## mmrotate-0.3.3/0.3.4安装

```shell
# 受到 https://github.com/state-spaces/mamba 要求：PyTorch 1.12+ CUDA 11.6+
wget https://developer.download.nvidia.com/compute/cuda/11.6.2/local_installers/cuda_11.6.2_510.47.03_linux.run
chmod +x cuda_11.6.2_510.47.03_linux.run
sudo sh cuda_11.6.2_510.47.03_linux.run
# cuda 11.6对应cudnn 8.4.0
# tar -xf cudnn-linux-x86_64-8.4.0.27_cuda11.6-archive.tar.xz
tar -xf cudnn-linux-x86_64-8.4.1.50_cuda11.6-archive.tar.xz
sudo cp cudnn-linux-x86_64-8.4.1.50_cuda11.6-archive/include/* /usr/local/cuda-11.6/include/
sudo cp cudnn-linux-x86_64-8.4.1.50_cuda11.6-archive/lib/* /usr/local/cuda-11.6/lib64/
# 
# vi ~/.bashrc
# Add CUDA path
export PATH=/usr/local/cuda-11.6/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.6/lib64:$LD_LIBRARY_PATH
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"
# 
source ~/.bashrc
nvcc -V
# 
conda create -n py38cu116 python=3.8 -y
conda activate py38cu116
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
pip install shapely tqdm timm
# 
pip install openmim
mim install mmcv-full==1.6.1
mim install mmdet==2.25.1
git clone https://github.com/open-mmlab/mmrotate.git
cd mmrotate
pip install -r requirements/build.txt
pip install -v -e .
# 
# 降版本
pip install numpy==1.21.5
pip install yapf==0.40.1
# 
# pip install causal-conv1d==1.1.0
wget https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.1.0/causal_conv1d-1.1.0+cu118torch1.12cxx11abiFALSE-cp38-cp38-linux_x86_64.whl
pip install causal_conv1d-1.1.0+cu118torch1.12cxx11abiFALSE-cp38-cp38-linux_x86_64.whl
# pip install mamba-ssm==1.1.1
wget https://github.com/state-spaces/mamba/releases/download/v1.1.1/mamba_ssm-1.1.1+cu118torch1.12cxx11abiFALSE-cp38-cp38-linux_x86_64.whl
pip install mamba_ssm-1.1.1+cu118torch1.12cxx11abiFALSE-cp38-cp38-linux_x86_64.whl
```

## DOTA_devkit安装

```shell
sudo apt install swig
swig -c++ -python polyiou.i
python setup.py build_ext --inplace
```

## DOTA数据集创建

使用官网下载的数据集解压创建

```shell
python md5_calc.py --path /Dataset/DOTA/train.tar.gz
# cfb5007ada913241e02c24484e12d5d2
python md5_calc.py --path /Dataset/DOTA/val.tar.gz
# a53e74b0d69dacf3ffcb438accd60c45
tar -xzf /Dataset/DOTA/train.tar.gz -C /Dataset/DOTA/
tar -xzf /Dataset/DOTA/val.tar.gz -C /Dataset/DOTA/
python dir_list.py --path /Dataset/DOTA/train/images/ --output /Dataset/DOTA/train/trainset.txt
# 1411
python dir_list.py --path /Dataset/DOTA/val/images/ --output /Dataset/DOTA/val/valset.txt
# 458
```

mmrotate分割处理

```
python tools/data/dota/split/img_split.py --base-json tools/data/dota/split/split_configs/ss_train.json
# Total images number: 15749
python tools/data/dota/split/img_split.py --base-json tools/data/dota/split/split_configs/ss_val.json
# Total images number: 5297
python tools/data/dota/split/img_split.py --base-json tools/data/dota/split/split_configs/ss_trainval.json
# Total images number: 21046
python tools/data/dota/split/img_split.py --base-json tools/data/dota/split/split_configs/ss_test.json
# Total images number: 10833
python tools/data/dota/split/img_split.py --base-json tools/data/dota/split/split_configs/ms_trainval.json
# Total images number: 138883
python tools/data/dota/split/img_split.py --base-json tools/data/dota/split/split_configs/ms_test.json
# Total images number: 71888
```

预训练模型放在mmrotate-main/data/pretrained/

## DOTA数据集训练与测试

多卡训练：

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./tools/dist_train.sh ./configs/rotated_faster_rcnn/rotated_faster_rcnn_r50_fpn_1x_dota_le90.py 8
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 nohup ./tools/dist_train.sh ./configs/rotated_faster_rcnn/rotated_faster_rcnn_r50_fpn_1x_dota_le90.py 8 > nohup.log 2>&1 &
```

多卡验证：

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./tools/dist_test.sh ./configs/rotated_faster_rcnn/rotated_faster_rcnn_r50_fpn_1x_dota_le90.py ./work_dirs/rotated_faster_rcnn_r50_fpn_1x_dota_le90/epoch_12.pth 8 --eval mAP
```

多卡测试：

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./tools/dist_test.sh ./configs/rotated_faster_rcnn/rotated_faster_rcnn_r50_fpn_1x_dota_le90.py ./work_dirs/rotated_faster_rcnn_r50_fpn_1x_dota_le90/epoch_12.pth 8 --format-only --eval-options submission_dir="./work_dirs/Task1_epoch_12/"
python "../DOTA_devkit-master/dota_evaluation_task1.py" --mergedir "./work_dirs/Task1_epoch_12/" --imagesetdir "./data/DOTA/val/" --use_07_metric True
```

Params&FLOPs计算：

```shell
python ./tools/analysis_tools/get_flops.py ./configs/rotated_faster_rcnn/rotated_faster_rcnn_r50_fpn_1x_dota_le90.py
```

## 性能报告

|    Model    | 12 epochs | split mAP | merge mAP |  Params  |  FLOPs  | Configs |
| :---------: | :-------: | :-------: | :-------: | :------: | :-----: | :-----: |
|  Swin-Tiny  |    1.8h   |   70.22   |   72.16   |  44.76M  | 215.54G | [cfg](./mmrotate-main/configs/_rotated_faster_rcnn_/rotated_faster_rcnn_swin_tiny_fpn_1x_dota_le90.py) |
|  Swin-Small |    2.8h   |   71.49   |   73.21   |  66.08M  | 308.28G | [cfg](./mmrotate-main/configs/_rotated_faster_rcnn_/rotated_faster_rcnn_swin_small_fpn_1x_dota_le90.py) |
|  Swin-Base  |    3.5h   |   71.43   |   73.08   | 104.11M  | 455.35G | [cfg](./mmrotate-main/configs/_rotated_faster_rcnn_/rotated_faster_rcnn_swin_base_fpn_1x_dota_le90.py) |
| VMamba-Tiny |    1.7h   |   72.12   |   74.09   |  39.37M  | 179.42G | [cfg](./mmrotate-main/configs/_rotated_faster_rcnn_/rotated_faster_rcnn_vmamba_tiny_fpn_1x_dota_le90.py) |
| VMamba-Small|    3.9h   |   72.24   |   74.52   |  60.89M  | 245.38G | [cfg](./mmrotate-main/configs/_rotated_faster_rcnn_/rotated_faster_rcnn_vmamba_small_fpn_1x_dota_le90.py) |
| VMamba-Base |    4.9h   |   73.20   |   75.07   |  92.59M  | 342.89G | [cfg](./mmrotate-main/configs/_rotated_faster_rcnn_/rotated_faster_rcnn_vmamba_base_fpn_1x_dota_le90.py) |

## 写在后面

本项目代码不一定会更新但VMamba的本家代码一定会更新，欢迎关注！

（其实我是VMamba的作者之一）

-----

Copyright (c) 2024 Marina Akitsuki. All rights reserved.

Date modified: 2024/01/22


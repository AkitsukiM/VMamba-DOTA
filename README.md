# VMamba+MMRotate+DOTA-v1.0旋转目标检测

## 写在前面

本项目仅为中文区的CVers提供快速实装VMamba视觉表征模型的参考例。

本项目以代码差分方式提交，请务必参考：

* MMRotate: [main](https://github.com/open-mmlab/mmrotate) [dev-1.x](https://github.com/open-mmlab/mmrotate/tree/dev-1.x)
* DOTA_devkit: [code](https://github.com/CAPTAIN-WHU/DOTA_devkit)
* VMamba: [paper](https://arxiv.org/abs/2401.10166) [code](https://github.com/MzeroMiko/VMamba)
* VHeat: [paper](https://arxiv.org/abs/2405.16555) [code](https://github.com/MzeroMiko/vHeat)

本项目基于2024/05/30版本的VMamba代码和VHeat代码编写。

本项目组织方式：

* 把"VMamba/classification/models/"文件夹作为"mmrotate/models/backbones/vmamba_models/"文件夹，把"vHeat/detection/vHeat/"文件夹作为"mmrotate/models/backbones/vheat_models/"文件夹
* 把"VMamba/detection/model.py"文件作为"mmrotate/models/backbones/vmamba_model.py"文件，把"vheat/detection/model.py"文件作为"mmrotate/models/backbones/vheat_model.py"文件，并修改"\_\_init\_\_.py"
* 制作vmamba和VHeat的config文件
* 迁移"VMamba/kernels/"文件夹

## mmrotate-0.3.3/0.3.4/dev-1.x安装

【说明】我们推荐使用mmrotate-0.3.3/0.3.4版本，它是一个较为稳定的版本。mmrotate-dev-1.x版本是基于mmcv-2与mmdet-3编写的未来主流版本，但它的多尺度测试可能存在一些问题。

```shell
# 受到 https://github.com/state-spaces/mamba 要求：PyTorch 1.12+ CUDA 11.6+
# wget https://developer.download.nvidia.com/compute/cuda/11.7.1/local_installers/cuda_11.7.1_515.65.01_linux.run
# chmod +x ./cuda_11.7.1_515.65.01_linux.run
# sudo sh cuda_11.7.1_515.65.01_linux.run
wget https://developer.download.nvidia.com/compute/cuda/11.6.2/local_installers/cuda_11.6.2_510.47.03_linux.run
chmod +x ./cuda_11.6.2_510.47.03_linux.run
sudo ./cuda_11.6.2_510.47.03_linux.run
# 
# vi ~/.bashrc
# Add CUDA path
# export PATH=/usr/local/cuda-11.7/bin:$PATH
# export LD_LIBRARY_PATH=/usr/local/cuda-11.7/lib64:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda-11.6/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.6/lib64:$LD_LIBRARY_PATH
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"
# 
source ~/.bashrc
nvcc -V
# 
# NO sudo when install anaconda
# chmod +x ./Anaconda3-2023.09-0-Linux-x86_64.sh
# ./Anaconda3-2023.09-0-Linux-x86_64.sh
# 
# conda create -n openmmlab1131 python=3.9 -y
# conda activate openmmlab1131
# # ref: https://pytorch.org/get-started/previous-versions/#v1131
# conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
conda create -n openmmlab1121 python=3.8 -y
conda activate openmmlab1121
# ref: https://pytorch.org/get-started/previous-versions/#v1121
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
# 
pip install shapely tqdm timm
# 
# # if mmrotate-dev-1.x
# pip install -U openmim
# mim install mmengine
# # 受到 dev-1.x 要求，可以安装 mmcv==2.0.0rc2 和 mmdet==3.0.0rc6 之后的那个版本
# mim install mmcv==2.0.1
# mim install mmdet==3.1.0
# # 我们所使用mmrotate-dev-1.x版本的提交码是fd60beff130a54e284a73651903de29fe728f97b，请注意核对
# git clone https://github.com/open-mmlab/mmrotate.git -b dev-1.x
# cd mmrotate
# pip install -r requirements/build.txt
# pip install -v -e .
# 
# if mmrotate-0.3.3/0.3.4
pip install openmim
mim install mmcv-full==1.6.1
mim install mmdet==2.25.1
git clone https://github.com/open-mmlab/mmrotate.git
cd mmrotate
pip install -r requirements/build.txt
pip install -v -e .
# 
# 为mmrotate-0.3.3/0.3.4降低部分包的版本
pip install numpy==1.21.5
pip install yapf==0.40.1
# 
# 安装必要的vmamba依赖
pip install einops fvcore triton ninja
cd kernels/selective_scan/ && pip install . && cd ../../
```

## DOTA数据集创建

使用官网下载的数据集解压创建

```shell
# 请先准备好与mmrotate主目录并列的mmrotate-data和mmrotate-tools文件夹
# 以下命令均在mmrotate主目录下执行
# 
# ln -s /Workspace/Dataset/DOTA/ ../mmrotate-data/data/
# 
ln -s ../mmrotate-data/data/ ./
ln -s ../mmrotate-data/work_dirs/ ./
# 
python ../mmrotate-tools/md5_calc.py --path ./data/DOTA/train.tar.gz
# cfb5007ada913241e02c24484e12d5d2
python ../mmrotate-tools/md5_calc.py --path ./data/DOTA/val.tar.gz
# a53e74b0d69dacf3ffcb438accd60c45
python ../mmrotate-tools/md5_calc.py --path ./data/DOTA/test/part1.zip
# d3028e48da64b37ad2f2f5f31059e0da
python ../mmrotate-tools/md5_calc.py --path ./data/DOTA/test/part2.zip
# 99f779850cc44b8f8b28d348494c6b41
# 
tar -xzf ./data/DOTA/train.tar.gz -C ./data/DOTA/
tar -xzf ./data/DOTA/val.tar.gz -C ./data/DOTA/
unzip ./data/DOTA/test/part1.zip -d ./data/DOTA/test/
unzip ./data/DOTA/test/part2.zip -d ./data/DOTA/test/
# 
python ../mmrotate-tools/dir_list.py --path ./data/DOTA/train/images/ --output ./data/DOTA/train/trainset.txt
# 1411
python ../mmrotate-tools/dir_list.py --path ./data/DOTA/val/images/ --output ./data/DOTA/val/valset.txt
# 458
python ../mmrotate-tools/dir_list.py --path ./data/DOTA/test/images/ --output ./data/DOTA/test/testset.txt
# 937
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

预训练模型目录为./data/pretrained/

## DOTA_devkit安装

```shell
sudo apt install swig
swig -c++ -python polyiou.i
python setup.py build_ext --inplace
```

## DOTA数据集训练与合并测试（示例）

多卡训练：

```shell
# if mmrotate-0.3.3/0.3.4
CUDA_VISIBLE_DEVICES=0,1 ./tools/dist_train.sh ./configs/_rotated_faster_rcnn_/rotated_faster_rcnn_r50_fpn_1x_dota_le90.py 2
CUDA_VISIBLE_DEVICES=0,1 nohup ./tools/dist_train.sh ./configs/_rotated_faster_rcnn_/rotated_faster_rcnn_r50_fpn_1x_dota_le90.py 2 > nohup.log 2>&1 &
# 
# if mmrotate-dev-1.x
CUDA_VISIBLE_DEVICES=0,1 ./tools/dist_train.sh ./configs/_rotated_faster_rcnn_/rotated-faster-rcnn-le90_r50_fpn_1x_dota.py 2
CUDA_VISIBLE_DEVICES=0,1 nohup ./tools/dist_train.sh ./configs/_rotated_faster_rcnn_/rotated-faster-rcnn-le90_r50_fpn_1x_dota.py 2 > nohup.log 2>&1 &
```

多卡合并测试：

```shell
# if mmrotate-0.3.3/0.3.4
CUDA_VISIBLE_DEVICES=0,1 ./tools/dist_test.sh ./configs/_rotated_faster_rcnn_/rotated_faster_rcnn_r50_fpn_1x_dota_le90.py ./work_dirs/rotated_faster_rcnn_r50_fpn_1x_dota_le90/rotated_faster_rcnn_r50_fpn_1x_dota_le90-0393aa5c.pth 2 --format-only --eval-options submission_dir="./work_dirs/Task1_r50_033"
python "../DOTA_devkit-master/dota_evaluation_task1.py" --mergedir "./work_dirs/Task1_r50_033/" --imagesetdir "./data/DOTA/val/" --use_07_metric True
# map: 0.820117064577964
# 
# if mmrotate-dev-1.x
CUDA_VISIBLE_DEVICES=0,1 ./tools/dist_test.sh ./configs/_rotated_faster_rcnn_/rotated-faster-rcnn-le90_r50_fpn_1x_dota.py ./work_dirs/rotated_faster_rcnn_r50_fpn_1x_dota_le90/rotated_faster_rcnn_r50_fpn_1x_dota_le90-0393aa5c.pth 2
python "../DOTA_devkit-master/dota_evaluation_task1.py" --mergedir "./work_dirs/Task1_rotated-faster-rcnn-le90_r50_fpn_1x_dota/" --imagesetdir "./data/DOTA/val/" --use_07_metric True
# map: 0.8193743727960783
```

Params&FLOPs计算：

```shell
python ./tools/analysis_tools/get_flops.py ./configs/rotated_faster_rcnn/rotated_faster_rcnn_r50_fpn_1x_dota_le90.py
```

## Performance

所有的训练和测试均在4×A100卡上进行。

* 表中split mAP是对ss-val的评测；merge mAP是对ss-val或ms-test的评测；
* 表中VMamba的FLOPs暂无计算。
* 表中VHeat的性能暂不发布。

<table border="11" align="center">
    <tr align="center">
        <td >Detector</td>
        <td >backbone_size</td>
        <td >batch_size</td>
        <td >init_lr</td>
        <td >
            split&nbsp;mAP
        </td>
        <td >
            merge&nbsp;mAP
        </td>
        <td >
            Training<br>
            Cost<br>
        </td>
        <td >
            Testing<br>
            FPS<br>
        </td>
        <td >Params</td>
        <td >FLOPs</td>
        <td >Configs</td>
    </tr>
    <tr align="center">
        <td rowspan="9">
            Rotated<br>
            RetinaNet<br>
            (1x&nbsp;ss)<br>
        </td>
        <td> Swin-T </td> <td>4*4</td> <td>1e-4</td> <td>67.14</td> <td>68.68</td> <td>0.7h</td> <td>106.6</td> <td> 37.13M</td> <td>222.08G</td> 
            <td><a href="https://github.com/AkitsukiM/VMamba-DOTA/blob/master/mmrotate-0.3.3/configs/_rotated_retinanet_/rotated_retinanet_obb_swin_tiny_fpn_1x_dota_le90.py">cfg</a></td>
    </tr>
    <tr align="center">
        <td> Swin-S </td> <td>4*4</td> <td>1e-4</td> <td>67.54</td> <td>69.66</td> <td>1.9h</td> <td> 61.1</td> <td> 58.45M</td> <td>314.82G</td> 
            <td><a href="https://github.com/AkitsukiM/VMamba-DOTA/blob/master/mmrotate-0.3.3/configs/_rotated_retinanet_/rotated_retinanet_obb_swin_small_fpn_1x_dota_le90.py">cfg</a></td>
    </tr>
    <tr align="center">
        <td> Swin-B </td> <td>4*4</td> <td>1e-4</td> <td>68.48</td> <td>70.56</td> <td>2.7h</td> <td> 45.3</td> <td> 97.06M</td> <td>461.50G</td> 
            <td><a href="https://github.com/AkitsukiM/VMamba-DOTA/blob/master/mmrotate-0.3.3/configs/_rotated_retinanet_/rotated_retinanet_obb_swin_base_fpn_1x_dota_le90.py">cfg</a></td>
    </tr>
    <tr align="center">
        <td> VHeat-T</td> <td>4*4</td> <td>1e-4</td> <td>  -  </td> <td>  -  </td> <td>  - </td> <td>  -  </td> <td>   -   </td> <td>   -   </td> 
            <td><a href="https://github.com/AkitsukiM/VMamba-DOTA/blob/master/mmrotate-0.3.3/configs/_rotated_retinanet_/rotated_retinanet_obb_vheat_tiny_fpn_1x_dota_le90.py">cfg</a></td>
    </tr>
    <tr align="center">
        <td> VHeat-S</td> <td>4*4</td> <td>1e-4</td> <td>  -  </td> <td>  -  </td> <td>  - </td> <td>  -  </td> <td>   -   </td> <td>   -   </td> 
            <td><a href="https://github.com/AkitsukiM/VMamba-DOTA/blob/master/mmrotate-0.3.3/configs/_rotated_retinanet_/rotated_retinanet_obb_vheat_small_fpn_1x_dota_le90.py">cfg</a></a></td>
    </tr>
    <tr align="center">
        <td> VHeat-B</td> <td>4*4</td> <td>1e-4</td> <td>  -  </td> <td>  -  </td> <td>  - </td> <td>  -  </td> <td>   -   </td> <td>   -   </td> 
            <td><a href="https://github.com/AkitsukiM/VMamba-DOTA/blob/master/mmrotate-0.3.3/configs/_rotated_retinanet_/rotated_retinanet_obb_vheat_base_fpn_1x_dota_le90.py">cfg</a></td>
    </tr>
    <tr align="center">
        <td>VMamba-T</td> <td>4*4</td> <td>1e-4</td> <td>69.15</td> <td>71.11</td> <td>1.0h</td> <td> 91.9</td> <td>   ?   </td> <td>   ?   </td> 
            <td><a href="https://github.com/AkitsukiM/VMamba-DOTA/blob/master/mmrotate-0.3.3/configs/_rotated_retinanet_/rotated_retinanet_obb_vmamba_tiny_fpn_1x_dota_le90.py">cfg</a></td>
    </tr>
    <tr align="center">
        <td>VMamba-S</td> <td>4*4</td> <td>1e-4</td> <td>69.78</td> <td>72.17</td> <td>1.8h</td> <td> 69.7</td> <td>   ?   </td> <td>   ?   </td> 
            <td><a href="https://github.com/AkitsukiM/VMamba-DOTA/blob/master/mmrotate-0.3.3/configs/_rotated_retinanet_/rotated_retinanet_obb_vmamba_small_fpn_1x_dota_le90.py">cfg</a></td>
    </tr>
    <tr align="center">
        <td>VMamba-B</td> <td>4*4</td> <td>1e-4</td> <td>69.70</td> <td>71.77</td> <td>2.2h</td> <td> 58.8</td> <td>   ?   </td> <td>   ?   </td> 
            <td><a href="https://github.com/AkitsukiM/VMamba-DOTA/blob/master/mmrotate-0.3.3/configs/_rotated_retinanet_/rotated_retinanet_obb_vmamba_base_fpn_1x_dota_le90.py">cfg</a></td>
    </tr>
    <tr align="center">
        <td rowspan="9">
            Rotated<br>
            Faster&nbsp;RCNN<br>
            (1x&nbsp;ss)<br>
        </td>
        <td> Swin-T </td> <td>4*4</td> <td>1e-4</td> <td>70.11</td> <td>72.62</td> <td>0.7h</td> <td>106.1</td> <td> 44.76M</td> <td>215.54G</td> 
            <td><a href="https://github.com/AkitsukiM/VMamba-DOTA/blob/master/mmrotate-0.3.3/configs/_rotated_faster_rcnn_/rotated_faster_rcnn_swin_tiny_fpn_1x_dota_le90.py">cfg</a></td>
    </tr>
    <tr align="center">
        <td> Swin-S </td> <td>4*4</td> <td>1e-4</td> <td>70.39</td> <td>73.22</td> <td>1.9h</td> <td> 58.7</td> <td> 66.08M</td> <td>308.28G</td> 
            <td><a href="https://github.com/AkitsukiM/VMamba-DOTA/blob/master/mmrotate-0.3.3/configs/_rotated_faster_rcnn_/rotated_faster_rcnn_swin_small_fpn_1x_dota_le90.py">cfg</a></td>
    </tr>
    <tr align="center">
        <td> Swin-B </td> <td>4*4</td> <td>1e-4</td> <td>71.73</td> <td>73.91</td> <td>2.7h</td> <td> 44.1</td> <td>104.11M</td> <td>455.35G</td> 
            <td><a href="https://github.com/AkitsukiM/VMamba-DOTA/blob/master/mmrotate-0.3.3/configs/_rotated_faster_rcnn_/rotated_faster_rcnn_swin_base_fpn_1x_dota_le90.py">cfg</a></td>
    </tr>
    <tr align="center">
        <td> VHeat-T</td> <td>4*4</td> <td>1e-4</td> <td>  -  </td> <td>73.16</td> <td>  - </td> <td>  -  </td> <td>49.71M</td> <td>219.11G</td> 
            <td><a href="https://github.com/AkitsukiM/VMamba-DOTA/blob/master/mmrotate-0.3.3/configs/_rotated_faster_rcnn_/rotated_faster_rcnn_vheat_tiny_fpn_1x_dota_le90.py">cfg</a></td>
    </tr>
    <tr align="center">
        <td> VHeat-S</td> <td>4*4</td> <td>1e-4</td> <td>72.86</td> <td>73.65</td> <td>  - </td> <td>  -  </td> <td>71.08M</td> <td>3.6.28G</td> 
            <td><a href="https://github.com/AkitsukiM/VMamba-DOTA/blob/master/mmrotate-0.3.3/configs/_rotated_faster_rcnn_/rotated_faster_rcnn_vheat_small_fpn_1x_dota_le90.py">cfg</a></td>
    </tr>
    <tr align="center">
        <td> VHeat-B</td> <td>4*4</td> <td>1e-4</td> <td>  -  </td> <td>73.56</td> <td>  - </td> <td>  -  </td> <td>111.65M</td> <td>451.47G</td> 
            <td><a href="https://github.com/AkitsukiM/VMamba-DOTA/blob/master/mmrotate-0.3.3/configs/_rotated_faster_rcnn_/rotated_faster_rcnn_vheat_base_fpn_1x_dota_le90.py">cfg</a></td>
    </tr>
    <tr align="center">
        <td>VMamba-T</td> <td>4*4</td> <td>1e-4</td> <td>73.13</td> <td>74.04</td> <td>1.1h</td> <td> 84.0</td> <td>   ?   </td> <td>   ?   </td> 
            <td><a href="https://github.com/AkitsukiM/VMamba-DOTA/blob/master/mmrotate-0.3.3/configs/_rotated_faster_rcnn_/rotated_faster_rcnn_vmamba_tiny_fpn_1x_dota_le90.py">cfg</a></td>
    </tr>
    <tr align="center">
        <td>VMamba-S</td> <td>4*4</td> <td>1e-4</td> <td>73.14</td> <td>74.16</td> <td>1.9h</td> <td> 63.3</td> <td>   ?   </td> <td>   ?   </td> 
            <td><a href="https://github.com/AkitsukiM/VMamba-DOTA/blob/master/mmrotate-0.3.3/configs/_rotated_faster_rcnn_/rotated_faster_rcnn_vmamba_small_fpn_1x_dota_le90.py">cfg</a></td>
    </tr>
    <tr align="center">
        <td>VMamba-B</td> <td>4*4</td> <td>1e-4</td> <td>73.30</td> <td>73.50</td> <td>2.3h</td> <td> 56.1</td> <td>   ?   </td> <td>   ?   </td> 
            <td><a href="https://github.com/AkitsukiM/VMamba-DOTA/blob/master/mmrotate-0.3.3/configs/_rotated_faster_rcnn_/rotated_faster_rcnn_vmamba_base_fpn_1x_dota_le90.py">cfg</a></td>
    </tr>
    <tr align="center">
        <td rowspan="9">
            Oriented<br>
            RCNN<br>
            (1x&nbsp;ss)<br>
        </td>
        <td> Swin-T </td> <td>4*4</td> <td>1e-4</td> <td>73.88</td> <td>75.92</td> <td>0.8h</td> <td>105.0</td> <td> 44.76M</td> <td>215.68G</td> 
            <td><a href="https://github.com/AkitsukiM/VMamba-DOTA/blob/master/mmrotate-0.3.3/configs/_oriented_rcnn_/oriented_rcnn_swin_tiny_fpn_1x_dota_le90.py">cfg</a></td>
    </tr>
    <tr align="center">
        <td> Swin-S </td> <td>4*4</td> <td>1e-4</td> <td>74.49</td> <td>76.07</td> <td>2.0h</td> <td> 58.7</td> <td> 66.08M</td> <td>308.42G</td> 
            <td><a href="https://github.com/AkitsukiM/VMamba-DOTA/blob/master/mmrotate-0.3.3/configs/_oriented_rcnn_/oriented_rcnn_swin_small_fpn_1x_dota_le90.py">cfg</a></td>
    </tr>
    <tr align="center">
        <td> Swin-B </td> <td>4*4</td> <td>1e-4</td> <td>74.88</td> <td>76.16</td> <td>2.8h</td> <td> 44.1</td> <td>104.11M</td> <td>455.49G</td> 
            <td><a href="https://github.com/AkitsukiM/VMamba-DOTA/blob/master/mmrotate-0.3.3/configs/_oriented_rcnn_/oriented_rcnn_swin_base_fpn_1x_dota_le90.py">cfg</a></td>
    </tr>
    <tr align="center">
        <td> VHeat-T</td> <td>4*4</td> <td>1e-4</td> <td>74.85</td> <td>76.56</td> <td>  - </td> <td>  -  </td> <td>49.71M</td> <td>219.11G</td> 
            <td><a href="https://github.com/AkitsukiM/VMamba-DOTA/blob/master/mmrotate-0.3.3/configs/_oriented_rcnn_/oriented_rcnn_vheat_tiny_fpn_1x_dota_le90.py">cfg</a></td>
    </tr>
    <tr align="center">
        <td> VHeat-S</td> <td>4*4</td> <td>1e-4</td> <td>74.96</td> <td>76.20</td> <td>  - </td> <td>  -  </td> <td>71.08M</td> <td> 306.42G</td> 
            <td><a href="https://github.com/AkitsukiM/VMamba-DOTA/blob/master/mmrotate-0.3.3/configs/_oriented_rcnn_/oriented_rcnn_vheat_small_fpn_1x_dota_le90.py">cfg</a></td>
    </tr>
    <tr align="center">
        <td> VHeat-B</td> <td>4*4</td> <td>1e-4</td> <td>74.58</td> <td>76.54</td> <td>  - </td> <td>  -  </td> <td>111.65</td> <td>451.60G</td> 
            <td><a href="https://github.com/AkitsukiM/VMamba-DOTA/blob/master/mmrotate-0.3.3/configs/_oriented_rcnn_/oriented_rcnn_vheat_base_fpn_1x_dota_le90.py">cfg</a></td>
    </tr>
    <tr align="center">
        <td>VMamba-T</td> <td>4*4</td> <td>1e-4</td> <td>75.95</td> <td>76.59</td> <td>1.1h</td> <td> 79.9</td> <td>   ?   </td> <td>   ?   </td> 
            <td><a href="https://github.com/AkitsukiM/VMamba-DOTA/blob/master/mmrotate-0.3.3/configs/_oriented_rcnn_/oriented_rcnn_vmamba_tiny_fpn_1x_dota_le90.py">cfg</a></td>
    </tr>
    <tr align="center">
        <td>VMamba-S</td> <td>4*4</td> <td>1e-4</td> <td>76.10</td> <td>76.70</td> <td>1.9h</td> <td> 62.4</td> <td>   ?   </td> <td>   ?   </td> 
            <td><a href="https://github.com/AkitsukiM/VMamba-DOTA/blob/master/mmrotate-0.3.3/configs/_oriented_rcnn_/oriented_rcnn_vmamba_small_fpn_1x_dota_le90.py">cfg</a></td>
    </tr>
    <tr align="center">
        <td>VMamba-B</td> <td>4*4</td> <td>1e-4</td> <td>76.27</td> <td>76.23</td> <td>2.3h</td> <td> 54.7</td> <td>   ?   </td> <td>   ?   </td> 
            <td><a href="https://github.com/AkitsukiM/VMamba-DOTA/blob/master/mmrotate-0.3.3/configs/_oriented_rcnn_/oriented_rcnn_vmamba_base_fpn_1x_dota_le90.py">cfg</a></td>
    </tr>
    <tr align="center">
        <td rowspan="9">
            Oriented<br>
            RCNN<br>
            (1x&nbsp;msrr)<br>
        </td>
        <td> Swin-T </td> <td>4*4</td> <td>1e-4</td> <td>88.45</td> <td>81.36</td> <td> 4.6h</td> <td></td> <td></td> <td></td> 
            <td><a href="https://github.com/AkitsukiM/VMamba-DOTA/blob/master/mmrotate-0.3.3/configs/_ms_/oriented_rcnn_swin_tiny_fpn_1x_dota_le90_ms.py">cfg</a></td>
    </tr>
    <tr align="center">
        <td> Swin-S </td> <td>4*4</td> <td>1e-4</td> <td>89.58</td> <td>81.08</td> <td>12.5h</td> <td></td> <td></td> <td></td> 
            <td><a href="https://github.com/AkitsukiM/VMamba-DOTA/blob/master/mmrotate-0.3.3/configs/_ms_/oriented_rcnn_swin_small_fpn_1x_dota_le90_ms.py">cfg</a></td>
    </tr>
    <tr align="center">
        <td> Swin-B </td> <td>4*4</td> <td>1e-4</td> <td>89.36</td> <td>81.04</td> <td>17.5h</td> <td></td> <td></td> <td></td> 
            <td><a href="https://github.com/AkitsukiM/VMamba-DOTA/blob/master/mmrotate-0.3.3/configs/_ms_/oriented_rcnn_swin_base_fpn_1x_dota_le90_ms.py">cfg</a></td>
    </tr>
    <tr align="center">
        <td> VHeat-T</td> <td>4*4</td> <td>1e-4</td> <td>89.73</td> <td>81.50</td> <td>  - </td> <td></td> <td></td> <td></td> 
            <td><a href="https://github.com/AkitsukiM/VMamba-DOTA/blob/master/mmrotate-0.3.3/configs/_ms_/oriented_rcnn_vheat_tiny_fpn_1x_dota_le90_ms.py">cfg</a></td>
    </tr>
    <tr align="center">
        <td> VHeat-S</td> <td>4*4</td> <td>1e-4</td> <td>89.57</td> <td>81.37</td> <td>  - </td> <td></td> <td></td> <td></td> 
            <td><a href="https://github.com/AkitsukiM/VMamba-DOTA/blob/master/mmrotate-0.3.3/configs/_ms_/oriented_rcnn_vheat_small_fpn_1x_dota_le90_ms.py">cfg</a></td>
    </tr>
    <tr align="center">
        <td> VHeat-B</td> <td>4*4</td> <td>1e-4</td> <td>90.71</td> <td>81.16</td> <td>  - </td> <td></td> <td></td> <td></td> 
            <td><a href="https://github.com/AkitsukiM/VMamba-DOTA/blob/master/mmrotate-0.3.3/configs/_ms_/oriented_rcnn_vheat_base_fpn_1x_dota_le90_ms.py">cfg</a></td>
    </tr>
    <tr align="center">
        <td>VMamba-T</td> <td>4*4</td> <td>1e-4</td> <td>89.78</td> <td>80.70</td> <td> 6.8h</td> <td></td> <td></td> <td></td> 
            <td><a href="https://github.com/AkitsukiM/VMamba-DOTA/blob/master/mmrotate-0.3.3/configs/_ms_/oriented_rcnn_vmamba_tiny_fpn_1x_dota_le90_ms.py">cfg</a></td>
    </tr>
    <tr align="center">
        <td>VMamba-S</td> <td>4*4</td> <td>1e-4</td> <td>90.56</td> <td>80.62</td> <td>12.2h</td> <td></td> <td></td> <td></td> 
            <td><a href="https://github.com/AkitsukiM/VMamba-DOTA/blob/master/mmrotate-0.3.3/configs/_ms_/oriented_rcnn_vmamba_small_fpn_1x_dota_le90_ms.py">cfg</a></td>
    </tr>
    <tr align="center">
        <td>VMamba-B</td> <td>4*4</td> <td>1e-4</td> <td>90.48</td> <td>80.97</td> <td>15.1h</td> <td></td> <td></td> <td></td> 
            <td><a href="https://github.com/AkitsukiM/VMamba-DOTA/blob/master/mmrotate-0.3.3/configs/_ms_/oriented_rcnn_vmamba_base_fpn_1x_dota_le90_ms.py">cfg</a></td>
    </tr>
</table>

## 写在后面

有错误请及时指出！虽然不会经常来看issue非常抱歉但是看到就一定会回复的。

更新代码好麻烦呜呜呜

-----

Copyright (c) 2024 Marina Akitsuki. All rights reserved.

Date modified: 2024/06/03


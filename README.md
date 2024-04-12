# VMamba+MMRotate+DOTA-v1.0旋转目标检测

## 写在前面

本项目仅为中文区的CVers提供快速实装VMamba视觉表征模型的参考例。

本项目以代码差分方式提交，请务必参考：

* MMRotate: [main](https://github.com/open-mmlab/mmrotate) [dev-1.x](https://github.com/open-mmlab/mmrotate/tree/dev-1.x)
* DOTA_devkit: [code](https://github.com/CAPTAIN-WHU/DOTA_devkit)
* VMamba: [paper](https://arxiv.org/abs/2401.10166) [code](https://github.com/MzeroMiko/VMamba)
* Mamba: [paper](https://arxiv.org/abs/2312.00752) [code](https://github.com/state-spaces/mamba)

本项目基于2024/04/04版本的VMamba代码编写。

本项目组织方式：

* 把"VMamba/classification/models/"文件夹作为"mmrotate/models/backbones/vmamba_models/"文件夹
* 把"VMamba/detection/model.py"文件作为"mmrotate/models/backbones/vmamba_model.py"文件并修改"\_\_init\_\_.py"
* 制作vmamba的config文件

## mmrotate-0.3.3/0.3.4/dev-1.x安装

【说明】我们推荐使用mmrotate-0.3.3/0.3.4版本，它是一个较为稳定的版本。mmrotate-dev-1.x版本是基于mmcv-2与mmdet-3编写的未来主流版本，但它的多尺度测试可能存在一些问题。

```shell
# 受到 https://github.com/state-spaces/mamba 要求：PyTorch 1.12+ CUDA 11.6+
wget https://developer.download.nvidia.com/compute/cuda/11.6.2/local_installers/cuda_11.6.2_510.47.03_linux.run
chmod +x ./cuda_11.6.2_510.47.03_linux.run
sudo ./cuda_11.6.2_510.47.03_linux.run
# cuda 11.6对应cudnn 8.4.0
# tar -xf cudnn-linux-x86_64-8.4.1.50_cuda11.6-archive.tar.xz
# sudo cp cudnn-linux-x86_64-8.4.1.50_cuda11.6-archive/include/* /usr/local/cuda-11.6/include/
# sudo cp cudnn-linux-x86_64-8.4.1.50_cuda11.6-archive/lib/* /usr/local/cuda-11.6/lib64/
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
# NO sudo when install anaconda
# chmod +x ./Anaconda3-2023.09-0-Linux-x86_64.sh
# ./Anaconda3-2023.09-0-Linux-x86_64.sh
# 
# 此处存疑，我自己使用的是pytorch==1.12.1，有报告称高版本PyTorch可能会对Swin Transformer有性能增益具体不明
# conda create -n openmmlab1131 python=3.8 -y
# conda activate openmmlab1131
# # ref: https://pytorch.org/get-started/previous-versions/#v1131
# conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia
conda create -n openmmlab1121 python=3.8 -y
conda activate openmmlab1121
# ref: https://pytorch.org/get-started/previous-versions/#v1121
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
pip install shapely tqdm timm
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
# 降低部分包的版本
pip install numpy==1.21.5
pip install yapf==0.40.1
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
# 安装必要的vmamba依赖
pip install einops fvcore triton
cd kernels/selective_scan && pip install .
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

## DOTA数据集训练与合并测试

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

<table border="11" align="center">
    <tr align="center">
        <td >Detector</td>
        <td >backbone_size</td>
        <td >batch_size</td>
        <td >
            init_lr<br>
            ×e-4<br>
        </td>
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
        <td rowspan="12">
            Rotated<br>
            RetinaNet<br>
            (1x&nbsp;ss)<br>
        </td>
        <td rowspan="2"> Swin-T </td> <td>4*4</td> <td>1</td> <td>67.14</td> <td>68.68</td> 
          <td rowspan="2">0.7h</td> <td rowspan="2">106.6</td> <td rowspan="2"> 37.13M</td> <td rowspan="2">222.08G</td> 
            <td rowspan="2"><a href="https://github.com/AkitsukiM/VMamba-DOTA/blob/master/mmrotate-0.3.3/configs/_rotated_retinanet_/rotated_retinanet_obb_swin_tiny_fpn_1x_dota_le90.py">cfg</a></td>
    </tr>
    <tr align="center">               <td>4*4</td> <td>2</td> <td>66.91</td> <td>68.72</td></tr>
    <tr align="center">
        <td rowspan="2"> Swin-S </td> <td>4*4</td> <td>1</td> <td>67.54</td> <td>69.66</td> 
          <td rowspan="2">1.9h</td> <td rowspan="2"> 61.1</td> <td rowspan="2"> 58.45M</td> <td rowspan="2">314.82G</td> 
            <td rowspan="2"><a href="https://github.com/AkitsukiM/VMamba-DOTA/blob/master/mmrotate-0.3.3/configs/_rotated_retinanet_/rotated_retinanet_obb_swin_small_fpn_1x_dota_le90.py">cfg</a></td>
    </tr>
    <tr align="center">               <td>4*4</td> <td>2</td> <td>68.22</td> <td>70.13</td></tr>
    <tr align="center">
        <td rowspan="2"> Swin-B </td> <td>4*4</td> <td>1</td> <td>68.48</td> <td>70.56</td> 
          <td rowspan="2">2.7h</td> <td rowspan="2"> 45.3</td> <td rowspan="2"> 97.06M</td> <td rowspan="2">461.50G</td> 
            <td rowspan="2"><a href="https://github.com/AkitsukiM/VMamba-DOTA/blob/master/mmrotate-0.3.3/configs/_rotated_retinanet_/rotated_retinanet_obb_swin_base_fpn_1x_dota_le90.py">cfg</a></td>
    </tr>
    <tr align="center">               <td>4*4</td> <td>2</td> <td>68.60</td> <td>70.65</td></tr>
    <tr align="center">
        <td rowspan="2">VMamba-T</td> <td>4*4</td> <td>1</td> <td>68.99</td> <td>71.28</td> 
          <td rowspan="2">0.9h</td> <td rowspan="2"> 94.4</td> <td rowspan="2">   ?   </td> <td rowspan="2">   ?   </td> 
            <td rowspan="2"><a href="https://github.com/AkitsukiM/VMamba-DOTA/blob/master/mmrotate-0.3.3/configs/_rotated_retinanet_/rotated_retinanet_obb_vmamba_tiny_fpn_1x_dota_le90.py">cfg</a></td>
    </tr>
    <tr align="center">               <td>4*4</td> <td>2</td> <td>70.74</td> <td>71.56</td></tr>
    <tr align="center">
        <td rowspan="2">VMamba-S</td> <td>4*4</td> <td>1</td> <td>68.48</td> <td>71.45</td> 
          <td rowspan="2">3.0h</td> <td rowspan="2"> 51.3</td> <td rowspan="2">   ?   </td> <td rowspan="2">   ?   </td> 
            <td rowspan="2"><a href="https://github.com/AkitsukiM/VMamba-DOTA/blob/master/mmrotate-0.3.3/configs/_rotated_retinanet_/rotated_retinanet_obb_vmamba_small_fpn_1x_dota_le90.py">cfg</a></td>
    </tr>
    <tr align="center">               <td>4*4</td> <td>2</td> <td>70.50</td> <td>71.80</td></tr>
    <tr align="center">
        <td rowspan="2">VMamba-B</td> <td>4*4</td> <td>1</td> <td>69.15</td> <td>71.69</td> 
          <td rowspan="2">3.8h</td> <td rowspan="2"> 38.5</td> <td rowspan="2">   ?   </td> <td rowspan="2">   ?   </td> 
            <td rowspan="2"><a href="https://github.com/AkitsukiM/VMamba-DOTA/blob/master/mmrotate-0.3.3/configs/_rotated_retinanet_/rotated_retinanet_obb_vmamba_base_fpn_1x_dota_le90.py">cfg</a></td>
    </tr>
    <tr align="center">               <td>4*4</td> <td>2</td> <td>69.99</td> <td>72.03</td></tr>
    <tr align="center">
        <td rowspan="12">
            Rotated<br>
            Faster&nbsp;RCNN<br>
            (1x&nbsp;ss)<br>
        </td>
        <td rowspan="2"> Swin-T </td> <td>4*4</td> <td>1</td> <td>70.11</td> <td>72.62</td> 
          <td rowspan="2">0.7h</td> <td rowspan="2">106.1</td> <td rowspan="2"> 44.76M</td> <td rowspan="2">215.54G</td> 
            <td rowspan="2"><a href="https://github.com/AkitsukiM/VMamba-DOTA/blob/master/mmrotate-0.3.3/configs/_rotated_faster_rcnn_/rotated_faster_rcnn_swin_tiny_fpn_1x_dota_le90.py">cfg</a></td>
    </tr>
    <tr align="center">               <td>4*4</td> <td>2</td> <td>70.55</td> <td>73.34</td></tr>
    <tr align="center">
        <td rowspan="2"> Swin-S </td> <td>4*4</td> <td>1</td> <td>70.39</td> <td>73.22</td> 
          <td rowspan="2">1.9h</td> <td rowspan="2"> 58.7</td> <td rowspan="2"> 66.08M</td> <td rowspan="2">308.28G</td> 
            <td rowspan="2"><a href="https://github.com/AkitsukiM/VMamba-DOTA/blob/master/mmrotate-0.3.3/configs/_rotated_faster_rcnn_/rotated_faster_rcnn_swin_small_fpn_1x_dota_le90.py">cfg</a></td>
    </tr>
    <tr align="center">               <td>4*4</td> <td>2</td> <td>72.23</td> <td>73.77</td></tr>
    <tr align="center">
        <td rowspan="2"> Swin-B </td> <td>4*4</td> <td>1</td> <td>71.73</td> <td>73.91</td> 
          <td rowspan="2">2.7h</td> <td rowspan="2"> 44.1</td> <td rowspan="2">104.11M</td> <td rowspan="2">455.35G</td> 
            <td rowspan="2"><a href="https://github.com/AkitsukiM/VMamba-DOTA/blob/master/mmrotate-0.3.3/configs/_rotated_faster_rcnn_/rotated_faster_rcnn_swin_base_fpn_1x_dota_le90.py">cfg</a></td>
    </tr>
    <tr align="center">               <td>4*4</td> <td>2</td> <td>73.16</td> <td>74.41</td></tr>
    <tr align="center">
        <td rowspan="2">VMamba-T</td> <td>4*4</td> <td>1</td> <td>73.72</td> <td>74.75</td> 
          <td rowspan="2">1.0h</td> <td rowspan="2"> 90.0</td> <td rowspan="2">   ?   </td> <td rowspan="2">   ?   </td> 
            <td rowspan="2"><a href="https://github.com/AkitsukiM/VMamba-DOTA/blob/master/mmrotate-0.3.3/configs/_rotated_faster_rcnn_/rotated_faster_rcnn_vmamba_tiny_fpn_1x_dota_le90.py">cfg</a></td>
    </tr>
    <tr align="center">               <td>4*4</td> <td>2</td> <td>74.91</td> <td>75.31</td></tr>
    <tr align="center">
        <td rowspan="2">VMamba-S</td> <td>4*4</td> <td>1</td> <td>73.26</td> <td>74.60</td> 
          <td rowspan="2">3.0h</td> <td rowspan="2"> 48.6</td> <td rowspan="2">   ?   </td> <td rowspan="2">   ?   </td> 
            <td rowspan="2"><a href="https://github.com/AkitsukiM/VMamba-DOTA/blob/master/mmrotate-0.3.3/configs/_rotated_faster_rcnn_/rotated_faster_rcnn_vmamba_small_fpn_1x_dota_le90.py">cfg</a></td>
    </tr>
    <tr align="center">               <td>4*4</td> <td>2</td> <td>73.05</td> <td>75.55</td></tr>
    <tr align="center">
        <td rowspan="2">VMamba-B</td> <td>4*4</td> <td>1</td> <td>73.43</td> <td>74.16</td> 
          <td rowspan="2">3.8h</td> <td rowspan="2"> 37.3</td> <td rowspan="2">   ?   </td> <td rowspan="2">   ?   </td> 
            <td rowspan="2"><a href="https://github.com/AkitsukiM/VMamba-DOTA/blob/master/mmrotate-0.3.3/configs/_rotated_faster_rcnn_/rotated_faster_rcnn_vmamba_base_fpn_1x_dota_le90.py">cfg</a></td>
    </tr>
    <tr align="center">               <td>4*4</td> <td>2</td> <td>74.27</td> <td>75.63</td></tr>
    <tr align="center">
        <td rowspan="12">
            Oriented<br>
            RCNN<br>
            (1x&nbsp;ss)<br>
        </td>
        <td rowspan="2"> Swin-T </td> <td>4*4</td> <td>1</td> <td>73.88</td> <td>75.92</td> 
          <td rowspan="2">0.8h</td> <td rowspan="2">105.0</td> <td rowspan="2"> 44.76M</td> <td rowspan="2">215.68G</td> 
            <td rowspan="2"><a href="https://github.com/AkitsukiM/VMamba-DOTA/blob/master/mmrotate-0.3.3/configs/_oriented_rcnn_/oriented_rcnn_swin_tiny_fpn_1x_dota_le90.py">cfg</a></td>
    </tr>
    <tr align="center">               <td>4*4</td> <td>2</td> <td>74.30</td> <td>75.84</td></tr>
    <tr align="center">
        <td rowspan="2"> Swin-S </td> <td>4*4</td> <td>1</td> <td>74.49</td> <td>76.07</td> 
          <td rowspan="2">2.0h</td> <td rowspan="2"> 58.7</td> <td rowspan="2"> 66.08M</td> <td rowspan="2">308.42G</td> 
            <td rowspan="2"><a href="https://github.com/AkitsukiM/VMamba-DOTA/blob/master/mmrotate-0.3.3/configs/_oriented_rcnn_/oriented_rcnn_swin_small_fpn_1x_dota_le90.py">cfg</a></td>
    </tr>
    <tr align="center">               <td>4*4</td> <td>2</td> <td>74.33</td> <td>76.26</td></tr>
    <tr align="center">
        <td rowspan="2"> Swin-B </td> <td>4*4</td> <td>1</td> <td>74.88</td> <td>76.16</td> 
          <td rowspan="2">2.8h</td> <td rowspan="2"> 44.1</td> <td rowspan="2">104.11M</td> <td rowspan="2">455.49G</td> 
            <td rowspan="2"><a href="https://github.com/AkitsukiM/VMamba-DOTA/blob/master/mmrotate-0.3.3/configs/_oriented_rcnn_/oriented_rcnn_swin_base_fpn_1x_dota_le90.py">cfg</a></td>
    </tr>
    <tr align="center">               <td>4*4</td> <td>2</td> <td>74.41</td> <td>76.38</td></tr>
    <tr align="center">
        <td rowspan="2">VMamba-T</td> <td>4*4</td> <td>1</td> <td>76.24</td> <td>76.51</td> 
          <td rowspan="2">1.0h</td> <td rowspan="2"> 86.6</td> <td rowspan="2">   ?   </td> <td rowspan="2">   ?   </td> 
            <td rowspan="2"><a href="https://github.com/AkitsukiM/VMamba-DOTA/blob/master/mmrotate-0.3.3/configs/_oriented_rcnn_/oriented_rcnn_vmamba_tiny_fpn_1x_dota_le90.py">cfg</a></td>
    </tr>
    <tr align="center">               <td>4*4</td> <td>2</td> <td>76.81</td> <td>77.01</td></tr>
    <tr align="center">
        <td rowspan="2">VMamba-S</td> <td>4*4</td> <td>1</td> <td>76.40</td> <td>76.32</td> 
          <td rowspan="2">3.1h</td> <td rowspan="2"> 47.8</td> <td rowspan="2">   ?   </td> <td rowspan="2">   ?   </td> 
            <td rowspan="2"><a href="https://github.com/AkitsukiM/VMamba-DOTA/blob/master/mmrotate-0.3.3/configs/_oriented_rcnn_/oriented_rcnn_vmamba_small_fpn_1x_dota_le90.py">cfg</a></td>
    </tr>
    <tr align="center">               <td>4*4</td> <td>2</td> <td>76.00</td> <td>77.32</td></tr>
    <tr align="center">
        <td rowspan="2">VMamba-B</td> <td>4*4</td> <td>1</td> <td>76.19</td> <td>76.17</td> 
          <td rowspan="2">3.8h</td> <td rowspan="2"> 36.5</td> <td rowspan="2">   ?   </td> <td rowspan="2">   ?   </td> 
            <td rowspan="2"><a href="https://github.com/AkitsukiM/VMamba-DOTA/blob/master/mmrotate-0.3.3/configs/_oriented_rcnn_/oriented_rcnn_vmamba_base_fpn_1x_dota_le90.py">cfg</a></td>
    </tr>
    <tr align="center">               <td>4*4</td> <td>2</td> <td>76.25</td> <td>77.61</td></tr>
    <tr align="center">
        <td rowspan="6">
            Oriented<br>
            RCNN<br>
            (1x&nbsp;msrr)<br>
        </td>
        <td            > Swin-T </td> <td>4*4</td> <td>2</td> <td>87.77</td> <td>81.25</td> 
          <td> 4.6h</td> <td></td> <td></td> <td></td> 
            <td><a href="https://github.com/AkitsukiM/VMamba-DOTA/blob/master/mmrotate-0.3.3/configs/_test_/oriented_rcnn_swin_tiny_fpn_1x_dota_le90_ms.py">cfg</a></td>
    </tr>
    <tr align="center">
        <td            > Swin-S </td> <td>4*4</td> <td>2</td> <td>89.11</td> <td>81.14</td> 
          <td>12.5h</td> <td></td> <td></td> <td></td> 
            <td><a href="https://github.com/AkitsukiM/VMamba-DOTA/blob/master/mmrotate-0.3.3/configs/_test_/oriented_rcnn_swin_small_fpn_1x_dota_le90_ms.py">cfg</a></td>
    </tr>
    <tr align="center">
        <td            > Swin-B </td> <td>4*4</td> <td>2</td> <td>89.12</td> <td>81.26</td> 
          <td>17.5h</td> <td></td> <td></td> <td></td> 
            <td><a href="https://github.com/AkitsukiM/VMamba-DOTA/blob/master/mmrotate-0.3.3/configs/_test_/oriented_rcnn_swin_base_fpn_1x_dota_le90_ms.py">cfg</a></td>
    </tr>
    <tr align="center">
        <td            >VMamba-T</td> <td>4*4</td> <td>2</td> <td>89.06</td> <td>81.20</td> 
          <td> 6.2h</td> <td></td> <td></td> <td></td> 
            <td><a href="https://github.com/AkitsukiM/VMamba-DOTA/blob/master/mmrotate-0.3.3/configs/_test_/oriented_rcnn_vmamba_tiny_fpn_1x_dota_le90_ms.py">cfg</a></td>
    </tr>
    <tr align="center">
        <td            >VMamba-S</td> <td>4*4</td> <td>2</td> <td>89.68</td> <td>79.81</td> 
          <td>20.2h</td> <td></td> <td></td> <td></td> 
            <td><a href="https://github.com/AkitsukiM/VMamba-DOTA/blob/master/mmrotate-0.3.3/configs/_test_/oriented_rcnn_vmamba_small_fpn_1x_dota_le90_ms.py">cfg</a></td>
    </tr>
    <tr align="center">
        <td            >VMamba-B</td> <td>4*4</td> <td>2</td> <td>89.86</td> <td>80.25</td> 
          <td>26.0h</td> <td></td> <td></td> <td></td> 
            <td><a href="https://github.com/AkitsukiM/VMamba-DOTA/blob/master/mmrotate-0.3.3/configs/_test_/oriented_rcnn_vmamba_base_fpn_1x_dota_le90_ms.py">cfg</a></td>
    </tr>
</table>

## 写在后面

更新代码好麻烦呜呜呜

-----

Copyright (c) 2024 Marina Akitsuki. All rights reserved.

Date modified: 2024/04/12


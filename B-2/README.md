# mmrotate-ui使用

## mmrotate-0.3.3/0.3.4安装

已验证在RTX 4090单卡上可实现

```shell
wget https://developer.download.nvidia.com/compute/cuda/11.0.3/local_installers/cuda_11.0.3_450.51.06_linux.run
chmod +x ./cuda_11.0.3_450.51.06_linux.run
sudo sh cuda_11.0.3_450.51.06_linux.run
# 
vi ~/.bashrc
# Add CUDA path
# export PATH=/usr/local/cuda-11.0/bin:$PATH
# export LD_LIBRARY_PATH=/usr/local/cuda-11.0/lib64:$LD_LIBRARY_PATH
source ~/.bashrc
nvcc -V
# 
# NO sudo when install anaconda
# wget https://mirror.tuna.tsinghua.edu.cn/anaconda/archive/Anaconda3-2023.09-0-Linux-x86_64.sh
# chmod +x ./Anaconda3-2023.09-0-Linux-x86_64.sh
# ./Anaconda3-2023.09-0-Linux-x86_64.sh
# 
conda create -n openmmlab171 python=3.7 -y
conda activate openmmlab171
# ref: https://pytorch.org/get-started/previous-versions/#v171
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch
# 
pip install shapely timm
# 
pip install openmim
mim install mmcv-full==1.6.1
mim install mmdet==2.25.1
git clone https://github.com/open-mmlab/mmrotate.git
cd mmrotate
pip install -r requirements/build.txt
pip install -v -e .
# 
# 为mmrotate-0.3.3/0.3.4降低部分包的版本
pip install yapf==0.40.1
```

## DOTA数据集创建

使用官网下载的数据集解压创建

```shell
ln -s ../../mm-data/data/ ./
ln -s ../../mm-data/work_dirs/ ./
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

请对应修改"./configs/\_base\_/datasets/dotav1.py"

预训练模型目录为"./data/pretrained/"

## DOTA数据集训练与合并测试（示例）

单卡训练和测试（5297测试报错是正常的）：

```shell
python ./tools/train.py ./configs/rotated_faster_rcnn/rotated_faster_rcnn_r50_fpn_1x_dota_le90.py
python ./tools/test.py  ./configs/rotated_faster_rcnn/rotated_faster_rcnn_r50_fpn_1x_dota_le90.py ./work_dirs/rotated_faster_rcnn_r50_fpn_1x_dota_le90/rotated_faster_rcnn_r50_fpn_1x_dota_le90-0393aa5c.pth --eval mAP
```

## PyQt5安装

```shell
# 执行：
pip install pyqt5
# 
# 报错：
# qt.qpa.plugin: Could not load the Qt platform plugin "xcb" in "..." even though it was found.
# 执行：
export QT_DEBUG_PLUGINS=1
# 报错：
# "Cannot load library .../libqxcb.so: (libxcb-xinerama.so.0: cannot open shared object file: No such file or directory)"
# 进入目录，执行：
ldd libqxcb.so
# 可以找到
# libxcb-xinerama.so.0 => not found
# libxcb-cursor.so.0 => not found
# 执行：
sudo apt install libxcb-xinerama0 libxcb-cursor0
```

-----

Copyright (c) 2024 Marina Akitsuki. All rights reserved.

Date modified: 2024/09/24


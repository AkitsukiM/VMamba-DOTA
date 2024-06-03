_base_ = './oriented-rcnn-le90_vheat-tiny_fpn_1x_dota_ms.py'

pretrained = 'data/pretrained/vHeat_base.pth'

model = dict(
    backbone=dict(
        # copied from https://github.com/MzeroMiko/vHeat/blob/main/detection/configs/vheat/mask_rcnn_fpn_coco_base.py
        post_norm=True,
        depths=(2, 2, 18, 2),
        dims=128,
        drop_path_rate=0.5,
        layer_scale=1.e-5,
        img_size=512,
        use_checkpoint=True,
        pretrained=pretrained),
    neck=dict(in_channels=[128, 256, 512, 1024]))

test_evaluator = dict(
    outfile_prefix='./work_dirs/Task1_oriented-rcnn-le90_vheat-base_fpn_1x_dota_ms')

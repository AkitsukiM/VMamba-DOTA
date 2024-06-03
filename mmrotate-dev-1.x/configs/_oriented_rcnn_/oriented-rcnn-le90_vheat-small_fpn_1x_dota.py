_base_ = './oriented-rcnn-le90_vheat-tiny_fpn_1x_dota.py'

pretrained = 'data/pretrained/vHeat_small.pth'

model = dict(
    backbone=dict(
        # copied from https://github.com/MzeroMiko/vHeat/blob/main/detection/configs/vheat/mask_rcnn_fpn_coco_small.py
        post_norm=True,
        depths=(2, 2, 18, 2),
        dims=96,
        drop_path_rate=0.3,
        layer_scale=1.e-5,
        img_size=512,
        use_checkpoint=True,
        pretrained=pretrained))

test_evaluator = dict(
    outfile_prefix='./work_dirs/Task1_oriented-rcnn-le90_vheat-small_fpn_1x_dota')

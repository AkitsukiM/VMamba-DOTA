_base_ = './rotated-faster-rcnn-le90_swin-tiny_fpn_1x_dota.py'

pretrained = 'data/pretrained/swin_base_patch4_window7_224.pth'

depths = [2, 2, 18, 2]
model = dict(
    backbone=dict(
        embed_dims=128,
        depths=depths,
        num_heads=[4, 8, 16, 32],
        drop_path_rate=0.3,
        with_cp=True,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=dict(in_channels=[128, 256, 512, 1024]))

test_evaluator = dict(
    outfile_prefix='./work_dirs/Task1_rotated-faster-rcnn-le90_swin-base_fpn_1x_dota')

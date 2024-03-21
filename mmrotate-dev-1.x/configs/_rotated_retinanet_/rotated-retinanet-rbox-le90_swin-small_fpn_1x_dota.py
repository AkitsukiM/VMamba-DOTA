_base_ = './rotated-retinanet-rbox-le90_swin-tiny_fpn_1x_dota.py'

pretrained = 'data/pretrained/swin_small_patch4_window7_224.pth'

depths = [2, 2, 18, 2]
model = dict(
    backbone=dict(
        depths=depths,
        with_cp=True,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)))

test_evaluator = dict(
    outfile_prefix='./work_dirs/Task1_rotated-retinanet-rbox-le90_swin-small_fpn_1x_dota')


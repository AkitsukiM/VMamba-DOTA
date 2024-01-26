_base_ = './rotated-faster-rcnn-le90_vmamba-tiny_fpn_1x_dota.py'

# configs from '../oriented_rcnn/oriented-rcnn-le90_swin-tiny_fpn_1x_dota.py'

pretrained = 'data/pretrained/vmamba_small_ckpt_epoch_238.pth'

depths = [2, 2, 27, 2]
model = dict(
    backbone=dict(
        depths=depths,
        with_cp=True,
        # init_cfg=dict(type='Pretrained', checkpoint=pretrained)))
        pretrained=pretrained))

test_evaluator = dict(
    outfile_prefix='./work_dirs/Task1_rotated-faster-rcnn-le90_vmamba-small_fpn_1x_dota')


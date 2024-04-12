_base_ = './rotated-faster-rcnn-le90_vmamba-tiny_fpn_1x_dota.py'

pretrained = 'data/pretrained/vssm_small_0229_ckpt_epoch_222.pth'

model = dict(
    backbone=dict(
        depths=(2, 2, 15, 2),
        use_checkpoint=True,
        pretrained=pretrained))

test_evaluator = dict(
    outfile_prefix='./work_dirs/Task1_rotated-faster-rcnn-le90_vmamba-small_fpn_1x_dota')

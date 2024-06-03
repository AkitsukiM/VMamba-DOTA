_base_ = './rotated-faster-rcnn-le90_vmamba-tiny_fpn_1x_dota.py'

pretrained = 'data/pretrained/vssm_small_0229_ckpt_epoch_222.pth'

model = dict(
    backbone=dict(
        # copied from https://github.com/MzeroMiko/VMamba/blob/main/detection/configs/vssm1/mask_rcnn_vssm_fpn_coco_small.py
        dims=96,
        depths=(2, 2, 15, 2),
        ssm_ratio=2.0,
        drop_path_rate=0.3,
        use_checkpoint=True,
        pretrained=pretrained))

test_evaluator = dict(
    outfile_prefix='./work_dirs/Task1_rotated-faster-rcnn-le90_vmamba-small_fpn_1x_dota')

_base_ = './rotated_faster_rcnn_vmamba_tiny_fpn_1x_dota_le90.py'

pretrained = 'data/pretrained/vssm_small_0229_ckpt_epoch_222.pth'

model = dict(
    backbone=dict(
        depths=(2, 2, 15, 2),
        use_checkpoint=True,
        pretrained=pretrained))

fp16 = None

_base_ = './oriented_rcnn_vmamba_tiny_fpn_1x_dota_ms_rr_le90.py'

pretrained = 'data/pretrained/vmamba_small_ckpt_epoch_238.pth'

model = dict(
    backbone=dict(
        depths=[2, 2, 27, 2],
        with_cp=True,
        # init_cfg=dict(type='Pretrained', checkpoint=pretrained)))
        pretrained=pretrained))

fp16 = None


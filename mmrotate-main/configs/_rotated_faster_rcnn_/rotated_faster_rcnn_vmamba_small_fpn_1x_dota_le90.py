_base_ = './rotated_faster_rcnn_vmamba_tiny_fpn_1x_dota_le90.py'

pretrained = 'data/pretrained/vmamba_small_ckpt_epoch_238.pth'

model = dict(
    backbone=dict(
        depths=[2, 2, 27, 2],
        with_cp=True,
        # init_cfg=dict(type='Pretrained', checkpoint=pretrained)))
        pretrained=pretrained))

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2)

optimizer = dict(lr=5e-5)

fp16 = None

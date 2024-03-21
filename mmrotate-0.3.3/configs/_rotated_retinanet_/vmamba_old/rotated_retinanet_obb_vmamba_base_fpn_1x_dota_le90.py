_base_ = './rotated_retinanet_obb_vmamba_tiny_fpn_1x_dota_le90.py'

pretrained = 'data/pretrained/vmamba_base_ckpt_epoch_260.pth'

model = dict(
    backbone=dict(
        embed_dims=128,
        depths=[2, 2, 27, 2],
        num_heads=[4, 8, 16, 32],
        drop_path_rate=0.3,
        with_cp=True,
        # init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
        dims=[128, 256, 512, 1024],
        pretrained=pretrained),
    neck=dict(in_channels=[256, 512, 1024]))

fp16 = None


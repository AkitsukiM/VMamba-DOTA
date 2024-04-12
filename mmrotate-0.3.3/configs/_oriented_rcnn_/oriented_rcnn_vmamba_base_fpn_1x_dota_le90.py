_base_ = './oriented_rcnn_vmamba_tiny_fpn_1x_dota_le90.py'

pretrained = 'data/pretrained/vssm_base_0229_ckpt_epoch_237.pth'

model = dict(
    backbone=dict(
        dims=128,
        depths=(2, 2, 15, 2),
        drop_path_rate=0.6,
        use_checkpoint=True,
        pretrained=pretrained),
    neck=dict(in_channels=[128, 256, 512, 1024]))

fp16 = None

_base_ = './oriented_rcnn_r50_fpn_1x_dota_ms_rr_le90.py'

# configs from 'mmdetection-2.25.1/configs/swin/mask_rcnn_swin-t-p4-w7_fpn_1x_coco.py'

pretrained = 'data/pretrained/swin_tiny_patch4_window7_224.pth'

model = dict(
    backbone=dict(
        _delete_=True,
        type='SwinTransformer',
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=dict(in_channels=[96, 192, 384, 768]))

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4)

# swin paper recommend: batch_size=8*2, init_lr=1e-4
# if with 8*2080Ti GPU: batch_size=8*1, init_lr=5e-5
# if with 4*A100   GPU: batch_size=4*4, init_lr=1e-4
optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=1e-4,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))

# you need to set mode='dynamic' if you are using pytorch<=1.5.0
fp16 = dict(loss_scale=dict(init_scale=512))


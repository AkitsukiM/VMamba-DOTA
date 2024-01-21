_base_ = './rotated_faster_rcnn_r50_fpn_1x_dota_le90.py'

pretrained = 'data/pretrained/ckpt_epoch_292.pth'

angle_version = 'le90'
model = dict(
    backbone=dict(
        _delete_=True,
        type='MMDET_VSSM',
        embed_dims=96,
        depths=[2, 2, 9, 2], # depths=[2, 2, 6, 2],
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
        # init_cfg=dict(type='Pretrained', checkpoint=pretrained)
        pretrained=pretrained),
    neck=dict(in_channels=[96, 192, 384, 768]))

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2)

# paper recommend: batch_size=16, init_lr=1e-4
# now with 8x12GB GPU: batch_size=8, init_lr=5e-5
optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=5e-5,
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

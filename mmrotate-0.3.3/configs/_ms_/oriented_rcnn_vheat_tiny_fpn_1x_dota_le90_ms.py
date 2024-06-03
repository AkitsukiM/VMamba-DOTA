_base_ = [
    '../_base_/_models_/oriented_rcnn_r50_fpn.py',
    '../_base_/_datasets_/dotav1_ms.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]

pretrained = 'data/pretrained/vHeat_tiny.pth'

angle_version = 'le90'
model = dict(
    backbone=dict(
        _delete_=True,
        # copied from https://github.com/MzeroMiko/vHeat/blob/main/detection/configs/vheat/mask_rcnn_fpn_coco_tiny.py
        type='MMDET_VHEAT',
        drop_path=0.1,
        post_norm=False,
        depths=(2, 2, 6, 2),
        dims=96,
        out_indices=(0, 1, 2, 3),
        img_size=512,
        pretrained=None,
        use_checkpoint=False,
        # ##### ##### swin-tiny! ##### ##### #
        embed_dims=96,
        # depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        # out_indices=(0, 1, 2, 3),
        with_cp=True, # False,
        convert_weights=True,
        init_cfg=None,
    ),
    neck=dict(in_channels=[96, 192, 384, 768]))

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4)

# NOTE
# swin paper recommend: batch_size=8*2, init_lr=1e-4
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

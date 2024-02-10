_base_ = './rotated-faster-rcnn-le90_r50_fpn_1x_dota.py'

# configs from '../oriented_rcnn/oriented-rcnn-le90_swin-tiny_fpn_1x_dota.py'

pretrained = 'data/pretrained/vmamba_tiny_ckpt_epoch_292.pth'

depths = [2, 2, 9, 2]
model = dict(
    backbone=dict(
        _delete_=True,
        type='MMDET_VSSM', # 'mmdet.SwinTransformer',
        embed_dims=96,
        depths=depths,
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
        # init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
        dims=[96, 192, 384, 768],
        pretrained=pretrained),
    neck=dict(in_channels=[96, 192, 384, 768]))

# optimizer
optim_wrapper = dict(
    optimizer=dict(
        _delete_=True,
        type='AdamW',
        lr=2e-4, # 0.0001,
        betas=(0.9, 0.999),
        weight_decay=0.05),
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))

# NOTE
# swin paper recommend: batch_size=8*2, init_lr=1e-4
# if with 4*A100   GPU: batch_size=4*4, init_lr=1e-4
train_dataloader = dict(
    batch_size=4,
    num_workers=4)

test_evaluator = dict(
    outfile_prefix='./work_dirs/Task1_rotated-faster-rcnn-le90_vmamba-tiny_fpn_1x_dota')


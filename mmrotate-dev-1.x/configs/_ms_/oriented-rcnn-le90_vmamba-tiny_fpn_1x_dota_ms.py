_base_ = [
    '../_base_/_models_/oriented-rcnn-le90_r50_fpn.py',
    '../_base_/_datasets_/dota_ms.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]

# pretrained = 'data/pretrained/vssm_tiny_0230_ckpt_epoch_262.pth'
pretrained = 'data/pretrained/vssm1_tiny_0230s_ckpt_epoch_264.pth'

angle_version = 'le90'
model = dict(
    backbone=dict(
        _delete_=True,
        type='MM_VSSM',
        out_indices=(0, 1, 2, 3),
        pretrained=pretrained,
        # copied from https://github.com/MzeroMiko/VMamba/blob/main/detection/configs/vssm1/mask_rcnn_vssm_fpn_coco_tiny.py
        dims=96,
        # depths=(2, 2, 5, 2),
        depths=(2, 2, 8, 2),
        ssm_d_state=1,
        ssm_dt_rank="auto",
        # ssm_ratio=2.0,
        ssm_ratio=1.0,
        ssm_conv=3,
        ssm_conv_bias=False,
        forward_type="v05_noz", # v3_noz
        mlp_ratio=4.0,
        downsample_version="v3",
        patchembed_version="v2",
        drop_path_rate=0.2,
        norm_layer="ln2d",
    ),
    neck=dict(in_channels=[96, 192, 384, 768]))

# optimizer
optim_wrapper = dict(
    optimizer=dict(
        _delete_=True,
        type='AdamW',
        lr=1e-4,
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
    outfile_prefix='./work_dirs/Task1_oriented-rcnn-le90_vmamba-tiny_fpn_1x_dota')

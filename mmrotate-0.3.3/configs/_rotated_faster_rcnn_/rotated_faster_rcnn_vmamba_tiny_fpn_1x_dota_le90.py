_base_ = [
    '../_base_/_models_/rotated_faster_rcnn_r50_fpn.py',
    '../_base_/_datasets_/dotav1_ss_valmerge.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]

pretrained = 'data/pretrained/vssm_tiny_0230_ckpt_epoch_262.pth'

angle_version = 'le90'
model = dict(
    backbone=dict(
        _delete_=True,
        type='MM_VSSM',
        out_indices=(0, 1, 2, 3),
        pretrained=pretrained,
        # copied from classification/configs/vssm/vssm_tiny_224.yaml
        dims=96,
        depths=(2, 2, 5, 2),
        ssm_d_state=1,
        ssm_dt_rank="auto",
        ssm_ratio=2.0,
        ssm_conv=3,
        ssm_conv_bias=False,
        forward_type="v3noz",
        mlp_ratio=4.0,
        downsample_version="v3",
        patchembed_version="v2",
        drop_path_rate=0.2,
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
    lr=2e-4, # 1e-4,
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

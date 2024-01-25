_base_ = './rotated-faster-rcnn-le90_vmamba-tiny_fpn_1x_dota.py'

# configs from '../oriented_rcnn/oriented-rcnn-le90_swin-tiny_fpn_1x_dota.py'

pretrained = 'data/pretrained/vmamba_small_ckpt_epoch_238.pth'

depths = [2, 2, 27, 2]
model = dict(
    backbone=dict(
        depths=depths,
        with_cp=True,
        # init_cfg=dict(type='Pretrained', checkpoint=pretrained)))
        pretrained=pretrained))

# set all layers in backbone to lr_mult=0.1
# set all norm layers, position_embeding,
# query_embeding, level_embeding to decay_multi=0.0
backbone_norm_multi = dict(lr_mult=0.1, decay_mult=0.0)
backbone_embed_multi = dict(lr_mult=0.1, decay_mult=0.0)
embed_multi = dict(lr_mult=1.0, decay_mult=0.0)
custom_keys = {
    'backbone': dict(lr_mult=0.1, decay_mult=1.0),
    'backbone.patch_embed.norm': backbone_norm_multi,
    'backbone.norm': backbone_norm_multi,
    'absolute_pos_embed': backbone_embed_multi,
    'relative_position_bias_table': backbone_embed_multi,
    'query_embed': embed_multi,
    'query_feat': embed_multi,
    'level_embed': embed_multi
}
custom_keys.update({
    f'backbone.stages.{stage_id}.blocks.{block_id}.norm': backbone_norm_multi
    for stage_id, num_blocks in enumerate(depths)
    for block_id in range(num_blocks)
})
custom_keys.update({
    f'backbone.stages.{stage_id}.downsample.norm': backbone_norm_multi
    for stage_id in range(len(depths) - 1)
})

# optimizer
optim_wrapper = dict(
    paramwise_cfg=dict(custom_keys=custom_keys, norm_decay_mult=0.0))

test_evaluator = dict(
    outfile_prefix='./work_dirs/Task1_rotated-faster-rcnn-le90_vmamba-small_fpn_1x_dota')


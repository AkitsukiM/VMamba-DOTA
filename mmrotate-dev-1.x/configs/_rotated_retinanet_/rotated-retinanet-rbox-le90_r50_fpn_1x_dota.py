_base_ = [
    '../_base_/_models_/rotated-retinanet-rbox-le90_r50_fpn.py',
    '../_base_/_datasets_/dota_ss_valmerge.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]

# optim_wrapper = dict(optimizer=dict(lr=0.005))
optim_wrapper = dict(optimizer=dict(lr=0.020))

# NOTE
# original recommend: batch_size=1*2, lr=0.005
# if with 4*A100 GPU: batch_size=4*4, lr=0.020 (?)
train_dataloader = dict(
    batch_size=4,
    num_workers=4)

test_evaluator = dict(
    outfile_prefix='./work_dirs/Task1_rotated-retinanet-rbox-le90_r50_fpn_1x_dota')

_base_ = [
    '../_base_/_models_/oriented_rcnn_r50_fpn.py',
    '../_base_/_datasets_/dotav1_ss_valmerge.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4)

# NOTE
# original recommend: batch_size=1*2, lr=0.005
# if with 4*A100 GPU: batch_size=4*4, lr=0.020 (?)
optimizer=dict(lr=0.020)

# Copyright (c) OpenMMLab. All rights reserved.
# Default runtime settings for MapTR training

default_scope = 'mmdet3d'

# Hooks
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        max_keep_ckpts=3,
        save_best='auto'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='Det3DVisualizationHook'))

# Environment
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

# Logging
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='Det3DLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer')

log_processor = dict(
    type='LogProcessor',
    window_size=50,
    by_epoch=True)

log_level = 'INFO'
load_from = None
resume = False

# Training and testing config
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=24,
    val_interval=1)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# Copyright (c) OpenMMLab. All rights reserved.
"""
MapTR Example Configuration for Argoverse2
This is a minimal example configuration for testing the migration.
"""

_base_ = ['./_base_/default_runtime.py']

default_scope = 'mmdet3d'

# Model configuration
model = dict(
    type='MapTR',
    use_grid_mask=True,
    video_test_mode=False,
    
    # Image backbone
    img_backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN2d', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    
    # Image neck
    img_neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',
        num_outs=4,
        relu_before_extra_convs=True),
    
    # MapTR Head
    pts_bbox_head=dict(
        type='MapTRHead',
        num_queries=100,
        num_classes=3,  # divider, ped_crossing, boundary
        in_channels=256,
        sync_cls_avg_factor=True,
        code_size=2,
        # More head configuration...
    ),
    
    # Training and testing configuration
    train_cfg=dict(
        pts=dict(
            grid_size=[512, 512, 1],
            voxel_size=[0.2, 0.2, 8],
            point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
            out_size_factor=4,
            assigner=dict(
                type='MapTRAssigner',
                cls_cost=dict(type='FocalLossCost', weight=2.0),
                reg_cost=dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                iou_cost=dict(type='IoUCost', iou_mode='giou', weight=2.0)))),
    
    test_cfg=dict(pts=dict(max_per_img=100))
)

# Dataset configuration
dataset_type = 'CustomAV2LocalMapDataset'
data_root = 'data/av2/'
ann_file = 'data/av2/av2_map_infos_train.pkl'
map_ann_file = 'data/av2/av2_map_ann.json'

# Dataset pipeline
train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='LoadAnnotations3D', with_bbox_3d=False, with_label_3d=False),
    dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(type='NormalizeMultiviewImage',
         mean=[123.675, 116.28, 103.53],
         std=[58.395, 57.12, 57.375],
         to_rgb=True),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=['divider', 'ped_crossing', 'boundary']),
    dict(type='CustomCollect3D', keys=['img', 'gt_vecs_pts_loc', 'gt_vecs_label'])
]

test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='NormalizeMultiviewImage',
         mean=[123.675, 116.28, 103.53],
         std=[58.395, 57.12, 57.375],
         to_rgb=True),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='MultiScaleFlipAug3D',
         img_scale=(1333, 800),
         pts_scale_ratio=1,
         flip=False,
         transforms=[
             dict(type='DefaultFormatBundle3D', class_names=['divider', 'ped_crossing', 'boundary'], with_label=False),
             dict(type='CustomCollect3D', keys=['img'])
         ])
]

# Data loaders
train_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=ann_file,
        map_ann_file=map_ann_file,
        pipeline=train_pipeline,
        test_mode=False,
        map_classes=['divider', 'ped_crossing', 'boundary'],
        queue_length=4,
        bev_size=(200, 200),
        pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
        fixed_ptsnum_per_line=20,
        eval_use_same_gt_sample_num_flag=True,
        padding_value=-10000,
    ))

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=ann_file.replace('train', 'val'),
        map_ann_file=map_ann_file.replace('train', 'val'),
        pipeline=test_pipeline,
        test_mode=True,
        map_classes=['divider', 'ped_crossing', 'boundary'],
    ))

test_dataloader = val_dataloader

# Evaluators
val_evaluator = dict(
    type='MapMetricWithGT',
    ann_file=map_ann_file.replace('train', 'val'),
    map_classes=('divider', 'ped_crossing', 'boundary'),
    fixed_num=20,
    eval_use_same_gt_sample_num_flag=True,
    pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
    metric=['chamfer'],
    prefix='AV2Map')

test_evaluator = val_evaluator

# Optimizer
optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=2e-4,
        weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),
        }),
    clip_grad=dict(max_norm=35, norm_type=2),
)

# Learning rate scheduler
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0 / 3,
        by_epoch=False,
        begin=0,
        end=500),
    dict(
        type='CosineAnnealingLR',
        by_epoch=True,
        begin=0,
        end=24,
        T_max=24,
        eta_min_ratio=1e-3)
]

# Training configuration
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=24,
    val_interval=1)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# Runtime settings
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        max_keep_ckpts=3,
        save_best='AV2Map/chamfer_mAP',
        rule='greater'),
    sampler_seed=dict(type='DistSamplerSeedHook'))

# Custom hooks
custom_hooks = []

# Load from checkpoint
load_from = None
resume = False

# Working directory
work_dir = './work_dirs/maptr_av2_example'

_base_ = [
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_1x.py',
    '../_base_/datasets/depth_hand_det.py', './rtmdet_tta.py'
]
model = dict(
    type='RTMDet',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[0., 0., 0.],
        std=[1., 1., 1.],
        bgr_to_rgb=False,
        batch_augments=None),
    backbone=dict(
        type='CSPNeXt',
        arch='P5',
        expand_ratio=0.5,
        deepen_factor=1,
        widen_factor=1,
        channel_attention=True,
        norm_cfg=dict(type='SyncBN'),
        act_cfg=dict(type='SiLU', inplace=True)),
    neck=dict(
        type='CSPNeXtPAFPN',
        in_channels=[256, 512, 1024],
        out_channels=256,
        num_csp_blocks=3,
        expand_ratio=0.5,
        norm_cfg=dict(type='SyncBN'),
        act_cfg=dict(type='SiLU', inplace=True)),
    bbox_head=dict(
        type='RTMDetSepBNHead',
        num_classes=2,
        in_channels=256,
        stacked_convs=2,
        feat_channels=256,
        anchor_generator=dict(
            type='MlvlPointGenerator', offset=0, strides=[8, 16, 32]),
        bbox_coder=dict(type='DistancePointBBoxCoder'),
        loss_cls=dict(
            type='QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
        with_objectness=False,
        exp_on_reg=True,
        share_conv=True,
        pred_kernel_size=1,
        norm_cfg=dict(type='SyncBN'),
        act_cfg=dict(type='SiLU', inplace=True)),
    train_cfg=dict(
        assigner=dict(type='DynamicSoftLabelAssigner', topk=13),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=30000,
        min_bbox_size=0,
        score_thr=0.001,
        nms=dict(type='nms', iou_threshold=0.65),
        max_per_img=300),
)

train_pipeline = [
    dict(type='LoadDepthFromNPY'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='DepthToTripleChannel'),
    dict(type='CachedMosaic', img_scale=(640, 640), pad_val=0.0),
    dict(
        type='RandomResize',
        scale=(1280, 1280),
        ratio_range=(0.1, 2.0),
        keep_ratio=True,
        interpolation='nearest'),
    dict(type='RandomCrop', crop_size=(640, 640)),
    dict(type='RandomFlip', prob=0.0),
    dict(type='Pad', size=(640, 640), pad_val=dict(img=(0., 0., 0.))),
    dict(type='PackDetInputs')
]

train_pipeline_stage2 = [
    dict(type='LoadDepthFromNPY'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='DepthToTripleChannel'),
    dict(
        type='RandomResize',
        scale=(640, 640),
        ratio_range=(0.1, 2.0),
        keep_ratio=True,
        interpolation='nearest'),
    dict(type='RandomCrop', crop_size=(640, 640)),
    dict(type='RandomFlip', prob=0.0),
    dict(type='Pad', size=(640, 640), pad_val=dict(img=(0., 0., 0.))),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadDepthFromNPY'),
    dict(type='DepthToTripleChannel'),
    dict(type='Resize', scale=(640, 640), keep_ratio=True, interpolation='nearest'),
    dict(type='Pad', size=(640, 640), pad_val=dict(img=(0., 0., 0.))),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

train_dataloader = dict(
    batch_size=10,
    num_workers=5,
    batch_sampler=None,
    pin_memory=True,
    dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=10, num_workers=5, dataset=dict(pipeline=test_pipeline))
test_dataloader = val_dataloader

max_epochs = 15
stage2_num_epochs = 2
base_lr = 0.001
interval = 1

train_cfg = dict(
    max_epochs=max_epochs,
    val_interval=interval,
    dynamic_intervals=[(max_epochs - stage2_num_epochs, 1)])

val_evaluator = dict(proposal_nums=(100, 1, 10))
test_evaluator = val_evaluator

# optimizer
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.05),
    paramwise_cfg=dict(
        norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True))

# learning rate
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0e-5,
        by_epoch=False,
        begin=0,
        end=1000),
    dict(
        type='MultiStepLR',
        begin=0,
        end=15,
        by_epoch=True,
        milestones=[7, 12],
        gamma=0.1)
    # dict(
    #     # use cosine lr from 150 to 300 epoch
    #     type='CosineAnnealingLR',
    #     eta_min=base_lr * 0.05,
    #     begin=max_epochs // 2,
    #     end=max_epochs,
    #     T_max=max_epochs // 2,
    #     by_epoch=True,
    #     convert_to_iter_based=True),
]

# hooks
default_hooks = dict(
    checkpoint=dict(
        interval=interval,
        max_keep_ckpts=20  # only keep latest 3 checkpoints
    ))
custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        update_buffers=True,
        priority=49),
    dict(
        type='PipelineSwitchHook',
        switch_epoch=max_epochs - stage2_num_epochs,
        switch_pipeline=train_pipeline_stage2)
]

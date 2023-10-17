_base_ = './rtmdet_s_8xb32-300e_ir_right_left.py'

checkpoint = 'work_dirs/rtmdet_tiny_ir_right_left2/epoch_50.pth'  # noqa

model = dict(
    backbone=dict(
        deepen_factor=0.167,
        widen_factor=0.375,
        init_cfg=dict(
            type='Pretrained', prefix='backbone.', checkpoint=checkpoint)),
    neck=dict(in_channels=[96, 192, 384], out_channels=96, num_csp_blocks=1),
    bbox_head=dict(in_channels=96, feat_channels=96, exp_on_reg=False))

train_pipeline = [
    dict(type='LoadImageFromFile', color_type='grayscale', backend_args={{_base_.backend_args}}),
    dict(type='SignalChannelToTripleChannel'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='CachedMosaic',
        img_scale=(640, 640),
        pad_val=114.0,
        max_cached_images=256,
        random_pop=False),
    dict(
        type='RandomResize',
        scale=(1280, 1280),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=(640, 640)),
    dict(type='YOLOXHSVRandomAug', hue_delta=0., saturation_delta=0.),
    dict(type='RandomFlip', direction='horizontal', prob=0.5),
    dict(type='Pad', size=(640, 640), pad_val=dict(img=(114, 114, 114))),
    dict(
        type='CachedMixUp',
        img_scale=(640, 640),
        ratio_range=(1.0, 1.0),
        max_cached_images=10,
        random_pop=False,
        pad_val=(114, 114, 114),
        prob=0.5),
    dict(type='PackDetInputs')
]

train_pipeline_stage2 = [
    dict(type='LoadImageFromFile', color_type='grayscale', backend_args={{_base_.backend_args}}),
    dict(type='SignalChannelToTripleChannel'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='RandomResize',
        scale=(640, 640),
        ratio_range=(0.1, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=(640, 640)),
    dict(type='YOLOXHSVRandomAug', hue_delta=0., saturation_delta=0.),
    dict(type='RandomFlip', direction='horizontal', prob=0.5),
    dict(type='Pad', size=(640, 640), pad_val=dict(img=(114, 114, 114))),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile', color_type='grayscale', backend_args={{_base_.backend_args}}),
    dict(type='SignalChannelToTripleChannel'),
    dict(type='Resize', scale=(640, 480), keep_ratio=True),
    dict(type='Pad', size=(640, 480), pad_val=dict(img=(114, 114, 114))),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

train_dataloader = dict(
    batch_size=64,
    num_workers=16,
    batch_sampler=None,
    pin_memory=True,
    dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=64, num_workers=8, dataset=dict(pipeline=test_pipeline))
test_dataloader = val_dataloader

max_epochs = 20
stage2_num_epochs = 2
base_lr = 0.004
interval = 1

train_cfg = dict(
    max_epochs=max_epochs,
    val_interval=interval,
    dynamic_intervals=[(max_epochs - stage2_num_epochs, 1)])

# learning rate
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0e-5,
        by_epoch=False,
        begin=0,
        end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=50,
        by_epoch=True,
        milestones=[10, 16],
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
        max_keep_ckpts=30  # only keep latest 3 checkpoints
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

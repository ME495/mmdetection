# dataset settings

metainfo = {
    'classes': ('right', 'left'),
    'palette': [
        (220, 20, 60),  (119, 11, 32)
    ]
}

# Example to use different file client
# Method 1: simply set the data root and let the file I/O module
# automatically infer from prefix (not support LMDB and Memcache yet)

# data_root = 's3://openmmlab/datasets/detection/coco/'

# Method 2: Use `backend_args`, `file_client_args` in versions before 3.0.0rc6
# backend_args = dict(
#     backend='petrel',
#     path_mapping=dict({
#         './data/': 's3://openmmlab/datasets/detection/',
#         'data/': 's3://openmmlab/datasets/detection/'
#     }))
backend_args = None

train_pipeline = [
    dict(type='LoadImageFromFile', color_type='grayscale', backend_args=backend_args),
    dict(type='SignalChannelToTripleChannel'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', prob=0.),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', color_type='grayscale', backend_args=backend_args),
    dict(type='SignalChannelToTripleChannel'),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

dataset_coco_ir_right_left_train = dict(
    type='CocoDataset',
    data_root='data/coco_ir_right_left/',
    ann_file='train.json',
    metainfo=metainfo,
    data_prefix=dict(img=''),
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=[],
    backend_args=backend_args
)
dataset_100DOH_ir_train = dict(
    type='CocoDataset',
    data_root='data/100DOH_ir/',
    ann_file='train.json',
    metainfo=metainfo,
    data_prefix=dict(img=''),
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=[],
    backend_args=backend_args
)
dataset_qiyuan_multi_view_train = dict(
    type='CocoDataset',
    data_root='data/qiyuan_multiview_right_left/',
    ann_file='train.json',
    metainfo=metainfo,
    data_prefix=dict(img=''),
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=[],
    backend_args=backend_args
)
dataset_qiyuan_ir_right_left_train = dict(
    type='CocoDataset',
    data_root='data/qiyuan_ir_right_left/',
    ann_file='train.json',
    metainfo=metainfo,
    data_prefix=dict(img=''),
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=[],
    backend_args=backend_args
)
dataset_train = dict(
    type='CombinedDataset',
    metainfo=metainfo,
    datasets=[
        dataset_coco_ir_right_left_train, 
        dataset_100DOH_ir_train,
        dataset_qiyuan_multi_view_train,
        dataset_qiyuan_ir_right_left_train
        ],
    pipeline=train_pipeline,
    test_mode=False
)

dataset_qiyuan_multi_view_val = dict(
    type='CocoDataset',
    data_root='data/qiyuan_multiview_right_left/',
    ann_file='val.json',
    metainfo=metainfo,
    data_prefix=dict(img=''),
    test_mode=True,
    pipeline=[],
    backend_args=backend_args
)
dataset_qiyuan_ir_right_left_val = dict(
    type='CocoDataset',
    data_root='data/qiyuan_ir_right_left/',
    ann_file='val.json',
    metainfo=metainfo,
    data_prefix=dict(img=''),
    test_mode=True,
    pipeline=[],
    backend_args=backend_args
)
dataset_val = dict(
    type='CombinedDataset',
    metainfo=metainfo,
    datasets=[dataset_qiyuan_multi_view_val],
    pipeline=test_pipeline,
    test_mode=True,
)

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dataset_train)
    # dataset=dataset_coco_ir_right_left_train)
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    # dataset=dataset_val)
    dataset=dataset_qiyuan_multi_view_val)
test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file='data/qiyuan_multiview_right_left/val.json',
    metric='bbox',
    format_only=False,
    backend_args=backend_args)
test_evaluator = val_evaluator

# inference on test dataset and
# format the output results for submission.
# test_dataloader = dict(
#     batch_size=1,
#     num_workers=2,
#     persistent_workers=True,
#     drop_last=False,
#     sampler=dict(type='DefaultSampler', shuffle=False),
#     dataset=dict(
#         type=dataset_type,
#         data_root=data_root,
#         ann_file=data_root + 'annotations/image_info_test-dev2017.json',
#         data_prefix=dict(img='test2017/'),
#         test_mode=True,
#         pipeline=test_pipeline))
# test_evaluator = dict(
#     type='CocoMetric',
#     metric='bbox',
#     format_only=True,
#     ann_file=data_root + 'annotations/image_info_test-dev2017.json',
#     outfile_prefix='./work_dirs/coco_detection/test')

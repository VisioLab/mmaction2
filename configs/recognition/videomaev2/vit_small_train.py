_base_ = ['../../_base_/default_runtime.py']

# model settings
model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='VisionTransformer',
        img_size=224,
        patch_size=16,
        embed_dims=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        num_frames=16,
        norm_cfg=dict(type='LN', eps=1e-6),
        pretrained = '/home/hamza/action_recognition/mmaction2/pretrained_weights/viT_small.pth',
        freeze=True,
    ),
    cls_head=dict(
        type='TimeSformerHead',
        num_classes=4,
        in_channels=384,
        average_clips='prob',
        topk=(1, 2),
        ),
    data_preprocessor=dict(
        type='ActionDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        format_shape='NCTHW'))



dataset_type = 'VideoDataset'
data_root = '../../datasets/ssv2_dataset/20bn-something-something-v2/'
data_root_val = '../../datasets/ssv2_dataset/20bn-something-something-v2/'
ann_file_train = './tools/data/sthv2/subsets/single_class_1/100_samples/train.txt'
ann_file_val = './tools/data/sthv2/subsets/single_class_1/100_samples/val.txt'

num_epochs = 80



train_pipeline = [
    dict(type='DecordInit'),
    dict(
        type='SampleFrames',
        clip_len=16,
        frame_interval=2,
    ),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    # dict(type='MultiScaleCrop', input_size=(224,224)),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]



val_pipeline = [
    dict(type='DecordInit'),
    dict(
        type='SampleFrames',
        clip_len=16,
        frame_interval=2,
        num_clips=3,
        test_mode=True),
    dict(type='DecordDecode'),
    # dict(type='Resize', scale=(-1, 224)),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='ThreeCrop', crop_size=224),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]

train_dataset_cfg =dict(
    type=dataset_type,
    ann_file=ann_file_train,
    data_prefix=dict(video=data_root),
    pipeline=train_pipeline)


val_dataset_cfg =dict(
    type=dataset_type,
    ann_file=ann_file_val,
    data_prefix=dict(video=data_root_val),
    pipeline=val_pipeline,
    test_mode=True)

train_dataloader = dict(
batch_size=16,
num_workers=8,
persistent_workers=True,
sampler=dict(type='DefaultSampler', shuffle=True),
dataset=train_dataset_cfg,
)

val_dataloader = dict(
    batch_size=2,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=val_dataset_cfg
)

train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=num_epochs, 
    val_interval=25,
    )

val_cfg = dict(type='ValLoop')

val_evaluator = [
    dict(type='AccMetric',metric_options=dict(top_k_accuracy=dict(topk=(1, 2)))),
    dict(type='ConfusionMatrix')
]


param_scheduler = dict(
    type='CosineAnnealingLR',
    by_epoch=True,
    T_max=num_epochs)


# optimizer
optim_wrapper = dict(
    optimizer=dict(
        type='SGD',
        lr=0.01,
        momentum=0.9,
        weight_decay=0.0001),
)

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook',save_best='auto')
)
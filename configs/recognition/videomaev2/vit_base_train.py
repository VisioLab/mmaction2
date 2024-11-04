_base_ = ['../../_base_/default_runtime.py']

# model settings
model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='VisionTransformer',
        img_size=224,
        patch_size=16,
        embed_dims=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        num_frames=16,
        drop_path_rate=0.1,
        norm_cfg=dict(type='LN', eps=1e-6),
        pretrained = '/home/hamza/action_recognition/mmaction2/pretrained_weights/viT_base.pth',
        freeze=True,
    ),
    cls_head=dict(
        type='TimeSformerHead',
        num_classes=4,
        in_channels=768,
        average_clips='prob',
        label_smooth_eps = 0.1,
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
tags = {
    "Issue_id":"AFC-3691",
    # "Model":"VTN_RES50",
    }


file_client_args = dict(io_backend='disk')
train_pipeline = [
    dict(type='DecordInit',**file_client_args),
    dict(
        type='SampleFrames',
        clip_len=16,
        frame_interval=2,
    ),
    # dict(type='UniformSampleFrames',clip_len=16,num_clips=1),
    dict(type='DecordDecode'),
    dict(type='PytorchVideoWrapper', op='RandAugment', magnitude=0, num_layers=4),
    dict(type='Resize', scale=(-1, 256)),
    dict(
        type='MultiScaleCrop',
        input_size=224,
        scales=(1, 0.875, 0.75, 0.66),
        random_crop=False,
        max_wh_scale_gap=1),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]

val_pipeline = [
    dict(type='DecordInit',**file_client_args),
    # dict(type='UniformSampleFrames',clip_len=16,num_clips=2,test_mode=True),
    dict(
        type='SampleFrames',
        clip_len=16,
        frame_interval=2,
        num_clips=3,
        test_mode=True),
    dict(type='DecordDecode'),
    
    dict(type='Resize', scale=(-1, 224)),
    # dict(type='Resize', scale=(224, 224), keep_ratio=False),
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
    val_interval=5,
    )

val_cfg = dict(type='ValLoop')

val_evaluator = [
    dict(type='AccMetric',metric_options=dict(top_k_accuracy=dict(topk=(1, 2)))),
]


param_scheduler = dict(
    type='CosineAnnealingLR',
    by_epoch=True,
    T_max=num_epochs)



base_lr = 3e-4
optim_wrapper = dict(
    optimizer=dict(
        type='AdamW', lr=base_lr, betas=(0.9, 0.999), weight_decay=0.05))


default_hooks = dict(
    checkpoint=dict(type='CheckpointHook',save_best='auto')
)



custom_hooks = [
    dict(type ='ConfusionMatrixHook',
         class_map="./tools/data/sthv2/subsets/class_map.json"),
    dict(
        type="MLflowHook",
        tags=tags,
        ),
    ]
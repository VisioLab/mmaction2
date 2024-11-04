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
data_root_val = '../../datasets/ssv2_dataset/20bn-something-something-v2/'
ann_file_val = './tools/data/sthv2/subsets/single_class_1/200_samples/val.txt'


file_client_args = dict(io_backend='disk')

test_pipeline = [
    dict(type='DecordInit',**file_client_args),
    dict(
        type='SampleFrames',
        clip_len=16,
        frame_interval=2,
        num_clips=2,
        test_mode=True),
    dict(type='DecordDecode'),
    
    dict(type='Resize', scale=(-1, 224)),
    dict(type='ThreeCrop', crop_size=224),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]

test_dataset_cfg =dict(
    type=dataset_type,
    ann_file=ann_file_val,
    data_prefix=dict(video=data_root_val),
    pipeline=test_pipeline,
    test_mode=True)

test_dataloader = dict(
    batch_size=2,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=test_dataset_cfg
)

test_cfg = dict(type='TestLoop')

test_evaluator = [
    dict(type='AccMetric',metric_options=dict(top_k_accuracy=dict(topk=(1, 2)))),
    dict(type='ConfusionMatrix')
]

custom_hooks = [dict(type ='MisclassificationHook',
                     enable=True,
                     interval=1,
                     class_map="./tools/data/sthv2/subsets/class_map.json"
                     ),
                ]
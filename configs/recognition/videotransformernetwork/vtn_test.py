_base_ = [
    '../../_base_/models/vtn_vit_b.py', '../../_base_/default_runtime.py'
]


dataset_type = 'VideoDataset'
data_root_val = '../../datasets/ssv2_dataset/20bn-something-something-v2/'
ann_file_val = './tools/data/sthv2/subsets/single_class_1/200_samples/val.txt'

tags={'Issue_id':"test"}


file_client_args = dict(io_backend='disk')

test_pipeline = [
    dict(type='DecordInit', **file_client_args),
    # dict(type='SampleFrames', clip_len=250, frame_interval=1, num_clips=1,test_mode=True),
    dict(type='UniformSampleFrames',clip_len=8,num_clips=1,test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]
test_dataset_cfg =dict(
    type=dataset_type,
    ann_file=ann_file_val,
    data_prefix=dict(video=data_root_val),
    pipeline=test_pipeline,
    test_mode=True
    )

test_dataloader = dict(
batch_size=16,
num_workers=8,
persistent_workers=True,
sampler=dict(type='DefaultSampler', shuffle=False),
dataset=test_dataset_cfg,
)

test_cfg  = dict(type='TestLoop')

test_evaluator = [
    dict(type='AccMetric',metric_options=dict(top_k_accuracy=dict(topk=(1, 2))),prefix='val')
]

custom_hooks = [
    dict(type="MLflowHook",
         tags=tags,
        ),
    dict(type ='MisclassificationHook',
    interval=1,
    class_map="./tools/data/sthv2/subsets/class_map.json"
    ),
    dict(type ='ConfusionMatrixHook',
         class_map="./tools/data/sthv2/subsets/class_map.json"
         ),
]
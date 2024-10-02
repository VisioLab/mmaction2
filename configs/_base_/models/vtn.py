
model = dict(
    type='VTN',
    backbone=dict(
        type='RESNET_BACKBONE',
        depth=50, 
        pretrained='/mmaction2/pretrained_weights/resnet50-11ad3fa6.pth',
        frozen_stages=4, ##freeze all stages
                    
    ),
    neck=dict(
        type='VTNLongformerModel',
        embed_dim=2048,
        max_position_embeddings=288,
        num_attention_heads=16,
        num_hidden_layers=1,
        attention_mode= 'sliding_chunks',
        pad_token_id=-1,
        attention_window=[32],
        intermediate_size=4096,
        attention_probs_dropout_prob=0.1,
        hidden_dropout_prob=0.1
    ),
    cls_head=dict(
        type='MLP',
        in_channels= 2048,
        hidden_channels= 768,
        dropout_rate= 0.5,
        num_classes=4,
        init_cfg = dict(type='Normal',mean=0,std=0.02,layer='Linear'),
    ),
    data_preprocessor=dict(
        type='ActionDataPreprocessor',
        combine_frame_inds=True,
        mean=[123.675, 116.28, 103.53], ## ImageNet mean
        std=[58.395, 57.12, 57.375], ## ImageNet std
        format_shape='NCHW'
        )
)
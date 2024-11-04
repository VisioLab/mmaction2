
model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='ViT2D',
        pretrained='./pretrained_weights/vit-b_2d.pth',
        freeze=True, ## freeze all stages
                    
    ),
    neck=dict(
        type='VTNLongformerModel',
        embed_dim=768,
        max_position_embeddings=288,
        num_attention_heads=12,
        num_hidden_layers=1,
        attention_mode= 'sliding_chunks',
        pad_token_id=-1,
        attention_window=[18],
        intermediate_size=3072,
        attention_probs_dropout_prob=0.1,
        hidden_dropout_prob=0.1,
    ),
    cls_head=dict(
        type='MLP',
        in_channels= 768,
        label_smooth_eps = 0.01,
        dropout_rate= 0.0,
        num_classes=4,
        average_clips='prob',
        topk=(1, 2),
    ),
    data_preprocessor=dict(
        type='ActionDataPreprocessor',
        combine_frame_inds=False,
        mean=[123.675, 116.28, 103.53], ## ImageNet mean
        std=[58.395, 57.12, 57.375], ## ImageNet std
        format_shape='NCTHW'
        ),
)
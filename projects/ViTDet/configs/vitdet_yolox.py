_base_ = [
    '/home/mlavery/scalemae_docker/mmdetection/projects/ViTDINO_FSDP/lsj-100e_coco-instance.py',
]

from mmdet.models import ViTMAE
from functools import partial
from torch.nn import LayerNorm as LN
from mmengine.model.weight_init import PretrainedInit

custom_imports = dict(imports=['projects.ViTDet.vitdet'])

backbone_norm_cfg = dict(type='LN', requires_grad=True, eps=1e-6)
norm_cfg = dict(type='LN2d', requires_grad=True)
image_size = (400, 400)
batch_augments = [
    dict(type='BatchFixedSizePad', size=image_size, pad_mask=True)
]

model = dict(
    type='YOLOX',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        pad_size_divisor=32,
        batch_augments=batch_augments),
    backbone=dict(
        type=ViTMAE,
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(LN, eps=1e-6),
        img_size=400,
        init_cfg=dict(type=PretrainedInit, checkpoint='/home/mlavery/scalemae_docker/weights/scalemae-vitlarge-unwrapped.pth')
    ),
    neck=dict(
        type='SimpleFPNYolo',
        backbone_channel=1024,
        in_channels=[256, 512, 1024],
        out_channels=256,
        num_outs=4,
        norm_cfg=norm_cfg
    ),
    bbox_head=dict(
        type='YOLOXHead',
        num_classes=60,
        in_channels=256,
        feat_channels=256,
        stacked_convs=2,
        strides=(4, 8, 16),
        use_depthwise=False,
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='Swish'),
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            reduction='sum',
            loss_weight=1.0),
        loss_bbox=dict(
            type='IoULoss',
            mode='square',
            eps=1e-16,
            reduction='sum',
            loss_weight=5.0),
        loss_obj=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            reduction='sum',
            loss_weight=1.0),
        loss_l1=dict(type='L1Loss', reduction='sum', loss_weight=1.0)),
    train_cfg=dict(assigner=dict(type='SimOTAAssigner', center_radius=2.5)),
    # In order to align the source code, the threshold of the val phase is
    # 0.01, and the threshold of the test phase is 0.001.
    test_cfg=dict(score_thr=0.01, nms=dict(type='nms', iou_threshold=0.65)))

base_lr = 0.01
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='SGD', lr=base_lr, momentum=0.9, weight_decay=5e-4,
        nesterov=True),
    paramwise_cfg=dict(norm_decay_mult=0., bias_decay_mult=0.))

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (4 samples per GPU)
auto_scale_lr = dict(base_batch_size=32)


# optim_wrapper = dict(
#     type='AmpOptimWrapper',
#     constructor='LayerDecayOptimizerConstructor',
#     paramwise_cfg={
#         'decay_rate': 0.7,
#         'decay_type': 'layer_wise',
#         'num_layers': 12,
#     },
#     optimizer=dict(
#         type='AdamW',
#         lr=0.0001,
#         betas=(0.9, 0.999),
#         weight_decay=0.1,
#     ))

# custom_hooks = [dict(type='Fp16CompresssionHook')]

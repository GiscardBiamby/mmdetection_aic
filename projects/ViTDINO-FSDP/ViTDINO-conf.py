_base_ = [
    './lsj-100e_coco-instance.py',
]

from torch.nn.modules.activation import ReLU
from torch.nn import LayerNorm as LN

from mmdet.models import (DETR, ChannelMapper, DetDataPreprocessor, DETRHead, ViT)
from mmdet.models.losses.cross_entropy_loss import CrossEntropyLoss
from mmdet.models.losses.smooth_l1_loss import L1Loss
from mmdet.models.losses.iou_loss import GIoULoss
from mmdet.models.task_modules import (BBoxL1Cost, ClassificationCost,
                                       HungarianAssigner, IoUCost)
from mmengine.model.weight_init import PretrainedInit

image_size = (256, 256)
backbone_norm_cfg = dict(type=LN, requires_grad=True)
norm_cfg = dict(type="LN2d", requires_grad=True)
batch_augments = [
    dict(type='BatchFixedSizePad', size=image_size, pad_mask=True)
]

data_preprocessor = dict(
    type=DetDataPreprocessor,
    mean=[0, 0, 0],
    std=[255., 255., 255.],
    bgr_to_rgb=True,
    pad_size_divisor=32)

model = dict(
    type=DETR,
    num_queries=100,
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type=ViT,
        img_size=256,
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        drop_path_rate=0.1,
        window_size=14,
        mlp_ratio=4,
        qkv_bias=True,
        norm_cfg=backbone_norm_cfg,
        window_block_indexes=[
            0,
            1,
            3,
            4,
            6,
            7,
            9,
            10,
        ],
        use_rel_pos=True,
        init_cfg=dict(
            type=PretrainedInit, checkpoint='/home/mlavery/scalemae_docker/weights/vitlarge-wrapped.pt')),
    neck=dict(
        type=ChannelMapper,
        in_channels=[1024],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=None,
        num_outs=1),
    encoder=dict(  # DetrTransformerEncoder
        num_layers=6,
        layer_cfg=dict(  # DetrTransformerEncoderLayer
            self_attn_cfg=dict(  # MultiheadAttention
                embed_dims=256,
                num_heads=8,
                dropout=0.1,
                batch_first=True),
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=2048,
                num_fcs=2,
                ffn_drop=0.1,
                act_cfg=dict(type=ReLU, inplace=True)))),
    decoder=dict(  # DetrTransformerDecoder
        num_layers=6,
        layer_cfg=dict(  # DetrTransformerDecoderLayer
            self_attn_cfg=dict(  # MultiheadAttention
                embed_dims=256,
                num_heads=8,
                dropout=0.1,
                batch_first=True),
            cross_attn_cfg=dict(  # MultiheadAttention
                embed_dims=256,
                num_heads=8,
                dropout=0.1,
                batch_first=True),
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=2048,
                num_fcs=2,
                ffn_drop=0.1,
                act_cfg=dict(type=ReLU, inplace=True))),
        return_intermediate=True),
    positional_encoding=dict(num_feats=128, normalize=True),
    bbox_head=dict(
        type=DETRHead,
        num_classes=80,
        embed_dims=256,
        loss_cls=dict(
            type=CrossEntropyLoss,
            bg_cls_weight=0.1,
            use_sigmoid=False,
            loss_weight=1.0,
            class_weight=1.0),
        loss_bbox=dict(type=L1Loss, loss_weight=5.0),
        loss_iou=dict(type=GIoULoss, loss_weight=2.0)),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type=HungarianAssigner,
            match_costs=[
                dict(type=ClassificationCost, weight=1.),
                dict(type=BBoxL1Cost, weight=5.0, box_format='xywh'),
                dict(type=IoUCost, iou_mode='giou', weight=2.0)
            ])),
    test_cfg=dict(max_per_img=100))

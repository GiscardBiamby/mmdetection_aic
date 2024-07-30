from mmengine.config import read_base

with read_base():
    # pylint: disable-next=E0402,W0401,W0614
    from ....configs._base_.models.maskrcnn_r50_fpn import *   # noqa: F401,F403,W0401
    # pylint: disable-next=E0402,W0401,W0614
    from ...ViTDINO_FSDP.xview_dataset_bs8 import *   # noqa: F401,F403,W0401


from functools import partial

from mmengine.model.weight_init import PretrainedInit
from mmengine.optim import AmpOptimWrapper
from torch.nn import LayerNorm as LN
from torch.optim import AdamW

from mmdet.models import ViTMAE
from mmdet.models.data_preprocessors import BatchFixedSizePad
from mmdet.models.roi_heads.bbox_heads import Shared4Conv1FCBBoxHead
from projects.ViTDet.vitdet import Fp16CompresssionHook, LayerDecayOptimizerConstructor
from projects.ViTDet.vitdet.simple_fpn import SimpleFPN
from projects.ViTDet.vitdet.util import _layernorm


custom_imports = dict(imports=['projects.ViTDet.vitdet'])

backbone_norm_cfg = dict(type='LN', requires_grad=True, eps=1e-6)
norm_cfg = dict(type='LN2d', requires_grad=True)
image_size = (400, 400)
batch_augments = [
    dict(type=BatchFixedSizePad, size=image_size, pad_mask=True)
]

# model settings
model.merge(dict(
    data_preprocessor=dict(pad_size_divisor=32, batch_augments=batch_augments),
    backbone=dict(
        _delete_=True,
        type=ViTMAE,
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        # norm_layer=partial(LN, eps=1e-6),
        # When using this config the partial throws an error (after switching to use
        # `with read_base()` at the top). Hacky solution, not ideal because the eps arg
        # is hardcoded in _layer_norm:
        norm_layer=_layernorm,
        img_size=224,
        init_cfg=dict(type=PretrainedInit, checkpoint='/home/mlavery/scalemae_docker/weights/scalemae-vitlarge-unwrapped.pth')),
    neck=dict(
        _delete_=True,
        type=SimpleFPN,
        backbone_channel=1024,
        in_channels=[256, 512, 1024, 1024],
        out_channels=256,
        num_outs=5,
        norm_cfg=norm_cfg),
    rpn_head=dict(
        num_convs=2,
    ),
    roi_head=dict(
        bbox_head=dict(
            type=Shared4Conv1FCBBoxHead,
            conv_out_channels=256,
            norm_cfg=norm_cfg,
            num_classes=60
        ),
        mask_head=dict(
            num_classes=60
        )
    ),
))


optim_wrapper = dict(
    type=AmpOptimWrapper,
    constructor=LayerDecayOptimizerConstructor,
    paramwise_cfg={
        'decay_rate': 0.7,
        'decay_type': 'layer_wise',
        'num_layers': 24,
    },
    optimizer=dict(
        type=AdamW,
        lr=0.0001,
        betas=(0.9, 0.999),
        weight_decay=0.1,
    ))
custom_hooks = [dict(type=Fp16CompresssionHook)]
auto_scale_lr = dict(base_batch_size=64)

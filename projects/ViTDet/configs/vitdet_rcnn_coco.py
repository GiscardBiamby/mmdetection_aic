# _base_ = [
#     '../../../configs/_base_/models/mask-rcnn_r50_fpn.py',
#     './coco_dataset.py',
# ]
# from mmdet.models.detectors import MaskRCNN

from mmengine.config import read_base

with read_base():
    # pylint: disable-next=E0402,W0401,W0614
    from ....configs._base_.models.maskrcnn_r50_fpn import *  # noqa: F401,F403,W0401,W0614

    # pylint: disable-next=E0402,W0401,W0614
    from ...ViTDet.configs.coco_dataset import *  # noqa: F401,F403,W0401,W0614

from torch.nn import LayerNorm as LN

from mmdet.models.backbones.vit import LN2d, ViT
from mmdet.models.data_preprocessors import BatchFixedSizePad
from mmdet.models.roi_heads.bbox_heads import Shared4Conv1FCBBoxHead
from projects.ViTDet.vitdet import Fp16CompresssionHook
from projects.ViTDet.vitdet.simple_fpn import SimpleFPN

custom_imports = dict(imports=["projects.ViTDet.vitdet"])

backbone_norm_cfg = dict(type=LN, requires_grad=True)
norm_cfg = dict(type=LN2d, requires_grad=True)
image_size = (512, 512)
batch_augments = [dict(type=BatchFixedSizePad, size=image_size, pad_mask=True)]

# # model settings
model.merge(
    dict(
        data_preprocessor=dict(pad_size_divisor=32, batch_augments=batch_augments),
        backbone=dict(
            _delete_=True,
            type=ViT,
            img_size=1024,
            patch_size=16,
            embed_dim=768,
            depth=12,
            num_heads=12,
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
                type="Pretrained",
                checkpoint="/home/mlavery/scalemae_docker/weights/vitdet_mask-rcnn_vit-b-mae_lsj-100e_20230328_153519-e15fe294.pth",
            ),
        ),
        neck=dict(
            _delete_=True,
            type=SimpleFPN,
            backbone_channel=768,
            in_channels=[192, 384, 768, 768],
            out_channels=256,
            num_outs=5,
            norm_cfg=norm_cfg,
        ),
        rpn_head=dict(num_convs=2),
        roi_head=dict(
            bbox_head=dict(
                type=Shared4Conv1FCBBoxHead, conv_out_channels=256, norm_cfg=norm_cfg
            ),
            mask_head=dict(norm_cfg=norm_cfg),
        ),
    )
)

# Don't know if these are the right settings, but I hadto add this to fix an error
# message, it needs optim_wrapper to be defined:
optim_wrapper = dict(
    type="AmpOptimWrapper",
    constructor="LayerDecayOptimizerConstructor",
    paramwise_cfg={
        "decay_rate": 0.7,
        "decay_type": "layer_wise",
        "num_layers": 12,
    },
    optimizer=dict(
        type="AdamW",
        lr=0.0001,
        betas=(0.9, 0.999),
        weight_decay=0.1,
    ),
)
custom_hooks = [dict(type=Fp16CompresssionHook)]

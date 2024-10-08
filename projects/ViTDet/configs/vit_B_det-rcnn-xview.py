from mmengine.config import read_base

with read_base():
    # pylint: disable-next=E0402,W0401,W0614
    from ....configs._base_.models.maskrcnn_r50_fpn import *  # noqa: F401,F403,W0401

    # pylint: disable-next=E0402,W0401,W0614
    from ...ViTDINO_FSDP.xview_dataset import *  # noqa: F401,F403,W0401

from torch.nn import LayerNorm as LN

from mmdet.models.backbones.vit import LN2d, ViT
from mmdet.models.data_preprocessors import BatchFixedSizePad
from mmdet.models.roi_heads.bbox_heads import Shared4Conv1FCBBoxHead
from projects.ViTDet.vitdet import Fp16CompresssionHook
from projects.ViTDet.vitdet.simple_fpn import SimpleFPN

custom_imports = dict(imports=["projects.ViTDet.vitdet"])

backbone_norm_cfg = dict(type=LN, requires_grad=True, eps=1e-6)
norm_cfg = dict(type=LN2d, requires_grad=True)
image_size = (512, 512)
batch_augments = [dict(type=BatchFixedSizePad, size=image_size, pad_mask=True)]

# model settings
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
                checkpoint="../weights/vitdet_mask-rcnn_vit-b-mae_lsj-100e_20230328_153519-e15fe294.pth",
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
        rpn_head=dict(
            num_convs=2,
        ),
        roi_head=dict(
            bbox_head=dict(
                type=Shared4Conv1FCBBoxHead,
                conv_out_channels=256,
                norm_cfg=norm_cfg,
                num_classes=60,
            ),
            mask_head=dict(num_classes=60),
        ),
    )
)

custom_hooks = [dict(type=Fp16CompresssionHook)]
auto_scale_lr = dict(base_batch_size=64)

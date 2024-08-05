from mmengine.config import read_base

with read_base():
    # pylint: disable-next=E0402,W0401,W0614
    from mmdet.configs._base_.models.mask_rcnn_r50_fpn import *  # noqa: F401,F403,W0401,W0614

    # pylint: disable-next=E0402,W0401,W0614
    from xview_dataset_dmp_100e import *  # noqa: F401,F403,W0401,W0614

from torch.nn import LayerNorm as LN

from mmdet.models.backbones.vit import LN2d
from mmdet.models.backbones import ViTGSD
from mmdet.models.data_preprocessors import BatchFixedSizePad
from mmdet.models.roi_heads.bbox_heads import Shared4Conv1FCBBoxHead
from projects.ViTDet.vitdet import Fp16CompresssionHook
from projects.ViTDet.vitdet.simple_fpn import SimpleFPN

custom_imports = dict(imports=["projects.ViTDet.vitdet"])

backbone_norm_cfg = dict(type=LN, requires_grad=True, eps=1e-6)
norm_cfg = dict(type=LN2d, requires_grad=True)
image_size = (400, 400)
batch_augments = [dict(type=BatchFixedSizePad, size=image_size, pad_mask=True)]

# model settings
model.merge(
    dict(
        data_preprocessor=dict(pad_size_divisor=32, batch_augments=batch_augments),
        backbone=dict(
            _delete_=True,
            type=ViTGSD,
            img_size=image_size[0],
            patch_size=16,
            embed_dim=1024,
            depth=24,
            num_heads=16,
            drop_path_rate=0.4,
            window_size=14,
            mlp_ratio=4,
            qkv_bias=True,
            norm_cfg=backbone_norm_cfg,
            window_block_indexes=list(range(0, 5))
            + list(range(6, 11))
            + list(range(12, 17))
            + list(range(18, 23)),
            use_rel_pos=True,
            init_cfg=dict(
                type="Pretrained", checkpoint="weights/scalemae-vitlarge-800.pth"
            ),
        ),
        neck=dict(
            _delete_=True,
            type=SimpleFPN,
            backbone_channel=1024,
            in_channels=[256, 512, 1024, 1024],
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
                num_classes=67,
            ),
            mask_head=dict(num_classes=67),
        ),
    )
)

custom_hooks = [dict(type=Fp16CompresssionHook)]

from mmengine.config import read_base

with read_base():
    # pylint: disable-next=E0402,W0401,W0614
    from mmdet.configs._base_.models.mask_rcnn_r50_fpn import *  # noqa: F401,F403,W0401,W0614

    # pylint: disable-next=E0402,W0401,W0614
    from ..datasets.xview400_dataset_parent import *  # noqa: F401,F403,W0401,W0614

from torch.nn import LayerNorm as LN

from mmdet.models.backbones import ViTGSD
from mmdet.models.backbones.vit import LN2d
from mmdet.models.data_preprocessors import BatchFixedSizePad
from mmdet.models.roi_heads.bbox_heads import Shared4Conv1FCBBoxHead
from projects.ViTDet.vitdet import Fp16CompresssionHook
from projects.ViTDet.vitdet.simple_fpn import SimpleFPN

custom_imports = dict(imports=["projects.ViTDet.vitdet"])

backbone_norm_cfg = dict(type=LN, requires_grad=True, eps=1e-6)
norm_cfg = dict(type=LN2d, requires_grad=True)
image_size = (400, 400)
batch_augments = [dict(type=BatchFixedSizePad, size=image_size, pad_mask=True)]
# NUM_CLASSES=7 # This is defined in the xviewXX_dataset_parents/child

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
                num_classes=NUM_CLASSES,
            ),
            mask_head=dict(num_classes=NUM_CLASSES),
        ),
    )
)
# * An epoch on 6x4090 takes ~12 minutes, and eval takes ~5min. We don't want eval to
#   take up a significant portion of training time so eval interval of 5 seems like the
#   lowest reasonable setting.
epochs = 60
val_interval = 5
param_scheduler = [
    dict(type="LinearLR", start_factor=0.001, begin=0, end=250, by_epoch=False),
    dict(
        type="MultiStepLR",
        begin=0,
        end=epochs,
        milestones=[40, 50, 55],
        gamma=0.5,
        by_epoch=True,
    ),
]
train_cfg = dict(
    type="EpochBasedTrainLoop",
    max_epochs=epochs,
    val_interval=val_interval,
)
custom_hooks = [dict(type=Fp16CompresssionHook)]

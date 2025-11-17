_base_ = [
    "../../../configs/_base_/models/retinanet_r50_fpn.py",
    # "../../../configs/_base_/schedules/schedule_1x.py",
    # "../../../configs/_base_/default_runtime.py",
    "./_base_/schedules/xv_schedule_1x.py",
    "./_base_/default_runtime.py",
    "./datasets/xview_200_0.py",
]
custom_imports = dict(
    imports=[
        "projects.nadirdet.nadirdet",
        "mmdet.visualization",
        "mmdet.visualization.local_visualizer",
        "projects.nadirdet.nadirdet.xview_coco_metric",
    ],
    allow_failed_imports=False,
)
NUM_CLASSES = 60


load_from = "checkpoints/retinanet_r101_fpn_1x_coco_20200130-7a93545f.pth"

model = dict(
    # Disable redundant backbone init since we load a full RetinaNet checkpoint
    backbone=dict(
        init_cfg=None,
        depth=101,
        # init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    ),
    bbox_head=dict(num_classes=NUM_CLASSES),
)


# * Base config
# *   - `enable` means enable scaling LR automatically
# *   - `base_batch_size` = (4 GPUs) x (128 samples per GPU).
# * effective lr = base_lr * scale_factor
# * scale_factor = actual_batch_size / base_batch_size
# * How the scaling works if you diverge from GPU count. E.g., if you use 2 GPUs:
# * global batch = 2 × 128 = 256, scale factor = 256 / 512 = 0.5
auto_scale_lr = dict(base_batch_size=256, enable=True)


# TODO: implement SyncBN if needed and doing multi-gpu training
# norm_cfg = dict(type='SyncBN', requires_grad=True, eps=1e-3, momentum=0.01)


# * From ViTDet: An epoch on 6x4090 takes ~12 minutes, and eval takes ~5min. We don't want eval to
# *   take up a significant portion of training time so eval interval of 5 seems like the
# *   lowest reasonable setting.
max_epochs = 90
val_interval = 5

default_hooks = dict(
    checkpoint=dict(
        interval=val_interval,
    ),
)
train_cfg = dict(
    max_epochs=max_epochs,
    val_interval=val_interval,
)

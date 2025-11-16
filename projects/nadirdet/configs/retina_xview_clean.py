_base_ = [
    "../../../configs/_base_/models/retinanet_r50_fpn.py",
    "../../../configs/_base_/schedules/schedule_1x.py",
    "../../../configs/_base_/default_runtime.py",
    "./datasets/xview_200_0.py",
]
custom_imports = dict(
    imports=[
        "projects.nadirdet.nadirdet",
        "mmdet.visualization",
        "mmdet.visualization.local_visualizer",
    ],
    allow_failed_imports=False,
)
NUM_CLASSES=60



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

# * ViTDet: An epoch on 6x4090 takes ~12 minutes, and eval takes ~5min. We don't want eval to
# *   take up a significant portion of training time so eval interval of 5 seems like the
# *   lowest reasonable setting.
epochs = 90
val_interval = 5

# TODO: implement SyncBN if needed and doing multi-gpu training
# norm_cfg = dict(type='SyncBN', requires_grad=True, eps=1e-3, momentum=0.01)

# TODO: Check on cudnn best practice
# * lets cuDNN autotune conv implementations. Can accelerate fix-size training. Don't use if input
# sizes vary a lot, e.g., if you have continuous variation. For example if you us e randomResize with
# (e.g. ratio_range=(0.8, 1.2)
# env_cfg = dict(cudnn_benchmark=True)

# * Optimizer and learning rate scheduler
param_scheduler = [
    dict(type="LinearLR", start_factor=0.001, begin=0, end=250, by_epoch=False),
    dict(
        type="MultiStepLR",
        begin=0,
        end=epochs,
        milestones=[50, 75, 85],
        gamma=0.5,
        by_epoch=True,
    ),
]
optim_wrapper = dict(
    type="AmpOptimWrapper",
    optimizer=dict(
        _delete_=True,
        type="AdamW",
        lr=5e-5,
        betas=(0.9, 0.999),
        weight_decay=0.05,
    ),
    clip_grad=dict(
        max_norm=1.0, norm_type=2
    ),  # enable to help with unstable training (since we use large batch size and AMP)
    # * Smaller LR on backbone (to preserve pretrained features).
    # * Larger LR on detection head (to adapt to xView’s object types & scales).
    paramwise_cfg=dict(
        custom_keys={
            "backbone": dict(lr_mult=0.1),  # LR = 1e-5 for backbone
            "norm": dict(decay_mult=0.0),  # optional: no weight decay on norms
            "bias": dict(decay_mult=0.0),
        }
    ),
)

# * Training, validation, testing settings
train_cfg = dict(
    type="EpochBasedTrainLoop",  # The training loop type. Refer to https://github.com/open-mmlab/mmengine/blob/main/mmengine/runner/loops.py
    max_epochs=epochs,
    val_interval=val_interval,
)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")

# * Logging and checkpointing
vis_backends = [
    dict(type="LocalVisBackend"),
    dict(
        type="WandbVisBackend",
        init_kwargs={
            "project": "geopose",
            "group": "geosesame",
        },
    ),
]
visualizer = dict(type="DetLocalVisualizer", vis_backends=vis_backends, name="visualizer")
default_hooks = dict(
    logger=dict(type="LoggerHook", interval=50),
    checkpoint=dict(
        type="CheckpointHook",
        by_epoch=True,
        save_last=True,
        interval=val_interval,
        max_keep_ckpts=3,
        save_best="auto",
    ),
    visualization=dict(
        # user visualization of validation and test results
        type="DetVisualizationHook",
        draw=True,
        interval=10,
        show=False,
    ),
)
log_processor = dict(
    type="LogProcessor",
    window_size=50,
    by_epoch=True,  # Whether to format logs with epoch type. Should be consistent with the train loop's type.
)

# custom_hooks = [dict(type=Fp16CompresssionHook)]

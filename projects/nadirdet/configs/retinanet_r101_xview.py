_base_ = "../../../configs/retinanet/retinanet_r101_fpn_1x_coco.py"

data_root = "data/xview/chipped/200_0/"
dataset_type = "CocoDataset"

# Root directory for the chipped xView dataset relative to the mmdetection repo
data_root = "data/xview/chipped/200_0/"

XVIEW_CLASSES = (
    "Fixed-wing Aircraft",
    "Small Aircraft",
    "Cargo Plane",
    "Helicopter",
    "Passenger Vehicle",
    "Small Car",
    "Bus",
    "Pickup Truck",
    "Utility Truck",
    "Truck",
    "Cargo Truck",
    "Truck w/Box",
    "Truck Tractor",
    "Trailer",
    "Truck w/Flatbed",
    "Truck w/Liquid",
    "Crane Truck",
    "Railway Vehicle",
    "Passenger Car",
    "Cargo Car",
    "Flat Car",
    "Tank car",
    "Locomotive",
    "Maritime Vessel",
    "Motorboat",
    "Sailboat",
    "Tugboat",
    "Barge",
    "Fishing Vessel",
    "Ferry",
    "Yacht",
    "Container Ship",
    "Oil Tanker",
    "Engineering Vehicle",
    "Tower crane",
    "Container Crane",
    "Reach Stacker",
    "Straddle Carrier",
    "Mobile Crane",
    "Dump Truck",
    "Haul Truck",
    "Scraper/Tractor",
    "Front loader/Bulldozer",
    "Excavator",
    "Cement Mixer",
    "Ground Grader",
    "Hut/Tent",
    "Shed",
    "Building",
    "Aircraft Hangar",
    "Damaged Building",
    "Facility",
    "Construction Site",
    "Vehicle Lot",
    "Helipad",
    "Storage Tank",
    "Shipping container lot",
    "Shipping Container",
    "Pylon",
    "Tower",
)

metainfo = dict(classes=XVIEW_CLASSES)
backend_args = None
chip_size = (200, 200)  # set to (512, 512) if you switch tile size

load_from = "checkpoints/retinanet_r101_fpn_1x_coco_20200130-7a93545f.pth"
model = dict(
    # Disable redundant backbone init since we load a full RetinaNet checkpoint
    backbone=dict(
        init_cfg=None,
        # init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    ),
    # xview 60 classes
    bbox_head=dict(num_classes=len(XVIEW_CLASSES)),
)

# * Data pipelines
train_pipeline = [
    dict(type="LoadImageFromFile", backend_args=backend_args),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(type="RandomFlip", prob=0.75, direction=["horizontal", "vertical", "diagonal"]),  # aicdet
    dict(type="RandomResize", scale=chip_size, ratio_range=(0.8, 1.3), keep_ratio=True),  # mild shrink so small objs don't vanish
    dict(type="FilterAnnotations", min_gt_bbox_wh=(1, 1)),
    dict(type="PackDetInputs"),
]
# train_dataloader = dict(
#     batch_size=2,
#     num_workers=8,
#     persistent_workers=True,
#     sampler=dict(type="DefaultSampler", shuffle=True),
#     # batch_sampler=dict(type="AspectRatioBatchSampler"), # what is this?
#     dataset=dict(
#         type="ClassBalancedDataset",
#         # TODO: What is this?:
#         oversample_thr=1e-3,
#         dataset=dict(
#             type=dataset_type,
#             data_root=data_root,
#             ann_file="xview_coco_train_200_0.json",
#             data_prefix=dict(img="xview_coco_train_images_200_0/"),
#             filter_cfg=dict(filter_empty_gt=True, min_size=1),  # Is this size area? or one edge?
#             pipeline=train_pipeline,
#             metainfo=metainfo,
#             backend_args=backend_args,
#         ),
#     ),
# )  # From: docker_mmdet/lib/mmdetection/projects/Detic_new/configs/detic_centernet2_r50_fpn_4x_lvis_boxsup.py

train_dataloader = dict(
    batch_size=64,
    num_workers=8,  # This setting is per-gpu
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file="xview_coco_train_200_0.json",
        data_prefix=dict(img="xview_coco_train_images_200_0/"),
        filter_cfg=dict(
            filter_empty_gt=False, min_size=1
        ),  # Allow training on empty chips so model learns to handle background
        pipeline=train_pipeline,
        metainfo=metainfo,
        backend_args=backend_args,
    ),
)
val_pipeline = [
    # Note: dont want LoadAnnotations or FIlterAnnotations here since val_dataloader has
    # test_mode=True
    dict(type="LoadImageFromFile", backend_args=backend_args),
    dict(type="Resize", scale=chip_size, keep_ratio=True),
    dict(
        type="PackDetInputs",
        meta_keys=("img_id", "img_path", "ori_shape", "img_shape", "scale_factor"),
    ),
]
val_dataloader = dict(
    batch_size=64,
    num_workers=4,  # This setting is per-gpu
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file="xview_coco_val_200_0.json",
        data_prefix=dict(img="xview_coco_val_images_200_0/"),
        test_mode=True,
        pipeline=val_pipeline,
        metainfo=metainfo,
        backend_args=backend_args,
    ),
)
test_dataloader = val_dataloader
val_evaluator = dict(
    type="CocoMetric",
    ann_file=data_root + "xview_coco_val_200_0.json",
    metric=["bbox"],
    format_only=False,
    backend_args=backend_args,
)
test_evaluator = val_evaluator

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
)
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
visualization = dict(  # user visualization of validation and test results
    type="DetVisualizationHook", draw=True, interval=10, show=False
)
log_processor = dict(
    type="LogProcessor",
    window_size=50,
    by_epoch=True,  # Whether to format logs with epoch type. Should be consistent with the train loop's type.
)

# custom_hooks = [dict(type=Fp16CompresssionHook)]

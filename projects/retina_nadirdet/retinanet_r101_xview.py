_base_ = "../../configs/retinanet/retinanet_r50_fpn_1x_coco.py"
model = dict(
    backbone=dict(depth=101, init_cfg=dict(type="Pretrained", checkpoint="torchvision://resnet101"))
)
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
# TODO: use the xview std and mean in data_preprocessor

model = dict(
    # Disable redundant backbone init since we load a full RetinaNet checkpoint
    backbone=dict(init_cfg=None),
    # xview 60 classes
    bbox_head=dict(num_classes=len(XVIEW_CLASSES)),
)
# Define train pipeline
train_pipeline = [
    dict(type="LoadImageFromFile", backend_args=backend_args),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(type="Resize", scale=chip_size, keep_ratio=True),
    dict(type="RandomFlip", prob=0.5),
    dict(type="PackDetInputs"),
]
train_dataloader = dict(
    batch_size=2,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file="xview_coco_train_200_0.json",
        data_prefix=dict(img="xview_coco_train_images_200_0/"),
        filter_cfg=dict(filter_empty_gt=True, min_size=16),
        pipeline=train_pipeline,
        metainfo=metainfo,
        backend_args=backend_args,
    ),
)
val_pipeline = [
    dict(type="LoadImageFromFile", backend_args=backend_args),
    dict(type="Resize", scale=chip_size, keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    dict(type="LoadAnnotations", with_bbox=True),
    dict(
        type="PackDetInputs",
        meta_keys=("img_id", "img_path", "ori_shape", "img_shape", "scale_factor"),
    ),
]
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
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
    metric="bbox",
    format_only=False,
    backend_args=backend_args,
)
test_evaluator = val_evaluator

# * NOTE: `auto_scale_lr` is for automatically scaling LR, basically 16 x 4 gpus = 64 batch size
# * coco1x Default setting for scaling LR automatically
# *   - `enable` means enable scaling LR automatically
# *       or not by default.
# *   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(base_batch_size=16, enable=True)

# * An epoch on 6x4090 takes ~12 minutes, and eval takes ~5min. We don't want eval to
# *   take up a significant portion of training time so eval interval of 5 seems like the
# *   lowest reasonable setting.
epochs = 60
val_interval = 5
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
train_cfg = dict(
    type="EpochBasedTrainLoop",
    max_epochs=epochs,
    val_interval=val_interval,
)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")

default_hooks = dict(
    logger=dict(type="LoggerHook", interval=50),
    checkpoint=dict(
        type="CheckpointHook",
        by_epoch=True,
        save_last=True,
        interval=val_interval,
        max_keep_ckpts=5,
        save_best="auto",
    ),
)

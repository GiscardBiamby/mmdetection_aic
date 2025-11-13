_base_ = "../../configs/retinanet/retinanet_r101_fpn_ms-640-800-3x_coco.py"


# Root directory for the chipped xView dataset relative to the mmdetection repo
dataset_type = "CocoDataset"
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
backend_args = None

data_preprocessor = dict(
    type="DetDataPreprocessor",
    # xView stats (adjust if needed) or switch to COCO stats if preferred
    mean=[38.754, 47.501, 57.333],
    std=[32.028, 34.528, 42.689],
    bgr_to_rgb=True,
    pad_size_divisor=32,
)

model = dict(
    # Disable redundant backbone init since we load a full RetinaNet checkpoint
    backbone=dict(init_cfg=None),
    data_preprocessor=data_preprocessor,
    # Keep your 60 classes
    bbox_head=dict(num_classes=len(XVIEW_CLASSES)),
)

# Define your custom multi-scale training pipeline
train_pipeline = [
    dict(type="LoadImageFromFile", backend_args=backend_args),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(type="RandomResize", scale=[(1333, 640), (1333, 800)], keep_ratio=True),
    dict(type="RandomFlip", prob=0.5),
    # No extra meta_keys needed for multi-scale training
    dict(type="PackDetInputs"),
]
val_pipeline = [
    dict(type="LoadImageFromFile", backend_args=backend_args),
    dict(type="Resize", scale=chip_size, keep_ratio=True),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(
        type="PackDetInputs",
        meta_keys=("img_id", "img_path", "ori_shape", "img_shape", "scale_factor"),
    ),
]

# Completely override the train_dataloader from the base config
train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    # Use AspectRatioBatchSampler for multi-scale training, like the base config.
    batch_sampler=dict(type="AspectRatioBatchSampler"),
    dataset=dict(
        # This key tells mmengine to ignore the base dataset config
        _delete_=True,
        type=dataset_type,
        data_root=data_root,
        ann_file="xview_coco_train_200_0.json",
        data_prefix=dict(img="xview_coco_train_images_200_0/"),
        filter_cfg=dict(filter_empty_gt=True, min_size=16),
        # Explicitly assign your new pipeline here
        pipeline=train_pipeline,
        metainfo=metainfo,
        backend_args=backend_args,
    ),
)

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

test_dataloader = dict(
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

val_evaluator = dict(
    type="CocoMetric",
    ann_file=data_root + "xview_coco_val_200_0.json",
    metric="bbox",
    format_only=False,
    backend_args=backend_args,
)

test_evaluator = val_evaluator

optim_wrapper = dict(
    optimizer=dict(type="SGD", lr=0.005, momentum=0.9, weight_decay=0.0001),
    paramwise_cfg=dict(norm_decay_mult=0.0, bias_decay_mult=0.0),
)

train_cfg = dict(max_epochs=24)

param_scheduler = [
    dict(type="LinearLR", start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type="MultiStepLR",
        begin=0,
        end=24,
        by_epoch=True,
        milestones=[16, 22],
        gamma=0.1,
    ),
]

default_hooks = dict(checkpoint=dict(type="CheckpointHook", interval=1, max_keep_ckpts=3))

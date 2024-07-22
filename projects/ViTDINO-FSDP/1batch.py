_base_ = [
    '../../configs/_base_/default_runtime.py',
]
from functools import partial
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

from mmengine.runner.loops import IterBasedTrainLoop
from mmdet.datasets import CocoDatasetGSD
from mmdet.datasets.transforms import Rotate

# dataset settings
data_root = 'data/coco/'
image_size = (400, 400)

CLASSES = (
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

backend_args = None

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', scale=image_size, keep_ratio=True),
    dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor', 'flip', 'flip_direction', "input_res"))
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', scale=image_size, keep_ratio=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', "input_res"))
]

train_dataloader = dict(
    batch_size=4,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=CocoDatasetGSD,
        metainfo=dict(classes=CLASSES),
        data_root="/home/mlavery/xview_400_0/",
        # ann_file='train_gsd.json',
        # data_prefix=dict(img='train_images_400_0/'),
        ann_file='val_gsd_20_80.json',
        data_prefix=dict(img='val_images_400_0/'),
        pipeline=train_pipeline,
    ))

val_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=CocoDatasetGSD,
        metainfo=dict(classes=CLASSES),
        data_root="/home/mlavery/xview_400_0/",
        ann_file='val_gsd_20_80.json',
        data_prefix=dict(img='val_images_400_0/'),
        #test_mode=True,
        pipeline=test_pipeline,
        ))

test_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=CocoDatasetGSD,
        metainfo=dict(classes=CLASSES),
        data_root="/home/mlavery/xview_400_0/",
        ann_file='val_gsd_20_80.json',
        data_prefix=dict(img='val_images_400_0/'),
        #test_mode=True,
        pipeline=test_pipeline,
        ))

val_evaluator = dict(
    type='CocoMetric',
    ann_file='/home/mlavery/xview_400_0/val_gsd_20_80.json',
    metric=['bbox'],
    format_only=False,
    classwise=False
)
test_evaluator = val_evaluator

# 100 ep = 184375 iters * 64 images/iter / 118000 images/ep
# xview 1 has 209222 images in train
# 10 * 209222 / 64 = 32,690 iters
max_iters = 64690
interval = 50

train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=max_iters,
    val_interval=interval
)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=50),
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        save_last=True,
        interval=interval,
        max_keep_ckpts=5),
    #visualization=dict(type="DetVisualizationHook", interval=interval, draw=True, show=True),
)
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='WandbVisBackend',
        init_kwargs={
        'project': 'ScaleMAE-DET',
        'group': 'ViTDINO-DDP',
        })
]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=False)
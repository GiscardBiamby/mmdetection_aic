_base_ = [
    '../../configs/_base_/default_runtime.py',
]

from mmengine.runner.loops import IterBasedTrainLoop
from mmdet.datasets import CocoDatasetGSD
from mmdet.datasets.transforms import Rotate

from mmcv.transforms import RandomGrayscale, BaseTransform

# dataset settings
data_root = "/home/mlavery/xview_400_0/"
image_size = (512, 512)

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

class AddScale(BaseTransform):
    def transform(self, results):
        results["scale"] = 1
        return results

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True, ),
    # dict(type="RandomGrayscale", prob=0.2, keep_channels=True),
    # dict(type='RandomFlip', prob=1., direction=['horizontal']),
    # dict(type='RandomFlip', prob=1, direction=['vertical']),
    dict(
        type='RandomResize',
        scale=image_size,
        ratio_range=(1, 1.2),
        keep_ratio=True),
    # dict(
    #     type='RandomCrop',
    #     crop_type='absolute_range',
    #     crop_size=image_size,
    #     recompute_bbox=True,
    #     allow_negative_crop=True),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
    # dict(type='Pad', size=image_size, pad_val=dict(img=(114, 114, 114))),
    dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor', 'flip', 'flip_direction', "input_res"))
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', scale=image_size, keep_ratio=True),
    dict(type='Pad', size=image_size, pad_val=dict(img=(114, 114, 114))),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', "input_res", 'instances'))
]

train_dataloader = dict(
    batch_size=16,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=CocoDatasetGSD,
        metainfo=dict(classes=CLASSES),
        data_root=data_root,
        ann_file='train_gsd.json',
        data_prefix=dict(img='train_images_400_0/'),
        pipeline=train_pipeline,
        indices=100
    ))

val_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=CocoDatasetGSD,
        metainfo=dict(classes=CLASSES),
        data_root=data_root,
        ann_file='val_gsd_4k_80.json',
        data_prefix=dict(img='val_images_400_0/'),
        test_mode=True,
        pipeline=test_pipeline,
        ))

test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    # ann_file='/home/mlavery/xview_400_0/val_gsd_4k_80.json',
    metric=['bbox'],
    format_only=False,
    classwise=True,
    )
test_evaluator = val_evaluator

# 100 ep = 184375 iters * 64 images/iter / 118000 images/ep
max_iters = 184375
interval = 5000
dynamic_intervals = [(max_iters // interval * interval + 1, max_iters)]
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=250),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_iters,
        by_epoch=False,
        # 88 ep = [163889 iters * 64 images/iter / 118000 images/ep
        # 96 ep = [177546 iters * 64 images/iter / 118000 images/ep
        milestones=[163889, 177546],
        gamma=0.1)
]

train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=max_iters,
    val_interval=interval,
    dynamic_intervals=dynamic_intervals)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=50),
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        save_last=True,
        interval=interval,
        max_keep_ckpts=5,
        save_best='auto'),
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

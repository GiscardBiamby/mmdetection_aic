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
image_size = (224, 224)

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
    dict(type='LoadAnnotations', with_bbox=True),
    # dict(type="RandomGrayscale", prob=0.2, keep_channels=True),
    # dict(type='RandomFlip', prob=0.5),
    # dict(
    #     type='RandomResize',
    #     scale=image_size,
    #     ratio_range=(0.1, 2.0),
    #     keep_ratio=True),
    # dict(
    #     type='RandomCrop',
    #     crop_type='absolute_range',
    #     crop_size=image_size,
    #     recompute_bbox=True,
    #     allow_negative_crop=True),
    #dict(type=Rotate),
    dict(type='Resize', scale=image_size, keep_ratio=True),
    # dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
    # dict(type='Pad', size=image_size, pad_val=dict(img=(114, 114, 114))),
    dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor', 'flip', 'flip_direction', "image_res"))
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=image_size, keep_ratio=True),
    # dict(type='Pad', size=image_size, pad_val=dict(img=(114, 114, 114))),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', "image_res"))
]

train_dataloader = dict(
    batch_size=4,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=CocoDatasetGSD,
        metainfo=dict(classes=CLASSES),
        data_root="/home/mlavery/xview_train/",
        ann_file='val_gsd.json',
        # ann_file='train_gsd.json',
        data_prefix=dict(img='val_images_224_02/'),
        # data_prefix=dict(img='train_images_224_02/'),
        # filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        indices=range(0, 200)))

val_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=CocoDatasetGSD,
        metainfo=dict(classes=CLASSES),
        data_root="/home/mlavery/xview_train/",
        ann_file='val_gsd.json',
        data_prefix=dict(img='val_images_224_02/'),
        #test_mode=True,
        pipeline=test_pipeline,
        indices=range(0, 200)))
val_dataloader = val_dataloader
test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file='/home/mlavery/xview_train/val_gsd.json',
    metric=['bbox'],
    format_only=False)
test_evaluator = val_evaluator


optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=0.0003,  # 0.0002 for DeformDETR
        weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.1)}),
)

# 100 ep = 184375 iters * 64 images/iter / 118000 images/ep
# xview 1 has 209222 images in train
# 10 * 209222 / 64 = 32,690 iters
max_iters = 32690
interval = 200
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.01, by_epoch=False, begin=0, end=250),
    dict(
        type='LinearLR',
        start_factor=1,
        by_epoch=False,
        begin=250,
        end=max_iters,
        end_factor=0.01),
]

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
        max_keep_ckpts=5))
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

auto_scale_lr = dict(base_batch_size=32)

wrap_policy = partial(size_based_auto_wrap_policy, min_num_params=100_000, recurse=True)

# runner_type = 'FlexibleRunner'
# strategy = dict(
#     type='FSDPStrategy',
#     model_wrapper=dict(auto_wrap_policy=dict(type=wrap_policy)))
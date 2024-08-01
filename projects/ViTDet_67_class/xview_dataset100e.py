from mmengine.config import read_base

with read_base():
    from ...configs._base_.default_runtime import *

from mmdet.datasets import CocoDatasetGSD
from mmengine.optim import AmpOptimWrapper
from projects.ViTDet.vitdet import LayerDecayOptimizerConstructor
from torch.optim import AdamW

# dataset settings - do not change, it should match up with the paths generated by enclave_scripts/scripts/slice_xview.sh, which is `data/xview/chipped/...`
# This should point to aic_det/data/, which should look like this:
#  (aic_det) ➜  dgx1:~/proj/aic_det/lib<main>tree ../data -L 4                                                                                                                                                                                                                                                                                  git:(main|) 
# ../data
# └── xview
#     ├── chipped
#     │   ├── 2048_0
#     │   │   ├── xview_coco_full.json
#     │   │   ├── xview_coco_train_2048_0.json
#     │   │   ├── xview_coco_train_images_2048_0
#     │   │   ├── xview_coco_train.json
#     │   │   ├── xview_coco_val_2048_0.json
#     │   │   ├── xview_coco_val_images_2048_0
#     │   │   └── xview_coco_val.json
#     │   └── 400_0
#     │       ├── xview_coco_full.json
#     │       ├── xview_coco_train_400_0.json
#     │       ├── xview_coco_train_images_400_0
#     │       ├── xview_coco_train.json
#     │       ├── xview_coco_val_400_0.json
#     │       ├── xview_coco_val_images_400_0
#     │       └── xview_coco_val.json
data_root = "data/xview/chipped/400_0"
image_size = (400, 400)

CLASSES = ('Fixed-wing Aircraft', 'Small Aircraft', 'Cargo Plane', 'Helicopter', 'Passenger Vehicle', 'Small Car', 'Bus', 'Pickup Truck', 'Utility Truck', 'Truck', 'Cargo Truck', 'Truck w/Box', 'Truck Tractor', 'Trailer', 'Truck w/Flatbed', 'Truck w/Liquid', 'Crane Truck', 'Railway Vehicle', 'Passenger Car', 'Cargo Car', 'Flat Car', 'Tank car', 'Locomotive', 'Maritime Vessel', 'Motorboat', 'Sailboat', 'Tugboat', 'Barge', 'Fishing Vessel', 'Ferry', 'Yacht', 'Container Ship', 'Oil Tanker', 'Engineering Vehicle', 'Tower crane', 'Container Crane', 'Reach Stacker', 'Straddle Carrier', 'Mobile Crane', 'Dump Truck', 'Haul Truck', 'Scraper/Tractor', 'Front loader/Bulldozer', 'Excavator', 'Cement Mixer', 'Ground Grader', 'Hut/Tent', 'Shed', 'Building', 'Aircraft Hangar', 'Damaged Building', 'Facility', 'Construction Site', 'Vehicle Lot', 'Helipad', 'Storage Tank', 'Shipping container lot', 'Shipping Container', 'Pylon', 'Tower', 'Passenger Car1', 'Passenger Car2', 'Passenger Car3', 'Passenger Car4', 'Passenger Car5', 'Passenger Car6', 'Passenger Car7')


backend_args = None

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type="RandomGrayscale", prob=0.2, keep_channels=True),
    dict(type='RandomFlip', prob=0.75, direction=['horizontal', 'vertical', 'diagonal']),
    dict(
        type='RandomResize',
        scale=image_size,
        ratio_range=(0.8, 1.2),
        keep_ratio=True),
    # dict(type=Rotate),
    dict(type='Pad', size=image_size, pad_val=dict(img=(114, 114, 114))),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1)),
    dict(type='FilterAnnotations', by_mask=True),
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
                   'scale_factor', "input_res", "instances"))
]

train_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=CocoDatasetGSD,
        metainfo=dict(classes=CLASSES),
        data_root=data_root,
        ann_file='xview_coco_train_400_0_8xcar.json',
        data_prefix=dict(img='xview_coco_train_images_400_0/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=1),
        pipeline=train_pipeline,
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
        ann_file='xview_coco_val_400_0_8xcar.json',
        data_prefix=dict(img='xview_coco_val_images_400_0/'),
        test_mode=True,
        pipeline=test_pipeline,
        ))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    metric=['bbox'],
    format_only=False,
    classwise=True,
)
test_evaluator = val_evaluator

optim_wrapper = dict(
    type=AmpOptimWrapper,
    constructor=LayerDecayOptimizerConstructor,
    paramwise_cfg={
        "decay_rate": 0.8,
        "decay_type": "layer_wise",
        "num_layers": 24,
    },
    optimizer=dict(
        type=AdamW,
        lr=0.0001,
        betas=(0.9, 0.999),
        weight_decay=0.1,
    ),
)
auto_scale_lr = dict(base_batch_size=64, enable=True)

epochs = 100
val_interval = 1
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, begin=0, end=1, by_epoch=True),
    dict(
        type='MultiStepLR',
        begin=0,
        end=100,
        milestones=[88, 94],
        gamma=0.1,
        by_epoch=True,
    )
]

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=100,
    val_interval=1,
)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=50),
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=True,
        save_last=True,
        interval=val_interval,
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

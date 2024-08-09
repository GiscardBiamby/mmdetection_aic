from mmengine.config import read_base

with read_base():
    from ...configs._base_.default_runtime import *
    from .util import GSDDropout, CheckAddGSD


from mmdet.datasets import CocoDatasetGSD
from mmengine.optim import AmpOptimWrapper
from projects.ViTDet.vitdet import LayerDecayOptimizerConstructor
from torch.optim import AdamW

# dataset settings - do not change, it should match up with the paths generated by enclave_scripts/scripts/slice_xview.sh, which is `data/xview/chipped/...`
# This should point to aic_det/data/, which should look like this:
#  (aic_det) ➜  dgx1:~/proj/aic_det/lib<main>tree ../data -L 4                                                                                                                                                                                                                                                                                  git:(main|) 
# ../data
# └── comsat
#     ├── chipped
#     │   └── 400_0
#     │       ├── comsat_coco_train_400_0.json
#     │       ├── comsat_coco_train_images_400_0
#     │       ├── comsat_coco_val_400_0.json
#     │       ├── comsat_coco_val_images_400_0
data_root = "data/comsat/chipped/400_0"
image_size = (400, 400)

# num_classes 67
CLASSES = ('Mechanized Combat Ground Motor Vehicle', 'Armored Personnel Carrier', 'Amphibious Armored Personnel Carrier', 'BRDM-2 Armored Reconnaissance Carrier', 'Infantry Fighting Vehicle', 'Amphibious Infantry Fighting Vehicle ', 'BMP Infantry Fighting Vehicle', 'Missile Transloader', 'Mobile Projectile Launcher', 'Mobile Rocket Launcher', 'Multiple Rocket Launcher System', 'BM-21 Multiple Rocket Launcher System', 'Self-Propelled Artillery', 'Self-Propelled Anti-Aircraft Artillery', 'SA-22 SPAAGM', 'ZSU-23-4 Self-Propelled Anti-Aircraft Artillery', 'Self-Propelled Howitzer', '2S1 Self-Propelled Howitzer', 'SH2 Self-Propelled Howitzer', 'Tank', 'T-Series Tank', 'T-72 Tank', 'T-90 Tank', 'ZTZ-Series Tank', 'Mobile Radar-Communications Vehicle', 'Communications Vehicle', 'Copper Jack Command Post Vehicle', 'Radar Vehicle ', 'Grave Stone Radar Vehicle', 'Spoon Rest Radar Vehicle Series', 'Big Bird Radar Vehicle', 'Cheese Board Radar Vehicle', 'Fan Song Radar Vehicle Series', 'Flap Lid Radar Vehicle', 'Nebo SVU Radar Vehicle', 'Tomb Stone Radar Vehicle', 'Ground Motor Passenger Vehicle', 'Automobile', 'Mobile Construction Equipment', 'Emergency Vehicle', 'Bus', 'Van', 'Sport Utility Vehicle', 'Cargo Automobile', 'Mobile Surface-to-Air Missile Launcher', 'CSA Surface-to-Air Missile Launcher Series', 'CSA-9b Transporter Erector Launcher', 'Mobile Surface-to-Air Missile Launcher with Canister', 'SA-11 Transporter Erector Launcher', 'SA-13 Transporter Erector Launcher', 'SA-21 Transporter Erector Launcher', 'SA-6 Transporter Erector Launcher', 'Mobile Surface-to-Surface Missile Launcher', 'CSS-5 Transporter Erector Launcher', 'DF-26 Transporter Erector Launcher Series', 'Hwasong-13 Transporter Erector Launcher', 'Mobile Surface-to-Surface Missile Launcher with Canister', 'Shahab Surface-to-Surface Missile Launcher Series', 'SS-21 Transporter Erector Launcher Series', 'SS-26 Transporter Erector Launcher', 'Transporter Erector Launcher with Canister', 'Bongo Truck', 'Dump Truck', 'Flat Bed Truck', 'Pickup Truck', 'Semi Truck', 'Tanker Truck')

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
    dict(type=CheckAddGSD, gsd=0.45608300352667663),
    dict(type=GSDDropout, prob=0.1, gsd=0.45608300352667663),
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

# define per gpu batch size here, lr and max-iter should will be scaled automatically.
train_dataloader = dict(
    batch_size=2,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=CocoDatasetGSD,
        metainfo=dict(classes=CLASSES),
        data_root=data_root,
        ann_file='comsat_coco_train_400_0.json',
        data_prefix=dict(img='comsat_coco_train_images_400_0/'),
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
        ann_file='comsat_coco_val_400_0.json',
        data_prefix=dict(img='comsat_coco_val_images_400_0/'),
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
# NOTE: `auto_scale_lr` is for automatically scaling LR, basically 16 x 4 gpus = 64 batch size
auto_scale_lr = dict(base_batch_size=64, enable=True)

epochs = 100
val_interval = 1
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, begin=0, end=250, by_epoch=False),
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

default_hooks.merge(dict(
    logger=dict(type='LoggerHook', interval=50),
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=True,
        save_last=True,
        interval=val_interval,
        max_keep_ckpts=5,
        save_best='auto'),
))
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend')
]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=False)

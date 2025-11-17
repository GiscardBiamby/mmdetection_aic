_base_ = ".xview_512_0.py"

# dataset settings
dataset_type = "CocoDataset"
data_root = "data/xview/chipped/512_0/"
chip_size = (512, 512)  # The actual image size, before any resizing/augmentations

# Example to use different file client
# Method 1: simply set the data root and let the file I/O module
# automatically infer from prefix (not support LMDB and Memcache yet)

# data_root = 's3://openmmlab/datasets/detection/coco/'

# Method 2: Use `backend_args`, `file_client_args` in versions before 3.0.0rc6
# backend_args = dict(
#     backend='petrel',
#     path_mapping=dict({
#         './data/': 's3://openmmlab/datasets/detection/',
#         'data/': 's3://openmmlab/datasets/detection/'
#     }))
backend_args = None


# * Data pipelines
train_pipeline = [
    dict(type="LoadImageFromFile", backend_args=backend_args),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(type="RandomFlip", prob=0.75, direction=["horizontal", "vertical", "diagonal"]),  # aicdet
    dict(
        type="RandomResize", scale=chip_size, ratio_range=(0.8, 1.3), keep_ratio=True
    ),  # mild shrink so small objs don't vanish
    dict(type="FilterAnnotations", min_gt_bbox_wh=(1, 1)),
    dict(type="PackDetInputs"),
]
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
test_pipeline = val_pipeline

train_dataloader = dict(
    batch_size=64,
    num_workers=8,  # This setting is per-gpu
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file="xview_coco_train_512_0.json",
        data_prefix=dict(img="xview_coco_train_images_512_0/"),
        filter_cfg=dict(
            filter_empty_gt=False, min_size=1
        ),  # Allow training on empty chips so model learns to handle background
        pipeline=train_pipeline,
        metainfo=metainfo,
        backend_args=backend_args,
    ),
)
val_dataloader = dict(
    batch_size=64,
    num_workers=4,  # This setting is per-gpu
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file="xview_coco_val_512_0.json",
        data_prefix=dict(img="xview_coco_val_images_512_0/"),
        test_mode=True,
        pipeline=val_pipeline,
        metainfo=metainfo,
        backend_args=backend_args,
    ),
)
test_dataloader = val_dataloader

val_evaluator = dict(
    type="XViewCocoMetric",
    ann_file=data_root + "xview_coco_val_512_0.json",
    metric=["bbox"],
    format_only=False,
    backend_args=backend_args,
    # TODO: update proposal_nums to match dataset stats
    # # again, only affects AR@K
    # proposal_nums=(100, 300, 1000),
)
test_evaluator = val_evaluator

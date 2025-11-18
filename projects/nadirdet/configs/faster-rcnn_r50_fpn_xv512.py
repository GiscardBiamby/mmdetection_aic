_base_ = [
    "../../../configs/_base_/models/faster-rcnn_r50_fpn.py",
    "./_base_/schedules/xv_schedule_1x.py",
    "./_base_/default_runtime.py",
    "./datasets/xview_512_0.py",
]
custom_imports = dict(
    imports=[
        "projects.nadirdet.nadirdet",
        "mmdet.visualization",
        "mmdet.visualization.local_visualizer",
        "projects.nadirdet.nadirdet.xview_coco_metric",
        "projects.nadirdet.nadirdet.fp16_compression_hook",
    ],
    allow_failed_imports=False,
)
NUM_CLASSES = 60

load_from = "https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_mstrain_3x_coco/faster_rcnn_r50_fpn_mstrain_3x_coco_20210524_110822-e10bd31c.pth"
model = dict(
    backbone=dict(norm_eval=False, frozen_stages=-1),
    # * Customize anchors for xview:
    # * Effective sizes (scale=[1, 2, 4, 8] × stride):
    #   * Level P2 (stride 4): 4, 8, 16 32 px
    #   * Level P3 (stride 8): 8, 16, 32 64 px
    #   * Level P4 (stride 16): 16, 32, 64 128 px
    #   * Level P5 (stride 32): 32, 64, 128 256 px
    #   * Level P6 (stride 64): 64, 128, 256 512 px

    # * Effective sizes (scale=[2, 4, 8] × stride):
    #   * Level P2 (stride 4): 8, 16 32 px
    #   * Level P3 (stride 8): 16, 32 64 px
    #   * Level P4 (stride 16): 32, 64 128 px
    #   * Level P5 (stride 32): 64, 128 256 px
    #   * Level P6 (stride 64): 128, 256 512 px
    # We could also try `ratios=[0.75, 1.0, 1.5]``
    #
    # Note: should not need to change anchor sizes if you change to different chip sizes. if you
    # rescale the data before inputing into the network then you'd have to reconsider anchors. For
    # example if you scaleup the images by 2x so a 10x10 car becomes 20x20, the anchors would have
    # to be updated.
    rpn_head=dict(
        anchor_generator=dict(
            type="AnchorGenerator",
            scales=[2, 4, 8],               # defaults are [8]
            # Ratios: Default is perfect. Covers 0.25 to 4.0 aspect ratios with >0.5 IoU.
            ratios=[0.5, 1.0, 2.0],         # defaults are [0.5, 1.0, 2.0]
            strides=[4, 8, 16, 32, 64],     # keep Standard FPN strides (P2-P6) defaults of [4, 8, 16, 32, 64]
        ),
    ),
    roi_head=dict(
        bbox_head=dict(
            num_classes=int(NUM_CLASSES),
        )
    ),
    # # Because xview_512 has a large number of objects per image (up to 995):
    train_cfg=dict(
        rpn_proposal=dict(
            # frcnn defaults are 2000/1000 for train/test
            nms_pre=2000,
            max_per_img=1000,
        ),
    ),
    test_cfg=dict(
        # frcnn defaults are 1000/100 for train/test
        rpn=dict(
            nms_pre=3000,
            max_per_img=2000,
        ),
        rcnn=dict(
            # frcnn default max_per_img=100
            max_per_img=2000,  # or higher, e.g. 2000 / 10000 for analysis
        ),
    ),
)


# * Base config
# *   - `enable` means enable scaling LR automatically
# *   - `base_batch_size` = (4 GPUs) x (128 samples per GPU).
# * effective lr = base_lr * scale_factor
# * scale_factor = actual_batch_size / base_batch_size
# * How the scaling works if you diverge from GPU count. E.g., if you use 2 GPUs:
# * global batch = 2 × 128 = 256, scale factor = 256 / 512 = 0.5
auto_scale_lr = dict(base_batch_size=64, enable=True)


optim_wrapper = dict(
    type="AmpOptimWrapper",
)


#  # * Debug mode:
# max_debug_epochs = 5

# train_cfg = dict(
#     type="EpochBasedTrainLoop",
#     max_epochs=max_debug_epochs,
#     val_interval=1,
# )
# # ---- Use only a small subset of data for fast debugging ----
# train_dataloader = dict(
#     # inherit everything else (batch_size, num_workers, etc.)
#     dataset=dict(
#         # use only first 256 training images
#         indices=range(1024),
#     ),
# )
# val_dataloader = dict(
#     dataset=dict(
#         # use only first 64 validation images
#         indices=range(128),
#     ),
# )



train_cfg = dict(
    type="EpochBasedTrainLoop",
    max_epochs=5,
    val_interval=1,
)
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
    ],
    allow_failed_imports=False,
)
NUM_CLASSES = 60

load_from = "https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_mstrain_3x_coco/faster_rcnn_r50_fpn_mstrain_3x_coco_20210524_110822-e10bd31c.pth"
model = dict(
    backbone=dict(norm_eval=False, frozen_stages=-1),
    roi_head=dict(bbox_head=dict(num_classes=int(NUM_CLASSES))),
)

# * Base config
# *   - `enable` means enable scaling LR automatically
# *   - `base_batch_size` = (4 GPUs) x (128 samples per GPU).
# * effective lr = base_lr * scale_factor
# * scale_factor = actual_batch_size / base_batch_size
# * How the scaling works if you diverge from GPU count. E.g., if you use 2 GPUs:
# * global batch = 2 × 128 = 256, scale factor = 256 / 512 = 0.5
auto_scale_lr = dict(base_batch_size=128, enable=True)


optim_wrapper = dict(
    type="AmpOptimWrapper",
)
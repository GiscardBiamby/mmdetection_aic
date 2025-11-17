_base_ = [
    "../../../configs/_base_/models/retinanet_r50_fpn.py",
    "../../../configs/_base_/schedules/schedule_1x.py",
    "../../../configs/_base_/default_runtime.py",
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
NUM_CLASSES=60

load_from = ""
model = dict(
    bbox_head=dict(num_classes=NUM_CLASSES),
)
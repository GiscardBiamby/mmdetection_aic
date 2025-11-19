default_scope = "mmdet"

# * Logging and checkpointing
default_hooks = dict(
    logger=dict(type="LoggerHook", interval=50),
    checkpoint=dict(
        type="CheckpointHook",
        by_epoch=True,
        save_last=True,
        interval=5,  # probably want to override from child config, set it to val_interval
        max_keep_ckpts=3,
        save_best="auto",
    ),
    visualization=dict(
        # user visualization of validation and test results
        type="DetVisualizationHook",
        draw=True,
        interval=10,
        show=False,
    ),
    wandb_epoch_logger=dict(
        type="WandbEpochLoggerHook",
    ),
)

# custom_hooks = [dict(type="Fp16CompresssionHook")]

# * lets cuDNN autotune conv implementations. Can accelerate fix-size training. Don't use if input
# sizes vary a lot, e.g., if you have continuous variation. For example if you us e randomResize with
# (e.g. ratio_range=(0.8, 1.2)
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
    dist_cfg=dict(backend="nccl"),
)

vis_backends = [
    dict(type="LocalVisBackend"),
    dict(
        type="WandbVisBackend",
        init_kwargs={
            "project": "geopose",
            "group": "geosesame",
        },
    ),
]
visualizer = dict(type="mmdet.DetLocalVisualizer", vis_backends=vis_backends, name="visualizer")
log_processor = dict(
    type="LogProcessor",
    window_size=50,
    by_epoch=True,  # Whether to format logs with epoch type. Should be consistent with the train loop's type.
)
log_level = "INFO"
load_from = None
resume = False

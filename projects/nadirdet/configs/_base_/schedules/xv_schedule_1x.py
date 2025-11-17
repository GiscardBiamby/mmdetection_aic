max_epochs = 90
val_interval = 5
train_cfg = dict(
    type="EpochBasedTrainLoop",  # The training loop type. Refer to https://github.com/open-mmlab/mmengine/blob/main/mmengine/runner/loops.py
    max_epochs=max_epochs,
    val_interval=val_interval,
)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")

# * Learning rate
param_scheduler = [
    dict(type="LinearLR", start_factor=0.001, begin=0, end=250, by_epoch=False),
    dict(
        type="MultiStepLR",
        begin=0,
        end=max_epochs,
        milestones=[50, 75, 85],
        gamma=0.5,
        by_epoch=True,
    ),
]

# * Optimizer
optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(
        type="AdamW",
        lr=5e-5,
        betas=(0.9, 0.999),
        weight_decay=0.05,
    ),
    clip_grad=dict(
        max_norm=1.0, norm_type=2
    ),  # enable to help with unstable training (since we use large batch size and AMP)
    # * Smaller LR on backbone (to preserve pretrained features).
    # * Larger LR on detection head (to adapt to xView’s object types & scales).
    paramwise_cfg=dict(
        custom_keys={
            "backbone": dict(lr_mult=0.1),  # LR = 1e-5 for backbone
            "norm": dict(decay_mult=0.0),  # optional: no weight decay on norms
            "bias": dict(decay_mult=0.0),
        }
    ),
)
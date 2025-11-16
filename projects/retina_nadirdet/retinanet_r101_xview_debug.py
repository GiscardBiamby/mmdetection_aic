_base_ = "./retinanet_r101_xview.py"

# ---- Debug training schedule ----
# Only 1 epoch, and run validation every epoch
max_debug_epochs = 1

train_cfg = dict(
    type="EpochBasedTrainLoop",
    max_epochs=max_debug_epochs,
    val_interval=1,
)

# ---- Use only a small subset of data for fast debugging ----
# This dramatically reduces the number of iterations per epoch.
# Adjust the ranges to taste (e.g. 64, 128, 256).
train_dataloader = dict(
    # inherit everything else (batch_size, num_workers, etc.)
    dataset=dict(
        # use only first 256 training images
        indices=range(256),
    ),
)

# val_dataloader = dict(
#     dataset=dict(
#         # use only first 64 validation images
#         indices=range(64),
#     ),
# )

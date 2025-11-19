#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MMDET_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONFIG_PATH="projects/nadirdet/configs/faster-rcnn_r50_fpn_xv512.py"
EXP_ID="faster-rcnn_r50_fpn_xv512_rebalanced_xview-new_anchors"
# Initialize an empty array for extra mmdet config options
EXTRA_CFG_OPTIONS=()

# If "debug" is passed in as command line parameter, use the debug config
if [[ "${1:-}" == "debug" ]]; then
    EXP_ID="${EXP_ID}_debug"
    # Define the overrides for a debug run. Note the quotes around range().
    EXTRA_CFG_OPTIONS=(
        "train_cfg.max_epochs=5"
        "train_cfg.val_interval=1"
        "train_dataloader.dataset.indices=1024"
        "val_dataloader.dataset.indices=128"
    )
fi
WORK_DIR="${MMDET_ROOT}/work_dirs/${EXP_ID}"

cd "${MMDET_ROOT}"

# * Multi-gpu training
export WANDB_NAME="${EXP_ID}"
GPUS=5
CUDA_VISIBLE_DEVICES=0,1,2,3,4 torchrun --nproc_per_node=$GPUS \
    tools/train.py "${CONFIG_PATH}" \
    --work-dir "${WORK_DIR}" \
    --launcher pytorch \
    --cfg-options \
        "${EXTRA_CFG_OPTIONS[@]}"
    #     vis_backends[1].init_kwargs.group='my-new-group-name'
        #  \
    #
    # * Ex.: Use custom cfg options but also pass debug args:
    # --cfg-options \
    #     load_from="SOMEPATH" \
    #     "${EXTRA_CFG_OPTIONS[@]}"
    # --resume
    #
    # --cfg-options \
    #   train_dataloader.batch_size=64 \
    #   val_dataloader.batch_size=64 \
    #   test_dataloader.batch_size=64

# * Example: run final test on GPU 0 only
# CUDA_VISIBLE_DEVICES=0 python tools/test.py \
#     configs/faster-rcnn_r50_fpn_xv512.py \
#     "${WORK_DIR}/epoch_90.pth" \
#     --out results.pkl \

# # * Single gpu training
# python tools/train.py "${CONFIG_PATH}" \
#     --work-dir "${WORK_DIR}" \
#     --cfg-options load_from="${WEIGHTS_FILE}"

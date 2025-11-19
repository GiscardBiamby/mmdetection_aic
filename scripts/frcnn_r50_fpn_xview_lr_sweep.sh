#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MMDET_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONFIG_PATH="projects/nadirdet/configs/faster-rcnn_r50_fpn_xv512.py"
EXP_ID_BASE="faster-rcnn_r50_fpn_xv512_lr_sweep"

cd "${MMDET_ROOT}"

echo "[INFO] Starting LR sweep run with CONFIG_PATH=${CONFIG_PATH}"

EXP_SUFFIX="${WANDB_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
EXP_ID="${EXP_ID_BASE}_${EXP_SUFFIX}"
WORK_DIR="${MMDET_ROOT}/work_dirs/${EXP_ID}"
mkdir -p "${WORK_DIR}"

# args_no_hyphens gives: base_lr=0.001 other_param=...

echo "[INFO] args_no_hyphens: '$@'"

GPUS=4
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=$GPUS \
    tools/train.py "${CONFIG_PATH}" \
    --work-dir "${WORK_DIR}" \
    --launcher pytorch \
    --cfg-options \
        auto_scale_lr.enable=False \
        train_cfg.max_epochs=5 \
        train_cfg.val_interval=1 \
        "$@"
        # train_dataloader.dataset.indices=4096 \
        # val_dataloader.dataset.indices=512

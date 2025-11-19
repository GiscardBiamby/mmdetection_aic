#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MMDET_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Config and experiment naming
CONFIG_PATH="projects/nadirdet/configs/faster-rcnn_r50_fpn_xv512.py"
EXP_ID_BASE="faster-rcnn_r50_fpn_xv512_lr_sweep"

# Default sweep parameters (can be overridden by CLI)
SWEEP_NAME="frcnn_lr_sweep"
NUM_TRIALS=${NUM_TRIALS:-10}

# Move into MMDetection root so relative paths work
cd "${MMDET_ROOT}"

cat <<'EOF'
To run the W&B sweep:

  1) Initialize the sweep (this prints a SWEEP_ID):

     cd docker_mmdet/lib/mmdetection
     wandb sweep projects/nadirdet/configs/wandb_lr_sweep.yaml

  2) Launch a sweep agent (this script) on 4 GPUs:

     cd docker_mmdet/lib/mmdetection
     wandb agent --count 1 <YOUR_SWEEP_ID>

This script will:
  * Read base_lr from W&B config (wandb.config.base_lr)
  * Run a short training (e.g., 4 epochs) on 4 GPUs
EOF

echo "[INFO] Starting LR sweep run with CONFIG_PATH=${CONFIG_PATH}"

# Expect W&B to set WANDB_RUN_ID and wandb.config via environment.

# Derive experiment ID suffix from W&B run name if available.
EXP_SUFFIX="${WANDB_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
EXP_ID="${EXP_ID_BASE}_${EXP_SUFFIX}"
WORK_DIR="${MMDET_ROOT}/work_dirs/${EXP_ID}"

mkdir -p "${WORK_DIR}"

# We will let W&B config control base_lr via cfg-options.
# W&B config should define "base_lr"; if not, fall back to a sane default.
BASE_LR_ENV=${WANDB_CONFIG_BASE_LR:-"0.005"}
echo "[INFO] Using base_lr=${BASE_LR_ENV} (from W&B or default)"

# Train only for a few epochs for LR evaluation. We override schedule via cfg-options.

GPUS=4
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=$GPUS \
    tools/train.py "${CONFIG_PATH}" \
    --work-dir "${WORK_DIR}" \
    --launcher pytorch \
    --cfg-options \
    optim_wrapper.optimizer.lr="${BASE_LR_ENV}" \
    auto_scale_lr.enable=False \
    train_cfg.max_epochs=5 \
    train_cfg.val_interval=1
    # train_dataloader.dataset.indices=4096 \
    # val_dataloader.dataset.indices=512

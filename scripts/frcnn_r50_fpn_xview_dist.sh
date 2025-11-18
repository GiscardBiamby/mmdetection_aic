#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MMDET_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

EXP_ID="faster-rcnn_r50_fpn_xv512_rebalanced_xview-new_anchors"

# If "debug" is passed in as command line parameter, use the debug config
if [[ "${1:-}" == "debug" ]]; then
    CONFIG_PATH="projects/nadirdet/configs/faster-rcnn_r50_fpn_xv512_debug.py"
    EXP_ID="${EXP_ID}_debug"
else
    CONFIG_PATH="projects/nadirdet/configs/faster-rcnn_r50_fpn_xv512.py"
fi
WORK_DIR="${MMDET_ROOT}/work_dirs/${EXP_ID}"


cd "${MMDET_ROOT}"

# # * Single gpu training
# python tools/train.py "${CONFIG_PATH}" \
#     --work-dir "${WORK_DIR}" \
#     --cfg-options load_from="${WEIGHTS_FILE}"

# * Multi-gpu training
GPUS=5
CUDA_VISIBLE_DEVICES=0,1,2,3,4 torchrun --nproc_per_node=$GPUS \
    tools/train.py "${CONFIG_PATH}" \
    --work-dir "${WORK_DIR}" \
    --launcher pytorch
    # --resume
    # --cfg-options \
    # train_dataloader.batch_size=64 \
    # val_dataloader.batch_size=64 \
    # test_dataloader.batch_size=64



# # Example command to run final test on GPU 0 only
# CUDA_VISIBLE_DEVICES=0 python tools/test.py \
#     configs/faster-rcnn_r50_fpn_xv512.py \
#     "${WORK_DIR}/epoch_90.pth" \
#     --out results.pkl \
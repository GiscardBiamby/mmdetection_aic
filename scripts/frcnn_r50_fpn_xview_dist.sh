#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MMDET_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CHECKPOINTS_DIR="${MMDET_ROOT}/checkpoints"
WEIGHTS_URL="https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_mstrain_3x_coco/faster_rcnn_r50_fpn_mstrain_3x_coco_20210524_110822-e10bd31c.pth"
WEIGHTS_FILE="${CHECKPOINTS_DIR}/faster_rcnn_r50_fpn_mstrain_3x_coco_20210524_110822-e10bd31c.pth"
WORK_DIR="${MMDET_ROOT}/work_dirs/faster-rcnn_r50_fpn_xv512_rebalanced_xview-new_anchors"
# "faster-rcnn_r50_fpn_xv512_v02"
mkdir -p "${CHECKPOINTS_DIR}" "${WORK_DIR}"

# If "debug" is passed in as command line parameter, use the debug config
if [[ "${1:-}" == "debug" ]]; then
    CONFIG_PATH="projects/nadirdet/configs/faster-rcnn_r50_fpn_xv512_debug.py"
    WORK_DIR="${WORK_DIR}_debug"
else
    CONFIG_PATH="projects/nadirdet/configs/faster-rcnn_r50_fpn_xv512.py"
fi

# if [[ ! -f "${WEIGHTS_FILE}" ]]; then
#     echo "Downloading pretrained RetinaNet weights to ${WEIGHTS_FILE} ..."
#     if command -v curl >/dev/null 2>&1; then
#         curl -L "${WEIGHTS_URL}" -o "${WEIGHTS_FILE}"
#     elif command -v wget >/dev/null 2>&1; then
#         wget -O "${WEIGHTS_FILE}" "${WEIGHTS_URL}"
#     else
#         echo "Error: neither curl nor wget is available to download weights." >&2
#         exit 1
#     fi
# fi

# if [[ ! -f "${WEIGHTS_FILE}" ]]; then
#     echo "Error: failed to obtain pretrained weights at ${WEIGHTS_FILE}." >&2
#     exit 1
# fi

cd "${MMDET_ROOT}"

# # * Single gpu training
# python tools/train.py "${CONFIG_PATH}" \
#     --work-dir "${WORK_DIR}" \
#     --cfg-options load_from="${WEIGHTS_FILE}"

# * Multi-gpu training
GPUS=4
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=$GPUS \
    tools/train.py "${CONFIG_PATH}" \
    --work-dir "${WORK_DIR}" \
    --launcher pytorch
    # --cfg-options \
    # train_dataloader.batch_size=64 \
    # val_dataloader.batch_size=64 \
    # test_dataloader.batch_size=64


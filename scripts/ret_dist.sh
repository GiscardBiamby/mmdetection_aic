#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MMDET_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_PATH="projects/nadirdet/configs/retina_xview_clean.py"
CONFIG_PATH="projects/nadirdet/configs/retinanet_r101_xview.py"
CHECKPOINTS_DIR="${MMDET_ROOT}/checkpoints"
WEIGHTS_URL="https://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_r101_fpn_1x_coco/retinanet_r101_fpn_1x_coco_20200130-7a93545f.pth"
WEIGHTS_FILE="${CHECKPOINTS_DIR}/retinanet_r101_fpn_1x_coco_20200130-7a93545f.pth"
WORK_DIR="${MMDET_ROOT}/work_dirs/xview_retinanet"
mkdir -p "${CHECKPOINTS_DIR}" "${WORK_DIR}"

if [[ ! -f "${WEIGHTS_FILE}" ]]; then
    echo "Downloading pretrained RetinaNet weights to ${WEIGHTS_FILE} ..."
    if command -v curl >/dev/null 2>&1; then
        curl -L "${WEIGHTS_URL}" -o "${WEIGHTS_FILE}"
    elif command -v wget >/dev/null 2>&1; then
        wget -O "${WEIGHTS_FILE}" "${WEIGHTS_URL}"
    else
        echo "Error: neither curl nor wget is available to download weights." >&2
        exit 1
    fi
fi

if [[ ! -f "${WEIGHTS_FILE}" ]]; then
    echo "Error: failed to obtain pretrained weights at ${WEIGHTS_FILE}." >&2
    exit 1
fi

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


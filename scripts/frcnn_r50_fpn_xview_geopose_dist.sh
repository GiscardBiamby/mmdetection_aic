#!/usr/bin/env bash
set -euo pipefail

# * Usage:
# * # Normal run
# * ./frcnn_r50_fpn_xview_geopose_dist.sh
# *
# * # With suffix
# * ./frcnn_r50_fpn_xview_geopose_dist.sh --exp-id-suffix v2
# *
# * # Debug mode
# * ./frcnn_r50_fpn_xview_geopose_dist.sh debug
# *
# * # Both (order doesn't matter)
# * ./frcnn_r50_fpn_xview_geopose_dist.sh debug --exp-id-suffix v2
# * ./frcnn_r50_fpn_xview_geopose_dist.sh --exp-id-suffix v2 debug
# *
# * # Alternative syntax
# * ./frcnn_r50_fpn_xview_geopose_dist.sh --exp-id-suffix=v2

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MMDET_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONFIG_PATH="projects/nadirdet/configs/faster-rcnn_r50_fpn_xv512_geopose.py"
EXP_ID="frcnn_r50_fpn_xv512_geopose_geoflip"

# Initialize an empty array for extra mmdet config options. Actually the array is not empty because
# empty args list can cause `train.py: error: argument --cfg-options: expected at least one
# argument` errors. So pick some value that never changes, like default_scope and "override" it to
# it'sdefault value.
EXTRA_CFG_OPTIONS=("default_scope=mmdet")

# Parse command line arguments
DEBUG_MODE=false
EXP_ID_SUFFIX=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        debug)
            DEBUG_MODE=true
            shift
            ;;
        --exp-id-suffix)
            if [[ -n "${2:-}" ]]; then
                EXP_ID_SUFFIX="$2"
                shift 2
            else
                echo "Error: --exp-id-suffix requires a value"
                exit 1
            fi
            ;;
        --exp-id-suffix=*)
            EXP_ID_SUFFIX="${1#*=}"
            shift
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 [debug] [--exp-id-suffix <suffix>]"
            exit 1
            ;;
    esac
done

# Apply exp_id_suffix if specified
if [[ -n "${EXP_ID_SUFFIX}" ]]; then
    EXP_ID="${EXP_ID}_${EXP_ID_SUFFIX}"
fi

# Apply debug mode settings
if [[ "${DEBUG_MODE}" == "true" ]]; then
    EXP_ID="${EXP_ID}_debug"
    EXTRA_CFG_OPTIONS+=(
        "train_cfg.max_epochs=5"
        "train_cfg.val_interval=1"
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
        # train_dataloader.batch_size=32 \
        # val_dataloader.batch_size=32
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

#!/usr/bin/env bash
set -euo pipefail

# * Usage:
# * # Normal run (3 seeds: 1000, 1001, 1002)
# * ./geopose_fpns.sh
# *
# * # With suffix
# * ./geopose_fpns.sh --exp-id-suffix v2
# *
# * # Debug mode
# * ./geopose_fpns.sh debug
# *
# * # Custom FPN levels (comma-separated, no spaces)
# * ./geopose_fpns.sh --fpn-levels 3,4
# * ./geopose_fpns.sh --fpn-levels=2,3,4
# *
# * # Custom seeds (comma-separated, no spaces)
# * ./geopose_fpns.sh --seeds 42,123,456
# *
# * # Single seed
# * ./geopose_fpns.sh --seeds 0
# *
# * # Custom GPUs (comma-separated device IDs and count)
# * ./geopose_fpns.sh --gpus 0,1,2,3
# *
# * # Custom batch sizes
# * ./geopose_fpns.sh --train-batch-size 16 --val-batch-size 8
# *
# * # Combined
# * ./geopose_fpns.sh debug --exp-id-suffix v2 --fpn-levels 3,4 --seeds 0,1,2 --gpus 0,1 --train-batch-size 8

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MMDET_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONFIG_PATH="projects/nadirdet/configs/faster-rcnn_r50_fpn_xv512_geopose.py"
EXP_ID_BASE="frcnn_r50_fpn_xv512_geopose_geoflip"

# Defaults
DEBUG_MODE=false
EXP_ID_SUFFIX=""
FPN_LEVELS=""
SEEDS="1000,1001,1002"
GPUS="0,1,2,3,4"
TRAIN_BATCH_SIZE=""
VAL_BATCH_SIZE=""

# Parse arguments with getopt
OPTS=$(getopt -o '' --long exp-id-suffix:,fpn-levels:,seeds:,gpus:,train-batch-size:,val-batch-size: -n "$0" -- "$@") || exit 1
eval set -- "$OPTS"

while true; do
    case "$1" in
        --exp-id-suffix)     EXP_ID_SUFFIX="$2"; shift 2 ;;
        --fpn-levels)        FPN_LEVELS="$2"; shift 2 ;;
        --seeds)             SEEDS="$2"; shift 2 ;;
        --gpus)              GPUS="$2"; shift 2 ;;
        --train-batch-size)  TRAIN_BATCH_SIZE="$2"; shift 2 ;;
        --val-batch-size)    VAL_BATCH_SIZE="$2"; shift 2 ;;
        --)                  shift; break ;;
    esac
done

# Handle positional args (like "debug")
for arg in "$@"; do
    case "$arg" in
        debug) DEBUG_MODE=true ;;
        *)     echo "Unknown argument: $arg"; exit 1 ;;
    esac
done

# Convert comma-separated seeds to array
IFS=',' read -ra SEED_ARRAY <<<"$SEEDS"

# Count GPUs
IFS=',' read -ra GPU_ARRAY <<<"$GPUS"
NUM_GPUS=${#GPU_ARRAY[@]}

cd "${MMDET_ROOT}"

# Run training for each seed
for SEED in "${SEED_ARRAY[@]}"; do
    echo "=========================================="
    echo "Running with seed: ${SEED}"
    echo "GPUs: ${GPUS} (${NUM_GPUS} devices)"
    echo "=========================================="

    # Build EXP_ID for this seed
    EXP_ID="${EXP_ID_BASE}"

    # Add FPN levels to exp_id if specified
    if [[ -n "${FPN_LEVELS}" ]]; then
        # Convert 3,4 to p3p4 for exp_id
        FPN_SUFFIX=$(echo "${FPN_LEVELS}" | sed 's/,/p/g' | sed 's/^/p/')
        EXP_ID="${EXP_ID}_fpn${FPN_SUFFIX}"
    fi

    # Add suffix if specified
    if [[ -n "${EXP_ID_SUFFIX}" ]]; then
        EXP_ID="${EXP_ID}_${EXP_ID_SUFFIX}"
    fi

    # Add seed to exp_id
    EXP_ID="${EXP_ID}_seed${SEED}"

    # Initialize cfg options array
    EXTRA_CFG_OPTIONS=("default_scope=mmdet")

    # Add seed to cfg options
    EXTRA_CFG_OPTIONS+=("randomness.seed=${SEED}")

    # Add FPN levels if specified
    if [[ -n "${FPN_LEVELS}" ]]; then
        # Convert "3,4" to "(3,4)" for Python tuple
        FPN_TUPLE="(${FPN_LEVELS})"
        EXTRA_CFG_OPTIONS+=("model.geo_pose_head.fpn_levels=${FPN_TUPLE}")
    fi

    # Add batch sizes if specified
    if [[ -n "${TRAIN_BATCH_SIZE}" ]]; then
        EXTRA_CFG_OPTIONS+=("train_dataloader.batch_size=${TRAIN_BATCH_SIZE}")
    fi
    if [[ -n "${VAL_BATCH_SIZE}" ]]; then
        EXTRA_CFG_OPTIONS+=("val_dataloader.batch_size=${VAL_BATCH_SIZE}")
        EXTRA_CFG_OPTIONS+=("test_dataloader.batch_size=${VAL_BATCH_SIZE}")
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

    # Multi-gpu training
    echo "GPUS: ${GPUS}"
    echo "NUM_GPUS: ${NUM_GPUS}"
    export WANDB_NAME="${EXP_ID}"
    CUDA_VISIBLE_DEVICES="${GPUS}" torchrun --nproc_per_node="${NUM_GPUS}" \
        tools/train.py "${CONFIG_PATH}" \
        --work-dir "${WORK_DIR}" \
        --launcher pytorch \
        --cfg-options \
            "${EXTRA_CFG_OPTIONS[@]}"

    echo "Finished seed ${SEED}"
    echo ""
done

echo "=========================================="
echo "All seeds completed!"
echo "=========================================="

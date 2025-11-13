#!/bin/bash

echo "env_check"
set -e

# Get the directory of this script so that we can reference paths correctly no matter which folder
# the script was launched from:
SCRIPTS_DIR="$(builtin cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(realpath "${SCRIPTS_DIR}"/../../../../)"
MMDET_ROOT="$(realpath "${SCRIPTS_DIR}"/../)"

# * Export all variabes from manifest file:
PYTHON_ENV_NAME="nadirdet"

echo ""
echo "========================================================================"
echo "create_env.sh"
echo "SCRIPTS_DIR: ${SCRIPTS_DIR}"
echo "PROJ_ROOT: ${PROJ_ROOT}"
echo "MMDET_ROOT: ${MMDET_ROOT}"
echo "CONDA_DIR: ${CONDA_DIR}"
echo "PYTHON_ENV_NAME: ${PYTHON_ENV_NAME}"

# * Load conda/mamba
if [ -d ~/anaconda3/etc/profile.d ]; then
    source ~/anaconda3/etc/profile.d/conda.sh
elif [ -d /opt/miniconda-latest/etc/profile.d ]; then
    source /opt/miniconda-latest/etc/profile.d/conda.sh
elif [ -d ~/miniconda/etc/profile.d ]; then
    source ~/miniconda/etc/profile.d/conda.sh
elif [ -d ~/mambaforge/etc/profile.d ]; then
    source ~/mambaforge/etc/profile.d/conda.sh
    source ~/mambaforge/etc/profile.d/mamba.sh
else
    echo "ERROR, no conda installation found"
    exit 1
fi

# * Activate conda environment:
conda activate "${PYTHON_ENV_NAME}"
pushd "${MMDET_ROOT}" || exit
CUDA_VISIBLE_DEVICES=0,1,2,3 ./tools/dist_train.sh \
    ./projects/ViTDet_67_class/parent_child/vit_L_gsd_det_rcnn_xview200_child_60e.py 4 \
    --amp \
    --cfg-options train_dataloader.batch_size=4

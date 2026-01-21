#!/bin/bash

# RDT + RLinf Training Script for LIBERO Spatial
# This script runs PPO training on LIBERO Spatial tasks using RDT policy

export EMBODIED_PATH="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"
export REPO_PATH=$(dirname $(dirname "$EMBODIED_PATH"))
export SRC_FILE="${EMBODIED_PATH}/train_embodied_agent.py"

# Set up environment variables
export PYTHONPATH=${REPO_PATH}:$PYTHONPATH
export MUJOCO_GL="egl"
export PYOPENGL_PLATFORM="egl"

# Optional: WandB API key for logging
# export WANDB_API_KEY="your_wandb_api_key_here"

# Configuration
CONFIG_NAME="libero_spatial_ppo_rdt"

# RDT checkpoint path - MUST UPDATE THIS!
# Download RDT checkpoint from: https://huggingface.co/robotics-diffusion-transformer/rdt-1b
export RDT_CHECKPOINT_PATH="/home/zhukefei/chensiqi/rlinf_workspace/Libero_RDT/RDT_libero_finetune/checkpoints/best_checkpoints/libero_spatial_best_ckpt"

echo "=========================================="
echo "RLinf + RDT Training on LIBERO Spatial"
echo "=========================================="
echo "Using Python at $(which python)"
echo "Repo Path: ${REPO_PATH}"
echo "Config: ${CONFIG_NAME}"
echo "RDT Checkpoint: ${RDT_CHECKPOINT_PATH}"
echo "=========================================="

# Create log directory
LOG_DIR="${REPO_PATH}/logs/$(date +'%Y%m%d-%H%M%S')-${CONFIG_NAME}"
MEGA_LOG_FILE="${LOG_DIR}/run_embodiment.log"
mkdir -p "${LOG_DIR}"

# Build command
CMD="python ${SRC_FILE} \
  --config-path ${EMBODIED_PATH}/config/ \
  --config-name ${CONFIG_NAME} \
  runner.logger.log_path=${LOG_DIR} \
  actor.model.model_path=${RDT_CHECKPOINT_PATH} \
  rollout.model.model_path=${RDT_CHECKPOINT_PATH}"

echo "Command: ${CMD}" > ${MEGA_LOG_FILE}
echo "=========================================="

# Run training
${CMD} 2>&1 | tee -a ${MEGA_LOG_FILE}



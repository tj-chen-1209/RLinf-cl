#!/bin/bash
# Run script for RLinf-cl RDT Docker container

set -e

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Image name
IMAGE_NAME="rlinf-cl-rdt"
IMAGE_TAG="latest"

# Set default paths
HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
LOG_DIR="${LOG_DIR:-$PROJECT_ROOT/logs}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/share_data/chensiqi/gr00t/best_checkpoints}"

echo "========================================="
echo "Running RLinf-cl RDT Docker Container"
echo "========================================="
echo "Image: ${IMAGE_NAME}:${IMAGE_TAG}"
echo "Project Root: $PROJECT_ROOT"
echo "HF_HOME: $HF_HOME"
echo "LOG_DIR: $LOG_DIR"
echo "CHECKPOINT_DIR: $CHECKPOINT_DIR"
echo "========================================="

# Create directories if they don't exist
mkdir -p "$HF_HOME"
mkdir -p "$LOG_DIR"

# Run the Docker container
docker run -it --rm \
    --gpus all \
    --shm-size 100g \
    --net=host \
    --name rlinf-cl-rdt \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
    -v "$PROJECT_ROOT":/workspace/RLinf-cl \
    -v "$HF_HOME":/workspace/hf \
    -v "$LOG_DIR":/workspace/RLinf-cl/logs \
    -v "$CHECKPOINT_DIR":/workspace/checkpoints \
    "${IMAGE_NAME}:${IMAGE_TAG}" \
    /bin/bash

echo ""
echo "========================================="
echo "Container exited"
echo "========================================="


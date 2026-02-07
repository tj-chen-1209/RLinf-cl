#!/bin/bash
# Build script for RLinf-cl RDT Docker image (simplified version)
# Uses pip instead of conda environment.yml

set -e

# Get workspace root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
WORKSPACE_ROOT="$( cd "$PROJECT_ROOT/.." && pwd )"

# Image name and tag
IMAGE_NAME="rlinf-cl-rdt"
IMAGE_TAG="latest"

echo "========================================="
echo "Building RLinf-cl RDT Docker Image (Simplified)"
echo "========================================="
echo "Workspace Root: $WORKSPACE_ROOT"
echo "Project Root: $PROJECT_ROOT"
echo "Image: ${IMAGE_NAME}:${IMAGE_TAG}"
echo "========================================="

# Check if core requirements file exists
if [ ! -f "$PROJECT_ROOT/docker/requirements_core.txt" ]; then
    echo "Error: docker/requirements_core.txt not found!"
    exit 1
fi

# Check if LIBERO directory exists
if [ ! -d "$WORKSPACE_ROOT/LIBERO" ]; then
    echo "Error: LIBERO directory not found at $WORKSPACE_ROOT/LIBERO"
    exit 1
fi

# Change to workspace root
cd "$WORKSPACE_ROOT"

# Build the Docker image
echo "Building Docker image (this may take 10-20 minutes)..."
echo "Note: You may be prompted for sudo password"
sudo docker build \
    -f RLinf-cl/docker/Dockerfile.rdt.simple \
    -t "${IMAGE_NAME}:${IMAGE_TAG}" \
    .

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================="
    echo "✓ Docker image built successfully!"
    echo "========================================="
    echo "Image: ${IMAGE_NAME}:${IMAGE_TAG}"
    echo ""
    echo "Image size:"
    sudo docker images "${IMAGE_NAME}:${IMAGE_TAG}"
    echo ""
    echo "To run the container:"
    echo "  bash docker/run_rdt.sh"
    echo ""
    echo "Or manually:"
    echo "  sudo docker run -it --rm --gpus all \\"
    echo "    -v \$(pwd):/workspace/RLinf-cl \\"
    echo "    -v \$HF_HOME:/workspace/hf \\"
    echo "    ${IMAGE_NAME}:${IMAGE_TAG} /bin/bash"
    echo "========================================="
else
    echo ""
    echo "========================================="
    echo "✗ Docker build failed!"
    echo "========================================="
    exit 1
fi


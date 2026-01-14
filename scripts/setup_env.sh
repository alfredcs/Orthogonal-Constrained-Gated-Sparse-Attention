#!/bin/bash
# OrthGSA Environment Setup Script using uv
# Fast Python package management with uv
# NOTE: This script uses the existing system CUDA installation.
#       It does NOT install CUDA - you must have CUDA already installed.

set -e

# Parse arguments
SKIP_CUDA_PACKAGES=false
SKIP_MODEL_DOWNLOAD=false
for arg in "$@"; do
    case $arg in
        --skip-cuda-packages)
            SKIP_CUDA_PACKAGES=true
            shift
            ;;
        --skip-model-download)
            SKIP_MODEL_DOWNLOAD=true
            shift
            ;;
    esac
done

echo "============================================"
echo "OrthGSA Environment Setup (using uv)"
echo "Using existing system CUDA installation"
echo "============================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo -e "${YELLOW}uv not found. Installing uv...${NC}"
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # Add uv to PATH for current session
    export PATH="$HOME/.cargo/bin:$PATH"

    echo -e "${GREEN}uv installed successfully!${NC}"
else
    echo -e "${GREEN}uv is already installed: $(uv --version)${NC}"
fi

# Get the script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo ""
echo "Project root: $PROJECT_ROOT"
echo ""

# Detect existing system CUDA version (does NOT install CUDA)
echo ""
echo "Detecting existing CUDA installation..."
CUDA_VERSION=""
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p')
    echo -e "${GREEN}Found existing CUDA: $CUDA_VERSION (from nvcc)${NC}"
elif [ -f /usr/local/cuda/version.txt ]; then
    CUDA_VERSION=$(cat /usr/local/cuda/version.txt | sed -n 's/CUDA Version \([0-9]*\.[0-9]*\).*/\1/p')
    echo -e "${GREEN}Found existing CUDA: $CUDA_VERSION (from /usr/local/cuda)${NC}"
elif [ -d /usr/local/cuda ]; then
    # Try to get version from cuda directory
    if [ -f /usr/local/cuda/version.json ]; then
        CUDA_VERSION=$(python3 -c "import json; print(json.load(open('/usr/local/cuda/version.json'))['cuda']['version'][:4])" 2>/dev/null || echo "")
    fi
    if [ -n "$CUDA_VERSION" ]; then
        echo -e "${GREEN}Found existing CUDA: $CUDA_VERSION (from version.json)${NC}"
    else
        echo -e "${YELLOW}CUDA directory exists but version unknown. Assuming CUDA 12.x${NC}"
        CUDA_VERSION="12.0"
    fi
else
    echo -e "${YELLOW}No existing CUDA installation found. Will install CPU-only PyTorch.${NC}"
    echo -e "${YELLOW}To use GPU, please install CUDA first and re-run this script.${NC}"
fi

echo ""
echo "Step 1: Creating virtual environment with uv..."
echo "============================================"

# Create virtual environment
uv venv .venv --python 3.11

echo -e "${GREEN}Virtual environment created at .venv${NC}"

echo ""
echo "Step 2: Installing latest PyTorch with CUDA support..."
echo "============================================"

# Install latest PyTorch, torchvision using existing system CUDA
# PyPI now includes CUDA 12.4 support by default in latest PyTorch
if [[ -n "$CUDA_VERSION" ]]; then
    echo "Installing latest PyTorch with CUDA support (using existing CUDA $CUDA_VERSION)..."
    uv pip install torch torchvision torchaudio
else
    echo "Installing latest PyTorch (CPU only)..."
    uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

echo -e "${GREEN}PyTorch installed successfully!${NC}"

echo ""
echo "Step 3: Installing OrthGSA and dependencies..."
echo "============================================"

# Install the project in editable mode with all dependencies
uv pip install -e ".[dev]"

echo -e "${GREEN}OrthGSA and dependencies installed successfully!${NC}"

echo ""
echo "Step 4: Installing optional CUDA dependencies..."
echo "============================================"

if [[ "$SKIP_CUDA_PACKAGES" == "true" ]]; then
    echo -e "${YELLOW}Skipping optional CUDA packages (--skip-cuda-packages flag)${NC}"
elif [[ -n "$CUDA_VERSION" ]]; then
    # Install triton (uses existing CUDA)
    echo "Installing triton (will use existing CUDA installation)..."
    uv pip install triton>=2.1.0
    echo -e "${GREEN}Triton installed successfully!${NC}"

    # Try to install flash-attn (may fail on some systems)
    echo ""
    echo "Attempting to install flash-attention (optional, uses existing CUDA)..."
    if uv pip install flash-attn --no-build-isolation 2>/dev/null; then
        echo -e "${GREEN}flash-attn installed successfully!${NC}"
    else
        echo -e "${YELLOW}flash-attn installation failed. This is optional.${NC}"
        echo -e "${YELLOW}You can try manually: uv pip install flash-attn --no-build-isolation${NC}"
    fi
else
    echo -e "${YELLOW}Skipping CUDA-specific packages (no existing CUDA detected)${NC}"
fi

echo ""
echo "Step 5: Downloading base model from Hugging Face..."
echo "============================================"

# Model to download
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-4B-Instruct-2507}"

if [[ "$SKIP_MODEL_DOWNLOAD" == "true" ]]; then
    echo -e "${YELLOW}Skipping model download (--skip-model-download flag)${NC}"
else
    echo "Downloading model: $BASE_MODEL"
    echo "This may take a while depending on your internet connection..."

    # Download the model using huggingface-cli
    if command -v huggingface-cli &> /dev/null; then
        huggingface-cli download "$BASE_MODEL" --local-dir-use-symlinks False
        echo -e "${GREEN}Model downloaded successfully!${NC}"
    else
        # Fallback to Python-based download
        python -c "
from huggingface_hub import snapshot_download
print('Downloading $BASE_MODEL...')
snapshot_download('$BASE_MODEL')
print('Download complete!')
"
        echo -e "${GREEN}Model downloaded successfully!${NC}"
    fi
fi

echo ""
echo "Step 6: Verifying installation..."
echo "============================================"

# Activate and verify
source .venv/bin/activate

python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
"

python -c "
import transformers
import datasets
import wandb
print(f'Transformers version: {transformers.__version__}')
print(f'Datasets version: {datasets.__version__}')
print(f'Wandb version: {wandb.__version__}')
"

python -c "
import orthgsa
print(f'OrthGSA version: {orthgsa.__version__}')
print('OrthGSA imported successfully!')
"

echo ""
echo "============================================"
echo -e "${GREEN}Setup complete!${NC}"
echo "============================================"
echo ""
echo "NOTE: This script uses your existing system CUDA installation."
echo "      It does NOT install CUDA itself."
echo ""
echo "To activate the environment:"
echo "  source .venv/bin/activate"
echo ""
echo "To start training:"
echo "  ./scripts/launch_train.sh"
echo ""
echo "Or with torchrun directly:"
echo "  torchrun --nproc_per_node=4 scripts/train.py --config configs/config_qwen3_4b.yaml"
echo ""
echo "Script options:"
echo "  --skip-cuda-packages   Skip installing triton/flash-attn"
echo "  --skip-model-download  Skip downloading the base model"
echo ""
echo "Environment variables:"
echo "  BASE_MODEL=<model>  Specify a different model to download (default: Qwen/Qwen3-4B-Instruct-2507)"
echo ""

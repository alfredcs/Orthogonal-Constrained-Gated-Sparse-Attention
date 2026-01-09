#!/bin/bash
# OrthGSA Environment Setup Script using uv
# Fast Python package management with uv

set -e

echo "============================================"
echo "OrthGSA Environment Setup (using uv)"
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

# Detect CUDA version
CUDA_VERSION=""
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p')
    echo -e "${GREEN}Detected CUDA version: $CUDA_VERSION${NC}"
elif [ -f /usr/local/cuda/version.txt ]; then
    CUDA_VERSION=$(cat /usr/local/cuda/version.txt | sed -n 's/CUDA Version \([0-9]*\.[0-9]*\).*/\1/p')
    echo -e "${GREEN}Detected CUDA version: $CUDA_VERSION${NC}"
else
    echo -e "${YELLOW}CUDA not detected. Installing CPU-only PyTorch.${NC}"
fi

# Determine PyTorch index URL based on CUDA version
TORCH_INDEX=""
if [[ "$CUDA_VERSION" == "12."* ]]; then
    TORCH_INDEX="https://download.pytorch.org/whl/cu121"
    echo "Using PyTorch CUDA 12.1 wheels"
elif [[ "$CUDA_VERSION" == "11.8"* ]] || [[ "$CUDA_VERSION" == "11.7"* ]]; then
    TORCH_INDEX="https://download.pytorch.org/whl/cu118"
    echo "Using PyTorch CUDA 11.8 wheels"
elif [[ "$CUDA_VERSION" == "11."* ]]; then
    TORCH_INDEX="https://download.pytorch.org/whl/cu118"
    echo "Using PyTorch CUDA 11.8 wheels (closest match)"
else
    TORCH_INDEX="https://download.pytorch.org/whl/cpu"
    echo "Using PyTorch CPU wheels"
fi

echo ""
echo "Step 1: Creating virtual environment with uv..."
echo "============================================"

# Create virtual environment
uv venv .venv --python 3.11

echo -e "${GREEN}Virtual environment created at .venv${NC}"

echo ""
echo "Step 2: Installing PyTorch with CUDA support..."
echo "============================================"

# Install PyTorch first with the correct index
uv pip install torch torchvision torchaudio --index-url "$TORCH_INDEX"

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

if [[ -n "$CUDA_VERSION" ]]; then
    # Install triton
    uv pip install triton>=2.1.0
    echo -e "${GREEN}Triton installed successfully!${NC}"

    # Try to install flash-attn (may fail on some systems)
    echo ""
    echo "Attempting to install flash-attention (optional)..."
    if uv pip install flash-attn --no-build-isolation 2>/dev/null; then
        echo -e "${GREEN}flash-attn installed successfully!${NC}"
    else
        echo -e "${YELLOW}flash-attn installation failed. This is optional.${NC}"
        echo -e "${YELLOW}You can try manually: uv pip install flash-attn --no-build-isolation${NC}"
    fi
else
    echo -e "${YELLOW}Skipping CUDA-specific packages (no CUDA detected)${NC}"
fi

echo ""
echo "Step 5: Verifying installation..."
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
echo "To activate the environment:"
echo "  source .venv/bin/activate"
echo ""
echo "To start training:"
echo "  ./scripts/launch_train.sh"
echo ""
echo "Or with torchrun directly:"
echo "  torchrun --nproc_per_node=4 scripts/train.py --config configs/config_qwen3_4b.yaml"
echo ""

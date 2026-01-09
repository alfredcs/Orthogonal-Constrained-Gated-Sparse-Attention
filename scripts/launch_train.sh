#!/bin/bash
# OrthGSA Training Launch Script
# For 4x 44GB GPUs (e.g., A100-44GB or A6000)
# Uses Cayley Transform for orthogonal constraints instead of Sinkhorn-Knopp

set -e

# Get the script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# Activate uv virtual environment if it exists
if [ -d ".venv" ]; then
    echo "Activating uv virtual environment..."
    source .venv/bin/activate
elif [ -d "venv" ]; then
    echo "Activating venv virtual environment..."
    source venv/bin/activate
else
    echo "Warning: No virtual environment found. Run ./scripts/setup_env.sh first."
    echo "Continuing with system Python..."
fi

# Configuration
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export TOKENIZERS_PARALLELISM=false

# Distributed training settings
NUM_GPUS="${NUM_GPUS:-4}"
MASTER_PORT="${MASTER_PORT:-29500}"

# Memory optimization
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Training configuration
CONFIG_FILE="${CONFIG_FILE:-configs/config_qwen3_4b.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/orthgsa-qwen3-4b}"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, Devices: {torch.cuda.device_count()}')"

echo "============================================"
echo "OrthGSA Training (Cayley Transform)"
echo "============================================"
echo "Python: $(which python)"
echo "GPUs: $NUM_GPUS"
echo "Config: $CONFIG_FILE"
echo "Output: $OUTPUT_DIR"
echo "============================================"

# Launch distributed training with torchrun
torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=$MASTER_PORT \
    --nnodes=1 \
    --node_rank=0 \
    scripts/train.py \
    --config "$CONFIG_FILE" \
    "$@"

echo "Training complete!"

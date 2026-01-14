#!/bin/bash
# OrthGSA Training Launch Script
# For multi-GPU training (e.g., 4x 44GB GPUs)
# Uses Cayley Transform for orthogonal constraints instead of Sinkhorn-Knopp
#
# Training Methods:
#   - DeepSpeed ZeRO-2 (default): ~17GB per GPU - recommended for 24-48GB GPUs
#   - DDP (torchrun): ~44GB per GPU - use only with 80GB+ GPUs
#
# Usage:
#   ./scripts/launch_train.sh                    # DeepSpeed (default)
#   USE_DDP=1 ./scripts/launch_train.sh          # DDP (requires more memory)
#   NUM_GPUS=2 ./scripts/launch_train.sh         # DeepSpeed with 2 GPUs

set -e

# Get the script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
echo "cd $PROJECT_ROOT"
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
USE_DDP="${USE_DDP:-0}"  # Set USE_DDP=1 to use DDP instead of DeepSpeed

# Memory optimization
export PYTORCH_ALLOC_CONF=max_split_size_mb:512

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

if [ "$USE_DDP" = "1" ]; then
    echo "Method: DDP (torchrun) - ~44GB/GPU"
    echo "============================================"

    # Launch distributed training with torchrun (DDP)
    torchrun \
        --nproc_per_node=$NUM_GPUS \
        --master_port=$MASTER_PORT \
        --nnodes=1 \
        --node_rank=0 \
        scripts/train.py \
        --config "$CONFIG_FILE" \
        "$@"
else
    echo "Method: DeepSpeed ZeRO-2 - ~17GB/GPU (recommended)"
    echo "============================================"

    # Launch distributed training with DeepSpeed ZeRO-2 (recommended)
    deepspeed \
        --num_gpus=$NUM_GPUS \
        --master_port=$MASTER_PORT \
        scripts/train_deepspeed.py \
        --config "$CONFIG_FILE" \
        "$@"
fi

echo "Training complete!"

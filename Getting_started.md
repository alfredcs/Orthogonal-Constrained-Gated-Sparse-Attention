# Getting Started with OrthGSA

This guide provides step-by-step instructions for setting up the environment using **uv** (fast Python package manager), preparing the dataset, and launching training and evaluation for OrthGSA.

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Quick Start (Recommended)](#2-quick-start-recommended)
3. [Manual Environment Setup](#3-manual-environment-setup)
4. [Dataset Preparation](#4-dataset-preparation)
5. [Configuration](#5-configuration)
6. [Training](#6-training)
7. [Evaluation](#7-evaluation)
8. [Troubleshooting](#8-troubleshooting)

---

## 1. Prerequisites

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPUs | 1x 24GB | 4x 40-48GB (A100/H100/L40S/A6000) |
| RAM | 64GB | 128GB |
| Storage | 500GB SSD | 2TB NVMe SSD |
| CPU | 8 cores | 16+ cores |

**Supported GPUs**:
- **A100/H100** (40GB/80GB): Best performance, full precision support
- **L40S** (48GB): Excellent for training, good cost-efficiency
- **A6000/RTX 4090** (48GB/24GB): Good for smaller runs or inference
- **RTX 3090/4090** (24GB): Requires DeepSpeed ZeRO-2/3

### Software Requirements

- Linux (Ubuntu 20.04+ recommended)
- Python 3.10 or higher
- CUDA 11.8 or higher
- cuDNN 8.6 or higher

### Verify CUDA Installation

```bash
# Check NVIDIA driver
nvidia-smi

# Check CUDA version
nvcc --version
```

---

## 2. Quick Start (Recommended)

The fastest way to get started is using our automated setup script with **uv**.

### One-Command Setup

```bash
# Clone the repository
git clone https://github.com/your-org/OrthGSA.git
cd OrthGSA

# Run automated setup (installs uv if needed, creates venv, installs all deps)
chmod +x scripts/setup_env.sh
./scripts/setup_env.sh

# Activate the environment
source .venv/bin/activate

# Start training
./scripts/launch_train.sh
```

That's it! The setup script will:
1. Install `uv` if not already installed
2. Create a virtual environment at `.venv`
3. Install PyTorch with the correct CUDA version
4. Install all OrthGSA dependencies
5. Verify the installation

---

## 3. Manual Environment Setup

If you prefer manual setup or need custom configuration:

### Step 3.1: Install uv

```bash
# Install uv (fast Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Add to PATH (add to your .bashrc/.zshrc for persistence)
export PATH="$HOME/.cargo/bin:$PATH"

# Verify installation
uv --version
```

### Step 3.2: Clone the Repository

```bash
git clone https://github.com/your-org/OrthGSA.git
cd OrthGSA
```

### Step 3.3: Create Virtual Environment

```bash
# Create virtual environment with Python 3.11
uv venv .venv --python 3.11

# Activate the environment
source .venv/bin/activate
```

### Step 3.4: Install PyTorch with CUDA

Choose the appropriate PyTorch version for your CUDA:

```bash
# For CUDA 12.1 (recommended for latest GPUs)
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CUDA 11.8
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CPU only
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

Verify PyTorch installation:

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"
```

Expected output:
```
PyTorch: 2.1.0
CUDA available: True
GPU count: 4
```

### Step 3.5: Install OrthGSA and Dependencies

```bash
# Install OrthGSA in editable mode with all dependencies
uv pip install -e ".[dev]"

# Or install from pyproject.toml without editable mode
uv pip install .
```

### Step 3.6: Install Optional CUDA Dependencies

```bash
# Install triton for custom kernels
uv pip install triton>=2.1.0

# Install flash-attention (optional, for faster attention)
# Note: Requires compatible GPU and may need compilation
uv pip install flash-attn --no-build-isolation
```

### Step 3.7: Setup Weights & Biases (wandb)

```bash
# Login to wandb
wandb login

# Enter your API key when prompted
# Get your key at: https://wandb.ai/authorize
```

### Step 3.8: Setup Hugging Face Hub

```bash
# Login to Hugging Face (required for gated models)
huggingface-cli login

# Enter your token from: https://huggingface.co/settings/tokens
```

---

## 4. Dataset Preparation

OrthGSA uses the **SlimPajama-627B** dataset from Cerebras, available on Hugging Face.

### Step 4.1: Option A - Use Streaming (Default)

By default, the dataset streams from Hugging Face. Verify access:

```bash
python -c "
from datasets import load_dataset
ds = load_dataset('cerebras/SlimPajama-627B', split='train', streaming=True)
sample = next(iter(ds))
print('Dataset accessible!')
print(f'Sample keys: {sample.keys()}')
print(f'Text preview: {sample[\"text\"][:200]}...')
"
```

### Step 4.2: Option B - Use Pre-Downloaded Dataset (Recommended)

For faster training and to avoid Hugging Face rate limits, pre-download the dataset:

```bash
# Download SlimPajama-627B to local storage
# This requires ~2TB of disk space
python -c "
from datasets import load_dataset
ds = load_dataset('cerebras/SlimPajama-627B', split='train')
ds.save_to_disk('~/datasets/SlimPajama-627B')
"
```

Then configure the local path in `configs/config_qwen3_4b.yaml`:

```yaml
data:
  dataset: "cerebras/SlimPajama-627B"
  local_path: "~/datasets/SlimPajama-627B"  # Pre-downloaded dataset path
  num_workers: 2                            # Reduced for local loading
```

The training script automatically uses the local path if the directory exists and is not empty. Otherwise, it falls back to streaming from Hugging Face.

### Step 4.3: Download Base Model (Optional)

Pre-download the Qwen3-4B model:

```bash
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = 'Qwen/Qwen3-4B-Instruct-2507'
print(f'Downloading {model_name}...')

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype='auto',
)

print('Model downloaded successfully!')
print(f'Model parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B')
"
```

---

## 5. Configuration

### Step 5.1: Review Default Configuration

The default configuration is at `configs/config_qwen3_4b.yaml`:

```yaml
# Model settings
model:
  base_model: "Qwen/Qwen3-4B-Instruct-2507"
  orthgsa:
    n_streams: 4          # Number of parallel streams
    alpha_init: 0.01      # Initial residual connection strength
    cayley_scaling: 0.1   # Cayley transform scaling
  gsa:
    k_base: 512           # Base number of selected tokens
    adaptive_k: true      # Enable adaptive top-k

# Training settings
training:
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 8  # Effective batch = 4 * 8 * 4 GPUs = 128
  learning_rate: 2.0e-5
  max_steps: 100000
  bf16: true
```

### Step 5.2: Configuration for Different GPU Setups

**Single GPU (24GB)**:
```yaml
training:
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 32
```

**2x GPUs (48GB each)**:
```yaml
training:
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 16
```

**4x GPUs (48GB each, L40S/A6000)**:
```yaml
training:
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 8
data:
  max_seq_length: 2048
```

**8x GPUs (80GB each, A100/H100)**:
```yaml
training:
  per_device_train_batch_size: 8
  gradient_accumulation_steps: 4
data:
  max_seq_length: 4096
```

---

## 6. Training

### Step 6.1: Quick Test Run

```bash
# Activate environment (if not already)
source .venv/bin/activate

# Test with 100 steps
python scripts/train.py --config configs/config_qwen3_4b.yaml
```

### Step 6.2: Launch Single-GPU Training

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train.py \
    --config configs/config_qwen3_4b.yaml
```

### Step 6.3: Launch Multi-GPU Training with DeepSpeed (Recommended)

**DeepSpeed ZeRO-2** is the recommended method for multi-GPU training. It shards optimizer states across GPUs, reducing memory usage from ~44GB to ~17GB per GPU for a 4B parameter model.

Using DeepSpeed directly:

```bash
# 4 GPUs with DeepSpeed ZeRO-2
deepspeed --num_gpus=4 scripts/train_deepspeed.py \
    --config configs/config_qwen3_4b.yaml

# 2 GPUs
deepspeed --num_gpus=2 scripts/train_deepspeed.py \
    --config configs/config_qwen3_4b.yaml

# Specific GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3 deepspeed --num_gpus=4 \
    scripts/train_deepspeed.py --config configs/config_qwen3_4b.yaml
```

Using the launch script:

```bash
# Make script executable
chmod +x scripts/launch_train.sh

# Launch with default 4 GPUs (uses DeepSpeed)
./scripts/launch_train.sh

# Or customize
NUM_GPUS=2 CUDA_VISIBLE_DEVICES=0,1 ./scripts/launch_train.sh
```

#### Memory Requirements per GPU

| Training Method | Memory per GPU (4B model) | Suitable GPUs |
|----------------|---------------------------|---------------|
| DDP (torchrun) | ~44GB | A100-80GB, H100 |
| **DeepSpeed ZeRO-2** | **~17-22GB** | **A100/H100/L40S/A6000** (recommended) |
| DeepSpeed ZeRO-3 | ~12GB | RTX 3090/4090, any 24GB+ GPU |

### Step 6.4: Alternative: Multi-GPU with torchrun (DDP)

> **Note**: DDP requires more memory per GPU than DeepSpeed. Use only if you have 80GB+ GPUs (A100-80GB, H100).

```bash
# 4 GPUs with DDP
torchrun \
    --nproc_per_node=4 \
    --master_port=29500 \
    scripts/train.py \
    --config configs/config_qwen3_4b.yaml

# 8 GPUs across 2 nodes
torchrun \
    --nproc_per_node=4 \
    --nnodes=2 \
    --node_rank=0 \
    --master_addr="master_node_ip" \
    --master_port=29500 \
    scripts/train.py \
    --config configs/config_qwen3_4b.yaml
```

### Step 6.5: Resume Training

```bash
# Resume DeepSpeed training
deepspeed --num_gpus=4 scripts/train_deepspeed.py \
    --config configs/config_qwen3_4b.yaml \
    --resume outputs/orthgsa-qwen3-4b/checkpoint-10000

# Resume DDP training
python scripts/train.py \
    --config configs/config_qwen3_4b.yaml \
    --resume outputs/orthgsa-qwen3-4b/checkpoint-10000
```

### Step 6.6: Monitor Training

```bash
# Watch GPU utilization
watch -n 1 nvidia-smi

# View wandb dashboard
# https://wandb.ai/your-username/orthgsa-qwen3-4b
```

---

## 7. Evaluation

### Step 7.1: Evaluate Perplexity

```bash
python scripts/evaluate.py \
    --checkpoint outputs/orthgsa-qwen3-4b/checkpoint-10000
```

### Step 7.2: Full Evaluation Suite

```bash
python scripts/evaluate.py \
    --checkpoint outputs/orthgsa-qwen3-4b/checkpoint-10000 \
    --max_eval_steps 1000 \
    --eval_throughput \
    --eval_generation \
    --output eval_results_full.json
```

---

## 8. Troubleshooting

### Issue: uv not found

```bash
# Reinstall uv
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.cargo/bin:$PATH"
```

### Issue: CUDA Out of Memory with DDP (torchrun)

DDP requires the full model + optimizer states on each GPU (~44GB for 4B model). **Use DeepSpeed ZeRO-2 instead**:

```bash
# Switch to DeepSpeed ZeRO-2 (reduces memory to ~17GB per GPU)
deepspeed --num_gpus=4 scripts/train_deepspeed.py \
    --config configs/config_qwen3_4b.yaml
```

If you must use DDP, reduce memory usage:

```yaml
# Reduce batch size and sequence length in config
training:
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 32
data:
  max_seq_length: 512
```

### Issue: CUDA Out of Memory with DeepSpeed

```yaml
# If still OOM with DeepSpeed ZeRO-2, try:
training:
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 64  # Increase to maintain effective batch size
```

Or switch to ZeRO-3 by editing the deepspeed config in `scripts/train_deepspeed.py`:

```python
"zero_optimization": {
    "stage": 3,  # Changed from 2 to 3
    ...
}
```

### Issue: PyTorch CUDA mismatch

```bash
# Check CUDA version
nvcc --version

# Reinstall PyTorch with correct CUDA
uv pip uninstall torch torchvision torchaudio
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Issue: flash-attn installation fails

```bash
# flash-attn requires specific CUDA/GPU compatibility
# Skip if not needed - OrthGSA works without it
# Or try:
MAX_JOBS=4 uv pip install flash-attn --no-build-isolation
```

### Issue: Hugging Face Rate Limits

Download the dataset locally to avoid rate limits:

```bash
# Download to local storage
python -c "
from datasets import load_dataset
ds = load_dataset('cerebras/SlimPajama-627B', split='train')
ds.save_to_disk('~/datasets/SlimPajama-627B')
"
```

Then set `local_path` in config:

```yaml
data:
  local_path: "~/datasets/SlimPajama-627B"
```

### Issue: DeepSpeed installation fails

```bash
# Install DeepSpeed with proper CUDA
uv pip install deepspeed>=0.14.0

# If compilation fails, try pre-built wheels
pip install deepspeed --no-build-isolation
```

---

## Quick Reference

### Essential Commands

```bash
# Setup (one time)
./scripts/setup_env.sh

# Activate environment
source .venv/bin/activate

# Start training (DeepSpeed - recommended)
deepspeed --num_gpus=4 scripts/train_deepspeed.py --config configs/config_qwen3_4b.yaml

# Or use launch script
./scripts/launch_train.sh

# Single GPU training
python scripts/train.py --config configs/config_qwen3_4b.yaml

# Evaluate
python scripts/evaluate.py --checkpoint outputs/orthgsa-qwen3-4b/checkpoint-10000

# Monitor GPUs
watch -n 1 nvidia-smi
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `CUDA_VISIBLE_DEVICES` | GPUs to use | `0,1,2,3` |
| `NUM_GPUS` | Number of GPUs | `4` |
| `OUTPUT_DIR` | Output directory | `outputs/orthgsa-qwen3-4b` |
| `CONFIG_FILE` | Config file path | `configs/config_qwen3_4b.yaml` |

### Directory Structure

```
OrthGSA/
├── .venv/                   # uv virtual environment
├── configs/
│   ├── config_qwen3_4b.yaml # Main training configuration
│   └── deepspeed_zero2.json # DeepSpeed ZeRO-2 config (optional)
├── orthgsa/                 # Source code
├── outputs/                 # Training outputs
├── scripts/
│   ├── setup_env.sh         # Environment setup script
│   ├── launch_train.sh      # Training launch script
│   ├── train.py             # Single-GPU / DDP training script
│   ├── train_deepspeed.py   # DeepSpeed multi-GPU training script (recommended)
│   └── evaluate.py          # Evaluation script
├── pyproject.toml           # Project configuration (uv/pip)
└── Getting_started.md       # This guide
```

### Training Script Comparison

| Script | Use Case | Memory Efficiency |
|--------|----------|-------------------|
| `train.py` | Single GPU, DDP with 80GB+ GPUs | Low (full model per GPU) |
| `train_deepspeed.py` | Multi-GPU with 24-48GB GPUs | High (optimizer sharding) |

---

**Happy Training with OrthGSA!**

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
| RAM | 64GB | 128GB+ (256GB for 128K context with CPU offload) |
| Storage | 500GB SSD | 2TB NVMe SSD |
| CPU | 8 cores | 16+ cores |

**Supported GPUs**:
- **A100/H100** (40GB/80GB): Best performance, full precision support
- **L40S** (48GB): Excellent for training, good cost-efficiency
- **A6000/RTX 4090** (48GB/24GB): Good for smaller runs or inference
- **RTX 3090/4090** (24GB): Requires DeepSpeed ZeRO-2/3

**Model-Specific Requirements**:

| Model | Context | GPUs | Memory/GPU | DeepSpeed Stage |
|-------|---------|------|------------|-----------------|
| Qwen3-4B-Instruct | 1K | 4x | 17-22GB | ZeRO-2 |
| Qwen3-8B | 8K | 8x | 30-35GB | ZeRO-3 + CPU offload |
| Qwen3-8B | 16K | 8x | 35-40GB | ZeRO-3 + CPU offload |
| Qwen3-8B | 32K | 8x | 38-42GB | ZeRO-3 + CPU offload |
| Qwen3-8B | 64K | 8x | 40-44GB | ZeRO-3 + CPU offload |
| Qwen3-8B | 128K | 8x | 42-44GB | ZeRO-3 + CPU offload |

> **Note**: The model automatically uses Flash Attention 2 when available, which significantly reduces memory usage for long context training.

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

OrthGSA uses the **SlimPajama-627B** dataset, available via S3 bucket for direct streaming (no download required).

### Step 4.1: Option A - S3 Streaming (Default, Recommended)

By default, the dataset streams directly from S3 without downloading to local storage. This is the recommended approach as it:
- Requires no local storage for the dataset (~2TB)
- Avoids HuggingFace rate limits
- Works out of the box with the default configuration

**Requirements**: Install `boto3` for S3 access:

```bash
pip install boto3 botocore zstandard
```

**AWS Credentials Setup**: Configure your AWS credentials for S3 access:

```bash
# Option 1: Using AWS CLI
aws configure

# Option 2: Manually create credentials file
mkdir -p ~/.aws
cat > ~/.aws/credentials << 'EOF'
[default]
aws_access_key_id = YOUR_ACCESS_KEY
aws_secret_access_key = YOUR_SECRET_KEY
EOF
```

Verify S3 streaming access:

```bash
python -c "
import boto3
session = boto3.Session(profile_name='default')
s3 = session.client('s3')
response = s3.list_objects_v2(
    Bucket='public-datasets-multimodality',
    Prefix='SlimPajama-627B/train/',
    MaxKeys=5
)
print('S3 bucket accessible!')
print(f'Sample files: {[obj[\"Key\"] for obj in response.get(\"Contents\", [])]}')
"
```

The default configuration in all config files uses S3 streaming:

```yaml
data:
  dataset: "cerebras/SlimPajama-627B"  # Fallback HuggingFace identifier
  dataset_path: "s3://public-datasets-multimodality/SlimPajama-627B/"  # S3 path (streams directly, no download)
  streaming: true
```

### Step 4.2: Option B - HuggingFace Streaming (Alternative)

If you prefer to stream from HuggingFace instead of S3, remove or comment out the `dataset_path` in your config:

```yaml
data:
  dataset: "cerebras/SlimPajama-627B"
  # dataset_path: "s3://..."  # Comment out to use HuggingFace
  streaming: true
```

Verify HuggingFace access:

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

### Step 4.3: Option C - Pre-Downloaded Local Dataset

For offline training or maximum throughput, you can pre-download the dataset locally:

```bash
# Download SlimPajama-627B to local storage
# This requires ~2TB of disk space
python -c "
from datasets import load_dataset
ds = load_dataset('cerebras/SlimPajama-627B', split='train')
ds.save_to_disk('~/datasets/SlimPajama-627B')
"
```

Then configure the local path in your config (remove `dataset_path` to use local):

```yaml
data:
  dataset: "cerebras/SlimPajama-627B"
  # dataset_path: "s3://..."  # Comment out to use local path
  local_path: "~/datasets/SlimPajama-627B"  # Pre-downloaded dataset path
  num_workers: 2
```

**Priority order**: `dataset_path` (S3) > `local_path` > `dataset` (HuggingFace hub)

### Step 4.4: Download Base Model (Optional)

Pre-download the base model. Choose based on your training configuration:

**Option A: Qwen3-4B-Instruct** (for smaller context training):

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

**Option B: Qwen3-8B** (for long context 32K/64K/128K training):

Download the model to `/home/alfred/models/Qwen3-8B`:

```bash
# Using huggingface-cli (recommended)
huggingface-cli download Qwen/Qwen3-8B --local-dir /home/alfred/models/Qwen3-8B

# Or using Python
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path

model_name = 'Qwen/Qwen3-8B'
save_path = Path.home() / 'models' / 'Qwen3-8B'
save_path.mkdir(parents=True, exist_ok=True)

print(f'Downloading {model_name} to {save_path}...')

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.save_pretrained(save_path)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype='auto',
)
model.save_pretrained(save_path)

print('Model downloaded successfully!')
print(f'Model parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B')
"
```

> **Note**: Qwen3-8B is an 8B parameter text model. The long context configs use a local path (`/home/alfred/models/Qwen3-8B`) for faster loading.

---

## 5. Configuration

### Step 5.1: Available Configuration Files

OrthGSA provides pre-configured files for different models and context windows:

| Config File | Model | Context | GPUs | Use Case |
|-------------|-------|---------|------|----------|
| `config_qwen3_4b.yaml` | Qwen3-4B-Instruct | 1K | 4x 24-48GB | Default, smaller experiments |
| `config_qwen3_8b_8k.yaml` | Qwen3-8B | 8K | 8x 44GB | Base long context (recommended start) |
| `config_qwen3_8b_16k.yaml` | Qwen3-8B | 16K | 8x 44GB | Medium long context |
| `config_qwen3_8b_32k.yaml` | Qwen3-8B | 32K | 8x 44GB | Long context training |
| `config_qwen3_8b_64k.yaml` | Qwen3-8B | 64K | 8x 44GB | Extended context training |
| `config_qwen3_8b_128k.yaml` | Qwen3-8B | 128K | 8x 44GB | Maximum context training |

> **Note**: All Qwen3-8B configs use DeepSpeed ZeRO-3 with CPU offload and Flash Attention 2 for memory efficiency.

### Step 5.2: Qwen3-4B Configuration (Default)

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

### Step 5.3: Qwen3-8B Long Context Configurations

For Qwen3-8B with long context windows on **8x 44GB GPUs**:

**8K Context** (`configs/config_qwen3_8b_8k.yaml`) - Recommended starting point:
```yaml
model:
  base_model: "/home/alfred/models/Qwen3-8B"
  orthgsa:
    n_streams: 2          # Memory-optimized
  gsa:
    k_base: 512
    k_max: 1024
    indexer_heads: 4

data:
  max_seq_length: 8192    # 8K context
  num_workers: 2

training:
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 16   # Effective batch = 128
  learning_rate: 1.5e-5

distributed:
  deepspeed_config: "configs/deepspeed_zero3_8k.json"  # ZeRO-3 with CPU offload
```

**16K Context** (`configs/config_qwen3_8b_16k.yaml`):
```yaml
model:
  base_model: "/home/alfred/models/Qwen3-8B"
  orthgsa:
    n_streams: 2          # Memory-optimized
  gsa:
    k_base: 512
    k_max: 1024
    indexer_heads: 4

data:
  max_seq_length: 16384   # 16K context
  num_workers: 2

training:
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 16   # Effective batch = 128
  learning_rate: 1.5e-5

distributed:
  deepspeed_config: "configs/deepspeed_zero3_16k.json"  # ZeRO-3 with CPU offload
```

**32K Context** (`configs/config_qwen3_8b_32k.yaml`):
```yaml
model:
  base_model: "/home/alfred/models/Qwen3-8B"
  orthgsa:
    n_streams: 2          # Memory-optimized for long context
  gsa:
    k_base: 512
    k_max: 1024
    indexer_heads: 4

data:
  max_seq_length: 32768   # 32K context
  num_workers: 2

training:
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 16   # Effective batch = 128
  learning_rate: 1.5e-5

distributed:
  deepspeed_config: "configs/deepspeed_zero3_32k.json"  # ZeRO-3 with CPU offload
```

**128K Context** (`configs/config_qwen3_8b_128k.yaml`):
```yaml
model:
  base_model: "/home/alfred/models/Qwen3-8B"
  orthgsa:
    n_streams: 2          # Minimum for memory constraints
  gsa:
    k_base: 512           # Conservative for 128K
    k_max: 1024
    indexer_heads: 4

data:
  max_seq_length: 131072  # 128K context

training:
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 16   # Effective batch = 128
  learning_rate: 1.0e-5             # Lower for stability
  warmup_ratio: 0.05                # Longer warmup
  max_grad_norm: 0.5                # Tighter clipping

distributed:
  deepspeed_config: "configs/deepspeed_zero3_128k.json"  # ZeRO-3 with CPU offload
```

> **Note**: All long context configs use DeepSpeed ZeRO-3 with CPU offloading. Ensure you have sufficient system RAM (256GB+ recommended for 128K context).

### Step 5.4: Launch Commands for Each Configuration

```bash
# Qwen3-4B (default, 4x GPUs)
deepspeed --num_gpus=4 scripts/train_deepspeed.py --config configs/config_qwen3_4b.yaml

# ============================================
# Qwen3-8B Long Context Training (8x 44GB GPUs)
# ============================================
# Use PYTORCH_CUDA_ALLOC_CONF for memory optimization

# Qwen3-8B with 8K context (recommended starting point)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.6 \
deepspeed --num_gpus=8 scripts/train_deepspeed.py --config configs/config_qwen3_8b_8k.yaml

# Qwen3-8B with 16K context
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.6 \
deepspeed --num_gpus=8 scripts/train_deepspeed.py --config configs/config_qwen3_8b_16k.yaml

# Qwen3-8B with 32K context
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.6 \
deepspeed --num_gpus=8 scripts/train_deepspeed.py --config configs/config_qwen3_8b_32k.yaml

# Qwen3-8B with 64K context
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.6 \
deepspeed --num_gpus=8 scripts/train_deepspeed.py --config configs/config_qwen3_8b_64k.yaml

# Qwen3-8B with 128K context
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.6 \
deepspeed --num_gpus=8 scripts/train_deepspeed.py --config configs/config_qwen3_8b_128k.yaml
```

> **Important**: The `PYTORCH_CUDA_ALLOC_CONF` environment variable is essential for long context training. It enables expandable memory segments and aggressive garbage collection to reduce memory fragmentation.

### Step 5.5: Configuration for Different GPU Setups (Qwen3-4B)

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

### Step 5.6: Parameter Summary Table

| Config | n_streams | k_base | LR | Batch | Grad Accum | Eff. Batch | DeepSpeed |
|--------|-----------|--------|-----|-------|------------|------------|-----------|
| Qwen3-4B (1K) | 2-4 | 512 | 2.0e-5 | 1 | 32 | 128 | ZeRO-2 |
| Qwen3-8B (8K) | 2 | 512 | 1.5e-5 | 1 | 16 | 128 | ZeRO-3 |
| Qwen3-8B (16K) | 2 | 512 | 1.5e-5 | 1 | 16 | 128 | ZeRO-3 |
| Qwen3-8B (32K) | 2 | 512 | 1.5e-5 | 1 | 16 | 128 | ZeRO-3 |
| Qwen3-8B (64K) | 2 | 512 | 1.5e-5 | 1 | 16 | 128 | ZeRO-3 |
| Qwen3-8B (128K) | 2 | 512 | 1.0e-5 | 1 | 16 | 128 | ZeRO-3 |

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

OrthGSA provides a dedicated script for resuming DeepSpeed training from checkpoints.

#### Auto-Resume from Latest Checkpoint (Recommended)

The `resume_deepspeed.py` script automatically finds and resumes from the latest checkpoint:

```bash
# Auto-detect and resume from the latest checkpoint in the output directory
deepspeed --num_gpus=4 scripts/resume_deepspeed.py \
    --config configs/config_qwen3_4b.yaml

# Single GPU
python scripts/resume_deepspeed.py --config configs/config_qwen3_4b.yaml
```

**Key Features of `resume_deepspeed.py`:**

1. **Auto-detection**: Scans the output directory for `checkpoint-{step}` folders and automatically selects the one with the highest step number
2. **Checkpoint Loading**: Uses DeepSpeed's native `load_checkpoint()` to restore model weights, optimizer states, and learning rate scheduler
3. **DataLoader Fast-forwarding**: Approximates the correct position in the dataset by skipping batches
4. **W&B Integration**: Marks resumed runs with `-resumed-{step}` suffix for easy identification
5. **Progress Tracking**: Progress bar starts from the resumed step, showing accurate completion percentage

#### Resume from a Specific Checkpoint

To resume from a specific checkpoint instead of the latest:

```bash
# Specify exact checkpoint path
deepspeed --num_gpus=4 scripts/resume_deepspeed.py \
    --config configs/config_qwen3_4b.yaml \
    --checkpoint outputs/orthgsa-qwen3-4b/checkpoint-5000
```

#### Resume DDP Training

For DDP (torchrun) training, use the `--resume` flag with `train.py`:

```bash
# Resume DDP training from specific checkpoint
python scripts/train.py \
    --config configs/config_qwen3_4b.yaml \
    --resume outputs/orthgsa-qwen3-4b/checkpoint-10000

# Multi-GPU DDP resume
torchrun --nproc_per_node=4 scripts/train.py \
    --config configs/config_qwen3_4b.yaml \
    --resume outputs/orthgsa-qwen3-4b/checkpoint-10000
```

#### Checkpoint Structure

Checkpoints are saved in the output directory with the following structure:

```
outputs/orthgsa-qwen3-4b/
├── checkpoint-1000/
│   ├── global_step1000/      # DeepSpeed checkpoint files
│   │   ├── mp_rank_00_model_states.pt
│   │   ├── zero_pp_rank_0_mp_rank_00_optim_states.pt
│   │   └── ...
│   └── latest                # Marker file
├── checkpoint-2000/
│   └── ...
└── checkpoint-3000/
    └── ...
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

The default configuration uses S3 streaming which avoids HuggingFace rate limits entirely. If you're still using HuggingFace streaming and encounter rate limits, switch to S3 (recommended) or download locally:

**Option 1: Use S3 streaming (recommended)**

Ensure your config has `dataset_path` set:

```yaml
data:
  dataset_path: "s3://public-datasets-multimodality/SlimPajama-627B/"
```

**Option 2: Download to local storage**

```bash
# Download to local storage
python -c "
from datasets import load_dataset
ds = load_dataset('cerebras/SlimPajama-627B', split='train')
ds.save_to_disk('~/datasets/SlimPajama-627B')
"
```

Then set `local_path` in config (and remove `dataset_path`):

```yaml
data:
  # dataset_path: "s3://..."  # Comment out
  local_path: "~/datasets/SlimPajama-627B"
```

### Issue: S3 Dataset Access Fails

If you encounter errors accessing the S3 dataset:

```
botocore.exceptions.NoCredentialsError: Unable to locate credentials
```

**Solution**: Configure AWS credentials. The S3 bucket requires authentication:

```bash
# Option 1: Using AWS CLI
aws configure

# Option 2: Manually create credentials file
mkdir -p ~/.aws
cat > ~/.aws/credentials << 'EOF'
[default]
aws_access_key_id = YOUR_ACCESS_KEY
aws_secret_access_key = YOUR_SECRET_KEY
EOF
```

Ensure `boto3` is installed:

```bash
pip install boto3 botocore zstandard
```

Verify the bucket is accessible:

```bash
python -c "
import boto3
session = boto3.Session(profile_name='default')
s3 = session.client('s3')
print('Testing S3 access...')
response = s3.list_objects_v2(
    Bucket='public-datasets-multimodality',
    Prefix='SlimPajama-627B/',
    MaxKeys=5
)
print(f'Found {len(response.get(\"Contents\", []))} items. S3 access working!')
"
```

If S3 is blocked in your network, fall back to HuggingFace streaming by removing `dataset_path` from your config:

```yaml
data:
  dataset: "cerebras/SlimPajama-627B"
  # dataset_path: "s3://..."  # Comment out to use HuggingFace
```

### Issue: DeepSpeed installation fails

```bash
# Install DeepSpeed with proper CUDA
uv pip install deepspeed>=0.14.0

# If compilation fails, try pre-built wheels
pip install deepspeed --no-build-isolation
```

### Issue: "Compression type zstd not supported" error

The SlimPajama dataset uses zstd compression. If you see this error:

```
ValueError: Compression type zstd not supported
```

**Solution**: Install the `zstandard` package:

```bash
pip install zstandard>=0.22.0
```

The `zstandard` package is listed in `requirements.txt` and `pyproject.toml`, but if you installed dependencies manually, you may have missed it. The data loading code automatically registers zstd with fsspec when loaded.

If the error persists after installing `zstandard`, ensure you're using the correct Python environment:

```bash
# Verify zstandard is installed in your active environment
python -c "import zstandard; print('zstandard version:', zstandard.__version__)"

# Verify zstd is registered with fsspec
python -c "
import fsspec.compression
from orthgsa.data.slimpajama import _register_zstd_compression
print('zstd registered:', 'zstd' in fsspec.compression.compr)
"
```

---

## Quick Reference

### Essential Commands

```bash
# Setup (one time)
./scripts/setup_env.sh

# Activate environment
source .venv/bin/activate

# ============================================
# Qwen3-4B Training (default, 1K context)
# ============================================
deepspeed --num_gpus=4 scripts/train_deepspeed.py --config configs/config_qwen3_4b.yaml

# Or use launch script
./scripts/launch_train.sh

# ============================================
# Qwen3-8B Long Context Training (8x 44GB GPUs)
# ============================================
# IMPORTANT: Always use PYTORCH_CUDA_ALLOC_CONF for long context training

# 8K context (recommended starting point)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.6 \
deepspeed --num_gpus=8 scripts/train_deepspeed.py --config configs/config_qwen3_8b_8k.yaml

# 16K context
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.6 \
deepspeed --num_gpus=8 scripts/train_deepspeed.py --config configs/config_qwen3_8b_16k.yaml

# 32K context
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.6 \
deepspeed --num_gpus=8 scripts/train_deepspeed.py --config configs/config_qwen3_8b_32k.yaml

# 64K context
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.6 \
deepspeed --num_gpus=8 scripts/train_deepspeed.py --config configs/config_qwen3_8b_64k.yaml

# 128K context (uses ZeRO-3 with CPU offload)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.6 \
deepspeed --num_gpus=8 scripts/train_deepspeed.py --config configs/config_qwen3_8b_128k.yaml

# ============================================
# Resume Training
# ============================================
# Resume Qwen3-4B from latest checkpoint
deepspeed --num_gpus=4 scripts/resume_deepspeed.py --config configs/config_qwen3_4b.yaml

# Resume Qwen3-8B 8K from latest checkpoint
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.6 \
deepspeed --num_gpus=8 scripts/resume_deepspeed.py --config configs/config_qwen3_8b_8k.yaml

# Resume Qwen3-8B 32K from specific checkpoint
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.6 \
deepspeed --num_gpus=8 scripts/resume_deepspeed.py --config configs/config_qwen3_8b_32k.yaml \
    --checkpoint outputs/orthgsa-qwen3-8b-32k/checkpoint-5000

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
│   ├── config_qwen3_4b.yaml          # Qwen3-4B config (default)
│   ├── config_qwen3_8b_8k.yaml       # Qwen3-8B 8K context config (recommended start)
│   ├── config_qwen3_8b_16k.yaml      # Qwen3-8B 16K context config
│   ├── config_qwen3_8b_32k.yaml      # Qwen3-8B 32K context config
│   ├── config_qwen3_8b_64k.yaml      # Qwen3-8B 64K context config
│   ├── config_qwen3_8b_128k.yaml     # Qwen3-8B 128K context config
│   ├── deepspeed_zero2.json          # DeepSpeed ZeRO-2 config (default)
│   ├── deepspeed_zero3_8k.json       # DeepSpeed ZeRO-3 for 8K context
│   ├── deepspeed_zero3_16k.json      # DeepSpeed ZeRO-3 for 16K context
│   ├── deepspeed_zero3_32k.json      # DeepSpeed ZeRO-3 for 32K context
│   ├── deepspeed_zero3_64k.json      # DeepSpeed ZeRO-3 for 64K context
│   └── deepspeed_zero3_128k.json     # DeepSpeed ZeRO-3 for 128K context
├── orthgsa/                 # Source code
├── outputs/                 # Training outputs
├── scripts/
│   ├── setup_env.sh         # Environment setup script
│   ├── launch_train.sh      # Training launch script
│   ├── train.py             # Single-GPU / DDP training script
│   ├── train_deepspeed.py   # DeepSpeed multi-GPU training script (recommended)
│   ├── resume_deepspeed.py  # Resume training from checkpoint (DeepSpeed)
│   └── evaluate.py          # Evaluation script
├── pyproject.toml           # Project configuration (uv/pip)
└── Getting_started.md       # This guide
```

### Training Script Comparison

| Script | Use Case | Memory Efficiency |
|--------|----------|-------------------|
| `train.py` | Single GPU, DDP with 80GB+ GPUs | Low (full model per GPU) |
| `train_deepspeed.py` | Multi-GPU with 24-48GB GPUs | High (optimizer sharding) |
| `resume_deepspeed.py` | Resume DeepSpeed training from checkpoint | High (optimizer sharding) |

---

**Happy Training with OrthGSA!**

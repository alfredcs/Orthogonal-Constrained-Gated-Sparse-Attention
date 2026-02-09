#!/usr/bin/env python3
"""
OrthGSA Training Script with Ring Attention for Ultra-Long Context

This script extends the DeepSpeed training with Ring Attention for
sequence parallelism, enabling training on 256K+ context windows.

Key features:
- Ring Attention: Distributes sequence across GPUs
- ZeRO-Infinity: Offloads optimizer and parameters to CPU/NVMe
- Gradient Checkpointing: Reduces activation memory

Usage:
    deepspeed --num_gpus=8 scripts/train_ring_attention.py --config configs/config_qwen3_1.7b_256k.yaml
"""

import os
import sys
import argparse
import logging
import math
import re
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

import torch
import torch.distributed as dist
import deepspeed
from transformers import AutoTokenizer
import yaml
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from orthgsa.models import OrthGSAForCausalLM, OrthGSAConfig
from orthgsa.data import get_slimpajama_dataloader
from orthgsa.utils import (
    log_metrics,
    count_parameters,
    get_gpu_memory_info,
)


def compute_gradient_norm(model_engine, norm_type: float = 2.0) -> float:
    """
    Compute gradient norm across all parameters with DeepSpeed ZeRO-3 support.

    Args:
        model_engine: DeepSpeed model engine
        norm_type: Type of norm (default: L2)

    Returns:
        Total gradient norm as a float
    """
    # DeepSpeed tracks gradient norm internally after clipping
    # We can also compute it manually for more detailed analysis
    total_norm = 0.0

    # For ZeRO-3, gradients are partitioned. We need to gather them or
    # use DeepSpeed's built-in tracking
    if hasattr(model_engine, 'get_global_grad_norm'):
        # DeepSpeed provides this after backward pass
        return model_engine.get_global_grad_norm()

    # Fallback: compute manually (works for ZeRO-1/2 or after gradient gathering)
    parameters = [p for p in model_engine.module.parameters() if p.grad is not None]

    if len(parameters) == 0:
        return 0.0

    device = parameters[0].grad.device

    for p in parameters:
        param_norm = p.grad.data.float().norm(norm_type)
        total_norm += param_norm.item() ** norm_type

    total_norm = total_norm ** (1.0 / norm_type)

    # All-reduce across ranks for distributed training
    if dist.is_initialized():
        total_norm_tensor = torch.tensor([total_norm], device=device)
        dist.all_reduce(total_norm_tensor, op=dist.ReduceOp.MAX)
        total_norm = total_norm_tensor.item()

    return total_norm


def compute_parameter_norms(model, prefix: str = "") -> Dict[str, float]:
    """
    Compute per-component parameter and gradient norms for diagnostics.

    Useful for identifying which component is causing gradient explosion.
    """
    norms = {}

    # Group parameters by component
    components = {
        "mhc": [],
        "base_model": [],
        "streams": [],
    }

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "mhc" in name:
            components["mhc"].append((name, param))
        elif "expand_streams" in name or "collapse_streams" in name:
            components["streams"].append((name, param))
        else:
            components["base_model"].append((name, param))

    for comp_name, params in components.items():
        if not params:
            continue

        # Parameter norm
        param_norm = sum(p.data.float().norm().item() ** 2 for _, p in params) ** 0.5
        norms[f"{prefix}{comp_name}_param_norm"] = param_norm

        # Gradient norm (if available)
        grads = [(n, p) for n, p in params if p.grad is not None]
        if grads:
            grad_norm = sum(p.grad.data.float().norm().item() ** 2 for _, p in grads) ** 0.5
            norms[f"{prefix}{comp_name}_grad_norm"] = grad_norm

    return norms
from orthgsa.layers.ring_attention import (
    split_sequence_for_ring_attention,
    gather_sequence_from_ring_attention,
    RingCommunicator,
)

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def setup_wandb(config: Dict[str, Any], rank: int, model=None) -> Optional[Any]:
    """Setup Weights & Biases logging."""
    wandb_config = config.get("logging", {}).get("wandb", {})
    enabled = wandb_config.get("enabled", False)

    # Use print for guaranteed visibility (logger might be suppressed)
    print(f"[WANDB DEBUG] setup_wandb called: enabled={enabled}, rank={rank}", flush=True)
    logger.info(f"Wandb setup: enabled={enabled}, rank={rank}")

    if not enabled:
        print(f"[WANDB DEBUG] Wandb disabled in config", flush=True)
        logger.info("Wandb disabled in config (logging.wandb.enabled=false)")
        return None

    if rank != 0:
        print(f"[WANDB DEBUG] Skipping wandb on rank {rank}", flush=True)
        logger.info(f"Wandb skipped on rank {rank} (only rank 0 logs to wandb)")
        return None

    try:
        import wandb
        print(f"[WANDB DEBUG] wandb module imported successfully", flush=True)

        run_name = wandb_config.get("run_name") or f"orthgsa-ring-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        project = wandb_config.get("project", "orthgsa")
        entity = wandb_config.get("entity")
        log_code = wandb_config.get("log_code", False)

        print(f"[WANDB DEBUG] Calling wandb.init(project={project}, run_name={run_name})", flush=True)
        logger.info(f"Initializing wandb: project={project}, run_name={run_name}")

        wandb.init(
            project=project,
            entity=entity,
            name=run_name,
            tags=wandb_config.get("tags", []) + ["ring-attention"],
            config=config,
            save_code=log_code,
        )

        # Watch model for gradient logging if specified
        watch_mode = wandb_config.get("watch")
        if watch_mode and model is not None:
            log_freq = wandb_config.get("watch_log_freq", 100)
            logger.info(f"Setting up wandb.watch with mode={watch_mode}, log_freq={log_freq}")
            wandb.watch(model, log=watch_mode, log_freq=log_freq)

        print(f"[WANDB DEBUG] wandb.init() completed: {wandb.run.url}", flush=True)
        logger.info(f"Wandb initialized successfully: {wandb.run.url}")
        return wandb

    except ImportError as e:
        print(f"[WANDB DEBUG] ImportError: {e}", flush=True)
        logger.warning("wandb not installed, skipping wandb logging")
        return None
    except Exception as e:
        print(f"[WANDB DEBUG] Exception: {e}", flush=True)
        logger.error(f"Failed to initialize wandb: {e}")
        return None


class SequenceParallelDataLoader:
    """
    Wrapper that handles sequence splitting for ring attention.

    Each GPU gets a chunk of the sequence, with proper handling
    of labels across chunk boundaries.
    """

    def __init__(
        self,
        dataloader: torch.utils.data.DataLoader,
        chunk_size: int,
        world_size: int,
        rank: int,
    ):
        self.dataloader = dataloader
        self.chunk_size = chunk_size
        self.world_size = world_size
        self.rank = rank
        self.total_seq_length = chunk_size * world_size

    def __iter__(self):
        for batch in self.dataloader:
            # Original batch: [batch_size, seq_len]
            input_ids = batch["input_ids"]
            labels = batch["labels"]

            batch_size, seq_len = input_ids.shape

            # Ensure sequence is properly padded/truncated
            if seq_len < self.total_seq_length:
                # Pad sequences
                pad_length = self.total_seq_length - seq_len
                input_ids = torch.nn.functional.pad(
                    input_ids, (0, pad_length), value=0
                )
                labels = torch.nn.functional.pad(
                    labels, (0, pad_length), value=-100
                )
            elif seq_len > self.total_seq_length:
                # Truncate sequences
                input_ids = input_ids[:, :self.total_seq_length]
                labels = labels[:, :self.total_seq_length]

            # Split for this rank
            start_idx = self.rank * self.chunk_size
            end_idx = start_idx + self.chunk_size

            chunk_input_ids = input_ids[:, start_idx:end_idx].contiguous()
            chunk_labels = labels[:, start_idx:end_idx].contiguous()

            yield {
                "input_ids": chunk_input_ids,
                "labels": chunk_labels,
                "original_seq_len": seq_len,
            }

    def __len__(self):
        return len(self.dataloader)


def ring_attention_forward(
    model: OrthGSAForCausalLM,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    communicator: RingCommunicator,
    device: torch.device,
) -> torch.Tensor:
    """
    Forward pass with ring attention.

    This function handles the distributed computation across GPUs,
    where each GPU processes a chunk of the sequence.

    Args:
        model: The OrthGSA model
        input_ids: Local chunk of input IDs [batch, chunk_len]
        labels: Local chunk of labels [batch, chunk_len]
        communicator: Ring communication handler
        device: Current device

    Returns:
        Loss tensor
    """
    batch_size, chunk_len = input_ids.shape
    world_size = communicator.world_size
    rank = communicator.rank

    # Compute global position IDs for this chunk
    global_offset = rank * chunk_len
    position_ids = torch.arange(
        global_offset,
        global_offset + chunk_len,
        device=device,
    ).unsqueeze(0).expand(batch_size, -1)

    # Forward pass through model
    # The model should use ring attention internally
    outputs = model(
        input_ids=input_ids,
        attention_mask=None,  # Ring attention handles masking
        position_ids=position_ids,
        labels=labels,
    )

    return outputs.loss


def evaluate_ring_attention(
    model_engine: deepspeed.DeepSpeedEngine,
    eval_dataloader: SequenceParallelDataLoader,
    device: torch.device,
    communicator: RingCommunicator,
    max_eval_steps: int = 50,
) -> Dict[str, float]:
    """Evaluate the model with ring attention."""
    model_engine.eval()

    total_loss = 0.0
    total_tokens = 0
    num_steps = 0

    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="Evaluating", total=max_eval_steps):
            if num_steps >= max_eval_steps:
                break

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            loss = ring_attention_forward(
                model_engine.module,
                input_ids,
                labels,
                communicator,
                device,
            )

            # Count valid tokens (non -100 labels)
            valid_tokens = (labels != -100).sum().item()
            total_loss += loss.item() * valid_tokens
            total_tokens += valid_tokens
            num_steps += 1

    # All-reduce loss across GPUs
    total_loss_tensor = torch.tensor([total_loss], device=device)
    total_tokens_tensor = torch.tensor([total_tokens], device=device)

    if dist.is_initialized():
        dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tokens_tensor, op=dist.ReduceOp.SUM)

    avg_loss = total_loss_tensor.item() / (total_tokens_tensor.item() + 1e-8)
    perplexity = math.exp(avg_loss) if avg_loss < 20 else float("inf")

    return {
        "eval_loss": avg_loss,
        "perplexity": perplexity,
    }


def save_hf_checkpoint(
    model_engine,
    model: OrthGSAForCausalLM,
    save_dir: str,
    rank: int,
    orthgsa_config: OrthGSAConfig,
):
    """Save checkpoint in HuggingFace-compatible format."""
    import json

    hf_save_dir = os.path.join(save_dir, "hf_model")
    os.makedirs(hf_save_dir, exist_ok=True)

    # Save full 16-bit model
    model_engine.save_16bit_model(hf_save_dir, save_filename="model.safetensors")

    # Save config (only rank 0)
    if rank == 0:
        config_dict = model.config.to_dict()
        config_dict['orthgsa'] = orthgsa_config.to_dict()
        with open(os.path.join(hf_save_dir, 'config.json'), 'w') as f:
            json.dump(config_dict, f, indent=2)

    # Save OrthGSA weights
    # GatheredParameters is a COLLECTIVE operation - ALL ranks must participate
    # Only rank 0 (modifier_rank=0) will do the actual save
    with deepspeed.zero.GatheredParameters(
        list(model.mhc_modules.parameters()) + [model.expand_streams, model.collapse_streams],
        modifier_rank=0
    ):
        if rank == 0:
            orthgsa_state = {
                'mhc_modules': model.mhc_modules.state_dict(),
                'expand_streams': model.expand_streams.data.clone(),
                'collapse_streams': model.collapse_streams.data.clone(),
            }
            torch.save(orthgsa_state, os.path.join(hf_save_dir, 'orthgsa_weights.pt'))
            logger.info(f"HuggingFace checkpoint saved to {hf_save_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train OrthGSA with Ring Attention")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank")
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Get distributed info
    # Check if we're in a distributed environment (DeepSpeed launcher sets these)
    is_distributed = (
        args.local_rank != -1 and
        os.environ.get("WORLD_SIZE") is not None and
        int(os.environ.get("WORLD_SIZE", "1")) > 1
    )

    if is_distributed:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        deepspeed.init_distributed()
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        # Single GPU mode - even if local_rank=0 is passed
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        rank = 0
        world_size = 1
        if args.local_rank != -1:
            torch.cuda.set_device(args.local_rank)
            device = torch.device("cuda", args.local_rank)
            # Initialize minimal distributed environment for DeepSpeed
            os.environ.setdefault("MASTER_ADDR", "localhost")
            os.environ.setdefault("MASTER_PORT", "29500")
            os.environ.setdefault("WORLD_SIZE", "1")
            os.environ.setdefault("RANK", "0")
            os.environ.setdefault("LOCAL_RANK", str(args.local_rank))
            deepspeed.init_distributed()

    # Setup logging
    if rank != 0:
        logging.getLogger().setLevel(logging.WARNING)

    logger.info(f"Starting Ring Attention training with config: {args.config}")
    logger.info(f"Distributed mode: {is_distributed}")
    logger.info(f"Rank: {rank}, World Size: {world_size}, Device: {device}")
    logger.info(f"Environment: WORLD_SIZE={os.environ.get('WORLD_SIZE')}, RANK={os.environ.get('RANK')}, LOCAL_RANK={os.environ.get('LOCAL_RANK')}")

    # Setup wandb EARLY - before model loading so we catch any crashes
    print(f"[WANDB DEBUG] About to call setup_wandb with rank={rank}", flush=True)
    wandb_run = setup_wandb(config, rank, model=None)
    print(f"[WANDB DEBUG] setup_wandb returned: {wandb_run is not None}", flush=True)

    # Get sequence parallel config
    sp_config = config.get("distributed", {}).get("sequence_parallel", {})
    sp_enabled = sp_config.get("enabled", False)
    sp_degree = sp_config.get("degree", world_size)
    chunk_size = sp_config.get("chunk_size", 32768)

    if sp_enabled:
        logger.info(f"Sequence Parallelism enabled: degree={sp_degree}, chunk_size={chunk_size}")
    else:
        logger.info("Sequence Parallelism disabled, using standard training")

    # Ring communicator
    communicator = RingCommunicator()

    # Load tokenizer
    model_config = config["model"]
    tokenizer = AutoTokenizer.from_pretrained(
        model_config["base_model"],
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create OrthGSA config
    orthgsa_cfg = OrthGSAConfig(
        n_streams=model_config["orthgsa"]["n_streams"],
        alpha_init=model_config["orthgsa"]["alpha_init"],
        k_base=model_config["gsa"]["k_base"],
        k_min=model_config["gsa"]["k_min"],
        k_max=model_config["gsa"]["k_max"],
        indexer_heads=model_config["gsa"]["indexer_heads"],
        indexer_dim=model_config["gsa"]["indexer_dim"],
        adaptive_k=model_config["gsa"]["adaptive_k"],
    )

    # Create model
    logger.info(f"Loading model: {model_config['base_model']}")
    torch.cuda.empty_cache()

    data_config = config["data"]
    rope_scaling = model_config.get("rope_scaling")

    # For ring attention, each GPU processes chunk_size tokens
    # Total sequence = chunk_size * world_size
    if sp_enabled:
        max_position_embeddings = chunk_size * sp_degree
    else:
        max_position_embeddings = data_config.get("max_seq_length")

    model = OrthGSAForCausalLM(
        base_model_name=model_config["base_model"],
        orthgsa_config=orthgsa_cfg,
        torch_dtype=torch.bfloat16 if config["training"]["bf16"] else torch.float32,
        device_map="cpu",
        low_cpu_mem_usage=True,
        max_position_embeddings=max_position_embeddings,
        rope_scaling=rope_scaling,
    )

    # Log parameters
    param_info = count_parameters(model)
    logger.info(f"Model parameters: {param_info['total_millions']:.2f}M total")

    # Enable gradient checkpointing
    if config["training"].get("gradient_checkpointing", True):
        if hasattr(model.base_model, "gradient_checkpointing_enable"):
            model.base_model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")

    # Setup wandb model watching for gradient logging (wandb was initialized earlier)
    if wandb_run is not None:
        wandb_config = config.get("logging", {}).get("wandb", {})
        watch_mode = wandb_config.get("watch")
        if watch_mode:
            log_freq = wandb_config.get("watch_log_freq", 100)
            logger.info(f"Setting up wandb.watch with mode={watch_mode}, log_freq={log_freq}")
            wandb_run.watch(model, log=watch_mode, log_freq=log_freq)

    # Create data loaders
    training_config = config["training"]
    dataset_name = data_config.get("dataset", "cerebras/SlimPajama-627B")
    dataset_path = data_config.get("dataset_path")
    local_path = data_config.get("local_path")

    # Base data loader - loads full sequences
    base_train_loader = get_slimpajama_dataloader(
        tokenizer=tokenizer,
        batch_size=training_config["per_device_train_batch_size"],
        max_length=data_config["max_seq_length"],
        split="train",
        num_workers=data_config["num_workers"],
        seed=config.get("seed", 42),
        rank=rank,
        world_size=world_size,
        packed=True,
        dataset_name=dataset_name,
        local_path=local_path,
        dataset_path=dataset_path,
    )

    # Wrap with sequence parallel splitter
    if sp_enabled:
        train_dataloader = SequenceParallelDataLoader(
            base_train_loader,
            chunk_size=chunk_size,
            world_size=sp_degree,
            rank=rank,
        )
    else:
        train_dataloader = base_train_loader

    # Eval dataloader
    eval_dataloader = None
    if config.get("evaluation", {}).get("do_eval", True):
        base_eval_loader = get_slimpajama_dataloader(
            tokenizer=tokenizer,
            batch_size=training_config["per_device_eval_batch_size"],
            max_length=data_config["max_seq_length"],
            split="train",
            num_workers=data_config["num_workers"],
            seed=config.get("seed", 42) + 1000,
            rank=rank,
            world_size=world_size,
            packed=True,
            dataset_name=dataset_name,
            local_path=local_path,
            dataset_path=dataset_path,
        )
        if sp_enabled:
            eval_dataloader = SequenceParallelDataLoader(
                base_eval_loader,
                chunk_size=chunk_size,
                world_size=sp_degree,
                rank=rank,
            )
        else:
            eval_dataloader = base_eval_loader

    # Load DeepSpeed config
    distributed_config = config.get("distributed", {})
    ds_config_path = distributed_config.get("deepspeed_config")

    if ds_config_path and not os.path.isabs(ds_config_path):
        project_root = Path(__file__).parent.parent
        ds_config_path = str(project_root / ds_config_path)

    if ds_config_path and os.path.exists(ds_config_path):
        logger.info(f"Loading DeepSpeed config from: {ds_config_path}")
        import json
        with open(ds_config_path, "r") as f:
            ds_config = json.load(f)

        # Override batch sizes
        ds_config["train_batch_size"] = (
            training_config["per_device_train_batch_size"]
            * world_size
            * training_config["gradient_accumulation_steps"]
        )
        ds_config["train_micro_batch_size_per_gpu"] = training_config["per_device_train_batch_size"]
        ds_config["gradient_accumulation_steps"] = training_config["gradient_accumulation_steps"]

        # Add optimizer
        if "optimizer" not in ds_config:
            ds_config["optimizer"] = {
                "type": "AdamW",
                "params": {
                    "lr": training_config["learning_rate"],
                    "betas": [training_config["adam_beta1"], training_config["adam_beta2"]],
                    "eps": training_config["adam_epsilon"],
                    "weight_decay": training_config["weight_decay"],
                },
            }

        # Add scheduler
        if "scheduler" not in ds_config:
            ds_config["scheduler"] = {
                "type": "WarmupDecayLR",
                "params": {
                    "warmup_min_lr": 1e-6,
                    "warmup_max_lr": training_config["learning_rate"],
                    "warmup_num_steps": int(
                        training_config["max_steps"] * training_config["warmup_ratio"]
                    ),
                    "total_num_steps": training_config["max_steps"],
                },
            }
    else:
        raise ValueError(f"DeepSpeed config not found: {ds_config_path}")

    # Initialize DeepSpeed
    model_engine, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=ds_config,
    )

    logger.info("DeepSpeed engine initialized")

    # Clear cache
    torch.cuda.empty_cache()
    if rank == 0:
        gpu_mem = get_gpu_memory_info()
        logger.info(
            f"GPU memory before training: {gpu_mem.get('allocated_gb', 0):.2f}GB allocated"
        )

    # Training loop
    max_steps = training_config["max_steps"]
    logging_steps = training_config["logging_steps"]
    eval_steps = training_config["eval_steps"]
    save_steps = training_config["save_steps"]

    output_dir = config["logging"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    global_step = 0

    # Resume from checkpoint if specified
    if args.resume:
        checkpoint_dir = args.resume

        # Convert to absolute path if relative
        if not os.path.isabs(checkpoint_dir):
            checkpoint_dir = os.path.abspath(checkpoint_dir)

        logger.info(f"Resuming from checkpoint: {checkpoint_dir}")

        # Verify checkpoint directory exists
        if not os.path.exists(checkpoint_dir):
            raise ValueError(f"Checkpoint directory not found: {checkpoint_dir}")

        # Check for 'latest' file
        latest_file = os.path.join(checkpoint_dir, "latest")
        load_tag = None
        if os.path.exists(latest_file):
            with open(latest_file, "r") as f:
                load_tag = f.read().strip()
            logger.info(f"Found checkpoint tag: {load_tag}")
            checkpoint_subdir = os.path.join(checkpoint_dir, load_tag)
            if os.path.exists(checkpoint_subdir):
                files = os.listdir(checkpoint_subdir)
                model_files = [f for f in files if "model_states" in f]
                logger.info(f"Checkpoint contains {len(model_files)} model state files")
            else:
                logger.warning(f"Checkpoint subdirectory not found: {checkpoint_subdir}")
        else:
            logger.warning(f"No 'latest' file found in checkpoint directory")

        # Load checkpoint - DeepSpeed will read the 'latest' file to find the actual checkpoint
        # Pass the tag explicitly if we found it to avoid any path resolution issues
        logger.info(f"Loading checkpoint with tag={load_tag}")

        # Clear CUDA cache before loading to avoid memory fragmentation
        torch.cuda.empty_cache()
        if dist.is_initialized():
            dist.barrier()
            logger.info("All ranks synchronized before checkpoint loading")

        # Load checkpoint with explicit error handling
        try:
            _, client_state = model_engine.load_checkpoint(checkpoint_dir, tag=load_tag)
            logger.info("Checkpoint loaded successfully")
        except ValueError as e:
            if "parameter group" in str(e):
                logger.warning(f"Optimizer state mismatch, trying to load model weights only: {e}")
                # Try loading without optimizer states
                _, client_state = model_engine.load_checkpoint(
                    checkpoint_dir, tag=load_tag, load_optimizer_states=False
                )
                logger.warning("Loaded model weights only (optimizer state skipped due to config mismatch)")
            else:
                logger.error(f"Failed to load checkpoint: {e}")
                raise
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise

        # Synchronize after loading
        if dist.is_initialized():
            dist.barrier()
            logger.info("All ranks synchronized after checkpoint loading")

        torch.cuda.empty_cache()

        # Get the global step from DeepSpeed engine
        global_step = model_engine.global_steps

        # If global_step wasn't restored properly, try to extract from path
        if global_step == 0:
            # Try to get step from checkpoint path (e.g., checkpoint-500)
            match = re.search(r"checkpoint-(\d+)", checkpoint_dir)
            if match:
                global_step = int(match.group(1))
            elif load_tag:
                # Extract step from tag like "global_step499"
                step_match = re.search(r"global_step(\d+)", load_tag)
                if step_match:
                    global_step = int(step_match.group(1)) + 1  # +1 because it's 0-indexed

        logger.info(f"Resumed training from step {global_step}")

        # Fix: Manually advance LR scheduler to match resumed step
        # DeepSpeed's WarmupDecayLR doesn't always restore step count properly
        if global_step > 0 and lr_scheduler is not None:
            # Get current scheduler step
            current_lr_step = getattr(lr_scheduler, 'num_steps', 0)
            if current_lr_step < global_step:
                steps_to_advance = global_step - current_lr_step
                logger.info(f"Advancing LR scheduler by {steps_to_advance} steps to sync with global_step={global_step}")
                for _ in range(steps_to_advance):
                    lr_scheduler.step()
                current_lr = lr_scheduler.get_last_lr()[0] if hasattr(lr_scheduler, 'get_last_lr') else lr_scheduler.get_lr()[0]
                logger.info(f"LR scheduler synced. Current learning rate: {current_lr:.2e}")

    logger.info("Starting training with Ring Attention...")

    accumulated_loss = 0.0
    accumulated_grad_norm = 0.0
    grad_norm_count = 0
    max_grad_norm_seen = 0.0
    prev_loss = None  # For spike detection
    train_iterator = iter(train_dataloader)

    # Fix: Fast-forward dataloader to resume position
    # When resuming, the dataloader starts from the beginning, so we need to skip
    # the batches that were already processed
    # NOTE: We add frequent barriers to keep ranks synchronized during S3 streaming
    #
    # Set SKIP_DATA_FASTFORWARD=1 to skip this step (faster resume, but may re-see some data)
    # For large streaming datasets like SlimPajama, this is usually acceptable
    skip_fastforward = os.environ.get("SKIP_DATA_FASTFORWARD", "0") == "1"

    if global_step > 0 and not skip_fastforward:
        # Calculate micro-batches to skip (each global step = gradient_accumulation_steps micro-batches)
        skip_micro_batches = global_step * training_config["gradient_accumulation_steps"]
        logger.info(f"Fast-forwarding dataloader: skipping {skip_micro_batches} micro-batches to resume at step {global_step}")
        logger.info("Set SKIP_DATA_FASTFORWARD=1 to skip this step for faster resume")

        # Synchronize before starting fast-forward
        if dist.is_initialized():
            dist.barrier()
            logger.info("All ranks synchronized before fast-forward")

        # Use a faster skip method - just iterate without processing
        # Add synchronization every N batches to prevent rank desynchronization
        sync_interval = 100  # Sync every 100 batches to prevent NCCL timeout
        skip_progress = tqdm(
            range(skip_micro_batches),
            desc="Fast-forwarding data",
            disable=rank != 0,
            mininterval=5.0  # Update progress less frequently
        )
        for i in skip_progress:
            try:
                next(train_iterator)
            except StopIteration:
                # Dataset exhausted, wrap around
                train_iterator = iter(train_dataloader)
                next(train_iterator)

            # Periodic synchronization to keep ranks aligned during S3 streaming
            if (i + 1) % sync_interval == 0 and dist.is_initialized():
                dist.barrier()

        logger.info(f"Dataloader fast-forward complete, resuming training from step {global_step}")

        # Final synchronization after fast-forward
        if dist.is_initialized():
            dist.barrier()
            logger.info("All ranks synchronized after fast-forward")

    elif global_step > 0 and skip_fastforward:
        logger.info(f"SKIP_DATA_FASTFORWARD=1: Skipping dataloader fast-forward (resuming from step {global_step})")
        logger.info("Note: Model may re-see some training data, which is acceptable for large streaming datasets")
        # Still synchronize all ranks
        if dist.is_initialized():
            dist.barrier()

    progress_bar = tqdm(total=max_steps, initial=global_step, desc="Training", disable=rank != 0)

    while global_step < max_steps:
        model_engine.train()

        try:
            batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_dataloader)
            batch = next(train_iterator)

        # Move to device
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        # Compute global position IDs for this chunk
        batch_size, chunk_len = input_ids.shape
        global_offset = rank * chunk_size if sp_enabled else 0
        position_ids = torch.arange(
            global_offset,
            global_offset + chunk_len,
            device=device,
        ).unsqueeze(0).expand(batch_size, -1)

        # Forward pass
        outputs = model_engine(
            input_ids=input_ids,
            attention_mask=None,
            position_ids=position_ids,
            labels=labels,
        )
        loss = outputs.loss

        # Backward
        model_engine.backward(loss)

        # Compute gradient norm BEFORE optimizer step (while gradients exist)
        # This is critical for diagnosing gradient explosion
        current_grad_norm = compute_gradient_norm(model_engine)
        accumulated_grad_norm += current_grad_norm
        grad_norm_count += 1
        max_grad_norm_seen = max(max_grad_norm_seen, current_grad_norm)

        model_engine.step()

        accumulated_loss += loss.item()

        # Free memory aggressively
        del outputs, batch, input_ids, labels, position_ids

        # Clear cache every micro-batch to prevent memory accumulation
        torch.cuda.empty_cache()

        # Check gradient accumulation boundary
        if model_engine.is_gradient_accumulation_boundary():
            global_step += 1
            progress_bar.update(1)

            # Aggressive memory management every step
            import gc
            gc.collect()
            torch.cuda.empty_cache()

            # Synchronize ranks periodically to catch hangs early
            if global_step % 10 == 0:
                if dist.is_initialized():
                    dist.barrier()

            # Log memory usage periodically to help diagnose issues
            if global_step % 50 == 0 and rank == 0:
                gpu_mem = get_gpu_memory_info()
                logger.info(f"Step {global_step}: GPU memory {gpu_mem.get('allocated_gb', 0):.2f}GB / {gpu_mem.get('total_gb', 0):.2f}GB")

            # Extra detailed logging in the problematic 1k-1.1k step range
            # This helps diagnose the spike issue
            if 950 <= global_step <= 1150 and global_step % 5 == 0 and rank == 0:
                instant_grad_norm = compute_gradient_norm(model_engine)
                logger.info(
                    f"[DETAILED] Step {global_step}: "
                    f"instant_grad_norm={instant_grad_norm:.6f}, "
                    f"accumulated_grad_norm={accumulated_grad_norm:.6f}, "
                    f"max_seen={max_grad_norm_seen:.6f}"
                )
                # Log component norms every 10 steps in this range
                if global_step % 10 == 0:
                    try:
                        component_norms = compute_parameter_norms(model, prefix="detail_")
                        for k, v in component_norms.items():
                            logger.info(f"  [DETAILED] {k}: {v:.6f}")
                    except Exception as e:
                        logger.warning(f"Could not compute detailed norms: {e}")

            # Logging
            if global_step % logging_steps == 0:
                num_micro_batches = logging_steps * training_config["gradient_accumulation_steps"]
                avg_loss = accumulated_loss / num_micro_batches
                accumulated_loss = 0.0

                # Compute average and max gradient norm over logging interval
                avg_grad_norm = accumulated_grad_norm / max(grad_norm_count, 1)
                max_grad_norm_interval = max_grad_norm_seen

                # Reset gradient norm tracking for next interval
                accumulated_grad_norm = 0.0
                grad_norm_count = 0
                max_grad_norm_seen = 0.0

                lr = (
                    lr_scheduler.get_last_lr()[0]
                    if lr_scheduler
                    else optimizer.param_groups[0]["lr"]
                )
                gpu_mem = get_gpu_memory_info()

                metrics = {
                    "loss": avg_loss,
                    "learning_rate": lr,
                    "gpu_memory_gb": gpu_mem.get("allocated_gb", 0),
                    "grad_norm_avg": avg_grad_norm,
                    "grad_norm_max": max_grad_norm_interval,
                }

                # Spike detection: flag if loss increased significantly
                if prev_loss is not None:
                    loss_ratio = avg_loss / (prev_loss + 1e-8)
                    metrics["loss_ratio"] = loss_ratio
                    if loss_ratio > 1.5:
                        logger.warning(
                            f"SPIKE DETECTED at step {global_step}: "
                            f"loss={avg_loss:.4f} (prev={prev_loss:.4f}, ratio={loss_ratio:.2f}), "
                            f"grad_norm_avg={avg_grad_norm:.4f}, grad_norm_max={max_grad_norm_interval:.4f}"
                        )
                        # Log detailed component norms when spike detected
                        if rank == 0:
                            try:
                                component_norms = compute_parameter_norms(model, prefix="spike_")
                                for k, v in component_norms.items():
                                    logger.warning(f"  {k}: {v:.4f}")
                                metrics.update(component_norms)
                            except Exception as e:
                                logger.warning(f"Could not compute component norms: {e}")
                prev_loss = avg_loss

                # Gradient explosion warning
                if avg_grad_norm > 10.0:
                    logger.warning(
                        f"HIGH GRADIENT NORM at step {global_step}: "
                        f"avg={avg_grad_norm:.4f}, max={max_grad_norm_interval:.4f}"
                    )

                log_metrics(metrics, global_step, prefix="train", wandb_run=wandb_run)

            # Evaluation
            if eval_dataloader is not None and global_step % eval_steps == 0:
                eval_metrics = evaluate_ring_attention(
                    model_engine=model_engine,
                    eval_dataloader=eval_dataloader,
                    device=device,
                    communicator=communicator,
                    max_eval_steps=50,
                )
                log_metrics(eval_metrics, global_step, prefix="eval", wandb_run=wandb_run)

            # Save checkpoint
            if global_step % save_steps == 0:
                checkpoint_dir = os.path.join(output_dir, f"checkpoint-{global_step}")
                model_engine.save_checkpoint(checkpoint_dir)
                logger.info(f"Checkpoint saved to {checkpoint_dir}")
                save_hf_checkpoint(model_engine, model, checkpoint_dir, rank, orthgsa_cfg)

                # Log checkpoint to wandb if enabled
                wandb_config = config.get("logging", {}).get("wandb", {})
                log_model = wandb_config.get("log_model")
                if wandb_run is not None and log_model and rank == 0:
                    try:
                        artifact = wandb_run.Artifact(
                            name=f"model-checkpoint-{global_step}",
                            type="model",
                            metadata={"step": global_step}
                        )
                        # Log the HF-compatible checkpoint
                        hf_checkpoint_dir = os.path.join(checkpoint_dir, "hf_model")
                        if os.path.exists(hf_checkpoint_dir):
                            artifact.add_dir(hf_checkpoint_dir)
                            wandb_run.log_artifact(artifact)
                            logger.info(f"Checkpoint logged to wandb as artifact")
                    except Exception as e:
                        logger.warning(f"Failed to log checkpoint to wandb: {e}")

    progress_bar.close()

    # Final save
    final_checkpoint_dir = os.path.join(output_dir, f"checkpoint-{global_step}")
    model_engine.save_checkpoint(final_checkpoint_dir)
    save_hf_checkpoint(model_engine, model, final_checkpoint_dir, rank, orthgsa_cfg)
    logger.info(f"Final checkpoint saved to {final_checkpoint_dir}")

    # Log final checkpoint to wandb
    wandb_config = config.get("logging", {}).get("wandb", {})
    log_model = wandb_config.get("log_model")
    if wandb_run is not None and log_model and rank == 0:
        try:
            artifact = wandb_run.Artifact(
                name=f"model-final",
                type="model",
                metadata={"step": global_step, "final": True}
            )
            hf_checkpoint_dir = os.path.join(final_checkpoint_dir, "hf_model")
            if os.path.exists(hf_checkpoint_dir):
                artifact.add_dir(hf_checkpoint_dir)
                wandb_run.log_artifact(artifact)
                logger.info(f"Final checkpoint logged to wandb as artifact")
        except Exception as e:
            logger.warning(f"Failed to log final checkpoint to wandb: {e}")

    if wandb_run is not None:
        wandb_run.finish()

    logger.info("Training complete!")


if __name__ == "__main__":
    main()

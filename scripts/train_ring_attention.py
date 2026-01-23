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


def setup_wandb(config: Dict[str, Any], rank: int) -> Optional[Any]:
    """Setup Weights & Biases logging."""
    wandb_config = config.get("logging", {}).get("wandb", {})
    enabled = wandb_config.get("enabled", False)

    logger.info(f"Wandb setup: enabled={enabled}, rank={rank}")

    if not enabled:
        logger.info("Wandb disabled in config (logging.wandb.enabled=false)")
        return None

    if rank != 0:
        logger.info(f"Wandb skipped on rank {rank} (only rank 0 logs to wandb)")
        return None

    try:
        import wandb

        run_name = wandb_config.get("run_name") or f"orthgsa-ring-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        project = wandb_config.get("project", "orthgsa")
        entity = wandb_config.get("entity")

        logger.info(f"Initializing wandb: project={project}, run_name={run_name}")

        wandb.init(
            project=project,
            entity=entity,
            name=run_name,
            tags=wandb_config.get("tags", []) + ["ring-attention"],
            config=config,
        )

        logger.info(f"Wandb initialized successfully: {wandb.run.url}")
        return wandb

    except ImportError:
        logger.warning("wandb not installed, skipping wandb logging")
        return None
    except Exception as e:
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

    if rank == 0:
        # Save config
        config_dict = model.config.to_dict()
        config_dict['orthgsa'] = orthgsa_config.to_dict()
        with open(os.path.join(hf_save_dir, 'config.json'), 'w') as f:
            json.dump(config_dict, f, indent=2)

        # Save OrthGSA weights
        with deepspeed.zero.GatheredParameters(
            list(model.mhc_modules.parameters()) + [model.expand_streams, model.collapse_streams],
            modifier_rank=0
        ):
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

    # Setup wandb
    wandb_run = setup_wandb(config, rank)

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
    logger.info("Starting training with Ring Attention...")

    max_steps = training_config["max_steps"]
    logging_steps = training_config["logging_steps"]
    eval_steps = training_config["eval_steps"]
    save_steps = training_config["save_steps"]

    output_dir = config["logging"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    global_step = 0
    accumulated_loss = 0.0
    train_iterator = iter(train_dataloader)

    progress_bar = tqdm(total=max_steps, desc="Training", disable=rank != 0)

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
        model_engine.step()

        accumulated_loss += loss.item()

        # Free memory
        del outputs, batch, input_ids, labels, position_ids

        # Check gradient accumulation boundary
        if model_engine.is_gradient_accumulation_boundary():
            global_step += 1
            progress_bar.update(1)

            # Periodic cache flush
            if global_step % 5 == 0:
                if dist.is_initialized():
                    dist.barrier()
                torch.cuda.empty_cache()

            # Logging
            if global_step % logging_steps == 0:
                num_micro_batches = logging_steps * training_config["gradient_accumulation_steps"]
                avg_loss = accumulated_loss / num_micro_batches
                accumulated_loss = 0.0

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
                }

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

    progress_bar.close()

    # Final save
    final_checkpoint_dir = os.path.join(output_dir, f"checkpoint-{global_step}")
    model_engine.save_checkpoint(final_checkpoint_dir)
    save_hf_checkpoint(model_engine, model, final_checkpoint_dir, rank, orthgsa_cfg)
    logger.info(f"Final checkpoint saved to {final_checkpoint_dir}")

    if wandb_run is not None:
        wandb_run.finish()

    logger.info("Training complete!")


if __name__ == "__main__":
    main()

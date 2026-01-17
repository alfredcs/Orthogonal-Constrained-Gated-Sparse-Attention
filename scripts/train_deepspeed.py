#!/usr/bin/env python3
"""
OrthGSA Training Script with DeepSpeed ZeRO

Train OrthGSA model on SlimPajama-627B with DeepSpeed ZeRO-2 for memory-efficient
distributed training across multiple GPUs.

Usage:
    deepspeed --num_gpus=4 scripts/train_deepspeed.py --config configs/config_qwen3_4b.yaml

For single GPU:
    python scripts/train_deepspeed.py --config configs/config_qwen3_4b.yaml
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

    if not wandb_config.get("enabled", False) or rank != 0:
        return None

    try:
        import wandb

        run_name = wandb_config.get("run_name") or f"orthgsa-ds-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

        wandb.init(
            project=wandb_config.get("project", "orthgsa"),
            entity=wandb_config.get("entity"),
            name=run_name,
            tags=wandb_config.get("tags", []) + ["deepspeed"],
            config=config,
        )

        logger.info(f"Wandb initialized: {wandb.run.url}")
        return wandb

    except ImportError:
        logger.warning("wandb not installed, skipping wandb logging")
        return None


def evaluate(
    model_engine: deepspeed.DeepSpeedEngine,
    eval_dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    max_eval_steps: int = 100,
) -> Dict[str, float]:
    """Evaluate the model."""
    model_engine.eval()

    total_loss = 0.0
    total_tokens = 0
    num_steps = 0

    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="Evaluating", total=max_eval_steps):
            if num_steps >= max_eval_steps:
                break

            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model_engine(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )

            total_loss += outputs.loss.item() * batch["input_ids"].numel()
            total_tokens += batch["input_ids"].numel()
            num_steps += 1

    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
    perplexity = math.exp(avg_loss) if avg_loss < 20 else float("inf")

    return {
        "eval_loss": avg_loss,
        "perplexity": perplexity,
    }


def main():
    parser = argparse.ArgumentParser(description="Train OrthGSA model with DeepSpeed")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Get distributed info
    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        rank = 0
        world_size = 1
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        deepspeed.init_distributed()
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()

    # Setup logging for rank 0 only
    if rank != 0:
        logging.getLogger().setLevel(logging.WARNING)

    logger.info(f"Starting training with config: {args.config}")
    logger.info(f"Rank: {rank}, World Size: {world_size}, Device: {device}")

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

    # Create OrthGSA configuration
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
    # Note: For ZeRO-3 with pretrained models, we load normally with low_cpu_mem_usage=True
    # and let deepspeed.initialize() handle the weight sharding.
    # deepspeed.zero.Init() is only for randomly initialized models, not from_pretrained().
    logger.info(f"Loading model: {model_config['base_model']}")

    model = OrthGSAForCausalLM(
        base_model_name=model_config["base_model"],
        orthgsa_config=orthgsa_cfg,
        torch_dtype=torch.bfloat16 if config["training"]["bf16"] else torch.float32,
        device_map=None,  # Let DeepSpeed handle device placement
        low_cpu_mem_usage=True,  # Reduce peak memory during loading for ZeRO-3
    )

    # Log parameter counts
    param_info = count_parameters(model)
    logger.info(f"Model parameters: {param_info['total_millions']:.2f}M total, {param_info['trainable_millions']:.2f}M trainable")

    # Enable gradient checkpointing if configured
    if config["training"].get("gradient_checkpointing", True):
        if hasattr(model.base_model, "gradient_checkpointing_enable"):
            model.base_model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")

    # Create data loaders
    training_config = config["training"]
    data_config = config["data"]
    dataset_name = data_config.get("dataset", "cerebras/SlimPajama-627B")
    local_path = data_config.get("local_path")

    train_dataloader = get_slimpajama_dataloader(
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
    )

    # Optional evaluation dataloader
    eval_dataloader = None
    if config.get("evaluation", {}).get("do_eval", True):
        eval_dataloader = get_slimpajama_dataloader(
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
        )

    # DeepSpeed config - load from external file if specified, otherwise use defaults
    distributed_config = config.get("distributed", {})
    ds_config_path = distributed_config.get("deepspeed_config")

    # Resolve relative path from project root
    if ds_config_path and not os.path.isabs(ds_config_path):
        project_root = Path(__file__).parent.parent
        ds_config_path = str(project_root / ds_config_path)

    if ds_config_path and os.path.exists(ds_config_path):
        # Load external DeepSpeed config
        logger.info(f"Loading DeepSpeed config from: {ds_config_path}")
        import json
        with open(ds_config_path, "r") as f:
            ds_config = json.load(f)

        # Override batch sizes with values from training config
        ds_config["train_batch_size"] = training_config["per_device_train_batch_size"] * world_size * training_config["gradient_accumulation_steps"]
        ds_config["train_micro_batch_size_per_gpu"] = training_config["per_device_train_batch_size"]
        ds_config["gradient_accumulation_steps"] = training_config["gradient_accumulation_steps"]

        # Add optimizer if not present
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

        # Add scheduler if not present
        if "scheduler" not in ds_config:
            ds_config["scheduler"] = {
                "type": "WarmupDecayLR",
                "params": {
                    "warmup_min_lr": 1e-6,
                    "warmup_max_lr": training_config["learning_rate"],
                    "warmup_num_steps": int(training_config["max_steps"] * training_config["warmup_ratio"]),
                    "total_num_steps": training_config["max_steps"],
                },
            }
    else:
        # Use default ZeRO-2 config
        logger.info("Using default DeepSpeed ZeRO-2 config")
        ds_config = {
            "train_batch_size": training_config["per_device_train_batch_size"] * world_size * training_config["gradient_accumulation_steps"],
            "train_micro_batch_size_per_gpu": training_config["per_device_train_batch_size"],
            "gradient_accumulation_steps": training_config["gradient_accumulation_steps"],
            "gradient_clipping": training_config["max_grad_norm"],
            "bf16": {
                "enabled": training_config.get("bf16", True),
            },
            "optimizer": {
                "type": "AdamW",
                "params": {
                    "lr": training_config["learning_rate"],
                    "betas": [training_config["adam_beta1"], training_config["adam_beta2"]],
                    "eps": training_config["adam_epsilon"],
                    "weight_decay": training_config["weight_decay"],
                },
            },
            "scheduler": {
                "type": "WarmupDecayLR",
                "params": {
                    "warmup_min_lr": 1e-6,
                    "warmup_max_lr": training_config["learning_rate"],
                    "warmup_num_steps": int(training_config["max_steps"] * training_config["warmup_ratio"]),
                    "total_num_steps": training_config["max_steps"],
                },
            },
            "zero_optimization": {
                "stage": 2,
                "offload_optimizer": {
                    "device": "none",
                },
                "allgather_partitions": True,
                "allgather_bucket_size": 2e8,
                "overlap_comm": True,
                "reduce_scatter": True,
                "reduce_bucket_size": 2e8,
                "contiguous_gradients": True,
            },
            "steps_per_print": training_config["logging_steps"],
            "wall_clock_breakdown": False,
        }

    # Initialize DeepSpeed
    model_engine, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=ds_config,
    )

    logger.info("DeepSpeed engine initialized")

    # Training loop
    logger.info("Starting training...")

    max_steps = training_config["max_steps"]
    logging_steps = training_config["logging_steps"]
    eval_steps = training_config["eval_steps"]
    save_steps = training_config["save_steps"]

    output_dir = config["logging"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    global_step = 0
    accumulated_loss = 0.0
    train_iterator = iter(train_dataloader)

    progress_bar = tqdm(
        total=max_steps,
        desc="Training",
        disable=rank != 0,
    )

    while global_step < max_steps:
        model_engine.train()

        try:
            batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_dataloader)
            batch = next(train_iterator)

        # Move batch to device
        batch = {k: v.to(device) for k, v in batch.items()}

        # Forward pass
        outputs = model_engine(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        loss = outputs.loss

        # Backward pass (DeepSpeed handles gradient accumulation)
        model_engine.backward(loss)
        model_engine.step()

        accumulated_loss += loss.item()

        # Check if we've completed a full gradient accumulation cycle
        if model_engine.is_gradient_accumulation_boundary():
            global_step += 1
            progress_bar.update(1)

            # Logging
            if global_step % logging_steps == 0:
                # Fix: divide by total micro-batches, not just logging_steps
                num_micro_batches = logging_steps * training_config["gradient_accumulation_steps"]
                avg_loss = accumulated_loss / num_micro_batches
                accumulated_loss = 0.0

                lr = lr_scheduler.get_last_lr()[0] if lr_scheduler else optimizer.param_groups[0]["lr"]
                gpu_mem = get_gpu_memory_info()

                metrics = {
                    "loss": avg_loss,
                    "learning_rate": lr,
                    "gpu_memory_gb": gpu_mem.get("allocated_gb", 0),
                }

                log_metrics(metrics, global_step, prefix="train", wandb_run=wandb_run)

            # Evaluation
            if eval_dataloader is not None and global_step % eval_steps == 0:
                eval_metrics = evaluate(
                    model_engine=model_engine,
                    eval_dataloader=eval_dataloader,
                    device=device,
                    max_eval_steps=100,
                )

                log_metrics(eval_metrics, global_step, prefix="eval", wandb_run=wandb_run)

            # Save checkpoint
            if global_step % save_steps == 0:
                checkpoint_dir = os.path.join(output_dir, f"checkpoint-{global_step}")
                model_engine.save_checkpoint(checkpoint_dir)
                logger.info(f"Checkpoint saved to {checkpoint_dir}")

    progress_bar.close()

    # Final save
    final_checkpoint_dir = os.path.join(output_dir, f"checkpoint-{global_step}")
    model_engine.save_checkpoint(final_checkpoint_dir)
    logger.info(f"Final checkpoint saved to {final_checkpoint_dir}")

    # Cleanup
    if wandb_run is not None:
        wandb_run.finish()

    logger.info("Training complete!")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
OrthGSA Training Script

Train OrthGSA model on SlimPajama-627B with FSDP for distributed training.
Uses Cayley Transform for orthogonal constraints instead of Sinkhorn-Knopp.

Usage:
    torchrun --nproc_per_node=4 scripts/train.py --config configs/config_qwen3_4b.yaml

For single GPU:
    python scripts/train.py --config configs/config_qwen3_4b.yaml
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
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.cuda.amp import GradScaler, autocast
from transformers import AutoTokenizer
import yaml
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from orthgsa.models import OrthGSAForCausalLM, OrthGSAConfig
from orthgsa.data import get_slimpajama_dataloader
from orthgsa.utils import (
    setup_distributed,
    cleanup_distributed,
    get_optimizer,
    get_lr_scheduler,
    save_checkpoint,
    load_checkpoint,
    log_metrics,
    count_parameters,
    estimate_memory_usage,
    get_gpu_memory_info,
    get_fsdp_config,
    wrap_model_fsdp,
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

        run_name = wandb_config.get("run_name") or f"orthgsa-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

        wandb.init(
            project=wandb_config.get("project", "orthgsa"),
            entity=wandb_config.get("entity"),
            name=run_name,
            tags=wandb_config.get("tags", []),
            config=config,
        )

        logger.info(f"Wandb initialized: {wandb.run.url}")
        return wandb

    except ImportError:
        logger.warning("wandb not installed, skipping wandb logging")
        return None


def train_step(
    model: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: Optional[GradScaler],
    gradient_accumulation_steps: int,
    max_grad_norm: float,
    device: torch.device,
    use_amp: bool = True,
) -> Dict[str, float]:
    """Execute a single training step."""
    model.train()

    # Move batch to device
    batch = {k: v.to(device) for k, v in batch.items()}

    # Forward pass with mixed precision
    with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        loss = outputs.loss / gradient_accumulation_steps

    # Backward pass
    if scaler is not None:
        scaler.scale(loss).backward()
    else:
        loss.backward()

    return {"loss": loss.item() * gradient_accumulation_steps}


def evaluate(
    model: torch.nn.Module,
    eval_dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    max_eval_steps: int = 100,
    use_amp: bool = True,
) -> Dict[str, float]:
    """Evaluate the model."""
    model.eval()

    total_loss = 0.0
    total_tokens = 0
    num_steps = 0

    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="Evaluating", total=max_eval_steps):
            if num_steps >= max_eval_steps:
                break

            batch = {k: v.to(device) for k, v in batch.items()}

            with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                outputs = model(
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
    parser = argparse.ArgumentParser(description="Train OrthGSA model")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Setup distributed training
    distributed = int(os.environ.get("WORLD_SIZE", 1)) > 1

    if distributed:
        rank, local_rank, world_size = setup_distributed()
    else:
        rank, local_rank, world_size = 0, 0, 1
        torch.cuda.set_device(0)

    device = torch.device(f"cuda:{local_rank}")

    # Setup logging for rank 0 only
    if rank != 0:
        logging.getLogger().setLevel(logging.WARNING)

    logger.info(f"Starting training with config: {args.config}")
    logger.info(f"Distributed: {distributed}, Rank: {rank}, World Size: {world_size}")

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

    # Create OrthGSA configuration (uses Cayley Transform for orthogonal constraints)
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

    model = OrthGSAForCausalLM(
        base_model_name=model_config["base_model"],
        orthgsa_config=orthgsa_cfg,
        torch_dtype=torch.bfloat16 if config["training"]["bf16"] else torch.float32,
    )

    # Log parameter counts
    param_info = count_parameters(model)
    logger.info(f"Model parameters: {param_info['total_millions']:.2f}M total, {param_info['trainable_millions']:.2f}M trainable")

    # Enable gradient checkpointing if configured
    if config["training"].get("gradient_checkpointing", True):
        if hasattr(model.base_model, "gradient_checkpointing_enable"):
            model.base_model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")

    # Wrap with FSDP for distributed training
    if distributed:
        fsdp_config = get_fsdp_config(
            model,
            sharding_strategy=config["distributed"]["fsdp_config"]["sharding_strategy"],
            use_bf16=config["training"]["bf16"],
            backward_prefetch=config["distributed"]["fsdp_config"]["backward_prefetch"],
        )
        model = wrap_model_fsdp(model, **fsdp_config)
        logger.info("Model wrapped with FSDP")
    else:
        model = model.to(device)

    # Create optimizer
    training_config = config["training"]
    optimizer = get_optimizer(
        model,
        learning_rate=training_config["learning_rate"],
        weight_decay=training_config["weight_decay"],
        beta1=training_config["adam_beta1"],
        beta2=training_config["adam_beta2"],
        eps=training_config["adam_epsilon"],
    )

    # Create learning rate scheduler
    scheduler = get_lr_scheduler(
        optimizer,
        scheduler_type=training_config["lr_scheduler_type"],
        num_training_steps=training_config["max_steps"],
        warmup_ratio=training_config["warmup_ratio"],
    )

    # Create data loaders
    data_config = config["data"]
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
    )

    # Optional evaluation dataloader
    eval_dataloader = None
    if config.get("evaluation", {}).get("do_eval", True):
        # Note: SlimPajama doesn't have a separate validation split
        # We use a portion of training data for evaluation
        eval_dataloader = get_slimpajama_dataloader(
            tokenizer=tokenizer,
            batch_size=training_config["per_device_eval_batch_size"],
            max_length=data_config["max_seq_length"],
            split="train",  # Use train split for eval (different seed/shard)
            num_workers=data_config["num_workers"],
            seed=config.get("seed", 42) + 1000,  # Different seed
            rank=rank,
            world_size=world_size,
            packed=True,
        )

    # Gradient scaler for mixed precision
    scaler = GradScaler() if training_config.get("bf16", True) else None

    # Resume from checkpoint if specified
    start_step = 0
    if args.resume:
        start_step = load_checkpoint(
            model, optimizer, scheduler, args.resume,
            is_fsdp=distributed,
        )
        logger.info(f"Resumed from step {start_step}")

    # Training loop
    logger.info("Starting training...")
    model.train()

    gradient_accumulation_steps = training_config["gradient_accumulation_steps"]
    max_steps = training_config["max_steps"]
    logging_steps = training_config["logging_steps"]
    eval_steps = training_config["eval_steps"]
    save_steps = training_config["save_steps"]
    max_grad_norm = training_config["max_grad_norm"]

    output_dir = config["logging"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    global_step = start_step
    accumulated_loss = 0.0
    train_iterator = iter(train_dataloader)

    progress_bar = tqdm(
        total=max_steps,
        initial=start_step,
        desc="Training",
        disable=rank != 0,
    )

    while global_step < max_steps:
        for micro_step in range(gradient_accumulation_steps):
            try:
                batch = next(train_iterator)
            except StopIteration:
                train_iterator = iter(train_dataloader)
                batch = next(train_iterator)

            # Training step
            step_metrics = train_step(
                model=model,
                batch=batch,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                gradient_accumulation_steps=gradient_accumulation_steps,
                max_grad_norm=max_grad_norm,
                device=device,
                use_amp=training_config.get("bf16", True),
            )

            accumulated_loss += step_metrics["loss"]

        # Gradient clipping and optimizer step
        if scaler is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

        scheduler.step()
        optimizer.zero_grad()

        global_step += 1
        progress_bar.update(1)

        # Logging
        if global_step % logging_steps == 0:
            avg_loss = accumulated_loss / logging_steps
            accumulated_loss = 0.0

            lr = scheduler.get_last_lr()[0]
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
                model=model,
                eval_dataloader=eval_dataloader,
                device=device,
                max_eval_steps=100,
                use_amp=training_config.get("bf16", True),
            )

            log_metrics(eval_metrics, global_step, prefix="eval", wandb_run=wandb_run)
            model.train()

        # Save checkpoint
        if global_step % save_steps == 0 and rank == 0:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                step=global_step,
                loss=avg_loss if "avg_loss" in dir() else 0,
                output_dir=output_dir,
                is_fsdp=distributed,
                rank=rank,
            )

    progress_bar.close()

    # Final save
    if rank == 0:
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            step=global_step,
            loss=avg_loss if "avg_loss" in dir() else 0,
            output_dir=output_dir,
            is_fsdp=distributed,
            rank=rank,
        )

    # Cleanup
    if wandb_run is not None:
        wandb_run.finish()

    if distributed:
        cleanup_distributed()

    logger.info("Training complete!")


if __name__ == "__main__":
    main()

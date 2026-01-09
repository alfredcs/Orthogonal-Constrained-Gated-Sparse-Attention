"""
Training Utilities for OrthGSA

Includes:
- Distributed training setup (FSDP/DDP)
- Learning rate schedulers
- Optimizer configuration
- Checkpointing
- Memory estimation
"""

import os
import math
import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    MixedPrecision,
    BackwardPrefetch,
    StateDictType,
    FullStateDictConfig,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    size_based_auto_wrap_policy,
)
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    SequentialLR,
    OneCycleLR,
)
from typing import Optional, Dict, Any, List, Tuple
import logging
import json
from datetime import datetime

logger = logging.getLogger(__name__)


def setup_distributed(
    backend: str = "nccl",
    init_method: str = "env://",
) -> Tuple[int, int, int]:
    """
    Setup distributed training environment.

    Returns:
        rank: Process rank
        local_rank: Local process rank (GPU index)
        world_size: Total number of processes
    """
    if not dist.is_initialized():
        dist.init_process_group(backend=backend, init_method=init_method)

    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = dist.get_world_size()

    # Set device
    torch.cuda.set_device(local_rank)

    logger.info(f"Distributed setup: rank={rank}, local_rank={local_rank}, world_size={world_size}")

    return rank, local_rank, world_size


def cleanup_distributed():
    """Cleanup distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def get_fsdp_config(
    model: torch.nn.Module,
    sharding_strategy: str = "FULL_SHARD",
    use_bf16: bool = True,
    cpu_offload: bool = False,
    backward_prefetch: str = "BACKWARD_PRE",
    auto_wrap_policy: str = "transformer",
    min_num_params: int = 1e6,
) -> Dict[str, Any]:
    """
    Get FSDP configuration.

    Args:
        model: Model to wrap
        sharding_strategy: Sharding strategy
        use_bf16: Use BF16 mixed precision
        cpu_offload: Offload to CPU
        backward_prefetch: Backward prefetch strategy
        auto_wrap_policy: Auto wrap policy type
        min_num_params: Minimum parameters for size-based wrapping

    Returns:
        FSDP configuration dictionary
    """
    # Sharding strategy
    strategy_map = {
        "FULL_SHARD": ShardingStrategy.FULL_SHARD,
        "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP,
        "NO_SHARD": ShardingStrategy.NO_SHARD,
        "HYBRID_SHARD": ShardingStrategy.HYBRID_SHARD,
    }
    sharding = strategy_map.get(sharding_strategy, ShardingStrategy.FULL_SHARD)

    # Mixed precision
    if use_bf16:
        mixed_precision = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )
    else:
        mixed_precision = None

    # Backward prefetch
    prefetch_map = {
        "BACKWARD_PRE": BackwardPrefetch.BACKWARD_PRE,
        "BACKWARD_POST": BackwardPrefetch.BACKWARD_POST,
        None: None,
    }
    prefetch = prefetch_map.get(backward_prefetch)

    # Auto wrap policy
    if auto_wrap_policy == "transformer":
        # Try to find transformer layer classes
        from ..models.orthgsa_layer import OrthGSALayer

        wrap_policy = transformer_auto_wrap_policy(
            transformer_layer_cls={OrthGSALayer},
        )
    else:
        wrap_policy = size_based_auto_wrap_policy(min_num_params=int(min_num_params))

    return {
        "sharding_strategy": sharding,
        "mixed_precision": mixed_precision,
        "backward_prefetch": prefetch,
        "auto_wrap_policy": wrap_policy,
        "cpu_offload": None,  # Set to CPUOffload() if needed
        "device_id": torch.cuda.current_device(),
        "sync_module_states": True,
        "forward_prefetch": True,
        "limit_all_gathers": True,
        "use_orig_params": True,
    }


def wrap_model_fsdp(
    model: torch.nn.Module,
    **fsdp_kwargs,
) -> FSDP:
    """
    Wrap model with FSDP.

    Args:
        model: Model to wrap
        **fsdp_kwargs: FSDP configuration

    Returns:
        FSDP-wrapped model
    """
    return FSDP(model, **fsdp_kwargs)


def get_optimizer(
    model: torch.nn.Module,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.1,
    beta1: float = 0.9,
    beta2: float = 0.95,
    eps: float = 1e-8,
    fused: bool = True,
    foreach: bool = True,
) -> torch.optim.Optimizer:
    """
    Get optimizer for training.

    Uses AdamW with decoupled weight decay.
    Applies weight decay only to non-bias, non-LayerNorm parameters.
    """
    # Separate parameters into decay and no-decay groups
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # No decay for biases and LayerNorm/RMSNorm weights
        if param.dim() == 1 or "bias" in name or "norm" in name.lower():
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    logger.info(f"Optimizer groups: {len(decay_params)} decay params, {len(no_decay_params)} no-decay params")

    # Use fused AdamW if available
    if fused and hasattr(torch.optim, "AdamW"):
        try:
            optimizer = AdamW(
                optimizer_groups,
                lr=learning_rate,
                betas=(beta1, beta2),
                eps=eps,
                fused=True,
            )
            logger.info("Using fused AdamW optimizer")
            return optimizer
        except Exception:
            pass

    # Fallback to regular AdamW
    optimizer = AdamW(
        optimizer_groups,
        lr=learning_rate,
        betas=(beta1, beta2),
        eps=eps,
        foreach=foreach,
    )
    logger.info("Using standard AdamW optimizer")
    return optimizer


def get_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str = "cosine",
    num_training_steps: int = 100000,
    warmup_steps: Optional[int] = None,
    warmup_ratio: float = 0.03,
    min_lr_ratio: float = 0.1,
) -> torch.optim.lr_scheduler.LRScheduler:
    """
    Get learning rate scheduler.

    Args:
        optimizer: Optimizer
        scheduler_type: Type of scheduler ("cosine", "linear", "constant")
        num_training_steps: Total training steps
        warmup_steps: Number of warmup steps (overrides warmup_ratio)
        warmup_ratio: Warmup ratio (fraction of total steps)
        min_lr_ratio: Minimum LR as fraction of max LR

    Returns:
        Learning rate scheduler
    """
    if warmup_steps is None:
        warmup_steps = int(num_training_steps * warmup_ratio)

    logger.info(f"LR scheduler: {scheduler_type}, warmup={warmup_steps}, total={num_training_steps}")

    if scheduler_type == "cosine":
        # Warmup + Cosine decay
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=1e-8,
            end_factor=1.0,
            total_iters=warmup_steps,
        )

        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=num_training_steps - warmup_steps,
            eta_min=optimizer.param_groups[0]["lr"] * min_lr_ratio,
        )

        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps],
        )

    elif scheduler_type == "linear":
        scheduler = LinearLR(
            optimizer,
            start_factor=1e-8,
            end_factor=min_lr_ratio,
            total_iters=num_training_steps,
        )

    elif scheduler_type == "one_cycle":
        scheduler = OneCycleLR(
            optimizer,
            max_lr=optimizer.param_groups[0]["lr"],
            total_steps=num_training_steps,
            pct_start=warmup_ratio,
            anneal_strategy="cos",
            div_factor=25.0,
            final_div_factor=10000.0,
        )

    else:  # constant with warmup
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=1e-8,
            end_factor=1.0,
            total_iters=warmup_steps,
        )

        constant_scheduler = LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=1.0,
            total_iters=num_training_steps - warmup_steps,
        )

        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, constant_scheduler],
            milestones=[warmup_steps],
        )

    return scheduler


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    step: int,
    loss: float,
    output_dir: str,
    is_fsdp: bool = False,
    rank: int = 0,
) -> str:
    """
    Save training checkpoint.

    Args:
        model: Model to save
        optimizer: Optimizer state
        scheduler: Scheduler state
        step: Current training step
        loss: Current loss
        output_dir: Output directory
        is_fsdp: Whether model is wrapped with FSDP
        rank: Process rank

    Returns:
        Checkpoint path
    """
    checkpoint_dir = os.path.join(output_dir, f"checkpoint-{step}")

    if rank == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)

    if is_fsdp:
        # Use FSDP state dict
        with FSDP.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
        ):
            model_state = model.state_dict()
            if rank == 0:
                torch.save(model_state, os.path.join(checkpoint_dir, "model.pt"))
    else:
        if rank == 0:
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "model.pt"))

    if rank == 0:
        # Save optimizer and scheduler
        torch.save({
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "step": step,
            "loss": loss,
        }, os.path.join(checkpoint_dir, "training_state.pt"))

        # Save training info
        info = {
            "step": step,
            "loss": loss,
            "timestamp": datetime.now().isoformat(),
        }
        with open(os.path.join(checkpoint_dir, "info.json"), "w") as f:
            json.dump(info, f, indent=2)

        logger.info(f"Checkpoint saved to {checkpoint_dir}")

    return checkpoint_dir


def load_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    checkpoint_dir: str,
    is_fsdp: bool = False,
) -> int:
    """
    Load training checkpoint.

    Returns:
        Starting step
    """
    # Load model state
    model_path = os.path.join(checkpoint_dir, "model.pt")
    if os.path.exists(model_path):
        model_state = torch.load(model_path, map_location="cpu")
        if is_fsdp:
            with FSDP.state_dict_type(
                model,
                StateDictType.FULL_STATE_DICT,
            ):
                model.load_state_dict(model_state)
        else:
            model.load_state_dict(model_state)
        logger.info(f"Loaded model from {model_path}")

    # Load training state
    training_state_path = os.path.join(checkpoint_dir, "training_state.pt")
    if os.path.exists(training_state_path):
        training_state = torch.load(training_state_path, map_location="cpu")
        optimizer.load_state_dict(training_state["optimizer"])
        scheduler.load_state_dict(training_state["scheduler"])
        step = training_state["step"]
        logger.info(f"Loaded training state from step {step}")
        return step

    return 0


def log_metrics(
    metrics: Dict[str, float],
    step: int,
    prefix: str = "train",
    wandb_run: Optional[Any] = None,
):
    """Log metrics to console and wandb."""
    # Format for console
    metrics_str = ", ".join(f"{k}={v:.4f}" for k, v in metrics.items())
    logger.info(f"Step {step} [{prefix}]: {metrics_str}")

    # Log to wandb
    if wandb_run is not None:
        wandb_metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}
        wandb_metrics["step"] = step
        wandb_run.log(wandb_metrics)


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """Count model parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
        "total_millions": total / 1e6,
        "trainable_millions": trainable / 1e6,
    }


def estimate_memory_usage(
    model: torch.nn.Module,
    batch_size: int,
    seq_length: int,
    hidden_size: int,
    num_layers: int,
    n_streams: int = 4,
    dtype: torch.dtype = torch.bfloat16,
    gradient_checkpointing: bool = True,
) -> Dict[str, float]:
    """
    Estimate GPU memory usage.

    Returns memory estimates in GB.
    """
    bytes_per_param = 2 if dtype in [torch.float16, torch.bfloat16] else 4

    # Model parameters
    param_count = sum(p.numel() for p in model.parameters())
    param_memory = param_count * bytes_per_param

    # Gradients (same size as parameters for full precision gradients)
    grad_memory = param_count * 4  # FP32 gradients

    # Optimizer states (AdamW: 2 states per parameter)
    optimizer_memory = param_count * 8  # Two FP32 states

    # Activations (rough estimate)
    # With gradient checkpointing, only store activations at checkpoint boundaries
    if gradient_checkpointing:
        activation_memory = batch_size * seq_length * hidden_size * n_streams * bytes_per_param * 2
    else:
        # Store all activations
        activation_memory = batch_size * seq_length * hidden_size * n_streams * num_layers * bytes_per_param * 4

    # KV cache (if using)
    kv_memory = batch_size * seq_length * hidden_size * num_layers * 2 * bytes_per_param

    total_memory = param_memory + grad_memory + optimizer_memory + activation_memory

    return {
        "parameters_gb": param_memory / 1e9,
        "gradients_gb": grad_memory / 1e9,
        "optimizer_gb": optimizer_memory / 1e9,
        "activations_gb": activation_memory / 1e9,
        "kv_cache_gb": kv_memory / 1e9,
        "total_estimated_gb": total_memory / 1e9,
    }


def get_gpu_memory_info() -> Dict[str, float]:
    """Get current GPU memory usage."""
    if not torch.cuda.is_available():
        return {}

    return {
        "allocated_gb": torch.cuda.memory_allocated() / 1e9,
        "reserved_gb": torch.cuda.memory_reserved() / 1e9,
        "max_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
    }

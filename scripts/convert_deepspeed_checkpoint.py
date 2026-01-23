#!/usr/bin/env python3
"""
Convert DeepSpeed ZeRO checkpoint to HuggingFace-compatible format.

This script converts DeepSpeed ZeRO-3 checkpoints (which store model weights
sharded across processes) to a standard HuggingFace format that can be loaded
with OrthGSAForCausalLM.from_pretrained().

Usage:
    python scripts/convert_deepspeed_checkpoint.py \
        --checkpoint outputs/orthgsa-qwen3-8b-8k/checkpoint-1000 \
        --config configs/config_qwen3_8b_8k.yaml \
        --output outputs/orthgsa-qwen3-8b-8k/checkpoint-1000/hf_model
"""

import os
import sys
import argparse
import json
import logging
from pathlib import Path

import torch
import yaml

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from orthgsa.models import OrthGSAConfig

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def convert_deepspeed_checkpoint(
    checkpoint_dir: str,
    config_path: str,
    output_dir: str,
    tag: str = None,
):
    """Convert DeepSpeed checkpoint to HuggingFace format.

    Args:
        checkpoint_dir: Path to DeepSpeed checkpoint directory
        config_path: Path to training config YAML file
        output_dir: Where to save the converted checkpoint
        tag: DeepSpeed checkpoint tag (default: auto-detect from 'latest' file)
    """
    # Import zero_to_fp32 from the checkpoint directory
    zero_to_fp32_path = os.path.join(checkpoint_dir, "zero_to_fp32.py")
    if not os.path.exists(zero_to_fp32_path):
        raise FileNotFoundError(
            f"zero_to_fp32.py not found in {checkpoint_dir}. "
            "Is this a DeepSpeed checkpoint?"
        )

    # Import the conversion module
    import importlib.util
    spec = importlib.util.spec_from_file_location("zero_to_fp32", zero_to_fp32_path)
    zero_to_fp32 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(zero_to_fp32)

    # Load training config to get model parameters
    logger.info(f"Loading config from {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    model_config = config["model"]

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Get checkpoint tag
    if tag is None:
        latest_file = os.path.join(checkpoint_dir, "latest")
        if os.path.exists(latest_file):
            with open(latest_file, 'r') as f:
                tag = f.read().strip()
        else:
            # Try to find global_step directory
            for item in os.listdir(checkpoint_dir):
                if item.startswith("global_step"):
                    tag = item
                    break

    if tag is None:
        raise ValueError(f"Could not determine checkpoint tag in {checkpoint_dir}")

    logger.info(f"Using checkpoint tag: {tag}")

    # Convert ZeRO checkpoint to FP32/BF16
    logger.info("Converting DeepSpeed checkpoint to consolidated weights...")

    # DeepSpeed may create a directory with sharded files or a single file
    # Use a temp path for the conversion output
    temp_output_path = os.path.join(output_dir, "_temp_converted")

    # Use zero_to_fp32's conversion
    zero_to_fp32.convert_zero_checkpoint_to_fp32_state_dict(
        checkpoint_dir,
        temp_output_path,
        tag=tag,
    )

    logger.info(f"Converted weights saved to {temp_output_path}")

    # Load the converted state dict to extract OrthGSA weights
    logger.info("Extracting OrthGSA-specific weights...")

    # Handle both single file and sharded checkpoint formats
    if os.path.isdir(temp_output_path):
        # Sharded checkpoint - load using index file
        index_file = os.path.join(temp_output_path, "pytorch_model.bin.index.json")
        if os.path.exists(index_file):
            with open(index_file, 'r') as f:
                index = json.load(f)

            # Load all shards
            state_dict = {}
            shard_files = set(index["weight_map"].values())
            for shard_file in sorted(shard_files):
                shard_path = os.path.join(temp_output_path, shard_file)
                logger.info(f"Loading shard: {shard_file}")
                shard_dict = torch.load(shard_path, map_location='cpu')
                state_dict.update(shard_dict)
        else:
            raise FileNotFoundError(f"No index file found in {temp_output_path}")
    else:
        # Single file checkpoint
        state_dict = torch.load(temp_output_path, map_location='cpu')

    # Separate OrthGSA weights from base model weights
    # The checkpoint may have duplicate keys with different prefixes (model., transformer., layers.)
    # all pointing to the same tensors. We keep only the canonical 'model.' prefix for HuggingFace.
    orthgsa_state = {
        'mhc_modules': {},
        'expand_streams': None,
        'collapse_streams': None,
    }
    base_model_state = {}

    for key, value in state_dict.items():
        if key.startswith('mhc_modules.'):
            # Remove 'mhc_modules.' prefix for mhc_modules state dict
            new_key = key[len('mhc_modules.'):]
            orthgsa_state['mhc_modules'][new_key] = value
        elif key == 'expand_streams':
            orthgsa_state['expand_streams'] = value
        elif key == 'collapse_streams':
            orthgsa_state['collapse_streams'] = value
        elif key.startswith('base_model.'):
            # Remove 'base_model.' prefix for HF format
            new_key = key[len('base_model.'):]
            # Only keep 'model.' prefixed keys (standard HF format), skip duplicates
            if new_key.startswith('model.') or new_key in ('lm_head.weight',):
                base_model_state[new_key] = value.clone()  # Clone to avoid shared memory issues
        elif key.startswith('model.') or key in ('lm_head.weight',):
            # Direct model keys without base_model prefix
            base_model_state[key] = value.clone()
        # Skip other prefixes (transformer., layers., embed_tokens., etc.) as they are duplicates

    # Free memory
    del state_dict

    # Save OrthGSA weights
    torch.save(orthgsa_state, os.path.join(output_dir, 'orthgsa_weights.pt'))
    logger.info(f"OrthGSA weights saved to {output_dir}/orthgsa_weights.pt")
    del orthgsa_state

    # Save base model weights (sharded for large models)
    logger.info("Saving base model weights...")
    from transformers import AutoConfig
    base_model_path = model_config["base_model"]
    base_config = AutoConfig.from_pretrained(base_model_path, trust_remote_code=True)

    # Use safetensors for efficient storage
    try:
        from safetensors.torch import save_file
        save_file(base_model_state, os.path.join(output_dir, 'model.safetensors'))
        logger.info(f"Base model weights saved to {output_dir}/model.safetensors")
    except ImportError:
        torch.save(base_model_state, os.path.join(output_dir, 'pytorch_model.bin'))
        logger.info(f"Base model weights saved to {output_dir}/pytorch_model.bin")

    del base_model_state

    # Clean up temp files
    import shutil
    if os.path.isdir(temp_output_path):
        shutil.rmtree(temp_output_path)
    elif os.path.exists(temp_output_path):
        os.remove(temp_output_path)

    # Create config.json
    logger.info("Creating config.json...")
    config_dict = base_config.to_dict()

    # Add OrthGSA config
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
    config_dict['orthgsa'] = orthgsa_cfg.to_dict()

    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config_dict, f, indent=2)

    logger.info(f"Config saved to {output_dir}/config.json")

    # Copy tokenizer files if they exist in base model
    logger.info("Conversion complete!")
    logger.info(f"\nTo evaluate, run:")
    logger.info(f"  python scripts/evaluate.py --checkpoint {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert DeepSpeed ZeRO checkpoint to HuggingFace format"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to DeepSpeed checkpoint directory"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to training config YAML file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory (default: <checkpoint>/hf_model)"
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="DeepSpeed checkpoint tag (default: auto-detect)"
    )

    args = parser.parse_args()

    output_dir = args.output
    if output_dir is None:
        output_dir = os.path.join(args.checkpoint, "hf_model")

    convert_deepspeed_checkpoint(
        checkpoint_dir=args.checkpoint,
        config_path=args.config,
        output_dir=output_dir,
        tag=args.tag,
    )


if __name__ == "__main__":
    main()

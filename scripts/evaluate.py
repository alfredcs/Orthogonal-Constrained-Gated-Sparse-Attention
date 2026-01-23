#!/usr/bin/env python3
"""
OrthGSA Evaluation Script

Evaluate a trained OrthGSA model on various benchmarks.
Uses Cayley Transform for orthogonal constraints instead of Sinkhorn-Knopp.

Usage:
    python scripts/evaluate.py --checkpoint outputs/orthgsa-qwen3-4b/checkpoint-10000
"""

import os
import sys
import argparse
import logging
import math
from pathlib import Path
from typing import Optional, Dict, Any, List
import json

import torch
from torch.amp import autocast
from transformers import AutoTokenizer
from tqdm import tqdm
import yaml

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from orthgsa.models import OrthGSAForCausalLM
from orthgsa.data import get_slimpajama_dataloader

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def compute_perplexity(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    max_steps: int = 500,
    use_amp: bool = True,
) -> Dict[str, float]:
    """
    Compute perplexity on a dataset.

    Args:
        model: Model to evaluate
        dataloader: Data loader
        device: Device to use
        max_steps: Maximum evaluation steps
        use_amp: Use automatic mixed precision

    Returns:
        Dictionary with perplexity metrics
    """
    model.eval()

    total_loss = 0.0
    total_tokens = 0
    num_steps = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing perplexity", total=max_steps):
            if num_steps >= max_steps:
                break

            batch = {k: v.to(device) for k, v in batch.items()}

            with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )

            # Count non-padding tokens
            non_pad_mask = batch["labels"] != -100
            num_tokens = non_pad_mask.sum().item()

            total_loss += outputs.loss.item() * num_tokens
            total_tokens += num_tokens
            num_steps += 1

    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
    perplexity = math.exp(avg_loss) if avg_loss < 100 else float("inf")

    return {
        "loss": avg_loss,
        "perplexity": perplexity,
        "total_tokens": total_tokens,
        "num_steps": num_steps,
    }


def evaluate_generation(
    model: torch.nn.Module,
    tokenizer,
    prompts: List[str],
    device: torch.device,
    max_new_tokens: int = 100,
) -> List[Dict[str, str]]:
    """
    Evaluate text generation quality.

    Args:
        model: Model to evaluate
        tokenizer: Tokenizer
        prompts: List of prompts
        device: Device to use
        max_new_tokens: Maximum new tokens to generate

    Returns:
        List of prompt-completion pairs
    """
    model.eval()
    results = []

    for prompt in tqdm(prompts, desc="Generating"):
        inputs = tokenizer(prompt, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
            )

        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)

        results.append({
            "prompt": prompt,
            "generated": generated,
            "completion": generated[len(prompt):].strip(),
        })

    return results


def evaluate_throughput(
    model: torch.nn.Module,
    tokenizer,
    device: torch.device,
    batch_sizes: List[int] = [1, 2, 4, 8],
    seq_lengths: List[int] = [512, 1024, 2048],
    num_iterations: int = 10,
) -> Dict[str, Any]:
    """
    Measure inference throughput.

    Returns tokens per second for different batch sizes and sequence lengths.
    """
    model.eval()
    results = {}

    for batch_size in batch_sizes:
        for seq_len in seq_lengths:
            key = f"bs{batch_size}_seq{seq_len}"

            # Create dummy input
            input_ids = torch.randint(
                0, tokenizer.vocab_size,
                (batch_size, seq_len),
                device=device,
            )
            attention_mask = torch.ones_like(input_ids)

            # Warmup
            for _ in range(3):
                with torch.no_grad(), autocast(device_type="cuda", dtype=torch.bfloat16):
                    _ = model(input_ids=input_ids, attention_mask=attention_mask)

            # Synchronize
            torch.cuda.synchronize()

            # Measure
            import time
            start_time = time.time()

            for _ in range(num_iterations):
                with torch.no_grad(), autocast(device_type="cuda", dtype=torch.bfloat16):
                    _ = model(input_ids=input_ids, attention_mask=attention_mask)

            torch.cuda.synchronize()
            elapsed = time.time() - start_time

            tokens_per_second = (batch_size * seq_len * num_iterations) / elapsed

            results[key] = {
                "batch_size": batch_size,
                "seq_length": seq_len,
                "tokens_per_second": tokens_per_second,
                "time_per_batch_ms": (elapsed / num_iterations) * 1000,
            }

            logger.info(f"{key}: {tokens_per_second:.0f} tokens/sec")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate OrthGSA model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--config", type=str, default=None, help="Path to config file")
    parser.add_argument("--output", type=str, default="eval_results.json", help="Output file")
    parser.add_argument("--max_eval_steps", type=int, default=500, help="Max evaluation steps")
    parser.add_argument("--eval_throughput", action="store_true", help="Evaluate throughput")
    parser.add_argument("--eval_generation", action="store_true", help="Evaluate generation")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load config if provided
    if args.config:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
        base_model = config["model"]["base_model"]
        max_seq_length = config["data"]["max_seq_length"]
    else:
        # Try to load from checkpoint
        config_path = os.path.join(args.checkpoint, "config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)
            base_model = config.get("_name_or_path", "Qwen/Qwen3-4B-Instruct-2507")
        else:
            base_model = "Qwen/Qwen3-4B-Instruct-2507"
        max_seq_length = 2048

    # Load tokenizer
    logger.info(f"Loading tokenizer: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    logger.info(f"Loading model from: {args.checkpoint}")
    model = OrthGSAForCausalLM.from_pretrained(
        args.checkpoint,
        torch_dtype=torch.bfloat16,
    )
    model = model.to(device)

    results = {
        "checkpoint": args.checkpoint,
        "model": base_model,
    }

    # Evaluate perplexity
    logger.info("Evaluating perplexity...")
    eval_dataloader = get_slimpajama_dataloader(
        tokenizer=tokenizer,
        batch_size=8,
        max_length=max_seq_length,
        split="train",
        num_workers=4,
        seed=12345,  # Different seed from training
        packed=True,
    )

    perplexity_results = compute_perplexity(
        model=model,
        dataloader=eval_dataloader,
        device=device,
        max_steps=args.max_eval_steps,
    )
    results["perplexity"] = perplexity_results
    logger.info(f"Perplexity: {perplexity_results['perplexity']:.2f}")

    # Evaluate throughput
    if args.eval_throughput:
        logger.info("Evaluating throughput...")
        throughput_results = evaluate_throughput(
            model=model,
            tokenizer=tokenizer,
            device=device,
        )
        results["throughput"] = throughput_results

    # Evaluate generation
    if args.eval_generation:
        logger.info("Evaluating generation...")
        test_prompts = [
            "The capital of France is",
            "def fibonacci(n):",
            "In the year 2050,",
            "The theory of relativity states that",
            "Machine learning is a field of",
        ]

        generation_results = evaluate_generation(
            model=model,
            tokenizer=tokenizer,
            prompts=test_prompts,
            device=device,
        )
        results["generation"] = generation_results

        for result in generation_results:
            logger.info(f"\nPrompt: {result['prompt']}")
            logger.info(f"Completion: {result['completion'][:200]}...")

    # Save results
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()

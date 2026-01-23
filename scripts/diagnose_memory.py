#!/usr/bin/env python3
"""
Memory Diagnostic Script

Identifies where GPU memory is being consumed during model loading and forward pass.
"""

import os
import sys
import gc
from pathlib import Path

import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).parent.parent))


def get_gpu_memory():
    """Get current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        return f"Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB"
    return "No GPU"


def clear_memory():
    """Aggressively clear GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def main():
    print("=" * 60)
    print("Memory Diagnostic for Long Context Training")
    print("=" * 60)

    # Check initial state
    print(f"\n[1] Initial state: {get_gpu_memory()}")

    # Check GPU info
    if torch.cuda.is_available():
        print(f"    GPU: {torch.cuda.get_device_name(0)}")
        print(f"    Total memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f}GB")

    # Load tokenizer
    print("\n[2] Loading tokenizer...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "/home/alfred/models/Qwen3-1.7B",
        trust_remote_code=True,
    )
    print(f"    After tokenizer: {get_gpu_memory()}")
    print(f"    Vocab size: {tokenizer.vocab_size}")

    # Calculate theoretical memory requirements
    print("\n[3] Theoretical Memory Calculation for Qwen3-1.7B:")
    hidden_size = 2048
    num_layers = 28
    vocab_size = tokenizer.vocab_size

    for seq_len in [16384, 32768, 65536, 131072, 262144]:
        # Embeddings
        embed_mem = seq_len * hidden_size * 2 / 1e9  # BF16

        # Hidden states (with gradient checkpointing, ~2 layers active)
        hidden_mem = 2 * seq_len * hidden_size * 2 / 1e9

        # Attention (Flash Attention avoids O(n²), but still needs some memory)
        # Approximate: 2 * batch * heads * seq * head_dim for Q, K
        attn_mem = 2 * seq_len * hidden_size * 2 / 1e9

        # Logits (chunked computation avoids full materialization)
        # But if not chunked: seq_len * vocab_size * 2 bytes
        logits_full = seq_len * vocab_size * 2 / 1e9
        logits_chunked = 512 * vocab_size * 2 / 1e9  # 512 token chunks

        print(f"\n    Sequence length: {seq_len:,} tokens")
        print(f"      Embeddings:      {embed_mem:.2f}GB")
        print(f"      Hidden states:   {hidden_mem:.2f}GB (with checkpointing)")
        print(f"      Attention:       {attn_mem:.2f}GB (Flash Attention)")
        print(f"      Logits (full):   {logits_full:.2f}GB ⚠️")
        print(f"      Logits (chunked):{logits_chunked:.4f}GB ✓")

    # Test sequence creation
    print("\n[4] Testing sequence creation memory...")
    clear_memory()
    print(f"    Before: {get_gpu_memory()}")

    # Create a 128K sequence
    seq_len = 131072
    input_ids = torch.randint(0, vocab_size, (1, seq_len), dtype=torch.long)
    print(f"    After creating {seq_len:,} token sequence (CPU): {get_gpu_memory()}")

    input_ids_gpu = input_ids.cuda()
    print(f"    After moving to GPU: {get_gpu_memory()}")

    del input_ids_gpu
    clear_memory()
    print(f"    After cleanup: {get_gpu_memory()}")

    # Test with smaller chunk
    print("\n[5] Testing chunked approach (16K per GPU with 8 GPUs)...")
    chunk_size = 16384
    chunk = torch.randint(0, vocab_size, (1, chunk_size), dtype=torch.long).cuda()
    print(f"    16K chunk on GPU: {get_gpu_memory()}")

    # Simulate embeddings
    embeddings = torch.randn(1, chunk_size, hidden_size, dtype=torch.bfloat16, device='cuda')
    print(f"    + Embeddings: {get_gpu_memory()}")

    # Simulate hidden states (2 layers active with checkpointing)
    hidden1 = torch.randn(1, chunk_size, hidden_size, dtype=torch.bfloat16, device='cuda')
    hidden2 = torch.randn(1, chunk_size, hidden_size, dtype=torch.bfloat16, device='cuda')
    print(f"    + 2 hidden layers: {get_gpu_memory()}")

    del chunk, embeddings, hidden1, hidden2
    clear_memory()

    # Test model loading with ZeRO
    print("\n[6] Testing model loading (CPU only)...")
    print(f"    Before: {get_gpu_memory()}")

    from orthgsa.models import OrthGSAForCausalLM, OrthGSAConfig

    orthgsa_cfg = OrthGSAConfig(
        n_streams=1,
        alpha_init=0.01,
        k_base=512,
        k_min=128,
        k_max=1024,
        indexer_heads=4,
        indexer_dim=64,
        adaptive_k=True,
    )

    print("    Loading model on CPU...")
    model = OrthGSAForCausalLM(
        base_model_name="/home/alfred/models/Qwen3-1.7B",
        orthgsa_config=orthgsa_cfg,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    print(f"    After loading on CPU: {get_gpu_memory()}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"    Total parameters: {total_params / 1e9:.2f}B")
    print(f"    Memory if all on GPU: {total_params * 2 / 1e9:.2f}GB (BF16)")
    print(f"    Memory per GPU (8-way ZeRO-3): {total_params * 2 / 8 / 1e9:.2f}GB")

    # Move a small part to GPU to test
    print("\n[7] Testing partial GPU usage...")
    # Just move the embedding layer
    if hasattr(model, 'embed_tokens') and model.embed_tokens is not None:
        embed_params = sum(p.numel() for p in model.embed_tokens.parameters())
        model.embed_tokens = model.embed_tokens.cuda()
        print(f"    Embeddings on GPU ({embed_params / 1e6:.2f}M params): {get_gpu_memory()}")
        model.embed_tokens = model.embed_tokens.cpu()
        clear_memory()

    del model
    clear_memory()

    print("\n" + "=" * 60)
    print("RECOMMENDATIONS:")
    print("=" * 60)
    print("""
1. For 128K context (16K per GPU with 8-way sequence parallel):
   - Chunk memory: ~0.5GB per GPU
   - With ZeRO-3: ~0.4GB model per GPU
   - Activations: ~5-10GB (with checkpointing)
   - Total: ~10-15GB should be achievable

2. Key memory hogs to avoid:
   - Full logits tensor (use chunked_lm_loss)
   - Full attention mask (pass None, let SDPA handle)
   - Materializing full sequence before chunking

3. If still OOM:
   - Check if model is loaded on GPU before ZeRO init
   - Verify chunked loss is being used
   - Enable CPU activation checkpointing
   - Reduce chunk_size to 8192 (8K per GPU = 64K total)
""")


if __name__ == "__main__":
    main()

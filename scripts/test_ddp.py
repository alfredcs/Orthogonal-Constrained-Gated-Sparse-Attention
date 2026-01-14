#!/usr/bin/env python3
"""Minimal DDP test for OrthGSA model."""

import os
import sys
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.insert(0, str(__file__).rsplit('/', 1)[0])

from orthgsa.models import OrthGSAForCausalLM, OrthGSAConfig
from transformers import AutoTokenizer


def main():
    # Setup distributed
    distributed = int(os.environ.get("WORLD_SIZE", 1)) > 1

    if distributed:
        dist.init_process_group(backend='nccl', init_method='env://')
        rank = dist.get_rank()
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        world_size = dist.get_world_size()
        torch.cuda.set_device(local_rank)
    else:
        rank = local_rank = 0
        world_size = 1
        torch.cuda.set_device(0)

    device = torch.device(f'cuda:{local_rank}')

    print(f'Rank {rank}: Starting test (world_size={world_size})')

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-4B-Instruct-2507', trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Create config
    config = OrthGSAConfig(n_streams=4, alpha_init=0.01, k_base=512)

    # Create model
    model = OrthGSAForCausalLM('Qwen/Qwen3-4B-Instruct-2507', config, torch_dtype=torch.bfloat16)
    model = model.to(device)

    # Wrap with DDP
    if distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                    find_unused_parameters=True, static_graph=True)

    # Test forward passes
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    for step in range(5):
        optimizer.zero_grad()

        input_ids = torch.randint(0, 10000, (1, 256)).to(device)
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        labels = input_ids.clone()

        with torch.autocast('cuda', dtype=torch.bfloat16):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        loss = outputs.loss
        loss.backward()
        optimizer.step()

        if rank == 0:
            print(f'Step {step}: loss = {outputs.loss.item():.4f}')

    print(f'Rank {rank}: DDP test completed!')

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

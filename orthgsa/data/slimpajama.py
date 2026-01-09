"""
SlimPajama Dataset Loading

Efficiently loads the cerebras/SlimPajama-627B dataset for training.
Uses streaming to handle the large dataset size.
"""

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
from transformers import AutoTokenizer, PreTrainedTokenizer
from datasets import load_dataset, IterableDataset as HFIterableDataset
from typing import Optional, Dict, List, Any, Iterator
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class DataCollatorForLanguageModeling:
    """
    Data collator for causal language modeling.

    Pads sequences to the same length and creates labels for CLM.
    """

    tokenizer: PreTrainedTokenizer
    max_length: int = 2048
    pad_to_multiple_of: Optional[int] = 8  # For tensor core efficiency

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Get input_ids from examples
        if isinstance(examples[0], dict):
            input_ids = [e["input_ids"] for e in examples]
        else:
            input_ids = examples

        # Find max length in batch
        max_len = max(len(ids) for ids in input_ids)

        # Pad to multiple of pad_to_multiple_of
        if self.pad_to_multiple_of:
            max_len = ((max_len + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of) * self.pad_to_multiple_of

        # Clip to max_length
        max_len = min(max_len, self.max_length)

        # Pad sequences
        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []

        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id

        for ids in input_ids:
            # Truncate if needed
            ids = ids[:max_len]

            # Calculate padding
            padding_length = max_len - len(ids)

            # Pad input_ids and create attention mask
            padded_ids = ids + [pad_token_id] * padding_length
            attention_mask = [1] * len(ids) + [0] * padding_length

            # Labels are same as input_ids, with padding set to -100
            labels = ids + [-100] * padding_length

            batch_input_ids.append(padded_ids)
            batch_attention_mask.append(attention_mask)
            batch_labels.append(labels)

        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
        }


class SlimPajamaDataset(IterableDataset):
    """
    Iterable dataset for SlimPajama-627B.

    Uses streaming to efficiently handle the large dataset.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 2048,
        split: str = "train",
        seed: int = 42,
        rank: int = 0,
        world_size: int = 1,
        buffer_size: int = 10000,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.split = split
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.buffer_size = buffer_size

        # Load dataset in streaming mode
        logger.info(f"Loading SlimPajama-627B (streaming, split={split})")
        self.dataset = load_dataset(
            "cerebras/SlimPajama-627B",
            split=split,
            streaming=True,
            trust_remote_code=True,
        )

        # Shuffle the dataset
        self.dataset = self.dataset.shuffle(seed=seed, buffer_size=buffer_size)

    def _tokenize(self, example: Dict[str, Any]) -> Dict[str, List[int]]:
        """Tokenize a single example."""
        text = example.get("text", "")

        # Tokenize with truncation
        tokens = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_attention_mask=False,
        )

        return {"input_ids": tokens["input_ids"]}

    def __iter__(self) -> Iterator[Dict[str, List[int]]]:
        """Iterate over tokenized examples."""
        worker_info = torch.utils.data.get_worker_info()

        # Calculate effective rank for distributed + multi-worker setup
        if worker_info is not None:
            # DataLoader with multiple workers
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            effective_rank = self.rank * num_workers + worker_id
            effective_world_size = self.world_size * num_workers
        else:
            effective_rank = self.rank
            effective_world_size = self.world_size

        # Shard the dataset
        for idx, example in enumerate(self.dataset):
            if idx % effective_world_size == effective_rank:
                tokenized = self._tokenize(example)
                if len(tokenized["input_ids"]) > 0:
                    yield tokenized


class PackedSlimPajamaDataset(IterableDataset):
    """
    Packed dataset that concatenates sequences for efficient training.

    Instead of padding each sequence, sequences are concatenated and
    chunked into fixed-length segments.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 2048,
        split: str = "train",
        seed: int = 42,
        rank: int = 0,
        world_size: int = 1,
        buffer_size: int = 10000,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.split = split
        self.seed = seed
        self.rank = rank
        self.world_size = world_size

        # Load dataset
        logger.info(f"Loading SlimPajama-627B (streaming, packed, split={split})")
        self.dataset = load_dataset(
            "cerebras/SlimPajama-627B",
            split=split,
            streaming=True,
            trust_remote_code=True,
        )
        self.dataset = self.dataset.shuffle(seed=seed, buffer_size=buffer_size)

        # EOS token for sequence separation
        self.eos_token_id = tokenizer.eos_token_id

    def __iter__(self) -> Iterator[Dict[str, List[int]]]:
        """Iterate over packed sequences."""
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is not None:
            effective_rank = self.rank * worker_info.num_workers + worker_info.id
            effective_world_size = self.world_size * worker_info.num_workers
        else:
            effective_rank = self.rank
            effective_world_size = self.world_size

        # Buffer for accumulating tokens
        token_buffer = []

        for idx, example in enumerate(self.dataset):
            if idx % effective_world_size != effective_rank:
                continue

            text = example.get("text", "")
            if not text:
                continue

            # Tokenize
            tokens = self.tokenizer(
                text,
                truncation=False,
                padding=False,
                return_attention_mask=False,
            )["input_ids"]

            # Add EOS token between sequences
            token_buffer.extend(tokens)
            token_buffer.append(self.eos_token_id)

            # Yield full chunks
            while len(token_buffer) >= self.max_length:
                chunk = token_buffer[:self.max_length]
                token_buffer = token_buffer[self.max_length:]
                yield {"input_ids": chunk}


def get_slimpajama_dataloader(
    tokenizer: PreTrainedTokenizer,
    batch_size: int = 4,
    max_length: int = 2048,
    split: str = "train",
    num_workers: int = 4,
    seed: int = 42,
    rank: int = 0,
    world_size: int = 1,
    packed: bool = True,
    pin_memory: bool = True,
    prefetch_factor: int = 2,
) -> DataLoader:
    """
    Create DataLoader for SlimPajama dataset.

    Args:
        tokenizer: Tokenizer to use
        batch_size: Batch size per device
        max_length: Maximum sequence length
        split: Dataset split ("train" or "validation")
        num_workers: Number of DataLoader workers
        seed: Random seed
        rank: Process rank for distributed training
        world_size: Total number of processes
        packed: Whether to use packed sequences
        pin_memory: Whether to pin memory
        prefetch_factor: Prefetch factor for DataLoader

    Returns:
        DataLoader for the dataset
    """
    # Create dataset
    if packed:
        dataset = PackedSlimPajamaDataset(
            tokenizer=tokenizer,
            max_length=max_length,
            split=split,
            seed=seed,
            rank=rank,
            world_size=world_size,
        )
    else:
        dataset = SlimPajamaDataset(
            tokenizer=tokenizer,
            max_length=max_length,
            split=split,
            seed=seed,
            rank=rank,
            world_size=world_size,
        )

    # Create data collator
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        max_length=max_length,
    )

    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )

    return dataloader


def create_data_collator(
    tokenizer: PreTrainedTokenizer,
    max_length: int = 2048,
) -> DataCollatorForLanguageModeling:
    """Create data collator for language modeling."""
    return DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        max_length=max_length,
    )


def test_slimpajama_loading():
    """Test SlimPajama data loading."""
    from transformers import AutoTokenizer

    print("Testing SlimPajama data loading...")

    # Load tokenizer (use a small model for testing)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Create dataset
    dataset = SlimPajamaDataset(
        tokenizer=tokenizer,
        max_length=512,
        split="train",
    )

    # Get a few examples
    print("Getting examples from dataset...")
    for i, example in enumerate(dataset):
        print(f"Example {i}: {len(example['input_ids'])} tokens")
        if i >= 2:
            break

    print("\nSlimPajama loading test passed!")


if __name__ == "__main__":
    test_slimpajama_loading()

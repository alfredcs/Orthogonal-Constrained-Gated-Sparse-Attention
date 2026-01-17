"""
SlimPajama Dataset Loading

Efficiently loads the SlimPajama-627B dataset for training.
Supports streaming from S3 bucket without downloading to local filesystem.
"""

import os
import io
import json
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
from transformers import AutoTokenizer, PreTrainedTokenizer
from datasets import load_dataset, IterableDataset as HFIterableDataset
from typing import Optional, Dict, List, Any, Iterator, Generator
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


def _is_s3_path(path: str) -> bool:
    """Check if the path is an S3 URI."""
    return path.startswith("s3://")


def _parse_s3_path(s3_path: str) -> tuple:
    """Parse S3 URI into bucket and prefix."""
    path = s3_path.replace("s3://", "").rstrip("/")
    parts = path.split("/", 1)
    bucket = parts[0]
    prefix = parts[1] if len(parts) > 1 else ""
    return bucket, prefix


def _get_s3_client():
    """Get boto3 S3 client using default credentials."""
    import boto3
    session = boto3.Session(profile_name="default")
    return session.client("s3")


def _list_s3_files(bucket: str, prefix: str, extensions: List[str]) -> List[str]:
    """List files in S3 bucket with given extensions."""
    s3 = _get_s3_client()
    files = []

    paginator = s3.get_paginator("list_objects_v2")

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        if "Contents" not in page:
            continue
        for obj in page["Contents"]:
            key = obj["Key"]
            if any(key.endswith(ext) for ext in extensions):
                files.append(key)

    return sorted(files)


def _stream_s3_jsonl_zst(bucket: str, key: str) -> Generator[Dict, None, None]:
    """Stream and decompress a .jsonl.zst file from S3."""
    import zstandard as zstd

    s3 = _get_s3_client()

    # Get the object
    response = s3.get_object(Bucket=bucket, Key=key)
    body = response["Body"]

    # Create zstd decompressor
    dctx = zstd.ZstdDecompressor()

    # Stream decompress and parse JSONL
    with dctx.stream_reader(body) as reader:
        text_stream = io.TextIOWrapper(reader, encoding="utf-8")
        for line in text_stream:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def _stream_s3_jsonl(bucket: str, key: str) -> Generator[Dict, None, None]:
    """Stream a .jsonl file from S3."""
    s3 = _get_s3_client()

    response = s3.get_object(Bucket=bucket, Key=key)
    body = response["Body"]

    for line in body.iter_lines():
        line = line.decode("utf-8").strip()
        if line:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _stream_s3_json_zst(bucket: str, key: str) -> Generator[Dict, None, None]:
    """Stream and decompress a .json.zst file from S3."""
    import zstandard as zstd

    s3 = _get_s3_client()

    response = s3.get_object(Bucket=bucket, Key=key)
    body = response["Body"]

    dctx = zstd.ZstdDecompressor()

    with dctx.stream_reader(body) as reader:
        data = reader.read()
        try:
            # Try as JSON array
            items = json.loads(data)
            if isinstance(items, list):
                for item in items:
                    yield item
            else:
                yield items
        except json.JSONDecodeError:
            # Try as JSONL
            for line in data.decode("utf-8").split("\n"):
                line = line.strip()
                if line:
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue


class S3StreamingDataset(IterableDataset):
    """
    Custom streaming dataset that reads directly from S3 using boto3.
    Avoids s3fs/aiobotocore compatibility issues.
    """

    def __init__(
        self,
        s3_path: str,
        split: str = "train",
        seed: int = 42,
        shuffle_files: bool = True,
    ):
        self.s3_path = s3_path
        self.split = split
        self.seed = seed
        self.shuffle_files = shuffle_files

        # Parse S3 path
        self.bucket, self.prefix = _parse_s3_path(s3_path)

        # Add split to prefix if needed
        split_prefix = f"{self.prefix}/{split}" if self.prefix else split

        # List files
        extensions = [".jsonl.zst", ".json.zst", ".jsonl", ".json"]
        self.files = _list_s3_files(self.bucket, split_prefix, extensions)

        # If no files found with split prefix, try without
        if not self.files:
            self.files = _list_s3_files(self.bucket, self.prefix, extensions)

        if not self.files:
            raise FileNotFoundError(
                f"No data files found in S3 path: {s3_path}\n"
                f"Bucket: {self.bucket}, Prefix: {self.prefix}\n"
                f"Tried extensions: {extensions}"
            )

        logger.info(f"Found {len(self.files)} files in s3://{self.bucket}/{self.prefix}")

    def _get_file_streamer(self, key: str):
        """Get appropriate streamer based on file extension."""
        if key.endswith(".jsonl.zst"):
            return _stream_s3_jsonl_zst(self.bucket, key)
        elif key.endswith(".json.zst"):
            return _stream_s3_json_zst(self.bucket, key)
        elif key.endswith(".jsonl"):
            return _stream_s3_jsonl(self.bucket, key)
        else:
            raise ValueError(f"Unsupported file format: {key}")

    def __iter__(self) -> Iterator[Dict]:
        worker_info = torch.utils.data.get_worker_info()

        files = self.files.copy()

        # Shuffle files if requested
        if self.shuffle_files:
            import random
            rng = random.Random(self.seed)
            rng.shuffle(files)

        # Shard files across workers
        if worker_info is not None:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            files = files[worker_id::num_workers]

        # Stream from each file
        for key in files:
            try:
                for example in self._get_file_streamer(key):
                    yield example
            except Exception as e:
                logger.warning(f"Error reading {key}: {e}")
                continue


def _load_dataset_from_s3(
    s3_path: str,
    split: str = "train",
    streaming: bool = True,
    seed: int = 42,
) -> S3StreamingDataset:
    """
    Load dataset directly from S3 bucket using boto3.

    Args:
        s3_path: S3 URI (e.g., s3://bucket-name/path/to/dataset)
        split: Dataset split to load
        streaming: Whether to stream the dataset (always True for S3)
        seed: Random seed for file shuffling

    Returns:
        S3StreamingDataset
    """
    logger.info(f"Loading dataset from S3: {s3_path} (split={split})")
    return S3StreamingDataset(s3_path, split=split, seed=seed)


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
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.bool),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
        }


class SlimPajamaDataset(IterableDataset):
    """
    Iterable dataset for SlimPajama-627B.

    Uses streaming to efficiently handle the large dataset.
    Supports loading from S3 bucket directly without downloading.
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
        dataset_path: Optional[str] = None,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.split = split
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.buffer_size = buffer_size
        self._is_s3 = False

        # Load dataset - supports S3 paths, local paths, and HuggingFace hub
        if dataset_path and _is_s3_path(dataset_path):
            # Load from S3 bucket using boto3 (avoids s3fs/aiobotocore issues)
            logger.info(f"Loading from S3: {dataset_path} (streaming, split={split})")
            self.dataset = _load_dataset_from_s3(dataset_path, split=split, seed=seed)
            self._is_s3 = True
        elif dataset_path:
            # Load from local or HuggingFace path
            logger.info(f"Loading from {dataset_path} (streaming, split={split})")
            self.dataset = load_dataset(dataset_path, split=split, streaming=True)
            self.dataset = self.dataset.shuffle(seed=seed, buffer_size=buffer_size)
        else:
            # Default to HuggingFace hub
            logger.info(f"Loading SlimPajama-627B from HuggingFace (streaming, split={split})")
            self.dataset = load_dataset("cerebras/SlimPajama-627B", split=split, streaming=True)
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

    Supports loading from S3 bucket directly without downloading.
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
        dataset_name: str = "cerebras/SlimPajama-627B",
        local_path: Optional[str] = None,
        dataset_path: Optional[str] = None,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.split = split
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.dataset_name = dataset_name
        self.local_path = local_path
        self.dataset_path = dataset_path
        self._is_s3 = False

        # Priority: dataset_path (S3 or explicit path) > local_path > dataset_name (HuggingFace)
        if dataset_path and _is_s3_path(dataset_path):
            # Load directly from S3 bucket using boto3 (avoids s3fs/aiobotocore issues)
            logger.info(f"Loading from S3: {dataset_path} (streaming, packed, split={split})")
            self.dataset = _load_dataset_from_s3(dataset_path, split=split, seed=seed)
            self._is_s3 = True
        elif dataset_path:
            # Load from explicit path (local or HuggingFace identifier)
            logger.info(f"Loading from {dataset_path} (streaming, packed, split={split})")
            self.dataset = load_dataset(dataset_path, split=split, streaming=True)
            self.dataset = self.dataset.shuffle(seed=seed, buffer_size=buffer_size)
        elif local_path:
            expanded_path = os.path.expanduser(local_path)
            if os.path.exists(expanded_path) and os.listdir(expanded_path):
                logger.info(f"Loading from local path: {expanded_path} (streaming, packed, split={split})")
                self.dataset = load_dataset(expanded_path, split=split, streaming=True)
                self.dataset = self.dataset.shuffle(seed=seed, buffer_size=buffer_size)
            else:
                logger.info(f"Local path not found or empty, using {dataset_name} (streaming, packed, split={split})")
                self.dataset = load_dataset(dataset_name, split=split, streaming=True)
                self.dataset = self.dataset.shuffle(seed=seed, buffer_size=buffer_size)
        else:
            logger.info(f"Loading {dataset_name} (streaming, packed, split={split})")
            self.dataset = load_dataset(dataset_name, split=split, streaming=True)
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
    dataset_name: str = "cerebras/SlimPajama-627B",
    local_path: Optional[str] = None,
    dataset_path: Optional[str] = None,
) -> DataLoader:
    """
    Create DataLoader for dataset.

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
        dataset_name: Name of the dataset to load (HuggingFace identifier)
        local_path: Local path to pre-downloaded dataset (optional)
        dataset_path: S3 URI or path to dataset (takes priority over local_path and dataset_name)

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
            dataset_name=dataset_name,
            local_path=local_path,
            dataset_path=dataset_path,
        )
    else:
        dataset = SlimPajamaDataset(
            tokenizer=tokenizer,
            max_length=max_length,
            split=split,
            seed=seed,
            rank=rank,
            world_size=world_size,
            dataset_path=dataset_path,
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

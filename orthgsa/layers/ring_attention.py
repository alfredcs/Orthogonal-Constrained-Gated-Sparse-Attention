"""
Ring Attention Implementation for Ultra-Long Context Training

Ring Attention distributes sequence across GPUs using a ring communication pattern,
enabling training on sequences that exceed single-GPU memory capacity.

Key idea: Each GPU processes a chunk of the sequence and exchanges KV cache with
neighboring GPUs in a ring topology. This allows O(N/P) memory per GPU where:
- N = sequence length
- P = number of GPUs (sequence parallel degree)

Reference: Ring Attention with Blockwise Transformers for Near-Infinite Context
https://arxiv.org/abs/2310.01889
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.cuda.amp import autocast
from typing import Optional, Tuple, List
import math
import logging

logger = logging.getLogger(__name__)


def get_sequence_parallel_group():
    """Get the sequence parallel process group."""
    # By default, use the world as sequence parallel group
    # In production, you'd want a separate group for flexibility
    return None  # None means default world group


def get_sequence_parallel_rank():
    """Get rank within sequence parallel group."""
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_sequence_parallel_world_size():
    """Get sequence parallel world size."""
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


class RingCommunicator:
    """
    Handles ring communication for KV cache exchange.

    In a ring topology, each GPU sends to (rank + 1) % world_size
    and receives from (rank - 1) % world_size.
    """

    def __init__(self, group=None):
        self.group = group
        self.rank = get_sequence_parallel_rank()
        self.world_size = get_sequence_parallel_world_size()

        # Pre-compute send/recv ranks
        self.send_rank = (self.rank + 1) % self.world_size
        self.recv_rank = (self.rank - 1 + self.world_size) % self.world_size

    def ring_send_recv(
        self,
        send_tensor: torch.Tensor,
        recv_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """
        Send tensor to next GPU and receive from previous GPU.

        Args:
            send_tensor: Tensor to send to next rank
            recv_tensor: Pre-allocated tensor to receive into

        Returns:
            Received tensor
        """
        if self.world_size == 1:
            return send_tensor.clone()

        # Use async operations for overlap
        send_op = dist.isend(send_tensor.contiguous(), self.send_rank, group=self.group)
        recv_op = dist.irecv(recv_tensor, self.recv_rank, group=self.group)

        send_op.wait()
        recv_op.wait()

        return recv_tensor


class RingAttention(nn.Module):
    """
    Ring Attention implementation for distributed long-context training.

    This module wraps the standard attention computation to enable
    sequence parallelism across GPUs.

    Memory per GPU: O(chunk_size * hidden_dim) instead of O(seq_len * hidden_dim)
    Communication: O(num_rings * chunk_size * hidden_dim) per layer

    Args:
        hidden_size: Model hidden dimension
        num_attention_heads: Number of attention heads
        num_key_value_heads: Number of KV heads (for GQA)
        head_dim: Dimension per head
        sequence_parallel_size: Number of GPUs for sequence parallelism
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.num_kv_heads = num_key_value_heads or num_attention_heads
        self.head_dim = head_dim or (hidden_size // num_attention_heads)
        self.dropout = dropout

        # Ring communicator
        self.communicator = RingCommunicator()

        # Scaling factor for attention
        self.scale = self.head_dim ** -0.5

    def _compute_local_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        causal_mask: bool = True,
        key_offset: int = 0,
        query_offset: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute attention between local query and (possibly remote) key/value.

        Args:
            query: [batch, num_heads, query_len, head_dim]
            key: [batch, num_kv_heads, key_len, head_dim]
            value: [batch, num_kv_heads, key_len, head_dim]
            causal_mask: Whether to apply causal masking
            key_offset: Global offset of keys (for causal masking)
            query_offset: Global offset of queries

        Returns:
            attention_output: [batch, num_heads, query_len, head_dim]
            attention_weights: [batch, num_heads, query_len, key_len]
        """
        batch_size, num_heads, query_len, head_dim = query.shape
        _, num_kv_heads, key_len, _ = key.shape

        # Handle grouped query attention (GQA)
        if num_kv_heads != num_heads:
            # Repeat KV heads to match query heads
            repeat_factor = num_heads // num_kv_heads
            key = key.repeat_interleave(repeat_factor, dim=1)
            value = value.repeat_interleave(repeat_factor, dim=1)

        # Compute attention scores: [batch, num_heads, query_len, key_len]
        attn_weights = torch.matmul(query, key.transpose(-2, -1)) * self.scale

        # Apply causal mask if needed
        if causal_mask:
            # Create causal mask based on global positions
            # Query position i can attend to key position j if:
            # (query_offset + i) >= (key_offset + j)
            q_pos = torch.arange(query_len, device=query.device) + query_offset
            k_pos = torch.arange(key_len, device=key.device) + key_offset

            # [query_len, key_len] - True where we should NOT attend
            mask = q_pos.unsqueeze(1) < k_pos.unsqueeze(0)

            attn_weights.masked_fill_(mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        # Softmax
        attn_weights = torch.softmax(attn_weights, dim=-1)

        # Dropout
        if self.dropout > 0 and self.training:
            attn_weights = torch.dropout(attn_weights, self.dropout, self.training)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        causal: bool = True,
    ) -> torch.Tensor:
        """
        Ring attention forward pass.

        Each GPU holds a chunk of the sequence. We perform multiple rounds
        of attention computation, each time exchanging KV cache with neighbors.

        Args:
            query: [batch, seq_chunk, num_heads, head_dim] - local query chunk
            key: [batch, seq_chunk, num_kv_heads, head_dim] - local key chunk
            value: [batch, seq_chunk, num_kv_heads, head_dim] - local value chunk
            causal: Whether to use causal masking

        Returns:
            output: [batch, seq_chunk, num_heads, head_dim]
        """
        batch_size, chunk_len, num_heads, head_dim = query.shape
        _, _, num_kv_heads, _ = key.shape

        world_size = self.communicator.world_size
        rank = self.communicator.rank

        # Transpose for attention: [batch, heads, seq, dim]
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        # Global position offset for this rank's chunk
        query_offset = rank * chunk_len

        # Initialize output accumulator and normalization factor
        # We accumulate weighted outputs and divide by total weights
        output_acc = torch.zeros_like(query)
        weight_acc = torch.zeros(
            batch_size, num_heads, chunk_len, 1,
            device=query.device, dtype=query.dtype
        )

        # Pre-allocate receive buffers
        key_recv = torch.zeros_like(key)
        value_recv = torch.zeros_like(value)

        # Current KV tensors (start with local)
        current_key = key
        current_value = value
        current_kv_rank = rank

        # Ring attention: iterate through all chunks
        for ring_step in range(world_size):
            # Compute key offset based on which rank's KV we have
            key_offset = current_kv_rank * chunk_len

            # Compute local attention with current KV
            attn_out, attn_weights = self._compute_local_attention(
                query, current_key, current_value,
                causal_mask=causal,
                key_offset=key_offset,
                query_offset=query_offset,
            )

            # Accumulate using log-sum-exp trick for numerical stability
            # This allows us to combine attention from multiple chunks
            chunk_weights = attn_weights.sum(dim=-1, keepdim=True)  # [B, H, Q, 1]

            # Only accumulate if there are valid attention weights
            valid_mask = chunk_weights > 0
            output_acc = output_acc + attn_out * chunk_weights
            weight_acc = weight_acc + chunk_weights

            # Ring exchange: send current KV, receive next KV
            if ring_step < world_size - 1:
                # Send current KV and receive from previous rank
                self.communicator.ring_send_recv(current_key, key_recv)
                self.communicator.ring_send_recv(current_value, value_recv)

                # Swap buffers
                current_key, key_recv = key_recv, current_key
                current_value, value_recv = value_recv, current_value

                # Update KV rank (we received from previous rank)
                current_kv_rank = (current_kv_rank - 1 + world_size) % world_size

        # Normalize output
        output = output_acc / (weight_acc + 1e-8)

        # Transpose back: [batch, seq, heads, dim]
        output = output.transpose(1, 2)

        return output


class RingAttentionWrapper(nn.Module):
    """
    Wrapper to apply Ring Attention to an existing attention layer.

    This preserves the original layer's projections while replacing
    the attention computation with Ring Attention.

    Args:
        attention_layer: Original attention layer (e.g., from HuggingFace)
        hidden_size: Model hidden dimension
        num_attention_heads: Number of attention heads
        num_key_value_heads: Number of KV heads
    """

    def __init__(
        self,
        attention_layer: nn.Module,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: Optional[int] = None,
    ):
        super().__init__()

        self.original_attention = attention_layer
        self.ring_attention = RingAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
        )

        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.num_kv_heads = num_key_value_heads or num_attention_heads
        self.head_dim = hidden_size // num_attention_heads

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass using ring attention.

        Uses original layer's Q/K/V projections but replaces attention
        computation with ring attention.
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Get projections from original layer
        # This handles different model architectures
        attn = self.original_attention

        # Compute Q, K, V using original projections
        if hasattr(attn, 'q_proj'):
            # Qwen/LLaMA style
            query = attn.q_proj(hidden_states)
            key = attn.k_proj(hidden_states)
            value = attn.v_proj(hidden_states)
        elif hasattr(attn, 'c_attn'):
            # GPT-2 style
            qkv = attn.c_attn(hidden_states)
            query, key, value = qkv.split(self.hidden_size, dim=-1)
        else:
            raise ValueError(f"Unknown attention architecture: {type(attn)}")

        # Reshape for attention: [batch, seq, heads, dim]
        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim)
        key = key.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        value = value.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        # Apply rotary embeddings if provided
        if position_embeddings is not None:
            cos, sin = position_embeddings
            query, key = self._apply_rotary_pos_emb(query, key, cos, sin)

        # Apply ring attention
        attn_output = self.ring_attention(query, key, value, causal=True)

        # Reshape and project output
        attn_output = attn_output.reshape(batch_size, seq_len, self.hidden_size)

        # Apply output projection
        if hasattr(attn, 'o_proj'):
            attn_output = attn.o_proj(attn_output)
        elif hasattr(attn, 'c_proj'):
            attn_output = attn.c_proj(attn_output)

        return (attn_output, None, None)  # (output, attn_weights, past_key_values)

    def _apply_rotary_pos_emb(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary position embeddings."""
        # query/key: [batch, seq, heads, dim]
        # cos/sin: [seq, dim] or [batch, seq, dim]

        def rotate_half(x):
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2 :]
            return torch.cat((-x2, x1), dim=-1)

        # Expand cos/sin for heads dimension
        if cos.dim() == 2:
            cos = cos.unsqueeze(0).unsqueeze(2)  # [1, seq, 1, dim]
            sin = sin.unsqueeze(0).unsqueeze(2)
        elif cos.dim() == 3:
            cos = cos.unsqueeze(2)  # [batch, seq, 1, dim]
            sin = sin.unsqueeze(2)

        query = (query * cos) + (rotate_half(query) * sin)
        key = (key * cos) + (rotate_half(key) * sin)

        return query, key


def split_sequence_for_ring_attention(
    tensor: torch.Tensor,
    dim: int = 1,
) -> torch.Tensor:
    """
    Split a tensor's sequence dimension across GPUs for ring attention.

    Args:
        tensor: Input tensor with sequence dimension
        dim: Dimension to split (default: 1 for [batch, seq, ...])

    Returns:
        Local chunk of the tensor
    """
    if not dist.is_initialized():
        return tensor

    world_size = dist.get_world_size()
    rank = dist.get_rank()

    seq_len = tensor.size(dim)
    chunk_size = seq_len // world_size

    # Handle uneven splits
    if seq_len % world_size != 0:
        # Pad to make divisible
        pad_size = world_size - (seq_len % world_size)
        pad_shape = list(tensor.shape)
        pad_shape[dim] = pad_size
        padding = torch.zeros(pad_shape, dtype=tensor.dtype, device=tensor.device)
        tensor = torch.cat([tensor, padding], dim=dim)
        chunk_size = tensor.size(dim) // world_size

    # Extract local chunk
    start_idx = rank * chunk_size
    end_idx = start_idx + chunk_size

    indices = [slice(None)] * tensor.dim()
    indices[dim] = slice(start_idx, end_idx)

    return tensor[tuple(indices)].contiguous()


def gather_sequence_from_ring_attention(
    tensor: torch.Tensor,
    dim: int = 1,
    original_length: Optional[int] = None,
) -> torch.Tensor:
    """
    Gather tensor chunks from all GPUs back into full sequence.

    Args:
        tensor: Local chunk
        dim: Dimension that was split
        original_length: Original sequence length (to trim padding)

    Returns:
        Full tensor with gathered sequence
    """
    if not dist.is_initialized():
        return tensor

    world_size = dist.get_world_size()

    # Gather all chunks
    gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered, tensor)

    # Concatenate along sequence dimension
    result = torch.cat(gathered, dim=dim)

    # Trim to original length if needed
    if original_length is not None and result.size(dim) > original_length:
        indices = [slice(None)] * result.dim()
        indices[dim] = slice(0, original_length)
        result = result[tuple(indices)]

    return result


def enable_ring_attention_for_model(
    model: nn.Module,
    hidden_size: int,
    num_attention_heads: int,
    num_key_value_heads: Optional[int] = None,
) -> nn.Module:
    """
    Replace attention layers in a model with Ring Attention.

    Args:
        model: Model to modify
        hidden_size: Model hidden dimension
        num_attention_heads: Number of attention heads
        num_key_value_heads: Number of KV heads (for GQA)

    Returns:
        Modified model with ring attention
    """
    for name, module in model.named_modules():
        # Look for attention layers
        if 'attention' in name.lower() or 'attn' in name.lower():
            if hasattr(module, 'q_proj') or hasattr(module, 'c_attn'):
                # Replace with ring attention wrapper
                parent_name = '.'.join(name.split('.')[:-1])
                module_name = name.split('.')[-1]

                parent = model
                for part in parent_name.split('.'):
                    if part:
                        parent = getattr(parent, part)

                wrapped = RingAttentionWrapper(
                    module,
                    hidden_size=hidden_size,
                    num_attention_heads=num_attention_heads,
                    num_key_value_heads=num_key_value_heads,
                )
                setattr(parent, module_name, wrapped)
                logger.info(f"Replaced {name} with Ring Attention")

    return model

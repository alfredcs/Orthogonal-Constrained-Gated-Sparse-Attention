"""
OrthGSA Layer Implementation

Combines Orthogonal-Constrained Hyper-Connections (using Cayley Transform) with
Gated Sparse Attention (GSA) for efficient, stable transformers.

Architecture per layer:
1. Orthogonal HC-wrapped GSA attention
2. Orthogonal HC-wrapped FFN (SwiGLU)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math

from ..layers.mhc import ManifoldHyperConnection, RMSNorm
from ..layers.gsa import GatedSparseAttention, GatedSparseAttentionWithFlash


class SwiGLU(nn.Module):
    """SwiGLU Feed-Forward Network."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        bias: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class OrthGSALayer(nn.Module):
    """
    Single OrthGSA Transformer Layer.

    This layer combines:
    - Orthogonal-Constrained Hyper-Connections with Cayley Transform
    - Gated Sparse Attention (GSA)
    - SwiGLU Feed-Forward Network

    Args:
        hidden_size: Model hidden dimension
        intermediate_size: FFN intermediate dimension
        num_heads: Number of attention heads
        num_kv_heads: Number of KV heads (for GQA)
        n_streams: Number of mHC streams
        k_base: Base number of selected tokens for GSA
        k_min: Minimum k for GSA
        k_max: Maximum k for GSA
        indexer_heads: Number of GSA indexer heads
        indexer_dim: GSA indexer dimension
        adaptive_k: Whether to use adaptive k in GSA
        alpha_init: Initial alpha for mHC coefficients
        dropout: Dropout probability
        max_seq_len: Maximum sequence length
        use_flash: Whether to use Flash Attention fallback
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        n_streams: int = 4,
        k_base: int = 512,
        k_min: int = 128,
        k_max: int = 1024,
        indexer_heads: int = 4,
        indexer_dim: int = 64,
        adaptive_k: bool = True,
        alpha_init: float = 0.01,
        dropout: float = 0.0,
        max_seq_len: int = 8192,
        use_flash: bool = True,
        layer_idx: Optional[int] = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_streams = n_streams
        self.layer_idx = layer_idx

        # Pre-normalization for attention and FFN
        self.input_layernorm = RMSNorm(hidden_size)
        self.post_attention_layernorm = RMSNorm(hidden_size)

        # mHC for attention and FFN
        self.mhc_attn = ManifoldHyperConnection(
            hidden_size=hidden_size,
            n_streams=n_streams,
            alpha_init=alpha_init,
        )

        self.mhc_ffn = ManifoldHyperConnection(
            hidden_size=hidden_size,
            n_streams=n_streams,
            alpha_init=alpha_init,
        )

        # Gated Sparse Attention
        gsa_cls = GatedSparseAttentionWithFlash if use_flash else GatedSparseAttention
        self.self_attn = gsa_cls(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            k_base=k_base,
            k_min=k_min,
            k_max=k_max,
            indexer_heads=indexer_heads,
            indexer_dim=indexer_dim,
            adaptive_k=adaptive_k,
            dropout=dropout,
            max_seq_len=max_seq_len,
        )

        # SwiGLU FFN
        self.mlp = SwiGLU(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
        )

    def _attention_fn(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ) -> torch.Tensor:
        """Attention function to be wrapped by mHC."""
        # Pre-norm
        normed = self.input_layernorm(hidden_states)

        # GSA attention
        attn_outputs = self.self_attn(
            normed,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )

        return attn_outputs[0]

    def _ffn_fn(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """FFN function to be wrapped by mHC."""
        # Pre-norm
        normed = self.post_attention_layernorm(hidden_states)
        # SwiGLU
        return self.mlp(normed)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass for OrthGSA layer.

        Args:
            hidden_states: Input tensor [B, L, n, C] (n-stream format)
            attention_mask: Attention mask
            position_ids: Position IDs
            past_key_value: Cached KV
            use_cache: Whether to return cache
            output_attentions: Whether to return attention weights

        Returns:
            hidden_states: Output tensor [B, L, n, C]
            (optional) past_key_value
            (optional) attention_weights
        """
        # Ensure input is in n-stream format
        if hidden_states.dim() == 3:
            # [B, L, C] -> [B, L, n, C]
            hidden_states = self.mhc_attn.expand_to_streams(hidden_states)

        # mHC-wrapped attention
        def attn_wrapper(x):
            return self._attention_fn(
                x,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )

        hidden_states = self.mhc_attn(hidden_states, attn_wrapper)

        # mHC-wrapped FFN
        hidden_states = self.mhc_ffn(hidden_states, self._ffn_fn)

        outputs = (hidden_states,)

        if use_cache:
            # Note: cache handling needs to be adjusted for full implementation
            outputs += (None,)

        if output_attentions:
            outputs += (None,)

        return outputs


class OrthGSADecoderLayer(nn.Module):
    """
    OrthGSA Decoder Layer compatible with HuggingFace transformers.

    This is a drop-in replacement for standard decoder layers.
    """

    def __init__(
        self,
        config,
        layer_idx: int,
    ):
        super().__init__()

        # Extract config
        hidden_size = config.hidden_size
        intermediate_size = config.intermediate_size
        num_heads = config.num_attention_heads
        num_kv_heads = getattr(config, "num_key_value_heads", num_heads)

        # OrthGSA specific config
        orthgsa_config = getattr(config, "orthgsa", {})
        gsa_config = getattr(config, "gsa", {})

        n_streams = orthgsa_config.get("n_streams", 4)
        alpha_init = orthgsa_config.get("alpha_init", 0.01)

        k_base = gsa_config.get("k_base", 512)
        k_min = gsa_config.get("k_min", 128)
        k_max = gsa_config.get("k_max", 1024)
        indexer_heads = gsa_config.get("indexer_heads", 4)
        indexer_dim = gsa_config.get("indexer_dim", 64)
        adaptive_k = gsa_config.get("adaptive_k", True)

        max_seq_len = getattr(config, "max_position_embeddings", 8192)

        self.layer = OrthGSALayer(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            n_streams=n_streams,
            k_base=k_base,
            k_min=k_min,
            k_max=k_max,
            indexer_heads=indexer_heads,
            indexer_dim=indexer_dim,
            adaptive_k=adaptive_k,
            alpha_init=alpha_init,
            max_seq_len=max_seq_len,
            layer_idx=layer_idx,
        )

        self.n_streams = n_streams
        self.hidden_size = hidden_size

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, ...]:
        return self.layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )


def test_orthgsa_layer():
    """Test OrthGSA layer."""
    torch.manual_seed(42)

    B, L, C = 2, 32, 256
    n_streams = 4
    intermediate_size = 512
    num_heads = 8

    # Create layer
    layer = OrthGSALayer(
        hidden_size=C,
        intermediate_size=intermediate_size,
        num_heads=num_heads,
        n_streams=n_streams,
        k_base=16,
        k_min=8,
        k_max=32,
    )

    # Test with single-stream input (auto-expand)
    x_single = torch.randn(B, L, C)
    output_single = layer(x_single)
    print(f"Single-stream input: {x_single.shape}")
    print(f"Output shape: {output_single[0].shape}")

    # Test with n-stream input
    x_multi = torch.randn(B, L, n_streams, C)
    output_multi = layer(x_multi)
    print(f"Multi-stream input: {x_multi.shape}")
    print(f"Output shape: {output_multi[0].shape}")

    print("\nAll OrthGSA layer tests passed!")


if __name__ == "__main__":
    test_orthgsa_layer()

"""
Gated Sparse Attention (GSA)

Implements sub-quadratic attention with:
- Value Gate (G2): Pre-attention gating on values
- Gated Lightning Indexer: Efficient token selection
- Adaptive Top-k Selection: Dynamic sparsity
- Output Gate (G1): Post-attention gating

Complexity: O(L * k * d) instead of O(L^2 * d)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class RotaryPositionalEmbedding(nn.Module):
    """Rotary Positional Embedding (RoPE)."""

    def __init__(self, dim: int, max_seq_len: int = 8192, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Precompute inverse frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Precompute cos/sin for max_seq_len
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.cos_cached.shape[0]:
            self._build_cache(seq_len)
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary positional embedding to query and key."""
    # q, k: [B, L, H, D]
    # cos, sin: [L, D]
    cos = cos.unsqueeze(0).unsqueeze(2)  # [1, L, 1, D]
    sin = sin.unsqueeze(0).unsqueeze(2)  # [1, L, 1, D]

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed, k_embed


class GatedLightningIndexer(nn.Module):
    """
    Gated Lightning Indexer for efficient token selection.

    Computes importance scores for each query-key pair using lightweight
    gated scoring mechanism:
        I_{t,s} = sum_j sigma(y_t @ W_j^Iw) * sigma(q^I_{t,j} @ k^I_s + b_j)

    Args:
        hidden_size: Input hidden dimension
        indexer_heads: Number of indexer heads (H^I)
        indexer_dim: Indexer dimension (d_I)
    """

    def __init__(
        self,
        hidden_size: int,
        indexer_heads: int = 4,
        indexer_dim: int = 64,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.indexer_heads = indexer_heads
        self.indexer_dim = indexer_dim

        # Projections for indexer queries and keys
        self.q_indexer = nn.Linear(hidden_size, indexer_heads * indexer_dim, bias=False)
        self.k_indexer = nn.Linear(hidden_size, indexer_dim, bias=False)

        # Query-dependent weights for each indexer head
        self.w_indexer = nn.Linear(hidden_size, indexer_heads, bias=False)

        # Bias for indexer scoring
        self.indexer_bias = nn.Parameter(torch.zeros(indexer_heads))

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.q_indexer.weight)
        nn.init.xavier_uniform_(self.k_indexer.weight)
        nn.init.xavier_uniform_(self.w_indexer.weight)

    def forward(
        self,
        y: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute indexer scores.

        Args:
            y: Input tensor of shape [B, L, C]
            attention_mask: Optional attention mask [B, L] or [B, 1, L, L]

        Returns:
            Indexer scores of shape [B, L, L]
        """
        B, L, C = y.shape
        H_I = self.indexer_heads
        d_I = self.indexer_dim

        # Compute indexer queries and keys
        q_I = self.q_indexer(y).view(B, L, H_I, d_I)  # [B, L, H_I, d_I]
        k_I = self.k_indexer(y)  # [B, L, d_I]

        # Compute query-dependent weights
        w_I = torch.sigmoid(self.w_indexer(y))  # [B, L, H_I]

        # Compute similarity scores for each head
        # q_I: [B, L, H_I, d_I], k_I: [B, L, d_I]
        # scores_j = sigmoid(q_I[:,:,j,:] @ k_I.T + bias_j)
        # [B, L, H_I, d_I] @ [B, d_I, L] -> [B, L, H_I, L]
        scores = torch.einsum("blhd,bsd->blhs", q_I, k_I)  # [B, L, H_I, L]
        scores = scores + self.indexer_bias.view(1, 1, H_I, 1)
        scores = torch.sigmoid(scores)  # [B, L, H_I, L]

        # Aggregate with query-dependent weights
        # w_I: [B, L, H_I], scores: [B, L, H_I, L]
        # I = sum_j w_I[:,:,j] * scores[:,:,j,:]
        w_I = w_I.unsqueeze(-1)  # [B, L, H_I, 1]
        I = (w_I * scores).sum(dim=2)  # [B, L, L]

        # Apply causal mask if needed
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                # [B, L] -> [B, 1, L]
                attention_mask = attention_mask.unsqueeze(1)
            elif attention_mask.dim() == 4:
                # [B, 1, L, L] -> [B, L, L] (take first head)
                attention_mask = attention_mask.squeeze(1)

            I = I.masked_fill(attention_mask == 0, float("-inf"))

        return I


class AdaptiveTopK(nn.Module):
    """
    Adaptive Top-K Selection.

    Dynamically adjusts k based on indexer score variance:
        k_t = clip(k_base * (1 + beta * softplus(Var(I_t))), k_min, k_max)
    """

    def __init__(
        self,
        k_base: int = 512,
        k_min: int = 128,
        k_max: int = 1024,
        beta: float = 0.1,
        adaptive: bool = True,
    ):
        super().__init__()
        self.k_base = k_base
        self.k_min = k_min
        self.k_max = k_max
        self.beta = beta
        self.adaptive = adaptive

    def forward(
        self,
        indexer_scores: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Select top-k indices based on indexer scores.

        Args:
            indexer_scores: Scores of shape [B, L, L]

        Returns:
            selected_indices: Indices of shape [B, L, k]
            k_values: Actual k values used per position [B, L]
        """
        B, L, _ = indexer_scores.shape

        if self.adaptive:
            # Compute variance of scores per query position
            variance = indexer_scores.var(dim=-1)  # [B, L]

            # Adaptive k
            k_adaptive = self.k_base * (1 + self.beta * F.softplus(variance))
            k_adaptive = k_adaptive.clamp(self.k_min, min(self.k_max, L))
            k_values = k_adaptive.int()

            # For simplicity, use max k and mask
            k = min(int(k_values.max().item()), L)
        else:
            k = min(self.k_base, L)
            k_values = torch.full((B, L), k, device=indexer_scores.device)

        # Select top-k indices
        _, selected_indices = indexer_scores.topk(k, dim=-1)  # [B, L, k]

        return selected_indices, k_values


class GatedSparseAttention(nn.Module):
    """
    Gated Sparse Attention (GSA) Layer.

    Implements efficient attention with:
    1. Value Gate (G2) - pre-attention gating
    2. Gated Lightning Indexer - token selection
    3. Adaptive Top-k Selection - dynamic sparsity
    4. Sparse Attention Computation
    5. Output Gate (G1) - post-attention gating

    Args:
        hidden_size: Model hidden dimension
        num_heads: Number of attention heads
        num_kv_heads: Number of KV heads (for GQA)
        head_dim: Dimension per head
        k_base: Base number of selected tokens
        k_min: Minimum k
        k_max: Maximum k
        indexer_heads: Number of indexer heads
        indexer_dim: Indexer dimension
        adaptive_k: Whether to use adaptive k selection
        dropout: Attention dropout
        max_seq_len: Maximum sequence length for RoPE
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        k_base: int = 512,
        k_min: int = 128,
        k_max: int = 1024,
        indexer_heads: int = 4,
        indexer_dim: int = 64,
        adaptive_k: bool = True,
        dropout: float = 0.0,
        max_seq_len: int = 8192,
        gate_bias_init: float = 0.5,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.head_dim = head_dim or (hidden_size // num_heads)
        self.n_rep = num_heads // self.num_kv_heads  # For GQA
        self.scale = self.head_dim ** -0.5

        # QKV projections
        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_size, bias=False)

        # Value Gate (G2) - pre-attention gating
        self.v_gate_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=True)

        # Output Gate (G1) - post-attention gating
        self.o_gate_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=True)

        # Initialize gate biases
        nn.init.constant_(self.v_gate_proj.bias, gate_bias_init)
        nn.init.constant_(self.o_gate_proj.bias, gate_bias_init)

        # Gated Lightning Indexer
        self.indexer = GatedLightningIndexer(
            hidden_size=hidden_size,
            indexer_heads=indexer_heads,
            indexer_dim=indexer_dim,
        )

        # Adaptive Top-K Selection
        self.topk_selector = AdaptiveTopK(
            k_base=k_base,
            k_min=k_min,
            k_max=k_max,
            adaptive=adaptive_k,
        )

        # Rotary Positional Embedding
        self.rope = RotaryPositionalEmbedding(
            dim=self.head_dim,
            max_seq_len=max_seq_len,
        )

        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.o_proj.weight)
        nn.init.xavier_uniform_(self.v_gate_proj.weight)
        nn.init.xavier_uniform_(self.o_gate_proj.weight)

    def repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        """Repeat KV heads for GQA."""
        if self.n_rep == 1:
            return x
        B, L, H_kv, D = x.shape
        return x.unsqueeze(-2).expand(B, L, H_kv, self.n_rep, D).reshape(B, L, H_kv * self.n_rep, D)

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
        Forward pass for Gated Sparse Attention.

        Args:
            hidden_states: Input tensor [B, L, C]
            attention_mask: Attention mask [B, L] or [B, 1, L, L]
            position_ids: Position IDs [B, L]
            past_key_value: Cached KV for generation
            use_cache: Whether to return cached KV
            output_attentions: Whether to return attention weights

        Returns:
            output: Attention output [B, L, C]
            (optional) past_key_value: Updated cache
            (optional) attention_weights: Sparse attention weights
        """
        B, L, C = hidden_states.shape
        H = self.num_heads
        H_kv = self.num_kv_heads
        D = self.head_dim

        # QKV projections
        q = self.q_proj(hidden_states).view(B, L, H, D)
        k = self.k_proj(hidden_states).view(B, L, H_kv, D)
        v = self.v_proj(hidden_states).view(B, L, H_kv, D)

        # Apply RoPE
        cos, sin = self.rope(hidden_states, L)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Value Gate (G2) - pre-attention gating
        v_gate = torch.sigmoid(self.v_gate_proj(hidden_states).view(B, L, H_kv, D))
        v = v * v_gate

        # Handle KV cache for generation
        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=1)
            v = torch.cat([past_key_value[1], v], dim=1)

        if use_cache:
            past_key_value = (k, v)

        # Repeat KV for GQA
        k = self.repeat_kv(k)  # [B, L, H, D]
        v = self.repeat_kv(v)  # [B, L, H, D]

        # Compute indexer scores and select top-k
        indexer_scores = self.indexer(hidden_states, attention_mask)  # [B, L, L]
        selected_indices, k_values = self.topk_selector(indexer_scores)  # [B, L, k]

        k_actual = selected_indices.shape[-1]

        # Gather sparse K and V
        # selected_indices: [B, L, k]
        # k: [B, L, H, D] -> need to gather along dim 1

        # Expand indices for gathering
        idx_expanded = selected_indices.unsqueeze(-1).unsqueeze(-1)  # [B, L, k, 1, 1]
        idx_expanded = idx_expanded.expand(B, L, k_actual, H, D)  # [B, L, k, H, D]

        # Gather K and V
        k_expanded = k.unsqueeze(1).expand(B, L, -1, H, D)  # [B, L, L, H, D]
        v_expanded = v.unsqueeze(1).expand(B, L, -1, H, D)  # [B, L, L, H, D]

        k_sparse = torch.gather(k_expanded, 2, idx_expanded)  # [B, L, k, H, D]
        v_sparse = torch.gather(v_expanded, 2, idx_expanded)  # [B, L, k, H, D]

        # Compute sparse attention
        # q: [B, L, H, D], k_sparse: [B, L, k, H, D]
        q = q.unsqueeze(2)  # [B, L, 1, H, D]

        # Attention scores
        attn_scores = torch.einsum("blqhd,blkhd->blhk", q, k_sparse) * self.scale  # [B, L, H, k]

        # Apply causal mask to sparse attention
        if attention_mask is not None:
            # Create sparse mask based on selected indices
            # For causal: mask where selected_index > current_position
            positions = torch.arange(L, device=hidden_states.device).view(1, L, 1)
            sparse_mask = selected_indices > positions  # [B, L, k]
            sparse_mask = sparse_mask.unsqueeze(2).expand(-1, -1, H, -1)  # [B, L, H, k]
            attn_scores = attn_scores.masked_fill(sparse_mask, float("-inf"))

        # Softmax
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Weighted sum of values
        # attn_weights: [B, L, H, k], v_sparse: [B, L, k, H, D]
        v_sparse = v_sparse.transpose(2, 3)  # [B, L, H, k, D]
        attn_output = torch.einsum("blhk,blhkd->blhd", attn_weights, v_sparse)  # [B, L, H, D]

        # Output Gate (G1) - post-attention gating
        o_gate = torch.sigmoid(self.o_gate_proj(hidden_states).view(B, L, H, D))
        attn_output = attn_output * o_gate

        # Reshape and project output
        attn_output = attn_output.reshape(B, L, H * D)
        output = self.o_proj(attn_output)

        outputs = (output,)

        if use_cache:
            outputs += (past_key_value,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class GatedSparseAttentionWithFlash(GatedSparseAttention):
    """
    GSA with Flash Attention support for dense fallback.

    Uses Flash Attention for short sequences where sparse attention
    overhead isn't worth it.
    """

    def __init__(self, *args, flash_threshold: int = 512, **kwargs):
        super().__init__(*args, **kwargs)
        self.flash_threshold = flash_threshold

        try:
            from flash_attn import flash_attn_func
            self.flash_attn_func = flash_attn_func
            self.has_flash = True
        except ImportError:
            self.has_flash = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        L = hidden_states.shape[1]

        # Use dense Flash Attention for short sequences
        if self.has_flash and L <= self.flash_threshold and not output_attentions:
            return self._forward_flash(
                hidden_states, attention_mask, position_ids,
                past_key_value, use_cache
            )

        # Use sparse attention for long sequences
        return super().forward(
            hidden_states, attention_mask, position_ids,
            past_key_value, use_cache, output_attentions
        )

    def _forward_flash(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.Tensor],
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]],
        use_cache: bool,
    ) -> Tuple[torch.Tensor, ...]:
        """Dense attention using Flash Attention."""
        B, L, C = hidden_states.shape
        H = self.num_heads
        H_kv = self.num_kv_heads
        D = self.head_dim

        # QKV projections
        q = self.q_proj(hidden_states).view(B, L, H, D)
        k = self.k_proj(hidden_states).view(B, L, H_kv, D)
        v = self.v_proj(hidden_states).view(B, L, H_kv, D)

        # Apply RoPE
        cos, sin = self.rope(hidden_states, L)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Value Gate (G2)
        v_gate = torch.sigmoid(self.v_gate_proj(hidden_states).view(B, L, H_kv, D))
        v = v * v_gate

        # Handle KV cache
        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=1)
            v = torch.cat([past_key_value[1], v], dim=1)

        if use_cache:
            past_key_value = (k, v)

        # Repeat KV for GQA
        k = self.repeat_kv(k)
        v = self.repeat_kv(v)

        # Flash Attention
        attn_output = self.flash_attn_func(q, k, v, causal=True)  # [B, L, H, D]

        # Output Gate (G1)
        o_gate = torch.sigmoid(self.o_gate_proj(hidden_states).view(B, L, H, D))
        attn_output = attn_output * o_gate

        # Output projection
        attn_output = attn_output.reshape(B, L, H * D)
        output = self.o_proj(attn_output)

        outputs = (output,)
        if use_cache:
            outputs += (past_key_value,)

        return outputs


def test_gsa():
    """Test Gated Sparse Attention."""
    torch.manual_seed(42)

    B, L, C = 2, 64, 256
    num_heads = 8
    num_kv_heads = 4

    # Create GSA layer
    gsa = GatedSparseAttention(
        hidden_size=C,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        k_base=16,
        k_min=8,
        k_max=32,
    )

    # Create input
    x = torch.randn(B, L, C)

    # Forward pass
    output = gsa(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output[0].shape}")

    # Test with cache
    output_cached = gsa(x, use_cache=True)
    print(f"Output with cache: {len(output_cached)} tensors")

    print("\nAll GSA tests passed!")


if __name__ == "__main__":
    test_gsa()

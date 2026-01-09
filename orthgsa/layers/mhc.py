"""
Manifold-Constrained Hyper-Connections (mHC)

Implements n-stream residual connections with:
- Pre-mapping: Aggregates n streams to single stream
- Post-mapping: Distributes single stream back to n streams
- Residual mapping: Mixes streams via orthogonal transformation (Cayley)

Key equations:
    x_{l+1} = H^res @ x_l + H^post^T @ F(H^pre @ x_l)

where:
    H^pre ∈ R^{1×n}: Pre-mapping coefficients (sigmoid)
    H^post ∈ R^{1×n}: Post-mapping coefficients (2×sigmoid)
    H^res ∈ O(n): Residual mapping (orthogonal via Cayley transform)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math

from .cayley import cayley_transform


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # RMS normalization
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        x_norm = x / rms
        return x_norm * self.weight


class ManifoldHyperConnection(nn.Module):
    """
    Manifold-Constrained Hyper-Connection Layer.

    This layer wraps a sub-layer (e.g., attention or FFN) with n-stream
    residual connections using Cayley transform for orthogonal mixing.

    Args:
        hidden_size: Model hidden dimension (C)
        n_streams: Number of residual streams (n)
        alpha_init: Initial scaling for learned coefficients
        eps: Epsilon for numerical stability
    """

    def __init__(
        self,
        hidden_size: int,
        n_streams: int = 4,
        alpha_init: float = 0.01,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_streams = n_streams
        self.eps = eps

        # Input dimension for coefficient computation
        input_dim = n_streams * hidden_size

        # RMSNorm for normalizing flattened input
        self.norm = RMSNorm(input_dim, eps=eps)

        # Learnable projections for computing coefficients
        # phi_pre: [nC, n] -> produces H_pre
        # phi_post: [nC, n] -> produces H_post
        # phi_res: [nC, n*n] -> produces H_res (before Cayley)
        self.phi_pre = nn.Linear(input_dim, n_streams, bias=False)
        self.phi_post = nn.Linear(input_dim, n_streams, bias=False)
        self.phi_res = nn.Linear(input_dim, n_streams * n_streams, bias=False)

        # Bias terms for coefficients
        self.b_pre = nn.Parameter(torch.ones(n_streams) / n_streams)  # Uniform
        self.b_post = nn.Parameter(torch.ones(n_streams))  # Uniform distribution
        self.b_res = nn.Parameter(torch.zeros(n_streams, n_streams))  # Identity via Cayley

        # Scaling factors (alpha)
        self.alpha_pre = nn.Parameter(torch.tensor(alpha_init))
        self.alpha_post = nn.Parameter(torch.tensor(alpha_init))
        self.alpha_res = nn.Parameter(torch.tensor(alpha_init))

        # Initialize projections to zero for identity-like behavior at start
        self._init_weights()

    def _init_weights(self):
        """Initialize weights for near-identity behavior."""
        nn.init.zeros_(self.phi_pre.weight)
        nn.init.zeros_(self.phi_post.weight)
        nn.init.zeros_(self.phi_res.weight)

    def compute_coefficients(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute mHC coefficients from input.

        Args:
            x: Input tensor of shape [B, L, n, C] (n-stream hidden state)

        Returns:
            H_pre: [B, L, 1, n] - Pre-mapping coefficients
            H_post: [B, L, n, 1] - Post-mapping coefficients
            H_res: [B, L, n, n] - Residual mapping (orthogonal)
        """
        B, L, n, C = x.shape

        # Flatten streams: [B, L, n, C] -> [B*L, n*C]
        x_flat = x.reshape(B * L, n * C)

        # Normalize
        x_norm = self.norm(x_flat)

        # Compute raw coefficients
        H_pre_raw = self.alpha_pre * self.phi_pre(x_norm) + self.b_pre  # [B*L, n]
        H_post_raw = self.alpha_post * self.phi_post(x_norm) + self.b_post  # [B*L, n]
        H_res_raw = self.alpha_res * self.phi_res(x_norm).view(B * L, n, n) + self.b_res  # [B*L, n, n]

        # Apply manifold projections
        H_pre = torch.sigmoid(H_pre_raw)  # [B*L, n] -> non-negative
        H_post = 2.0 * torch.sigmoid(H_post_raw)  # [B*L, n] -> scaled for amplification
        H_res = cayley_transform(H_res_raw)  # [B*L, n, n] -> orthogonal

        # Reshape for batch operations
        H_pre = H_pre.view(B, L, 1, n)  # [B, L, 1, n]
        H_post = H_post.view(B, L, n, 1)  # [B, L, n, 1]
        H_res = H_res.view(B, L, n, n)  # [B, L, n, n]

        return H_pre, H_post, H_res

    def pre_mapping(self, x: torch.Tensor, H_pre: torch.Tensor) -> torch.Tensor:
        """
        Aggregate n streams to single stream.

        Args:
            x: Input tensor of shape [B, L, n, C]
            H_pre: Pre-mapping coefficients [B, L, 1, n]

        Returns:
            y: Aggregated tensor of shape [B, L, C]
        """
        # y = H_pre @ x: [B, L, 1, n] @ [B, L, n, C] -> [B, L, 1, C]
        y = torch.matmul(H_pre, x)  # [B, L, 1, C]
        return y.squeeze(-2)  # [B, L, C]

    def post_mapping(self, z: torch.Tensor, H_post: torch.Tensor) -> torch.Tensor:
        """
        Distribute single stream back to n streams.

        Args:
            z: Sub-layer output of shape [B, L, C]
            H_post: Post-mapping coefficients [B, L, n, 1]

        Returns:
            o: Distributed tensor of shape [B, L, n, C]
        """
        # o = H_post @ z: [B, L, n, 1] @ [B, L, 1, C] -> [B, L, n, C]
        z_expanded = z.unsqueeze(-2)  # [B, L, 1, C]
        o = H_post * z_expanded  # Broadcasting: [B, L, n, C]
        return o

    def residual_mapping(self, x: torch.Tensor, H_res: torch.Tensor) -> torch.Tensor:
        """
        Mix streams via orthogonal transformation.

        Args:
            x: Input tensor of shape [B, L, n, C]
            H_res: Residual mapping (orthogonal) [B, L, n, n]

        Returns:
            r: Mixed tensor of shape [B, L, n, C]
        """
        # r = H_res @ x: [B, L, n, n] @ [B, L, n, C] -> [B, L, n, C]
        r = torch.matmul(H_res, x)
        return r

    def forward(
        self,
        x: torch.Tensor,
        sublayer_fn: callable,
        **sublayer_kwargs,
    ) -> torch.Tensor:
        """
        Apply mHC-wrapped sub-layer.

        Args:
            x: Input tensor of shape [B, L, n, C] (n-stream hidden state)
            sublayer_fn: Sub-layer function (attention or FFN)
            **sublayer_kwargs: Additional arguments for sublayer

        Returns:
            Output tensor of shape [B, L, n, C]
        """
        # Compute mHC coefficients
        H_pre, H_post, H_res = self.compute_coefficients(x)

        # Pre-mapping: aggregate n streams to single stream
        y = self.pre_mapping(x, H_pre)  # [B, L, C]

        # Apply sub-layer
        z = sublayer_fn(y, **sublayer_kwargs)  # [B, L, C]

        # Post-mapping: distribute to n streams
        o = self.post_mapping(z, H_post)  # [B, L, n, C]

        # Residual mapping: mix streams orthogonally
        r = self.residual_mapping(x, H_res)  # [B, L, n, C]

        # Combine: x_{l+1} = r + o
        return r + o

    def expand_to_streams(self, x: torch.Tensor) -> torch.Tensor:
        """
        Expand single-stream tensor to n-stream format.

        Args:
            x: Input tensor of shape [B, L, C]

        Returns:
            Expanded tensor of shape [B, L, n, C]
        """
        # Option 1: Tile (all streams start the same)
        return x.unsqueeze(-2).expand(-1, -1, self.n_streams, -1).clone()

    def collapse_from_streams(self, x: torch.Tensor, method: str = "mean") -> torch.Tensor:
        """
        Collapse n-stream tensor to single-stream format.

        Args:
            x: Input tensor of shape [B, L, n, C]
            method: Collapse method ("mean", "first", "last", "sum")

        Returns:
            Collapsed tensor of shape [B, L, C]
        """
        if method == "mean":
            return x.mean(dim=-2)
        elif method == "first":
            return x[..., 0, :]
        elif method == "last":
            return x[..., -1, :]
        elif method == "sum":
            return x.sum(dim=-2)
        else:
            raise ValueError(f"Unknown collapse method: {method}")


class ManifoldHyperConnectionBlock(nn.Module):
    """
    Complete mHC block with both attention and FFN sub-layers.

    This is a convenience wrapper that applies mHC to both attention and FFN.
    """

    def __init__(
        self,
        hidden_size: int,
        n_streams: int = 4,
        alpha_init: float = 0.01,
        eps: float = 1e-6,
    ):
        super().__init__()

        # Separate mHC for attention and FFN
        self.mhc_attn = ManifoldHyperConnection(
            hidden_size=hidden_size,
            n_streams=n_streams,
            alpha_init=alpha_init,
            eps=eps,
        )

        self.mhc_ffn = ManifoldHyperConnection(
            hidden_size=hidden_size,
            n_streams=n_streams,
            alpha_init=alpha_init,
            eps=eps,
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_fn: callable,
        ffn_fn: callable,
        **kwargs,
    ) -> torch.Tensor:
        """
        Apply mHC-wrapped attention and FFN.

        Args:
            x: Input tensor of shape [B, L, n, C]
            attention_fn: Attention function
            ffn_fn: FFN function
            **kwargs: Additional arguments for attention

        Returns:
            Output tensor of shape [B, L, n, C]
        """
        # Attention sub-block
        x = self.mhc_attn(x, attention_fn, **kwargs)

        # FFN sub-block
        x = self.mhc_ffn(x, ffn_fn)

        return x


def test_mhc():
    """Test ManifoldHyperConnection."""
    torch.manual_seed(42)

    B, L, n, C = 2, 16, 4, 64
    hidden_size = C
    n_streams = n

    # Create mHC layer
    mhc = ManifoldHyperConnection(hidden_size=hidden_size, n_streams=n_streams)

    # Create n-stream input
    x = torch.randn(B, L, n, C)

    # Define a simple sublayer
    def simple_sublayer(y):
        return y * 0.5  # Just scale for testing

    # Forward pass
    output = mhc(x, simple_sublayer)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    assert output.shape == x.shape, "Shape mismatch!"

    # Test coefficient computation
    H_pre, H_post, H_res = mhc.compute_coefficients(x)
    print(f"H_pre shape: {H_pre.shape}")
    print(f"H_post shape: {H_post.shape}")
    print(f"H_res shape: {H_res.shape}")

    # Verify H_res is orthogonal
    H_res_flat = H_res.view(-1, n, n)
    I = torch.eye(n)
    ortho_error = torch.norm(H_res_flat @ H_res_flat.transpose(-2, -1) - I, dim=(-2, -1)).max()
    print(f"Max orthogonality error in H_res: {ortho_error.item():.2e}")

    # Test expand/collapse
    single_stream = torch.randn(B, L, C)
    expanded = mhc.expand_to_streams(single_stream)
    collapsed = mhc.collapse_from_streams(expanded, method="mean")
    print(f"Expanded shape: {expanded.shape}")
    print(f"Collapsed shape: {collapsed.shape}")

    print("\nAll mHC tests passed!")


if __name__ == "__main__":
    test_mhc()

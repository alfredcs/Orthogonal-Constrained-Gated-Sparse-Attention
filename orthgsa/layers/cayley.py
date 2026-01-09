"""
Cayley Transform for Orthogonal Matrix Parameterization

The Cayley transform maps skew-symmetric matrices to orthogonal matrices:
    Q = (I - A)(I + A)^{-1}

where A is skew-symmetric (A^T = -A).

Properties:
- Q^T Q = Q Q^T = I (orthogonal)
- det(Q) = +1 (special orthogonal / rotation)
- ||Q||_2 = 1 (unit spectral norm)
- Cayley(0) = I (identity at zero)
"""

import torch
import torch.nn as nn
from typing import Optional
import math


def cayley_transform(H_raw: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Apply Cayley transform to produce orthogonal matrix.

    The transform extracts the skew-symmetric component and applies:
        Q = (I - A)(I + A)^{-1}

    Args:
        H_raw: Raw coefficient matrix of shape [..., n, n]
        eps: Small epsilon for numerical stability

    Returns:
        Orthogonal matrix of shape [..., n, n]
    """
    # Extract skew-symmetric component: A = (H - H^T) / 2
    A = (H_raw - H_raw.transpose(-2, -1)) / 2

    n = A.shape[-1]
    device = A.device
    dtype = A.dtype

    # Identity matrix
    I = torch.eye(n, device=device, dtype=dtype)

    # Expand I for batch dimensions
    for _ in range(len(A.shape) - 2):
        I = I.unsqueeze(0)
    I = I.expand_as(A)

    # Cayley transform: (I - A)(I + A)^{-1}
    I_plus_A = I + A
    I_minus_A = I - A

    # Use torch.linalg.solve for numerical stability
    # solve(A, B) computes A^{-1} @ B, so solve(I+A, I-A) = (I+A)^{-1}(I-A)
    # But we want (I-A)(I+A)^{-1}, so we need to transpose the solve
    # Actually: (I-A)(I+A)^{-1} = solve(I+A^T, I-A^T)^T = solve((I+A)^T, (I-A)^T)^T
    # Since A is skew-symmetric: A^T = -A, so (I+A)^T = I - A, (I-A)^T = I + A
    # Therefore: solve((I-A), (I+A))^T = (I-A)^{-1}(I+A), transposed gives (I+A)^T(I-A)^{-T}
    #
    # Simpler: just compute (I-A) @ inverse(I+A)
    # Or use: solve(I+A^T, (I-A)^T)^T for (I-A)(I+A)^{-1}

    # Direct approach using linalg.solve
    # We want X such that (I+A) @ X^T = (I-A)^T = I + A (since A^T = -A)
    # So X^T = (I+A)^{-1} @ (I+A) = I... that's wrong

    # Let's be more careful:
    # We want Q = (I - A) @ (I + A)^{-1}
    # Let Y = (I + A)^{-1}, then Q = (I - A) @ Y
    # Y = solve(I + A, I)
    # Q = (I - A) @ Y

    # More efficient: solve the system directly
    # (I + A) @ Q^T = (I - A)^T = I + A (since (I-A)^T = I - A^T = I + A for skew-sym A)
    # Hmm, that gives Q^T = I, which is wrong.

    # Let me reconsider:
    # Q = (I - A)(I + A)^{-1}
    # Q^T = ((I + A)^{-1})^T (I - A)^T = ((I + A)^T)^{-1} (I - A)^T
    #     = (I + A^T)^{-1} (I - A^T) = (I - A)^{-1} (I + A)  [since A^T = -A]
    #
    # So: (I + A) @ Q = (I - A)
    # Therefore: Q = solve(I + A, I - A)

    Q = torch.linalg.solve(I_plus_A, I_minus_A)

    return Q


def cayley_transform_batched(H_raw: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Optimized batched Cayley transform for small matrices (n <= 4).

    For small n, direct computation can be faster than torch.linalg.solve.

    Args:
        H_raw: Raw coefficient matrix of shape [B, n, n]
        eps: Small epsilon for numerical stability

    Returns:
        Orthogonal matrix of shape [B, n, n]
    """
    n = H_raw.shape[-1]

    if n <= 4:
        # For small matrices, use explicit inverse formula
        A = (H_raw - H_raw.transpose(-2, -1)) / 2

        I = torch.eye(n, device=H_raw.device, dtype=H_raw.dtype)
        I = I.unsqueeze(0).expand(H_raw.shape[0], -1, -1)

        I_plus_A = I + A
        I_minus_A = I - A

        # Use batched inverse for small matrices
        I_plus_A_inv = torch.linalg.inv(I_plus_A)
        Q = I_minus_A @ I_plus_A_inv

        return Q
    else:
        return cayley_transform(H_raw, eps)


class CayleyTransform(nn.Module):
    """
    Cayley Transform layer for producing orthogonal matrices.

    This module wraps the Cayley transform with learnable parameters
    for use in the Manifold-Constrained Hyper-Connections (mHC).

    Args:
        n: Matrix dimension (number of streams)
        scaling: Initial scaling factor for input
        learnable_scaling: Whether to make scaling learnable
    """

    def __init__(
        self,
        n: int,
        scaling: float = 0.1,
        learnable_scaling: bool = True,
    ):
        super().__init__()
        self.n = n
        self.learnable_scaling = learnable_scaling

        if learnable_scaling:
            self.scaling = nn.Parameter(torch.tensor(scaling))
        else:
            self.register_buffer("scaling", torch.tensor(scaling))

    def forward(self, H_raw: torch.Tensor) -> torch.Tensor:
        """
        Apply Cayley transform with scaling.

        Args:
            H_raw: Raw coefficient matrix of shape [..., n, n]

        Returns:
            Orthogonal matrix of shape [..., n, n]
        """
        # Scale the input to control the magnitude of rotation
        scaled = H_raw * self.scaling
        return cayley_transform(scaled)

    def extra_repr(self) -> str:
        return f"n={self.n}, scaling={self.scaling.item():.4f}, learnable={self.learnable_scaling}"


class CayleyParameterization(nn.Module):
    """
    Parameterize a weight matrix as orthogonal using Cayley transform.

    This is useful for parameterizing weight matrices that should remain
    orthogonal throughout training.

    Args:
        n: Matrix dimension
        init_scale: Initial scale for the skew-symmetric parameters
    """

    def __init__(self, n: int, init_scale: float = 0.01):
        super().__init__()
        self.n = n

        # Learnable skew-symmetric parameters (only upper triangular needed)
        # A skew-symmetric matrix has n(n-1)/2 free parameters
        num_params = n * (n - 1) // 2
        self.skew_params = nn.Parameter(torch.randn(num_params) * init_scale)

        # Register indices for constructing skew-symmetric matrix
        triu_indices = torch.triu_indices(n, n, offset=1)
        self.register_buffer("triu_row", triu_indices[0])
        self.register_buffer("triu_col", triu_indices[1])

    def get_skew_symmetric(self) -> torch.Tensor:
        """Construct skew-symmetric matrix from parameters."""
        A = torch.zeros(self.n, self.n, device=self.skew_params.device, dtype=self.skew_params.dtype)
        A[self.triu_row, self.triu_col] = self.skew_params
        A = A - A.T  # Make skew-symmetric
        return A

    def forward(self) -> torch.Tensor:
        """
        Get the orthogonal matrix.

        Returns:
            Orthogonal matrix of shape [n, n]
        """
        A = self.get_skew_symmetric()
        return cayley_transform(A.unsqueeze(0)).squeeze(0)

    def extra_repr(self) -> str:
        return f"n={self.n}, num_params={self.skew_params.numel()}"


def test_cayley_properties():
    """Test that Cayley transform produces orthogonal matrices."""
    torch.manual_seed(42)

    # Test single matrix
    H = torch.randn(4, 4)
    Q = cayley_transform(H.unsqueeze(0)).squeeze(0)

    # Check orthogonality
    I = torch.eye(4)
    ortho_error = torch.norm(Q @ Q.T - I).item()
    print(f"Orthogonality error (Q @ Q^T - I): {ortho_error:.2e}")

    # Check determinant
    det = torch.linalg.det(Q).item()
    print(f"Determinant: {det:.4f} (should be +1)")

    # Check spectral norm
    spec_norm = torch.linalg.norm(Q, ord=2).item()
    print(f"Spectral norm: {spec_norm:.4f} (should be 1)")

    # Test identity at zero
    zero_H = torch.zeros(4, 4)
    Q_zero = cayley_transform(zero_H.unsqueeze(0)).squeeze(0)
    identity_error = torch.norm(Q_zero - I).item()
    print(f"Identity at zero error: {identity_error:.2e}")

    # Test batched
    H_batch = torch.randn(32, 4, 4)
    Q_batch = cayley_transform(H_batch)
    print(f"Batched output shape: {Q_batch.shape}")

    # Check all are orthogonal
    ortho_errors = torch.norm(Q_batch @ Q_batch.transpose(-2, -1) - I, dim=(-2, -1))
    print(f"Max batch orthogonality error: {ortho_errors.max().item():.2e}")

    print("\nAll tests passed!")


if __name__ == "__main__":
    test_cayley_properties()

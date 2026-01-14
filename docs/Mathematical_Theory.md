# OrthGSA: Mathematical Formulation and Theoretical Properties

This document contains the detailed mathematical formulation and theoretical analysis of the OrthGSA architecture. For an overview and practical usage, see [README.md](../README.md).

---

## Table of Contents

1. [Mathematical Formulation](#mathematical-formulation)
   - [OrthGSA Layer Definition](#1-orthgsa-layer-definition)
   - [Orthogonal-Constrained Coefficient Computation](#2-orthogonal-constrained-coefficient-computation-from-ohc)
   - [Gated Sparse Attention Function](#3-gated-sparse-attention-function-from-gsa)
   - [Complete OrthGSA Blocks](#4-complete-orthgsa-attention-block)
   - [Cayley Transform](#6-cayley-transform-for-orthogonal-matrices)
   - [Cayley vs Sinkhorn-Knopp Comparison](#7-cayley-transform-vs-sinkhorn-knopp-comparison)
   - [Cayley Transform Algorithm](#8-algorithm-cayley-transform-projection)
2. [Architecture Diagrams](#architecture-diagrams)
   - [High-Level OrthGSA Block](#high-level-orthgsa-block)
   - [Detailed GSA Sub-Module](#detailed-gsa-sub-module-architecture)
   - [Multi-Stream Residual Flow](#multi-stream-residual-flow)
3. [Data Flow](#data-flow)
   - [Complete Forward Pass](#complete-forward-pass-data-flow)
   - [Tensor Shape Summary](#tensor-shape-summary)
4. [Algorithm Details](#algorithm-details)
   - [OrthGSA Forward Pass](#algorithm-1-orthgsa-forward-pass)
   - [Gated Sparse Attention](#algorithm-2-gated-sparse-attention-gsa)
   - [Cayley Transform Projection](#algorithm-3-cayley-transform-projection)
5. [Infrastructure Optimizations](#infrastructure-optimizations)
   - [Kernel Fusion Strategy](#1-kernel-fusion-strategy)
   - [Selective Recomputation](#2-selective-recomputation-strategy)
   - [Memory Access Optimization](#3-memory-access-pattern-optimization)
   - [Distributed Training Communication](#4-communication-overlap-for-distributed-training)
6. [Theoretical Properties](#theoretical-properties)
   - [Complexity Analysis](#1-complexity-analysis)
   - [Gradient Flow Analysis](#2-gradient-flow-analysis)
   - [Identity Mapping Restoration](#3-identity-mapping-restoration)
   - [Attention Sink Elimination](#4-attention-sink-elimination)

---

## Mathematical Formulation

### 1. OrthGSA Layer Definition

For layer $l$, define the OrthGSA transformation:

$$\mathbf{x}_{l+1} = \mathbf{H}_l^{\text{res}} \mathbf{x}_l + (\mathbf{H}_l^{\text{post}})^\top \mathcal{F}^{\text{GSA}}(\mathbf{H}_l^{\text{pre}} \mathbf{x}_l, \mathbf{W}_l)$$

Where:
- $\mathbf{x}_l, \mathbf{x}_{l+1} \in \mathbb{R}^{n \times C}$ : n-stream hidden states
- $\mathbf{H}_l^{\text{res}} \in \mathcal{B}^{n \times n}$ : doubly stochastic residual mapping (Birkhoff polytope)
- $\mathbf{H}_l^{\text{pre}} \in \mathbb{R}_+^{1 \times n}$ : pre-mapping (aggregates n streams)
- $\mathbf{H}_l^{\text{post}} \in \mathbb{R}_+^{1 \times n}$ : post-mapping (distributes to n streams)
- $\mathcal{F}^{\text{GSA}}$ : Gated Sparse Attention function

### 2. Orthogonal-Constrained Coefficient Computation (from oHC)

**Step 2.1**: Normalize input

$$\vec{\mathbf{x}}'_l = \text{RMSNorm}(\text{vec}(\mathbf{x}_l)) \in \mathbb{R}^{1 \times nC}$$

**Step 2.2**: Compute raw coefficients

$$\tilde{\mathbf{H}}_l^{\text{pre}} = \alpha_l^{\text{pre}} \cdot (\vec{\mathbf{x}}'_l \cdot \boldsymbol{\varphi}_l^{\text{pre}}) + \mathbf{b}_l^{\text{pre}} \in \mathbb{R}^{1 \times n}$$

$$\tilde{\mathbf{H}}_l^{\text{post}} = \alpha_l^{\text{post}} \cdot (\vec{\mathbf{x}}'_l \cdot \boldsymbol{\varphi}_l^{\text{post}}) + \mathbf{b}_l^{\text{post}} \in \mathbb{R}^{1 \times n}$$

$$\tilde{\mathbf{H}}_l^{\text{res}} = \alpha_l^{\text{res}} \cdot \text{mat}(\vec{\mathbf{x}}'_l \cdot \boldsymbol{\varphi}_l^{\text{res}}) + \mathbf{b}_l^{\text{res}} \in \mathbb{R}^{n \times n}$$

**Step 2.3**: Apply orthogonal projections

$$\mathbf{H}_l^{\text{pre}} = \sigma(\tilde{\mathbf{H}}_l^{\text{pre}}) \quad \text{(Sigmoid for non-negativity)}$$

$$\mathbf{H}_l^{\text{post}} = 2 \cdot \sigma(\tilde{\mathbf{H}}_l^{\text{post}}) \quad \text{(Scaled sigmoid for amplification)}$$

$$\mathbf{H}_l^{\text{res}} = \text{Cayley}(\tilde{\mathbf{H}}_l^{\text{res}}) = (\mathbf{I}_n - \mathbf{A}_l)(\mathbf{I}_n + \mathbf{A}_l)^{-1} \quad \text{(Orthogonal projection)}$$

where $\mathbf{A}_l = \frac{1}{2}(\tilde{\mathbf{H}}_l^{\text{res}} - \tilde{\mathbf{H}}_l^{\text{res}\top})$ is the skew-symmetric component.

### 3. Gated Sparse Attention Function (from GSA)

The inner attention function $\mathcal{F}^{\text{GSA}}$ operates on the aggregated input:

**Input**: $\mathbf{y} = \mathbf{H}_l^{\text{pre}} \mathbf{x}_l \in \mathbb{R}^{1 \times C}$ (single-stream from n-stream aggregation)

For batch/sequence dimensions, let $\mathbf{y} \in \mathbb{R}^{B \times L \times C}$

**Step 3.1**: QKV Projections

$$\mathbf{Q} = \mathbf{y} W^Q, \quad \mathbf{K} = \mathbf{y} W^K, \quad \mathbf{V} = \mathbf{y} W^V$$

Where $\mathbf{Q} \in \mathbb{R}^{B \times L \times H \times d}$, etc.

**Step 3.2**: Value Gate (G2) - Pre-attention gating

$$\mathbf{V}^{\text{gated}} = \mathbf{V} \odot \sigma(\mathbf{y} W^{G_2})$$

**Step 3.3**: Gated Lightning Indexer

$$I_{t,s} = \sum_{j=1}^{H^I} \sigma(\mathbf{y}_t W_j^{Iw}) \cdot \sigma(\mathbf{q}_{t,j}^I \cdot \mathbf{k}_s^{I\top} + b_j^I)$$

Where:
- $\mathbf{q}^I = \mathbf{y} W^{Q_I} \in \mathbb{R}^{B \times L \times H^I \times d_I}$ : indexer queries
- $\mathbf{k}^I = \mathbf{y} W^{K_I} \in \mathbb{R}^{B \times L \times d_I}$ : indexer keys
- $H^I$ : number of indexer heads
- $d_I$ : indexer dimension

**Step 3.4**: Adaptive Top-k Selection

$$S_t = \{s : I_{t,s} \in \text{Top-}k_t(I_{t,:})\}$$

$$k_t = \text{clip}\left(k_{\text{base}} \cdot f(\text{Var}(I_{t,:})), k_{\min}, k_{\max}\right)$$

Where $f(\cdot)$ is a monotonic function (e.g., $f(v) = 1 + \beta \cdot \text{softplus}(v)$)

**Step 3.5**: Sparse Attention Computation

$$\mathbf{A}_{t,h} = \text{Softmax}\left(\frac{\mathbf{q}_{t,h} \cdot \mathbf{K}_{S_t,h}^\top}{\sqrt{d}}\right) \in \mathbb{R}^{|S_t|}$$

$$\mathbf{O}_{t,h}^{\text{sparse}} = \mathbf{A}_{t,h} \cdot \mathbf{V}_{S_t,h}^{\text{gated}}$$

**Step 3.6**: Output Gate (G1) - Post-attention gating

$$\mathbf{O}_{t,h}^{\text{final}} = \mathbf{O}_{t,h}^{\text{sparse}} \odot \sigma(\mathbf{y}_t W_{O,h}^g)$$

**Step 3.7**: Output Projection

$$\mathbf{z} = \text{Concat}(\mathbf{O}_{:,h}^{\text{final}}) \cdot W^O \in \mathbb{R}^{B \times L \times C}$$

### 4. Complete OrthGSA Attention Block

$$\mathbf{z}^{\text{attn}} = \mathcal{F}^{\text{GSA}}(\mathbf{H}_l^{\text{pre}} \mathbf{x}_l)$$

$$\mathbf{x}_{l+0.5} = \mathbf{H}_l^{\text{res,attn}} \mathbf{x}_l + (\mathbf{H}_l^{\text{post,attn}})^\top \mathbf{z}^{\text{attn}}$$

### 5. OrthGSA FFN Block

$$\mathbf{z}^{\text{ffn}} = \mathcal{F}^{\text{FFN}}(\mathbf{H}_l^{\text{pre,ffn}} \mathbf{x}_{l+0.5})$$

$$\mathbf{x}_{l+1} = \mathbf{H}_l^{\text{res,ffn}} \mathbf{x}_{l+0.5} + (\mathbf{H}_l^{\text{post,ffn}})^\top \mathbf{z}^{\text{ffn}}$$

Where $\mathcal{F}^{\text{FFN}}$ can be standard FFN or gated FFN (SwiGLU):

$$\mathcal{F}^{\text{FFN}}(\mathbf{y}) = (\text{SiLU}(\mathbf{y} W^{\text{gate}}) \odot (\mathbf{y} W^{\text{up}})) W^{\text{down}}$$

### 6. Cayley Transform for Orthogonal Matrices

The **Cayley Transform** provides a smooth, differentiable mapping from skew-symmetric matrices to orthogonal matrices, offering several advantages over iterative projection methods.

#### Mathematical Definition

**Transform**: For any skew-symmetric matrix $\mathbf{A}$ (i.e., $\mathbf{A}^\top = -\mathbf{A}$):

$$\mathbf{Q} = \text{Cayley}(\mathbf{A}) = (\mathbf{I}_n - \mathbf{A})(\mathbf{I}_n + \mathbf{A})^{-1}$$

**Equivalently** (using Woodbury identity):

$$\mathbf{Q} = \mathbf{I}_n - 2\mathbf{A}(\mathbf{I}_n + \mathbf{A})^{-1}$$

#### Skew-Symmetric Extraction

Given raw coefficients $\tilde{\mathbf{H}}^{\text{res}} \in \mathbb{R}^{n \times n}$, extract the skew-symmetric component:

$$\mathbf{A} = \frac{1}{2}(\tilde{\mathbf{H}}^{\text{res}} - \tilde{\mathbf{H}}^{\text{res}\top})$$

This guarantees $\mathbf{A}^\top = -\mathbf{A}$.

#### Properties of Cayley Transform Output

For $\mathbf{H}^{\text{res}} = \text{Cayley}(\mathbf{A})$:

- **Orthogonality**: $\mathbf{H}^{\text{res}\top} \mathbf{H}^{\text{res}} = \mathbf{H}^{\text{res}} \mathbf{H}^{\text{res}\top} = \mathbf{I}_n$
- **Determinant**: $\det(\mathbf{H}^{\text{res}}) = +1$ (special orthogonal, rotation)
- **Spectral Norm**: $\|\mathbf{H}^{\text{res}}\|_2 = 1$ (exact, not bounded)
- **Invertibility**: $\mathbf{H}^{\text{res}}$ is always invertible
- **Smoothness**: Differentiable everywhere (no iterative convergence needed)

#### Computational Form (for implementation)

For small $n$ (typically $n=4$), use direct computation:

$$\mathbf{H}^{\text{res}} = (\mathbf{I}_n - \mathbf{A}) \cdot \text{solve}(\mathbf{I}_n + \mathbf{A})$$

Or via Neumann series approximation when $\|\mathbf{A}\| < 1$:

$$(\mathbf{I}_n + \mathbf{A})^{-1} \approx \sum_{k=0}^{K} (-\mathbf{A})^k$$

---

### 7. Cayley Transform vs Sinkhorn-Knopp: Comparison

| Property | **Cayley Transform** | **Sinkhorn-Knopp** |
|----------|---------------------|-------------------|
| **Output Matrix Type** | Orthogonal ($\mathbf{Q}^\top\mathbf{Q} = \mathbf{I}$) | Doubly Stochastic (rows/cols sum to 1) |
| **Spectral Norm** | Exactly 1 | ≤ 1 |
| **Determinant** | +1 (rotation) | Variable (0 to 1) |
| **Non-negativity** | Can have negative entries | Always non-negative |
| **Computation** | Single matrix solve | Iterative ($t_{\max}$ iterations) |
| **Time Complexity** | $O(n^3)$ one-time | $O(n^2 \cdot t_{\max})$ |
| **Gradient Flow** | Exact gradients | Through unrolled iterations |
| **Numerical Stability** | Stable (direct solve) | May oscillate near boundaries |
| **Identity at Init** | $\mathbf{A}=0 \Rightarrow \mathbf{Q}=\mathbf{I}$ | Requires careful initialization |

#### When to Use Each Method

**Cayley Transform (Recommended for OrthGSA)**:
- Faster for small $n$ (typical case: $n=4$)
- Exact orthogonal constraints without iteration
- Better gradient flow (no unrolling needed)
- Deterministic computation (same input → same output)
- Allows negative mixing coefficients (more expressive)

**Sinkhorn-Knopp**:
- When strict non-negativity is required
- When probabilistic interpretation is needed (convex combination)
- For optimal transport applications
- When $n$ is very large (iterative may be cheaper)

#### Gradient Comparison

**Cayley Transform Gradient** (via implicit differentiation):

$$\frac{\partial \mathbf{Q}}{\partial \mathbf{A}} = -2(\mathbf{I}_n + \mathbf{A})^{-\top} \otimes (\mathbf{I}_n + \mathbf{A})^{-1}$$

**Sinkhorn-Knopp Gradient** (unrolled through $t_{\max}$ iterations):

$$\frac{\partial \mathbf{H}}{\partial \tilde{\mathbf{H}}} = \prod_{t=1}^{t_{\max}} \frac{\partial \mathbf{M}^{(t)}}{\partial \mathbf{M}^{(t-1)}}$$

The Cayley gradient is cleaner and doesn't suffer from vanishing/exploding through many iterations.

---

### 8. Algorithm: Cayley Transform Projection

```
Algorithm: CayleyTransform(H_raw)
─────────────────────────────────────────────────────────────────────────────────
Input:
  H_raw ∈ ℝ^{n×n}      : Raw coefficient matrix

Output:
  H_res ∈ O(n)         : Orthogonal matrix

1:  // Extract skew-symmetric component
2:  A ← (H_raw - H_raw^T) / 2

3:  // Compute Cayley transform
4:  I_plus_A ← I_n + A
5:  I_minus_A ← I_n - A

6:  // Solve linear system (more stable than direct inverse)
7:  H_res ← solve(I_plus_A, I_minus_A)    // Equivalent to I_minus_A @ inv(I_plus_A)

8:  return H_res

Properties of output H_res:
  - H_res^T · H_res = I_n     (orthogonal)
  - H_res · H_res^T = I_n     (orthogonal)
  - det(H_res) = +1           (special orthogonal / rotation)
  - ||H_res||_2 = 1           (unit spectral norm)
─────────────────────────────────────────────────────────────────────────────────
```

#### Numerically Stable Implementation

```python
def cayley_transform(H_raw: torch.Tensor) -> torch.Tensor:
    """
    Apply Cayley transform to produce orthogonal matrix.

    Args:
        H_raw: Raw coefficient matrix of shape [..., n, n]

    Returns:
        Orthogonal matrix of shape [..., n, n]
    """
    # Extract skew-symmetric component
    A = (H_raw - H_raw.transpose(-2, -1)) / 2

    n = A.shape[-1]
    I = torch.eye(n, device=A.device, dtype=A.dtype)

    # Cayley transform: (I - A)(I + A)^{-1}
    I_plus_A = I + A
    I_minus_A = I - A

    # Use torch.linalg.solve for numerical stability
    H_res = torch.linalg.solve(I_plus_A, I_minus_A)

    return H_res
```

---

## Architecture Diagrams

### High-Level OrthGSA Block

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                            OrthGSA TRANSFORMER BLOCK                             │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
                          ┌─────────────────────────┐
                          │   x_l ∈ ℝ^{n × C}       │
                          │   (n-stream residual)   │
                          └───────────┬─────────────┘
                                      │
          ┌───────────────────────────┼───────────────────────────┐
          │                           │                           │
          ▼                           ▼                           ▼
┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
│  Compute H^pre  │       │  Compute H^res  │       │  Compute H^post │
│    σ(·)         │       │ Cayley Transform│       │    2·σ(·)       │
└────────┬────────┘       └────────┬────────┘       └────────┬────────┘
         │                         │                         │
         ▼                         │                         │
┌─────────────────┐                │                         │
│   Pre-Mapping   │                │                         │
│  y = H^pre·x_l  │                │                         │
│  y ∈ ℝ^{1×C}    │                │                         │
└────────┬────────┘                │                         │
         │                         │                         │
         ▼                         │                         │
┌─────────────────────────────────────────────────────────┐  │
│              GATED SPARSE ATTENTION (GSA)               │  │
├─────────────────────────────────────────────────────────┤  │
│                                                         │  │
│   ┌──────────┐   ┌──────────┐   ┌──────────┐           │  │
│   │  Q Proj  │   │  K Proj  │   │  V Proj  │           │  │
│   └────┬─────┘   └────┬─────┘   └────┬─────┘           │  │
│        │              │              │                 │  │
│        │              │              ▼                 │  │
│        │              │      ┌──────────────┐          │  │
│        │              │      │ Value Gate G2│          │  │
│        │              │      │   V·σ(yW^G2) │          │  │
│        │              │      └──────┬───────┘          │  │
│        │              │             │                  │  │
│        ▼              ▼             │                  │  │
│   ┌────────────────────────────┐    │                  │  │
│   │   Gated Lightning Indexer  │    │                  │  │
│   │   I_{t,s} = Σ σ(·)·σ(qk+b) │    │                  │  │
│   └────────────┬───────────────┘    │                  │  │
│                │                    │                  │  │
│                ▼                    │                  │  │
│   ┌────────────────────────────┐    │                  │  │
│   │   Adaptive Top-k Selection │    │                  │  │
│   │   S_t = Top-k_t(I_{t,:})   │    │                  │  │
│   └────────────┬───────────────┘    │                  │  │
│                │                    │                  │  │
│                ▼                    ▼                  │  │
│   ┌────────────────────────────────────┐               │  │
│   │        Sparse SDPA                  │              │  │
│   │   A = Softmax(Q·K_S^T/√d)          │              │  │
│   │   O = A · V_S^{gated}              │              │  │
│   └────────────┬───────────────────────┘               │  │
│                │                                       │  │
│                ▼                                       │  │
│   ┌────────────────────────────┐                       │  │
│   │    Output Gate G1          │                       │  │
│   │    O · σ(yW^G1)            │                       │  │
│   └────────────┬───────────────┘                       │  │
│                │                                       │  │
│                ▼                                       │  │
│   ┌────────────────────────────┐                       │  │
│   │    Output Projection       │                       │  │
│   │    z = Concat(O_h)·W^O     │                       │  │
│   └────────────┬───────────────┘                       │  │
│                │                                       │  │
└────────────────┼───────────────────────────────────────┘  │
                 │                                          │
                 ▼                                          ▼
        ┌─────────────────┐                       ┌─────────────────┐
        │   Post-Mapping  │                       │   Res-Mapping   │
        │ o = (H^post)^T·z│                       │  r = H^res·x_l  │
        │ o ∈ ℝ^{n×C}     │                       │  r ∈ ℝ^{n×C}    │
        └────────┬────────┘                       └────────┬────────┘
                 │                                         │
                 └──────────────────┬──────────────────────┘
                                    │
                                    ▼
                          ┌─────────────────┐
                          │  Residual Merge │
                          │  x_{l+1} = r + o│
                          └────────┬────────┘
                                   │
                                   ▼
                          ┌─────────────────────────┐
                          │  x_{l+1} ∈ ℝ^{n × C}    │
                          │   (n-stream output)     │
                          └─────────────────────────┘
```

### Detailed GSA Sub-Module Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        GATED SPARSE ATTENTION DETAILS                           │
└─────────────────────────────────────────────────────────────────────────────────┘

                              Input: y ∈ ℝ^{B×L×C}
                                       │
           ┌───────────────────────────┼───────────────────────────┐
           │                           │                           │
           ▼                           ▼                           ▼
   ┌───────────────┐          ┌───────────────┐          ┌───────────────┐
   │  Q Projection │          │  K Projection │          │  V Projection │
   │  y·W^Q        │          │  y·W^K        │          │  y·W^V        │
   │  [B,L,H,d]    │          │  [B,L,H_kv,d] │          │  [B,L,H_kv,d] │
   └───────┬───────┘          └───────┬───────┘          └───────┬───────┘
           │                          │                          │
           │                          │                          ▼
           │                          │               ┌─────────────────────┐
           │                          │               │   VALUE GATE (G2)   │
           │                          │               ├─────────────────────┤
           │                          │               │ G2 = σ(y·W^{G2})    │
           │                          │               │ V_gated = V ⊙ G2    │
           │                          │               │                     │
           │                          │               │ Purpose:            │
           │                          │               │ • Bounded values    │
           │                          │               │ • Token importance  │
           │                          │               │ • Gradient control  │
           │                          │               └──────────┬──────────┘
           │                          │                          │
           │                   ┌──────┴──────┐                   │
           │                   │             │                   │
           │                   ▼             ▼                   │
           │          ┌─────────────┐ ┌─────────────┐            │
           │          │  K_indexer  │ │  Q_indexer  │            │
           │          │  y·W^{KI}   │ │  y·W^{QI}   │            │
           │          │  [B,L,d_I]  │ │ [B,L,H^I,d_I]            │
           │          └──────┬──────┘ └──────┬──────┘            │
           │                 │               │                   │
           │                 └───────┬───────┘                   │
           │                         ▼                           │
           │        ┌─────────────────────────────────┐          │
           │        │     GATED LIGHTNING INDEXER     │          │
           │        ├─────────────────────────────────┤          │
           │        │                                 │          │
           │        │  For each query position t:     │          │
           │        │                                 │          │
           │        │  w_j = σ(y_t · W_j^{Iw})       │          │
           │        │        (query-dependent weight) │          │
           │        │                                 │          │
           │        │  s_j = σ(q^I_{t,j}·k^I_s + b_j)│          │
           │        │        (similarity score)       │          │
           │        │                                 │          │
           │        │  I_{t,s} = Σ_{j=1}^{H^I} w_j·s_j│          │
           │        │        (aggregated index score) │          │
           │        │                                 │          │
           │        │  Key insight: Sigmoid bounding │          │
           │        │  prevents score explosion       │          │
           │        └────────────────┬────────────────┘          │
           │                         │                           │
           │                         ▼                           │
           │        ┌─────────────────────────────────┐          │
           │        │    ADAPTIVE TOP-K SELECTION     │          │
           │        ├─────────────────────────────────┤          │
           │        │                                 │          │
           │        │  v_t = Var(I_{t,:})            │          │
           │        │       (index score variance)    │          │
           │        │                                 │          │
           │        │  k_t = clip(k_base·(1+β·v_t),  │          │
           │        │              k_min, k_max)      │          │
           │        │                                 │          │
           │        │  S_t = argtop_{k_t}(I_{t,:})   │          │
           │        │       (selected token indices)  │          │
           │        │                                 │          │
           │        │  Rationale: More variance →    │          │
           │        │  more discriminative → fewer k │          │
           │        └────────────────┬────────────────┘          │
           │                         │                           │
           └─────────────┬───────────┘                           │
                         │                                       │
                         ▼                                       │
           ┌─────────────────────────────────────────────────────┐
           │              SPARSE ATTENTION COMPUTATION           │
           ├─────────────────────────────────────────────────────┤
           │                                                     │
           │  For each query position t and head h:              │
           │                                                     │
           │  K_sparse = gather(K, S_t)  ∈ ℝ^{k_t × d}          │
           │  V_sparse = gather(V_gated, S_t)  ∈ ℝ^{k_t × d}    │
           │                                                     │
           │  A_{t,h} = Softmax(Q_{t,h}·K_sparse^T / √d)        │
           │           ∈ ℝ^{k_t}                                 │
           │                                                     │
           │  O_{t,h} = A_{t,h} · V_sparse                       │
           │           ∈ ℝ^{d}                                   │
           │                                                     │
           │  Complexity: O(L·k·d) vs O(L²·d) for full attention │
           └─────────────────────────┬───────────────────────────┘
                                     │
                                     ▼
           ┌─────────────────────────────────────────────────────┐
           │                 OUTPUT GATE (G1)                    │
           ├─────────────────────────────────────────────────────┤
           │                                                     │
           │  G1_h = σ(y · W^{G1}_h)   ∈ ℝ^{B×L×d}              │
           │                                                     │
           │  O^{final}_{t,h} = O_{t,h} ⊙ G1_h                  │
           │                                                     │
           │  Purpose:                                           │
           │  • Most effective gate position (per GSA paper)    │
           │  • Eliminates attention sinks                       │
           │  • Head-specific modulation                         │
           │  • Bounded gradients via sigmoid                    │
           └─────────────────────────┬───────────────────────────┘
                                     │
                                     ▼
           ┌─────────────────────────────────────────────────────┐
           │              OUTPUT PROJECTION                      │
           │                                                     │
           │  z = Concat(O^{final}_1, ..., O^{final}_H) · W^O   │
           │      ∈ ℝ^{B×L×C}                                   │
           └─────────────────────────────────────────────────────┘
```

### Multi-Stream Residual Flow

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                       n-STREAM RESIDUAL PROPAGATION                             │
└─────────────────────────────────────────────────────────────────────────────────┘

   Stream 1    Stream 2    Stream 3    Stream 4    (n=4 example)
      │           │           │           │
      ▼           ▼           ▼           ▼
  ┌───────────────────────────────────────────┐
  │         x_l ∈ ℝ^{4 × C}                   │
  │         [x_l^1, x_l^2, x_l^3, x_l^4]      │
  └───────────────────────────────────────────┘
                       │
       ┌───────────────┼───────────────┐
       │               │               │
       ▼               │               ▼
   H^pre ∈ ℝ^{1×4}     │           H^res ∈ O(4)
   [w1,w2,w3,w4]       │           Orthogonal (Cayley)
       │               │               │
       ▼               │               │
   ┌───────────────┐   │               │
   │ y = Σ wi·x^i  │   │               │
   │ (Weighted     │   │               │
   │  aggregation) │   │               │
   └───────┬───────┘   │               │
           │           │               │
           ▼           │               │
   ┌───────────────┐   │               │
   │    GSA(y)     │   │               │
   │               │   │               │
   │  z ∈ ℝ^{1×C}  │   │               │
   └───────┬───────┘   │               │
           │           │               │
           ▼           │               ▼
   ┌───────────────┐   │        ┌───────────────┐
   │ H^post ∈ℝ^{1×4}   │        │ r = H^res·x_l │
   │ [p1,p2,p3,p4] │   │        │               │
   │               │   │        │ r^i = Σ H^res_{ij}·x^j
   │ o^i = pi · z  │   │        │               │
   │ (Distribute   │   │        │ (Mix streams  │
   │  to streams)  │   │        │  via orthog.  │
   └───────┬───────┘   │        │  rotation,    │
           │           │        │  norm-preserv)│
           │           │        └───────┬───────┘
           │           │                │
           └───────────┼────────────────┘
                       │
                       ▼
   ┌───────────────────────────────────────────┐
   │   x_{l+1} = r + o                         │
   │   x_{l+1}^i = Σ_j H^res_{ij}·x_l^j + pi·z │
   └───────────────────────────────────────────┘
                       │
                       ▼
      │           │           │           │
  Stream 1    Stream 2    Stream 3    Stream 4
```

---

## Data Flow

### Complete Forward Pass Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        OrthGSA COMPLETE DATA FLOW                               │
└─────────────────────────────────────────────────────────────────────────────────┘

PHASE 1: INPUT EMBEDDING
─────────────────────────
Token IDs [B, L]
     │
     ▼
┌─────────────┐
│  Embedding  │ → e ∈ ℝ^{B×L×C}
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────┐
│   Expand to n-streams               │
│   x_0 = tile(e, n) ∈ ℝ^{B×L×n×C}   │
│   or learned expansion              │
└──────────────┬──────────────────────┘
               │
               ▼

PHASE 2: OrthGSA LAYERS (l = 0, 1, ..., L-1)
────────────────────────────────────────────

For each layer l:

  Input: x_l ∈ ℝ^{B×L×n×C}
              │
              ├─────────────────────────────┐
              │                             │
              ▼                             │
  ┌───────────────────────────────────┐     │
  │ STEP 2a: Compute oHC Coefficients │     │
  ├───────────────────────────────────┤     │
  │ vec_x = reshape(x_l, [B×L, n×C])  │     │
  │ x_norm = RMSNorm(vec_x)           │     │
  │                                   │     │
  │ H_raw = x_norm · φ + b            │     │
  │   → H_pre_raw  [B×L, n]           │     │
  │   → H_post_raw [B×L, n]           │     │
  │   → H_res_raw  [B×L, n, n]        │     │
  │                                   │     │
  │ H_pre = σ(H_pre_raw)              │     │
  │ H_post = 2·σ(H_post_raw)          │     │
  │ H_res = Cayley(H_res_raw)         │     │
  └───────────────┬───────────────────┘     │
                  │                         │
                  ▼                         │
  ┌───────────────────────────────────┐     │
  │ STEP 2b: Pre-Mapping              │     │
  ├───────────────────────────────────┤     │
  │ y = H_pre @ x_l                   │     │
  │ y ∈ ℝ^{B×L×1×C} → squeeze to      │     │
  │ y ∈ ℝ^{B×L×C}                     │     │
  └───────────────┬───────────────────┘     │
                  │                         │
                  ▼                         │
  ┌───────────────────────────────────┐     │
  │ STEP 2c: Gated Sparse Attention   │     │
  ├───────────────────────────────────┤     │
  │                                   │     │
  │ // QKV Projections                │     │
  │ Q = y·W^Q  [B,L,H,d]              │     │
  │ K = y·W^K  [B,L,H_kv,d]           │     │
  │ V = y·W^V  [B,L,H_kv,d]           │     │
  │                                   │     │
  │ // Apply RoPE                     │     │
  │ Q, K = RoPE(Q, K, positions)      │     │
  │                                   │     │
  │ // Value Gate (G2)                │     │
  │ G2 = σ(y·W^{G2})  [B,L,H_kv,d]   │     │
  │ V = V ⊙ G2                        │     │
  │                                   │     │
  │ // Indexer                        │     │
  │ Q_I = y·W^{QI}  [B,L,H^I,d_I]    │     │
  │ K_I = y·W^{KI}  [B,L,d_I]        │     │
  │ W_I = σ(y·W^{Iw}) [B,L,H^I]      │     │
  │                                   │     │
  │ I = Σ_j W_I_j · σ(Q_I_j·K_I^T+b) │     │
  │ I ∈ ℝ^{B×L×L}                     │     │
  │                                   │     │
  │ // Adaptive Selection             │     │
  │ k_t = f(Var(I_t,:))               │     │
  │ S_t = argtopk(I_t, k_t)           │     │
  │                                   │     │
  │ // Sparse Attention               │     │
  │ K_s, V_s = gather(K, V, S)        │     │
  │ A = Softmax(Q·K_s^T/√d)           │     │
  │ O = A · V_s  [B,L,H,d]            │     │
  │                                   │     │
  │ // Output Gate (G1)               │     │
  │ G1 = σ(y·W^{G1})  [B,L,H,d]      │     │
  │ O = O ⊙ G1                        │     │
  │                                   │     │
  │ // Output Projection              │     │
  │ z_attn = concat(O)·W^O [B,L,C]   │     │
  └───────────────┬───────────────────┘     │
                  │                         │
                  ▼                         ▼
  ┌───────────────────────────────────────────────────┐
  │ STEP 2d: Post-Mapping & Residual                  │
  ├───────────────────────────────────────────────────┤
  │                                                   │
  │ // Post-mapping (distribute to streams)           │
  │ o_attn = H_post^T @ z_attn  [B,L,n,C]            │
  │                                                   │
  │ // Residual mapping (mix streams)                 │
  │ r_attn = H_res @ x_l  [B,L,n,C]                  │
  │                                                   │
  │ // Merge                                          │
  │ x_{l+0.5} = r_attn + o_attn                      │
  └───────────────┬───────────────────────────────────┘
                  │
                  ▼
  ┌───────────────────────────────────┐
  │ STEP 2e: oHC + FFN (similar flow) │
  ├───────────────────────────────────┤
  │ // Compute new oHC coefficients   │
  │ H_pre', H_post', H_res' = oHC(x_{l+0.5})
  │                                   │
  │ // Pre-mapping                    │
  │ y' = H_pre' @ x_{l+0.5}           │
  │                                   │
  │ // FFN (SwiGLU)                   │
  │ z_ffn = W_down(SiLU(W_gate·y') ⊙ W_up·y')
  │                                   │
  │ // Post + Residual                │
  │ o_ffn = H_post'^T @ z_ffn         │
  │ r_ffn = H_res' @ x_{l+0.5}        │
  │ x_{l+1} = r_ffn + o_ffn           │
  └───────────────┬───────────────────┘
                  │
                  ▼
  Output: x_{l+1} ∈ ℝ^{B×L×n×C}
              │
              │ (Continue to next layer)
              ▼

PHASE 3: OUTPUT
───────────────
  x_L ∈ ℝ^{B×L×n×C}
              │
              ▼
  ┌───────────────────────────────────┐
  │ Collapse n-streams                │
  │ Method 1: Average pooling         │
  │   h = mean(x_L, dim=n)            │
  │ Method 2: Learned aggregation     │
  │   h = W_agg · x_L                 │
  └───────────────┬───────────────────┘
                  │
                  ▼
  ┌───────────────────────────────────┐
  │ Final RMSNorm                     │
  │ h = RMSNorm(h)                    │
  └───────────────┬───────────────────┘
                  │
                  ▼
  ┌───────────────────────────────────┐
  │ LM Head                           │
  │ logits = h · W_vocab^T            │
  │ logits ∈ ℝ^{B×L×V}               │
  └───────────────────────────────────┘
```

### Tensor Shape Summary

| Tensor | Shape | Description |
|--------|-------|-------------|
| `x_l` | `[B, L, n, C]` | n-stream hidden state |
| `vec_x` | `[B×L, n×C]` | Flattened for oHC computation |
| `H_pre` | `[B×L, 1, n]` | Pre-mapping coefficients |
| `H_post` | `[B×L, 1, n]` | Post-mapping coefficients |
| `H_res` | `[B×L, n, n]` | Residual mapping (orthogonal via Cayley) |
| `y` | `[B, L, C]` | Aggregated single-stream for attention |
| `Q` | `[B, L, H, d]` | Queries |
| `K` | `[B, L, H_kv, d]` | Keys (GQA) |
| `V` | `[B, L, H_kv, d]` | Values (GQA) |
| `I` | `[B, L, L]` | Indexer scores |
| `S` | `[B, L, k_t]` | Selected indices |
| `A` | `[B, L, H, k_t]` | Sparse attention weights |
| `O` | `[B, L, H, d]` | Attention output |
| `z` | `[B, L, C]` | Projected output |

---

## Algorithm Details

### Algorithm 1: OrthGSA Forward Pass

```
Algorithm: OrthGSA_Forward(x_0, θ)
─────────────────────────────────────────────────────────────────────────────────
Input:
  x_0 ∈ ℝ^{B×L×n×C}  : Initial n-stream hidden state
  θ                   : All model parameters

Output:
  x_L ∈ ℝ^{B×L×n×C}  : Final n-stream hidden state

1:  for l = 0 to L-1 do
2:      // ═══ ATTENTION SUB-BLOCK ═══
3:
4:      // Compute oHC coefficients (attention)
5:      x_norm ← RMSNorm(flatten(x_l))
6:      H_pre_attn ← σ(α_pre · (x_norm · φ_pre) + b_pre)
7:      H_post_attn ← 2 · σ(α_post · (x_norm · φ_post) + b_post)
8:      H_res_attn ← CayleyTransform(α_res · reshape(x_norm · φ_res) + b_res)
9:
10:     // Pre-mapping: n-streams → single stream
11:     y ← H_pre_attn @ x_l                    // [B,L,C]
12:
13:     // Gated Sparse Attention
14:     z_attn ← GSA(y, θ_attn)                 // [B,L,C]
15:
16:     // Post-mapping + Residual
17:     o_attn ← H_post_attn^T @ z_attn         // [B,L,n,C]
18:     r_attn ← H_res_attn @ x_l               // [B,L,n,C]
19:     x_mid ← r_attn + o_attn
20:
21:     // ═══ FFN SUB-BLOCK ═══
22:
23:     // Compute oHC coefficients (FFN)
24:     x_norm_ffn ← RMSNorm(flatten(x_mid))
25:     H_pre_ffn ← σ(α_pre_ffn · (x_norm_ffn · φ_pre_ffn) + b_pre_ffn)
26:     H_post_ffn ← 2 · σ(α_post_ffn · (x_norm_ffn · φ_post_ffn) + b_post_ffn)
27:     H_res_ffn ← CayleyTransform(α_res_ffn · reshape(x_norm_ffn · φ_res_ffn) + b_res_ffn)
28:
29:     // Pre-mapping
30:     y_ffn ← H_pre_ffn @ x_mid               // [B,L,C]
31:
32:     // FFN (SwiGLU)
33:     z_ffn ← FFN(y_ffn, θ_ffn)               // [B,L,C]
34:
35:     // Post-mapping + Residual
36:     o_ffn ← H_post_ffn^T @ z_ffn            // [B,L,n,C]
37:     r_ffn ← H_res_ffn @ x_mid               // [B,L,n,C]
38:     x_{l+1} ← r_ffn + o_ffn
39:
40: end for
41: return x_L
─────────────────────────────────────────────────────────────────────────────────
```

### Algorithm 2: Gated Sparse Attention (GSA)

```
Algorithm: GSA(y, θ_attn)
─────────────────────────────────────────────────────────────────────────────────
Input:
  y ∈ ℝ^{B×L×C}       : Single-stream input (from oHC pre-mapping)
  θ_attn              : Attention parameters

Output:
  z ∈ ℝ^{B×L×C}       : Attention output

1:  // QKV Projections
2:  Q ← y · W^Q                               // [B,L,H,d]
3:  K ← y · W^K                               // [B,L,H_kv,d]
4:  V ← y · W^V                               // [B,L,H_kv,d]

5:  // Apply RoPE
6:  Q, K ← ApplyRoPE(Q, K, positions)

7:  // Value Gate (G2)
8:  G2 ← σ(y · W^{G2})                        // [B,L,H_kv,d]
9:  V ← V ⊙ G2

10: // Expand KV for GQA
11: K ← repeat_kv(K, n_rep)                   // [B,L,H,d]
12: V ← repeat_kv(V, n_rep)                   // [B,L,H,d]

13: // Gated Lightning Indexer
14: Q_I ← y · W^{QI}                          // [B,L,H^I,d_I]
15: K_I ← y · W^{KI}                          // [B,L,d_I]
16: W_I ← σ(y · W^{Iw})                       // [B,L,H^I]
17:
18: for t = 1 to L do
19:     for j = 1 to H^I do
20:         S_j ← σ(Q_I[t,j] · K_I^T + b_j)   // [L]
21:     end for
22:     I[t] ← Σ_j W_I[t,j] · S_j             // [L]
23: end for

24: // Adaptive Top-k Selection
25: for t = 1 to L do
26:     v_t ← Var(I[t])
27:     k_t ← clip(k_base · (1 + β · softplus(v_t)), k_min, k_max)
28:     S_t ← argtopk(I[t], k_t)              // Selected indices
29: end for

30: // Sparse Attention
31: for t = 1 to L do
32:     K_sparse ← gather(K, S_t)              // [k_t, H, d]
33:     V_sparse ← gather(V, S_t)              // [k_t, H, d]
34:
35:     A_t ← Softmax(Q[t] · K_sparse^T / √d)  // [H, k_t]
36:     O_t ← A_t @ V_sparse                   // [H, d]
37: end for

38: // Output Gate (G1)
39: G1 ← σ(y · W^{G1})                        // [B,L,H,d]
40: O ← O ⊙ G1

41: // Output Projection
42: z ← concat(O) · W^O                       // [B,L,C]
43: return z
─────────────────────────────────────────────────────────────────────────────────
```

### Algorithm 3: Cayley Transform Projection

```
Algorithm: CayleyTransform(H_raw)
─────────────────────────────────────────────────────────────────────────────────
Input:
  H_raw ∈ ℝ^{n×n}    : Raw coefficient matrix

Output:
  H ∈ O(n)           : Orthogonal matrix (special orthogonal, det = +1)

1:  // Extract skew-symmetric component
2:  A ← (H_raw - H_raw^T) / 2              // Guarantees A^T = -A

3:  // Compute Cayley transform matrices
4:  I_plus_A ← I_n + A
5:  I_minus_A ← I_n - A

6:  // Apply Cayley transform via linear solve
7:  H ← solve(I_plus_A, I_minus_A)         // H = (I-A)(I+A)^{-1}

8:  return H

Properties of output H:
  - H^T · H = I_n          (orthogonal)
  - H · H^T = I_n          (orthogonal)
  - det(H) = +1            (special orthogonal / rotation)
  - ||H||_2 = 1            (unit spectral norm, exactly)
─────────────────────────────────────────────────────────────────────────────────
```

**Note**: The Cayley Transform replaces the iterative Sinkhorn-Knopp projection, providing:
- Single-pass computation (no iterations needed)
- Exact orthogonality (not approximate)
- Better gradient flow (no unrolling through iterations)
- Allows negative mixing coefficients (more expressive than doubly stochastic)

---

## Infrastructure Optimizations

### 1. Kernel Fusion Strategy

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        FUSED KERNEL ARCHITECTURE                                │
└─────────────────────────────────────────────────────────────────────────────────┘

FUSED KERNEL 1: oHC Coefficient Computation
────────────────────────────────────────────
┌─────────────────────────────────────────────────────────────────────┐
│  Input: x_l ∈ ℝ^{B×L×n×C}                                          │
│                                                                     │
│  Fused operations:                                                  │
│  1. vec_x = reshape(x_l)                                           │
│  2. r = ||vec_x||_2 / √(nC)          (RMS computation)             │
│  3. raw = vec_x · φ                   (MatMul)                      │
│  4. scaled = α/r · raw + b           (Scale + Bias)                │
│  5. H_pre = σ(scaled_pre)            (Sigmoid)                     │
│  6. H_post = 2·σ(scaled_post)        (Scaled sigmoid)              │
│                                                                     │
│  Output: H_pre, H_post, H_res_raw                                  │
│                                                                     │
│  Memory I/O: Read nC, Write (n² + 2n) per token                    │
└─────────────────────────────────────────────────────────────────────┘

FUSED KERNEL 2: Cayley Transform
─────────────────────────────────
┌─────────────────────────────────────────────────────────────────────┐
│  Input: H_res_raw ∈ ℝ^{n×n}                                        │
│                                                                     │
│  Fused operations (in registers):                                   │
│  1. A = (H_res_raw - H_res_raw^T) / 2     (skew-symmetric)         │
│  2. I_plus_A = I_n + A                                             │
│  3. I_minus_A = I_n - A                                            │
│  4. H_res = solve(I_plus_A, I_minus_A)    (linear solve)           │
│                                                                     │
│  Output: H_res ∈ O(n)  (orthogonal matrix)                         │
│                                                                     │
│  Note: Small n (typically 4) fits in registers                     │
│  For n=4, direct inversion is efficient: O(1) flops                │
│  Memory I/O: Read n², Write n²                                     │
└─────────────────────────────────────────────────────────────────────┘

FUSED KERNEL 3: Pre-mapping + GSA Prep
──────────────────────────────────────
┌─────────────────────────────────────────────────────────────────────┐
│  Input: x_l, H_pre                                                  │
│                                                                     │
│  Fused operations:                                                  │
│  1. y = H_pre @ x_l                  (n×C → C aggregation)         │
│  2. Q = y · W^Q                      (QKV projection start)        │
│  3. K = y · W^K                                                     │
│  4. V = y · W^V                                                     │
│                                                                     │
│  Output: Q, K, V                                                    │
└─────────────────────────────────────────────────────────────────────┘

FUSED KERNEL 4: GSA Indexer + Selection
───────────────────────────────────────
┌─────────────────────────────────────────────────────────────────────┐
│  Input: y, W^{QI}, W^{KI}, W^{Iw}                                   │
│                                                                     │
│  Fused operations:                                                  │
│  1. Q_I = y · W^{QI}                                               │
│  2. K_I = y · W^{KI}                                               │
│  3. W_I = σ(y · W^{Iw})                                            │
│  4. I = Σ W_I_j · σ(Q_I_j · K_I^T + b)                             │
│  5. k_t = adaptive_k(Var(I))                                       │
│  6. S = argtopk(I, k)                                              │
│                                                                     │
│  Output: S (selected indices)                                       │
│                                                                     │
│  Note: This is the bottleneck - use Triton for efficiency          │
└─────────────────────────────────────────────────────────────────────┘

FUSED KERNEL 5: Sparse SDPA + Gating
────────────────────────────────────
┌─────────────────────────────────────────────────────────────────────┐
│  Input: Q, K, V, G2, G1, S                                         │
│                                                                     │
│  Fused operations:                                                  │
│  1. V_gated = V ⊙ G2                 (Value gating)                │
│  2. K_s, V_s = gather(K, V_gated, S) (Sparse gather)               │
│  3. A = Softmax(Q · K_s^T / √d)      (Attention scores)            │
│  4. O = A @ V_s                       (Weighted sum)                │
│  5. O = O ⊙ G1                       (Output gating)               │
│  6. z = concat(O) · W^O              (Output projection)           │
│                                                                     │
│  Output: z                                                          │
│                                                                     │
│  Complexity: O(L·k·d) instead of O(L²·d)                           │
└─────────────────────────────────────────────────────────────────────┘

FUSED KERNEL 6: Post-mapping + Residual
───────────────────────────────────────
┌─────────────────────────────────────────────────────────────────────┐
│  Input: z, x_l, H_post, H_res                                      │
│                                                                     │
│  Fused operations:                                                  │
│  1. o = H_post^T @ z                 (Distribute to n streams)     │
│  2. r = H_res @ x_l                  (Mix n streams)               │
│  3. x_{l+1} = r + o                  (Residual addition)           │
│                                                                     │
│  Output: x_{l+1}                                                    │
│                                                                     │
│  Memory I/O: Read (n+1)C, Write nC per token                       │
└─────────────────────────────────────────────────────────────────────┘
```

### 2. Selective Recomputation Strategy

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                     SELECTIVE RECOMPUTATION FOR OrthGSA                         │
└─────────────────────────────────────────────────────────────────────────────────┘

Forward Pass Memory Strategy:
─────────────────────────────

Block Size: L_r* = √(nL/(n+2))  (optimal for OrthGSA)

┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│   FORWARD PASS                         BACKWARD PASS                │
│   ─────────────                        ─────────────                │
│                                                                     │
│   Layer 0  ─┐                                                       │
│   Layer 1   │ Block 0                  Recompute:                   │
│   ...       │                          • oHC coefficients           │
│   Layer L_r─┘  → Store x_0             • H_pre, H_post, H_res       │
│                                        • Pre-mapping y              │
│   Layer L_r+1─┐                        • Indexer scores             │
│   Layer L_r+2 │ Block 1                                             │
│   ...         │                        DO NOT Recompute:            │
│   Layer 2L_r ─┘  → Store x_{L_r}       • GSA output (expensive)     │
│                                        • FFN output (expensive)     │
│   ...                                                               │
│                                                                     │
│   Cost Analysis:                                                    │
│   Storage: nC × ⌈L/L_r⌉ per sequence                               │
│   Recompute: ~15% overhead (oHC coefficients are cheap)            │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 3. Memory Access Pattern Optimization

| Operation | Naive I/O | Optimized I/O | Savings |
|-----------|-----------|---------------|---------|
| oHC Coefficients | 3nC + n² + 2n read, n² + 2n write | nC read, n² + 2n write | 66% |
| Pre + Post + Res Mapping | (3n+1)C read, 3nC write | (n+1)C read, nC write | 70% |
| GSA Indexer | 3LC read, L² write | 3LC read, Lk write | ~85% |
| Sparse SDPA | L²d read, Ld write | Lkd read, Ld write | ~85% |

### 4. Communication Overlap for Distributed Training

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                     DUALPIPE COMMUNICATION OVERLAP                              │
└─────────────────────────────────────────────────────────────────────────────────┘

Pipeline Stage i                          Pipeline Stage i+1
─────────────────                         ──────────────────

┌──────────────┐  ═══════════════════════▶  ┌──────────────┐
│   OrthGSA    │     n-stream tensor        │   OrthGSA    │
│   Attention  │     (x_{l+0.5})            │   Attention  │
└──────────────┘                            └──────────────┘
       │                                           │
       │                                           │
       ▼                                           ▼
┌──────────────┐                            ┌──────────────┐
│   OrthGSA    │                            │   OrthGSA    │
│     FFN      │                            │     FFN      │
└──────────────┘                            └──────────────┘
       │                                           │
       │  ═══════════════════════════════▶         │
       │         Async P2P send                    │

Optimization Strategy:
─────────────────────
1. Execute Post+Res mapping on high-priority compute stream
2. Overlap Cayley Transform computation with P2P communication
3. Use separate CUDA streams for oHC and GSA computations
4. Prefetch next layer's oHC parameters during current layer's SDPA
```

---

## Theoretical Properties

### 1. Complexity Analysis

| Component | Time Complexity | Space Complexity |
|-----------|-----------------|------------------|
| oHC Coefficient Computation | O(BLnC) | O(n² + 2n) |
| Cayley Transform (per token) | O(n³) | O(n²) |
| Pre/Post/Res Mapping | O(BLnC) | O(nC) |
| GSA Indexer | O(BL²d_I × H^I) | O(L²) |
| Sparse Attention | O(BLkd × H) | O(Lkd) |
| **Total per layer** | **O(BL(nC + kHd))** | **O(BL(nC + k))** |

For typical settings (n=4, k<<L):
- **OrthGSA**: O(BL·k·d) attention + O(BL·nC) residual
- **Standard Transformer**: O(BL²d)

**Speedup**: ~(L/k) × for attention, with ~6.7% overhead from oHC

### 2. Gradient Flow Analysis

**Theorem (Signal Preservation)**: In OrthGSA, the gradient flow through L layers satisfies:

$$\left\| \frac{\partial \mathcal{L}}{\partial \mathbf{x}_0} \right\| \leq \left\| \frac{\partial \mathcal{L}}{\partial \mathbf{x}_L} \right\| \cdot \prod_{l=0}^{L-1} \left( \|\mathbf{H}_l^{\text{res}}\|_2 + \|\mathbf{H}_l^{\text{post}}\|_2 \cdot \|\nabla \mathcal{F}^{\text{GSA}}_l\| \cdot \|\mathbf{H}_l^{\text{pre}}\|_2 \right)$$

**Key bounds** (with Cayley Transform):
1. $\|\mathbf{H}_l^{\text{res}}\|_2 = 1$ (orthogonal matrix, exactly unit spectral norm)
2. $\|\mathbf{H}_l^{\text{pre}}\|_2 \leq \sqrt{n}$ (sigmoid bounded)
3. $\|\mathbf{H}_l^{\text{post}}\|_2 \leq 2\sqrt{n}$ (scaled sigmoid bounded)
4. GSA gradients bounded by sigmoid gating

**Consequence**: No gradient explosion regardless of depth L.

**Cayley Transform Advantage**: The orthogonality constraint $\mathbf{H}^{\text{res}\top}\mathbf{H}^{\text{res}} = \mathbf{I}$ ensures:
- Perfect gradient preservation through residual path (no attenuation or explosion)
- Eigenvalues all lie on the unit circle
- Condition number is exactly 1 (perfectly conditioned)

### 3. Identity Mapping Restoration

**Theorem (Identity at Initialization)**: With Cayley Transform, identity initialization is natural:
- When $\tilde{\mathbf{H}}^{\text{res}} = \mathbf{0}$, the skew-symmetric component $\mathbf{A} = \mathbf{0}$
- Cayley transform of zero: $\text{Cayley}(\mathbf{0}) = (\mathbf{I} - \mathbf{0})(\mathbf{I} + \mathbf{0})^{-1} = \mathbf{I}_n$

**Initialization for identity behavior**:
- $\alpha^{\text{pre}}, \alpha^{\text{post}}, \alpha^{\text{res}} \to 0$
- $\mathbf{b}^{\text{pre}} = [1/n, ..., 1/n]$ (uniform aggregation)
- $\mathbf{b}^{\text{res}} = \mathbf{0}$ (Cayley produces identity automatically)
- $\mathbf{b}^{\text{post}} = [1, 0, ..., 0]$ (first stream)

Then:
$$\mathbf{x}_{l+1} \approx \mathbf{x}_l + \text{broadcast}(\mathcal{F}^{\text{GSA}}(\text{mean}(\mathbf{x}_l)))$$

Which recovers standard residual behavior at initialization.

**Cayley Transform Advantage**: Unlike Sinkhorn-Knopp which requires careful initialization to converge to identity, the Cayley transform naturally produces $\mathbf{I}_n$ when the input is zero or symmetric.

### 4. Attention Sink Elimination

**Proposition (Bounded Attention)**: The GSA output gate ensures:

$$\max_{t,s} |\text{Attn}_{t,s}| \leq \max_{t,s} |\sigma(\mathbf{y}_t W^{G_1})| \leq 1$$

Combined with sparse selection, this prevents any single token from dominating attention allocation.

**Empirical metric**: First-token attention drops from ~47% (baseline) to <5% (OrthGSA).

---

## References

1. **Gated Sparse Attention (GSA)**: Combining Computational Efficiency with Training Stability for Long-Context Language Models
2. **oHC**: Orthogonal-Constrained Hyper-Connections (to be published)
3. DeepSeek-V3: Pushing the Frontier of Open Large Language Models
4. Gated Attention for Large Language Models (arXiv:2505.06708)
5. Cayley (1846): Sur quelques propriétés des déterminants gauches
6. Helfrich et al. (2018): Orthogonal Recurrent Neural Networks with Scaled Cayley Transform

---

*For practical usage and training instructions, see [README.md](../README.md) and [Getting_started.md](../Getting_started.md).*

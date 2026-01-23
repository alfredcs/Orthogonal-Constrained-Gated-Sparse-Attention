# OrthGSA: Orthogonal-Constrained Gated Sparse Attention

A unified architecture combining **Gated Sparse Attention (GSA)** with **Orthogonal-Constrained Hyper-Connections** using the **Cayley Transform** to achieve computational efficiency, training stability, and restored identity mapping properties.

---

## Table of Contents

1. [Overview](#overview)
2. [Design Rationale](#design-rationale)
3. [Mathematical Formulation](#mathematical-formulation) *(summary - [full details](docs/Mathematical_Theory.md))*
4. [Architecture Diagram](#architecture-diagram) *(summary - [full details](docs/Mathematical_Theory.md#architecture-diagrams))*
5. [Theoretical Properties](#theoretical-properties) *(summary - [full details](docs/Mathematical_Theory.md#theoretical-properties))*
6. [Implementation Considerations](#implementation-considerations)

---

## Overview

**OrthGSA** (Orthogonally-constrained Gated Sparse Attention) is a novel transformer architecture that synergistically integrates two complementary innovations:

| Component | Source | Contribution |
|-----------|--------|--------------|
| **Gated Sparse Attention** | GSA | Sub-quadratic attention complexity, attention sink elimination |
| **Orthogonal-Constrained Hyper-Connections** | oHC | Identity mapping preservation via Cayley Transform, multi-stream residual expressiveness |

### Key Benefits

| Property | Standard Transformer | GSA Only | oHC Only | **OrthGSA** |
|----------|---------------------|----------|----------|-------------|
| Attention Complexity | O(L²d) | **O(Lkd)** | O(L²d) | **O(Lkd)** |
| Attention Sinks | Severe | **Eliminated** | Moderate | **Eliminated** |
| Residual Identity Mapping | Preserved | Preserved | **Enhanced** | **Enhanced** |
| Training Stability | Moderate | Good | **Excellent** | **Excellent** |
| Length Extrapolation | Poor | Good | Moderate | **Excellent** |
| Representational Capacity | Baseline | Baseline | **Expanded** | **Expanded** |

---

## Design Rationale

### Why Combine GSA and oHC?

The two innovations address **complementary** aspects of transformer architecture:

**GSA addresses attention-level concerns:**
- Quadratic complexity of self-attention → Sparse top-k selection
- Attention sink phenomenon → Sigmoid gating (bounded activations)
- Training instability from attention → Dual gating (G1/G2)

**oHC addresses residual connection concerns:**
- Single-stream residual bottleneck → n-stream expansion
- Signal explosion/vanishing in deep networks → Orthogonal constraint via Cayley Transform
- Loss of identity mapping property → Orthogonal residual matrices (unit spectral norm)

### Synergistic Integration Points

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     COMPLEMENTARY INNOVATION LAYERS                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Layer Level        │    Component         │    Innovation                 │
│   ─────────────────────────────────────────────────────────                │
│   Residual Stream    │    oHC               │    n-stream expansion        │
│                      │                      │    Cayley Transform          │
│   ─────────────────────────────────────────────────────────                │
│   Attention Layer    │    GSA               │    Sparse selection          │
│                      │                      │    Dual gating (G1, G2)       │
│   ─────────────────────────────────────────────────────────                │
│   FFN Layer          │    Standard/oHC      │    Multi-stream aggregation  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Key Design Decisions

1. **Apply oHC at residual level**: Wrap the entire GSA block with orthogonally-constrained hyper-connections
2. **Apply GSA within the attention layer**: Replace standard attention with gated sparse attention
3. **Unified gating philosophy**: Both oHC and GSA use sigmoid-based gating for stability
4. **Shared infrastructure**: Leverage kernel fusion from both approaches

---

## Mathematical Formulation

> **Full details**: See [docs/Mathematical_Theory.md](docs/Mathematical_Theory.md) for complete mathematical derivations.

### Core Equations

**OrthGSA Layer Transformation**:

$$\mathbf{x}_{l+1} = \mathbf{H}_l^{\text{res}} \mathbf{x}_l + (\mathbf{H}_l^{\text{post}})^\top \mathcal{F}^{\text{GSA}}(\mathbf{H}_l^{\text{pre}} \mathbf{x}_l, \mathbf{W}_l)$$

Where:
- $\mathbf{H}_l^{\text{res}}$ : Orthogonal residual mapping via **Cayley Transform**
- $\mathbf{H}_l^{\text{pre}}, \mathbf{H}_l^{\text{post}}$ : Pre/post mapping coefficients
- $\mathcal{F}^{\text{GSA}}$ : Gated Sparse Attention function

**Cayley Transform** (for orthogonal constraint):

$$\mathbf{H}^{\text{res}} = (\mathbf{I}_n - \mathbf{A})(\mathbf{I}_n + \mathbf{A})^{-1}$$

where $\mathbf{A}$ is the skew-symmetric component of the raw coefficients.

### Key Properties

| Property | Value |
|----------|-------|
| Orthogonality | $\mathbf{H}^{\text{res}\top} \mathbf{H}^{\text{res}} = \mathbf{I}_n$ (exact) |
| Spectral Norm | $\|\mathbf{H}^{\text{res}}\|_2 = 1$ |
| Identity at Init | $\text{Cayley}(\mathbf{0}) = \mathbf{I}_n$ |
| Computation | Single matrix solve (no iteration) |

---

## Architecture Diagram

> **Full details**: See [docs/Mathematical_Theory.md](docs/Mathematical_Theory.md#architecture-diagrams) for detailed diagrams including GSA sub-module architecture, multi-stream residual flow, data flow, and algorithms.

### Simplified Block Structure

```
Input: x_l ∈ ℝ^{n×C}  (n-stream)
           │
    ┌──────┼──────┐
    │      │      │
    ▼      ▼      ▼
  H^pre  H^res  H^post
  (σ)  (Cayley) (2σ)
    │      │      │
    └──┬───┘      │
       ▼          │
    GSA(y)        │
       │          │
       ▼          ▼
   Post-Map   Res-Map
       │          │
       └────┬─────┘
            ▼
Output: x_{l+1} = H^res·x_l + H^post'·GSA(H^pre·x_l)
```

### Key Components

| Stage | Operation | Complexity |
|-------|-----------|------------|
| **Pre-mapping** | Aggregate n streams → 1 | O(nC) |
| **GSA** | Sparse attention (top-k) | O(L·k·d) |
| **Post-mapping** | Distribute 1 → n streams | O(nC) |
| **Residual** | Orthogonal mixing (Cayley) | O(n³) |

### Data Flow Summary

1. **Input**: Token embeddings → expand to n-streams
2. **Per Layer**: oHC coefficients → Pre-map → GSA → Post-map → Residual merge
3. **Output**: Collapse n-streams → RMSNorm → LM Head

---

## Theoretical Properties

> **Full details**: See [docs/Mathematical_Theory.md](docs/Mathematical_Theory.md#theoretical-properties) for complete proofs and analysis.

### Complexity Summary

| Metric | OrthGSA | Standard Transformer |
|--------|---------|---------------------|
| Attention Complexity | **O(BL·k·d)** | O(BL²d) |
| Speedup at 128K context | **~11×** | 1× |
| oHC Overhead | ~6.7% | - |

### Key Theoretical Results

1. **Gradient Flow**: No gradient explosion regardless of depth L due to:
   - Orthogonal residual matrices ($\|\mathbf{H}^{\text{res}}\|_2 = 1$ exactly)
   - Sigmoid-bounded gates in GSA

2. **Identity Mapping**: Natural identity initialization via Cayley Transform
   - $\text{Cayley}(\mathbf{0}) = \mathbf{I}_n$ (no special initialization needed)

3. **Attention Sink Elimination**: First-token attention drops from ~47% (baseline) to <5%

---

## Implementation Considerations

### 1. Hyperparameter Recommendations

| Parameter | Symbol | Recommended Value | Notes |
|-----------|--------|-------------------|-------|
| Stream expansion | n | 4 | Balance between expressiveness and overhead |
| Cayley scaling | α_cayley | 0.1 | Initial skew-symmetric scaling |
| Base selected tokens | k_base | 2048 | For 128K context |
| Min selected tokens | k_min | 256 | Lower bound |
| Max selected tokens | k_max | 4096 | Upper bound |
| Indexer heads | H^I | 4 | Lightweight |
| Indexer dimension | d_I | 64 | Compact |
| Gate bias init | - | 0.5 | Moderate initial gating |
| oHC alpha init | α | 0.01 | Small for near-identity start |

### 2. Initialization Strategy

```python
def initialize_orthgsa_layer(layer):
    # oHC coefficients - start near identity
    nn.init.zeros_(layer.phi_pre)
    nn.init.zeros_(layer.phi_post)
    nn.init.zeros_(layer.phi_res)

    # Biases for near-identity behavior
    nn.init.constant_(layer.b_pre, 1.0 / layer.n_streams)  # Uniform aggregation
    nn.init.constant_(layer.b_post, 1.0)  # Uniform distribution
    # For Cayley Transform: b_res = 0 produces identity matrix
    # (Cayley(0) = (I-0)(I+0)^{-1} = I)
    nn.init.zeros_(layer.b_res)  # Identity via Cayley transform

    # Small alpha for gradual learning
    nn.init.constant_(layer.alpha_pre, 0.01)
    nn.init.constant_(layer.alpha_post, 0.01)
    nn.init.constant_(layer.alpha_res, 0.01)

    # GSA gates - moderate initial gating
    nn.init.xavier_uniform_(layer.W_G1)
    nn.init.xavier_uniform_(layer.W_G2)
    nn.init.constant_(layer.gate_bias, 0.5)

    # Indexer
    nn.init.xavier_uniform_(layer.W_QI)
    nn.init.xavier_uniform_(layer.W_KI)
    nn.init.zeros_(layer.indexer_bias)


def cayley_transform(H_raw: torch.Tensor) -> torch.Tensor:
    """
    Apply Cayley transform: (I - A)(I + A)^{-1}
    where A is the skew-symmetric part of H_raw.
    """
    # Extract skew-symmetric component
    A = (H_raw - H_raw.transpose(-2, -1)) / 2

    n = A.shape[-1]
    I = torch.eye(n, device=A.device, dtype=A.dtype)

    # Cayley transform via linear solve (more stable than inverse)
    return torch.linalg.solve(I + A, I - A)
```

### 3. Training Stability Techniques

1. **Warmup schedule for oHC**:
   - First 1000 steps: α = 0.001
   - Ramp up over 5000 steps to target α

2. **Separate learning rates**:
   - oHC parameters: 10× base LR (they're cheap to compute)
   - GSA parameters: 1× base LR

3. **Gradient clipping**:
   - Global norm clipping at 1.0
   - Additional per-layer clipping for oHC coefficients

4. **Mixed precision**:
   - Cayley Transform linear solve in FP32 (numerical stability)
   - Everything else in BF16

### 4. Distributed Training

For multi-GPU training, we recommend **DeepSpeed ZeRO-3** with **Ring Attention** for long-context training:

| Method | Memory per GPU (8B model) | Use Case |
|--------|---------------------------|----------|
| DDP | ~64GB | Single GPU or 80GB+ GPUs |
| DeepSpeed ZeRO-2 | ~30GB | Short context (≤8K) |
| **DeepSpeed ZeRO-3** | **~20GB** | Recommended for 24-48GB GPUs |
| **ZeRO-3 + Ring Attention** | **~25GB** | Ultra-long context (64K-128K) |

```bash
# DeepSpeed ZeRO-3 (recommended for standard training)
deepspeed --num_gpus=8 scripts/train_deepspeed.py --config configs/config_qwen3_8b_64k.yaml

# DeepSpeed ZeRO-3 + Ring Attention (for ultra-long context 64K-128K)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
deepspeed --num_gpus=8 scripts/train_ring_attention.py --config configs/config_qwen3_8b_64k_ring.yaml
```

**Ring Attention** enables sequence parallelism by distributing the sequence across GPUs:
- Each GPU processes `seq_len / num_gpus` tokens
- KV cache is exchanged in a ring topology
- Enables 64K-128K context on 8x 44GB GPUs

See `Getting_started.md` for detailed setup instructions.

### 5. Expected Performance

| Model Size | Standard Transformer | OrthGSA | Speedup | Memory |
|------------|---------------------|---------|---------|--------|
| 1.7B @ 8K | 1.00× | 0.90× | 1.1× | 0.95× |
| 8B @ 32K | 1.00× | 0.42× | 2.4× | 0.85× |
| 8B @ 64K (Ring) | 1.00× | 0.25× | 4× | 0.70× |
| 8B @ 128K (Ring) | 1.00× | 0.09× | 11× | 0.65× |

*Note: Speedup increases with sequence length due to sparse attention. Ring Attention enables training on contexts that would otherwise exceed GPU memory.*

---

## Summary

**OrthGSA** unifies two complementary architectural innovations:

1. **From GSA**: Sub-quadratic sparse attention, attention sink elimination via dual gating
2. **From oHC**: Multi-stream residual connections with orthogonal constraints via **Cayley Transform**

The combination yields a transformer architecture that is:
- **Computationally efficient**: O(Lkd) instead of O(L²d) for attention
- **Training stable**: Bounded gradients from both sigmoid gating and orthogonal (unit spectral norm) constraints
- **Expressive**: n-stream residuals increase representational capacity, with orthogonal mixing allowing negative coefficients
- **Scalable**: Designed for distributed training with communication overlap

### Cayley Transform Highlights

The Cayley Transform $(\mathbf{I} - \mathbf{A})(\mathbf{I} + \mathbf{A})^{-1}$ provides:
- **Exact orthogonality**: $\mathbf{H}^{\text{res}\top}\mathbf{H}^{\text{res}} = \mathbf{I}$ (not approximate)
- **Single-pass computation**: No iterative projection needed (unlike Sinkhorn-Knopp)
- **Natural identity initialization**: $\text{Cayley}(\mathbf{0}) = \mathbf{I}$
- **Better gradient flow**: Clean analytical gradients without unrolling

---

## References

1. **Gated Sparse Attention (GSA)**: Combining Computational Efficiency with Training Stability for Long-Context Language Models
2. **oHC**: Orthogonal-Constrained Hyper-Connections (to be published)
3. DeepSeek-V3: Pushing the Frontier of Open Large Language Models
4. Gated Attention for Large Language Models (arXiv:2505.06708)
5. Cayley (1846): Sur quelques propriétés des déterminants gauches
6. Helfrich et al. (2018): Orthogonal Recurrent Neural Networks with Scaled Cayley Transform

---

*OrthGSA: Where efficient sparse attention meets stable hyper-connections.*

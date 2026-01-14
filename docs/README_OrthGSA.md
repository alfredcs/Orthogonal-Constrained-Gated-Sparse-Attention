# OrthGSA: Orthogonal-Constrained Gated Sparse Attention

A unified architecture combining **Gated Sparse Attention (GSA)** with **Orthogonal-Constrained Hyper-Connections** using the **Cayley Transform** to achieve computational efficiency, training stability, and restored identity mapping properties.

---

## Table of Contents

1. [Overview](#overview)
2. [Design Rationale](#design-rationale)
3. [Mathematical Formulation](#mathematical-formulation) *(summary - [full details](docs/Mathematical_Theory.md))*
4. [Architecture Diagram](#architecture-diagram)
5. [Data Flow](#data-flow)
6. [Algorithm Details](#algorithm-details)
7. [Infrastructure Optimizations](#infrastructure-optimizations)
8. [Theoretical Properties](#theoretical-properties) *(summary - [full details](docs/Mathematical_Theory.md#theoretical-properties))*
9. [Implementation Considerations](#implementation-considerations)

---

## Overview

**OrthGSA** (Manifold-constrained Gated Sparse Attention) is a novel transformer architecture that synergistically integrates two complementary innovations:

| Component | Source | Contribution |
|-----------|--------|--------------|
| **Gated Sparse Attention** | GSA | Sub-quadratic attention complexity, attention sink elimination |
| **Manifold-Constrained Hyper-Connections** | mHC | Identity mapping preservation via Cayley Transform, multi-stream residual expressiveness |

### Key Benefits

| Property | Standard Transformer | GSA Only | mHC Only | **OrthGSA** |
|----------|---------------------|----------|----------|-------------|
| Attention Complexity | O(L²d) | **O(Lkd)** | O(L²d) | **O(Lkd)** |
| Attention Sinks | Severe | **Eliminated** | Moderate | **Eliminated** |
| Residual Identity Mapping | Preserved | Preserved | **Enhanced** | **Enhanced** |
| Training Stability | Moderate | Good | **Excellent** | **Excellent** |
| Length Extrapolation | Poor | Good | Moderate | **Excellent** |
| Representational Capacity | Baseline | Baseline | **Expanded** | **Expanded** |

---

## Design Rationale

### Why Combine GSA and mHC?

The two innovations address **complementary** aspects of transformer architecture:

**GSA addresses attention-level concerns:**
- Quadratic complexity of self-attention → Sparse top-k selection
- Attention sink phenomenon → Sigmoid gating (bounded activations)
- Training instability from attention → Dual gating (G1/G2)

**mHC addresses residual connection concerns:**
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
│   Residual Stream    │    mHC               │    n-stream expansion        │
│                      │                      │    Cayley Transform          │
│   ─────────────────────────────────────────────────────────                │
│   Attention Layer    │    GSA               │    Sparse selection          │
│                      │                      │    Dual gating (G1, G2)       │
│   ─────────────────────────────────────────────────────────                │
│   FFN Layer          │    Standard/mHC      │    Multi-stream aggregation  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Key Design Decisions

1. **Apply mHC at residual level**: Wrap the entire GSA block with manifold-constrained hyper-connections
2. **Apply GSA within the attention layer**: Replace standard attention with gated sparse attention
3. **Unified gating philosophy**: Both mHC and GSA use sigmoid-based gating for stability
4. **Shared infrastructure**: Leverage kernel fusion from both approaches

---

## Mathematical Formulation

> **Full details**: See [docs/Mathematical_Theory.md](docs/Mathematical_Theory.md) for complete mathematical derivations.

### Core Equations

**OrthGSA Layer Transformation**:

$$\mathbf{x}_{l+1} = \mathbf{H}_l^{\text{res}} \mathbf{x}_l + (\mathbf{H}_l^{\text{post}})^\top \mathcal{F}^{\text{GSA}}(\mathbf{H}_l^{\text{pre}} \mathbf{x}_l, \mathbf{W}_l)$$

**Cayley Transform** (orthogonal projection):

$$\mathbf{H}_l^{\text{res}} = \text{Cayley}(\tilde{\mathbf{H}}_l^{\text{res}}) = (\mathbf{I}_n - \mathbf{A}_l)(\mathbf{I}_n + \mathbf{A}_l)^{-1}$$

where $\mathbf{A}_l = \frac{1}{2}(\tilde{\mathbf{H}}_l^{\text{res}} - \tilde{\mathbf{H}}_l^{\text{res}\top})$ is the skew-symmetric component.

### Key Components

| Component | Formula | Properties |
|-----------|---------|------------|
| **Pre-mapping** | $\mathbf{H}_l^{\text{pre}} = \sigma(\tilde{\mathbf{H}}_l^{\text{pre}})$ | Aggregates n streams |
| **Post-mapping** | $\mathbf{H}_l^{\text{post}} = 2 \cdot \sigma(\tilde{\mathbf{H}}_l^{\text{post}})$ | Distributes to n streams |
| **Residual mapping** | $\mathbf{H}_l^{\text{res}} = \text{Cayley}(\cdot)$ | Orthogonal, $\|\cdot\|_2 = 1$ |

### Cayley Transform Properties

- **Orthogonality**: $\mathbf{H}^{\text{res}\top} \mathbf{H}^{\text{res}} = \mathbf{I}_n$ (exact)
- **Spectral Norm**: $\|\mathbf{H}^{\text{res}}\|_2 = 1$ (unit, not bounded)
- **Identity at Init**: $\text{Cayley}(\mathbf{0}) = \mathbf{I}_n$
- **Single-pass**: No iterative projection needed

---

## Architecture Diagram

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
  │ STEP 2a: Compute mHC Coefficients │     │
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
  │ STEP 2e: mHC + FFN (similar flow) │
  ├───────────────────────────────────┤
  │ // Compute new mHC coefficients   │
  │ H_pre', H_post', H_res' = mHC(x_{l+0.5})
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
| `vec_x` | `[B×L, n×C]` | Flattened for mHC computation |
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
4:      // Compute mHC coefficients (attention)
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
23:     // Compute mHC coefficients (FFN)
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
  y ∈ ℝ^{B×L×C}       : Single-stream input (from mHC pre-mapping)
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

FUSED KERNEL 1: mHC Coefficient Computation
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
│   ...       │                          • mHC coefficients           │
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
│   Recompute: ~15% overhead (mHC coefficients are cheap)            │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 3. Memory Access Pattern Optimization

| Operation | Naive I/O | Optimized I/O | Savings |
|-----------|-----------|---------------|---------|
| mHC Coefficients | 3nC + n² + 2n read, n² + 2n write | nC read, n² + 2n write | 66% |
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
3. Use separate CUDA streams for mHC and GSA computations
4. Prefetch next layer's mHC parameters during current layer's SDPA
```

---

## Theoretical Properties

> **Full details**: See [docs/Mathematical_Theory.md](docs/Mathematical_Theory.md#theoretical-properties) for complete proofs and analysis.

### Summary

| Property | Standard Transformer | OrthGSA Benefit |
|----------|---------------------|-----------------|
| **Attention Complexity** | O(L²d) | **O(Lkd)** - ~(L/k)× speedup |
| **Gradient Flow** | May explode/vanish | **Bounded** - orthogonal $\|\mathbf{H}^{\text{res}}\|_2 = 1$ |
| **Identity Init** | N/A | **Natural** - $\text{Cayley}(\mathbf{0}) = \mathbf{I}$ |
| **Attention Sinks** | ~47% first-token | **<5%** - eliminated via gating |

### Key Theorems

1. **Signal Preservation**: Gradient flow bounded by orthogonal constraint - no explosion regardless of depth
2. **Identity Mapping**: Cayley transform naturally produces $\mathbf{I}_n$ at zero initialization
3. **Bounded Attention**: GSA output gate bounds attention: $\max_{t,s} |\text{Attn}_{t,s}| \leq 1$

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
| mHC alpha init | α | 0.01 | Small for near-identity start |

### 2. Initialization Strategy

```python
def initialize_orthgsa_layer(layer):
    # mHC coefficients - start near identity
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

1. **Warmup schedule for mHC**:
   - First 1000 steps: α = 0.001
   - Ramp up over 5000 steps to target α

2. **Separate learning rates**:
   - mHC parameters: 10× base LR (they're cheap to compute)
   - GSA parameters: 1× base LR

3. **Gradient clipping**:
   - Global norm clipping at 1.0
   - Additional per-layer clipping for mHC coefficients

4. **Mixed precision**:
   - Cayley Transform linear solve in FP32 (numerical stability)
   - Everything else in BF16

### 4. Distributed Training

For multi-GPU training, we recommend **DeepSpeed ZeRO-2** over DDP/FSDP:

| Method | Memory per GPU (4B model) | Use Case |
|--------|---------------------------|----------|
| DDP | ~44GB | Single GPU or 80GB+ GPUs |
| **DeepSpeed ZeRO-2** | **~17GB** | Recommended for 24-48GB GPUs |
| DeepSpeed ZeRO-3 | ~12GB | Memory-constrained setups |

```bash
# DeepSpeed ZeRO-2 (recommended)
deepspeed --num_gpus=4 scripts/train_deepspeed.py --config configs/config_qwen3_4b.yaml

# DDP (requires more memory)
torchrun --nproc_per_node=4 scripts/train.py --config configs/config_qwen3_4b.yaml
```

See `Getting_started.md` for detailed setup instructions.

### 5. Expected Performance

| Model Size | Standard Transformer | OrthGSA | Speedup | Memory |
|------------|---------------------|---------|---------|--------|
| 1B @ 4K | 1.00× | 0.95× | 1.05× | 1.03× |
| 7B @ 32K | 1.00× | 0.42× | 2.4× | 0.85× |
| 7B @ 128K | 1.00× | 0.09× | 11× | 0.65× |
| 27B @ 128K | 1.00× | 0.08× | 12.5× | 0.60× |

*Note: Speedup increases with sequence length due to sparse attention.*

---

## Summary

**OrthGSA** unifies two complementary architectural innovations:

1. **From GSA**: Sub-quadratic sparse attention, attention sink elimination via dual gating
2. **From mHC**: Multi-stream residual connections with orthogonal constraints via **Cayley Transform**

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
2. **mHC**: Manifold-Constrained Hyper-Connections (arXiv:2512.24880)
3. DeepSeek-V3: Pushing the Frontier of Open Large Language Models
4. Gated Attention for Large Language Models (arXiv:2505.06708)
5. Cayley (1846): Sur quelques propriétés des déterminants gauches
6. Helfrich et al. (2018): Orthogonal Recurrent Neural Networks with Scaled Cayley Transform

---

*OrthGSA: Where efficient sparse attention meets stable hyper-connections.*

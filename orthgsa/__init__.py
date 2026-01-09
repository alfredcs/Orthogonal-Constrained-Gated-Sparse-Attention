"""
OrthGSA: Orthogonal-Constrained Gated Sparse Attention

A unified architecture combining Gated Sparse Attention (GSA) with
Orthogonal-Constrained Hyper-Connections using Cayley Transform.
"""

__version__ = "0.1.0"

from .layers.cayley import CayleyTransform, cayley_transform
from .layers.mhc import ManifoldHyperConnection
from .layers.gsa import GatedSparseAttention
from .models.orthgsa_layer import OrthGSALayer
from .models.orthgsa_model import OrthGSAForCausalLM

__all__ = [
    "CayleyTransform",
    "cayley_transform",
    "ManifoldHyperConnection",
    "GatedSparseAttention",
    "OrthGSALayer",
    "OrthGSAForCausalLM",
]

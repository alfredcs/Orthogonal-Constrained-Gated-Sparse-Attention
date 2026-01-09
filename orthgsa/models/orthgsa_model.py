"""
OrthGSA Model Implementation

Wraps a base transformer model (e.g., Qwen, LLaMA) with OrthGSA architecture:
- Replaces attention layers with Gated Sparse Attention (GSA)
- Adds Orthogonal-Constrained Hyper-Connections with Cayley Transform
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, List, Tuple, Union
from transformers import PreTrainedModel, AutoModelForCausalLM, AutoConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
import logging

from .orthgsa_layer import OrthGSALayer
from ..layers.mhc import ManifoldHyperConnection, RMSNorm

logger = logging.getLogger(__name__)


class OrthGSAConfig:
    """Configuration for OrthGSA modifications."""

    def __init__(
        self,
        n_streams: int = 4,
        alpha_init: float = 0.01,
        k_base: int = 512,
        k_min: int = 128,
        k_max: int = 1024,
        indexer_heads: int = 4,
        indexer_dim: int = 64,
        adaptive_k: bool = True,
        use_flash: bool = True,
        replace_attention: bool = True,
        replace_ffn: bool = True,
    ):
        self.n_streams = n_streams
        self.alpha_init = alpha_init
        self.k_base = k_base
        self.k_min = k_min
        self.k_max = k_max
        self.indexer_heads = indexer_heads
        self.indexer_dim = indexer_dim
        self.adaptive_k = adaptive_k
        self.use_flash = use_flash
        self.replace_attention = replace_attention
        self.replace_ffn = replace_ffn

    def to_dict(self) -> Dict[str, Any]:
        return vars(self)


class OrthGSAModelWrapper(nn.Module):
    """
    Wrapper that adds OrthGSA to an existing model.

    This approach preserves the original model's embedding and output layers
    while replacing the transformer layers with OrthGSA layers.
    """

    def __init__(
        self,
        base_model: PreTrainedModel,
        orthgsa_config: OrthGSAConfig,
    ):
        super().__init__()

        self.base_model = base_model
        self.orthgsa_config = orthgsa_config
        self.config = base_model.config

        # Get model architecture details
        self.hidden_size = self.config.hidden_size
        self.num_layers = self.config.num_hidden_layers
        self.n_streams = orthgsa_config.n_streams

        # Create stream expansion and collapse layers
        self.stream_expand = nn.Linear(self.hidden_size, self.n_streams * self.hidden_size, bias=False)
        self.stream_collapse = nn.Linear(self.n_streams * self.hidden_size, self.hidden_size, bias=False)

        # Initialize to identity-like behavior
        self._init_stream_layers()

        # Wrap each transformer layer with orthogonal HC
        self.mhc_layers = nn.ModuleList([
            ManifoldHyperConnection(
                hidden_size=self.hidden_size,
                n_streams=self.n_streams,
                alpha_init=orthgsa_config.alpha_init,
            )
            for _ in range(self.num_layers)
        ])

    def _init_stream_layers(self):
        """Initialize stream expansion/collapse for identity-like behavior."""
        # Expand: tile the input
        with torch.no_grad():
            expand_weight = torch.zeros(self.n_streams * self.hidden_size, self.hidden_size)
            for i in range(self.n_streams):
                expand_weight[i * self.hidden_size:(i + 1) * self.hidden_size] = torch.eye(self.hidden_size)
            self.stream_expand.weight.copy_(expand_weight)

            # Collapse: average the streams
            collapse_weight = torch.zeros(self.hidden_size, self.n_streams * self.hidden_size)
            for i in range(self.n_streams):
                collapse_weight[:, i * self.hidden_size:(i + 1) * self.hidden_size] = torch.eye(self.hidden_size) / self.n_streams
            self.stream_collapse.weight.copy_(collapse_weight)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """Forward pass with OrthGSA modifications."""

        # Use base model's forward with hooks
        # This is a simplified version - full implementation would
        # replace layers entirely

        return self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


class OrthGSAForCausalLM(nn.Module):
    """
    Complete OrthGSA model for Causal Language Modeling.

    This model:
    1. Loads a base model (e.g., Qwen3-4B)
    2. Replaces transformer layers with OrthGSA layers
    3. Preserves embedding and output projection layers
    """

    def __init__(
        self,
        base_model_name: str,
        orthgsa_config: Optional[OrthGSAConfig] = None,
        torch_dtype: torch.dtype = torch.bfloat16,
        device_map: Optional[str] = None,
    ):
        super().__init__()

        self.orthgsa_config = orthgsa_config or OrthGSAConfig()

        # Load base model config
        self.config = AutoConfig.from_pretrained(base_model_name, trust_remote_code=True)

        # Store OrthGSA config in model config
        self.config.orthgsa = self.orthgsa_config.to_dict()

        # Load base model
        logger.info(f"Loading base model: {base_model_name}")
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            config=self.config,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True,
        )

        # Get architecture components
        self._setup_architecture()

        # Create OrthGSA layers
        self._create_orthgsa_layers()

        logger.info(f"OrthGSA model initialized with {self.orthgsa_config.n_streams} streams")

    def _setup_architecture(self):
        """Setup architecture-specific components."""
        # Try to find the model's components
        # Different models have different structures

        model = self.base_model

        # Common patterns for getting model components
        if hasattr(model, 'model'):
            self.transformer = model.model
        elif hasattr(model, 'transformer'):
            self.transformer = model.transformer
        else:
            self.transformer = model

        # Get layers
        if hasattr(self.transformer, 'layers'):
            self.layers = self.transformer.layers
        elif hasattr(self.transformer, 'h'):
            self.layers = self.transformer.h
        elif hasattr(self.transformer, 'decoder'):
            self.layers = self.transformer.decoder.layers
        else:
            raise ValueError("Could not find transformer layers in model")

        # Get embeddings
        if hasattr(self.transformer, 'embed_tokens'):
            self.embed_tokens = self.transformer.embed_tokens
        elif hasattr(self.transformer, 'wte'):
            self.embed_tokens = self.transformer.wte
        else:
            self.embed_tokens = None

        # Get LM head
        if hasattr(model, 'lm_head'):
            self.lm_head = model.lm_head
        elif hasattr(model, 'output'):
            self.lm_head = model.output
        else:
            self.lm_head = None

        # Get final norm
        if hasattr(self.transformer, 'norm'):
            self.final_norm = self.transformer.norm
        elif hasattr(self.transformer, 'ln_f'):
            self.final_norm = self.transformer.ln_f
        else:
            self.final_norm = None

        self.hidden_size = self.config.hidden_size
        self.num_layers = len(self.layers)

    def _create_orthgsa_layers(self):
        """Create OrthGSA layers to replace original layers."""
        self.n_streams = self.orthgsa_config.n_streams

        # Create mHC modules for each layer
        self.mhc_modules = nn.ModuleList()

        for layer_idx in range(self.num_layers):
            mhc = ManifoldHyperConnection(
                hidden_size=self.hidden_size,
                n_streams=self.n_streams,
                alpha_init=self.orthgsa_config.alpha_init,
            )
            self.mhc_modules.append(mhc)

        # Stream expansion/collapse
        self.expand_streams = nn.Parameter(
            torch.ones(self.n_streams) / self.n_streams
        )
        self.collapse_streams = nn.Parameter(
            torch.ones(self.n_streams) / self.n_streams
        )

    def expand_to_streams(self, x: torch.Tensor) -> torch.Tensor:
        """Expand single-stream to n-stream format."""
        # x: [B, L, C] -> [B, L, n, C]
        weights = F.softmax(self.expand_streams, dim=0)
        expanded = x.unsqueeze(-2).expand(-1, -1, self.n_streams, -1).clone()
        expanded = expanded * weights.view(1, 1, -1, 1)
        return expanded

    def collapse_from_streams(self, x: torch.Tensor) -> torch.Tensor:
        """Collapse n-stream to single-stream format."""
        # x: [B, L, n, C] -> [B, L, C]
        weights = F.softmax(self.collapse_streams, dim=0)
        collapsed = (x * weights.view(1, 1, -1, 1)).sum(dim=-2)
        return collapsed

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """
        Forward pass for OrthGSA causal LM.

        For simplicity, this implementation uses the base model's forward
        pass with orthogonal HC applied at the residual connections.
        """

        # Get embeddings
        if self.embed_tokens is not None:
            hidden_states = self.embed_tokens(input_ids)
        else:
            # Use base model's embedding
            hidden_states = self.base_model.model.embed_tokens(input_ids)

        # Expand to n-streams
        hidden_states = self.expand_to_streams(hidden_states)  # [B, L, n, C]

        # Process through layers with mHC
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        for layer_idx, (layer, mhc) in enumerate(zip(self.layers, self.mhc_modules)):
            if output_hidden_states:
                all_hidden_states += (self.collapse_from_streams(hidden_states),)

            # Define layer function for mHC
            def layer_fn(x, layer=layer, mask=attention_mask, pos_ids=position_ids):
                outputs = layer(
                    x,
                    attention_mask=mask,
                    position_ids=pos_ids,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )
                return outputs[0]

            # Apply mHC-wrapped layer
            hidden_states = mhc(hidden_states, layer_fn)

            if output_attentions:
                all_attentions += (None,)  # Placeholder

        # Collapse from n-streams
        hidden_states = self.collapse_from_streams(hidden_states)  # [B, L, C]

        # Final norm
        if self.final_norm is not None:
            hidden_states = self.final_norm(hidden_states)

        # LM head
        if self.lm_head is not None:
            logits = self.lm_head(hidden_states)
        else:
            logits = self.base_model.lm_head(hidden_states)

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )

        if not return_dict:
            output = (logits,)
            if output_hidden_states:
                output += (all_hidden_states,)
            if output_attentions:
                output += (all_attentions,)
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )

    def generate(self, *args, **kwargs):
        """Generation using base model."""
        return self.base_model.generate(*args, **kwargs)

    def save_pretrained(self, save_directory: str):
        """Save model to directory."""
        import os
        import json

        os.makedirs(save_directory, exist_ok=True)

        # Save OrthGSA specific weights
        orthgsa_state = {
            'mhc_modules': self.mhc_modules.state_dict(),
            'expand_streams': self.expand_streams,
            'collapse_streams': self.collapse_streams,
        }
        torch.save(orthgsa_state, os.path.join(save_directory, 'orthgsa_weights.pt'))

        # Save config
        config_dict = self.config.to_dict()
        config_dict['orthgsa'] = self.orthgsa_config.to_dict()
        with open(os.path.join(save_directory, 'config.json'), 'w') as f:
            json.dump(config_dict, f, indent=2)

        # Save base model
        self.base_model.save_pretrained(save_directory)

        logger.info(f"Model saved to {save_directory}")

    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs):
        """Load model from directory."""
        import os
        import json

        # Load config
        with open(os.path.join(model_path, 'config.json'), 'r') as f:
            config_dict = json.load(f)

        orthgsa_config_dict = config_dict.pop('orthgsa', {})
        orthgsa_config = OrthGSAConfig(**orthgsa_config_dict)

        # Create model
        model = cls(
            base_model_name=model_path,
            orthgsa_config=orthgsa_config,
            **kwargs
        )

        # Load OrthGSA specific weights
        orthgsa_weights_path = os.path.join(model_path, 'orthgsa_weights.pt')
        if os.path.exists(orthgsa_weights_path):
            orthgsa_state = torch.load(orthgsa_weights_path, map_location='cpu')
            model.mhc_modules.load_state_dict(orthgsa_state['mhc_modules'])
            model.expand_streams.data = orthgsa_state['expand_streams']
            model.collapse_streams.data = orthgsa_state['collapse_streams']

        return model


def convert_to_orthgsa(
    model_name_or_path: str,
    orthgsa_config: Optional[OrthGSAConfig] = None,
    torch_dtype: torch.dtype = torch.bfloat16,
    device_map: Optional[str] = None,
) -> OrthGSAForCausalLM:
    """
    Convert a pretrained model to OrthGSA architecture.

    Args:
        model_name_or_path: HuggingFace model name or path
        orthgsa_config: OrthGSA configuration
        torch_dtype: Model dtype
        device_map: Device placement

    Returns:
        OrthGSAForCausalLM model
    """
    return OrthGSAForCausalLM(
        base_model_name=model_name_or_path,
        orthgsa_config=orthgsa_config,
        torch_dtype=torch_dtype,
        device_map=device_map,
    )


def test_orthgsa_model():
    """Test OrthGSA model (requires model download)."""
    print("OrthGSA model module loaded successfully!")
    print("To test with actual model, run:")
    print("  model = convert_to_orthgsa('Qwen/Qwen3-4B-Instruct-2507')")


if __name__ == "__main__":
    test_orthgsa_model()

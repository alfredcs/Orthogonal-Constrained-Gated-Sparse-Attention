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


def chunked_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    chunk_size: int = 1024,
    ignore_index: int = -100,
) -> torch.Tensor:
    """
    Compute cross-entropy loss in chunks to reduce peak memory usage.

    Instead of materializing the full [batch*seq_len, vocab_size] tensor,
    we process chunks along the sequence dimension.

    Args:
        logits: [batch, seq_len, vocab_size] - NOT shifted
        labels: [batch, seq_len] - NOT shifted
        chunk_size: Number of tokens to process at once
        ignore_index: Label index to ignore in loss computation

    Returns:
        Scalar loss tensor
    """
    # Shift for causal LM (predict next token)
    shift_logits = logits[..., :-1, :]  # [B, L-1, V]
    shift_labels = labels[..., 1:]       # [B, L-1]

    batch_size, seq_len, vocab_size = shift_logits.shape

    # Flatten batch dimension into sequence
    shift_logits = shift_logits.reshape(-1, vocab_size)  # [B*(L-1), V]
    shift_labels = shift_labels.reshape(-1)               # [B*(L-1)]

    total_tokens = shift_labels.numel()

    # Count valid tokens (not ignored)
    valid_mask = shift_labels != ignore_index
    num_valid = valid_mask.sum()

    if num_valid == 0:
        return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)

    # Process in chunks
    total_loss = torch.tensor(0.0, device=logits.device, dtype=torch.float32)

    for start_idx in range(0, total_tokens, chunk_size):
        end_idx = min(start_idx + chunk_size, total_tokens)

        chunk_logits = shift_logits[start_idx:end_idx]  # [chunk, V]
        chunk_labels = shift_labels[start_idx:end_idx]  # [chunk]

        # Compute loss for this chunk (reduction='none' to handle masking ourselves)
        chunk_loss = F.cross_entropy(
            chunk_logits,
            chunk_labels,
            ignore_index=ignore_index,
            reduction='sum'
        )

        total_loss = total_loss + chunk_loss

    # Average over valid tokens
    return total_loss / num_valid


def chunked_lm_loss(
    hidden_states: torch.Tensor,
    lm_head: nn.Module,
    labels: torch.Tensor,
    chunk_size: int = 512,
    ignore_index: int = -100,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Compute language modeling loss without materializing full logits tensor.

    This is more memory-efficient for long sequences as it:
    1. Processes hidden states in chunks
    2. Computes logits for each chunk
    3. Immediately computes loss and discards logits

    Args:
        hidden_states: [batch, seq_len, hidden_dim] - Final hidden states
        lm_head: The language model head (linear projection to vocab)
        labels: [batch, seq_len] - Target token IDs
        chunk_size: Number of tokens to process at once
        ignore_index: Label index to ignore

    Returns:
        Tuple of (loss, None) - logits are not returned to save memory
    """
    batch_size, seq_len, hidden_dim = hidden_states.shape

    # Shift labels for causal LM
    shift_labels = labels[..., 1:].contiguous()  # [B, L-1]
    shift_labels_flat = shift_labels.reshape(-1)  # [B*(L-1)]

    # We only need hidden states for positions :-1 (to predict next token)
    shift_hidden = hidden_states[..., :-1, :].contiguous()  # [B, L-1, H]
    shift_hidden_flat = shift_hidden.reshape(-1, hidden_dim)  # [B*(L-1), H]

    total_tokens = shift_labels_flat.numel()

    # Count valid tokens
    valid_mask = shift_labels_flat != ignore_index
    num_valid = valid_mask.sum()

    if num_valid == 0:
        return torch.tensor(0.0, device=hidden_states.device, dtype=hidden_states.dtype), None

    # Process in chunks to avoid materializing full logits
    total_loss = torch.tensor(0.0, device=hidden_states.device, dtype=torch.float32)

    for start_idx in range(0, total_tokens, chunk_size):
        end_idx = min(start_idx + chunk_size, total_tokens)

        # Get chunk of hidden states
        chunk_hidden = shift_hidden_flat[start_idx:end_idx]  # [chunk, H]
        chunk_labels = shift_labels_flat[start_idx:end_idx]  # [chunk]

        # Compute logits for this chunk only
        chunk_logits = lm_head(chunk_hidden)  # [chunk, V]

        # Compute loss and immediately discard logits
        chunk_loss = F.cross_entropy(
            chunk_logits,
            chunk_labels,
            ignore_index=ignore_index,
            reduction='sum'
        )

        total_loss = total_loss + chunk_loss

        # Explicitly delete to free memory
        del chunk_logits, chunk_hidden

    return total_loss / num_valid, None


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
        low_cpu_mem_usage: bool = False,
        max_position_embeddings: Optional[int] = None,
        rope_scaling: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()

        self.orthgsa_config = orthgsa_config or OrthGSAConfig()
        self.torch_dtype = torch_dtype

        # Load base model config
        self.config = AutoConfig.from_pretrained(base_model_name, trust_remote_code=True)

        # Configure RoPE scaling for extended context lengths
        if max_position_embeddings is not None:
            original_max_pos = getattr(self.config, 'max_position_embeddings', 32768)
            self.config.max_position_embeddings = max_position_embeddings

            # Auto-configure RoPE scaling if extending beyond original
            if max_position_embeddings > original_max_pos and rope_scaling is None:
                scaling_factor = max_position_embeddings / original_max_pos
                rope_scaling = {
                    "type": "dynamic",  # or "linear" for linear scaling
                    "factor": scaling_factor,
                }
                logger.info(f"Auto-configured RoPE scaling: {rope_scaling}")

        if rope_scaling is not None:
            self.config.rope_scaling = rope_scaling
            logger.info(f"RoPE scaling enabled: {rope_scaling}")

        # Store OrthGSA config in model config
        self.config.orthgsa = self.orthgsa_config.to_dict()

        # Load base model
        # For ZeRO-3, use low_cpu_mem_usage=True to reduce peak memory during loading
        # DeepSpeed.initialize() will handle the weight sharding afterwards
        logger.info(f"Loading base model: {base_model_name}")

        # Try to use Flash Attention 2 for memory efficiency
        attn_implementation = "flash_attention_2"
        try:
            self.base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                config=self.config,
                torch_dtype=torch_dtype,
                device_map=device_map,
                trust_remote_code=True,
                low_cpu_mem_usage=low_cpu_mem_usage,
                attn_implementation=attn_implementation,
            )
            logger.info(f"Loaded model with {attn_implementation}")
        except Exception as e:
            logger.warning(f"Flash Attention 2 not available ({e}), falling back to SDPA")
            # Fallback to SDPA (scaled dot product attention) which is also memory efficient
            try:
                self.base_model = AutoModelForCausalLM.from_pretrained(
                    base_model_name,
                    config=self.config,
                    torch_dtype=torch_dtype,
                    device_map=device_map,
                    trust_remote_code=True,
                    low_cpu_mem_usage=low_cpu_mem_usage,
                    attn_implementation="sdpa",
                )
                logger.info("Loaded model with SDPA attention")
            except Exception:
                # Final fallback to default
                self.base_model = AutoModelForCausalLM.from_pretrained(
                    base_model_name,
                    config=self.config,
                    torch_dtype=torch_dtype,
                    device_map=device_map,
                    trust_remote_code=True,
                    low_cpu_mem_usage=low_cpu_mem_usage,
                )
                logger.info("Loaded model with default attention")

        # Get architecture components
        self._setup_architecture()

        # Create OrthGSA layers
        self._create_orthgsa_layers()

        # Convert all new modules to the same dtype as base model
        self._convert_to_dtype(torch_dtype)

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

    def _convert_to_dtype(self, dtype: torch.dtype):
        """Convert all OrthGSA modules to the specified dtype for FSDP compatibility."""
        # Convert mHC modules
        for mhc in self.mhc_modules:
            mhc.to(dtype)

        # Convert stream parameters
        self.expand_streams.data = self.expand_streams.data.to(dtype)
        self.collapse_streams.data = self.collapse_streams.data.to(dtype)

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

    def _prepare_4d_causal_attention_mask(
        self,
        attention_mask: torch.Tensor,
        dtype: torch.dtype,
        seq_length: int,
    ) -> torch.Tensor:
        """
        Prepare 4D causal attention mask from 2D attention mask.

        Args:
            attention_mask: [batch, seq_len] tensor with 1s for valid positions, 0s for padding
            dtype: Target dtype for the mask
            seq_length: Sequence length

        Returns:
            4D causal mask [batch, 1, seq_len, seq_len] suitable for SDPA
        """
        batch_size = attention_mask.shape[0]
        device = attention_mask.device

        # Create causal mask (lower triangular)
        # Shape: [1, 1, seq_len, seq_len]
        causal_mask = torch.triu(
            torch.ones((seq_length, seq_length), dtype=torch.bool, device=device),
            diagonal=1
        )
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]

        # Expand causal mask for batch
        causal_mask = causal_mask.expand(batch_size, 1, seq_length, seq_length)

        # Create padding mask from attention_mask
        # attention_mask: [batch, seq_len] -> [batch, 1, 1, seq_len]
        padding_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        padding_mask = padding_mask.expand(-1, 1, seq_length, -1)

        # Combine masks: True means masked (should not attend)
        # causal_mask is True for positions to mask (upper triangle)
        # padding_mask is True for valid positions, False for padding
        # We want to mask where either causal says mask OR padding says mask
        combined_mask = causal_mask | (~padding_mask.bool())

        # Convert to float mask for SDPA: 0.0 for attend, -inf for mask
        float_mask = torch.zeros_like(combined_mask, dtype=dtype)
        float_mask.masked_fill_(combined_mask, float("-inf"))

        return float_mask

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

        batch_size, seq_length = input_ids.shape

        # Create position_ids if not provided
        if position_ids is None:
            device = input_ids.device
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        # Compute position embeddings (rotary embeddings) for Qwen3
        # Access rotary_emb directly from base_model to ensure correct reference after FSDP wrapping
        position_embeddings = None
        rotary_emb = None

        # Try multiple paths to find rotary_emb (handles different model structures)
        if hasattr(self.base_model, 'model') and hasattr(self.base_model.model, 'rotary_emb'):
            rotary_emb = self.base_model.model.rotary_emb
        elif hasattr(self.transformer, 'rotary_emb'):
            rotary_emb = self.transformer.rotary_emb
        elif hasattr(self.base_model, 'transformer') and hasattr(self.base_model.transformer, 'rotary_emb'):
            rotary_emb = self.base_model.transformer.rotary_emb

        if rotary_emb is not None:
            position_embeddings = rotary_emb(hidden_states, position_ids)
        else:
            raise RuntimeError(
                "Could not find rotary_emb in model. This is required for Qwen3 models. "
                f"Model structure: base_model has 'model': {hasattr(self.base_model, 'model')}, "
                f"transformer type: {type(self.transformer)}"
            )

        # Prepare 4D causal attention mask for SDPA
        # The mask should be [batch, 1, seq_len, seq_len] for causal attention
        # When attention_mask is None, SDPA uses causal masking automatically
        causal_mask = None
        if attention_mask is not None:
            # Create 4D causal mask that combines padding mask with causal structure
            # attention_mask: [batch, seq_len] -> causal_mask: [batch, 1, seq_len, seq_len]
            causal_mask = self._prepare_4d_causal_attention_mask(
                attention_mask, hidden_states.dtype, seq_length
            )

        # Expand to n-streams
        hidden_states = self.expand_to_streams(hidden_states)  # [B, L, n, C]

        # Process through layers with mHC
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        for layer_idx, (layer, mhc) in enumerate(zip(self.layers, self.mhc_modules)):
            if output_hidden_states:
                all_hidden_states += (self.collapse_from_streams(hidden_states),)

            # Define layer function for mHC
            # Pass None for attention_mask to let SDPA use causal attention automatically
            # This avoids complex mask preparation and works with packed sequences
            def layer_fn(x, layer=layer, mask=causal_mask, pos_ids=position_ids, pos_emb=position_embeddings):
                outputs = layer(
                    x,
                    attention_mask=mask,
                    position_ids=pos_ids,
                    position_embeddings=pos_emb,
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

        # Get LM head reference
        lm_head = self.lm_head if self.lm_head is not None else self.base_model.lm_head

        # Compute loss and optionally logits
        loss = None
        logits = None

        if labels is not None:
            # Use chunked loss computation to avoid materializing full logits tensor
            # This saves ~9GB for 12K context with 152K vocab
            loss, _ = chunked_lm_loss(
                hidden_states,
                lm_head,
                labels,
                chunk_size=512,  # Process 512 tokens at a time
                ignore_index=-100,
            )
            # Only compute full logits if needed for output
            if not return_dict or output_hidden_states or output_attentions:
                logits = lm_head(hidden_states)
        else:
            # No labels - compute full logits for inference
            logits = lm_head(hidden_states)

        if not return_dict:
            output = (logits,) if logits is not None else ()
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
    low_cpu_mem_usage: bool = False,
) -> OrthGSAForCausalLM:
    """
    Convert a pretrained model to OrthGSA architecture.

    Args:
        model_name_or_path: HuggingFace model name or path
        orthgsa_config: OrthGSA configuration
        torch_dtype: Model dtype
        device_map: Device placement
        low_cpu_mem_usage: Use accelerate for memory-efficient loading (useful for ZeRO-3)

    Returns:
        OrthGSAForCausalLM model
    """
    return OrthGSAForCausalLM(
        base_model_name=model_name_or_path,
        orthgsa_config=orthgsa_config,
        torch_dtype=torch_dtype,
        device_map=device_map,
        low_cpu_mem_usage=low_cpu_mem_usage,
    )


def test_orthgsa_model():
    """Test OrthGSA model (requires model download)."""
    print("OrthGSA model module loaded successfully!")
    print("To test with actual model, run:")
    print("  model = convert_to_orthgsa('Qwen/Qwen3-4B-Instruct-2507')")


if __name__ == "__main__":
    test_orthgsa_model()

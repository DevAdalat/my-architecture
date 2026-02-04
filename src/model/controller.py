"""TinyController: The executive center of DPSN-R.

The controller is a minimal, high-speed Transformer that handles:
1. Encoding input context into latent representations
2. Generating retrieval queries for the Parameter Pool
3. Integrating retrieved knowledge into the thought vector
4. Deciding when to stop thinking (halt prediction for ACT)

XLA Compliance:
- All shapes are static (no data-dependent operations)
- Dimensions are multiples of 128 for TPU efficiency
- No .item() calls or Python control flow on tensors
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .config import DPSNRConfig


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    RMSNorm is simpler and faster than LayerNorm while achieving
    similar performance. It normalizes by RMS without centering.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        """Initialize RMSNorm.

        Args:
            dim: Feature dimension.
            eps: Small constant for numerical stability.
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMS normalization.

        Args:
            x: Input tensor of shape [..., dim].

        Returns:
            Normalized tensor of same shape.
        """
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class RotaryPositionalEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) for position-aware attention.

    RoPE encodes position information directly into query/key representations
    using rotation matrices, enabling better length generalization.
    """

    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0) -> None:
        """Initialize RoPE.

        Args:
            dim: Dimension of the embedding (must be even).
            max_seq_len: Maximum sequence length.
            base: Base for frequency computation.
        """
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len

        # Compute inverse frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Precompute cos/sin cache for max_seq_len
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        """Build cos/sin cache for given sequence length."""
        positions = torch.arange(seq_len, dtype=torch.float32)
        freqs = torch.outer(positions, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cache", emb.cos(), persistent=False)
        self.register_buffer("sin_cache", emb.sin(), persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get cos/sin embeddings for sequence length.

        Args:
            x: Input tensor (used for device/dtype).
            seq_len: Current sequence length.

        Returns:
            Tuple of (cos, sin) embeddings of shape [seq_len, dim].
        """
        return (
            self.cos_cache[:seq_len].to(x.dtype),
            self.sin_cache[:seq_len].to(x.dtype),
        )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary positional embedding to query and key tensors.

    Args:
        q: Query tensor of shape [batch, heads, seq, head_dim].
        k: Key tensor of shape [batch, heads, seq, head_dim].
        cos: Cosine embeddings of shape [seq, head_dim].
        sin: Sine embeddings of shape [seq, head_dim].

    Returns:
        Tuple of rotated (query, key) tensors.
    """
    # Reshape cos/sin for broadcasting: [1, 1, seq, dim]
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class MultiHeadAttention(nn.Module):
    """Multi-Head Self-Attention with RoPE.

    XLA-compliant implementation with static shapes and no data-dependent ops.
    """

    def __init__(self, config: DPSNRConfig) -> None:
        """Initialize attention module.

        Args:
            config: Model configuration.
        """
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.hidden_dim = config.hidden_dim
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # QKV projection
        self.qkv_proj = nn.Linear(config.hidden_dim, 3 * config.hidden_dim, bias=False)
        self.out_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)

        # RoPE
        self.rope = RotaryPositionalEmbedding(config.head_dim, config.max_seq_len)

        # Dropout
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape [batch, seq, hidden_dim].
            attention_mask: Optional mask of shape [batch, 1, seq, seq].

        Returns:
            Output tensor of shape [batch, seq, hidden_dim].
        """
        batch_size, seq_len, _ = x.shape

        # QKV projection
        qkv = self.qkv_proj(x)
        qkv = rearrange(qkv, "b s (three h d) -> three b h s d", three=3, h=self.num_heads)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Apply RoPE
        cos, sin = self.rope(x, seq_len)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Scaled dot-product attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Apply causal mask for autoregressive attention
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device),
            diagonal=1,
        )
        attn_weights = attn_weights.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), -1e9)

        # Apply optional attention mask
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Attention output
        attn_output = torch.matmul(attn_weights, v)
        attn_output = rearrange(attn_output, "b h s d -> b s (h d)")

        return self.out_proj(attn_output)


class FeedForward(nn.Module):
    """Feed-forward network with SwiGLU activation.

    SwiGLU provides better performance than standard ReLU/GELU activations.
    """

    def __init__(self, config: DPSNRConfig) -> None:
        """Initialize FFN.

        Args:
            config: Model configuration.
        """
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_dim, config.intermediate_dim, bias=False)
        self.up_proj = nn.Linear(config.hidden_dim, config.intermediate_dim, bias=False)
        self.down_proj = nn.Linear(config.intermediate_dim, config.hidden_dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with SwiGLU activation.

        Args:
            x: Input tensor of shape [batch, seq, hidden_dim].

        Returns:
            Output tensor of shape [batch, seq, hidden_dim].
        """
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.dropout(self.down_proj(gate * up))


class TransformerBlock(nn.Module):
    """Single Transformer block with pre-norm architecture."""

    def __init__(self, config: DPSNRConfig) -> None:
        """Initialize transformer block.

        Args:
            config: Model configuration.
        """
        super().__init__()
        self.attn_norm = RMSNorm(config.hidden_dim)
        self.attention = MultiHeadAttention(config)
        self.ffn_norm = RMSNorm(config.hidden_dim)
        self.ffn = FeedForward(config)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape [batch, seq, hidden_dim].
            attention_mask: Optional attention mask.

        Returns:
            Output tensor of shape [batch, seq, hidden_dim].
        """
        # Pre-norm attention with residual
        x = x + self.attention(self.attn_norm(x), attention_mask)
        # Pre-norm FFN with residual
        x = x + self.ffn(self.ffn_norm(x))
        return x


class TinyController(nn.Module):
    """The executive center of DPSN-R.

    A minimal, high-speed Transformer that encodes context, generates
    retrieval queries, and decides when to halt reasoning.
    """

    def __init__(self, config: DPSNRConfig) -> None:
        """Initialize controller.

        Args:
            config: Model configuration.
        """
        super().__init__()
        self.config = config

        # Token and step embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.step_embedding = nn.Embedding(config.max_reasoning_steps, config.hidden_dim)

        # Transformer blocks
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])

        # Final norm
        self.final_norm = RMSNorm(config.hidden_dim)

        # Query generator for pool retrieval
        self.query_proj = nn.Linear(config.hidden_dim, config.retrieval_dim, bias=False)

        # Halt predictor for ACT
        self.halt_proj = nn.Linear(config.hidden_dim, 1, bias=True)

        # Integration layer for retrieved knowledge
        self.integration_gate = nn.Linear(config.hidden_dim + config.pool_dim, config.hidden_dim)
        self.integration_proj = nn.Linear(config.pool_dim, config.hidden_dim, bias=False)

        # Output projection (LM head)
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)

        # Tie weights between embedding and LM head
        self.lm_head.weight = self.token_embedding.weight

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights with small values for stability."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def encode(
        self,
        input_ids: torch.Tensor,
        step: int = 0,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode input tokens into hidden representations.

        Args:
            input_ids: Token IDs of shape [batch, seq].
            step: Current reasoning step (0 to max_reasoning_steps-1).
            attention_mask: Optional attention mask.

        Returns:
            Hidden states of shape [batch, seq, hidden_dim].
        """
        batch_size, seq_len = input_ids.shape

        # Token embeddings
        x = self.token_embedding(input_ids)

        # Add step embedding (broadcast across sequence)
        step_idx = torch.tensor(
            min(step, self.config.max_reasoning_steps - 1),
            device=input_ids.device,
        )
        step_emb = self.step_embedding(step_idx)
        x = x + step_emb.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1)

        # Apply transformer blocks
        for layer in self.layers:
            x = layer(x, attention_mask)

        return self.final_norm(x)

    def generate_query(self, hidden: torch.Tensor) -> torch.Tensor:
        """Generate retrieval query from hidden states.

        Args:
            hidden: Hidden states of shape [batch, seq, hidden_dim].

        Returns:
            Query vectors of shape [batch, seq, retrieval_dim].
        """
        return self.query_proj(hidden)

    def predict_halt(self, hidden: torch.Tensor) -> torch.Tensor:
        """Predict halting probability for ACT.

        Args:
            hidden: Hidden states of shape [batch, seq, hidden_dim].

        Returns:
            Halt probabilities of shape [batch, seq, 1] in range [0, 1].
        """
        return torch.sigmoid(self.halt_proj(hidden))

    def integrate(
        self,
        hidden: torch.Tensor,
        retrieved: torch.Tensor,
    ) -> torch.Tensor:
        """Integrate retrieved knowledge into hidden state.

        Uses a gated mechanism to selectively incorporate retrieved information.

        Args:
            hidden: Current hidden state of shape [batch, seq, hidden_dim].
            retrieved: Retrieved vectors of shape [batch, seq, pool_dim].

        Returns:
            Updated hidden state of shape [batch, seq, hidden_dim].
        """
        # Project retrieved to hidden dimension
        retrieved_proj = self.integration_proj(retrieved)

        # Compute gate
        gate_input = torch.cat([hidden, retrieved], dim=-1)
        gate = torch.sigmoid(self.integration_gate(gate_input))

        # Gated update with residual
        return hidden + gate * retrieved_proj

    def decode(self, hidden: torch.Tensor) -> torch.Tensor:
        """Decode hidden states to vocabulary logits.

        Args:
            hidden: Hidden states of shape [batch, seq, hidden_dim].

        Returns:
            Logits of shape [batch, seq, vocab_size].
        """
        return self.lm_head(hidden)

"""TinyController: The executive center of DPSN-R.

XLA Compliance:
- All shapes are static (no data-dependent operations)
- Dimensions are multiples of 128 for TPU efficiency
- No .item() calls or Python control flow on tensors
- Causal masks are precomputed and cached as buffers
- No einops - use native torch reshape/transpose
- Step embeddings use direct indexing (no torch.tensor in forward)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import DPSNRConfig


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class RotaryPositionalEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) with precomputed cache."""

    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        positions = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cache", emb.cos(), persistent=False)
        self.register_buffer("sin_cache", emb.sin(), persistent=False)

    def forward(self, seq_len: int, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        """Get cached cos/sin for seq_len. No dynamic computation."""
        return self.cos_cache[:seq_len].to(dtype), self.sin_cache[:seq_len].to(dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class MultiHeadAttention(nn.Module):
    """Multi-Head Self-Attention with RoPE and precomputed causal mask."""

    def __init__(self, config: DPSNRConfig) -> None:
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.hidden_dim = config.hidden_dim
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.qkv_proj = nn.Linear(config.hidden_dim, 3 * config.hidden_dim, bias=False)
        self.out_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)

        self.rope = RotaryPositionalEmbedding(config.head_dim, config.max_seq_len)
        self.dropout = nn.Dropout(config.dropout)

        # Precompute causal mask as buffer (XLA: no dynamic creation)
        causal_mask = torch.triu(
            torch.ones(config.max_seq_len, config.max_seq_len, dtype=torch.bool),
            diagonal=1,
        )
        self.register_buffer("causal_mask", causal_mask, persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        # QKV projection - use native reshape instead of einops
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch, heads, seq, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Apply RoPE
        cos, sin = self.rope(seq_len, x.dtype)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Scaled dot-product attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Use precomputed causal mask (slice to current seq_len)
        mask = self.causal_mask[:seq_len, :seq_len]

        # Use min value for current dtype to avoid overflow in float16
        min_value = torch.finfo(attn_weights.dtype).min
        attn_weights = attn_weights.masked_fill(mask.unsqueeze(0).unsqueeze(0), min_value)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Attention output - use native reshape instead of einops
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.hidden_dim)

        return self.out_proj(attn_output)


class FeedForward(nn.Module):
    """Feed-forward network with SwiGLU activation."""

    def __init__(self, config: DPSNRConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_dim, config.intermediate_dim, bias=False)
        self.up_proj = nn.Linear(config.hidden_dim, config.intermediate_dim, bias=False)
        self.down_proj = nn.Linear(config.intermediate_dim, config.hidden_dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.dropout(self.down_proj(gate * up))


class TransformerBlock(nn.Module):
    """Single Transformer block with pre-norm architecture."""

    def __init__(self, config: DPSNRConfig) -> None:
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
        x = x + self.attention(self.attn_norm(x), attention_mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class TinyController(nn.Module):
    """The executive center of DPSN-R.

    XLA-Optimized: No dynamic tensor creation, precomputed buffers.
    """

    def __init__(self, config: DPSNRConfig) -> None:
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.step_embedding = nn.Embedding(config.max_reasoning_steps, config.hidden_dim)

        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])
        self.final_norm = RMSNorm(config.hidden_dim)

        self.query_proj = nn.Linear(config.hidden_dim, config.retrieval_dim, bias=False)
        self.halt_proj = nn.Linear(config.hidden_dim, 1, bias=True)

        self.integration_gate = nn.Linear(config.hidden_dim + config.pool_dim, config.hidden_dim)
        self.integration_proj = nn.Linear(config.pool_dim, config.hidden_dim, bias=False)

        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

        # Precompute step indices as buffer (XLA: avoid torch.tensor in forward)
        step_indices = torch.arange(config.max_reasoning_steps, dtype=torch.long)
        self.register_buffer("step_indices", step_indices, persistent=False)

        self._init_weights()

    def _init_weights(self) -> None:
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
        """Encode input tokens. Step is a Python int (compile-time constant)."""
        batch_size, seq_len = input_ids.shape

        x = self.token_embedding(input_ids)

        # Use precomputed step index buffer (no torch.tensor call)
        step_idx = min(step, self.config.max_reasoning_steps - 1)
        step_emb = self.step_embedding(self.step_indices[step_idx])
        x = x + step_emb.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1)

        for layer in self.layers:
            x = layer(x, attention_mask)

        return self.final_norm(x)

    def generate_query(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.query_proj(hidden)

    def predict_halt(self, hidden: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.halt_proj(hidden))

    def integrate(
        self,
        hidden: torch.Tensor,
        retrieved: torch.Tensor,
    ) -> torch.Tensor:
        retrieved_proj = self.integration_proj(retrieved)
        gate_input = torch.cat([hidden, retrieved], dim=-1)
        gate = torch.sigmoid(self.integration_gate(gate_input))
        return hidden + gate * retrieved_proj

    def decode(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden)

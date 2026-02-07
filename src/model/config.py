"""Configuration for DPSN-R model.

All dimensions follow XLA guidelines:
- Hidden dimensions are multiples of 128 for TPU systolic array efficiency.
- Shapes are static to enable graph compilation and caching.
"""

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class DPSNRConfig:
    """Configuration for DPSN-R architecture.

    Attributes:
        vocab_size: Size of the vocabulary.
        hidden_dim: Hidden dimension of the controller (must be multiple of 128).
        num_layers: Number of transformer layers in the controller.
        num_heads: Number of attention heads.
        head_dim: Dimension per attention head.
        intermediate_dim: FFN intermediate dimension.
        pool_size: Total number of vectors in the parameter pool.
        pool_dim: Dimension of each pool vector (must be multiple of 128).
        top_k: Number of vectors to retrieve per query (must be multiple of 128).
        retrieval_dim: Dimension of retrieval query vectors.
        max_reasoning_steps: Maximum recurrent reasoning loops per token.
        max_seq_len: Maximum sequence length.
        dropout: Dropout probability.
        knowledge_ratio: Fraction of pool for knowledge partition.
        reasoning_ratio: Fraction of pool for reasoning partition.
        grammar_ratio: Fraction of pool for grammar partition.
        act_threshold: Halting probability threshold for ACT.
        ponder_lambda: Weight for ponder loss in total loss.
    """

    # Vocabulary and sequence
    vocab_size: int = 32000
    max_seq_len: int = 2048

    # Controller dimensions (all multiples of 128)
    hidden_dim: int = 512
    num_layers: int = 4
    num_heads: int = 8
    head_dim: int = 64
    intermediate_dim: int = 2048
    dropout: float = 0.1

    # Pool dimensions (all multiples of 128)
    pool_size: int = 100000
    pool_dim: int = 256
    top_k: int = 512
    retrieval_dim: int = 128

    # Recurrent reasoning
    max_reasoning_steps: int = 8
    act_threshold: float = 0.99
    ponder_lambda: float = 0.01

    # Pool partitions (ratios)
    knowledge_ratio: float = 0.7
    reasoning_ratio: float = 0.2
    grammar_ratio: float = 0.1

    # Computed partition sizes (set in __post_init__)
    knowledge_size: int = field(init=False)
    reasoning_size: int = field(init=False)
    grammar_size: int = field(init=False)

    def __post_init__(self) -> None:
        """Validate config and compute derived values."""
        # Validate multiples of 128 for XLA efficiency
        if self.hidden_dim % 128 != 0:
            raise ValueError(f"hidden_dim must be multiple of 128, got {self.hidden_dim}")
        if self.pool_dim % 128 != 0:
            raise ValueError(f"pool_dim must be multiple of 128, got {self.pool_dim}")
        if self.top_k % 32 != 0:
            raise ValueError(f"top_k must be multiple of 32, got {self.top_k}")

        # Validate ratios sum to 1
        total_ratio = self.knowledge_ratio + self.reasoning_ratio + self.grammar_ratio
        if abs(total_ratio - 1.0) > 1e-6:
            raise ValueError(f"Partition ratios must sum to 1.0, got {total_ratio}")

        # Compute partition sizes
        self.knowledge_size = int(self.pool_size * self.knowledge_ratio)
        self.reasoning_size = int(self.pool_size * self.reasoning_ratio)
        self.grammar_size = self.pool_size - self.knowledge_size - self.reasoning_size

    @classmethod
    def from_preset(cls, preset: Literal["nano", "mini", "tiny", "small", "base"]) -> "DPSNRConfig":
        """Create config from a preset.

        Args:
            preset: One of 'nano', 'mini', 'tiny', 'small', or 'base'.

        Returns:
            DPSNRConfig with preset values.

        Raises:
            ValueError: If preset is not recognized.
        """
        presets = {
            "nano": {
                "vocab_size": 256,
                "max_seq_len": 128,
                "hidden_dim": 128,
                "num_layers": 2,
                "num_heads": 2,
                "head_dim": 64,
                "intermediate_dim": 512,
                "pool_size": 1024,
                "pool_dim": 128,
                "top_k": 128,
                "retrieval_dim": 128,
                "max_reasoning_steps": 4,
                "dropout": 0.0,
            },
            "mini": {
                "vocab_size": 512,
                "max_seq_len": 256,
                "hidden_dim": 256,
                "num_layers": 3,
                "num_heads": 4,
                "head_dim": 64,
                "intermediate_dim": 1024,
                "pool_size": 8192,
                "pool_dim": 256,
                "top_k": 256,
                "retrieval_dim": 128,
                "max_reasoning_steps": 6,
                "dropout": 0.0,
            },
            "tiny": {
                "hidden_dim": 512,
                "num_layers": 4,
                "num_heads": 8,
                "head_dim": 64,
                "intermediate_dim": 2048,
                "pool_size": 100000,
                "pool_dim": 256,
                "top_k": 512,
            },
            "small": {
                "hidden_dim": 768,
                "num_layers": 6,
                "num_heads": 12,
                "head_dim": 64,
                "intermediate_dim": 3072,
                "pool_size": 250000,
                "pool_dim": 384,
                "top_k": 512,
            },
            "base": {
                "hidden_dim": 1024,
                "num_layers": 8,
                "num_heads": 16,
                "head_dim": 64,
                "intermediate_dim": 4096,
                "pool_size": 500000,
                "pool_dim": 512,
                "top_k": 512,
            },
        }

        if preset not in presets:
            raise ValueError(f"Unknown preset: {preset}. Choose from {list(presets.keys())}")

        return cls(**presets[preset])

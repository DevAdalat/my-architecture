"""MassivePool: The externalized knowledge storage of DPSN-R.

XLA Compliance:
- Fixed top-k retrieval with static shapes
- No data-dependent operations (no torch.nonzero, torch.unique)
- Uses torch.topk for deterministic shape selection
- Partition slicing uses precomputed indices
- Uses index_select instead of advanced indexing
"""

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import DPSNRConfig

PhaseType = Literal["knowledge", "reasoning", "grammar", "all"]


class MassivePool(nn.Module):
    """Massive partitioned parameter pool.

    XLA-Optimized: Static shapes, precomputed partition tensors.
    """

    def __init__(self, config: DPSNRConfig) -> None:
        super().__init__()
        self.config = config
        self.pool_size = config.pool_size
        self.pool_dim = config.pool_dim
        self.retrieval_dim = config.retrieval_dim
        self.top_k = config.top_k

        # Partition boundaries (Python ints - compile-time constants)
        self.knowledge_start = 0
        self.knowledge_end = config.knowledge_size
        self.reasoning_start = config.knowledge_size
        self.reasoning_end = config.knowledge_size + config.reasoning_size
        self.grammar_start = config.knowledge_size + config.reasoning_size
        self.grammar_end = config.pool_size

        # Precompute partition sizes for static top_k
        self.knowledge_k = min(config.top_k, config.knowledge_size)
        self.reasoning_k = min(config.top_k, config.reasoning_size)
        self.grammar_k = min(config.top_k, config.grammar_size)
        self.all_k = min(config.top_k, config.pool_size)

        # The massive parameter pool [pool_size, pool_dim]
        self.pool = nn.Parameter(torch.randn(config.pool_size, config.pool_dim) * 0.02)

        # Keys for retrieval matching [pool_size, retrieval_dim]
        self.keys = nn.Parameter(torch.randn(config.pool_size, config.retrieval_dim) * 0.02)

        self.output_proj = nn.Linear(config.pool_dim, config.pool_dim, bias=False)

    def _retrieve_partition(
        self,
        query: torch.Tensor,
        keys: torch.Tensor,
        pool: torch.Tensor,
        effective_k: int,
    ) -> torch.Tensor:
        """Retrieve from a partition with static k value."""
        batch_size, seq_len, _ = query.shape

        # Compute scores: [batch, seq, partition_size]
        scores = torch.matmul(query, keys.T)

        # Static top-k selection
        topk_scores, topk_indices = torch.topk(scores, k=effective_k, dim=-1)

        # Softmax weights
        topk_weights = F.softmax(topk_scores, dim=-1)

        # Gather using index_select pattern (more XLA-friendly than advanced indexing)
        # Flatten for gathering
        flat_indices = topk_indices.reshape(-1)  # [batch * seq * k]

        # Use index_select (static operation)
        gathered = torch.index_select(pool, dim=0, index=flat_indices)
        gathered = gathered.reshape(batch_size, seq_len, effective_k, self.pool_dim)

        # Weighted sum
        topk_weights = topk_weights.unsqueeze(-1)  # [batch, seq, k, 1]
        aggregated = (gathered * topk_weights).sum(dim=2)  # [batch, seq, pool_dim]

        return aggregated

    def retrieve_knowledge(self, query: torch.Tensor) -> torch.Tensor:
        """Retrieve from knowledge partition (static shape)."""
        keys = self.keys[self.knowledge_start : self.knowledge_end]
        pool = self.pool[self.knowledge_start : self.knowledge_end]
        result = self._retrieve_partition(query, keys, pool, self.knowledge_k)
        return self.output_proj(result)

    def retrieve_reasoning(self, query: torch.Tensor) -> torch.Tensor:
        """Retrieve from reasoning partition (static shape)."""
        keys = self.keys[self.reasoning_start : self.reasoning_end]
        pool = self.pool[self.reasoning_start : self.reasoning_end]
        result = self._retrieve_partition(query, keys, pool, self.reasoning_k)
        return self.output_proj(result)

    def retrieve_grammar(self, query: torch.Tensor) -> torch.Tensor:
        """Retrieve from grammar partition (static shape)."""
        keys = self.keys[self.grammar_start : self.grammar_end]
        pool = self.pool[self.grammar_start : self.grammar_end]
        result = self._retrieve_partition(query, keys, pool, self.grammar_k)
        return self.output_proj(result)

    def retrieve_all(self, query: torch.Tensor) -> torch.Tensor:
        """Retrieve from entire pool (static shape)."""
        result = self._retrieve_partition(query, self.keys, self.pool, self.all_k)
        return self.output_proj(result)

    def retrieve(self, query: torch.Tensor, phase: PhaseType = "all") -> torch.Tensor:
        """Retrieve from pool based on phase.

        Note: For XLA, prefer calling retrieve_knowledge/reasoning/grammar directly
        to avoid Python dispatch overhead. This method exists for compatibility.
        """
        if phase == "knowledge":
            return self.retrieve_knowledge(query)
        elif phase == "reasoning":
            return self.retrieve_reasoning(query)
        elif phase == "grammar":
            return self.retrieve_grammar(query)
        else:
            return self.retrieve_all(query)

    def get_statistics(self) -> dict[str, int]:
        return {
            "total_size": self.pool_size,
            "knowledge_size": self.knowledge_end - self.knowledge_start,
            "reasoning_size": self.reasoning_end - self.reasoning_start,
            "grammar_size": self.grammar_end - self.grammar_start,
            "pool_dim": self.pool_dim,
            "top_k": self.top_k,
        }

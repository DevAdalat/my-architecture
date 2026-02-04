"""MassivePool: The externalized knowledge storage of DPSN-R.

The pool stores the model's total parameter capacity, organized into
three semantic partitions:
- Knowledge Partition (70%): Facts, concepts, domain knowledge
- Reasoning Partition (20%): Abstract logic patterns, inference rules
- Grammar Partition (10%): Language structure, fluency markers

XLA Compliance:
- Fixed top-k retrieval with static shapes
- No data-dependent operations (no torch.nonzero, torch.unique)
- Uses torch.topk for deterministic shape selection
"""

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import DPSNRConfig

PhaseType = Literal["knowledge", "reasoning", "grammar", "all"]


class MassivePool(nn.Module):
    """Massive partitioned parameter pool.

    The pool contains vectors that can be dynamically retrieved based
    on query relevance. It supports phase-specific retrieval where
    different reasoning phases access different partitions.
    """

    def __init__(self, config: DPSNRConfig) -> None:
        """Initialize the parameter pool.

        Args:
            config: Model configuration.
        """
        super().__init__()
        self.config = config
        self.pool_size = config.pool_size
        self.pool_dim = config.pool_dim
        self.retrieval_dim = config.retrieval_dim
        self.top_k = config.top_k

        # Partition boundaries
        self.knowledge_start = 0
        self.knowledge_end = config.knowledge_size
        self.reasoning_start = config.knowledge_size
        self.reasoning_end = config.knowledge_size + config.reasoning_size
        self.grammar_start = config.knowledge_size + config.reasoning_size
        self.grammar_end = config.pool_size

        # The massive parameter pool
        # Shape: [pool_size, pool_dim]
        self.pool = nn.Parameter(torch.randn(config.pool_size, config.pool_dim) * 0.02)

        # Keys for retrieval matching (separate from values for efficiency)
        # Shape: [pool_size, retrieval_dim]
        self.keys = nn.Parameter(torch.randn(config.pool_size, config.retrieval_dim) * 0.02)

        # Output projection to integrate retrieved vectors
        self.output_proj = nn.Linear(config.pool_dim, config.pool_dim, bias=False)

    def _get_partition_mask(
        self,
        phase: PhaseType,
        device: torch.device,
    ) -> torch.Tensor:
        """Get a mask for the specified partition.

        Args:
            phase: The reasoning phase ('knowledge', 'reasoning', 'grammar', 'all').
            device: Device to create mask on.

        Returns:
            Boolean mask of shape [pool_size].
        """
        mask = torch.zeros(self.pool_size, dtype=torch.bool, device=device)

        if phase == "knowledge":
            mask[self.knowledge_start : self.knowledge_end] = True
        elif phase == "reasoning":
            mask[self.reasoning_start : self.reasoning_end] = True
        elif phase == "grammar":
            mask[self.grammar_start : self.grammar_end] = True
        else:  # "all"
            mask[:] = True

        return mask

    def _get_partition_range(self, phase: PhaseType) -> tuple[int, int]:
        """Get start/end indices for a partition.

        Args:
            phase: The reasoning phase.

        Returns:
            Tuple of (start_idx, end_idx).
        """
        if phase == "knowledge":
            return self.knowledge_start, self.knowledge_end
        elif phase == "reasoning":
            return self.reasoning_start, self.reasoning_end
        elif phase == "grammar":
            return self.grammar_start, self.grammar_end
        else:  # "all"
            return 0, self.pool_size

    def retrieve(
        self,
        query: torch.Tensor,
        phase: PhaseType = "all",
    ) -> torch.Tensor:
        """Retrieve top-k vectors from the pool based on query relevance.

        Uses static top-k selection for XLA compatibility.

        Args:
            query: Query vectors of shape [batch, seq, retrieval_dim].
            phase: Which partition to search ('knowledge', 'reasoning', 'grammar', 'all').

        Returns:
            Retrieved and aggregated vectors of shape [batch, seq, pool_dim].
        """
        batch_size, seq_len, _ = query.shape

        # Get partition range
        start_idx, end_idx = self._get_partition_range(phase)
        partition_size = end_idx - start_idx

        # Slice keys and pool for the partition
        partition_keys = self.keys[start_idx:end_idx]  # [partition_size, retrieval_dim]
        partition_pool = self.pool[start_idx:end_idx]  # [partition_size, pool_dim]

        # Compute relevance scores
        # query: [batch, seq, retrieval_dim]
        # partition_keys: [partition_size, retrieval_dim]
        # scores: [batch, seq, partition_size]
        scores = torch.matmul(query, partition_keys.T)

        # Determine effective top_k (can't exceed partition size)
        effective_k = min(self.top_k, partition_size)

        # Top-k selection (static shape operation)
        topk_scores, topk_indices = torch.topk(scores, k=effective_k, dim=-1)

        # Softmax over top-k scores for weighted aggregation
        topk_weights = F.softmax(topk_scores, dim=-1)  # [batch, seq, k]

        # Gather top-k vectors from pool
        # Flatten batch and seq for indexing
        topk_indices_flat = topk_indices.reshape(-1, effective_k)  # [batch*seq, k]

        # Gather vectors: we need to index partition_pool with topk_indices
        # partition_pool: [partition_size, pool_dim]
        # topk_indices_flat: [batch*seq, k]
        # Result: [batch*seq, k, pool_dim]
        retrieved = partition_pool[topk_indices_flat.reshape(-1)].reshape(
            batch_size * seq_len, effective_k, self.pool_dim
        )

        # Weighted aggregation
        topk_weights_flat = topk_weights.reshape(-1, effective_k, 1)  # [batch*seq, k, 1]
        aggregated = (retrieved * topk_weights_flat).sum(dim=1)  # [batch*seq, pool_dim]

        # Reshape back
        aggregated = aggregated.reshape(batch_size, seq_len, self.pool_dim)

        # Apply output projection
        return self.output_proj(aggregated)

    def get_statistics(self) -> dict[str, int]:
        """Get pool statistics.

        Returns:
            Dictionary with partition sizes.
        """
        return {
            "total_size": self.pool_size,
            "knowledge_size": self.knowledge_end - self.knowledge_start,
            "reasoning_size": self.reasoning_end - self.reasoning_start,
            "grammar_size": self.grammar_end - self.grammar_start,
            "pool_dim": self.pool_dim,
            "top_k": self.top_k,
        }

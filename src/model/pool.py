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
import torch.distributed as dist
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

        # Distributed State
        self.is_distributed = False
        self.rank = 0
        self.world_size = 1
        self.local_start = 0
        self.local_end = config.pool_size
        self.local_keys: torch.Tensor | None = None
        self.local_pool: torch.Tensor | None = None

    def enable_distributed(self, rank: int, world_size: int) -> None:
        """Enable distributed mode with manual sharding.

        Args:
            rank: Current process rank.
            world_size: Total number of processes.
        """
        if world_size <= 1:
            return

        self.is_distributed = True
        self.rank = rank
        self.world_size = world_size

        # 1. Shard the Pool Logic
        # Calculate local shard range [start, end)
        shard_size = self.pool_size // world_size
        remainder = self.pool_size % world_size

        # Simple block sharding
        if rank < remainder:
            shard_size += 1
            start = rank * shard_size
        else:
            start = rank * shard_size + remainder

        self.local_start = start
        self.local_end = start + shard_size

        # 2. Slice the Parameters (Views) -> REAL SHARDING
        # Create new leaf parameters for the shard to free memory
        # We use .data to avoid tracking history, then wrap in Parameter
        local_keys_data = self.keys.data[self.local_start : self.local_end].clone()
        local_pool_data = self.pool.data[self.local_start : self.local_end].clone()

        # Delete the full massive parameters to free memory
        del self.keys
        del self.pool

        # Re-register the shards as the primary parameters
        self.keys = nn.Parameter(local_keys_data)
        self.pool = nn.Parameter(local_pool_data)

        # Update local pointers to match
        self.local_keys = self.keys
        self.local_pool = self.pool

    def _retrieve_partition_distributed(
        self,
        query: torch.Tensor,
        partition_start: int,
        partition_end: int,
        effective_k: int,
    ) -> torch.Tensor:
        """Distributed retrieval: Local Scan -> All-Gather Candidates -> Global Select -> Sparse Fetch."""

        batch_size, seq_len, _ = query.shape

        # --- Step 1: Local Search ---
        # Determine overlap between this GPU's shard and the requested partition
        # Partition: [p_start, p_end)
        # My Shard:  [my_start, my_end)

        overlap_start = max(self.local_start, partition_start)
        overlap_end = min(self.local_end, partition_end)

        if overlap_start < overlap_end:
            # We have relevant data
            # Adjust indices to be relative to local shard
            local_slice_start = overlap_start - self.local_start
            local_slice_end = overlap_end - self.local_start

            # Get local keys for this partition
            # self.local_keys is [shard_size, dim]
            local_keys_part = self.local_keys[local_slice_start:local_slice_end]  # type: ignore

            # Chunked Score Computation to avoid OOM
            # [B, S, local_part_size] is too big (e.g. 5GB).
            # We compute scores in blocks of keys.
            chunk_size = 20000  # Process 20k keys at a time (~320MB for scores)
            num_keys = local_keys_part.size(0)

            best_scores = None
            best_indices = None

            for i in range(0, num_keys, chunk_size):
                end_idx = min(i + chunk_size, num_keys)
                key_chunk = local_keys_part[i:end_idx]

                # [B, S, chunk_size]
                # DETACH to prevent massive graph buildup.
                # Since dist.all_gather breaks gradients anyway, keeping this graph is useless memory cost.
                chunk_scores = torch.matmul(query, key_chunk.T).detach()

                # Local Top-K for this chunk
                curr_k_chunk = min(effective_k, chunk_scores.size(-1))
                chunk_top_scores, chunk_top_indices_local = torch.topk(
                    chunk_scores, k=curr_k_chunk, dim=-1
                )

                # Adjust indices to be relative to the full local part start
                chunk_top_indices_local = chunk_top_indices_local + i

                if best_scores is None:
                    best_scores = chunk_top_scores
                    best_indices = chunk_top_indices_local
                else:
                    # Merge with previous best
                    combined_scores = torch.cat([best_scores, chunk_top_scores], dim=-1)
                    combined_indices = torch.cat([best_indices, chunk_top_indices_local], dim=-1)

                    # Take top-k of combined
                    # Note: combined size is at most 2*effective_k
                    curr_k_combined = min(effective_k, combined_scores.size(-1))
                    best_scores, best_indices_idx = torch.topk(
                        combined_scores, k=curr_k_combined, dim=-1
                    )
                    best_indices = torch.gather(combined_indices, -1, best_indices_idx)

            my_top_scores = best_scores
            my_top_indices_local = best_indices

            # Convert local indices to global indices
            # my_top_indices_local is 0..local_part_size
            # global index = my_top_indices_local + overlap_start
            my_top_indices_global = my_top_indices_local + overlap_start

            curr_k = my_top_indices_local.size(-1)

        else:
            # We have NO data for this partition (e.g., I'm GPU 0, partition is strictly on GPU 1)
            # Return dummy results (score -inf)
            curr_k = effective_k  # Default fallback
            my_top_scores = torch.full(
                (batch_size, seq_len, effective_k),
                float("-inf"),
                device=query.device,
                dtype=query.dtype,
            )
            my_top_indices_global = torch.zeros(
                (batch_size, seq_len, effective_k), device=query.device, dtype=torch.long
            )
            my_top_indices_local = torch.zeros(
                (batch_size, seq_len, effective_k), device=query.device, dtype=torch.long
            )

        # --- Step 2: Gather Candidates (Scores & Indices) ---
        # We need to gather (scores, indices) from all ranks
        # List of tensors from each rank
        gathered_scores = [torch.zeros_like(my_top_scores) for _ in range(self.world_size)]
        gathered_indices = [torch.zeros_like(my_top_indices_global) for _ in range(self.world_size)]

        dist.all_gather(gathered_scores, my_top_scores)
        dist.all_gather(gathered_indices, my_top_indices_global)

        # Concatenate: [B, S, world_size * k]
        all_scores = torch.cat(gathered_scores, dim=-1)

        # --- Step 3: Global Selection ---

        # Find the global top-k from the candidates
        final_scores, best_candidate_indices = torch.topk(all_scores, k=effective_k, dim=-1)

        # Calculate Softmax Weights
        final_weights = F.softmax(final_scores, dim=-1)

        # --- Step 4: Sparse Fetch (The "Vector Routing") ---
        # Logic:
        # 1. Local Search -> Get top K candidates (Score, Index, AND Vector).
        # 2. All-Gather (Score, Index, Vector).
        # 3. Global Top-K on Score.
        # 4. Select corresponding Vector.

        # Re-do Local Step to include Vectors
        if overlap_start < overlap_end:
            # We already have my_top_indices_local
            # Gather vectors: [B, S, k, dim]
            flat_local_indices = my_top_indices_local.reshape(-1)
            my_top_vectors = torch.index_select(self.local_pool, 0, flat_local_indices)  # type: ignore
            my_top_vectors = my_top_vectors.reshape(batch_size, seq_len, curr_k, self.pool_dim)
        else:
            my_top_vectors = torch.zeros(
                (batch_size, seq_len, curr_k, self.pool_dim),
                device=query.device,
                dtype=query.dtype,
            )

        # Gather Vectors
        gathered_vectors = [torch.zeros_like(my_top_vectors) for _ in range(self.world_size)]
        dist.all_gather(gathered_vectors, my_top_vectors)
        all_vectors = torch.cat(gathered_vectors, dim=2)  # [B, S, world_size*k, dim]

        # Use the best_candidate_indices from before to select the vectors
        # Expand indices for dim dimension
        # best_candidate_indices: [B, S, k] -> [B, S, k, dim]
        vector_gather_indices = best_candidate_indices.unsqueeze(-1).expand(
            -1, -1, -1, self.pool_dim
        )

        final_vectors = torch.gather(all_vectors, 2, vector_gather_indices)

        # Weighted sum optimization to avoid massive intermediate tensor [B, S, k, dim]
        # Previous: aggregated = (final_vectors * final_weights.unsqueeze(-1)).sum(dim=2)
        #
        # Memory Efficient Approach using Batch Matrix Multiply (BMM):
        # final_weights: [B, S, k] -> [B*S, 1, k]
        # final_vectors: [B, S, k, D] -> [B*S, k, D]
        # bmm([B*S, 1, k], [B*S, k, D]) -> [B*S, 1, D] -> [B, S, D]

        weights_flat = final_weights.reshape(-1, 1, effective_k)
        vectors_flat = final_vectors.reshape(-1, effective_k, self.pool_dim)

        aggregated_flat = torch.bmm(weights_flat, vectors_flat)
        aggregated = aggregated_flat.reshape(batch_size, seq_len, self.pool_dim)

        return aggregated

    def _retrieve_partition(
        self,
        query: torch.Tensor,
        keys: torch.Tensor | None,
        pool: torch.Tensor | None,
        effective_k: int,
        partition_range: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        """Retrieve from a partition with static k value."""

        # Dispatch to distributed logic if enabled
        if self.is_distributed:
            # Use explicit range if provided, otherwise fallback to inference (though callers should provide range)
            if partition_range:
                start, end = partition_range
            else:
                # Fallback logic (should ideally be removed if all callers are updated)
                # But since we changed signature to allow None for keys/pool, let's just default to All
                start, end = 0, self.pool_size

            return self._retrieve_partition_distributed(query, start, end, effective_k)

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
        start, end = self.knowledge_start, self.knowledge_end
        if self.is_distributed:
            keys, pool = None, None
        else:
            keys = self.keys[start:end]
            pool = self.pool[start:end]

        result = self._retrieve_partition(
            query, keys, pool, self.knowledge_k, partition_range=(start, end)
        )
        return self.output_proj(result)

    def retrieve_reasoning(self, query: torch.Tensor) -> torch.Tensor:
        """Retrieve from reasoning partition (static shape)."""
        start, end = self.reasoning_start, self.reasoning_end
        if self.is_distributed:
            keys, pool = None, None
        else:
            keys = self.keys[start:end]
            pool = self.pool[start:end]

        result = self._retrieve_partition(
            query, keys, pool, self.reasoning_k, partition_range=(start, end)
        )
        return self.output_proj(result)

    def retrieve_grammar(self, query: torch.Tensor) -> torch.Tensor:
        """Retrieve from grammar partition (static shape)."""
        start, end = self.grammar_start, self.grammar_end
        if self.is_distributed:
            keys, pool = None, None
        else:
            keys = self.keys[start:end]
            pool = self.pool[start:end]

        result = self._retrieve_partition(
            query, keys, pool, self.grammar_k, partition_range=(start, end)
        )
        return self.output_proj(result)

    def retrieve_all(self, query: torch.Tensor) -> torch.Tensor:
        """Retrieve from entire pool (static shape)."""
        start, end = 0, self.pool_size
        if self.is_distributed:
            keys, pool = None, None
        else:
            keys = self.keys
            pool = self.pool

        result = self._retrieve_partition(
            query, keys, pool, self.all_k, partition_range=(start, end)
        )
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

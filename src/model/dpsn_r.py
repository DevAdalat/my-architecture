"""DPSN-R: Dynamic Parameter Selection Network with Recurrent Reasoning.

The main model class that orchestrates:
1. TinyController - encodes context and generates queries
2. MassivePool - stores and retrieves knowledge vectors
3. ACT - adaptive compute for variable reasoning depth

XLA Compliance:
- All loops run to fixed max iterations (no early exit)
- Shapes are static throughout the forward pass
- Phase selection uses integer indices, not dynamic branching
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .act import AdaptiveComputeTime, compute_ponder_loss
from .config import DPSNRConfig
from .controller import TinyController
from .pool import MassivePool, PhaseType


class DPSNR(nn.Module):
    """Dynamic Parameter Selection Network with Recurrent Reasoning.

    This model implements a sparse recurrent architecture where:
    - A tiny controller encodes context and routes queries
    - A massive pool stores externalized knowledge
    - Recurrent reasoning loops refine understanding
    - ACT enables variable compute per token
    """

    # Phase schedule: maps step index to pool partition
    PHASE_SCHEDULE: list[PhaseType] = [
        "knowledge",  # Step 0-1: Understanding phase
        "knowledge",
        "reasoning",  # Step 2-5: Reasoning phase
        "reasoning",
        "reasoning",
        "reasoning",
        "grammar",  # Step 6-7: Expression phase
        "grammar",
    ]

    def __init__(self, config: DPSNRConfig) -> None:
        """Initialize DPSN-R model.

        Args:
            config: Model configuration.
        """
        super().__init__()
        self.config = config

        # Core components
        self.controller = TinyController(config)
        self.pool = MassivePool(config)
        self.act = AdaptiveComputeTime(config)

        # Extend phase schedule if max_steps exceeds default
        while len(self.PHASE_SCHEDULE) < config.max_reasoning_steps:
            self.PHASE_SCHEDULE.append("grammar")

    @classmethod
    def from_preset(cls, preset: str) -> "DPSNR":
        """Create model from a preset configuration.

        Args:
            preset: One of 'tiny', 'small', or 'base'.

        Returns:
            Initialized DPSNR model.
        """
        config = DPSNRConfig.from_preset(preset)  # type: ignore[arg-type]
        return cls(config)

    def _get_phase(self, step: int) -> PhaseType:
        """Get the reasoning phase for a given step.

        Args:
            step: Current reasoning step (0-indexed).

        Returns:
            Phase type for pool retrieval.
        """
        step = min(step, len(self.PHASE_SCHEDULE) - 1)
        return self.PHASE_SCHEDULE[step]

    def _reasoning_step(
        self,
        hidden: torch.Tensor,
        input_ids: torch.Tensor,
        step: int,
    ) -> torch.Tensor:
        """Perform a single reasoning step.

        Args:
            hidden: Current hidden state [batch, seq, hidden_dim].
            input_ids: Input token IDs [batch, seq].
            step: Current reasoning step.

        Returns:
            Updated hidden state [batch, seq, hidden_dim].
        """
        # Generate query for pool retrieval
        query = self.controller.generate_query(hidden)

        # Retrieve from pool based on current phase
        phase = self._get_phase(step)
        retrieved = self.pool.retrieve(query, phase=phase)

        # Integrate retrieved knowledge
        hidden = self.controller.integrate(hidden, retrieved)

        return hidden

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        use_act: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Forward pass with recurrent reasoning.

        Args:
            input_ids: Token IDs of shape [batch, seq].
            labels: Optional target labels for loss computation.
            use_act: Whether to use adaptive compute (default True).

        Returns:
            Dictionary containing:
                - logits: Output logits [batch, seq, vocab_size]
                - loss: Total loss (if labels provided)
                - ce_loss: Cross-entropy loss (if labels provided)
                - ponder_loss: Ponder regularization loss (if labels provided)
                - ponder_cost: Steps used per token [batch, seq]
        """
        batch_size, seq_len = input_ids.shape

        # Initial encoding
        hidden = self.controller.encode(input_ids, step=0)

        if use_act:
            # Define step function for ACT
            # Captures step count via closure
            step_counter = [0]

            def step_fn(h: torch.Tensor) -> torch.Tensor:
                step_counter[0] += 1
                step = step_counter[0]
                return self._reasoning_step(h, input_ids, step)

            def halt_fn(h: torch.Tensor) -> torch.Tensor:
                return self.controller.predict_halt(h)

            # Run ACT loop
            hidden, ponder_cost, halted = self.act(hidden, step_fn, halt_fn)
        else:
            # Run fixed number of steps without ACT
            ponder_cost = torch.full(
                (batch_size, seq_len),
                self.config.max_reasoning_steps,
                device=input_ids.device,
                dtype=hidden.dtype,
            )

            for step in range(self.config.max_reasoning_steps):
                hidden = self._reasoning_step(hidden, input_ids, step)

        # Decode to logits
        logits = self.controller.decode(hidden)

        outputs: dict[str, torch.Tensor] = {
            "logits": logits,
            "ponder_cost": ponder_cost,
        }

        # Compute loss if labels provided
        if labels is not None:
            # Cross-entropy loss
            ce_loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                labels.view(-1),
                ignore_index=-100,
            )

            # Ponder loss
            ponder_loss = compute_ponder_loss(ponder_cost, target_ponder=1.0)

            # Total loss
            total_loss = ce_loss + self.config.ponder_lambda * ponder_loss

            outputs["loss"] = total_loss
            outputs["ce_loss"] = ce_loss
            outputs["ponder_loss"] = ponder_loss

        return outputs

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
    ) -> torch.Tensor:
        """Generate text autoregressively.

        Args:
            input_ids: Initial token IDs [batch, seq].
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_k: Top-k filtering (optional).
            top_p: Nucleus sampling threshold (optional).

        Returns:
            Generated token IDs [batch, seq + max_new_tokens].
        """
        self.eval()

        for _ in range(max_new_tokens):
            # Get logits for last position
            outputs = self.forward(input_ids, use_act=True)
            next_logits = outputs["logits"][:, -1, :]  # [batch, vocab]

            # Apply temperature
            if temperature != 1.0:
                next_logits = next_logits / temperature

            # Apply top-k filtering
            if top_k is not None:
                top_k_logits, top_k_indices = torch.topk(next_logits, k=top_k, dim=-1)
                next_logits = torch.full_like(next_logits, float("-inf"))
                next_logits.scatter_(-1, top_k_indices, top_k_logits)

            # Apply top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices_to_remove.scatter(
                    -1, sorted_indices, sorted_indices_to_remove
                )
                next_logits = next_logits.masked_fill(indices_to_remove, float("-inf"))

            # Sample
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids

    def count_parameters(self) -> dict[str, int]:
        """Count model parameters by component.

        Returns:
            Dictionary with parameter counts.
        """
        controller_params = sum(p.numel() for p in self.controller.parameters())
        pool_params = sum(p.numel() for p in self.pool.parameters())
        total_params = sum(p.numel() for p in self.parameters())

        return {
            "controller": controller_params,
            "pool": pool_params,
            "total": total_params,
        }

    def get_active_parameters(self) -> int:
        """Estimate active parameters per token (approximate).

        Returns:
            Estimated number of active parameters per forward pass.
        """
        # Controller is fully active
        controller_params = sum(p.numel() for p in self.controller.parameters())

        # Pool: only top_k vectors are retrieved per step
        active_pool_per_step = self.config.top_k * self.config.pool_dim

        # Approximate: average 4 steps per token
        avg_steps = 4
        active_pool_total = active_pool_per_step * avg_steps

        return controller_params + active_pool_total

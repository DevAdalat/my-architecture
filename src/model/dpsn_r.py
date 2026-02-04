"""DPSN-R: Dynamic Parameter Selection Network with Recurrent Reasoning.

XLA Compliance:
- All loops are UNROLLED with static Python iteration (no dynamic callbacks)
- Shapes are static throughout the forward pass
- Phase indices are precomputed as Python constants
- No mutable closures or step counters
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

    XLA-Optimized: Uses explicit loop unrolling, no callbacks.
    """

    PHASE_SCHEDULE: list[PhaseType] = [
        "knowledge",
        "knowledge",
        "reasoning",
        "reasoning",
        "reasoning",
        "reasoning",
        "grammar",
        "grammar",
    ]

    def __init__(self, config: DPSNRConfig) -> None:
        super().__init__()
        self.config = config

        self.controller = TinyController(config)
        self.pool = MassivePool(config)
        self.act = AdaptiveComputeTime(config)

        while len(self.PHASE_SCHEDULE) < config.max_reasoning_steps:
            self.PHASE_SCHEDULE.append("grammar")

        self._phase_list = self.PHASE_SCHEDULE[: config.max_reasoning_steps]

    @classmethod
    def from_preset(cls, preset: str) -> "DPSNR":
        config = DPSNRConfig.from_preset(preset)  # type: ignore[arg-type]
        return cls(config)

    def _reasoning_step(
        self,
        hidden: torch.Tensor,
        phase: PhaseType,
    ) -> torch.Tensor:
        """Single reasoning step with fixed phase (no dynamic lookup)."""
        query = self.controller.generate_query(hidden)
        retrieved = self.pool.retrieve(query, phase=phase)
        return self.controller.integrate(hidden, retrieved)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        use_act: bool = True,
    ) -> dict[str, torch.Tensor]:
        batch_size, seq_len = input_ids.shape

        hidden = self.controller.encode(input_ids, step=0)

        if use_act:
            acc = self.act.init_state(
                batch_size,
                seq_len,
                self.config.hidden_dim,
                input_ids.device,
                hidden.dtype,
            )

            # UNROLLED LOOP - each iteration is a static graph node
            # Step 0: knowledge phase
            halt_prob = self.controller.predict_halt(hidden).squeeze(-1)
            acc = self.act.step(hidden, halt_prob, acc, 0)
            hidden = self._reasoning_step(hidden, "knowledge")

            # Step 1: knowledge phase
            halt_prob = self.controller.predict_halt(hidden).squeeze(-1)
            acc = self.act.step(hidden, halt_prob, acc, 1)
            hidden = self._reasoning_step(hidden, "knowledge")

            # Step 2: reasoning phase
            halt_prob = self.controller.predict_halt(hidden).squeeze(-1)
            acc = self.act.step(hidden, halt_prob, acc, 2)
            hidden = self._reasoning_step(hidden, "reasoning")

            # Step 3: reasoning phase
            halt_prob = self.controller.predict_halt(hidden).squeeze(-1)
            acc = self.act.step(hidden, halt_prob, acc, 3)
            hidden = self._reasoning_step(hidden, "reasoning")

            # Step 4: reasoning phase (for max_steps >= 5)
            if self.config.max_reasoning_steps >= 5:
                halt_prob = self.controller.predict_halt(hidden).squeeze(-1)
                acc = self.act.step(hidden, halt_prob, acc, 4)
                hidden = self._reasoning_step(hidden, "reasoning")

            # Step 5: reasoning phase (for max_steps >= 6)
            if self.config.max_reasoning_steps >= 6:
                halt_prob = self.controller.predict_halt(hidden).squeeze(-1)
                acc = self.act.step(hidden, halt_prob, acc, 5)
                hidden = self._reasoning_step(hidden, "reasoning")

            # Step 6: grammar phase (for max_steps >= 7)
            if self.config.max_reasoning_steps >= 7:
                halt_prob = self.controller.predict_halt(hidden).squeeze(-1)
                acc = self.act.step(hidden, halt_prob, acc, 6)
                hidden = self._reasoning_step(hidden, "grammar")

            # Step 7: grammar phase (for max_steps >= 8)
            if self.config.max_reasoning_steps >= 8:
                halt_prob = self.controller.predict_halt(hidden).squeeze(-1)
                acc = self.act.step(hidden, halt_prob, acc, 7)
                hidden = self._reasoning_step(hidden, "grammar")

            hidden, ponder_cost, _ = self.act.finalize(hidden, acc)
        else:
            ponder_cost = torch.full(
                (batch_size, seq_len),
                self.config.max_reasoning_steps,
                device=input_ids.device,
                dtype=hidden.dtype,
            )

            for step_idx, phase in enumerate(self._phase_list):
                hidden = self._reasoning_step(hidden, phase)

        logits = self.controller.decode(hidden)

        outputs: dict[str, torch.Tensor] = {
            "logits": logits,
            "ponder_cost": ponder_cost,
        }

        if labels is not None:
            ce_loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                labels.view(-1),
                ignore_index=-100,
            )
            ponder_loss = compute_ponder_loss(ponder_cost, target_ponder=1.0)
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
        self.eval()

        for _ in range(max_new_tokens):
            outputs = self.forward(input_ids, use_act=True)
            next_logits = outputs["logits"][:, -1, :]

            if temperature != 1.0:
                next_logits = next_logits / temperature

            if top_k is not None:
                top_k_logits, top_k_indices = torch.topk(next_logits, k=top_k, dim=-1)
                next_logits = torch.full_like(next_logits, float("-inf"))
                next_logits.scatter_(-1, top_k_indices, top_k_logits)

            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(
                    -1, sorted_indices, sorted_indices_to_remove
                )
                next_logits = next_logits.masked_fill(indices_to_remove, float("-inf"))

            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids

    def count_parameters(self) -> dict[str, int]:
        controller_params = sum(p.numel() for p in self.controller.parameters())
        pool_params = sum(p.numel() for p in self.pool.parameters())
        total_params = sum(p.numel() for p in self.parameters())
        return {
            "controller": controller_params,
            "pool": pool_params,
            "total": total_params,
        }

    def get_active_parameters(self) -> int:
        controller_params = sum(p.numel() for p in self.controller.parameters())
        active_pool_per_step = self.config.top_k * self.config.pool_dim
        avg_steps = 4
        return controller_params + active_pool_per_step * avg_steps

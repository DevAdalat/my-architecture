"""Adaptive Compute Time (ACT) for DPSN-R.

XLA Compliance:
- Static maximum iteration count (no early exit)
- Uses masks and weighted sums instead of conditional execution
- No .item() calls or Python control flow on tensor values
- All shapes remain constant throughout execution
- NO CALLBACKS - logic is inlined for graph tracing
"""

import torch
import torch.nn as nn

from .config import DPSNRConfig


class AdaptiveComputeTime(nn.Module):
    """Adaptive Compute Time mechanism for dynamic depth.

    XLA-Optimized: No callbacks, pure tensor operations only.
    The step execution is handled externally with explicit unrolling.
    """

    def __init__(self, config: DPSNRConfig) -> None:
        super().__init__()
        self.max_steps = config.max_reasoning_steps
        self.threshold = config.act_threshold

    def init_state(
        self,
        batch_size: int,
        seq_len: int,
        hidden_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[str, torch.Tensor]:
        """Initialize ACT accumulators. Call once before the loop."""
        return {
            "cumulative_halt": torch.zeros(batch_size, seq_len, device=device, dtype=dtype),
            "remainder": torch.ones(batch_size, seq_len, device=device, dtype=dtype),
            "ponder_cost": torch.zeros(batch_size, seq_len, device=device, dtype=dtype),
            "final_state": torch.zeros(batch_size, seq_len, hidden_dim, device=device, dtype=dtype),
        }

    def step(
        self,
        state: torch.Tensor,
        halt_prob: torch.Tensor,
        acc: dict[str, torch.Tensor],
        step_idx: int,
    ) -> dict[str, torch.Tensor]:
        """Process one ACT step. Call inside an UNROLLED loop (not dynamic)."""
        cumulative_halt = acc["cumulative_halt"]
        remainder = acc["remainder"]
        ponder_cost = acc["ponder_cost"]
        final_state = acc["final_state"]

        should_halt = (cumulative_halt + halt_prob) >= self.threshold

        step_weight = torch.where(should_halt, remainder, halt_prob)

        new_remainder = torch.where(
            should_halt,
            torch.zeros_like(remainder),
            remainder - halt_prob,
        ).clamp(min=0)

        new_final_state = final_state + step_weight.unsqueeze(-1) * state
        new_ponder_cost = ponder_cost + step_weight * (step_idx + 1)
        new_cumulative_halt = (cumulative_halt + halt_prob).clamp(max=1.0)

        return {
            "cumulative_halt": new_cumulative_halt,
            "remainder": new_remainder,
            "ponder_cost": new_ponder_cost,
            "final_state": new_final_state,
        }

    def finalize(
        self,
        state: torch.Tensor,
        acc: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Finalize ACT after all steps. Returns final outputs."""
        remainder = acc["remainder"]
        final_state = acc["final_state"] + remainder.unsqueeze(-1) * state
        ponder_cost = acc["ponder_cost"] + remainder * self.max_steps
        halted = acc["cumulative_halt"] >= self.threshold
        return final_state, ponder_cost, halted


def compute_ponder_loss(
    ponder_cost: torch.Tensor,
    target_ponder: float = 1.0,
) -> torch.Tensor:
    """Compute ponder loss to regularize computation usage."""
    mean_ponder = ponder_cost.mean()
    return (mean_ponder - target_ponder) ** 2

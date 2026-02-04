"""Adaptive Compute Time (ACT) for DPSN-R.

ACT allows the model to dynamically decide how many reasoning steps
to perform per token based on confidence. Easy tokens halt early,
while complex tokens use more computation.

XLA Compliance:
- Static maximum iteration count (no early exit)
- Uses masks and weighted sums instead of conditional execution
- No .item() calls or Python control flow on tensor values
- All shapes remain constant throughout execution
"""

from collections.abc import Callable

import torch
import torch.nn as nn

from .config import DPSNRConfig

StepFn = Callable[[torch.Tensor], torch.Tensor]
HaltFn = Callable[[torch.Tensor], torch.Tensor]


class AdaptiveComputeTime(nn.Module):
    """Adaptive Compute Time mechanism for dynamic depth.

    Instead of running a fixed number of steps, ACT accumulates halting
    probabilities and uses a weighted combination of intermediate states.
    This allows gradient flow while enabling variable compute.

    XLA Note: We always run max_steps iterations but use masks to
    effectively "halt" tokens. This maintains static shapes.
    """

    def __init__(self, config: DPSNRConfig) -> None:
        super().__init__()
        self.max_steps = config.max_reasoning_steps
        self.threshold = config.act_threshold
        self.epsilon = 1e-6

    def forward(
        self,
        initial_state: torch.Tensor,
        step_fn: StepFn,
        halt_fn: HaltFn,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run ACT loop with static shapes.

        Args:
            initial_state: Initial hidden state of shape [batch, seq, dim].
            step_fn: Function that takes state and returns updated state.
            halt_fn: Function that takes state and returns halt prob [batch, seq, 1].

        Returns:
            Tuple of:
                - final_state: Weighted combination of states [batch, seq, dim].
                - ponder_cost: Number of steps used per token [batch, seq].
                - halted: Whether each token has halted [batch, seq].
        """
        batch_size, seq_len, hidden_dim = initial_state.shape
        device = initial_state.device
        dtype = initial_state.dtype

        # Initialize accumulators
        # cumulative_halt: tracks accumulated halting probability
        cumulative_halt = torch.zeros(batch_size, seq_len, device=device, dtype=dtype)
        # remainder: probability mass remaining when we halt
        remainder = torch.ones(batch_size, seq_len, device=device, dtype=dtype)
        # ponder_cost: weighted sum of step indices (for loss)
        ponder_cost = torch.zeros(batch_size, seq_len, device=device, dtype=dtype)
        # final_state: weighted combination of intermediate states
        final_state = torch.zeros_like(initial_state)

        state = initial_state

        for step in range(self.max_steps):
            # Compute halt probability for current state
            halt_prob = halt_fn(state).squeeze(-1)  # [batch, seq]

            # Determine which tokens should halt at this step
            # A token halts when cumulative probability exceeds threshold
            should_halt = (cumulative_halt + halt_prob) >= self.threshold

            # For tokens that halt at this step, use remainder as their weight
            # For tokens that continue, add halt_prob to their cumulative
            step_weight = torch.where(
                should_halt,
                remainder,  # Use all remaining probability
                halt_prob,  # Use this step's probability
            )

            # Update remainder for tokens that don't halt
            remainder = torch.where(
                should_halt,
                torch.zeros_like(remainder),
                remainder - halt_prob,
            )
            remainder = remainder.clamp(min=0)  # Numerical stability

            # Accumulate weighted state
            final_state = final_state + step_weight.unsqueeze(-1) * state

            # Update ponder cost (step index weighted by probability used)
            ponder_cost = ponder_cost + step_weight * (step + 1)

            # Update cumulative halt probability
            cumulative_halt = cumulative_halt + halt_prob
            cumulative_halt = cumulative_halt.clamp(max=1.0)

            # Take a reasoning step (always execute for static graph)
            state = step_fn(state)

        # Handle any remaining probability mass (numerical edge cases)
        # If we've used all steps, distribute remaining probability to final state
        final_state = final_state + remainder.unsqueeze(-1) * state
        ponder_cost = ponder_cost + remainder * self.max_steps

        # Compute halted mask (tokens that reached threshold before max_steps)
        halted = cumulative_halt >= self.threshold

        return final_state, ponder_cost, halted


def compute_ponder_loss(
    ponder_cost: torch.Tensor,
    target_ponder: float = 1.0,
) -> torch.Tensor:
    """Compute ponder loss to regularize computation usage.

    The ponder loss encourages the model to use an appropriate amount
    of computation, penalizing both under-thinking and over-thinking.

    Args:
        ponder_cost: Ponder cost per token of shape [batch, seq].
        target_ponder: Target average ponder cost.

    Returns:
        Scalar ponder loss.
    """
    # Mean ponder cost across all tokens
    mean_ponder = ponder_cost.mean()

    # L2 penalty for deviation from target
    ponder_loss = (mean_ponder - target_ponder) ** 2

    return ponder_loss

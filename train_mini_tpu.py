#!/usr/bin/env python3
"""Train DPSN-R mini model on TPU with XLA compilation.

This script follows XLA compilation rules from AGENTS.md:
- Static shapes (fixed batch size, sequence length)
- No .item() calls in training loop
- Uses xm.optimizer_step() for gradient sync
- Dimensions are multiples of 128
- Uses drop_last=True for fixed batch sizes
"""

import sys
import time
from functools import partial
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))

from src.data.synthetic import collate_fn, create_data_splits
from src.model import DPSNR, DPSNRConfig

try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl

    HAS_XLA = True
except ImportError:
    HAS_XLA = False
    print("WARNING: torch_xla not available. Falling back to CPU/CUDA.")


def get_device():
    """Get the appropriate device (TPU, CUDA, or CPU)."""
    if HAS_XLA:
        return xm.xla_device()
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def sync_print(msg: str):
    """Print with XLA barrier synchronization."""
    if HAS_XLA:
        xm.master_print(msg)
    else:
        print(msg)


def train_epoch_xla(
    model: DPSNR,
    dataloader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """Train for one epoch with XLA-compliant operations.

    Returns accumulated loss and accuracy tensors (no .item() calls).
    """
    model.train()

    total_loss = torch.tensor(0.0, device=device)
    total_correct = torch.tensor(0, device=device)
    total_tokens = torch.tensor(0, device=device)
    num_batches = torch.tensor(0, device=device)

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, labels=labels, use_act=True)

        loss = outputs["loss"]
        loss.backward()

        # XLA-compliant gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        if HAS_XLA:
            xm.optimizer_step(optimizer)
        else:
            optimizer.step()

        # Accumulate metrics without .item()
        total_loss = total_loss + loss.detach()

        # Compute accuracy on device
        mask = labels != -100
        predictions = outputs["logits"].argmax(dim=-1)
        correct = ((predictions == labels) & mask).sum()
        total_correct = total_correct + correct
        total_tokens = total_tokens + mask.sum()
        num_batches = num_batches + 1

    # Return tensors for later aggregation
    return total_loss, total_correct, total_tokens, num_batches


@torch.no_grad()
def evaluate_xla(
    model: DPSNR,
    dataloader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluate model with XLA-compliant operations."""
    model.eval()

    total_loss = torch.tensor(0.0, device=device)
    total_correct = torch.tensor(0, device=device)
    total_tokens = torch.tensor(0, device=device)
    total_ponder = torch.tensor(0.0, device=device)
    num_batches = torch.tensor(0, device=device)

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids, labels=labels, use_act=True)

        total_loss = total_loss + outputs["loss"]
        total_ponder = total_ponder + outputs["ponder_cost"].mean()

        mask = labels != -100
        predictions = outputs["logits"].argmax(dim=-1)
        correct = ((predictions == labels) & mask).sum()
        total_correct = total_correct + correct
        total_tokens = total_tokens + mask.sum()
        num_batches = num_batches + 1

    return total_loss, total_correct, total_tokens, total_ponder, num_batches


def main():
    sync_print("=" * 60)
    sync_print("DPSN-R Mini Model Training on TPU")
    sync_print("=" * 60)

    device = get_device()
    device_type = "TPU" if HAS_XLA else ("CUDA" if torch.cuda.is_available() else "CPU")
    sync_print(f"\nDevice: {device_type}")

    # Create mini config
    config = DPSNRConfig.from_preset("mini")
    sync_print("\nModel Config (mini):")
    sync_print(f"  vocab_size: {config.vocab_size}")
    sync_print(f"  max_seq_len: {config.max_seq_len}")
    sync_print(f"  hidden_dim: {config.hidden_dim}")
    sync_print(f"  num_layers: {config.num_layers}")
    sync_print(f"  pool_size: {config.pool_size}")
    sync_print(f"  max_reasoning_steps: {config.max_reasoning_steps}")

    # Create model
    model = DPSNR(config).to(device)
    param_counts = model.count_parameters()
    sync_print("\nParameter Counts:")
    sync_print(f"  Controller: {param_counts['controller']:,}")
    sync_print(f"  Pool: {param_counts['pool']:,}")
    sync_print(f"  Total: {param_counts['total']:,}")

    # Create datasets
    sync_print("\nCreating synthetic datasets...")
    train_ds, val_ds, test_ds = create_data_splits(
        total_samples=10000,
        train_ratio=0.8,
        val_ratio=0.1,
        vocab_size=config.vocab_size,
        max_seq_len=config.max_seq_len,
        seed=42,
    )
    sync_print(f"  Train: {len(train_ds)} samples")
    sync_print(f"  Val: {len(val_ds)} samples")
    sync_print(f"  Test: {len(test_ds)} samples")

    # XLA-compliant: fixed batch size, drop_last=True
    batch_size = 64  # Multiple of 8 for TPU efficiency
    collate = partial(collate_fn, max_len=config.max_seq_len)

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate,
        drop_last=True,  # XLA: Fixed batch size
        num_workers=0,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate,
        drop_last=True,
        num_workers=0,
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate,
        drop_last=True,
        num_workers=0,
    )

    # Wrap with XLA parallel loader if available
    if HAS_XLA:
        train_loader = pl.MpDeviceLoader(train_loader, device)
        val_loader = pl.MpDeviceLoader(val_loader, device)
        test_loader = pl.MpDeviceLoader(test_loader, device)

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    num_epochs = 30
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-6
    )

    best_val_loss = float("inf")
    best_epoch = 0

    sync_print("\n" + "=" * 60)
    sync_print("Training")
    sync_print("=" * 60)

    start_time = time.time()

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()

        # Train
        train_loss, train_correct, train_tokens, train_batches = train_epoch_xla(
            model, train_loader, optimizer, device
        )

        # Validate
        val_loss, val_correct, val_tokens, val_ponder, val_batches = evaluate_xla(
            model, val_loader, device
        )

        scheduler.step()

        # Synchronize and extract metrics (single sync point per epoch)
        if HAS_XLA:
            xm.mark_step()

        # Now safe to call .item() after mark_step
        train_loss_val = (train_loss / train_batches).item()
        train_acc_val = (train_correct / train_tokens).item() if train_tokens.item() > 0 else 0.0
        val_loss_val = (val_loss / val_batches).item()
        val_acc_val = (val_correct / val_tokens).item() if val_tokens.item() > 0 else 0.0
        val_ponder_val = (val_ponder / val_batches).item()

        epoch_time = time.time() - epoch_start

        if val_loss_val < best_val_loss:
            best_val_loss = val_loss_val
            best_epoch = epoch
            best_marker = " *"
        else:
            best_marker = ""

        sync_print(
            f"Epoch {epoch:2d}/{num_epochs} | "
            f"Train Loss: {train_loss_val:.4f} | "
            f"Train Acc: {train_acc_val:.2%} | "
            f"Val Loss: {val_loss_val:.4f} | "
            f"Val Acc: {val_acc_val:.2%} | "
            f"Ponder: {val_ponder_val:.2f} | "
            f"Time: {epoch_time:.1f}s{best_marker}"
        )

    total_time = time.time() - start_time
    sync_print(f"\nTraining completed in {total_time:.1f}s")
    sync_print(f"Best validation loss: {best_val_loss:.4f} at epoch {best_epoch}")

    # Final test evaluation
    sync_print("\n" + "=" * 60)
    sync_print("Final Test Evaluation")
    sync_print("=" * 60)

    test_loss, test_correct, test_tokens, test_ponder, test_batches = evaluate_xla(
        model, test_loader, device
    )

    if HAS_XLA:
        xm.mark_step()

    test_loss_val = (test_loss / test_batches).item()
    test_acc_val = (test_correct / test_tokens).item() if test_tokens.item() > 0 else 0.0
    test_ponder_val = (test_ponder / test_batches).item()

    sync_print("\nTest Results:")
    sync_print(f"  Loss: {test_loss_val:.4f}")
    sync_print(f"  Accuracy: {test_acc_val:.2%}")
    sync_print(f"  Avg Ponder Steps: {test_ponder_val:.2f}")

    sync_print("\n" + "=" * 60)
    sync_print("Training Complete!")
    sync_print("=" * 60)

    return {
        "test_loss": test_loss_val,
        "test_accuracy": test_acc_val,
        "test_ponder": test_ponder_val,
    }


if __name__ == "__main__":
    main()

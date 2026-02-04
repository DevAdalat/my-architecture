#!/usr/bin/env python3
"""Train DPSN-R nano model on synthetic algorithmic data."""

import sys
import time
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))

from src.data.synthetic import collate_fn, create_data_splits
from src.model import DPSNR, DPSNRConfig


def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute token-level accuracy ignoring padding."""
    mask = labels != -100
    if mask.sum() == 0:
        return 0.0
    predictions = logits.argmax(dim=-1)
    correct = (predictions == labels) & mask
    return (correct.sum() / mask.sum()).item()


def train_epoch(
    model: DPSNR,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_ce_loss = 0.0
    total_ponder_loss = 0.0
    total_acc = 0.0
    num_batches = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, labels=labels, use_act=True)

        loss = outputs["loss"]
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total_ce_loss += outputs["ce_loss"].item()
        total_ponder_loss += outputs["ponder_loss"].item()
        total_acc += compute_accuracy(outputs["logits"], labels)
        num_batches += 1

    return {
        "loss": total_loss / num_batches,
        "ce_loss": total_ce_loss / num_batches,
        "ponder_loss": total_ponder_loss / num_batches,
        "accuracy": total_acc / num_batches,
    }


@torch.no_grad()
def evaluate(
    model: DPSNR,
    dataloader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate model on dataloader."""
    model.eval()
    total_loss = 0.0
    total_ce_loss = 0.0
    total_acc = 0.0
    total_ponder = 0.0
    num_batches = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids, labels=labels, use_act=True)

        total_loss += outputs["loss"].item()
        total_ce_loss += outputs["ce_loss"].item()
        total_acc += compute_accuracy(outputs["logits"], labels)
        total_ponder += outputs["ponder_cost"].mean().item()
        num_batches += 1

    return {
        "loss": total_loss / num_batches,
        "ce_loss": total_ce_loss / num_batches,
        "accuracy": total_acc / num_batches,
        "avg_ponder": total_ponder / num_batches,
    }


def main():
    print("=" * 60)
    print("DPSN-R Nano Model Training on Synthetic Data")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    config = DPSNRConfig.from_preset("nano")
    print("\nModel Config (nano):")
    print(f"  vocab_size: {config.vocab_size}")
    print(f"  hidden_dim: {config.hidden_dim}")
    print(f"  num_layers: {config.num_layers}")
    print(f"  pool_size: {config.pool_size}")
    print(f"  max_reasoning_steps: {config.max_reasoning_steps}")

    model = DPSNR(config).to(device)
    param_counts = model.count_parameters()
    print("\nParameter Counts:")
    print(f"  Controller: {param_counts['controller']:,}")
    print(f"  Pool: {param_counts['pool']:,}")
    print(f"  Total: {param_counts['total']:,}")

    print("\nCreating synthetic datasets...")
    train_ds, val_ds, test_ds = create_data_splits(
        total_samples=5000,
        train_ratio=0.8,
        val_ratio=0.1,
        vocab_size=config.vocab_size,
        max_seq_len=config.max_seq_len,
        seed=42,
    )
    print(f"  Train: {len(train_ds)} samples")
    print(f"  Val: {len(val_ds)} samples")
    print(f"  Test: {len(test_ds)} samples")

    batch_size = 32
    collate = partial(collate_fn, max_len=config.max_seq_len)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate, drop_last=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate, drop_last=True
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-5)

    num_epochs = 20
    best_val_loss = float("inf")
    best_epoch = 0

    print("\n" + "=" * 60)
    print("Training")
    print("=" * 60)

    start_time = time.time()

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()

        train_metrics = train_epoch(model, train_loader, optimizer, device)
        val_metrics = evaluate(model, val_loader, device)

        scheduler.step()

        epoch_time = time.time() - epoch_start

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            best_marker = " *"
        else:
            best_marker = ""

        print(
            f"Epoch {epoch:2d}/{num_epochs} | "
            f"Train Loss: {train_metrics['loss']:.4f} | "
            f"Train Acc: {train_metrics['accuracy']:.2%} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.2%} | "
            f"Ponder: {val_metrics['avg_ponder']:.2f} | "
            f"Time: {epoch_time:.1f}s{best_marker}"
        )

    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time:.1f}s")
    print(f"Best validation loss: {best_val_loss:.4f} at epoch {best_epoch}")

    print("\n" + "=" * 60)
    print("Final Test Evaluation")
    print("=" * 60)

    test_metrics = evaluate(model, test_loader, device)
    print("\nTest Results:")
    print(f"  Loss: {test_metrics['loss']:.4f}")
    print(f"  CE Loss: {test_metrics['ce_loss']:.4f}")
    print(f"  Accuracy: {test_metrics['accuracy']:.2%}")
    print(f"  Avg Ponder Steps: {test_metrics['avg_ponder']:.2f}")

    print("\n" + "=" * 60)
    print("Sample Predictions")
    print("=" * 60)

    model.eval()
    sample_batch = next(iter(test_loader))
    sample_input = sample_batch["input_ids"][:3].to(device)
    sample_labels = sample_batch["labels"][:3].to(device)

    with torch.no_grad():
        outputs = model(sample_input, use_act=True)
        predictions = outputs["logits"].argmax(dim=-1)
        ponder_costs = outputs["ponder_cost"]

    for i in range(min(3, len(sample_input))):
        mask = sample_labels[i] != -100
        if mask.sum() == 0:
            continue

        target_tokens = sample_labels[i][mask].tolist()
        pred_tokens = predictions[i][mask].tolist()
        avg_ponder = ponder_costs[i].mean().item()

        match = target_tokens == pred_tokens
        match_str = "CORRECT" if match else "WRONG"

        print(f"\nSample {i + 1} [{match_str}] (ponder: {avg_ponder:.2f}):")
        print(f"  Target: {target_tokens[:20]}{'...' if len(target_tokens) > 20 else ''}")
        print(f"  Pred:   {pred_tokens[:20]}{'...' if len(pred_tokens) > 20 else ''}")

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)

    return test_metrics


if __name__ == "__main__":
    main()

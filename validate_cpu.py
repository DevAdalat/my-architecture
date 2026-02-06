"""
Validation Script for Mini DPSN-R Model
Trains a small model on synthetic data (including Associative Recall) to verify architecture.
Runs on CPU/CUDA (Standard PyTorch).
"""

import time
from functools import partial
import sys
import os

# Ensure project root is in sys.path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.synthetic import SyntheticDataset, collate_fn
from src.model.config import DPSNRConfig
from src.model.dpsn_r import DPSNR


def validate():
    # 1. Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running validation on {device}...")

    # 2. Config (Mini)
    # create small enough to run fast on CPU
    config = DPSNRConfig(
        vocab_size=1000,  # Small vocab for synthetic
        hidden_dim=128,
        num_heads=4,
        head_dim=32,
        pool_size=1024,
        pool_dim=128,  # Must be multiple of 128
        top_k=128,  # Must be multiple of 128 (XLA constraint)
        knowledge_ratio=0.5,
        reasoning_ratio=0.3,
        grammar_ratio=0.2,
    )

    # 3. Model
    model = DPSNR(config).to(device)
    print(f"Model created. Params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # 4. Data
    # 1000 samples, len 32 (enough for k-v pairs)
    dataset = SyntheticDataset(num_samples=1000, vocab_size=1000, max_seq_len=32)
    # Use custom collate_fn to handle padding
    collate = partial(collate_fn, max_len=32)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate)

    # 5. Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # 6. Training Loop
    model.train()
    print("Starting training loop...")

    t0 = time.time()
    for epoch in range(2):
        total_loss = 0
        steps = 0
        for step, batch in enumerate(dataloader):
            # batch is a dict now
            input_ids = batch["input_ids"].to(device)
            targets = batch["labels"].to(device)

            optimizer.zero_grad()

            # Forward
            # forward(input_ids, labels=None)
            # We want to pass labels to get internal loss?
            # Or calc manually.
            # DPSNR returns {"logits": ..., "loss": ...} if labels passed?
            # Let's assume we pass labels.
            outputs = model(input_ids, labels=targets)

            loss = outputs.get("loss")
            if loss is None:
                # Calc manually
                logits = outputs["logits"]
                loss = nn.functional.cross_entropy(
                    logits.view(-1, config.vocab_size), targets.view(-1)
                )

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            steps += 1

            if step % 10 == 0:
                print(f"Epoch {epoch} | Step {step} | Loss: {loss.item():.4f}")

        avg_loss = total_loss / steps
        print(f"Epoch {epoch} Finished. Avg Loss: {avg_loss:.4f}")

    print(f"Validation Complete in {time.time() - t0:.2f}s")


if __name__ == "__main__":
    validate()

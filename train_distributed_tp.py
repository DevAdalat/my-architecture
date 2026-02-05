"""
Multi-GPU Distributed Training Script for DPSN-R
Implements "Sparse Retrieval Tensor Parallelism" for the Massive Pool.

Usage:
    torchrun --nproc_per_node=2 train_distributed_tp.py
"""

import os
import sys
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data.dataset import TensorDataset

from src.model.config import DPSNRConfig
from src.model.dpsn_r import DPSNR
from src.data.synthetic import SyntheticDataset, collate_fn
from functools import partial
import time


def setup_distributed():
    """Initialize PyTorch Distributed."""
    # Check if we have CUDA
    if torch.cuda.is_available():
        backend = "nccl"
        device = "cuda"
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
    else:
        # Fallback to Gloo for CPU testing
        backend = "gloo"
        device = "cpu"
        # On CPU, local_rank is just an index, no specific device setting needed
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        # Fix for macOS Gloo issues
        if sys.platform == "darwin":
            os.environ["GLOO_SOCKET_IFNAME"] = "lo0"

    dist.init_process_group(backend=backend)
    return device, local_rank


def cleanup_distributed():
    dist.destroy_process_group()


def train():
    # 1. Setup Distributed Environment
    device, local_rank = setup_distributed()
    global_rank = dist.get_rank()
    world_size = dist.get_world_size()

    print(f"[Rank {global_rank}] Initializing... Device: {device}")

    # 2. Configuration
    # NOTE: Mini configuration for testing with Synthetic Data
    config = DPSNRConfig(
        vocab_size=1000,
        hidden_dim=128,
        num_heads=4,
        head_dim=32,
        pool_size=1024,
        pool_dim=128,  # Must be multiple of 128
        top_k=32,  # Small k for sparse fetch
        # Partition ratios (not sizes)
        knowledge_ratio=0.5,
        reasoning_ratio=0.3,
        grammar_ratio=0.2,
    )

    # 3. Model Initialization
    model = DPSNR(config).to(device)

    # Calculate Model Size
    total_params = sum(p.numel() for p in model.parameters())
    pool_params = sum(p.numel() for p in model.pool.parameters())
    if global_rank == 0:
        print(f"\n{'=' * 40}")
        print(f"Model Configuration")
        print(f"Total Parameters: {total_params / 1e6:.2f}M")
        print(f"Pool Parameters:  {pool_params / 1e6:.2f}M (Sharded across {world_size} GPUs)")
        print(f"Controller Params: {(total_params - pool_params) / 1e6:.2f}M")
        print(f"{'=' * 40}\n")

    # 4. Enable Custom Tensor Parallelism for the Pool
    print(f"[Rank {global_rank}] Sharding Massive Pool...")
    model.pool.enable_distributed(global_rank, world_size)

    # 5. Wrap Controller in DDP (Data Parallel)
    # FIX: We cannot wrap model.controller in DDP because dpsn_r.py calls custom methods
    # like 'encode', 'predict_halt' which DDP hides.
    # Instead, we will manually sync gradients for the controller in the training loop.
    print(f"[Rank {global_rank}] Using Manual Gradient Sync for Controller...")

    # 6. Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # 7. Data Loader
    # Use Synthetic Data with Associative Recall
    dataset = SyntheticDataset(num_samples=1000, vocab_size=1000, max_seq_len=32)
    collate = partial(collate_fn, max_len=32)
    sampler = DistributedSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=16, sampler=sampler, collate_fn=collate)

    # 8. Training Loop
    model.train()

    print(f"[Rank {global_rank}] Starting Training Loop...")

    # Metrics
    seq_len = 32  # Matches max_seq_len
    batch_size = 16
    recurrent_steps = 8  # Default for DPSN-R

    # Calculate Theoretical Bandwidth per Step (Forward Only)
    # We gather (Scores + Indices + Vectors) for (WorldSize * K) candidates
    # Done 'recurrent_steps' times per forward pass
    # Scores: Float32 (4 bytes), Indices: Int64 (8 bytes), Vectors: Float32 (4 bytes * dim)
    candidates_per_step = world_size * config.top_k
    bytes_scores = candidates_per_step * 4
    bytes_indices = candidates_per_step * 8
    bytes_vectors = candidates_per_step * config.pool_dim * 4

    payload_per_token_pass = bytes_scores + bytes_indices + bytes_vectors
    total_bytes_per_step_fwd = payload_per_token_pass * batch_size * seq_len * recurrent_steps
    # Backward pass usually involves similar or double traffic for gradients
    total_bytes_per_step_est = total_bytes_per_step_fwd * 2

    t0 = time.time()

    for epoch in range(1):
        sampler.set_epoch(epoch)
        for step, batch in enumerate(dataloader):
            # Unpack dict batch from collate_fn
            inputs = batch["input_ids"].to(device)
            targets = batch["labels"].to(device)

            optimizer.zero_grad()

            # Forward Pass
            # The model internally calls model.pool.retrieve(...)
            # which now triggers the distributed "Sparse Fetch" logic.
            outputs = model(inputs)

            # Loss Calculation (Standard dummy loss)
            # Assuming output is [B, S, V]
            logits = outputs["logits"]
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, config.vocab_size), targets.view(-1)
            )

            # Backward Pass
            loss.backward()

            # Manual Gradient Sync for Controller (Data Parallel)
            # Since we didn't use DDP, we must average gradients for the replicated parts.
            for param in model.controller.parameters():
                if param.grad is not None:
                    dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                    param.grad.data /= world_size

            # Optimizer Step
            optimizer.step()

            if step % 10 == 0 and global_rank == 0:
                t1 = time.time()
                dt = t1 - t0
                t0 = t1

                # Calculate Metrics
                if step == 0:
                    dt = 1.0  # Avoid div/0 on first step

                # 10 steps elapsed (except first)
                steps_elapsed = 10 if step > 0 else 1

                tps = (batch_size * seq_len * steps_elapsed) / dt

                # Bandwidth (MB/s)
                mb_moved = (total_bytes_per_step_est * steps_elapsed) / 1e6
                mbps = mb_moved / dt

                print(
                    f"Ep {epoch} | Step {step:3d} | Loss: {loss.item():.4f} | "
                    f"TPS: {tps:7.1f} tok/s | "
                    f"Comm: {mbps:6.1f} MB/s (Est)"
                )

    print(f"[Rank {global_rank}] Training Complete.")
    cleanup_distributed()


if __name__ == "__main__":
    train()

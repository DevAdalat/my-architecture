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


def generate_dummy_data(config, samples=1000):
    """Generate dummy data for testing training loop."""
    # Vocab size assumption: 50257 (GPT-2 style)
    input_ids = torch.randint(0, 50257, (samples, 128))
    labels = torch.randint(0, 50257, (samples, 128))
    return TensorDataset(input_ids, labels)


def train():
    # 1. Setup Distributed Environment
    device, local_rank = setup_distributed()
    global_rank = dist.get_rank()
    world_size = dist.get_world_size()

    print(f"[Rank {global_rank}] Initializing... Device: {device}")

    # 2. Configuration
    config = DPSNRConfig(
        vocab_size=50257,
        hidden_dim=512,
        pool_size=100000,  # Large pool
        pool_dim=256,
        top_k=128,  # Must be multiple of 128
        # Partition ratios (not sizes)
        knowledge_ratio=0.7,
        reasoning_ratio=0.2,
        grammar_ratio=0.1,
    )

    # 3. Model Initialization
    model = DPSNR(config).to(device)

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
    dataset = generate_dummy_data(config)
    sampler = DistributedSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=16, sampler=sampler)

    # 8. Training Loop
    model.train()

    print(f"[Rank {global_rank}] Starting Training Loop...")

    for epoch in range(1):
        sampler.set_epoch(epoch)
        for step, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)

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
                print(f"Epoch {epoch} | Step {step} | Loss: {loss.item():.4f}")

    print(f"[Rank {global_rank}] Training Complete.")
    cleanup_distributed()


if __name__ == "__main__":
    train()

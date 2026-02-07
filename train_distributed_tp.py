"""
Multi-GPU Distributed Training Script for DPSN-R
Implements "Sparse Retrieval Tensor Parallelism" for the Massive Pool.

Usage:
    torchrun --nproc_per_node=2 train_distributed_tp.py
"""

import argparse
import os
import sys
import time
from dataclasses import asdict
from functools import partial

# Ensure project root is in sys.path for distributed processes
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer

from src.data.hf_loader import get_hf_dataloader
from src.data.synthetic import SyntheticDataset, collate_fn
from src.model.config import DPSNRConfig
from src.model.dpsn_r import DPSNR
from src.utils.config_loader import load_config


def parse_args():
    parser = argparse.ArgumentParser(description="DPSN-R Distributed Training")
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml", help="Path to YAML config file"
    )
    parser.add_argument("--steps", type=int, default=None, help="Max steps per epoch (for testing)")
    return parser.parse_args()


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
    # On macOS with Gloo, destroy_process_group can hang for single-process runs
    if sys.platform != "darwin":
        dist.destroy_process_group()


def train():
    # 1. Setup Distributed Environment
    device, local_rank = setup_distributed()
    global_rank = dist.get_rank()
    world_size = dist.get_world_size()

    print(f"[Rank {global_rank}] Initializing... Device: {device}")

    # 2. Configuration
    args = parse_args()
    full_config = load_config(args.config)

    # Map ModelConfig to DPSNRConfig
    config = DPSNRConfig(**asdict(full_config.model))

    # 3. Model Initialization
    model = DPSNR(config).to(device)

    # Calculate Model Size
    total_params = sum(p.numel() for p in model.parameters())
    pool_params = sum(p.numel() for p in model.pool.parameters())
    if global_rank == 0:
        print(f"\n{'=' * 40}")
        print("Model Configuration")
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=full_config.training.lr)

    # 7. Data Loader
    if full_config.dataset.name == "synthetic":
        train_dataset = SyntheticDataset(
            num_samples=5000,
            vocab_size=config.vocab_size,
            max_seq_len=full_config.training.seq_len,
            seed=42,
        )
        val_dataset = SyntheticDataset(
            num_samples=500,
            vocab_size=config.vocab_size,
            max_seq_len=full_config.training.seq_len,
            seed=43,
        )
        train_sampler = DistributedSampler(train_dataset)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
        train_loader = DataLoader(
            train_dataset,
            batch_size=full_config.training.batch_size,
            sampler=train_sampler,
            collate_fn=partial(collate_fn, max_len=full_config.training.seq_len),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=full_config.training.batch_size,
            sampler=val_sampler,
            collate_fn=partial(collate_fn, max_len=full_config.training.seq_len),
        )
    else:
        # HuggingFace Loader (Streaming supported)
        train_loader = get_hf_dataloader(full_config.dataset, full_config.training)
        # Use same loader logic for validation (approximate)
        val_loader = get_hf_dataloader(full_config.dataset, full_config.training)
        train_sampler = None

    # 8. Training Loop
    model.train()

    # Load tokenizer for decoding
    tokenizer = None
    if global_rank == 0:
        try:
            tokenizer = AutoTokenizer.from_pretrained(full_config.dataset.tokenizer_name)
        except Exception as e:
            print(f"Warning: Could not load tokenizer for decoding: {e}")

    print(f"[Rank {global_rank}] Starting Training Loop...")

    # Metrics
    seq_len = full_config.training.seq_len
    batch_size = full_config.training.batch_size
    recurrent_steps = full_config.training.recurrent_steps

    num_epochs = full_config.training.epochs

    candidates_per_step = world_size * config.top_k
    bytes_scores = candidates_per_step * 4
    bytes_indices = candidates_per_step * 8
    bytes_vectors = candidates_per_step * config.pool_dim * 4

    payload_per_token_pass = bytes_scores + bytes_indices + bytes_vectors
    total_bytes_per_step_fwd = payload_per_token_pass * batch_size * seq_len * recurrent_steps
    total_bytes_per_step_est = total_bytes_per_step_fwd * 2

    t0 = time.time()

    for epoch in range(num_epochs):
        if train_sampler:
            train_sampler.set_epoch(epoch)
        model.train()
        total_loss = 0.0

        for step, batch in enumerate(train_loader):
            inputs = batch["input_ids"].to(device)
            targets = batch["labels"].to(device)

            optimizer.zero_grad()

            outputs = model(inputs)

            logits = outputs["logits"]
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, config.vocab_size), targets.view(-1)
            )

            loss.backward()

            for param in model.controller.parameters():
                if param.grad is not None:
                    dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                    param.grad.data /= world_size

            optimizer.step()

            total_loss += loss.item()

            if step % 10 == 0 and global_rank == 0:
                t1 = time.time()
                dt = t1 - t0
                if step == 0:
                    dt = 1.0  # Avoid division by zero
                t0 = t1

                # Calculate metrics for the last 10 steps (or 1 for first step)
                steps_elapsed = 10 if step > 0 else 1

                tps = (batch_size * seq_len * steps_elapsed) / dt

                # Estimate communication bandwidth (very rough)
                mb_moved = (total_bytes_per_step_est * steps_elapsed) / 1e6
                mbps = mb_moved / dt

                print(
                    f"Ep {epoch} | Step {step:3d} | Loss: {loss.item():.4f} | "
                    f"TPS: {tps:7.1f} tok/s | "
                    f"Comm: {mbps:6.1f} MB/s (Est)"
                )

            if step > 0 and step % full_config.training.generate_steps == 0:
                if global_rank == 0:
                    print(f"\n[Step {step}] Generating sample...")

                model.eval()
                with torch.no_grad():
                    prompt_text = "Once upon a time"
                    if global_rank == 0:
                        prompt_ids = tokenizer.encode(prompt_text, return_tensors="pt").to(device)
                    else:
                        prompt_ids = torch.zeros((1, 4), dtype=torch.long).to(device)

                    dist.broadcast(prompt_ids, src=0)

                    gen_len = 20
                    curr_ids = prompt_ids

                    for _ in range(gen_len):
                        out = model(curr_ids)
                        temperature = 0.8
                        logits = out["logits"][:, -1, :] / temperature
                        probs = torch.softmax(logits, dim=-1)
                        next_token = torch.multinomial(probs, num_samples=1)
                        curr_ids = torch.cat([curr_ids, next_token], dim=1)

                    if global_rank == 0:
                        print(f'Prompt: "{prompt_text}"')
                        print(f"Generated Tokens: {curr_ids[0].tolist()}")
                        if tokenizer:
                            decoded_text = tokenizer.decode(
                                curr_ids[0].tolist(), skip_special_tokens=True
                            )
                            print(f"Generated Text: {decoded_text}")
                        print(f"{'-' * 40}")

                model.train()

            if (
                full_config.training.save_steps > 0
                and step > 0
                and step % full_config.training.save_steps == 0
            ):
                if global_rank == 0:
                    print(f"\n[Step {step}] Saving checkpoint...")
                    checkpoint_path = f"checkpoints/step_{step}.pt"
                    torch.save(
                        {
                            "epoch": epoch,
                            "step": step,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "loss": loss.item(),
                        },
                        checkpoint_path,
                    )
                    print(f"Checkpoint saved to {checkpoint_path}")

            if args.steps and step >= args.steps:
                break

        # Calculate training loss
        avg_loss = 0.0
        try:
            # If dataset has length
            avg_loss = total_loss / len(train_loader)
        except TypeError:
            # If streaming/iterable
            steps_taken = step + 1 if "step" in locals() else 1
            avg_loss = total_loss / steps_taken

        if global_rank == 0:
            print(f"Ep {epoch} | Training Avg Loss: {avg_loss:.4f}")

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                if args.steps and i >= args.steps:
                    break
                inputs = batch["input_ids"].to(device)
                targets = batch["labels"].to(device)
                outputs = model(inputs)
                logits = outputs["logits"]
                loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, config.vocab_size), targets.view(-1)
                )
                val_loss += loss.item()

        try:
            num_val_batches = len(val_loader)
        except TypeError:
            num_val_batches = args.steps if args.steps else 100  # Fallback

        avg_val_loss = val_loss / num_val_batches
        val_loss_tensor = torch.tensor(avg_val_loss).to(device)
        dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
        avg_val_loss = val_loss_tensor.item() / world_size

        if global_rank == 0:
            print(f"Ep {epoch} | Validation Loss: {avg_val_loss:.4f}")

    if global_rank == 0:
        print("\n" + "=" * 40)
        print("Running Generation Demo...")
        print("=" * 40)

    model.eval()

    prompt_text = "Once upon a time"
    if global_rank == 0:
        start_tokens = tokenizer.encode(prompt_text, return_tensors="pt").to(device)
    else:
        start_tokens = torch.zeros((1, 4), dtype=torch.long).to(device)

    dist.broadcast(start_tokens, src=0)

    generated = start_tokens
    max_new_tokens = 10

    with torch.no_grad():
        for _ in range(max_new_tokens):
            outputs = model(generated)
            next_token_logits = outputs["logits"][:, -1, :]
            temperature = 0.8
            logits = next_token_logits / temperature
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            dist.broadcast(next_token, src=0)

            generated = torch.cat([generated, next_token], dim=1)

    if global_rank == 0:
        print(f"Input: {tokenizer.decode(start_tokens[0].tolist())}")
        print(f"Generated: {tokenizer.decode(generated[0].tolist())}")
        print("=" * 40)

    print(f"[Rank {global_rank}] Training Complete.")
    print(f"[Rank {global_rank}] Cleaning up distributed process group...")
    # On macOS with Gloo, destroy_process_group can hang.
    # We set a timeout or force exit if it takes too long.
    if sys.platform == "darwin" and dist.get_backend() == "gloo":
        print(
            f"[Rank {global_rank}] macOS/Gloo detected: skipping destroy_process_group to avoid hang."
        )
    else:
        cleanup_distributed()
    print(f"[Rank {global_rank}] Cleanup complete.")
    # Force exit to kill any lingering background threads (e.g. from datasets streaming)
    import os

    os._exit(0)
    sys.exit(0)


if __name__ == "__main__":
    train()

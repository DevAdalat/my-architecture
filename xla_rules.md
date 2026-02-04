# PyTorch XLA & TPU Development Rules

This document outlines the critical rules, best practices, and performance guidelines for developing models with PyTorch XLA, specifically targeting TPU execution.

## 1. The Core Principle: "Compile Once, Execute Many"

TPUs use the XLA (Accelerated Linear Algebra) compiler. Unlike standard PyTorch (eager execution), XLA uses a **Lazy Tensor** model.
- Operations are recorded into a graph.
- The graph is **compiled** into a TPU executable only when a result is needed.
- Compilation is expensive. Performance depends on reusing the same compiled graph.

**Rule:** You must write code that produces a stable, static computation graph that can be cached and reused.

---

## 2. Static Shapes (The Golden Rule)

XLA compiles a new kernel for every unique input shape. If your batch size or sequence length changes, XLA recompiles. This causes "compilation thrashing" and makes training 100x slower.

### Rules:
1.  **Fixed Batch Size**:
    - Always use `drop_last=True` in your DataLoaders.
    - Never allow the final batch to be smaller than the others.
2.  **Fixed Sequence Length**:
    - Pad all inputs to a fixed maximum length (e.g., 512, 1024).
    - If variable lengths are unavoidable, use "bucketing" (pad to the nearest multiple of 128) to limit the number of unique shapes.
3.  **Avoid Data-Dependent Shapes**:
    - Operations where the output shape depends on the *values* of the input are forbidden in the graph.
    - **BANNED**: `torch.nonzero()`, `torch.unique()`, boolean indexing `tensor[mask]`.
    - **ALTERNATIVE**: Use masks (`torch.where`) and keep tensors at full size.

---

## 3. Graph Breaks & CPU Fallback

A "Graph Break" occurs when the TPU execution must stop, return data to the CPU (Python), and potentially restart. This kills performance.

### Rules:
1.  **NO `.item()`**:
    - Never call `.item()` on a tensor inside the training loop. This forces an immediate blocking read from device to CPU.
    - *Exception*: Logging loss *after* `xm.mark_step()` or `optimizer_step()`.
2.  **NO Python Control Flow on Tensors**:
    - **Bad**: `if tensor.mean() > 0.5: ...` (Requires value on CPU).
    - **Good**: `torch.where(tensor.mean() > 0.5, true_branch, false_branch)`.
3.  **Printing**:
    - Do not print tensors for debugging inside the loop unless necessary. It triggers a graph execution.

---

## 4. TPU Memory Layout

TPUs are systolic arrays that operate most efficiently on specific dimensions.

### Rules:
1.  **Multiples of 128**:
    - Ensure feature dimensions (hidden size, embedding dim) are multiples of 128 (or at least 8).
    - Padding dimensions to 128 can significantly improve throughput (up to 3x).
2.  **Precision**:
    - TPUs natively support **BF16** (BFloat16).
    - Use `torch.autocast` or simply allow XLA to handle BF16. It offers float32 range with float16 performance.

---

## 5. Implementation Guidelines

### Device Management
- **Never hardcode "cuda" or "cpu"**.
- Use the standard `xm` utility:
```python
import torch_xla.core.xla_model as xm
device = xm.xla_device()
model.to(device)
```

### Optimization Step
- Use `xm.optimizer_step(optimizer)` instead of `optimizer.step()`.
- This function handles the critical `mark_step()` call, triggering XLA execution at the optimal time.
```python
loss.backward()
xm.optimizer_step(optimizer) # Includes barrier & mark_step
```

### Data Loading
- Use `MpDeviceLoader` for high-performance parallel data loading to the TPU.
```python
import torch_xla.distributed.parallel_loader as pl
train_loader = pl.MpDeviceLoader(original_loader, device)
```

## 6. Debugging

If performance is slow, check for recompilation:
1.  **Metrics Report**: Call `xm.master_print(met.metrics_report())` periodically.
2.  **CompileTime**: If this metric keeps increasing after the first few steps, you have a dynamic shape or graph break issue.

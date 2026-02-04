# AGENTS.md

## Project Context
**Name**: DPSN-R (Dynamic Parameter Selection Network with Recurrent Reasoning)
**Description**: A novel architecture decoupling model knowledge capacity from active computational cost using Recurrent Reasoning and Adaptive Compute (ACT).
**Architecture Type**: Sparse Recurrent Neural Network with Externalized Parameter Pool.
**Current Status**: Proof of Concept / Research Demo (Loss ~1.4 on TinyStories).

---

## 🏗️ Architecture Guidelines (DPSN-R)
The codebase must strictly implement the components described in the system architecture.

### 1. The Controller
The "Executive Center" of the model.
- **Responsibility**: Encoding context, generating retrieval queries, and deciding when to halt (ACT).
- **Constraint**: Designed to be computationally efficient relative to the total knowledge capacity; primarily handles logic and routing.

### 2. The Massive Partitioned Pool
The "Externalized Brain." Organized into semantic regions:
- **Knowledge Partition**: Facts, concepts, domain knowledge.
- **Reasoning Partition**: Abstract logic patterns, inference rules.
- **Grammar Partition**: Language structure, fluency markers.

### 3. Recurrent Reasoning Loop
- Every token hidden state cycles through the architecture (default max: 8 loops).
- **Phases**:
    1. **Understanding** (Knowledge focus)
    2. **Reasoning** (Logic focus)
    3. **Expression** (Grammar focus)

---

## ⚡ TPU & XLA Development Guidelines (CRITICAL)
TPU support via `torch_xla` is the primary optimization target. **Failure to follow these rules will cause 100x performance degradation.**

### 1. The Core Principle: "Compile Once, Execute Many"
XLA uses a **Lazy Tensor** model. Performance relies on reusing cached, compiled graphs. Any shape change or graph break forces an expensive recompilation.

### 2. Static Shapes (The Golden Rule)
- **Fixed Batch Size**: Always use `drop_last=True` in DataLoaders.
- **Fixed Sequence Length**: Pad all inputs to a fixed length or use static buckets (multiples of 128).
- **BANNED**: Data-dependent shapes like `torch.nonzero()`, `torch.unique()`, or boolean indexing `tensor[mask]`.
- **ALTERNATIVE**: Use `torch.where()` masks and keep tensors at full size.

### 3. Graph Breaks & CPU Fallback
- **NO `.item()`**: Never call `.item()` inside the training loop. It forces a blocking sync to CPU.
- **NO Python Control Flow on Tensors**: Use `torch.where()` instead of `if tensor.mean() > 0.5:`.
- **Mark Steps**: Use `xm.optimizer_step(optimizer)` which handles `xm.mark_step()` internally.

### 4. Implementation Standards
- **Device Agnostic**: Never hardcode "cuda" or "cpu". Use `device = xm.xla_device()`.
- **Multiples of 128**: Ensure hidden dimensions and embedding sizes are multiples of 128 for systolic array efficiency.
- **Precision**: Use **BF16** natively.

---

## 🛠️ Environment & Build
- **Python**: >= 3.11
- **Dependency Management**: `uv` (via `pyproject.toml`). **NEVER use `pip` directly.**
- **Linter/Formatter**: `ruff` (Strictly enforced).

### Essential Commands
- **Install**: `uv pip install -e .`
- **Lint/Format**: `ruff check .` and `ruff format .`
- **Test**: `pytest`
- **Benchmark XLA**: `python benchmark_xla_compilation.py`

---

## 📏 Code Style & Standards

### Typing & Docs
- **Mandatory Type Hints**: All function signatures and complex variables must be typed.
- **Google/NumPy Style Docstrings**: Required for all public modules and classes.

### Naming Conventions
- **Classes**: `PascalCase`
- **Functions/Variables**: `snake_case`
- **Constants**: `UPPER_CASE`

### Error Handling
- Use specific exceptions (e.g., `ValueError`, `RuntimeError`).
- **Fail Fast**: Provide descriptive errors early in the execution chain.

---

## 🧪 Testing & Git Workflow
- **Unit Tests**: Focus on Controller and Retrieval logic.
- **Integration Tests**: Verify the full recurrent generation loop.
- **Commits**: Follow [Conventional Commits](https://www.conventionalcommits.org/).
- **Cleanliness**: Run `ruff` and `lsp_diagnostics` before committing.

# Dynamic Parameter Selection Network (DPSN) Architecture

The DPSN is a specialized neural architecture designed to decouple **model size** from **computational cost**. Unlike traditional Transformers where every parameter is used for every token, DPSN dynamically selects a tiny subset of "active" parameters from a massive pool for each computation step.

This architecture behaves like a **Recurrent Neural Network (RNN)** that has access to a massive, external "brain" (the Parameter Pool) and a "router" that decides which parts of that brain to use at any given moment.

## 1. High-Level Data Flow

The model processes input tokens through a recurrent loop. Instead of stacking fixed layers (Layer 1 → Layer 2 → ... → Layer N), the DPSN reuses the **same components** multiple times (Recurrence Steps).

**The Forward Pass Loop:**
1.  **Input**: Token embeddings + Step Embeddings (to tell the model "this is step 3 of 10").
2.  **Self-Attention**: Standard mechanism to gather context from other tokens.
3.  **Router**: Analyzes the current context and asks: *"Which parameters do I need?"*
4.  **Parameter Retrieval**: Fetches only the requested parameters (e.g., 32 out of 1,000,000) from the Pool.
5.  **Execution Engine**: Synthesizes the fetched parameters into a result and updates the token state.
6.  **Repeat**: This updated state is fed back into step 2 for the next recurrence step.

---

## 2. Component Deep Dive

### A. The Embedding Model (`StepEmbedding`)
**File:** `src/model/step_embedding.rs`

In a recurrent model, the network needs to know "where" it is in the computation process. Since the same physical layers are reused, a signal is required to differentiate Step 1 (initial understanding) from Step 10 (final refinement).

*   **How it works**: It maintains a lookup table of learnable vectors, one for each recurrence step.
*   **Mechanism**: Before entering the attention layer, the embedding for the current step `t` is added to the token's hidden state.
    ```rust
    // Simplified Logic
    x = x + step_embeddings[step_index];
    ```
*   **Why it's important**: It allows the single set of weights to specialize. The model can learn to perform "rough processing" at Step 0 and "fine-tuning" at Step N, effectively creating a deep network with shared parameters.

### B. The Router Model (`Router` & `HierarchicalRouter`)
**File:** `src/model/router.rs`, `src/model/hierarchical_router.rs`

This is the brain's "manager." It determines the sparsity and specialization of the model.

*   **Dynamic Budgeting**:
    The router doesn't just pick parameters; it decides *how many* it needs. It projects the input to a scalar **complexity score** (0.0 to 1.0) using a sigmoid function.
    *   *Low complexity input* (e.g., "the"): Select fewer parameters (lower compute).
    *   *High complexity input* (e.g., "quantum physics"): Select more parameters (higher compute).

*   **Selection Mechanism**:
    1.  **Relevance Scoring**: It computes a score for every parameter vector in the pool against the current input `x`.
    2.  **Top-K Selection**: It selects the top `k` indices with the highest scores.

*   **Hierarchical Routing (Optimization)**:
    Searching a pool of 1 million vectors is slow. The `HierarchicalRouter` organizes parameters into **clusters**.
    1.  **Coarse Router**: Selects the top relevant *clusters*.
    2.  **Fine Router**: Searches only within those selected clusters.
    *   **Efficiency**: Reduces search complexity from $O(N)$ to $O(\sqrt{N})$.

### C. The Parameter Pool (`ParameterPool`)
**File:** `src/model/parameter_pool.rs`, `src/model/offloaded_pool.rs`

This is the massive storage of knowledge. It is a simple, large tensor of shape `[pool_size, parameter_dim]`.

*   **The "Key-Value" Concept**: Think of the pool as a dictionary. The indices chosen by the Router are the "keys," and the vectors returned are the "values."
*   **Offloading (The "Infinite" Model)**:
    Because the pool is just storage and not matrix multiplication, it doesn't need to live on the GPU.
    *   **CPU Offloading**: The `OffloadedPool` keeps the 10GB+ parameter tensor on system RAM.
    *   **On-Demand Transfer**: Only the tiny fraction of active vectors (e.g., 32 vectors) are copied to the GPU during the forward pass.
    *   **Why it's efficient**: You can run a 100 Billion parameter model on a consumer GPU, provided you only use ~1 Billion active parameters at a time. The bottleneck becomes PCIe bandwidth, not VRAM.

### D. The Execution Engine
**File:** `src/model/execution_engine.rs`

Once the parameters are fetched, the Execution Engine applies them.

*   **Weighted Aggregation**:
    The router provides `scores` (how relevant a parameter is) and `indices` (which parameter it is). The engine uses a softmax over the scores to create **attention weights** for the parameters themselves.
    ```rust
    // Concept
    active_params = pool.retrieve(indices) // shape: [k, dim]
    weights = softmax(router_scores)       // shape: [k, 1]
    combined_param = sum(active_params * weights)
    ```
*   **Interaction**: The `combined_param` is essentially a dynamically generated weight vector specific to the current input token. This vector is then added to the residual stream.

---

## 3. Why This Architecture is Important

### 1. Decoupled Compute vs. Capacity
In standard Transformers (GPT-4, Llama 3), doubling the parameter count doubles the inference cost (FLOPs).
*   **DPSN**: You can increase the Parameter Pool size from 1 million to 1 billion. The **inference cost remains constant** because the router still only selects `k` parameters. You get a smarter model for the same speed.

### 2. Adaptive Computation
Not all tokens are created equal. Processing the word "the" shouldn't take as much brainpower as "thermodynamics."
*   **DPSN**: The dynamic budget mechanism allows the model to "rest" on easy tokens and "focus" on hard ones, saving energy and time.

### 3. Hardware Accessibility (Offloading)
By keeping the inactive 99% of parameters on CPU RAM, DPSN enables running massive models on commodity hardware. The GPU only needs to hold the "working memory" (activations) and the tiny slice of "active knowledge."

## 4. Summary of Data Flow

```mermaid
graph TD
    Input[Input Token] -->|Recurrence Start| Embedding
    Embedding --> Attention
    Attention -->|Context Vector| Router
    
    subgraph "Dynamic Selection"
        Router -->|1. Calculate Complexity| Budget[K Budget]
        Router -->|2. Compute Scores| Selector[Top-K Selector]
        Budget --> Selector
        Selector -->|Indices| Pool[Parameter Pool (CPU/RAM)]
    end
    
    Pool -->|Active Params (GPU)| Engine[Execution Engine]
    Selector -->|Weights| Engine
    Attention -->|Residual| Engine
    
    Engine -->|Update State| Output
    Output -->|Next Step| Embedding
```

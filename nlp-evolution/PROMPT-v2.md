**System Role & Persona:**
Act as an Expert AI Educator, Senior LLM Researcher, and Data Visualization Engineer. Your goal is to break down the architectural, compute, and coding-paradigm shifts behind modern Large Language Models (LLMs) and Agentic AI into an intuitive, highly visual, technically accurate, and reusable educational asset.

**Target Audience:**
Software engineers, data scientists, and technical students who understand basic ML/deep learning and want to be job-ready for today's LLM and Agentic AI ecosystem — able to read, implement, modify, and reason about modern Transformer-based systems.

**Scope & Time Focus (IMPORTANT):**
Many of these algorithms were coined decades ago (e.g., RNNs in the 1980s, LSTM in 1997, attention mechanisms in 2014). Mention the **original date of inception** of each algorithm in one line, but **focus the content on the last 5–8 years (≈2017–2026)** — the period where **compute** (GPU/TPU scale, mixed precision, distributed training), **data** (web-scale pre-training corpora, instruction and preference data), and the **coding paradigm** (PyTorch 2.x, Hugging Face, scaled-dot-product attention kernels, PEFT, inference servers, agent frameworks) shifted fundamentally. Prioritize the latest trends, code, and architecture examples that genuinely prepare the reader for modern LLMs and Agentic AI. Pre-2017 history is context only, never the focus.

**Task Overview:**
Generate a comprehensive, reusable educational guide and interactive asset that explains how we got to modern LLMs and agents, bridging high-level intuition and low-level mechanics (math and code). Structure the response strictly into the following five deliverables.

### Deliverable 1: The Evolutionary Timeline & Paradigm Shifts
Open with a short **Prelude (pre-2017, one screen max)** covering only *why the Transformer was needed*: sparse representations (BoW/TF-IDF), dense embeddings (Word2Vec 2013), recurrent models (RNN 1986, LSTM 1997 — popular 2014–15), and Seq2Seq + attention (Bahdanau 2014, Luong 2015). One line each, with inception dates.

Then cover the main eras in depth. For each, state the **Bottleneck of the Previous Era**, the **Core Breakthrough**, the **New Capabilities Unlocked**, and **What Changed in Compute / Data / Code**:
1. **The Paradigm Shift (2017):** The Transformer & Self-Attention — parallelism, all-to-all token connections, GPU-friendly matrix math.
2. **Pre-train, then Adapt (2018–2020):** Encoder-only (BERT, masked LM) vs. Decoder-only (GPT-2/GPT-3, autoregressive); the rise of Hugging Face Transformers; in-context / few-shot learning.
3. **The Scaling Era (2020–2022):** Scaling laws (Kaplan 2020, Chinchilla 2022); mixed precision (bf16), FSDP/DeepSpeed ZeRO, FlashAttention.
4. **The Alignment Era (2022–2023):** Instruction tuning, RLHF (InstructGPT), and DPO; the shift from base models to chat models.
5. **The Modern Decoder Block (2023–2025):** Anatomy of a Llama-style model — pre-norm, **RMSNorm**, **RoPE**, **SwiGLU**, **GQA/MQA**, **KV cache**; why each replaced its predecessor.
6. **Efficiency at Scale (2023–2026):** **Mixture of Experts** (Mixtral, DeepSeek-V3), **LoRA/QLoRA**, quantization, inference serving (vLLM/PagedAttention, speculative decoding).
7. **Grounding & Memory (2020–2026):** **RAG** (Lewis et al. 2020 → production hybrid/graph RAG), long-context models, embeddings and vector search.
8. **Reasoning & Agentic AI (2024–2026):** Test-time compute and reasoning models, RL with verifiable rewards (e.g., GRPO), **tool/function calling**, the **Model Context Protocol (MCP)**, and multi-agent orchestration (e.g., LangGraph).

### Deliverable 2: Architectural Deep-Dives (Mermaid Diagrams & Math)
For the most critical modern leaps, provide **Mermaid.js diagrams** (`flowchart TD`, valid syntax) showing architecture, data flow, and tensor shapes, plus **mathematical intuition** in LaTeX:
1. **Original Transformer block (2017) vs. Modern Llama-style decoder block** — side by side, highlighting every component that changed.
2. **Scaled Dot-Product & Multi-Head Attention** — break down $Attention(Q,K,V) = softmax(\frac{QK^T}{\sqrt{d_k}} + M)V$, including the causal mask $M$; explain in plain English what Q, K, and V represent and why we divide by $\sqrt{d_k}$.
3. **Inference with KV cache & GQA** — prefill vs. decode phases and the memory math.
4. **Mixture of Experts** — router/gating equations, top-k selection, load-balancing loss.
5. **RAG pipeline and an Agent loop** — retrieve → augment → generate, and plan → tool call → observe → reflect.

### Deliverable 3: Minimal, Heavily-Commented Modern PyTorch Code
Provide clean, minimal, executable **PyTorch 2.x** snippets that demystify the "black box." Do not use high-level wrappers (`nn.Transformer`, `nn.MultiheadAttention`, `nn.LSTM`); write the underlying tensor operations, then **verify** each against the official reference (e.g., `F.scaled_dot_product_attention`) with an `assert torch.allclose(...)`. Include:
1. **Causal Multi-Head Self-Attention** — Q/K/V projections, head split, scaled dot-product, causal mask, softmax, output projection; then show the one-line `F.scaled_dot_product_attention` equivalent and why it is faster.
2. **RoPE** (rotary positional embeddings).
3. **A modern decoder block** — RMSNorm + GQA attention with KV cache + SwiGLU MLP + residuals.
4. **A top-k MoE layer** — router, expert dispatch, load-balancing loss.
5. **A LoRA linear layer** — frozen base weight + low-rank adapters.
6. **A minimal agent loop** — framework-agnostic tool/function calling with a mocked LLM and a tiny retrieval tool (so it runs offline).
7. *(Optional, ≤15 lines)* a single RNN/LSTM step purely for contrast with attention.

### Deliverable 4: Visual Animation Generation Script (Manim)
Provide a complete, executable Python script using **Manim Community Edition** that renders:
* **Scene 1:** An animated timeline from **2017 to 2026**, with a small "pre-2017 prelude" stub on the left.
* **Scene 2:** **RNN sequential processing** (tokens lighting up one by one) vs. **Transformer self-attention** (all tokens lighting up simultaneously with connecting lines representing the all-to-all attention matrix).
* **Scene 3:** **MoE routing** — tokens flowing through a router to their top-2 experts — or **KV-cache decoding** growing token by token.
* Use only standard Manim imports, and include comments on how to run each scene via the CLI.

### Deliverable 5: Interactive Reusable Asset
Package everything into a single self-contained, light-themed interactive HTML page with:
* a clickable era timeline (bottleneck → breakthrough → capabilities → compute/data/code cards);
* live-rendered Mermaid diagrams and KaTeX math, with hover-to-explain Q/K/V terms;
* an **attention playground** (type a sentence, see the attention heatmap, toggle heads and the causal mask);
* an in-browser RNN-vs-Transformer animation and a MoE routing visual;
* tabbed, copyable code from Deliverable 3, and the rendered Manim video embedded.

**Formatting & Quality Constraints:**
* Use bolding for key terminology (e.g., **KV cache**, **autoregressive**, **mixture of experts**).
* Keep the tone engaging, authoritative, and accessible.
* All code blocks must specify their language (`python`, `mermaid`, `bash`).
* Format LaTeX with `$` for inline and `$$` for display equations.
* Cite the original paper and year for every milestone; prefer post-2017 sources and examples.
* All code must run on CPU in under a minute; the Mermaid must parse without errors.

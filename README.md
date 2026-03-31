# tinyllm

<p align="center">
  <img src="images/robot_shakespeare.png" alt="A tiny robot reading Shakespeare by candlelight, with transformer layers glowing inside its transparent head" width="700">
</p>

Building a tiny LLM from scratch in PyTorch to understand how modern language models work — from first principles to a published model.

**The idea**: GPT-4, Claude, and LLaMA are all scaled-up versions of the same architecture. Build it small, understand it completely, then scale up.

## What this is

A ~10M parameter decoder-only transformer trained on Shakespeare, progressively modernized with the same components used in LLaMA, Qwen, and Mistral:

| Component | GPT-2 era (vanilla) | Modern (what we upgrade to) | Why |
|-----------|--------------------|-----------------------------|-----|
| Normalization | LayerNorm | **RMSNorm** | Simpler, faster, equally effective |
| FFN activation | ReLU | **SwiGLU** | Gated mechanism, better gradient flow |
| Position encoding | Learned embeddings | **RoPE** | Relative positions, length generalization |
| Inference | Recompute everything | **KV Cache** | O(n) generation instead of O(n²) |

After all four upgrades, the architecture is structurally identical to LLaMA / Qwen — just 1000× smaller.

## Project structure

```
tokenizer.py         # Character-level tokenizer + data loading
attention.py         # Single-head causal self-attention from scratch
transformer.py       # Multi-head attention, FFN, transformer Block
model.py             # Full vanilla GPT (~10.8M params)
train.py             # Training loop (AdamW, 5000 steps)
generate.py          # Text generation with temperature, top-k, top-p sampling

swap1_rmsnorm.py     # Isolated test: LayerNorm → RMSNorm
swap2_swiglu.py      # Isolated test: ReLU FFN → SwiGLU
swap3_rope.py        # Isolated test: Learned pos emb → RoPE
swap4_kvcache.py     # Benchmark: generation with/without KV cache

modernize.py         # All modern components (RMSNorm, SwiGLU, RoPE, KV cache)
model_modern.py      # Full modernized GPT (~10.6M params)
train_modern.py      # Train the modernized model
train_bpe.py         # BPE tokenizer + mixed precision + gradient accumulation

DEVLOG.md            # Full learning journal with explanations
```

## How to run

```bash
# Setup
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Train the vanilla transformer (5000 steps, ~85 min on M4 Mac)
python -u train.py

# Generate Shakespeare
python generate.py --demo                # all sampling configs
python generate.py --prompt "ROMEO:" --temp 0.8

# Run individual swap comparisons (2000 steps each)
python -u swap1_rmsnorm.py
python -u swap2_swiglu.py
python -u swap3_rope.py
python swap4_kvcache.py                  # benchmarks, no training needed

# Train the full modern model
python -u train_modern.py

# Train with BPE + mixed precision (Phase 4)
python -u train_bpe.py
```

## The learning journey

### Step 1: Build a vanilla transformer from scratch

Started with the absolute basics: read Shakespeare, map each of 65 unique characters to a number, and build the training loop. Then built attention from the ground up — the Q/K/V projections, the scaled dot product, the causal mask that prevents the model from seeing the future.

Stacked 6 transformer blocks with multi-head attention (6 heads), ReLU feed-forward networks, pre-norm architecture, and residual connections. Total: **10.8M parameters**.

### Step 2: Train it and watch it learn

The untrained model starts with loss ~4.2 (randomly guessing among 65 characters). After 5000 training steps, loss drops to ~1.5 — the model learns word shapes, common phrases, character names, and basic Shakespeare structure.

### Step 3: Swap components one at a time

The key insight from this project: **change one thing at a time and measure the effect.** Each swap script modifies exactly one component, retrains for 2000 steps with identical hyperparameters, and prints the loss curve for comparison.

- **RMSNorm**: Dropping the mean subtraction from LayerNorm. Same quality, fewer operations.
- **SwiGLU**: Replacing the hard ReLU cutoff with a smooth learned gate. Noticeably better loss.
- **RoPE**: Rotating Q and K vectors instead of adding position embeddings. Encodes relative position for free, generalizes to longer sequences.
- **KV Cache**: Caching Key/Value tensors during generation so each new token only needs one forward pass instead of reprocessing the entire sequence.

### Step 4: Scale up

Switched from character-level to BPE tokenization (50,257 tokens), added mixed precision training (float16), and gradient accumulation to simulate larger batch sizes.

## Results

### Training (Google Colab T4)

| Model | Params | Best Val Loss | Best Step | Training Time |
|-------|--------|-------------|-----------|--------------|
| Vanilla (char-level) | 10.8M | **1.4837** | 3000 | 59.7 min |
| Modern (char-level) | 10.6M | **1.4783** | 2500 | 66.9 min |
| Modern (BPE) | 29.9M | 4.6414 | 1000 | 68.2 min |

Modern architecture beats vanilla — better loss, reached faster, with fewer parameters.

### Swap comparison (each tested in isolation, 2000 steps)

| Swap | Val loss at step 500 | vs Vanilla |
|------|---------------------|-----------|
| Vanilla (baseline) | 1.99 | — |
| RMSNorm | 1.99 | same |
| SwiGLU | 1.88 | **-0.11** |
| **RoPE** | **1.68** | **-0.31** |

RoPE was the biggest single improvement.

### Sample output (vanilla model, temp=0.8)

```
ROMEO:
Nay, be too be so head: but I am as betimes;
There is no man with her pleasure attentience,
She doth behold our queen arms.

PAULINA:
I'll not too woe to die for the law to the world,
```

### Throughput (Colab T4)

| Model | tok/s | 300 tokens |
|-------|-------|-----------|
| Vanilla (no cache) | 72.2 | 4.16s |
| Modern (KV cache) | 40.7 | 7.37s |

## What I learned

1. **RoPE is the most impactful modern architecture change** — 0.31 better at step 500, fewer params, relative position for free
2. **More powerful models overfit faster on small data** — modern hit best val at step 2500 then degraded. Early stopping is essential.
3. **MPS (Apple Silicon) has a memory leak** that silently kills training after 60-80 min. Move to CUDA for anything serious.
4. **Mixed precision float16 diverges on MPS** with large vocabularies. CUDA's GradScaler handles this; MPS doesn't have it.
5. **BPE only pays off at scale** — 50K vocab on 1MB of data means most embeddings never train properly
6. **When loss is good but output is garbage, the bug is in inference** — we had a RoPE position bug in KV cache generation that took hours to find
7. **Change one thing at a time and measure** — the per-swap comparison approach showed exactly what each architectural change contributes

See [DEVLOG.md](DEVLOG.md) for the full learning journal with detailed explanations of every concept, error, and fix.

## Architecture details

```
ModernGPT(
  token_emb:   Embedding(50257, 384)     # BPE token embeddings
  blocks × 6:                             # 6 transformer layers
    rn1:       RMSNorm(384)               #   pre-norm (not LayerNorm)
    attn:      MultiHeadAttention(        #   6 heads × 64 dims
                 Q, K with RoPE           #     rotary position encoding
                 KV Cache for inference   #     cached keys/values
               )
    rn2:       RMSNorm(384)
    ffn:       SwiGLU(384 → 1024 → 384)  #   gated FFN (not ReLU)
  ln_f:        RMSNorm(384)               # final norm
  lm_head:     Linear(384, 50257)         # tied with token_emb
)
```

## Resources

- [build-nanogpt](https://github.com/karpathy/build-nanogpt) — primary reference
- [Karpathy's "Let's build GPT"](https://www.youtube.com/watch?v=kCc8FmEb1nY) — video walkthrough
- [RoPE paper](https://arxiv.org/abs/2104.09864) — Su et al., "RoFormer"
- [SwiGLU paper](https://arxiv.org/abs/2002.05202) — Shazeer, "GLU Variants Improve Transformer"

## Hardware

- Apple Silicon Mac Mini M4, 16GB unified memory
- PyTorch 2.11 with MPS backend
- Training time: ~85 min for 5000 steps (vanilla, char-level)

## License

MIT

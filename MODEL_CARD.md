---
license: mit
language:
- en
tags:
- pytorch
- transformer
- language-model
- from-scratch
- educational
- shakespeare
- rope
- swiglu
- rmsnorm
- kv-cache
datasets:
- tiny-shakespeare
pipeline_tag: text-generation
---

# tiny-gpt-shakespeare

<p align="center">
  <img src="images/brain_book.png" alt="A glowing neural network brain floating above an open Shakespeare book" width="600">
</p>

A ~10M parameter language model built **entirely from scratch** in PyTorch — no HuggingFace Transformers, no pretrained weights, no shortcuts. Trained on Shakespeare, modernized with the same architecture used in LLaMA, Qwen, and Mistral.

> Built as a learning project to understand modern LLM architectures from first principles. Every line of code is written from scratch and explained in the [DEVLOG](DEVLOG.md).

## Architecture

```
                    tinyllm — Modern Decoder-Only Transformer
    ┌──────────────────────────────────────────────────────────────┐
    │                                                              │
    │   Input: "To be or not to"                                   │
    │       │                                                      │
    │       ▼                                                      │
    │   ┌──────────────────┐                                       │
    │   │ Token Embedding   │  50,257 tokens → 384 dims            │
    │   │ (no pos_emb!)     │  Position via RoPE, not learned      │
    │   └────────┬─────────┘                                       │
    │            │                                                  │
    │            ▼                                                  │
    │   ┌──────────────────────────────────────────────┐  × 6      │
    │   │  RMSNorm ──→ Multi-Head Attention (6 heads)  │  layers   │
    │   │                  Q,K rotated by RoPE         │           │
    │   │                  KV cached for inference     │           │
    │   │              + residual connection            │           │
    │   │                                              │           │
    │   │  RMSNorm ──→ SwiGLU FFN (384→1024→384)      │           │
    │   │                  gate·up·down with SiLU      │           │
    │   │              + residual connection            │           │
    │   └──────────────────────────────────────────────┘           │
    │            │                                                  │
    │            ▼                                                  │
    │   ┌──────────────────┐                                       │
    │   │ RMSNorm → Linear  │  384 → 50,257 (tied with embedding) │
    │   └────────┬─────────┘                                       │
    │            │                                                  │
    │            ▼                                                  │
    │   Output: "be" (predicted next token)                        │
    │                                                              │
    └──────────────────────────────────────────────────────────────┘
```

## What makes this "modern"?

We built a vanilla GPT-2-style transformer first, then swapped in four improvements — one at a time, measuring the effect of each:

| Swap | Old (GPT-2 era) | New (LLaMA/Qwen era) | Effect on val loss at 500 steps |
|------|-----------------|---------------------|-------------------------------|
| Normalization | LayerNorm | **RMSNorm** | Same (free efficiency win) |
| FFN activation | ReLU | **SwiGLU** | **-0.11** (faster learning) |
| Position encoding | Learned embeddings | **RoPE** | **-0.31** (huge improvement) |
| Inference | Recompute all | **KV Cache** | N/A (1.3x faster generation) |

**RoPE was the star** — the biggest single improvement, and it achieves this with *fewer* parameters (no positional embedding table needed).

## Training results

### Vanilla vs Modern loss curves (character-level, 5000 steps)

**Vanilla GPT (LayerNorm + ReLU + learned pos):**
| Step | Train | Val |
|------|-------|-----|
| 0 | 4.19 | 4.19 |
| 2500 | 1.15 | 1.49 |
| 5000 | 0.88 | 1.54 |

Best val: **1.48** at step 3000 (overfit after)

## Sample outputs

**Prompt: "ROMEO:" at temperature 0.8:**
```
ROMEO:
Thither the forest world they are.

MERCUTIO:
No better for the court.

MERCUTIO:
Let us always be for contented: have you not slander'd
therein like less behind than than offends it, and he
discharged in Verona his report.
```

**Prompt: "ROMEO:" at temperature 0.5:**
```
ROMEO:
I would be so straitly for thee for thy heart.

BENVOLIO:
By this and look on thee, who were thy son
As if thou couldst desire to thy love.
```

A 10M parameter model trained for 88 minutes on 1MB of text — producing recognizable Shakespeare with proper character names, dialogue formatting, and verse rhythm.

## How to use

```python
import torch
from model_modern import ModernGPT

# Load
device = "mps" if torch.backends.mps.is_available() else "cpu"
ckpt = torch.load("model.pt", map_location=device, weights_only=False)
model = ModernGPT(**ckpt["config"]).to(device)
model.load_state_dict(ckpt["model_state"])
model.eval()

# Generate
from tokenizer import encode, decode
idx = torch.tensor([encode("ROMEO:")], dtype=torch.long, device=device)
out = model.generate(idx, max_new_tokens=200, temperature=0.8)
print(decode(out[0].tolist()))
```

## Training details

| Parameter | Value |
|-----------|-------|
| Parameters | ~10.6M |
| Architecture | Decoder-only transformer |
| Layers | 6 |
| Heads | 6 |
| Embedding dim | 384 |
| Context length | 256 tokens |
| Vocab | 65 (char-level) / 50,257 (BPE) |
| Optimizer | AdamW (lr=3e-4) |
| Training steps | 5,000 |
| Hardware | Apple M4 Mac Mini (16GB, MPS) |
| Training time | ~88 minutes |
| Dataset | Tiny Shakespeare (~1.1MB) |

## The journey

This model was built as a weekend learning project following [Karpathy's build-nanogpt](https://github.com/karpathy/build-nanogpt) as a reference. Every component was implemented from scratch:

1. **Character-level tokenizer** — 65 unique chars, encode/decode, train/val split
2. **Scaled dot-product attention** — Q/K/V projections, causal mask, dropout
3. **Multi-head attention** — 6 parallel heads, concatenate + project
4. **Transformer block** — pre-norm, residual connections, feed-forward network
5. **Full GPT model** — embeddings, 6 blocks, language model head, weight tying
6. **Modernization** — RMSNorm, SwiGLU, RoPE, KV cache (each tested in isolation)
7. **Scaling** — BPE tokenization, mixed precision, gradient accumulation

Full learning journal with detailed explanations of every concept: [DEVLOG.md](DEVLOG.md)

## Key learnings

- **RoPE > learned positional embeddings** by a significant margin, even on tiny data
- **SwiGLU learns faster** than ReLU but converges to the same floor on small datasets
- **RMSNorm is a free upgrade** — identical quality, simpler code
- **MPS (Apple Silicon) works** but kills processes silently on OOM — don't run GPU tasks in parallel on 16GB
- **Best val loss was at step 3000**, not step 5000 — the model started memorizing Shakespeare after that

## References

- [build-nanogpt](https://github.com/karpathy/build-nanogpt) — Karpathy's step-by-step GPT build
- [nanochat](https://github.com/karpathy/nanochat) — nanoGPT successor
- [RoPE paper](https://arxiv.org/abs/2104.09864) — Su et al., "RoFormer"
- [SwiGLU paper](https://arxiv.org/abs/2002.05202) — Shazeer, "GLU Variants Improve Transformer"
- [RMSNorm paper](https://arxiv.org/abs/1910.07467) — Zhang & Sennrich

## License

MIT

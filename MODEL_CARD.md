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

## Training results (Google Colab T4)

| Model | Params | Best Val Loss | Best Step | Time |
|-------|--------|-------------|-----------|------|
| Vanilla (char-level) | 10.8M | 1.4837 | 3000 | 59.7 min |
| **Modern (char-level)** | **10.6M** | **1.4783** | **2500** | **66.9 min** |
| Modern (BPE) | 29.9M | 4.6414 | 1000 | 68.2 min |

Modern beats vanilla with fewer params and reaches best loss 500 steps sooner.

### Throughput

| Model | tok/s | 300 tokens |
|-------|-------|-----------|
| Vanilla (no cache) | 72.2 | 4.16s |
| Modern (KV cache) | 40.7 | 7.37s |

## Sample outputs

**Vanilla model, temp=0.8:**
```
ROMEO:
Nay, be too be so head: but I am as betimes;
There is no man with her pleasure attentience,
She doth behold our queen arms.

PAULINA:
I'll not too woe to die for the law to the world,
```

**Modern model, temp=0.8 (KV cached):**
```
ROMEO:
A gallant-house! what says the woe?

MERCUTIO:
Good madam, my lord.

ROMEO:
Villain, for I do not say it is true,
Which hath a sin by him come to the crown,
That he is reports for me; for ever is he.
```

**Vanilla model, temp=0.5 (focused):**
```
KING HENRY:
The father of the marriage of my son,
And then we will be no longer to be then,
And but the Lord Hastings of Semiram Stanley.
```

A 10M parameter model trained for ~60 minutes on 1MB of text — producing recognizable Shakespeare with proper character names, dialogue formatting, and verse rhythm.

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

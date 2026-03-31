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

<p align="center">
  <strong>I built a tiny LLM from scratch to understand how GPT-4 and LLaMA actually work.</strong>
</p>

<p align="center">
  <em>10M parameters. Trained on Shakespeare. Every line of code written from scratch. Every mistake documented.</em>
</p>

<p align="center">
  <a href="https://github.com/brianmeyer/tinyllm">GitHub</a> |
  <a href="https://github.com/brianmeyer/tinyllm/blob/main/DEVLOG.md">Learning Journal</a>
</p>

---

## What is this?

A ~10M parameter decoder-only transformer — no HuggingFace Transformers library, no pretrained weights, no shortcuts. Built from an empty file to a working Shakespeare generator, then modernized with the same architecture used in LLaMA, Qwen, and Mistral.

This is a learning project. The model itself is tiny and toy-scale. The value is in the code, the [DEVLOG](https://github.com/brianmeyer/tinyllm/blob/main/DEVLOG.md), and the 9 things that went wrong along the way.

## It generates Shakespeare

**Modern model, temp=0.8 (RMSNorm + SwiGLU + RoPE + KV cache):**
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

**Vanilla model, temp=0.5:**
```
KING HENRY:
The father of the marriage of my son,
And then we will be no longer to be then,
And but the Lord Hastings of Semiram Stanley.
```

Not perfect. But recognizable Shakespeare — proper character names, dialogue formatting, verse rhythm — from a 10M param model trained for ~60 minutes on 1MB of text.

## Architecture

```
ModernGPT (10.6M params)
  token_emb:   Embedding(65, 384)
  blocks × 6:
    RMSNorm → MultiHeadAttention(6 heads, RoPE, KV cache) → residual
    RMSNorm → SwiGLU(384 → 1024 → 384) → residual
  RMSNorm → lm_head (tied with token_emb)
```

Four upgrades over vanilla GPT-2, each tested in isolation:

| Upgrade | What changed | Impact |
|---------|-------------|--------|
| **RMSNorm** | Drop mean subtraction from LayerNorm | Free efficiency win |
| **SwiGLU** | Smooth gating replaces hard ReLU cutoff | **-0.11** val loss at step 500 |
| **RoPE** | Rotate Q/K vectors instead of adding position embeddings | **-0.31** val loss at step 500 |
| **KV Cache** | Cache keys/values during generation | Faster inference |

## Results

| Model | Params | Best Val Loss | Time |
|-------|--------|-------------|------|
| Vanilla | 10.8M | 1.4837 | 57 min |
| **Modern** | **10.6M** | **1.4754** | **67 min** |

Modern beats vanilla with fewer params. RoPE was the star — biggest single improvement.

## 9 things that went wrong

Building this was not smooth. Every failure is documented in the [DEVLOG](https://github.com/brianmeyer/tinyllm/blob/main/DEVLOG.md):

1. MPS training died silently (memory leak)
2. Bundled all 4 architecture swaps together instead of testing one at a time
3. Python stdout buffering hid training progress
4. RoPE position bug in KV cache made the model generate garbage
5. Modern model memorized Shakespeare (overfitting on 1MB)
6. Float16 diverged on MPS with 50K BPE vocab
7. MPS kept killing every retrain attempt
8. Lost all Colab checkpoints when runtime disconnected
9. Ran out of free Colab GPU quota

## Training details

| | |
|---|---|
| Dataset | Tiny Shakespeare (~1.1MB, 65 unique characters) |
| Optimizer | AdamW, lr=3e-4 |
| Batch size | 64, block size 256 |
| Steps | 5,000 (best checkpoint via early stopping) |
| Hardware | Google Colab T4 (and an M4 Mac that kept crashing) |
| Dropout | 0.3 (increased from 0.2 to fight overfitting) |

## How to use

```python
import torch
import sys
sys.path.append('src')
from model_modern import ModernGPT
from tokenizer import encode, decode

device = "cuda" if torch.cuda.is_available() else "cpu"
ckpt = torch.load("model.pt", map_location=device, weights_only=False)
model = ModernGPT(**ckpt["config"]).to(device)
model.load_state_dict(ckpt["model_state"])
model.eval()

idx = torch.tensor([encode("ROMEO:")], dtype=torch.long, device=device)
out = model.generate(idx, max_new_tokens=200, temperature=0.8)
print(decode(out[0].tolist()))
```

## What I learned

1. **RoPE is the most impactful modern architecture change** — beautiful math, fewer params, better results
2. **More powerful models overfit faster on small data** — early stopping is essential
3. **When loss is good but output is garbage, the bug is in inference code** — not the model
4. **MPS is not ready for serious training** — use CUDA
5. **Always save checkpoints to persistent storage** — Colab runtimes are ephemeral
6. **Change one thing at a time and measure** — this is how real ML research works

## References

- [build-nanogpt](https://github.com/karpathy/build-nanogpt) — Karpathy's step-by-step GPT build
- [RoPE paper](https://arxiv.org/abs/2104.09864) — Su et al.
- [SwiGLU paper](https://arxiv.org/abs/2002.05202) — Shazeer
- [RMSNorm paper](https://arxiv.org/abs/1910.07467) — Zhang & Sennrich

## License

MIT

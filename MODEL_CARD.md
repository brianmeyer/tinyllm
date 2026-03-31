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

A 10M parameter decoder-only transformer trained on the Tiny Shakespeare dataset. Built from scratch in PyTorch as an educational project — no pretrained weights or external libraries used for the model itself.

## Model Description

- **Architecture:** Decoder-only transformer with modern components (RMSNorm, SwiGLU, RoPE, KV cache)
- **Parameters:** 10.6M (modern) / 10.8M (vanilla)
- **Training data:** [Tiny Shakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt) (~1.1MB, 65 unique characters)
- **Tokenization:** Character-level (65 tokens)
- **Context length:** 256 tokens
- **License:** MIT

## Architecture Details

| Component | Implementation |
|-----------|---------------|
| Layers | 6 transformer blocks |
| Attention | 6 heads, 64 dims each, with RoPE |
| FFN | SwiGLU (384 → 1024 → 384) |
| Normalization | RMSNorm (pre-norm) |
| Position encoding | Rotary Position Embeddings (RoPE) |
| Inference | KV cache for autoregressive generation |
| Weight tying | lm_head shares weights with token embedding |

## Training

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning rate | 3e-4 |
| Batch size | 64 |
| Block size | 256 |
| Dropout | 0.3 |
| Training steps | 5,000 (best checkpoint at step 2,500) |
| Hardware | Google Colab T4 GPU |
| Training time | ~64 minutes |

### Training Results

| Model | Parameters | Best Val Loss | Best Step |
|-------|-----------|-------------|-----------|
| Vanilla (LayerNorm + ReLU + learned pos) | 10.8M | 1.4804 | 3,000 |
| Modern (RMSNorm + SwiGLU + RoPE + KV cache) | 10.6M | 1.4754 | 2,500 |

Early stopping was used — the model checkpointed at the lowest validation loss.

### Component Comparison

Each modern component was tested in isolation against the vanilla baseline (2,000 training steps each):

| Component | Val Loss at Step 500 | vs Vanilla |
|-----------|---------------------|-----------|
| Vanilla (baseline) | 1.99 | — |
| RMSNorm | 1.99 | No change |
| SwiGLU | 1.88 | -0.11 |
| RoPE | 1.68 | -0.31 |

## Intended Use

This is an **educational model**. It is not intended for production use. It generates Shakespeare-style text and serves as a reference implementation for understanding transformer architectures.

## Sample Outputs

**Prompt: "ROMEO:", temperature=0.8:**
```
ROMEO:
Marry, good day with me!

ROMEO:
And not your lady command her at arms.

THOMAS MOWBRAY:
My dear lord, but go on.

MERCUTIO:
Hence will not speak against a marriage.
```

**Prompt: "KING HENRY:", temperature=0.5:**
```
KING HENRY:
The father of the marriage of my son,
And then we will be no longer to be then,
And but the Lord Hastings of Semiram Stanley.
```

## Limitations

- **Tiny dataset:** Trained on only 1.1MB of text. The model overfits after ~2,500 steps.
- **Character-level tokenization:** Inefficient compared to BPE. Each character is a separate token.
- **No instruction tuning:** This is a base model — it completes text, it does not follow instructions or answer questions.
- **Small context window:** 256 tokens maximum.
- **Quality:** Output is recognizably Shakespeare-like but contains grammatical errors and occasionally mixes characters from different plays.

## How to Use

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

## Source Code

Full implementation with detailed documentation: [github.com/brianmeyer/tinyllm](https://github.com/brianmeyer/tinyllm)

## References

- Karpathy, A. [build-nanogpt](https://github.com/karpathy/build-nanogpt) — primary reference
- Su et al. (2021). [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
- Shazeer, N. (2020). [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)
- Zhang & Sennrich (2019). [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467)

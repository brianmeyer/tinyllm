# tinyllm

<p align="center">
  <img src="images/robot_shakespeare.png" alt="A tiny robot reading Shakespeare by candlelight, with transformer layers glowing inside its transparent head" width="700">
</p>

<p align="center">
  <strong>I built a tiny LLM from scratch to understand how GPT-4, Claude, and LLaMA actually work.</strong>
</p>

<p align="center">
  <em>10M parameters. Trained on Shakespeare. Modernized with the same architecture as LLaMA and Qwen. Every line of code written from scratch.</em>
</p>

<p align="center">
  <a href="DEVLOG.md">Learning Journal</a> |
  <a href="https://huggingface.co/bmeyer2025/tiny-gpt-shakespeare">HuggingFace</a> |
  <a href="MODEL_CARD.md">Model Card</a>
</p>

---

## The idea

GPT-4, Claude, and LLaMA are all scaled-up versions of the same architecture. I wanted to understand it from the ground up — not by reading papers, but by building it myself.

So I built a 10M parameter transformer, trained it on Shakespeare, then upgraded it piece by piece with the same components used in production LLMs. Every mistake, crash, and debugging session is documented in the [DEVLOG](DEVLOG.md).

## What makes it "modern"?

I started with a vanilla GPT-2-style transformer, then swapped in four upgrades — one at a time, measuring each:

| Component | GPT-2 era | Modern (LLaMA/Qwen) | Impact |
|-----------|-----------|---------------------|--------|
| Normalization | LayerNorm | **RMSNorm** | Free efficiency win |
| FFN | ReLU | **SwiGLU** | **-0.11** val loss |
| Position | Learned embeddings | **RoPE** | **-0.31** val loss |
| Inference | Recompute all | **KV Cache** | Faster generation |

**RoPE was the star** — biggest improvement, fewer parameters, and the position encoding math is genuinely beautiful.

## Results

| Model | Best Val Loss | Training Time |
|-------|-------------|--------------|
| Vanilla (10.8M params) | 1.4804 | 57 min |
| **Modern (10.6M params)** | **1.4754** | **64 min** |

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

*A 10M parameter model generating Shakespeare dialogue after 67 minutes of training.*

## Project structure

```
tinyllm/
├── src/                        # Core model code (built from scratch)
│   ├── tokenizer.py            #   Character-level tokenizer + data loading
│   ├── attention.py            #   Single-head causal self-attention
│   ├── transformer.py          #   Multi-head attention, FFN, transformer Block
│   ├── model.py                #   Full vanilla GPT (10.8M params)
│   ├── modernize.py            #   Modern components: RMSNorm, SwiGLU, RoPE, KV cache
│   ├── model_modern.py         #   Modernized GPT (10.6M params)
│   └── generate.py             #   Text generation with sampling
│
├── experiments/                # Per-swap A/B comparisons
│   ├── swap1_rmsnorm.py        #   LayerNorm → RMSNorm (2000 steps)
│   ├── swap2_swiglu.py         #   ReLU → SwiGLU (2000 steps)
│   ├── swap3_rope.py           #   Learned pos → RoPE (2000 steps)
│   └── swap4_kvcache.py        #   KV cache speed benchmark
│
├── training/                   # Training scripts
│   ├── train.py                #   Vanilla GPT (5000 steps)
│   ├── train_modern.py         #   Modern GPT with early stopping
│   ├── train_bpe.py            #   BPE + gradient accumulation
│   └── benchmark.py            #   Samples, latency, throughput comparison
│
├── colab/                      # Google Colab
│   └── train_colab.py          #   All-in-one: vanilla + modern + BPE + benchmarks
│
├── data/input.txt              # Tiny Shakespeare (~1.1MB)
├── images/                     # Generated graphics
├── DEVLOG.md                   # Full learning journal (the real value)
├── MODEL_CARD.md               # HuggingFace model card
└── publish.py                  # Upload to HuggingFace
```

## Quick start

**On Google Colab (recommended):**
```python
!git clone https://github.com/brianmeyer/tinyllm.git
%cd tinyllm
!pip install tiktoken
!python -u colab/train_colab.py
```

**Locally (M4 Mac / any GPU):**
```bash
git clone https://github.com/brianmeyer/tinyllm.git
cd tinyllm
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -u training/train.py              # vanilla, ~60 min
python -u training/train_modern.py       # modern, ~67 min
python src/generate.py --demo            # see the output
```

## The learning journey

**Phase 1 — Build from scratch:** Tokenizer, attention mechanism, multi-head attention, feed-forward network, transformer block, full GPT model. Every component explained in the [DEVLOG](DEVLOG.md).

**Phase 2 — Modernize one swap at a time:** Replace LayerNorm, ReLU, learned positions, and naive inference with RMSNorm, SwiGLU, RoPE, and KV cache. Each swap tested in isolation so you can see exactly what it does.

**Phase 3 — Scale up:** BPE tokenization (50K vocab), mixed precision, gradient accumulation. Learned why BPE needs way more data than 1MB of Shakespeare.

**Phase 4 — Break everything:** MPS memory leaks, silent process kills, float16 divergence, RoPE position bugs, Colab runtime evictions, lost checkpoints. Each failure documented with root cause and fix.

## What I learned

1. **RoPE is the most impactful modern change** — 0.31 better loss, fewer params, beautiful math
2. **More powerful models overfit faster on small data** — early stopping is essential
3. **MPS (Apple Silicon) silently kills training** after 60-80 min due to memory leaks
4. **When loss is good but output is garbage, the bug is in inference** — our RoPE position bug only appeared during KV cache generation
5. **Change one thing at a time** — the per-swap comparison approach is how real ML research works
6. **Always save checkpoints to persistent storage** — we lost 3 hours of Colab training to a runtime disconnect

## Architecture

```
ModernGPT (10.6M params)
  token_emb:   Embedding(65, 384)
  blocks × 6:
    RMSNorm → MultiHeadAttention(6 heads, RoPE, KV cache) → residual
    RMSNorm → SwiGLU(384 → 1024 → 384) → residual
  RMSNorm → lm_head (tied with token_emb)
```

## 9 things that went wrong

| # | What happened | Root cause |
|---|--------------|-----------|
| 1 | MPS training died silently | Memory leak in PyTorch MPS backend |
| 2 | Bundled all 4 swaps together | Rushing — should test one at a time |
| 3 | Python output hidden during training | stdout buffering — use `python -u` |
| 4 | Modern model generated garbage | RoPE position bug in KV cache inference |
| 5 | Modern model memorized Shakespeare | 10M params too powerful for 1MB data |
| 6 | BPE training diverged | float16 on MPS overflows with 50K vocab |
| 7 | MPS kept killing all retrains | Memory leak unfixable on 16GB |
| 8 | Lost all Colab checkpoints | Runtime disconnected — ephemeral storage |
| 9 | Colab GPU quota exhausted | Used all free T4 hours in one session |

Full analysis of each: [DEVLOG.md](DEVLOG.md)

## References

- [build-nanogpt](https://github.com/karpathy/build-nanogpt) — Karpathy's step-by-step GPT build
- [nanochat](https://github.com/karpathy/nanochat) — nanoGPT successor
- [RoPE paper](https://arxiv.org/abs/2104.09864) — Su et al.
- [SwiGLU paper](https://arxiv.org/abs/2002.05202) — Shazeer

## License

MIT

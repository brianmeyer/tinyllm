"""
All-in-one training script for Google Colab.

Runs ALL training in one shot on a T4 GPU:
  1. Vanilla char-level GPT (5000 steps)
  2. Modern char-level GPT with early stopping (5000 steps)
  3. Modern BPE GPT with gradient accumulation (3000 steps)
  4. Benchmarks: samples, latency, throughput for all models
  5. Saves all checkpoints

Usage in Colab:
  !git clone https://github.com/brianmeyer/tinyllm.git
  %cd tinyllm
  !pip install tiktoken
  !python train_colab.py

WHY COLAB:
  MPS (Apple Silicon) has a memory leak that silently kills training after
  60-80 minutes. We tried empty_cache(), watermark ratios — nothing worked
  reliably on 16GB. CUDA on a T4 is rock-solid. Lesson: use the right tool.
"""

import os
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Device setup ──────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

print(f"Device: {DEVICE}")
if DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

# ── Download data if needed ───────────────────────────────────────────────────
os.makedirs("data", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

if not os.path.exists("data/input.txt"):
    import urllib.request
    print("Downloading tiny Shakespeare...")
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        "data/input.txt"
    )

# ── Patch tokenizer device ───────────────────────────────────────────────────
# The tokenizer module hardcodes device detection. Override it.
import tokenizer as tok
tok.DEVICE = DEVICE

from tokenizer import encode, decode, VOCAB_SIZE, BLOCK_SIZE, get_batch
from model import GPT
from model_modern import ModernGPT


# ══════════════════════════════════════════════════════════════════════════════
# PART 1: Vanilla GPT (5000 steps)
# ══════════════════════════════════════════════════════════════════════════════
def train_vanilla():
    print("\n" + "=" * 70)
    print("PART 1: VANILLA GPT (LayerNorm + ReLU + learned pos embeddings)")
    print("=" * 70)

    model = GPT(
        vocab_size=VOCAB_SIZE, n_embd=384, n_heads=6, n_layer=6,
        block_size=BLOCK_SIZE, dropout=0.2,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,} ({n_params/1e6:.1f}M)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    @torch.no_grad()
    def estimate_loss():
        model.eval()
        out = {}
        for split in ["train", "val"]:
            losses = torch.zeros(200)
            for k in range(200):
                x, y = get_batch(split)
                _, loss = model(x, y)
                losses[k] = loss.item()
            out[split] = losses.mean().item()
        model.train()
        return out

    t0 = time.time()
    best_val = float("inf")

    for step in range(5001):
        if step % 500 == 0:
            losses = estimate_loss()
            elapsed = time.time() - t0
            marker = ""
            if losses["val"] < best_val:
                best_val = losses["val"]
                torch.save({
                    "model_state": model.state_dict(),
                    "config": {"vocab_size": VOCAB_SIZE, "n_embd": 384, "n_heads": 6,
                               "n_layer": 6, "block_size": BLOCK_SIZE, "dropout": 0.2},
                }, "checkpoints/vanilla_gpt.pt")
                marker = " ← best, saved!"
            print(f"step {step:5d} | train: {losses['train']:.4f} | val: {losses['val']:.4f} | {elapsed:.1f}s{marker}")

        if step == 5000:
            break

        x, y = get_batch("train")
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    total = time.time() - t0
    print(f"Vanilla training complete in {total/60:.1f} min (best val: {best_val:.4f})")
    return model


# ══════════════════════════════════════════════════════════════════════════════
# PART 2: Modern GPT — char-level (5000 steps, early stopping)
# ══════════════════════════════════════════════════════════════════════════════
def train_modern():
    print("\n" + "=" * 70)
    print("PART 2: MODERN GPT (RMSNorm + SwiGLU + RoPE + KV Cache)")
    print("=" * 70)

    model = ModernGPT(
        vocab_size=VOCAB_SIZE, n_embd=384, n_heads=6, n_layer=6,
        block_size=BLOCK_SIZE, dropout=0.3,  # higher dropout to prevent overfitting
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,} ({n_params/1e6:.1f}M)")
    print(f"Dropout: 0.3 (increased to fight overfitting on tiny data)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    @torch.no_grad()
    def estimate_loss():
        model.eval()
        out = {}
        for split in ["train", "val"]:
            losses = torch.zeros(200)
            for k in range(200):
                x, y = get_batch(split)
                _, loss = model(x, y)
                losses[k] = loss.item()
            out[split] = losses.mean().item()
        model.train()
        return out

    t0 = time.time()
    best_val = float("inf")

    for step in range(5001):
        if step % 500 == 0:
            losses = estimate_loss()
            elapsed = time.time() - t0
            marker = ""
            if losses["val"] < best_val:
                best_val = losses["val"]
                torch.save({
                    "model_state": model.state_dict(),
                    "config": {"vocab_size": VOCAB_SIZE, "n_embd": 384, "n_heads": 6,
                               "n_layer": 6, "block_size": BLOCK_SIZE, "dropout": 0.0},
                    "model_type": "modern",
                    "step": step,
                }, "checkpoints/modern_gpt.pt")
                marker = " ← best, saved!"
            print(f"step {step:5d} | train: {losses['train']:.4f} | val: {losses['val']:.4f} | {elapsed:.1f}s{marker}")

        if step == 5000:
            break

        x, y = get_batch("train")
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    total = time.time() - t0
    print(f"Modern training complete in {total/60:.1f} min (best val: {best_val:.4f})")
    return model


# ══════════════════════════════════════════════════════════════════════════════
# PART 3: Modern GPT — BPE (3000 steps, gradient accumulation)
# ══════════════════════════════════════════════════════════════════════════════
def train_bpe():
    import tiktoken

    print("\n" + "=" * 70)
    print("PART 3: MODERN GPT + BPE + GRADIENT ACCUMULATION")
    print("=" * 70)

    enc = tiktoken.get_encoding("gpt2")
    bpe_vocab = enc.n_vocab  # 50,257

    with open("data/input.txt", "r") as f:
        text = f.read()

    data = torch.tensor(enc.encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data, val_data = data[:n], data[n:]

    print(f"BPE tokens: {len(data):,} (was {len(text):,} chars, {len(text)/len(data):.1f}x compression)")

    MICRO_BATCH = 16
    ACCUM_STEPS = 4
    BPE_BLOCK = 256

    def get_bpe_batch(split):
        src = train_data if split == "train" else val_data
        ix = torch.randint(len(src) - BPE_BLOCK, (MICRO_BATCH,))
        x = torch.stack([src[i:i+BPE_BLOCK] for i in ix])
        y = torch.stack([src[i+1:i+BPE_BLOCK+1] for i in ix])
        return x.to(DEVICE), y.to(DEVICE)

    model = ModernGPT(
        vocab_size=bpe_vocab, n_embd=384, n_heads=6, n_layer=6,
        block_size=BPE_BLOCK, dropout=0.3,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,} ({n_params/1e6:.1f}M)")
    print(f"Effective batch: {MICRO_BATCH} × {ACCUM_STEPS} = {MICRO_BATCH * ACCUM_STEPS}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    @torch.no_grad()
    def estimate_loss():
        model.eval()
        out = {}
        for split in ["train", "val"]:
            losses = torch.zeros(100)
            for k in range(100):
                x, y = get_bpe_batch(split)
                _, loss = model(x, y)
                losses[k] = loss.item()
            out[split] = losses.mean().item()
        model.train()
        return out

    config = {"vocab_size": bpe_vocab, "n_embd": 384, "n_heads": 6,
              "n_layer": 6, "block_size": BPE_BLOCK, "dropout": 0.0}

    t0 = time.time()
    best_val = float("inf")

    for step in range(3001):
        if step % 500 == 0:
            losses = estimate_loss()
            elapsed = time.time() - t0
            marker = ""
            if losses["val"] < best_val:
                best_val = losses["val"]
                torch.save({
                    "model_state": model.state_dict(),
                    "config": config,
                    "model_type": "modern",
                    "tokenizer": "gpt2",
                    "step": step,
                }, "checkpoints/modern_bpe_gpt.pt")
                with open("checkpoints/config.json", "w") as f:
                    json.dump(config, f, indent=2)
                marker = " ← best, saved!"
            print(f"step {step:5d} | train: {losses['train']:.4f} | val: {losses['val']:.4f} | {elapsed:.1f}s{marker}")

        if step == 3000:
            break

        optimizer.zero_grad(set_to_none=True)
        for _ in range(ACCUM_STEPS):
            x, y = get_bpe_batch("train")
            _, loss = model(x, y)
            (loss / ACCUM_STEPS).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    total = time.time() - t0
    print(f"BPE training complete in {total/60:.1f} min (best val: {best_val:.4f})")
    return model, enc


# ══════════════════════════════════════════════════════════════════════════════
# PART 4: Benchmarks
# ══════════════════════════════════════════════════════════════════════════════
def run_benchmarks():
    print("\n" + "=" * 70)
    print("PART 4: BENCHMARKS")
    print("=" * 70)

    # ── Load vanilla ──────────────────────────────────────────────────────
    print("\n--- VANILLA MODEL ---")
    ckpt = torch.load("checkpoints/vanilla_gpt.pt", map_location=DEVICE, weights_only=False)
    vanilla = GPT(**ckpt["config"]).to(DEVICE)
    vanilla.load_state_dict(ckpt["model_state"])
    vanilla.eval()

    prompts = ["ROMEO:", "To be", "KING HENRY:"]
    temps = [0.5, 0.8, 1.0]

    for prompt in prompts:
        for temp in temps:
            idx = torch.tensor([encode(prompt)], dtype=torch.long, device=DEVICE)
            if DEVICE == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            with torch.no_grad():
                for _ in range(200):
                    idx_c = idx[:, -vanilla.block_size:]
                    logits, _ = vanilla(idx_c)
                    logits = logits[:, -1, :] / temp
                    probs = F.softmax(logits, dim=-1)
                    idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
            if DEVICE == "cuda":
                torch.cuda.synchronize()
            elapsed = time.time() - t0
            text = decode(idx[0].tolist())
            print(f"\n[prompt=\"{prompt}\", temp={temp}, {200/elapsed:.0f} tok/s]")
            print(text[:250])

    # ── Load modern ───────────────────────────────────────────────────────
    if os.path.exists("checkpoints/modern_gpt.pt"):
        print("\n--- MODERN MODEL (char-level, KV cache) ---")
        ckpt = torch.load("checkpoints/modern_gpt.pt", map_location=DEVICE, weights_only=False)
        modern = ModernGPT(**ckpt["config"]).to(DEVICE)
        modern.load_state_dict(ckpt["model_state"])
        modern.eval()

        for prompt in ["ROMEO:", "To be"]:
            for temp in [0.5, 0.8, 1.0]:
                idx = torch.tensor([encode(prompt)], dtype=torch.long, device=DEVICE)
                out = modern.generate(idx, max_new_tokens=200, temperature=temp)
                text = decode(out[0].tolist())
                print(f"\n[prompt=\"{prompt}\", temp={temp}, KV cached]")
                print(text[:250])

        # Throughput comparison
        print("\n--- THROUGHPUT ---")
        for name, mdl, use_cache in [("Vanilla (no cache)", vanilla, False), ("Modern (KV cache)", modern, True)]:
            times = []
            for _ in range(3):
                idx = torch.tensor([encode("ROMEO:")], dtype=torch.long, device=DEVICE)
                if DEVICE == "cuda":
                    torch.cuda.synchronize()
                t0 = time.time()
                if use_cache:
                    out = mdl.generate(idx, max_new_tokens=300, temperature=0.8)
                else:
                    with torch.no_grad():
                        for _ in range(300):
                            idx_c = idx[:, -mdl.block_size:]
                            logits, _ = mdl(idx_c)
                            logits = logits[:, -1, :] / 0.8
                            probs = F.softmax(logits, dim=-1)
                            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
                if DEVICE == "cuda":
                    torch.cuda.synchronize()
                times.append(time.time() - t0)
            avg = sum(times) / len(times)
            print(f"  {name}: {300/avg:.1f} tok/s ({avg:.2f}s for 300 tokens)")

    print("\nBenchmarks complete!")


# ══════════════════════════════════════════════════════════════════════════════
# RUN EVERYTHING
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    train_vanilla()
    train_modern()
    train_bpe()
    run_benchmarks()

    print("\n" + "=" * 70)
    print("ALL DONE!")
    print("=" * 70)
    print("Checkpoints saved to checkpoints/")
    print("  vanilla_gpt.pt      — vanilla char-level model (best val)")
    print("  modern_gpt.pt       — modern char-level model (best val)")
    print("  modern_bpe_gpt.pt   — modern BPE model (best val)")
    print("  config.json         — BPE model config for HuggingFace")
    print()
    print("Next steps:")
    print("  1. Download checkpoints from Colab")
    print("  2. Run: python publish.py")
    print("  3. Push to GitHub")

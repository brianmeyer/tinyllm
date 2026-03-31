"""
Phase 4: Scaled training with BPE + mixed precision + gradient accumulation.

THREE upgrades over train.py, each explained:

1. BPE TOKENIZATION (tiktoken, GPT-2 tokenizer)
   - Character-level: "the" = 3 tokens. Wasteful.
   - BPE: "the" = 1 token. Each token carries more information.
   - Vocab jumps from 65 to 50,257, but sequences are ~4x shorter.
   - Net effect: the model sees more "meaning" per training step.

2. MIXED PRECISION (float16)
   - Default: all math in float32 (32 bits per number)
   - Mixed precision: most math in float16 (half the bits)
   - ~2x faster, ~half the memory, minimal quality loss
   - The "mixed" part: some ops (like loss computation) stay in float32
     for numerical stability.
   - On MPS (Apple Silicon), we use torch.autocast — no GradScaler needed
     (GradScaler is CUDA-only).

3. GRADIENT ACCUMULATION
   - Problem: bigger batches train better, but don't fit in memory
   - Solution: run several small "micro-batches", accumulate the gradients,
     then do one optimizer step
   - effective_batch = micro_batch × accumulation_steps
   - Also adds gradient clipping (max_norm=1.0) to prevent training instability

Uses the ModernGPT architecture (all 4 swaps applied).
"""

import os
import time
import json
import torch
import tiktoken

from model_modern import ModernGPT

# ── Config ────────────────────────────────────────────────────────────────────
DEVICE     = "mps" if torch.backends.mps.is_available() else "cpu"
DATA_PATH  = "data/input.txt"

# BPE tokenizer (GPT-2)
ENC        = tiktoken.get_encoding("gpt2")
VOCAB_SIZE = ENC.n_vocab     # 50,257

# Model config (slightly larger than vanilla to use the bigger vocab well)
N_EMBD     = 384
N_HEADS    = 6
N_LAYER    = 6
BLOCK_SIZE = 256
DROPOUT    = 0.3   # increased from 0.2 — more regularization for bigger model

# Training config
MAX_ITERS          = 3000      # reduced from 5000 (MPS stability + overfitting)
EVAL_EVERY         = 500
EVAL_ITERS         = 100       # fewer eval iters (each BPE batch covers more text)
LR                 = 1e-4      # reduced from 3e-4 — bigger model needs smaller steps
MICRO_BATCH_SIZE   = 16        # fits in memory
ACCUM_STEPS        = 4         # effective batch = 16 × 4 = 64
MAX_GRAD_NORM      = 1.0       # gradient clipping
USE_MIXED_PRECISION = False    # DISABLED — float16 diverges on MPS with 50K vocab

SAVE_PATH = "checkpoints/modern_bpe_gpt.pt"

# ── Load + tokenize data ──────────────────────────────────────────────────────
print("=" * 60)
print("PHASE 4: BPE + Mixed Precision + Gradient Accumulation")
print("=" * 60)

with open(DATA_PATH, "r") as f:
    text = f.read()

# Show the BPE difference
sample = "ROMEO:\nTo be, or not to be"
char_tokens = list(sample)
bpe_tokens  = ENC.encode(sample)
print(f"\nBPE demo: \"{sample}\"")
print(f"  Character tokens: {len(char_tokens)} tokens → {char_tokens}")
print(f"  BPE tokens:       {len(bpe_tokens)} tokens → {bpe_tokens}")
print(f"  BPE decoded:      {[ENC.decode([t]) for t in bpe_tokens]}")
print(f"  Compression:      {len(char_tokens)/len(bpe_tokens):.1f}x fewer tokens\n")

# Tokenize full dataset
data = torch.tensor(ENC.encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data   = data[n:]

print(f"Character tokens:  {len(text):,}")
print(f"BPE tokens:        {len(data):,}  ({len(text)/len(data):.1f}x compression)")
print(f"Train BPE tokens:  {len(train_data):,}")
print(f"Val BPE tokens:    {len(val_data):,}")
print(f"Vocab size:        {VOCAB_SIZE:,} (was 65 with char-level)")


def get_batch(split):
    src = train_data if split == "train" else val_data
    ix  = torch.randint(len(src) - BLOCK_SIZE, (MICRO_BATCH_SIZE,))
    x   = torch.stack([src[i     : i + BLOCK_SIZE    ] for i in ix])
    y   = torch.stack([src[i + 1 : i + BLOCK_SIZE + 1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)


# ── Model ─────────────────────────────────────────────────────────────────────
model = ModernGPT(
    vocab_size = VOCAB_SIZE,
    n_embd     = N_EMBD,
    n_heads    = N_HEADS,
    n_layer    = N_LAYER,
    block_size = BLOCK_SIZE,
    dropout    = DROPOUT,
).to(DEVICE)

n_params = sum(p.numel() for p in model.parameters())
print(f"\nModernGPT (BPE): {n_params:,} params ({n_params/1e6:.1f}M)")
print(f"  (Much bigger than char-level due to 50k embedding table)")
print(f"Device: {DEVICE}")
print(f"Mixed precision: {USE_MIXED_PRECISION}")
print(f"Effective batch: {MICRO_BATCH_SIZE} × {ACCUM_STEPS} = {MICRO_BATCH_SIZE * ACCUM_STEPS}")
print(f"Gradient clipping: max_norm={MAX_GRAD_NORM}")
print(f"Training for {MAX_ITERS} steps...\n")

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

config = {
    "vocab_size":  VOCAB_SIZE,
    "n_embd":      N_EMBD,
    "n_heads":     N_HEADS,
    "n_layer":     N_LAYER,
    "block_size":  BLOCK_SIZE,
    "dropout":     0.0,
}


# ── Loss estimation ───────────────────────────────────────────────────────────
@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {}
    for split in ["train", "val"]:
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            x, y = get_batch(split)
            with torch.autocast(device_type=DEVICE, dtype=torch.float16, enabled=USE_MIXED_PRECISION):
                _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


# ── Training loop with gradient accumulation ──────────────────────────────────
os.makedirs("checkpoints", exist_ok=True)
best_val_loss = float("inf")
t0 = time.time()

for step in range(MAX_ITERS + 1):

    if step % EVAL_EVERY == 0:
        losses = estimate_loss()
        elapsed = time.time() - t0
        marker = ""
        if losses["val"] < best_val_loss:
            best_val_loss = losses["val"]
            torch.save({
                "model_state": model.state_dict(),
                "config": config,
                "model_type": "modern",
                "tokenizer": "gpt2",
                "step": step,
            }, SAVE_PATH)
            with open("checkpoints/config.json", "w") as f:
                json.dump(config, f, indent=2)
            marker = " ← best, saved!"
        print(f"step {step:5d} | train: {losses['train']:.4f} | val: {losses['val']:.4f} | {elapsed:.1f}s{marker}")

    if step == MAX_ITERS:
        break

    # ── Gradient accumulation loop ────────────────────────────────────────
    # Instead of one big batch, we run ACCUM_STEPS small batches and
    # accumulate gradients before doing one optimizer step.
    # This simulates a larger batch without needing more memory.
    optimizer.zero_grad(set_to_none=True)

    for micro_step in range(ACCUM_STEPS):
        x, y = get_batch("train")

        # Mixed precision: autocast runs most ops in float16
        with torch.autocast(device_type=DEVICE, dtype=torch.float16, enabled=USE_MIXED_PRECISION):
            _, loss = model(x, y)

        # Scale loss by accumulation steps so the effective learning rate
        # is the same regardless of how many micro-batches we use
        loss = loss / ACCUM_STEPS
        loss.backward()

    # Gradient clipping: prevent any single gradient from being too large
    # This stabilizes training, especially early on when gradients can spike
    torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)

    optimizer.step()

    # MPS memory leak workaround: periodically clear leaked GPU memory
    # Without this, MPS slowly eats all 16GB and macOS kills the process
    if DEVICE == "mps" and step % 100 == 0:
        torch.mps.empty_cache()


total = time.time() - t0
print(f"\nTraining complete in {total/60:.1f} min")
print(f"Best val loss: {best_val_loss:.4f}")
print(f"Best checkpoint saved to {SAVE_PATH}")

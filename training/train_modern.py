"""
Training loop for the modernized GPT (RMSNorm + SwiGLU + RoPE + KV cache).

Same hyperparameters as train.py so we can do a fair comparison.
Saves checkpoint to checkpoints/modern_gpt.pt.
"""

import time
import os
import torch

from tokenizer import DEVICE, VOCAB_SIZE, BLOCK_SIZE, BATCH_SIZE, get_batch
from model_modern import ModernGPT

# ── Hyperparameters (same as vanilla for fair comparison) ─────────────────────
MAX_ITERS   = 5000
EVAL_ITERS  = 200
EVAL_EVERY  = 500
LR          = 3e-4
SAVE_PATH   = "checkpoints/modern_gpt.pt"

# ── Model ─────────────────────────────────────────────────────────────────────
model = ModernGPT(
    vocab_size  = VOCAB_SIZE,
    n_embd      = 384,
    n_heads     = 6,
    n_layer     = 6,
    block_size  = BLOCK_SIZE,
    dropout     = 0.3,   # increased from 0.2 to fight overfitting on tiny data
).to(DEVICE)

n_params = sum(p.numel() for p in model.parameters())
print(f"ModernGPT: {n_params:,} params ({n_params/1e6:.1f}M)")
print(f"Device: {DEVICE}")
print(f"Training for {MAX_ITERS} steps...")
print()

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

# ── Loss estimation ───────────────────────────────────────────────────────────
@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {}
    for split in ["train", "val"]:
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            x, y = get_batch(split)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

# ── Training loop ─────────────────────────────────────────────────────────────
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
            # Save best checkpoint
            torch.save({
                "model_state": model.state_dict(),
                "config": {
                    "vocab_size":  VOCAB_SIZE,
                    "n_embd":      384,
                    "n_heads":     6,
                    "n_layer":     6,
                    "block_size":  BLOCK_SIZE,
                    "dropout":     0.0,
                },
                "model_type": "modern",
                "step": step,
            }, SAVE_PATH)
            marker = " ← best, saved!"
        print(f"step {step:5d} | train loss: {losses['train']:.4f} | val loss: {losses['val']:.4f} | {elapsed:.1f}s{marker}")

    if step == MAX_ITERS:
        break

    x, y = get_batch("train")
    logits, loss = model(x, y)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    # MPS memory leak workaround
    if DEVICE == "mps" and step % 100 == 0:
        torch.mps.empty_cache()

# ── Save checkpoint ───────────────────────────────────────────────────────────
torch.save({
    "model_state": model.state_dict(),
    "config": {
        "vocab_size":  VOCAB_SIZE,
        "n_embd":      384,
        "n_heads":     6,
        "n_layer":     6,
        "block_size":  BLOCK_SIZE,
        "dropout":     0.0,   # disable dropout at inference
    },
    "model_type": "modern",
}, SAVE_PATH)

total = time.time() - t0
print(f"\nTraining complete in {total/60:.1f} min")
print(f"Checkpoint saved to {SAVE_PATH}")

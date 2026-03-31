"""
Training loop for the vanilla GPT model.

Config: ~10M params, 5000 iterations, AdamW lr=3e-4.
Prints train/val loss every 500 steps and saves the final checkpoint.
"""

import time
import torch

from tokenizer import DEVICE, VOCAB_SIZE, BLOCK_SIZE, BATCH_SIZE, get_batch
from model import GPT

# ── Hyperparameters ───────────────────────────────────────────────────────────
MAX_ITERS   = 5000
EVAL_ITERS  = 200      # batches averaged for loss estimate
EVAL_EVERY  = 500      # steps between evaluations
LR          = 3e-4
SAVE_PATH   = "checkpoints/vanilla_gpt.pt"

# ── Model ─────────────────────────────────────────────────────────────────────
model = GPT(
    vocab_size  = VOCAB_SIZE,
    n_embd      = 384,
    n_heads     = 6,
    n_layer     = 6,
    block_size  = BLOCK_SIZE,
    dropout     = 0.2,
).to(DEVICE)

n_params = sum(p.numel() for p in model.parameters())
print(f"Model: {n_params:,} params ({n_params/1e6:.1f}M)")
print(f"Device: {DEVICE}")
print(f"Training for {MAX_ITERS} steps...")
print()

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

# ── Loss estimation ───────────────────────────────────────────────────────────
@torch.no_grad()
def estimate_loss() -> dict[str, float]:
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
import os
os.makedirs("checkpoints", exist_ok=True)

t0 = time.time()
for step in range(MAX_ITERS + 1):

    # Evaluate periodically
    if step % EVAL_EVERY == 0:
        losses = estimate_loss()
        elapsed = time.time() - t0
        print(f"step {step:5d} | train loss: {losses['train']:.4f} | val loss: {losses['val']:.4f} | {elapsed:.1f}s")

    if step == MAX_ITERS:
        break

    # Forward + backward + update
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
        "dropout":     0.2,
    },
}, SAVE_PATH)

total_time = time.time() - t0
print(f"\nTraining complete in {total_time/60:.1f} min")
print(f"Checkpoint saved to {SAVE_PATH}")

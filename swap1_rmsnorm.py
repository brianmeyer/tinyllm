"""
Swap 1: Replace LayerNorm with RMSNorm.

WHY THIS MATTERS:
  LayerNorm normalizes by: (x - mean) / std
  RMSNorm normalizes by:   x / rms    (skips the mean subtraction)

  Turns out subtracting the mean doesn't help much in practice, and
  removing it saves compute. All modern LLMs (LLaMA, Qwen, Mistral)
  use RMSNorm. The original paper showed equivalent or better results
  with 7-64% speedup depending on the layer.

WHAT WE CHANGE:
  - Replace nn.LayerNorm with our RMSNorm class in the Block
  - Everything else (attention, FFN, model architecture) stays vanilla

WHAT TO LOOK FOR:
  - Loss should be very similar to vanilla (maybe slightly better)
  - This swap is more about efficiency than quality at this scale
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from tokenizer import DEVICE, VOCAB_SIZE, BLOCK_SIZE, BATCH_SIZE, get_batch
from attention import Head


# ── RMSNorm implementation ────────────────────────────────────────────────────
class RMSNorm(nn.Module):
    """Root Mean Square Normalization.

    Standard LayerNorm:
        y = (x - mean(x)) / sqrt(var(x) + eps) * weight + bias
        → Computes mean AND variance, has both weight and bias

    RMSNorm:
        y = x / sqrt(mean(x^2) + eps) * weight
        → Just computes RMS (root mean square), only has weight (no bias)

    The insight: the re-centering (subtracting mean) doesn't contribute much.
    The re-scaling (dividing by magnitude) is what matters.
    """
    def __init__(self, n_embd, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(n_embd))

    def forward(self, x):
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return (x / rms) * self.weight


# ── Vanilla components (unchanged) ────────────────────────────────────────────
class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, head_size, n_embd, block_size, dropout):
        super().__init__()
        self.heads = nn.ModuleList([
            Head(head_size, n_embd, block_size, dropout) for _ in range(n_heads)
        ])
        self.proj = nn.Linear(n_heads * head_size, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    """Same vanilla ReLU FFN — NOT swapped yet."""
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


# ── Block with RMSNorm (THE ONLY CHANGE) ─────────────────────────────────────
class Block(nn.Module):
    """Transformer block — ONLY difference from vanilla: RMSNorm instead of LayerNorm."""
    def __init__(self, n_embd, n_heads, block_size, dropout):
        super().__init__()
        head_size = n_embd // n_heads
        self.attn = MultiHeadAttention(n_heads, head_size, n_embd, block_size, dropout)
        self.ffn = FeedForward(n_embd, dropout)
        # ↓↓↓ THIS IS THE SWAP: RMSNorm instead of nn.LayerNorm ↓↓↓
        self.ln1 = RMSNorm(n_embd)
        self.ln2 = RMSNorm(n_embd)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class GPT_RMSNorm(nn.Module):
    """Vanilla GPT but with RMSNorm everywhere."""
    def __init__(self, vocab_size, n_embd=384, n_heads=6, n_layer=6, block_size=256, dropout=0.2):
        super().__init__()
        self.block_size = block_size
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[
            Block(n_embd, n_heads, block_size, dropout) for _ in range(n_layer)
        ])
        self.ln_f = RMSNorm(n_embd)  # Final norm is also RMSNorm
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, 0.0, 0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.token_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device))
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


# ── Training: 2000 steps, then compare ────────────────────────────────────────
if __name__ == "__main__":
    STEPS = 2000
    EVAL_EVERY = 500
    EVAL_ITERS = 200

    model = GPT_RMSNorm(VOCAB_SIZE, block_size=BLOCK_SIZE).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    print("=" * 60)
    print("SWAP 1: LayerNorm → RMSNorm")
    print("=" * 60)
    print(f"Parameters: {n_params:,}")
    print(f"Training for {STEPS} steps...")
    print()

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

    t0 = time.time()
    for step in range(STEPS + 1):
        if step % EVAL_EVERY == 0:
            losses = estimate_loss()
            elapsed = time.time() - t0
            print(f"step {step:5d} | train: {losses['train']:.4f} | val: {losses['val']:.4f} | {elapsed:.1f}s")

        if step == STEPS:
            break

        x, y = get_batch("train")
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # MPS memory leak workaround
        if DEVICE == "mps" and step % 100 == 0:
            torch.mps.empty_cache()

    total = time.time() - t0
    print(f"\nDone in {total/60:.1f} min")
    print()
    print("COMPARE: Look at val loss at step 2000 vs vanilla's val loss at step 2000")
    print("(Vanilla step 2000 is interpolated between step 1500 and 2500)")
    print("If similar or slightly better → RMSNorm is a free upgrade.")

"""
Swap 3: Replace learned positional embeddings with RoPE.

WHY THIS MATTERS:
  Vanilla approach: each position gets a learned embedding vector that's ADDED
  to the token embedding. Position 5 always adds the same vector. The model has
  to learn from scratch that nearby tokens matter more than far-away ones.

  RoPE approach: instead of adding position info, we ROTATE the Q and K vectors
  in attention. Each position rotates by a different angle. The math works out
  so that when you compute Q·K (the attention score), it only depends on the
  RELATIVE distance between the two tokens, not their absolute positions.

  Why this is better:
  1. Relative positions are what actually matter for language understanding
  2. Generalizes to sequence lengths longer than training (vanilla can't)
  3. No extra parameters (the learned pos_emb table is removed entirely)

HOW THE ROTATION WORKS:
  - Take consecutive pairs of dimensions: (dim0, dim1), (dim2, dim3), ...
  - Each pair gets rotated by angle = position × frequency
  - Frequencies decrease geometrically: dim0 rotates fast, last dim barely moves
  - Fast rotation = captures local patterns (nearby words)
  - Slow rotation = captures global structure (long-range dependencies)

  The rotation formula for a pair (x1, x2) at position m:
    x1' = x1 × cos(m×θ) - x2 × sin(m×θ)
    x2' = x1 × sin(m×θ) + x2 × cos(m×θ)

  This is literally a 2D rotation matrix applied to each pair of dimensions.

WHAT WE CHANGE:
  - Remove positional embedding table entirely
  - Add RoPE rotation to Q and K in every attention head
  - Everything else stays vanilla (LayerNorm, ReLU FFN)

WHAT TO LOOK FOR:
  - Loss should be comparable or slightly better than vanilla
  - The real benefit (length generalization) can't be tested on tiny Shakespeare
  - But understanding RoPE is critical for Qwen3 audio head grafting
"""

import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from tokenizer import DEVICE, VOCAB_SIZE, BLOCK_SIZE, BATCH_SIZE, get_batch


# ── RoPE helper functions ─────────────────────────────────────────────────────
def precompute_rope_freqs(head_size, seq_len, device, theta=10000.0):
    """Precompute cos and sin tables for RoPE.

    Each pair of dimensions gets a frequency: freq_i = 1/θ^(2i/d)
    - Pair 0 (dims 0,1): highest frequency → rotates fast → local patterns
    - Last pair: lowest frequency → rotates slow → global patterns

    θ (theta) = 10000 by default. Larger θ = slower rotation = longer effective
    context. LLaMA uses 10000, some models use 500000 for very long contexts.
    """
    # Frequencies for each pair of dimensions
    i = torch.arange(0, head_size, 2, device=device).float()  # [0, 2, 4, ...]
    freqs = 1.0 / (theta ** (i / head_size))                  # (head_size//2,)

    # Angles: position × frequency
    positions = torch.arange(seq_len, device=device).float()   # [0, 1, 2, ...]
    angles = torch.outer(positions, freqs)                     # (seq_len, head_size//2)

    return angles.cos(), angles.sin()


def apply_rope(x, cos, sin):
    """Rotate Q or K vectors by position-dependent angles.

    x: (B, T, head_size)  — the Q or K tensor for one attention head
    cos, sin: (T, head_size//2)  — precomputed rotation tables

    For each pair of dimensions (x1, x2), we apply a 2D rotation:
      x1_new = x1 * cos - x2 * sin
      x2_new = x1 * sin + x2 * cos
    """
    B, T, C = x.shape
    x1 = x[..., 0::2]  # even dimensions: (B, T, C//2)
    x2 = x[..., 1::2]  # odd dimensions:  (B, T, C//2)

    cos = cos[:T].unsqueeze(0)  # (1, T, C//2) for broadcasting
    sin = sin[:T].unsqueeze(0)

    # Apply rotation to each pair
    out = torch.stack([
        x1 * cos - x2 * sin,   # rotated x1
        x1 * sin + x2 * cos,   # rotated x2
    ], dim=-1)                  # (B, T, C//2, 2)

    return out.flatten(-2)      # (B, T, C) — interleave pairs back


# ── Attention head with RoPE ──────────────────────────────────────────────────
class HeadWithRoPE(nn.Module):
    """Same as vanilla Head, but applies RoPE to Q and K (not V).

    Why only Q and K, not V?
    - RoPE's purpose is to make the ATTENTION PATTERN position-aware
    - The attention pattern comes from Q·K
    - V carries the actual information — it doesn't need position encoding
    """
    def __init__(self, head_size, n_embd, block_size, dropout):
        super().__init__()
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.drop  = nn.Dropout(dropout)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        # ↓↓↓ THIS IS THE SWAP: apply RoPE to Q and K ↓↓↓
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        head_size = k.shape[-1]
        scores = q @ k.transpose(-2, -1) * (head_size ** -0.5)
        scores = scores.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        weights = self.drop(weights)
        return weights @ v


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, head_size, n_embd, block_size, dropout):
        super().__init__()
        self.heads = nn.ModuleList([
            HeadWithRoPE(head_size, n_embd, block_size, dropout) for _ in range(n_heads)
        ])
        self.proj = nn.Linear(n_heads * head_size, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, cos, sin):
        out = torch.cat([h(x, cos, sin) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    """Vanilla ReLU FFN (not swapped)."""
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd), nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embd, n_heads, block_size, dropout):
        super().__init__()
        head_size = n_embd // n_heads
        self.attn = MultiHeadAttention(n_heads, head_size, n_embd, block_size, dropout)
        self.ffn = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.ln1(x), cos, sin)
        x = x + self.ffn(self.ln2(x))
        return x


class GPT_RoPE(nn.Module):
    """Vanilla GPT but with RoPE instead of learned positional embeddings.

    Key structural change: no more self.pos_emb table!
    Position is encoded via rotations in each attention head.
    """
    def __init__(self, vocab_size, n_embd=384, n_heads=6, n_layer=6, block_size=256, dropout=0.2):
        super().__init__()
        self.block_size = block_size
        self.head_size = n_embd // n_heads

        self.token_emb = nn.Embedding(vocab_size, n_embd)
        # ↓↓↓ NO pos_emb table! RoPE handles position ↓↓↓
        self.blocks = nn.ModuleList([
            Block(n_embd, n_heads, block_size, dropout) for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_embd)
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
        cos, sin = precompute_rope_freqs(self.head_size, T, idx.device)

        x = self.token_emb(idx)  # (B, T, n_embd) — NO position added!
        for block in self.blocks:
            x = block(x, cos, sin)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


# ── Training: 2000 steps ─────────────────────────────────────────────────────
if __name__ == "__main__":
    STEPS = 2000
    EVAL_EVERY = 500
    EVAL_ITERS = 200

    model = GPT_RoPE(VOCAB_SIZE, block_size=BLOCK_SIZE).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    print("=" * 60)
    print("SWAP 3: Learned Positional Embeddings → RoPE")
    print("=" * 60)
    print(f"Parameters: {n_params:,}  (fewer than vanilla — no pos_emb table!)")
    print(f"Removed: pos_emb table was {BLOCK_SIZE}×384 = {BLOCK_SIZE*384:,} params")
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
    print("COMPARE: RoPE val loss at step 2000 vs vanilla val loss at step 2000")
    print("Loss should be similar. The real RoPE win is length generalization,")
    print("which doesn't show on fixed-length tiny Shakespeare.")

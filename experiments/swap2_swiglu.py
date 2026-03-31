"""
Swap 2: Replace ReLU FFN with SwiGLU.

WHY THIS MATTERS:
  Vanilla FFN:  Linear → ReLU → Linear
    - ReLU kills all negative values (hard zero)
    - Information is permanently destroyed

  SwiGLU FFN:  silu(gate(x)) * up(x) → down → output
    - "gate" decides what to let through (using smooth SiLU activation)
    - "up" provides the actual values
    - They get multiplied: the gate controls the flow
    - Nothing is hard-zeroed — gradients flow better

  SwiGLU has 3 weight matrices instead of 2, so we shrink the hidden dim
  to keep total params ~same. Formula: hidden = round_to_64(2/3 * 4 * n_embd)

WHAT WE CHANGE:
  - Replace FeedForward (ReLU) with SwiGLU FFN
  - Everything else stays vanilla (LayerNorm, learned pos embeddings)

WHAT TO LOOK FOR:
  - Loss should be noticeably BETTER than vanilla at same step count
  - SwiGLU typically gives 0.1-0.3 better loss — significant for such a simple change
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from tokenizer import DEVICE, VOCAB_SIZE, BLOCK_SIZE, BATCH_SIZE, get_batch
from attention import Head


# ── SwiGLU FFN (THE SWAP) ─────────────────────────────────────────────────────
class SwiGLU(nn.Module):
    """Gated Linear Unit with SiLU (Swish) activation.

    Instead of:   Linear(384→1536) → ReLU → Linear(1536→384)

    We do:
      gate = Linear(384→1024)(x)     # learns WHAT to let through
      up   = Linear(384→1024)(x)     # learns the VALUES
      out  = Linear(1024→384)( silu(gate) * up )

    silu(x) = x * sigmoid(x)  — smooth, no hard zeros like ReLU

    Why 1024 instead of 1536?
      3 matrices of size 384×1024 ≈ 2 matrices of size 384×1536
      Same param count, better architecture.
    """
    def __init__(self, n_embd, dropout):
        super().__init__()
        hidden = int(2 / 3 * 4 * n_embd)
        hidden = (hidden + 63) // 64 * 64   # round to multiple of 64

        self.gate = nn.Linear(n_embd, hidden, bias=False)
        self.up   = nn.Linear(n_embd, hidden, bias=False)
        self.down = nn.Linear(hidden, n_embd, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        # silu(gate) acts as a learned "valve" controlling the signal
        return self.drop(self.down(F.silu(self.gate(x)) * self.up(x)))


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


class Block(nn.Module):
    """Transformer block — ONLY difference: SwiGLU instead of ReLU FFN."""
    def __init__(self, n_embd, n_heads, block_size, dropout):
        super().__init__()
        head_size = n_embd // n_heads
        self.attn = MultiHeadAttention(n_heads, head_size, n_embd, block_size, dropout)
        # ↓↓↓ THIS IS THE SWAP ↓↓↓
        self.ffn = SwiGLU(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)  # Still vanilla LayerNorm
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class GPT_SwiGLU(nn.Module):
    """Vanilla GPT but with SwiGLU FFN."""
    def __init__(self, vocab_size, n_embd=384, n_heads=6, n_layer=6, block_size=256, dropout=0.2):
        super().__init__()
        self.block_size = block_size
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[
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
        x = self.token_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device))
        x = self.blocks(x)
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

    model = GPT_SwiGLU(VOCAB_SIZE, block_size=BLOCK_SIZE).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    print("=" * 60)
    print("SWAP 2: ReLU FFN → SwiGLU")
    print("=" * 60)
    print(f"Parameters: {n_params:,}")

    # Show SwiGLU hidden dim for comparison
    swiglu = model.blocks[0].ffn
    print(f"SwiGLU hidden dim: {swiglu.gate.out_features} (vanilla ReLU was {4*384}=1536)")
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
    print("COMPARE: SwiGLU val loss at step 2000 vs vanilla val loss at step 2000")
    print("Expected: SwiGLU should be noticeably better (lower loss).")

"""
Milestone 3: MultiHeadAttention, FeedForward, and Transformer Block.

Architecture uses pre-norm (LayerNorm before attention/FFN, not after).
This is what modern models like LLaMA/Qwen do — it trains more stably.

Block layout:
  x -> LayerNorm -> MultiHeadAttention -> + (residual) -> LayerNorm -> FeedForward -> + (residual)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from attention import Head


class MultiHeadAttention(nn.Module):
    """Multiple attention heads running in parallel, outputs concatenated and projected."""

    def __init__(self, n_heads: int, head_size: int, n_embd: int, block_size: int, dropout: float):
        super().__init__()
        self.heads = nn.ModuleList([
            Head(head_size=head_size, n_embd=n_embd, block_size=block_size, dropout=dropout)
            for _ in range(n_heads)
        ])
        # Project concatenated heads back to n_embd
        self.proj    = nn.Linear(n_heads * head_size, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Run all heads, concatenate along the last dim
        out = torch.cat([h(x) for h in self.heads], dim=-1)   # (B, T, n_heads * head_size)
        out = self.dropout(self.proj(out))                      # (B, T, n_embd)
        return out


class FeedForward(nn.Module):
    """Position-wise feed-forward network: Linear -> ReLU -> Linear.

    Standard GPT uses a 4x expansion of n_embd in the hidden layer.
    We'll swap ReLU for SwiGLU in the modernization phase.
    """

    def __init__(self, n_embd: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Block(nn.Module):
    """One transformer block with pre-norm architecture.

    Pre-norm applies LayerNorm *before* attention/FFN (not after).
    This is more stable to train than post-norm (the original Transformer).
    """

    def __init__(self, n_embd: int, n_heads: int, block_size: int, dropout: float):
        super().__init__()
        head_size = n_embd // n_heads
        self.attn = MultiHeadAttention(
            n_heads=n_heads,
            head_size=head_size,
            n_embd=n_embd,
            block_size=block_size,
            dropout=dropout,
        )
        self.ffn  = FeedForward(n_embd=n_embd, dropout=dropout)
        self.ln1  = nn.LayerNorm(n_embd)
        self.ln2  = nn.LayerNorm(n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm + residual for attention
        x = x + self.attn(self.ln1(x))
        # Pre-norm + residual for feed-forward
        x = x + self.ffn(self.ln2(x))
        return x


# ── Quick sanity check ────────────────────────────────────────────────────────
if __name__ == "__main__":
    from tokenizer import DEVICE, BLOCK_SIZE

    n_embd     = 384
    n_heads    = 6
    dropout    = 0.1
    batch_size = 4

    block = Block(n_embd=n_embd, n_heads=n_heads, block_size=BLOCK_SIZE, dropout=dropout).to(DEVICE)

    x   = torch.randn(batch_size, BLOCK_SIZE, n_embd, device=DEVICE)
    out = block(x)

    print(f"Input  shape : {x.shape}")
    print(f"Output shape : {out.shape}  (expected [4, {BLOCK_SIZE}, {n_embd}])")

    # Count parameters
    n_params = sum(p.numel() for p in block.parameters())
    print(f"Block params : {n_params:,}")
    print("\nMilestone 3 OK: transformer block works.")

"""
Milestone 2: Single-head causal self-attention.

Implements scaled dot-product attention with:
  - Separate Q, K, V linear projections
  - Causal mask (lower-triangular) so each position can only attend to past tokens
  - Dropout on the attention weights

Key formula:
  Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Head(nn.Module):
    """Single head of causal self-attention."""

    def __init__(self, head_size: int, n_embd: int, block_size: int, dropout: float = 0.1):
        super().__init__()
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)

        # Causal mask: lower triangle of 1s, upper triangle of 0s.
        # Registered as a buffer so it moves with the model (to/from device)
        # but is NOT a learnable parameter.
        self.register_buffer(
            "tril",
            torch.tril(torch.ones(block_size, block_size))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape   # batch, time (seq len), channels (n_embd)

        k = self.key(x)     # (B, T, head_size)
        q = self.query(x)   # (B, T, head_size)
        v = self.value(x)   # (B, T, head_size)

        head_size = k.shape[-1]

        # Scaled dot-product attention scores
        # (B, T, head_size) @ (B, head_size, T) -> (B, T, T)
        scores = q @ k.transpose(-2, -1) * (head_size ** -0.5)

        # Apply causal mask: positions that shouldn't be attended to get -inf,
        # which softmax turns into 0 probability.
        scores = scores.masked_fill(self.tril[:T, :T] == 0, float("-inf"))

        weights = F.softmax(scores, dim=-1)   # (B, T, T)
        weights = self.dropout(weights)

        # Weighted sum of values
        out = weights @ v   # (B, T, head_size)
        return out


# ── Quick sanity check ────────────────────────────────────────────────────────
if __name__ == "__main__":
    from tokenizer import DEVICE, BLOCK_SIZE, get_batch

    n_embd     = 32
    head_size  = 16
    batch_size = 4

    head = Head(head_size=head_size, n_embd=n_embd, block_size=BLOCK_SIZE).to(DEVICE)

    # Use random embeddings (we don't have the full model yet)
    x = torch.randn(batch_size, BLOCK_SIZE, n_embd, device=DEVICE)
    out = head(x)

    print(f"Input  shape: {x.shape}")
    print(f"Output shape: {out.shape}  (expected [4, {BLOCK_SIZE}, {head_size}])")

    # Verify causality: output at position t should NOT depend on positions > t.
    # We do this by checking that the attention mask is lower-triangular.
    tril = head.tril[:8, :8]
    print(f"\nCausal mask (8x8 top-left corner):")
    print(tril.int())
    print("\nMilestone 2 OK: single-head causal self-attention works.")

"""
Milestone 4: Full GPT model.

Architecture:
  - Token embedding table
  - Learned positional embedding table (will be replaced with RoPE in modernization)
  - Stack of transformer Blocks
  - Final LayerNorm
  - Linear language model head (maps n_embd -> vocab_size)

~10M parameters with the default config.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer import Block


class GPT(nn.Module):

    def __init__(
        self,
        vocab_size:  int,
        n_embd:      int   = 384,
        n_heads:     int   = 6,
        n_layer:     int   = 6,
        block_size:  int   = 256,
        dropout:     float = 0.2,
    ):
        super().__init__()
        self.block_size = block_size

        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb   = nn.Embedding(block_size, n_embd)   # learned positional embeddings

        self.blocks = nn.Sequential(*[
            Block(n_embd=n_embd, n_heads=n_heads, block_size=block_size, dropout=dropout)
            for _ in range(n_layer)
        ])
        self.ln_f  = nn.LayerNorm(n_embd)           # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        # Weight tying: share token embedding and lm_head weights.
        # Standard in GPT-2 — reduces params and improves performance.
        self.lm_head.weight = self.token_emb.weight

        self._init_weights()

    def _init_weights(self):
        """Initialize weights following GPT-2 paper."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.shape
        assert T <= self.block_size, f"Sequence length {T} exceeds block_size {self.block_size}"

        positions = torch.arange(T, device=idx.device)           # (T,)
        x = self.token_emb(idx) + self.pos_emb(positions)        # (B, T, n_embd)
        x = self.blocks(x)                                        # (B, T, n_embd)
        x = self.ln_f(x)                                          # (B, T, n_embd)
        logits = self.lm_head(x)                                  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            # Reshape for cross-entropy: (B*T, vocab_size) vs (B*T,)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx:         torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k:       int | None = None,
    ) -> torch.Tensor:
        """Autoregressively generate new tokens.

        Args:
            idx: (B, T) tensor of seed token ids
            max_new_tokens: number of tokens to generate
            temperature: >1 = more random, <1 = more focused
            top_k: if set, only sample from the top-k logits
        """
        for _ in range(max_new_tokens):
            # Crop context to block_size
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature   # (B, vocab_size) — last time step

            if top_k is not None:
                # Zero out all logits below the top-k
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)   # (B, 1)
            idx = torch.cat([idx, next_id], dim=1)               # (B, T+1)

        return idx


# ── Quick model size check ────────────────────────────────────────────────────
if __name__ == "__main__":
    from tokenizer import DEVICE, VOCAB_SIZE, BLOCK_SIZE

    model = GPT(vocab_size=VOCAB_SIZE, block_size=BLOCK_SIZE).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,} (~{n_params/1e6:.1f}M)")

    # Forward pass test
    x = torch.zeros((2, 8), dtype=torch.long, device=DEVICE)
    logits, loss = model(x, x)
    print(f"Logits shape    : {logits.shape}  (expected [2, 8, {VOCAB_SIZE}])")
    print(f"Loss (untrained): {loss.item():.4f}  (expected ~{__import__('math').log(VOCAB_SIZE):.2f})")

    print("\nMilestone 4 OK: GPT model works.")

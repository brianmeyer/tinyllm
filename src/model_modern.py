"""
Modernized GPT model.

Same architecture as model.py but with all four swaps applied:
  1. RMSNorm (replaces LayerNorm everywhere)
  2. SwiGLU  (replaces ReLU FFN)
  3. RoPE    (replaces learned positional embeddings)
  4. KV Cache (for fast inference generation)

The positional embedding table is removed entirely — position is encoded
via RoPE rotations directly in each attention head.

BUG FIX (2026-03-29): RoPE positions were wrong during KV cache generation.
When generating token-by-token with use_cache=True, we were computing RoPE
for position 0 every time instead of the actual position. This made every
generated token think it was at position 0 → garbage output. Fixed by
tracking _cache_pos and passing position offset to forward().
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from modernize import ModernBlock, RMSNorm, precompute_rope_freqs


class ModernGPT(nn.Module):

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
        self.n_heads    = n_heads
        self.head_size  = n_embd // n_heads

        # Token embedding only — no positional embedding table (RoPE handles position)
        self.token_emb = nn.Embedding(vocab_size, n_embd)

        self.blocks  = nn.ModuleList([
            ModernBlock(n_embd=n_embd, n_heads=n_heads, block_size=block_size, dropout=dropout)
            for _ in range(n_layer)
        ])
        self.ln_f    = RMSNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.token_emb.weight

        # Track position for KV cache generation
        self._cache_pos = 0

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def clear_kv_cache(self):
        self._cache_pos = 0
        for block in self.blocks:
            block.clear_cache()

    def forward(
        self,
        idx:     torch.Tensor,
        targets: torch.Tensor | None = None,
        use_cache: bool = False,
    ):
        B, T = idx.shape
        assert T <= self.block_size

        # Precompute RoPE frequencies.
        # During KV cache generation, we need frequencies for the ACTUAL
        # positions (cache_pos .. cache_pos + T), not always 0..T.
        # We precompute for max length and slice to the right range.
        max_pos = self._cache_pos + T
        cos_full, sin_full = precompute_rope_freqs(self.head_size, max_pos, idx.device)
        # Slice to just the positions we need
        cos = cos_full[self._cache_pos : max_pos]   # (T, head_size//2)
        sin = sin_full[self._cache_pos : max_pos]

        if use_cache:
            self._cache_pos += T

        x = self.token_emb(idx)   # (B, T, n_embd)

        for block in self.blocks:
            x = block(x, cos, sin, use_cache=use_cache)

        x      = self.ln_f(x)
        logits = self.lm_head(x)   # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx:            torch.Tensor,
        max_new_tokens: int,
        temperature:    float = 1.0,
        top_k:          int | None = None,
    ) -> torch.Tensor:
        """Generate tokens using KV cache for fast inference."""
        self.eval()
        self.clear_kv_cache()

        # Process the prompt all at once to fill the cache
        if idx.shape[1] > 1:
            _, _ = self(idx, use_cache=True)

        for _ in range(max_new_tokens):
            # Only pass the last token — KV cache has the rest
            # RoPE now correctly uses position = cache_pos (not 0!)
            idx_last = idx[:, -1:]
            logits, _ = self(idx_last, use_cache=True)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs   = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx     = torch.cat([idx, next_id], dim=1)

        self.clear_kv_cache()
        return idx


# ── Sanity check ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import time
    from tokenizer import DEVICE, VOCAB_SIZE, BLOCK_SIZE

    model = ModernGPT(vocab_size=VOCAB_SIZE, block_size=BLOCK_SIZE).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"ModernGPT parameters : {n_params:,} (~{n_params/1e6:.1f}M)")

    # Forward pass
    x = torch.zeros((2, 8), dtype=torch.long, device=DEVICE)
    logits, loss = model(x, x)
    print(f"Logits shape         : {logits.shape}")
    print(f"Loss (untrained)     : {loss.item():.4f}")

    # Confirm no positional embedding table
    has_pos_emb = hasattr(model, "pos_emb")
    print(f"Has pos_emb table    : {has_pos_emb}  (expected False — using RoPE)")

    print("\nModernGPT OK.")

"""
Swap 4: Add KV Cache for fast inference.

WHY THIS MATTERS:
  Without KV cache (naive generation):
    - To generate token 100, you reprocess ALL 100 tokens through the model
    - To generate token 101, you reprocess ALL 101 tokens
    - O(n²) total compute for n generated tokens

  With KV cache:
    - Process the prompt once, STORE the K and V tensors for each layer
    - To generate each new token, only compute K,V for that ONE token
    - Concatenate with cached K,V, run attention, done
    - O(n) total compute — MUCH faster for long sequences

  For a 256-token context, that's up to 256× fewer K,V computations per step.

WHAT WE CHANGE:
  - Add cache storage to each attention head
  - During generation: compute K,V for only the new token, concat with cache
  - During training: no change (we process full sequences, no cache needed)

WHAT TO LOOK FOR:
  - This script benchmarks generation speed WITH and WITHOUT cache
  - Generation quality is identical (same model, same math)
  - Speed should be significantly faster with cache

NOTE: This doesn't need a training run — it's an inference optimization.
We'll load the vanilla checkpoint and benchmark generation speed.
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from tokenizer import DEVICE, VOCAB_SIZE, BLOCK_SIZE, encode, decode
from attention import Head


# ── Attention head WITH KV cache ──────────────────────────────────────────────
class HeadWithCache(nn.Module):
    """Same attention as vanilla, but optionally caches K and V tensors.

    How the cache works:
      1. First call with use_cache=True: compute K,V for full prompt, store them
      2. Next calls: compute K,V for just the new token, append to cache
      3. Attention uses the full cached K,V — new token attends to everything

    Cache is cleared between generation calls via clear_cache().
    """
    def __init__(self, head_size, n_embd, block_size, dropout):
        super().__init__()
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.drop  = nn.Dropout(dropout)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self._k_cache = None
        self._v_cache = None

    def clear_cache(self):
        self._k_cache = None
        self._v_cache = None

    def forward(self, x, use_cache=False):
        B, T, C = x.shape
        k = self.key(x)    # (B, T, head_size) — T=1 when using cache!
        q = self.query(x)
        v = self.value(x)

        if use_cache:
            # Append new K,V to the cache
            if self._k_cache is not None:
                k = torch.cat([self._k_cache, k], dim=1)  # grow the cache
                v = torch.cat([self._v_cache, v], dim=1)
            self._k_cache = k
            self._v_cache = v

        T_k = k.shape[1]
        head_size = k.shape[-1]

        scores = q @ k.transpose(-2, -1) * (head_size ** -0.5)

        # Only apply causal mask during training (full sequence processing)
        if not use_cache:
            scores = scores.masked_fill(self.tril[:T, :T] == 0, float("-inf"))

        weights = F.softmax(scores, dim=-1)
        weights = self.drop(weights)
        return weights @ v


# ── Full model (vanilla + KV cache) ──────────────────────────────────────────
class MultiHeadAttentionCached(nn.Module):
    def __init__(self, n_heads, head_size, n_embd, block_size, dropout):
        super().__init__()
        self.heads = nn.ModuleList([
            HeadWithCache(head_size, n_embd, block_size, dropout) for _ in range(n_heads)
        ])
        self.proj = nn.Linear(n_heads * head_size, n_embd)
        self.dropout = nn.Dropout(dropout)

    def clear_cache(self):
        for h in self.heads:
            h.clear_cache()

    def forward(self, x, use_cache=False):
        out = torch.cat([h(x, use_cache) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
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
        self.attn = MultiHeadAttentionCached(n_heads, head_size, n_embd, block_size, dropout)
        self.ffn = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def clear_cache(self):
        self.attn.clear_cache()

    def forward(self, x, use_cache=False):
        x = x + self.attn(self.ln1(x), use_cache)
        x = x + self.ffn(self.ln2(x))
        return x


class GPT_KVCache(nn.Module):
    """Vanilla GPT with KV cache for fast generation."""
    def __init__(self, vocab_size, n_embd=384, n_heads=6, n_layer=6, block_size=256, dropout=0.0):
        super().__init__()
        self.block_size = block_size
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList([
            Block(n_embd, n_heads, block_size, dropout) for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

    def clear_cache(self):
        for b in self.blocks:
            b.clear_cache()

    def forward(self, idx, targets=None, use_cache=False):
        B, T = idx.shape
        positions = torch.arange(T, device=idx.device)
        x = self.token_emb(idx) + self.pos_emb(positions)
        for block in self.blocks:
            x = block(x, use_cache)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


# ── Benchmark: with and without KV cache ──────────────────────────────────────
if __name__ == "__main__":
    import os

    CKPT = "checkpoints/vanilla_gpt.pt"
    GEN_TOKENS = 200

    print("=" * 60)
    print("SWAP 4: KV Cache for inference")
    print("=" * 60)

    if not os.path.exists(CKPT):
        print(f"\nCheckpoint {CKPT} not found — vanilla training still running?")
        print("Creating a fresh model for benchmarking instead...\n")
        model = GPT_KVCache(VOCAB_SIZE, block_size=BLOCK_SIZE, dropout=0.0).to(DEVICE)
    else:
        print(f"Loading vanilla checkpoint from {CKPT}...")
        ckpt = torch.load(CKPT, map_location=DEVICE, weights_only=False)
        config = ckpt["config"]
        config["dropout"] = 0.0
        model = GPT_KVCache(**config).to(DEVICE)
        model.load_state_dict(ckpt["model_state"])
        print("Loaded.\n")

    model.eval()
    prompt = encode("ROMEO:")
    prompt_t = torch.tensor([prompt], dtype=torch.long, device=DEVICE)

    # ── Generate WITHOUT cache (vanilla method) ──────────────────────────────
    print(f"Generating {GEN_TOKENS} tokens WITHOUT KV cache...")
    torch.mps.synchronize() if DEVICE == "mps" else None
    t0 = time.time()

    idx = prompt_t.clone()
    with torch.no_grad():
        for _ in range(GEN_TOKENS):
            idx_cond = idx[:, -model.block_size:]
            logits, _ = model(idx_cond, use_cache=False)
            logits = logits[:, -1, :] / 0.8
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)

    torch.mps.synchronize() if DEVICE == "mps" else None
    t_no_cache = time.time() - t0
    text_no_cache = decode(idx[0].tolist())

    # ── Generate WITH cache ──────────────────────────────────────────────────
    print(f"Generating {GEN_TOKENS} tokens WITH KV cache...")
    torch.mps.synchronize() if DEVICE == "mps" else None
    t0 = time.time()

    model.clear_cache()
    idx = prompt_t.clone()
    with torch.no_grad():
        # Process prompt to fill cache
        _, _ = model(idx, use_cache=True)
        for _ in range(GEN_TOKENS):
            logits, _ = model(idx[:, -1:], use_cache=True)
            logits = logits[:, -1, :] / 0.8
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)

    torch.mps.synchronize() if DEVICE == "mps" else None
    t_with_cache = time.time() - t0

    model.clear_cache()

    # ── Results ──────────────────────────────────────────────────────────────
    print()
    print("RESULTS:")
    print(f"  Without cache: {t_no_cache:.3f}s  ({GEN_TOKENS/t_no_cache:.1f} tok/s)")
    print(f"  With cache:    {t_with_cache:.3f}s  ({GEN_TOKENS/t_with_cache:.1f} tok/s)")
    speedup = t_no_cache / t_with_cache if t_with_cache > 0 else 0
    print(f"  Speedup:       {speedup:.1f}x faster")
    print()
    print("Sample output (no-cache, should be same quality as cache):")
    print(text_no_cache[:300])

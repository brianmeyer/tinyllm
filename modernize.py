"""
Phase 3: Modern architecture components.

Four swaps over the vanilla transformer:
  1. RMSNorm      — replaces LayerNorm (simpler, faster)
  2. SwiGLU       — replaces ReLU FFN (better gradient flow, used in LLaMA/Qwen)
  3. RoPE         — replaces learned positional embeddings (better length generalization)
  4. KV Cache     — enables fast autoregressive inference

These are the components that make a "modern" LLM. After swapping all four,
the architecture is structurally similar to LLaMA / Qwen at tiny scale.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Swap 1: RMSNorm ────────────────────────────────────────────────────────────
class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Simpler than LayerNorm: skips the mean-subtraction step, just divides by
    the RMS of the activations and applies a learnable scale.

    LayerNorm:  y = (x - mean(x)) / sqrt(var(x) + eps) * weight + bias
    RMSNorm:    y = x / sqrt(mean(x^2) + eps) * weight   (no mean, no bias)

    Used in: LLaMA, Qwen, Mistral, Gemma.
    Paper: "Root Mean Square Layer Normalization" (Zhang & Sennrich, 2019)
    """

    def __init__(self, n_embd: int, eps: float = 1e-6):
        super().__init__()
        self.eps    = eps
        self.weight = nn.Parameter(torch.ones(n_embd))   # learnable scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return (x / rms) * self.weight


# ── Swap 2: SwiGLU Feed-Forward ───────────────────────────────────────────────
class SwiGLU(nn.Module):
    """SwiGLU feed-forward network.

    Replaces the standard FFN: Linear -> ReLU -> Linear

    SwiGLU uses a gated mechanism:
        gate = xW_gate
        up   = xW_up
        out  = (gate * silu(up)) @ W_down    ← silu(x) = x * sigmoid(x)

    Three weight matrices instead of two. To keep param count similar to a
    standard 4x FFN, we use hidden_dim = (2/3 * 4 * n_embd) rounded to nearest
    multiple of 64 (hardware-friendly).

    Used in: LLaMA, Qwen, Mistral, PaLM.
    Paper: "GLU Variants Improve Transformer" (Shazeer, 2020)
    """

    def __init__(self, n_embd: int, dropout: float):
        super().__init__()
        # Target hidden dim: 2/3 of 4x expansion, rounded to multiple of 64
        hidden = int(2 / 3 * 4 * n_embd)
        hidden = (hidden + 63) // 64 * 64   # round up to multiple of 64

        self.gate = nn.Linear(n_embd, hidden, bias=False)
        self.up   = nn.Linear(n_embd, hidden, bias=False)
        self.down = nn.Linear(hidden, n_embd, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.down(F.silu(self.gate(x)) * self.up(x)))


# ── Swap 3: RoPE (Rotary Position Embeddings) ─────────────────────────────────
def precompute_rope_freqs(head_size: int, seq_len: int, device: torch.device, theta: float = 10000.0):
    """Precompute the RoPE rotation frequencies.

    For each pair of dimensions (2i, 2i+1) in the head, we use frequency:
        freq_i = 1 / theta^(2i / head_size)

    Returns cos and sin tables of shape (seq_len, head_size//2).
    """
    # Frequencies decrease geometrically: dim 0 rotates fast, last dim barely moves
    i      = torch.arange(0, head_size, 2, device=device).float()   # (head_size//2,)
    freqs  = 1.0 / (theta ** (i / head_size))                        # (head_size//2,)
    pos    = torch.arange(seq_len, device=device).float()            # (seq_len,)
    angles = torch.outer(pos, freqs)                                  # (seq_len, head_size//2)
    return angles.cos(), angles.sin()                                 # each (seq_len, head_size//2)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary position embeddings to a query or key tensor.

    x:   (B, n_heads, T, head_size)
    cos: (T, head_size//2)
    sin: (T, head_size//2)

    RoPE rotates each consecutive pair of dimensions (x1, x2) by:
        x1' = x1*cos - x2*sin
        x2' = x1*sin + x2*cos

    This encodes relative position into the dot product Q·K without adding
    a separate positional embedding to the token embedding.
    """
    B, H, T, C = x.shape
    x1 = x[..., 0::2]   # even dims  (B, H, T, C//2)
    x2 = x[..., 1::2]   # odd  dims  (B, H, T, C//2)

    cos = cos[:T].unsqueeze(0).unsqueeze(0)   # (1, 1, T, C//2)
    sin = sin[:T].unsqueeze(0).unsqueeze(0)   # (1, 1, T, C//2)

    x_rot = torch.stack([
        x1 * cos - x2 * sin,
        x1 * sin + x2 * cos,
    ], dim=-1)                                # (B, H, T, C//2, 2)

    return x_rot.flatten(-2)                  # (B, H, T, C)


# ── Swap 4: Attention with RoPE + KV Cache ────────────────────────────────────
class ModernHead(nn.Module):
    """Single attention head with RoPE and optional KV cache.

    KV cache stores past (key, value) tensors so during generation we only
    compute attention for the new token, not the entire sequence.
    Disabled during training (we process full sequences with the causal mask).
    """

    def __init__(self, head_size: int, n_embd: int, block_size: int, dropout: float):
        super().__init__()
        self.head_size  = head_size
        self.block_size = block_size

        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.drop  = nn.Dropout(dropout)

        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

        # KV cache (None = disabled, set during inference)
        self._kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None

    def clear_cache(self):
        self._kv_cache = None

    def forward(
        self,
        x:    torch.Tensor,
        cos:  torch.Tensor,
        sin:  torch.Tensor,
        use_cache: bool = False,
    ) -> torch.Tensor:
        B, T, C = x.shape

        k = self.key(x)    # (B, T, head_size)
        q = self.query(x)  # (B, T, head_size)
        v = self.value(x)  # (B, T, head_size)

        # Reshape for RoPE: (B, 1, T, head_size)
        k = k.unsqueeze(1)
        q = q.unsqueeze(1)

        # Apply RoPE to Q and K (not V — position only affects attention pattern)
        k = apply_rope(k, cos, sin).squeeze(1)   # (B, T, head_size)
        q = apply_rope(q, cos, sin).squeeze(1)

        # KV cache: append new K/V to cache during inference
        if use_cache:
            if self._kv_cache is not None:
                k_cache, v_cache = self._kv_cache
                k = torch.cat([k_cache, k], dim=1)
                v = torch.cat([v_cache, v], dim=1)
            self._kv_cache = (k, v)

        T_k = k.shape[1]   # key sequence length (may be longer than T with cache)

        # Scaled dot-product attention
        scores = q @ k.transpose(-2, -1) * (self.head_size ** -0.5)  # (B, T, T_k)

        # Causal mask — only needed during training (full sequence)
        if not use_cache:
            scores = scores.masked_fill(self.tril[:T, :T] == 0, float("-inf"))

        weights = F.softmax(scores, dim=-1)
        weights = self.drop(weights)
        return weights @ v   # (B, T, head_size)


class ModernMultiHeadAttention(nn.Module):
    """Multi-head attention using ModernHead (RoPE + KV cache)."""

    def __init__(self, n_heads: int, head_size: int, n_embd: int, block_size: int, dropout: float):
        super().__init__()
        self.heads = nn.ModuleList([
            ModernHead(head_size, n_embd, block_size, dropout)
            for _ in range(n_heads)
        ])
        self.proj = nn.Linear(n_heads * head_size, n_embd, bias=False)
        self.drop = nn.Dropout(dropout)

    def clear_cache(self):
        for h in self.heads:
            h.clear_cache()

    def forward(self, x, cos, sin, use_cache=False):
        out = torch.cat([h(x, cos, sin, use_cache) for h in self.heads], dim=-1)
        return self.drop(self.proj(out))


# ── Modern Transformer Block ───────────────────────────────────────────────────
class ModernBlock(nn.Module):
    """Transformer block with all four modern swaps:
        RMSNorm + ModernMultiHeadAttention (RoPE + KV cache) + SwiGLU
    """

    def __init__(self, n_embd: int, n_heads: int, block_size: int, dropout: float):
        super().__init__()
        head_size = n_embd // n_heads
        self.attn = ModernMultiHeadAttention(n_heads, head_size, n_embd, block_size, dropout)
        self.ffn  = SwiGLU(n_embd, dropout)
        self.rn1  = RMSNorm(n_embd)
        self.rn2  = RMSNorm(n_embd)

    def clear_cache(self):
        self.attn.clear_cache()

    def forward(self, x, cos, sin, use_cache=False):
        x = x + self.attn(self.rn1(x), cos, sin, use_cache)
        x = x + self.ffn(self.rn2(x))
        return x


# ── Quick sanity check ────────────────────────────────────────────────────────
if __name__ == "__main__":
    from tokenizer import DEVICE, BLOCK_SIZE

    n_embd  = 384
    n_heads = 6
    dropout = 0.1
    B, T    = 2, 64

    head_size = n_embd // n_heads

    # Test RMSNorm
    rms = RMSNorm(n_embd).to(DEVICE)
    x   = torch.randn(B, T, n_embd, device=DEVICE)
    print(f"RMSNorm output shape : {rms(x).shape}")

    # Test SwiGLU
    ffn = SwiGLU(n_embd, dropout).to(DEVICE)
    print(f"SwiGLU output shape  : {ffn(x).shape}")
    swiglu_params = sum(p.numel() for p in ffn.parameters())
    relu_params   = 2 * n_embd * (4 * n_embd)   # approximate for comparison
    print(f"SwiGLU params        : {swiglu_params:,}  (vs ReLU FFN ~{relu_params:,})")

    # Test RoPE
    cos, sin = precompute_rope_freqs(head_size, BLOCK_SIZE, DEVICE)
    print(f"RoPE cos/sin shape   : {cos.shape}")

    # Test ModernBlock
    block = ModernBlock(n_embd, n_heads, BLOCK_SIZE, dropout).to(DEVICE)
    x     = torch.randn(B, T, n_embd, device=DEVICE)
    cos_t, sin_t = precompute_rope_freqs(head_size, T, DEVICE)
    out   = block(x, cos_t, sin_t)
    print(f"ModernBlock output   : {out.shape}  (expected [{B}, {T}, {n_embd}])")

    block_params = sum(p.numel() for p in block.parameters())
    print(f"ModernBlock params   : {block_params:,}")

    print("\nAll modernize.py components OK.")

"""
Full benchmarking suite: samples, latency, throughput, comparisons.

Runs against both vanilla and modern checkpoints and produces a complete
report for the DEVLOG, README, and model card.
"""

import time
import torch
import torch.nn.functional as F

from tokenizer import encode, decode, DEVICE, VOCAB_SIZE, BLOCK_SIZE

# ── Load both models ──────────────────────────────────────────────────────────
def load_vanilla():
    from model import GPT
    ckpt = torch.load("checkpoints/vanilla_gpt.pt", map_location=DEVICE, weights_only=False)
    model = GPT(**ckpt["config"]).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model

def load_modern():
    from model_modern import ModernGPT
    ckpt = torch.load("checkpoints/modern_gpt.pt", map_location=DEVICE, weights_only=False)
    model = ModernGPT(**ckpt["config"]).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


# ── Generation helpers ────────────────────────────────────────────────────────
@torch.no_grad()
def generate_text(model, prompt, max_tokens=300, temperature=0.8, top_k=None, use_cache=False):
    """Generate text, return (text, elapsed_seconds)."""
    idx = torch.tensor([encode(prompt)], dtype=torch.long, device=DEVICE)

    if DEVICE == "mps":
        torch.mps.synchronize()
    t0 = time.time()

    if use_cache and hasattr(model, 'clear_kv_cache'):
        model.clear_kv_cache()
        # Process prompt to fill cache
        if idx.shape[1] > 1:
            _, _ = model(idx, use_cache=True)
        for _ in range(max_tokens):
            logits, _ = model(idx[:, -1:], use_cache=True)
            logits = logits[:, -1, :] / temperature
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        model.clear_kv_cache()
    else:
        for _ in range(max_tokens):
            idx_cond = idx[:, -model.block_size:]
            logits, _ = model(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)

    if DEVICE == "mps":
        torch.mps.synchronize()
    elapsed = time.time() - t0

    return decode(idx[0].tolist()), elapsed


@torch.no_grad()
def measure_throughput(model, n_tokens=500, n_runs=3, use_cache=False):
    """Measure average generation throughput over multiple runs."""
    times = []
    for _ in range(n_runs):
        _, elapsed = generate_text(model, "ROMEO:", max_tokens=n_tokens, temperature=0.8, use_cache=use_cache)
        times.append(elapsed)
    avg = sum(times) / len(times)
    return n_tokens / avg, avg  # tok/s, avg_time


# ── Main benchmark ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os

    print("=" * 70)
    print("TINYLLM BENCHMARK SUITE")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print()

    results = {}

    # ── Vanilla model ─────────────────────────────────────────────────────
    if os.path.exists("checkpoints/vanilla_gpt.pt"):
        print("-" * 70)
        print("VANILLA MODEL (LayerNorm + ReLU + learned pos embeddings)")
        print("-" * 70)
        vanilla = load_vanilla()
        n_params = sum(p.numel() for p in vanilla.parameters())
        print(f"Parameters: {n_params:,}")
        print()

        # Samples at different temperatures
        prompts = ["ROMEO:", "To be", "KING HENRY:"]
        temps = [0.5, 0.8, 1.0]

        print("### Samples ###")
        for prompt in prompts:
            for temp in temps:
                text, elapsed = generate_text(vanilla, prompt, max_tokens=150, temperature=temp)
                tok_s = 150 / elapsed
                print(f"\n[prompt=\"{prompt}\", temp={temp}, {tok_s:.0f} tok/s]")
                # Show just the generated part (after prompt)
                print(text[:len(prompt) + 200])

        # Throughput
        print("\n### Throughput ###")
        tok_s, avg_time = measure_throughput(vanilla, n_tokens=300, n_runs=3)
        print(f"  Vanilla (no cache):  {tok_s:.1f} tok/s  ({avg_time:.2f}s for 300 tokens)")
        results["vanilla_toks"] = tok_s
        results["vanilla_time"] = avg_time

        del vanilla
        torch.mps.empty_cache() if DEVICE == "mps" else None
    else:
        print("SKIP: checkpoints/vanilla_gpt.pt not found")

    print()

    # ── Modern model ──────────────────────────────────────────────────────
    if os.path.exists("checkpoints/modern_gpt.pt"):
        print("-" * 70)
        print("MODERN MODEL (RMSNorm + SwiGLU + RoPE + KV Cache)")
        print("-" * 70)
        modern = load_modern()
        n_params = sum(p.numel() for p in modern.parameters())
        print(f"Parameters: {n_params:,}")
        print()

        # Samples at different temperatures
        print("### Samples ###")
        for prompt in prompts:
            for temp in temps:
                text, elapsed = generate_text(modern, prompt, max_tokens=150, temperature=temp, use_cache=True)
                tok_s = 150 / elapsed
                print(f"\n[prompt=\"{prompt}\", temp={temp}, {tok_s:.0f} tok/s, KV cached]")
                print(text[:len(prompt) + 200])

        # Throughput: with and without cache
        print("\n### Throughput ###")
        tok_s_no, avg_no = measure_throughput(modern, n_tokens=300, n_runs=3, use_cache=False)
        tok_s_cache, avg_cache = measure_throughput(modern, n_tokens=300, n_runs=3, use_cache=True)
        print(f"  Modern (no cache):   {tok_s_no:.1f} tok/s  ({avg_no:.2f}s for 300 tokens)")
        print(f"  Modern (KV cache):   {tok_s_cache:.1f} tok/s  ({avg_cache:.2f}s for 300 tokens)")
        speedup = tok_s_cache / tok_s_no if tok_s_no > 0 else 0
        print(f"  Cache speedup:       {speedup:.2f}x")
        results["modern_toks_no_cache"] = tok_s_no
        results["modern_toks_cache"] = tok_s_cache
        results["modern_time_no_cache"] = avg_no
        results["modern_time_cache"] = avg_cache

        del modern
        torch.mps.empty_cache() if DEVICE == "mps" else None
    else:
        print("SKIP: checkpoints/modern_gpt.pt not found")

    # ── Summary ───────────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    if results:
        print(f"  {'Metric':<30} {'Vanilla':<15} {'Modern':<15} {'Modern+Cache':<15}")
        print(f"  {'-'*30} {'-'*15} {'-'*15} {'-'*15}")
        if "vanilla_toks" in results:
            print(f"  {'Throughput (tok/s)':<30} {results.get('vanilla_toks', 0):<15.1f} {results.get('modern_toks_no_cache', 0):<15.1f} {results.get('modern_toks_cache', 0):<15.1f}")
            print(f"  {'Time for 300 tokens (s)':<30} {results.get('vanilla_time', 0):<15.2f} {results.get('modern_time_no_cache', 0):<15.2f} {results.get('modern_time_cache', 0):<15.2f}")
    print()
    print("Done! Use these results to update DEVLOG.md, README.md, and MODEL_CARD.md")

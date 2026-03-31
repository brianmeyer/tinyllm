"""
Text generation from a trained GPT checkpoint.

Supports temperature, top-k, and top-p (nucleus) sampling.
Run: python generate.py --checkpoint checkpoints/vanilla_gpt.pt
"""

import argparse
import torch
import torch.nn.functional as F

from tokenizer import encode, decode, DEVICE
from model import GPT


def load_model(checkpoint_path: str):
    from model import GPT
    from model_modern import ModernGPT

    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    config = ckpt["config"]
    model_type = ckpt.get("model_type", "vanilla")

    if model_type == "modern":
        model = ModernGPT(**config).to(DEVICE)
    else:
        model = GPT(**config).to(DEVICE)

    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


@torch.no_grad()
def generate(
    model:          GPT,
    prompt:         str,
    max_new_tokens: int   = 500,
    temperature:    float = 1.0,
    top_k:          int | None = None,
    top_p:          float | None = None,
) -> str:
    """Generate text from a prompt using the given model.

    Args:
        temperature: 0.5 = focused/conservative, 1.0 = default, 1.2 = creative/chaotic
        top_k: restrict sampling to top-k most likely tokens (e.g. 50)
        top_p: nucleus sampling — restrict to smallest set of tokens whose cumulative prob >= p
    """
    idx = torch.tensor([encode(prompt)], dtype=torch.long, device=DEVICE)

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.block_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature   # (1, vocab_size)

        # Top-k filtering
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float("-inf")

        # Top-p (nucleus) filtering
        if top_p is not None:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            probs_sorted = F.softmax(sorted_logits, dim=-1)
            cumprobs = torch.cumsum(probs_sorted, dim=-1)
            # Remove tokens where cumulative prob exceeds top_p
            remove = cumprobs - probs_sorted > top_p
            sorted_logits[remove] = float("-inf")
            # Unsort back
            logits.scatter_(1, sorted_idx, sorted_logits)

        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_id], dim=1)

    return decode(idx[0].tolist())


def demo(checkpoint_path: str):
    print(f"Loading model from {checkpoint_path}...")
    model = load_model(checkpoint_path)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded: {n_params:,} params\n")

    prompt = "ROMEO:"
    configs = [
        dict(temperature=0.5, top_k=None, label="temp=0.5 (focused)"),
        dict(temperature=0.8, top_k=None, label="temp=0.8 (balanced)"),
        dict(temperature=1.0, top_k=None, label="temp=1.0 (default)"),
        dict(temperature=1.0, top_k=50,   label="temp=1.0 + top_k=50"),
        dict(temperature=1.0, top_p=0.9,  label="temp=1.0 + top_p=0.9"),
    ]

    for cfg in configs:
        label = cfg.pop("label")
        print(f"{'='*60}")
        print(f"Settings: {label}")
        print(f"{'='*60}")
        text = generate(model, prompt, max_new_tokens=300, **cfg)
        print(text)
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/vanilla_gpt.pt")
    parser.add_argument("--prompt", default="ROMEO:")
    parser.add_argument("--tokens", type=int, default=500)
    parser.add_argument("--temp", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--demo", action="store_true", help="Run all sampling configs")
    args = parser.parse_args()

    if args.demo:
        demo(args.checkpoint)
    else:
        model = load_model(args.checkpoint)
        text = generate(model, args.prompt, args.tokens, args.temp, args.top_k, args.top_p)
        print(text)

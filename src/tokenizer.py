"""
Milestone 1: Character-level tokenizer and data loading.

Reads tiny Shakespeare, builds a vocab from all unique characters,
provides encode/decode, and a get_batch() function for training.
"""

import torch

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH  = "data/input.txt"
BLOCK_SIZE = 256   # context length (tokens per sample)
BATCH_SIZE = 64    # samples per batch
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

# ── Load data ─────────────────────────────────────────────────────────────────
with open(DATA_PATH, "r") as f:
    text = f.read()

# ── Build vocab ───────────────────────────────────────────────────────────────
chars  = sorted(set(text))
VOCAB_SIZE = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}   # char -> int
itos = {i: ch for i, ch in enumerate(chars)}   # int  -> char

def encode(s: str) -> list[int]:
    return [stoi[c] for c in s]

def decode(ids: list[int]) -> str:
    return "".join(itos[i] for i in ids)

# ── Train / val split ─────────────────────────────────────────────────────────
data   = torch.tensor(encode(text), dtype=torch.long)
n      = int(0.9 * len(data))
train_data = data[:n]
val_data   = data[n:]

# ── Batch sampler ─────────────────────────────────────────────────────────────
def get_batch(split: str):
    """Return a random batch of (x, y) pairs.

    x: (BATCH_SIZE, BLOCK_SIZE) input token ids
    y: (BATCH_SIZE, BLOCK_SIZE) target token ids (x shifted right by 1)
    """
    src = train_data if split == "train" else val_data
    ix  = torch.randint(len(src) - BLOCK_SIZE, (BATCH_SIZE,))
    x   = torch.stack([src[i     : i + BLOCK_SIZE    ] for i in ix])
    y   = torch.stack([src[i + 1 : i + BLOCK_SIZE + 1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)


# ── Quick sanity check ────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Dataset length : {len(text):,} characters")
    print(f"Vocab size     : {VOCAB_SIZE} unique chars")
    print(f"Train tokens   : {len(train_data):,}")
    print(f"Val tokens     : {len(val_data):,}")
    print(f"Device         : {DEVICE}")
    print()

    sample = text[:100]
    encoded = encode(sample)
    decoded = decode(encoded)
    print(f"Sample text    : {repr(sample[:40])}")
    print(f"Encoded[:10]   : {encoded[:10]}")
    print(f"Round-trip OK  : {decoded == sample}")
    print()

    x, y = get_batch("train")
    print(f"Batch x shape  : {x.shape}  (on {x.device})")
    print(f"Batch y shape  : {y.shape}  (on {y.device})")
    print(f"x[0,:8]        : {x[0,:8].tolist()}")
    print(f"y[0,:8]        : {y[0,:8].tolist()}  (x shifted by 1)")

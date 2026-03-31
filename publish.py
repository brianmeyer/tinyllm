"""
Phase 5: Publish to HuggingFace.

Uploads the trained model, config, and README to HuggingFace Hub
as bmeyer2025/tiny-gpt-shakespeare.

Run this AFTER training is complete and you have a checkpoint.
"""

import os
import json
from huggingface_hub import HfApi, login

REPO_ID = "bmeyer2025/tiny-gpt-shakespeare"

# Files to upload
FILES = {
    "checkpoints/modern_bpe_gpt.pt": "model.pt",
    "checkpoints/config.json":       "config.json",
    "README.md":                     "README.md",
    "DEVLOG.md":                     "DEVLOG.md",

    # Source code (so people can load and run the model)
    "tokenizer.py":                  "src/tokenizer.py",
    "attention.py":                  "src/attention.py",
    "transformer.py":                "src/transformer.py",
    "model.py":                      "src/model.py",
    "modernize.py":                  "src/modernize.py",
    "model_modern.py":               "src/model_modern.py",
    "generate.py":                   "src/generate.py",
}


def main():
    print("=" * 60)
    print("Publishing to HuggingFace")
    print("=" * 60)

    # Check that checkpoint exists
    ckpt = "checkpoints/modern_bpe_gpt.pt"
    if not os.path.exists(ckpt):
        # Fall back to other checkpoints
        for alt in ["checkpoints/modern_gpt.pt", "checkpoints/vanilla_gpt.pt"]:
            if os.path.exists(alt):
                ckpt = alt
                FILES[alt] = "model.pt"
                print(f"Using checkpoint: {alt}")
                break
        else:
            print("ERROR: No checkpoint found. Train a model first.")
            print("  python -u train.py           (vanilla)")
            print("  python -u train_modern.py    (modern)")
            print("  python -u train_bpe.py       (modern + BPE)")
            return

    api = HfApi()

    # Create repo (no-op if it already exists)
    print(f"\nCreating repo {REPO_ID}...")
    api.create_repo(REPO_ID, exist_ok=True)

    # Upload each file
    for local_path, remote_path in FILES.items():
        if not os.path.exists(local_path):
            print(f"  SKIP {local_path} (not found)")
            continue
        size = os.path.getsize(local_path)
        print(f"  Uploading {local_path} → {remote_path} ({size:,} bytes)")
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=remote_path,
            repo_id=REPO_ID,
        )

    print(f"\nDone! Model published to:")
    print(f"  https://huggingface.co/{REPO_ID}")


if __name__ == "__main__":
    main()

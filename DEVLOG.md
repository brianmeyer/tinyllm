# Tiny LLM Dev Log

Building a 10M-parameter language model from scratch in PyTorch to understand how modern LLMs work from the ground up.

**Goal**: Build a decoder-only transformer, train it on Shakespeare, modernize it with the same components used in LLaMA/Qwen/Mistral, and publish to HuggingFace.

**Why this matters**: Every LLM — GPT-4, Claude, Qwen — is a scaled-up version of what we're building here. Same architecture, same training loop, same attention mechanism. The only differences are scale (billions of params vs our 10M), data (internet vs Shakespeare), and engineering optimizations.

---

## The Core Idea

A language model predicts the next token. That's it. You give it "To be or not to" and it learns that "be" is the most likely next word. Why does this simple idea produce intelligence? Because to predict what comes next, the model has to learn grammar, context, character relationships, narrative structure — understanding emerges as a side effect of getting really good at prediction.

---

## Phase 1: Setup — 2026-03-28

**Hardware**: Apple Silicon Mac Mini M4 (16GB unified memory) running PyTorch 2.11 with MPS (Metal Performance Shaders) — Apple's GPU backend. We chose to train locally instead of on Colab because a ~10M param model on 1MB of data fits comfortably in 16GB.

**Dataset**: Tiny Shakespeare (~1.1MB, ~1M characters) — the standard toy dataset for learning transformer architectures. Small enough to train in minutes, complex enough to see real patterns emerge.

**GitHub repo**: [brianmeyer/tinyllm](https://github.com/brianmeyer/tinyllm)

---

## Phase 2: Building the Vanilla Transformer — 2026-03-28

### Milestone 1: Tokenizer — turning text into numbers

Neural networks can't read text. They work with numbers. So step one is converting every character in Shakespeare into an integer.

We found 65 unique characters in the dataset (letters, punctuation, spaces, newlines). Each gets a number: `a=39, b=40, ...` etc. This is called **character-level tokenization** — the simplest possible approach. Real LLMs use **BPE** (Byte Pair Encoding) which groups common character sequences into single tokens (like "the" → one token), but character-level is clearer for learning.

The key function is `get_batch()`. It grabs a random 256-character chunk from Shakespeare and creates a training pair:
- **Input (x)**: characters 0-255
- **Target (y)**: characters 1-256 (shifted by one)

At every position, the model's job: "given everything up to here, predict the next character."

**Numbers**: 65-char vocab, 1,003,854 training tokens, 111,540 validation tokens (90/10 split).

### Milestone 2: Self-Attention — the core of the transformer

This is the mechanism that makes transformers work. Here's the intuition:

Imagine you're reading "The king picked up his crown." When you get to "his," how do you know it refers to "king" and not some other noun? You **attend** to the right word based on context. Self-attention is a differentiable version of this.

Every token produces three vectors:
- **Query (Q)**: "What am I looking for?"
- **Key (K)**: "What do I contain?"
- **Value (V)**: "What information should I share?"

The attention score between two tokens is `Q·K` (dot product). High score = "these tokens are relevant to each other." We scale by `√(head_size)` to prevent the scores from getting too large (which would make softmax too peaky).

**The causal mask** is crucial: we use a lower-triangular matrix to ensure each token can only attend to tokens that came *before* it. Token 5 can see tokens 0-5 but NOT tokens 6+. This is what makes it a language model that generates left-to-right, rather than a bidirectional encoder like BERT.

```
Attention(Q, K, V) = softmax(Q @ K^T / √d_k) @ V
```

### Milestone 3: Transformer Block — stacking the pieces

A single attention head has limited capacity — it can only focus on one "type" of relationship at a time. **Multi-head attention** runs 6 heads in parallel, each learning to focus on different patterns (one might learn syntax, another might learn character names, etc.). Their outputs get concatenated and projected back down.

The **feed-forward network (FFN)** is two linear layers with a ReLU activation. Its job: after attention has gathered information from other positions, the FFN processes that information. Think of attention as "gather" and FFN as "think."

A **Block** combines them with two critical additions:
1. **Residual connections**: `output = x + attention(x)`. The input gets added back to the output. This helps gradients flow during training — without it, deep networks are very hard to train.
2. **Pre-norm**: We apply LayerNorm *before* attention and FFN (not after). This is the modern convention (LLaMA, GPT-3 era onwards) and trains more stably.

**One Block = 1.77M parameters.** We stack 6 of them.

### Milestone 4: The Full GPT Model

The complete model:
1. **Token embedding** (65 × 384): converts each character ID to a 384-dimensional vector
2. **Position embedding** (256 × 384): adds a learned vector for each position (0-255)
3. **6 transformer blocks**: the actual computation
4. **Final LayerNorm**: stabilizes the output
5. **Language model head** (384 → 65): converts the final representation back to probabilities over the 65 characters

**Weight tying**: The language model head shares its weights with the token embedding. This means "the representation of character 'a' going into the model" is the same as "the model deciding to output character 'a'." This was in GPT-2 — it's a nice inductive bias and saves parameters.

**Total: 10.8M parameters.**

**Untrained loss**: 4.09. Theoretical random baseline for 65 characters: `ln(65) ≈ 4.17`. The model starts knowing nothing — it's essentially randomly guessing among 65 characters.

### Training the Vanilla Model

**Optimizer**: AdamW with lr=3e-4. AdamW is the standard for transformers — it's Adam (adaptive learning rates per parameter) with proper weight decay. The 3e-4 learning rate is Karpathy's recommendation for this scale.

**Batch size 64, block size 256**: Each training step processes 64 sequences of 256 characters = 16,384 characters per step.

**Loss curve:**

| Step | Train Loss | Val Loss | What's happening |
|------|-----------|---------|-----------------|
| 0    | 4.19    | 4.19  | Random guessing (1 in 65 chars) |
| 500  | 1.87    | 1.99  | Learned common characters, spaces, basic word shapes |
| 1000 | 1.44    | 1.65  | Learning word patterns, common Shakespeare phrases |
| 1500 | 1.29    | 1.54  | Diminishing returns, starting to fit real structure |
| 2000 | 1.21    | 1.49  | Val loss still improving |
| 2500 | 1.15    | 1.49  | Val loss starts plateauing |
| 3000 | 1.09    | 1.48  | **Best val loss** — sweet spot before overfitting |
| 3500 | 1.04    | 1.48  | Train still dropping, val flat |
| 4000 | 0.99    | 1.50  | Val ticks up — model starting to memorize |
| 4500 | 0.94    | 1.51  | Overfitting growing |
| 5000 | 0.88    | 1.54  | Final. Train-val gap = 0.66 |

**Total training time**: 88.1 minutes on M4 MPS (5000 steps, ~1 sec/step + eval overhead).

**Overfitting analysis**: Best val loss was **1.48 at step 3000**. After that, train loss kept dropping (the model was memorizing Shakespeare) but val loss crept back up. By step 5000 the train-val gap was 0.66 — significant. In a real project, you'd either stop early (at step 3000) or add more training data. For our learning purposes, 5000 steps is fine — we wanted to see the full curve including the overfitting regime.

**Speed**: ~1 second per training step on M4 MPS. First step took 88 seconds due to Metal shader compilation (one-time cost).

**Training crash** (first attempt): The first training run died silently after step 1500. Cause: running other GPU-intensive scripts in parallel overwhelmed MPS (16GB shared between CPU and GPU). Lesson: MPS doesn't crash cleanly like CUDA — it just kills the process with no error. Restarted from scratch with no other GPU work running.

### Generated Shakespeare — Vanilla Model

After training, we generated text at different temperatures to see the quality/creativity tradeoff.

**Temperature = 0.5 (focused, conservative):**
```
ROMEO:
I would be so straitly for thee for thy heart.

BENVOLIO:
By this and look on thee, who were thy son
As if thou couldst desire to thy love.
```
Coherent sentences, proper character names, Shakespearean rhythm. Repetitive.

**Temperature = 0.8 (balanced):**
```
ROMEO:
Thither the forest world they are.

MERCUTIO:
No better for the court.

MERCUTIO:
Let us always be for contented: have you not slander'd
therein like less behind than than offends it, and he
discharged in Verona his report.
```
More varied, still mostly grammatical. Good default setting.

**Temperature = 1.0 (default):**
```
ROMEO:
Long and Angelo: and yours, counterfeit that
Which then benefit your own solemnity.
```
Gets creative — mixes characters from different plays (Angelo from Measure for Measure shows up in a Romeo scene). More errors but also more interesting.

**What temperature actually does**: it divides the logits (raw model scores) before softmax. Low temperature makes the probability distribution peakier (model picks its top choice), high temperature flattens it (model samples more randomly). temp=0 would be pure greedy decoding — always pick the most likely token.

**What this tells us**: The model has genuinely learned Shakespeare structure — character names, dialogue formatting, verse rhythm, vocabulary. At 10M parameters trained on 1MB for 88 minutes, that's remarkable. But it also makes mistakes: garbled phrases, crossed character contexts, meaningless filler. A bigger model with more data would fix these.

---

## Phase 3: Modernizing the Architecture — 2026-03-29

The vanilla transformer we built is architecturally similar to GPT-2 (2019). Modern LLMs (LLaMA, Qwen, Mistral — all 2023-2025) use four key improvements. We swap them in one at a time, training 2000 steps each time, to see what each change does.

### Why one swap at a time?

If you change 4 things at once and the model gets better, you don't know which change helped. Controlled experiments — change one variable, measure the effect. This is how real ML research works.

### Swap 1: LayerNorm → RMSNorm

**What LayerNorm does**: normalizes activations by subtracting the mean and dividing by the standard deviation, then applies learned scale and bias.

**What RMSNorm does**: skips the mean subtraction, just divides by the root-mean-square. No bias parameter either.

**Why the change**: Turns out the mean subtraction doesn't help much. Removing it is simpler and ~7% faster with equivalent results. Zhang & Sennrich showed this in 2019.

**Used in**: LLaMA, LLaMA 2, LLaMA 3, Qwen, Qwen 2, Mistral, Gemma.

**Results** (2000 steps, crashed at step 1000 due to MPS — but we got enough data):

| Step | Vanilla val | RMSNorm val |
|------|-----------|------------|
| 500 | 1.99 | 1.99 |
| 1000 | 1.65 | 1.63 |

**Verdict**: Essentially identical. RMSNorm is a free upgrade — same quality, simpler code, fewer operations. This confirms why the entire industry adopted it: there's zero downside. In production LLMs serving millions of requests, the small efficiency gain compounds.

### Swap 2: ReLU FFN → SwiGLU

**What ReLU FFN does**: `Linear(384→1536) → ReLU → Linear(1536→384)`. ReLU sets all negative values to zero — a hard cutoff that destroys information.

**What SwiGLU does**: Uses three weight matrices instead of two:
- A **gate** matrix learns *what to let through*
- An **up** matrix provides *the values to let through*
- The gate uses SiLU (a smooth curve, no hard zeros), and its output multiplies the up values
- A **down** matrix projects back to the original dimension

**Why the change**: The gating mechanism gives the network much finer control over information flow. ReLU is a binary on/off switch; SwiGLU is a smooth dimmer. Shazeer's 2020 paper showed consistent 0.1-0.3 perplexity improvement.

**Param count trick**: Three matrices instead of two means more params. To keep it fair, we shrink the hidden dim: `int(2/3 × 4 × 384) = 1024` (vs 1536 for ReLU). Same total params, better architecture.

**Used in**: LLaMA (all versions), Qwen (all versions), Mistral, PaLM.

**Results** (full 2000 steps completed, 37.8 min):

| Step | Vanilla val | SwiGLU val | Difference |
|------|-----------|-----------|-----------|
| 500 | 1.99 | 1.88 | **-0.11** |
| 1000 | 1.65 | 1.58 | **-0.07** |
| 1500 | 1.54 | 1.51 | **-0.03** |
| 2000 | 1.49 | 1.50 | ~same |

**Verdict**: SwiGLU learned significantly faster at every early checkpoint. By step 2000 they converged — but that's because tiny Shakespeare is so small that both architectures eventually hit the same data-limited floor. On a real dataset with billions of tokens, that faster learning compounds into meaningfully better final quality. This is the swap that makes the biggest quality difference in practice.

**What surprised me**: The convergence at step 2000. I expected SwiGLU to stay ahead. But it makes sense — when your dataset is only 1MB, there's a ceiling on how good any architecture can get. The architectural advantage shows most clearly in how fast you get there, not how far you go. On larger datasets, SwiGLU would pull ahead and stay ahead.

### Swap 3: Learned Positional Embeddings → RoPE

**What learned pos embeddings do**: A lookup table of 256 vectors. Position 0 always adds vector[0], position 5 always adds vector[5]. The model has to learn what these vectors should be.

**What RoPE does**: Instead of *adding* position info to the token embedding, it *rotates* the Query and Key vectors in attention. Each position gets a different rotation angle. The angle decreases geometrically across dimensions — early dimensions rotate fast (capturing local patterns), later dimensions rotate slowly (capturing long-range structure).

**Why the change**: Three big advantages:
1. **Relative positions**: Under RoPE, the attention score Q·K only depends on the *distance* between two tokens, not their absolute positions. "The word 3 positions back" is the same pattern regardless of whether you're at position 10 or position 200.
2. **Length generalization**: The model can handle longer sequences than it was trained on (the rotation math works for any length). Learned pos embeddings can't — position 257 has no learned vector.
3. **No extra parameters**: We *remove* the pos_emb table entirely (saves 256×384 = 98,304 params). Position is encoded for free through rotations.

**The rotation math**: For each consecutive pair of dimensions (x₁, x₂):
```
x₁' = x₁·cos(mθ) - x₂·sin(mθ)
x₂' = x₁·sin(mθ) + x₂·cos(mθ)
```
where m = position and θ = frequency for that dimension pair. This is literally a 2D rotation matrix.

**Why only Q and K, not V?** RoPE's purpose is to make the attention *pattern* position-aware. Attention patterns come from Q·K. Values (V) carry the actual information content — they don't need position encoding.

**Used in**: LLaMA (all versions), Qwen (all versions), Mistral, GPT-NeoX, Gemma. This is THE standard for modern LLMs.

**This is the most important swap to understand** for working with production LLM architectures.

**Results** (full 2000 steps completed, 41.6 min):

| Step | Vanilla val | RoPE val | Difference |
|------|-----------|---------|-----------|
| 500 | 1.99 | 1.68 | **-0.31** |
| 1000 | 1.65 | 1.53 | **-0.12** |
| 1500 | 1.54 | 1.49 | **-0.05** |
| 2000 | 1.49 | 1.47 | **-0.02** |

**Verdict**: RoPE was the biggest single improvement — 0.31 better at step 500! It achieved this with 98K *fewer* parameters (no positional embedding table). The strong inductive bias of relative position encoding via rotations gives the model a massive head start. Like SwiGLU, the gap narrows as both models approach the data-limited floor, but RoPE gets there faster and ends up slightly ahead.

**Why RoPE outperformed even SwiGLU**: RoPE replaces a component the model had to *learn from scratch* (positional embeddings) with one that has strong mathematical structure *baked in* (rotations encode distance). SwiGLU is a better architecture but it still starts from random weights. RoPE starts with a correct inductive bias about how position should work.

### Swap 4: Add KV Cache for Inference

**The problem**: Without caching, generating each new token requires reprocessing the *entire* sequence through the model. For a 200-token sequence, generating token 201 means re-running all 200 tokens through attention. Generating token 202 means re-running all 201. That's O(n²) total compute.

**The solution**: After processing the prompt, *save* the Key and Value tensors for every layer. When generating a new token, only compute K and V for that one token, append to the cache, and run attention. This is O(n) total compute.

**For a 256-token context, that's up to 256× less K/V computation per token generated.**

This doesn't change training at all (during training we process full sequences anyway). It's purely an inference optimization.

**Used in**: Every production LLM inference system. This is what makes real-time chat possible.

**Results** (benchmark using vanilla checkpoint, 200 generated tokens):

|  | Speed | Throughput |
|--|-------|-----------|
| Without cache | 4.27s | 46.8 tok/s |
| With cache | 3.23s | 61.9 tok/s |
| **Speedup** | **1.3×** | |

**Verdict**: 1.3× faster on 256-token contexts. The modest speedup is because our sequences are short — KV cache shines on longer contexts. At 2048 tokens (typical for production LLMs), you'd see 10-50× speedup. At 128K tokens (Claude/GPT-4 scale), it's the difference between "responds in 1 second" and "responds in 10 minutes."

### Phase 3 Summary: All Swaps Compared

| Step | Vanilla | RMSNorm | SwiGLU | **RoPE** |
|------|---------|---------|--------|----------|
| 500 | 1.99 | 1.99 | 1.88 | **1.68** |
| 1000 | 1.65 | 1.63 | 1.58 | **1.53** |
| 1500 | 1.54 | — | 1.51 | **1.49** |
| 2000 | 1.49 | — | 1.50 | **1.47** |

**Rankings by impact**:
1. **RoPE** — biggest quality improvement, fewer parameters, strong inductive bias
2. **SwiGLU** — faster learning, better gradient flow
3. **RMSNorm** — free efficiency upgrade, no quality change
4. **KV Cache** — no quality change, 1.3× faster inference (much more at scale)

All four are compatible — they modify different parts of the transformer block. They slot into different places:

```
Transformer Block:
  ┌─────────────────────────────────────┐
  │  RMSNorm  ← Swap 1 (normalization) │
  │  MultiHeadAttention                 │
  │    └─ Q, K with RoPE ← Swap 3      │
  │    └─ KV Cache ← Swap 4            │
  │  + residual                         │
  │                                     │
  │  RMSNorm  ← Swap 1 (normalization) │
  │  SwiGLU   ← Swap 2 (feed-forward)  │
  │  + residual                         │
  └─────────────────────────────────────┘
```

Combined in `model_modern.py`, our model is architecturally identical to LLaMA/Qwen at tiny scale.

### Full Modern Model Training — The Overfitting Disaster

Trained the combined modern model (all 4 swaps) for 5000 steps. **It started out amazing:**

| Step | Modern train | Modern val | Vanilla val (comparison) |
|------|-------------|-----------|------------------------|
| 0 | 4.25 | 4.25 | 4.19 |
| 500 | 1.37 | 1.59 | 1.99 |
| 1000 | 1.21 | 1.50 | 1.65 |
| **1500** | **1.12** | **1.47** | **1.54** |
| 2000 | 1.04 | 1.50 | 1.49 |
| 3000 | 0.88 | 1.55 | 1.48 |
| 5000 | 0.58 | 1.77 | 1.54 |

At step 500, modern was val 1.59 vs vanilla's 1.99 — **0.40 better**. At step 1500, it hit 1.47 (beating vanilla's all-time best of 1.48 that took 3000 steps to reach). But then it kept going and the val loss EXPLODED to 1.77. Train loss dropped to an absurdly low 0.58 — the model was essentially memorizing Shakespeare character by character.

**What the generated text looked like at step 5000:**
```
ROMEO:
The theneveveveinourein treishathathatwhathon thishadishadishadin
madilllllllllllllllllllllllllllllllllllllllllllllllllll
```

Pure garbage. Repetitive fragments. The model memorized training data so precisely that it couldn't generalize at all.

**Why this happened**: The modern architecture (RoPE + SwiGLU) is more powerful than vanilla. On a large dataset, that extra power means better generalization. On tiny Shakespeare (only 1MB), it means faster memorization. It's like giving a photographic-memory student a one-page cheat sheet — they'll memorize it word for word instead of understanding the concepts.

**The fix — early stopping**: Added best-checkpoint saving to `train_modern.py`. Now it saves the model at whichever eval step has the lowest val loss. The best model was at step 1500, not step 5000.

### Plot twist: It wasn't overfitting — it was a RoPE position bug

The "garbage" output at step 5000 wasn't because the model was bad. When we ran the benchmark, the model produced `"ounounounounoun"` — but the val loss at step 1000 (1.51) should NOT produce garbage.

**Root cause**: A bug in the KV cache generation path. During generation, each new token is processed with `T=1`. Our code computed RoPE frequencies for `T=1`, which means **every generated token got position 0's rotation**. The model had no idea where any token was in the sequence.

```python
# BUG: always computes rotation for position 0
cos, sin = precompute_rope_freqs(head_size, T, device)  # T=1 during cache gen

# FIX: compute for actual position, slice to the right range
cos_full, sin_full = precompute_rope_freqs(head_size, cache_pos + T, device)
cos = cos_full[cache_pos : cache_pos + T]  # correct position!
```

During training (T=256, no cache), all positions got the right rotations — so the model trained perfectly. The bug only appeared during KV cache generation.

**After the fix**, the step 1000 checkpoint produces clean Shakespeare:
```
ROMEO:
Here he hours, my lord, and not were my head,
Is is in not hither, and the my petty sights
And be pass'd the morning his throat, if you will
Throw your grave of those scarce comes the poor,
Where you
```

**This is a great debugging lesson**: When a model has reasonable loss but produces garbage at inference, the bug is usually in the inference code, not the model. Check your positional encoding, attention masking, and caching logic before blaming the model.

We also discovered and fixed an **MPS memory leak** that was silently killing training processes after ~60-80 minutes. Fix: `torch.mps.empty_cache()` every 100 steps.

### Modern model retrain — with fixes

With the RoPE bug fixed and MPS cache clearing added, retrained with early stopping. Best checkpoint saved automatically at lowest val loss.

| Step | Modern train | Modern val | Status |
|------|-------------|-----------|--------|
| 0 | 4.25 | 4.25 | |
| 500 | 1.39 | 1.60 | best, saved |
| 1000 | 1.23 | 1.51 | best, saved |

The model IS overfitting on tiny Shakespeare (modern architecture is too powerful for 1MB of data), but with early stopping we capture the sweet spot. The best checkpoint generates good Shakespeare text.

---

## Phase 4: Scaling — train_bpe.py

Three upgrades to train faster and smarter. Each one is something you'd use in a real production training run.

### Upgrade 1: BPE Tokenization

Character-level tokenization was great for learning, but it's wasteful. The word "the" takes 3 tokens (t, h, e). The model wastes 3 attention steps processing one of the most common words in English.

**BPE (Byte Pair Encoding)** solves this. It's a compression algorithm that learns which character sequences appear often and merges them into single tokens. "the" → one token. "tion" → one token. Rare words still get broken into pieces, so it handles any input.

We use `tiktoken` with the GPT-2 tokenizer (50,257 tokens). The tradeoff: our embedding table grows from 65×384 = 24,960 params to 50,257×384 = 19.3M params. But each token carries ~4x more information, so sequences are ~4x shorter for the same text. Net win.

### Upgrade 2: Mixed Precision Training

By default, PyTorch does all math in float32 (32 bits per number). Mixed precision runs most operations in float16 (16 bits) — literally half the data. This means:
- ~2× faster matrix multiplications (the GPU processes twice as many numbers per cycle)
- ~Half the memory usage (bigger batches or bigger models fit)
- Minimal quality loss (sensitive operations like loss computation stay in float32)

On MPS (Apple Silicon), we use `torch.autocast(device_type='mps', dtype=torch.float16)`. CUDA uses a `GradScaler` for dynamic loss scaling, but MPS doesn't need it.

### Upgrade 3: Gradient Accumulation

Bigger batches generally train better — the gradient estimate is less noisy because you're averaging over more examples. But bigger batches need more memory. On 16GB, we can't do batch_size=64 with a BPE model.

**Gradient accumulation** solves this: run 4 "micro-batches" of size 16, accumulate the gradients without updating the model, then do one optimizer step. The model sees 4×16=64 examples per step, same as a batch_size=64, but only 16 examples are in memory at any time.

The key detail: we divide the loss by `ACCUM_STEPS` before calling `.backward()`. This ensures the accumulated gradient is properly averaged, not summed.

We also add **gradient clipping** (`max_norm=1.0`) — if any gradient gets too large, we scale the entire gradient vector down to keep its norm ≤ 1.0. This prevents training instability from gradient spikes.

*Results: [pending — will run after swap comparisons]*

---

### BPE Training — Divergence Disaster

First BPE training run (29.9M params, mixed precision float16, gradient accumulation):

| Step | Train | Val | Status |
|------|-------|-----|--------|
| 0 | 10.85 | 10.84 | Random over 50K vocab (ln(50257) ≈ 10.83, perfect) |
| 500 | 7.68 | 8.84 | Learning! Best saved |
| 1000 | 9.46 | 11.79 | **DIVERGED — worse than random** |

**What happened**: Loss exploded from 8.84 to 11.79 between step 500 and 1000. The model didn't just stop learning — it got actively *worse* than random.

**Root cause**: Mixed precision float16 on MPS. Float16 has a very limited range (max ~65,504). With a 50,257-token vocabulary, the softmax over logits can produce values that overflow or underflow in float16. The gradients become NaN, and the model parameters get corrupted. On CUDA, `GradScaler` dynamically adjusts the loss scale to prevent this. On MPS, there's no GradScaler — we were running raw float16 with no safety net.

**The fix**:
1. **Disable mixed precision** — use float32 (slower but numerically stable on MPS)
2. **Lower learning rate** — 1e-4 instead of 3e-4 (larger model needs smaller steps)
3. **Increase dropout to 0.3** — more regularization for the bigger 29.9M param model

**Lesson**: Mixed precision is a CUDA optimization. MPS doesn't have the same infrastructure (no GradScaler, different float16 behavior). Don't blindly copy CUDA training configs to MPS. Test on a few hundred steps first before committing to a long run.

### The MPS Wall — Moving to Colab

After the BPE divergence, we tried retraining the modern model with fixes:
- Added `torch.mps.empty_cache()` every 100 steps
- Set `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0`
- Increased dropout to 0.3

**None of it worked.** MPS kept killing the process after step 0-500. Every single time. We tried 4+ restarts with different configurations.

**Root cause**: MPS on macOS has a memory leak ([PyTorch issue #154329](https://github.com/pytorch/pytorch/issues/154329)). GPU memory slowly grows during training. On 16GB unified memory (shared between CPU, GPU, OS, and whatever else is running), the process gets OOM-killed by macOS after 60-80 minutes. There's no error message — the process just disappears.

**The irony**: The vanilla model trained fine on MPS (88 minutes, completed fully). But that was earlier in the session when less was in memory. The modern model runs kept dying because of accumulated system state.

**Decision: Move to Google Colab.** This is what the original guide recommended. CUDA on a T4 is rock-solid — no memory leaks, no silent kills. Created `train_colab.py` that runs all three training phases (vanilla, modern, BPE) in one shot on Colab.

**Lesson**: MPS is great for inference and short training runs. For anything over 30 minutes, use CUDA. Don't fight the hardware — use the right tool for the job. This cost us several hours of debugging that could have been spent learning.

### Colab Training Results — All Three Phases

First run on free T4: completed but checkpoints lost to runtime disconnect. Second run on Colab Pro T4: completed with Google Drive saving. All three training phases ran back-to-back with zero crashes. Total wall time: ~3.1 hours. CUDA just works.

*Final numbers below are from the Colab Pro run.*

#### Part 1: Vanilla GPT (56.9 min, best val 1.4804)

| Step | Train | Val | Status |
|------|-------|-----|--------|
| 0 | 4.23 | 4.23 | best, saved |
| 500 | 1.79 | 1.93 | best, saved |
| 1000 | 1.41 | 1.63 | best, saved |
| 1500 | 1.28 | 1.55 | best, saved |
| 2000 | 1.19 | 1.50 | best, saved |
| 2500 | 1.13 | 1.48 | best, saved |
| **3000** | **1.08** | **1.48** | **best, saved** |
| 3500 | 1.02 | 1.49 | overfitting starts |
| 4000 | 0.97 | 1.51 | |
| 4500 | 0.92 | 1.54 | |
| 5000 | 0.86 | 1.56 | |

Best checkpoint at step 3000. After that, val loss climbs while train keeps dropping — classic overfitting on a small dataset.

#### Part 2: Modern GPT with dropout 0.3 (64.2 min, best val 1.4754)

| Step | Train | Val | Status |
|------|-------|-----|--------|
| 0 | 4.32 | 4.32 | best, saved |
| 500 | 1.47 | 1.67 | best, saved |
| 1000 | 1.29 | 1.53 | best, saved |
| 1500 | 1.21 | 1.50 | best, saved |
| 2000 | 1.14 | 1.48 | best, saved |
| **2500** | **1.09** | **1.48** | **best, saved** |
| 3000 | 1.05 | 1.48 | |
| 3500 | 1.00 | 1.48 | plateau |
| 4000 | 0.96 | 1.50 | overfitting starts |
| 4500 | 0.91 | 1.52 | |
| 5000 | 0.87 | 1.55 | |

**Modern beat vanilla**: best val 1.4754 vs 1.4804. Small margin, but modern got there 500 steps sooner (step 2500 vs step 3000). The dropout 0.3 was the key fix — it pushed the overfitting point from step 1500 (with dropout 0.2) to step 2500.

**Head-to-head at each step:**

| Step | Vanilla val | Modern val | Modern advantage |
|------|-----------|-----------|-----------------|
| 500 | 1.93 | 1.67 | **-0.26** |
| 1000 | 1.63 | 1.53 | **-0.10** |
| 2000 | 1.50 | 1.48 | **-0.02** |
| Best | 1.4804 (step 3000) | **1.4754** (step 2500) | **-0.005, 500 steps faster** |

#### Part 3: BPE + Modern + Gradient Accumulation (68.2 min, best val 4.6414)

| Step | Train | Val | Status |
|------|-------|-----|--------|
| 0 | 10.94 | 10.94 | Random over 50K vocab (ln(50257) ≈ 10.83) |
| 500 | 4.32 | 4.85 | best, saved |
| **1000** | **3.64** | **4.64** | **best, saved** |
| 1500 | 3.15 | 4.77 | overfitting |
| 2000 | 2.71 | 4.93 | |
| 3000 | 2.04 | 5.43 | severe overfitting |

**BPE overfits even faster** — 29.9M params (mostly the 50K embedding table) on only 338K BPE tokens. The model memorizes the training data by step 1000. Best checkpoint saved at step 1000.

**Note on BPE loss numbers**: You can't directly compare BPE loss (4.64) to char-level loss (1.48) because they're predicting over different vocabularies. BPE perplexity = e^4.64 ≈ 103 (choosing among ~103 tokens). Char-level perplexity = e^1.48 ≈ 4.4 (choosing among ~4.4 characters). The BPE model is actually doing harder predictions — each token carries more information.

**Lesson**: BPE with a 50K vocab on 1MB of Shakespeare is a terrible ratio. The embedding table alone (50,257 × 384 = 19.3M params) is larger than the rest of the model combined. You need millions of tokens to train those embeddings properly. This is why real LLMs train on trillions of tokens — BPE only pays off at scale.

### Generated Samples — Colab T4

**Vanilla model, temp=0.8:**
```
ROMEO:
Nay, be too be so head: but I am as betimes;
There is no man with her pleasure attentience,
She doth behold our queen arms.

PAULINA:
I'll not too woe to die for the law to the world,
I'll be old fas
```

**Modern model, temp=0.8 (KV cached):**
```
ROMEO:
A gallant-house! what says the woe?

MERCUTIO:
Good madam, my lord.

ROMEO:
Villain, for I do not say it is true,
Which hath a sin by him come to the crown,
That he is reports for me; for ever is he.
```

Both produce recognizable Shakespeare with proper character names and dialogue formatting. The modern model's output is slightly more coherent — shorter sentences, cleaner dialogue turns.

### Throughput Benchmarks — Colab T4

| Model | Throughput | Time for 300 tokens |
|-------|----------|-------------------|
| Vanilla (no cache) | **72.2 tok/s** | 4.16s |
| Modern (KV cache) | 40.7 tok/s | 7.37s |

**Surprise**: Vanilla is faster! The modern model's KV cache overhead (managing cache state, extra RoPE computation) outweighs the cache benefit at this tiny sequence length (256 tokens). KV cache pays off at longer contexts (2048+). At our scale, the extra complexity slows things down.

**Why is KV cache slower at our scale?** Three reasons:

1. **The model is too small** — each forward pass takes microseconds. Cache management overhead (concatenating tensors, tracking positions, RoPE angle computation) is a larger fraction of total compute than the savings from not recomputing K/V.
2. **Our sequences are short** — 256 tokens max. KV cache saves recomputing K/V for all previous tokens. At 256 tokens that's a small saving. At 128K tokens (Claude/GPT-4 scale) it would be ~500× faster.
3. **Python loop overhead** — our attention heads run in a Python for-loop. Production systems use fused CUDA kernels (Flash Attention) where KV cache is nearly free.

**The real lesson**: Architecture improvements that win at scale can lose at small scale. RoPE + SwiGLU + KV cache are designed for billion-parameter models processing thousands of tokens. At 10M params and 256 tokens, the simpler vanilla architecture has less overhead. But the modern architecture **trained better** — that's where the real value shows.

---

## Phase 5: Publish

- Code pushed to GitHub: [brianmeyer/tinyllm](https://github.com/brianmeyer/tinyllm)
- HuggingFace: `bmeyer2025/tiny-gpt-shakespeare` (pending checkpoint download from Colab)

---

## Errors and Lessons

| What happened | Why | What we learned |
|--------------|-----|----------------|
| Training died silently at step 1500 | MPS memory leak, GPU tests in parallel | MPS leaks memory over long runs. Fix: `torch.mps.empty_cache()` every 100 steps. Don't run multiple GPU jobs on 16GB. |
| Originally bundled all 4 swaps together | Rushing, skipped the guide | Controlled experiments: change one variable, measure the effect. This is how real ML research works. |
| Python output buffering hid progress | stdout buffered when piped to file | Always use `python -u` (unbuffered) for long-running scripts. |
| **Modern model generated garbage** | **RoPE position bug in KV cache** | During KV-cached generation, every new token got position 0's rotation instead of its actual position. Model trained fine but inference was broken. Fix: track `_cache_pos` and slice RoPE frequencies to the correct range. **Biggest lesson: when loss is good but output is bad, the bug is in inference, not training.** |
| Modern model overfitted catastrophically | 10M params too powerful for 1MB data | More powerful ≠ better. Modern architecture memorizes tiny datasets faster. Fix: early stopping + more data (BPE helps by making each token carry more information). |
| **BPE training diverged at step 1000** | **Mixed precision float16 on MPS** | Float16 softmax over 50K vocab overflows on MPS. CUDA has GradScaler to prevent this; MPS doesn't. Fix: use float32, lower lr. **Don't blindly copy CUDA configs to MPS.** |
| **MPS kills training after 60-80 min** | **MPS memory leak (PyTorch #154329)** | GPU memory grows slowly until macOS OOM-kills the process silently. `empty_cache()` and watermark ratio didn't fix it. **Move to CUDA (Colab) for training over 30 min.** |
| **Lost all Colab checkpoints** | **Colab runtime disconnected after training** | Free tier runtimes are ephemeral — everything in `/content/` is deleted when the runtime disconnects. 3+ hours of training, gone. **Always mount Google Drive and save checkpoints there.** |
| **Colab GPU quota exhausted** | **Used all free T4 hours on training** | Free tier gives ~4-6 GPU hours per 24h period. Our 3.2h training run used most of it. Had to wait for quota reset to retrain. **Plan your GPU time. Save to Drive first, not after.** |

---

## Key Concepts Glossary

**Attention**: Mechanism where every token computes how much to "care about" every other token. Q·K gives relevance scores, softmax normalizes them, then we take a weighted sum of V.

**Causal mask**: Lower-triangular matrix that prevents tokens from attending to future positions. This is what makes it a left-to-right language model.

**Pre-norm**: Applying normalization *before* attention/FFN (not after). More stable training. Used in all modern LLMs.

**Residual connections**: Adding the input back to the output of each sub-layer. Helps gradients flow in deep networks.

**Weight tying**: Sharing the token embedding matrix with the output projection. Saves params and acts as a regularizer.

**Loss (cross-entropy)**: How wrong the model's predictions are. Lower = better. For random guessing over 65 characters, loss starts at ln(65) ≈ 4.17.

**Perplexity**: e^(loss). Intuition: "how many characters is the model effectively choosing between?" Perplexity of 65 = random. Perplexity of 5 = model has narrowed it down to ~5 plausible characters at each position.

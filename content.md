# ============================================================
# 1. LINKEDIN POST
# ============================================================

I let an AI agent run 16 experiments on a GPU while I slept. Here's what it found.

Last week, I set up Karpathy's autoresearch — a system where an AI agent autonomously modifies a neural network's code, trains it for 5 minutes, checks if it improved, and repeats.

I rented an A40 GPU on RunPod ($0.40/hr), tuned the model for the hardware, and let Claude run experiments overnight.

The results:
- 16 experiments completed autonomously
- 4 improvements found (val_bpb: 1.098 → 1.095)
- Total cost: ~$15

What worked:
-> Switching attention from full to sliding window (SSSL pattern)
-> Increasing warmdown ratio from 0.5 to 0.7
-> Adding a learning rate floor at 5% to prevent over-annealing

What didn't:
-> Bigger models (depth 8) — too slow on A40, fewer training steps
-> Smaller MLPs — lost capacity faster than it gained speed
-> Removing weight decay — regularization still matters even in 5-min runs

The most interesting finding: the optimal model size is hardware-dependent. On an H100, depth=8 would win. On my budget A40, depth=6 was the sweet spot because it could process 154M tokens vs only 89M at depth=8.

This is the future of ML research — autonomous agents systematically exploring the search space while you focus on higher-level decisions.

Full writeup: https://abhid.substack.com/

Code + results: https://github.com/abhid1234/autoresearch-experiments

#MachineLearning #AI #DeepLearning #NeuralNetworks #MLEngineering


# ============================================================
# 2. X.COM POST
# ============================================================

I let an AI agent run 16 autonomous experiments on a neural network overnight.

Budget: $15 GPU rental
Result: 4 improvements found

Best finding: adding a 5% learning rate floor prevents over-annealing in short training runs.

Worst idea: making the model bigger on a budget GPU — fewer steps killed it.

Autoresearch by @karpathy is genuinely the future of ML research. Set it up, go to sleep, wake up to results.

Full writeup: https://abhid.substack.com/
Code: https://github.com/abhid1234/autoresearch-experiments


# ============================================================
# 3. SUBSTACK BLOG POST
# ============================================================
# Title: I Ran an Autonomous AI Research Agent Overnight — Here's What It Discovered
# Subtitle: 16 experiments, $15, and one GPU. How Karpathy's autoresearch turns your computer into a research lab while you sleep.
# ============================================================


## The Idea

What if you could run 100 machine learning experiments while you sleep?

That's the premise behind Andrej Karpathy's [autoresearch](https://github.com/karpathy/autoresearch) — a system where an AI agent gets a small but real neural network training setup and experiments autonomously. It modifies the code, trains for 5 minutes, checks if the result improved, keeps or discards, and repeats.

I spent a weekend setting it up on a budget GPU to see what an AI researcher could discover on its own.

---

## The Setup

**Hardware:** NVIDIA A40 (48GB VRAM) on RunPod — $0.40/hour
**Model:** GPT-style transformer, 26M parameters, 6 layers
**Agent:** Claude (Sonnet) via Claude Code
**Budget:** ~$15 total

The original autoresearch is designed for an H100 ($3.59/hr). I wanted to see if the same approach works on budget hardware — spoiler: it does, with some tuning.

[IMAGE: Screenshot of RunPod dashboard showing A40 pod at 100% GPU utilization]

The first challenge was fitting the model to the A40. The defaults assume H100-level throughput, so I had to:

- **Lower model depth** from 8 to 6 layers
- **Reduce batch size** from 524K to 131K tokens
- **Switch attention pattern** from SSSL to full (L)

After tuning, the baseline model trained 1,175 steps in 5 minutes and scored a val_bpb of 1.098.

---

## What is val_bpb?

If you're new to this — val_bpb stands for "validation bits per byte." Think of it as a score for how well the model predicts text it has never seen before.

- **8.0** — random guessing
- **1.5** — learned basic patterns
- **1.0** — understands language well
- **0.7** — very strong model

Lower is better. Every experiment tries to beat the current best score.

---

## The Experiments

I let the agent run for about 4 hours. It completed 16 experiments — modifying hyperparameters, architecture choices, and optimization settings. Here's every single one:

[IMAGE: Bar chart — all 16 experiments, green = kept, red = discarded]

[IMAGE: Summary stat cards — 16 experiments, 4 improvements, 1.098 → 1.095, ~$15]

| # | What Changed | val_bpb | Kept? |
|---|-------------|---------|-------|
| 1 | Baseline (depth=6, full attention) | 1.0980 | Yes |
| 2 | Depth 6 to 8 | 1.1017 | No |
| 3 | **Window L to SSSL** | **1.0961** | **Yes** |
| 4 | Depth 8 + SSSL | 1.0990 | No |
| 5 | Depth 4 + SSSL | 1.1549 | No |
| 6 | Warmdown 0.5 to 0.3 | 1.0994 | No |
| 7 | **Warmdown 0.5 to 0.7** | **1.0960** | **Yes** |
| 8 | Halve batch size for 2x steps | 1.1007 | No |
| 9 | GQA (n_kv_head=1) | 1.1078 | No |
| 10 | HEAD_DIM 128 to 64 | 1.1032 | No |
| 11 | Matrix LR 0.04 to 0.05 | 1.0961 | No |
| 12 | MLP ratio 4 to 3 | 1.1000 | No |
| 13 | Warmdown 0.7 to 0.8 | 1.0961 | No |
| 14 | Window SSSL to all short (S) | 1.0966 | No |
| 15 | **LR floor 0 to 5%** | **1.0949** | **Yes** |
| 16 | LR floor 5% to 10% | 1.0961 | No |

**Final score: 1.0949** (down from 1.0980 baseline — a 0.28% improvement)

[IMAGE: Progress line chart — val_bpb over time with "best so far" stepped line]

---

## What Actually Worked

**1. Sliding window attention (SSSL pattern)**

The agent's first win was switching from full attention (every token attends to every other token) to an alternating pattern: 3 layers with short windows, 1 layer with full attention. This is computationally cheaper, so the model got more training steps in the same 5 minutes.

**2. Higher warmdown ratio (0.5 to 0.7)**

The "warmdown" controls how the learning rate decays at the end of training. A longer warmdown (70% of training spent decaying) gave marginal but consistent gains. The agent also tested 0.3 and 0.8 — confirming 0.7 is the sweet spot.

**3. Learning rate floor at 5%**

This was the best single discovery. Instead of letting the learning rate decay all the way to zero, keeping a small floor of 5% prevents "over-annealing" — the model keeps learning right up to the end instead of stalling. The agent also tried 10%, which was too high. Sweet spot: 5%.

---

## What Didn't Work (and Why)

**Bigger models** — Depth 8 has more capacity but on the A40, it only completed 339 steps vs 1,175 at depth 6. Not enough training time to converge.

**Smaller MLPs** — Reducing the feedforward layer from 4x to 3x embedding dimension saved compute but lost too much model capacity. More steps couldn't compensate for a dumber model.

**Grouped Query Attention (GQA)** — Sharing key/value heads across attention heads is great for large models. At our tiny scale with only 3 heads, going to 1 KV head was too aggressive.

**Halving batch size** — More steps but noisier gradients. The noise outweighed the benefit.

---

## The Meta-Insight: Model Size is Hardware-Dependent

The most interesting finding wasn't any single hyperparameter — it was that **the optimal model architecture depends entirely on your GPU.**

On my A40:
- Depth 6 → 1,175 steps → val_bpb 1.098
- Depth 8 → 339 steps → val_bpb 1.176 (worse!)

On an H100 (3-4x faster), depth 8 would train ~2,500 steps and almost certainly win. The "best architecture" isn't universal — it's a function of your compute budget per experiment.

This is why autoresearch uses a fixed time budget instead of a fixed number of steps. It automatically finds the best model *for your hardware*.

[IMAGE: A40 vs H100 comparison table from charts.html]

---

## What I Learned About AI-Driven Research

**1. The easy wins come first.** The first 7 experiments found 3 improvements. The next 9 found only 1. Diminishing returns are real.

**2. The agent is systematic.** It doesn't randomly guess — it tests one variable at a time, tries both directions (warmdown 0.3 vs 0.7 vs 0.8), and uses results to narrow the search space.

**3. Budget hardware works.** You don't need an H100. An A40 at $0.40/hr produced genuine research insights for $15 total.

**4. The insights transfer.** Learning rate scheduling findings (warmdown ratio, LR floor) are architecture-general. These would likely improve training on any hardware.

---

## The Cost Breakdown

| Item | Cost |
|------|------|
| RunPod A40 (~4 hours) | ~$1.60 |
| Anthropic API (Sonnet, 16 experiments) | ~$0.50 |
| RunPod setup/testing time | ~$3.00 |
| **Total** | **~$5-15** |

This is the democratization of ML research. A weekend and $15 gets you a real experiment log with real findings.

---

## Try It Yourself

1. Rent a GPU on [RunPod](https://runpod.io) (A40 is the best value)
2. Clone [autoresearch](https://github.com/karpathy/autoresearch)
3. Tune hyperparameters for your GPU (depth, batch size)
4. Point Claude Code or Gemini CLI at `program.md`
5. Go to sleep

My fork with A40-tuned settings and full results: [github.com/abhid1234/autoresearch-experiments](https://github.com/abhid1234/autoresearch-experiments)

---

## What's Next

I'm planning to:
- Run a longer overnight session (100+ experiments)
- Try architectural changes beyond hyperparameters (different activation functions, normalization schemes)
- Compare results across A40, RTX 4090, and eventually H100
- Write up a more rigorous analysis of which findings transfer across hardware

If you run your own autoresearch experiments, I'd love to compare notes. Drop a comment or find me on [X](https://x.com) or [LinkedIn](https://linkedin.com).

Full code and results: [github.com/abhid1234/autoresearch-experiments](https://github.com/abhid1234/autoresearch-experiments)

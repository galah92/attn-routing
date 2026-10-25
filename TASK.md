# Task: Does attention predict MoE expert routing?

## Hypothesis
In an MoE LLM, tokens that attend strongly to each other tend to be routed to the same experts,
beyond what token distance (temporal locality) already explains. If true, attention can be used to
prefetch experts at decode time (useful when experts are offloaded or bandwidth-bound).

## Environment
- Single NVIDIA L4 (24 GB VRAM). Python, PyTorch, Hugging Face `transformers`, `datasets`.
- Work in this directory. Put code in `src/`, outputs in `results/`, final write-up in `results/REPORT.md`.
- Use a venv. Pin versions in `requirements.txt`.

## Model
- `allenai/OLMoE-1B-7B-0125` (16 layers, 64 experts, top-8 routing), bf16 (~14 GB).
- Load with `attn_implementation="eager"` so attention weights are available.
- Router logits: `output_router_logits=True`. Verify their shape and that top-8 of them reproduces
  the experts the model actually uses (check the OLMoE modeling code, incl. whether top-k probs are renormalized).
- Prefer forward hooks over `output_attentions=True` for all layers at once, to control memory:
  reduce each layer's attention to what you need, then drop it.

## Data
- ~300 sequences of 1024 tokens (raise to 2048 only if memory allows), split evenly across
  three domains: general web, code, math. Use ungated datasets with `streaming=True`
  (e.g. FineWeb-Edu sample, a code dataset such as codeparrot-clean-valid, OpenWebMath).
  If a dataset is gated or unavailable, substitute an equivalent and note it in the report.
- Do not download full datasets. Keep total disk use for data under 5 GB.

## Step 0: Sanity checks (report before continuing)
1. Model loads, forward pass on one sequence, peak memory reported.
2. Router top-8 per token per layer extracted; matches the model's internal expert choice.
3. Attention per layer per head `[H, n, n]` extracted via hooks; rows sum to 1, causal.
4. Expert usage histogram per layer (to confirm routing isn't degenerate).

**Stop after Step 0 and print a short summary for the user before running Steps 1 and 2.**

## Step 1: Does the signal exist?
For each layer l and each pair (t, j) with j < t:
- `overlap(t, j) = |E_t ∩ E_j| / 8` where E are top-8 expert sets at layer l.
- `a(t, j)` = attention weight from t to j at layer l: (i) mean over heads, (ii) per head.

Analysis:
- Bin pairs by distance d = t − j (log-spaced bins, e.g. 1, 2, 3–4, 5–8, …, 513–1024).
- Within each distance bin, Spearman correlation between `a(t, j)` and `overlap(t, j)`.
  Subsample pairs if needed; report sample sizes.
- Exclude j = 0 (attention sink) in a second run and compare.
- Per-head: rank heads by within-bin correlation. Identify whether a few heads carry the signal.
- Report per layer and per domain.

Outputs: `results/step1_*.csv`, plots of correlation vs distance per layer, top-heads table.

## Step 2: Is it useful for prefetch?
For each token t and layer l, a predictor outputs a set of B experts (B in {8, 12, 16, 24, 32}).
Metric: recall = |predicted ∩ true top-8| / 8, averaged over tokens. Report per layer, per domain.

Lead time matters: the real router runs right after layer l's attention, so a predictor is only
useful if it is available earlier. Label each predictor with its lead time.

Predictors:
- **A. Attention, same layer** (lead: overlaps with layer l attention compute): score each expert by
  sum over j < t of `a_l(t, j) * 1[e in E_j^l]`; take top-B. Try mean-head and best-heads-from-Step-1
  (select heads on a held-out split to avoid leakage).
- **B. Attention, previous layer** (lead: a full layer): same, but weights from layer l−1 attention,
  expert sets from layer l of earlier tokens.

Baselines:
- **Prev-token**: experts of token t−1 at layer l (pad to B with most frequent in sequence).
- **Frequency**: B most frequent experts at layer l over tokens < t in this sequence.
- **Pre-gating** (strong baseline, lead: a full layer): take the residual stream entering layer l,
  apply layer l's pre-MLP RMSNorm and layer l's router, top-B. This is training-free.
- **Combined**: best attention predictor's scores + pre-gating logits (simple normalized sum),
  to test whether attention adds information on top of pre-gating.

Outputs: `results/step2_recall.csv`, recall-vs-B plots per layer with all predictors.

## Report (`results/REPORT.md`)
- Setup, data actually used, any deviations.
- Step 1: does attention predict overlap beyond distance? Which layers, which heads?
- Step 2: recall table at B = 8 and 16 for all predictors; does attention beat pre-gating,
  and does it add to it?
- Honest verdict: supported / not supported / mixed, with the key numbers.
- Limitations and a suggested next experiment.

## Git workflow
This directory is the repo `galah92/attn-routing`. A reviewer reads results from GitHub, so:
- Commit and push to `main` after Step 0, after Step 1, and after Step 2 (plus whenever useful).
- Commit: `src/`, `requirements.txt`, `results/*.csv`, `results/*.png`, `results/*.md`, logs under 1 MB.
- Never commit: model weights, HF cache, datasets, raw attention/router dumps, `.venv/`. Keep any
  single results file under 50 MB; summarize instead of dumping.
- After Step 0, write the sanity-check summary to `results/STEP0.md` (numbers, shapes, peak memory,
  router-match result, usage histogram plot) before pushing.
- Before starting a step, `git pull` and check `REVIEW.md` for reviewer feedback; follow it.

## Rules
- Fix random seeds. Keep scripts rerunnable end-to-end (`python -m src.run_all`).
- Prefer small runs first (10 sequences) to validate the pipeline, then scale.
- If something is ambiguous or a result looks suspicious (e.g. recall near 1.0, correlations
  identical across layers), investigate and note it rather than silently proceeding.

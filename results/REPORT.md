# Does attention predict MoE expert routing? (OLMoE-1B-7B)

## Verdict: mixed

- **Supported: attention predicts routing beyond distance.** Tokens that attend strongly to each other
  share more experts than other pairs at the same distance. Within a distance bin, Spearman
  ρ(attention, expert overlap) is 0.13–0.27 in layers 2–15 (averaged over bins), with CIs of about
  ±0.01. Controlling for exact distance, removing repeated tokens and dropping the attention sink each
  barely change it. As a predictor, attention beats tuned recency by +0.069 recall@8.
- **Not supported in practice: attention as a prefetch signal.** Training-free pre-gating gets 0.875
  recall@8 (0.976 @16), against ~0.50 for attention. Added to pre-gating + recency, previous-layer
  attention gains only **+0.0049 [+0.0046, +0.0052] recall@8** and **+0.0018 @16**. The gain is real,
  positive in every layer and domain, but negligible. It is probably not worth computing and
  transferring attention-derived scores for prefetch when pre-gating is available.

All numbers come from the eval split (150 of 300 sequences), with 95% bootstrap CIs over sequences.
Everything tuned or selected used the other 150 sequences (the selection split). Per-step details:
[`STEP0.md`](STEP0.md), [`STEP1.md`](STEP1.md), [`STEP2.md`](STEP2.md).

## Setup
- **Model:** `allenai/OLMoE-1B-7B-0125`, bf16, eager attention, on a single L4. It has 16 layers, 64
  experts with top-8 routing, and 16 heads; `norm_topk_prob=False`. Peak GPU memory is 13.5 GB for a
  1024-token forward pass with all hooks.
- **Ground-truth expert sets:** the `top_k_index` captured by a forward pre-hook on each layer's
  `mlp.experts`. Top-8 of `output_router_logits` matches it for 100% of tokens in all layers.
- **Attention:** taken from the eager attention output via forward hooks and reduced inside the hook,
  so the full `[H, n, n]` tensor is never kept. Rows sum to 1 (within bf16 rounding of 3e-3), and the
  upper triangle is exactly 0.
- **Data:** 300 sequences × 1024 tokens, 100 per domain, all ungated and streamed (no substitutions):
  - web: FineWeb-Edu `sample-10BT`
  - code: `codeparrot/codeparrot-clean-valid`
  - math: `open-web-math/open-web-math`

  Each sequence is one document: `<|endoftext|>` plus its first 1023 tokens; shorter documents were
  skipped. Data uses 1.2 MB on disk. The split is 50/50 selection/eval, stratified by domain, seed 0
  (`split.json`).
- **Reproduce:** `python -m src.run_all` (data build if missing → split → Step 0 → Step 1 → Step 2). It
  takes 16.5 min on the L4 with data cached. Seeds are fixed, and the data build is deterministic
  (byte-identical on rebuild). A final end-to-end rerun reproduced every Step 1/2 CSV and JSON
  byte-for-byte.

### Deviations and decisions
- **Start token:** the tokenizer adds no BOS, so I prepend `<|endoftext|>`. Position 0 is therefore
  always the same token, so following REVIEW.md, "exclude j = 0" is the main variant everywhere.
- **`data.py` crash at exit:** the `datasets` streaming threads abort interpreter teardown with a
  `PyGILState_Release` fatal error. `del ds; gc.collect()` did not fix it, so the fix is `os._exit(0)`
  after saving (`data.hard_exit`).
- **NVML:** nvidia-smi is broken on this VM (driver/library version mismatch), but CUDA works. Memory
  was measured with `torch.cuda.max_memory_allocated`.
- **Additions from REVIEW.md:** the token-identity control, per-sequence bootstrap CIs, stratified
  pair sampling, the recency baseline, an exact pre-gating definition, z-scored combinations with
  selection-tuned weights, attention + recency, and pre-gating + recency + attention.
- **My own additions:**
  - a partial Spearman controlling for exact distance within each bin (Step 1);
  - an oracle (the router's own logits) as a sanity check;
  - B's heads (layer l−1) ranked by B's own selection-split recall, added after review.
- **Tuning scope:** all tuned parameters are per layer. The objective is mean recall over
  B ∈ {8, 12, 16, 24, 32}.
- **Pre-gating lead time:** TASK.md labels pre-gating "a full layer" of lead. Its real lead is layer
  l's attention block only: it needs the residual after layer l−1. Previous-layer attention (B) is
  available slightly earlier, after layer l−1's attention.
- **Token range:** recall is measured over tokens t ∈ [2, 1024). Token 0 has no history, and token 1's
  only key is the sink.

## Step 1: does attention predict expert overlap beyond distance?
**Method.**
- For each sequence and each log-spaced distance bin, I take all d = 1 pairs, or up to 2000 sampled
  pairs. The same pairs are used for every layer.
- I compute Spearman ρ between a(t, j) and |E_t ∩ E_j| / 8 per sequence, then average over sequences.
- Main variant: j = 0 excluded and ids[t] ≠ ids[j]. Full tables: `STEP1_tables.md`, `step1_corr.csv`.

**Head-mean attention, eval:**

| layers | ρ, mean over bins d ≥ 2 | notes |
|---|---|---|
| 0–1 | 0.02–0.05 | signal only at d ≤ 2; layer 0 turns negative at long range (−0.10) |
| 2–8 | 0.18–0.27 (peak: layer 5, 0.268) | peak at d = 2–4 (up to 0.36), still 0.12–0.21 at d = 513–1023 |
| 9–10 | 0.13–0.16 | |
| 11–15 | 0.18–0.24 | |

- **Beyond distance:** within a bin, the partial ρ given the exact d stays within ~0.003 of ρ when
  averaged over bins. Single cells differ by ≤ 0.03.
- **d = 1 is the blind spot:** at d = 1, ρ is weak or negative in middle layers (down to −0.13). Nearly
  every token attends heavily to its predecessor, so attention doesn't separate tokens there. Locality
  carries that case (mean overlap 0.37–0.49 at d = 1).
- **Controls barely matter:** same-token pairs are only 0.7–3.4% of pairs, and removing them lowers ρ
  by ~0.01. Excluding the sink barely changes pair-level ρ, because few sampled pairs have j = 0.
- **Domains:** similar across domains. Code is slightly stronger in middle and late layers, and web is
  the weakest.
- **Heads:** a handful of heads carry a stronger signal than the layer average, but most heads in
  layers 2–15 are positive.
  - Top heads, ranked on the selection split and scored on eval: 5.6 (0.350), 14.12 (0.350),
    6.2 (0.345), 5.3 (0.344), 2.2 (0.310).
  - The best head beats its layer's head-mean by up to +0.14. In layers 11–13 no single head beats the
    head-mean.
  - Selection vs. eval head ranks agree almost perfectly (Spearman 0.999). That reflects stable head
    behaviour; the splits are disjoint documents, so it isn't leakage.
  - Head 0.4 is strongly negative (−0.29).

## Step 2: is attention useful for prefetch?
recall@B = |top-B prediction ∩ true top-8| / 8. The table shows the eval split, all domains, averaged
over layers 1–15, with 95% CI. Per-layer and per-domain tables are in `STEP2_tables.md`; the full data
is in `step2_recall.csv`.

| predictor | lead time | recall@8 | recall@16 |
|---|---|---|---|
| random | — | 0.125 | 0.250 |
| frequency (j < t) | before token t | 0.397 [0.383, 0.412] | 0.581 [0.566, 0.597] |
| prev-token | before token t | 0.425 [0.417, 0.434] | 0.637 [0.627, 0.650] |
| recency (λ tuned) | before token t | 0.434 [0.422, 0.445] | 0.627 [0.615, 0.638] |
| A: same-layer attention, head-mean | ≈ none | 0.474 [0.461, 0.487] | 0.683 [0.670, 0.695] |
| A: same-layer attention, best heads | ≈ none | 0.503 [0.489, 0.515] | 0.708 [0.696, 0.720] |
| A, head-mean, sink included | ≈ none | 0.327 [0.311, 0.341] | 0.607 [0.591, 0.622] |
| B: prev-layer attention, 3 heads | ~1 layer | 0.522 [0.509, 0.535] | 0.721 [0.710, 0.733] |
| B, head-mean | ~1 layer | 0.472 | 0.680 |
| recency + A | ≈ none | 0.507 [0.495, 0.520] | 0.712 [0.700, 0.723] |
| recency + B | ~1 layer | 0.525 [0.513, 0.538] | 0.725 [0.713, 0.737] |
| **pre-gating** | layer l's attention block | **0.875 [0.873, 0.877]** | **0.976 [0.975, 0.976]** |
| pre-gating + A | ≈ none | 0.888 [0.886, 0.889] | 0.982 [0.982, 0.983] |
| pre-gating + B | layer l's attention block | 0.888 [0.887, 0.889] | 0.983 [0.983, 0.983] |
| pre-gating + recency | layer l's attention block | 0.885 [0.884, 0.887] | 0.982 [0.982, 0.983] |
| pre-gating + recency + A | ≈ none | 0.890 [0.888, 0.891] | 0.984 [0.983, 0.984] |
| **pre-gating + recency + B** | layer l's attention block | **0.890 [0.889, 0.892]** | **0.984 [0.984, 0.984]** |
| oracle (router logits, sanity check) | — | 1.000 | 1.000 |

**Does attention beat pre-gating?** No, by a wide margin: A − P = −0.373 [−0.388, −0.357] @8. This holds
in every layer and domain; even code, where attention is strongest, gives −0.276.

**Does it add to pre-gating?** Only marginally. All gains below are paired per sequence, eval, averaged
over layers 1–15:

| comparison | gain@8 | gain@16 |
|---|---|---|
| attention adds to recency (R+A − R) | +0.073 [+0.071, +0.076] | +0.085 [+0.082, +0.088] |
| prev-layer attention adds to pre-gating (P+B − P) | +0.013 [+0.012, +0.014] | +0.007 [+0.007, +0.008] |
| recency adds to pre-gating (P+R − P) | +0.010 [+0.009, +0.011] | +0.007 [+0.006, +0.007] |
| **prev-layer attention adds to P+R (P+R+B − P+R)** | **+0.0049 [+0.0046, +0.0052]** | **+0.0018 [+0.0017, +0.0020]** |
| same-layer attention adds to P+R (P+R+A − P+R) | +0.0043 [+0.0041, +0.0045] | +0.0015 [+0.0014, +0.0016] |

- **By domain (P+R+B − P+R @8):** code +0.0069, math +0.0044, web +0.0034.
- **By layer:**
  - The gain is largest in early layers, where pre-gating is weakest (layers 1, 2, 4: pre-gating
    0.67–0.82, gain +0.010 to +0.014).
  - It is smallest in layers 12–14, where pre-gating is 0.94–0.95 @8 (gain +0.0003 to +0.0012).
  - In practice, layer 1 is where attention could matter most: P+R+B gives 0.70 vs. 0.67 for
    pre-gating.
- **Sink handling is essential:** dropping j = 0 and renormalizing adds +0.148 recall@8 to head-mean A.
  With the sink included, attention falls below recency.
- **Lead time:** the previous layer (B) is at least as good as the same layer (A), which has ≈ no
  usable lead. See the caveat on head selection under Limitations.

## Limitations
- **One model, one context length, prefill-style teacher forcing.** Decode is equivalent here by
  causality, but there are no batching effects and no long contexts (> 1024 tokens).
- **Recall is not latency.** I did not measure bytes transferred, prefetch timing or cache policies.
  A +0.005 recall gain could matter only in a severely bandwidth-bound setup, and the cost of
  computing attention-weighted expert scores was not measured either.
- **Linear z-score mixing on a grid, tuned per layer.** A learned combiner (e.g. a small model over
  pre-gating logits plus attention features) could extract more. The Step 1 correlations (ρ ≈ 0.2)
  bound how much attention can add.
- **Head selection is not symmetric between A and B.**
  - A uses the Step 1 correlation ranking, as REVIEW.md specifies. B's heads are ranked by B's own
    recall, which was added after the independent review.
  - Part of B's +0.020 edge over A may come from that.
  - The negative heads (e.g. 0.4, ρ = −0.29) were not used, per REVIEW.md. A signed or learned head
    weighting might help a little.
- **Pre-gating's lead is short.** It only covers layer l's attention block, and the comparison is at
  that lead. At longer leads (2+ layers ahead) pre-gating is expected to degrade, and attention's
  relative value may grow; this was not measured.
- **bf16 ties at the router's 8/9 boundary** (~2% of tokens) make the ground-truth set itself slightly
  arbitrary at that margin.

## Suggested next experiment
**Matched lead-time comparison at longer horizons.** Predict layer l's experts from layer l−k for
k = 1…4:
- pre-gating: the residual after layer l−k, through layer l's norm and gate;
- attention: layer l−k attention × E_j^l, j < t, with heads selected per (k, l) on the selection split;
- their tuned combination.

Report recall vs. k together with a simple offload simulator that tracks expert bytes moved and stall
time. If attention's unique contribution grows as pre-gating decays with k, attention could still earn
a place in multi-layer-ahead prefetch. If it stays at ~0.5 points, the hypothesis can be closed for
practical purposes.

# Step 3: matched lead-time comparison (k = 1..4) and offload simulator

Code: `src/step3.py` (`python -m src.step3`; also run by `src.run_all`). Full tables: `STEP3_tables.md`.
Data: `step3_recall.csv`, `step3_gains.csv`, `step3_sim.csv`, `step3_choice.json`, `step3_k1_check.json`.
All numbers: eval split (150 seqs), 95% bootstrap CIs over sequences (1000 resamples). Everything
selected or tuned (heads, head variant, mixing weights) uses the selection split (150 seqs) only.

## Method
**Lead k.** A predictor of token t's experts at layer l reads only what exists once layer l−k has
fully finished for token t, plus anything from tokens j < t at any layer.

| predictor | reads (for token t, target layer l) | available |
|---|---|---|
| P(k) pre-gating | residual after layer l−k (= input of layer l−k+1) → layer l's `post_attention_layernorm` → layer l's `mlp.gate` | after layer l−k finishes |
| A(k) attention | layer l−k attention row of t, keys j < t, j = 0 excluded and renormalized, × 1[e ∈ E_j^l] | after layer l−k's attention |
| R recency | Σ_{j<t} λ^(t−j) 1[e ∈ E_j^l], λ per layer from Step 2 | before token t starts |
| P(k)+R, P(k)+R+A(k) | z-scored sums | after layer l−k finishes |
| PRw(k) (control, added after review) | P(k)+R at P(k)+R+A(k)'s own (w_P, w_R), with w_A = 0 | after layer l−k finishes |

- **Heads of A(k)**, per (k, l): pass 1 ranks every head of layer l−k by A(k)'s own single-head recall
  on the selection split (mean over B ∈ {8, 12, 16, 24, 32}); the top 3 are averaged. Pass 2 compares
  top-3 with the head-mean on the selection split and keeps the better. Top-3 won in every (k, l).
- **Weights**, per (k, l), from Step 2's triple grid (220 points of (w_P, w_R, w_A), w = 0..1, finer
  near 0), same objective. P(k)+R is the w_A = 0 subset of the same grid, so the two are tuned in the
  same way.
- **Ranking** of all predictors as in Step 2: by score, exact ties broken by in-sequence frequency
  over j < t. Recall over tokens t ∈ [2, 1024).
- **Layers:** P(k) and A(k) need l ≥ k. The main averages use layers 4–15 for every k (same layers),
  with averages over all defined l ≥ k as a secondary column.
- **Pass 3 (eval)** recomputes every predictor at the chosen settings and saves each predictor's top-16
  ranked experts per token (`results/raw/`, not committed) for the simulator.

**k = 1 reproduces Step 2 exactly** (`step3_k1_check.json`, per-sequence max |Δ| on the eval split):
P(1) vs. Step 2 pre-gating 0.0; A(1) vs. Step 2 B 0.0 (the selected heads are identical in all 15
layers); P(1)+R and P(1)+R+A(1) vs. Step 2's pregate+recency and pregate+recency+B 0.0 (identical
tuned weights in all 15 layers). Hence the k = 1 column of every Step 3 table equals Step 2. This confirms that the code paths match;
since Step 3 reruns the same selection procedure on the same selection data, identical heads and
weights are expected rather than an independent replication.

**Offload simulator** (CPU, replays the saved rankings; layers 4–15 of all 150 eval sequences, 1024
tokens each):
- One LRU cache of C ∈ {16, 24, 32} experts per layer, starting empty; decode token by token.
- Before token t's layer l, prefetch the predictor's top B = C/2 into the cache (lowest-ranked first,
  so the top-ranked expert ends most recent; experts already resident are only touched).
- Then the true 8 experts are used: resident ones are hits (touched first), missing ones are loaded
  (misses). Every load into a full cache evicts the least recently used expert.
- Bytes moved = prefetch loads of non-resident experts + miss loads. One expert = 3 × 2048 × 1024 × 2 B
  = 12.58 MB. "Experts moved per token" sums over the 12 simulated layers.
- Since each layer has its own cache, the lead k only changes which prediction is used, not the timing.
  The simulator counts bytes and misses; it does not model bandwidth or stall time.
- Counts cover tokens t ∈ [2, 1024), as for recall; tokens 0 and 1 still run, to warm the cache.
- R is not really a prefetcher: all recency scores decay by the same λ, so its top-B consists of
  experts used by token t−1 or already in its previous top-B, which are resident whenever C ≥ 2B.
  R therefore loads nothing (`prefetched` = 0) and only reorders LRU recency.

## Results

### 1. Recall vs. k (eval, layers 4–15)

| predictor | k=1 @8 | k=2 @8 | k=3 @8 | k=4 @8 | k=1 @16 | k=4 @16 |
|---|---|---|---|---|---|---|
| P(k) | 0.903 | 0.822 | 0.753 | 0.685 | 0.991 | 0.865 |
| A(k) | 0.534 | 0.528 | 0.544 | 0.549 | 0.738 | 0.751 |
| R | 0.456 | 0.456 | 0.456 | 0.456 | 0.651 | 0.651 |
| P(k)+R | 0.911 | 0.843 | 0.793 | 0.749 | 0.995 | 0.919 |
| P(k)+R+A(k) | 0.914 | 0.850 | 0.807 | 0.768 | 0.996 | 0.932 |
| PRw(k) (control, see below) | 0.909 | 0.840 | 0.786 | 0.735 | 0.994 | 0.906 |

CI half-widths at @8 are ±0.0015–0.0045 for P and the combinations and ±0.013 for A and R
(`STEP3_tables.md`).

- Pre-gating decays steadily with lead: about −0.07 recall@8 per extra layer (−0.081, −0.069, −0.068),
  0.903 → 0.685.
- Attention does not decay at all; A(k) is flat to slightly rising (0.534 → 0.549). The previous
  layers' attention, paired with layer l's expert sets of earlier tokens, stays equally informative.
- Recency gains in relative weight as pre-gating decays: P(k)+R − P(k) grows from +0.008 to +0.064 @8.
- Even at k = 4, attention alone is far below pre-gating (A − P = −0.136 @8); in code it nearly
  catches up (−0.026 [−0.033, −0.019]).

### 2. Attention's unique gain, P(k)+R+A(k) − P(k)+R (main result)

| k | gain@8, layers 4–15 | gain@16, layers 4–15 | gain@8, all l ≥ k |
|---|---|---|---|
| 1 | +0.0038 [+0.0036, +0.0041] | +0.0008 [+0.0007, +0.0009] | +0.0049 [+0.0046, +0.0052] |
| 2 | +0.0071 [+0.0067, +0.0075] | +0.0030 [+0.0028, +0.0031] | +0.0088 [+0.0084, +0.0093] |
| 3 | +0.0134 [+0.0126, +0.0142] | +0.0081 [+0.0077, +0.0084] | +0.0148 [+0.0140, +0.0157] |
| 4 | **+0.0188 [+0.0178, +0.0198]** | +0.0130 [+0.0125, +0.0135] | +0.0188 [+0.0178, +0.0198] |

The gain grows roughly linearly with k, about 5× from k = 1 to k = 4, as expected if attention is
complementary to a decaying pre-gating signal. It stays just below +0.02 at k = 4 (CI upper bound 0.0198).

### 3. Per domain and per layer
- **Domain (k = 4, layers 4–15, @8):** code +0.0238 [+0.0224, +0.0253], math +0.0190 [+0.0176, +0.0205],
  web +0.0135 [+0.0125, +0.0145]. Same order at every k.
- **Per layer, k = 2:** +0.021 in layer 3 (the only layer above +0.02), +0.010 to +0.019 in layers 2–7,
  +0.001 to +0.007 in layers 8–15.
- **Per layer, k = 4:** above +0.02 in layers 4, 5, 6, 7, 9, 10 (max +0.037 in layer 5), +0.004 to +0.015
  in layers 8 and 11–15. Tuned w_A rises with k (k = 2: 0.05–0.2; k = 4: 0.1–0.4).
- At every k and layer the unique gain at B = 8 is positive with the CI excluding 0. At B = 16 the CI
  touches 0 in four cells: layers 12–14 at k = 1 and layer 14 at k = 2 (all |gain| ≤ 0.0001).

### 4. Offload simulator (eval, layers 4–15)

Misses per token per layer (of 8) and experts moved per token (sum over 12 layers); CIs in
`STEP3_tables.md`.

| predictor | C=16 misses | C=24 misses | C=32 misses | C=24 moved | C=24 MB/token |
|---|---|---|---|---|---|
| no prefetch (LRU) | 2.94 | 1.92 | 1.19 | 23.1 | 291 |
| R | 2.79 | 1.83 | 1.14 | 22.0 | 277 |
| P(1) / P(4) | 0.40 / 1.24 | 0.066 / 0.63 | 0.017 / 0.34 | 51.8 / 65.7 | 651 / 826 |
| A(1) / A(4) | 2.34 / 2.30 | 1.47 / 1.43 | 0.90 / 0.87 | 28.7 / 29.4 | 361 / 369 |
| P(1)+R / P(4)+R | 0.48 / 1.35 | 0.079 / 0.65 | 0.021 / 0.34 | 47.4 / 45.5 | 596 / 572 |
| PRw(1) / PRw(4) | 0.43 / 1.26 | 0.071 / 0.61 | 0.019 / 0.33 | 49.6 / 55.3 | 624 / 695 |
| P(1)+R+A(1) / P(4)+R+A(4) | 0.45 / 1.23 | 0.068 / 0.55 | 0.017 / 0.28 | 47.3 / 44.0 | 595 / 554 |

Relative miss reduction of P(k)+R+A(k), 1 − misses_PRA / misses_ref (95% CI in `STEP3_tables.md`):

| reference | C | k=1 | k=2 | k=3 | k=4 |
|---|---|---|---|---|---|
| P(k)+R (as in the rule) | 16 | +5.9% | +5.1% | +7.1% | +8.8% |
| P(k)+R (as in the rule) | 24 | +13.5% | +9.5% | +12.9% | +14.3% |
| P(k)+R (as in the rule) | 32 | +17.1% | +11.7% | +16.8% | +18.0% |
| best non-attention (min misses of P, P+R, PRw) | 16 | −11.6% (P) | −8.6% (P) | −4.0% (P) | +0.7% [−0.2, +1.5] (P) |
| best non-attention | 24 | −2.9% (P) | −2.1% (P) | +6.0% (PRw) | **+9.4% [8.9, 9.9]** (PRw) |
| best non-attention | 32 | +2.6% (P) | +0.8% (P) | **+10.9% [10.4, 11.5]** (PRw) | **+14.6% [14.2, 15.1]** (PRw) |

- **Prefetch trades bytes for misses.** Pre-gating prefetch at C = 24 cuts misses from 1.92 to 0.07
  per token per layer, but more than doubles the experts moved (23 → 52 per token): most of the 12
  prefetched experts per layer per token are loads of non-resident experts that go unused.
- **Recall is not the cache objective.** P(k)+R has higher recall than P(k) but *more* misses: R's
  top-ranked experts are already resident (see Method), so recency weight spends prefetch slots
  without saving misses. P(k)+R+A(k) is tuned with much less recency than P(k)+R (e.g. k = 4,
  l = 4: w_R 0.15 vs. 0.4; k = 1, l = 1: 0.02 vs. 0.2), so part of its advantage over P(k)+R is just less recency. That is why
  the review added PRw(k) and the "best non-attention" reference; all weights are tuned for recall, not
  misses.
- **Against the best non-attention reference, attention helps only at long lead and larger caches:**
  −10.9% misses at k = 3 and −14.6% at k = 4 for C = 32, −9.4% at k = 4 for C = 24, and it moves
  fewer bytes than every reference there (k = 4, C = 24: 554 MB/token vs. 572 for P+R, 695 for
  PRw, 826 for P). At k ≤ 2 it is worse than P(k) alone for C ≤ 24 and at most 2.6% better at C = 32.
- **In absolute terms the k = 4, C = 24 saving is small:** 0.058 misses per token per layer vs. PRw(4)
  (0.611 → 0.553), i.e. about 0.7 fewer expert loads per token over 12 layers (≈ 9 MB/token).

## Decision rule (REVIEW.md)
Attention is worth pursuing if, at some k ≤ 4, (a) its unique gain is ≥ +0.02 recall@8 with the CI
excluding 0, or (b) it cuts simulated misses by ≥ 10% relative to P(k)+R.

- **(a) Not met, narrowly:** the best is +0.0188 [+0.0178, +0.0198] at k = 4 (layers 4–15; also
  +0.0188 over all l ≥ 4). It is met in the code domain (+0.0238) and in 6 of 12 layers at k = 4,
  but those are post-hoc subsets. (Against PRw, the attention term is +0.021 at k = 3 and +0.033 at
  k = 4, but PRw is not recall-tuned, so for recall the rule's P(k)+R is the right reference.)
- **(b) Met as written:** −14.3% at k = 4, C = 24 and −18.0% at C = 32 vs. P(4)+R. Against P(k)+R,
  k = 1–3 also exceed 10% at C ≥ 24, but at k ≤ 2 that is an artefact of P(k)+R's recency weight:
  against P(k) alone the k ≤ 2 effect vanishes.
- **Under the stricter reading** (best non-attention reference), (b) holds only at C = 32 for k = 3
  (−10.9%) and k = 4 (−14.6%), and narrowly misses at k = 4, C = 24 (−9.4%).
- **Outcome: by the stated rule, attention is worth pursuing, but only for multi-layer-ahead
  prefetch (k ≥ 3) with caches of ≥ 24–32 of 64 experts per layer.** At k ≤ 2, pre-gating alone is as
  good or better. The effect is modest in absolute terms, and it is only partly robust to the choice
  of reference.

## Notes and caveats
- Attention at larger k is not weaker; the slight rise of A(k) from k = 1 to 4 (+0.015 @8) is
  within the range of differences between head sets of different source layers, so I read it as
  "flat", not as a real improvement.
- Pass 3 recomputes on the eval split with settings fixed from passes 1–2; no eval sequence is used in
  any selection. CIs are over 150 sequences and tight because the per-sequence differences are paired.
- The "best non-attention" reference picks, per (k, C) cell, whichever of P, P+R and PRw has the
  fewest eval misses. That favors the reference, so it is conservative for attention.
- The simulator does not model stall time or overlap of transfers with compute; at k = 4 the
  prefetch has four layers of compute to hide behind, at k = 1 only one, which favors k = 4 further.
- Runtime: 23 min for the three model passes plus the simulator.

## Review fixes
An independent subagent with fresh context audited the code, results and this file against TASK.md,
REVIEW.md and the CLAUDE.md checklist. It found no bugs:
- lead time is correct (pre-gating reads `layer_in[l−k+1]`, attention reads layer l−k, keys j < t only);
- no leakage;
- P(k)+R is tuned like P(k)+R+A(k);
- the k = 1 check compares all B values;
- an independent OrderedDict LRU reimplementation matched the simulator exactly;
- every quoted recall/gain number matched the CSVs.

Fixes:
1. **Overclaim, decision rule (b):** P(k)+R is a weakened reference for misses (its recency weight
   spends prefetch slots on resident experts, and P(k)+R+A(k) uses less recency). The pre-review text
   cited ≥ 10% at k = 1 and 3 as support.
   - Fix: added the control PRw(k) (P(k)+R at PRA(k)'s own weights, w_A = 0, computed in pass 3 with
     its ranks saved for the simulator; a GPU rerun) and a "best non-attention" reference.
   - The decision-rule section now separates "met as written" from the stricter reading and drops the
     k = 1 support. The conclusion narrowed from "k ≥ 3, C ≥ 24" to "k ≥ 3 and C = 32 robustly; k = 4,
     C = 24 narrowly misses".
2. **Omitted cell:** k = 2 at C = 32 is also ≥ 10% vs. P(k)+R. It is now in the table, and per fix 1
   it does not count as support.
3. **Overclaim:** "at k ≤ 2 no better than P(k)" is now "worse at C ≤ 24, at most 2.6% better at
   C = 32" (C = 32: +2.6% at k = 1, +0.8% at k = 2, CIs excluding 0).
4. **Pre-gating decay:** corrected to ≈ −0.07 per layer (was −0.08).
5. **CI widths:** corrected to ±0.0015–0.0045 at @8.
6. **Token range:** the simulator now counts tokens t ≥ 2, like recall (it previously counted from t = 0).
   Simulator numbers moved by ≤ 0.01.
7. **R in the simulator:** documented that R never prefetches (`prefetched` = 0 for t ≥ 2) and only
   reorders LRU recency.
8. **k = 1 check:** noted that it confirms the code paths match, not an independent replication.

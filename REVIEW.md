# Review notes

## Step 0 — approved (reviewed commit 3e43443)

Checks are correct and sufficient. Using the `mlp.experts` pre-hook `top_k_index` as ground truth is the
right call; keep it as the source of expert sets everywhere (not top-k of the bf16 logits).
Keep 1024 tokens per sequence.

Before Step 1:
- Run `python -m src.run_all` end-to-end once (data exists, so it should skip the build). For the crash on
  exit in `data.py`: if it's the streaming iterator's background threads, explicitly `del ds` and call
  `gc.collect()` per domain, or call `os._exit(0)` after saving. Note in the report which fix you used.
- Split sequences into a **head-selection split** and an **eval split**, 50/50, stratified by domain, fixed
  seed. Save the split to `results/split.json`. All "best heads" and any tuned parameter are chosen on the
  selection split and reported on the eval split only.

## Changes to Step 1

1. **Token-identity control.** Repeated tokens (identifiers in code, repeated words) get both high
   attention (duplicate/induction heads) and near-identical routing, especially in early layers, where
   routing is largely token-driven. That would make attention look predictive for a trivial reason.
   Report every within-bin correlation twice: all pairs, and pairs with `ids[t] != ids[j]`. Also report
   the fraction of same-token pairs per bin.
2. **Sink exclusion.** Run as planned (j = 0 excluded). Since position 0 is always `<|endoftext|>`,
   make "exclude j = 0" the main result and the all-pairs version secondary.
3. **Uncertainty.** Pairs within a sequence aren't independent. Compute the correlation per sequence,
   then report the mean and a bootstrap 95% CI over sequences (1000 resamples). Report the number of
   sequences and pairs per cell.
4. **Pair sampling.** Sample pairs stratified per distance bin (for example up to 2000 pairs per bin per
   sequence per layer) so short distances don't dominate. Bin 1 (d = 1) has only n − 1 pairs per sequence;
   use all of them.
5. **Per head.** Rank heads on the selection split by mean within-bin correlation (averaged over bins
   with d ≥ 2, sink excluded, distinct tokens). Report the top 10 (layer, head) with their eval-split
   values, and a heatmap of correlation per [layer, head].

## Changes to Step 2

1. **Add a recency baseline.** This is the key control for "attention vs. distance":
   score(e) = Σ_{j<t} λ^(t−j) · 1[e ∈ E_j], with λ tuned on the selection split
   (grid: 0.5, 0.7, 0.8, 0.9, 0.95, 0.99). Attention predictors must beat this to count.
2. **Pre-gating, exact definition.** Input = residual stream entering layer l (output of layer l−1,
   i.e. before layer l's attention). Apply layer l's `post_attention_layernorm` (in HF this is the
   pre-MLP norm) and layer l's `mlp.gate`. Not defined for l = 0; report it as N/A there.
3. **Combined predictor.** Combine z-scored pre-gating logits with z-scored attention scores, using the
   mixing weight tuned on the selection split. Report the gain over pre-gating alone, with a bootstrap CI.
4. **Also exclude j = 0** from attention-based scores (its expert set is a constant token's), and report
   both versions for predictor A.
5. Report recall on the eval split only, per layer and per domain, and an all-layer average excluding l = 0.

## Process
- Run on 10 sequences first, check the outputs make sense, then do the full run.
- Push after Step 1 with a short `results/STEP1.md` (headline numbers + plots) and stop, so this can be
  reviewed before Step 2.

## Step 1 — approved (reviewed commit bdadbbb)

Well done; the partial-ρ-given-d control was a good addition. Notes that change Step 2:

1. **Sink in scores.** Per your note, j = 0 carries most long-range mass. In predictors A and B, the main
   variant excludes j = 0 and renormalizes the remaining weights; report the with-sink variant secondary.
2. **Per-layer heads.** For layer l, predictor A uses the top-3 heads *of layer l* from
   `step1_best_heads.json` (selection split), averaged. Also report head-mean. In layers 11–13, where no
   head beats the mean, head-mean is expected to win; report whichever happens.
3. **d = 1 is attention's blind spot, recency's strength.** Add **Attention + recency**: z-scored
   attention scores + z-scored recency scores, mixing weight tuned on the selection split. The final
   question is then: does attention add to (a) recency and (b) pre-gating? Report gain over each with a
   bootstrap CI over sequences.
4. **Best combined predictor.** Also report pre-gating + recency + attention (two mixing weights,
   small grid on the selection split). This is the number that decides whether attention is worth
   computing.
5. **Negative heads** (e.g. 0.4): skip for now; mention in limitations.
6. **Lead-time summary table** in STEP2.md: one row per predictor with what it reads and when.

You now have autonomy: see `CLAUDE.md`. Run Step 2 and the final report without stopping.

## Final report — approved (reviewed commit 9ad8362). Step 3 follows.

Good work, especially catching that pre-gating's lead is one attention block, not a full layer.
Step 3 tests the case that matters for offloading: predicting experts several layers ahead.

## Step 3: matched lead-time comparison, k = 1..4

**Definition of lead k.** A predictor for token t's experts at layer l may use only what exists once
layer l−k has fully finished for token t (attention and MLP), plus anything from tokens j < t at any
layer. k = 1 must reproduce Step 2's pre-gating and B numbers; check this and report it.
Target layers: l ≥ k (skip undefined pairs; report averages over l = 4..15 so every k is
compared on the same layers, plus the per-k average over all defined l).

**Predictors, at each k:**
- **Pre-gating(k):** residual stream after layer l−k → layer l's `post_attention_layernorm` →
  layer l's `mlp.gate`, top-B.
- **Attention(k):** layer (l−k)'s attention for token t, sink excluded and renormalized,
  × 1[e ∈ E_j^l] over j < t. Head choice per (k, l): top-3 heads of layer l−k ranked by this
  predictor's own recall on the selection split, and head-mean; keep whichever is better on selection.
- **Recency:** unchanged (λ tuned per l); its value doesn't depend on k.
- **Combinations:** P(k)+R, P(k)+R+Attn(k), z-scored, weights tuned per (k, l) on the selection split,
  same grid as the final Step 2 version.

**Report**, eval split, B ∈ {8, 16}, 95% bootstrap CIs over sequences:
1. Recall vs. k for every predictor (table + one plot).
2. Attention's unique gain, P(k)+R+Attn(k) − P(k)+R, vs. k. This is the main result.
3. The same per domain, and per layer for k = 2 and 4.

**Offload simulator (cheap, CPU only, uses the saved predictions):** per layer, a GPU expert cache of
capacity C ∈ {16, 24, 32} experts (of 64), LRU eviction. Decode token by token. Before token t's
layer l, at lead k, prefetch the predictor's top-B (B = C/2) into the cache, evicting LRU.
Then count misses = true experts not resident, and bytes moved = prefetched + on-miss loads.
Report, per predictor and k: misses per token per layer, and total experts moved per token
(prefetch waste counts). Include "no prefetch, LRU only" as the baseline.
One OLMoE expert is 3 × 2048 × 1024 × 2 bytes ≈ 12.6 MB in bf16; report MB/token too.

**Decision rule** (write it in the report): attention is worth pursuing if, at some k ≤ 4, its unique
gain is ≥ +0.02 recall@8 with the CI excluding 0, or it cuts simulated misses by ≥ 10% relative to
P(k)+R. Otherwise close the prefetch hypothesis.

**Process:** same loop as CLAUDE.md (pilot, full run, independent reviewer, fixes, push). Write
`results/STEP3.md`, then update `REPORT.md`: add a Step 3 section, revise the verdict and the
"Suggested next experiment" accordingly. Final commit message starts with `Step 3 final:`.
Budget: stop and write BLOCKED.md if this would exceed 4 GPU-hours.

# Step 2 summary: is attention useful for expert prefetch?

**Short answer: attention clearly beats every locality baseline, but it is far below pre-gating. On top
of pre-gating + recency, it adds only +0.5 points of recall@8 (+0.2 at B = 16).** That gain is
statistically clear, positive in every layer and domain, and practically negligible.

All numbers are on the eval split (150 sequences), averaged over layers 1–15, with 95% bootstrap CIs over
sequences in brackets. Full tables are in [`STEP2_tables.md`](STEP2_tables.md). The raw numbers are in
`step2_recall.csv` (every predictor × layer × domain × B) and `step2_gains.csv` (paired differences with
CIs). Choices made on the selection split are in `step2_choice.json` and `step2_tuned_weights.json`.

## Headline (eval, all domains, average over layers 1–15)

| predictor | usable lead time | recall@8 | recall@16 |
|---|---|---|---|
| random | — | 0.125 | 0.250 |
| frequency (j < t) | before token t | 0.397 [0.383, 0.412] | 0.581 [0.566, 0.597] |
| prev-token | before token t | 0.425 [0.417, 0.434] | 0.637 [0.627, 0.650] |
| recency (λ tuned per layer) | before token t | 0.434 [0.422, 0.445] | 0.627 [0.615, 0.638] |
| A: same-layer attention, j = 0 excluded *(diagnostic)* | ≈ none | 0.503 [0.489, 0.515] | 0.708 [0.696, 0.720] |
| A, head-mean, **with** sink (secondary) | ≈ none | 0.327 [0.311, 0.341] | 0.607 [0.591, 0.622] |
| **B**: previous-layer attention, j = 0 excluded | ~1 layer | 0.522 [0.509, 0.535] | 0.721 [0.710, 0.733] |
| recency + B | ~1 layer | 0.525 [0.513, 0.538] | 0.725 [0.713, 0.737] |
| **pre-gating** | 1 attention block | **0.875 [0.873, 0.877]** | **0.976 [0.975, 0.976]** |
| pre-gating + recency | 1 attention block | 0.885 [0.884, 0.887] | 0.982 [0.982, 0.983] |
| pre-gating + B | 1 attention block | 0.888 [0.887, 0.889] | 0.983 [0.983, 0.983] |
| **pre-gating + recency + B** (best deployable) | 1 attention block | **0.890 [0.889, 0.892]** | **0.984 [0.984, 0.984]** |
| pre-gating + recency + A *(diagnostic)* | ≈ none | 0.890 [0.888, 0.891] | 0.984 [0.983, 0.984] |

A and any combination containing A only become available after layer l's attention softmax. At that
point just o_proj, the residual add, a norm and a 2048×64 gate stand between it and the exact router.
They are therefore reported as diagnostics of information content, not as usable prefetchers.

## Does attention add information? (paired gains, eval, average over layers 1–15)

| question | gain@8 | gain@16 |
|---|---|---|
| attention alone vs. recency (A − R) | +0.069 [+0.066, +0.071] | +0.081 [+0.079, +0.084] |
| (a) attention adds to recency (R+A − R) | +0.073 [+0.071, +0.076] | +0.085 [+0.082, +0.088] |
| (b) prev-layer attention adds to pre-gating (P+B − P) | +0.013 [+0.012, +0.014] | +0.007 [+0.007, +0.008] |
| recency adds to pre-gating (P+R − P) | +0.010 [+0.009, +0.011] | +0.007 [+0.006, +0.007] |
| **prev-layer attention adds to pre-gating + recency (P+R+B − P+R)** | **+0.0049 [+0.0046, +0.0052]** | **+0.0018 [+0.0017, +0.0020]** |
| same-layer attention adds to P+R (P+R+A − P+R) | +0.0043 [+0.0041, +0.0045] | +0.0015 [+0.0014, +0.0016] |
| attention vs. pre-gating (A − P) | −0.373 [−0.388, −0.357] | −0.268 [−0.280, −0.255] |
| prev-layer vs. same-layer attention alone (B − A) | +0.020 [+0.019, +0.021] | +0.014 [+0.013, +0.014] |

- **Attention vs. distance:** attention carries routing information beyond locality. It beats tuned
  recency by ~7 points of recall@8 in every domain (web +0.062, code +0.077, math +0.068), which
  confirms Step 1. Adding recency on top of attention adds little (R+A vs. A: +0.004).
- **Attention vs. pre-gating:** pre-gating dominates (0.875 vs. ~0.5 at B = 8). At B = 16 pre-gating
  already reaches 0.976, and ≥ 0.994 from layer 8 on.
  - Previous-layer attention and recency each add ~1 point to pre-gating. They overlap, so attention's
    unique contribution on top of P+R is +0.5 points at B = 8 and +0.2 at B = 16.
  - By domain (P+R+B − P+R, @8): code +0.0069, math +0.0044, web +0.0034.
- **Where the gain is:** the gain is largest in early layers, where pre-gating is weakest (layers 1, 2
  and 4 at 0.67–0.82). There P+R+B − P+R is +0.010 to +0.014 at B = 8. In layers 12–14 it is
  +0.0003 to +0.0012. With the refined weight grid every layer is positive, and the lower CI bound is
  above 0 everywhere.
- **Same vs. previous layer:** A's shorter lead buys nothing. B is as good as or better than A in
  every comparison, alone (+0.020) and in combination. **Caveat:** B's 3 heads per layer are ranked by
  B's own selection-split recall (review fix 3). A's heads use the Step 1 correlation ranking, as
  REVIEW.md specifies. Part of B's edge may come from that difference in head selection.

## Ablations (eval)
- **Sink:** excluding j = 0 and renormalizing is essential. For head-mean A it is worth
  +0.148 [+0.144, +0.152] recall@8. With the sink included, A falls below recency.
- **Heads:**
  - A with the top-3 Step 1 heads beats A with the head-mean by +0.027 [+0.026, +0.028] at B = 8.
  - B with its 3 recall-ranked heads beats B with the head-mean by +0.050 (0.522 vs. 0.472).
  - Selection picked head-mean for A only in layers 0, 1 and 3.
  - In layers 11–13, which REVIEW.md expected to favour head-mean, selection also picked top-3, and
    top-3 is better on eval there too (+0.013 to +0.020).
- **Recency λ (selection):** 0.7 in layers 4–9 and 11; 0.8 in layer 2; 0.9 in layers 3, 12 and 13;
  0.95 in layers 0, 1, 10, 14 and 15.
- **Mixing weights (selection):** the attention weight in P+R+B is small, 0.02–0.2. It is larger in
  early layers (0.1–0.2 in layers 1, 2, 4 and 5; 0.05 in layer 3) and 0.02–0.075 in layers 8–15. The
  recency weight is 0.01–0.15.

## Lead-time summary
| predictor | reads (for token t, layer l) | available when |
|---|---|---|
| prev-token, frequency, recency | E_j^l for j < t (earlier tokens' layer-l routing) | before token t's forward starts |
| pre-gating | residual entering layer l (output of layer l−1) → layer l `post_attention_layernorm` → layer l `mlp.gate` | after layer l−1 finishes; hides layer l's attention block |
| A (same layer) | a_l(t, j), j < t (no diagonal, j = 0 dropped, renormalized) × E_j^l | after layer l's attention softmax; ≈ no lead |
| B (prev layer) | a_{l−1}(t, j), j < t, 3 heads of layer l−1 (j = 0 dropped, renormalized) × E_j^l | after layer l−1's attention; hides layer l−1's MoE + layer l's attention |
| P+B, P+R, P+R+B | z-scored sum | limited by pre-gating: after layer l−1 |
| R+A, P+A, P+R+A | z-scored sum | limited by A: ≈ no lead |

Pre-gating and B are not defined at layer 0. They are reported as N/A there, and layer 0 is excluded from
all averages.

## Method
- **Recall:** recall = |top-B(score) ∩ E_t^l| / 8, averaged over tokens t ∈ [2, 1024), then over
  sequences.
- **Ranking:** lexicographic, by score first and then by in-sequence expert frequency over j < t, used
  only among exactly equal scores (two stable sorts).
  - The tie-break applies to every predictor except the oracle, including pre-gating, where bf16
    logits tie at the 8/9 boundary for ~2% of tokens. That information is available at prediction
    time.
  - Prev-token is E_{t−1}^l, padded to B by this same frequency tie-break.
- **No predictor reads token t's own layer-l routing.**
- **Tuned per layer on the selection split only** (objective: mean recall over B ∈ {8, 12, 16, 24, 32}):
  - the recency λ (grid from REVIEW.md);
  - A's head set (Step 1 top-3 vs. head-mean);
  - B's 3 heads of layer l−1, ranked by per-head recall of layer-l routing;
  - the mixing weights.
- **Weight grid:** per weight, {0, .01, .02, .03, .05, .075, .1, .15, .2, .3, …, 1}. Pairs are
  (1−w)·z_X + w·z_Y. Triples are (w_P, w_R, w_X) on the simplex built from that grid. "Pre-gating +
  recency" is the w_X = 0 subset of the same grid, tuned the same way, so P+R+X vs. P+R isolates the
  attention term.
- **z-scores** are per token, over the 64 experts.
- **Two passes:** pass 1 runs on the selection split to choose λ and heads. Pass 2 runs all 300
  sequences, and the mixing weights are chosen on its selection rows.

## Sanity checks
- **Pre-gating wiring:** `gate(post_attention_layernorm(post-attention residual))` reproduces the
  router's expert set in all 16 layers. Pre-gating then feeds the *pre*-attention residual through the
  same path.
- **Oracle:** predicting from the router's own logits gives recall 1.000 for every B.
  - An earlier version took the top-32 once and used its prefixes. That gave 0.997, because `topk`
    orders bf16 ties differently for different k.
  - The oracle now uses one plain `topk` per B.
- **Tie-break:** recency with λ = 0.5 equals prev-token at B = 8 on synthetic data. This identity must
  hold, and it failed under the old epsilon tie-break.
- **Recall of 1.0:** for B ≤ 16, nothing except the oracle reaches 1.0. The highest is pre-gating in
  layers 12–14 (0.94–0.95 at B = 8, ≥ 0.999 at B = 16). At B = 24/32 (a third to half of all experts),
  pre-gating and its combinations reach 1.000 in deep layers. That is expected for a strong predictor
  and does not indicate leakage.
- **Pilots:** 12-sequence pilots ran before each full run.

## Review fixes
An independent subagent audited the pre-review version (commit 8b547b5) against TASK.md, REVIEW.md and
the CLAUDE.md checklist. It found no leakage and no lead-time violations; it confirmed this by
recomputing the choices from the selection rows and the headline numbers from the raw per-sequence
recalls. The fixes:

1. **Tie-break epsilon was not below all real score gaps (bug, minor).**
   - The old `1e-12 × freq` could exceed genuine recency/attention gaps (λ^d < 1e-9 at long range). That
     mixed some frequency signal into near-ties, which the docs did not admit.
   - Replaced with a true lexicographic tie-break (`recall_at`), checked on the λ = 0.5 / prev-token
     identity.
   - Effect: changes of ~1e-4 to 4e-4. Prev-token is now pure E_{t−1} + frequency padding.
2. **Mixing-weight grid too coarse near the optimum (methodological).**
   - The old 0.1-step grid put the tuned attention weight on its first point or at 0. That produced
     exact-zero gains in layers 10 and 12–14 and negative transfer in layers 3 and 9.
   - The refined grid (dense near 0) fixes this: every layer now has a positive gain.
   - Headline P+R+A − P+R moved from +0.0036 to +0.0043 @8. P+R itself moved from 0.883 to 0.885.
3. **B's heads were ranked for the wrong target (methodological).**
   - B used the Step 1 top-3 heads of layer l−1, which were ranked on layer l−1 overlap, not layer l
     routing.
   - B's heads are now ranked by B's own per-head recall on the selection split (pass 1;
     `step2_choice.json`).
   - B alone went from 0.497 to 0.522 @8, and P+R+B − P+R from +0.0031 to +0.0049. B's head-set choice
     (top-3 vs. mean) was dropped: B always uses its 3 recall-ranked heads, and head-mean B is reported
     separately.
4. **A has ≈ no lead time (framing).** The headline best is now pre-gating + recency + B. A and its
   combinations are marked diagnostic.
5. **Misreports in the text** (the λ list, the layers where the gain was 0, the layer 11–13 head range,
   "≥ 0.996" at B = 16, and the "recall < 1" wording for large B) were rewritten from the new results.
6. **Nits:**
   - Layer 0 is now `null` in `step2_tuned_weights.json`.
   - The redundant pair `pregate+recency` was dropped; only the nested `pregate+recency(grid3)` remains.
   - `top3_heads()` asserts the Step 1 selection-split criterion.

## Plots
![recall vs B](step2_recall_vs_B.png)
![per layer](step2_recall_per_layer.png)

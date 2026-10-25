# Step 2 summary: is attention useful for expert prefetch?

**Short answer: attention beats every locality baseline by a clear margin, but it is far below
pre-gating. On top of pre-gating + recency it adds only +0.3–0.4 points of recall@8.** The gain is
statistically clear but practically negligible.

Numbers are on the eval split (150 sequences), averaged over layers 1–15. Brackets are 95% bootstrap CIs
over sequences. Full tables are in [`STEP2_tables.md`](STEP2_tables.md); the raw numbers are in
`step2_recall.csv` (every predictor × layer × domain × B) and `step2_gains.csv` (paired differences with
CIs). Choices made on the selection split are in `step2_choice.json` and `step2_tuned_weights.json`.

## Headline (eval, all domains, average over layers 1–15)

| predictor | lead time | recall@8 | recall@16 |
|---|---|---|---|
| random | — | 0.125 | 0.250 |
| frequency (j < t) | before token t | 0.397 [0.383, 0.412] | 0.581 [0.566, 0.597] |
| prev-token | before token t | 0.425 [0.417, 0.434] | 0.637 [0.627, 0.650] |
| recency (λ tuned per layer) | before token t | 0.434 [0.422, 0.445] | 0.627 [0.615, 0.638] |
| **A**: same-layer attention, top-3 or mean heads, j = 0 excluded | same layer | 0.503 [0.489, 0.515] | 0.708 [0.696, 0.720] |
| A, head-mean, **with** sink (secondary) | same layer | 0.327 [0.311, 0.341] | 0.607 [0.591, 0.622] |
| **B**: previous-layer attention, j = 0 excluded | ~1 layer | 0.497 [0.484, 0.510] | 0.702 [0.690, 0.713] |
| recency + A | same layer | 0.507 [0.495, 0.520] | 0.712 [0.700, 0.723] |
| **pre-gating** | 1 layer's attention | **0.875 [0.873, 0.877]** | **0.976 [0.975, 0.976]** |
| pre-gating + A | same layer | 0.886 [0.884, 0.887] | 0.982 [0.982, 0.983] |
| pre-gating + B | 1 layer's attention | 0.886 [0.884, 0.887] | 0.982 [0.982, 0.983] |
| pre-gating + recency | 1 layer's attention | 0.883 [0.882, 0.885] | 0.982 [0.981, 0.982] |
| **pre-gating + recency + A** (best) | same layer | **0.887 [0.886, 0.888]** | **0.983 [0.983, 0.984]** |
| pre-gating + recency + B | 1 layer's attention | 0.887 [0.885, 0.888] | 0.983 [0.983, 0.983] |

## Does attention add information? (paired gains, eval, avg layers 1–15)

| question | gain@8 | gain@16 |
|---|---|---|
| attention alone vs. recency (A − R) | +0.069 [+0.066, +0.071] | +0.081 [+0.078, +0.084] |
| (a) attention adds to recency (R+A − R) | +0.073 [+0.071, +0.076] | +0.085 [+0.083, +0.087] |
| (b) attention adds to pre-gating (P+A − P) | +0.010 [+0.009, +0.011] | +0.007 [+0.006, +0.007] |
| recency adds to pre-gating (P+R − P) | +0.008 [+0.008, +0.009] | +0.006 [+0.006, +0.007] |
| **attention adds to pre-gating + recency (P+R+A − P+R)** | **+0.0036 [+0.0034, +0.0040]** | **+0.0016 [+0.0014, +0.0017]** |
| same, prev-layer attention (P+R+B − P+R) | +0.0031 [+0.0029, +0.0034] | +0.0012 [+0.0011, +0.0013] |
| attention vs. pre-gating (A − P) | −0.373 [−0.387, −0.358] | −0.268 [−0.280, −0.255] |

- **Attention vs. distance:** attention does carry routing information beyond locality. It beats tuned
  recency by about 7 points of recall@8 in every domain (+0.062 web, +0.077 code, +0.068 math).
  Adding recency on top of attention gains little (R+A vs. A: +0.004). That fits the Step 1 picture:
  attention mostly covers the short-range information too, except at d = 1.
- **Attention vs. pre-gating:** pre-gating is far stronger, at 0.875 vs. 0.50 recall@8. Recall@16 is
  0.976, and from layer 8 on it is ≥ 0.994, so B = 16 already covers almost every true expert.
  Attention and recency each add about 1 point to pre-gating at B = 8. They largely overlap with each
  other, so attention's unique contribution on top of P+R is 0.4 points at B = 8 and 0.2 at B = 16.
- **Where the gain is:** the gain from attention comes mostly from early layers (1, 2, 4: +0.009 to
  +0.013 over P+R at B = 8), where pre-gating is weakest (0.67–0.82). In layers 12–14 the tuned weight
  on attention is 0 and the gain is exactly 0. Code gains the most (+0.006 over P+R), web the least
  (+0.002).
- **Lead time:** the previous-layer variant (B) does almost as well as the same-layer one (A) in every
  combination (0.497 vs. 0.503 alone, +0.0031 vs. +0.0036 over P+R). So the shorter lead of A buys
  nothing.

## Ablations (eval)
- **Sink:** excluding j = 0 and renormalizing is essential. For the head-mean variant of A it is worth
  +0.148 [+0.143, +0.152] recall@8; with the sink included, A falls below recency.
- **Heads:** the top-3 heads per layer (ranked on the selection split in Step 1) beat the head-mean by
  +0.027 [+0.026, +0.028] at B = 8. Selection picked head-mean for A only in layers 0, 1 and 3. Layers 11–13,
  which REVIEW.md expected to favour head-mean, also picked top-3, and top-3 is better on eval there
  too (+0.014 to +0.020).
- **Recency λ:** the tuned λ is 0.7–0.95, varying by layer. Short memory (0.7) wins in layers 4–8 and
  11, 0.9 in layers 3, 9, 12, 13 and 15, and long memory (0.95) in layers 0, 1, 10 and 14.

## Lead-time summary
| predictor | reads (for token t, layer l) | available when |
|---|---|---|
| prev-token, frequency, recency | E_j^l for j < t (earlier tokens' layer-l routing) | before token t's forward starts |
| pre-gating | residual entering layer l (output of layer l−1) → layer l `post_attention_layernorm` → layer l `mlp.gate` | after layer l−1 finishes; hides layer l's attention |
| A (same layer) | a_l(t, j), j < t (no diagonal, j = 0 dropped, renormalized) × E_j^l | after layer l's attention softmax; only o_proj + residual + norm remain before the router |
| B (prev layer) | a_{l−1}(t, j), j < t × E_j^l | after layer l−1's attention; hides layer l−1's MoE + layer l's attention |
| combinations | z-scored sum of the above | the latest of the parts (P+A: same layer; P+B, P+R: after layer l−1) |

Not defined for layer 0: pre-gating and B. They are reported as N/A, and layer 0 is excluded from all
averages.

## Method
- **Recall:** recall = |top-B(score) ∩ E_t^l| / 8, averaged over tokens t ∈ [2, 1024), then over
  sequences. Ties in history-based scores are broken by in-sequence frequency over j < t, using an
  epsilon below any real score gap.
- **No predictor reads token t's own layer-l routing.** Prev-token pads with frequency over j < t, not
  the whole sequence.
- **Tuned per layer on the selection split only** (objective: mean recall over B ∈ {8, 12, 16, 24, 32}):
  - the recency λ (grid from REVIEW.md);
  - top-3 heads vs. head-mean, for A and for B (B uses layer l−1's top-3 heads from Step 1);
  - the mixing weights. Pair weights use a grid with step 0.1. The three-way weights (w_P, w_R, w_X)
    use the simplex with step 0.1. "pre-gating + recency" is the w_X = 0 subset of that grid, tuned the
    same way, so P+R+X vs. P+R isolates the attention term.
- **Combinations** z-score each score over the 64 experts, per token.
- **Two passes:** pass 1 runs on the selection split to choose λ and heads. Pass 2 runs all 300
  sequences, and the mixing weights are then chosen on the selection rows of pass 2.

## Sanity checks
- **Pre-gating wiring:** `gate(post_attention_layernorm(post-attention residual))` reproduces the
  router's expert set in all 16 layers. Pre-gating then feeds the *pre*-attention residual through the
  same path.
- **Oracle:** predicting from the router's own logits gives recall 1.000 for every B.
  - The first full run gave 0.997. That came from taking the top-32 once and using its first B entries.
    bf16 router logits tie at the 8/9 boundary for 1.5–2.6% of tokens, and `topk` orders ties
    differently for different k.
  - Recall now uses a separate `topk` per B. Only the oracle row moved; every other predictor's value
    is identical to 3 decimals between the two runs.
- **Recall stays below 1.0:** apart from the oracle, the highest value is pre-gating in deep layers
  (layers 12–14: 0.94–0.95 at B = 8, ≥ 0.999 at B = 16). This is plausible: the residual barely changes across one
  attention block that late in the network.
- **Layer 3 anomaly:** the selection-tuned P+R+A is −0.003 below P+R on eval. The selection objective
  averages over B, and the triple grid is coarser near the optimum. This is a small tuning-transfer
  loss, not leakage.
- **Pilot:** a 12-sequence pilot ran before the full run, with consistent results.

## Plots
![recall vs B](step2_recall_vs_B.png)
![per layer](step2_recall_per_layer.png)

"""Step 2: can attention prefetch experts? Recall of the true top-8 by a predicted set of B experts.

For token t at layer l every predictor scores the 64 experts; the prediction is the top-B.
recall = |top-B ∩ E_t^l| / 8, averaged over tokens t in [T_MIN, n). Tokens' own layer-l routing is never
used for t: history-based scores only read j < t (the attention diagonal j = t is dropped).

Pass 1 (selection split): standalone predictors, to pick per layer the recency λ and the head set
(top-3 vs. head-mean) of the attention predictors used in the combinations.
Pass 2 (all sequences): standalone predictors plus z-scored combinations over a weight grid.
Mixing weights are chosen per layer on the selection split; every reported number is eval split.
"""

import argparse
import itertools
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from . import data
from .common import RESULTS, SEED, capture, load_model, seed_all

BS = [8, 12, 16, 24, 32]
LAMBDAS = [0.5, 0.7, 0.8, 0.9, 0.95, 0.99]
T_MIN = 2  # t = 0 has no history; t = 1 has only the sink j = 0
N_BOOT = 1000
RAW = RESULTS / "raw" / "step2_perseq.npz"
W2 = np.round(np.linspace(0, 1, 11), 2)  # pair weight on the second component
W3 = np.array([(a, b, round(1 - a - b, 2)) for a in W2 for b in W2 if a + b <= 1 + 1e-9])  # (wP, wR, wX)

# Standalone predictors, in storage order. A = attention of layer l, B = attention of layer l-1.
ATTN_VARIANTS = [f"{p}_{h}_{s}" for p in "AB" for h in ("top3", "mean") for s in ("nosink", "sink")]
FIXED = ["oracle_router", "prev_token", "frequency", "pregate", *[f"recency_{x}" for x in LAMBDAS],
         *ATTN_VARIANTS]
PAIRS = [("recency", "A"), ("pregate", "A"), ("recency", "B"), ("pregate", "B"), ("pregate", "recency")]
TRIPLES = ["A", "B"]  # pregate + recency + {A, B}
COMBO_KEYS = [f"{a}+{b}" for a, b in PAIRS] + [f"pregate+recency+{x}" for x in TRIPLES]


def top3_heads():
    h = pd.read_csv(RESULTS / "step1_heads.csv")
    h = h[h["head"] != "mean"].astype({"head": int})
    return {l: g.sort_values("sel_score", ascending=False)["head"].head(3).tolist()
            for l, g in h.groupby("layer")}


def zscore(s):
    return (s - s.mean(-1, keepdim=True)) / (s.std(-1, keepdim=True) + 1e-12)


def recall_at(scores, oh, tie):
    """scores [..., n, E], oh [n, E] bool -> recall [..., len(BS)] averaged over t >= T_MIN."""
    sc = scores[..., T_MIN:, :] + tie[T_MIN:]
    tgt = oh[T_MIN:].expand(*scores.shape[:-2], -1, -1)
    k = oh[0].sum()  # top-k size (8)
    # One topk per B (not prefixes of a top-32): topk orders tied values differently for different k,
    # and bf16 router logits tie at the 8/9 boundary for ~2% of tokens.
    return torch.stack([tgt.gather(-1, torch.topk(sc, b, dim=-1).indices).double().sum(-1).mean(-1)
                        for b in BS], -1) / k


def attn_scores(W, oh, sink):
    """Σ_{j<t} W[t, j] 1[e ∈ E_j], strictly-past keys only, optionally without j = 0, renormalized."""
    W = torch.tril(W, diagonal=-1)
    if not sink:
        W = W.clone()
        W[:, 0] = 0
    W = W / W.sum(-1, keepdim=True).clamp(min=1e-30)
    return W @ oh.double()


def per_sequence(model, ids_row, heads, choice=None):
    n = len(ids_row)
    L = model.config.num_hidden_layers
    E = model.config.num_experts
    hidx = {l: torch.tensor(heads[l], device="cuda") for l in range(L)}

    def reduce(l, w):
        w = w[0].double()
        return {"mean": w.mean(0), "top3": w[hidx[l]].mean(0)}

    x = torch.tensor(ids_row[None], dtype=torch.long, device="cuda")
    with torch.no_grad(), capture(model, attn_fn=reduce, keep_layer_input=True) as rec:
        o = model(x, output_router_logits=True)
        pregate = {}
        for l in range(1, L):
            layer = model.model.layers[l]
            h = layer.post_attention_layernorm(rec["layer_in"][l][0])
            pregate[l] = layer.mlp.gate(h)[0].double()  # router logits [n, E]
    oh = torch.zeros(L, n, E, dtype=torch.bool, device="cuda")
    for l in range(L):
        oh[l].scatter_(1, rec["experts"][l], True)

    tri = torch.tril(torch.ones(n, n, dtype=torch.float64, device="cuda"), diagonal=-1)
    d = torch.arange(n, device="cuda", dtype=torch.float64)
    dist = d[:, None] - d[None, :]
    Wrec = {lam: torch.where(dist > 0, lam ** dist.clamp(min=0), 0.0) for lam in LAMBDAS}
    fixed = np.full((L, len(FIXED), len(BS)), np.nan)
    combos = {}
    for l in range(L):
        ohl = oh[l]
        freq = tri @ ohl.double()  # counts over j < t
        tie = 1e-12 * freq  # tie-break by in-sequence frequency; below any real score gap
        s = {
            "oracle_router": o.router_logits[l].double(),
            "prev_token": torch.cat([torch.zeros(1, E, device="cuda", dtype=torch.float64),
                                     ohl[:-1].double()]) + 1e-3 * freq / freq.sum(-1, keepdim=True).clamp(min=1),
            "frequency": freq,
        }
        if l > 0:
            s["pregate"] = pregate[l]
        rec_s = {}
        for lam in LAMBDAS:
            rec_s[lam] = Wrec[lam] @ ohl.double()
            s[f"recency_{lam}"] = rec_s[lam]
        att = {}
        for v in ATTN_VARIANTS:
            src, hs, sk = v.split("_")
            la = l if src == "A" else l - 1
            if la < 0:
                continue
            att[v] = attn_scores(rec["attn"][la][hs], ohl, sk == "sink")
            s[v] = att[v]
        for i, k in enumerate(FIXED):
            if k in s:
                # oracle: no tie-break, so bf16-tied logits resolve as in the router's own topk
                fixed[l, i] = recall_at(s[k], ohl, 0 * tie if k == "oracle_router" else tie).cpu().numpy()

        if choice is None:
            continue
        c = choice[str(l)]
        z = {"recency": zscore(rec_s[c["lambda"]]), "A": zscore(att[f"A_{c['A_heads']}_nosink"])}
        if l > 0:
            z["pregate"] = zscore(pregate[l])
            z["B"] = zscore(att[f"B_{c['B_heads']}_nosink"])
        w2 = torch.tensor(W2, device="cuda", dtype=torch.float64)[:, None, None]
        for a, b in PAIRS:
            if a in z and b in z:
                combos[(f"{a}+{b}", l)] = recall_at((1 - w2) * z[a] + w2 * z[b], ohl, tie).cpu().numpy()
        w3 = torch.tensor(W3, device="cuda", dtype=torch.float64)[:, :, None, None]
        for xk in TRIPLES:
            if l > 0:
                sc = w3[:, 0] * z["pregate"] + w3[:, 1] * z["recency"] + w3[:, 2] * z[xk]
                combos[(f"pregate+recency+{xk}", l)] = recall_at(sc, ohl, tie).cpu().numpy()
    out = {"fixed": fixed}
    if choice is not None:
        for k in COMBO_KEYS:
            G = len(W2) if k.count("+") == 1 else len(W3)
            out[k] = np.full((L, G, len(BS)), np.nan)
            for l in range(L):
                if (k, l) in combos:
                    out[k][l] = combos[(k, l)]
    return out


def run_pass(model, seqs, ids, heads, choice=None):
    outs = []
    for c, i in enumerate(seqs):
        outs.append(per_sequence(model, ids[i], heads, choice))
        if c % 25 == 0:
            print(f"  seq {c + 1}/{len(seqs)}", flush=True)
    return {k: np.stack([o[k] for o in outs]) for k in outs[0]}


def choose(fixed_sel, L):
    """Per layer: λ and head sets maximizing mean recall over B on the selection split."""
    m = np.nanmean(fixed_sel, 0).mean(-1)  # [L, NP]
    ix = {k: i for i, k in enumerate(FIXED)}
    choice = {}
    for l in range(L):
        lam = max(LAMBDAS, key=lambda x: m[l, ix[f"recency_{x}"]])
        a = max(["top3", "mean"], key=lambda h: m[l, ix[f"A_{h}_nosink"]])
        b = max(["top3", "mean"], key=lambda h: m[l, ix[f"B_{h}_nosink"]]) if l > 0 else None
        choice[str(l)] = {"lambda": lam, "A_heads": a, "B_heads": b}
    return choice


def collect(limit=None):
    seed_all()
    ids, domains = data.load(1024)
    split = data.load_split()
    sel, ev = split["selection"], split["eval"]
    if limit:
        k = max(1, limit // 6)
        pick = lambda part: [i for d in data.DOMAINS for i in [s for s in part if domains[s] == d][:k]]
        sel, ev = pick(sel), pick(ev)
    heads = top3_heads()
    model = load_model()
    L = model.config.num_hidden_layers
    check_pregate_wiring(model, ids[sel[0]])
    print(f"pass 1: {len(sel)} selection seqs")
    p1 = run_pass(model, sel, ids, heads)
    choice = choose(p1["fixed"], L)
    seqs = sorted(sel + ev)
    print(f"pass 2: {len(seqs)} seqs")
    raw = run_pass(model, seqs, ids, heads, choice)
    raw["seq"] = np.array(seqs)
    raw["domain"] = domains[seqs]
    raw["is_sel"] = np.isin(seqs, sel)
    RAW.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(RAW, **raw)
    (RESULTS / "step2_choice.json").write_text(json.dumps(
        {"criterion": "selection split, mean recall over B", "top3_heads": heads, "per_layer": choice}, indent=2))
    del model
    torch.cuda.empty_cache()
    return raw


def check_pregate_wiring(model, ids_row):
    """Applying layer l's post-attention norm + gate to the post-attention residual must reproduce the
    router's expert set; this validates the pre-gating code path (which feeds it the pre-attention one)."""
    x = torch.tensor(ids_row[None], dtype=torch.long, device="cuda")
    resid = {}
    hooks = [layer.post_attention_layernorm.register_forward_pre_hook(
        lambda m, a, l=l: resid.__setitem__(l, a[0].detach())) for l, layer in enumerate(model.model.layers)]
    try:
        with torch.no_grad(), capture(model) as rec:
            model(x)
            for l, layer in enumerate(model.model.layers):
                idx = layer.mlp.gate(layer.post_attention_layernorm(resid[l][0]))[2]
                same = (idx.sort(-1).values == rec["experts"][l].sort(-1).values).all()
                assert same, f"pre-gating wiring mismatch at layer {l}"
    finally:
        for h in hooks:
            h.remove()
    print("pre-gating wiring check: OK (gate(norm(post-attn residual)) == router, all layers)")


def boot(vals, rng):
    """vals [S, ...] -> mean, lo, hi over a bootstrap of sequences."""
    S = vals.shape[0]
    idx = rng.integers(0, S, (N_BOOT, S))
    bm = np.nanmean(vals[idx], 1)
    lo, hi = np.nanpercentile(bm, [2.5, 97.5], 0)
    return np.nanmean(vals, 0), lo, hi


def aggregate(raw=None):
    import warnings

    warnings.filterwarnings("ignore", category=RuntimeWarning)
    raw = raw if raw is not None else dict(np.load(RAW))
    choice = json.loads((RESULTS / "step2_choice.json").read_text())["per_layer"]
    sel = raw["is_sel"]
    ev = ~sel
    S, L = raw["fixed"].shape[:2]
    rng = np.random.default_rng(SEED)

    # Per-seq recall [S, L, B] for every reported predictor
    preds, tuned = {}, {}
    for i, k in enumerate(FIXED):
        preds[k] = raw["fixed"][:, :, i]
    ix = {k: i for i, k in enumerate(FIXED)}
    preds["recency"] = np.stack([raw["fixed"][:, l, ix[f"recency_{choice[str(l)]['lambda']}"]] for l in range(L)], 1)
    preds["A_best"] = np.stack([raw["fixed"][:, l, ix[f"A_{choice[str(l)]['A_heads']}_nosink"]] for l in range(L)], 1)
    preds["B_best"] = np.stack([raw["fixed"][:, l, ix[f"B_{choice[str(l)]['B_heads']}_nosink"]]
                                if l > 0 else np.full((S, len(BS)), np.nan) for l in range(L)], 1)
    for k in COMBO_KEYS:
        arr = raw[k]  # [S, L, G, B]
        obj = np.nanmean(arr[sel], 0).mean(-1)  # [L, G]
        best = [int(np.nanargmax(obj[l])) if np.isfinite(obj[l]).any() else 0 for l in range(L)]
        preds[k] = np.stack([arr[:, l, best[l]] for l in range(L)], 1)
        grid = W2 if k.count("+") == 1 else W3
        tuned[k] = {l: (grid[best[l]].tolist() if np.isfinite(obj[l]).any() else None) for l in range(L)}
    # pregate+recency (no attention) as the triple-grid baseline: wX = 0 subset, tuned the same way
    arr = raw["pregate+recency+A"]
    noatt = np.where(W3[:, 2] == 0)[0]
    obj = np.nanmean(arr[sel][:, :, noatt], 0).mean(-1)
    best = [int(noatt[np.nanargmax(obj[l])]) if np.isfinite(obj[l]).any() else 0 for l in range(L)]
    preds["pregate+recency(grid3)"] = np.stack([arr[:, l, best[l]] for l in range(L)], 1)
    tuned["pregate+recency(grid3)"] = {l: W3[best[l]].tolist() for l in range(L)}

    # --- recall table: eval split, per domain, per layer + avg over l >= 1
    rows = []
    for dom in ["all", *data.DOMAINS]:
        m = ev & ((raw["domain"] == dom) if dom != "all" else True)
        for k, v in preds.items():
            vv = v[m]  # [s, L, B]
            mean, lo, hi = boot(vv, rng)
            avg = np.nanmean(vv[:, 1:], 1)  # per-seq mean over layers 1..15
            am, alo, ahi = boot(avg, rng)
            for bi, B in enumerate(BS):
                for l in range(L):
                    rows.append(dict(domain=dom, predictor=k, layer=str(l), B=B, recall=mean[l, bi],
                                     ci_lo=lo[l, bi], ci_hi=hi[l, bi], n_seq=int(m.sum())))
                rows.append(dict(domain=dom, predictor=k, layer="avg1-15", B=B, recall=am[bi],
                                 ci_lo=alo[bi], ci_hi=ahi[bi], n_seq=int(m.sum())))
    rec = pd.DataFrame(rows)
    rec.to_csv(RESULTS / "step2_recall.csv", index=False, float_format="%.4f")

    # --- gains (eval, per-seq paired differences, bootstrap CI)
    comparisons = [
        ("A_best", "recency", "attention alone vs recency"),
        ("recency+A", "recency", "attention adds to recency"),
        ("pregate+A", "pregate", "attention (same layer) adds to pre-gating"),
        ("pregate+B", "pregate", "attention (prev layer) adds to pre-gating"),
        ("pregate+recency+A", "pregate+recency(grid3)", "attention (same layer) adds to pre-gating+recency"),
        ("pregate+recency+B", "pregate+recency(grid3)", "attention (prev layer) adds to pre-gating+recency"),
        ("A_best", "pregate", "attention (same layer) vs pre-gating"),
        ("pregate+recency(grid3)", "pregate", "recency adds to pre-gating"),
        ("A_top3_nosink", "A_mean_nosink", "top-3 heads vs head-mean (A)"),
        ("A_mean_nosink", "A_mean_sink", "sink excluded vs included (A, head-mean)"),
    ]
    grows = []
    for a, b, desc in comparisons:
        for dom in ["all", *data.DOMAINS]:
            m = ev & ((raw["domain"] == dom) if dom != "all" else True)
            diff = preds[a][m] - preds[b][m]  # [s, L, B]
            mean, lo, hi = boot(diff, rng)
            am, alo, ahi = boot(np.nanmean(diff[:, 1:], 1), rng)
            for bi, B in enumerate(BS):
                if B not in (8, 16):
                    continue
                for l in range(L):
                    grows.append(dict(comparison=desc, a=a, b=b, domain=dom, layer=str(l), B=B,
                                      gain=mean[l, bi], ci_lo=lo[l, bi], ci_hi=hi[l, bi]))
                grows.append(dict(comparison=desc, a=a, b=b, domain=dom, layer="avg1-15", B=B,
                                  gain=am[bi], ci_lo=alo[bi], ci_hi=ahi[bi]))
    gains = pd.DataFrame(grows)
    gains.to_csv(RESULTS / "step2_gains.csv", index=False, float_format="%.4f")
    (RESULTS / "step2_tuned_weights.json").write_text(json.dumps(
        {"note": "per layer, selection split, max mean recall over B; pairs: weight on 2nd component; "
                 "triples: (w_pregate, w_recency, w_attn)", "weights": tuned}, indent=2))
    plots(rec, L)
    write_md(rec, gains, choice, tuned, L, int(ev.sum()), int(sel.sum()))
    return rec, gains


MAIN = ["oracle_router", "prev_token", "frequency", "recency", "pregate", "A_mean_nosink", "A_top3_nosink",
        "A_mean_sink", "B_best", "recency+A", "pregate+A", "pregate+recency(grid3)", "pregate+recency+A",
        "pregate+recency+B"]


def plots(rec, L):
    r = rec[(rec.domain == "all")]
    fig, axes = plt.subplots(4, 4, figsize=(20, 16), sharex=True)
    for l, ax in enumerate(axes.flat):
        for i, k in enumerate(MAIN):
            if k == "oracle_router":
                continue
            v = r[(r.predictor == k) & (r.layer == l)].sort_values("B")
            if v.recall.isna().all():
                continue
            ls = "-" if ("+" in k) else ("--" if k.startswith(("A_", "B_")) else ":")
            ax.plot(v.B, v.recall, ls=ls, marker="o", ms=3, color=f"C{i % 10}", label=k)
        ax.plot(BS, [b / 64 for b in BS], color="gray", lw=0.8, label="random (B/64)")
        ax.set_title(f"layer {l}")
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=7, loc="lower right")
    fig.supxlabel("B (predicted set size)")
    fig.supylabel("recall of true top-8 (eval split)")
    fig.tight_layout()
    fig.savefig(RESULTS / "step2_recall_vs_B.png", dpi=90)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(18, 6), sharey=True)
    for ax, B in zip(axes, (8, 16)):
        for i, k in enumerate(MAIN[1:]):
            v = r[(r.predictor == k) & (r.B == B) & (r.layer != "avg1-15")]
            ax.plot(v.layer.astype(int), v.recall, marker="o", ms=3, color=f"C{(i + 1) % 10}",
                    ls="-" if "+" in k else ("--" if k.startswith(("A_", "B_")) else ":"), label=k)
        ax.set(title=f"recall@{B} per layer (eval, all domains)", xlabel="layer", ylim=(0, 1))
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    axes[0].set_ylabel("recall")
    fig.tight_layout()
    fig.savefig(RESULTS / "step2_recall_per_layer.png", dpi=100)
    plt.close(fig)


LEAD = [
    ("oracle_router", "layer l router logits (sanity check, not a predictor)", "none"),
    ("prev_token", "E_{t-1}^l (+ in-seq frequency over j < t to pad)", "previous token's step: before token t starts"),
    ("frequency", "counts of e in E_j^l, j < t", "before token t starts"),
    ("recency", "sum_{j<t} lambda^(t-j) 1[e in E_j^l]; lambda tuned per layer", "before token t starts"),
    ("pregate", "layer l's post_attention_layernorm + gate on the residual entering layer l (= output of layer l-1)",
     "after layer l-1 finishes; covers layer l's attention"),
    ("A_* (same layer)", "layer l attention a_l(t, j), j < t, x E_j^l", "after layer l's attention softmax; only o_proj + norm before the router"),
    ("B_* (prev layer)", "layer l-1 attention a_{l-1}(t, j), j < t, x E_j^l", "after layer l-1's attention; covers layer l-1's MoE + layer l's attention"),
    ("recency+A, pregate+A, pregate+recency+A", "combination", "limited by A: same-layer"),
    ("recency+B, pregate+B, pregate+recency+B", "combination", "limited by pregate (after layer l-1) or B"),
    ("pregate+recency", "combination", "after layer l-1 finishes"),
]


def _ci(r):
    return f"{r.recall:.3f} [{r.ci_lo:.3f}, {r.ci_hi:.3f}]"


def write_md(rec, gains, choice, tuned, L, n_ev, n_sel):
    r = rec[rec.domain == "all"]
    order = ["random", *MAIN[:4], "B_best", "A_mean_nosink", "A_top3_nosink", "A_best", "A_mean_sink",
             "A_top3_sink", "recency+A", "recency+B", "pregate", "pregate+A", "pregate+B",
             "pregate+recency(grid3)", "pregate+recency+A", "pregate+recency+B"]
    lines = ["# Step 2 tables (auto-generated)", "",
             f"Eval split ({n_ev} seqs) unless stated; tuning on the selection split ({n_sel} seqs). "
             f"Recall averaged over tokens t in [{T_MIN}, 1024). 95% bootstrap CIs over sequences. "
             "`avg1-15` = per-sequence mean over layers 1..15 (layer 0 excluded; pre-gating and B are N/A there).",
             "", "## Lead time", "", "| predictor | reads | available |", "|---|---|---|"]
    lines += [f"| {a} | {b} | {c} |" for a, b, c in LEAD]
    lines += ["", "## Recall, all domains, average over layers 1-15", "",
              "| predictor | recall@8 | recall@16 | recall@32 |", "|---|---|---|---|"]
    for k in order:
        if k == "random":
            lines.append("| random | 0.125 | 0.250 | 0.500 |")
            continue
        v = r[(r.predictor == k) & (r.layer == "avg1-15")].set_index("B")
        lines.append(f"| {k} | {_ci(v.loc[8])} | {_ci(v.loc[16])} | {_ci(v.loc[32])} |")
    for B in (8, 16):
        lines += ["", f"## Recall@{B} per layer (all domains)", "",
                  "| layer | " + " | ".join(order[1:]) + " |", "|---" * len(order) + "|"]
        for l in range(L):
            vals = []
            for k in order[1:]:
                v = r[(r.predictor == k) & (r.layer == str(l)) & (r.B == B)].recall
                vals.append("N/A" if v.isna().all() else f"{v.iloc[0]:.3f}")
            lines.append(f"| {l} | " + " | ".join(vals) + " |")
    lines += ["", "## Per domain, average over layers 1-15", "",
              "| predictor | web @8 | code @8 | math @8 | web @16 | code @16 | math @16 |", "|---|---|---|---|---|---|---|"]
    for k in order[1:]:
        vals = []
        for B in (8, 16):
            for d in data.DOMAINS:
                v = rec[(rec.domain == d) & (rec.predictor == k) & (rec.layer == "avg1-15") & (rec.B == B)]
                vals.append(f"{v.recall.iloc[0]:.3f}")
        lines.append(f"| {k} | " + " | ".join(vals) + " |")
    lines += ["", "## Gains (paired per sequence, eval)", "",
              "| comparison | a - b | domain | gain@8 [95% CI] | gain@16 [95% CI] |", "|---|---|---|---|---|"]
    g = gains[gains.layer == "avg1-15"]
    for (desc, a, b), gg in g.groupby(["comparison", "a", "b"], sort=False):
        for d in ["all", *data.DOMAINS]:
            x = gg[gg.domain == d].set_index("B")
            lines.append(f"| {desc} | {a} - {b} | {d} | {x.loc[8].gain:+.4f} [{x.loc[8].ci_lo:+.4f}, "
                         f"{x.loc[8].ci_hi:+.4f}] | {x.loc[16].gain:+.4f} [{x.loc[16].ci_lo:+.4f}, {x.loc[16].ci_hi:+.4f}] |")
    key = ["attention adds to recency", "attention (same layer) adds to pre-gating",
           "attention (same layer) adds to pre-gating+recency", "attention (prev layer) adds to pre-gating+recency"]
    lines += ["", "## Per-layer gains at B = 8 (all domains)", "",
              "| layer | " + " | ".join(key) + " |", "|---" * (len(key) + 1) + "|"]
    for l in range(L):
        vals = []
        for c in key:
            x = gains[(gains.comparison == c) & (gains.layer == str(l)) & (gains.B == 8) & (gains.domain == "all")]
            vals.append("N/A" if x.gain.isna().all() else
                        f"{x.gain.iloc[0]:+.4f} [{x.ci_lo.iloc[0]:+.4f}, {x.ci_hi.iloc[0]:+.4f}]")
        lines.append(f"| {l} | " + " | ".join(vals) + " |")
    lines += ["", "## Tuned parameters (selection split)", "",
              "| layer | lambda | A heads | B heads | recency+A w_A | pregate+A w_A | pregate+recency+A (wP,wR,wA) | "
              "pregate+recency+B (wP,wR,wB) | pregate+recency (wP,wR) |", "|---" * 9 + "|"]
    for l in range(L):
        c = choice[str(l)]
        t3a = tuned["pregate+recency+A"][l]
        t3b = tuned["pregate+recency+B"][l]
        t3 = tuned["pregate+recency(grid3)"][l]
        lines.append(f"| {l} | {c['lambda']} | {c['A_heads']} | {c['B_heads']} | {tuned['recency+A'][l]} | "
                     f"{tuned['pregate+A'][l]} | {t3a} | {t3b} | {t3[:2] if l > 0 else None} |")
    lines += ["", "![recall vs B](step2_recall_vs_B.png)", "", "![per layer](step2_recall_per_layer.png)", ""]
    (RESULTS / "STEP2_tables.md").write_text("\n".join(lines) + "\n")


def run(limit=None):
    return aggregate(collect(limit))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--aggregate-only", action="store_true")
    a = p.parse_args()
    aggregate() if a.aggregate_only else run(a.limit)

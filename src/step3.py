"""Step 3: matched lead-time comparison, k = 1..4, plus an expert-offload cache simulator.

Lead k: a predictor for token t's experts at layer l may read only what exists once layer l-k has fully
finished for token t, plus anything from tokens j < t at any layer.
  P(k)    pre-gating: residual after layer l-k (= input of layer l-k+1) -> layer l's
          post_attention_layernorm -> layer l's mlp.gate. k = 1 is Step 2's pre-gating.
  A(k)    attention of layer l-k for token t, j < t, sink (j = 0) excluded and renormalized, x 1[e in E_j^l].
          Heads per (k, l): top-3 heads of layer l-k ranked by A(k)'s own per-head recall, or head-mean,
          whichever is better on the selection split. k = 1 is Step 2's predictor B.
  R       recency, lambda per layer from Step 2 (selection split); independent of k.
  PR(k), PRA(k)  z-scored P(k)+R and P(k)+R+A(k), weights per (k, l) on Step 2's triple grid
          (PR(k) = the w_A = 0 subset of the same grid).

Pass 1 (selection): per-head recall of A(k) -> top-3 heads per (k, l).
Pass 2 (selection): standalone recall of both head variants + the full weight grid -> choices.
Pass 3 (eval): recall at the chosen settings, and the top-16 ranked experts of every predictor for every
token, which the CPU cache simulator replays. Every reported number is eval split.
"""

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from . import data
from .common import RESULTS, SEED, capture, load_model, seed_all
from .step2 import BS, T_MIN, W3, attn_scores, boot, check_pregate_wiring, recall_at, zscore

KS = [1, 2, 3, 4]
L0 = max(KS)  # common target layers l = L0..15 for comparing k
STAND = ["P", "A_top3", "A_mean"]
RAW = RESULTS / "raw" / "step3_eval.npz"
RAW_SEL = RESULTS / "raw" / "step3_sel.npz"
NOATT = np.where(W3[:, 2] == 0)[0]
TOPM = 16  # ranks kept for the simulator (B = C/2 <= 16)
CS = [16, 24, 32]
EXPERT_MB = 3 * 2048 * 1024 * 2 / 1e6  # one OLMoE expert in bf16 (gate, up, down), MB
SIM_PREDS = ["none", "R", *[f"{p}{k}" for k in KS for p in ("P", "A", "PR", "PRA")]]


def ranked(scores, freq, m=TOPM):
    """Top-m experts per token with recall_at's lexicographic (score, then frequency) ranking."""
    o1 = torch.sort(freq.expand_as(scores), dim=-1, descending=True, stable=True).indices
    o2 = torch.sort(scores.gather(-1, o1), dim=-1, descending=True, stable=True).indices
    return o1.gather(-1, o2)[..., :m]


def forward(model, ids_row):
    """One pass: expert one-hots [L, n, E], per-head attention of every layer, pre-gating logits P(k)."""
    L, E = model.config.num_hidden_layers, model.config.num_experts
    n = len(ids_row)
    x = torch.tensor(ids_row[None], dtype=torch.long, device="cuda")
    with torch.no_grad(), capture(model, attn_fn=lambda l, w: w[0].float(), keep_layer_input=True) as rec:
        model(x)
        pg = {}
        for k in KS:
            for l in range(k, L):
                layer = model.model.layers[l]
                h = layer.post_attention_layernorm(rec["layer_in"][l - k + 1][0])  # residual after layer l-k
                pg[k, l] = layer.mlp.gate(h)[0].double()
    oh = torch.zeros(L, n, E, dtype=torch.bool, device="cuda")
    for l in range(L):
        oh[l].scatter_(1, rec["experts"][l], True)
    return oh, rec["attn"], pg, rec["experts"]


def recency(n, lam, ohl):
    d = torch.arange(n, device="cuda", dtype=torch.float64)
    dist = d[:, None] - d[None, :]
    return torch.where(dist > 0, lam ** dist.clamp(min=0), 0.0) @ ohl.double()


def per_sequence(model, ids_row, lams, heads=None, cfg=None, final=False):
    """heads None: pass 1 (per-head A(k) recall). cfg None: pass 2 (standalone + grid). final: pass 3."""
    L, H = model.config.num_hidden_layers, model.config.num_attention_heads
    nK, nB = len(KS), len(BS)
    oh, attn, pg, experts = forward(model, ids_row)
    n = oh.shape[1]
    tri = torch.tril(torch.ones(n, n, dtype=torch.float64, device="cuda"), diagonal=-1)
    out = {}
    if heads is None:
        out["head"] = np.full((nK, L, H, nB), np.nan)
        out["mean"] = np.full((nK, L, nB), np.nan)
    else:
        out["stand"] = np.full((nK, L, len(STAND), nB), np.nan)
        out["R"] = np.full((L, nB), np.nan)
    if heads is not None and cfg is None:
        out["grid"] = np.full((nK, L, 2, len(W3), nB), np.nan)
    if final:
        out["combo"] = np.full((nK, L, 2, nB), np.nan)  # PR, PRA at the chosen weights
        out["ranks"] = {p: np.full((L, n, TOPM), -1, dtype=np.int8) for p in SIM_PREDS[1:]}
        out["true"] = torch.stack([experts[l] for l in range(L)]).to(torch.int8).cpu().numpy()  # [L, n, 8]
    w3 = torch.tensor(W3, device="cuda", dtype=torch.float64)[:, :, None, None]
    for l in range(L):
        ohl = oh[l]
        freq = tri @ ohl.double()
        if heads is None:
            for ki, k in enumerate(KS):
                if l >= k:
                    W = attn[l - k].double()
                    out["head"][ki, l] = recall_at(attn_scores(W, ohl, False), ohl, freq).cpu().numpy()
                    out["mean"][ki, l] = recall_at(attn_scores(W.mean(0), ohl, False), ohl, freq).cpu().numpy()
            continue
        R = recency(n, lams[l], ohl)
        out["R"][l] = recall_at(R, ohl, freq).cpu().numpy()
        if final:
            out["ranks"]["R"][l] = ranked(R, freq).to(torch.int8).cpu().numpy()
        zR = zscore(R)
        for ki, k in enumerate(KS):
            if l < k:
                continue
            W = attn[l - k].double()
            s = {"P": pg[k, l],
                 "A_top3": attn_scores(W[heads[k][l]].mean(0), ohl, False),
                 "A_mean": attn_scores(W.mean(0), ohl, False)}
            for si, name in enumerate(STAND):
                out["stand"][ki, l, si] = recall_at(s[name], ohl, freq).cpu().numpy()
            zP = zscore(s["P"])
            if cfg is None:
                for vi, v in enumerate(("top3", "mean")):
                    sc = w3[:, 0] * zP + w3[:, 1] * zR + w3[:, 2] * zscore(s[f"A_{v}"])
                    out["grid"][ki, l, vi] = recall_at(sc, ohl, freq).cpu().numpy()
                continue
            c = cfg[f"{k},{l}"]
            zA = zscore(s[f"A_{c['heads']}"])
            pr = c["w_PR"][0] * zP + c["w_PR"][1] * zR
            pra = c["w_PRA"][0] * zP + c["w_PRA"][1] * zR + c["w_PRA"][2] * zA
            out["combo"][ki, l] = torch.stack([recall_at(pr, ohl, freq), recall_at(pra, ohl, freq)]).cpu().numpy()
            for p, sc in (("P", s["P"]), ("A", s[f"A_{c['heads']}"]), ("PR", pr), ("PRA", pra)):
                out["ranks"][f"{p}{k}"][l] = ranked(sc, freq).to(torch.int8).cpu().numpy()
    return out


def run_pass(model, seqs, ids, **kw):
    outs = []
    for c, i in enumerate(seqs):
        outs.append(per_sequence(model, ids[i], **kw))
        if c % 25 == 0:
            print(f"  seq {c + 1}/{len(seqs)}", flush=True)
    res = {k: np.stack([o[k] for o in outs]) for k in outs[0] if k != "ranks"}
    if "ranks" in outs[0]:
        res["ranks"] = {p: np.stack([o["ranks"][p] for o in outs]) for p in outs[0]["ranks"]}
    return res


def choose_heads(p1, L):
    """Top-3 heads of layer l-k per (k, l), by A(k)'s own per-head recall on the selection split."""
    hr = np.nanmean(p1["head"], 0).mean(-1)  # [K, L, H]
    return {k: {l: [int(h) for h in np.argsort(-hr[ki, l])[:3]] for l in range(k, L)} for ki, k in enumerate(KS)}


def choose_cfg(p2, L):
    """Per (k, l) on the selection split, objective mean recall over B: head variant of A(k) by standalone
    recall, then PR and PRA weights on the triple grid (PRA's grid uses the chosen A variant)."""
    st = np.nanmean(p2["stand"], 0).mean(-1)  # [K, L, 3]
    gr = np.nanmean(p2["grid"], 0).mean(-1)  # [K, L, 2, G]
    cfg = {}
    for ki, k in enumerate(KS):
        for l in range(k, L):
            vi = 0 if st[ki, l, 1] >= st[ki, l, 2] else 1
            g = gr[ki, l, vi]
            cfg[f"{k},{l}"] = {"heads": ("top3", "mean")[vi],
                               "w_PR": W3[NOATT[np.argmax(g[NOATT])]].tolist(),
                               "w_PRA": W3[int(np.argmax(g))].tolist()}
    return cfg


def _nan2none(a):
    return [_nan2none(x) for x in a] if isinstance(a, list) else (None if a != a else a)


def collect(limit=None):
    import warnings

    warnings.filterwarnings("ignore", category=RuntimeWarning)  # nanmean over undefined (k, l) cells
    seed_all()
    ids, domains = data.load(1024)
    split = data.load_split()
    sel, ev = split["selection"], split["eval"]
    if limit:
        k = max(1, limit // 6)
        pick = lambda part: [i for d in data.DOMAINS for i in [s for s in part if domains[s] == d][:k]]
        sel, ev = pick(sel), pick(ev)
    s2 = json.loads((RESULTS / "step2_choice.json").read_text())["per_layer"]
    model = load_model()
    L = model.config.num_hidden_layers
    lams = [s2[str(l)]["lambda"] for l in range(L)]
    check_pregate_wiring(model, ids[sel[0]])
    print(f"pass 1 (head ranking): {len(sel)} selection seqs")
    p1 = run_pass(model, sel, ids, lams=lams)
    heads = choose_heads(p1, L)
    print(f"pass 2 (standalone + weight grid): {len(sel)} selection seqs")
    p2 = run_pass(model, sel, ids, lams=lams, heads=heads)
    cfg = choose_cfg(p2, L)
    (RESULTS / "step3_choice.json").write_text(json.dumps(
        {"criterion": "selection split, mean recall over B in " + str(BS),
         "lambda_per_layer (Step 2)": lams,
         "A_top3_heads_of_layer_l-k": {str(k): {str(l): h for l, h in v.items()} for k, v in heads.items()},
         "A_head_mean_recall_selection [k][l]": _nan2none(np.round(np.nanmean(p1["mean"], 0).mean(-1), 4).tolist()),
         "A_per_head_recall_selection [k][l][head]": _nan2none(np.round(np.nanmean(p1["head"], 0).mean(-1), 4).tolist()),
         "per_k_l (weights: w_P, w_R, w_A)": cfg}, indent=2))
    print(f"pass 3 (eval, chosen settings + ranks): {len(ev)} eval seqs")
    p3 = run_pass(model, ev, ids, lams=lams, heads=heads, cfg=cfg, final=True)
    del model
    torch.cuda.empty_cache()
    RAW.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(RAW_SEL, stand=p2["stand"], R=p2["R"], grid=p2["grid"], seq=np.array(sel))
    ranks = p3.pop("ranks")
    np.savez_compressed(RAW, seq=np.array(ev), domain=domains[ev], **p3, **{f"ranks_{p}": v for p, v in ranks.items()})
    return load_raw()


def load_raw():
    d = dict(np.load(RAW))
    d["ranks"] = {p: d.pop(f"ranks_{p}") for p in SIM_PREDS[1:]}
    return d


# ---------------------------------------------------------------- offload simulator (CPU)

def simulate(true, ranks, C, B):
    """LRU expert cache of capacity C, one per (row = sequence x layer); decode token by token.

    true [N, n, 8] experts used; ranks [N, n, >=B] prefetch order (None = no prefetch).
    Per token: prefetch the top-B in reverse rank order (top-ranked ends most recent; already-resident
    ones are only touched), then touch the resident true experts (hits), then load the missing ones
    (misses). A load into a full cache evicts the least recently used expert. The true experts are touched
    before any miss is loaded, so a miss load can never evict another of the same token's experts.
    Returns per-row totals: prefetch loads, misses (= on-miss loads)."""
    N, n, k = true.shape
    ts = np.full((N, 64), -1, dtype=np.int64)  # last-use clock; -1 = not resident
    cnt = np.zeros(N, dtype=np.int64)
    ar = np.arange(N)
    pre = np.zeros(N, dtype=np.int64)
    miss = np.zeros(N, dtype=np.int64)
    clock = 0

    def load(e, want):
        """Load expert e[row] for rows in want (not resident), evicting LRU where full."""
        nonlocal clock
        full = want & (cnt >= C)
        if full.any():
            r = np.where(full)[0]
            sub = ts[r]
            v = np.where(sub >= 0, sub, np.iinfo(np.int64).max).argmin(1)
            ts[r, v] = -1
            cnt[r] -= 1
        ts[ar[want], e[want]] = clock
        cnt[want] += 1
        clock += 1

    for t in range(n):
        if ranks is not None:
            for i in range(B - 1, -1, -1):
                e = ranks[:, t, i].astype(np.int64)
                res = ts[ar, e] >= 0
                ts[ar[res], e[res]] = clock  # touch
                load(e, ~res)
                pre += ~res
        e_t = true[:, t].astype(np.int64)
        res = ts[ar[:, None], e_t] >= 0  # [N, 8]
        for j in range(k):
            ts[ar[res[:, j]], e_t[res[:, j], j]] = clock
            clock += 1
        for j in range(k):
            load(e_t[:, j], ~res[:, j])
        miss += (~res).sum(1)
    return pre, miss


def run_sim(raw):
    """Simulate layers L0..15 of every eval sequence for each predictor and C. Returns per-seq totals."""
    true = raw["true"]  # [S, L, n, 8]
    S, L, n, _ = true.shape
    layers = list(range(L0, L))
    rows = true[:, layers].reshape(S * len(layers), n, -1)
    out = {}
    for C in CS:
        B = C // 2
        for p in SIM_PREDS:
            r = None if p == "none" else raw["ranks"][p][:, layers].reshape(S * len(layers), n, -1)
            assert r is None or (r[..., :B] >= 0).all(), p
            pre, miss = simulate(rows, r, C, B)
            out[(p, C)] = (pre.reshape(S, len(layers)), miss.reshape(S, len(layers)))
        print(f"  simulated C = {C}", flush=True)
    return out, n, layers


# ---------------------------------------------------------------- aggregation

def check_k1(raw, choice):
    """k = 1 must reproduce Step 2 (eval split, per sequence): P(1) = pregate, A(1) top-3 = B_top3_nosink,
    PR(1) and PRA(1) = Step 2's tuned pregate+recency(grid3) / pregate+recency+B when the same weights
    and A variant are chosen."""
    from .step2 import FIXED

    s2 = dict(np.load(RESULTS / "raw" / "step2_perseq.npz"))
    pos = {int(s): i for i, s in enumerate(s2["seq"])}
    idx = [pos[int(s)] for s in raw["seq"]]
    ix = {k: i for i, k in enumerate(FIXED)}
    tw = json.loads((RESULTS / "step2_tuned_weights.json").read_text())["weights"]
    L = raw["stand"].shape[2]
    s2heads = json.loads((RESULTS / "step2_choice.json").read_text())["per_layer"]
    sh = json.loads((RESULTS / "step3_choice.json").read_text())["A_top3_heads_of_layer_l-k"]["1"]
    hl = [l for l in range(1, L) if sh[str(l)] == s2heads[str(l)]["B_heads_of_prev_layer"]]
    res = {"A(1) top-3 heads == Step 2 B heads (layers)": hl,
           "P(1) vs Step 2 pregate": np.nanmax(np.abs(raw["stand"][:, 0, 1:, 0] - s2["fixed"][idx, 1:, ix["pregate"]])),
           "A(1) top3 vs Step 2 B_top3_nosink, layers with the same heads": np.nanmax(
               np.abs(raw["stand"][:, 0, hl, 1] - s2["fixed"][idx][:, hl, ix["B_top3_nosink"]]))}
    same_pr, same_pra, d_pr, d_pra = [], [], [], []
    for l in range(1, L):
        c = choice[f"1,{l}"]
        w2pr = tw["pregate+recency(grid3)"][str(l)]
        w2b = tw["pregate+recency+B"][str(l)]
        if np.allclose(c["w_PR"], w2pr):
            same_pr.append(l)
            d_pr.append(np.abs(raw["combo"][:, 0, l, 0] - s2["pregate+recency+A"][idx, l, _w3i(w2pr)]).max())
        if l in hl and c["heads"] == "top3" and np.allclose(c["w_PRA"], w2b):
            same_pra.append(l)
            d_pra.append(np.abs(raw["combo"][:, 0, l, 1] - s2["pregate+recency+B"][idx, l, _w3i(w2b)]).max())
    res["PR(1) layers with Step 2's weights"] = same_pr
    res["PR(1) max abs diff on those layers"] = max(d_pr) if d_pr else None
    res["PRA(1) layers with Step 2's weights and same top-3 heads"] = same_pra
    res["PRA(1) max abs diff on those layers"] = max(d_pra) if d_pra else None
    return {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in res.items()}


def _w3i(w):
    return int(np.argmin(np.abs(W3 - np.array(w)).sum(1)))


def predictors(raw, choice):
    """Per-seq recall [S, K, L, B] per predictor name."""
    st, cb = raw["stand"], raw["combo"]
    K, L = st.shape[1:3]
    sel = np.array([[0 if choice.get(f"{k},{l}", {}).get("heads", "top3") == "top3" else 1
                     for l in range(L)] for k in KS])
    A = np.where(sel[None, :, :, None] == 0, st[:, :, :, 1], st[:, :, :, 2])
    R = np.repeat(raw["R"][:, None], K, 1)
    return {"P": st[:, :, :, 0], "A": A, "A_top3": st[:, :, :, 1], "A_mean": st[:, :, :, 2], "R": R,
            "PR": cb[:, :, :, 0], "PRA": cb[:, :, :, 1]}


def aggregate(raw=None):
    import warnings

    warnings.filterwarnings("ignore", category=RuntimeWarning)
    raw = raw if raw is not None else load_raw()
    choice = json.loads((RESULTS / "step3_choice.json").read_text())["per_k_l (weights: w_P, w_R, w_A)"]
    rng = np.random.default_rng(SEED)
    pr = predictors(raw, choice)
    S, K, L, _ = pr["P"].shape
    dom = raw["domain"]

    def summaries(v, m):
        """v [S, K, L, B] -> rows per k: per layer, avg over l = L0..15, avg over all defined l >= k."""
        out = []
        for ki, k in enumerate(KS):
            x = v[m, ki]
            mean, lo, hi = boot(x, rng)
            for l in range(L):
                out.append((k, str(l), mean[l], lo[l], hi[l]))
            for name, a in ((f"avg{L0}-15", L0), ("avg_defined", k)):
                am, alo, ahi = boot(np.nanmean(x[:, a:], 1), rng)
                out.append((k, name, am, alo, ahi))
        return out

    rows = []
    for d in ["all", *data.DOMAINS]:
        m = np.ones(S, bool) if d == "all" else dom == d
        for p, v in pr.items():
            for k, layer, mean, lo, hi in summaries(v, m):
                for bi, B in enumerate(BS):
                    rows.append(dict(domain=d, predictor=p, k=k, layer=layer, B=B, recall=mean[bi],
                                     ci_lo=lo[bi], ci_hi=hi[bi], n_seq=int(m.sum())))
    rec = pd.DataFrame(rows)
    rec.to_csv(RESULTS / "step3_recall.csv", index=False, float_format="%.4f")

    comps = [("PRA", "PR", "attention's unique gain: PRA(k) - PR(k)"),
             ("PR", "P", "recency adds to pre-gating: PR(k) - P(k)"),
             ("A", "R", "attention alone vs recency: A(k) - R"),
             ("A", "P", "attention vs pre-gating: A(k) - P(k)"),
             ("A_top3", "A_mean", "top-3 heads vs head-mean: A_top3(k) - A_mean(k)")]
    grows = []
    for a, b, desc in comps:
        for d in ["all", *data.DOMAINS]:
            m = np.ones(S, bool) if d == "all" else dom == d
            for k, layer, mean, lo, hi in summaries(pr[a] - pr[b], m):
                for bi, B in enumerate(BS):
                    if B in (8, 16):
                        grows.append(dict(comparison=desc, a=a, b=b, domain=d, k=k, layer=layer, B=B,
                                          gain=mean[bi], ci_lo=lo[bi], ci_hi=hi[bi]))
    gains = pd.DataFrame(grows)
    gains.to_csv(RESULTS / "step3_gains.csv", index=False, float_format="%.4f")

    print("simulating offload cache ...", flush=True)
    sim, n, layers = run_sim(raw)
    srows = []
    nl = len(layers)
    base = {}
    for (p, C), (pre, miss) in sim.items():
        miss_tl = miss.sum(1) / (n * nl)  # per seq: misses per token per layer
        moved_t = (pre + miss).sum(1) / n  # per seq: experts moved per token, summed over layers
        pre_t = pre.sum(1) / n
        vals = {"misses_per_token_layer": miss_tl, "experts_moved_per_token": moved_t,
                "prefetched_per_token": pre_t, "miss_loads_per_token": miss.sum(1) / n,
                "MB_per_token": moved_t * EXPERT_MB}
        for name, v in vals.items():
            mean, lo, hi = boot(v, rng)
            srows.append(dict(predictor=p, C=C, B=C // 2 if p != "none" else 0, metric=name,
                              value=mean, ci_lo=lo, ci_hi=hi))
        base[p, C] = miss.sum(1)
    # Relative miss reduction of PRA(k) vs PR(k) (the decision rule's reference) and vs P(k) (which can have
    # fewer misses than PR(k)); ratio of totals over sequences, bootstrap CI.
    for C in CS:
        for k in KS:
            for ref in ("PR", "P"):
                a, b = base[f"PRA{k}", C], base[f"{ref}{k}", C]
                idx = rng.integers(0, S, (1000, S))
                lo, hi = np.percentile(1 - a[idx].sum(1) / b[idx].sum(1), [2.5, 97.5])
                srows.append(dict(predictor=f"PRA{k} vs {ref}{k}", C=C, B=C // 2, metric="relative_miss_reduction",
                                  value=1 - a.sum() / b.sum(), ci_lo=lo, ci_hi=hi))
    simdf = pd.DataFrame(srows)
    simdf.to_csv(RESULTS / "step3_sim.csv", index=False, float_format="%.4f")

    chk = check_k1(raw, choice)
    (RESULTS / "step3_k1_check.json").write_text(json.dumps(chk, indent=2))
    plots(rec, gains, simdf)
    write_md(rec, gains, simdf, chk, choice, S, L)
    return rec, gains, simdf


LINES = [("P", "pre-gating P(k)"), ("A", "attention A(k)"), ("R", "recency R"), ("PR", "P(k)+R"),
         ("PRA", "P(k)+R+A(k)")]


def plots(rec, gains, simdf):
    r = rec[(rec.domain == "all") & (rec.layer == f"avg{L0}-15")]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, B in zip(axes[:2], (8, 16)):
        for i, (p, lab) in enumerate(LINES):
            v = r[(r.predictor == p) & (r.B == B)].sort_values("k")
            ax.errorbar(v.k, v.recall, yerr=[v.recall - v.ci_lo, v.ci_hi - v.recall], marker="o", capsize=3,
                        color=f"C{i}", label=lab, ls="--" if p == "R" else "-")
        ax.set(title=f"recall@{B} vs lead k (eval, layers {L0}-15)", xlabel="lead k (layers)", ylabel="recall",
               xticks=KS)
        ax.grid(alpha=0.3)
    axes[0].legend()
    g = gains[(gains.a == "PRA") & (gains.b == "PR") & (gains.domain == "all") & (gains.layer == f"avg{L0}-15")]
    for B, c in ((8, "C0"), (16, "C1")):
        v = g[g.B == B].sort_values("k")
        axes[2].errorbar(v.k, v.gain, yerr=[v.gain - v.ci_lo, v.ci_hi - v.gain], marker="o", capsize=3,
                         color=c, label=f"B = {B}")
    axes[2].axhline(0.02, color="red", ls=":", label="decision threshold (+0.02 @8)")
    axes[2].axhline(0, color="gray", lw=0.8)
    axes[2].set(title="attention's unique gain PRA(k) - PR(k)", xlabel="lead k (layers)", ylabel="recall gain",
                xticks=KS)
    axes[2].legend()
    axes[2].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS / "step3_recall_vs_k.png", dpi=100)
    plt.close(fig)

    s = simdf[simdf.metric == "misses_per_token_layer"]
    fig, axes = plt.subplots(1, len(CS), figsize=(18, 5), sharey=True)
    for ax, C in zip(axes, CS):
        sc = s[s.C == C].set_index("predictor")
        ax.axhline(sc.loc["none"].value, color="gray", ls=":", label="no prefetch (LRU)")
        ax.axhline(sc.loc["R"].value, color="C2", ls="--", label="R")
        for i, (p, lab) in enumerate(LINES):
            if p == "R":
                continue
            ax.plot(KS, [sc.loc[f"{p}{k}"].value for k in KS], marker="o", color=f"C{i}", label=lab)
        ax.set(title=f"cache C = {C}, prefetch B = {C // 2} (layers {L0}-15)", xlabel="lead k", xticks=KS)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("misses per token per layer (of 8)")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "step3_sim.png", dpi=100)
    plt.close(fig)


def _ci(v, x="recall", fmt=".3f"):
    return f"{v[x]:{fmt}} [{v.ci_lo:{fmt}}, {v.ci_hi:{fmt}}]"


def write_md(rec, gains, simdf, chk, choice, S, L):
    r = rec[rec.domain == "all"]
    lines = ["# Step 3 tables (auto-generated)", "",
             f"Eval split ({S} seqs). Recall over tokens t in [{T_MIN}, 1024); 95% bootstrap CIs over sequences. "
             f"`avg{L0}-15` = per-sequence mean over target layers {L0}..15 (same layers for every k); "
             "`avg_defined` = over all l >= k.", "",
             "## k = 1 reproduces Step 2 (eval, per-sequence max abs difference)", "",
             *[f"- {k}: {v}" for k, v in chk.items()]]
    for avg in (f"avg{L0}-15", "avg_defined"):
        lines += ["", f"## Recall vs k, all domains, {avg}", "",
                  "| predictor | " + " | ".join(f"k={k} @8" for k in KS) + " | " + " | ".join(f"k={k} @16" for k in KS) + " |",
                  "|---" * 9 + "|"]
        for p, lab in LINES + [("A_top3", "A_top3(k)"), ("A_mean", "A_mean(k)")]:
            vals = [_ci(r[(r.predictor == p) & (r.k == k) & (r.layer == avg) & (r.B == B)].iloc[0])
                    for B in (8, 16) for k in KS]
            lines.append(f"| {lab} | " + " | ".join(vals) + " |")
    g = gains[gains.layer.isin([f"avg{L0}-15", "avg_defined"])]
    lines += ["", "## Gains vs k (paired per sequence, eval)", "",
              "| comparison | domain | layers | " + " | ".join(f"k={k} @8" for k in KS) + " | "
              + " | ".join(f"k={k} @16" for k in KS) + " |", "|---" * 11 + "|"]
    for (desc, d, layer), gg in g.groupby(["comparison", "domain", "layer"], sort=False):
        vals = [_ci(gg[(gg.k == k) & (gg.B == B)].iloc[0], "gain", "+.4f") for B in (8, 16) for k in KS]
        lines.append(f"| {desc} | {d} | {layer} | " + " | ".join(vals) + " |")
    for k in (2, 4):
        lines += ["", f"## Per layer, k = {k}, all domains (recall@8; unique gain PRA - PR at B = 8 and 16)", "",
                  "| layer | P | A | R | PR | PRA | PRA-PR @8 | PRA-PR @16 | A heads | w (P,R,A) |", "|---" * 10 + "|"]
        for l in range(k, L):
            vals = [f"{r[(r.predictor == p) & (r.k == k) & (r.layer == str(l)) & (r.B == 8)].recall.iloc[0]:.3f}"
                    for p in ("P", "A", "R", "PR", "PRA")]
            for B in (8, 16):
                x = gains[(gains.a == "PRA") & (gains.b == "PR") & (gains.domain == "all") & (gains.k == k)
                          & (gains.layer == str(l)) & (gains.B == B)].iloc[0]
                vals.append(_ci(x, "gain", "+.4f"))
            c = choice[f"{k},{l}"]
            lines.append(f"| {l} | " + " | ".join(vals) + f" | {c['heads']} | {c['w_PRA']} |")
    lines += ["", f"## Offload simulator (eval, layers {L0}-15, LRU per layer, B = C/2; expert = {EXPERT_MB:.2f} MB)", "",
              "| predictor | C | misses / token / layer | experts moved / token | of which prefetched | MB / token |",
              "|---|---|---|---|---|---|"]
    for C in CS:
        for p in SIM_PREDS:
            x = simdf[(simdf.predictor == p) & (simdf.C == C)].set_index("metric")
            lines.append(f"| {p} | {C} | {_ci(x.loc['misses_per_token_layer'], 'value')} | "
                         f"{_ci(x.loc['experts_moved_per_token'], 'value', '.2f')} | "
                         f"{x.loc['prefetched_per_token'].value:.2f} | {_ci(x.loc['MB_per_token'], 'value', '.0f')} |")
    lines += ["", "## Relative miss reduction of PRA(k) (1 - misses_PRA / misses_ref)", "",
              "| reference | C | " + " | ".join(f"k={k}" for k in KS) + " |", "|---" * (len(KS) + 2) + "|"]
    for ref in ("PR", "P"):
        for C in CS:
            x = simdf[(simdf.metric == "relative_miss_reduction") & (simdf.C == C)]
            lines.append(f"| {ref}(k) | {C} | " + " | ".join(
                _ci(x[x.predictor == f"PRA{k} vs {ref}{k}"].iloc[0], "value", "+.4f") for k in KS) + " |")
    lines += ["", "![recall vs k](step3_recall_vs_k.png)", "", "![simulator](step3_sim.png)", ""]
    (RESULTS / "STEP3_tables.md").write_text("\n".join(lines) + "\n")


def run(limit=None):
    return aggregate(collect(limit))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--aggregate-only", action="store_true")
    a = p.parse_args()
    aggregate() if a.aggregate_only else run(a.limit)

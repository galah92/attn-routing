"""Step 1: within a distance bin, does attention a(t, j) predict expert overlap |E_t ∩ E_j| / 8?

Per sequence: sample pairs stratified by distance bin (all d = 1 pairs, up to PAIRS_PER_BIN otherwise;
the same pairs are used for every layer), gather per-head attention for them inside the attention hook,
and compute Spearman correlations per (layer, head, bin, variant). Variants: sink {incl, excl j = 0} x
tokens {all, distinct ids[t] != ids[j]}. Main result: sink excluded, distinct tokens.
Aggregation: mean over sequences with a bootstrap 95% CI over sequences.
"""

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from . import data
from .common import RESULTS, SEED, capture, load_model, seed_all

BINS = [(1, 1), (2, 2), (3, 4), (5, 8), (9, 16), (17, 32), (33, 64), (65, 128), (129, 256),
        (257, 512), (513, 1023)]
BIN_LABELS = [f"{lo}" if lo == hi else f"{lo}-{hi}" for lo, hi in BINS]
PAIRS_PER_BIN = 2000
MIN_PAIRS = 20  # fewer pairs than this in a (seq, bin, variant) cell -> NaN
N_BOOT = 1000
SINK = ["incl", "excl"]  # axis index 0/1
TOKENS = ["all", "distinct"]
RAW = RESULTS / "raw" / "step1_perseq.npz"


def sample_pairs(n, rng):
    """Return t, j, bin index arrays (j < t), stratified by distance bin."""
    ts, js, bs = [], [], []
    for b, (lo, hi) in enumerate(BINS):
        d = np.arange(lo, min(hi, n - 1) + 1)
        counts = n - d  # pairs at distance d: t in [d, n-1]
        cum = np.cumsum(counts)
        total = int(cum[-1])
        k = total if lo == 1 else min(PAIRS_PER_BIN, total)
        flat = np.sort(rng.choice(total, size=k, replace=False))
        di = np.searchsorted(cum, flat, side="right")
        offset = flat - np.concatenate([[0], cum[:-1]])[di]
        t = d[di] + offset
        ts.append(t), js.append(t - d[di]), bs.append(np.full(k, b))
    return np.concatenate(ts), np.concatenate(js), np.concatenate(bs)


def avg_rank(x):
    """Average ranks (ties share the mean rank) along the last dim."""
    s, idx = x.sort(-1)
    new = torch.ones_like(s, dtype=torch.bool)
    new[..., 1:] = s[..., 1:] != s[..., :-1]
    gid = new.long().cumsum(-1) - 1
    pos = torch.arange(x.shape[-1], device=x.device, dtype=torch.float64).expand_as(s)
    sums = torch.zeros_like(pos).scatter_add_(-1, gid, pos)
    cnt = torch.zeros_like(pos).scatter_add_(-1, gid, torch.ones_like(pos))
    r_sorted = (sums / cnt.clamp(min=1)).gather(-1, gid)
    return torch.empty_like(r_sorted).scatter_(-1, idx, r_sorted)


def pearson(a, b):
    a = a - a.mean(-1, keepdim=True)
    b = b - b.mean(-1, keepdim=True)
    den = (a.pow(2).sum(-1) * b.pow(2).sum(-1)).sqrt()
    r = (a * b).sum(-1) / den
    return torch.where(den > 0, r, torch.full_like(r, float("nan")))


def spearman(a, b):
    return pearson(avg_rank(a), avg_rank(b))


def check_spearman():
    g = np.random.default_rng(1)
    a = g.integers(0, 9, 500).astype(float) / 8
    b = np.round(a + g.normal(0, 0.5, 500), 1)
    ours = spearman(torch.tensor(a), torch.tensor(b)).item()
    ref = spearmanr(a, b).statistic
    assert abs(ours - ref) < 1e-9, (ours, ref)
    return ours, ref


def per_sequence(model, ids_row, seq_idx):
    n = len(ids_row)
    rng = np.random.default_rng([SEED, seq_idx])
    t_np, j_np, b_np = sample_pairs(n, rng)
    T = torch.tensor(t_np, device="cuda")
    J = torch.tensor(j_np, device="cuda")
    x = torch.tensor(ids_row[None], dtype=torch.long, device="cuda")
    with torch.no_grad(), capture(model, attn_fn=lambda l, w: w[0][:, T, J].float()) as rec:
        model(x)
    L = len(rec["experts"])
    E = model.config.num_experts
    K = model.config.num_experts_per_tok
    oh = torch.zeros(L, n, E, dtype=torch.bool, device="cuda")
    for l in range(L):
        oh[l].scatter_(1, rec["experts"][l], True)
    overlap = (oh[:, T] & oh[:, J]).sum(-1).double() / K  # [L, P]
    A = torch.stack([rec["attn"][l] for l in range(L)]).double()  # [L, H, P]
    A = torch.cat([A, A.mean(1, keepdim=True)], 1)  # [L, H+1, P]; last = head mean
    H1 = A.shape[1]

    tok = torch.tensor(ids_row, device="cuda")
    distinct = tok[T] != tok[J]
    notsink = J > 0
    bins = torch.tensor(b_np, device="cuda")
    dist = (T - J).double()

    NB = len(BINS)
    rho = np.full((L, H1, NB, 2, 2), np.nan, np.float32)
    partial = np.full((L, NB, 2, 2), np.nan, np.float32)
    npairs = np.zeros((NB, 2, 2), np.int32)
    same_frac = np.full((NB, 2), np.nan, np.float32)
    mov = np.full((L, NB, 2, 2), np.nan, np.float32)
    mattn = np.full((L, NB, 2, 2), np.nan, np.float32)
    for b in range(NB):
        inb = bins == b
        for s in range(2):
            ms = inb & notsink if s == 1 else inb
            same_frac[b, s] = (~distinct[ms]).float().mean().item()
            for k in range(2):
                m = ms & distinct if k == 1 else ms
                p = int(m.sum())
                npairs[b, s, k] = p
                if p == 0:
                    continue
                a, o = A[:, :, m], overlap[:, m]
                mov[:, b, s, k] = o.mean(-1).cpu().numpy()
                mattn[:, b, s, k] = a[:, -1].mean(-1).cpu().numpy()
                if p < MIN_PAIRS:
                    continue
                ro = avg_rank(o)  # [L, p]
                r = pearson(avg_rank(a), ro[:, None, :])  # [L, H+1]
                rho[:, :, b, s, k] = r.cpu().numpy()
                # Partial Spearman of (attn, overlap) controlling for exact distance within the bin
                dd = dist[m]
                if dd.min() == dd.max():
                    partial[:, b, s, k] = r[:, -1].cpu().numpy()
                else:
                    rd = avg_rank(dd)
                    r_ao = r[:, -1]
                    r_ad = pearson(avg_rank(a[:, -1]), rd)
                    r_od = pearson(ro, rd)
                    pr = (r_ao - r_ad * r_od) / ((1 - r_ad**2) * (1 - r_od**2)).sqrt()
                    partial[:, b, s, k] = pr.cpu().numpy()
    return dict(rho=rho, partial=partial, npairs=npairs, same_frac=same_frac, mov=mov, mattn=mattn)


def collect(limit=None):
    seed_all()
    print("spearman check (ours, scipy):", check_spearman())
    ids, domains = data.load(1024)
    split = data.load_split()
    seqs = sorted(split["selection"] + split["eval"])
    if limit:
        k = max(1, limit // 6)
        seqs = sorted(i for part in ("selection", "eval") for d in data.DOMAINS
                      for i in [s for s in split[part] if domains[s] == d][:k])
    model = load_model()
    outs = []
    for c, i in enumerate(seqs):
        outs.append(per_sequence(model, ids[i], i))
        if c % 25 == 0:
            print(f"seq {c + 1}/{len(seqs)}", flush=True)
    del model
    torch.cuda.empty_cache()
    raw = {k: np.stack([o[k] for o in outs]) for k in outs[0]}
    raw["seq"] = np.array(seqs)
    raw["domain"] = domains[seqs]
    RAW.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(RAW, **raw)
    return raw


def boot_ci(vals, rng):
    """vals [S, C] -> mean [C], lo [C], hi [C], n [C] with a bootstrap over sequences (NaN-aware)."""
    S = vals.shape[0]
    mean = np.nanmean(vals, 0)
    n = (~np.isnan(vals)).sum(0)
    idx = rng.integers(0, S, (N_BOOT, S))
    lo, hi = np.empty_like(mean), np.empty_like(mean)
    for c0 in range(0, vals.shape[1], 64):
        bm = np.nanmean(vals[idx, c0 : c0 + 64], 1)  # [N_BOOT, chunk]
        lo[c0 : c0 + 64], hi[c0 : c0 + 64] = np.nanpercentile(bm, [2.5, 97.5], 0)
    return mean, lo, hi, n


def aggregate(raw=None):
    import warnings

    warnings.filterwarnings("ignore", category=RuntimeWarning)  # all-NaN slices in sparse cells
    raw = raw if raw is not None else dict(np.load(RAW))
    split = data.load_split()
    sel = np.isin(raw["seq"], split["selection"])
    groups = {"eval": ~sel, "selection": sel}
    S, L, H1, NB = raw["rho"].shape[:4]
    H = H1 - 1
    rng = np.random.default_rng(SEED)

    # --- head-mean correlations per (split, domain, layer, bin, variant)
    rows = []
    for gname, gmask in groups.items():
        for dom in ["all", *data.DOMAINS]:
            m = gmask & ((raw["domain"] == dom) if dom != "all" else True)
            rho = raw["rho"][m][:, :, -1]  # [s, L, NB, 2, 2]
            s = rho.shape[0]
            mean, lo, hi, n = boot_ci(rho.reshape(s, -1), rng)
            pm, plo, phi, _ = boot_ci(raw["partial"][m].reshape(s, -1), rng)
            shp = rho.shape[1:]
            mean, lo, hi, n, pm, plo, phi = (a.reshape(shp) for a in (mean, lo, hi, n, pm, plo, phi))
            npairs = raw["npairs"][m].sum(0)
            same = np.nanmean(raw["same_frac"][m], 0)
            mov = np.nanmean(raw["mov"][m], 0)
            mattn = np.nanmean(raw["mattn"][m], 0)
            for l in range(L):
                for b in range(NB):
                    for si in range(2):
                        for ki in range(2):
                            ix = (l, b, si, ki)
                            rows.append(dict(
                                split=gname, domain=dom, layer=l, bin=BIN_LABELS[b], d_lo=BINS[b][0],
                                d_hi=BINS[b][1], sink=SINK[si], tokens=TOKENS[ki], n_seq=int(n[ix]),
                                n_pairs=int(npairs[b, si, ki]), rho=mean[ix], ci_lo=lo[ix], ci_hi=hi[ix],
                                partial_rho=pm[ix], partial_ci_lo=plo[ix], partial_ci_hi=phi[ix],
                                same_token_frac=same[b, si], mean_overlap=mov[ix], mean_attn=mattn[ix]))
    corr = pd.DataFrame(rows)
    corr.to_csv(RESULTS / "step1_corr.csv", index=False, float_format="%.4f")

    # --- per-head score: mean over bins d >= 2 of rho (sink excl, distinct tokens), per sequence
    score = np.nanmean(raw["rho"][:, :, :, 1:, 1, 1], -1)  # [S, L, H+1]
    head_rows = []
    sm = np.nanmean(score[sel], 0)
    em, elo, ehi, en = (a.reshape(L, H1) for a in boot_ci(score[~sel].reshape((~sel).sum(), -1), rng))
    dom_eval = {d: np.nanmean(score[~sel & (raw["domain"] == d)], 0) for d in data.DOMAINS}
    for l in range(L):
        for h in range(H1):
            head_rows.append(dict(layer=l, head="mean" if h == H else h, sel_score=sm[l, h],
                                  eval_score=em[l, h], eval_ci_lo=elo[l, h], eval_ci_hi=ehi[l, h],
                                  eval_n_seq=int(en[l, h]),
                                  **{f"eval_{d}": dom_eval[d][l, h] for d in data.DOMAINS}))
    heads = pd.DataFrame(head_rows)
    ind = heads[heads["head"] != "mean"].copy()
    ind["sel_rank"] = ind["sel_score"].rank(ascending=False).astype(int)
    heads = heads.merge(ind[["layer", "head", "sel_rank"]], on=["layer", "head"], how="left")
    heads.to_csv(RESULTS / "step1_heads.csv", index=False, float_format="%.4f")
    top = ind.sort_values("sel_score", ascending=False).head(10)
    top.to_csv(RESULTS / "step1_top_heads.csv", index=False, float_format="%.4f")
    best_sel = {l: int(ind[ind.layer == l].sort_values("sel_score").iloc[-1]["head"]) for l in range(L)}
    (RESULTS / "step1_best_heads.json").write_text(json.dumps(
        {"criterion": "selection split, mean rho over bins d>=2, sink excl, distinct tokens",
         "best_head_per_layer": best_sel,
         "top10": [[int(r.layer), int(r.head)] for r in top.itertuples()]}, indent=2))

    plots(corr, sm[:, :H], em[:, :H], L)
    write_md(corr, heads, top, raw, sel, L)
    return corr, heads


def plots(corr, sel_heads, eval_heads, L):
    ev = corr[(corr.split == "eval")]
    x = np.arange(len(BINS))
    styles = {("excl", "distinct"): ("C0", "-", 2.2), ("excl", "all"): ("C1", "--", 1.2),
              ("incl", "distinct"): ("C2", ":", 1.2), ("incl", "all"): ("C3", "-.", 1.2)}

    def grid(title, fname, series):
        fig, axes = plt.subplots(4, 4, figsize=(18, 14), sharex=True, sharey=True)
        for l, ax in enumerate(axes.flat):
            for label, df, (c, ls, lw) in series(l):
                ax.plot(x, df.rho.values, color=c, ls=ls, lw=lw, label=label)
                ax.fill_between(x, df.ci_lo.values, df.ci_hi.values, color=c, alpha=0.15)
            ax.axhline(0, color="k", lw=0.5)
            ax.set_title(f"layer {l}")
            ax.set_xticks(x, BIN_LABELS, rotation=60, fontsize=7)
        axes[0, 0].legend(fontsize=8)
        fig.supxlabel("distance bin d = t - j")
        fig.supylabel("Spearman(attn head-mean, expert overlap), mean over seqs, 95% CI")
        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(RESULTS / fname, dpi=100)
        plt.close(fig)

    grid("Within-bin correlation per layer (eval split, all domains)", "step1_corr_vs_distance.png",
         lambda l: [(f"sink {s}, {k} tokens", ev[(ev.domain == "all") & (ev.layer == l) & (ev.sink == s)
                     & (ev.tokens == k)], st) for (s, k), st in styles.items()])
    grid("Within-bin correlation per domain (eval, sink excl, distinct tokens)", "step1_corr_by_domain.png",
         lambda l: [(d, ev[(ev.domain == d) & (ev.layer == l) & (ev.sink == "excl") & (ev.tokens == "distinct")],
                     (f"C{i}", "-", 1.5)) for i, d in enumerate(["all", *data.DOMAINS])])

    main = ev[(ev.domain == "all") & (ev.sink == "excl") & (ev.tokens == "distinct")]
    fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))
    for ax, col, title in [(axes[0], "rho", "Spearman"), (axes[1], "partial_rho", "partial Spearman | d")]:
        M = main.pivot(index="layer", columns="d_lo", values=col).values
        im = ax.imshow(M, aspect="auto", cmap="RdBu_r", vmin=-0.5, vmax=0.5)
        ax.set(xticks=x, yticks=range(L), xlabel="distance bin", ylabel="layer",
               title=f"{title} (eval, sink excl, distinct)")
        ax.set_xticklabels(BIN_LABELS, rotation=60)
        fig.colorbar(im, ax=ax)
    M = main.pivot(index="layer", columns="d_lo", values="mean_overlap").values
    im = axes[2].imshow(M, aspect="auto", cmap="viridis")
    axes[2].set(xticks=x, yticks=range(L), xlabel="distance bin", ylabel="layer",
                title="mean expert overlap (temporal locality)")
    axes[2].set_xticklabels(BIN_LABELS, rotation=60)
    fig.colorbar(im, ax=axes[2])
    fig.tight_layout()
    fig.savefig(RESULTS / "step1_layer_bin_heatmap.png", dpi=110)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    vmax = np.nanmax(np.abs(np.concatenate([sel_heads, eval_heads])))
    for ax, M, t in [(axes[0], sel_heads, "selection"), (axes[1], eval_heads, "eval")]:
        im = ax.imshow(M, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set(xlabel="head", ylabel="layer", xticks=range(M.shape[1]), yticks=range(L),
               title=f"per-head score ({t}): mean rho over bins d>=2, sink excl, distinct")
        fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(RESULTS / "step1_heads_heatmap.png", dpi=110)
    plt.close(fig)

    st = ev[(ev.layer == 0) & (ev.sink == "excl") & (ev.tokens == "all")]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for d in ["all", *data.DOMAINS]:
        ax.plot(x, st[st.domain == d].same_token_frac.values, marker="o", label=d)
    ax.set(xticks=x, xlabel="distance bin", ylabel="fraction of pairs with ids[t] == ids[j]",
           title="Same-token pair fraction (eval, sink excl)")
    ax.set_xticklabels(BIN_LABELS, rotation=60)
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "step1_same_token_frac.png", dpi=110)
    plt.close(fig)


def _fmt(r, col="rho"):
    lo, hi = ("ci_lo", "ci_hi") if col == "rho" else ("partial_ci_lo", "partial_ci_hi")
    return f"{r[col]:+.3f} [{r[lo]:+.3f}, {r[hi]:+.3f}]"


def write_md(corr, heads, top, raw, sel, L):
    ev = corr[corr.split == "eval"]
    main = ev[(ev.domain == "all") & (ev.sink == "excl") & (ev.tokens == "distinct")]
    n_eval = int((~sel).sum())
    lines = [
        "# Step 1: does attention predict expert overlap beyond distance?",
        "",
        f"{len(raw['seq'])} sequences x 1024 tokens ({n_eval} eval / {int(sel.sum())} selection, stratified "
        "by domain, `results/split.json`). Pairs sampled per sequence and distance bin (all d = 1 pairs, "
        f"else up to {PAIRS_PER_BIN}), same pairs for every layer. Spearman per sequence, then mean over "
        f"sequences with a {N_BOOT}-resample bootstrap 95% CI over sequences. "
        "**Main variant: sink (j = 0) excluded, distinct tokens (ids[t] != ids[j]).** "
        "All numbers below are on the eval split unless stated. Full table: `step1_corr.csv`.",
        "",
        "`partial` = partial Spearman controlling for the exact distance d within the bin "
        "(wide bins still contain distance variation, which drives both attention and overlap).",
        "",
        "## Head-mean attention, main variant, all domains (eval)",
        "",
        "Spearman rho [95% CI] per layer, selected bins:",
        "",
    ]
    show = ["1", "2", "5-8", "33-64", "513-1023"]
    lines += ["| layer | " + " | ".join(f"d={b}" for b in show) + " | mean over bins d>=2 | partial, mean d>=2 |",
              "|---|" + "---|" * (len(show) + 2)]
    for l in range(L):
        r = main[main.layer == l].set_index("bin")
        m2 = r[r.d_lo >= 2]
        lines.append(f"| {l} | " + " | ".join(_fmt(r.loc[b]) for b in show)
                     + f" | {m2.rho.mean():+.3f} | {m2.partial_rho.mean():+.3f} |")
    lines += ["", "![corr vs distance](step1_corr_vs_distance.png)", "",
              "![layer x bin](step1_layer_bin_heatmap.png)", ""]

    # Variant comparison
    lines += ["## Effect of the controls (eval, all domains, mean over layers and bins d>=2)", "",
              "| sink | tokens | mean rho | mean partial rho |", "|---|---|---|---|"]
    for s in SINK:
        for k in TOKENS:
            v = ev[(ev.domain == "all") & (ev.sink == s) & (ev.tokens == k) & (ev.d_lo >= 2)]
            lines.append(f"| {s} | {k} | {v.rho.mean():+.3f} | {v.partial_rho.mean():+.3f} |")
    st = ev[(ev.layer == 0) & (ev.sink == "excl") & (ev.tokens == "all")]
    lines += ["", "Same-token pair fraction per bin (sink excl): " + ", ".join(
        f"{d}: " + " / ".join(f"{v:.3f}" for v in st[st.domain == d].same_token_frac.values)
        for d in ["all", *data.DOMAINS]) + f" (bins {', '.join(BIN_LABELS)}).",
        "", "![same token](step1_same_token_frac.png)", ""]

    # Per domain
    lines += ["## Per domain (eval, main variant, mean over bins d>=2)", "",
              "| layer | web | code | math |", "|---|---|---|---|"]
    for l in range(L):
        vals = []
        for d in data.DOMAINS:
            v = ev[(ev.domain == d) & (ev.layer == l) & (ev.sink == "excl") & (ev.tokens == "distinct")
                   & (ev.d_lo >= 2)]
            vals.append(f"{v.rho.mean():+.3f}")
        lines.append(f"| {l} | " + " | ".join(vals) + " |")
    lines += ["", "![by domain](step1_corr_by_domain.png)", ""]

    # Heads
    hm = heads[heads["head"] == "mean"].set_index("layer")
    lines += ["## Per head", "",
              "Score = mean rho over bins d >= 2 (sink excl, distinct tokens). Ranked on the selection split; "
              "eval values with bootstrap CI. Head-mean score of the same layer for reference.", "",
              "| sel rank | layer | head | sel score | eval score [95% CI] | eval web / code / math | layer head-mean (eval) |",
              "|---|---|---|---|---|---|---|"]
    for r in top.itertuples():
        lines.append(f"| {r.sel_rank} | {r.layer} | {r.head} | {r.sel_score:+.3f} | {r.eval_score:+.3f} "
                     f"[{r.eval_ci_lo:+.3f}, {r.eval_ci_hi:+.3f}] | {r.eval_web:+.3f} / {r.eval_code:+.3f} / "
                     f"{r.eval_math:+.3f} | {hm.loc[r.layer, 'eval_score']:+.3f} |")
    ind = heads[heads["head"] != "mean"]
    rs = ind[["sel_score", "eval_score"]].corr(method="spearman").iloc[0, 1]
    lines += ["", f"Selection-vs-eval rank agreement over all {len(ind)} heads: Spearman {rs:.3f}.", "",
              "![heads](step1_heads_heatmap.png)", ""]
    (RESULTS / "STEP1_tables.md").write_text("\n".join(lines) + "\n")


def run(limit=None):
    raw = collect(limit)
    return aggregate(raw)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=None, help="approx. number of sequences (pipeline test)")
    p.add_argument("--aggregate-only", action="store_true")
    a = p.parse_args()
    aggregate() if a.aggregate_only else run(a.limit)

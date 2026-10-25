"""Step 0 sanity checks: load, router top-8 vs. actual expert choice, attention via hooks, usage."""

import json
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from . import data
from .common import RESULTS, capture, load_model, seed_all

GB = 1024**3
N_USAGE_PER_DOMAIN = 10


def attn_checks(l, w):
    """Reduce one layer's attention [1, H, n, n] to scalar checks plus head-mean sink mass."""
    w = w[0].float()
    n = w.shape[-1]
    upper = torch.triu(torch.ones(n, n, dtype=torch.bool, device=w.device), diagonal=1)
    return {
        "shape": list(w.shape),
        "dtype": "bf16->f32",
        "row_sum_max_abs_dev": (w.sum(-1) - 1).abs().max().item(),
        "upper_tri_max": w[:, upper].max().item(),
        "min": w.min().item(),
        "sink_mass_mean": w[:, 1:, 0].mean().item(),  # mean attention to position 0, over heads, t>=1
        "diag_mass_mean": w.diagonal(dim1=-2, dim2=-1).mean().item(),
    }


def run():
    seed_all()
    RESULTS.mkdir(exist_ok=True)
    ids, domains = data.load(1024)
    out = {}

    # 1. Load + forward on one sequence
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    model = load_model()
    cfg = model.config
    out["load"] = {
        "seconds": round(time.time() - t0, 1),
        "mem_allocated_gb": round(torch.cuda.memory_allocated() / GB, 2),
        "peak_gb": round(torch.cuda.max_memory_allocated() / GB, 2),
        "config": {k: getattr(cfg, k) for k in [
            "num_hidden_layers", "num_experts", "num_experts_per_tok", "norm_topk_prob",
            "num_attention_heads", "num_key_value_heads", "hidden_size"]},
        "experts_implementation": getattr(cfg, "_experts_implementation", None),
        "attn_implementation": cfg._attn_implementation,
    }
    L, E, K = cfg.num_hidden_layers, cfg.num_experts, cfg.num_experts_per_tok

    x = torch.tensor(ids[:1], dtype=torch.long, device="cuda")
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad(), capture(model, attn_fn=attn_checks) as rec:
        o = model(x, labels=x, output_router_logits=True)
    out["forward_1x1024"] = {
        "domain": str(domains[0]),
        "loss": round(o.loss.item(), 3),
        "peak_gb": round(torch.cuda.max_memory_allocated() / GB, 2),
    }

    # 2. Router logits: shape + top-8 vs. the experts the model actually used
    rl = o.router_logits
    out["router_logits"] = {"type": type(rl).__name__, "len": len(rl), "shape": list(rl[0].shape),
                            "dtype": str(rl[0].dtype)}
    match = []
    for l in range(L):
        logits = rl[l].float()  # [n, E]
        top_from_logits = torch.topk(logits, K, dim=-1).indices
        used = rec["experts"][l]
        same_set = (torch.sort(top_from_logits, -1).values == torch.sort(used, -1).values).all(-1)
        # Weights: softmax over all 64, gather top-8, renormalize only if norm_topk_prob
        p = torch.softmax(logits, -1).gather(-1, used)
        if cfg.norm_topk_prob:
            p = p / p.sum(-1, keepdim=True)
        w = rec["weights"][l].float()
        match.append({
            "layer": l,
            "set_match_frac": same_set.float().mean().item(),
            "weight_max_abs_err": (p - w).abs().max().item(),
            "used_weight_sum_mean": w.sum(-1).mean().item(),
        })
    out["router_match"] = match
    out["router_match_all"] = all(m["set_match_frac"] == 1.0 for m in match)

    # 3. Attention checks from hooks
    out["attention"] = [{"layer": l, **rec["attn"][l]} for l in range(L)]

    # Memory headroom at 2048 tokens (two sequences concatenated, just to measure)
    x2 = torch.tensor(np.concatenate([ids[0], ids[1]])[None], dtype=torch.long, device="cuda")
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad(), capture(model, attn_fn=lambda l, w: w[0].float().mean(0).half()):
        model(x2)
    out["forward_1x2048_peak_gb"] = round(torch.cuda.max_memory_allocated() / GB, 2)
    del rec, o

    # 4. Expert usage histogram over 10 sequences per domain
    usage = {d: np.zeros((L, E), dtype=np.int64) for d in data.DOMAINS}
    for d in data.DOMAINS:
        for i in np.where(domains == d)[0][:N_USAGE_PER_DOMAIN]:
            x = torch.tensor(ids[i : i + 1], dtype=torch.long, device="cuda")
            with torch.no_grad(), capture(model) as rec:
                model(x)
            for l in range(L):
                usage[d][l] += np.bincount(rec["experts"][l].flatten().cpu().numpy(), minlength=E)
    total = sum(usage.values())
    np.savetxt(RESULTS / "step0_usage_counts.csv", total, fmt="%d", delimiter=",",
               header=f"rows=layers 0..{L-1}, cols=experts 0..{E-1}; {3*N_USAGE_PER_DOMAIN} seqs x 1024 tok")
    stats = []
    for l in range(L):
        f = total[l] / total[l].sum()
        nz = f[f > 0]
        stats.append({
            "layer": l,
            "unused_experts": int((f == 0).sum()),
            "max_frac": round(float(f.max()), 4),
            "min_frac": round(float(f.min()), 4),
            "max_over_uniform": round(float(f.max() * E), 2),
            "norm_entropy": round(float(-(nz * np.log(nz)).sum() / np.log(E)), 4),
            **{f"norm_entropy_{d}": round(float(_ent(usage[d][l])), 4) for d in usage},
        })
    out["usage"] = stats
    plot_usage(total, usage)

    (RESULTS / "step0.json").write_text(json.dumps(out, indent=2))
    write_md(out)
    return out


def _ent(c):
    f = c / c.sum()
    f = f[f > 0]
    return -(f * np.log(f)).sum() / np.log(len(c))


def plot_usage(total, usage):
    L, E = total.shape
    frac = total / total.sum(1, keepdims=True)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), gridspec_kw={"width_ratios": [3, 2]})
    im = axes[0].imshow(frac * E, aspect="auto", cmap="viridis", vmin=0)
    axes[0].set(xlabel="expert", ylabel="layer", title="Expert usage / uniform (all domains, 30 seqs)")
    fig.colorbar(im, ax=axes[0])
    for d, c in usage.items():
        ent = [_ent(c[l]) for l in range(L)]
        axes[1].plot(range(L), ent, marker="o", label=d)
    axes[1].set(xlabel="layer", ylabel="normalized entropy (1 = uniform)",
                title="Usage entropy per layer", ylim=(0.8, 1.0))
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "step0_usage.png", dpi=120)
    plt.close(fig)


def write_md(o):
    ld = o["load"]
    lines = [
        "# Step 0: sanity checks",
        "",
        f"Model `allenai/OLMoE-1B-7B-0125`, bf16, attn=`{ld['attn_implementation']}`, "
        f"experts impl=`{ld['experts_implementation']}`. Config: `{ld['config']}`.",
        "",
        "## 1. Load and forward",
        f"- Load: {ld['seconds']} s, {ld['mem_allocated_gb']} GB allocated after load.",
        f"- Forward 1x1024 ({o['forward_1x1024']['domain']}), with router logits and attention hooks: "
        f"peak {o['forward_1x1024']['peak_gb']} GB, LM loss {o['forward_1x1024']['loss']}.",
        f"- Forward 1x2048 (headroom check): peak {o['forward_1x2048_peak_gb']} GB of 24 GB.",
        "- Peak memory from `torch.cuda.max_memory_allocated` (nvidia-smi/NVML is broken on this host: "
        "driver/library version mismatch; CUDA itself works).",
        "",
        "## 2. Router top-8 vs. actual expert choice",
        f"- `output_router_logits`: {o['router_logits']['type']} of {o['router_logits']['len']} tensors, "
        f"each {o['router_logits']['shape']} ({o['router_logits']['dtype']}) = [tokens, experts], pre-softmax.",
        "- Ground truth: `top_k_index` captured by a forward pre-hook on each layer's `mlp.experts`.",
        "- Router (modeling_olmoe.py): softmax(logits) -> top-8; top-8 of softmax == top-8 of logits. "
        f"`norm_topk_prob={ld['config']['norm_topk_prob']}`, so mixing weights are raw softmax probs "
        "(not renormalized); this affects weights only, not the expert set.",
        f"- **Set match all layers, all tokens: {o['router_match_all']}**",
        "",
        "| layer | set match | weight max abs err | mean sum of top-8 weights |",
        "|---|---|---|---|",
    ]
    lines += [f"| {m['layer']} | {m['set_match_frac']:.4f} | {m['weight_max_abs_err']:.2e} | "
              f"{m['used_weight_sum_mean']:.3f} |" for m in o["router_match"]]
    lines += [
        "",
        "## 3. Attention via hooks",
        "Forward hook on each `self_attn` gets `attn_weights` `[1, H, n, n]` (bf16), reduces it, "
        "and returns `None` in its place so the full tensor is freed.",
        "",
        "| layer | shape [H,n,n] | max row-sum dev | max upper-tri | mean attn to pos 0 | mean diag |",
        "|---|---|---|---|---|---|",
    ]
    lines += [f"| {a['layer']} | {a['shape']} | {a['row_sum_max_abs_dev']:.2e} | {a['upper_tri_max']:.1e} | "
              f"{a['sink_mass_mean']:.3f} | {a['diag_mass_mean']:.3f} |" for a in o["attention"]]
    lines += [
        "",
        "## 4. Expert usage (30 seqs = 10/domain x 1024 tokens)",
        "![usage](step0_usage.png)",
        "",
        "| layer | unused | max frac | min frac | max/uniform | norm. entropy | web | code | math |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    lines += [f"| {s['layer']} | {s['unused_experts']} | {s['max_frac']} | {s['min_frac']} | "
              f"{s['max_over_uniform']} | {s['norm_entropy']} | {s['norm_entropy_web']} | "
              f"{s['norm_entropy_code']} | {s['norm_entropy_math']} |" for s in o["usage"]]
    lines += [
        "",
        "## Notes",
        "- Data: 100 docs/domain, one document per sequence, `<|endoftext|>` + first 1023 tokens; "
        "shorter docs skipped. web = FineWeb-Edu `sample-10BT`, code = `codeparrot/codeparrot-clean-valid`, "
        "math = `open-web-math/open-web-math`, all streamed (no substitutions needed). "
        "The tokenizer adds no BOS, hence the prepended `<|endoftext|>`.",
        "- Row-sum deviation (~3e-3) and router weight error (~2e-3) are bf16 rounding: eager attention "
        "computes softmax in fp32 then casts to bf16. Upper triangle is exactly 0, so masking is causal.",
        "- Attention to position 0 rises with depth (up to ~0.57 mean in layer 14), so the j=0 exclusion "
        "run in Step 1 matters most for late layers.",
        "- Code routing is clearly more concentrated than web/math (norm. entropy ~0.81-0.85 from layer 4 on), "
        "so frequency-based baselines will be stronger on code.",
        "- 2048 tokens fit easily (peak ~14 GB); kept 1024 per TASK.md unless the reviewer asks otherwise.",
    ]
    (RESULTS / "STEP0.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    run()

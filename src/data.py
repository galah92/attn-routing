"""Build fixed-length token sequences from three domains via streaming (no full downloads).

Each sequence is a single document: <|endoftext|> + the document's first SEQ_LEN-1 tokens.
Documents shorter than that are skipped. The tokenizer adds no BOS, so the leading
<|endoftext|> (OLMo's document separator) gives every sequence the same position-0 token.
"""

import argparse
import json

import numpy as np
from datasets import load_dataset

from .common import DATA, load_tokenizer

DOMAINS = {
    "web": ("HuggingFaceFW/fineweb-edu", "sample-10BT", "text"),
    "code": ("codeparrot/codeparrot-clean-valid", None, "content"),
    "math": ("open-web-math/open-web-math", None, "text"),
}
MAX_SCAN = 20000  # documents scanned per domain before giving up


def build(n_per_domain=100, seq_len=1024):
    tok = load_tokenizer()
    ids, domains, meta = [], [], {}
    for dom, (name, cfg, field) in DOMAINS.items():
        ds = load_dataset(name, cfg, split="train", streaming=True)
        got = scanned = 0
        for ex in ds:
            scanned += 1
            if scanned > MAX_SCAN:
                break
            toks = tok(ex[field], add_special_tokens=False).input_ids
            if len(toks) < seq_len - 1:
                continue
            ids.append([tok.eos_token_id] + toks[: seq_len - 1])
            domains.append(dom)
            got += 1
            if got == n_per_domain:
                break
        meta[dom] = {"dataset": name, "config": cfg, "kept": got, "scanned": scanned}
        print(f"{dom}: kept {got} of {scanned} scanned docs from {name}")
    DATA.mkdir(exist_ok=True)
    path = DATA / f"seqs_{seq_len}.npz"
    np.savez(path, ids=np.array(ids, dtype=np.int32), domains=np.array(domains))
    (DATA / f"seqs_{seq_len}.json").write_text(json.dumps(meta, indent=2))
    return path


def load(seq_len=1024):
    d = np.load(DATA / f"seqs_{seq_len}.npz")
    return d["ids"], d["domains"]


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n-per-domain", type=int, default=100)
    p.add_argument("--seq-len", type=int, default=1024)
    a = p.parse_args()
    build(a.n_per_domain, a.seq_len)

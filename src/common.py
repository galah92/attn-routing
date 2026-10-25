"""Shared model loading and hook-based capture of routing + attention."""

import random
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "allenai/OLMoE-1B-7B-0125"
ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
DATA = ROOT / "data"
SEED = 0


def seed_all(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_ID)


def load_model():
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, attn_implementation="eager", device_map="cuda"
    )
    model.eval()
    return model


@contextmanager
def capture(model, attn_fn=None, keep_router_input=False):
    """Register hooks on every decoder layer for one forward pass.

    Records into the yielded dict:
      experts[l]: top_k_index [n, k] actually passed to layer l's experts module
      weights[l]: top_k_weights [n, k] actually used to mix expert outputs
      attn[l]:    attn_fn(l, attn_weights[B, H, n, n]) if attn_fn is given
      router_in[l]: hidden state fed to layer l's router (post-attention RMSNorm output)
    The attention tensor is reduced by attn_fn inside the hook and then dropped, so only
    the reduction is kept in memory.
    """
    rec = {"experts": {}, "weights": {}, "attn": {}, "router_in": {}}
    handles = []
    for l, layer in enumerate(model.model.layers):

        def experts_pre(mod, args, kwargs, l=l):
            names = ["hidden_states", "top_k_index", "top_k_weights"]
            vals = dict(zip(names, args)) | kwargs
            rec["experts"][l] = vals["top_k_index"].detach()
            rec["weights"][l] = vals["top_k_weights"].detach()
            if keep_router_input:
                rec["router_in"][l] = vals["hidden_states"].detach()

        handles.append(layer.mlp.experts.register_forward_pre_hook(experts_pre, with_kwargs=True))

        if attn_fn is not None:

            def attn_post(mod, args, output, l=l):
                attn_out, attn_w = output
                assert attn_w is not None, "attention weights missing; need eager attention"
                rec["attn"][l] = attn_fn(l, attn_w.detach())
                return attn_out, None

            handles.append(layer.self_attn.register_forward_hook(attn_post))
    try:
        yield rec
    finally:
        for h in handles:
            h.remove()

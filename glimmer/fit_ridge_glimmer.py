"""Closed-form bridge from Llama 3.2 1B (reader) to Muse Glimmer 30B (solver), across two tokenizers.

The Llama-to-Llama bridge regresses the reader's hidden state for a token onto the solver's embedding of
the SAME token id. Glimmer has its own tokenizer (202k vocab), so the target is matched by character
span instead: for each Llama token, the mean of Glimmer's (RMS-normalized) input embeddings of the
Glimmer tokens that overlap the same characters of the context. Then the same ridge fit as fit_ridge.py.

Runs on the laptop CPU: needs data/hidden_train.pt and data/glimmer_embed.pt (glimmer/fetch_embed.py).

python glimmer/fit_ridge_glimmer.py   ->  checkpoints/ridge_glimmer_L8.pt, checkpoints/bridge_glimmer_ridge.pt
"""
import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bridge import TokenBridge  # noqa: E402
from common import CKPT, DATA, SENDER_ID, SENDER_INSTR, USER_HDR, content_span, sender_prompt  # noqa: E402

GLIMMER_ID = "meta-models/Muse-Glimmer-30B"


def rms_norm(x, eps=1e-5):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


def llama_spans(ltok, context):
    """(start, end) character spans, in context coordinates, of the reader's content tokens."""
    p = sender_prompt(context)
    enc = ltok(p, add_special_tokens=False, return_offsets_mapping=True)
    s, e = content_span(ltok, len(enc.input_ids) - 1)
    off = enc.offset_mapping[1 + s:1 + e]  # same slice as common.content_token_ids
    base = len(USER_HDR + SENDER_INSTR)
    return [(max(0, a - base), min(len(context), b - base)) for a, b in off]


def glimmer_targets(gtok, E, context, spans):
    """Mean normalized Glimmer embedding over the Glimmer tokens overlapping each Llama span."""
    enc = gtok(context, add_special_tokens=False, return_offsets_mapping=True)
    gids, goff = enc.input_ids, enc.offset_mapping
    rows, first = [], []
    for a, b in spans:
        hit = [k for k, (ga, gb) in enumerate(goff) if ga < max(b, a + 1) and gb > a]
        if not hit:  # zero-width / whitespace-only span: take the nearest Glimmer token
            hit = [min(range(len(goff)), key=lambda k: abs(goff[k][0] - a))]
        rows.append(rms_norm(E[[gids[k] for k in hit]].float()).mean(0))
        first.append(gids[hit[0]])
    return torch.stack(rows), first


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=8)
    ap.add_argument("--n", type=int, default=1450)
    ap.add_argument("--lam", type=float, default=10.0)
    args = ap.parse_args()

    E = torch.load(DATA / "glimmer_embed.pt")  # bf16 [202048, 6656]
    ltok = AutoTokenizer.from_pretrained(SENDER_ID)
    gtok = AutoTokenizer.from_pretrained(GLIMMER_ID)
    rows = json.loads((DATA / "pairs.json").read_text(encoding="utf-8"))
    hid = torch.load(DATA / "hidden_train.pt", weights_only=False, mmap=True)
    X, Y, first, multi = [], [], [], 0
    for i in range(min(args.n, len(rows) - 50)):
        s, e = content_span(ltok, hid["ctx_len"][i])
        x = hid["layers"][args.layer][i][s:e].float()
        spans = llama_spans(ltok, rows[i]["context"])
        assert len(spans) == len(x), (i, len(spans), len(x))
        y, f = glimmer_targets(gtok, E, rows[i]["context"], spans)
        X.append(x); Y.append(y); first += f
    X, Y, first = torch.cat(X), torch.cat(Y), torch.tensor(first)
    print(f"{len(X)} aligned tokens", flush=True)
    mu, sd = X.mean(0), X.std(0).clamp_min(1e-3)
    A = torch.cat([(X - mu) / sd, torch.ones(len(X), 1)], 1)
    n = int(len(A) * 0.9)
    W = torch.linalg.solve(A[:n].T @ A[:n] + args.lam * torch.eye(A.shape[1]), A[:n].T @ Y[:n])
    P = A[n:] @ W
    cos = F.cosine_similarity(P, Y[n:], dim=-1).mean().item()
    # nearest Glimmer token for the first 1,000 held-out positions, vocab in chunks (keeps RAM small)
    Pn, m = F.normalize(P[:1000], dim=-1), min(1000, len(P))
    best_s, best_i = torch.full((m,), -2.0), torch.zeros(m, dtype=torch.long)
    for v in range(0, E.shape[0], 32768):
        sc = Pn @ F.normalize(E[v:v + 32768].float(), dim=-1).T
        s_, i_ = sc.max(-1)
        upd = s_ > best_s
        best_s[upd], best_i[upd] = s_[upd], i_[upd] + v
    top1 = best_i.eq(first[n:n + m]).float().mean().item()
    ridge = {"W": W, "mu": mu, "sd": sd, "layer": args.layer, "heldout_cos": cos, "heldout_top1": top1,
             "receiver": GLIMMER_ID}
    torch.save(ridge, CKPT / f"ridge_glimmer_L{args.layer}.pt")
    print(f"layer {args.layer}: held-out cos {cos:.3f} | nearest Glimmer token top-1 {top1:.1%}", flush=True)

    # Ridge-only bridge (no training), same as the Llama ridge-only ablation; vectors at Glimmer's
    # embedding scale (RMS 1 -> L2 norm sqrt(d)).
    br = TokenBridge(X.shape[1], E.shape[1], stride=1, target_norm=math.sqrt(E.shape[1]))
    br.init_from_ridge(ridge)
    br.save(CKPT / "bridge_glimmer_ridge.pt", layer=args.layer, src="ctx", teacher="none", skip_template=True,
            content_only=True, arch="token", receiver=GLIMMER_ID, sender=SENDER_ID, ridge_only=True,
            heldout_cos=cos, heldout_top1=top1)
    print("saved checkpoints/bridge_glimmer_ridge.pt", flush=True)


if __name__ == "__main__":
    main()

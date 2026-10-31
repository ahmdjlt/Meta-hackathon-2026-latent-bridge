"""Closed-form initialization for the token bridge.

Fits a ridge regression from standardized sender hidden states (layer L, context tokens only) to the
receiver's input embeddings of the same tokens (the two Llama 3.2 models share a tokenizer).
Saves checkpoints/ridge_L{L}.pt = {W (d_in+1 x d_out, last row = bias), mu, sd, heldout_cos, heldout_top1}.

python fit_ridge.py --layer 8
"""
import argparse
import glob
import json

import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import AutoTokenizer

from common import CKPT, DATA, RECEIVER_ID, SENDER_ID, content_span, content_token_ids


def receiver_embeddings():
    snap = snapshot_download(RECEIVER_ID, allow_patterns=["*.json", "*.safetensors"])
    idx = json.load(open(f"{snap}/model.safetensors.index.json"))["weight_map"]
    with safe_open(f"{snap}/{idx['model.embed_tokens.weight']}", "pt") as f:
        return f.get_tensor("model.embed_tokens.weight").float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=8)
    ap.add_argument("--n", type=int, default=1450, help="training rows used (last 50 are validation)")
    ap.add_argument("--lam", type=float, default=10.0)
    args = ap.parse_args()

    E = receiver_embeddings()
    tok = AutoTokenizer.from_pretrained(SENDER_ID)
    rows = json.loads((DATA / "pairs.json").read_text(encoding="utf-8"))
    hid = torch.load(DATA / "hidden_train.pt", weights_only=False, mmap=True)
    X, ids = [], []
    for i in range(min(args.n, len(rows) - 50)):
        s, e = content_span(tok, hid["ctx_len"][i])
        t = content_token_ids(tok, rows[i]["context"])
        x = hid["layers"][args.layer][i][s:e].float()
        assert len(t) == len(x)
        X.append(x)
        ids += t
    X, ids = torch.cat(X), torch.tensor(ids)
    Y = E[ids]
    mu, sd = X.mean(0), X.std(0).clamp_min(1e-3)
    A = torch.cat([(X - mu) / sd, torch.ones(len(X), 1)], 1)
    n = int(len(A) * 0.9)  # hold out the last 10% of tokens to report fit quality
    W = torch.linalg.solve(A[:n].T @ A[:n] + args.lam * torch.eye(A.shape[1]), A[:n].T @ Y[:n])
    P = A[n:] @ W
    cos = F.cosine_similarity(P, Y[n:], dim=-1).mean().item()
    En = F.normalize(E, dim=-1)
    top1 = sum((F.normalize(P[j:j + 512], dim=-1) @ En.T).argmax(-1).eq(ids[n:][j:j + 512]).sum().item()
               for j in range(0, len(P), 512)) / len(P)
    torch.save({"W": W, "mu": mu, "sd": sd, "layer": args.layer, "heldout_cos": cos, "heldout_top1": top1},
               CKPT / f"ridge_L{args.layer}.pt")
    print(f"layer {args.layer}: {n} tokens | held-out cos {cos:.3f} | nearest-token top-1 {top1:.1%}")


if __name__ == "__main__":
    main()

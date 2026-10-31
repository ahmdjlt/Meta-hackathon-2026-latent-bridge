"""Check: can the receiver solve problems from closed-form (ridge) mapped sender states?

  oracle : receiver's own embeddings of the context tokens, inserted as soft tokens (upper bound)
  ridge  : sender layer-L states -> linear map fit by ridge regression -> soft tokens
"""
import json
import sys

import torch
import torch.nn.functional as F

from common import DATA, DEVICE, RECEIVER_ID, content_span, content_token_ids, extract_number, load_model
from receiver_io import Receiver

L = int(sys.argv[1]) if len(sys.argv) > 1 else 8
n = int(sys.argv[2]) if len(sys.argv) > 2 else 20
rows = json.loads((DATA / "test.json").read_text(encoding="utf-8"))
hid = torch.load(DATA / "hidden_test.pt", weights_only=False, mmap=True)
rid = torch.load(f"checkpoints/ridge_L{L}.pt")
rec = Receiver(*load_model(RECEIVER_ID))
norm = rec.mean_embed_norm()
hits = {"oracle": 0, "ridge": 0}
for i in range(n):
    r = rows[i]
    ids = content_token_ids(rec.tok, r["context"])
    s, e = content_span(rec.tok, hid["ctx_len"][i])
    x = (hid["layers"][L][i][s:e].float() - rid["mu"]) / rid["sd"]
    ridge = torch.cat([x, torch.ones(len(x), 1)], 1) @ rid["W"]
    lat = {"oracle": rec.embed(ids).float(), "ridge": (F.normalize(ridge, dim=-1) * norm)[None].to(DEVICE)}
    for k, v in lat.items():
        out = rec.generate(rec.latent_inputs(v, r["final_q"])[0], 200)
        hits[k] += extract_number(out) == r["answer"]
    print(f"{i + 1}/{n} " + " ".join(f"{k}={v}" for k, v in hits.items()), flush=True)
print({k: f"{v / n:.0%}" for k, v in hits.items()})

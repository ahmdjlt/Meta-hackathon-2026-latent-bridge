"""Debug: what does the receiver generate from latents, and do latents carry information?"""
import json
import sys

import torch
import torch.nn.functional as F

from bridge import LatentBridge
from common import DATA, DEVICE, RECEIVER_ID, answer_text, extract_number, load_model
from receiver_io import Receiver

ckpt = sys.argv[1]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 4
rows = json.loads((DATA / "test.json").read_text(encoding="utf-8"))
hid = torch.load(DATA / "hidden_test.pt", weights_only=False, mmap=True)
rec = Receiver(*load_model(RECEIVER_ID))
bridge, meta = LatentBridge.load(ckpt, DEVICE)
L = meta["layer"]


@torch.no_grad()
def nll(x, n_prefix, sol):
    ids = torch.tensor(rec.ids(answer_text(sol)), device=DEVICE)
    lg = rec.model(inputs_embeds=x, logits_to_keep=len(ids) + 1).logits[0, :-1].float()
    return F.cross_entropy(lg, ids).item()


tot = {"latent": 0, "shuffled": 0, "random": 0, "text": 0, "none": 0}
for i in range(n):
    r = rows[i]
    h = hid["layers"][L][i][:hid["ctx_len"][i]].unsqueeze(0).to(DEVICE)
    h_other = hid["layers"][L][i + 1][:hid["ctx_len"][i + 1]].unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        lat = bridge(h)
        lat_other = bridge(h_other)
    rnd = F.normalize(torch.randn_like(lat), dim=-1) * bridge.target_norm
    xs = {"latent": rec.latent_inputs(lat, r["final_q"], r["solution"]),
          "shuffled": rec.latent_inputs(lat_other, r["final_q"], r["solution"]),
          "random": rec.latent_inputs(rnd, r["final_q"], r["solution"]),
          "text": rec.text_inputs(r["msg"], r["final_q"], r["solution"])}
    print(f"\n=== {r['id']} gold {r['answer']} | {r['context'][:90]}... Q: {r['final_q']}")
    for k, (x, npre) in xs.items():
        v = nll(x, npre, r["solution"])
        tot[k] += v
        print(f"  NLL[{k}] {v:.3f}")
    out = rec.generate(rec.latent_inputs(lat, r["final_q"])[0], 160)
    print(f"  GEN latent -> {extract_number(out)!r}: {out[:300]!r}")
    print(f"  latent norms {lat.norm(dim=-1).mean():.3f} | pairwise cos between soft tokens "
          f"{(F.normalize(lat[0], dim=-1) @ F.normalize(lat[0], dim=-1).T).mean():.3f} | "
          f"cos to other example {F.cosine_similarity(lat.flatten(), lat_other.flatten(), dim=0):.3f}")
print("\nmean solution NLL:", {k: round(v / n, 3) for k, v in tot.items()})

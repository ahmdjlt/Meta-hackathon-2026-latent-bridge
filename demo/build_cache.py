"""Run the real pipeline once on a few GSM8K test problems and store everything the demos show.

python demo/build_cache.py            -> demo/cache.json   (GPU, ~15 s per problem)

demo_cli.py replays this file, so the recorded demo starts instantly and needs no GPU.
Nothing in results/ or checkpoints/ is written. Muse Spark calls go through the usual on-disk cache.
"""
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import CKPT, DATA, content_token_ids, gold_number  # noqa: E402
from pipeline import Pipeline  # noqa: E402

OUT = Path(__file__).resolve().parent / "cache.json"
# 8 short test problems: 5 where latent beats text, 2 where both are right, 1 where Muse Spark rejects a latent error.
IDS = ["test-0", "test-36", "test-68", "test-32", "test-55", "test-104", "test-3", "test-2"]


@torch.no_grad()
def probe(pipe, lat, facts):
    """Nearest 3B vocabulary token (cosine) for each soft token, next to the context token it stands for."""
    E = pipe.receiver.emb.weight
    sims = F.normalize(lat.to(E.device, E.dtype), dim=-1) @ F.normalize(E, dim=-1).T
    cos, ids = sims.max(-1)
    src = content_token_ids(pipe.sender.tok, facts)
    tok = pipe.receiver.tok
    rows = []
    for i, (c, t) in enumerate(zip(cos.tolist(), ids.tolist())):
        s = src[i] if i < len(src) else None
        rows.append({"nearest": tok.decode([t]), "cos": round(c, 3),
                     "source": tok.decode([s]) if s is not None else "", "match": s == t})
    return rows, src


@torch.no_grad()
def pca_coords(pipe, lat, src, n_bg=1500, seed=0):
    """2-D PCA of (background 3B token embeddings + the message's context-token embeddings + soft tokens).
    Everything is L2-normalized first, so the plot shows directions (what cosine compares)."""
    E = pipe.receiver.emb.weight
    g = random.Random(seed)
    bg_ids = g.sample(range(1000, 120000), n_bg)
    emb_bg = F.normalize(E[bg_ids].float(), dim=-1).cpu()
    emb_src = F.normalize(E[src].float(), dim=-1).cpu()
    soft = F.normalize(lat.float(), dim=-1).cpu()
    pca = PCA(n_components=2, random_state=0).fit(torch.cat([emb_src, soft]).numpy())
    r = lambda x: [[round(float(a), 4), round(float(b), 4)] for a, b in pca.transform(x.numpy())]
    return {"background": r(emb_bg), "tokens": r(emb_src), "soft": r(soft)}


def to_example(pipe, q, gold=None, ex_id="live", verify=True):
    """Run both pipelines on one problem and return the dict the demos display."""
    r = pipe.run(q, verify=verify)
    p, t, l = r["plan"], r["text"], r["latent"]
    lat = l.pop("latents")
    rows, src = probe(pipe, lat, p["facts"])
    ok = (lambda a: a == gold) if gold is not None else (lambda a: None)
    return {"id": ex_id, "question": q, "gold": gold,
            "plan": {k: p.get(k) for k in ("facts", "question", "subtasks", "orchestrator", "orchestrator_s")},
            "text": {**t, "correct": ok(t["answer"])},
            "latent": {**l, "correct": ok(l["answer"]), "final_correct": ok(l.get("final_answer")),
                       "dim": int(lat.shape[1]), "probe": rows},
            "pca": pca_coords(pipe, lat, src)}


def main():
    test = {f"test-{i}": r for i, r in enumerate(json.loads((DATA / "test.json").read_text(encoding="utf-8")))}
    pipe = Pipeline(CKPT / "bridge_main_token.pt")
    pipe.run(test[IDS[0]]["question"], verify=False)  # warm-up (CUDA kernels), not recorded
    out = []
    for i in IDS:
        ex = to_example(pipe, test[i]["question"], gold_number(test[i]["answer"]), i)
        out.append(ex)
        t, l = ex["text"], ex["latent"]
        print(f"{i}: gold {ex['gold']} | text {t['answer']} ({t['total_s']:.1f}s) | latent {l['answer']} "
              f"({l['total_s']:.1f}s) | verify {l['verify']['verdict']}", flush=True)
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()

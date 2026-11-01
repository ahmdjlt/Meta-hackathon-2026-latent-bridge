"""FINAL test touch (run ONCE): score candidate checkpoints on the 500 test problems.

For each --cand TAG=CKPT: latent condition on all 500 test problems using CACHED
sender states (same computation as eval_big.py's live forward; verify with
val_full.py --sanity before running this). Compares against the recorded old
results in results/big_preds.jsonl (never re-runs them, never touches that file).

Outputs: results/ext_test_<TAG>.json (+ .jsonl preds), results/ext_test_summary.json/.md
with accuracy and the paired McNemar test vs the old checkpoint (and vs text).

Example:
  python test_final.py --cand ext_s2=checkpoints/bridge_ext_s2_best.pt --old-cond latent_s2
"""
import argparse
import json
import math
import time

import torch

from bridge import LatentBridge
from common import DATA, DEVICE, RECEIVER_ID, RESULTS, content_span, ctx_start, extract_number, load_model
from receiver_io import Receiver
from train import sender_states


def mcnemar(a, b):
    n01 = sum(1 for x, y in zip(a, b) if not x and y)
    n10 = sum(1 for x, y in zip(a, b) if x and not y)
    n = n01 + n10
    if n == 0:
        return n01, n10, 1.0
    k = min(n01, n10)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return n01, n10, min(1.0, 2 * p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", action="append", required=True, help="TAG=CKPT (repeatable, all run in this one pass)")
    ap.add_argument("--old-cond", required=True, help="big_preds.jsonl cond of the old checkpoint (latent_s1/latent_s2)")
    ap.add_argument("--layer", type=int, default=8)
    ap.add_argument("--max_new", type=int, default=256)
    args = ap.parse_args()
    cands = []
    for c in args.cand:
        tag, _, ckpt = c.partition("=")
        assert tag and ckpt, f"bad --cand {c!r}, want TAG=CKPT"
        cands.append((tag, ckpt))

    rows = json.loads((DATA / "test.json").read_text(encoding="utf-8"))
    hid = torch.load(DATA / "hidden_test.pt", weights_only=False, mmap=True)
    hid["layers"] = {args.layer: hid["layers"][args.layer]}
    old = {}
    for line in (RESULTS / "big_preds.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        old[(r["cond"], r["id"])] = r
    assert all((args.old_cond, r["id"]) in old and ("text", r["id"]) in old for r in rows), "old results incomplete"

    model, tok = load_model(RECEIVER_ID)
    rec = Receiver(model, tok)
    start = ctx_start(tok)
    content_end = lambda n: content_span(tok, n)[1]  # noqa: E731

    def states(i):
        return sender_states(hid, i, args.layer, "ctx", start, content_end)

    summary = {}
    for tag, ckpt in cands:
        bridge, meta = LatentBridge.load(ckpt, DEVICE)
        t0 = time.time()
        outs = []
        with torch.no_grad():
            for j, r in enumerate(rows):
                out = rec.generate(rec.latent_inputs(bridge(states(j)), r["final_q"])[0], args.max_new)
                ok = extract_number(out) == r["answer"]
                outs.append({"id": r["id"], "cond": f"ext_{tag}", "out": out, "correct": ok,
                             "gold": r["answer"], "msg_size": bridge(states(j)).shape[1]})
                if (j + 1) % 50 == 0:
                    print(f"  [{tag}] {j + 1}/{len(rows)} ({time.time() - t0:.0f}s)", flush=True)
        acc = sum(o["correct"] for o in outs) / len(outs)
        new = [o["correct"] for o in outs]
        o_old = [old[(args.old_cond, r["id"])]["correct"] for r in rows]
        o_text = [old[("text", r["id"])]["correct"] for r in rows]
        n01, n10, p = mcnemar(o_old, new)
        t01, t10, tp = mcnemar(o_text, new)
        old_acc = sum(o_old) / len(o_old)
        kept = acc > old_acc and p < 0.05
        summary[tag] = {"ckpt": ckpt, "n": len(rows), "accuracy": acc, "old_cond": args.old_cond,
                        "old_accuracy": old_acc,
                        "mcnemar_vs_old": {"old_wrong_new_right": n01, "old_right_new_wrong": n10, "p_value": p},
                        "mcnemar_vs_text": {"text_wrong_new_right": t01, "text_right_new_wrong": t10, "p_value": tp},
                        "kept": kept, "seconds": round(time.time() - t0)}
        (RESULTS / f"ext_test_{tag}.json").write_text(json.dumps(
            {"tag": tag, **summary[tag], "preds": outs}, indent=1))
        print(f"[{tag}] test {acc:.1%} vs old {old_acc:.1%} | McNemar vs old: {n01}/{n10} p={p:.3g} | kept={kept}",
              flush=True)
        del bridge
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
    (RESULTS / "ext_test_summary.json").write_text(json.dumps(summary, indent=1))
    lines = ["| Candidate | Test acc | Old acc | McNemar vs old (n01/n10/p) | Kept |",
             "|---|---|---|---|---|"]
    for tag, s in summary.items():
        m = s["mcnemar_vs_old"]
        lines.append(f"| {tag} | {s['accuracy']:.1%} | {s['old_accuracy']:.1%} | "
                     f"{m['old_wrong_new_right']}/{m['old_right_new_wrong']}/{m['p_value']:.3g} | {s['kept']} |")
    (RESULTS / "ext_test_summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

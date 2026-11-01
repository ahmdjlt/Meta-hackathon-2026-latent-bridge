"""Measure the Muse Spark verify step on the big eval's latent answers.

For each problem, Muse Spark checks the latent pipeline's worked answer against the full problem. On a
reject, the pipeline retries once over the text channel. Decoding is greedy (deterministic), so the retry's
answer is exactly the text condition's answer already recorded in results/big_preds.jsonl; no GPU is needed.
Verdicts are cached in data/orchestrator_cache.json, so reruns cost nothing.

python verify_eval.py --n 500 --cond latent_s1
Writes results/verify_<cond>.json and .md
"""
import argparse
import json

from common import DATA, RESULTS
from pipeline import Orchestrator


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--cond", default="latent_s1")
    args = ap.parse_args()
    rows = json.loads((DATA / "test.json").read_text(encoding="utf-8"))[:args.n]
    preds = {}
    for line in (RESULTS / "big_preds.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        preds[(r["cond"], r["id"])] = r
    orch = Orchestrator(max_calls=args.n + 10)
    stats = {"n": 0, "rejects": 0, "unavailable": 0, "reject_and_latent_wrong": 0, "reject_and_latent_right": 0,
             "acc_before": 0, "acc_after": 0, "verify_tokens": 0, "verify_s": 0.0}
    details = []
    for k, r in enumerate(rows):
        lat, txt = preds.get((args.cond, r["id"])), preds.get(("text", r["id"]))
        if not lat or not txt:
            continue
        v = orch.verify(r["question"], lat["out"])
        stats["n"] += 1
        stats["acc_before"] += lat["correct"]
        final_correct = lat["correct"]
        if v["verdict"] == "reject":
            stats["rejects"] += 1
            stats["reject_and_latent_wrong" if not lat["correct"] else "reject_and_latent_right"] += 1
            final_correct = txt["correct"]
        elif v["verdict"] == "unavailable":
            stats["unavailable"] += 1
        stats["acc_after"] += final_correct
        stats["verify_tokens"] += (v.get("usage") or {}).get("total_tokens") or 0
        stats["verify_s"] += v.get("verify_s") or 0.0
        details.append({"id": r["id"], "verdict": v["verdict"], "reason": v.get("reason"),
                        "latent_correct": lat["correct"], "text_correct": txt["correct"], "final_correct": final_correct})
        if (k + 1) % 50 == 0:
            print(f"  {k + 1}/{len(rows)} rejects {stats['rejects']}", flush=True)
    n = max(1, stats["n"])
    out = {**stats, "trigger_rate": stats["rejects"] / n, "accuracy_before": stats["acc_before"] / n,
           "accuracy_after": stats["acc_after"] / n,
           "reject_precision": stats["reject_and_latent_wrong"] / max(1, stats["rejects"]),
           "latent_errors_caught": stats["reject_and_latent_wrong"] / max(1, n - stats["acc_before"]),
           "details": details}
    (RESULTS / f"verify_{args.cond}.json").write_text(json.dumps(out, indent=1))
    md = (f"Muse Spark verify on `{args.cond}`, n = {stats['n']}\n\n"
          f"| Metric | Value |\n|---|---|\n"
          f"| Trigger rate (rejects -> text retry) | {out['trigger_rate']:.1%} ({stats['rejects']}) |\n"
          f"| Accuracy before verify | {out['accuracy_before']:.1%} |\n"
          f"| Accuracy after verify + retry | {out['accuracy_after']:.1%} |\n"
          f"| Rejects where the latent answer was wrong | {out['reject_precision']:.1%} |\n"
          f"| Latent errors caught | {out['latent_errors_caught']:.1%} |\n"
          f"| Unavailable (API errors) | {stats['unavailable']} |\n"
          f"| Mean verify cost | {stats['verify_tokens'] / n:.0f} tokens |\n")
    (RESULTS / f"verify_{args.cond}.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()

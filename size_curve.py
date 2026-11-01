"""Accuracy vs message size, on the same held-out problems.

Conditions (message size = positions the solver receives for the message):
  latent_s{S}  : token bridge with stride S (cached reader states; size = vectors sent)
  text_b{N}    : the 1B restates the facts "in at most N tokens", hard-truncated at N generated tokens
  ctx_t{N}     : the original context, truncated to its first N tokens

Resumable: per-problem results go to results/size_curve_preds.jsonl; finished (condition, id) pairs are
skipped on restart. Stride-1/2 latent results can be imported from the live eval prediction files.
Writes results/size_curve.csv and results/size_curve.png.

python size_curve.py --n 100
"""
import argparse
import csv
import json

import torch

from bridge import LatentBridge
from common import (ASSIST_HDR, CKPT, DATA, DEVICE, RECEIVER_ID, RESULTS, SENDER_ID, USER_HDR, content_span,
                    ctx_start, extract_number, load_model)
from receiver_io import Receiver

PREDS = RESULTS / "size_curve_preds.jsonl"


def budget_prompt(context, n):
    words = max(5, round(n * 0.75))
    return (USER_HDR + f"Rewrite the facts in the text below as short notes for a teammate, in at most {n} tokens "
            f"(about {words} words). Keep every number exactly as written. Do not calculate anything. "
            f"Output only the notes, with no introduction.\n\nText: " + context + ASSIST_HDR + "-")


@torch.no_grad()
def budget_message(model, tok, context, n):
    enc = tok(budget_prompt(context, n), return_tensors="pt", add_special_tokens=False).to(DEVICE)
    # the reply is seeded with "-" (counts toward the budget) so no tokens go to a preamble
    out = model.generate(**enc, max_new_tokens=n - 1, do_sample=False, temperature=None, top_p=None,
                         pad_token_id=tok.pad_token_id)
    return "-" + tok.decode(out[0, enc.input_ids.shape[1]:], skip_special_tokens=True).rstrip()


def load_done():
    done = {}
    if PREDS.exists():
        for line in PREDS.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            done[(r["cond"], r["id"])] = r
    return done


def append(rec):
    with open(PREDS, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


def import_live(done, path, cond, size_key="tokens_sent"):
    """Reuse per-problem latent results from an eval.py live run (same greedy decoding)."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r["mode"] == "latent" and (cond, r["id"]) not in done:
            rec = {"cond": cond, "id": r["id"], "size": r[size_key], "correct": r["correct"], "source": path.name}
            append(rec)
            done[(cond, r["id"])] = rec


def summarize(rows_ids):
    done = load_done()
    by = {}
    for (cond, i), r in done.items():
        if i in rows_ids:
            by.setdefault(cond, []).append(r)
    out = []
    for cond, rs in sorted(by.items()):
        fam = cond.split("_")[0]
        out.append({"condition": cond, "family": fam, "n": len(rs),
                    "mean_size": sum(r["size"] for r in rs) / len(rs),
                    "accuracy": sum(r["correct"] for r in rs) / len(rs)})
    with open(RESULTS / "size_curve.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["condition", "family", "n", "mean_size", "accuracy"])
        w.writeheader()
        w.writerows(out)
    return out


def plot(summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker
    names = {"latent": ("Latent (ours)", "#2a78d6", "o"), "text": ("Text message, token budget", "#eb6834", "s"),
             "ctx": ("Original context, truncated", "#1baf7a", "^")}
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    fig.patch.set_facecolor("#fcfcfb"); ax.set_facecolor("#fcfcfb")
    for fam, (label, color, marker) in names.items():
        pts = sorted((r["mean_size"], r["accuracy"]) for r in summary if r["family"] == fam)
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color=color, lw=2, marker=marker, ms=8,
                    markeredgecolor="#fcfcfb", markeredgewidth=1.5, label=label)
    ax.set_xlabel("message size (positions the solver reads)", color="#52514e")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_ylim(0, 0.85)
    ax.set_title(f"Accuracy vs message size (GSM8K, n={summary[0]['n'] if summary else 0})", loc="left", color="#0b0b0b")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(color="#e4e3df", lw=0.8); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(RESULTS / "size_curve.png", dpi=150)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--budgets", default="10,20,40,60")
    ap.add_argument("--strides", default="3:bridge_abl_s3_L8.pt,4:bridge_abl_s4_L8.pt",
                    help="stride:checkpoint pairs evaluated here from cached states")
    ap.add_argument("--max_new", type=int, default=256)
    ap.add_argument("--plot_only", action="store_true")
    args = ap.parse_args()
    rows = json.loads((DATA / "test.json").read_text(encoding="utf-8"))[:args.n]
    ids = {r["id"] for r in rows}
    if args.plot_only:
        plot(summarize(ids))
        return

    done = load_done()
    import_live(done, RESULTS / "preds_live_main.jsonl", "latent_s1")
    import_live(done, RESULTS / "preds_live_s2.jsonl", "latent_s2")

    rec = Receiver(*load_model(RECEIVER_ID))
    hid = torch.load(DATA / "hidden_test.pt", weights_only=False, mmap=True)
    for spec in filter(None, args.strides.split(",")):
        s, ck = spec.split(":")
        cond = f"latent_s{s}"
        if not (CKPT / ck).exists():
            print(f"skip {cond}: {ck} missing", flush=True)
            continue
        bridge, meta = LatentBridge.load(CKPT / ck, DEVICE)
        for i, r in enumerate(rows):
            if (cond, r["id"]) in done:
                continue
            a, b = content_span(rec.tok, hid["ctx_len"][i])
            h = hid["layers"][meta["layer"]][i][a:b].unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                lat = bridge(h)
            out = rec.generate(rec.latent_inputs(lat, r["final_q"])[0], args.max_new)
            append({"cond": cond, "id": r["id"], "size": lat.shape[1], "correct": extract_number(out) == r["answer"], "out": out})
            done[(cond, r["id"])] = True
        print(f"{cond} done", flush=True)

    for n in [int(x) for x in args.budgets.split(",")]:
        cond = f"ctx_t{n}"
        for r in rows:
            if (cond, r["id"]) in done:
                continue
            ids_ = rec.tok(r["context"], add_special_tokens=False).input_ids[:n]
            ctx = rec.tok.decode(ids_)
            out = rec.generate(rec.text_inputs(ctx, r["final_q"])[0], args.max_new)
            append({"cond": cond, "id": r["id"], "size": len(ids_), "correct": extract_number(out) == r["answer"], "out": out})
            done[(cond, r["id"])] = True
        print(f"{cond} done", flush=True)

    sender, stok = load_model(SENDER_ID)
    for n in [int(x) for x in args.budgets.split(",")]:
        cond = f"text_b{n}"
        for k, r in enumerate(rows):
            if (cond, r["id"]) in done:
                continue
            msg = budget_message(sender, stok, r["context"], n)
            size = len(rec.tok(msg, add_special_tokens=False).input_ids)
            out = rec.generate(rec.text_inputs(msg, r["final_q"])[0], args.max_new)
            append({"cond": cond, "id": r["id"], "size": size, "correct": extract_number(out) == r["answer"],
                    "msg": msg, "out": out})
            done[(cond, r["id"])] = True
            if (k + 1) % 25 == 0:
                print(f"  {cond} {k + 1}/{len(rows)}", flush=True)
        print(f"{cond} done", flush=True)

    summary = summarize(ids)
    plot(summary)
    for s in summary:
        print(f"{s['condition']:12s} n={s['n']:3d} size={s['mean_size']:5.1f} acc={s['accuracy']:.1%}")


if __name__ == "__main__":
    main()

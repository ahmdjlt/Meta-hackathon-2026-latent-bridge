"""Phase 3: evaluate on held-out GSM8K test rows.

Live mode (default): sender + receiver + bridge loaded together, each condition run end to end and timed.
  text    : sender writes a text message (greedy, <=96 tokens), receiver reads it      (Baseline A)
  none    : receiver gets only the final question                                       (Baseline B)
  latent  : sender's hidden states -> bridge -> K soft tokens -> receiver               (Ours)
  context : receiver reads the original context text (reference upper bound)
Ablation mode (--ablate ckpt ...): latent condition only, from cached sender states, accuracy only.
"""
import argparse
import json
import time

import torch

from bridge import LatentBridge
from common import (CKPT, DATA, DEVICE, RECEIVER_ID, RESULTS, SENDER_ID, content_span, ctx_start, extract_number, load_model)
from receiver_io import Receiver, Sender


def sync():
    if DEVICE == "cuda":
        torch.cuda.synchronize()


def now():
    sync()
    return time.perf_counter()


def summarize(recs):
    n = len(recs)
    keys = [k for k in recs[0] if isinstance(recs[0][k], (int, float)) and not isinstance(recs[0][k], bool)]
    out = {"n": n, "accuracy": sum(r["correct"] for r in recs) / n}
    for k in keys:
        out[f"mean_{k}"] = sum(r[k] for r in recs) / n
    return out


def run_live(args, rows):
    bridge, meta = LatentBridge.load(args.ckpt, DEVICE)
    sender = Sender(*load_model(SENDER_ID))
    rec = Receiver(*load_model(RECEIVER_ID))
    print(f"loaded all models, VRAM {torch.cuda.memory_allocated() / 2**20 if DEVICE == 'cuda' else 0:.0f} MB")
    layer, src, K = meta["layer"], meta["src"], bridge.config["k"]
    skip = meta.get("skip_template", False)
    d_out = bridge.config["d_out"]
    modes = args.modes.split(",")
    res = {m: [] for m in modes}
    pooled, labels = [], []

    # warm-up so CUDA kernels/compilation do not pollute the first timing
    rec.generate(rec.none_inputs("What is 1+1?")[0], 2)
    sender.hidden("Warm up.", layer)

    for n, r in enumerate(rows):
        gold, fq = r["answer"], r["final_q"]
        msg, msg_s = None, 0.0
        if "text" in modes or src == "full":
            t0 = now()
            msg = sender.message(r["context"])
            msg_s = now() - t0
        if "text" in modes:
            t0 = now()
            x, _ = rec.text_inputs(msg, fq)
            out = rec.generate(x, args.max_new)
            recv_s = now() - t0
            n_tok = len(rec.tok(msg, add_special_tokens=False).input_ids)
            res["text"].append(dict(id=r["id"], out=out, correct=extract_number(out) == gold, sender_s=msg_s,
                                    receiver_s=recv_s, total_s=msg_s + recv_s, tokens_sent=n_tok,
                                    bytes_sent=len(msg.encode()), receiver_input_len=x.shape[1], msg=msg))
        if "latent" in modes:
            t0 = now()
            h = sender.hidden(r["context"], layer, msg if src == "full" else None, skip_template=skip,
                              content_only=meta.get("content_only", False))
            with torch.no_grad():
                lat = bridge(h)
            s_s = now() - t0 + (msg_s if src == "full" else 0.0)
            t0 = now()
            x, _ = rec.latent_inputs(lat, fq)
            out = rec.generate(x, args.max_new)
            recv_s = now() - t0
            res["latent"].append(dict(id=r["id"], out=out, correct=extract_number(out) == gold, sender_s=s_s,
                                      receiver_s=recv_s, total_s=s_s + recv_s, tokens_sent=lat.shape[1],
                                      bytes_sent=lat.shape[1] * d_out * 2, receiver_input_len=x.shape[1]))
            pooled.append(lat[0].mean(0).float().cpu())
            labels.append(r["id"])
        if "none" in modes:
            t0 = now()
            x, _ = rec.none_inputs(fq)
            out = rec.generate(x, args.max_new)
            recv_s = now() - t0
            res["none"].append(dict(id=r["id"], out=out, correct=extract_number(out) == gold, sender_s=0.0,
                                    receiver_s=recv_s, total_s=recv_s, tokens_sent=0, bytes_sent=0,
                                    receiver_input_len=x.shape[1]))
        if "context" in modes:
            t0 = now()
            x, _ = rec.context_inputs(r["context"], fq)
            out = rec.generate(x, args.max_new)
            recv_s = now() - t0
            n_tok = len(rec.tok(r["context"], add_special_tokens=False).input_ids)
            res["context"].append(dict(id=r["id"], out=out, correct=extract_number(out) == gold, sender_s=0.0,
                                       receiver_s=recv_s, total_s=recv_s, tokens_sent=n_tok,
                                       bytes_sent=len(r["context"].encode()), receiver_input_len=x.shape[1]))
        if (n + 1) % 10 == 0:
            print(f"  {n + 1}/{len(rows)} " + " ".join(
                f"{m}={sum(x['correct'] for x in res[m]) / len(res[m]):.2f}" for m in modes), flush=True)

    tag = args.tag or f"live_{args.ckpt.stem}"
    summary = {m: summarize(res[m]) for m in modes}
    summary["_meta"] = {"ckpt": str(args.ckpt.name), "layer": layer, "src": src, "k": K, "sender": SENDER_ID,
                        "receiver": RECEIVER_ID, "n": len(rows),
                        "peak_vram_mb": round(torch.cuda.max_memory_allocated() / 2**20) if DEVICE == "cuda" else 0}
    (RESULTS / f"eval_{tag}.json").write_text(json.dumps(summary, indent=1))
    with open(RESULTS / f"preds_{tag}.jsonl", "w", encoding="utf-8") as f:
        for m in modes:
            for x in res[m]:
                f.write(json.dumps({"mode": m, **x}) + "\n")
    if pooled:
        torch.save({"pooled": torch.stack(pooled), "ids": labels,
                    "correct": [x["correct"] for x in res["latent"]]}, RESULTS / f"latents_{tag}.pt")
    names = {"none": "B: no message", "text": "A: text message", "latent": (f"Ours: latent (token bridge, stride {bridge.config.get('stride', 1)}, L{layer})"
                         if meta.get("arch") == "token" else f"Ours: latent (K={K}, L{layer}, {src})"),
             "context": "Ref: raw context text"}
    lines = ["| Condition | Accuracy | Tokens sent | Bytes sent | Receiver input len | Sender s | Receiver s | Total s |",
             "|---|---|---|---|---|---|---|---|"]
    for m in ["none", "text", "latent", "context"]:
        if m in summary:
            s = summary[m]
            lines.append(f"| {names[m]} | {s['accuracy']:.1%} | {s['mean_tokens_sent']:.1f} | {s['mean_bytes_sent']:.0f} | "
                         f"{s['mean_receiver_input_len']:.1f} | {s['mean_sender_s']:.3f} | {s['mean_receiver_s']:.3f} | {s['mean_total_s']:.3f} |")
    table = "\n".join(lines)
    (RESULTS / f"eval_{tag}.md").write_text(table + "\n")
    print(table)


@torch.no_grad()
def run_ablate(args, rows):
    hid = torch.load(DATA / "hidden_test.pt", weights_only=False, mmap=True)
    rec = Receiver(*load_model(RECEIVER_ID))
    out_rows = []
    for ck in args.ablate:
        bridge, meta = LatentBridge.load(ck, DEVICE)
        hits = 0
        n_sent = 0
        for i, r in enumerate(rows):
            h = hid["layers"][meta["layer"]][i]
            if meta["src"] == "ctx":
                h = h[:hid["ctx_len"][i]]
            if meta.get("content_only"):
                h = h[ctx_start(rec.tok):content_span(rec.tok, hid["ctx_len"][i])[1]]
            elif meta.get("skip_template"):
                h = h[ctx_start(rec.tok):]
            lat = bridge(h.unsqueeze(0).to(DEVICE))
            n_sent += lat.shape[1]
            out = rec.generate(rec.latent_inputs(lat, r["final_q"])[0], args.max_new)
            hits += extract_number(out) == r["answer"]
        acc = hits / len(rows)
        out_rows.append({"ckpt": ck.name, "arch": meta.get("arch", "perceiver"), "stride": bridge.config.get("stride"),
                         "k": bridge.config["k"], "layer": meta["layer"], "src": meta["src"],
                         "accuracy": acc, "mean_vectors_sent": n_sent / len(rows),
                         "val_shuffle_gap": meta.get("val", {}).get("val_shuffle_gap")})
        print(f"  {ck.name}: acc {acc:.1%}", flush=True)
    tag = args.tag or "ablation"
    (RESULTS / f"{tag}.json").write_text(json.dumps(out_rows, indent=1))
    lines = ["| Bridge | Sender layer | Vectors sent (mean) | Accuracy | Val shuffle gap |", "|---|---|---|---|---|"]
    for o in out_rows:
        name = f"token, stride {o['stride']}" if o["arch"] == "token" else f"perceiver, K={o['k']}"
        gap = "n/a" if o["val_shuffle_gap"] is None else f"{o['val_shuffle_gap']:.3f}"
        lines.append(f"| {name} | {o['layer']} | {o['mean_vectors_sent']:.1f} | {o['accuracy']:.1%} | {gap} |")
    (RESULTS / f"{tag}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    from pathlib import Path
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=CKPT / "bridge_k16_L16_ctx.pt")
    ap.add_argument("--modes", default="none,text,latent,context")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--max_new", type=int, default=256)
    ap.add_argument("--ablate", type=Path, nargs="*")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    rows = json.loads((DATA / "test.json").read_text(encoding="utf-8"))[:args.n]
    if args.ablate:
        run_ablate(args, rows)
    else:
        run_live(args, rows)


if __name__ == "__main__":
    main()

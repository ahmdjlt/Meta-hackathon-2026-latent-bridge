"""Score a bridge checkpoint by free-generation accuracy on the FULL 200-problem val set.

Uses cached sender states (data/hidden_val.pt): no sender model on GPU, same
computation as live forward (verify once with --sanity, which loads the sender
and compares latents on a few problems).

Appends one line per scored checkpoint to results/ext_val.jsonl:
  {tag, ckpt, n, accuracy, ts}
"""
import argparse
import json
import time

import torch

from bridge import LatentBridge
from common import DATA, DEVICE, RECEIVER_ID, RESULTS, SENDER_ID, content_span, ctx_start, extract_number, load_model
from receiver_io import Receiver, Sender
from train import sender_states


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--layer", type=int, default=8)
    ap.add_argument("--max_new", type=int, default=200)
    ap.add_argument("--sanity", action="store_true", help="also compare cached vs live sender latents on 3 problems")
    args = ap.parse_args()

    vrows = json.loads((DATA / "val.json").read_text(encoding="utf-8"))
    vhid = torch.load(DATA / "hidden_val.pt", weights_only=False, mmap=True)
    vhid["layers"] = {args.layer: vhid["layers"][args.layer]}
    model, tok = load_model(RECEIVER_ID)
    rec = Receiver(model, tok)
    bridge, meta = LatentBridge.load(args.ckpt, DEVICE)
    start = ctx_start(tok)
    content_end = lambda n: content_span(tok, n)[1]  # noqa: E731

    def vstates(i):
        return sender_states(vhid, i, args.layer, "ctx", start, content_end)

    if args.sanity:
        smodel, stok = load_model(SENDER_ID)
        sender = Sender(smodel, stok)
        with torch.no_grad():
            for i in [0, 100, 199]:
                live = sender.hidden(vrows[i]["context"], args.layer, skip_template=meta.get("skip_template", False),
                                     content_only=meta.get("content_only", False))
                a, b = bridge(live), bridge(vstates(i))
                print(f"  sanity {vrows[i]['id']}: shape {tuple(a.shape)} vs {tuple(b.shape)}, "
                      f"max|diff| {(a - b).abs().max().item():.2e}, cos {(torch.nn.functional.cosine_similarity(a, b, dim=-1)).mean().item():.6f}",
                      flush=True)
                assert a.shape == b.shape, "live vs cached shape mismatch!"
        del smodel
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    t0 = time.time()
    hits, outs = 0, []
    with torch.no_grad():
        for j in range(len(vrows)):
            out = rec.generate(rec.latent_inputs(bridge(vstates(j)), vrows[j]["final_q"])[0], args.max_new)
            ok = extract_number(out) == vrows[j]["answer"]
            hits += ok
            outs.append({"id": vrows[j]["id"], "out": out, "correct": ok, "gold": vrows[j]["answer"]})
            if (j + 1) % 25 == 0:
                print(f"  {j + 1}/{len(vrows)} acc {hits / (j + 1):.3f} ({time.time() - t0:.0f}s)", flush=True)
    acc = hits / len(vrows)
    print(f"[{args.tag}] val accuracy {hits}/{len(vrows)} = {acc:.1%} in {time.time() - t0:.0f}s", flush=True)
    (RESULTS / f"ext_val_{args.tag}.json").write_text(json.dumps(
        {"tag": args.tag, "ckpt": args.ckpt, "n": len(vrows), "accuracy": acc, "hits": hits,
         "seconds": round(time.time() - t0), "preds": outs}, indent=1))
    with open(RESULTS / "ext_val.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"tag": args.tag, "ckpt": args.ckpt, "n": len(vrows), "accuracy": acc, "ts": time.time()}) + "\n")


if __name__ == "__main__":
    main()

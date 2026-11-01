"""Large, resumable, fairly-timed evaluation (default 500 GSM8K test problems).

Conditions: text (1B writes, 3B reads), latent_s1, latent_s2 (live 1B forward -> bridge), context (raw facts).
Timing per answer, measured separately (CUDA-synchronized):
  reader_s   : 1B time (text: generating the message; latent: one forward pass + bridge)
  prefill_s  : 3B reading its whole input (one forward pass)
  decode_s   : 3B generating its answer token by token (manual greedy loop with KV cache)
  gen_tokens : number of answer tokens the 3B generated
Results append to results/big_preds.jsonl after every problem; restarts skip finished (cond, id) pairs.
Report: results/big_eval.json / .md with accuracy, timing split, and McNemar tests vs text.

python eval_big.py --n 500            # run (resumable)
python eval_big.py --n 500 --report   # only rebuild the report
"""
import argparse
import json
import math
import time

import torch

from bridge import LatentBridge
from common import CKPT, DATA, DEVICE, RECEIVER_ID, RESULTS, SENDER_ID, extract_number, load_model
from receiver_io import Receiver, Sender

PREDS = RESULTS / "big_preds.jsonl"
LATENTS = {"latent_s1": "bridge_main_token.pt", "latent_s2": "bridge_main_token_s2.pt"}
CONDS = ["text", "latent_s1", "latent_s2", "context"]


def now():
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter()


@torch.no_grad()
def timed_generate(rec, x, max_new=256):
    """Greedy decode with an explicit prefill / decode split. Returns (text, prefill_s, decode_s, n_tokens)."""
    model = rec.model
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    t0 = now()
    out = model(inputs_embeds=x, use_cache=True)
    nxt = out.logits[:, -1].argmax(-1)
    t1 = now()
    past, toks = out.past_key_values, []
    while len(toks) < max_new and nxt.item() not in eos:
        toks.append(nxt.item())
        out = model(input_ids=nxt[:, None], past_key_values=past, use_cache=True)
        past, nxt = out.past_key_values, out.logits[:, -1].argmax(-1)
    t2 = now()
    return rec.tok.decode(toks, skip_special_tokens=True).strip(), t1 - t0, t2 - t1, len(toks)


def load_done():
    done = {}
    if PREDS.exists():
        for line in PREDS.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            done[(r["cond"], r["id"])] = r
    return done


def mcnemar(a, b):
    """Exact two-sided McNemar test on paired correctness lists. Returns (b01, b10, p)."""
    n01 = sum(1 for x, y in zip(a, b) if not x and y)   # a wrong, b right
    n10 = sum(1 for x, y in zip(a, b) if x and not y)   # a right, b wrong
    n = n01 + n10
    if n == 0:
        return n01, n10, 1.0
    k = min(n01, n10)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return n01, n10, min(1.0, 2 * p)


def report(n_rows):
    rows = json.loads((DATA / "test.json").read_text(encoding="utf-8"))[:n_rows]
    ids = [r["id"] for r in rows]
    done = load_done()
    complete = [i for i in ids if all((c, i) in done for c in CONDS)]
    out = {"n_complete": len(complete), "conditions": {}, "mcnemar_vs_text": {}}
    keys = ["reader_s", "prefill_s", "decode_s", "total_s", "gen_tokens", "msg_size", "solver_input_len"]
    for c in CONDS:
        rs = [done[(c, i)] for i in complete]
        if not rs:
            continue
        d = {"accuracy": sum(r["correct"] for r in rs) / len(rs)}
        for k in keys:
            d[k] = sum(r[k] for r in rs) / len(rs)
        d["decode_s_per_token"] = sum(r["decode_s"] for r in rs) / max(1, sum(r["gen_tokens"] for r in rs))
        out["conditions"][c] = d
    text = [done[("text", i)]["correct"] for i in complete]
    for c in CONDS[1:]:
        other = [done[(c, i)]["correct"] for i in complete]
        n01, n10, p = mcnemar(text, other)
        out["mcnemar_vs_text"][c] = {"text_wrong_other_right": n01, "text_right_other_wrong": n10, "p_value": p}
    (RESULTS / "big_eval.json").write_text(json.dumps(out, indent=1))
    names = {"text": "Text message", "latent_s1": "Latent, stride 1", "latent_s2": "Latent, stride 2",
             "context": "Raw context (ref)"}
    lines = [f"n = {len(complete)} problems (all conditions complete)", "",
             "| Condition | Accuracy | Msg size | Solver input | Reader s | 3B prefill s | 3B decode s | Answer tokens | Total s |",
             "|---|---|---|---|---|---|---|---|---|"]
    for c, d in out["conditions"].items():
        lines.append(f"| {names[c]} | {d['accuracy']:.1%} | {d['msg_size']:.1f} | {d['solver_input_len']:.1f} | "
                     f"{d['reader_s']:.3f} | {d['prefill_s']:.3f} | {d['decode_s']:.2f} | {d['gen_tokens']:.0f} | "
                     f"{d['total_s']:.2f} |")
    lines += ["", "McNemar (exact, two-sided) vs text message:", "",
              "| Condition | Text wrong, it right | Text right, it wrong | p |", "|---|---|---|---|"]
    for c, m in out["mcnemar_vs_text"].items():
        lines.append(f"| {names[c]} | {m['text_wrong_other_right']} | {m['text_right_other_wrong']} | {m['p_value']:.2g} |")
    (RESULTS / "big_eval.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--max_new", type=int, default=256)
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    if args.report:
        report(args.n)
        return
    rows = json.loads((DATA / "test.json").read_text(encoding="utf-8"))[:args.n]
    assert len(rows) >= args.n, f"test.json has {len(rows)} rows; run build_dataset.py --only_test --n_test {args.n}"
    done = load_done()
    sender = Sender(*load_model(SENDER_ID))
    rec = Receiver(*load_model(RECEIVER_ID))
    bridges = {c: LatentBridge.load(CKPT / ck, DEVICE) for c, ck in LATENTS.items()}
    timed_generate(rec, rec.none_inputs("What is 1+1?")[0], 4)  # warm-up
    f = open(PREDS, "a", encoding="utf-8")
    t_start = time.time()
    for k, r in enumerate(rows):
        if all((c, r["id"]) in done for c in CONDS):
            continue
        base = {"id": r["id"], "gold": r["answer"]}
        msg, reader_s = None, 0.0
        if ("text", r["id"]) not in done:
            t0 = now()
            msg = sender.message(r["context"])
            reader_s = now() - t0
            x, _ = rec.text_inputs(msg, r["final_q"])
            out, pre, dec, ntok = timed_generate(rec, x, args.max_new)
            size = len(rec.tok(msg, add_special_tokens=False).input_ids)
            rec_ = dict(base, cond="text", msg=msg, msg_size=size, reader_s=reader_s)
            rec_.update(out=out, correct=extract_number(out) == r["answer"], prefill_s=pre, decode_s=dec,
                        gen_tokens=ntok, total_s=reader_s + pre + dec, solver_input_len=x.shape[1])
            f.write(json.dumps(rec_) + "\n")
        for c, (bridge, meta) in bridges.items():
            if (c, r["id"]) in done:
                continue
            t0 = now()
            h = sender.hidden(r["context"], meta["layer"], skip_template=meta.get("skip_template", False),
                              content_only=meta.get("content_only", False))
            with torch.no_grad():
                lat = bridge(h)
            reader_s = now() - t0
            x, _ = rec.latent_inputs(lat, r["final_q"])
            out, pre, dec, ntok = timed_generate(rec, x, args.max_new)
            f.write(json.dumps(dict(base, cond=c, msg_size=lat.shape[1], reader_s=reader_s, out=out,
                                    correct=extract_number(out) == r["answer"], prefill_s=pre, decode_s=dec,
                                    gen_tokens=ntok, total_s=reader_s + pre + dec, solver_input_len=x.shape[1])) + "\n")
        if ("context", r["id"]) not in done:
            x, _ = rec.context_inputs(r["context"], r["final_q"])
            out, pre, dec, ntok = timed_generate(rec, x, args.max_new)
            size = len(rec.tok(r["context"], add_special_tokens=False).input_ids)
            f.write(json.dumps(dict(base, cond="context", msg_size=size, reader_s=0.0, out=out,
                                    correct=extract_number(out) == r["answer"], prefill_s=pre, decode_s=dec,
                                    gen_tokens=ntok, total_s=pre + dec, solver_input_len=x.shape[1])) + "\n")
        f.flush()
        if (k + 1) % 50 == 0:  # checkpoint: refresh the report every 50 problems
            f.close()
            print(f"[{k + 1}/{len(rows)}] {time.time() - t_start:.0f}s", flush=True)
            report(args.n)
            f = open(PREDS, "a", encoding="utf-8")
    f.close()
    report(args.n)


if __name__ == "__main__":
    main()

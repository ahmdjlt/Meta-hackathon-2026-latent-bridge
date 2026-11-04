"""Muse Spark as the improver: the local team gets better from Muse Spark's feedback, with no answer key.

One round:
  1. pool    the 1B reads NEW problems (GSM8K train rows after the 200 val rows, never used before);
             its layer-8 states are cached
  2. solve   1B -> bridge -> 3B answers each one (latent channel), exactly as in the pipeline
  3. teach   Muse Spark checks every worked answer. Accept -> the team's own answer becomes a training
             target. Reject -> Muse Spark writes a corrected step-by-step solution, which becomes the target.
             GSM8K gold answers are NEVER used for training; they are only read to report how good
             Muse Spark's labels were.
  4. train   fine-tune only the bridge on those targets (same losses as train.py; the KL teacher is the 3B
             reading the raw context, which needs no labels). Both Llamas stay frozen.
  5. eval    score before/after on the 200 held-out val problems: python val_full.py --ckpt <out> --tag <tag>

python improve.py --tag muse_r1 --ckpt checkpoints/best_backup/bridge_main_token_s2.pt --n 300
python improve.py --tag muse_r2 --ckpt checkpoints/bridge_muse_r1.pt --n 300 --offset 300   (next round)

Every stage writes to data/improve_<tag>/ and is skipped if its output exists, so a crash resumes.
"""
import argparse
import hashlib
import json
import math
import os
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import torch

from common import (CKPT, DATA, DEVICE, RECEIVER_ID, RESULTS, SENDER_ID, content_span, ctx_start, extract_number,
                    load_model)

MUSE_BASE = os.getenv("MUSE_BASE_URL", "https://api.meta.ai/v1")
POOL_START = 1700  # splittable GSM8K train rows: 0-1499 train, 1500-1699 val, so the pool starts after

TEACH_INSTR = """You are the teacher of a team of small local models (Llama 3.2 1B reads, Llama 3.2 3B solves).
Check the team's worked solution to the word problem: is the final number the correct answer?
Return ONLY JSON:
{"verdict": "accept" or "reject",
 "reason": "<one short sentence>",
 "solution": "<if reject: a short correct step-by-step solution in plain sentences (no LaTeX), ending with '#### <number>'; if accept: empty>"}"""


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--ckpt", required=True, help="bridge to improve")
    ap.add_argument("--n", type=int, default=300, help="new problems in this round")
    ap.add_argument("--offset", type=int, default=0, help="skip this many pool problems (earlier rounds)")
    ap.add_argument("--muse", default=os.getenv("IMPROVER_MODEL", "muse-spark-1.3"))
    ap.add_argument("--workers", type=int, default=6, help="parallel Muse Spark calls")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--layer", type=int, default=8)
    ap.add_argument("--max_new", type=int, default=200)
    ap.add_argument("--vram_gb", type=float, default=3.9)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stop_after", choices=["pool", "solve", "teach", "train"], default="train")
    return ap.parse_args()


# ---------------------------------------------------------------- 1. pool
def stage_pool(args, out):
    if (out / "pool.json").exists():
        return
    from build_dataset import make_rows, sender_hidden
    rows = make_rows("train", POOL_START + args.offset + args.n)[POOL_START + args.offset:]
    val_ids = {r["id"] for r in json.loads((DATA / "val.json").read_text(encoding="utf-8"))}
    train_ids = {r["id"] for r in json.loads((DATA / "pairs.json").read_text(encoding="utf-8"))}
    assert not ({r["id"] for r in rows} & (val_ids | train_ids)), "pool overlaps train/val!"
    print(f"[pool] {len(rows)} new problems: {rows[0]['id']} .. {rows[-1]['id']}", flush=True)
    model, tok = load_model(SENDER_ID)
    layers, ctx_len = [], []
    t0 = time.time()
    for r in rows:
        h, c = sender_hidden(model, tok, r["context"], None)
        layers.append(h[args.layer])
        ctx_len.append(c)
    torch.save({"layers": {args.layer: layers}, "ctx_len": ctx_len, "sender": SENDER_ID}, out / "hidden_pool.pt")
    # gold stays in the file for REPORTING ONLY; training never reads it
    (out / "pool.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print(f"[pool] 1B read {len(rows)} problems in {time.time() - t0:.0f}s", flush=True)
    del model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()


def load_receiver(args):
    if DEVICE == "cuda":
        torch.cuda.set_per_process_memory_fraction(
            min(1.0, args.vram_gb * 2**30 / torch.cuda.get_device_properties(0).total_memory))
    from receiver_io import Receiver
    return Receiver(*load_model(RECEIVER_ID))


def pool_states(args, out, tok):
    from train import sender_states
    hid = torch.load(out / "hidden_pool.pt", weights_only=False)
    start = ctx_start(tok)
    content_end = lambda n: content_span(tok, n)[1]  # noqa: E731
    return lambda i: sender_states(hid, i, args.layer, "ctx", start, content_end)


# ---------------------------------------------------------------- 2. solve
def stage_solve(args, out, rec):
    path = out / "solve.jsonl"
    rows = json.loads((out / "pool.json").read_text(encoding="utf-8"))
    done = {json.loads(l)["id"] for l in path.read_text(encoding="utf-8").splitlines()} if path.exists() else set()
    if len(done) >= len(rows):
        return
    from bridge import LatentBridge
    bridge, _ = LatentBridge.load(args.ckpt, DEVICE)
    bridge.eval()
    states = pool_states(args, out, rec.tok)
    t0 = time.time()
    with open(path, "a", encoding="utf-8") as f, torch.no_grad():
        for i, r in enumerate(rows):
            if r["id"] in done:
                continue
            ans = rec.generate(rec.latent_inputs(bridge(states(i)), r["final_q"])[0], args.max_new)
            f.write(json.dumps({"id": r["id"], "answer_raw": ans, "answer": extract_number(ans)}) + "\n")
            f.flush()
            if (i + 1) % 25 == 0:
                print(f"[solve] {i + 1}/{len(rows)} ({time.time() - t0:.0f}s)", flush=True)
    del bridge


# ---------------------------------------------------------------- 3. teach
class Teacher:
    """Muse Spark check-and-correct, one call per problem, cached on disk (thread-safe)."""

    def __init__(self, model, out):
        from openai import OpenAI
        self.model, self.path, self.lock = model, out / "teach_cache.json", threading.Lock()
        self.cache = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {}
        self.client = OpenAI(base_url=MUSE_BASE, api_key=os.environ["MODEL_API_KEY"])

    def __call__(self, problem, solution):
        h = hashlib.sha256((self.model + problem + "\n##\n" + solution).encode()).hexdigest()[:16]
        if h in self.cache:
            return self.cache[h]
        last = None
        for attempt in range(3):
            try:
                t0 = time.perf_counter()
                resp = self.client.chat.completions.create(
                    model=self.model, reasoning_effort="low",
                    messages=[{"role": "developer", "content": TEACH_INSTR},
                              {"role": "user", "content": f"Problem:\n{problem}\n\nTeam's solution:\n{solution}"}])
                v = json.loads(re.search(r"\{.*\}", resp.choices[0].message.content, re.S).group(0))
                assert v.get("verdict") in ("accept", "reject")
                if v["verdict"] == "reject":
                    assert "####" in (v.get("solution") or ""), "reject without a '#### N' solution"
                v.update(seconds=round(time.perf_counter() - t0, 2),
                         tokens=(resp.usage.total_tokens if resp.usage else None))
                with self.lock:
                    self.cache[h] = v
                    self.path.write_text(json.dumps(self.cache, indent=1), encoding="utf-8")
                return v
            except Exception as e:  # network / bad JSON: retry, then give up on this problem
                last = f"{type(e).__name__}: {e}"
                time.sleep(2 * (attempt + 1))
        return {"verdict": "unavailable", "reason": last}


def tidy(sol):
    """One '#### N' at the end, in the GSM8K format the training targets use."""
    body, num = sol.rsplit("####", 1)
    return body.strip() + f"\n#### {extract_number('#### ' + num)}"


def stage_teach(args, out):
    path = out / "teach.json"
    if path.exists():
        return
    rows = json.loads((out / "pool.json").read_text(encoding="utf-8"))
    solved = {}
    for l in (out / "solve.jsonl").read_text(encoding="utf-8").splitlines():
        s = json.loads(l)
        solved[s["id"]] = s
    teacher = Teacher(args.muse, out)
    t0 = time.time()
    with ThreadPoolExecutor(args.workers) as ex:
        verdicts = list(ex.map(lambda r: teacher(r["question"], solved[r["id"]]["answer_raw"]), rows))
    labeled, stats = [], {"n": len(rows), "accept": 0, "reject": 0, "unavailable": 0, "tokens": 0,
                          "team_right_before": 0, "label_right": 0, "accept_right": 0, "reject_fixed": 0,
                          "reject_of_right": 0}
    for pi, (r, v) in enumerate(zip(rows, verdicts)):
        s = solved[r["id"]]
        team_right = s["answer"] == r["answer"]  # reporting only
        stats["team_right_before"] += team_right
        stats["tokens"] += v.get("tokens") or 0
        stats[v["verdict"]] += 1
        if v["verdict"] == "unavailable":
            continue
        if v["verdict"] == "accept":
            if "####" not in s["answer_raw"]:
                continue  # no clean final line: not a usable target
            target, source = tidy(s["answer_raw"]), "team (accepted)"
            stats["accept_right"] += team_right
        else:
            target, source = tidy(v["solution"]), "muse correction"
            stats["reject_of_right"] += team_right
        label_right = extract_number(target) == r["answer"]
        stats["label_right"] += label_right
        stats["reject_fixed"] += (v["verdict"] == "reject" and label_right and not team_right)
        labeled.append({**{k: r[k] for k in ("id", "question", "context", "final_q")}, "pool_index": pi,
                        "solution": target, "label_source": source, "reason": v.get("reason"),
                        "label_right_vs_gold": label_right})
    n_lab = max(1, len(labeled))
    stats.update(labeled=len(labeled), label_accuracy=stats["label_right"] / n_lab,
                 team_acc_before=stats["team_right_before"] / max(1, stats["n"]), muse=args.muse,
                 teach_seconds=round(time.time() - t0))
    path.write_text(json.dumps({"stats": stats, "rows": labeled}, indent=1), encoding="utf-8")
    print(f"[teach] {json.dumps(stats)}", flush=True)


# ---------------------------------------------------------------- 4. train
class NS:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def stage_train(args, out, rec):
    dst = CKPT / f"bridge_{args.tag}.pt"
    if dst.exists():
        return
    from bridge import LatentBridge
    from train import losses
    data = json.loads((out / "teach.json").read_text(encoding="utf-8"))["rows"]
    bridge, meta = LatentBridge.load(args.ckpt, DEVICE)
    states = pool_states(args, out, rec.tok)
    rec.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    lin = [p for n, p in bridge.named_parameters() if n.startswith("lin.")]
    rest = [p for n, p in bridge.named_parameters() if not n.startswith("lin.")]
    opt = torch.optim.AdamW([{"params": rest}, {"params": lin}], lr=args.lr, weight_decay=0.01)
    total = math.ceil(len(data) * args.epochs / args.accum)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * min(1.0, s / max(1, total)))))
    # same loss weights as the main runs; the KL teacher is the 3B reading the raw context (label-free)
    largs = NS(kl_w=1.0, ce_w=0.5, recon_w=1.0, align_w=0.2, temp=2.0, teacher="context", arch="token",
               stride=bridge.config["stride"])
    random.seed(args.seed)
    order = list(range(len(data)))
    step, t0, log = 0, time.time(), []
    bridge.train()
    rec.model.train()
    for ep in range(args.epochs):
        random.shuffle(order)
        run = []
        for j, k in enumerate(order):
            row = data[k]
            o, _ = losses(rec, bridge, row, states(row["pool_index"]), largs)
            (o["loss"] / args.accum).backward()
            run.append(o["loss"].item())
            if (j + 1) % args.accum == 0 or j == len(order) - 1:
                torch.nn.utils.clip_grad_norm_(bridge.parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
                step += 1
                if DEVICE == "cuda" and step % 5 == 0:
                    torch.cuda.empty_cache()
                log.append({"step": step, "epoch": ep, "loss": sum(run) / len(run), "s": round(time.time() - t0)})
                if step % 10 == 0:
                    print(f"[train] ep {ep} step {step}/{total} loss {log[-1]['loss']:.3f} ({time.time() - t0:.0f}s)",
                          flush=True)
                run = []
    meta = {**meta, "improved_from": str(args.ckpt), "improver": args.muse, "improve_tag": args.tag,
            "improve_rows": len(data), "args": {**meta.get("args", {}), "improve": vars(args)}}
    bridge.save(dst, **meta)
    (out / "train_log.json").write_text(json.dumps({"log": log, "seconds": round(time.time() - t0)}, indent=1))
    print(f"[train] {len(data)} Muse-labeled problems x {args.epochs} epochs in {time.time() - t0:.0f}s -> {dst}",
          flush=True)


def main():
    args = parse_args()
    out = DATA / f"improve_{args.tag}"
    out.mkdir(exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=1))
    torch.manual_seed(args.seed)
    stage_pool(args, out)
    if args.stop_after == "pool":
        return
    rec = None
    if not (out / "teach.json").exists() or not (CKPT / f"bridge_{args.tag}.pt").exists():
        rec = load_receiver(args)
    stage_solve(args, out, rec)
    if args.stop_after == "solve":
        return
    stage_teach(args, out)
    if args.stop_after == "teach":
        return
    stage_train(args, out, rec)
    print(f"done. next: python val_full.py --ckpt checkpoints/bridge_{args.tag}.pt --tag {args.tag}", flush=True)


if __name__ == "__main__":
    main()

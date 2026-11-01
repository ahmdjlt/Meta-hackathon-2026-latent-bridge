"""Extended training (more-training branch): continue a token bridge from a checkpoint.

Differences from train.py:
  - starts from --resume (a bridge_*.pt); stride may differ (weights are position-wise)
  - trains on ALL 1500 pairs.json rows; selects on data/val.json (200 held-out problems)
  - per-epoch checkpoints bridge_<tag>_ep<N>.pt (with optimizer/scheduler state: resumable)
  - best-by-validation bridge_<tag>_best.pt; early stopping, patience 2, on
    val_gen_acc (+ 0.01 * shuffle_gap tie-break, same score as train.py)
  - cosine lr decay from --lr to 0 over the run (no warmup: we continue training)

Per-epoch validation: teacher-forced losses + shuffle gap on all 200 val problems
(cheap), free generation on a FIXED seeded subset (--val_subset, default 60).
Full-200 generation accuracy for finalists is scored separately (val_full.py).

Resuming an interrupted run:
  python train_ext.py --tag ext_s2 --resume_train checkpoints/ext_s2_ep2.pt --epochs 6
continues from epoch 3 with the saved optimizer, scheduler, best and patience state.
"""
import argparse
import csv
import json
import math
import random
import time

import torch
import torch.nn.functional as F

from bridge import TokenBridge
from common import (CKPT, DATA, DEVICE, RECEIVER_ID, RESULTS, SENDER_ID, answer_text, content_span,
                    ctx_start, extract_number, load_model)
from receiver_io import Receiver
from train import losses, mem_gb, sender_states, tail_ce


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--resume", default=None, help="bridge checkpoint to continue from")
    ap.add_argument("--resume_train", default=None, help="per-epoch ckpt of this script to resume (overrides --resume)")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--layer", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--vram_gb", type=float, default=3.9)
    ap.add_argument("--val_subset", type=int, default=60, help="fixed seeded val subset for per-epoch generation")
    ap.add_argument("--val_seed", type=int, default=1234)
    ap.add_argument("--max_new", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1)
    # loss weights identical to the main runs
    ap.add_argument("--kl_w", type=float, default=1.0)
    ap.add_argument("--ce_w", type=float, default=0.5)
    ap.add_argument("--recon_w", type=float, default=1.0)
    ap.add_argument("--align_w", type=float, default=0.2)
    ap.add_argument("--temp", type=float, default=2.0)
    ap.add_argument("--teacher", choices=["msg", "context"], default="context")
    ap.add_argument("--lin_lr_mult", type=float, default=1.0)
    ap.add_argument("--limit", type=int, default=0, help="use only the first N training rows (smoke tests)")
    ap.add_argument("--d", type=int, default=1024)
    return ap.parse_args()


class NS:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    rows = json.loads((DATA / "pairs.json").read_text(encoding="utf-8"))
    hid = torch.load(DATA / "hidden_train.pt", weights_only=False, mmap=True)
    hid["layers"] = {args.layer: hid["layers"][args.layer]}
    vrows = json.loads((DATA / "val.json").read_text(encoding="utf-8"))
    vhid = torch.load(DATA / "hidden_val.pt", weights_only=False, mmap=True)
    vhid["layers"] = {args.layer: vhid["layers"][args.layer]}
    tr_idx = list(range(len(rows)))
    if args.limit:
        tr_idx = tr_idx[:args.limit]
    sub_rng = random.Random(args.val_seed)
    sub_idx = sub_rng.sample(range(len(vrows)), min(args.val_subset, len(vrows)))
    print(f"[{args.tag}] train {len(tr_idx)} val {len(vrows)} (gen subset {len(sub_idx)})", flush=True)

    if DEVICE == "cuda":
        torch.cuda.set_per_process_memory_fraction(min(1.0, args.vram_gb * 2**30 / torch.cuda.get_device_properties(0).total_memory))
    model, tok = load_model(RECEIVER_ID)
    rec = Receiver(model, tok)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    start = ctx_start(tok)
    content_end = lambda n: content_span(tok, n)[1]  # noqa: E731
    d_in = hid["layers"][args.layer][0].shape[-1]
    bridge = TokenBridge(d_in, rec.emb.weight.shape[1], d=args.d, stride=args.stride,
                         target_norm=rec.mean_embed_norm())

    start_ep, best, wait, history, total_done_s = 0, -float("inf"), 0, [], 0.0
    train_state = None
    ckpt_path = args.resume_train or args.resume
    if ckpt_path:
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = ck.get("state_dict", ck)
        missing, unexpected = bridge.load_state_dict(sd, strict=False), None
        assert not missing.missing_keys and not missing.unexpected_keys, f"bad load: {missing}"
        print(f"  resumed bridge weights from {ckpt_path} (meta stride {ck.get('meta', {}).get('args', {}).get('stride', '?')} -> {args.stride})",
              flush=True)
        if args.resume_train and "train_state" in ck:
            train_state = ck["train_state"]

    lin = [p for n, p in bridge.named_parameters() if n.startswith("lin.")]
    rest = [p for n, p in bridge.named_parameters() if not n.startswith("lin.")]
    opt = torch.optim.AdamW([{"params": rest, "lr": args.lr}, {"params": lin, "lr": args.lr * args.lin_lr_mult}],
                            weight_decay=0.01)
    total = math.ceil(len(tr_idx) * args.epochs / args.accum)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * min(1.0, s / max(1, total)))))
    step = 0
    if train_state:
        opt.load_state_dict(train_state["opt"])
        sched.load_state_dict(train_state["sched"])
        start_ep = train_state["epoch"] + 1
        best, wait = train_state["best"], train_state["wait"]
        history = train_state["history"]
        step = train_state["step"]
        total_done_s = train_state.get("elapsed_s", 0.0)
        # rescale the schedule to the full run (same total => consistent cosine)
        print(f"  resumed training state: epochs done {start_ep}, best {best:.3f}, wait {wait}, step {step}", flush=True)
    bridge = bridge.to(DEVICE)
    largs = NS(kl_w=args.kl_w, ce_w=args.ce_w, recon_w=args.recon_w, align_w=args.align_w,
               temp=args.temp, teacher=args.teacher, arch="token", stride=args.stride)
    print(f"  bridge {sum(p.numel() for p in bridge.parameters()) / 1e6:.1f}M params | {total} optimizer steps "
          f"| cosine {args.lr:.0e} -> 0 | patience {args.patience}", flush=True)

    logf = open(RESULTS / f"train_log_{args.tag}.csv", "a" if train_state else "w", newline="")
    log = csv.writer(logf)
    if not train_state:
        log.writerow(["step", "epoch", "loss", "kl", "ce", "recon", "align", "lr", "elapsed_s"])

    def states(i):
        return sender_states(hid, i, args.layer, "ctx", start, content_end)

    def vstates(i):
        return sender_states(vhid, i, args.layer, "ctx", start, content_end)

    def validate():
        bridge.eval()
        model.eval()
        m = {"kl": 0.0, "ce": 0.0, "recon": 0.0, "align": 0.0, "loss": 0.0, "ce_shuffled": 0.0}
        with torch.no_grad():
            for j in range(len(vrows)):
                o, _ = losses(rec, bridge, vrows[j], vstates(j), largs)
                for k in ("kl", "ce", "recon", "align", "loss"):
                    m[k] += o[k].item()
                other = (j + 1) % len(vrows)
                lat_o = bridge(vstates(other))
                xo, no = rec.latent_inputs(lat_o, vrows[j]["final_q"], vrows[j]["solution"])
                m["ce_shuffled"] += tail_ce(rec, xo, no, answer_text(vrows[j]["solution"])).item()
            hits = 0
            for j in sub_idx:
                out = rec.generate(rec.latent_inputs(bridge(vstates(j)), vrows[j]["final_q"])[0], args.max_new)
                hits += extract_number(out) == vrows[j]["answer"]
        n = len(vrows)
        v = {f"val_{k}": x / n for k, x in m.items()}
        v["val_shuffle_gap"] = v["val_ce_shuffled"] - v["val_ce"]
        v["val_sub_acc"] = hits / len(sub_idx)
        v["score"] = v["val_sub_acc"] + v["val_shuffle_gap"] * 0.01
        bridge.train()
        model.train()
        return v

    t0 = time.time()
    bridge.train()
    model.train()
    stopped_early = False
    for ep in range(start_ep, args.epochs):
        random.shuffle(tr_idx)
        run = []
        for j, i in enumerate(tr_idx):
            o, _ = losses(rec, bridge, rows[i], states(i), largs)
            (o["loss"] / args.accum).backward()
            run.append([o[k].item() for k in ("loss", "kl", "ce", "recon", "align")])
            if (j + 1) % args.accum == 0 or j == len(tr_idx) - 1:
                torch.nn.utils.clip_grad_norm_(bridge.parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
                if DEVICE == "cuda" and step % 5 == 0:
                    torch.cuda.empty_cache()
                step += 1
                mean = [sum(c) / len(run) for c in zip(*run)]
                log.writerow([step, ep, *[f"{x:.4f}" for x in mean], f"{sched.get_last_lr()[0]:.2e}",
                              f"{total_done_s + time.time() - t0:.0f}"])
                logf.flush()
                if step % 10 == 0:
                    print(f"  ep {ep} step {step}/{total} loss {mean[0]:.3f} kl {mean[1]:.3f} ce {mean[2]:.3f} "
                          f"recon {mean[3]:.3f} align {mean[4]:.3f} ({total_done_s + time.time() - t0:.0f}s) | "
                          f"RAM {mem_gb():.1f} GB, VRAM {torch.cuda.memory_reserved() / 2**30 if DEVICE == 'cuda' else 0:.2f} GB reserved",
                          flush=True)
                run = []
        v = validate()
        v["epoch"] = ep
        history.append(v)
        print(f"  [val ep {ep}] " + " ".join(f"{k} {x:.3f}" for k, x in v.items() if k != "epoch"), flush=True)
        # per-epoch checkpoint WITH training state (resumable)
        torch.save({"config": bridge.config, "state_dict": bridge.state_dict(),
                    "meta": {"layer": args.layer, "src": "ctx", "teacher": args.teacher, "skip_template": True,
                             "content_only": True, "arch": "token", "receiver": RECEIVER_ID, "sender": SENDER_ID,
                             "val": v, "args": vars(args)},
                    "train_state": {"opt": opt.state_dict(), "sched": sched.state_dict(), "epoch": ep,
                                    "best": best, "wait": wait, "history": history, "step": step,
                                    "elapsed_s": total_done_s + time.time() - t0}},
                   CKPT / f"bridge_{args.tag}_ep{ep}.pt")
        if v["score"] > best:
            best = v["score"]
            wait = 0
            bridge.save(CKPT / f"bridge_{args.tag}_best.pt", layer=args.layer, src="ctx", teacher=args.teacher,
                        skip_template=True, content_only=True, arch="token", receiver=RECEIVER_ID,
                        sender=SENDER_ID, val=v, args=vars(args))
            print(f"  new best (score {best:.3f}) -> bridge_{args.tag}_best.pt", flush=True)
        else:
            wait += 1
            print(f"  no improvement (wait {wait}/{args.patience})", flush=True)
            if wait >= args.patience:
                stopped_early = True
                break
    logf.close()
    peak = torch.cuda.max_memory_allocated() / 2**20 if DEVICE == "cuda" else 0
    summary = {"tag": args.tag, "args": vars(args), "history": history,
               "train_seconds": round(total_done_s + time.time() - t0), "peak_vram_mb": round(peak),
               "stopped_early": stopped_early, "best_score": best}
    (RESULTS / f"train_summary_{args.tag}.json").write_text(json.dumps(summary, indent=1))
    print(f"done in {total_done_s + time.time() - t0:.0f}s, peak VRAM {peak:.0f} MB, early stop {stopped_early}", flush=True)


if __name__ == "__main__":
    main()

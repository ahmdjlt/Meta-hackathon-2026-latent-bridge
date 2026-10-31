"""Phase 2: train the bridge with KL distillation (teacher = receiver reading the TEXT message,
student = receiver reading LATENTS) + ce_w * cross-entropy on the gold answer tokens.

Uses cached sender hidden states (data/hidden_train.pt), so only the 3B receiver sits on the GPU.
"""
import argparse
import csv
import json
import math
import random
import time

import torch
import torch.nn.functional as F

from bridge import LatentBridge
from common import CKPT, DATA, DEVICE, RECEIVER_ID, RESULTS, answer_text, load_model
from receiver_io import Receiver


def sender_states(hid, i, layer, src):
    h = hid["layers"][layer][i]
    if src == "ctx":
        h = h[:hid["ctx_len"][i]]
    return h.unsqueeze(0).to(DEVICE)


def tail_logits(model, x, n_prefix):
    """Logits that predict the tail tokens (positions n_prefix-1 .. L-2). Only those positions go
    through lm_head (logits_to_keep), which keeps the 128k-vocab logits small."""
    keep = x.shape[1] - n_prefix + 1
    return model(inputs_embeds=x, logits_to_keep=keep, use_cache=False).logits[0, :-1].float()


def losses(rec, bridge, row, h, args):
    with torch.no_grad():
        if args.teacher == "msg":   # teacher = receiver reading the sender's text message (plan default)
            xt, nt = rec.text_inputs(row["msg"], row["final_q"], row["solution"])
        else:                       # teacher = receiver reading the original context (privileged teacher)
            xt, nt = rec.text_inputs(row["context"], row["final_q"], row["solution"])
        t_logits = tail_logits(rec.model, xt, nt)
    lat = bridge(h)
    xs, ns = rec.latent_inputs(lat, row["final_q"], row["solution"])
    s_logits = tail_logits(rec.model, xs, ns)
    T = args.temp
    kl = F.kl_div(F.log_softmax(s_logits / T, -1), F.log_softmax(t_logits / T, -1),
                  log_target=True, reduction="batchmean") * T * T
    ans_ids = torch.tensor(rec.ids(answer_text(row["solution"])), device=DEVICE)
    n_ans = len(ans_ids)
    ce = F.cross_entropy(s_logits[-n_ans:], ans_ids)
    # teacher-forced token accuracy on the gold solution (cheap proxy for answer quality)
    s_hit = (s_logits[-n_ans:].argmax(-1) == ans_ids).float().mean().item()
    t_hit = (t_logits[-n_ans:].argmax(-1) == ans_ids).float().mean().item()
    return kl, ce, s_hit, t_hit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--layer", type=int, default=16)
    ap.add_argument("--src", choices=["ctx", "full"], default="ctx",
                    help="ctx: sender only reads the context (one forward pass, no decoding); "
                         "full: sender states over context + its generated message")
    ap.add_argument("--teacher", choices=["msg", "context"], default="msg",
                    help="msg: receiver reading the text message; context: receiver reading the raw context")
    ap.add_argument("--no_ckpt", action="store_true", help="disable gradient checkpointing")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--temp", type=float, default=2.0)
    ap.add_argument("--ce_w", type=float, default=0.5)
    ap.add_argument("--n_val", type=int, default=50)
    ap.add_argument("--d", type=int, default=1024)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0, help="debug: use only N training rows")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    tag = args.tag or f"k{args.k}_L{args.layer}_{args.src}_{args.teacher}"
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    rows = json.loads((DATA / "pairs.json").read_text(encoding="utf-8"))
    # mmap: tensors stay on disk until used, so the unused sender layer never fills RAM
    hid = torch.load(DATA / "hidden_train.pt", weights_only=False, mmap=True)
    hid["layers"] = {args.layer: hid["layers"][args.layer]}
    idx = list(range(len(rows)))
    val_idx, tr_idx = idx[-args.n_val:], idx[:-args.n_val]
    if args.limit:
        tr_idx = tr_idx[:args.limit]

    model, tok = load_model(RECEIVER_ID)
    rec = Receiver(model, tok)
    if not args.no_ckpt:  # recompute receiver activations in backward to stay under 4 GB
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.train()  # checkpointing only runs in train mode; Llama has no dropout so outputs are unchanged
    d_in = hid["layers"][args.layer][0].shape[-1]
    bridge = LatentBridge(d_in, rec.emb.weight.shape[1], k=args.k, d=args.d, depth=args.depth,
                          target_norm=rec.mean_embed_norm()).to(DEVICE)
    opt = torch.optim.AdamW(bridge.parameters(), lr=args.lr, weight_decay=0.01)
    total = math.ceil(len(tr_idx) * args.epochs / args.accum)
    warm = max(1, total // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))
    print(f"[{tag}] train {len(tr_idx)} val {len(val_idx)} | bridge {sum(p.numel() for p in bridge.parameters()) / 1e6:.1f}M params"
          f" | {total} optimizer steps | VRAM {torch.cuda.memory_allocated() / 2**20 if DEVICE == 'cuda' else 0:.0f} MB")

    log_path = RESULTS / f"train_log_{tag}.csv"
    logf = open(log_path, "w", newline="")
    log = csv.writer(logf)
    log.writerow(["step", "epoch", "loss", "kl", "ce", "lr", "elapsed_s"])

    def validate():
        bridge.eval()
        tot = kl_s = ce_s = s_acc = t_acc = 0.0
        with torch.no_grad():
            for i in val_idx:
                kl, ce, sh, th = losses(rec, bridge, rows[i], sender_states(hid, i, args.layer, args.src), args)
                kl_s += kl.item(); ce_s += ce.item(); s_acc += sh; t_acc += th
        n = len(val_idx)
        bridge.train()
        return {"val_loss": (kl_s + args.ce_w * ce_s) / n, "val_kl": kl_s / n, "val_ce": ce_s / n,
                "val_tf_acc_latent": s_acc / n, "val_tf_acc_text": t_acc / n}

    best = float("inf")
    step = 0
    t0 = time.time()
    history = []
    bridge.train()
    for ep in range(args.epochs):
        random.shuffle(tr_idx)
        run = []
        for j, i in enumerate(tr_idx):
            kl, ce, _, _ = losses(rec, bridge, rows[i], sender_states(hid, i, args.layer, args.src), args)
            loss = kl + args.ce_w * ce
            (loss / args.accum).backward()
            run.append((loss.item(), kl.item(), ce.item()))
            if (j + 1) % args.accum == 0 or j == len(tr_idx) - 1:
                torch.nn.utils.clip_grad_norm_(bridge.parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
                step += 1
                m = [sum(c) / len(run) for c in zip(*run)]
                log.writerow([step, ep, *[f"{v:.4f}" for v in m], f"{sched.get_last_lr()[0]:.2e}", f"{time.time() - t0:.0f}"])
                logf.flush()
                if step % 10 == 0:
                    print(f"  ep {ep} step {step}/{total} loss {m[0]:.3f} kl {m[1]:.3f} ce {m[2]:.3f} "
                          f"({time.time() - t0:.0f}s)", flush=True)
                run = []
        v = validate()
        v["epoch"] = ep
        history.append(v)
        print(f"  [val ep {ep}] " + " ".join(f"{k} {x:.3f}" for k, x in v.items() if k != "epoch"), flush=True)
        if v["val_loss"] < best:
            best = v["val_loss"]
            bridge.save(CKPT / f"bridge_{tag}.pt", layer=args.layer, src=args.src, teacher=args.teacher, receiver=RECEIVER_ID, val=v, args=vars(args))
    logf.close()
    peak = torch.cuda.max_memory_allocated() / 2**20 if DEVICE == "cuda" else 0
    summary = {"tag": tag, "args": vars(args), "history": history, "best_val_loss": best,
               "train_seconds": round(time.time() - t0), "peak_vram_mb": round(peak)}
    (RESULTS / f"train_summary_{tag}.json").write_text(json.dumps(summary, indent=1))
    print(f"done in {time.time() - t0:.0f}s, peak VRAM {peak:.0f} MB -> checkpoints/bridge_{tag}.pt")


if __name__ == "__main__":
    main()

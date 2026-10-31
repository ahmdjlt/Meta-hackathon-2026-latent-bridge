"""Phase 2: train the bridge. Both LLMs frozen; only the bridge learns.

Loss = KL(teacher || student) * T^2  +  ce_w * CE(gold solution)  +  recon_w * CE(reconstruct context)
  student : receiver reading the LATENT message
  teacher : receiver reading text: the sender's message (--teacher msg) or the raw context (--teacher context)
  recon   : receiver must repeat the original context from the latents alone (forces the facts through)

Validation reports the shuffle gap: solution loss with ANOTHER example's latents minus loss with the
correct latents. Near zero means the bridge collapsed to a generic message. It also generates
full answers for a few validation problems and scores them.

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

from bridge import LatentBridge, TokenBridge
from common import (CKPT, DATA, DEVICE, RECEIVER_ID, RESULTS, SENDER_ID, answer_text, content_span, content_token_ids, ctx_start,
                    extract_number, load_model)
from receiver_io import Receiver


def mem_gb():
    import psutil
    return psutil.Process().memory_info().rss / 2**30


def sender_states(hid, i, layer, src, start, content_end=None):
    h = hid["layers"][layer][i]
    end = hid["ctx_len"][i] if src == "ctx" else len(h)
    if content_end is not None:  # token bridge: context text only (drop the assistant header)
        end = content_end(hid["ctx_len"][i])
    return h[start:end].unsqueeze(0).to(DEVICE)


def pooled_targets(rec, row, stride):
    """Receiver embeddings of the context tokens, mean-pooled like the bridge pools its positions."""
    with torch.no_grad():
        e = rec.embed(content_token_ids(rec.tok, row["context"]))[0].float()
        pad = (-e.shape[0]) % stride
        if pad:
            e = torch.cat([e, e[-1:].expand(pad, -1)])
        return e.view(-1, stride, e.shape[-1]).mean(1)


def tail_logits(model, x, n_prefix):
    """Logits that predict the tail tokens (positions n_prefix-1 .. L-2). Only those positions go
    through lm_head (logits_to_keep), which keeps the 128k-vocab logits small."""
    keep = x.shape[1] - n_prefix + 1
    return model(inputs_embeds=x, logits_to_keep=keep, use_cache=False).logits[0, :-1].float()


def tail_ce(rec, x, n_prefix, target_text):
    ids = torch.tensor(rec.tok(target_text, add_special_tokens=False).input_ids, device=DEVICE)
    lg = tail_logits(rec.model, x, n_prefix)
    return F.cross_entropy(lg[-len(ids):], ids)


def losses(rec, bridge, row, h, args):
    out = {}
    lat = bridge(h)
    xs, ns = rec.latent_inputs(lat, row["final_q"], row["solution"])
    s_logits = tail_logits(rec.model, xs, ns)
    ans_ids = torch.tensor(rec.ids(answer_text(row["solution"])), device=DEVICE)
    out["ce"] = F.cross_entropy(s_logits[-len(ans_ids):], ans_ids)
    if args.kl_w > 0:
        with torch.no_grad():
            src = row["msg"] if args.teacher == "msg" else row["context"]
            xt, nt = rec.text_inputs(src, row["final_q"], row["solution"])
            t_logits = tail_logits(rec.model, xt, nt)
        T = args.temp
        out["kl"] = F.kl_div(F.log_softmax(s_logits / T, -1), F.log_softmax(t_logits / T, -1),
                             log_target=True, reduction="batchmean") * T * T
    else:
        out["kl"] = torch.zeros((), device=DEVICE)
    if args.recon_w > 0:
        xr, nr = rec.recon_inputs(lat, row["context"])
        out["recon"] = tail_ce(rec, xr, nr, row["context"] + "<|eot_id|>")
    else:
        out["recon"] = torch.zeros((), device=DEVICE)
    if args.align_w > 0 and args.arch == "token":
        tgt = pooled_targets(rec, row, args.stride)
        out["align"] = (1 - F.cosine_similarity(lat[0], tgt, dim=-1)).mean()
    else:
        out["align"] = torch.zeros((), device=DEVICE)
    out["loss"] = (args.kl_w * out["kl"] + args.ce_w * out["ce"] + args.recon_w * out["recon"]
                   + args.align_w * out["align"])
    return out, lat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["token", "perceiver"], default="token",
                    help="token: one soft token per context token (or per --stride tokens), anchored to the "
                         "receiver's embeddings; perceiver: K learned queries (collapses, see README)")
    ap.add_argument("--stride", type=int, default=1, help="token arch: pool this many positions per soft token")
    ap.add_argument("--align_w", type=float, default=0.2, help="token arch: weight of the embedding alignment loss")
    ap.add_argument("--no_ridge", action="store_true", help="token arch: random init instead of fit_ridge.py")
    ap.add_argument("--lin_lr_mult", type=float, default=1.0, help="lr multiplier for the ridge-initialized path")
    ap.add_argument("--vram_gb", type=float, default=3.9, help="cap on CUDA memory (GB)")
    ap.add_argument("--k", type=int, default=16, help="perceiver arch: number of soft tokens")
    ap.add_argument("--layer", type=int, default=8)
    ap.add_argument("--src", choices=["ctx", "full"], default="ctx",
                    help="ctx: sender only reads the context (one forward pass, no decoding); "
                         "full: sender states over context + its generated message")
    ap.add_argument("--teacher", choices=["msg", "context"], default="context",
                    help="msg: receiver reading the text message; context: receiver reading the raw context")
    ap.add_argument("--kl_w", type=float, default=1.0)
    ap.add_argument("--ce_w", type=float, default=0.5)
    ap.add_argument("--recon_w", type=float, default=1.0)
    ap.add_argument("--no_skip_template", action="store_true", help="also feed the constant instruction states")
    ap.add_argument("--no_pos", action="store_true", help="no positional encoding in the bridge")
    ap.add_argument("--no_ckpt", action="store_true", help="disable gradient checkpointing")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--temp", type=float, default=2.0)
    ap.add_argument("--n_val", type=int, default=50)
    ap.add_argument("--val_gen", type=int, default=20, help="validation problems answered by free generation")
    ap.add_argument("--d", type=int, default=1024)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0, help="use only the first N training rows")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    shape = f"s{args.stride}" if args.arch == "token" else f"k{args.k}"
    tag = args.tag or f"{args.arch}_{shape}_L{args.layer}_{args.src}_{args.teacher}"
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

    if DEVICE == "cuda":
        # stay inside the real 4 GB: without a cap Windows spills the CUDA cache into system RAM
        torch.cuda.set_per_process_memory_fraction(min(1.0, args.vram_gb * 2**30 / torch.cuda.get_device_properties(0).total_memory))
    model, tok = load_model(RECEIVER_ID)
    rec = Receiver(model, tok)
    skip = not args.no_skip_template
    start = ctx_start(tok) if skip else 0  # sender and receiver share the Llama 3 tokenizer
    if not args.no_ckpt:  # recompute receiver activations in backward to stay under 4 GB
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    d_in = hid["layers"][args.layer][0].shape[-1]
    content_end = None
    if args.arch == "token":
        assert args.src == "ctx", "token bridge reads the context only"
        start = ctx_start(tok)
        content_end = lambda n: content_span(tok, n)[1]  # noqa: E731
        bridge = TokenBridge(d_in, rec.emb.weight.shape[1], d=args.d, stride=args.stride,
                             target_norm=rec.mean_embed_norm())
    else:
        bridge = LatentBridge(d_in, rec.emb.weight.shape[1], k=args.k, d=args.d, depth=args.depth,
                              target_norm=rec.mean_embed_norm(), pos=not args.no_pos)
    if args.arch == "token" and not args.no_ridge:
        rp = CKPT / f"ridge_L{args.layer}.pt"
        assert rp.exists(), f"run: python fit_ridge.py --layer {args.layer}"
        bridge.init_from_ridge(torch.load(rp))
    else:
        bridge.set_input_stats([sender_states(hid, i, args.layer, args.src, start, content_end)[0].cpu()
                                for i in tr_idx[:300]])
    bridge = bridge.to(DEVICE)
    # the closed-form linear path is already good: refine it gently, train the rest at full lr
    lin = [p for n, p in bridge.named_parameters() if n.startswith("lin.")]
    rest = [p for n, p in bridge.named_parameters() if not n.startswith("lin.")]
    opt = torch.optim.AdamW([{"params": rest, "lr": args.lr}, {"params": lin, "lr": args.lr * args.lin_lr_mult}],
                            weight_decay=0.01)
    total = math.ceil(len(tr_idx) * args.epochs / args.accum)
    warm = max(1, total // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))
    print(f"[{tag}] train {len(tr_idx)} val {len(val_idx)} | bridge {sum(p.numel() for p in bridge.parameters()) / 1e6:.1f}M params"
          f" | {total} optimizer steps | skip_template={skip} pos={not args.no_pos} kl_w={args.kl_w} recon_w={args.recon_w}",
          flush=True)

    logf = open(RESULTS / f"train_log_{tag}.csv", "w", newline="")
    log = csv.writer(logf)
    log.writerow(["step", "epoch", "loss", "kl", "ce", "recon", "align", "lr", "elapsed_s"])

    def states(i):
        return sender_states(hid, i, args.layer, args.src, start, content_end)

    def validate():
        bridge.eval()
        model.eval()
        m = {"kl": 0.0, "ce": 0.0, "recon": 0.0, "align": 0.0, "loss": 0.0, "ce_shuffled": 0.0}
        hits = 0
        with torch.no_grad():
            for j, i in enumerate(val_idx):
                o, _ = losses(rec, bridge, rows[i], states(i), args)
                for k in ("kl", "ce", "recon", "align", "loss"):
                    m[k] += o[k].item()
                other = val_idx[(j + 1) % len(val_idx)]
                lat_o = bridge(states(other))
                xo, no = rec.latent_inputs(lat_o, rows[i]["final_q"], rows[i]["solution"])
                m["ce_shuffled"] += tail_ce(rec, xo, no, answer_text(rows[i]["solution"])).item()
            for i in val_idx[:args.val_gen]:
                out = rec.generate(rec.latent_inputs(bridge(states(i)), rows[i]["final_q"])[0], 200)
                hits += extract_number(out) == rows[i]["answer"]
        n = len(val_idx)
        v = {f"val_{k}": x / n for k, x in m.items()}
        v["val_shuffle_gap"] = v["val_ce_shuffled"] - v["val_ce"]
        v["val_gen_acc"] = hits / max(1, min(args.val_gen, n))
        bridge.train()
        model.train()
        return v

    best = -float("inf")
    step = 0
    t0 = time.time()
    history = []
    bridge.train()
    model.train()  # gradient checkpointing only runs in train mode; Llama has no dropout
    for ep in range(args.epochs):
        random.shuffle(tr_idx)
        run = []
        for j, i in enumerate(tr_idx):
            o, _ = losses(rec, bridge, rows[i], states(i), args)
            (o["loss"] / args.accum).backward()
            run.append([o[k].item() for k in ("loss", "kl", "ce", "recon", "align")])
            if (j + 1) % args.accum == 0 or j == len(tr_idx) - 1:
                torch.nn.utils.clip_grad_norm_(bridge.parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
                if DEVICE == "cuda" and step % 5 == 0:
                    torch.cuda.empty_cache()
                step += 1
                mean = [sum(c) / len(run) for c in zip(*run)]
                log.writerow([step, ep, *[f"{x:.4f}" for x in mean], f"{sched.get_last_lr()[0]:.2e}", f"{time.time() - t0:.0f}"])
                logf.flush()
                if step % 10 == 0:
                    print(f"  ep {ep} step {step}/{total} loss {mean[0]:.3f} kl {mean[1]:.3f} ce {mean[2]:.3f} "
                          f"recon {mean[3]:.3f} align {mean[4]:.3f} ({time.time() - t0:.0f}s) | RAM {mem_gb():.1f} GB, "
                          f"VRAM {torch.cuda.memory_reserved() / 2**30 if DEVICE == 'cuda' else 0:.2f} GB reserved", flush=True)
                run = []
        v = validate()
        v["epoch"] = ep
        history.append(v)
        print(f"  [val ep {ep}] " + " ".join(f"{k} {x:.3f}" for k, x in v.items() if k != "epoch"), flush=True)
        score = v["val_gen_acc"] + v["val_shuffle_gap"] * 0.01  # generation accuracy first, gap breaks ties
        if score > best:
            best = score
            bridge.save(CKPT / f"bridge_{tag}.pt", layer=args.layer, src=args.src, teacher=args.teacher,
                        skip_template=skip, content_only=args.arch == "token",
                        arch=args.arch, receiver=RECEIVER_ID, sender=SENDER_ID, val=v, args=vars(args))
    logf.close()
    peak = torch.cuda.max_memory_allocated() / 2**20 if DEVICE == "cuda" else 0
    summary = {"tag": tag, "args": vars(args), "history": history, "train_seconds": round(time.time() - t0),
               "peak_vram_mb": round(peak)}
    (RESULTS / f"train_summary_{tag}.json").write_text(json.dumps(summary, indent=1))
    print(f"done in {time.time() - t0:.0f}s, peak VRAM {peak:.0f} MB -> checkpoints/bridge_{tag}.pt", flush=True)


if __name__ == "__main__":
    main()

"""Phase 1: build (sender message, query, answer) pairs from GSM8K and cache sender hidden states.

Outputs
  data/pairs.json, data/test.json        rows {id, context, final_q, msg, query, answer, ...}
  data/hidden_train.pt, hidden_test.pt   {"layers": {L: [T_i x d_sender fp16]}, "ctx_len": [...]}
Hidden states cover the sender's whole sequence (prompt + generated message, BOS dropped);
ctx_len marks where the prompt ends, so a bridge can read the prompt only ("ctx") or everything ("full").
"""
import argparse
import json
import time

import torch
from datasets import load_dataset

from common import (DATA, DEVICE, SENDER_ID, SENDER_LAYERS, gold_number, load_model,
                    sender_prompt, split_question)


def make_rows(split, n, offset=0):
    ds = load_dataset("openai/gsm8k", "main", split=split)
    rows = []
    for i in range(offset, len(ds)):
        ctx, fq = split_question(ds[i]["question"])
        if ctx is None:
            continue
        rows.append({"id": f"{split}-{i}", "question": ds[i]["question"], "context": ctx,
                     "final_q": fq, "answer": gold_number(ds[i]["answer"])})
        if len(rows) == n:
            break
    return rows


@torch.no_grad()
def sender_messages(model, tok, contexts, batch, max_new_tokens):
    tok.padding_side = "left"
    msgs = []
    for b in range(0, len(contexts), batch):
        prompts = [sender_prompt(c) for c in contexts[b:b + batch]]
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(DEVICE)
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.pad_token_id, temperature=None, top_p=None)
        for seq in out[:, enc.input_ids.shape[1]:]:
            msgs.append(tok.decode(seq, skip_special_tokens=True).strip())
        print(f"  generated {min(b + batch, len(contexts))}/{len(contexts)}", flush=True)
    return msgs


@torch.no_grad()
def sender_hidden(model, tok, context, msg):
    """Hidden states of the sender over prompt+message (BOS dropped). Returns ({L: T x d}, ctx_len)."""
    p_ids = tok(sender_prompt(context), add_special_tokens=False).input_ids
    m_ids = tok(msg, add_special_tokens=False).input_ids if msg else []
    ids = torch.tensor([p_ids + m_ids], device=DEVICE)
    hs = model(ids, output_hidden_states=True).hidden_states
    return {L: hs[L][0, 1:].to(torch.float16).cpu() for L in SENDER_LAYERS}, len(p_ids) - 1


def build(name, rows, model, tok, args):
    t = time.time()
    msgs = sender_messages(model, tok, [r["context"] for r in rows], args.batch, args.max_new_tokens)
    gen_s = time.time() - t
    layers = {L: [] for L in SENDER_LAYERS}
    ctx_len = []
    for r, m in zip(rows, msgs):
        r["msg"] = m
        r["query"] = f"Using the info above: {r['final_q']} Answer:"
        r["msg_tokens"] = len(tok(m, add_special_tokens=False).input_ids)
        h, c = sender_hidden(model, tok, r["context"], m)
        for L in SENDER_LAYERS:
            layers[L].append(h[L])
        ctx_len.append(c)
    out_json = DATA / ("pairs.json" if name == "train" else "test.json")
    out_json.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    torch.save({"layers": layers, "ctx_len": ctx_len, "sender": SENDER_ID}, DATA / f"hidden_{name}.pt")
    avg = sum(r["msg_tokens"] for r in rows) / len(rows)
    print(f"[{name}] {len(rows)} rows -> {out_json.name}; avg msg tokens {avg:.1f}; "
          f"sender generation {gen_s:.0f}s ({gen_s / len(rows):.2f}s/example batched)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=500)
    ap.add_argument("--n_test", type=int, default=100)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max_new_tokens", type=int, default=96)
    args = ap.parse_args()

    model, tok = load_model(SENDER_ID)
    build("test", make_rows("test", args.n_test), model, tok, args)
    build("train", make_rows("train", args.n_train), model, tok, args)
    print("example message:\n", json.loads((DATA / "pairs.json").read_text(encoding="utf-8"))[0]["msg"])


if __name__ == "__main__":
    main()

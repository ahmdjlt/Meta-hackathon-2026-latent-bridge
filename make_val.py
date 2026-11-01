"""Extended training: carve a 200-problem validation set out of GSM8K train.

Takes the 200 splittable problems immediately AFTER the 1,500 training rows
(same order/filter as build_dataset.make_rows, so zero overlap), generates the
sender messages, and caches sender hidden states.

Outputs: data/val.json, data/hidden_val.pt
"""
import json
import time

import torch

from build_dataset import make_rows, sender_hidden, sender_messages
from common import DATA, SENDER_ID, SENDER_LAYERS, load_model


def main():
    model, tok = load_model(SENDER_ID)
    all_rows = make_rows("train", 1700)
    assert len(all_rows) == 1700, f"expected 1700 splittable rows, got {len(all_rows)}"
    train_rows = json.loads((DATA / "pairs.json").read_text(encoding="utf-8"))
    train_ids = {r["id"] for r in train_rows}
    assert len(train_rows) == 1500, f"expected 1500 train rows, got {len(train_rows)}"
    # the first 1500 splittable rows must be exactly the training set (no overlap by construction)
    assert [r["id"] for r in all_rows[:1500]] == [r["id"] for r in train_rows], \
        "GSM8K order/filter drift: first 1500 splittable rows != pairs.json"
    val_rows = all_rows[1500:]
    assert not (train_ids & {r["id"] for r in val_rows}), "overlap with training set!"
    print(f"val ids: {val_rows[0]['id']} .. {val_rows[-1]['id']}", flush=True)

    t = time.time()
    msgs = sender_messages(model, tok, [r["context"] for r in val_rows], 16, 96)
    gen_s = time.time() - t
    layers = {L: [] for L in SENDER_LAYERS}
    ctx_len = []
    for r, m in zip(val_rows, msgs):
        r["msg"] = m
        r["query"] = f"Using the info above: {r['final_q']}"
        r["msg_tokens"] = len(tok(m, add_special_tokens=False).input_ids)
        h, c = sender_hidden(model, tok, r["context"], m)
        for L in SENDER_LAYERS:
            layers[L].append(h[L])
        ctx_len.append(c)
    (DATA / "val.json").write_text(json.dumps(val_rows, indent=1), encoding="utf-8")
    torch.save({"layers": layers, "ctx_len": ctx_len, "sender": SENDER_ID}, DATA / "hidden_val.pt")
    print(f"[val] {len(val_rows)} rows; sender generation {gen_s:.0f}s ({gen_s / len(val_rows):.2f}s/example batched)")


if __name__ == "__main__":
    main()

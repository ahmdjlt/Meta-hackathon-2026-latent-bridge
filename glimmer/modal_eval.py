"""Muse Glimmer 30B as the solver, reading Llama 3.2 1B's thoughts as vectors. Runs on a Modal H100.

The laptop side is already done: the 1B's layer-8 states for the test problems (data/hidden_test.pt) go
through the zero-training ridge bridge (glimmer/fit_ridge_glimmer.py) into 6656-d vectors at Glimmer's
input-embedding scale. This script ships those vectors, plus the 1B's text messages, to one H100 and asks
Glimmer each question four ways:

  none     the final question only
  text     the 1B's written notes (~60 tokens) + question
  latent   the bridge vectors spliced in as soft tokens + question   (the 1B never writes a word)
  context  the original facts verbatim + question (reference)

Setup once:  pip install modal  &&  modal token new
Run:         modal run glimmer/modal_eval.py --n 100
Writes results/glimmer_eval.json and results/glimmer_eval.md
"""
import json
import sys
import time
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
GLIMMER_ID = "meta-models/Muse-Glimmer-30B"
ANSWER_FMT = "Solve it in a few short steps, then end with '#### <number>'."
SLOT = "<<<NOTES>>>"

app = modal.App("latentbridge-glimmer")
weights = modal.Volume.from_name("glimmer-weights", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch==2.11.0", "transformers==5.22.1", "accelerate", "safetensors", "hf_transfer",
                      "huggingface_hub")
         .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/weights"}))


def prompt_parts(tok, mode, q, notes=""):
    """Glimmer chat prompt split around the message slot (so vectors can be spliced in)."""
    if mode == "none":
        user = f"{q}\n{ANSWER_FMT}"
    else:
        user = f"Notes from a teammate:\n{SLOT}\n\nUsing the info above: {q}\n{ANSWER_FMT}"
    full = tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True,
                                   reasoning_strength="low")
    if mode in ("text", "context"):
        return full.replace(SLOT, notes), None
    if mode == "none":
        return full, None
    return full.split(SLOT)


@app.function(image=image, gpu="H100", volumes={"/weights": weights}, timeout=3 * 3600)
def evaluate(payload: bytes, n: int, max_new: int = 384):
    import io
    import re

    import torch
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    data = torch.load(io.BytesIO(payload), weights_only=False)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(GLIMMER_ID)
    model = AutoModelForImageTextToText.from_pretrained(GLIMMER_ID, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    weights.commit()
    load_s = time.time() - t0
    emb = model.get_input_embeddings()

    def ids(text):
        return torch.tensor([tok(text, add_special_tokens=False).input_ids], device="cuda")

    @torch.no_grad()
    def gen(x):
        torch.cuda.synchronize()
        t = time.perf_counter()
        out = model.generate(inputs_embeds=x, attention_mask=torch.ones(x.shape[:2], dtype=torch.long, device="cuda"),
                             max_new_tokens=max_new, do_sample=False, temperature=None, top_p=None, top_k=None)
        torch.cuda.synchronize()
        return tok.decode(out[0], skip_special_tokens=True).strip(), time.perf_counter() - t, out.shape[1]

    num = re.compile(r"-?\d[\d,]*\.?\d*")

    def extract(text):
        text = text.replace("$", "")
        tail = text.rsplit("####", 1)[1] if "####" in text else text
        found = num.findall(tail)
        if not found:
            return ""
        s = found[0 if "####" in text else -1].rstrip(".").replace(",", "")
        try:
            f = float(s)
            return str(int(f)) if f == int(f) else str(f)
        except ValueError:
            return s

    rows, results = data["rows"][:n], []
    for i, r in enumerate(rows):
        rec = {"id": r["id"], "gold": r["answer"]}
        for mode in ("none", "text", "latent", "context"):
            if mode == "latent":
                pre, post = prompt_parts(tok, mode, r["final_q"])
                lat = data["latents"][i].to("cuda", torch.bfloat16).unsqueeze(0)
                x = torch.cat([emb(ids(pre)), lat, emb(ids(post))], dim=1)
                sent = lat.shape[1]
            else:
                notes = r["msg"] if mode == "text" else r["context"]
                text, _ = prompt_parts(tok, mode, r["final_q"], notes)
                x = emb(ids(text))
                sent = len(tok(notes, add_special_tokens=False).input_ids) if mode != "none" else 0
            out, secs, _ = gen(x)
            pred = extract(out)
            rec[mode] = {"pred": pred, "correct": pred == r["answer"], "seconds": round(secs, 2),
                         "sent": sent, "input_len": x.shape[1], "out": out[-600:]}
        results.append(rec)
        acc = {m: sum(z[m]["correct"] for z in results) / len(results) for m in ("none", "text", "latent", "context")}
        print(f"{i + 1}/{len(rows)} " + " ".join(f"{m} {a:.0%}" for m, a in acc.items()), flush=True)
    return {"model": GLIMMER_ID, "n": len(results), "load_s": round(load_s), "results": results}


@app.local_entrypoint()
def main(n: int = 100):
    sys.path.insert(0, str(ROOT))
    import io

    import torch

    from bridge import LatentBridge
    from common import CKPT, DATA, RESULTS, content_span, ctx_start
    from train import sender_states
    from transformers import AutoTokenizer
    from common import SENDER_ID

    rows = json.loads((DATA / "test.json").read_text(encoding="utf-8"))[:n]
    hid = torch.load(DATA / "hidden_test.pt", weights_only=False, mmap=True)
    bridge, meta = LatentBridge.load(CKPT / "bridge_glimmer_ridge.pt", "cpu")
    ltok = AutoTokenizer.from_pretrained(SENDER_ID)
    start, end = ctx_start(ltok), (lambda c: content_span(ltok, c)[1])
    with torch.no_grad():
        lats = [bridge(sender_states(hid, i, meta["layer"], "ctx", start, end).cpu())[0].to(torch.bfloat16)
                for i in range(len(rows))]
    buf = io.BytesIO()
    torch.save({"rows": [{k: r[k] for k in ("id", "context", "final_q", "msg", "answer")} for r in rows],
                "latents": lats}, buf)
    print(f"sending {len(rows)} problems, {sum(l.shape[0] for l in lats)} vectors ({buf.tell() / 2**20:.0f} MB)")
    res = evaluate.remote(buf.getvalue(), n)
    (RESULTS / "glimmer_eval.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    R = res["results"]
    lines = [f"Muse Glimmer 30B solver, {len(R)} GSM8K test problems, greedy, bf16 on one H100 "
             f"(reader: Llama 3.2 1B; bridge: zero-training ridge map, glimmer/fit_ridge_glimmer.py)\n",
             "| Condition | Accuracy | Message sent | Glimmer time |", "|---|---|---|---|"]
    for m, name in [("none", "No message"), ("text", "Text message (1B writes)"),
                    ("latent", "**Latent vectors (ours, no training)**"), ("context", "Raw context (reference)")]:
        acc = sum(z[m]["correct"] for z in R) / len(R)
        sent = sum(z[m]["sent"] for z in R) / len(R)
        secs = sum(z[m]["seconds"] for z in R) / len(R)
        unit = "vectors" if m == "latent" else "tokens"
        lines.append(f"| {name} | {acc:.0%} | {sent:.0f} {unit if m != 'none' else ''} | {secs:.2f} s |")
    (RESULTS / "glimmer_eval.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))

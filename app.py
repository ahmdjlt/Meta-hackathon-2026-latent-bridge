"""Phase 6: Gradio demo. Text pipeline vs latent pipeline, side by side.

python app.py  ->  http://127.0.0.1:7860
"""
import glob
import json
import os

import gradio as gr
import matplotlib
import torch
import torch.nn.functional as F

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from common import CKPT, DATA, RESULTS
from pipeline import Pipeline

CKPT_PATH = os.getenv("BRIDGE_CKPT", str(CKPT / "bridge_k16_L16_ctx.pt"))
pipe = Pipeline(CKPT_PATH)

# background points for the latent-space plot: pooled latents from the eval run
bg = None
files = sorted(glob.glob(str(RESULTS / "latents_*.pt")))
if files:
    bg = torch.load(files[-1], weights_only=False)
    pca = PCA(n_components=2).fit(bg["pooled"].numpy())
    bg_xy = pca.transform(bg["pooled"].numpy())

examples = [[r["question"]] for r in json.loads((DATA / "test.json").read_text(encoding="utf-8"))[:6]]
table_md = ""
for f in sorted(glob.glob(str(RESULTS / "eval_live_*.md"))):
    table_md = open(f, encoding="utf-8").read()


def nearest_tokens(lat):
    """Probe: closest receiver vocabulary token (cosine) for each soft token."""
    E = pipe.receiver.emb.weight
    sims = F.normalize(lat.to(E.device, E.dtype), dim=-1) @ F.normalize(E, dim=-1).T
    ids = sims.argmax(-1).tolist()
    return " | ".join(repr(pipe.receiver.tok.decode([i])) for i in ids)


def plot(lat):
    fig, ax = plt.subplots(figsize=(5, 4))
    if bg is None:
        ax.text(0.5, 0.5, "run eval.py first for background points", ha="center")
    else:
        c = ["#2a9d8f" if ok else "#e76f51" for ok in bg["correct"]]
        ax.scatter(bg_xy[:, 0], bg_xy[:, 1], c=c, s=18, alpha=0.6, label="test messages (green = solved)")
        p = pca.transform(lat.mean(0, keepdim=True).numpy())
        ax.scatter(p[:, 0], p[:, 1], c="black", marker="*", s=250, label="this message")
        ax.legend(fontsize=8, loc="best")
    ax.set_title("Latent messages (PCA of mean soft token)")
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    return fig


def run(task):
    r = pipe.run(task)
    p, t, l = r["plan"], r["text"], r["latent"]
    plan_md = (f"**Orchestrator:** `{p['orchestrator']}`{' (cached)' if p.get('cached') else ''}\n\n"
               f"**Facts → reader (Llama 3.2 1B):** {p['facts']}\n\n"
               f"**Question → solver (Llama 3.2 3B):** {p['question']}\n\n"
               + "\n".join(f"- `{s['agent']}`: {s['task']}" for s in p.get("subtasks", [])))
    if p.get("error"):
        plan_md += f"\n\n_Muse Spark call failed, used fallback: {p['error'][:200]}_"
    text_md = (f"### Answer: **{t['answer']}**\n\nMessage sent ({t['tokens_sent']} tokens, {t['bytes_sent']} bytes):\n"
               f"```\n{t['message']}\n```\nReader {t['reader_s']:.2f}s · Solver {t['solver_s']:.2f}s · "
               f"**Total {t['total_s']:.2f}s**")
    lat_md = (f"### Answer: **{l['answer']}**\n\nMessage sent: {l['tokens_sent']} soft tokens "
              f"({l['bytes_sent']:,} bytes fp16), no decoding by the reader.\n\n"
              f"Reader+bridge {l['reader_s']:.2f}s · Solver {l['solver_s']:.2f}s · **Total {l['total_s']:.2f}s**\n\n"
              f"Speed-up vs text: **{t['total_s'] / max(l['total_s'], 1e-6):.1f}x**\n\n"
              f"<small>Nearest-token probe: {nearest_tokens(l['latents'])}</small>")
    return plan_md, text_md, lat_md, plot(l["latents"])


with gr.Blocks(title="LatentBridge") as demo:
    gr.Markdown("# LatentBridge\nSmall Llama agents talking in **vectors** instead of text. "
                "Muse Spark plans, Llama 3.2 1B reads, a trained bridge turns its hidden states into "
                "soft tokens, Llama 3.2 3B answers.")
    task = gr.Textbox(label="Word problem", lines=3)
    go = gr.Button("Run both pipelines", variant="primary")
    gr.Examples(examples, task)
    plan = gr.Markdown()
    with gr.Row():
        with gr.Column():
            gr.Markdown("## Text pipeline")
            text_out = gr.Markdown()
        with gr.Column():
            gr.Markdown("## Latent pipeline")
            lat_out = gr.Markdown()
    fig = gr.Plot()
    if table_md:
        gr.Markdown("## Held-out GSM8K results\n" + table_md)
    go.click(run, task, [plan, text_out, lat_out, fig])

if __name__ == "__main__":
    demo.launch()

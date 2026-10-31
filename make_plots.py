"""Make README figures from results/: training curves, accuracy and latency per condition."""
import csv
import glob
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import RESULTS

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
                     "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb"})


def smooth(xs, w=10):
    out = []
    for i in range(len(xs)):
        seg = xs[max(0, i - w + 1):i + 1]
        out.append(sum(seg) / len(seg))
    return out


def training_curves(tags):
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for c, tag in zip(SERIES, tags):
        rows = list(csv.DictReader(open(RESULTS / f"train_log_{tag}.csv")))
        if not rows:
            continue
        x = [int(r["step"]) for r in rows]
        y = smooth([float(r["loss"]) for r in rows])
        ax.plot(x, y, color=c, lw=2, label=tag)
        ax.annotate(tag, (x[-1], y[-1]), xytext=(4, 0), textcoords="offset points", color=INK2, fontsize=8, va="center")
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("loss (KL + 0.5 CE), smoothed")
    ax.set_title("Bridge training loss", loc="left", color=INK)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / "fig_training_loss.png", dpi=150)


def bars(path, key, title, fmt, out):
    s = json.loads(open(path).read())
    names = {"none": "No message", "text": "Text message", "latent": "Latent (ours)", "context": "Raw context (ref)"}
    modes = [m for m in ["none", "text", "latent", "context"] if m in s]
    vals = [s[m][key] for m in modes]
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    b = ax.bar([names[m] for m in modes], vals, color=SERIES[0], width=0.55, edgecolor="#fcfcfb", linewidth=2)
    for rect, v in zip(b, vals):
        ax.annotate(fmt(v), (rect.get_x() + rect.get_width() / 2, rect.get_height()), xytext=(0, 3),
                    textcoords="offset points", ha="center", color=INK, fontsize=9)
    ax.set_title(title, loc="left", color=INK)
    ax.grid(axis="x", visible=False)
    ax.set_ylim(0, max(vals) * 1.18 if max(vals) > 0 else 1)
    fig.tight_layout()
    fig.savefig(RESULTS / out, dpi=150)


if __name__ == "__main__":
    tags = sys.argv[1:] or [p.split("train_log_")[1][:-4] for p in sorted(glob.glob(str(RESULTS / "train_log_k*.csv")))]
    training_curves(tags)
    evals = sorted(glob.glob(str(RESULTS / "eval_live_*.json")))
    if evals:
        main = evals[-1] if len(sys.argv) < 2 else evals[-1]
        bars(main, "accuracy", "GSM8K held-out accuracy (n=100)", lambda v: f"{v:.0%}", "fig_accuracy.png")
        bars(main, "mean_total_s", "End-to-end latency per question (s)", lambda v: f"{v:.2f}s", "fig_latency.png")
    print("figures written to results/")

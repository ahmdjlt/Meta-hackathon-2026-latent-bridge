"""HTML + figure builders for the demos (demo_cli.py). Pure functions of one cached/live example dict
(the format written by demo/build_cache.py), so the app can replay without a GPU."""
import csv
import html
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
CACHE = Path(__file__).resolve().parent / "cache.json"

TEXT_C, LAT_C, CTX_C, AUX_C = "#eb6834", "#2a78d6", "#1baf7a", "#eda100"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"font.size": 14, "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
                     "figure.facecolor": SURF, "axes.facecolor": SURF, "axes.titlesize": 16,
                     "axes.titleweight": "bold", "axes.titlelocation": "left"})

e = html.escape


def load_cache():
    return json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else []


# ------------------------------------------------------------------ Try it: two columns
def _verdict(ok, gold):
    if gold is None:
        return '<span class="verdict na">no gold answer</span>'
    return (f'<span class="verdict ok">✅ correct (gold {e(gold)})</span>' if ok
            else f'<span class="verdict bad">❌ wrong (gold {e(gold)})</span>')


def _bar(label, secs, color, nonce, scale_s):
    """Progress bar that fills in exactly `secs` seconds (CSS animation restarts on each render)."""
    width = max(1.5, 100 * secs / scale_s)
    return (f'<div class="bar-label">{e(label)} <b>{secs:.2f} s</b></div>'
            f'<div class="bar-track"><div class="bar-fill" data-n="{nonce}" style="background:{color};'
            f'--w:{width:.1f}%;animation-duration:{max(secs, 0.05):.2f}s"></div></div>')


def column_html(ex, which, nonce):
    """One side of the comparison: what was sent, the 3B's answer, timing bars."""
    r = ex[which]
    scale = max(ex["text"]["reader_s"], ex["latent"]["reader_s"], 0.1)
    gold = ex.get("gold")
    ok = r["answer"] == gold if gold is not None else None
    if which == "text":
        sent = (f'<div class="sent-head">Message: <b>{r["tokens_sent"]} text tokens</b> written by the 1B</div>'
                f'<pre class="msg">{e(r["message"])}</pre>')
        bar = _bar("1B writes the message token by token", r["reader_s"], TEXT_C, nonce, scale)
        color, title = TEXT_C, "Text pipeline"
    else:
        n, dim = r["tokens_sent"], r.get("dim", 3072)
        cells = "".join(f'<span class="cell" title="{e(p["source"])} → {e(p["nearest"])}" '
                        f'style="opacity:{0.35 + 0.65 * max(0.0, min(1.0, p["cos"])):.2f}"></span>'
                        for p in r.get("probe", []))
        sent = (f'<div class="sent-head">Message: <b>{n} vectors</b> × {dim} numbers, never decoded to text</div>'
                f'<div class="cells">{cells}</div>')
        bar = _bar("1B one forward pass + bridge", r["reader_s"], LAT_C, nonce, scale)
        color, title = LAT_C, "Latent pipeline"
    return (f'<div class="col" style="--accent:{color}"><div class="col-title">{title}</div>{sent}'
            f'<div class="answer">3B answer: <span class="num">{e(r["answer"] or "—")}</span> {_verdict(ok, gold)}</div>'
            f'{bar}<div class="times">3B solver {r["solver_s"]:.1f} s · <b>total {r["total_s"]:.1f} s</b></div></div>')


def _short(w, n=9):
    w = w.strip() or "␣"
    return w if len(w) <= n else w[:n - 1] + "…"


PCA_NOTE = ("2-D PCA of 3,072-D unit vectors. Squashed to 2-D the vectors sit apart from their words, "
            "but in the full space most of them are closest to exactly that word (see the boxes above). "
            "Grey: random other words from the 3B's vocabulary.")


def probe_html(ex):
    """'What the 3B received': nearest 3B word for each vector, under the source word it came from."""
    rows = ex["latent"].get("probe", [])
    if not rows:
        return ""
    hit = sum(p["match"] for p in rows)
    chips = "".join(
        f'<span class="chip{" miss" if not p["match"] else ""}" title="cosine {p["cos"]:.2f}">'
        f'<span class="near">{e(_short(p["nearest"]))}</span>'
        f'<span class="src">{e(p["source"].strip() or "␣")}</span></span>' for p in rows)
    return (f'<div class="probe"><div class="probe-head">What the 3B received, decoded by a probe '
            f'(nearest word to each vector; small grey text = the word the 1B read). '
            f'<b>{hit}/{len(rows)}</b> vectors are closest to exactly the source word (solid boxes); dashed boxes decode to '
            f'another word, yet the 3B still reads them in context.</div>'
            f'<div class="chips">{chips}</div></div>')


def muse_html(ex):
    p, v = ex["plan"], ex["latent"].get("verify") or {}
    subs = "".join(f'<li><b>{e(s["agent"])}</b>: {e(s["task"])}</li>' for s in p.get("subtasks") or [])
    verdict = v.get("verdict", "unavailable")
    icon = {"accept": "✅ accept", "reject": "❌ reject → retry over text"}.get(verdict, "— unavailable")
    fb = ""
    if ex["latent"].get("fallback") == "text":
        fb = f' Final answer after retry: <b>{e(ex["latent"]["final_answer"])}</b>.'
    t = f' ({p["orchestrator_s"]:.1f} s)' if p.get("orchestrator_s") else ""
    return (f'<div class="muse"><div class="muse-title">Muse Spark · {e(p.get("orchestrator") or "")}</div>'
            f'<div class="muse-grid"><div><div class="muse-step">1 · Plan{t}</div>'
            f'<div><span class="tag" style="background:{LAT_C}">facts → 1B reader</span> {e(p["facts"])}</div>'
            f'<div style="margin-top:6px"><span class="tag" style="background:{INK2}">question → 3B solver</span> '
            f'{e(p["question"])}</div><ul>{subs}</ul></div>'
            f'<div><div class="muse-step">2 · Verify the latent answer</div><div class="verify {e(verdict)}">{icon}</div>'
            f'<div class="reason">{e(v.get("reason", ""))}{fb}</div></div></div></div>')


def pca_figure(ex):
    """Soft tokens next to the 3B's own embeddings of the words they stand for (2-D PCA, unit vectors)."""
    pc, probe = ex["pca"], ex["latent"]["probe"]
    fig, ax = plt.subplots(figsize=(13, 6.5))
    bx, by = zip(*pc["background"])
    ax.scatter(bx, by, s=8, c="#c9c8c3", alpha=0.6, linewidths=0, label="other 3B vocabulary words")
    for (tx, ty), (sx, sy) in zip(pc["tokens"], pc["soft"]):
        ax.plot([tx, sx], [ty, sy], color=INK2, lw=0.8, alpha=0.5, zorder=2)
    tx, ty = zip(*pc["tokens"])
    sx, sy = zip(*pc["soft"])
    ax.scatter(tx, ty, s=70, facecolors=SURF, edgecolors=TEXT_C, linewidths=2, zorder=3,
               label="3B embedding of the source word")
    ax.scatter(sx, sy, s=70, c=LAT_C, edgecolors=SURF, linewidths=2, zorder=4, label="vector the bridge sent")
    seen, placed = set(), []
    span = max(max(p[0] for p in pc["soft"]) - min(p[0] for p in pc["soft"]), 1e-6)
    for (x, y), p in zip(pc["soft"], probe):  # label numbers and longer words once, skipping crowded spots
        w = p["source"].strip()
        if not w or w in seen or not (any(c.isdigit() for c in w) or len(w) >= 5):
            continue
        if any(abs(x - a) < 0.08 * span and abs(y - b) < 0.04 * span for a, b in placed):
            continue
        seen.add(w); placed.append((x, y))
        ax.annotate(w, (x, y), xytext=(6, 4), textcoords="offset points", fontsize=12, color=INK)
    allx, ally = tx + sx, ty + sy
    padx, pady = (max(allx) - min(allx)) * 0.15, (max(ally) - min(ally)) * 0.15
    ax.set_xlim(min(allx) - padx, max(allx) + padx)
    ax.set_ylim(min(ally) - pady, max(ally) + pady)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("Each vector (blue) joined to the 3B's embedding of the word it carries (orange)")
    ax.legend(loc="lower right", fontsize=11, frameon=False)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------ Results tab
def size_curve_figure():
    rows = list(csv.DictReader(open(RESULTS / "size_curve.csv", encoding="utf-8")))
    series = [("latent", "Latent (ours)", LAT_C, "o"), ("trunc", "Text message, truncated", TEXT_C, "s"),
              ("text", "Text, budget in prompt", AUX_C, "D"), ("ctx", "Original context, truncated", CTX_C, "^")]
    fig, ax = plt.subplots(figsize=(10, 5.6))
    for fam, label, c, m in series:
        pts = sorted((float(r["mean_size"]), 100 * float(r["accuracy"])) for r in rows if r["family"] == fam)
        x, y = zip(*pts)
        ax.plot(x, y, color=c, lw=2.5, marker=m, ms=9, mec=SURF, mew=2, label=label)
        ax.annotate(label, (x[-1], y[-1]), xytext=(8, 0), textcoords="offset points", va="center",
                    fontsize=12, color=INK2)
    ax.axvline(20, color=INK2, lw=1, ls=(0, (3, 3)))
    ax.text(71, 3, "At ~20 positions (dashed line):\nlatent 60%  ·  text message 22%\n"
            "original context cut to 20 tokens 25%", fontsize=13, color=INK, ha="right", va="bottom",
            linespacing=1.5)
    ax.set_xlabel("Message size (positions the 3B reads)")
    ax.set_ylabel("GSM8K accuracy, %")
    ax.set_ylim(0, 80); ax.set_xlim(5, 72)
    ax.set_title("Same message size, far more of the problem kept (n = 100)")
    ax.legend(loc="upper left", fontsize=11, frameon=False)
    fig.tight_layout()
    return fig


RESULTS_MD = """
### 500 GSM8K test problems, everything run live on the laptop (RTX 3050, 4 GB)

| Condition | Accuracy | Message | Reader time | Total per question |
|---|---|---|---|---|
| Text message (1B writes, 3B reads) | 60.2% | 59 tokens | 2.73 s | 13.4 s |
| **Latent, stride 1 (ours)** | **70.4%** | **41 vectors** | **0.06 s** | **10.1 s** |
| Latent, stride 2 (ours, compressed) | 58.6% | 21 vectors | 0.06 s | 10.0 s |
| Raw context (upper reference) | 77.6% | 41 tokens | — | 10.8 s |
| Latent + Muse Spark verify, text retry | **78.0%** | | | |

**Paired McNemar test vs the text message (same 500 problems)**

| Condition | Text wrong, it right | Text right, it wrong | p |
|---|---|---|---|
| **Latent, stride 1** | 94 | 43 | **1.6 × 10⁻⁵** |
| Latent, stride 2 | 75 | 83 | 0.58 (tie) |
| Raw context | 111 | 24 | 1.4 × 10⁻¹⁴ |

### Ablations

| Bridge | Accuracy | Note |
|---|---|---|
| Ridge matrix only, **no training** (n = 100) | 57% | one closed-form matrix already ≈ text (59%) |
| Trained token bridge, stride 1 (n = 100) | 68% | training adds 11 points |
| Reader layer 16 instead of 8, 1 epoch (n = 100) | 63% vs 69% | middle layer transfers better |
| Perceiver, 16 learned queries (original plan) | 4% | collapsed: same output for every problem |
| Extended training, stride 2 (n = 500) | 62.6% vs 58.6% | McNemar p = 0.057 → **not adopted** (bar was p < 0.05) |

What we do **not** claim: bandwidth (41 × 3072 fp16 ≈ 245 KB vs ≈ 240 bytes of text), or beating the raw context.
"""


# ------------------------------------------------------------------ How it works tab
def _box(x, y, w, h, title, sub, fill, ink="#ffffff"):
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="{fill}"/>'
            f'<text x="{x + w / 2}" y="{y + h / 2 - 4}" text-anchor="middle" font-size="19" font-weight="700" '
            f'fill="{ink}">{title}</text><text x="{x + w / 2}" y="{y + h / 2 + 20}" text-anchor="middle" '
            f'font-size="14" fill="{ink}" opacity="0.92">{sub}</text>')


def _arrow(x1, y1, x2, y2, label="", color=INK2, dash=False):
    d = ' stroke-dasharray="6 5"' if dash else ""
    lab = (f'<text x="{(x1 + x2) / 2}" y="{min(y1, y2) - 10}" text-anchor="middle" font-size="14" '
           f'fill="{INK2}">{label}</text>') if label else ""
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="2.5"{d} '
            f'marker-end="url(#ah)"/>{lab}')


def architecture_svg():
    W = 1180
    s = [f'<svg viewBox="0 0 {W} 520" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;'
         f'font-family:Inter,Segoe UI,sans-serif"><defs><marker id="ah" markerWidth="10" markerHeight="10" '
         f'refX="9" refY="5" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{INK2}"/></marker></defs>']
    # Muse Spark plan + verify
    s.append(_box(20, 190, 170, 90, "Muse Spark", "1 · plans the task", INK))
    s.append(_box(990, 190, 170, 90, "Muse Spark", "5 · verifies answer", INK))
    # latent lane
    s.append(f'<text x="230" y="40" font-size="17" font-weight="700" fill="{LAT_C}">LATENT PIPELINE (ours)</text>')
    s.append(_box(230, 60, 190, 90, "Llama 3.2 1B", "2 · reads, 1 forward pass", LAT_C))
    s.append(_box(490, 60, 170, 90, "Bridge", "3 · 17M params, trained", "#1d5fae"))
    s.append(_box(740, 60, 190, 90, "Llama 3.2 3B", "4 · solves", LAT_C))
    s.append(_arrow(420, 105, 487, 105, "hidden states"))
    s.append(_arrow(660, 105, 737, 105, "41 vectors"))
    s.append(f'<text x="325" y="172" text-anchor="middle" font-size="15" font-weight="700" fill="{LAT_C}">0.06 s</text>')
    # text lane
    s.append(f'<text x="230" y="335" font-size="17" font-weight="700" fill="{TEXT_C}">TEXT PIPELINE (baseline)</text>')
    s.append(_box(230, 355, 190, 90, "Llama 3.2 1B", "2 · writes a message", TEXT_C))
    s.append(_box(740, 355, 190, 90, "Llama 3.2 3B", "4 · solves", TEXT_C))
    s.append(_arrow(420, 400, 737, 400, "~60 text tokens, generated one by one"))
    s.append(f'<text x="325" y="467" text-anchor="middle" font-size="15" font-weight="700" fill="{TEXT_C}">2.73 s</text>')
    # plan -> readers, solvers -> verify
    s.append(_arrow(190, 220, 228, 115, "", dash=True))
    s.append(_arrow(190, 250, 228, 395, "", dash=True))
    s.append(_arrow(930, 115, 988, 220, "", dash=True))
    s.append(_arrow(930, 395, 988, 250, "", dash=True))
    s.append(f'<text x="105" y="305" text-anchor="middle" font-size="13" fill="{INK2}">facts → reader</text>'
             f'<text x="105" y="323" text-anchor="middle" font-size="13" fill="{INK2}">question → solver</text>'
             f'<text x="1075" y="305" text-anchor="middle" font-size="13" fill="{INK2}">reject → retry</text>'
             f'<text x="1075" y="323" text-anchor="middle" font-size="13" fill="{INK2}">over text</text>')
    s.append(f'<text x="590" y="505" text-anchor="middle" font-size="15" fill="{INK2}">Both Llamas stay frozen '
             f'(4-bit, RTX 3050 4 GB). Only the bridge is trained.</text></svg>')
    return "".join(s)


HOW_MD = """
1. **Muse Spark plans.** It splits the word problem into *facts* (for the 1B reader) and the *question* (for the 3B solver).
2. **The 1B reads the facts once.** Instead of writing a summary, its middle-layer hidden states go to the bridge: one forward pass, 0.06 s instead of 2.7 s.
3. **The bridge turns those states into vectors the 3B can read as if they were words** (one closed-form ridge matrix + a small trained correction). The 3B reads 41 vectors instead of ~60 written tokens and answers.
4. **Muse Spark verifies.** If it rejects the answer, the system retries once over text. Latent alone: 70.4%, text: 60.2%, with verify: 78.0% (500 problems).
"""

CSS = """
.gradio-container {font-size: 18px !important; max-width: 1500px !important}
.prose, .prose p, .prose li, .prose td, .prose th {font-size: 18px !important}
h1 {font-size: 34px !important}
.col {border: 2px solid var(--accent); border-radius: 14px; padding: 16px 18px; background: #fff; min-height: 330px}
.col-title {font-size: 24px; font-weight: 800; color: var(--accent); margin-bottom: 8px}
.sent-head {font-size: 17px; color: #52514e; margin-bottom: 6px}
.msg {white-space: pre-wrap; font-size: 15px; background: #fbf4ef; border-radius: 8px; padding: 10px;
      max-height: 190px; overflow: auto; margin: 0; font-family: Consolas, monospace; color: #0b0b0b}
.cells {display: flex; flex-wrap: wrap; gap: 3px; padding: 6px 0}
.cell {width: 20px; height: 34px; border-radius: 4px; background: #2a78d6}
.answer {font-size: 20px; margin: 14px 0 10px; color: #0b0b0b}
.answer .num {font-size: 34px; font-weight: 800; margin: 0 6px}
.verdict {font-size: 17px; font-weight: 700; padding: 3px 10px; border-radius: 999px; white-space: nowrap}
.verdict.ok {background: #e3f6ec; color: #0b6b43} .verdict.bad {background: #fde8e7; color: #a3201c}
.verdict.na {background: #eee; color: #52514e}
.bar-label {font-size: 16px; color: #52514e}
.bar-track {height: 22px; background: #efeee9; border-radius: 6px; overflow: hidden; margin: 4px 0 8px}
.bar-fill {height: 100%; width: 0; border-radius: 6px; animation-name: fillbar; animation-timing-function: linear;
           animation-fill-mode: forwards}
@keyframes fillbar {from {width: 0} to {width: var(--w)}}
.times {font-size: 16px; color: #52514e}
.probe {margin-top: 6px} .probe-head {font-size: 16px; color: #52514e; margin-bottom: 8px}
.chips {display: flex; flex-wrap: wrap; gap: 6px}
.chip {display: inline-flex; flex-direction: column; align-items: center; padding: 4px 8px; border-radius: 8px;
       background: #e8f0fb; border: 1px solid #2a78d6}
.chip.miss {background: #fff; border-style: dashed}
.chip .near {font-family: Consolas, monospace; font-size: 17px; color: #0b0b0b; font-weight: 700}
.chip .src {font-size: 12px; color: #8a8984}
.muse {border-radius: 14px; background: #f3f2ee; padding: 14px 18px}
.muse-title {font-weight: 800; font-size: 20px; margin-bottom: 8px; color: #0b0b0b}
.muse-grid {display: grid; grid-template-columns: 3fr 2fr; gap: 24px; font-size: 16px; color: #0b0b0b}
.muse-step {font-weight: 700; color: #52514e; margin-bottom: 6px}
.muse ul {margin: 6px 0 0 18px}
.tag {color: #fff; border-radius: 6px; padding: 1px 8px; font-size: 14px; font-weight: 700; margin-right: 4px}
.verify {font-size: 22px; font-weight: 800; margin-bottom: 4px}
.reason {color: #52514e}
.problem {font-size: 20px; line-height: 1.5; background: #fff; border-radius: 12px; padding: 14px 18px;
          border: 1px solid #e4e3df; color: #0b0b0b}
footer {display: none !important}
"""

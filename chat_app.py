"""LatentBridge chat: a ChatGPT-style chat over the multi-agent pipeline.

    python chat_app.py              ->  http://127.0.0.1:7860
    python chat_app.py --port 7861 --share

The two Llamas run locally; afterwards Muse Spark (Meta Model API) checks the answer (skip with --no-muse).
Per message:
  split   : last sentence = question (goes to the 3B), everything before it = facts (go to the 1B)
  read    : Llama 3.2 1B reads the facts
  channel : text message (1B writes notes) or a bridge (1B hidden states -> vectors the 3B reads directly)
  answer  : Llama 3.2 3B answers, streamed
An animation shows the hand-off; under each answer a grey line sums it up, and a "see each word" panel
shows the actual numbers: every token's 1B hidden state, the vector the bridge sends, and the 3B word
that vector is closest to.
"""
import argparse
import html
import os
import re
import time
from pathlib import Path
from threading import Thread

# Models load only from the files already on this computer: no Hugging Face / Meta network calls.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import gradio as gr  # noqa: E402
import torch

from bridge import LatentBridge
from common import (CKPT, DEVICE, RECEIVER_ID, SENDER_ID, content_token_ids, load_model,
                    sender_prompt, split_question)
from receiver_io import Receiver, Sender

MUSE = None  # MuseCheck when MODEL_API_KEY is set (see main)
ASSETS = Path(__file__).resolve().parent / "chat_assets"
ACCENT = "#a78bfa"  # bridge colour: dropdown, dots, result line
SOURCE = "#60a5fa"  # 1B-side vectors (2048-d) before the bridge

# label -> checkpoint file (None = text message). Only files that exist are offered.
CHANNELS = {
    "Text message": None,
    "Ridge only": "bridge_ridge_only_L8.pt",
    "Trained bridge (stride 1)": "bridge_main_token.pt",
    "Trained bridge (stride 2)": "bridge_main_token_s2.pt",
}
CHANNELS = {k: v for k, v in CHANNELS.items() if v is None or (CKPT / v).exists()}
SHORT = {"Text message": "text", "Ridge only": "ridge", "Trained bridge (stride 1)": "latent",
         "Trained bridge (stride 2)": "latent s2"}
HINTS = {"Text message": "1B writes notes in words (~60 tokens), slowest",
         "Ridge only": "closed-form linear map, no training, 1 vector per token",
         "Trained bridge (stride 1)": "~40 vectors, 1 per token, best accuracy",
         "Trained bridge (stride 2)": "~20 vectors, 2x compressed"}
DEFAULT = "Trained bridge (stride 1)" if "Trained bridge (stride 1)" in CHANNELS else "Text message"

EXAMPLES = [  # every one solved by the trained bridge (stride 1); tested on all channels
    ("💻", "Laptop sale",
     "A laptop costs $1,200. It is on sale for 25% off, and then a 10% tax is added to the sale price. "
     "What is the final price in dollars?"),
    ("🚆", "Train trip",
     "A train travels at 80 km per hour for 3 hours, then at 60 km per hour for 2 more hours. "
     "What is its average speed in km per hour for the whole trip?"),
    ("🧥", "Jacket profit",
     "A shop buys a jacket for $80 and marks the price up by 50%. Later it sells the jacket at 20% off the "
     "marked price. How much profit does the shop make in dollars?"),
    ("🐞", "Bug backlog",
     "A bug tracker has 84 open bugs. The team fixes 12 bugs per day, but testers report 5 new bugs each day. "
     "How many days until all bugs are fixed?"),
    ("🗄️", "Database loop",
     "A program has a loop that runs 25 times. Each run of the loop makes 4 database queries, and each query "
     "takes 3 milliseconds. How many milliseconds does the loop take in total?"),
    ("⌨️", "Lines of code",
     "A developer writes 120 lines of code per hour. She codes for 6 hours a day, 5 days a week. Later, 15% of "
     "the lines are deleted. How many lines remain after one week?"),
    ("🔢", "Even numbers",
     "The sum of three consecutive even numbers is 78. What is the largest of the three numbers?"),
    ("📐", "Rectangle area",
     "A rectangle is 3 times as long as it is wide. Its perimeter is 96 cm. What is its area in square cm?"),
    ("🎂", "Age puzzle",
     "Tom is twice as old as his sister. In 6 years, the sum of their ages will be 42. How old is Tom now?"),
]
ANIM_S = 2.9  # length of the latent animation (the 3B starts answering in the background meanwhile)
MAX_ANSWER = 384  # tokens the 3B may use to work a problem out
STRIP = 24  # dimensions drawn per vector in the "see each word" panel


# ------------------------------------------------------------------ models
class Models:
    """Both Llamas loaded once; every bridge on the dropdown loaded up front (they are small)."""

    def __init__(self):
        t0 = time.perf_counter()
        print(f"[load] {SENDER_ID} + {RECEIVER_ID} (4-bit on {DEVICE}) ...", flush=True)
        self.sender = Sender(*load_model(SENDER_ID))
        self.receiver = Receiver(*load_model(RECEIVER_ID))
        self.bridges = {label: LatentBridge.load(CKPT / f, DEVICE) for label, f in CHANNELS.items() if f}
        # length of every 3B word embedding, for "which 3B word is this vector closest to"
        w = self.receiver.emb.weight
        self.emb_norm = torch.cat([w[i:i + 8192].float().norm(dim=-1) for i in range(0, w.shape[0], 8192)])
        print(f"[load] done in {time.perf_counter() - t0:.1f} s; channels: {', '.join(CHANNELS)}", flush=True)
        self._warmup()

    def _t(self):
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        return time.perf_counter()

    def _warmup(self):
        t0 = time.perf_counter()
        for label in self.bridges:
            self.read_latent("Sam has 2 apples.", label)
        for _ in self.stream_message("Sam has 2 apples.", max_new_tokens=4):
            pass
        self.receiver.generate(self.receiver.text_inputs("Sam has 2 apples.", "How many apples?")[0], max_new_tokens=4)
        print(f"[load] warm-up {time.perf_counter() - t0:.1f} s", flush=True)

    def read_latent(self, facts, label):
        """1B reads the facts; the bridge maps its hidden states to 3B input vectors.
        Returns (latents [1, n, 3072], reader_s, hidden states [1, T, 2048])."""
        t0 = self._t()
        bridge, meta = self.bridges[label]
        h = self.sender.hidden(facts, meta["layer"], skip_template=meta.get("skip_template", False),
                               content_only=meta.get("content_only", False))
        with torch.no_grad():
            lat = bridge(h)
        return lat, self._t() - t0, h

    def tokens(self, facts):
        """The facts as the 1B's tokens, exactly the positions the bridge reads (one hidden state each)."""
        tok = self.sender.tok
        return [tok.decode([i]) for i in content_token_ids(tok, facts)]

    @torch.no_grad()
    def nearest_words(self, lat):
        """For each sent vector: the 3B word embedding it points at most (cosine), and that cosine."""
        w = self.receiver.emb.weight
        v = lat[0]
        cos = (v.to(w.dtype) @ w.T).float() / (self.emb_norm[None] * v.float().norm(dim=-1, keepdim=True))
        best = cos.max(-1)
        return [self.receiver.tok.decode([i]) for i in best.indices.tolist()], best.values.tolist()

    def _stream(self, model, tok, kw):
        from transformers import TextIteratorStreamer
        streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
        Thread(target=model.generate, kwargs={**kw, "streamer": streamer, "do_sample": False, "temperature": None,
                                              "top_p": None, "pad_token_id": tok.pad_token_id}, daemon=True).start()
        return streamer

    def stream_message(self, facts, max_new_tokens=96):
        """Text channel: the 1B writes its notes (same prompt and settings as Sender.message), streamed."""
        s = self.sender
        enc = s.tok(sender_prompt(facts), return_tensors="pt", add_special_tokens=False).to(DEVICE)
        out = ""
        for piece in self._stream(s.model, s.tok, dict(**enc, max_new_tokens=max_new_tokens)):
            out += piece
            yield out

    def start_answer(self, msg, question):
        """3B answers from a text message (str) or latents (tensor). Starts now; returns a text-piece iterator."""
        r = self.receiver
        x = r.text_inputs(msg, question)[0] if isinstance(msg, str) else r.latent_inputs(msg, question)[0]
        mask = torch.ones(x.shape[:2], dtype=torch.long, device=DEVICE)
        return self._stream(r.model, r.tok, dict(inputs_embeds=x, attention_mask=mask, max_new_tokens=MAX_ANSWER))


M = None  # set in main()


def split(message):
    """Last sentence = question for the 3B; the sentences before it = facts for the 1B."""
    facts, q = split_question(message)
    if not facts:  # a single sentence: the 1B reads all of it and the 3B answers it
        return message, message
    return facts, q


# ------------------------------------------------------------------ formatting
def secs(s):
    return f"{s:.2f} s" if s < 1 else f"{s:.1f} s"


def meta(channel, size, reader_s):
    """The grey result line, e.g. 'latent · 41 vectors · reader 0.06 s'."""
    cls = "lb-ch lb-acc" if CHANNELS[channel] else "lb-ch"
    return (f'<div class="lb-meta"><span class="{cls}">{SHORT[channel]}</span> · {size} · '
            f'reader {secs(reader_s)}</div>')


def pretty(ans):
    """3B output -> Markdown: the '#### N' line becomes a bold final answer. An unclosed ``` code block is
    closed, so the result line and the calculation panel after it render as HTML, not as code."""
    ans = ans.strip()
    if ans.count("```") % 2:
        ans += "\n```"
    m = re.search(r"####\s*(.*)$", ans, re.S)
    if m:
        body = ans[:m.start()].rstrip()
        final = m.group(1).strip()
        return (body + "\n\n" if body else "") + (f"**Answer: {final}**" if final else "")
    return ans


# ------------------------------------------------------------------ animation (HTML + CSS keyframes below)
def _words(facts, n=22):
    w = facts.split()
    return [html.escape(x) for x in w[:n]] + (["…"] if len(w) > n else [])


def _tok(t):
    """A token for display: visible text, escaped (leading space dropped, the chip shows the boundary)."""
    t = t.strip() or "␣"
    return html.escape(t)


def _arrow(cls="", d="0s"):
    return f'<div class="lb-arrow {cls}" style="--d:{d}">→</div>'


def anim_latent(toks, n, channel, question):
    """One-shot CSS animation (~2.9 s): the 1B's tokens turn into dots one by one, n dots flow through the
    bridge (2048-d blue -> 3072-d accent, growing), then enter the 3B."""
    shown_t = toks[:48]
    wd = min(0.03, 0.75 / max(len(shown_t), 1))
    words = "".join(f'<span class="lb-w" style="--i:{i}">{_tok(t)}<i></i></span>' for i, t in enumerate(shown_t))
    if len(toks) > len(shown_t):
        words += " …"
    shown = min(n, 64)
    g = 0.75 / max(shown, 1)
    flow = "".join(f'<i style="--i:{i};--y:{8 + (i * 37) % 80}%"></i>' for i in range(shown))
    dots = "".join(f'<i style="--i:{i}"></i>' for i in range(shown))
    stride = "2 tokens → 1 vector" if "stride 2" in channel else (
        "untrained ridge map" if "Ridge" in channel else "1 token → 1 vector")
    return (f'<div class="lb-anim" style="--g:{g:.4f}s;--wd:{wd:.4f}s">'
            + f'<div class="lb-step" style="--d:0s"><div class="lb-h">Llama 1B</div>'
              f'<div class="lb-sub">reads {len(toks)} tokens</div><div class="lb-body lb-words">{words}</div></div>'
            + _arrow(d=".95s")
            + f'<div class="lb-step lb-bridge" style="--d:1s"><div class="lb-h">Bridge</div>'
              f'<div class="lb-sub">{n} vectors · {stride}</div><div class="lb-flow">{flow}</div>'
              f'<div class="lb-dims"><span>2048-d</span><span>3072-d</span></div></div>'
            + _arrow(d="1.8s")
            + f'<div class="lb-step" style="--d:1.85s"><div class="lb-h">Llama 3B</div>'
              f'<div class="lb-sub">answers: {html.escape(question[:60])}</div>'
              f'<div class="lb-body lb-in">{dots}</div></div>'
            + '</div>')


def anim_text(facts, typed, elapsed, done, question):
    """Text channel, re-rendered as the 1B types: static boxes (no keyframes) so updates don't restart anything."""
    words = typed.split()
    tail = ("… " if len(words) > 40 else "") + html.escape(" ".join(words[-40:]))
    cursor = "" if done else '<span class="lb-cur"></span>'
    on3 = "on" if done else ""
    return ('<div class="lb-anim static">'
            + f'<div class="lb-step on"><div class="lb-h">Llama 1B</div><div class="lb-sub">reads</div>'
              f'<div class="lb-body lb-words">{" ".join(_words(facts))}</div></div>'
            + _arrow("on")
            + f'<div class="lb-step on lb-textmsg"><div class="lb-h">Text message</div>'
              f'<div class="lb-sub">1B typing · {len(words)} words · {secs(elapsed)}</div>'
              f'<div class="lb-typed">{tail}{cursor}</div></div>'
            + _arrow(on3)
            + f'<div class="lb-step {on3}"><div class="lb-h">Llama 3B</div>'
              f'<div class="lb-sub">answers: {html.escape(question[:60])}</div>'
              f'<div class="lb-body">{"reading the notes" if done else "waiting for the notes"}</div></div>'
            + '</div>')


# ------------------------------------------------------------------ "see each word": the actual numbers
def _strip(v):
    """First STRIP numbers of a vector as coloured cells: blue = negative, violet = positive, brighter = bigger."""
    x = v[:STRIP]
    m = float(x.abs().max()) or 1.0
    cells = []
    for val in x.tolist():
        a = min(abs(val) / m, 1.0) * 0.9 + 0.08
        rgb = "167,139,250" if val >= 0 else "96,165,250"
        cells.append(f'<i style="background:rgba({rgb},{a:.2f})"></i>')
    return f'<div class="lb-strip">{"".join(cells)}</div>'


def _nums(v, k=3):
    return "[" + ", ".join(f"{x:+.2f}" for x in v[:k].tolist()) + ", …]"


def breakdown_latent(toks, h, lat, channel):
    """Per vector: the token(s) the 1B read, their 1B hidden state, the vector the bridge sends, and the 3B
    word that vector is closest to. All numbers are the real ones from this run."""
    bridge, meta = M.bridges[channel]
    stride = bridge.stride
    near, cos = M.nearest_words(lat)
    h, v = h[0].float().cpu(), lat[0].float().cpu()
    T = min(len(toks), h.shape[0])
    ridge = "Ridge" in channel
    steps = [
        f"<b>1.</b> The 1B reads the facts. At layer {meta['layer']} every token is a list of <b>2048 numbers</b> "
        f"(its hidden state h).",
        "<b>2.</b> Standardize: h' = (h − mean) / std, per dimension.",
        "<b>3.</b> Multiply by a 2048×3072 matrix W (fitted in closed form, ridge regression)"
        + (" — nothing else; this bridge was never trained." if ridge else
           " + a small learned correction (MLP + attention, ×0.1)."),
    ]
    if stride > 1:
        steps.append(f"<b>4.</b> Average every {stride} neighbouring tokens into one vector (half as many vectors).")
    steps.append(f"<b>{len(steps) + 1}.</b> Scale to length {float(bridge.target_norm):.2f} (the 3B's typical word "
                 f"length) → <b>3072 numbers</b> per vector, fed to the 3B in place of word embeddings.")
    rows = []
    for j in range(v.shape[0]):
        idx = list(range(j * stride, min((j + 1) * stride, T)))
        if not idx:
            break
        tk = "".join(f'<span class="lb-tok">{_tok(toks[i])}</span>' for i in idx)
        hs = "".join(f'<div class="lb-cell">{_strip(h[i])}<div class="lb-num">{_nums(h[i])} ‖h‖={float(h[i].norm()):.1f}'
                     f'</div></div>' for i in idx)
        same = any(near[j].strip().lower() == toks[i].strip().lower() for i in idx) and near[j].strip()
        rows.append(
            f'<div class="lb-row"><div class="lb-rn">{j + 1}</div><div>{tk}</div><div>{hs}</div>'
            f'<div class="lb-arr">→</div>'
            f'<div class="lb-cell">{_strip(v[j])}<div class="lb-num">{_nums(v[j])} ‖v‖={float(v[j].norm()):.2f}</div></div>'
            f'<div><span class="lb-tok lb-near">{_tok(near[j])}</span>'
            f'<span class="lb-num"> cos {cos[j]:.2f}{" ✓" if same else ""}</span></div></div>')
    match = sum(1 for r in rows if " ✓</span>" in r)
    head = ('<div class="lb-row lb-head"><div>#</div><div>token(s) the 1B read</div>'
            f'<div>1B hidden state · first {STRIP} of 2048</div><div></div>'
            f'<div>vector sent · first {STRIP} of 3072</div><div>closest 3B word</div></div>')
    return (f'<details class="lb-x" open><summary>🔍 How each word became a vector: {T} tokens → {v.shape[0]} vectors '
            f'({match} point straight at the same word in the 3B)</summary>'
            f'<div class="lb-calc">{"".join(f"<div>{s}</div>" for s in steps)}</div>'
            f'<div class="lb-legend">cells: <span class="lb-neg">■</span> negative · '
            f'<span class="lb-pos">■</span> positive · brighter = larger · ✓ = the 3B reads this vector as the '
            f'same word the 1B read</div>'
            f'<div class="lb-table">{head}{"".join(rows)}</div></details>')


def breakdown_text(msg):
    """Text channel: the tokens the 1B wrote; the 3B looks each one up in its own embedding table."""
    tok = M.receiver.tok
    ids = tok(msg, add_special_tokens=False).input_ids
    chips = "".join(f'<span class="lb-tok">{_tok(tok.decode([i]))}</span>' for i in ids)
    return (f'<details class="lb-x" open><summary>🔍 The text message: {len(ids)} tokens written one at a time'
            f'</summary><div class="lb-calc"><div>The 1B had to <b>generate</b> these tokens one by one (that is the '
            f'slow part). The 3B then looks each one up in its embedding table: 1 token → 3072 numbers.</div></div>'
            f'<div class="lb-toks">{chips}</div></details>')


# ------------------------------------------------------------------ chat handler
def respond(message, history, channel):
    message = (message or "").strip()
    if not message:
        return
    if channel not in CHANNELS:
        channel = DEFAULT
    facts, q = split(message)

    if CHANNELS[channel] is None:  # text: the 1B really types its notes, word by word
        t0, msg = time.perf_counter(), ""
        for msg in M.stream_message(facts):
            yield anim_text(facts, msg, time.perf_counter() - t0, False, q)
        reader_s = time.perf_counter() - t0
        msg = msg.strip()
        size = f"{len(M.sender.tok(msg, add_special_tokens=False).input_ids)} tokens"
        yield anim_text(facts, msg, reader_s, True, q)
        pieces = M.start_answer(msg, q)
        time.sleep(0.6)
        detail = lambda: breakdown_text(msg)
    else:  # latent: one-shot animation while the 3B already starts answering
        lat, reader_s, h = M.read_latent(facts, channel)
        toks = M.tokens(facts)
        size = f"{lat.shape[1]} vectors"
        yield anim_latent(toks, lat.shape[1], channel, q)
        pieces = M.start_answer(lat, q)
        time.sleep(ANIM_S)
        detail = lambda: breakdown_latent(toks, h, lat, channel)

    line = meta(channel, size, reader_s)
    ans = ""
    for p in pieces:
        ans += p
        yield pretty(ans) + "\n\n" + line
    if MUSE is None:
        yield pretty(ans) + "\n\n" + line + "\n\n" + detail()
        return
    yield pretty(ans) + "\n\n" + line + "\n\n" + muse_line(None)
    v = MUSE.check(message, ans, channel)
    yield pretty(ans) + "\n\n" + line + "\n\n" + muse_line(v) + "\n\n" + detail()


# ------------------------------------------------------------------ Muse Spark check (optional, needs MODEL_API_KEY)
class MuseCheck:
    """After the local answer, Muse Spark checks it. On a reject it writes a corrected solution, which is
    shown and appended to data/chat_corrections.jsonl: training targets for the next improve.py round."""

    def __init__(self, model):
        from openai import OpenAI

        from improve import MUSE_BASE, TEACH_INSTR
        self.model, self.instr = model, TEACH_INSTR
        self.client = OpenAI(base_url=MUSE_BASE, api_key=os.environ["MODEL_API_KEY"])
        self.log = Path(__file__).resolve().parent / "data" / "chat_corrections.jsonl"

    def check(self, problem, answer, channel):
        import json
        t0 = time.perf_counter()
        try:
            r = self.client.chat.completions.create(
                model=self.model, reasoning_effort="low",
                messages=[{"role": "developer", "content": self.instr},
                          {"role": "user", "content": f"Problem:\n{problem}\n\nTeam's solution:\n{answer}"}])
            v = json.loads(re.search(r"\{.*\}", r.choices[0].message.content, re.S).group(0))
            assert v.get("verdict") in ("accept", "reject")
        except Exception as e:  # offline / bad JSON: the local answer still stands
            return {"verdict": "unavailable", "reason": type(e).__name__, "seconds": time.perf_counter() - t0}
        v["seconds"] = time.perf_counter() - t0
        if v["verdict"] == "reject" and "####" in (v.get("solution") or ""):
            with open(self.log, "a", encoding="utf-8") as f:
                f.write(json.dumps({"problem": problem, "team_answer": answer, "channel": channel,
                                    "muse": self.model, "reason": v.get("reason"), "solution": v["solution"]}) + "\n")
        return v


def muse_line(v):
    name = MUSE.model if MUSE else "Muse Spark"
    if v is None:
        return f'<span style="color:{ACCENT}">◆ {name} is checking the answer…</span>'
    reason = html.escape(v.get("reason") or "")
    if v["verdict"] == "accept":
        return (f'<span style="color:{ACCENT}">◆ {name} checked it ✓</span> '
                f'<span style="color:#9ca3af">{reason} ({v["seconds"]:.1f} s)</span>')
    if v["verdict"] == "reject":
        sol = v.get("solution") or ""
        fix = sol.rsplit("####", 1)[-1].strip() if "####" in sol else "?"
        return (f'<span style="color:#f87171">◆ {name} rejected it ✗</span> <span style="color:#9ca3af">{reason}</span><br>'
                f'<b>Corrected answer: {html.escape(fix)}</b> <span style="color:#9ca3af">— saved as a lesson for the '
                f'next training round ({v["seconds"]:.1f} s)</span>')
    return f'<span style="color:#9ca3af">◆ {name} unavailable ({html.escape(v.get("reason") or "")}); local answer shown</span>'


# ------------------------------------------------------------------ UI
def empty_state():
    chips = "".join(
        f'<div class="lb-chip" role="button"><div class="lb-chip-q">{html.escape(q)}</div>'
        f'<div class="lb-chip-t">{e} {html.escape(t)}</div>'
        f'<div class="lb-chip-s">{html.escape(q[:70])}…</div></div>' for e, t, q in EXAMPLES)
    return (f'<div class="lb-empty"><div class="lb-tag">Two Llama models share thoughts as vectors instead of text.'
            f'</div><div class="lb-chips">{chips}</div></div>')


def option_hint_css():
    """One-line hint under each dropdown option (pseudo-element per option, in CHANNELS order)."""
    return "#lb-channel ul.option-list li {flex-wrap: wrap !important}\n" + "\n".join(
        f'#lb-channel ul.option-list li:nth-child({i})::after '
        f'{{content: "{HINTS[k]}"; display: block; flex-basis: 100%; padding-left: 18px; font-size: 13px; color: #8e8e8e; margin-top: 2px}}'
        for i, k in enumerate(CHANNELS, 1))


CSS = """
html, body, .gradio-container, gradio-app {background: #212121 !important}
.gradio-container {width: 100% !important; max-width: 1200px !important; margin: 0 auto !important;
  padding: 0 12px !important; box-sizing: border-box; font-size: 20px !important}
#lb-title {text-align: center; color: #ececec; margin: 2px 0 0 0}
#lb-title h1 {font-size: 28px !important; font-weight: 600 !important; margin: 0 !important}
#lb-title p {color: #9b9b9b; font-size: 16px; margin: 4px 0 0 0}
/* composer: one rounded card = text on top, bar (local label · channel · send) below */
.gr-group:has(#lb-input), div:has(> #lb-input), .row:has(#lb-input), .form:has(#lb-input), .group:has(#lb-input), .styler:has(#lb-input) {
  background: transparent !important; border: none !important; box-shadow: none !important; padding: 0 !important;
  gap: 0 !important}
#lb-input {background: #2f2f2f !important; border: 1px solid #3e3e3e !important; border-radius: 26px !important;
  padding: 14px 14px 10px 18px !important; box-shadow: 0 6px 28px rgba(0, 0, 0, .28) !important;
  transition: border-color .15s}
#lb-input:focus-within {border-color: #575757 !important}
#lb-input label, #lb-input .input-container {background: transparent !important; border: none !important;
  box-shadow: none !important; padding: 0 !important}
#lb-input textarea {background: transparent !important; border: none !important; box-shadow: none !important;
  border-radius: 0 !important; padding: 2px 0 !important; font-size: 19px !important; color: #ececec !important;
  min-height: 30px}
#lb-input textarea::placeholder {color: #8e8e8e !important}
#lb-bar {display: none}
#lb-input #lb-bar {display: flex !important; align-items: center; gap: 8px; margin-top: 8px; padding: 0 !important;
  background: transparent !important; border: none !important; flex-wrap: nowrap !important}
#lb-bar > * {flex: 0 0 auto !important; min-width: 0 !important; width: auto !important; padding: 0 !important;
  background: transparent !important; border: none !important; box-shadow: none !important}
#lb-bar > #lb-left {flex: 1 1 auto !important}
#lb-bar .html-container, #lb-bar .prose {padding: 0 !important}
.lb-local {display: flex; align-items: center; gap: 7px; font-size: 14px; white-space: nowrap}
.lb-local, .lb-local span {color: #9b9b9b !important}
.lb-local b {width: 7px; height: 7px; border-radius: 50%; background: #34d399; box-shadow: 0 0 6px #34d399}
#lb-channel .wrap {background: transparent !important; border: none !important; box-shadow: none !important;
  border-radius: 10px !important; padding: 2px 4px 2px 10px !important; transition: background .15s}
#lb-channel:hover .wrap {background: #3a3a3a !important}
#lb-channel input {font-size: 15px !important; font-weight: 500; color: ACC !important; background: transparent !important;
  text-align: right; width: 215px !important; padding-right: 26px !important; cursor: pointer}
#lb-channel .icon-wrap {color: #8e8e8e !important; right: 6px !important}
#lb-channel ul.option-list {min-width: 0 !important; width: 300px !important; left: auto !important; right: 0 !important;
  background: #2a2a2a !important; border: 1px solid #3e3e3e !important; border-radius: 12px !important;
  box-shadow: 0 10px 30px rgba(0,0,0,.45); padding: 4px !important}
#lb-channel ul.option-list li {font-size: 15px !important; padding: 8px 12px !important; width: 100% !important;
  box-sizing: border-box !important; border-radius: 8px}
#lb-channel ul.option-list li.selected, #lb-channel ul.option-list li.active {color: ACC !important}
.lb-send {width: 36px; height: 36px; border-radius: 50%; border: none; background: ACC; cursor: pointer;
  display: grid; place-items: center; transition: transform .1s, filter .15s; padding: 0}
.lb-send:hover {filter: brightness(1.1)} .lb-send:active {transform: scale(.94)}
.lb-send svg {width: 18px; height: 18px}

/* thin dark scrollbars everywhere (no white bars) */
* {scrollbar-width: thin; scrollbar-color: #4a4a4a transparent}
::-webkit-scrollbar {width: 8px; height: 8px}
::-webkit-scrollbar-track {background: transparent}
::-webkit-scrollbar-thumb {background: #4a4a4a; border-radius: 8px}
::-webkit-scrollbar-thumb:hover {background: #5c5c5c}
::-webkit-scrollbar-button {display: none; height: 0}
.message-wrap .prose :is(p, li, ol, ul, strong, em, code, h1, h2, h3, h4, td, th, blockquote) {
  font-size: 20px !important; line-height: 1.6 !important}
.message-wrap .message {border: none !important}
.message-row.user-row .message {background: #303030 !important; border-radius: 22px !important}
.message-row.bot-row .message {background: transparent !important}
.block, .form, .row, .bubble-wrap, .chatbot {background: transparent !important; border: none !important;
  box-shadow: none !important}
.avatar-container img {border-radius: 50% !important}
footer {display: none !important}
/* keep only the copy buttons */
button[aria-label="Clear" i], button[title="Clear" i], button[aria-label="Retry" i], button[title="Retry" i],
button[aria-label="Undo" i], button[title="Undo" i], button[aria-label*="share" i] {display: none !important}

/* result + status lines */
.message-wrap .lb-meta {color: #8e8e8e !important; font-size: 15px !important; margin-top: 12px}
.message-wrap .lb-meta .lb-acc {color: ACC !important; font-weight: 600}
.message-wrap .lb-status {color: #8e8e8e !important; font-size: 15px !important; font-style: italic; margin-bottom: 10px}

/* empty state */
.lb-empty {display: flex; flex-direction: column; align-items: center; gap: 12px; padding: 4px}
.lb-tag {color: #b4b4b4; font-size: 18px; text-align: center}
.lb-tag2 {color: #8e8e8e; font-size: 14px}
.lb-chips {display: flex; flex-wrap: wrap; gap: 8px; justify-content: center; max-width: 900px}
.lb-chip {flex: 1 1 240px; max-width: 290px; cursor: pointer; text-align: left; background: #2a2a2a;
  border: 1px solid #3a3a3a; border-radius: 14px; padding: 8px 12px; transition: border-color .15s, background .15s}
.lb-chip:hover {border-color: ACC; background: #2f2b3a}
.lb-chip-q {display: none !important}
.lb-chip-t {color: #ececec; font-size: 16px; font-weight: 600}
.lb-chip-s {color: #8e8e8e; font-size: 13px; margin-top: 2px; line-height: 1.35; white-space: nowrap; overflow: hidden; text-overflow: ellipsis}

/* bridge animation */
.message-wrap .lb-anim {display: flex; align-items: stretch; gap: 6px; flex-wrap: wrap; margin: 2px 0 6px}
.message-wrap .lb-step {flex: 1 1 140px; min-width: 120px; background: #2a2a2a; border: 1px solid #3a3a3a;
  border-radius: 14px; padding: 10px 12px; opacity: .3; animation: lbOn .35s ease var(--d, 0s) forwards}
.message-wrap .lb-anim.static .lb-step, .message-wrap .lb-anim.static .lb-arrow {animation: none}
.message-wrap .lb-step.on {opacity: 1; border-color: ACC; box-shadow: 0 0 14px ACC40}
.message-wrap .lb-arrow {align-self: center; color: #666; font-size: 20px !important; opacity: .3;
  animation: lbArrow .3s ease var(--d, 0s) forwards}
.message-wrap .lb-arrow.on {opacity: 1; color: ACC}
.message-wrap .lb-h {font-weight: 600; font-size: 15px !important; color: #ececec; line-height: 1.3 !important}
.message-wrap .lb-sub {font-size: 12px !important; color: #8e8e8e; margin-bottom: 6px; line-height: 1.3 !important}
.message-wrap .lb-body, .message-wrap .lb-body * {font-size: 13px !important; color: #bdbdbd; line-height: 1.5 !important}
@keyframes lbOn {to {opacity: 1; border-color: ACC; box-shadow: 0 0 14px ACC40}}
@keyframes lbArrow {to {opacity: 1; color: ACC}}
/* 1B: each word fades out while a glowing dot pops up in its place */
.message-wrap .lb-w {position: relative; display: inline-block; margin: 0 3px 1px 0;
  animation: lbFade .3s ease calc(.3s + var(--i) * var(--wd, .03s)) forwards}
.message-wrap .lb-w i {position: absolute; left: 50%; top: 50%; width: 7px; height: 7px; margin: -3.5px 0 0 -3.5px;
  border-radius: 50%; background: SRC; box-shadow: 0 0 8px SRC; transform: scale(0);
  animation: lbPop .3s ease calc(.3s + var(--i) * var(--wd, .03s)) forwards}
@keyframes lbFade {to {color: transparent}}
@keyframes lbPop {to {transform: scale(1)}}
/* bridge: dots flow left -> right, 2048-d (small, blue) -> 3072-d (bigger, accent) */
.message-wrap .lb-flow {position: relative; height: 58px; border-radius: 10px; overflow: hidden;
  background: linear-gradient(90deg, SRC1a, ACC1a)}
.message-wrap .lb-flow i {position: absolute; left: 0; top: var(--y); width: 5px; height: 5px; border-radius: 50%;
  background: SRC; opacity: 0; animation: lbFlow .75s ease-in-out calc(1.05s + var(--i) * var(--g)) both}
@keyframes lbFlow {0% {left: 0; opacity: 0; width: 5px; height: 5px; background: SRC}
  15% {opacity: 1}
  85% {left: calc(100% - 10px); opacity: 1; width: 9px; height: 9px; background: ACC; box-shadow: 0 0 8px ACC}
  100% {left: calc(100% - 4px); opacity: 0; width: 9px; height: 9px; background: ACC}}
.message-wrap .lb-dims {display: flex; justify-content: space-between; font-size: 11px !important; color: #8e8e8e;
  margin-top: 4px; line-height: 1.2 !important}
.message-wrap .lb-dims span {font-size: 11px !important}
/* 3B: the vectors arrive */
.message-wrap .lb-in i {display: inline-block; width: 7px; height: 7px; margin: 2px; border-radius: 50%;
  background: ACC; box-shadow: 0 0 6px ACC; opacity: 0;
  animation: lbIn .25s ease calc(1.9s + var(--i) * var(--g) * .8) forwards}
@keyframes lbIn {to {opacity: 1}}
/* text channel: typed notes with a blinking cursor */
.message-wrap .lb-typed {font-size: 13px !important; color: #e5e5e5; line-height: 1.45 !important; min-height: 58px;
  max-height: 120px; overflow: hidden; font-family: ui-monospace, Consolas, monospace}
.message-wrap .lb-cur {display: inline-block; width: 7px; height: 14px; margin-left: 2px; vertical-align: -2px;
  background: #e5e5e5; animation: lbBlink .8s steps(1) infinite}
@keyframes lbBlink {50% {opacity: 0}}
.message-wrap .lb-textmsg {flex: 2 1 220px}

/* "see each word" panel */
.message-wrap .lb-x {margin-top: 14px; border: 1px solid #3a3a3a; border-radius: 14px; padding: 10px 14px; background: #262626}
.message-wrap .lb-x summary {cursor: pointer; color: #c4b5fd !important; font-size: 16px !important; font-weight: 600}
.message-wrap .lb-x div, .message-wrap .lb-x span {font-size: 13px !important; line-height: 1.45 !important}
.message-wrap .lb-calc {margin: 10px 0; color: #cfcfcf; display: flex; flex-direction: column; gap: 3px}
.message-wrap .lb-calc div {font-size: 14px !important}
.message-wrap .lb-legend {color: #8e8e8e; margin-bottom: 8px}
.message-wrap .lb-neg {color: SRC} .message-wrap .lb-pos {color: ACC}
.message-wrap .lb-table {max-height: 420px; overflow-y: auto; padding-right: 4px}
.message-wrap .lb-row {display: grid; grid-template-columns: 26px minmax(90px, 1fr) minmax(150px, 2fr) 16px minmax(150px, 2fr) minmax(90px, 1fr);
  gap: 8px; align-items: center; padding: 5px 0; border-bottom: 1px solid #303030}
.message-wrap .lb-head div {color: #8e8e8e; font-size: 12px !important}
.message-wrap .lb-rn {color: #6b6b6b}
.message-wrap .lb-arr {color: ACC}
.message-wrap .lb-cell {display: flex; flex-direction: column; gap: 2px; margin: 2px 0}
.message-wrap .lb-strip {display: flex; height: 12px; border-radius: 3px; overflow: hidden; background: #1b1b1b}
.message-wrap .lb-strip i {flex: 1; margin-right: 1px}
.message-wrap .lb-num {font-family: ui-monospace, Consolas, monospace; font-size: 11px !important; color: #9b9b9b}
.message-wrap .lb-tok {display: inline-block; font-family: ui-monospace, Consolas, monospace; background: #333;
  border-radius: 5px; padding: 0 5px; margin: 1px 3px 1px 0; color: #ececec; white-space: pre}
.message-wrap .lb-near {background: ACC33; color: #ede9fe; max-width: 110px; overflow: hidden; text-overflow: ellipsis; vertical-align: middle}
.message-wrap .lb-toks {display: flex; flex-wrap: wrap; gap: 2px; margin-top: 6px}

/* narrow screens */
@media (max-width: 640px) {
  .gradio-container {font-size: 17px !important; padding: 0 6px !important}
  #lb-title h1 {font-size: 22px !important}
  #lb-title p {font-size: 13px}
  .message-wrap .prose :is(p, li, ol, ul, strong, em, code, h1, h2, h3, h4, td, th, blockquote) {font-size: 17px !important}
  #lb-input textarea {font-size: 17px !important}
  #lb-channel input {width: 150px !important; font-size: 14px !important}
  .lb-local span {display: none}
  .message-wrap .lb-arrow {display: none}
  .message-wrap .lb-step {flex: 1 1 45%; min-width: 0}
  .lb-chip {max-width: none}
  .message-wrap .lb-row {grid-template-columns: 22px 1fr}
  .message-wrap .lb-arr, .message-wrap .lb-head {display: none}
}
""".replace("ACC", ACCENT).replace("SRC", SOURCE)

JS = """() => {
  document.body.classList.add('dark');
  document.documentElement.classList.add('dark');
  const send = (ta) => setTimeout(() => {  // same as pressing Enter in the message box
    for (const t of ['keydown', 'keypress'])
      ta.dispatchEvent(new KeyboardEvent(t, {key: 'Enter', code: 'Enter', keyCode: 13, which: 13,
                                             bubbles: true, cancelable: true}));
  }, 50);
  // put the bar (local label · channel · send) inside the message box, like a chat composer
  const dock = () => {
    const box = document.querySelector('#lb-input'), bar = document.querySelector('#lb-bar');
    if (!box || !bar) return setTimeout(dock, 100);
    if (bar.parentElement !== box) box.appendChild(bar);
  };
  dock();
  document.addEventListener('click', (e) => {
    const ta = document.querySelector('#lb-input textarea');
    if (!ta) return;
    if (e.target.closest('.lb-send')) {  // send button
      if (ta.value.trim()) send(ta);
      return;
    }
    const chip = e.target.closest('.lb-chip');  // example cards: fill the message box and send
    const q = chip && chip.querySelector('.lb-chip-q');
    if (!q) return;
    Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, 'value').set.call(ta, q.textContent.trim());
    ta.dispatchEvent(new Event('input', {bubbles: true}));
    ta.focus();
    send(ta);
  });
}"""


def build():
    with gr.Blocks(title="LatentBridge Chat", fill_height=True) as demo:
        gr.HTML('<div id="lb-title"><h1>LatentBridge Chat</h1></div>')
        with gr.Row(elem_id="lb-bar"):  # moved inside the message box by JS (see dock())
            gr.HTML('<div class="lb-local"><span>Llama 3.2 1B → 3B' + (' · checked by Muse Spark' if MUSE else '') + '</span></div>',
                    elem_id="lb-left")
            channel = gr.Dropdown(list(CHANNELS), value=DEFAULT, show_label=False, container=False,
                                  elem_id="lb-channel", interactive=True)
            gr.HTML('<button class="lb-send" title="Send" aria-label="Send"><svg viewBox="0 0 24 24" fill="none" '
                    'stroke="#17121f" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round">'
                    '<path d="M12 19V5M5 12l7-7 7 7"/></svg></button>', elem_id="lb-send-wrap")
        chatbot = gr.Chatbot(show_label=False, height="calc(100vh - 250px)", sanitize_html=False, render_markdown=True,
                             layout="bubble", placeholder=empty_state(),
                             avatar_images=(str(ASSETS / "user.svg"), str(ASSETS / "bot.svg")),
                             buttons=["copy", "copy_all"])
        gr.ChatInterface(respond, chatbot=chatbot, additional_inputs=[channel],
                         textbox=gr.Textbox(placeholder="Ask a math or dev problem…", show_label=False, lines=1,
                                            max_lines=8, autofocus=True, elem_id="lb-input"),
                         fill_height=True, show_progress="hidden")
    return demo


def main():
    global M
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--no-muse", action="store_true", help="skip the Muse Spark check (fully offline)")
    ap.add_argument("--muse", default=os.getenv("IMPROVER_MODEL", "muse-spark-1.3"))
    args = ap.parse_args()
    global MUSE
    MUSE = MuseCheck(args.muse) if os.getenv("MODEL_API_KEY") and not args.no_muse else None
    M = Models()
    theme = gr.themes.Base(primary_hue="violet", neutral_hue="neutral",
                           font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"])
    build().queue().launch(server_port=args.port, share=args.share, theme=theme, css=CSS + option_hint_css(),
                           js=JS, footer_links=[], allowed_paths=[str(ASSETS)])


if __name__ == "__main__":
    main()

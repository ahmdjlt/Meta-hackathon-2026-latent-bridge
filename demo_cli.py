"""LatentBridge terminal demo (rich). Built for a dark terminal and screen recording.

python demo_cli.py --example 3            replay cached example 3 (instant, no GPU; demo/cache.json)
python demo_cli.py --live "<problem>"     run the real pipeline step by step (GPU)
python demo_cli.py --list                 list cached examples

Cached replays wait for the recorded reader times exactly (2.7 s vs 0.06 s) and shorten the long 3B
answer step to 1.5 s (the printed time is the recorded one); add --realtime to replay at recorded speed.
"""
import argparse
import sys
import time

from rich import box
from rich.align import Align
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from demo.views import load_cache

LAT, TXT, MUSE, DIM = "#4c9bf0", "#f08a52", "#c9a0ff", "grey62"
con = Console(highlight=False)


class Replay:
    """Cached mode: each step takes its recorded time (long 3B steps shortened unless realtime)."""

    def __init__(self, realtime):
        self.realtime = realtime

    def step(self, label, color, key, secs):
        wait = secs if (self.realtime or secs <= 3.0) else 1.5
        with con.status(Text.assemble((f"[{label}] ", f"bold {color}"), ("working…", DIM)), spinner="dots"):
            time.sleep(wait)
        return secs


class Live:
    """Live mode: each step does the real work (work[key]() fills in the example dict)."""

    def __init__(self, work):
        self.work = work

    def step(self, label, color, key, _secs):
        with con.status(Text.assemble((f"[{label}] ", f"bold {color}"), ("working…", DIM)), spinner="dots"):
            t0 = time.perf_counter()
            self.work[key]()
            return time.perf_counter() - t0


def done(label, color, secs, msg=""):
    con.print(Text.assemble((f"[{label}]", f"bold {color}"), (f" {secs:5.2f} s  ", "bold white"), msg))


def vector_strip(probe):
    """One block per vector; brightness = how close it is to a real 3B word (cosine)."""
    shades = ["#1d3f66", "#24548a", "#2a6cb3", "#3a86d6", "#6aa8ec", "#a9cdf6"]
    t = Text()
    for p in probe:
        t.append("█", style=shades[min(5, max(0, int(p["cos"] * 6)))])
    return t


def probe_text(probe):
    """Nearest 3B word per vector: bold = exactly the word the 1B read, italic grey = a nearby word."""
    t = Text()
    for p in probe:
        t.append(p["nearest"].replace("\n", "⏎"), style=("bold white" if p["match"] else "italic grey70"))
        t.append("·", style="grey35")
    return t


def mark(ans, gold):
    if gold is None:
        return Text("no gold", style=DIM)
    return Text("✔ correct", style="bold green") if ans == gold else Text("✘ wrong", style="bold red")


def show(ex, runner):
    con.rule(Text("LatentBridge", style="bold white"), style=LAT)
    con.print(Align.center(Text("small Llama agents that talk in vectors instead of text", style="italic " + DIM)))
    con.print(Panel(Text(ex["question"], style="white"), title="problem", title_align="left",
                    border_style="grey50", padding=(0, 2)))
    gold = ex.get("gold")

    # 1 plan
    s = runner.step("Muse Spark", MUSE, "plan", ex["plan"].get("orchestrator_s") or 0.4)
    plan = ex["plan"]
    done("Muse Spark", MUSE, s, f"plan by {plan.get('orchestrator')}")
    g = Table.grid(padding=(0, 2))
    g.add_column(width=4); g.add_column(no_wrap=True); g.add_column()
    g.add_row("", Text("facts → 1B", style=LAT), Text(plan["facts"], style="white"))
    g.add_row("", Text("question → 3B", style="grey70"), Text(plan["question"], style="white"))
    con.print(g)

    # 2-5 latent pipeline
    con.print()
    con.print(Text("  LATENT PIPELINE", style=f"bold {LAT}"))
    runner.step("1B reader", LAT, "latent_read", ex["latent"].get("reader_s", 0))
    L = ex["latent"]
    done("1B reader", LAT, L["reader_s"], "one forward pass over the facts, no text written")
    done("bridge   ", LAT, 0.0, f"→ {L['tokens_sent']} vectors × {L.get('dim', 3072)} numbers")
    con.print(Text("      ").append(vector_strip(L["probe"])))
    con.print(Text("      what the 3B received (nearest word per vector): ", style=DIM)
              .append(probe_text(L["probe"])))
    runner.step("3B solver", LAT, "latent_solve", L.get("solver_s", 0))
    done("3B solver", LAT, L["solver_s"], Text.assemble(("answer ", DIM), (L["answer"] or "—", "bold white"),
                                                        "  ", mark(L["answer"], gold)))
    s = runner.step("Muse Spark", MUSE, "verify", (L.get("verify") or {}).get("verify_s") or 0.5)
    v = L.get("verify") or {}
    vtxt = {"accept": ("accept ✔", "bold green"), "reject": ("reject ✘ → retry over text", "bold red")}.get(
        v.get("verdict"), ("unavailable", DIM))
    done("Muse Spark", MUSE, s, Text.assemble(("verify: ", DIM), vtxt))
    if v.get("reason"):
        con.print(Text("      " + v["reason"], style="italic grey70"))

    # text pipeline, same problem
    con.print()
    con.print(Text("  TEXT PIPELINE (baseline)", style=f"bold {TXT}"))
    runner.step("1B reader", TXT, "text_read", ex["text"].get("reader_s", 0))
    T = ex["text"]
    done("1B reader", TXT, T["reader_s"], f"wrote {T['tokens_sent']} tokens, one at a time:")
    con.print(Panel(Text(T["message"], style="grey85"), border_style=TXT, padding=(0, 1), width=100))
    runner.step("3B solver", TXT, "text_solve", T.get("solver_s", 0))
    done("3B solver", TXT, T["solver_s"], Text.assemble(("answer ", DIM), (T["answer"] or "—", "bold white"),
                                                        "  ", mark(T["answer"], gold)))

    # side by side
    con.print()
    tb = Table(box=box.ROUNDED, border_style="grey50", header_style="bold white")
    tb.add_column("", style=DIM)
    tb.add_column("TEXT", style=TXT, justify="right")
    tb.add_column("LATENT", style=f"bold {LAT}", justify="right")
    tb.add_row("message size", f"{T['tokens_sent']} tokens", f"{L['tokens_sent']} vectors")
    tb.add_row("reader time", f"{T['reader_s']:.2f} s", f"{L['reader_s']:.2f} s")
    tb.add_row("total time", f"{T['total_s']:.1f} s", f"{L['total_s']:.1f} s")
    tb.add_row("answer", T["answer"] or "—", L["answer"] or "—")
    tb.add_row("correct?", mark(T["answer"], gold), mark(L["answer"], gold))
    speed = T["reader_s"] / max(L["reader_s"], 1e-3)
    foot = Text.assemble(("gold answer ", DIM), (str(gold if gold is not None else "n/a"), "bold white"),
                         ("  ·  reader ", DIM), (f"{speed:.0f}× faster", f"bold {LAT}"),
                         ("  ·  500-problem test: latent 70.4% vs text 60.2%, p = 1.6e-5", DIM))
    con.print(Align.center(tb))
    con.print(Align.center(foot))
    con.rule(style=LAT)


def live_example(question):
    """Load the models, then return (example dict, Live runner) whose steps fill the dict for real."""
    import torch
    from common import CKPT, DEVICE, extract_number
    from demo.build_cache import probe
    from pipeline import Pipeline
    with con.status(Text("loading Llama 3.2 1B + 3B (4-bit) and the bridge…", style=DIM)):
        pipe = Pipeline(CKPT / "bridge_main_token.pt")
        pipe.sender.hidden("Warm-up.", pipe.meta["layer"])  # first CUDA call is slow; keep it out of the timings
    ex = {"question": question, "gold": None, "plan": {}, "latent": {}, "text": {}}
    st = {}

    def timed(fn):
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = fn()
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        return out, time.perf_counter() - t0

    def plan():
        ex["plan"] = pipe.orch.plan(question)

    def latent_read():
        m, facts = pipe.meta, ex["plan"]["facts"]

        def go():
            h = pipe.sender.hidden(facts, m["layer"], skip_template=m.get("skip_template", False),
                                   content_only=m.get("content_only", False))
            with torch.no_grad():
                return pipe.bridge(h)
        st["lat"], s = timed(go)
        rows, _ = probe(pipe, st["lat"][0].float().cpu(), facts)
        ex["latent"].update(reader_s=s, tokens_sent=st["lat"].shape[1], dim=st["lat"].shape[2], probe=rows)

    def latent_solve():
        q = ex["plan"]["question"]
        ans, s = timed(lambda: pipe.receiver.generate(pipe.receiver.latent_inputs(st["lat"], q)[0]))
        L = ex["latent"]
        L.update(answer_raw=ans, answer=extract_number(ans), solver_s=s, total_s=L["reader_s"] + s)

    def verify():
        ex["latent"]["verify"] = pipe.orch.verify(question, ex["latent"]["answer_raw"])

    def text_read():
        msg, s = timed(lambda: pipe.sender.message(ex["plan"]["facts"]))
        ex["text"].update(message=msg, reader_s=s,
                          tokens_sent=len(pipe.sender.tok(msg, add_special_tokens=False).input_ids))

    def text_solve():
        T, q = ex["text"], ex["plan"]["question"]
        ans, s = timed(lambda: pipe.receiver.generate(pipe.receiver.text_inputs(T["message"], q)[0]))
        T.update(answer_raw=ans, answer=extract_number(ans), solver_s=s, total_s=T["reader_s"] + s)

    work = dict(plan=plan, latent_read=latent_read, latent_solve=latent_solve, verify=verify,
                text_read=text_read, text_solve=text_solve)
    return ex, Live(work)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--example", type=int, help="cached example number (1-based, see --list)")
    g.add_argument("--live", metavar="PROBLEM", help="run the real pipeline on this problem (GPU)")
    g.add_argument("--list", action="store_true")
    ap.add_argument("--realtime", action="store_true", help="replay every cached step at its recorded speed")
    a = ap.parse_args()
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
    cache = load_cache()
    if a.list:
        for i, ex in enumerate(cache, 1):
            con.print(f"[bold]{i}[/]. {ex['question'][:110]}")
        return
    if a.live:
        ex, runner = live_example(a.live)
        ex["gold"] = next((c["gold"] for c in cache if c["question"].strip() == a.live.strip()), None)
        con.clear()
        show(ex, runner)
        return
    if not 1 <= a.example <= len(cache):
        sys.exit(f"--example must be 1..{len(cache)} (python demo_cli.py --list)")
    show(cache[a.example - 1], Replay(a.realtime))


if __name__ == "__main__":
    main()

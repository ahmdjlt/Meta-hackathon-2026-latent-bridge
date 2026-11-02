"""LatentBridge CLI agent.

Brain: Muse Spark (Meta Model API, OpenAI-compatible tool calling) plans, calls tools, and verifies the
final answer. Reader: the latent bridge (Llama 3.2 1B -> bridge -> Llama 3.2 3B) reads fetched pages
locally, so long page text never goes to the brain.

Tools: web_search, fetch_url, read_and_answer, run_python (asks before running).

Usage:
  python agent/cli.py "What is the maximum depth of Lake Baikal?"
  python agent/cli.py --channel text "..."         # 1B writes text notes instead of latents
  python agent/cli.py --transcript agent/demo_runs/a.txt "..."
Keys come only from .env (MODEL_API_KEY, optional MUSE_MODEL / MUSE_BASE_URL).
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SANDBOX = HERE / "sandbox"
MAX_STEPS = 8

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

MUSE_MODEL = os.getenv("MUSE_MODEL") or "muse-spark-1.1"
MUSE_BASE = os.getenv("MUSE_BASE_URL", "https://api.meta.ai/v1")

SYSTEM = """You are the planner of a small agent running on a laptop. You have tools:
- web_search(query): find pages. Returns titles, URLs and snippets.
- fetch_url(url): downloads a page and stores its cleaned text locally as a document id (doc1, doc2, ...).
  You only see a short preview, not the full text.
- read_and_answer(source, question): a local reader (Llama 3.2 1B + a latent bridge + Llama 3.2 3B) reads
  the document (pass the doc id as source) and answers one focused, factual question about it.
- run_python(code): runs a Python script (10 s limit) in a sandbox folder after the user approves it.
  Print the results you need; you get stdout and stderr back.
Rules: For facts from the web, search, fetch the best page, then ask read_and_answer a short specific
question. Prefer read_and_answer over relying on snippets. For coding tasks, write a complete script and
run it. You have at most 8 tool calls. Before each tool call, say in one short sentence what you plan.
When done, reply with the final answer in one or two sentences, without calling a tool."""

VERIFY = """You verify an agent's final answer. You get the user's task, the tool results the agent saw,
and its proposed answer. Accept if the answer is supported by the tool results and answers the task.
Return ONLY JSON: {"verdict": "accept" or "reject", "reason": "<one short sentence>",
"final_answer": "<the answer to print, corrected only if the tool results clearly support a correction>"}"""

TOOLS = [
    {"type": "function", "function": {
        "name": "web_search", "description": "Search the web (DuckDuckGo). Returns top results.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string"}, "max_results": {"type": "integer", "default": 5}},
            "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "fetch_url", "description": "Download a web page, store its cleaned text, return a doc id and preview.",
        "parameters": {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]}}},
    {"type": "function", "function": {
        "name": "read_and_answer",
        "description": "Local small-model reader answers a factual question about a stored document (doc id) or given text.",
        "parameters": {"type": "object", "properties": {
            "source": {"type": "string", "description": "doc id from fetch_url (e.g. doc1) or raw text"},
            "question": {"type": "string"}}, "required": ["source", "question"]}}},
    {"type": "function", "function": {
        "name": "run_python", "description": "Run a Python script in agent/sandbox/ with a 10 s timeout (user must approve).",
        "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}}},
]


class Tee:
    """Mirror stdout to a transcript file."""

    def __init__(self, path):
        self.f = open(path, "w", encoding="utf-8")
        self.out = sys.stdout

    def write(self, s):
        self.out.write(s)
        self.f.write(s)

    def flush(self):
        self.out.flush()
        self.f.flush()


def short(s, n=160):
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[:n] + " ..."


class Agent:
    def __init__(self, channel="latent", top_k=6):
        from openai import OpenAI
        key = os.getenv("MODEL_API_KEY")
        if not key:
            sys.exit("MODEL_API_KEY missing: put it in .env (see .env.example).")
        self.client = OpenAI(base_url=MUSE_BASE, api_key=key)
        self.channel, self.top_k = channel, top_k
        self.docs, self.reader, self.n_script = {}, None, 0
        self.evidence = []  # (tool, args, result) for the verifier
        self.stats = {"muse_calls": 0, "muse_s": 0.0, "reader_calls": []}

    # ------------------------------------------------------------- Muse Spark
    def muse(self, messages, tools=None, effort="low"):
        t0 = time.perf_counter()
        kw = {"tools": tools} if tools else {}
        r = self.client.chat.completions.create(model=MUSE_MODEL, reasoning_effort=effort, messages=messages, **kw)
        dt = time.perf_counter() - t0
        self.stats["muse_calls"] += 1
        self.stats["muse_s"] += dt
        return r.choices[0].message, dt

    # ------------------------------------------------------------- tools
    def web_search(self, query, max_results=5):
        from ddgs import DDGS
        res = DDGS().text(query, max_results=int(max_results)) or []
        out = [{"title": x.get("title"), "url": x.get("href"), "snippet": short(x.get("body", ""), 200)} for x in res]
        return out, f"{len(out)} results"

    def fetch_url(self, url):
        import trafilatura
        raw = trafilatura.fetch_url(url)
        text = trafilatura.extract(raw, include_tables=False, include_comments=False) if raw else None
        if not text:
            return {"error": f"could not fetch or extract {url}"}, "failed"
        doc_id = f"doc{len(self.docs) + 1}"
        self.docs[doc_id] = {"url": url, "text": text}
        return ({"doc_id": doc_id, "url": url, "chars": len(text), "preview": short(text, 300)},
                f"{doc_id}, {len(text):,} chars")

    def read_and_answer(self, source, question):
        text = self.docs[source]["text"] if source in self.docs else source
        if self.reader is None:
            print("[reader] loading Llama 3.2 1B + bridge + Llama 3.2 3B (4-bit) ...", flush=True)
            sys.path.insert(0, str(HERE))
            from reader import Reader
            self.reader = Reader()
            print(f"[reader] loaded in {self.reader.load_s:.1f} s, warm-up {self.reader.warmup():.1f} s", flush=True)
        chunks, n_all = self.reader.select(text, question, top_k=self.top_k)
        out = self.reader.answer(text, question, channel=self.channel, chunks=chunks, max_new_tokens=64)
        size = (f"{out['msg_positions']} vectors ({out['msg_bytes'] / 1024:.0f} KB)" if self.channel == "latent"
                else f"{out['msg_positions']} tokens ({out['msg_bytes']} B)")
        print(f"[reader] {self.channel}, 1B {out['reader_s']:.2f} s, {len(chunks)} chunks (of {n_all}), "
              f"message {size}, 3B {out['solver_s']:.2f} s")
        self.stats["reader_calls"].append({k: out[k] for k in ("channel", "chunks", "msg_positions", "msg_bytes",
                                                                 "reader_s", "solver_s", "total_s")})
        return {"answer": out["answer"], "chunks_read": len(chunks)}, short(out["answer"], 100)

    def run_python(self, code):
        SANDBOX.mkdir(exist_ok=True)
        self.n_script += 1
        path = SANDBOX / f"script_{self.n_script}.py"
        print("-" * 60 + f"\n{code}\n" + "-" * 60)
        ans = input("Run? [y/N] ").strip().lower()
        if not sys.stdin.isatty():
            print(ans)  # echo piped input so it shows in the transcript
        elif isinstance(sys.stdout, Tee):
            sys.stdout.f.write(ans + "\n")
        if ans not in ("y", "yes"):
            return {"error": "user declined to run the code"}, "declined"
        path.write_text(code, encoding="utf-8")
        try:
            p = subprocess.run([sys.executable, "-I", path.name], cwd=SANDBOX, capture_output=True, text=True,
                               timeout=10, encoding="utf-8", errors="replace")
            res = {"exit_code": p.returncode, "stdout": p.stdout[-3000:], "stderr": p.stderr[-2000:]}
        except subprocess.TimeoutExpired:
            res = {"error": "timeout after 10 s"}
        if res.get("stdout"):
            print(res["stdout"].rstrip())
        if res.get("stderr"):
            print(res["stderr"].rstrip())
        return res, f"exit {res.get('exit_code', 'timeout')}"

    # ------------------------------------------------------------- loop
    def run(self, task, max_steps=MAX_STEPS):
        t_start = time.perf_counter()
        print(f"[task] {task}\n[config] brain {MUSE_MODEL}, reader channel = {self.channel}, max {max_steps} tool steps")
        messages = [{"role": "developer", "content": SYSTEM}, {"role": "user", "content": task}]
        steps, answer = 0, None
        while True:
            budget_left = steps < max_steps
            msg, dt = self.muse(messages, tools=TOOLS if budget_left else None)
            if msg.content and msg.content.strip():
                label = "plan" if msg.tool_calls else "answer"
                print(f"[Muse Spark] {label} ({dt:.1f} s): {short(msg.content, 300)}")
            if not msg.tool_calls:
                answer = (msg.content or "").strip()
                break
            messages.append({"role": "assistant", "content": msg.content or "",
                             "tool_calls": [{"id": tc.id, "type": "function",
                                             "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                                            for tc in msg.tool_calls]})
            for tc in msg.tool_calls:
                steps += 1
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                shown = {k: (short(v, 80) if k != "code" else f"<{len(v)} chars>") for k, v in args.items()}
                print(f"[tool {steps}/{max_steps}] {name} {json.dumps(shown, ensure_ascii=False)}", flush=True)
                t0 = time.perf_counter()
                if steps > max_steps:
                    result, note = {"error": "tool step limit reached; answer now"}, "skipped (limit)"
                else:
                    try:
                        result, note = getattr(self, name)(**args)
                    except Exception as e:  # bad args, network, ... -> report back to the brain
                        result, note = {"error": f"{type(e).__name__}: {e}"}, f"error: {short(e, 100)}"
                print(f"[tool {min(steps, max_steps)}/{max_steps}] {name} -> {note} ({time.perf_counter() - t0:.2f} s)")
                self.evidence.append({"tool": name, "args": args, "result": result})
                messages.append({"role": "tool", "tool_call_id": tc.id,
                                 "content": json.dumps(result, ensure_ascii=False)[:6000]})
        # ---- verify
        ev = json.dumps(self.evidence, ensure_ascii=False)[-12000:]
        vmsg, dt = self.muse([{"role": "developer", "content": VERIFY},
                              {"role": "user", "content": f"Task:\n{task}\n\nTool results:\n{ev}\n\nProposed answer:\n{answer}"}],
                             effort="low")
        try:
            v = json.loads(re.search(r"\{.*\}", vmsg.content or "", re.S).group(0))
        except (AttributeError, json.JSONDecodeError):
            v = {"verdict": "unparsed", "reason": short(vmsg.content, 200), "final_answer": answer}
        print(f"[Muse Spark] verify ({dt:.1f} s): {v.get('verdict')} - {v.get('reason')}")
        final = v.get("final_answer") or answer
        if v.get("verdict") == "reject":
            final = f"(unverified) {answer}"
        rc = self.stats["reader_calls"]
        print(f"\nANSWER: {final}")
        reader = (f"reader {self.channel} x{len(rc)}: {sum(r['reader_s'] for r in rc):.2f} s 1B + "
                  f"{sum(r['solver_s'] for r in rc):.2f} s 3B" if rc else "reader not used")
        print(f"[summary] {steps} tool steps, {self.stats['muse_calls']} Muse Spark calls ({self.stats['muse_s']:.1f} s), "
              f"{reader}, total {time.perf_counter() - t_start:.1f} s")
        return final


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("task", nargs="?")
    ap.add_argument("--channel", choices=("latent", "text"), default="latent")
    ap.add_argument("--max-steps", type=int, default=MAX_STEPS)
    ap.add_argument("--top-k", type=int, default=6, help="document chunks the reader reads per question")
    ap.add_argument("--transcript", help="also write everything printed to this file")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if args.transcript:
        sys.stdout = Tee(args.transcript)
    task = args.task or input("Task: ").strip()
    Agent(channel=args.channel, top_k=args.top_k).run(task, max_steps=min(args.max_steps, MAX_STEPS))


if __name__ == "__main__":
    main()

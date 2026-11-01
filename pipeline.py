"""Phase 4: multi-agent pipeline.

  Orchestrator : Meta Muse Spark (Meta Model API) splits the task into a reader subtask and a solver subtask.
                 Falls back to a rule-based splitter when MODEL_API_KEY is not set or the call fails.
  Reader       : Llama 3.2 1B (sender) reads the facts.
  Solver       : Llama 3.2 3B (receiver) answers the question.
  Channel      : text (reader writes a message) or latent (reader's hidden states -> bridge -> soft tokens).

Usage: python pipeline.py "Tom has 3 boxes with 12 pens each. He gives away 7 pens. How many pens are left?"
"""
import argparse
import hashlib
import json
import os
import re
import time

import torch

from bridge import LatentBridge
from common import CKPT, DATA, DEVICE, RECEIVER_ID, SENDER_ID, extract_number, load_model, split_question
from receiver_io import Receiver, Sender

MUSE_MODEL = os.getenv("MUSE_MODEL", "muse-spark-1.1")
MUSE_BASE = os.getenv("MUSE_BASE_URL", "https://api.meta.ai/v1")
CACHE = DATA / "orchestrator_cache.json"

ORCH_INSTR = """You orchestrate a team of two small local models that solve a word problem together.
- reader (Llama 3.2 1B): reads the facts and passes what it understood to the solver.
- solver (Llama 3.2 3B): receives the reader's message and answers the final question with one number.
Split the problem. Return ONLY JSON, no prose:
{"facts": "<every sentence that gives information, copied verbatim>",
 "question": "<the one sentence that asks what to find, copied verbatim>",
 "subtasks": [{"agent": "reader", "task": "<short instruction>"}, {"agent": "solver", "task": "<short instruction>"}]}"""


class Orchestrator:
    """Muse Spark planner with an on-disk cache (each distinct task costs at most one API call)."""

    def __init__(self, max_calls=50):
        self.key = os.getenv("MODEL_API_KEY")
        self.max_calls, self.calls = max_calls, 0
        self.cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
        self.client = None
        if self.key:
            from openai import OpenAI
            self.client = OpenAI(base_url=MUSE_BASE, api_key=self.key)

    def plan(self, task):
        h = hashlib.sha256((MUSE_MODEL + task).encode()).hexdigest()[:16]
        if h in self.cache:
            return {**self.cache[h], "cached": True}
        plan, err = None, None
        if self.client and self.calls < self.max_calls:
            try:
                self.calls += 1
                t0 = time.perf_counter()
                resp = self.client.chat.completions.create(
                    model=MUSE_MODEL, reasoning_effort="minimal",
                    messages=[{"role": "developer", "content": ORCH_INSTR}, {"role": "user", "content": task}])
                text = resp.choices[0].message.content
                plan = json.loads(re.search(r"\{.*\}", text, re.S).group(0))
                assert plan.get("facts") and plan.get("question")
                plan.update(orchestrator=MUSE_MODEL, orchestrator_s=round(time.perf_counter() - t0, 2),
                            usage=resp.usage.model_dump() if resp.usage else None)
                self.cache[h] = plan
                CACHE.write_text(json.dumps(self.cache, indent=1), encoding="utf-8")
            except Exception as e:  # network, auth, bad JSON -> fall back
                err, plan = f"{type(e).__name__}: {e}", None
        if plan is None:
            facts, q = split_question(task)
            if facts is None:
                facts, q = "", task
            plan = {"facts": facts, "question": q, "orchestrator": "rule-based fallback",
                    "subtasks": [{"agent": "reader", "task": "extract the key quantities"},
                                 {"agent": "solver", "task": "answer the final question"}]}
            if err:
                plan["error"] = err
        return {**plan, "cached": False}

    def verify(self, task, solution):
        """Muse Spark checks the solver's worked answer against the full problem.
        Returns {"verdict": "accept"|"reject"|"unavailable", "reason": str, "cached": bool}."""
        h = "v-" + hashlib.sha256((MUSE_MODEL + task + "\n##\n" + solution).encode()).hexdigest()[:16]
        if h in self.cache:
            return {**self.cache[h], "cached": True}
        if not self.client or self.calls >= self.max_calls:
            return {"verdict": "unavailable", "reason": "no API key or call cap reached", "cached": False}
        try:
            self.calls += 1
            t0 = time.perf_counter()
            resp = self.client.chat.completions.create(
                model=MUSE_MODEL, reasoning_effort="low",
                messages=[{"role": "developer", "content": VERIFY_INSTR},
                          {"role": "user", "content": f"Problem:\n{task}\n\nProposed solution:\n{solution}"}])
            v = json.loads(re.search(r"\{.*\}", resp.choices[0].message.content, re.S).group(0))
            assert v.get("verdict") in ("accept", "reject")
            v.update(verify_s=round(time.perf_counter() - t0, 2),
                     usage=resp.usage.model_dump() if resp.usage else None)
            self.cache[h] = v
            CACHE.write_text(json.dumps(self.cache, indent=1), encoding="utf-8")
            return {**v, "cached": False}
        except Exception as e:
            return {"verdict": "unavailable", "reason": f"{type(e).__name__}: {e}", "cached": False}


VERIFY_INSTR = """You check answers produced by a small model. Given a word problem and a proposed worked
solution ending in '#### <number>', decide whether the final number correctly answers the problem.
Return ONLY JSON: {"verdict": "accept" or "reject", "reason": "<one short sentence>"}"""


class Pipeline:
    def __init__(self, ckpt=CKPT / "bridge_main_token.pt", sender=None, receiver=None):
        self.bridge, self.meta = LatentBridge.load(ckpt, DEVICE)
        self.sender = sender or Sender(*load_model(SENDER_ID))
        self.receiver = receiver or Receiver(*load_model(RECEIVER_ID))
        self.orch = Orchestrator()

    def _t(self):
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        return time.perf_counter()

    def run(self, task, modes=("text", "latent"), verify=True):
        plan = self.orch.plan(task)
        facts, q = plan["facts"], plan["question"]
        out = {"plan": plan}
        if "text" in modes:
            t0 = self._t()
            msg = self.sender.message(facts)
            t1 = self._t()
            ans = self.receiver.generate(self.receiver.text_inputs(msg, q)[0])
            t2 = self._t()
            out["text"] = {"message": msg, "tokens_sent": len(self.sender.tok(msg, add_special_tokens=False).input_ids),
                           "bytes_sent": len(msg.encode()), "answer_raw": ans, "answer": extract_number(ans),
                           "reader_s": t1 - t0, "solver_s": t2 - t1, "total_s": t2 - t0}
        if "latent" in modes:
            t0 = self._t()
            h = self.sender.hidden(facts, self.meta["layer"], skip_template=self.meta.get("skip_template", False),
                                   content_only=self.meta.get("content_only", False))
            with torch.no_grad():
                lat = self.bridge(h)
            t1 = self._t()
            ans = self.receiver.generate(self.receiver.latent_inputs(lat, q)[0])
            t2 = self._t()
            out["latent"] = {"latents": lat[0].float().cpu(), "tokens_sent": lat.shape[1],
                             "bytes_sent": lat.shape[1] * lat.shape[2] * 2, "answer_raw": ans,
                             "answer": extract_number(ans), "reader_s": t1 - t0, "solver_s": t2 - t1,
                             "total_s": t2 - t0}
            if verify:
                v = self.orch.verify(task, ans)
                out["latent"]["verify"] = v
                if v["verdict"] == "reject":  # retry once over the text channel
                    if "text" not in out:
                        out.update(self.run(task, modes=("text",), verify=False))
                    out["latent"]["final_answer"] = out["text"]["answer"]
                    out["latent"]["fallback"] = "text"
                else:
                    out["latent"]["final_answer"] = out["latent"]["answer"]
                    out["latent"]["fallback"] = None
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", nargs="?", default=(
        "A bakery sells muffins for $3 each and cookies for $2 each. On Monday it sold 40 muffins and "
        "25 cookies. On Tuesday it sold twice as many muffins and the same number of cookies. "
        "How much money did the bakery make on Tuesday?"))
    ap.add_argument("--ckpt", default=str(CKPT / "bridge_main_token.pt"))
    ap.add_argument("--plan_only", action="store_true", help="only call the orchestrator (no GPU needed)")
    args = ap.parse_args()
    if args.plan_only:
        print(json.dumps(Orchestrator().plan(args.task), indent=1))
        return
    res = Pipeline(args.ckpt).run(args.task)
    res["latent"].pop("latents", None)
    print(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()

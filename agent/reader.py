"""Reader tool for the agent: Llama 3.2 1B reads text, the latent bridge carries it to Llama 3.2 3B,
which answers a question about it. Reuses the main project's loaders; never writes to checkpoints/.

Channels (same prompt tail for all, so only the message differs):
  latent : 1B forward pass per chunk -> bridge (stride 1) -> soft tokens, concatenated
  text   : 1B writes bullet notes per chunk, concatenated
  raw    : 3B reads the original text directly (reference)
  none   : 3B gets only the question (prior-knowledge control)
"""
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from bridge import LatentBridge  # noqa: E402
from common import (ASSIST_HDR, CKPT, DEVICE, RECEIVER_ID, RECV_PRE, RECV_PRE_NONE, SENDER_ID,  # noqa: E402
                    load_model)
from receiver_io import Receiver, Sender  # noqa: E402

DEFAULT_CKPT = CKPT / "bridge_main_token.pt"  # official stride-1 bridge, loaded read-only
CHUNK_TOKENS = 48  # the bridge was trained on GSM8K contexts of ~40 tokens
QA_FMT = "Answer in a few words."
_SENT = re.compile(r"(?<=[.!?])\s+")
_WORD = re.compile(r"[a-z0-9]+")
_STOP = set("the a an of in on at to for and or is are was were be by with from as that this which who what "
            "when where how why did does do its it their his her than then into about".split())


def qa_post(question):
    return f"\n\nUsing the info above: {question}\n{QA_FMT}" + ASSIST_HDR


class Reader:
    def __init__(self, ckpt=DEFAULT_CKPT):
        t0 = time.perf_counter()
        self.bridge, self.meta = LatentBridge.load(ckpt, DEVICE)
        self.sender = Sender(*load_model(SENDER_ID))
        self.receiver = Receiver(*load_model(RECEIVER_ID))
        self.tok = self.sender.tok
        self.load_s = time.perf_counter() - t0

    def warmup(self):
        """One tiny pass through every path so CUDA kernel setup is not billed to the first real call."""
        t0 = time.perf_counter()
        for ch in ("latent", "text"):
            self.answer("Warm-up text. It has two sentences.", "What is this?", channel=ch, max_new_tokens=4)
        return time.perf_counter() - t0

    def _sync(self):
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        return time.perf_counter()

    def ntok(self, text):
        return len(self.tok(text, add_special_tokens=False).input_ids)

    def chunk(self, text, max_tokens=CHUNK_TOKENS):
        """Greedy sentence packing into pieces of <= max_tokens; over-long sentences split by tokens."""
        pieces = []
        for s in (s.strip() for s in _SENT.split(" ".join(text.split()))):
            if not s:
                continue
            ids = self.tok(s, add_special_tokens=False).input_ids
            if len(ids) <= max_tokens:
                pieces.append(s)
            else:
                pieces += [self.tok.decode(ids[i:i + max_tokens]).strip() for i in range(0, len(ids), max_tokens)]
        chunks, cur = [], ""
        for p in pieces:
            cand = f"{cur} {p}".strip()
            if cur and self.ntok(cand) > max_tokens:
                chunks.append(cur)
                cur = p
            else:
                cur = cand
        if cur:
            chunks.append(cur)
        return chunks

    def select(self, text, question, top_k=6):
        """Chunk a long document and keep the top_k chunks sharing the most words with the question
        (kept in document order). Same selection for every channel, so comparisons stay fair."""
        chunks = self.chunk(text)
        if len(chunks) <= top_k:
            return chunks, len(chunks)
        q = {w for w in _WORD.findall(question.lower()) if w not in _STOP}
        scored = []
        for i, c in enumerate(chunks):
            words = set(_WORD.findall(c.lower()))
            scored.append((len(q & words) + 0.5 * any(ch.isdigit() for ch in c), -i))
        keep = sorted(-i for _, i in sorted(scored, reverse=True)[:top_k])
        return [chunks[i] for i in keep], len(chunks)

    def _gen(self, x, max_new_tokens):
        return self.receiver.generate(x, max_new_tokens=max_new_tokens)

    @torch.no_grad()
    def answer(self, text, question, channel="latent", max_new_tokens=48, max_chunks=24, chunks=None):
        """Returns dict: answer, channel, chunks, reader_s, solver_s, total_s, msg_positions, msg_bytes.
        Pass `chunks` to skip chunking (e.g. after select())."""
        r = self.receiver
        tail = r.embed(r.ids(qa_post(question)))
        if channel in ("latent", "text"):
            chunks = (chunks or self.chunk(text))[:max_chunks]
        else:
            text = " ".join(chunks) if chunks else text
            chunks = []
        t0 = self._sync()
        info = {"channel": channel, "chunks": len(chunks)}
        if channel == "latent":
            lats = []
            for c in chunks:
                h = self.sender.hidden(c, self.meta["layer"], skip_template=self.meta.get("skip_template", False),
                                       content_only=self.meta.get("content_only", False))
                lats.append(self.bridge(h).to(r.dtype))
            lat = torch.cat(lats, dim=1)
            x = torch.cat([r.embed(r.ids(RECV_PRE)), lat, tail], dim=1)
            info.update(msg_positions=lat.shape[1], msg_bytes=lat.shape[1] * lat.shape[2] * 2)
        elif channel == "text":
            msg = "\n".join(self.sender.message(c) for c in chunks)
            ids = r.ids(RECV_PRE) + self.tok(msg, add_special_tokens=False).input_ids
            x = torch.cat([r.embed(ids), tail], dim=1)
            info.update(message=msg, msg_positions=self.ntok(msg), msg_bytes=len(msg.encode()))
        elif channel == "raw":
            ids = r.ids(RECV_PRE) + self.tok(text, add_special_tokens=False).input_ids
            x = torch.cat([r.embed(ids), tail], dim=1)
            info.update(msg_positions=self.ntok(text), msg_bytes=len(text.encode()))
        elif channel == "none":
            x = r.embed(r.ids(RECV_PRE_NONE) + r.ids(f"{question}\n{QA_FMT}" + ASSIST_HDR))
            info.update(msg_positions=0, msg_bytes=0)
        else:
            raise ValueError(channel)
        t1 = self._sync()
        ans = self._gen(x, max_new_tokens)
        t2 = self._sync()
        info.update(answer=ans, reader_s=t1 - t0, solver_s=t2 - t1, total_s=t2 - t0)
        return info

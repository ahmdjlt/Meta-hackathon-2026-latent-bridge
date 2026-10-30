"""Shared config, model loading and prompt building for LatentBridge."""
import os
import re
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
CKPT = ROOT / "checkpoints"
RESULTS = ROOT / "results"
for d in (DATA, CKPT, RESULTS):
    d.mkdir(exist_ok=True)

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

# Official gated Meta repos: needs `hf auth login` or HF_TOKEN in .env.
SENDER_ID = os.getenv("SENDER_ID", "meta-llama/Llama-3.2-1B-Instruct")
RECEIVER_ID = os.getenv("RECEIVER_ID", "meta-llama/Llama-3.2-3B-Instruct")

# Sender layers we cache hidden states for: 8 = middle, 16 = last (1B has 16 layers).
SENDER_LAYERS = (8, 16)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def compute_dtype():
    if DEVICE == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16 if DEVICE == "cuda" else torch.float32


def load_model(model_id, four_bit=True):
    """Load a frozen causal LM (4-bit NF4 on GPU) plus its tokenizer."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = "<|finetune_right_pad_id|>"
    kwargs = {"dtype": compute_dtype()}
    if DEVICE == "cuda":
        kwargs["device_map"] = {"": 0}
        if four_bit:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype(),
                bnb_4bit_use_double_quant=True,
            )
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tok


@torch.no_grad()
def mean_embed_norm(weight, chunk=8192):
    """Mean L2 norm of embedding rows, chunked so we never copy the whole table to fp32."""
    return torch.cat([weight[i:i + chunk].float().norm(dim=-1) for i in range(0, weight.shape[0], chunk)]).mean().item()


def vram_mb():
    if DEVICE != "cuda":
        return 0.0
    return torch.cuda.memory_allocated() / 2**20


# ---------------------------------------------------------------- prompts
USER_HDR = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
ASSIST_HDR = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

SENDER_INSTR = (
    "List the key quantities in the text below and compute any partial results "
    "that follow directly from them. Be brief, use short lines, do not ask questions.\n\nText: "
)


def sender_prompt(context):
    return USER_HDR + SENDER_INSTR + context + ASSIST_HDR


# Receiver prompt = PRE + <message: text, latents, or nothing> + POST(final_q) + answer
RECV_PRE = USER_HDR + "Notes from a teammate:\n"
RECV_PRE_NONE = USER_HDR


def recv_post(final_q):
    return f"\n\nUsing the info above: {final_q} Answer:" + ASSIST_HDR


def recv_post_none(final_q):
    return f"{final_q} Answer:" + ASSIST_HDR


def answer_text(ans):
    return f"{ans}<|eot_id|>"


# ---------------------------------------------------------------- GSM8K helpers
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z$0-9\"'(])")


def split_question(q):
    """context = every sentence but the last, final_q = last sentence."""
    sents = [s.strip() for s in _SENT.split(q.strip()) if s.strip()]
    if len(sents) < 2:
        return None, None
    return " ".join(sents[:-1]), sents[-1]


def gold_number(answer_field):
    return normalize_num(answer_field.split("####")[-1])


def normalize_num(s):
    s = s.replace(",", "").replace("$", "").strip()
    try:
        f = float(s)
    except ValueError:
        return s
    return str(int(f)) if f == int(f) else str(f)


_NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def extract_number(text):
    m = _NUM.search(text.replace("$", ""))
    return normalize_num(m.group(0).rstrip(".")) if m else ""

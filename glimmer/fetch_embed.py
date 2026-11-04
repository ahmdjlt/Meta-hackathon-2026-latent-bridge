"""Download ONLY Muse Glimmer 30B's input-embedding table (2.7 GB of a 60 GB checkpoint) with an HTTP
range request into the safetensors file. Needed to fit the cross-tokenizer ridge map on the laptop.

python glimmer/fetch_embed.py   ->  data/glimmer_embed.pt  (bf16, 202048 x 6656)
"""
import json
import struct
import sys
from pathlib import Path

import requests
import torch
from huggingface_hub import hf_hub_url
from huggingface_hub.utils import build_hf_headers

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import DATA  # noqa: E402

REPO = "meta-models/Muse-Glimmer-30B"
FILE = "model-00001-of-00002.safetensors"
KEY = "model.language_model.embed_tokens.weight"
OUT = DATA / "glimmer_embed.pt"


def main():
    url = hf_hub_url(REPO, FILE)
    hdr = build_hf_headers()
    n = struct.unpack("<Q", requests.get(url, headers={**hdr, "Range": "bytes=0-7"}).content)[0]
    meta = json.loads(requests.get(url, headers={**hdr, "Range": f"bytes=8-{7 + n}"}).content)
    info = meta[KEY]
    a, b = info["data_offsets"]
    start, end = 8 + n + a, 8 + n + b - 1
    print(f"{KEY}: {info['dtype']} {info['shape']}, {(b - a) / 2**30:.2f} GB", flush=True)
    assert info["dtype"] == "BF16"
    tmp = OUT.with_suffix(".bin")
    have = tmp.stat().st_size if tmp.exists() else 0
    with requests.get(url, headers={**hdr, "Range": f"bytes={start + have}-{end}"}, stream=True, timeout=60) as r, \
            open(tmp, "ab") as f:
        r.raise_for_status()
        done = have
        for chunk in r.iter_content(1 << 22):
            f.write(chunk)
            done += len(chunk)
            if done % (256 << 20) < (1 << 22):
                print(f"  {done / 2**30:.2f} / {(b - a) / 2**30:.2f} GB", flush=True)
    raw = bytearray(tmp.read_bytes())
    assert len(raw) == b - a, f"size mismatch {len(raw)} vs {b - a}"
    E = torch.frombuffer(raw, dtype=torch.bfloat16).view(*info["shape"]).clone()
    torch.save(E, OUT)
    tmp.unlink()
    print(f"saved {OUT} {tuple(E.shape)} mean row norm {E[:5000].float().norm(dim=-1).mean():.4f}")


if __name__ == "__main__":
    main()

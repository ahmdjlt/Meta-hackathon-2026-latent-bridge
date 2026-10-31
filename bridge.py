"""Phase 2: the latent bridge (Perceiver-style resampler).

Maps a variable-length sequence of sender hidden states (T x d_sender) to K soft tokens in the
receiver's input-embedding space (K x d_receiver). Both LLMs stay frozen; only this module trains.
Internally it works at width d (default 1024) so params + AdamW state fit next to a 4-bit 3B model in 4 GB.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.ln_q = nn.LayerNorm(d)
        self.ln_kv = nn.LayerNorm(d)
        self.xattn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ln_s = nn.LayerNorm(d)
        self.sattn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ln_ff = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))

    def forward(self, q, kv, pad_mask):
        kvn = self.ln_kv(kv)
        q = q + self.xattn(self.ln_q(q), kvn, kvn, key_padding_mask=pad_mask, need_weights=False)[0]
        s = self.ln_s(q)
        q = q + self.sattn(s, s, s, need_weights=False)[0]
        return q + self.ff(self.ln_ff(q))


class LatentBridge(nn.Module):
    def __init__(self, d_in, d_out, k=16, d=1024, heads=8, depth=2, target_norm=1.0, pos=False):
        super().__init__()
        self.config = dict(d_in=d_in, d_out=d_out, k=k, d=d, heads=heads, depth=depth, target_norm=target_norm, pos=pos)
        self.pos = pos
        # Per-dimension standardization of sender states (set from training data via set_input_stats).
        # LLM hidden states share a large common direction; without this every input looks alike.
        self.register_buffer("in_mean", torch.zeros(d_in))
        self.register_buffer("in_std", torch.ones(d_in))
        self.pos_scale = nn.Parameter(torch.tensor(0.1))
        self.in_norm = nn.LayerNorm(d_in)
        self.in_proj = nn.Linear(d_in, d)
        self.queries = nn.Parameter(torch.randn(k, d) * 0.02)
        self.blocks = nn.ModuleList(Block(d, heads) for _ in range(depth))
        self.out_norm = nn.LayerNorm(d)
        self.out_proj = nn.Linear(d, d_out)
        self.register_buffer("target_norm", torch.tensor(float(target_norm)))

    def forward(self, h, mask=None):
        """h: (B, T, d_in); mask: (B, T) bool, True = real token. Returns (B, K, d_out) fp32."""
        h = (h.float() - self.in_mean) / self.in_std
        kv = self.in_proj(self.in_norm(h))
        if self.pos:  # sinusoidal source positions (learned strength), so the bridge knows token order
            kv = kv + self.pos_scale * self.sinusoid(h.shape[1], kv.shape[-1], kv.device)
        pad = None if mask is None else ~mask
        q = self.queries.unsqueeze(0).expand(h.shape[0], -1, -1)
        for blk in self.blocks:
            q = blk(q, kv, pad)
        out = self.out_proj(self.out_norm(q))
        # L2-normalize each soft token, then scale to the receiver's mean token-embedding norm.
        return F.normalize(out, dim=-1) * self.target_norm

    @torch.no_grad()
    def set_input_stats(self, states):
        """states: list of (T_i, d_in) tensors from the training set."""
        x = torch.cat([t.float() for t in states])
        self.in_mean.copy_(x.mean(0))
        self.in_std.copy_(x.std(0).clamp_min(1e-3))

    @staticmethod
    def sinusoid(n, d, device):
        pos = torch.arange(n, device=device, dtype=torch.float32)[:, None]
        div = torch.exp(torch.arange(0, d, 2, device=device, dtype=torch.float32) * (-math.log(10000.0) / d))
        pe = torch.zeros(n, d, device=device)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe

    def save(self, path, **meta):
        torch.save({"config": self.config, "state_dict": self.state_dict(), "meta": meta}, path)

    @classmethod
    def load(cls, path, device="cpu"):
        ck = torch.load(path, map_location=device, weights_only=False)
        cfg = dict(ck["config"])
        kind = cfg.pop("kind", "perceiver")
        m = (TokenBridge if kind == "token" else LatentBridge)(**cfg)
        m.load_state_dict(ck["state_dict"], strict=False)  # older checkpoints lack the input stats
        return m.to(device).eval(), ck.get("meta", {})


class TokenBridge(LatentBridge):
    """Token-aligned bridge: one soft token per sender position (or per `stride` positions).

    standardize -> per-token MLP -> one self-attention layer (mixes context) -> mean-pool groups of
    `stride` positions -> project to receiver width -> L2-normalize * mean embedding norm.
    Sender and receiver share the Llama 3 tokenizer, so during training each soft token can be
    anchored to the receiver's embedding of the same token (alignment loss in train.py).
    """

    def __init__(self, d_in, d_out, d=1024, heads=8, stride=1, target_norm=1.0, **_):
        nn.Module.__init__(self)
        self.config = dict(kind="token", d_in=d_in, d_out=d_out, d=d, heads=heads, stride=stride,
                           target_norm=target_norm, k=0)
        self.stride = stride
        self.register_buffer("in_mean", torch.zeros(d_in))
        self.register_buffer("in_std", torch.ones(d_in))
        self.register_buffer("target_norm", torch.tensor(float(target_norm)))
        self.in_norm = nn.LayerNorm(d_in)
        self.mlp = nn.Sequential(nn.Linear(d_in, d), nn.GELU(), nn.Linear(d, d))
        self.ln_s = nn.LayerNorm(d)
        self.sattn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.out_norm = nn.LayerNorm(d)
        self.out_proj = nn.Linear(d, d_out)

    def forward(self, h, mask=None):
        h = (h.float() - self.in_mean) / self.in_std
        x = self.mlp(self.in_norm(h))
        s = self.ln_s(x)
        T = x.shape[1]
        causal = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), 1)
        x = x + self.sattn(s, s, s, attn_mask=causal, need_weights=False)[0]
        if self.stride > 1:  # compress: average each group of `stride` neighbouring positions
            pad = (-T) % self.stride
            if pad:
                x = torch.cat([x, x[:, -1:].expand(-1, pad, -1)], dim=1)
            x = x.view(x.shape[0], -1, self.stride, x.shape[-1]).mean(2)
        out = self.out_proj(self.out_norm(x))
        return F.normalize(out, dim=-1) * self.target_norm


if __name__ == "__main__":
    for b in (LatentBridge(2048, 3072, k=16), TokenBridge(2048, 3072, stride=2)):
        n = sum(p.numel() for p in b.parameters())
        y = b(torch.randn(2, 37, 2048), torch.ones(2, 37, dtype=torch.bool))
        print(f"{type(b).__name__}: params {n / 1e6:.1f}M | out {tuple(y.shape)} | norms {y.norm(dim=-1).mean():.3f}")

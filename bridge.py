"""Phase 2: the latent bridge (Perceiver-style resampler).

Maps a variable-length sequence of sender hidden states (T x d_sender) to K soft tokens in the
receiver's input-embedding space (K x d_receiver). Both LLMs stay frozen; only this module trains.
Internally it works at width d (default 1024) so params + AdamW state fit next to a 4-bit 3B model in 4 GB.
"""
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
    def __init__(self, d_in, d_out, k=16, d=1024, heads=8, depth=2, target_norm=1.0):
        super().__init__()
        self.config = dict(d_in=d_in, d_out=d_out, k=k, d=d, heads=heads, depth=depth, target_norm=target_norm)
        self.in_norm = nn.LayerNorm(d_in)
        self.in_proj = nn.Linear(d_in, d)
        self.queries = nn.Parameter(torch.randn(k, d) * 0.02)
        self.blocks = nn.ModuleList(Block(d, heads) for _ in range(depth))
        self.out_norm = nn.LayerNorm(d)
        self.out_proj = nn.Linear(d, d_out)
        self.register_buffer("target_norm", torch.tensor(float(target_norm)))

    def forward(self, h, mask=None):
        """h: (B, T, d_in); mask: (B, T) bool, True = real token. Returns (B, K, d_out) fp32."""
        kv = self.in_proj(self.in_norm(h.float()))
        pad = None if mask is None else ~mask
        q = self.queries.unsqueeze(0).expand(h.shape[0], -1, -1)
        for blk in self.blocks:
            q = blk(q, kv, pad)
        out = self.out_proj(self.out_norm(q))
        # L2-normalize each soft token, then scale to the receiver's mean token-embedding norm.
        return F.normalize(out, dim=-1) * self.target_norm

    def save(self, path, **meta):
        torch.save({"config": self.config, "state_dict": self.state_dict(), "meta": meta}, path)

    @classmethod
    def load(cls, path, device="cpu"):
        ck = torch.load(path, map_location=device, weights_only=False)
        m = cls(**ck["config"])
        m.load_state_dict(ck["state_dict"])
        return m.to(device).eval(), ck.get("meta", {})


if __name__ == "__main__":
    b = LatentBridge(2048, 3072, k=16)
    n = sum(p.numel() for p in b.parameters())
    y = b(torch.randn(2, 37, 2048), torch.ones(2, 37, dtype=torch.bool))
    print(f"params {n / 1e6:.1f}M | out {tuple(y.shape)} | norms {y.norm(dim=-1).mean():.3f}")

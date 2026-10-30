"""Phase 0: check the GPU, load both models, run a random-latent sanity test."""
import time

import torch

from common import DEVICE, RECEIVER_ID, SENDER_ID, load_model, mean_embed_norm, vram_mb


def main():
    print(f"torch {torch.__version__} | CUDA available: {torch.cuda.is_available()}")
    if DEVICE == "cuda":
        p = torch.cuda.get_device_properties(0)
        print(f"GPU: {p.name} | VRAM: {p.total_memory / 2**30:.2f} GB | bf16: {torch.cuda.is_bf16_supported()}")

    t = time.time()
    sender, stok = load_model(SENDER_ID)
    print(f"sender  {SENDER_ID}: loaded in {time.time() - t:.1f}s, VRAM {vram_mb():.0f} MB")
    t = time.time()
    receiver, rtok = load_model(RECEIVER_ID)
    print(f"receiver {RECEIVER_ID}: loaded in {time.time() - t:.1f}s, VRAM {vram_mb():.0f} MB (both)")

    if DEVICE == "cuda":
        print(f"peak VRAM during loading: {torch.cuda.max_memory_allocated() / 2**20:.0f} MB")
        torch.cuda.reset_peak_memory_stats()

    # Sanity: receiver reads random vectors (scaled like real token embeddings) via inputs_embeds.
    emb = receiver.get_input_embeddings()
    mean_norm = mean_embed_norm(emb.weight)
    prefix = rtok("<|begin_of_text|>Here is a message: ", return_tensors="pt", add_special_tokens=False).input_ids.to(DEVICE)
    suffix = rtok(" What does it say?", return_tensors="pt", add_special_tokens=False).input_ids.to(DEVICE)
    rand = torch.randn(1, 8, emb.weight.shape[1], device=DEVICE)
    rand = torch.nn.functional.normalize(rand, dim=-1) * mean_norm
    x = torch.cat([emb(prefix), rand.to(emb.weight.dtype), emb(suffix)], dim=1)
    with torch.no_grad():
        out = receiver.generate(inputs_embeds=x, attention_mask=torch.ones(x.shape[:2], device=DEVICE, dtype=torch.long),
                                max_new_tokens=12, do_sample=False, pad_token_id=rtok.pad_token_id)
    print(f"receiver mean token-embedding norm: {mean_norm:.4f}")
    print("generate() from random inputs_embeds OK ->", repr(rtok.decode(out[0], skip_special_tokens=True)))

    # Sender hidden states shape check.
    ids = stok("<|begin_of_text|>Tom has 3 apples.", return_tensors="pt", add_special_tokens=False).input_ids.to(DEVICE)
    with torch.no_grad():
        hs = sender(ids, output_hidden_states=True).hidden_states
    print(f"sender hidden states: {len(hs)} tensors of shape {tuple(hs[-1].shape)}")
    if DEVICE == "cuda":
        print(f"peak VRAM after loading (inference): {torch.cuda.max_memory_allocated() / 2**20:.0f} MB")


if __name__ == "__main__":
    main()

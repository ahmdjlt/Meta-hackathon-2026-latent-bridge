"""Build receiver inputs for the three message types: text, latent (soft tokens), none."""
import torch

from common import (DEVICE, RECON_POST, content_span, ctx_start, mean_embed_norm, RECV_PRE, RECV_PRE_NONE, answer_text, recv_post, recv_post_none,
                    sender_prompt)


class Receiver:
    def __init__(self, model, tok):
        self.model, self.tok = model, tok
        self.emb = model.get_input_embeddings()
        self.dtype = self.emb.weight.dtype
        self._cache = {}

    def ids(self, text):
        if text not in self._cache:
            self._cache[text] = self.tok(text, add_special_tokens=False).input_ids
        return self._cache[text]

    def embed(self, ids):
        return self.emb(torch.tensor([ids], device=DEVICE))

    def mean_embed_norm(self):
        return mean_embed_norm(self.emb.weight)

    # Each builder returns (inputs_embeds [1, L, d], n_prefix) where n_prefix = tokens before the
    # shared POST+answer tail. The tail (and its logits) line up across text and latent inputs.
    def text_inputs(self, msg, final_q, ans=None):
        tail = self.ids(recv_post(final_q)) + (self.ids(answer_text(ans)) if ans is not None else [])
        head = self.ids(RECV_PRE) + self.tok(msg, add_special_tokens=False).input_ids
        return self.embed(head + tail), len(head)

    def latent_inputs(self, latents, final_q, ans=None):
        tail = self.ids(recv_post(final_q)) + (self.ids(answer_text(ans)) if ans is not None else [])
        pre = self.embed(self.ids(RECV_PRE))
        x = torch.cat([pre, latents.to(self.dtype), self.embed(tail)], dim=1)
        return x, pre.shape[1] + latents.shape[1]

    def recon_inputs(self, latents, context):
        """Auxiliary task: receiver must repeat the original context from the latents alone."""
        pre = self.embed(self.ids(RECV_PRE))
        tail = self.ids(RECON_POST) + self.tok(context + "<|eot_id|>", add_special_tokens=False).input_ids
        x = torch.cat([pre, latents.to(self.dtype), self.embed(tail)], dim=1)
        return x, pre.shape[1] + latents.shape[1] + len(self.ids(RECON_POST))

    def none_inputs(self, final_q):
        ids = self.ids(RECV_PRE_NONE) + self.ids(recv_post_none(final_q))
        return self.embed(ids), 0

    def context_inputs(self, context, final_q):
        """Upper-bound reference: receiver reads the raw original context."""
        return self.text_inputs(context, final_q)

    @torch.no_grad()
    def generate(self, x, max_new_tokens=256):
        out = self.model.generate(inputs_embeds=x, attention_mask=torch.ones(x.shape[:2], dtype=torch.long, device=DEVICE),
                                  max_new_tokens=max_new_tokens, do_sample=False, temperature=None, top_p=None,
                                  pad_token_id=self.tok.pad_token_id)
        return self.tok.decode(out[0], skip_special_tokens=True).strip()


class Sender:
    def __init__(self, model, tok):
        self.model, self.tok = model, tok

    @torch.no_grad()
    def hidden(self, context, layer, msg=None, skip_template=False, content_only=False):
        """Sender hidden states at `layer` over its prompt (+ message if given), BOS dropped: [1, T, d].
        skip_template drops the fixed instruction tokens before the context (same for every input);
        content_only keeps just the context text (no instruction, no assistant header)."""
        ids = self.tok(sender_prompt(context), add_special_tokens=False).input_ids
        n_prompt = len(ids) - 1
        if msg:
            ids = ids + self.tok(msg, add_special_tokens=False).input_ids
        hs = self.model(torch.tensor([ids], device=DEVICE), output_hidden_states=True).hidden_states
        h = hs[layer][:, 1:]
        if content_only:
            s, e = content_span(self.tok, n_prompt)
            return h[:, s:e]
        return h[:, ctx_start(self.tok):] if skip_template else h

    @torch.no_grad()
    def message(self, context, max_new_tokens=96):
        enc = self.tok(sender_prompt(context), return_tensors="pt", add_special_tokens=False).to(DEVICE)
        out = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, temperature=None,
                                  top_p=None, pad_token_id=self.tok.pad_token_id)
        return self.tok.decode(out[0, enc.input_ids.shape[1]:], skip_special_tokens=True).strip()

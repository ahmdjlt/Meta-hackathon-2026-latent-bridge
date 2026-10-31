# LatentBridge

**Small Llama agents that talk in vectors instead of text.**
Built for the Meta Global AI Developer Hackathon (track: *Push the Frontier*).

Multi-agent LLM systems pass messages as text: one model writes a message token by token, the next model
reads it back. LatentBridge replaces that text channel with a learned **latent channel**. A sender model
reads its input once, and a small trained *bridge* maps the sender's hidden states into the receiver's
input-embedding space as **K soft tokens**. Both LLMs stay frozen; only the bridge (30M params) trains.

```
            Muse Spark (Meta Model API)            plans the task, routes subtasks
                    │
        facts ──────┴────── question
          │                    │
 Llama 3.2 1B (reader)         │
   one forward pass            │
          │ hidden states      │
   ┌──────▼──────┐             │
   │   Bridge    │ K=16 soft tokens
   └──────┬──────┘             │
          └──────► Llama 3.2 3B (solver) ──► answer
```

Text pipeline for comparison: the 1B *writes* a ~60-token message, the 3B reads it.

## Meta models used

| Model | Role |
|---|---|
| **Muse Spark** (`muse-spark-1.1`, Meta Model API) | Orchestrator: splits each task into reader/solver subtasks |
| **Llama 3.2 1B Instruct** (`meta-llama/Llama-3.2-1B-Instruct`) | Reader / sender, 4-bit NF4 |
| **Llama 3.2 3B Instruct** (`meta-llama/Llama-3.2-3B-Instruct`) | Solver / receiver, 4-bit NF4 |

Everything runs on a laptop **RTX 3050 with 4 GB VRAM** (both models + bridge: ~3.2 GB; training peak ~3.4 GB).

## How it works

**Task.** GSM8K word problems. Each question is split into *context* (all sentences but the last) and
*final question* (last sentence). The reader sees only the context; the solver sees only the final question
plus the reader's message. So the solver can only succeed if the message carries the facts.

**Bridge** (`bridge.py`). Perceiver-style resampler: LayerNorm + Linear on the 1B's hidden states
(last layer, 2048-d), K learned queries, 2 blocks of cross-attention + self-attention + MLP at width 1024,
projection to 3072-d, each soft token L2-normalized and scaled to the 3B's mean token-embedding norm.

**Training** (`train.py`). Both LLMs frozen. Loss = KL distillation (temperature 2) + 0.5 × cross-entropy
on the gold GSM8K solution. The *student* is the 3B reading the latent message; the *teacher* is the
same 3B reading text: either the 1B's text message (`--teacher msg`, as in the original plan) or the raw
context (`--teacher context`). AdamW 2e-4, wd 0.01, grad clip 1.0, batch 1 × 8 accumulation, gradient
checkpointing through the frozen 4-bit receiver. Sender hidden states are cached once
(`build_dataset.py`), so training only keeps the 3B on the GPU.

**Latent source.** By default (`--src ctx`) the bridge reads the 1B's hidden states from a single forward
pass over the context: the reader never decodes a single token. That is where the latency win comes from.

## Results

_Filled in from `results/` after the runs finish._

## Reproduce

```bash
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt   # torch CUDA 12.8 build
cp .env.example .env        # HF_TOKEN (gated Llama repos) and MODEL_API_KEY (Muse Spark)
python check_gpu.py         # Phase 0: GPU + both models + random-latent sanity test
python build_dataset.py     # Phase 1: 1500 train / 100 test pairs + cached sender hidden states
bash run_experiments.sh     # Phase 2: main bridges + ablations
python eval.py --ckpt checkpoints/bridge_k16_L16_ctx_context.pt   # Phase 3: live, timed evaluation
python pipeline.py "your word problem"                             # Phase 4: Muse Spark + agents
python app.py               # Phase 6: Gradio demo at http://127.0.0.1:7860
```

## Repo layout

| File | Purpose |
|---|---|
| `common.py` | config, 4-bit model loading, prompts, GSM8K helpers |
| `check_gpu.py` | Phase 0 environment check |
| `build_dataset.py` | Phase 1 dataset + cached sender hidden states |
| `bridge.py` | the latent bridge module |
| `receiver_io.py` | builds receiver inputs for text / latent / no message |
| `train.py` | Phase 2 training |
| `eval.py` | Phase 3 evaluation and ablations |
| `pipeline.py` | Phase 4 Muse Spark orchestrator + reader/solver agents |
| `app.py` | Phase 6 Gradio demo |
| `make_plots.py` | figures for this README |

## Related work

- Moschella et al., *Relative Representations Enable Zero-Shot Latent Space Communication* (ICLR 2023)
- Fu et al., *Cache-to-Cache: Direct Semantic Communication Between Large Language Models* (C2C, 2025)
- Zou et al., *Latent Collaboration in Multi-Agent Systems* (LatentMAS, 2025)
- Alayrac et al., *Flamingo* (Perceiver resampler)

# Handoff — state as of 2026-10-31 17:10 (Chișinău)

Deadline: Nov 4, 9:00 PM ET = Nov 5, 04:00 Chișinău. Plan PDF: `C:\Users\jalil\Downloads\Latent Bridge Hackathon Plan.pdf`.

## Done and verified
- Phase 0: venv `.venv` (torch 2.11 cu128), official `meta-llama/Llama-3.2-1B/3B-Instruct` in 4-bit, both load in 3.2 GB VRAM. `.env` holds HF_TOKEN and MODEL_API_KEY (git-ignored).
- Phase 1: `data/pairs.json` (1500 train), `data/test.json` (100 test), cached sender hidden states for layers 8 and 16 (`data/hidden_*.pt`).
- Muse Spark orchestrator works (`python pipeline.py --plan_only`): ~2.5 s, ~760 tokens per plan, cached on disk.
- Receiver format: short reasoning ending in `#### N`. Sender prompt: restate facts, no arithmetic.
- Baselines on first 10 test problems: no message 0/10, text message 4/10, raw context 6/10.

## Finding: the Perceiver bridge (plan's design, K=16 learned queries) collapses
- Outputs ~identical for every input (cos 0.994–0.999 between examples); another example's latents work as well as the right ones; 0/10 on eval. Teacher-forced token accuracy hid it.
- Cause: LLM hidden states are anisotropic (mean-pooled states of different problems cos 0.85); the resampler averages them and training settles on a generic "math problem" soft prompt.
- Tried: dropping the constant instruction states + position encoding + reconstruction loss (exp v1): gap 0.000. Plus per-dim input standardization (exp v2): gap 0.002, 2/20.
- Debug tool: `python debug_latent.py <ckpt> 6`.

## In progress: token-aligned bridge (`TokenBridge` in bridge.py, now the default in train.py)
- One soft token per context token (or per `--stride` tokens), sender layer 8, anchored to the receiver's embedding of the same token (shared Llama 3 tokenizer) via `--align_w`, plus KL + CE + recon losses.
- Quick test command (was killed by Claude Code's low-memory guard at step 21/50):
  `python train.py --limit 400 --epochs 1 --tag exp_v3_token`
  Success = `val_shuffle_gap` clearly > 0 and `val_gen_acc` well above 0.

## Next steps
1. Rerun exp v3. If it works: full run `python train.py --epochs 3 --tag main_token` (~1.5 h).
2. `python eval.py --ckpt checkpoints/bridge_main_token.pt --tag live_main` (~1 h).
3. Ablations: stride 1/2/4, layer 8 vs 16 (`run_experiments.sh` needs updating to the token arch).
4. Test `pipeline.py` and `app.py` (Gradio) end to end; `make_plots.py`; fill README results + collapse finding.
5. User creates the GitHub repo; push; record 3-minute video on Nov 4.

## RAM note
Training uses ~5 GB system RAM; Claude Code kills background jobs when RAM is critically low. Close Chrome/Figma/Edge while training, or start Claude Code with `CLAUDE_CODE_DISABLE_BG_SHELL_PRESSURE_REAP=1`.

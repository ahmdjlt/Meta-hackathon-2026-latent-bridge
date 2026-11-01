# Handoff — state as of 2026-11-01 (Chișinău)

Deadline: Nov 4, 9:00 PM ET = Nov 5, 04:00 Chișinău. Plan PDF: `C:\Users\jalil\Downloads\Latent Bridge Hackathon Plan.pdf`.
README.md has the full write-up and results.

## Done and verified
- Phases 0–4 and 6 work end to end on the RTX 3050 (4 GB). `.env` holds HF_TOKEN and MODEL_API_KEY (git-ignored).
- Main bridge `checkpoints/bridge_main_token.pt` (token bridge, stride 1, reader layer 8, ridge-initialized).
- Live eval, 100 GSM8K test problems: latent 68% vs text 59% vs raw context 73% vs none 2%;
  9.9 s vs 13.7 s per question; reader 0.06 s vs 2.72 s. Files: `results/eval_live_main.*`.
- Ablations (`results/ablation.md`): layer 8 69% > layer 16 63%; stride 2 60%; stride 4 21%; Perceiver 4%.
- `pipeline.py` (Muse Spark orchestrator) and `app.py` (Gradio, http://127.0.0.1:7860) tested end to end.
- Perceiver collapse diagnosed and documented in README ("What did not work").

## In progress
- DONE: stride-2 bridge `checkpoints/bridge_main_token_s2.pt`: 60% with 20 vectors (text: 59% with 60 tokens), 11.0 s per question.

## Remaining
1. Add the stride-2 result to the README; optionally a demo toggle for it (`BRIDGE_CKPT` env var).
2. User creates the GitHub repo -> push (checkpoints and data are git-ignored; note how to rebuild).
3. Record the 3-minute demo video on Nov 4 (`python app.py`), submit via Cerebral Valley.
4. Ask on the hackathon Discord whether Llama 3.2 counts as a Meta model (Muse Spark is used either way).

## RAM note
Training now stays at ~2 GB RAM / 3.5 GB VRAM (CUDA memory capped at 3.9 GB in train.py).
Close Chrome/Figma/Edge for long runs to be safe.

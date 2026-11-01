# Handoff — state as of 2026-11-01 13:40 (Chișinău)

Deadline: Nov 4, 9:00 PM ET = Nov 5, 04:00 Chișinău. README.md has the full write-up and all results.

## Headline results
- 500 GSM8K test problems (eval_big.py): latent 70.4% vs text 60.2%, McNemar p = 1.6e-5; stride 2 ties text
  (58.6%, p = 0.58) at a third of the message size; raw context 77.6%. 13.4 s -> 10.1 s per question; the 3B's
  decode time is the same across conditions, the gain is the 1B not generating (2.73 s -> 0.06 s).
- Size curve (size_curve.py, n = 100): at ~20 positions latent 60% vs text 20-22% vs truncated context 25%.
- Ridge matrix only (no training): 57%. Layer 8 > 16. Perceiver (plan's design) collapses: 4%.
- Muse Spark verify (verify_eval.py): 30% trigger rate, catches 97% of latent errors at 95% precision,
  accuracy 70.4% -> 78.0% with a text retry (whole-system number; Muse Spark is a much larger model).

## Checkpoints (git-ignored)
bridge_main_token.pt (stride 1, main), bridge_main_token_s2.pt (stride 2), bridge_abl_*.pt, ridge_L8/L16.pt.

## Remaining (user)
1. Create the public GitHub repo -> push.
2. Record the 3-minute video on Nov 4 (`python app.py`), submit via Cerebral Valley.
3. Ask on Discord whether Llama 3.2 counts as a Meta model (Muse Spark is used either way).
4. Rotate/delete the shared HF token and Meta API key after the hackathon.
Not started by request: Phase 5 (shared hub).

# Handoff — state as of 2026-11-02 (Chișinău)

Deadline: Nov 4, 9:00 PM ET = Nov 5, 04:00 Chișinău. README.md has the full write-up and all results.

## Headline results (unchanged)
- 500 GSM8K test problems (eval_big.py): latent stride 1 70.4% vs text 60.2%, McNemar p = 1.6e-5; stride 2 ties
  text (58.6%); raw context 77.6%. 13.4 s -> 10.1 s per question (1B not generating: 2.73 s -> 0.06 s).
- Size curve (n = 100): at ~20 positions latent 60% vs text 20-22% vs truncated context 25%.
- Muse Spark verify + text retry: 70.4% -> 78.0% (whole-system number).
- Extended stride-2 training (Nov 1): 62.6% vs 58.6%, McNemar p = 0.057 -> NOT adopted, reported as ablation.

## Checkpoints (git-ignored, never overwrite)
checkpoints/bridge_main_token.pt (stride 1, official), bridge_main_token_s2.pt (stride 2, official), copies in
checkpoints/best_backup/. bridge_ext_s2_* are the not-adopted extended run.

## Session 2026-11-02: agent side feature (branch `agent-demo`, not merged/pushed)
Commits: step 0 (go/no-go), step 1 (CLI agent), step 2 (demos + number check), README "Agent demo" + this file.
- Step 0, agent/ood_test.py: 20 self-written encyclopedia passages -> raw 100%, text 100%, latent 95%,
  none 30%. Within 15 points -> GO. Results in agent/ood_results.md/.json.
- Step 1, agent/cli.py: Muse Spark (tool calling works on muse-spark-1.1) + web_search (ddgs), fetch_url
  (trafilatura, docs stored locally, brain sees a preview), read_and_answer (agent/reader.py: <=48-token
  chunks, top-6 chunks by word overlap, --channel latent|text), run_python (prints code, asks y/N, 10 s,
  agent/sandbox/). Max 8 tool steps, verify before printing.
- Found: latent channel sometimes shifts a digit inside comma-separated 3-digit groups (1,642 -> 1,542).
  Added a number check: answer numbers not in the source chunks -> retry over text. Documented in README.
- Step 2: agent/demo_runs/ a_web_latent.txt (tardigrades, 1B 0.36 s), c_web_text.txt (same, 1B 14.88 s),
  b_coding.txt (Friday-13th script), plus two Baikal runs showing the number issue and the retry.
- Agent is working; no need to fall back to the Gradio-only plan.

## Next
1. Review branch `agent-demo`, then merge into main and push (user decides).
2. Record the 3-minute video on Nov 4 (`python app.py` and/or `python agent/cli.py ...`), submit via
   Cerebral Valley.
3. Ask on Discord whether Llama 3.2 counts as a Meta model (Muse Spark is used either way).
4. Check the README line "Extended training run with Muse Code" is accurate (only claim Muse Code for work it did).
5. Rotate/delete the shared HF token and Meta API key after the hackathon.
Not started by request: Phase 5 (shared hub).

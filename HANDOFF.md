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

## Session 2026-11-04 (05:30-06:30 Chisinau) -- deadline TONIGHT Nov 5 04:00 Chisinau (= Nov 4 9 PM ET)

### Gap review (vs judging info; source = third-party wiki github.com/meta-ai-hackathon/meta-hack, not official)
- Pass/fail: must "genuinely use Meta Model API / open-weight models". Wiki lists Muse Spark/Wonder/Image/
  SAM 3.1 + open weights Muse Glimmer / Muse Spark Open. Llama 3.2 NOT listed -> Muse must be visible in demo.
  chat_app.py says "no API model" -> if used for the video, Muse disappears. Still: ask Discord re Llama.
- Scoring: Impact 30%, Meta AI tooling 25%, Demo 25% (3-min video = primary artifact), Technical depth 20%.
- Rules: public open-source repo, list all contributors, "unclear contribution = disqualification".
  Banned: "basic RAG apps" -> frame agent reader as latent communication, not retrieval.
- Bonuses: Best Agentic ($5K, our agent fits), Most Viral ($20K, X post link), Open Weight ($10K, Glimmer).
- TODO found:
  1. origin/main lacks agent/, chat_app.py, demo_cli.py, demo/; agent-demo branch never pushed.
  2. app.py deleted locally but README + demo/views.py + demo/build_cache.py still reference it.
  3. No LICENSE (add MIT). 4. requirements.txt lacks `rich` (demo_cli.py).
  5. Checkpoints git-ignored (65 MB each) -> upload to HF (account ahmdjlt) and link.
  6. README needs a short "who it helps / impact" section at the top.
  7. README line "Extended training run with Muse Code" -- remove unless true; add note on AI coding tools used
     (Nov 2 commits co-authored by Claude).
  8. Video not recorded. 9. Submit on Cerebral Valley + post on X + claim API credits before 9 PM ET.
- API key works; available models: muse-spark-1.1/1.2/1.3, muse-wonder-1.0, muse-image-1.0, sam-3.1, voice.

### New: Muse Spark as IMPROVER (improve.py, uncommitted)
Loop with no answer key: 1B->bridge->3B solves NEW GSM8K train problems (train rows after 1709, zero overlap
with train/val, asserted) -> Muse Spark 1.3 checks each answer; reject => writes corrected solution ending
'#### N' -> bridge fine-tuned on accepted team answers + Muse corrections (same losses; KL teacher = 3B on
raw context, label-free). Gold used only to report label accuracy. Stages resumable in data/improve_<tag>/.
- Smoke test (8 problems): 3 accept / 5 reject, Muse labels 8/8 correct, ~900 tokens per problem.
- Round 1 STOPPED by user at 120/300 solved (data/improve_muse_r1/solve.jsonl). Resume:
    python improve.py --tag muse_r1 --ckpt checkpoints/best_backup/bridge_main_token_s2.pt --n 300
    python val_full.py --ckpt checkpoints/bridge_muse_r1.pt --tag muse_r1
  (~35 min solve left, then teach ~5 min, train ~15 min, val 200 ~34 min). Compare on val 200:
  old stride-2 62.5% (baseline), gold-label extended training 65.0%.
- Needs the GPU: stop chat_app.py first (it holds 3.6 GB).

### New: Muse Glimmer 30B as SOLVER (glimmer/, uncommitted)
- Glimmer: 30B dense, own tokenizer (vocab 202048, Llama-4 style), embeddings RMS-normed (norm sqrt(6656)=81.6),
  chat template with reasoning_strength (we pass "low"). Spark Open is 816B (out of reach).
- glimmer/fetch_embed.py: range-download of only the embedding table -> data/glimmer_embed.pt (DONE).
- glimmer/fit_ridge_glimmer.py: cross-tokenizer ridge (target = mean Glimmer embedding of tokens overlapping
  the same characters). DONE: held-out cos 0.711 (Llama->Llama was 0.72), nearest-token top-1 66.7% (vs 80%).
  -> checkpoints/ridge_glimmer_L8.pt, checkpoints/bridge_glimmer_ridge.pt (zero training).
- glimmer/modal_eval.py: Modal H100, bf16, 100 test problems x {none, text, latent, context}.
  Tested code path with a tiny random Glimmer on CPU (generate with inputs_embeds OK); latents built OK (51 MB).
  NOT RUN YET. Needs: modal.com account (free $30/mo Starter is enough; hackathon Modal credit not guaranteed),
  then `! .venv/Scripts/modal token new`, then `modal run glimmer/modal_eval.py --n 100` (~1-1.5 h, ~$5-8).
  Output: results/glimmer_eval.json/.md. Hard cutoff idea: if not working by mid-afternoon, drop it.

### Plan for the rest of today
1. Resume improver (above). 2. Modal + Glimmer run (optional). 3. Repo cleanup (TODO list), README sections
for improver + Glimmer results, commit, merge agent-demo -> main, push. 4. Record 3-min video (show Muse Spark
planning/verifying/improving + latent vs text). 5. Submit before 9 PM ET (aim 6 PM ET), post on X.

### Improver round 1 result (09:55 Chișinău)
- Done: solve 300 → Muse teach (174 accept / 126 reject, label acc 96.3%, 115/123 wrong answers fixed, 5 right ones wrongly rejected, 260K tokens) → train 2 ep → val_full 200.
- Val: **58.0%** (116/200) vs baseline 62.5%, gold-ext 65.0%. Not kept. README section "Muse Spark improver" written honestly; VIDEO_SCRIPT 1:40 line filled.
- Still uncommitted (user: "dont commit anything yet"). GPU is free → chat_app.py can be restarted.

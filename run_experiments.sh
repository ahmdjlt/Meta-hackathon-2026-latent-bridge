#!/usr/bin/env bash
# Experiment queue (Git Bash on Windows, or any POSIX shell). Logs go to results/_*.log.
#   main   : token-aligned bridge, stride 1, sender layer 8, 3 epochs
#   eval   : live, timed evaluation of the main bridge on 100 held-out GSM8K problems
#   ablate : 1-epoch ablations (stride = compression, sender layer) + generation accuracy for each
set -u
cd "$(dirname "$0")"
PY=./.venv/Scripts/python
[ -x "$PY" ] || PY=./.venv/bin/python
export PYTHONIOENCODING=utf-8
STAGE=${1:-all}

train() { echo ">> train $*"; $PY train.py "$@" > "results/_train_$(echo "$*" | tr ' -' '__').log" 2>&1 || echo "FAILED train $*"; }

if [ "$STAGE" = all ] || [ "$STAGE" = main ]; then
  train --arch token --stride 1 --layer 8 --epochs 3 --tag main_token
fi
if [ "$STAGE" = all ] || [ "$STAGE" = eval ]; then
  echo ">> eval main"
  $PY eval.py --ckpt checkpoints/bridge_main_token.pt --tag live_main > results/_eval_main.log 2>&1 || echo "FAILED eval"
fi
if [ "$STAGE" = all ] || [ "$STAGE" = ablate ]; then
  for S in 1 2 4; do train --arch token --stride $S --layer 8 --epochs 1 --tag abl_s${S}_L8; done
  train --arch token --stride 1 --layer 16 --epochs 1 --tag abl_s1_L16
  echo ">> eval ablations"
  $PY eval.py --tag ablation --ablate checkpoints/bridge_abl_*.pt checkpoints/bridge_k16_L16_ctx_context.pt \
    > results/_eval_ablation.log 2>&1 || echo "FAILED ablation eval"
fi
echo ">> done stage $STAGE"

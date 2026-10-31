#!/usr/bin/env bash
# Experiment queue (Git Bash on Windows, or any POSIX shell). Logs go to results/_*.log.
#   main   : the two main bridges (3 epochs): text-message teacher vs raw-context teacher
#   eval   : live, timed evaluation of the main bridge on 100 held-out GSM8K problems
#   ablate : 1-epoch ablations (K, sender layer, latent source) + generation accuracy for every bridge
set -u
cd "$(dirname "$0")"
PY=./.venv/Scripts/python
[ -x "$PY" ] || PY=./.venv/bin/python
export PYTHONIOENCODING=utf-8
STAGE=${1:-all}
T=${TEACHER:-context}

train() { echo ">> train $*"; $PY train.py "$@" > "results/_train_$(echo "$*" | tr ' -' '__').log" 2>&1 || echo "FAILED train $*"; }

if [ "$STAGE" = all ] || [ "$STAGE" = main ]; then
  train --k 16 --layer 16 --src ctx --teacher msg
  train --k 16 --layer 16 --src ctx --teacher context
fi
if [ "$STAGE" = all ] || [ "$STAGE" = eval ]; then
  echo ">> eval main"
  $PY eval.py --ckpt checkpoints/bridge_k16_L16_ctx_$T.pt --tag live_main > results/_eval_main.log 2>&1 || echo "FAILED eval"
fi
if [ "$STAGE" = all ] || [ "$STAGE" = ablate ]; then
  for K in 4 8 16 32; do train --k $K --layer 16 --src ctx --teacher $T --epochs 1 --tag abl_k${K}_L16_ctx; done
  train --k 16 --layer 8 --src ctx --teacher $T --epochs 1 --tag abl_k16_L8_ctx
  train --k 16 --layer 16 --src full --teacher $T --epochs 1 --tag abl_k16_L16_full
  echo ">> eval ablations"
  $PY eval.py --tag ablation --ablate checkpoints/bridge_k16_L16_ctx_msg.pt checkpoints/bridge_k16_L16_ctx_context.pt \
    checkpoints/bridge_abl_*.pt > results/_eval_ablation.log 2>&1 || echo "FAILED ablation eval"
fi
echo ">> done stage $STAGE"

#!/usr/bin/env bash
# Full experiment queue (Git Bash on Windows, or any POSIX shell). Logs go to results/_*.log.
# Stage 1: main bridge with both teachers. Stage 2: ablations (K, layer, source). Stage 3: evals.
set -u
cd "$(dirname "$0")"
PY=./.venv/Scripts/python
[ -x "$PY" ] || PY=./.venv/bin/python
export PYTHONIOENCODING=utf-8
STAGE=${1:-all}

train() { echo ">> train $*"; $PY train.py "$@" > "results/_train_$(echo "$*" | tr ' -' '__').log" 2>&1 || echo "FAILED train $*"; }

if [ "$STAGE" = all ] || [ "$STAGE" = main ]; then
  train --k 16 --layer 16 --src ctx --teacher msg
  train --k 16 --layer 16 --src ctx --teacher context
fi
if [ "$STAGE" = all ] || [ "$STAGE" = ablate ]; then
  T=${TEACHER:-context}
  for K in 4 8 32; do train --k $K --layer 16 --src ctx --teacher $T --epochs 2; done
  train --k 16 --layer 8 --src ctx --teacher $T --epochs 2
  train --k 16 --layer 16 --src full --teacher $T --epochs 2
fi
echo ">> done stage $STAGE"

#!/usr/bin/env bash
# Overnight chain for the compression study. Each step is resumable / idempotent.
set -u
cd "$(dirname "$0")"
PY=./.venv/Scripts/python
export PYTHONIOENCODING=utf-8
step() { echo ">> $(date +%H:%M) $1"; }
step "build 500-problem test set"
[ "$($PY -c "import json;print(len(json.load(open('data/test.json',encoding='utf-8'))))")" -ge 500 ] || \
  $PY build_dataset.py --only_test --n_test 500 > results/_build_test500.log 2>&1 || echo "FAILED build"
step "train stride-3 bridge (1 epoch)"
[ -f checkpoints/bridge_abl_s3_L8.pt ] || \
  $PY train.py --arch token --stride 3 --layer 8 --epochs 1 --tag abl_s3_L8 > results/_train_abl_s3.log 2>&1 || echo "FAILED s3"
step "size curve (n=100)"
$PY size_curve.py --n 100 > results/_size_curve.log 2>&1 || echo "FAILED size curve"
step "big eval (n=500)"
$PY eval_big.py --n 500 > results/_eval_big.log 2>&1 || echo "FAILED big eval"
step "done"
step "size curve: truncated normal text messages"
$PY size_curve.py --n 100 --strides "" > results/_size_curve2.log 2>&1 || echo "FAILED size curve 2"
step "done (2)"

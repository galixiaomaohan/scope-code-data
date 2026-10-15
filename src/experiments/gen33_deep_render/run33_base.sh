#!/bin/bash
# run33_base.sh -- gen33 baseline-arm training (no attention; --pa-enable 0)
# GPU0 user redline: structurally refused (g33_common.gpu_gate SystemExits directly on idx==0)
# single-GPU exclusivity gate: free > 4500MiB
set -u
P="${SCOPE_PYTHON:-python3}"
R="${SCOPE_RESEARCH_ROOT:-.}"
G=$R/scratch/ccp/gen33
DEV=${1:-cuda:1}
mkdir -p $G/logs $G/ckpt $G/out
cd $R || exit 3
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${SCOPE_SUPEROPERATOR:-../device}"
nice -n 19 $P $G/gen33_hybrid.py \
  --cache  $G/cache/deep_all.bin \
  --side   $G/out/DEEP_RENDER_SIDECAR.jsonl \
  --vocab  $G/cache/CCP_VOCAB_V4.json \
  --allow  $G/cache/ccp_allow_v4.bin \
  --init   ckpt/sla_v31f.pt \
  --lat31  cppcuda/cache/lat_tab_v31.bin \
  --steps 6000 --bs 6 --lr 4e-5 --wd 0.01 --clip 1.0 --warmup 50 \
  --lam-bind 1.0 --lam-txt 1.0 --seed 20260923 --body-dtype fp32 \
  --grad-ckpt 1 --micro-token-cap 9000 --det-train 1 \
  --eval-every 50 --eval-rows 48 --snap-every 100 \
  --args-mode bridge --render-only-eval 1 --stopnan 1 \
  --pa-enable 0 \
  --run-kind formal --tag deep_base --dev "$DEV" \
  --out     $G/ckpt/deep_base_formal_6000.pt \
  --curve   $G/out/DEEP_BASE_CURVE.jsonl \
  --plan-out $G/out/batch_idx_deep.txt \
  --summary $G/out/DEEP_BASE_SUMMARY.json \
  --samples $G/out/DEEP_BASE_SAMPLES.json \
  --log     $G/logs/gen33_hybrid_base.log
echo "BASE_TRAIN_RC=$?"

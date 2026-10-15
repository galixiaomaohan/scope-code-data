#!/bin/bash
# gen36_gs_launch.sh -- G-S third-arm launch (pre-registered V36 M3_G+A-1 path, 2026-09-25 02:5x main run)
#   2x2 factorial complete: gen35 = no-shift x equal weight 117/218 . G-W = no-shift x weighted 110/218 .
#   G-SW = shift x weighted 172/218 . G-S = shift x equal weight = this arm
#   Flag-for-flag isomorphic with G-SW, only --w-tail 1.0 (equal weight) + new tag; same a2 cache / same seed / same expected batch order 89de9e65
set -u
P="${SCOPE_PYTHON:-python3}"
R="${SCOPE_RESEARCH_ROOT:-.}"
G36=$R/scratch/ccp/gen36
cd $R || exit 3
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${SCOPE_SUPEROPERATOR:-../device}"
unset CUDA_VISIBLE_DEVICES
export OMP_NUM_THREADS=4
TAG=deep_plan_formal_a2wt1_6000
F=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 2 | head -1)
echo "[gate] GPU2 free=${F}MiB"
[ -z "$F" ] && exit 21
[ "$F" -lt 4500 ] && { echo "[gate] GPU2 free insufficient"; exit 21; }
nice -n 19 $P $G36/gen36_hybrid.py \
  --cache $G36/cache/deep_all_a2.bin \
  --side $R/scratch/ccp/gen33/out/DEEP_RENDER_SIDECAR.jsonl \
  --vocab $R/scratch/ccp/gen33/cache/CCP_VOCAB_V4.json \
  --allow $R/scratch/ccp/gen33/cache/ccp_allow_v4.bin \
  --init ckpt/sla_v31f.pt \
  --lat31 cppcuda/cache/lat_tab_v31.bin \
  --steps 6000 --bs 6 --lr 4e-5 --wd 0.01 --clip 1.0 --warmup 50 \
  --lam-bind 1.0 --lam-txt 0.0 --w-tail 1.0 \
  --seed 20260923 --body-dtype fp32 --grad-ckpt 1 --micro-token-cap 9000 \
  --det-train 1 --eval-every 50 --eval-rows 48 --snap-every 100 \
  --args-mode bridge --render-only-eval 1 --stopnan 1 --pa-enable 0 \
  --slot-cond 1 --slot-dev 1 --slot-side $R/scratch/ccp/gen34/out/DEEP_SLOT_SIDECAR.jsonl \
  --slot-beta 8.0 --lam-slot 3.0 --run-kind formal \
  --tag $TAG --dev cuda:2 \
  --out $G36/ckpt/$TAG.pt --curve $G36/out/CURVE_$TAG.jsonl \
  --plan-out $G36/out/batch_idx_$TAG.txt --summary $G36/out/SUMMARY_$TAG.json \
  --samples $G36/out/SAMPLES_$TAG.json --log $G36/logs/gen36_hybrid_$TAG.log
RC=$?
echo "GS_TRAIN_RC=$RC"
exit $RC

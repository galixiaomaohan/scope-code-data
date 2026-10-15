#!/bin/bash
# run36_train.sh -- gen36 training launch (V36_DESIGN §4 gates + §3 diff list)
#   Usage: bash run36_train.sh <arm>   arm in {w, sw, s, smoke_w, smoke_sw}
#     w   = G-W  main arm: gen35 cache (frame state R_in) + terminal-step chain CE x5   (main reading of the task)
#     sw  = G-SW second arm: shifted cache deep_all_a2 (post-step state) + terminal step x5
#     s   = G-S third arm (optional single variable): shifted cache + terminal step x1.0 (run only if stop condition A-1 allows)
#   GPU allowlist: cuda:2 only (GPU0 user redline / GPU1 reserved zone structurally rejected)
set -u
P="${SCOPE_PYTHON:-python3}"
R="${SCOPE_RESEARCH_ROOT:-.}"
G36=$R/scratch/ccp/gen36
G33=$R/scratch/ccp/gen33
G34=$R/scratch/ccp/gen34
ARM=${1:-w}
case "$ARM" in
  w)        CACHE=$G33/cache/deep_all.bin;   WT=5.0; STEPS=6000; NAME=deep_plan_formal_wt5_6000;   RK=formal ;;
  sw)       CACHE=$G36/cache/deep_all_a2.bin; WT=5.0; STEPS=6000; NAME=deep_plan_formal_a2wt5_6000; RK=formal ;;
  s)        CACHE=$G36/cache/deep_all_a2.bin; WT=1.0; STEPS=6000; NAME=deep_plan_formal_a2wt1_6000; RK=formal ;;
  smoke_w)  CACHE=$G33/cache/deep_all.bin;   WT=5.0; STEPS=100;  NAME=smoke_wt5; RK=smoke ;;
  eqv)      CACHE=$G33/cache/deep_all.bin;   WT=1.0; STEPS=100;  NAME=smoke_eqv_wt1; RK=smoke ;;
  smoke_sw) CACHE=$G36/cache/deep_all_a2.bin; WT=5.0; STEPS=100; NAME=smoke_a2wt5; RK=smoke ;;
  *) echo "unknown arm: $ARM"; exit 3 ;;
esac
mkdir -p $G36/logs $G36/ckpt $G36/out $G36/lean
cd $R || exit 3
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${SCOPE_SUPEROPERATOR:-../device}"
unset CUDA_VISIBLE_DEVICES             # keep it unset = the allowlisted value (gpu_gate asserts this; an empty string masks all visible GPUs)
export OMP_NUM_THREADS=4

# ---- pre-launch gates (G-0 / G-1 / G-4 / G-5) ----
$P - <<PYEOF || exit 21
import hashlib, json, os, sys
sys.path.insert(0, "$G36")
import gen36_common as C
jm = C.md5f(C.DRCT + "/drct_common.py")
assert jm == C.JUDGE_MD5_EXPECT, ("G-0 判卷件", jm)
assert os.path.exists("$CACHE"), "缓存缺失 $CACHE"
n, det = C.n_lean_jobs()
assert n < 5, ("G-5 Lean 并发", n)
g = C.gpu_gate("cuda:2", logp=None)
led = C.parent_ledger()
print("[gate] G-0 judge=%s | G-1 %s | G-5 lean=%d | cache=%s md5=%s"
      % (jm[:8], g, n, "$CACHE", C.md5f("$CACHE")[:8]))
print("[gate] parent ledger n=%d (drct=%s gen35_hybrid=%s)"
      % (len(led), led["drct_common"]["md5"][:8],
         (led.get("gen35_hybrid") or {}).get("md5", "")[:8]))
PYEOF
[ $? -ne 0 ] && exit 21

nice -n 19 $P $G36/gen36_hybrid.py \
  --cache  $CACHE \
  --side   $G33/out/DEEP_RENDER_SIDECAR.jsonl \
  --vocab  $G33/cache/CCP_VOCAB_V4.json \
  --allow  $G33/cache/ccp_allow_v4.bin \
  --init   ckpt/sla_v31f.pt \
  --lat31  cppcuda/cache/lat_tab_v31.bin \
  --steps $STEPS --bs 6 --lr 4e-5 --wd 0.01 --clip 1.0 --warmup 50 \
  --lam-bind 1.0 --lam-txt 0.0 --w-tail $WT --seed 20260923 --body-dtype fp32 \
  --grad-ckpt 1 --micro-token-cap 9000 --det-train 1 \
  --eval-every 50 --eval-rows 48 --snap-every 100 \
  --args-mode bridge --render-only-eval 1 --stopnan 1 \
  --pa-enable 0 --slot-cond 1 --slot-dev 1 \
  --slot-side $G34/out/DEEP_SLOT_SIDECAR.jsonl --slot-beta 8.0 --lam-slot 3.0 \
  --run-kind $RK --tag $NAME --dev cuda:2 \
  --out     $G36/ckpt/$NAME.pt \
  --curve   $G36/out/CURVE_$NAME.jsonl \
  --plan-out $G36/out/batch_idx_$NAME.txt \
  --summary $G36/out/SUMMARY_$NAME.json \
  --samples $G36/out/SAMPLES_$NAME.json \
  --log     $G36/logs/gen36_hybrid_$NAME.log
RC=$?
echo "TRAIN_${ARM}_RC=$RC"
# ---- batch-order gate (G-4; same value as gen34/gen35) ----
if [ -f $G36/out/batch_idx_$NAME.txt ]; then
  PM=$(md5sum $G36/out/batch_idx_$NAME.txt | cut -d' ' -f1)
  echo "PLAN_MD5_$ARM=$PM (expect 89de9e657b1986c6acf3cdf944e721ca)"
fi
exit $RC

#!/bin/bash
# gen36_cpu2.sh -- G-SW closing pair (CPU/Lean, 2026-09-25 04:0x main run, V36 criteria companion)
#   [1] migration matrix: anchor vs G-SW (first-error step position / terminal-class confusion / confident errors -- core of the remaining 46 errors)
#   [2] dual convention: certified stratified literal re-judging 66 + reverse 12 (G-SW 172 pool)
set -u
P="${SCOPE_PYTHON:-python3}"
R="${SCOPE_RESEARCH_ROOT:-.}"
G36=$R/scratch/ccp/gen36
cd $R || exit 3
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${SCOPE_SUPEROPERATOR:-../device}"
unset CUDA_VISIBLE_DEVICES
export OMP_NUM_THREADS=4

echo "== [1/2] MIGRATION gsw vs anchor =="
nice -n 19 $P $G36/gen36_migration.py \
  --arm "gsw=$G36/out/GEN36_WALK_gsw.json" \
  --out $G36/out/GEN36_MIGRATION_gsw.json
RC1=$?
echo "MIG_RC=$RC1"

echo "== [2/2] DUAL literal+reverse for gsw =="
nice -n 19 $P $G36/gen36_judge.py \
  --src $G36/out/GEN36_PLAN_gsw_ROWS.jsonl \
  --src-tag gsw --pool certified --n-literal 66 --n-reverse 12 \
  --out $G36/out/GEN36_JUDGE_dual_gsw.json \
  --log $G36/logs/gen36_judge_dual_gsw.log
RC2=$?
echo "DUAL_RC=$RC2"
echo "CPU2_DONE rc1=$RC1 rc2=$RC2"

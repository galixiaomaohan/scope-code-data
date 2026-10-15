#!/bin/bash
# gen36_cpu3.sh -- G-S closing pair (CPU/Lean, 2026-09-25 05:0x main run, four-cell table completion companion)
#   [1] migration matrix: anchor vs G-S (core of the remaining 27 errors)
#   [2] dual convention: certified stratified literal re-judging 66 + reverse 12 (G-S 191 pool)
set -u
P="${SCOPE_PYTHON:-python3}"
R="${SCOPE_RESEARCH_ROOT:-.}"
G36=$R/scratch/ccp/gen36
cd $R || exit 3
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${SCOPE_SUPEROPERATOR:-../device}"
unset CUDA_VISIBLE_DEVICES
export OMP_NUM_THREADS=4

echo "== [1/2] MIGRATION gs vs anchor =="
nice -n 19 $P $G36/gen36_migration.py \
  --arm "gs=$G36/out/GEN36_WALK_gs.json" \
  --out $G36/out/GEN36_MIGRATION_gs.json
RC1=$?
echo "MIG_RC=$RC1"

echo "== [2/2] DUAL literal+reverse for gs =="
nice -n 19 $P $G36/gen36_judge.py \
  --src $G36/out/GEN36_PLAN_gs_ROWS.jsonl \
  --src-tag gs --pool certified --n-literal 66 --n-reverse 12 \
  --out $G36/out/GEN36_JUDGE_dual_gs.json \
  --log $G36/logs/gen36_judge_dual_gs.log
RC2=$?
echo "DUAL_RC=$RC2"
echo "CPU3_DONE rc1=$RC1 rc2=$RC2"

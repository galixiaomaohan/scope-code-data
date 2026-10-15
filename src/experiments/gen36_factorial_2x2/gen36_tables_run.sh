#!/bin/bash
# gen36_tables_run.sh -- V36 close-out summary table (all three arms + anchors, 2026-09-25 05:3x main run)
set -u
P="${SCOPE_PYTHON:-python3}"
R="${SCOPE_RESEARCH_ROOT:-.}"
G36=$R/scratch/ccp/gen36
cd $R || exit 3
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${SCOPE_SUPEROPERATOR:-../device}"
unset CUDA_VISIBLE_DEVICES
export OMP_NUM_THREADS=4
nice -n 19 $P $G36/gen36_tables.py \
  --arm "gw=$G36/out/GEN36_WALK_GW.json,$G36/out/GEN36_PLAN_GW.json,$G36/out/GEN36_PLAN_GW_ROWS.jsonl" \
  --arm "gsw=$G36/out/GEN36_WALK_gsw.json,$G36/out/GEN36_PLAN_gsw.json,$G36/out/GEN36_PLAN_gsw_ROWS.jsonl" \
  --arm "gs=$G36/out/GEN36_WALK_gs.json,$G36/out/GEN36_PLAN_gs.json,$G36/out/GEN36_PLAN_gs_ROWS.jsonl" \
  --migration $G36/out/GEN36_MIGRATION_gsw.json \
  --out $G36/out/GEN36_TABLES.json
echo "TABLES_RC=$?"

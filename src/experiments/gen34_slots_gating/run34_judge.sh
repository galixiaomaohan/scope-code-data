#!/bin/bash
# run34_judge.sh -- gen34 judging: batch full run + stratified literal re-judging of the
#                  certified pool + reverse verification
# Usage: bash run34_judge.sh <batch|literal|reverse|all> [tags]
# Frozen judging file drct_common.py 7c31b58d (import only, zero edits; measured once before
# and once after judging)
set -u
P="${SCOPE_PYTHON:-python3}"
R="${SCOPE_RESEARCH_ROOT:-.}"
G=$R/scratch/ccp/gen34
MODE=${1:-all}; TAGS=${2:-slotA,slotB,base}
mkdir -p $G/logs $G/lean
cd $R || exit 3
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${SCOPE_SUPEROPERATOR:-../device}"

JM0=$(md5sum $R/scratch/drct/drct_common.py | cut -d' ' -f1)
echo "[judge gate] drct_common md5=$JM0 (expected 7c31b58da9d7228cee31b9be581fe5dd)"
[ "$JM0" != "7c31b58da9d7228cee31b9be581fe5dd" ] && { echo "JUDGE_FREEZE_FAIL"; exit 24; }
NJ=$(ps -eo cmd --no-headers | grep -F 'lake env lean' | grep -Ev 'grep' | grep -Ec 'tcaboost|\.elan' || true)
echo "[lean gate] our jobs=$NJ"
[ "$NJ" -ge 5 ] && { echo "LEAN_GATE_FAIL $NJ"; exit 22; }

nice -n 19 $P $G/gen34_judge.py --mode "$MODE" --tags "$TAGS" \
  --n-literal 90 --n-reverse 12 --chunk 60 \
  --out "" --log $G/logs/gen34_judge.log
RC=$?
JM1=$(md5sum $R/scratch/drct/drct_common.py | cut -d' ' -f1)
echo "[judge after] drct_common md5=$JM1 rc=$RC"
[ "$JM0" != "$JM1" ] && echo "JUDGE_CHANGED_FATAL"
echo "JUDGE_DONE mode=$MODE rc=$RC"

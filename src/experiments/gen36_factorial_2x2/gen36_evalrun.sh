#!/bin/bash
# gen36_evalrun.sh -- exam-section launch (chain walk -> compile -> frozen judging), one command per arm
#   Usage: bash gen36_evalrun.sh <tag> <shift> <ckpt>
#     tag   = arm label (gw / gsw / ...), artifacts out/GEN36_WALK_<TAG>.json / GEN36_PLAN_<TAG>.json / _ROWS.jsonl
#     shift = 0 (parent payload convention, with gen35-cache training) | 1 (post-step residual convention, with a2-cache training)
#     ckpt  = full path to that arm's ckpt
set -u
P="${SCOPE_PYTHON:-python3}"
R="${SCOPE_RESEARCH_ROOT:-.}"
G36=$R/scratch/ccp/gen36
TAG=${1:?tag}
SHIFT=${2:-0}
CKPT=${3:?ckpt}
mkdir -p $G36/out $G36/logs $G36/lean
cd $R || exit 3
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${SCOPE_SUPEROPERATOR:-../device}"
unset CUDA_VISIBLE_DEVICES
export OMP_NUM_THREADS=4

WALK=$G36/out/GEN36_WALK_${TAG}.json
PLAN=$G36/out/GEN36_PLAN_${TAG}.json
ROWS=$G36/out/GEN36_PLAN_${TAG}_ROWS.jsonl
LOG=$G36/logs/gen36_eval_${TAG}.log

# ---- pre-launch gates: judge artifact frozen + GPU allowlist + Lean concurrency ----
$P - <<PYEOF || exit 21
import sys
sys.path.insert(0, "$G36")
import gen36_common as C
assert C.md5f(C.DRCT + "/drct_common.py") == C.JUDGE_MD5_EXPECT
assert C.n_lean_jobs()[0] < 5, "Lean 并发 >=5"
print("[gate]", C.gpu_gate("cuda:2"))
PYEOF
[ $? -ne 0 ] && exit 21

echo "== [1/2] WALK tag=$TAG shift=$SHIFT ckpt=$CKPT =="
nice -n 19 $P $G36/gen36_eva[l].py --phase walk --state-shift $SHIFT --dev cuda:2 \
  --ckpt "$CKPT" --walk-out "$WALK" --log "$LOG"
RC1=$?
echo "WALK_RC=$RC1"
[ $RC1 -ne 0 ] && exit $RC1

echo "== [2/2] COMPILE+JUDGE tag=$TAG =="
nice -n 19 $P $G36/gen36_eva[l].py --phase compile --judge 1 --state-shift $SHIFT \
  --ckpt "$CKPT" --walk-out "$WALK" --out "$PLAN" --rows-out "$ROWS" --log "$LOG"
RC2=$?
echo "COMPILE_JUDGE_RC=$RC2 (walk=$WALK plan=$PLAN rows=$ROWS)"
if [ -f "$ROWS" ]; then md5sum "$WALK" "$PLAN" "$ROWS"; fi
exit $RC2

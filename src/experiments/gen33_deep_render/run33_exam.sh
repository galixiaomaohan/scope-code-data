#!/bin/bash
# run33_exam.sh -- gen33 exam segment: both arms × valid deep exam set (chunks of 100; zhenti-style stratification) + naked control
# Usage: bash run33_exam.sh <hybrid|naked> <ckpt path|''> <dev> <tag> [n]
# The pre-exam gate (GEN33_EXAM_STATE_GATE pass) and the change-97 gate must already have passed (this script re-checks the pass flags before launching)
set -u
P="${SCOPE_PYTHON:-python3}"
R="${SCOPE_RESEARCH_ROOT:-.}"
G=$R/scratch/ccp/gen33
ARM=${1:-hybrid}; CKPT=${2:-}; DEV=${3:-cuda:1}; TAG=${4:-base}; N=${5:-0}
cd $R || exit 3
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${SCOPE_SUPEROPERATOR:-../device}"
mkdir -p $G/lean $G/logs $G/out

# ---- pre-launch gate re-check ----
$P - <<'EOF' || exit 21
import json,sys
g=json.load(open('scratch/ccp/gen33/out/GEN33_EXAM_STATE_GATE.json'))
f=json.load(open('scratch/ccp/gen33/out/GEN33_FRAME_CONSISTENCY.json'))
assert g['pass'] and f['pass'], ('门未过', g['pass'], f['pass'])
print('[gate] state_gate pass n=%d ; frame_consistency pass n=%d' % (g['n_rows'], f['n_sample']))
EOF
[ $? -ne 0 ] && exit 21

# ---- pre-launch Lean gate (global count of our jobs < 5) ----
NJ=$(ps -eo cmd --no-headers | grep -F 'lake env lean' | grep -Ev 'grep' | grep -Ec 'tcaboost|/.elan' || true)
echo "[lean gate] our_jobs=$NJ"
[ "$NJ" -ge 5 ] && { echo "LEAN_GATE_FAIL $NJ"; exit 22; }

TOT=218
[ "$N" -gt 0 ] && TOT=$N
if [ "$ARM" = "naked" ]; then
  nice -n 19 $P $G/gen33_eval.py --arm naked --seg exam_deep --n 30 --chunk-offset 0 \
    --dev "$DEV" --vocab $G/cache/CCP_VOCAB_V4.json --allow $G/cache/ccp_allow_v4.bin \
    --exam-side $G/out/DEEP_EXAM_SIDECAR.jsonl --naked-base 135M --max-new 384 \
    --out $G/out/GEN33_EVAL_naked_exam_deep_n30.json \
    --log $G/logs/gen33_eval.log
  echo "NAKED_DONE rc=$?"
  exit 0
fi

OFF=0
while [ "$OFF" -lt "$TOT" ]; do
  nice -n 19 $P $G/gen33_eval.py --arm "$ARM" --seg exam_deep --args-mode device \
    --ckpt "$CKPT" --n 100 --chunk-offset "$OFF" \
    --cache $G/cache/deep_all.bin --side $G/out/DEEP_RENDER_SIDECAR.jsonl \
    --exam-side $G/out/DEEP_EXAM_SIDECAR.jsonl \
    --vocab $G/cache/CCP_VOCAB_V4.json --allow $G/cache/ccp_allow_v4.bin \
    --pa-enable 1 --pa-state 1 --pa-heads 2 --pa-dhead 64 --plan-summary 0 \
    --dev "$DEV" --max-new 384 --judge 0 --attn-export 0 \
    --out $G/out/GEN33_EVAL_${TAG}_exam_deep_c$(printf '%04d' $OFF).json \
    --log $G/logs/gen33_eval.log
  echo "CHUNK_DONE $TAG off=$OFF rc=$?"
  OFF=$((OFF+100))
done
echo "EXAM_ALL_DONE $TAG"

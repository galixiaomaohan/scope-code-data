#!/bin/bash
# run34_exam.sh -- gen34 exam segment: three arms x effective deep exam 218 (chunked by 100)
#                 + same-distribution control a_held n=20
# Usage: bash run34_exam.sh <A|B|C> <dev> [n]
#   A = slots + device (ckpt deep_slotA_formal_6000, --slot-cond 1 --slot-dev 1)
#   B = slots + LM fills (ckpt deep_slotB_formal_6000, --slot-cond 1 --slot-dev 0)
#   C = baseline control (gen33 ckpt deep_base_formal_6000, --slot-cond 0)
#       [C-arm equivalence: see GEN34_EQUIV]
# Pre-launch gates: PA-EQ pass + slot coverage gate pass + our Lean jobs < 5
set -u
P="${SCOPE_PYTHON:-python3}"
R="${SCOPE_RESEARCH_ROOT:-.}"
G=$R/scratch/ccp/gen34
G33=$R/scratch/ccp/gen33
ARM=${1:-A}; DEV=${2:-cuda:1}; N=${3:-0}
mkdir -p $G/logs $G/out $G/lean
cd $R || exit 3
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${SCOPE_SUPEROPERATOR:-../device}"

$P - <<'EOF' || exit 21
import json, os
G = os.environ.get("SCOPE_RESEARCH_ROOT", ".") + "/scratch/ccp/gen34"
assert json.load(open(G+'/out/GEN34_PAEQ.json'))['pass'], 'PA-EQ not passed'
s=json.load(open(G+'/out/GEN34_SLOTS.json'))
assert s['gate']['pass'] and not s['gate']['stopline_hit'], ('slot coverage gate', s['gate'])
print('[gate] PA-EQ pass ; slot cov corpus=%.4f exam=%.4f'
      % (s['gate']['coverage_corpus'], s['gate']['coverage_exam']))
EOF
[ $? -ne 0 ] && exit 21
NJ=$(ps -eo cmd --no-headers | grep -F 'lake env lean' | grep -Ev 'grep' | grep -Ec 'tcaboost|\.elan' || true)
echo "[lean gate] our jobs=$NJ"
[ "$NJ" -ge 5 ] && { echo "LEAN_GATE_FAIL $NJ"; exit 22; }

case "$ARM" in
  A) SC=1; SD=1; CK=$G/ckpt/deep_slotA_formal_6000.pt; TAG=slotA ;;
  B) SC=1; SD=0; CK=$G/ckpt/deep_slotB_formal_6000.pt; TAG=slotB ;;
  C) SC=0; SD=0; CK=$G33/ckpt/deep_base_formal_6000.pt; TAG=base ;;
  *) echo "unknown arm $ARM"; exit 23 ;;
esac
echo "[exam] arm=$TAG slot_cond=$SC slot_dev=$SD ckpt=$CK"

# ---- same-distribution control a_held n=20 (run first, it is fast) ----
nice -n 19 $P $G/gen34_eval.py --arm hybrid --seg a_held --args-mode device \
  --ckpt "$CK" --n 20 --chunk-offset 0 \
  --cache $G33/cache/deep_all.bin --side $G33/out/DEEP_RENDER_SIDECAR.jsonl \
  --vocab $G33/cache/CCP_VOCAB_V4.json --allow $G33/cache/ccp_allow_v4.bin \
  --pa-enable 0 --plan-summary 0 --slot-cond "$SC" --slot-dev "$SD" \
  --dev "$DEV" --max-new 384 --judge 0 --attn-export 0 \
  --out $G/out/GEN34_EVAL_${TAG}_a_held_n20.json \
  --log $G/logs/gen34_eval.log
echo "CTRL_DONE $TAG rc=$?"

TOT=218
[ "$N" -gt 0 ] && TOT=$N
OFF=0
while [ "$OFF" -lt "$TOT" ]; do
  nice -n 19 $P $G/gen34_eval.py --arm hybrid --seg exam_deep --args-mode device \
    --ckpt "$CK" --n 100 --chunk-offset "$OFF" \
    --cache $G33/cache/deep_all.bin --side $G33/out/DEEP_RENDER_SIDECAR.jsonl \
    --exam-side $G33/out/DEEP_EXAM_SIDECAR.jsonl \
    --vocab $G33/cache/CCP_VOCAB_V4.json --allow $G33/cache/ccp_allow_v4.bin \
    --pa-enable 0 --plan-summary 0 --slot-cond "$SC" --slot-dev "$SD" \
    --dev "$DEV" --max-new 384 --judge 0 --attn-export 0 \
    --out $G/out/GEN34_EVAL_${TAG}_exam_deep_c$(printf '%04d' $OFF).json \
    --log $G/logs/gen34_eval.log
  echo "CHUNK_DONE $TAG off=$OFF rc=$?"
  OFF=$((OFF+100))
done
echo "EXAM_ALL_DONE $TAG"

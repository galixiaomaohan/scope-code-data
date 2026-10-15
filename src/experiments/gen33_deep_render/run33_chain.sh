#!/bin/bash
# run33_chain.sh -- gen33 post-training automatic chain (exam segment → AUD-1 → judging → report)
# Lean strictly serialized (global count of our jobs <5); the two arms' exam segments run in parallel (different cards: baseline GPU1 / attention GPU2); training has stopped ⇒ no contention
set -u
P="${SCOPE_PYTHON:-python3}"
R="${SCOPE_RESEARCH_ROOT:-.}"
G=$R/scratch/ccp/gen33
L=$G/logs/chain33.log
mkdir -p $G/logs $G/out $G/lean
cd $R || exit 3
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${SCOPE_SUPEROPERATOR:-../device}"
say() { echo "[$(date +%H:%M:%S)] $*" >> $L; }

say "CHAIN START"
# ---------- 1) wait for both arms' training to finish (ckpt + summary) ----------
for i in $(seq 1 480); do
  if [ -f $G/out/DEEP_BASE_SUMMARY.json ] && [ -f $G/out/DEEP_ATTN_SUMMARY.json ]; then
    break
  fi
  sleep 30
done
say "TRAIN_WAIT_DONE base=$( [ -f $G/out/DEEP_BASE_SUMMARY.json ] && echo 1 || echo 0 ) attn=$( [ -f $G/out/DEEP_ATTN_SUMMARY.json ] && echo 1 || echo 0 )"

# ---------- 2) both arms' exam segments (parallel, different cards) ----------
nohup bash $G/run33_exam.sh hybrid $G/ckpt/deep_base_formal_6000.pt cuda:1 base \
  > $G/logs/exam_base.stdout 2>&1 &
nohup bash $G/run33_exam.sh hybrid $G/ckpt/deep_attn_formal_6000.pt cuda:2 attn \
  > $G/logs/exam_attn.stdout 2>&1 &
for i in $(seq 1 360); do
  b=$(grep -c 'EXAM_ALL_DONE base' $G/logs/exam_base.stdout 2>/dev/null || true); b=${b:-0}
  a=$(grep -c 'EXAM_ALL_DONE attn' $G/logs/exam_attn.stdout 2>/dev/null || true); a=${a:-0}
  if [ "$b" -ge 1 ] && [ "$a" -ge 1 ]; then break; fi
  sleep 30
done
sB=$(grep -c 'EXAM_ALL_DONE base' $G/logs/exam_base.stdout 2>/dev/null || true); sA=$(grep -c 'EXAM_ALL_DONE attn' $G/logs/exam_attn.stdout 2>/dev/null || true)
say "EXAM_WAIT_DONE base=${sB:-0} attn=${sA:-0}"

# ---------- 3) naked zero-training base control (n=30 disclosed slice) ----------
nice -n 19 bash $G/run33_exam.sh naked '' cuda:1 naked 30 > $G/logs/exam_naked.stdout 2>&1
say "NAKED_DONE rc=$?"

# ---------- 4) AUD-1 (both arms' curves present) ----------
nice -n 19 $P $G/gen33_audit.py --curve-attn $G/out/DEEP_ATTN_CURVE.jsonl \
  --curve-base $G/out/DEEP_BASE_CURVE.jsonl >> $G/logs/chain33.log 2>&1
say "AUD1_DONE rc=$?"

# ---------- 5) judging (strictly serialized: batch both arms -> literal >=60 -> reverse 12) ----------
# Batch judging: the two arms run **in parallel** (2 Lean jobs < the 5 cap; separate --out paths avoid contention)
nice -n 19 $P $G/gen33_judge.py --mode batch --tags base --chunk 80   --out $G/out/GEN33_JUDGE_batch_base.json >> $G/logs/chain33.log 2>&1 &
PB1=$!
nice -n 19 $P $G/gen33_judge.py --mode batch --tags attn --chunk 80   --out $G/out/GEN33_JUDGE_batch_attn.json >> $G/logs/chain33.log 2>&1 &
PB2=$!
wait $PB1; say "JUDGE_BATCH_BASE_DONE rc=$?"
wait $PB2; say "JUDGE_BATCH_ATTN_DONE rc=$?"
nice -n 19 $P $G/gen33_judge.py --mode literal --tags base,attn --n-literal 60   --out $G/out/GEN33_JUDGE_literal.json >> $G/logs/chain33.log 2>&1
say "JUDGE_LITERAL_DONE rc=$?"
nice -n 19 $P $G/gen33_judge.py --mode reverse --tags base,attn --n-reverse 12   --out $G/out/GEN33_JUDGE_reverse.json >> $G/logs/chain33.log 2>&1
say "JUDGE_REVERSE_DONE rc=$?"

# ---------- 6) report ----------
nice -n 19 $P $G/gen33_report.py --tags base,attn >> $G/logs/chain33.log 2>&1
say "REPORT_DONE rc=$?"
say "CHAIN ALL DONE"

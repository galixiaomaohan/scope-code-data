#!/bin/bash
# gen44_barm_run.sh -- supplementary run, arm-B launcher (items / walk / manifest / summary / judge / all)
#   Discipline: all Lean under `nice -n 19`; unique shard tags (lesson of D-S8F-1); max-lean-jobs=1 per
#          shard => our global Lean count = number of shards (default 4, satisfying "global ours <5");
#          yield when sharing with others.
set -u
P="${SCOPE_PYTHON:-python3}"
R="${SCOPE_RESEARCH_ROOT:-.}"
G44=$R/scratch/ccp/gen44
OUT=$G44/out
LOGS=$G44/logs
cd "$R" || exit 3
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${SCOPE_SUPEROPERATOR:-../device}:$R/scratch/ccp/gen41:$R/scratch/ccp/gen42"
export OMP_NUM_THREADS=4
unset CUDA_VISIBLE_DEVICES
mkdir -p "$LOGS"

NS=${NS:-4}                 # number of judging shards (= cap on our global Lean count)
K=${K:-8}
TIMEOUT=${TIMEOUT:-300}
# MLJ = concurrency gate threshold. Sharing-period note (registered): our_lean_jobs() counts **all**
#   lake/lean processes globally (including the 5-8 of the final-audit slot)
#   => if MLJ is small (1/2), every row times out in gate_wait 900s -> reasons=gate_stop (reading
#   contamination). Hence MLJ is 24 (> the pre-existing global load), while this arm's own
#   concurrency is still limited by workers=1 => our Lean count = number of shards (4, satisfying the
#   <5 red line), with `nice -n 19` throughout.
MLJ=${MLJ:-24}
DEV=${DEV:-cuda:0}

cmd=${1:-help}
case "$cmd" in
  items)
    nice -n 19 $P "$G44/gen44_barm_model.py" --stage items 2>&1 | tail -20
    ;;
  walk)
    echo "== [BARM-2] GPU chain walk dev=$DEV $(date -Is) =="
    nice -n 19 $P "$G44/gen44_barm_model.py" --stage walk --dev "$DEV" \
        > "$LOGS/barm_walk.log" 2>&1
    echo "WALK_RC=$?"; tail -3 "$LOGS/barm_walk.log"
    ;;
  manifest)
    nice -n 19 $P "$G44/gen44_barm_model.py" --stage manifest 2>&1 | tail -10
    ;;
  judge)
    echo "== [BARM-3] sharded judging nshard=$NS K=$K timeout=$TIMEOUT $(date -Is) =="
    for i in $(seq 0 $((NS-1))); do
      out="$OUT/GEN44_BARM_JUDGE_s${i}.json"
      if [ -s "$out" ]; then echo "[skip] s$i already exists"; continue; fi
      nohup nice -n 19 $P "$G44/gen44_batchconfirm.py" --stage run \
        --manifest "$OUT/GEN44_BARM_MANIFEST_JUDGE.jsonl" \
        --K "$K" --workers 1 --max-lean-jobs "$MLJ" --timeout "$TIMEOUT" \
        --nshard "$NS" --shard "$i" --tag "barm_s${i}" --out "$out" \
        > "$LOGS/barm_judge_s${i}.log" 2>&1 &
      echo "[launch] s$i pid=$! tag=barm_s$i"
      sleep 3
    done
    echo "launched $NS shards (nohup); logs $LOGS/barm_judge_s*.log"
    ;;
  summary)
    nice -n 19 $P "$G44/gen44_barm_model.py" --stage summary 2>&1 | tail -5
    ;;
  ps)
    ps -eo pid,ni,etime,comm | grep -E 'lean|python' | grep -v grep | head -20
    ;;
  *)
    echo "usage: $0 {items|walk|manifest|judge|summary|ps}"
    ;;
esac

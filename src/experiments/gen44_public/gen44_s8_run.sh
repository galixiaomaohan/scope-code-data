#!/bin/bash
# gen44_s8_run.sh -- [S8] full-corpus final-test launcher (walk / grid / summary / rejudge)
#   Convention: volume A1 = model chain walk (gen44_eval phase_walk, unchanged) + exam single
#         rendering v3 -> manifest
#         volumes B1-B9 = family exam single rendering (import-only, unchanged) -> manifest
#         judging engine = gen44_batchconfirm.py (d5a02f1d... host value before
#                         the package comment translation; current digest in
#                         MANIFEST_SHA256.txt), K=8, timeout 300
#   Concurrency: exclusive period 5 slots x (workers 1 + max-lean-jobs 5) => ~5 Lean globally
#         (registered exception)
#         when sharing with others: drop the slot count to 2 and add --gate-wait-s 900
#         (yield-over semantics)
#   Discipline: the judging module md5 is self-verified before/after by batchconfirm; long runs go
#         through nohup; shards are resumable (existing shard outputs are skipped automatically)
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

SLOTS=${SLOTS:-5}          # concurrency slots (= 5 in the exclusive period; 2 while sharing)
K=${K:-8}
TIMEOUT=${TIMEOUT:-300}
NS_VAD=${NS_VAD:-8}         # shard count of the A1 device-chain reference arm
NS_VB=${NS_VB:-8}           # shard count of the B volumes

cmd=${1:-help}
case "$cmd" in
  walk)
    dev=${2:-cuda:0}
    echo "== [S8-1] GPU chain walk dev=$dev =="
    nice -n 19 $P "$G44/gen44_exam_bulk.py" --stage walk_va --dev "$dev" \
        --log "$LOGS/gen44_s8_walk.log" > "$LOGS/s8_walk_nohup.log" 2>&1
    echo "WALK_RC=$?"
    tail -3 "$LOGS/s8_walk_nohup.log"
    ;;
  items)
    $P "$G44/gen44_exam_bulk.py" --stage items_va | tail -30
    ;;
  man_va)
    $P "$G44/gen44_exam_bulk.py" --stage manifest_va | tail -40
    ;;
  man_vb)
    $P "$G44/gen44_exam_bulk.py" --stage manifest_vb | tail -40
    ;;
  classify)
    $P "$G44/gen44_exam_bulk.py" --stage classify | tail -60
    ;;
  grid)
    # The three volumes' sharded judging merges into one 5-slot pool (resumable: existing shard
    # outputs are skipped automatically)
    #   VAM = A1 model chain-walk volume (2050 -> 36 renderable) ; VAD = A1 device-chain reference
    #   arm ; VB = B1-B9 family tree volume
    echo "== [S8-2] sharded judging grid (slots=$SLOTS K=$K timeout=$TIMEOUT VAM=1 VAD=$NS_VAD VB=$NS_VB) =="
    run_one () {
      local pref=$1 sh=$2 ns=$3 man=$4
      local out="$OUT/GEN44_S8_MAIN_${pref}_s${sh}.json"
      if [ -s "$out" ]; then echo "[skip] $pref s$sh already exists"; return 0; fi
      echo "[launch] $pref s$sh/$ns $(date -Is)"
      nice -n 19 $P "$G44/gen44_batchconfirm.py" --stage run --manifest "$man" \
        --K "$K" --workers 1 --max-lean-jobs 5 --timeout "$TIMEOUT" \
        --nshard "$ns" --shard "$sh" --tag "s8m_$(echo "$pref" | tr 'A-Z' 'a-z')_s${sh}" --out "$out" \
        > "$LOGS/s8m_${pref}_s${sh}.log" 2>&1
      echo "[done] $pref s$sh rc=$? $(date -Is)"
    }
    for spec in "VAM:1" "VAD:$NS_VAD" "VB:$NS_VB"; do
      pref=${spec%%:*}; ns=${spec##*:}
      case "$pref" in
        VAM) man="$OUT/GEN44_S8_MANIFEST_VA_JUDGE.jsonl" ;;
        VAD) man="$OUT/GEN44_S8_MANIFEST_VA_DEV_JUDGE.jsonl" ;;
        VB)  man="$OUT/GEN44_S8_MANIFEST_VB_JUDGE.jsonl" ;;
      esac
      for i in $(seq 0 $((ns-1))); do
        while [ "$(jobs -rp | wc -l)" -ge "$SLOTS" ]; do sleep 15; done
        run_one "$pref" "$i" "$ns" "$man" &
      done
    done
    wait
    echo "GRID_DONE $(date -Is)"
    ;;
  summary)
    $P "$G44/gen44_exam_bulk.py" --stage summary \
        --vols "GEN44_S8_MAIN_VAM:1,GEN44_S8_MAIN_VAD:$NS_VAD,GEN44_S8_MAIN_VB:$NS_VB" | tail -70
    ;;
  rejudge)
    n=${2:-60}
    nice -n 19 $P "$G44/gen44_exam_bulk.py" --stage rejudge --n "$n" \
        --vols "GEN44_S8_MAIN_VAM:1,GEN44_S8_MAIN_VAD:$NS_VAD,GEN44_S8_MAIN_VB:$NS_VB" \
        > "$LOGS/s8_rejudge.log" 2>&1
    echo "REJUDGE_RC=$?"
    tail -5 "$LOGS/s8_rejudge.log"
    ;;
  status)
    echo "== shard progress =="
    for pref in VAM VAD VB; do
      echo "$pref done: $(ls "$OUT"/GEN44_S8_MAIN_${pref}_s*.json 2>/dev/null | wc -l)"
    done
    echo "== our Lean jobs (exact comm convention) =="
    ps -eo comm | grep -cx 'lean' || true
    ;;
  *)
    echo "usage: $0 {items|walk|man_va|man_vb|classify|grid|summary|rejudge|status}"
    ;;
esac

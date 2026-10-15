#!/bin/bash
# run37_chain_gpu1.sh -- fork of run37_chain.sh (change #104: all three GPUs allowed, gen37 launches immediately on GPU1; the old "wait for GPU2 release" gate is retired)
#   external baseline full chain: gate_wait -> smoke(3) -> infer(218) -> judge -> dual -> report
#   usage: bash run37_chain_gpu1.sh {build|smoke|gate|infer|judge|dual|report|all}
#   discipline: writes only under scratch/ccp/gen37/**; GPU cuda:1 only (change #104, one job per GPU); whole chain under nice -n 19; zero sudo / zero /tmp.
set -u
P="${SCOPE_PYTHON:-python3}"
R="${SCOPE_RESEARCH_ROOT:-.}"
G37=$R/scratch/ccp/gen37
L=$G37/logs
STEP=${1:-all}
MIN_FREE=${MIN_FREE:-11400}
mkdir -p $L $G37/out $G37/lean $G37/cache
cd $R || exit 3
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${SCOPE_SUPEROPERATOR:-../device}"
unset CUDA_VISIBLE_DEVICES
export OMP_NUM_THREADS=4
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_HOME=$G37/cache/hf
export TOKENIZERS_PARALLELISM=false

say() { echo "[$(date '+%F %T')] $*" >> $L/chain.log; }
free2() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 1 | head -1; }

run_build() {
  say "STEP build start"
  nice -n 19 $P $G37/gen37_build.py --tok-check >> $L/build.out 2>&1
  rc=$?; say "STEP build rc=$rc"; [ $rc -ne 0 ] && exit $rc
}

run_gate_wait() {   # GPU1 gate wait: until free>=MIN_FREE (change #104: launch as soon as GPU1 is idle)
  say "GATE_WAIT start need free>=${MIN_FREE}MiB"
  for i in $(seq 1 2880); do
    F=$(free2)
    say "GATE_WAIT free=$F MiB iter=$i"
    if [ -n "${F:-}" ] && [ "$F" -ge "$MIN_FREE" ]; then
      say "GATE_WAIT RELEASED free=$F"; return 0
    fi
    sleep 30
  done
  say "GATE_WAIT TIMEOUT"; exit 22
}

run_smoke() {       # 3-problem smoke run: verify int8 load + generation + metering; stop on failure (do not waste the window)
  say "STEP smoke start (limit=3)"
  nice -n 19 $P $G37/gen37_infer.py --limit 3 --out $G37/out/GEN37_DSRL_smoke \
    --dev cuda:1 --min-free $MIN_FREE >> $L/smoke.out 2>&1
  rc=$?; say "STEP smoke rc=$rc"
  [ $rc -ne 0 ] && { tail -6 $L/smoke.out >> $L/chain.log; exit 23; }
  nice -n 19 $P - <<PYEOF >> $L/chain.log 2>&1
import json
rows=[json.loads(l) for l in open("$G37/out/GEN37_DSRL_smoke_attempts.jsonl") if l.strip()]
ok=[r for r in rows if r["out_tokens"]>0 and len(r.get("text",""))>20]
print("[smoke] rows=%d nonempty=%d out_tokens=%s" % (len(rows), len(ok), [r["out_tokens"] for r in rows]))
print("[smoke] first160=%r" % (rows[0]["text"][:160] if rows else None))
assert len(rows)==3 and len(ok)==3, "smoke failed: empty/short generations"
print("[smoke] PASS")
PYEOF
  rc=$?; say "STEP smoke verify rc=$rc"; [ $rc -ne 0 ] && exit 23
}

run_infer() {
  say "STEP infer start"
  nice -n 19 $P $G37/gen37_infer.py --resume --max-new 1024 --dev cuda:1 --min-free $MIN_FREE \
    >> $L/infer.out 2>&1
  rc=$?; say "STEP infer rc=$rc"; [ $rc -ne 0 ] && exit $rc
}

run_judge() {
  say "STEP judge start"
  nice -n 19 $P $G37/gen37_judge.py --jobs 32 >> $L/judge.out 2>&1
  rc=$?; say "STEP judge rc=$rc"; [ $rc -ne 0 ] && exit $rc
}

run_dual() {
  say "STEP dual start"
  nice -n 19 $P $G37/gen37_dual.py >> $L/dual.out 2>&1
  rc=$?; say "STEP dual rc=$rc"; [ $rc -ne 0 ] && exit $rc
}

run_report() {
  say "STEP report start"
  nice -n 19 $P $G37/gen37_report.py >> $L/report.out 2>&1
  rc=$?; say "STEP report rc=$rc"; return $rc
}

case "$STEP" in
  build)  run_build ;;
  gate)   run_gate_wait ;;
  smoke)  run_gate_wait; run_smoke ;;
  infer)  run_gate_wait; run_infer ;;
  judge)  run_judge ;;
  dual)   run_dual ;;
  report) run_report ;;
  all)
    run_build
    run_gate_wait
    run_smoke          # stop on smoke failure (exit 23)
    run_infer          # full 218 (--resume; smoke problems are not merged in, problem sets stay separate)
    run_judge
    run_dual
    run_report
    say "CHAIN ALL DONE"
    ;;
  *) echo "unknown step: $STEP"; exit 3 ;;
esac
say "STEP $STEP end"

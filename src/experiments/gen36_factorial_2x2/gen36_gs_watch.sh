#!/bin/bash
# gen36_gs_watch.sh -- auto-run the exam section after G-S training finishes (watch mode modeled on gen36_watch.sh; 2026-09-25 04:2x main run)
#   Wait until the a2wt1 curve stays unchanged for 120 s and has >=6000 rows -> attach GPU2 metering -> run evalrun gs 1 (produces GEN36_PLAN_gs.json)
set -u
R="${SCOPE_RESEARCH_ROOT:-.}"
G36=$R/scratch/ccp/gen36
CCP=$R/scratch/ccp
CURVE=$G36/out/CURVE_deep_plan_formal_a2wt1_6000.jsonl
CKPT=$G36/ckpt/deep_plan_formal_a2wt1_6000.pt
L=$G36/logs/gs_watch.log
echo "WATCH start $(date +%T)" >> $L
for i in $(seq 1 90); do
  N1=$(wc -l < "$CURVE" 2>/dev/null || echo 0)
  sleep 120
  N2=$(wc -l < "$CURVE" 2>/dev/null || echo 0)
  echo "iter=$i n=$N2 $(date +%T)" >> $L
  if [ "$N1" = "$N2" ]; then
    if [ "$N2" -ge 6000 ]; then
      echo "GS_TRAIN_SETTLED $(date +%T) n=$N2" >> $L
      sleep 10
      setsid nohup bash $CCP/gpu_meter.sh 2 $G36/out/GPU2_METER_gs_eval.csv "gen36_eva[l]" 30 > /dev/null 2>&1 < /dev/null &
      bash $G36/gen36_evalrun.sh gs 1 "$CKPT" >> $L 2>&1
      echo "GS_EVALRUN_DONE rc=$? $(date +%T)" >> $L
      exit 0
    else
      echo "GS_STALLED_BELOW_6000 n=$N2 -- no auto exam run; await manual reading by the main run" >> $L
      exit 23
    fi
  fi
done
echo "WATCH TIMEOUT" >> $L

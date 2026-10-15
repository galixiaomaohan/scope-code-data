#!/bin/bash
# gen36_watch.sh -- fallback watch chain for the main run (takeover after the agent died / the host hung): wait for the G-SW arm to finish -> auto-run the exam section
cd "${SCOPE_RESEARCH_ROOT:-.}"
G36=scratch/ccp/gen36
C=$G36/out/CURVE_deep_plan_formal_a2wt5_6000.jsonl
# 1) wait for training to finish: curve size unchanged for 120 s (wait up to 3 hours)
for i in $(seq 1 90); do
  S1=$(stat -c %s "$C" 2>/dev/null || echo 0); sleep 120
  S2=$(stat -c %s "$C" 2>/dev/null || echo 0)
  if [ "$S1" = "$S2" ] && [ "$S1" != "0" ]; then break; fi
done
echo "G_SW_TRAIN_SETTLED $(date +%T) size=$S2" >> $G36/logs/watch.log
sleep 60
# 2) locate the a2 ckpt
CK=$(ls $G36/ckpt/*a2wt5*6000*.pt 2>/dev/null | head -1)
if [ -z "$CK" ]; then echo "CKPT_NOT_FOUND $(ls $G36/ckpt/ 2>/dev/null)" >> $G36/logs/watch.log; exit 41; fi
echo "CKPT=$CK" >> $G36/logs/watch.log
# 3) run the G-SW exam section (tag=gsw, shift=1 post-step residual convention)
bash $G36/gen36_evalru[n].sh gsw 1 "$CK" >> $G36/logs/watch.log 2>&1
echo "GEN36_GSW_EVAL_RC=$? $(date +%T)" >> $G36/logs/watch.log

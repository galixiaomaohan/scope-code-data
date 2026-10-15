#!/bin/bash
# run34_after.sh -- one-shot tail chain after training: wait for ckpt -> device-channel
#                  probe -> three-arm exam segment -> judging -> summary
# Usage: (nohup bash run34_after.sh > logs/after.out 2>&1 &)
set -u
P="${SCOPE_PYTHON:-python3}"
R="${SCOPE_RESEARCH_ROOT:-.}"
G=$R/scratch/ccp/gen34
G33=$R/scratch/ccp/gen33
cd $R || exit 3
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${SCOPE_SUPEROPERATOR:-../device}"

echo "== [0] wait for A/B training to finish =="
for i in $(seq 1 240); do
  A=$(grep -c 'slotA_TRAIN_RC=' $G/logs/train_A.out 2>/dev/null || echo 0)
  B=$(grep -c 'slotB_TRAIN_RC=' $G/logs/train_B.out 2>/dev/null || echo 0)
  [ "$A" -ge 1 ] && [ "$B" -ge 1 ] && break
  sleep 60
done
echo "A=$(grep 'slotA_TRAIN_RC=' $G/logs/train_A.out | tail -1)"
echo "B=$(grep 'slotB_TRAIN_RC=' $G/logs/train_B.out | tail -1)"
ls -la $G/ckpt/

echo "== [1] post-training device-channel probe =="
$P $G/gen34_probe.py --ckpt $G/ckpt/deep_slotA_formal_6000.pt --dev cuda:1 \
  --out $G/out/GEN34_PROBE_slotA.json --log $G/logs/gen34.log 2>&1 | tail -3
$P $G/gen34_probe.py --ckpt $G/ckpt/deep_slotB_formal_6000.pt --dev cuda:1 \
  --out $G/out/GEN34_PROBE_slotB.json --log $G/logs/gen34.log 2>&1 | tail -3

echo "== [2] exam segment, three arms (218 + a_held n=20) =="
bash $G/run34_exam.sh A cuda:1 2>&1 | tail -8
bash $G/run34_exam.sh B cuda:2 2>&1 | tail -8
bash $G/run34_exam.sh C cuda:1 2>&1 | tail -8

echo "== [3] judging, all three conventions =="
bash $G/run34_judge.sh all slotA,slotB,base 2>&1 | tail -10

echo "== [3b] reverse verification (12 gold-substrate cases) =="
$P $G/gen34_rev_gold.py --n 12 --log $G/logs/gen34.log 2>&1 | tail -3

echo "== [4] reading summary =="
$P $G/gen34_tables.py --tags slotA,slotB,base --log $G/logs/gen34.log 2>&1 | tail -8

echo "== [5] slot sidecar md5 ledger (cross-check against the design doc) =="
md5sum $G/out/DEEP_SLOT_SIDECAR.jsonl $G/out/DEEP_EXAM_SLOT_SIDECAR.jsonl \
       $G/out/GEN34_SLOTS.json $G/out/GEN34_PAEQ.json $G/out/GEN34_EQUIV.json
md5sum $R/scratch/drct/drct_common.py
echo "AFTER_ALL_DONE"

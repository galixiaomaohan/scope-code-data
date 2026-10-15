#!/bin/bash
# run34_train.sh -- gen34 A/B arm training (slot-conditioned render head; no attention)
#   A arm = slots + device supply (--slot-cond 1 --slot-dev 1)      [main]
#   B arm = slots + LM fills slots itself (--slot-cond 1 --slot-dev 0) [ablation]
# Preconditions: PA-EQ (GEN34_PAEQ.json pass=True) + slot coverage gate
#   (GEN34_SLOTS.json gate.pass)
# GPU0 user red line: structurally refused (gen34_common.gpu_gate raises SystemExit for idx==0)
# Exclusive single-GPU gate: free > 4500MiB
set -u
P="${SCOPE_PYTHON:-python3}"
R="${SCOPE_RESEARCH_ROOT:-.}"
G=$R/scratch/ccp/gen34
G33=$R/scratch/ccp/gen33
ARM=${1:-slotA}
DEV=${2:-cuda:1}
mkdir -p $G/logs $G/ckpt $G/out
cd $R || exit 3
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${SCOPE_SUPEROPERATOR:-../device}"

# ---- pre-training gates ----
$P - <<'EOF' || exit 21
import json, os
G = os.environ.get("SCOPE_RESEARCH_ROOT", ".") + "/scratch/ccp/gen34"
p=json.load(open(G+'/out/GEN34_PAEQ.json'))
s=json.load(open(G+'/out/GEN34_SLOTS.json'))
assert p['pass'], ('PA-EQ not passed', p['pass'])
assert s['gate']['pass'], ('slot coverage gate not passed', s['gate'])
assert not s['gate']['stopline_hit'], ('stop condition hit', s['gate'])
print('[gate] PA-EQ pass + slot coverage corpus=%.4f exam=%.4f'
      % (s['gate']['coverage_corpus'], s['gate']['coverage_exam']))
EOF
[ $? -ne 0 ] && exit 21

case "$ARM" in
  slotA) SD=1 ;;
  slotB) SD=0 ;;
  *) echo "unknown arm $ARM"; exit 23 ;;
esac

nice -n 19 $P $G/gen34_hybrid.py \
  --cache  $G33/cache/deep_all.bin \
  --side   $G33/out/DEEP_RENDER_SIDECAR.jsonl \
  --vocab  $G33/cache/CCP_VOCAB_V4.json \
  --allow  $G33/cache/ccp_allow_v4.bin \
  --init   ckpt/sla_v31f.pt \
  --lat31  cppcuda/cache/lat_tab_v31.bin \
  --steps 6000 --bs 6 --lr 4e-5 --wd 0.01 --clip 1.0 --warmup 50 \
  --lam-bind 1.0 --lam-txt 1.0 --seed 20260923 --body-dtype fp32 \
  --grad-ckpt 1 --micro-token-cap 9000 --det-train 1 \
  --eval-every 50 --eval-rows 48 --snap-every 100 \
  --args-mode bridge --render-only-eval 1 --stopnan 1 \
  --pa-enable 0 --slot-cond 1 --slot-dev "$SD" \
  --slot-side $G/out/DEEP_SLOT_SIDECAR.jsonl --slot-beta 8.0 --lam-slot 3.0 \
  --run-kind formal --tag deep_$ARM --dev "$DEV" \
  --out     $G/ckpt/deep_${ARM}_formal_6000.pt \
  --curve   $G/out/DEEP_${ARM}_CURVE.jsonl \
  --plan-out $G/out/batch_idx_deep_${ARM}.txt \
  --summary $G/out/DEEP_${ARM}_SUMMARY.json \
  --samples $G/out/DEEP_${ARM}_SAMPLES.json \
  --log     $G/logs/gen34_hybrid_${ARM}.log
echo "${ARM}_TRAIN_RC=$?"

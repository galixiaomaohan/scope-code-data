#!/bin/bash
# run35_train.sh -- gen35 pure-planning arm training (change 98: render text loss removed, chain + sensing supervision kept)
#   **Flag-for-flag isomorphic** with the gen34 A arm; sole difference: --lam-txt 0.0 (V35_DESIGN §3 P1)
#   Pre-gates: PA-EQ pass (including the plan_md5-equals-gen34 assertion) + judge artifact frozen + single-GPU exclusive + Lean < 5
#   GPU0 user redline: gen35_common.gpu_gate SystemExits when idx==0; single-GPU exclusive gate free > 4500MiB
set -u
P="${SCOPE_PYTHON:-python3}"
R="${SCOPE_RESEARCH_ROOT:-.}"
G=$R/scratch/ccp/gen35
G33=$R/scratch/ccp/gen33
G34=$R/scratch/ccp/gen34
DEV=${1:-cuda:1}
mkdir -p $G/logs $G/ckpt $G/out
cd $R || exit 3
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${SCOPE_SUPEROPERATOR:-../device}"

# ---- pre-training gates ----
$P - <<'EOF' || exit 21
import json, hashlib, os
R = os.environ.get('SCOPE_RESEARCH_ROOT', '.')
G = os.path.join(R, 'scratch/ccp/gen35')
p = json.load(open(G + '/out/GEN35_PAEQ.json'))
assert p['pass'], ('PA-EQ 未过', p['pass'])
jm = hashlib.md5(open(os.path.join(R, 'scratch/drct/drct_common.py'), 'rb').read()).hexdigest()
assert jm == '7c31b58da9d7228cee31b9be581fe5dd', jm
print('[gate] PA-EQ pass sd_diff=%d logits_eq=%s plan_same_gen34=%s judge=%s'
      % (p['state_dict']['n_diff'], p['logits_first_step']['torch_equal'],
         p['plan_md5_same_as_gen34'], jm[:8]))
EOF
[ $? -ne 0 ] && exit 21

nice -n 19 $P $G/gen35_hybrid.py \
  --cache  $G33/cache/deep_all.bin \
  --side   $G33/out/DEEP_RENDER_SIDECAR.jsonl \
  --vocab  $G33/cache/CCP_VOCAB_V4.json \
  --allow  $G33/cache/ccp_allow_v4.bin \
  --init   ckpt/sla_v31f.pt \
  --lat31  cppcuda/cache/lat_tab_v31.bin \
  --steps 6000 --bs 6 --lr 4e-5 --wd 0.01 --clip 1.0 --warmup 50 \
  --lam-bind 1.0 --lam-txt 0.0 --seed 20260923 --body-dtype fp32 \
  --grad-ckpt 1 --micro-token-cap 9000 --det-train 1 \
  --eval-every 50 --eval-rows 48 --snap-every 100 \
  --args-mode bridge --render-only-eval 1 --stopnan 1 \
  --pa-enable 0 --slot-cond 1 --slot-dev 1 \
  --slot-side $G34/out/DEEP_SLOT_SIDECAR.jsonl --slot-beta 8.0 --lam-slot 3.0 \
  --run-kind formal --tag deep_plan --dev "$DEV" \
  --out     $G/ckpt/deep_plan_formal_6000.pt \
  --curve   $G/out/DEEP_PLAN_CURVE.jsonl \
  --plan-out $G/out/batch_idx_deep_plan.txt \
  --summary $G/out/DEEP_PLAN_SUMMARY.json \
  --samples $G/out/DEEP_PLAN_SAMPLES.json \
  --log     $G/logs/gen35_hybrid_plan.log
echo "PLAN_TRAIN_RC=$?"

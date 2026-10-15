#!/bin/bash
# run33_attn.sh -- gen33 attention-arm training (plan-sequence self-attention P1+P2; --pa-enable 1, --pa-state 1)
# Prerequisite: PA-EQ assertion passed (both arms bitwise-equivalent at start, GEN33_PAEQ.json pass=True) + change-97 frame-consistency gate passed
# **D-7 (registered deviation)**: the P4-lite plan-summary gate is off (--plan-summary 0). Reason = measured divergence:
#   in the first round (--plan-summary 1) ps_delta_max went 647.9 → 133961.1 (step 100) while pa_delta_max only
#   0.017→0.088 ⇒ render-head text loss 968.98 → 63398 (step 160), i.e. P4-lite adds the **shared summary vector**
#   to **every render position**, and the deep-corpus render segment (max 768 tok, mean 470) is far longer than V2F ⇒ the
#   CE gradient sums over summ and amplifies into positive-feedback divergence. The module is **unchanged** (planattn
#   inherited byte for byte); only the CLI closes this gate; divergence evidence = out/EVID_DIVERGE_ATTN_psdev_curve.jsonl.
#   P1/P2 (this generation's objects of study: step-level attention + dynamic operator embedding + state channel) are all
#   kept ⇒ the M2_D pairing remains valid (the difference list narrows from P1+P2+P4-lite to P1+P2).
# GPU0 user redline: structurally refused; single-GPU exclusivity gate free > 4500MiB
set -u
P="${SCOPE_PYTHON:-python3}"
R="${SCOPE_RESEARCH_ROOT:-.}"
G=$R/scratch/ccp/gen33
DEV=${1:-cuda:2}
mkdir -p $G/logs $G/ckpt $G/out
cd $R || exit 3
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${SCOPE_SUPEROPERATOR:-../device}"
nice -n 19 $P $G/gen33_hybrid.py \
  --cache  $G/cache/deep_all.bin \
  --side   $G/out/DEEP_RENDER_SIDECAR.jsonl \
  --vocab  $G/cache/CCP_VOCAB_V4.json \
  --allow  $G/cache/ccp_allow_v4.bin \
  --init   ckpt/sla_v31f.pt \
  --lat31  cppcuda/cache/lat_tab_v31.bin \
  --steps 6000 --bs 6 --lr 4e-5 --wd 0.01 --clip 1.0 --warmup 50 \
  --lam-bind 1.0 --lam-txt 1.0 --seed 20260923 --body-dtype fp32 \
  --grad-ckpt 1 --micro-token-cap 9000 --det-train 1 \
  --eval-every 50 --eval-rows 48 --snap-every 100 \
  --args-mode bridge --render-only-eval 1 --stopnan 1 \
  --pa-enable 1 --pa-state 1 --pa-heads 2 --pa-dhead 64 --plan-summary 0 --pa-seed 20260923 \
  --run-kind formal --tag deep_attn --dev "$DEV" \
  --out     $G/ckpt/deep_attn_formal_6000.pt \
  --curve   $G/out/DEEP_ATTN_CURVE.jsonl \
  --plan-out $G/out/batch_idx_deep_attn.txt \
  --summary $G/out/DEEP_ATTN_SUMMARY.json \
  --samples $G/out/DEEP_ATTN_SAMPLES.json \
  --log     $G/logs/gen33_hybrid_attn.log
echo "ATTN_TRAIN_RC=$?"

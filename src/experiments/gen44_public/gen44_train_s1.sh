#!/bin/bash
# gen44_train_s1.sh -- [S6R/R2] isomorphic retrain with slot_cond=1 (main fix arm)
#   Purpose: reproduce the parent gen42 recipe (--slot-cond 1 + slot sidecar) => test the main
#        suspect for the [S6] 13-row drop, "recipe difference (missing slot head)", against the
#        secondary suspect "joint training on the new-family corpus interferes".
#   Differences from the launch command in GEN44_S6B_HANDOFF §3.3 (registered item by item, only 4):
#     [1] --slot-side -> $G44/cache/GEN44_V7_SLOT_SIDECAR.jsonl (newly built for this slot, full
#         9599-row coverage; the parent recipe uses $G42/cache/g42_slot_sidecar.jsonl, but that file
#         covers only the parent's 3299-row pool, and reusing it would fail the training-side
#         assertion `n_ok == len(train_idx)` -- S6B §3.3 already registers this point)
#     [2] --slot-cond 1  (S6B used 0; this arm restores the parent recipe)
#     [3] new tag/output name g44_v7_formal_s1_a2wt1_6000 (overwrites no existing file: it does not
#         reuse g44_v7_formal_a2wt1_6000.*)
#     [4] --dev cuda:1 (GPU measured idle by nvidia-smi before launch; one job per GPU,
#         constitution 108)
#   Everything else is copied verbatim (cache/side/vocab/allow/init/bs/lr/wd/clip/warmup/lam-*/
#   w-tail/seed/fp32/grad-ckpt/micro-token-cap/det-train/eval-*/snap-every/args-mode/
#   render-only-eval/stopnan/pa-enable/steps).
set -u
R="${SCOPE_RESEARCH_ROOT:-.}"
PY="${SCOPE_PYTHON:-python3}"
G44=$R/scratch/ccp/gen44
G42=$R/scratch/ccp/gen42
DEV=${1:-cuda:1}
TAG=g44_v7_formal_s1_a2wt1_6000
cd "$R" || exit 3
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${SCOPE_SUPEROPERATOR:-../device}:$R/scratch/ccp/gen41:$R/scratch/ccp/gen42"
echo "== [R2-0] pre-flight md5 =="
md5sum "$G44/cache/GEN44_V7_SLOT_SIDECAR.jsonl"
md5sum "$G44/gen44_hybrid.py" "$G44/gen44_eval.py"
md5sum "$G42/ckpt/g42_minif2f_formal_a2wt1_6000.pt"
echo "== [R2-0b] one job per GPU check (target GPU $DEV) =="
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
echo "== [R2-1] slot_cond=1 retrain, 6000 steps =="
nohup env PYTHONDONTWRITEBYTECODE=1 nice -n 19 $PY "$G44/gen44_hybrid.py" \
  --cache "$G44/cache/GEN44_V7_ALL.bin" --side "$G44/cache/GEN44_V7_SIDECAR.jsonl" \
  --vocab "$G44/cache/CCP_VOCAB_V7.json" --allow "$G44/cache/ccp_allow_v7.bin" \
  --slot-side "$G44/cache/GEN44_V7_SLOT_SIDECAR.jsonl" \
  --init "$G42/ckpt/g42_minif2f_formal_a2wt1_6000.pt" \
  --lat31 cppcuda/cache/lat_tab_v31.bin --bs 6 --lr 4e-5 --wd 0.01 --clip 1.0 \
  --warmup 50 --lam-bind 1.0 --lam-txt 0.0 --w-tail 1.0 --lam-slot 3.0 \
  --seed 20260923 --body-dtype fp32 --grad-ckpt 1 --micro-token-cap 9000 --det-train 1 \
  --eval-every 50 --eval-rows 48 --eval-rotate 1 --snap-every 100 --args-mode bridge \
  --render-only-eval 1 --stopnan 1 --pa-enable 0 --slot-cond 1 \
  --steps 6000 --run-kind formal --tag $TAG --dev "$DEV" \
  --out "$G44/ckpt/${TAG}.pt" --curve "$G44/out/CURVE_${TAG}.jsonl" \
  --plan-out "$G44/out/batch_idx_${TAG}.txt" --summary "$G44/out/SUMMARY_${TAG}.json" \
  --log "$G44/logs/gen44_hybrid_${TAG}.log"  > "$G44/logs/s5_${TAG}_stdout.log" 2>&1 &
echo "TRAIN_PID=$!"
echo "== [R2-2] closing criteria: rc=0 and the log shows the final step 6000/6000 and SUMMARY_${TAG}.json exists =="
echo "TRAIN_LAUNCHED"

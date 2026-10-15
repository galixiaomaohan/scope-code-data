# Data README

Column semantics, units, source experiment, and caveats for every dataset in
this package. All raw files are unchanged copies of the experiment outputs;
nothing in `data/raw/` was edited, resampled, or filtered after the runs.

## 1. data/raw/exam/

`GEN33_EXAM.json` (schema `gen33-exam/...`): the 218-problem exam set used by
every generation from gen33 on. `valid` is the admission census (n, chain
length histogram, per-family and per-class counts); the problem rows
themselves are the `sidecar` export `data/raw/results_gen33/DEEP_EXAM_SIDECAR.jsonl`
(also present under results_gen34). `c4` records the source census per family,
`state_gate`/`state_source` the admission gates, `wall_s` the build time.
`C4_EXAM_PREFILTER.json`: the prefilter that removed near-duplicate exam
candidates (n-gram gate with the conservative threshold and the sensitivity
check recorded inside). All pass rates with denominator 218 in the paper use
this exam.

## 2. data/raw/corpora/

`A_CORPUS_V2F.jsonl`: 9,366 training pairs of the formal arm (fields per row:
statement head/conclusion, gold operator chain, rendered frames). One epoch
equals 1,862,149 tokens. `A_CORPUS_V2F_STATS.json`: family and segment
counts. `GEN35_COMPILED_CORPUS.jsonl`: 2,399 compiled rows of the deep-chain
corpus (fields: `name`, `head`, `concl`, `proof_text`, `family`, `byte_eq`,
`n_lines`, `ref_lines`); the compiler gate reproduced all 2,399 proofs
byte-identically. Corpus and exam have zero overlap (four-way identifier and
semantic-twin gates; the gates run again inside the training scripts).

## 3. data/raw/ccp_course/

`CCP_COURSE_{IND,INEQ,NT}.json`: the three course families (1,154 / 973 / 750
problems; fields include `master_seed`, `n_problems`, `n_rows`, `kind_mix`
with the pos/endfail/neg split around 80/11/8-10 per family, `by_family`,
`step_hist`, `op_whitelist`). `CCP_GATE_D_*.json`: the three
floor/pollution/seed gates. `ccp_*_chains.jsonl` and `ccp_*_rows.jsonl`:
chain and row exports per family.

## 4. data/raw/latent/

`dyn_latq_v15.log`: the complete run log of the latent-space carrier
experiment (Table IV/V source). Two different quantities are named T in this
log and must never be merged: `[gate-probe] tmax` is the maximum depth the
gate has opened (reaches 16), while `[dyn_latq] T` is the sampled training
episode depth (mostly 1; three rows at T = 15 and a single episode at T = 16,
step 18100). The reading 0.096 at the tail is the T = 3 training error; the
gate probe at depth 16 reads 0.167. This file is the only surviving copy.

## 5. data/raw/results_gen33/ through results_gen37/

One directory per generation; file names follow the run scripts in
`src/experiments/`. The headline files:

* `results_gen35/GEN35_JUDGE_pure_plan.json`: pure-planning arm, certified
  117/218. `pool` is the arm tag with `pool_n` = 117; `literal` holds the
  literal re-judging of the certified pool (n = 66 within it),
  `reverse` the adversarial control (12/12 rejected),
  `judge_md5_before` the freeze check.
* `results_gen36/GEN36_PLAN_{GW,gs,gsw}.json`: the three factorial arms
  (110/191/172 of 218). Each carries `ckpt_md5`, `walk_table`,
  `compile`, `judge`, `rows_md5`.
* `results_gen36/GEN36_WALK_*.json`: per-problem chain walks with
  per-step traces (`chain.trace[k].p_max` is the quantity behind the
  error-confidence figures).
* `results_gen36/GEN36_ERRANALYSIS.json`: first-error position census
  (101/101 at the last step), confidence split (55 of 101 at p >= 0.9),
  tail-class identity on the exam.
* `results_gen36/GEN36_EXAM_TAILRHO.json`: the tail-class against
  residual-rho identity on the exam (181/181); the corpus side (799/799) is
  `GEN36_STATE_PROBE.json`, field `A_corpus_tailcls_vs_rho`.
* `results_gen36/GEN36_MIGRATION_*.json`: error migration matrices between
  arms (which problems the shift fixed or broke).
* `results_gen36/GPU2_METER_gs_eval.csv` (and the gsw twins): 1 Hz GPU
  meter, columns timestamp / memory.used MiB / utilisation % / power W.
  During the eval passes memory.used plateaus at 833 MiB (18 samples) with a
  peak of 837; the file's floor is 11-14 MiB. The paper's 833 MiB is that
  observed eval-phase plateau. `GEN37_DSRL_resource.json` carries the
  DeepSeek side (peak 8,567 MiB).
* `results_gen37/GEN37_JUDGE_summary.json`: DeepSeek-Prover-V1.5-RL,
  18/218 with rejection reasons (177 sorry, 23 reject).
* `results_gen37/GEN37_COMPARE.json`: the per-problem resource comparison
  (`ours_135m` against `deepseek`: decision steps, seconds, token flows).
* `CURVE_*.jsonl` in each directory: per-step training curves (one row per
  step; the held-out `eval` blocks appear every 50 steps).
* `results_gen33/DEEP_RENDER_SIDECAR.jsonl` and
  `results_gen34/DEEP_{EXAM,SLOT}_SIDECAR.jsonl`: render and slot inputs
  consumed by the training and extraction scripts.

`results_lineage/`: eleven archived report files of the earlier corpus
lineage (gen26/gen30).

## 6. data/raw/server_only/

Pulled read-only from the experiment host; the md5 values that the
generation report registers were verified before copying.

`gen44/`:

* `GEN44_S8_CLASSIFICATION.json`: the funnel. The `classes` block holds
  48,106 in-domain, 84,267 parseable-no-family, 5 unparseable;
  `denominator` 132,378 = 139,826 minus the 7,448 library-intrinsic C6
  damage (`c6`). The library document itself sums to 132,375; the 3-row
  delta is registered inside this file (`doc_value`, `delta_vs_doc`) and
  the paper uses the reproducible 132,378. None of the excluded classes is
  counted as zero.
* `GEN44_S8_LITALL_SUMMARY.json`: the literal re-judging that produces the
  certified 2,132 of 3,536 (60.29%, Wilson 58.67 to 61.89), with the
  per-family B-table (b1 to b9; family-tree arm 672 of 1,525) and the
  bidirectional disagreement census against the first judging pass.
* `GEN44_BARM_SUMMARY.json`: the model-arm re-test on B-family held-out rows
  (281/623 = 45.10%, Wilson 41.24 to 49.03; `by_fam` per family; the
  shard-level provenance with per-shard md5 is inside). The two arms have
  different denominators and are never divided against each other.
* `GEN44_FIX_LWB_GOLD_SUMMARY.json`: gold enumeration of the A1 routing
  volume (1,496 of 2,050 certified).
* `CURVE_g44_v7_formal_s1_a2wt1_6000.jsonl`: the training curve of the final
  configuration (6,000 steps; the direct input of Fig. A1).
* `GEN44_WALK_DEEP218_S1.json`: the 218-suite walk of the final model
  (27 errors, 13 confident; the gen36-series pairing uses the gen36 walks).
* `GEN44_FINAL_REPORT.md`: the generation report; its file table registers
  the md5 of `GEN44_S8_CLASSIFICATION.json`, `GEN44_S8_LITALL_SUMMARY.json`,
  and the report itself. The remaining gen44 files above are byte-copies
  whose digests are not restated in that table; their SHA-256 values are in
  `MANIFEST_SHA256.txt`.

`gen39/T4_DAMAGE.json`: the unit-level damage census behind Law 1 (24 cells:
six corruption levels p (0.50 to 0.99, drawn as "Integer accuracy p" on the
paper figure) by four payload-integer bins; `rate_by_cell` with
Wilson intervals, reconstruction cross-checks). The pooled whole-proof block
(0.558 measured against 0.530 theory, n_pool 2,617, 3,570 draws) and the
24-cell table read by `fig3_law1.R` are transcriptions in
`figures_src/data_cited.json`; that block has no separate raw file in this
package.

`lineage/`: the judge outputs behind the corpus-era readings. `gen26/
FORMAL_A_LITERAL_REJUDGE.json` carries `n_certified_total = 180` (formal
arm; the eleven `FORMAL_JUDGE_hybrid_exam_v1_c*.json` shards sum to 180 over
1,080 judged rows; the gen26 report registers the same 3-row shortfall
against the 1,083-problem suite for the probe batch and leaves it open, and
these formal shards total 1,080 the same way), and `gen30/
GEN30_LITERAL_REJUDGE.json` carries `n_certified_total = 118` (attention
arm, same suite convention). `FORMAL_BASELINE.json`, `FORMAL_SUMMARY.json`,
and the two attention summaries round out the set.

## 7. figures_src/data_cited.json

The arbiter file for the numbers that appear on figures: the funnel counts,
the domain-extension arms, and the 24 damage cells. The `_provenance` block
names the source file of each block (with md5 where the source registers
one); figure scripts read from this file or from the raw outputs it names.
A small set of annotation literals inside the scripts (for example the
"55 / 101" and "12 / 27" counts) is cross-checked against the same raw
files rather than computed at draw time.

## 8. data/processed/figures/

Regenerable exports; run `python run_all.py data` (or the individual scripts
in `figures_src/extract/`) to rebuild them from `data/raw/`. Byte-identity
was verified for every regenerable CSV except the UMAP coordinates noted
below. Two frozen artifacts are inputs rather than outputs of the runnable
scripts and are shipped as-is: `fig_llm_extract.json` (provenance of the
`hidden_frames.npz` capture, md5 of the checkpoint it read) and
`fig_hidden_entropy_frames.csv` (the 1,552-frame entropy table written by
the checkpoint-dependent extractor).

* `fig_err_confidence.csv`: one row per erroneous item (config baseline or
  final; first divergence step `k`, `p_max`, `margin`, predicted and gold
  operator). Counts: baseline 101 errors with 55 at p >= 0.9; final 27 with
  12. The `margin >= 0.9` columns (46 and 7) are completeness registrations,
  not the paper's reading.
* `fig_step_distribution.csv`: 218 per-item rows from the 436-row two-arm
  source `GEN35_CHAINREAD_ROWS.jsonl` (gold-chain arm; the stored-chain arm
  supplies panel (b) of the figure).
* `fig_step_entropy.csv`: per-step mean, standard error, and count of the
  decision entropy over the masked operator distribution actually used for
  decisions (1,552 frames).
* `fig_train_dynamics.csv`: 6,000 per-step rows from the final
  configuration curve; 120 of them carry the held-out eval blocks. The four
  loss columns have different denominators (total/chain/slot per token,
  text only logged); `eval_sel_top1` saturates at 1.00 by step 50 and is
  not a learning curve.
* `fig_v15_ladder_{gate,train,events}.csv`: parsed views of the run log
  (gate ladder, sampled episode depths, unlock and stall events).
* `fig_hidden_trajectory.csv` + `hidden_frames.npz`: the 1,552 decision
  frames (1152-d hidden states, verified to be the exact head inputs,
  max |diff| = 0), per-frame family and step metadata, and the UMAP
  coordinates. The 1-NN family purity 0.999 and the chance level 0.659 are
  properties of the 2-D embedding (umap-learn 0.5.12 with the numpy 2.0.x
  line, seed 20260930, as on the experiment host); the cosine comparison
  (0.607 adjacent against 0.154 random) lives in the full 1152-d space and
  is version-independent. With a newer umap-learn or numpy the embedding
  purity reads lower (0.957 under 0.5.19, and 0.957 under 0.5.12 with numpy
  2.5) while the 1152-d numbers are unchanged; requirements.txt notes the
  versions for exact reproduction, the shipped CSV is the frozen extraction,
  and a rerun writes `fig_hidden_trajectory.regenerated.csv` (with its
  `.regenerated.json` meta twin) beside it instead of replacing it.
* `fig_head_geometry.csv` + `op_index_segments_v7.csv`: the 220-row operator
  head matrix reading (row norms, segment labels, name provenance). The
  negative segmentation result (seven-segment silhouette below zero) is the
  measured reading, not an anomaly.
* `fig_loss_landscape.csv`: the 51 x 51 filtered-normalized loss grid with
  the delta calibration scan inside the JSON.

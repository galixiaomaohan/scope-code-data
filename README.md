# SCOPE code and data package

Supplementary code and data for the manuscript "SCOPE: language models as
policy planners for certified theorem proving" (IEEE TNNLS submission). The
package contains the experiment pipeline, the frozen judging module, the raw
experiment outputs behind every number in the paper, and the complete
generation chain of the data figures.

## 1. Layout and artifacts

```
SCOPE_code_data/
├── README.md                 this file
├── DATA_README.md            per-dataset semantics, sources, caveats
├── requirements.txt          Python dependencies (versions pinned where exact
│                             reproduction depends on them)
├── environment_R.txt         R version and package list for the figure scripts
├── LICENSE                   MIT
├── run_all.py / run_all.sh   local reproduction entry point
├── src/
│   ├── device/               the operator engine (three kernel generations,
│   │                         TCA engine, tests)
│   ├── judge/                frozen Lean judging module and protocol
│   ├── discretization/       problem-domain taxonomy, course builder, gates
│   ├── latent/               trainer of the latent-space carrier experiment
│   └── experiments/
│       ├── gen33_deep_render/       deep direct generation (collapse, 0/218)
│       ├── gen34_slots_gating/      integer slots with device supply (1/218)
│       ├── gen35_plan_compile/      pure planning with the proof compiler
│       │                            (117/218)
│       ├── gen36_factorial_2x2/     state-shift factorial (191/218)
│       ├── gen37_deepseek_baseline/ external DeepSeek-Prover comparison
│       │                            (18/218)
│       ├── gen44_public/            Lean-Workbook funnel and domain
│       │                            extension (server-side generation)
│       ├── gen23_support/, gen30_support/, gen32_support/,
│       │   gen41_support/, gen42_support/, bh_support/,
│       │   cppcuda_tools/           import closure of the experiment tree
├── data/
│   ├── raw/                  original experiment outputs, copied unchanged
│   └── processed/figures/    per-figure CSV/JSON exports (regenerable from
│                             raw; see section 4)
└── figures_src/
    ├── scripts_r/            R/ggplot2 figure scripts (paper track)
    ├── extract/              raw-to-CSV extraction scripts (data track)
    ├── fix_a1_a2.py          A1/A2 strip generator (overwrites the two
    │                         same-name PNGs from the R track by design)
    ├── svg/                  two directly edited SVG figure sources
    └── data_cited.json       the number-arbiter file every figure reads
```

`SCOPE_code_data.zip` and `MANIFEST_SHA256.txt` describe the zip payload
only: every file in the zip is listed with its SHA-256. Regenerating figures
creates `figures_src/output/` and bytecode caches next to the sources; those
are local build artifacts, not package content.

## 2. Environment

Python: see requirements.txt. The experiment host ran Python 3.13 with
torch 2.8.0 (cu128) and transformers 4.57.6 on a single Linux host with
three 12 GB GeForce RTX 40-series GPUs (the per-run records name the card
used: GPU1 is an RTX 4070 SUPER in the gen37 resource file, an RTX 4070 Ti
in the gen37 run report) and Lean 4 available for judging. The local
reproduction path (section 4) needs only the CPU-side packages.

R: R 4.6.1 with the packages listed in environment_R.txt (ggnewscale
included; patchwork was a development build on the host, see the note there).

Lean: the judging module writes proof scripts and checks them with the Lean
toolchain of the experiment host (`lean` on PATH during the judged runs).

Environment variables used by the launch scripts in `src/experiments/`:

| Variable | Meaning | Value with this package |
|---|---|---|
| `SCOPE_RESEARCH_ROOT` | root of the experiment tree (where training caches, checkpoints, and Lean live) | your checkout of the experiment tree; the gen44 scripts resolve `scratch/ccp/...` under it |
| `SCOPE_SUPEROPERATOR` | directory of the operator engine | `<pkg>/src/device` |
| `SCOPE_PYTHON` | interpreter used by the shell launchers | any Python with torch/transformers |

## 3. Quick start

```
pip install -r requirements.txt
python run_all.py --list     # inspect the plan
python run_all.py            # regenerate figure data exports, then figures
```

Outputs land in `data/processed/figures/` and `figures_src/output/`.
`run_all.py` is idempotent: rerunning overwrites the regenerated files and
leaves everything else untouched. The frozen UMAP export is never
overwritten; a rerun writes `fig_hidden_trajectory.regenerated.csv` and
`.regenerated.json` beside it (see DATA_README.md section 8).

## 4. What reproduces locally, and what does not

Runs on a CPU-only machine with this package alone:

| Step | Script | Verified result |
|---|---|---|
| error-step confidence export | `extract/fig_err_confidence.py` | 101 errors, 55 at p >= 0.9 (baseline); 27 errors, 12 (final); byte-identical CSV |
| decision-step distribution export | `extract/fig_step_distribution.py` | mean 6.12 steps; byte-identical CSV |
| training-dynamics export | `extract/fig_train_dynamics.py` | 6000 steps, loss 4.03 to 0.025 nats; byte-identical CSV |
| depth-ladder export | `extract/fig_v15_ladder.py` | gate 1 to 16, 15 unlocks in steps 11000 to 17500; byte-identical CSV |
| hidden-state figure data | `extract/fig_hidden_figs.py` | entropy CSV byte-identical; UMAP coordinates are version-sensitive (DATA_README.md) |
| the 11 manuscript data figures | `scripts_r/*.R` + `fix_a1_a2.py` | figure bundles in `figures_src/output/` |

Figure bundles and their scripts: Fig. 2 `fig_llm_06`, Fig. 3 `fig3_law1`,
Fig. 4 `fig_err_composite`, Fig. 5 `fig_results_grid`, Fig. A1
`fig_llm_01` then `fix_a1_a2.py`, Fig. A2 `fig_llm_04` then `fix_a1_a2.py`,
Fig. A3 `fig_llm_05`, Fig. A4 `fig_llm_08`, Fig. A5 `fig_llm_02`, Fig. A6
`fig_llm_07`, Fig. A7 `fig_llm_03`. The architecture
diagram (Fig. 1) is a drawing and has no generator in this package.
`fix_a1_a2.py` re-renders the A1/A2 strips from the same CSVs and replaces
the two same-name PNGs written by the R scripts.

Needs the original GPU host (checkpoints and caches are not shipped; their
md5 values are registered in the included summary files):

* training runs (`gen3*_hybrid.py`, `src/latent/dyn_train.py`),
* exam walks (`gen3*_eval.py`, `gen44_eval.py`, `gen44_exam_bulk.py`),
* Lean judging (`gen3*_judge.py`, `src/judge/`, `gen44_batchconfirm.py`),
* checkpoint-dependent extractions (`fig_llm_extract.py`,
  `fig_loss_landscape.py`, `fig_head_geometry.py`; run them with
  `SCOPE_RESEARCH_ROOT` pointing at a tree that holds the artifacts).

Device regression test: `PYTHONPATH=src/device python3
src/device/tests/audit_closure.py` (expects 26 assertions, 26 passed).

## 5. Experiment-to-artifact map

| Paper claim | Code | Raw data |
|---|---|---|
| Direct generation collapses on deep chains (0/218; token accuracy 0.9637 against certified 0) | `experiments/gen33_deep_render/` | `data/raw/results_gen33/` |
| Integer slots with device-computed supply break the zero (1/218) | `experiments/gen34_slots_gating/` | `data/raw/results_gen34/` |
| Pure planning with the proof compiler (117/218; compiler ceiling 218/218) | `experiments/gen35_plan_compile/` | `data/raw/results_gen35/`, corpora |
| State shift closes 74 of 101 failures (117 to 191 of 218) | `experiments/gen36_factorial_2x2/` | `data/raw/results_gen36/` (four arms), `GEN36_ERRANALYSIS.json` |
| Terminal-step error structure; rho identity (exam 181/181, corpus 799/799) | `gen36_erranalysis.py` | exam: `GEN36_EXAM_TAILRHO.json`; corpus: `GEN36_STATE_PROBE.json` (`A_corpus_tailcls_vs_rho`) |
| External comparison 18/218, resource ratios | `experiments/gen37_deepseek_baseline/` | `data/raw/results_gen37/` (`GEN37_COMPARE.json`; DeepSeek memory in `GEN37_DSRL_resource.json`, ours in the gen36 `GPU2_METER_*` files) |
| Latent carrier table (unified-protocol arms) | `src/latent/dyn_train.py` | `data/raw/latent/dyn_latq_v15.log` |
| Attention negative result (118 against 180, suite 1,083) | `gen30_support/gen30_planattn.py` | `data/raw/server_only/lineage/` (`gen30/GEN30_LITERAL_REJUDGE.json` 118; `gen26/FORMAL_A_LITERAL_REJUDGE.json` 180) |
| Lean-Workbook funnel 132,378 to 48,106, judged 3,536, certified 2,132 (60.29%, Wilson 58.67 to 61.89) | `experiments/gen44_public/` (`gen44_exam_bulk.py` walks, `gen44_batchconfirm.py` judges) | `data/raw/server_only/gen44/` (`GEN44_S8_CLASSIFICATION.json`, `GEN44_S8_LITALL_SUMMARY.json`, `GEN44_FINAL_REPORT.md`) |
| Domain extension, model arm 281/623 (45.10%) and family-tree arm 672/1,525 (44.07%) | `gen44_barm.py`, `gen44_census_v3.py` | `GEN44_BARM_SUMMARY.json`, `GEN44_S8_LITALL_SUMMARY.json` |
| Law 1 damage cells (24 cells, 1,680 units) | `figures_src/scripts_r/fig3_law1.R` | `data/raw/server_only/gen39/T4_DAMAGE.json`, `figures_src/data_cited.json` |
| Operator vocabulary (194 to 220 rows) | `gen33_deep_render/gen33s_vocab4.py` (194), `gen44_public/gen44_vocab7.py` (220); base 161 registry in `src/device/svokernel*/` and `src/discretization/` | vocab exports in `data/raw/results_gen33/` and the gen44 summaries |

## 6. Reading conventions

* Every pass rate carries its denominator (n of N); arms with different
  denominators are never compared by division (the funnel reading 60.29% has
  denominator 3,536 judged rows, not the full 132,378).
* "Certified" means the frozen Lean judge accepted the proof: `src/judge/
  drct_common.py`, md5 `7c31b58da9d7228cee31b9be581fe5dd`, verified before
  and after every judged batch (the digest is registered in
  `src/judge/DRCT_PROTOCOL.lock.md5`; the module ships byte-identical to it).
* Training arms share the exam set (218 problems, `data/raw/exam/GEN33_EXAM.json`),
  the random seed 20260923, and the batch order (md5
  `89de9e657b1986c6acf3cdf944e721ca`) unless a table states otherwise. The
  `MASTER = 20261005` constants in the gen32-36 common modules record the
  corpus batch, not the training seed.
* Negative results are reported as measured: the attention arm (118 against
  180 on the 1,083-problem suite), the UMAP-segmentation reading in
  Fig. A3, and the OOD walk arm of the funnel are all part of the record.
* Files under `data/raw/` are byte-identical copies of the experiment
  outputs; their internal provenance fields keep the original host paths of
  the experiment tree (they are part of the run records, and the file md5s
  are cross-registered in the shipped reports). Only `src/`, the processed
  exports, and the documentation were rewritten for this package.
* Source comments and docstrings are English. Runtime strings keep the
  original run language where they are data rather than prose: operator
  registry descriptors and exam-item text in `src/device/`, checker error
  details, the frozen judge module (kept byte-identical to its registered
  md5, comments included), and the report bodies that `ccp_docs.py` and the
  makefork benches emit. The generation reports under `data/raw/` are
  Chinese-language run records.

## 7. Not shipped

Model checkpoints (4 x 542 MB for gen35/36 plus the gen44 final model,
md5 values registered in `GEN36_PLAN_*.json` and `GEN44_BARM_SUMMARY.json`),
the DeepSeek-Prover models (public on HuggingFace), the sealed DRCT problem
bank (`drct_problems.json`, only its lock md5 is included), the training
caches, the Lean-Workbook corpus itself, and the gen13-era trainer stack
that the course tools reach only through training-time entry points
(`v32ccp_train`, `cppcuda/tools/v32b_train`, `skel32/v32lib`, and the
`probes/sla_train_*` modules, all resolved from `SCOPE_RESEARCH_ROOT` on the
experiment host). The repeated per-family
builders b2 to b9 of that corpus follow the pattern of the included b1 chain
(`gen44_fam_b1.py`, `gen44_payload_b1.py`) and are available from the
authors on request.

## 8. License

MIT (LICENSE). The DeepSeek-Prover models carry their own HuggingFace
license; SmolLM2 is Apache 2.0. No third-party code is bundled beyond the
standard scientific Python and R stacks.

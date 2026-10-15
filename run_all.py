#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reproduction entry point for the SCOPE code and data package.

Two stages, both local and CPU-only:

  stage data   regenerate the per-figure CSV/JSON exports in
               data/processed/figures/ from the raw outputs in data/raw/
  stage figs   regenerate the manuscript figures with the R scripts into
               figures_src/output/

Usage:
  python run_all.py            # both stages
  python run_all.py data       # stage data only
  python run_all.py figs       # stage figs only
  python run_all.py --list     # show what runs and what is skipped

The three checkpoint-dependent extraction scripts (fig_llm_extract,
fig_loss_landscape, fig_head_geometry) are skipped unless SCOPE_RESEARCH_ROOT
points at a tree that contains the 542 MB training artifacts; their shipped
outputs in data/processed/figures/ are frozen extractions with the checkpoint
md5 registered inside each JSON. Training and judging runs under
src/experiments/ need the original GPU host; README.md section 5 lists what
each script consumes.

Executables are resolved from PATH ("python", "Rscript"): run this script
with the interpreter you want it to reuse, and make sure that interpreter is
also reachable as "python" on PATH.
"""
from __future__ import annotations

import os
import pathlib
import subprocess
import sys

PKG = pathlib.Path(__file__).resolve().parent
EXTRACT = PKG / "figures_src" / "extract"
FIGS = PKG / "figures_src"


def stage_data(ckpt_root):
    rc = 0
    print("[run ] fig_err_confidence")
    rc |= subprocess.run(["python", "fig_err_confidence.py"],
                         cwd=str(EXTRACT), shell=False).returncode
    print("[run ] fig_step_distribution")
    rc |= subprocess.run(["python", "fig_step_distribution.py"],
                         cwd=str(EXTRACT), shell=False).returncode
    print("[run ] fig_train_dynamics")
    rc |= subprocess.run(["python", "fig_train_dynamics.py"],
                         cwd=str(EXTRACT), shell=False).returncode
    print("[run ] fig_v15_ladder")
    rc |= subprocess.run(["python", "fig_v15_ladder.py"],
                         cwd=str(EXTRACT), shell=False).returncode
    print("[run ] fig_hidden_figs")
    rc |= subprocess.run(["python", "fig_hidden_figs.py"],
                         cwd=str(EXTRACT), shell=False).returncode
    if ckpt_root:
        print("[run ] fig_llm_extract")
        rc |= subprocess.run(["python", "fig_llm_extract.py"],
                             cwd=str(EXTRACT), shell=False).returncode
        print("[run ] fig_loss_landscape")
        rc |= subprocess.run(["python", "fig_loss_landscape.py"],
                             cwd=str(EXTRACT), shell=False).returncode
        print("[run ] fig_head_geometry")
        rc |= subprocess.run(["python", "fig_head_geometry.py"],
                             cwd=str(EXTRACT), shell=False).returncode
    else:
        print("[skip] fig_llm_extract.py, fig_loss_landscape.py, "
              "fig_head_geometry.py (need the training checkpoint; set "
              "SCOPE_RESEARCH_ROOT)")
    return rc


def stage_figs():
    rc = 0
    print("[run ] fig3_law1.R")
    rc |= subprocess.run(["Rscript", "scripts_r/fig3_law1.R"],
                         cwd=str(FIGS), shell=False).returncode
    print("[run ] fig_err_composite.R")
    rc |= subprocess.run(["Rscript", "scripts_r/fig_err_composite.R"],
                         cwd=str(FIGS), shell=False).returncode
    print("[run ] fig_results_grid.R")
    rc |= subprocess.run(["Rscript", "scripts_r/fig_results_grid.R"],
                         cwd=str(FIGS), shell=False).returncode
    print("[run ] fig_llm_01_train_dynamics.R")
    rc |= subprocess.run(["Rscript", "scripts_r/fig_llm_01_train_dynamics.R"],
                         cwd=str(FIGS), shell=False).returncode
    print("[run ] fig_llm_02_err_confidence.R")
    rc |= subprocess.run(["Rscript", "scripts_r/fig_llm_02_err_confidence.R"],
                         cwd=str(FIGS), shell=False).returncode
    print("[run ] fig_llm_03_step_distribution.R")
    rc |= subprocess.run(["Rscript", "scripts_r/fig_llm_03_step_distribution.R"],
                         cwd=str(FIGS), shell=False).returncode
    print("[run ] fig_llm_04_v15_ladder.R")
    rc |= subprocess.run(["Rscript", "scripts_r/fig_llm_04_v15_ladder.R"],
                         cwd=str(FIGS), shell=False).returncode
    print("[run ] fig_llm_05_head_geometry.R")
    rc |= subprocess.run(["Rscript", "scripts_r/fig_llm_05_head_geometry.R"],
                         cwd=str(FIGS), shell=False).returncode
    print("[run ] fig_llm_06_hidden_trajectory.R")
    rc |= subprocess.run(["Rscript", "scripts_r/fig_llm_06_hidden_trajectory.R"],
                         cwd=str(FIGS), shell=False).returncode
    print("[run ] fig_llm_07_step_entropy.R")
    rc |= subprocess.run(["Rscript", "scripts_r/fig_llm_07_step_entropy.R"],
                         cwd=str(FIGS), shell=False).returncode
    print("[run ] fig_llm_08_loss_landscape.R")
    rc |= subprocess.run(["Rscript", "scripts_r/fig_llm_08_loss_landscape.R"],
                         cwd=str(FIGS), shell=False).returncode
    print("[run ] fix_a1_a2.py")
    rc |= subprocess.run(["python", "fix_a1_a2.py"],
                         cwd=str(FIGS), shell=False).returncode
    return rc


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    ckpt_root = os.environ.get("SCOPE_RESEARCH_ROOT", "")
    if "--list" in sys.argv:
        print("stage data : fig_err_confidence, fig_step_distribution,")
        print("             fig_train_dynamics, fig_v15_ladder, fig_hidden_figs")
        print("ckpt-gated : fig_llm_extract, fig_loss_landscape, "
              "fig_head_geometry (SCOPE_RESEARCH_ROOT)")
        print("stage figs : 11 R scripts + fix_a1_a2.py")
        return 0
    rc = 0
    try:
        if not args or "data" in args:
            rc |= stage_data(ckpt_root)
        if not args or "figs" in args:
            rc |= stage_figs()
    except FileNotFoundError as exc:
        # a launcher executable is missing from PATH (e.g. python on a
        # python3-only host); report it the documented way instead of a stack
        print("[FAIL] launcher not found: %s" % exc)
        print("        ensure 'python' (or 'Rscript' for stage figs) is on PATH,")
        print("        or run the stage scripts directly with your interpreter")
        return 2
    print("[done] exit %d" % rc)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())

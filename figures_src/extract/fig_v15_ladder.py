#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""fig_v15_ladder -- the depth-curriculum ladder of run dyn_latq_v15 (T = 1 -> 16).

Source  : data/raw/latent/dyn_latq_v15.log  (the only surviving copy, shipped
          with this package)
Outputs : figs/fig_v15_ladder.png
          data/fig_v15_ladder_gate.csv    [gate-probe] readings (the ladder itself)
          data/fig_v15_ladder_train.csv   [dyn_latq] per-log-interval training records
          data/fig_v15_ladder_events.csv  [gate-dep] / [gate-tf] gate transitions

Two DIFFERENT quantities both called T live in this log and are drawn as separate marks,
never merged into one series:
  * [gate-probe] tmax  -- the maximum depth the gate probe has actually unlocked
                          (5 rows at tmax=16, i.e. the ladder tops out at 16).
  * [dyn_latq]  T      -- the depth of the *training episode* at that moment; the sampler
                          keeps drawing shallow episodes throughout
                          (241 rows at T=1, down to 3 rows at T=15).

Layout note: e_fin is given its own panel rather than a twin axis. A twin axis forced the
ladder (0..16) and e_fin (0.06..1.4) into one frame and the two curves tangled; three
stacked panels give each quantity a clean scale.
"""
from __future__ import annotations

import csv
import json
import os
import re
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figcommon as FC
from figcommon import smooth

SRC = FC.raw("latent", "dyn_latq_v15.log")
NAME = "fig_v15_ladder"

RE_TRAIN = re.compile(r"^\[dyn_latq\] it=(\d+) T=(\d+) loss=([\d.]+) e1=([\d.]+) "
                      r"edeep=([\d.]+) efin=([\d.]+) warm=(\w+)")
RE_GATE = re.compile(r"^\[gate-probe\] it=(\d+) tmax=(\d+) tf_p=([\d.]+) "
                     r"e2_tf=([\d.]+) e_fin@tmax=([\d.]+)")
RE_DEP = re.compile(r"^\[gate-dep\] it=(\d+) e_fin@(\d+)=([\d.]+)<([\d.]+) tmax->(\d+)")
RE_STALL = re.compile(r"^\[gate-dep\] STALL it=(\d+) .*tmax=(\d+) e_fin=([\d.]+)")
RE_TF = re.compile(r"^\[gate-tf\] it=(\d+) e2_tf=([\d.]+) tf_p->([\d.]+)")


def parse():
    tr, ga, ev = [], [], []
    for line in open(SRC, "r", encoding="utf-8", errors="replace"):
        line = line.rstrip("\n")
        m = RE_TRAIN.match(line)
        if m:
            tr.append((int(m.group(1)), int(m.group(2)), float(m.group(3)),
                       float(m.group(4)), float(m.group(5)), float(m.group(6)),
                       m.group(7)))
            continue
        m = RE_GATE.match(line)
        if m:
            ga.append((int(m.group(1)), int(m.group(2)), float(m.group(3)),
                       float(m.group(4)), float(m.group(5))))
            continue
        m = RE_DEP.match(line)
        if m:
            ev.append((int(m.group(1)), "DEPTH_UNLOCK", int(m.group(5)), float(m.group(3))))
            continue
        m = RE_STALL.match(line)
        if m:
            ev.append((int(m.group(1)), "STALL", int(m.group(2)), float(m.group(3))))
            continue
        m = RE_TF.match(line)
        if m:
            ev.append((int(m.group(1)), "TF_P", None, float(m.group(3))))
    return tr, ga, ev


def main():
    FC.setup()
    tr, ga, ev = parse()
    src_md5 = FC.md5(SRC)
    assert tr and ga, "log parse produced nothing"

    def wcsv(name, header, rows):
        os.makedirs(FC.DATA, exist_ok=True)
        p = os.path.join(FC.DATA, name + ".csv")
        with open(p, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(header)
            w.writerows(rows)
        print("[csv ] %s  (%d rows)" % (p, len(rows)))
        return p

    p_tr = wcsv(NAME + "_train", ["step", "train_depth_T", "loss", "e1", "edeep",
                                  "efin", "warm"], tr)
    p_ga = wcsv(NAME + "_gate", ["step", "gate_tmax", "tf_p", "e2_tf",
                                 "e_fin_at_tmax"], ga)
    p_ev = wcsv(NAME + "_events", ["step", "event", "value", "reading"], ev)

    it_g = np.array([g[0] for g in ga], float)
    tm = np.array([g[1] for g in ga], float)
    efin_g = np.array([g[4] for g in ga], float)
    it_t = np.array([t[0] for t in tr], float)
    T_t = np.array([t[1] for t in tr], float)
    efin_t = np.array([t[5] for t in tr], float)
    loss_t = np.array([t[2] for t in tr], float)
    depth_rows = sorted((s, v, r) for s, k, v, r in ev if k == "DEPTH_UNLOCK")
    stalls = [s for s, k, _, _ in ev if k == "STALL"]

    # ---------------------------------------------------------- figure ----
    fig = plt.figure(figsize=(7.2, 5.9))
    gs = fig.add_gridspec(3, 1, height_ratios=[2.15, 0.95, 0.62], hspace=0.30)
    ax = fig.add_subplot(gs[0])
    axe = fig.add_subplot(gs[1], sharex=ax)
    axd = fig.add_subplot(gs[2], sharex=ax)

    jit = np.random.RandomState(0).uniform(-0.22, 0.22, T_t.size)
    ax.scatter(it_t, T_t + jit, s=3.0, color=FC.LGREY, lw=0, zorder=1)
    ax.annotate("sampled training-episode depth $T$  (light grey)",
                xy=(0.015, 0.93), xycoords="axes fraction", fontsize=8, color=FC.GREY)
    ax.step(it_g, tm, where="post", color=FC.BLUE, lw=2.0, zorder=4)
    ax.plot(it_g, tm, ls="none", marker="o", ms=3.4, mfc="white", mec=FC.BLUE,
            mew=1.1, zorder=5)
    last = -10 ** 9
    for s, v, r in depth_rows:
        ax.axvline(s, color=FC.BLUE, lw=0.6, ls=":", alpha=0.55, zorder=2)
        if s - last >= 1800:
            ax.annotate("$T\\!\\to\\!%d$" % v, xy=(s, v), xytext=(4, 9),
                        textcoords="offset points", fontsize=8, color=FC.BLUE,
                        fontweight="bold", zorder=7)
            last = s
    for s in stalls:
        ax.axvline(s, color=FC.RED, lw=0.7, ls=":", alpha=0.5, zorder=2)
    FC.endlabel(ax, it_g[-1], float(tm[-1]), "gate depth  $t_{max}$  (solid)", FC.BLUE,
                dx=6, dy=2)
    ax.set_ylim(0.2, 18.6)
    ax.set_yticks([1, 2, 4, 6, 8, 10, 12, 14, 16])
    ax.set_ylabel("Depth  $T$")
    ax.set_title("(a) Depth curriculum: the gate unlocks climb 1 $\\to$ 16")
    ax.annotate("%d unlock events, all between step 11 000 and 17 500\n"
                "dotted verticals: blue = unlock, red = gate stall ($\\times$%d)"
                % (len(depth_rows), len(stalls)),
                xy=(0.035, 0.55), xycoords="axes fraction", fontsize=8,
                color=FC.BLACK, fontweight="bold",
                bbox=dict(fc="white", ec="none", alpha=0.8, pad=1.4))

    # (b) final-step error, its own panel
    axe.plot(it_g, efin_g, color=FC.GREEN, lw=1.4, marker="o", ms=2.8, mfc="white",
             mec=FC.GREEN, mew=0.9)
    axe.plot(it_t, smooth(efin_t, 21), color=FC.ORANGE, lw=1.2)
    axe.set_yscale("log")
    axe.set_ylim(0.055, 1.9)
    axe.set_ylabel("$e_{fin}$  (log)")
    FC.endlabel(axe, it_g[-1], float(efin_g[-1]), "probe  $e_{fin}$ = %.3f" % efin_g[-1],
                FC.GREEN, dx=6, dy=-3)
    axe.annotate("training $e_{fin}$ (21-step mean)", xy=(0.34, 0.13),
                 xycoords="axes fraction", fontsize=8, color=FC.ORANGE,
                 fontweight="bold")
    axe.set_title("(b) Final-step error  (log scale)", fontsize=9)

    # (c) sampled depth
    axd.scatter(it_t, T_t, s=4.0, color=FC.ORANGE, alpha=0.7, lw=0)
    axd.set_ylim(0.2, 17.0)
    axd.set_yticks([1, 5, 9, 13])
    axd.set_xlim(-350, 19800)
    axd.set_xlabel("Training step  $it$")
    axd.set_ylabel("Sampled $T$", fontsize=8)
    axd.annotate("the sampler keeps drawing shallow episodes\n"
                 "(241 rows at $T$=1) even after depth 16 unlocks",
                 xy=(0.30, 0.44), xycoords="axes fraction",
                 fontsize=8, color=FC.GREY)

    fig.tight_layout()
    fig_path = FC.savefig(fig, NAME)

    meta = {
        "figure": NAME,
        "what": ("Depth-curriculum ladder of run dyn_latq_v15, 18 800 steps. "
                 "(a) the gate-probe depth t_max as a step function (dotted verticals = "
                 "gate unlocks, red = gate stalls) with the sampled training-episode depth "
                 "as a light backdrop; (b) probe e_fin and the 21-step mean of the training "
                 "e_fin, log scale; (c) the sampled training depth T over time."),
        "one_line_reading": ("The gate unlocks depth monotonically 1 -> 16 over 15 events "
                             "ending at step 17 500, and the probe error falls to 0.167 "
                             "while the training step reaches e_fin = 0.096; the sampler "
                             "never stops drawing shallow episodes, so this curve measures "
                             "the gate trajectory, not the training mixture."),
        "source": [{"path": SRC, "md5": src_md5,
                    "origin_note": ("the only surviving copy of this run log is the "
                                    "workstation copy shipped in this package "
                                    "(data/raw/latent/)"),
                    "role": "run log"}],
        "outputs": {"png": fig_path, "gate_csv": p_ga, "train_csv": p_tr,
                    "events_csv": p_ev},
        "method": ("line-type regexes over the raw log; [gate-probe] gives the ladder, "
                   "[dyn_latq] gives training episodes, [gate-dep]/[gate-tf] give gate "
                   "transitions. The two depth quantities are separate marks."),
        "readings": {
            "n_train_rows": len(tr), "n_gate_rows": len(ga), "n_events": len(ev),
            "train_depth_histogram": {str(k): int(v) for k, v in
                                      zip(*np.unique(T_t, return_counts=True))},
            "gate_tmax_max": int(tm.max()),
            "depth16_unlock_step": min([s for s, v, _ in depth_rows if v == 16] or [-1]),
            "first_gate_row_at_tmax16": int(it_g[int(np.argmax(tm))]),
            "n_rows_at_tmax": int((tm == tm.max()).sum()),
            "final_train_row": {"step": int(it_t[-1]), "T": int(T_t[-1]),
                                "loss": float(loss_t[-1]), "efin": float(efin_t[-1])},
            "final_gate_row": {"step": int(it_g[-1]), "tmax": int(tm[-1]),
                               "tf_p": float(ga[-1][2]), "efin": float(efin_g[-1])},
            "depth_unlocks": [{"step": s, "depth": v, "efin": r} for s, v, r in depth_rows],
            "n_stalls": len(stalls), "last_step": int(it_t[-1]),
        },
        "caveats": [
            "dyn_latq_v15.log is NOT on the server; docs/DEVIATIONS.md:425 states it was "
            "kept, but the file no longer exists there. Only the workstation copy survives "
            "(this figure's source).",
            "'T=16' appears only in the gate-probe line; the deepest *training* episode in "
            "the log is T=15 (3 rows). The run was stopped at step 18 800 before its "
            "32 000-step schedule finished.",
            "the ladder's own reading is e_fin at the gate (0.167 at step 18 750); the "
            "training-step reading e_fin = 0.096 belongs to a T=3 episode and is not the "
            "depth-16 result. Do not quote 0.096 as the depth-16 error.",
            "deviates from the brief's 'twin axis' suggestion: e_fin got its own panel "
            "because a shared frame made the ladder and the error curves unreadable.",
        ],
    }
    FC.save_meta(NAME, meta)
    print(json.dumps({k: v for k, v in meta["readings"].items()
                      if k not in ("depth_unlocks", "train_depth_histogram")}, indent=1))
    print("depth unlocks:", [(s, v) for s, v, _ in depth_rows])


if __name__ == "__main__":
    main()

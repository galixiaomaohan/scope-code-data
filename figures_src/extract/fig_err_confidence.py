#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""fig_err_confidence -- confidence of the first erroneous decision step, two configurations.

Sources (read-only):
  baseline : scratch/ccp/gen35/out/GEN35_EVAL_PLAN_WALK.json   ckpt deep_plan_formal_6000
  final    : scratch/ccp/gen36/out/GEN36_WALK_gs.json          ckpt deep_plan_formal_a2wt1_6000

Both walks come from the SAME gen36 factorial series (same exam, same seed, same batch
order), which is exactly the pair Section 8 of the manuscript quotes: baseline 117/218
with 101 errors of which 55 are high-confidence, final (G-S) 191/218 with 27 errors of
which 12 are high-confidence.  Nothing is re-estimated: the per-error probability is
read verbatim from chain.trace[k].p_max at the first divergent decision step k, the
same rule used by gen36_erranalysis.py:104-111.

The section is self-consistent under a single rule (p_max >= 0.9).  An earlier
analysis note claimed the "12" should be "13"; that was wrong and is withdrawn -- see
meta["caveats"] for the full four-arm table and for the separate gen44-s1 reading.
"""
from __future__ import annotations

import csv
import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figcommon as FC

NAME = "fig_err_confidence"
CFG = [
    ("baseline", "117 / 218 certified", FC.BLUE,
     FC.raw("gen35", "GEN35_EVAL_PLAN_WALK.json")),
    ("final", "191 / 218 certified", FC.ORANGE,
     FC.raw("gen36", "GEN36_WALK_gs.json")),
]

# the rest of the gen36 factorial series + the gen44 re-test, for the registered table
ARMS = [
    ("G-W  (w-tail only)", FC.raw("gen36", "GEN36_WALK_GW.json")),
    ("G-SW (shift+w-tail)", FC.raw("gen36", "GEN36_WALK_gsw.json")),
    ("G-S  (shift only) = final", FC.raw("gen36", "GEN36_WALK_gs.json")),
    ("g44_v7 s1 re-test", FC.raw("gen44", "GEN44_WALK_DEEP218_S1.json")),
]


def first_div(walked, gold):
    for i in range(min(len(walked), len(gold))):
        if walked[i] != gold[i]:
            return i
    if len(walked) != len(gold):
        return min(len(walked), len(gold))
    return None


def extract(path):
    d = json.load(open(path, "r", encoding="utf-8"))
    out, n = [], len(d["rows"])
    for r in d["rows"]:
        ch = r["chain"]
        if ch.get("exact_hit") and ch.get("first_hit"):
            continue
        fd = first_div(ch.get("walked") or [], ch.get("gold") or [])
        tr = [t for t in (ch.get("trace") or []) if t.get("k") == fd]
        t = tr[0] if tr else {}
        out.append({
            "ri": r["ri"], "name": r["name"], "family": r.get("family"),
            "first_div": fd, "p_max": t.get("p_max"), "margin": t.get("margin"),
            "pred": t.get("pred"), "gold": t.get("gold"),
            "gold_in_space": t.get("gold_in_space"),
        })
    return d, out, n


def ecdf(v):
    v = np.sort(np.asarray(v, float))
    return v, np.arange(1, v.size + 1) / float(v.size)


def main():
    FC.setup()
    res, csv_rows, sources = {}, [], []
    for tag, sub, col, path in CFG:
        d, errs, n = extract(path)
        src_md5 = FC.md5(path)
        sources.append({"path": path, "md5": src_md5, "n_items": n,
                        "ckpt": d.get("ckpt"), "ckpt_md5": d.get("ckpt_md5"),
                        "role": tag})
        pm = [e["p_max"] for e in errs if e["p_max"] is not None]
        mg = [e["margin"] for e in errs if e["margin"] is not None]
        res[tag] = {
            "label": sub, "colour": col, "n_items": n, "n_err": len(errs),
            "rate": 1.0 - len(errs) / float(n),
            "p_max": np.array(pm), "margin": np.array(mg),
            "n_hi_p": int(sum(1 for x in pm if x >= 0.9)),
            "n_hi_m": int(sum(1 for x in mg if x >= 0.9)),
            "n_hi_p_baseline_rule": int(sum(1 for x in pm if x > 0.9)),
        }
        for i, e in enumerate(errs):
            csv_rows.append([tag, e["ri"], e["name"], e["family"], e["first_div"],
                             e["p_max"], e["margin"], e["pred"], e["gold"],
                             e["gold_in_space"]])

    # ------------------------------------------------------------- CSV ----
    os.makedirs(FC.DATA, exist_ok=True)
    csv_path = os.path.join(FC.DATA, NAME + ".csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["config", "ri", "name", "family", "first_div_step",
                    "p_max", "margin", "pred_op", "gold_op", "gold_in_space"])
        w.writerows(csv_rows)
    print("[csv ] %s" % csv_path)

    # ---------------------------------------------------------- figure ----
    fig = plt.figure(figsize=(7.2, 5.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.30, 1.0], hspace=0.55, wspace=0.22)
    axE = fig.add_subplot(gs[0, :])
    axB = fig.add_subplot(gs[1, 0])
    axF = fig.add_subplot(gs[1, 1], sharex=axB)

    # (a) ECDF, shape comparison
    axE.axvspan(0.9, 1.02, color=FC.RED, alpha=0.07, lw=0)
    axE.axvline(0.9, color=FC.GREY, lw=0.9, ls="--")
    axE.annotate("high-confidence zone\n$p_{max}\\geq0.9$", xy=(0.9, 0.06), xytext=(3, 0),
                 textcoords="offset points", fontsize=8, color=FC.GREY, va="bottom")
    yq = {}
    for tag, r in res.items():
        x, y = ecdf(r["p_max"])
        axE.step(x, y, where="post", color=r["colour"], lw=1.9)
        yq[tag] = float(np.interp(0.9, x, y))
        axE.plot([0.9], [yq[tag]], marker="o", ms=5, mfc="white", mew=1.4,
                 color=r["colour"], zorder=5)
    axE.set_xlim(0.30, 1.025)
    axE.set_ylim(0, 1.06)
    axE.set_xlabel("Top-1 probability at the first erroneous decision step  $p_{max}$")
    axE.set_ylabel("Cumulative fraction of errors")
    axE.set_title("(a) Error confidence profile: the same shape, a quarter of the mass")
    FC.endlabel(axE, 0.905, yq["baseline"], "baseline: %d / %d"
                % (res["baseline"]["n_hi_p"], res["baseline"]["n_err"]),
                FC.BLUE, dx=8, dy=10)
    FC.endlabel(axE, 0.905, yq["final"], "final: %d / %d"
                % (res["final"]["n_hi_p"], res["final"]["n_err"]),
                FC.ORANGE, dx=8, dy=-14)
    _b, _f = res["baseline"]["n_hi_p"], res["final"]["n_hi_p"]
    axE.annotate("confidently-wrong decisions\n%d $\\rightarrow$ %d  ($-$%.0f%%)"
                 % (_b, _f, 100.0 * (1 - _f / float(_b))),
                 xy=(0.36, 0.80), fontsize=8, color=FC.BLACK, fontweight="bold")

    # (b, c) absolute-count histograms
    bins = np.arange(0.0, 1.001, 0.05)
    for ax, (tag, r) in zip((axB, axF), res.items()):
        ax.axvspan(0.9, 1.0, color=FC.RED, alpha=0.07, lw=0)
        ax.hist(r["p_max"], bins=bins, color=r["colour"], edgecolor="white", lw=0.5)
        ax.plot(r["p_max"], np.full(r["p_max"].size, -0.6), "|", color=r["colour"],
                ms=6, mew=0.8, clip_on=False)
        ax.set_xlim(0.35, 1.02)
        ax.set_ylim(-1.6, None)
        ax.axhline(0, color="#333333", lw=0.8)
        ax.set_xlabel("$p_{max}$")
        ax.set_title("(%s) %s configuration" % ("b" if tag == "baseline" else "c", tag), fontsize=9)
        ax.annotate("%d errors\n%d with $p_{max}\\geq0.9$\n(%d with margin $\\geq0.9$)\n"
                    "n = %d items, %s"
                    % (r["n_err"], r["n_hi_p"], r["n_hi_m"], r["n_items"], r["label"]),
                    xy=(0.035, 0.97), xycoords="axes fraction", fontsize=8,
                    color=r["colour"], fontweight="bold", va="top",
                    bbox=dict(fc="white", ec="none", alpha=0.75, pad=1.2))
    axB.set_ylabel("Erroneous items (count)")

    fig_path = FC.savefig(fig, NAME)

    # ---- the rest of the factorial series, registered (not drawn) ----
    arm_tbl = {}
    for label, p in ARMS:
        if not os.path.exists(p):
            arm_tbl[label] = None
            continue
        _d, _errs, _n = extract(p)
        _pm = [e["p_max"] for e in _errs if e["p_max"] is not None]
        arm_tbl[label] = {"src": p, "n_errors": len(_errs),
                          "certified": _n - len(_errs),
                          "n_p_ge_0.9": int(sum(1 for x in _pm if x >= 0.9)),
                          "md5": FC.md5(p)}

    meta = {
        "figure": NAME,
        "what": ("Confidence of the first erroneous decision step for the baseline "
                 "(117/218) and the final G-S configuration (191/218), on the same 218-item "
                 "deep-chain exam, both from the gen36 factorial series. (a) ECDF of the "
                 "top-1 operator probability at the first divergent step; (b,c) "
                 "absolute-count histograms with a rug of the raw values."),
        "one_line_reading": ("The final configuration does not make its remaining mistakes "
                             "less confidently -- the ECDF shape is unchanged -- it makes "
                             "far fewer of them: confidently-wrong decisions drop from 55 "
                             "to 12."),
        "source": sources,
        "outputs": {"png": fig_path, "csv": csv_path},
        "method": ("p_max is read verbatim from chain.trace[k].p_max of the two persisted "
                   "walk files at the first divergent decision step k (identical rule to "
                   "gen36_erranalysis.py); nothing is re-estimated."),
        "readings": {
            "baseline": {"certified": "117/218", "n_err": res["baseline"]["n_err"],
                         "n_p_ge_0.9": res["baseline"]["n_hi_p"],
                         "n_margin_ge_0.9": res["baseline"]["n_hi_m"],
                         "frac_p_ge_0.9": round(res["baseline"]["n_hi_p"] / 101.0, 4)},
            "final": {"certified": "191/218", "n_err": res["final"]["n_err"],
                      "n_p_ge_0.9": res["final"]["n_hi_p"],
                      "n_margin_ge_0.9": res["final"]["n_hi_m"],
                      "frac_p_ge_0.9": round(res["final"]["n_hi_p"] / 27.0, 4)},
            "confidently_wrong_reduction": "%d -> %d (%.0f%% fewer)" % (
                res["baseline"]["n_hi_p"], res["final"]["n_hi_p"],
                100.0 * (1 - res["final"]["n_hi_p"] / float(res["baseline"]["n_hi_p"]))),
            "gen36_factorial_series_arms": arm_tbl,
        },
        "caveats": [
            "CORRECTION (an earlier analysis note suggested changing the section's 12 to "
            "13 -- that suggestion is WITHDRAWN). The section's 12 comes from "
            "gen36/out/GEN36_WALK_gs.json, which is the SAME series as its baseline; under "
            "the single rule p_max>=0.9 that file gives exactly 12/27. Section 8 is "
            "self-consistent and needs no change. The earlier 13/27 came from pairing the "
            "gen35 baseline with a DIFFERENT checkpoint family (gen44 v7 s1, 220-op head).",
            "Both readings coexist and are registered above: p_max>=0.9 gives 12/27 on the "
            "gen36 G-S walk and 13/27 on the gen44 s1 walk (27 errors in both). The single "
            "row that moves is ri=171 (p_max=0.9449, margin=0.8900). The four factorial arms "
            "reproduce Section 8's table exactly: errors 101/108/46/27 -> certified "
            "117/110/172/191.",
            "margin>=0.9 is NOT the section's rule; it is reported only for completeness "
            "(baseline 46/101, G-S 7/27). Do not mix the two.",
            "This figure uses the gen36 in-series pair, so it is now directly citable by "
            "Section 8. The gen44 s1 re-test is the subject of fig_train_dynamics and "
            "fig_head_geometry instead.",
            "The baseline walk is a different checkpoint (deep_plan_formal_6000) than the "
            "final one (deep_plan_formal_a2wt1_6000), but they belong to the same factorial "
            "series: same exam rows, same seed 20260923, same batch order md5.",
            "'error' is defined as NOT (exact_hit and first_hit) on the operator chain, "
            "which reproduces the 101 / 27 counts exactly; it is a strict superset view of "
            "Lean certification, but on these files the two error sets coincide.",
            "The ECDF shapes are almost indistinguishable; the effect in this comparison is "
            "in the COUNT of confidently-wrong decisions (55 -> 12), not in their "
            "probability distribution. A caption claiming better-calibrated errors would "
            "overreach.",
        ],
    }
    FC.save_meta(NAME, meta)
    for tag, r in res.items():
        print("%-9s n_err=%3d  p>=0.9: %2d  margin>=0.9: %2d" %
              (tag, r["n_err"], r["n_hi_p"], r["n_hi_m"]))


if __name__ == "__main__":
    main()

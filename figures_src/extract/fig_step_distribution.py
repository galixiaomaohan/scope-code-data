#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""fig_step_distribution -- decision-chain length of the 218-item deep exam, by problem family.

Source  : scratch/ccp/gen35/out/GEN35_CHAINREAD_ROWS.jsonl (436 rows = 218 items x 2 chain arms)
Outputs : figs/fig_step_distribution.png
          data/fig_step_distribution.csv    per-item table (218 rows)
          data/fig_step_distribution.json   provenance + readings

The file holds two arms per item: arm_tag='gold_chain' (chain_src='gold', the reference
derivation) and arm_tag='stored_chain' (chain_src='gen34_slotA', the chain a model actually
produced).  gold_ops is identical in the two arms; it is the gold_ops length that defines the
number of decision steps.  The certified rate annotated on the right is taken from the
stored_chain arm, because the gold arm is certified 218/218 by construction.
"""
from __future__ import annotations

import collections
import csv
import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figcommon as FC

SRC = FC.raw("gen35", "GEN35_CHAINREAD_ROWS.jsonl")
NAME = "fig_step_distribution"


def main():
    FC.setup()
    rows = [json.loads(l) for l in open(SRC, "r", encoding="utf-8")]
    src_md5 = FC.md5(SRC)

    gold = [r for r in rows if r["arm_tag"] == "gold_chain"]
    stored = {r["ri"]: r for r in rows if r["arm_tag"] == "stored_chain"}
    gold.sort(key=lambda r: r["ri"])
    n = len(gold)

    steps = np.array([len(r["gold_ops"]) for r in gold], int)
    fams = [r["family"] for r in gold]
    mean_steps = float(steps.mean())
    assert steps.mean().round(4) == 6.1193, steps.mean()

    # per-family certified rate from the model arm (stored_chain)
    fam_cert = collections.defaultdict(lambda: [0, 0])
    for r in gold:
        c = stored.get(r["ri"], {})
        fam_cert[r["family"]][0] += int(bool(c.get("certified")))
        fam_cert[r["family"]][1] += 1
    fam_n = collections.Counter(fams)

    # ------------------------------------------------------------- CSV ----
    os.makedirs(FC.DATA, exist_ok=True)
    csv_path = os.path.join(FC.DATA, NAME + ".csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["ri", "name", "family", "cls", "cls_code", "n_lines",
                    "n_decision_steps", "gold_ops", "certified_stored_arm",
                    "exact_hit_stored_arm", "n_mismatch_stored_arm"])
        for r in gold:
            c = stored.get(r["ri"], {})
            w.writerow([r["ri"], r["name"], r["family"], r["cls"], r["cls_code"],
                        r["n_lines"], len(r["gold_ops"]), "|".join(r["gold_ops"]),
                        c.get("certified"), c.get("exact_hit"), c.get("n_mismatch")])
    print("[csv ] %s" % csv_path)

    # ---------------------------------------------------------- figure ----
    fig, (ax, axr) = plt.subplots(1, 2, figsize=(7.2, 3.1),
                                  gridspec_kw={"width_ratios": [1.55, 1.0], "wspace": 0.55})

    xs = np.arange(5, 10)
    bottom = np.zeros(xs.size)
    for fam in FC.FAM_ORDER:
        cnt = np.array([sum(1 for s, f in zip(steps, fams) if s == x and f == fam)
                        for x in xs], float)
        ax.bar(xs, cnt, bottom=bottom, color=FC.FAM_COLOR[fam], width=0.68,
               edgecolor="white", lw=0.6, label=FC.FAM_EN[fam])
        bottom += cnt
    for i, tot in enumerate(bottom):
        ax.annotate("%d" % tot, xy=(xs[i], tot), xytext=(0, 3), textcoords="offset points",
                    ha="center", fontsize=8, color=FC.BLACK)
    ax.legend(loc="upper left", frameon=False, fontsize=8, handlelength=1.5,
              labelspacing=0.3, borderaxespad=0.5, bbox_to_anchor=(0.005, 0.995))
    ax.axvline(mean_steps, color=FC.RED, lw=1.4, ls="--")
    ax.annotate("mean = %.2f steps" % mean_steps, xy=(mean_steps, 150), xytext=(6, 0),
                textcoords="offset points", fontsize=8, color=FC.RED, fontweight="bold")
    ax.set_xlim(4.4, 9.6)
    ax.set_ylim(0, 205)
    ax.set_xticks(xs)
    ax.set_xlabel("Decision steps per item  (length of the gold operator chain)")
    ax.set_ylabel("Items (count)")
    ax.set_title("(a) Chain length over the 218-item deep exam")
    ax.annotate("gold arm: certified 218 / 218\nthe 9-step tail is lemma tower only",
                xy=(0.035, 0.60), xycoords="axes fraction", fontsize=8, color=FC.GREY)

    # (b) per-family certification rate, model-produced (stored) arm
    ys = np.arange(len(FC.FAM_ORDER))[::-1]
    for y, fam in zip(ys, FC.FAM_ORDER):
        c, t = fam_cert[fam]
        r = c / float(t)
        axr.barh(y, r, color=FC.FAM_COLOR[fam], height=0.55)
        axr.annotate("%d / %d = %.3f" % (c, t, r), xy=(r + 0.03, y), va="center",
                     ha="left", fontsize=8, color=FC.BLACK, fontweight="bold")
    axr.set_yticks(ys)
    axr.set_yticklabels(["Nested (%d)" % fam_n[FC.FAM_ORDER[0]],
                         "Tower (%d)" % fam_n[FC.FAM_ORDER[1]],
                         "SoS (%d)" % fam_n[FC.FAM_ORDER[2]]], fontsize=8)
    axr.set_xlim(0, 1.42)
    axr.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    axr.set_xlabel("Certified rate (model-produced chain)")
    axr.set_title("(b) Certification by family")

    fig.tight_layout()
    fig_path = FC.savefig(fig, NAME)

    meta = {
        "figure": NAME,
        "what": ("Distribution of decision-chain length over the 218-item deep exam, "
                 "stacked by problem family (nested induction / lemma tower / "
                 "sum-of-squares cascade), with the per-family compile-and-judge pass rate "
                 "of the model-produced chain."),
        "one_line_reading": ("Chain length is 5-9 steps with mean %.2f; the two synthetic "
                             "families are rigid (5 and 9 steps) and certify perfectly, "
                             "while the 181-item inequality cascade spreads over 5-7 steps "
                             "and carries every failure." % mean_steps),
        "source": [{"path": SRC, "md5": src_md5, "rows": len(rows),
                    "used_arms": {"gold_chain": n, "stored_chain": len(stored)},
                    "role": "per-item chain read-out; gold_ops defines the step count"}],
        "outputs": {"png": fig_path, "csv": csv_path},
        "method": ("step count = len(gold_ops) on the gold_chain arm; identical gold_ops in "
                   "both arms so the count is arm-independent. Panel (b) uses the "
                   "certified flag of the stored_chain arm (chain_src='gen34_slotA')."),
        "readings": {
            "n_items": n, "mean_steps": round(mean_steps, 4),
            "min_steps": int(steps.min()), "max_steps": int(steps.max()),
            "step_histogram": {int(k): int(v) for k, v in
                               zip(*np.unique(steps, return_counts=True))},
            "per_family_n": dict(fam_n),
            "per_family_certified_stored_arm": {
                f: {"certified": fam_cert[f][0], "n": fam_cert[f][1],
                    "rate": round(fam_cert[f][0] / float(fam_cert[f][1]), 4)}
                for f in FC.FAM_ORDER},
        },
        "caveats": [
            "gold_chain is certified 218/218 by construction, so panel (b) deliberately "
            "uses the stored_chain arm; do not quote 218/218 as a capability number.",
            "the SPEC-suggested mean 6.12 is reproduced exactly (6.1193 on both arms).",
            "family labels in-figure are English (Nested induction / Lemma tower / "
            "Sum-of-squares cascade).",
        ],
    }
    FC.save_meta(NAME, meta)
    print(json.dumps(meta["readings"], indent=1))


if __name__ == "__main__":
    main()

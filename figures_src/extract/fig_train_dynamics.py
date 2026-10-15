#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""fig_train_dynamics -- CCP gen44 v7 / config-S1 training dynamics over 6000 steps.

Source  : scratch/ccp/gen44/out/CURVE_g44_v7_formal_s1_a2wt1_6000.jsonl (6000 rows, 1 per step)
Outputs : figs/fig_train_dynamics.png
          data/fig_train_dynamics.csv    tidy per-step table
          data/fig_train_dynamics.json   provenance + headline readings

Loss semantics (verified against gen44_hybrid.py:866-915):
  loss  = the objective actually optimised        (acc["loss"] / n_valid)
  chain = operator-selection cross-entropy        (acc["ls"]   / n_valid)
  slot  = slot-argument cross-entropy             (acc["lb"]   / n_binder_free)
  text  = render/text cross-entropy               (acc["lt"]   / n_text_tokens)
`text` lives on a different denominator and sits at 7-8 nats throughout, so it is
preserved in the CSV but deliberately not drawn -- it would flatten the panel.
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

SRC = FC.raw("gen44", "CURVE_g44_v7_formal_s1_a2wt1_6000.jsonl")
NAME = "fig_train_dynamics"
EVAL_COLS = ["n_rows", "sel_top1", "slot_top1", "slot_top1_nonunk", "render_tok_acc"]


def main():
    FC.setup()
    rows = [json.loads(l) for l in open(SRC, "r", encoding="utf-8")]
    src_md5 = FC.md5(SRC)

    step = np.array([r["step"] for r in rows], float)
    loss = np.array([r["loss"] for r in rows], float)
    chain = np.array([r["chain"] for r in rows], float)
    slot = np.array([r["slot"] for r in rows], float)
    text = np.array([r["text"] for r in rows], float)
    gn = np.array([r["gn"] for r in rows], float)
    gpos = np.array([r["slot_g_pos"] for r in rows], float)

    ev_steps = np.array([r["step"] for r in rows if "eval" in r], float)
    ev_sel = np.array([r["eval"]["sel_top1"] for r in rows if "eval" in r], float)
    ev_ren = np.array([r["eval"]["render_tok_acc"] for r in rows if "eval" in r], float)

    # ------------------------------------------------------------- CSV ----
    os.makedirs(FC.DATA, exist_ok=True)
    csv_path = os.path.join(FC.DATA, NAME + ".csv")
    cols = (["step", "loss", "chain", "slot", "text", "gn", "lr", "slot_g_pos",
             "slot_g_dev", "lt_slot", "L", "micro", "nv", "nbf", "ntxt", "secs",
             "n_ind", "n_ineq", "n_g41", "seen_ind", "seen_ineq", "seen_g41"]
            + ["eval_" + c for c in EVAL_COLS])
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r in rows:
            bc = r["batch_cls_counts"]
            sn = r["seen"]
            ev = r.get("eval", {})
            w.writerow([r["step"], r["loss"], r["chain"], r["slot"], r["text"], r["gn"],
                        r["lr"], r["slot_g_pos"], r["slot_g_dev"], r["lt_slot"],
                        r["L"], r["micro"], r["nv"], r["nbf"], r["ntxt"], r["secs"],
                        bc.get("ind"), bc.get("ineq"), bc.get("g41"),
                        sn.get("ind"), sn.get("ineq"), sn.get("g41")]
                       + [ev.get(c) for c in EVAL_COLS])
    print("[csv ] %s" % csv_path)

    # ---------------------------------------------------------- figure ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.5))

    sm_loss, sm_chain, sm_slot = FC.smooth(loss), FC.smooth(chain), FC.smooth(slot)
    # panel (a): objective + components, log y
    ax1.plot(step, loss, color=FC.LGREY, lw=0.6, alpha=0.95, zorder=1)
    ax1.plot(step, chain, color=FC.LGREY, lw=0.6, alpha=0.95, zorder=1)
    ax1.plot(step, slot, color=FC.LGREY, lw=0.6, alpha=0.95, zorder=1)
    ax1.plot(step, sm_loss, color=FC.BLACK, lw=1.8, zorder=3)
    ax1.plot(step, sm_chain, color=FC.BLUE, lw=1.5, zorder=3)
    ax1.plot(step, sm_slot, color=FC.ORANGE, lw=1.5, zorder=3)
    ax1.set_yscale("log")
    ax1.set_xlim(-90, 6600)
    ax1.set_ylim(8e-6, 12)
    ax1.set_xlabel("Training step")
    ax1.set_ylabel("Loss  (nats, log scale)")
    ax1.set_title("(a) Objective and chain/slot components")
    h_raw, = ax1.plot([], [], color=FC.LGREY, lw=0.8)
    h_loss, = ax1.plot([], [], color=FC.BLACK, lw=1.8)
    h_chain, = ax1.plot([], [], color=FC.BLUE, lw=1.5)
    h_slot, = ax1.plot([], [], color=FC.ORANGE, lw=1.5)
    ax1.legend([h_loss, h_chain, h_slot, h_raw],
               ["total objective", "chain (operator selection)", "slot (arguments)",
                "per-step, unsmoothed"],
               loc="upper right", frameon=False, fontsize=8, handlelength=1.7,
               labelspacing=0.35, borderaxespad=0.4)
    ax1.axvline(6000, color=FC.GREY, lw=0.7, ls=":")
    ax1.annotate("4.03 $\\rightarrow$ 0.025 nats\nover 6 000 steps",
                 xy=(0.035, 0.05), xycoords="axes fraction", fontsize=8,
                 color=FC.BLACK, fontweight="bold")

    # panel (b): gradient norm (log, left) + accuracies (right)
    sm_gn = FC.smooth(gn, 51)
    ax2.plot(step, gn, color=FC.LGREY, lw=0.5)
    ax2.plot(step, sm_gn, color=FC.RED, lw=1.6, zorder=3)
    ax2.set_yscale("log")
    ax2.set_xlim(-90, 6600)
    ax2.set_xlabel("Training step")
    ax2.set_ylabel("Gradient norm  (clipped, log scale)", color=FC.RED)
    ax2.tick_params(axis="y", colors=FC.RED)
    ax2.set_title("(b) Gradient norm and held-out accuracies")
    ax2.annotate("gradient norm\n(left axis)", xy=(0.80, 0.63), xycoords="axes fraction",
                 fontsize=8, color=FC.RED, fontweight="bold", ha="center", va="center",
                 bbox=dict(fc="white", ec="none", alpha=0.8, pad=1.4), zorder=8)

    ax3 = ax2.twinx()
    ax3.grid(False)
    ax3.plot(step, gpos, color=FC.GREEN, lw=1.5)
    ax3.plot(ev_steps, ev_sel, color=FC.PURPLE, lw=1.3, ls="--", marker="o",
             ms=2.4, mfc="white", mew=0.6)
    ax3.set_ylim(-0.03, 1.10)
    ax3.set_ylabel("Accuracy  (fraction)", color=FC.GREY)
    ax3.tick_params(axis="y", colors=FC.GREY)
    FC.endlabel(ax3, step[-1], float(gpos[-1]), "slot head $g_{pos}$", FC.GREEN, dx=6, dy=-9)
    FC.endlabel(ax3, ev_steps[-1], float(ev_sel[-1]), "eval operator top-1 (saturated)",
                FC.PURPLE, dx=-8, dy=-16, ha="right",
                bbox=dict(fc="white", ec="none", alpha=0.8, pad=1.2))
    ax3.annotate("$g_{pos}$: slot-head bias (right axis)",
                 xy=(0.045, 0.03), xycoords="axes fraction", fontsize=8,
                 color=FC.GREEN, style="italic")

    fig.tight_layout()
    fig.text(0.5, -0.045,
             "Note: the held-out operator top-1 (purple) is saturated at 1.00 from step 50 "
             "-- 86 of the 120 eval points are exactly 1.00 -- so it is a too-easy probe and "
             "not a learning curve.",
             ha="center", fontsize=7.5, color=FC.PURPLE)
    fig_path = FC.savefig(fig, NAME)

    ev_slot = np.array([r["eval"]["slot_top1"] for r in rows if "eval" in r], float)

    meta = {
        "figure": NAME,
        "what": ("Two-panel training-dynamics view of the final CCP configuration "
                 "(v7 head=220, arm s1, a2wt1, 6000 steps). "
                 "(a) optimised objective with its chain and slot components, log scale; "
                 "(b) clipped gradient norm (left log axis) against the slot-head bias "
                 "g_pos and the held-out operator top-1 accuracy (right axis)."),
        "one_line_reading": ("The objective falls 4.03 -> 0.025 nats over 6000 steps and the "
                             "clipped gradient norm decays 98 -> 1.9 with it; the two "
                             "held-out accuracies on the right axis are inert (operator "
                             "top-1 is saturated at 1.00 from step 50, g_pos stays near "
                             "0.37), so this run must be read off the loss, not off eval."),
        "source": [{"path": SRC, "md5": src_md5, "rows": len(rows),
                    "role": "per-training-step curve (1 line per step)"}],
        "outputs": {"png": fig_path, "csv": csv_path},
        "method": ("loss/chain/slot/text taken verbatim from the jsonl; curves drawn as "
                   "51-step centred moving average over the raw trace; eval points are the "
                   "rotating 48-row held-out pool evaluated every 50 steps."),
        "readings": {
            "loss_step1": loss[0], "loss_step6000": float(loss[-1]),
            "chain_step6000": float(chain[-1]), "slot_step6000": float(slot[-1]),
            "text_step6000": float(text[-1]), "text_min": float(text.min()),
            "gn_max": float(gn.max()), "gn_step6000": float(gn[-1]),
            "slot_g_pos_first": float(gpos[0]), "slot_g_pos_last": float(gpos[-1]),
            "eval_sel_top1_first": float(ev_sel[0]), "eval_sel_top1_last": float(ev_sel[-1]),
            "eval_sel_top1_n_exactly_1": int((ev_sel == 1.0).sum()),
            "eval_slot_top1_first": float(ev_slot[0]),
            "eval_slot_top1_last": float(ev_slot[-1]),
            "eval_render_tok_acc_last": float(ev_ren[-1]),
            "n_eval_points": int(len(ev_steps)),
            "wall_hours": round(sum(r["secs"] for r in rows) / 3600.0, 1),
        },
        "caveats": [
            "'text' (render-token CE, ~7-8 nats) uses a different denominator than "
            "loss/chain/slot and is kept only in the CSV.",
            "eval_sel_top1 is 1.00 already at step 50 and exactly 1.00 at 86 of the 120 "
            "eval points -- a saturated / too-easy probe that must NOT be presented as a "
            "learning curve. Its residual variance is n_frames rounding.",
            "eval_slot_top1 decays to 0.0 by step 6000 (from 0.400 at step 50) even though "
            "the slot training loss falls to 0.019 -- the two slot read-outs disagree; "
            "eval_render_tok_acc stays flat at 0.26-0.31. Neither is quoted in the panel.",
            "the eval pool rotates (eval_rotate), so successive eval points are not on an "
            "identical row set.",
        ],
    }
    FC.save_meta(NAME, meta)
    print(json.dumps(meta["readings"], indent=1))


if __name__ == "__main__":
    main()

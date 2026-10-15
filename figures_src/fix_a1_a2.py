# -*- coding: utf-8 -*-
# fix_a1_a2.py -- v6 (2026-10-09) acceptance-regression pass: bold panel ids
# (A1 a/b, A2 a/b/c) added above each panel (T re-solved, squares kept); the
# DEPTH_UNLOCK event match fixed (was startswith("UNLOCK") and never fired, so
# the 15 blue curriculum-opening dotted lines had never been drawn); A2 canvas
# re-solved to 14.4x4.42in. v5 (2026-10-09) minimal-figure pass: in-figure
# panel titles, reading annotations and the two-line figure footnotes are
# removed; their content moved to the paper captions. Square-panel contract
# (v4) kept by re-solving FH for the new top/bottom margins. Legend gains a
# "raw per-step values" entry so the grey background series stays
# self-explanatory.
# v4 (2026-10-03) per user review: EACH individual panel must
# be 1:1 (square plot area); panels sit in a horizontal row; the composite
# canvas is therefore a wide strip (A1 2 panels, A2 3 panels). Panel squares
# are set numerically via subplots_adjust so width_in == height_in per axes.
# Data: ../data/processed/figures CSV exports (unchanged originals).
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import os

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "STIXGeneral"],
    "mathtext.fontset": "stix",
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size": 3.5,
    "ytick.major.size": 3.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 10.5,
})
BLUE, ORANGE, TEAL, AMBER, GREY, INK = ("#1f6fb4", "#EEA236", "#2a9d8f",
                                        "#f0a63a", "#c9c9c9", "#262626")
# Paths resolve from this file's location, so the script runs from any cwd.
_HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(_HERE), "data", "processed", "figures")
PNG_OUT = os.path.join(_HERE, "output")
os.makedirs(PNG_OUT, exist_ok=True)

def roll(x, w=51):
    return pd.Series(x).rolling(w, center=True, min_periods=1).mean().to_numpy()

# ---------------- A1: training dynamics (2 square panels in a row) ----------
d = pd.read_csv(os.path.join(OUT, "fig_train_dynamics.csv"))
FW, FH = 10.8, 4.59
L, R, T, B, WS = 0.075, 0.945, 0.93, 0.14, 0.30
fig, axes = plt.subplots(1, 2, figsize=(FW, FH), dpi=300)
fig.subplots_adjust(left=L, right=R, top=T, bottom=B, wspace=WS)
for _ax, _tag in zip(axes, "ab"):
    _bb = _ax.get_position()
    fig.text(_bb.x0, 0.955, _tag, fontsize=13, fontweight="bold",
             ha="left", va="center")

ax = axes[0]
ax.plot(d.step, d.loss, color=GREY, lw=0.6, alpha=0.45,
        label="raw per-step values")
ax.plot(d.step, d.chain, color=GREY, lw=0.6, alpha=0.45)
ax.plot(d.step, d.slot, color=GREY, lw=0.6, alpha=0.45)
ax.plot(d.step, roll(d.loss), color=INK, lw=1.6, label="total objective")
ax.plot(d.step, roll(d.chain), color=BLUE, lw=1.6, label="chain (operator sel.)")
ax.plot(d.step, roll(d.slot), color=ORANGE, lw=1.6, label="slot (arguments)")
ax.set_yscale("log")
ax.set_xlabel("Training step")
ax.set_ylabel("Loss (nats, log scale)")
ax.legend(loc="lower left", frameon=False, fontsize=9.5)

ax = axes[1]
ax.plot(d.step, d.gn, color=GREY, lw=0.6, alpha=0.45)
ax.plot(d.step, roll(d.gn), color=ORANGE, lw=1.6, label="gradient norm (51-step mean)")
ax.plot(d.step, d.slot_g_pos, color=TEAL, lw=1.5,
        label="g_pos (slot-head bias)")
ax.set_yscale("log")
ax.set_xlabel("Training step")
ax.set_ylabel("Gradient norm (log scale)")
ax2 = ax.twinx()
ax2.spines["right"].set_visible(True)
ax2.spines["right"].set_linewidth(0.8)
ax2.plot(d.step, d.eval_sel_top1, "o", mfc="none", ms=4.2, mew=1.0,
         color=BLUE, label="eval operator top-1")
ax2.set_ylabel("Eval accuracy (fraction)")
ax2.set_ylim(-0.04, 1.08)
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, loc="lower left", frameon=False, fontsize=9)

fig.savefig(os.path.join(PNG_OUT, "fig_train_dynamics.png"))
plt.close(fig)

# ---------------- A2: v15 ladder (3 square panels in a row) -----------------
g = pd.read_csv(os.path.join(OUT, "fig_v15_ladder_gate.csv"))
t = pd.read_csv(os.path.join(OUT, "fig_v15_ladder_train.csv"))
ev = pd.read_csv(os.path.join(OUT, "fig_v15_ladder_events.csv"))

FW, FH = 14.4, 4.42
L2, R2, T2, B2, WS2 = 0.050, 0.990, 0.93, 0.16, 0.33
fig, axes = plt.subplots(1, 3, figsize=(FW, FH), dpi=300)
fig.subplots_adjust(left=L2, right=R2, top=T2, bottom=B2, wspace=WS2)
for _ax, _tag in zip(axes, "abc"):
    _bb = _ax.get_position()
    fig.text(_bb.x0, 0.955, _tag, fontsize=13, fontweight="bold",
             ha="left", va="center")
axa, axb, axc = axes

axa.plot(t.step, t.train_depth_T, "o", color=GREY, ms=3.0, alpha=0.45,
         label="sampled episode depth T")
axa.plot(g.step, g.gate_tmax, drawstyle="steps-post", color=BLUE, lw=1.9,
         label="gate probe depth")
for k, (_, r) in enumerate(ev.iterrows()):
    if str(r.event).startswith("STALL"):
        axa.axvline(r.step, color="#c0504d", ls=":", lw=0.9, alpha=0.8)
    elif "UNLOCK" in str(r.event):
        axa.axvline(r.step, color=BLUE, ls=":", lw=0.9, alpha=0.8)
axa.set_ylim(0, 16.8)
axa.set_xlabel("Training step")
axa.set_ylabel("Depth")
axa.legend(loc="upper left", frameon=False, fontsize=8.5)

axb.plot(t.step, t.efin.rolling(21, center=True, min_periods=1).mean(),
         color=ORANGE, lw=1.5, label="training e_fin, 21-step mean")
axb.plot(g.step, g.e_fin_at_tmax, "o-", color=TEAL, lw=1.1, ms=3.0,
         mfc="white", label="gate probe")
axb.set_yscale("log")
axb.set_xlabel("Training step")
axb.set_ylabel(r"$e_\mathrm{fin}$")
axb.legend(loc="upper right", frameon=False, fontsize=8.5)

axc.plot(t.step, t.train_depth_T, "o", color=AMBER, ms=3.8, alpha=0.75)
axc.set_ylim(0, 17.5)
axc.set_yticks([1, 5, 9, 13])
axc.set_xlabel("Training step")
axc.set_ylabel("Sampled depth T")

fig.savefig(os.path.join(PNG_OUT, "fig_v15_ladder.png"))
plt.close(fig)

# achieved panel aspect ratios (plot area, inches)
def panel_in(fw, fh, l, r, t, b, ws, n):
    w = (r - l) / (n * (1 + ws)) * fw
    h = (t - b) * fh
    return w, h
for name, fw, fh, l, r, tt, bb, ws, n in [
        ("A1", 10.8, 4.59, 0.075, 0.945, 0.93, 0.14, 0.30, 2),
        ("A2", 14.4, 4.42, 0.050, 0.990, 0.93, 0.16, 0.33, 3)]:
    w, h = panel_in(fw, fh, l, r, tt, bb, ws, n)
    print(f"{name}: panel {w:.2f} x {h:.2f} in  (aspect {w/h:.3f})")

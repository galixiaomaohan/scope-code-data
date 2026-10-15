# -*- coding: utf-8 -*-
"""Shared style + provenance helpers for the SCOPE / CCP LLM-analysis figure set.

Paper spec honoured here:
  * PNG at 300 dpi, no watermark
  * every in-figure string is English (axis labels, curve names, annotations);
    Chinese captions are added at typesetting time
  * colourblind-safe palette (Okabe-Ito anchors, a subset of tab10)
  * curves are labelled directly on the plot rather than through a legend box
  * font size >= 8 pt; axes always carry quantity + unit
"""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Package layout: this file lives in figures_src/extract/, so the package root
# is two directory levels above it. Curated per-figure CSV/JSON exports (the
# direct inputs of the R figure scripts) live in data/processed/figures;
# rendered PNGs go to figures_src/output. Raw experiment outputs live under
# data/raw/. All exported paths are resolved once here and validated against
# their owning directory (safe_data_path, open_data_writer).
_EXTRACT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(_EXTRACT_DIR))
FIGS = os.path.join(ROOT, "figures_src", "output")
DATA = os.path.join(ROOT, "data", "processed", "figures")
RAW = {
    "gen33": os.path.join(ROOT, "data", "raw", "results_gen33"),
    "gen34": os.path.join(ROOT, "data", "raw", "results_gen34"),
    "gen35": os.path.join(ROOT, "data", "raw", "results_gen35"),
    "gen36": os.path.join(ROOT, "data", "raw", "results_gen36"),
    "gen37": os.path.join(ROOT, "data", "raw", "results_gen37"),
    "gen44": os.path.join(ROOT, "data", "raw", "server_only", "gen44"),
    "latent": os.path.join(ROOT, "data", "raw", "latent"),
}


def raw(gen, name):
    """Path of a raw experiment output shipped with the package."""
    return os.path.join(RAW[gen], name)


def safe_data_path(name):
    """Resolve a file name inside DATA and refuse anything that escapes it."""
    root = os.path.realpath(DATA)
    p = os.path.realpath(os.path.join(root, os.path.basename(name)))
    if not p.startswith(root + os.sep):
        raise ValueError("path escapes the data directory: %r" % name)
    return p


def server_only(what):
    """Error message for artifacts that exist only on the original server."""
    raise SystemExit(
        "%s is not shipped with this package (model checkpoint / training "
        "cache). Point the corresponding --argument at a local copy or "
        "re-run the extraction on the original training host." % what)

# --------------------------------------------------------------- palette ----
# Okabe-Ito colourblind-safe palette
BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
RED = "#D55E00"      # vermillion
PURPLE = "#CC79A7"
SKY = "#56B4E9"
YELLOW = "#F0E442"
BLACK = "#1A1A1A"
GREY = "#5A5A5A"
LGREY = "#C8C8C8"

FAM_ORDER = ["gen32_ind_nest", "gen32_ind_tower", "gen32_ineq_cascade"]
FAM_COLOR = {
    "gen32_ind_nest": BLUE,
    "gen32_ind_tower": ORANGE,
    "gen32_ineq_cascade": GREEN,
}
# English in-figure labels; the family names used in the paper text are
# nested induction / lemma tower / sum-of-squares cascade respectively.
FAM_EN = {
    "gen32_ind_nest": "Nested induction",
    "gen32_ind_tower": "Lemma tower",
    "gen32_ineq_cascade": "Sum-of-squares cascade",
}


def setup():
    plt.rcParams.update({
        "figure.dpi": 120, "savefig.dpi": 300,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
        "figure.facecolor": "white",
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.titlesize": 10, "axes.labelsize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "axes.linewidth": 0.9, "axes.edgecolor": "#333333",
        "xtick.major.width": 0.8, "ytick.major.width": 0.8,
        "xtick.direction": "out", "ytick.direction": "out",
        "axes.grid": True, "grid.color": "#DDDDDD", "grid.linewidth": 0.5,
        "grid.alpha": 0.9, "axes.axisbelow": True,
        "lines.linewidth": 1.4, "lines.solid_capstyle": "round",
    })


# ------------------------------------------------------------ provenance ----
def md5(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def savefig(fig, name):
    os.makedirs(FIGS, exist_ok=True)
    p = os.path.join(FIGS, name + ".png")
    fig.savefig(p, dpi=300, facecolor="white")
    plt.close(fig)
    print("[fig ] %s  (%.1f KB)" % (p, os.path.getsize(p) / 1024.0))
    return p


def save_meta(name, obj):
    os.makedirs(DATA, exist_ok=True)
    p = os.path.join(DATA, name + ".json")
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=1)
    print("[meta] %s" % p)
    return p


# --------------------------------------------------------------- helpers ----
def smooth(y, w=51):
    """Centred moving average with edge padding (w made odd)."""
    y = np.asarray(y, dtype=float)
    if w <= 1 or y.size < w:
        return y.copy()
    w = w if w % 2 else w + 1
    pad = w // 2
    k = np.ones(w) / w
    return np.convolve(np.pad(y, pad, mode="edge"), k, mode="valid")


def endlabel(ax, x, y, text, color, dx=6, dy=0, size=8, ha="left", **kw):
    return ax.annotate(text, xy=(x, y), xytext=(dx, dy),
                       textcoords="offset points", color=color,
                       fontsize=size, fontweight="bold", va="center", ha=ha,
                       annotation_clip=False, **kw)


def pick_anchor(series, lo=0, hi=None):
    """Index in [lo, hi) to anchor direct labels at.

    With several series: the index where the (log-scale) values are most separated, so
    labels do not collide.  With a single series: the index whose value is closest to the
    median of the window, i.e. a mid-height anchor that does not sit on a spike.
    """
    n = len(series[0])
    hi = n if hi is None else min(hi, n)
    if hi <= lo:
        lo, hi = 0, n
    if len(series) == 1:
        v = np.asarray(series[0][lo:hi], float)
        pos = v > 0
        if not pos.any():
            return lo
        lv = np.log10(v[pos])
        tgt = float(np.median(lv))
        return lo + int(np.argmin(np.abs(lv - tgt)))
    best, besti = -1.0, lo
    for i in range(lo, hi):
        v = [float(s[i]) for s in series]
        if min(v) <= 0:
            continue
        lv = np.sort(np.log10(v))
        gap = float(np.min(np.diff(lv)))
        if gap > best:
            best, besti = gap, i
    return besti

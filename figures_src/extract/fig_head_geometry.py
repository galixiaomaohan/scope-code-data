#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""fig_head_geometry -- geometry of the 220-row operator-selection head of the final checkpoint.

Checkpoint : scratch/ccp/gen44/ckpt/g44_v7_formal_s1_a2wt1_6000.pt
             keys used: model.step_head.weight (220 x 1152)   <- the head output projection
                        model.codebook.lat_tab  (220 x 64)     <- per-operator latent table
Vocabulary : scratch/ccp/gen44/cache/CCP_VOCAB_V7.json  (op -> index for 161..219)
             plus V32REGISTRY.names() for the base block 0..160.
Read-only throughout: the checkpoint is never written back and no forward pass is run.

Segment partition (the paper's 7-way split, taken verbatim from the figure brief):
  0..160 base(161) | 161..177 deep-chain extension(17) | 178..184 construction curriculum(7)
  185..193 deep-chain family(9) | 194..197 substitution/evaluation(4) | 198..206 routing(9)
  207..219 expansion(13)

MEASURED RESULT (opposite to the generation-aligned expectation):
The head does NOT organise itself into seven generation-aligned clusters.  Measured on
W_head, the silhouette score of the 7-way partition is NEGATIVE (-0.094 over all 220 rows,
-0.083 over the 59 added rows), i.e. rows sit on average closer to another segment than to
their own.  What the geometry does encode is a single large legacy block: 156 of the 161
base rows have their nearest neighbour inside the base block (96.9%), whereas none of the
deep-chain-extension / construction-curriculum / deep-chain-family / routing rows does
(0.000 each).  Among the 59 added rows the 1-NN purity is 0.407 against a chance level of
0.197, so there is real but coarse structure there -- just not the 7-way one.

Outputs : figs/fig_head_geometry.png
          data/fig_head_geometry.csv        per-operator coordinates + segment
          data/op_index_segments_v7.csv     index -> operator name -> segment table
          data/fig_head_geometry.json       provenance + separability readings
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

# The 542 MB gen44 checkpoint and its vocab cache are server-side artifacts
# (md5-registered in the included summaries); SCOPE_RESEARCH_ROOT restores the
# original layout when they are available locally. The operator registry code
# (svokernel_v32) ships inside this package under src/device/.
_RES = os.environ.get("SCOPE_RESEARCH_ROOT", "")
if not _RES:
    FC.server_only("gen44 checkpoint + CCP_VOCAB_V7.json cache")
CKPT = os.path.join(_RES, "scratch/ccp/gen44/ckpt/g44_v7_formal_s1_a2wt1_6000.pt")
VOCAB = os.path.join(_RES, "scratch/ccp/gen44/cache/CCP_VOCAB_V7.json")
SUPER = os.path.join(FC.ROOT, "src", "device", "svokernel_v32")
NAME = "fig_head_geometry"
HEAD = 220
UMAP_SEED = 20260930

# (label, lo, hi, colour)  -- indices inclusive
SEGS = [
    ("Base (legacy 161)", 0, 160, FC.GREY),
    ("Deep-chain ext.", 161, 177, FC.BLUE),
    ("Constr. curriculum", 178, 184, FC.ORANGE),
    ("Deep-chain family", 185, 193, FC.GREEN),
    ("Subst. / eval.", 194, 197, FC.RED),
    ("Routing", 198, 206, FC.PURPLE),
    ("Expansion", 207, 219, FC.SKY),
]


def seg_of(i):
    for lab, lo, hi, col in SEGS:
        if lo <= i <= hi:
            return lab, col
    raise IndexError(i)


def build_names():
    """index -> operator name for all 220 rows, with the provenance of each source."""
    v = json.load(open(VOCAB, "r", encoding="utf-8"))
    names, src = {}, {}
    reg_ok = True
    try:
        sys.path.insert(0, SUPER)
        sys.path.insert(0, os.path.dirname(SUPER))
        from svokernel_v32 import V32REGISTRY
        L = list(V32REGISTRY.names())
        assert len(L) == 160, len(L)
        for i in range(154):
            names[i], src[i] = L[i], "v32registry_insertion_order"
        for j in range(6):                      # the 6 skel ops occupy index 155..160
            names[155 + j], src[155 + j] = L[154 + j], "v32registry_skel_block"
        names[154], src[154] = "v31_end_marker", "v32registry_end_sentinel"
    except Exception as exc:                    # pragma: no cover
        reg_ok = False
        print("[warn] V32REGISTRY unavailable (%s); base rows left unnamed" % exc)
        for i in range(161):
            names[i], src[i] = "base[%d]" % i, "unnamed"
    for key in ["ops_ccp", "ops_bh", "ops_g24", "ops_g41", "ops_gen32", "ops_g42", "ops_g44"]:
        for k, idx in (v.get(key) or {}).items():
            names[idx], src[idx] = k, "CCP_VOCAB_V7:" + key
    missing = [i for i in range(HEAD) if i not in names]
    for i in missing:
        names[i], src[i] = "op[%d]" % i, "unnamed"
    if missing:
        print("[warn] unnamed indices: %s" % missing)
    return names, src, reg_ok, v


def pca2(X):
    Xc = X - X.mean(0, keepdims=True)
    U, S, _ = np.linalg.svd(Xc, full_matrices=False)
    ev = (S ** 2) / float((S ** 2).sum())
    return U[:, :2] * S[:2], ev[:2]


def _dist(X):
    sq = (X ** 2).sum(1)
    d2 = sq[:, None] + sq[None, :] - 2.0 * (X @ X.T)
    return np.sqrt(np.maximum(d2, 0.0))


def silhouette(X, labels):
    """Mean silhouette of an arbitrary label vector (no sklearn dependency)."""
    D = _dist(X)
    n = len(labels)
    labs = list(dict.fromkeys(labels))
    s = []
    for i in range(n):
        own = [j for j in range(n) if j != i and labels[j] == labels[i]]
        if not own:
            s.append(0.0)
            continue
        a = float(D[i, own].mean())
        b = min(float(D[i, [j for j in range(n)
                            if labels[j] == L2]].mean())
                for L2 in labs if L2 != labels[i] and any(labels[j] == L2 for j in range(n)))
        s.append((b - a) / max(a, b) if max(a, b) > 0 else 0.0)
    return float(np.mean(s))


def purity(X, labels):
    """Leave-one-out 1-NN same-label fraction (cosine metric)."""
    Xn = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    S = Xn @ Xn.T
    np.fill_diagonal(S, -np.inf)
    nn = S.argmax(1)
    return float(np.mean([labels[i] == labels[nn[i]] for i in range(len(labels))]))


def chance_purity(labels):
    f = np.array([labels.count(k) for k in set(labels)], float) / len(labels)
    return float((f ** 2).sum())


def self_nn(X, labels):
    Xn = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    S = Xn @ Xn.T
    np.fill_diagonal(S, -np.inf)
    nn = S.argmax(1)
    out = {}
    for lab in dict.fromkeys(labels):
        m = [i for i in range(len(labels)) if labels[i] == lab]
        out[lab] = [int(sum(1 for i in m if labels[nn[i]] == lab)), len(m)]
    return out


def main():
    FC.setup()
    os.environ.setdefault("OMP_NUM_THREADS", "24")
    import torch
    torch.set_num_threads(24)

    ck_md5 = FC.md5(CKPT)
    sd = torch.load(CKPT, map_location="cpu", weights_only=False)
    model = sd["model"]
    W = model["step_head.weight"].detach().float().numpy()
    B = model["step_head.bias"].detach().float().numpy()
    LT = model["codebook.lat_tab"].detach().float().numpy()
    print("[ckpt] step_head.weight %s  bias %s  lat_tab %s  (declared head=%s)"
          % (W.shape, B.shape, LT.shape, sd.get("head")))
    assert W.shape[0] == HEAD, W.shape

    names, nsrc, reg_ok, _ = build_names()
    labels = [seg_of(i)[0] for i in range(HEAD)]
    add = np.array([i > 160 for i in range(HEAD)])          # the 59 added rows
    lab_add = [labels[i] for i in range(HEAD) if add[i]]

    # ------------------------------------------------------------- CSV ----
    os.makedirs(FC.DATA, exist_ok=True)
    tpath = os.path.join(FC.DATA, "op_index_segments_v7.csv")
    with open(tpath, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["index", "operator_name", "segment", "name_source",
                    "weight_row_l2", "bias"])
        for i in range(HEAD):
            w.writerow([i, names[i], labels[i], nsrc[i],
                        round(float(np.linalg.norm(W[i])), 6), round(float(B[i]), 6)])
    print("[csv ] %s" % tpath)

    P_w, ev_w = pca2(W)
    P_l, ev_l = pca2(LT)
    P_add, ev_add = pca2(W[add])

    U = None
    try:
        import umap
        U = umap.UMAP(n_neighbors=15, min_dist=0.15, n_components=2,
                      random_state=UMAP_SEED, metric="euclidean").fit_transform(W)
        print("[umap] ok")
    except Exception as exc:                     # pragma: no cover
        print("[warn] umap unavailable: %s" % exc)

    csv_path = os.path.join(FC.DATA, NAME + ".csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["index", "operator_name", "segment", "added_after_base",
                    "pc1_weight", "pc2_weight", "pc1_addzoom", "pc2_addzoom",
                    "pc1_lattab", "pc2_lattab", "umap1_weight", "umap2_weight"])
        for i in range(HEAD):
            z = (round(float(P_add[i - 161, 0]), 6), round(float(P_add[i - 161, 1]), 6)) \
                if add[i] else ("", "")
            w.writerow([i, names[i], labels[i], bool(add[i]),
                        round(float(P_w[i, 0]), 6), round(float(P_w[i, 1]), 6),
                        z[0], z[1],
                        round(float(P_l[i, 0]), 6), round(float(P_l[i, 1]), 6),
                        (round(float(U[i, 0]), 6) if U is not None else ""),
                        (round(float(U[i, 1]), 6) if U is not None else "")])
    print("[csv ] %s" % csv_path)

    # ---------------------------------------------------------- figure -----
    ncol = 3 if U is not None else 2
    fig, axes = plt.subplots(1, ncol, figsize=(7.2, 3.75))
    axes = np.atleast_1d(axes)
    panels = [
        ("(a) $W_{head}$: PCA, all 220 rows", P_w[:, 0], P_w[:, 1], False,
         "PC1  (%.0f%% var)" % (100 * ev_w[0]), "PC2  (%.0f%%)" % (100 * ev_w[1])),
    ]
    if U is not None:
        panels.append(("(b) $W_{head}$: UMAP, all 220 rows", U[:, 0], U[:, 1], False,
                       "UMAP-1", "UMAP-2"))
    panels.append(("(%s) the 59 added rows only: PCA" % ("c" if U is not None else "b"),
                   P_add[:, 0], P_add[:, 1], True,
                   "PC1  (%.0f%% var)" % (100 * ev_add[0]),
                   "PC2  (%.0f%%)" % (100 * ev_add[1])))

    handles = leg_labels = None
    for ax, (title, x, y, zoom, xl, yl) in zip(axes, panels):
        segs = SEGS[1:] if zoom else SEGS
        sub = list(range(161, HEAD)) if zoom else list(range(HEAD))
        ms = 34 if zoom else 17
        hs = []
        for lab, lo, hi, col in segs:
            m = np.array([lo <= i <= hi for i in sub])
            hs.append(ax.scatter(x[m], y[m], s=ms, color=col,
                                 alpha=0.95 if zoom else 0.92, lw=0.4,
                                 edgecolor="white", zorder=3, label=lab))
        if not zoom:                     # tag the one block that does self-group
            lab, lo, hi, col = SEGS[0]
            m = np.array([lo <= i <= hi for i in range(HEAD)])
            ax.annotate("legacy base\n%d rows" % int(m.sum()),
                        xy=(float(np.mean(x[m])), float(np.mean(y[m]))),
                        fontsize=8, fontweight="bold", color=col, ha="center",
                        va="center", zorder=6,
                        bbox=dict(fc="white", ec="none", alpha=0.8, pad=0.8))
        ax.set_xlabel(xl, fontsize=8)
        ax.set_ylabel(yl, fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.tick_params(labelsize=7.5)

    handles = hs
    leg_labels = ["Base (161)", "Deep-chain ext. (17)", "Constr. curriculum (7)",
                  "Deep-chain family (9)", "Subst. / eval. (4)", "Routing (9)",
                  "Expansion (13)"]
    fig.tight_layout(rect=[0.0, 0.30, 1.0, 1.0])
    fig.legend(handles, leg_labels, loc="lower center", ncol=2, frameon=False,
               fontsize=8, handletextpad=0.4, columnspacing=2.2, labelspacing=0.35,
               bbox_to_anchor=(0.5, 0.075), markerscale=1.1)
    fig.text(0.5, 0.008,
             "leave-one-out 1-NN purity: base 0.969 (156/161); every other segment "
             "$\\leq$ 0.462 (0.000 for four of them).\n"
             "7-way silhouette is negative ($-0.094$ over all 220 rows, $-0.083$ over the "
             "added 59): the head separates legacy from added operators, not the seven "
             "curriculum generations.",
             ha="center", va="bottom", fontsize=7.5, color=FC.BLACK)
    fig_path = FC.savefig(fig, NAME)

    # ------------------------------------------------- readings / meta -----
    pur_all, pur_add = purity(W, labels), purity(W[add], lab_add)
    sil_all, sil_add = silhouette(W, labels), silhouette(W[add], lab_add)
    sil_lt, sil_u = silhouette(LT, labels), (silhouette(U, labels) if U is not None else None)
    sself = self_nn(W, labels)
    per_seg = {}
    for lab, lo, hi, col in SEGS:
        m = np.array([lo <= i <= hi for i in range(HEAD)])
        Ws = W[m]
        Wn = Ws / np.maximum(np.linalg.norm(Ws, axis=1, keepdims=True), 1e-12)
        C = Wn @ Wn.T
        iu = np.triu_indices(Ws.shape[0], 1)
        per_seg[lab] = {
            "n": int(m.sum()),
            "mean_row_l2": round(float(np.linalg.norm(Ws, axis=1).mean()), 4),
            "mean_pairwise_cosine_within": (round(float(C[iu].mean()), 4)
                                            if len(iu[0]) else None),
            "self_nn": sself[lab][0], "self_nn_frac": round(sself[lab][0] / sself[lab][1], 4),
        }

    meta = {
        "figure": NAME,
        "what": ("Two-dimensional embeddings of the 220 operator rows of the final "
                 "checkpoint's selection head, coloured by the paper's 7-way vocabulary "
                 "partition. (a) PCA of the head output projection $W_{head}$ (220x1152), "
                 "all rows; (b) UMAP of the same matrix; (%s) PCA recomputed on the 59 "
                 "rows added after the legacy base only, which is the only zoom in which "
                 "the added operators are visible."
                 % ("c" if U is not None else "b")),
        "one_line_reading": ("The head's geometry does NOT carry the 7-way curriculum split: "
                             "silhouette is negative (%.3f over all 220 rows, %.3f over the "
                             "added 59). What it does carry is a legacy/added boundary -- "
                             "156/161 base rows (0.969) have their nearest neighbour inside "
                             "the base block while the extension, curriculum, family and "
                             "routing blocks score 0.000 each. Among the added rows the 1-NN "
                             "purity is %.3f against a chance level of %.3f, so there is real "
                             "but coarse structure, not generation-aligned clusters."
                             % (sil_all, sil_add, pur_add, chance_purity(lab_add))),
        "source": [
            {"path": CKPT, "md5": ck_md5, "role": "checkpoint",
             "keys": {"model.step_head.weight": list(W.shape),
                      "model.step_head.bias": list(B.shape),
                      "model.codebook.lat_tab": list(LT.shape)},
             "declared_head": sd.get("head"), "step": sd.get("step")},
            {"path": VOCAB, "md5": FC.md5(VOCAB), "role": "operator index registry 161..219"},
            {"path": os.path.join(SUPER, "v32codebook.py"),
             "role": ("base-name source (V32REGISTRY.names(), insertion order verified "
                      "against the documented anchors 6/7/8/15/95)")},
        ],
        "outputs": {"png": fig_path, "csv": csv_path, "index_table": tpath},
        "method": ("weights read straight out of the checkpoint on CPU (no forward pass); "
                   "PCA by SVD on column-centred rows; UMAP n_neighbors=15, min_dist=0.15, "
                   "random_state=%d; separability by leave-one-out 1-NN purity (cosine) and "
                   "by the mean silhouette computed directly from the distance matrix."
                   % UMAP_SEED),
        "readings": {
            "head_rows": HEAD, "head_dim": int(W.shape[1]), "n_added": int(add.sum()),
            "pc1_pc2_var_weight": [round(float(ev_w[0]), 4), round(float(ev_w[1]), 4)],
            "pc1_pc2_var_addzoom": [round(float(ev_add[0]), 4), round(float(ev_add[1]), 4)],
            "pc1_pc2_var_lattab": [round(float(ev_l[0]), 4), round(float(ev_l[1]), 4)],
            "nn1_purity_all": round(pur_all, 4),
            "nn1_purity_added": round(pur_add, 4),
            "chance_purity_all": round(chance_purity(labels), 4),
            "chance_purity_added": round(chance_purity(lab_add), 4),
            "silhouette_weight_all": round(sil_all, 4),
            "silhouette_weight_added": round(sil_add, 4),
            "silhouette_lattab_all": round(sil_lt, 4),
            "silhouette_umap_all": (round(sil_u, 4) if sil_u is not None else None),
            "umap_available": U is not None, "umap_seed": UMAP_SEED,
            "base_names_resolved": reg_ok,
            "per_segment": per_seg,
        },
        "caveats": [
            "NEGATIVE RESULT, registered deliberately: the naive claim 'the 7-way "
            "vocabulary partition is written into the head weights' is FALSE at this scale. "
            "The figure and this record state the measured alternative instead.",
            "PC1+PC2 explain only 25% of the variance of $W_{head}$; a 2-D view is a weak "
            "summary and the negative silhouette is measured in the full 1152-d space, not "
            "in the projection.",
            "The brief's index ranges are used verbatim. Index 178 ('bh_ineq_tail_other') is "
            "the one row that does not sit with its own bh_* family: it falls inside the "
            "178..184 'construction curriculum' block because g24_* starts at 179.",
            "Segment labels are the English forms used throughout the paper "
            "(base, deep-chain extension, construction curriculum, deep-chain "
            "family, substitution and evaluation, routing, expansion).",
            "UMAP coordinates depend on the seed and on n_neighbors; panel (a) PCA is "
            "deterministic and is the one to quote quantitatively.",
            "Base-block names for 0..153 come from V32REGISTRY insertion order, verified "
            "against the documented anchors (6=math_solve_linear_system, 7=math_simplify_expr, "
            "8=math_expand_expr, 15=math_evaluate_expr, 95=mini_kernel_arith), because "
            "v32_op_index_map() returns an empty map without its data-loading build step.",
        ],
    }
    FC.save_meta(NAME, meta)
    print(json.dumps({k: v for k, v in meta["readings"].items()
                      if k != "per_segment"}, indent=1))


if __name__ == "__main__":
    main()

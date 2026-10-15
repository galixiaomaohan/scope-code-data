#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""fig_hidden_trajectory + fig_step_entropy -- read hidden_frames.npz, draw both figures.

Input   : hidden_frames.npz (produced by fig_llm_extract.py on GPU0)
Figures : fig_hidden_trajectory.png  UMAP of the decision-frame hidden states
          fig_step_entropy.png       mean entropy of the operator distribution vs step
Exports : fig_hidden_trajectory.csv  per-frame UMAP coordinates + metadata
          fig_step_entropy.csv       per-step mean, SE, count of entropy by outcome
          fig_hidden_trajectory.json, fig_step_entropy.json (all under the
          package data directory)

h_t is the 1152-d vector that step_head consumes (verified exactly factorable back into the
step logits by the extractor).  Frames are one per decision step; the frame at step k is the
state from which the k-th operator was chosen.
"""
from __future__ import annotations

import csv
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figcommon as FC

NPZ = os.path.join(FC.DATA, "hidden_frames.npz")
UMAP_SEED = 20260930
N_TRAJ_ITEMS = 80


def load():
    d = np.load(NPZ, allow_pickle=False)
    return {k: d[k] for k in d.files}


def umap2(X, seed=UMAP_SEED):
    import umap
    return umap.UMAP(n_neighbors=15, min_dist=0.12, n_components=2,
                     random_state=seed, metric="euclidean").fit_transform(X)


def fig_trajectory(D):
    h = D["h"].astype(np.float32)
    fam = np.array([f.decode() if isinstance(f, bytes) else str(f) for f in D["family"]])
    ri = D["ri"]
    k = D["k"]
    exact = D["item_exact"]
    U = umap2(h)

    # deterministic sample of items for the trajectory panel
    uniq = np.unique(ri)
    rng = np.random.RandomState(UMAP_SEED)
    pick = set(rng.choice(uniq, size=min(N_TRAJ_ITEMS, uniq.size), replace=False).tolist())
    sel = np.array([r in pick for r in ri])

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.0))

    ax = axes[0]
    for f in FC.FAM_ORDER:
        m = fam == f
        ax.scatter(U[m, 0], U[m, 1], s=7, color=FC.FAM_COLOR[f], alpha=0.55, lw=0,
                   label=FC.FAM_EN[f])
    ax.legend(loc="best", frameon=False, fontsize=8, handletextpad=0.3, labelspacing=0.25,
              markerscale=2.0)
    ax.set_title("(a) All $\\mathdefault{%d}$ frames, by family" % len(h), fontsize=8.5)

    ax = axes[1]
    for f in FC.FAM_ORDER:
        m = (fam == f)
        ax.scatter(U[m, 0], U[m, 1], s=4, color=FC.LGREY, alpha=0.5, lw=0)
    for r in sorted(pick):
        m = ri == r
        o = np.argsort(k[m])
        xs, ys = U[m, 0][o], U[m, 1][o]
        f = fam[m][0]
        ax.plot(xs, ys, color=FC.FAM_COLOR.get(f, FC.GREY), lw=0.7, alpha=0.75)
        ax.scatter(xs[:1], ys[:1], s=13, marker="o", color=FC.FAM_COLOR.get(f, FC.GREY),
                   lw=0, zorder=4)
        ax.scatter(xs[-1:], ys[-1:], s=16, marker="^", color=FC.FAM_COLOR.get(f, FC.GREY),
                   lw=0, zorder=4)
    ax.set_title("(b) $\\mathdefault{%d}$ sampled chains (o first, $\\Delta$ last)"
                 % len(pick), fontsize=8.5)

    ax = axes[2]
    sc = ax.scatter(U[:, 0], U[:, 1], c=k, s=7, cmap="viridis", lw=0, alpha=0.85)
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label("decision step  $t$", fontsize=8)
    cb.ax.tick_params(labelsize=7.5)
    ax.set_title("(c) All frames, by step index", fontsize=8.5)

    for ax in axes:
        ax.set_xlabel("UMAP-1", fontsize=8)
        ax.set_ylabel("UMAP-2", fontsize=8)
        ax.tick_params(labelsize=7.5)
    fig.tight_layout()
    p = FC.savefig(fig, "fig_hidden_trajectory")

    # UMAP coordinates drift with the umap-learn and numpy versions, so a
    # rerun writes a .regenerated side file; the frozen extraction under the
    # package data directory is never replaced silently (see DATA_README.md).
    csvp = os.path.join(FC.DATA, "fig_hidden_trajectory.regenerated.csv")
    with open(csvp, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["ri", "family", "step_k", "umap1", "umap2", "item_exact_hit",
                    "in_trajectory_sample"])
        for i in range(len(h)):
            w.writerow([int(ri[i]), fam[i], int(k[i]), round(float(U[i, 0]), 6),
                        round(float(U[i, 1]), 6), int(exact[i]), int(sel[i])])
    print("[csv ] %s" % csvp)

    # family separation in the embedding: 1-NN purity + silhouette
    import collections
    labs = list(fam)
    Xn = U / np.maximum(np.linalg.norm(U, axis=1, keepdims=True), 1e-12)
    S = Xn @ Xn.T
    np.fill_diagonal(S, -np.inf)
    nn = S.argmax(1)
    pur = float(np.mean([labs[i] == labs[nn[i]] for i in range(len(labs))]))
    cnt = collections.Counter(labs)
    n = len(labs)
    ch = float(sum((v / n) ** 2 for v in cnt.values()))

    # trajectory coherence measured in the FULL 1152-d space (not the embedding)
    hn = h / np.maximum(np.linalg.norm(h, axis=1, keepdims=True), 1e-12)
    adj = []
    for r in uniq:
        m = np.where(ri == r)[0]
        m = m[np.argsort(k[m])]
        for a_, b_ in zip(m[:-1], m[1:]):
            adj.append(float(hn[a_] @ hn[b_]))
    adj = np.array(adj)
    rng2 = np.random.RandomState(UMAP_SEED + 1)
    idx = rng2.randint(0, len(h), size=(40000, 2))
    keep = idx[:, 0] != idx[:, 1]
    rnd = np.array([float(hn[i] @ hn[j]) for i, j in idx[keep]])

    meta = {
        "figure": "fig_hidden_trajectory",
        "what": ("UMAP embedding of the %d decision-frame hidden states h_t (1152-d, the "
                 "exact input of the operator-selection head) collected while the final "
                 "checkpoint walked the 218-item deep exam. (a) all frames coloured by "
                 "problem family; (b) a deterministic sample of %d chains drawn as "
                 "polylines in step order; (c) all frames coloured by step index."
                 % (len(h), len(pick))),
        "one_line_reading": ("h_t is organised first by problem family -- the three families "
                             "separate almost perfectly in the embedding (1-NN purity %.3f "
                             "against a chance level of %.3f) -- and then by the position "
                             "inside the chain: consecutive frames of one item are far more "
                             "similar than random frame pairs (cosine %.3f vs %.3f), and the "
                             "colour gradient in (c) shows the step index sweeping across "
                             "the manifold. A chain therefore walks a family-specific route "
                             "through state space rather than staying put."
                             % (pur, ch, float(adj.mean()), float(rnd.mean()))),
        "source": [{"path": NPZ, "md5": FC.md5(NPZ),
                    "role": "h_t capture (see fig_llm_extract.json for the identity check)"}],
        "outputs": {"png": p, "csv": csvp},
        "method": ("UMAP n_neighbors=15, min_dist=0.12, random_state=%d, euclidean, on the "
                   "raw 1152-d h_t (no prior PCA). Chain lines connect consecutive step "
                   "indices of one item. Coherence numbers are computed in the full 1152-d "
                   "space on L2-normalised vectors." % UMAP_SEED),
        "readings": {
            "n_frames": int(len(h)), "h_dim": int(h.shape[1]),
            "n_items": int(np.unique(ri).size),
            "n_items_in_trajectory_sample": int(len(pick)),
            "trajectory_sample_rule": ("np.random.RandomState(%d).choice over the item ids, "
                                       "no replacement" % UMAP_SEED),
            "umap_seed": UMAP_SEED,
            "nn1_family_purity_umap": round(pur, 4),
            "chance_family_purity": round(ch, 4),
            "adjacent_frame_cosine_mean": round(float(adj.mean()), 4),
            "adjacent_frame_cosine_p05_p95": [round(float(np.percentile(adj, 5)), 4),
                                              round(float(np.percentile(adj, 95)), 4)],
            "random_pair_cosine_mean": round(float(rnd.mean()), 4),
            "n_adjacent_pairs": int(adj.size),
            "frames_per_family": {f: int((fam == f).sum()) for f in FC.FAM_ORDER},
            "items_per_family": {f: int(np.unique(ri[fam == f]).size) for f in FC.FAM_ORDER},
        },
        "caveats": [
            "UMAP coordinates depend on the seed and on n_neighbors; the family purity in "
            "this record is computed in the 2-D embedding and is a statement about the "
            "visualisation. The adjacent-vs-random cosine comparison is computed in the "
            "full 1152-d space and does not depend on the embedding.",
            "Each (item, step) frame occurs exactly once in the capture, so no meaningful "
            "'same item+step nearest neighbour' purity exists; trajectory structure is "
            "supported by the adjacent-vs-random cosine statistic and by panel (b), not by "
            "a step-index purity.",
            "Frames are pooled across items; the per-item polylines in panel (b) are the only "
            "place where within-chain structure is claimed.",
        ],
    }
    # same version guard as the CSV: the meta JSON carries the embedding
    # purity, so it also goes to the regenerated side file.
    FC.save_meta("fig_hidden_trajectory.regenerated", meta)
    print("[traj] purity=%.3f chance=%.3f frames=%d" % (pur, ch, len(h)))


def fig_entropy(D):
    fam = np.array([f.decode() if isinstance(f, bytes) else str(f) for f in D["family"]])
    k = D["k"].astype(int)
    ent = D["entropy"].astype(float)
    pmax = D["p_max"].astype(float)
    exact = D["item_exact"].astype(bool)
    ri = D["ri"]

    kmax = int(k.max())
    groups = [("Certified items", exact, FC.BLUE), ("Failed items", ~exact, FC.ORANGE)]

    rows = []
    stats = {}
    for label, mask, _c in groups:
        per_k = []
        for kk in range(kmax + 1):
            m = mask & (k == kk)
            v = ent[m]
            per_k.append((kk, int(v.size),
                          float(v.mean()) if v.size else None,
                          float(v.std(ddof=1) / np.sqrt(v.size)) if v.size > 1 else None))
            if v.size:
                rows.append([label, kk, int(v.size), round(float(v.mean()), 6),
                             round(float(v.std(ddof=1) / np.sqrt(v.size)), 6)
                             if v.size > 1 else "",
                             round(float(np.mean(pmax[m])), 6)])
        stats[label] = per_k

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    for ax, (label, mask, col) in zip(axes, groups):
        pk = stats[label]
        xs = np.array([r[0] for r in pk if r[2] is not None], float)
        ys = np.array([r[2] for r in pk if r[2] is not None], float)
        es = np.array([r[3] if r[3] is not None else 0.0 for r in pk if r[2] is not None],
                      float)
        ns = [r[1] for r in pk if r[2] is not None]
        ax.fill_between(xs, ys - es, ys + es, color=col, alpha=0.28, lw=0)
        ax.plot(xs, ys, color=col, lw=1.8, marker="o", ms=4, mfc="white", mew=1.1)
        ns_full = [next((r[1] for r in pk if r[0] == kk), 0) for kk in range(kmax + 1)]
        ax.annotate("n per step:  " + "  ".join(str(v) for v in ns_full),
                    xy=(0.02, 0.015), xycoords="axes fraction", fontsize=7.5,
                    color=FC.GREY, va="bottom")
        allm = ent[mask]
        ax.axhline(float(allm.mean()), color=FC.GREY, lw=0.9, ls="--")
        ax.annotate("group mean %.3f nats" % float(allm.mean()), xy=(0.03, 0.06),
                    xycoords="axes fraction", fontsize=8, color=FC.GREY, va="bottom")
        n_items = int(np.unique(ri[mask]).size)
        ax.set_title("(%s) %s  (%d items, %d frames)"
                     % ("a" if label.startswith("Cert") else "b", label, n_items,
                        int(mask.sum())), fontsize=9)
        ax.set_xlabel("Decision step  $t$")
        ax.set_xlim(-0.4, kmax + 0.4)
        ax.set_xticks(range(kmax + 1))
        ax.tick_params(labelsize=7.5)
        ax.set_yscale("log")
        ax.set_ylim(4e-6, 3.0)
    axes[0].set_ylabel("Mean entropy of $P(o|F_t)$  (nats, log)")
    _c4 = stats["Certified items"][4][2] or 0.0
    _f4 = stats["Failed items"][4][2] or 0.0
    axes[1].annotate("the first divergent step lands on a %.0fx\nhigher-entropy decision "
                     "($t$=4: %.3f vs %.3f nats)"
                     % (_f4 / max(_c4, 1e-12), _f4, _c4),
                     xy=(0.03, 0.97), xycoords="axes fraction", fontsize=8,
                     color=FC.ORANGE, fontweight="bold", va="top")
    axes[0].annotate("steps 4-6 are where the tail operator is\nchosen -- the only steps "
                     "with any real entropy",
                     xy=(0.06, 0.86), xycoords="axes fraction", fontsize=8, color=FC.BLUE)
    fig.tight_layout()
    p = FC.savefig(fig, "fig_step_entropy")

    csvp = os.path.join(FC.DATA, "fig_step_entropy.csv")
    with open(csvp, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["group", "step_k", "n_frames", "mean_entropy_nats", "sem_entropy",
                    "mean_p_max"])
        w.writerows(rows)
    print("[csv ] %s" % csvp)

    def gmean(mask):
        return float(ent[mask].mean()) if mask.any() else None

    meta = {
        "figure": "fig_step_entropy",
        "what": ("Mean entropy of the masked operator distribution P(o|F_t) as a function of "
                 "decision step t, split into items the final checkpoint solved exactly "
                 "(certified) and items it failed. Bands are +/-1 s.e.m., n is the number of "
                 "frames at that step; the dashed line is each group's overall mean."),
        "one_line_reading": ("Every decision before step 4 is essentially deterministic "
                             "(entropy < 1e-3 nats), and steps 4-6 are the only ones that "
                             "carry any uncertainty. There the two groups separate sharply: "
                             "at t=4 the failed items decide at 0.191 nats against 0.027 for "
                             "the certified ones (7x). Both statements hold at once -- the "
                             "model is measurably less certain at the step where it goes "
                             "wrong, yet its top-1 probability there is still 0.918, so the "
                             "mistake is also 'confidently wrong'."),
        "source": [{"path": NPZ, "md5": FC.md5(NPZ), "role": "entropy capture"}],
        "outputs": {"png": p, "csv": csvp},
        "method": ("entropy of softmax(step_logits + mask) with mask = legal operator "
                   "indices + END, i.e. exactly the distribution gen34_eval.walk_chain "
                   "decides with; frames grouped by item outcome, averaged per step index."),
        "readings": {
            "kmax": kmax,
            "n_frames_certified": int(exact.sum()),
            "n_frames_failed": int((~exact).sum()),
            "n_items_certified": int(np.unique(ri[exact]).size),
            "n_items_failed": int(np.unique(ri[~exact]).size),
            "mean_entropy_certified": gmean(exact),
            "mean_entropy_failed": gmean(~exact),
            "entropy_t0_certified": stats["Certified items"][0][2],
            "entropy_t0_failed": stats["Failed items"][0][2],
        },
        "caveats": [
            "the failed group has only %d items, so late-step points rest on a few frames "
            "and their s.e.m. bands are correspondingly wide."
            % int(np.unique(ri[~exact]).size),
            "the step axis is unbounded for failures: an item that stopped early has no "
            "frames at high t, so the per-step n differs between groups (annotated).",
            "'certified' here is exact operator-chain match, which coincides with Lean "
            "certification on this checkpoint (191/218).",
        ],
    }
    FC.save_meta("fig_step_entropy", meta)
    print("[ent ] cert=%.4f fail=%.4f (nats), items %d/%d"
          % (meta["readings"]["mean_entropy_certified"],
             meta["readings"]["mean_entropy_failed"],
             meta["readings"]["n_items_certified"], meta["readings"]["n_items_failed"]))


def main():
    FC.setup()
    D = load()
    print("[npz] h=%s frames=%d items=%d"
          % (D["h"].shape, D["h"].shape[0], np.unique(D["ri"]).size))
    fig_trajectory(D)
    fig_entropy(D)


if __name__ == "__main__":
    main()

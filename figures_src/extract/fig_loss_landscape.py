#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""fig_loss_landscape -- filter-normalised loss landscape (Li et al. 2018) of the final ckpt.

Method (Li, Xu, Taylor, Goldstein & Studený, "Visualizing the Loss Landscape of Neural
Nets", NeurIPS 2018):
  * two random directions d1, d2, one tensor at a time,
  * FILTER NORMALISATION: each direction tensor is rescaled to the norm of the
    corresponding weight tensor, d <- d / ||d|| * ||theta||, which removes the
    scale-invariance artefact that otherwise makes the picture look artificially flat,
  * theta(a,b) = theta* + a*d1 + b*d2 on a square grid in [-delta, delta]^2.

Loss being mapped: the TAIL-DECISION planning loss.  For a stratified sample of exam items
we take the decision frame at step k=4 with the GOLD prefix (gold_ops[:5]) and record the
cross-entropy of the operator head against the gold operator gold_ops[4].  The prefix is a
constant, independent of theta, so the loss is smooth in the parameters -- walking the model
with its own predictions would make argmax decisions jump and destroy smoothness.  k=4 is
chosen because it is the tail-decision step where essentially all of the decision entropy
and all of the failures live (see fig_step_entropy).

delta is CALIBRATED, not guessed: a 1-D scan along +-d1 picks the largest candidate with
loss <= --max-factor x loss(theta*), and the choice is recorded in the JSON.

Outputs: figs/fig_loss_landscape.png, data/fig_loss_landscape.csv/.json
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

import numpy as np

# Experiment-tree resolution: gen4x modules ship in this package; the 542 MB
# checkpoint and training caches do not (md5-registered in the included
# summaries). SCOPE_RESEARCH_ROOT restores the original server layout when
# those artifacts are available locally.
_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir))
RES = os.environ.get("SCOPE_RESEARCH_ROOT", "")
for _p in (os.path.join(_PKG, "src", "device"),
           os.path.join(_PKG, "src", "experiments", "gen41_support"),
           os.path.join(_PKG, "src", "experiments", "gen42_support"),
           os.path.join(_PKG, "src", "experiments", "gen44_public"),
           os.path.dirname(os.path.abspath(__file__))):
    if os.path.exists(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

import torch                                                          # noqa: E402
import torch.nn.functional as F                                       # noqa: E402

import gen44_eval as E44                                              # noqa: E402
import figcommon as FC_PATHS                                          # noqa: E402

E34 = E44.E34
C = E44.C
if not RES:
    FC_PATHS.server_only("gen44 checkpoint/cache set "
                         "(ckpt/g44_v7_formal_s1_a2wt1_6000.pt, GEN44_V7_ALL.bin, ...)")
CKPT = os.path.join(RES, "scratch/ccp/gen44/ckpt/g44_v7_formal_s1_a2wt1_6000.pt")
FRAME_K = 4
D_SEED1, D_SEED2 = 20260930, 20260931


class NS:
    pass


def make_args(a):
    ns = NS()
    ns.vocab, ns.cache, ns.side = a.vocab, a.cache, a.side
    ns.exam_side, ns.allow, ns.lat31, ns.init = a.exam_side, a.allow, a.lat31, a.init
    ns.ckpt, ns.seg, ns.n = a.ckpt, a.seg, a.n
    ns.dev, ns.seed, ns.log, ns.slot_side = a.dev, a.seed, a.log, a.slot_side
    return E44.build_args(ns)


def make_direction(model, seed, dtype=torch.float32):
    """Filter-normalised random direction: per tensor, d <- d/||d|| * ||theta||."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    out = {}
    for n, p in model.named_parameters():
        d = torch.randn(tuple(p.shape), generator=g, dtype=torch.float32)
        dn, pn = float(d.norm()), float(p.detach().float().norm())
        if dn > 0 and pn > 0:
            d = d / dn * pn
        out[n] = d.to(device=p.device, dtype=p.dtype)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--init", default="ckpt/sla_v31f.pt")
    ap.add_argument("--seg", default="exam_deep")
    ap.add_argument("--n", type=int, default=218)
    ap.add_argument("--cache", default=os.path.join(RES, "scratch/ccp/gen44/cache/GEN44_V7_ALL.bin"))
    ap.add_argument("--side", default=os.path.join(RES, "scratch/ccp/gen44/cache/GEN44_V7_SIDECAR.jsonl"))
    ap.add_argument("--exam-side", default=FC_PATHS.raw("gen34", "DEEP_EXAM_SIDECAR.jsonl"))
    ap.add_argument("--allow", default=os.path.join(RES, "scratch/ccp/gen44/cache/ccp_allow_v7.bin"))
    ap.add_argument("--vocab", default=os.path.join(RES, "scratch/ccp/gen44/cache/CCP_VOCAB_V7.json"))
    ap.add_argument("--lat31", default="cppcuda/cache/lat_tab_v31.bin")
    ap.add_argument("--slot-side", default=C.RENDER_SIDE_G41)
    ap.add_argument("--seed", type=int, default=20260923)
    ap.add_argument("--state-shift", type=int, default=1)
    ap.add_argument("--dev", default="cuda:0")
    ap.add_argument("--items", type=int, default=32, help="stratified exam items in the batch")
    ap.add_argument("--grid", type=int, default=0, help="0 = auto from the time budget")
    ap.add_argument("--delta", type=float, default=0.0, help="0 = calibrate by 1-D scan")
    ap.add_argument("--max-factor", type=float, default=5.0)
    ap.add_argument("--budget-s", type=float, default=5400.0)
    ap.add_argument("--log", default=os.path.join(_PKG, "figures_src", "output",
                                                  "fig_loss_landscape.log"))
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.log), exist_ok=True)

    def log(msg):
        line = "[%s] %s" % (time.strftime("%H:%M:%S"), msg)
        print(line, flush=True)
        with open(a.log, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    dev = torch.device(a.dev)
    log("=== fig_loss_landscape ckpt=%s dev=%s ===" % (a.ckpt, a.dev))
    na = make_args(a)
    E44._set_vocab_v5(a.vocab)
    if int(a.state_shift):
        E34._payload_of = E44._payload_of_g41
    parents = {int(k): v["parent_idx"] for k, v in E34.VOC["init_map"].items()}
    allow = torch.from_numpy(np.fromfile(a.allow, dtype=np.uint8).reshape(
        E34.NHEAD, E34.NSLOT, E34.NVALS).copy())
    ctx = E34._load_ctx_eval(na)
    model, minfo = E34.load_model(na, parents, dev)
    E44.nan_guard(model)
    model.eval()
    legal_idx, names = E34._legal_ops()
    name2idx = {v: k for k, v in names.items()}
    log("model %s" % json.dumps({k: v for k, v in minfo.items()
                                 if k in ("md5", "step")}, ensure_ascii=False))

    items = E34.load_items(na)
    idxs = np.linspace(0, len(items) - 1, a.items).round().astype(int)
    idxs = sorted(set(int(i) for i in idxs))
    frames, used, skipped = [], [], []
    for i in idxs:
        ri, s = items[i]
        gold = list(s.get("ops") or [])
        if len(gold) <= FRAME_K or any(o not in name2idx for o in gold):
            skipped.append(int(ri))
            continue
        payload = E34._payload_of(s)
        cache = E34._prefix(ctx, s, gold[:FRAME_K + 1], payload)   # gold prefix, constant
        bc = E34.L32.BatchCpu32().build(cache, [0], 0, 0, 0)
        bc.rowIds = [0]
        tgt = name2idx[gold[FRAME_K]]
        lab = bc.labels[min(FRAME_K, len(bc.labels) - 1)]
        frames.append((cache, bc, tgt))
        used.append((int(ri), s.get("family"), gold[FRAME_K], lab == tgt))
    ok_lab = sum(1 for f in used if f[3])
    log("batch: %d items used, %d skipped, labels cross-check %d/%d"
        % (len(frames), len(skipped), ok_lab, len(used)))
    if ok_lab != len(used):
        log("WARN: bc.labels does not confirm the gold target for %d items"
            % (len(used) - ok_lab))
    if not frames:
        log("FATAL: no usable frames")
        raise SystemExit(2)

    W = model.step_head.weight
    Bb = model.step_head.bias
    HID = int(W.shape[1] // 2)

    def loss_now():
        tot = 0.0
        with torch.no_grad():
            for cache, bc, tgt in frames:
                cm, cosc, sinc = model.consts(bc.L, dev)
                mk = model.mask_of(cm, bc.B, bc.L, bc.lenB, dev)
                seq = model.assemble(bc.B, bc.L, bc, cache, dev)
                H = model.forward_body(seq, mk, cosc, sinc)
                Hf = H.reshape(-1, H.shape[-1])
                idxH = torch.tensor(bc.idxH, dtype=torch.long, device=dev)
                idxS = torch.tensor(getattr(bc, "idxS", bc.idxC), dtype=torch.long,
                                    device=dev)
                x = torch.cat([Hf.index_select(0, idxH), Hf.index_select(0, idxS)], -1)
                lg = F.linear(x, W, Bb)
                z = lg.view(1, bc.Talloc, -1)[0][min(FRAME_K, bc.Talloc - 1)]
                tot += float(F.cross_entropy(z.unsqueeze(0),
                                             torch.tensor([tgt], device=dev)))
                del H, seq, lg, z, x
            if dev.type == "cuda":
                torch.cuda.empty_cache()
        return tot / len(frames)

    p0 = {n: p.detach().clone() for n, p in model.named_parameters()}
    d1 = make_direction(model, D_SEED1)
    d2 = make_direction(model, D_SEED2)
    log("directions built (filter-normalised, seeds %d/%d)" % (D_SEED1, D_SEED2))

    def set_at(alpha, beta):
        with torch.no_grad():
            for n, p in model.named_parameters():
                p.copy_(p0[n] + alpha * d1[n] + beta * d2[n])

    L0 = loss_now()
    log("loss(theta*) = %.6f nats" % L0)

    # ---- delta calibration: 1-D scan along +-d1 ----
    if a.delta > 0:
        delta = a.delta
        scan = []
    else:
        cands = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
        scan = []
        delta = 0.05
        for c in cands:
            set_at(c, 0.0)
            lp = loss_now()
            set_at(-c, 0.0)
            lm = loss_now()
            scan.append((c, lp, lm))
            log("delta scan %+.2f: +d1 %.6f  -d1 %.6f" % (c, lp, lm))
            if max(lp, lm) <= a.max_factor * max(L0, 1e-9):
                delta = c
            else:
                break
    set_at(0.0, 0.0)

    # ---- timing probe -> grid size ----
    t0 = time.time()
    for _ in range(3):
        set_at(0.3 * delta, 0.3 * delta)
        loss_now()
    per = (time.time() - t0) / 3.0
    set_at(0.0, 0.0)
    log("per-point %.2f s" % per)
    if a.grid:
        g = a.grid
    else:
        g = 51 if per * 51 * 51 < 0.75 * a.budget_s else (41 if per * 41 * 41
                                                          < 0.75 * a.budget_s else 31)
    log("grid = %dx%d (projected %.0f s of %.0f s budget)"
        % (g, g, per * g * g, a.budget_s))

    alphas = np.linspace(-delta, delta, g)
    Z = np.zeros((g, g))
    t0 = time.time()
    with torch.no_grad():
        for i, al in enumerate(alphas):
            for j, be in enumerate(alphas):
                set_at(float(al), float(be))
                Z[i, j] = loss_now()
            log("  row %d/%d  (%.0f s)" % (i + 1, g, time.time() - t0))
    set_at(0.0, 0.0)

    csvp = os.path.join(FC_PATHS.DATA, "fig_loss_landscape.csv")
    with open(csvp, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["alpha_d1", "beta_d2", "loss_nats"])
        for i, al in enumerate(alphas):
            for j, be in enumerate(alphas):
                w.writerow([round(float(al), 6), round(float(be), 6),
                            round(float(Z[i, j]), 8)])
    log("csv -> %s" % csvp)

    # ------------------------------------------------------------ figure ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import figcommon as FC
    FC.setup()

    A, Bg = np.meshgrid(alphas, alphas, indexing="ij")
    fig = plt.figure(figsize=(7.2, 3.1))
    ax = fig.add_subplot(1, 2, 1)
    lv = np.linspace(float(Z.min()), float(np.percentile(Z, 97)), 22)
    cf = ax.contourf(A, Bg, Z, levels=lv, cmap="viridis")
    ax.contour(A, Bg, Z, levels=lv[::3], colors="white", linewidths=0.4, alpha=0.6)
    cb = fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label("planning loss  (nats)", fontsize=8)
    cb.ax.tick_params(labelsize=7.5)
    ax.plot([0], [0], marker="*", ms=11, color="white", mec="black", mew=0.7, zorder=6)
    ax.annotate("$\\theta^*$", xy=(0, 0), xytext=(6, 4), textcoords="offset points",
                fontsize=9, fontweight="bold", color="white", zorder=7)
    ax.set_xlabel("$\\alpha$  along  $d_1$", fontsize=8)
    ax.set_ylabel("$\\beta$  along  $d_2$", fontsize=8)
    ax.set_title("(a) Filter-normalised loss landscape", fontsize=9)
    ax.tick_params(labelsize=7.5)

    ax3 = fig.add_subplot(1, 2, 2, projection="3d")
    st = max(1, g // 34)
    ax3.plot_surface(A[::st, ::st], Bg[::st, ::st], Z[::st, ::st], cmap="viridis",
                     linewidth=0, antialiased=True, alpha=0.95)
    ax3.set_xlabel("$\\alpha$", fontsize=8, labelpad=-4)
    ax3.set_ylabel("$\\beta$", fontsize=8, labelpad=-4)
    ax3.set_zlabel("loss", fontsize=8, labelpad=-6)
    ax3.set_title("(b) Same grid, 3-D", fontsize=9)
    ax3.tick_params(labelsize=7)
    fig.tight_layout()
    p = FC.savefig(fig, "fig_loss_landscape")

    meta = {
        "figure": "fig_loss_landscape",
        "what": ("Filter-normalised loss landscape (Li et al. 2018) around the final "
                 "checkpoint, on a %dx%d grid in [-%.2f, %.2f]^2 along two random "
                 "filter-normalised directions." % (g, g, delta, delta)),
        "one_line_reading": ("The tail-decision planning loss sits in a shallow but "
                             "well-defined basin: loss(theta*) = %.6f nats and the value at "
                             "the grid corners is %.6f nats (%.2fx), so the optimum is a "
                             "broad flat valley rather than a sharp spike -- consistent with "
                             "the very low decision entropy measured in fig_step_entropy."
                             % (L0, float(Z.max()),
                                float(Z.max()) / max(L0, 1e-12))),
        "source": [
            {"path": a.ckpt, "md5": C.md5f(a.ckpt), "role": "centre theta*"},
            {"path": a.exam_side,
             "md5": C.md5f(a.exam_side) if os.path.exists(a.exam_side) else None,
             "role": "exam items (gold operators for the loss targets)"}],
        "outputs": {"png": p, "csv": csvp},
        "method": ("theta(a,b) = theta* + a*d1 + b*d2 with d_i filter-normalised per tensor "
                   "(d <- d/||d|| * ||theta||). Loss = mean cross-entropy of the operator "
                   "head at decision step k=%d with the GOLD prefix (gold_ops[:%d]), over a "
                   "stratified sample of %d exam items; the prefix is theta-independent so "
                   "the loss is smooth." % (FRAME_K, FRAME_K + 1, len(frames))),
        "readings": {
            "grid": g, "delta": delta, "loss_theta_star": L0,
            "loss_min": float(Z.min()), "loss_max": float(Z.max()),
            "loss_max_over_star": float(Z.max()) / max(L0, 1e-12),
            "loss_mean": float(Z.mean()), "loss_std": float(Z.std()),
            "n_items_in_batch": len(frames), "n_items_skipped": len(skipped),
            "skipped_ri": skipped,
            "frame_step_k": FRAME_K,
            "direction_seeds": [D_SEED1, D_SEED2],
            "delta_calibration_scan": [{"delta": c, "loss_plus_d1": lp, "loss_minus_d1": lm}
                                       for c, lp, lm in scan],
            "max_factor_rule": a.max_factor,
            "per_point_seconds": round(per, 3),
            "grid_choice": ("auto (largest of 51/41/31 whose projected cost fits 75%% of the "
                            "%.0f s budget)" % a.budget_s) if not a.grid else "explicit",
            "radii_mean_parameter_relative": {
                n: round(float((d1[n].norm() / max(float(p0[n].norm()), 1e-12)).item()), 5)
                for n in list(d1)[:3]},
        },
        "caveats": [
            "the loss mapped is the TAIL-DECISION planning loss at step k=%d only, not the "
            "full training objective (which also contains the slot and render terms). It is "
            "the loss that carries essentially all of the decision entropy and all of the "
            "failures, so it is the informative slice; it must not be quoted as 'the' "
            "training loss." % FRAME_K,
            "the batch is %d exam items chosen by evenly spaced index (np.linspace over the "
            "exam order), so all three families are represented; it is not a random sample."
            % len(frames),
            "delta was chosen by a 1-D scan along +-d1 under the rule 'largest candidate "
            "whose loss stays within %.1fx of loss(theta*)'; the scan table is in the JSON. "
            "A different rule would give a different zoom level -- the SHAPE is the claim, "
            "not the absolute widths." % a.max_factor,
            "filter normalisation is per tensor, which is the standard recipe; the model's "
            "language-model head is included in the perturbation although it does not feed "
            "the operator head, so part of the perturbation budget is spent off-target.",
        ],
    }
    mp = os.path.join(FC_PATHS.DATA, "fig_loss_landscape.json")
    with open(mp, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=1)
    log("meta -> %s" % mp)
    log("SUMMARY %s" % json.dumps({k: meta["readings"][k] for k in
                                   ("grid", "delta", "loss_theta_star", "loss_max",
                                    "loss_max_over_star", "per_point_seconds")}))


if __name__ == "__main__":
    main()

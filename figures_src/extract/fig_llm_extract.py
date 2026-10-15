#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""fig_llm_extract -- per-decision-frame hidden state h_t + operator-distribution entropy.

ONE forward-pass pass over the 218-item deep exam with the final checkpoint, capturing for
every decision frame:
    h_t   = concat(H[idxH], H[idxS])   (1152-d)  <- exactly what step_head consumes
    lg    = step_logits                (220-d, pre-mask)
    p     = softmax(lg + mask)         with mask = legal_idx + {END31}, i.e. the distribution
                                         the walk actually decides with (gen34_eval.py:386-392)
    entropy(p), p_max, margin
plus the walk outcome (hit / exact_hit / certified) per item.

The walk pipeline is reused VERBATIM from the frozen evaluation stack
(gen44_eval.py phase_walk setup + gen34_eval.py walk_chain loop).  This script imports
those modules read-only and re-implements only the ~6-line inner loop so that it can also
capture h_t and lg.  Nothing in scratch/ccp/gen*/ is written.

EXACTNESS CHECK: with pa_enable=0 in this eval
path the plan-attention delta is absent, so
    F.linear(h_t, step_head.weight, step_head.bias) == lg
must hold identically.  The residual is measured for every frame and the run aborts if it
exceeds --tol, so h_t is provably the head's input rather than a plausible-looking proxy.

Outputs (all under scratch/ccp/figllm/data/):
    hidden_frames.npz    h (N,1152) fp32, per-frame scalars, item metadata
    fig_hidden_entropy_frames.csv   the same scalars without the 1152 dims
    fig_llm_extract.json provenance + verification + coverage
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

import numpy as np

# Experiment-tree resolution. The gen4x modules ship inside this package; the
# checkpoint and training caches do not (542 MB class artifacts, md5-registered
# in the included summaries). SCOPE_RESEARCH_ROOT restores the original server
# layout when the artifacts are available locally.
_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir))
RES = os.environ.get("SCOPE_RESEARCH_ROOT", "")
G44 = os.path.join(RES, "scratch/ccp/gen44") if RES else ""
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
import figcommon as FC                                                # noqa: E402

E34 = E44.E34
C = E44.C          # gen44_common

if not RES:
    FC.server_only("gen44 checkpoint/cache set "
                   "(ckpt/g44_v7_formal_s1_a2wt1_6000.pt, GEN44_V7_ALL.bin, ...)")
CKPT = os.path.join(G44, "ckpt/g44_v7_formal_s1_a2wt1_6000.pt")
EXAM_SIDE = FC.raw("gen34", "DEEP_EXAM_SIDECAR.jsonl")
CACHE = os.path.join(G44, "cache/GEN44_V7_ALL.bin")
SIDE = os.path.join(G44, "cache/GEN44_V7_SIDECAR.jsonl")
VOCAB = os.path.join(G44, "cache/CCP_VOCAB_V7.json")
ALLOW = os.path.join(G44, "cache/ccp_allow_v7.bin")

FAM_EN = {"gen32_ind_nest": "Nested induction",
          "gen32_ind_tower": "Lemma tower",
          "gen32_ineq_cascade": "Sum-of-squares cascade"}


class NS:
    pass


def _mean_or_none(v):
    return float(np.mean(v)) if len(v) else None


def make_args(a):
    ns = NS()
    ns.vocab = a.vocab
    ns.cache = a.cache
    ns.side = a.side
    ns.exam_side = a.exam_side
    ns.allow = a.allow
    ns.lat31 = a.lat31
    ns.init = a.init
    ns.ckpt = a.ckpt
    ns.seg = a.seg
    ns.n = a.n
    ns.dev = a.dev
    ns.seed = a.seed
    ns.log = a.log
    ns.slot_side = a.slot_side
    return E44.build_args(ns)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--init", default="ckpt/sla_v31f.pt")
    ap.add_argument("--seg", default="exam_deep")
    ap.add_argument("--n", type=int, default=218)
    ap.add_argument("--cache", default=CACHE)
    ap.add_argument("--side", default=SIDE)
    ap.add_argument("--exam-side", default=EXAM_SIDE)
    ap.add_argument("--allow", default=ALLOW)
    ap.add_argument("--vocab", default=VOCAB)
    ap.add_argument("--lat31", default="cppcuda/cache/lat_tab_v31.bin")
    ap.add_argument("--slot-side", default=C.RENDER_SIDE_G41)
    ap.add_argument("--seed", type=int, default=20260923)
    ap.add_argument("--state-shift", type=int, default=1)
    ap.add_argument("--dev", default="cuda:0")
    ap.add_argument("--max-items", type=int, default=0, help="0 = all")
    ap.add_argument("--tol", type=float, default=1e-3)
    ap.add_argument("--out-tag", default="hidden_frames")
    ap.add_argument("--log", default=os.path.join(_PKG, "figures_src", "output",
                                                  "fig_llm_extract.log"))
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.log), exist_ok=True)

    def log(msg):
        line = "[%s] %s" % (time.strftime("%H:%M:%S"), msg)
        print(line, flush=True)
        with open(a.log, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    dev = torch.device(a.dev)
    log("=== fig_llm_extract ckpt=%s dev=%s ===" % (a.ckpt, a.dev))
    _free = str(torch.cuda.mem_get_info(0)) if dev.type == "cuda" else "n/a"
    log("cuda:0 mem (free, total): %s" % _free)
    na = make_args(a)
    E44._set_vocab_v5(a.vocab)
    if int(a.state_shift):
        E34._payload_of = E44._payload_of_g41
        log("state-shift payload caliber ON (_payload_of_g41)")
    parents = {int(k): v["parent_idx"] for k, v in E34.VOC["init_map"].items()}
    allow = torch.from_numpy(np.fromfile(a.allow, dtype=np.uint8).reshape(
        E34.NHEAD, E34.NSLOT, E34.NVALS).copy())
    ctx = E34._load_ctx_eval(na)
    model, minfo = E34.load_model(na, parents, dev)
    E44.nan_guard(model)
    model.eval()
    log("model: %s" % json.dumps({k: v for k, v in minfo.items()
                                 if k in ("md5", "step", "missing", "unexpected")},
                                ensure_ascii=False))
    items = E34.load_items(na)
    if a.max_items:
        items = items[:a.max_items]
    log("exam rows = %d" % len(items))

    legal_idx, names = E34._legal_ops()
    legal_idx = list(legal_idx)
    W = model.step_head.weight
    Bb = model.step_head.bias
    HID = int(W.shape[1] // 2)

    recs, H_all, hdrs = [], [], []
    max_diff = 0.0
    t0 = time.time()
    sup_n = None

    with torch.no_grad():
        for ii, (ri, s) in enumerate(items):
            gold_ops = list(s.get("ops") or [])
            payload = E34._payload_of(s)
            ops = []
            for k in range(E34.WALK_BUDGET):
                cache = E34._prefix(ctx, s, ops, payload)
                bc = E34.L32.BatchCpu32().build(cache, [0], 0, 0, 0)
                bc.rowIds = [0]
                cm, cosc, sinc = model.consts(bc.L, dev)
                mk = model.mask_of(cm, bc.B, bc.L, bc.lenB, dev)
                seq = model.assemble(bc.B, bc.L, bc, cache, dev)
                H = model.forward_body(seq, mk, cosc, sinc)
                lg, _zb, _tgt = model.heads(H, bc, cache, dev, allow)
                T = bc.Talloc
                fi = min(k, T - 1)
                # ---- h_t: the exact vector step_head consumes (pa_enable=0 => no delta) ----
                Hf = H.reshape(-1, H.shape[-1])
                idxH = torch.tensor(bc.idxH, dtype=torch.long, device=dev)
                idxS = torch.tensor(getattr(bc, "idxS", bc.idxC), dtype=torch.long,
                                    device=dev)
                x = torch.cat([Hf.index_select(0, idxH), Hf.index_select(0, idxS)], -1)
                if HID * 2 == x.shape[-1]:
                    lg_hat = F.linear(x.to(W.dtype), W, Bb)
                    max_diff = max(max_diff, float((lg_hat - lg).abs().max()))
                # ---- the decision distribution actually used (gen34_eval.py:386-392) ----
                z = lg.view(1, T, lg.shape[-1])[0][fi].clone()
                mask = torch.full_like(z, float("-inf"))
                mask[legal_idx] = 0.0
                mask[E34.END31] = 0.0
                z = z + mask
                p = torch.softmax(z, -1)
                ent = float(-(p * (p + 1e-12).log()).sum())
                pmax = float(p.max())
                top2 = torch.topk(p, 2)
                pred = int(z.argmax())
                sup_n = int(torch.isfinite(z).sum())
                gold_k = gold_ops[k] if k < len(gold_ops) else None
                hit = bool(gold_k is not None and pred != E34.END31
                           and names.get(pred) == gold_k)
                recs.append({
                    "ri": int(ri), "name": s["name"], "family": s.get("family"),
                    "k": k, "n_gold": len(gold_ops),
                    "pred": ("v31_end_marker" if pred == E34.END31
                             else names.get(pred, "?")),
                    "gold": gold_k, "hit": hit, "is_end": pred == E34.END31,
                    "p_max": round(pmax, 6),
                    "margin": round(float(top2.values[0] - top2.values[1]), 6),
                    "entropy": round(ent, 6),
                    "n_legal": sup_n,
                })
                H_all.append(x.view(1, T, -1)[0][fi].detach().to(torch.float32).cpu().numpy())
                if pred == E34.END31:
                    break
                ops.append(names.get(pred))
            exact = (ops == gold_ops)
            for r in recs:
                if r["ri"] == int(ri):
                    r["item_exact_hit"] = bool(exact)
            if (ii + 1) % 25 == 0 or ii + 1 == len(items):
                log("progress %d/%d  frames=%d  max|dF.linear|=%.2e  %.1f s"
                    % (ii + 1, len(items), len(recs), max_diff, time.time() - t0))
            del H, seq, lg, z, p

    Hm = np.stack(H_all)
    log("h_t stack %s  finite=%s" % (Hm.shape, bool(np.isfinite(Hm).all())))
    if max_diff > a.tol:
        log("FATAL: h_t is NOT the head input (max|diff|=%.3e > tol %.1e)" % (max_diff, a.tol))
        raise SystemExit(2)

    # ---------------------------------------------------------------- save ----
    npz = os.path.join(FC.DATA, a.out_tag + ".npz")
    np.savez_compressed(
        npz, h=Hm,
        ri=np.array([r["ri"] for r in recs], np.int32),
        k=np.array([r["k"] for r in recs], np.int16),
        n_gold=np.array([r["n_gold"] for r in recs], np.int16),
        family=np.array([r["family"] for r in recs]),
        p_max=np.array([r["p_max"] for r in recs], np.float32),
        margin=np.array([r["margin"] for r in recs], np.float32),
        entropy=np.array([r["entropy"] for r in recs], np.float32),
        hit=np.array([r["hit"] for r in recs], bool),
        is_end=np.array([r["is_end"] for r in recs], bool),
        item_exact=np.array([r["item_exact_hit"] for r in recs], bool),
        name=np.array([r["name"] for r in recs]),
        pred=np.array([r["pred"] for r in recs]),
    )
    log("npz -> %s (%.1f MB)" % (npz, os.path.getsize(npz) / 1e6))

    csvp = os.path.join(FC.DATA, "fig_hidden_entropy_frames.csv")
    with open(csvp, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["ri", "name", "family", "step_k", "n_gold_steps", "pred_op", "gold_op",
                    "hit", "is_end", "p_max", "margin", "entropy", "n_legal_support",
                    "item_exact_hit"])
        for r in recs:
            w.writerow([r["ri"], r["name"], r["family"], r["k"], r["n_gold"], r["pred"],
                        r["gold"], int(r["hit"]), int(r["is_end"]), r["p_max"], r["margin"],
                        r["entropy"], r["n_legal"], int(r["item_exact_hit"])])
    log("csv -> %s" % csvp)

    n_items = len({r["ri"] for r in recs})
    n_pass = len({r["ri"] for r in recs if r["item_exact_hit"]})
    meta = {
        "schema": "figllm-hidden-extract/1",
        "ckpt": a.ckpt, "ckpt_md5": C.md5f(a.ckpt),
        "dev": a.dev, "seg": a.seg,
        "state_shift": int(a.state_shift),
        "n_items_requested": int(a.n), "n_items_covered": n_items,
        "sampling": ("all items" if not a.max_items else "first %d items" % a.max_items),
        "n_frames": len(recs),
        "n_items_pass": n_pass, "n_items_fail": n_items - n_pass,
        "n_legal_support": sup_n,
        "h_dim": int(Hm.shape[1]),
        "verify": {"max_abs_Flinear_minus_head_logits": max_diff, "tol": a.tol,
                   "passed": bool(max_diff <= a.tol),
                   "note": ("h_t = concat(H[idxH], H[idxS]) reproduces the step logits the "
                            "model used; pa_enable=0 in this eval path so the plan-attention "
                            "delta is absent and the identity is exact up to fp error.")},
        "entropy_stats": {
            "all_mean": _mean_or_none([r["entropy"] for r in recs]),
            "pass_mean": _mean_or_none([r["entropy"] for r in recs
                                        if r["item_exact_hit"]]),
            "fail_mean": _mean_or_none([r["entropy"] for r in recs
                                        if not r["item_exact_hit"]]),
            "pass_n_frames": int(sum(1 for r in recs if r["item_exact_hit"])),
            "fail_n_frames": int(sum(1 for r in recs if not r["item_exact_hit"])),
        },
        "wall_s": round(time.time() - t0, 1),
        "model_info": {k: v for k, v in minfo.items()
                       if k in ("md5", "step", "missing", "unexpected")},
        "outputs": {"npz": npz, "csv": csvp},
    }
    mp = os.path.join(FC.DATA, "fig_llm_extract.json")
    with open(mp, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=1)
    log("meta -> %s" % mp)
    log("SUMMARY %s" % json.dumps({k: meta[k] for k in
                                   ("n_items_covered", "n_frames", "n_items_pass",
                                    "entropy_stats", "wall_s")}, ensure_ascii=False))


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
# gen44_barm.py -- family-B end-to-end certification arm (real LWB rows x exam-set rendering x
# frozen judging module, under `nice -n 19`)
#
# Measurement convention:
#   * Admission set = a **random sample** (fixed seed) of this family's accept domain
#     (rows with state in {certifiable, claim_uncert} in the census artifact)
#   * exam arm = single rendering (gen44_render_exam_v3 semantics, no enumeration)
#     -> exam-convention reading
#   * gold arm = bounded envelope (base + single-leaf deviation, cap 8)
#     -> device upper-bound reading (must be labelled as such)
#   * per-row trace omits the problem statement text; every reading carries n/total + Wilson 95%
import argparse
import collections
import json
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import gen44_bcore as B                                                  # noqa: E402
import gen44_bgate as G                                                  # noqa: E402


def load_claim(fam):
    p = os.path.join(B.OUT, "GEN44_%s_CENSUS.jsonl" % fam.upper())
    rows = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
    return [r for r in rows if r.get("state") in ("certifiable", "claim_uncert")]


def strat_sample(claim, n_cert, n_unc, seed):
    """Stratified sampling: draw n_cert from the device's self-claimed certifiable subset and
    n_unc from its self-claimed non-certifiable subset."""
    cert = [r for r in claim if r.get("state") == "certifiable"]
    unc = [r for r in claim if r.get("state") == "claim_uncert"]
    rng = random.Random(seed)
    sc = sorted(rng.sample(cert, min(n_cert, len(cert))), key=lambda r: r["idx"])
    su = sorted(rng.sample(unc, min(n_unc, len(unc))), key=lambda r: r["idx"])
    return sc + su


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fam", required=True, choices=("b1", "b2", "b6"))
    ap.add_argument("--mode", default="exam", choices=("exam", "gold"))
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--n-cert", type=int, default=24, dest="n_cert")
    ap.add_argument("--n-unc", type=int, default=8, dest="n_unc")
    ap.add_argument("--seed", type=int, default=20260926)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--max-lean-jobs", type=int, default=5)
    ap.add_argument("--gate-wait-s", type=float, default=900.0)
    ap.add_argument("--timeout", type=float, default=300.0)
    a = ap.parse_args()
    fam = a.fam
    up = fam.upper()
    claim = load_claim(fam)
    sel = strat_sample(claim, a.n_cert, a.n_unc, a.seed)
    sel = sel[a.shard::a.nshard]
    lwb = B.load_lwb()
    # lean workdir on the experiment host tree (SCOPE_RESEARCH_ROOT), not
    # inside the package source directory
    wd = os.path.join(B.RES, "scratch", "ccp", "gen44", "lean",
                      "barm_%s_%s" % (fam, a.mode))
    out, t0 = [], time.time()
    for i, r in enumerate(sel):
        lrec = lwb.get(r["idx"])
        ent = {"idx": r["idx"], "name": r["name"], "sub": r.get("sub"), "rs": r.get("rs"),
               "fam": fam, "mode": a.mode, "struct_state": r.get("state"),
               "kind_struct": r.get("kind")}
        if lrec is None:
            ent["state"] = "missing"
            out.append(ent)
            continue
        d = B.parse_stmt_ext(B.P(lrec["name"], lrec["head"], lrec["concl"],
                                 lrec.get("subset"), lrec["idx"]))
        if not d["ok"] or not G.fam_module(fam).claim(d["tree"]):
            ent["state"] = "not_claim_rerun"
            ent["reason"] = str(d.get("reason"))
            out.append(ent)
            continue
        texts = ([B.render_tree(d["tree"], "exam")] if a.mode == "exam"
                 else B.render_tree_multi(d["tree"], "gold", cap=8))
        texts = ["\n".join(t) for t in texts if t]
        if not texts:
            ent["state"] = "render_skip"
            out.append(ent)
            continue
        ent["n_text"] = len(texts)
        ent["ops"] = B.tree_ops(d["tree"]) + B.tree_leaf_ops(d["tree"])
        ok = False
        for ci, txt in enumerate(texts):
            jr = B.judge_nice(lrec["name"], lrec["head"], lrec["concl"], txt, wd,
                              "ba%s%s" % (fam, a.mode[0]), timeout=a.timeout,
                              maxj=a.max_lean_jobs, gate_wait_s=a.gate_wait_s)
            ent["cand_idx"] = ci
            ent["proof_md5"] = B.md5s(txt)
            ent["certified"] = bool(jr["certified"])
            ent["reasons"] = jr["reasons"]
            ent["lean"] = jr["lean"]
            ent["n1n2"] = jr["n1n2"]
            ent["state"] = "certified" if ent["certified"] else "judged_reject"
            if ent["certified"]:
                ok = True
                break
        out.append(ent)
        if (i + 1) % 5 == 0:
            print("[arm %s %s %d/%d] cert=%d %.0fs" % (
                fam, a.mode, i + 1, len(sel),
                sum(1 for x in out if x.get("certified")), time.time() - t0), flush=True)
            _flush(out, up, a)
    _flush(out, up, a)
    judged = [x for x in out if x.get("state") in ("certified", "judged_reject")]
    nc = sum(1 for x in judged if x["certified"])
    by_st = {}
    for stx in ("certifiable", "claim_uncert"):
        g = [x for x in judged if x.get("struct_state") == stx]
        k = sum(1 for x in g if x["certified"])
        by_st[stx] = {"n": len(g), "cert": k, "rate": round(k / max(1, len(g)), 4),
                      "wilson95": B.wilson(k, len(g))}
    rep = {"schema": "gen44-b-arm/1", "fam": fam, "mode": a.mode, "shard": a.shard,
           "nshard": a.nshard, "created": B.now_iso(), "n_claim_pop": len(claim),
           "n_sample": len(sel), "n_judged": len(judged), "n_certified": nc,
           "rate": round(nc / max(1, len(judged)), 4),
           "wilson95": B.wilson(nc, len(judged)),
           "state_hist": dict(collections.Counter(x.get("state") for x in out)),
           "reasons_hist": dict(collections.Counter(
               (x.get("reasons") or ["?"])[0] for x in judged if not x["certified"])),
           "by_struct_state": by_st,
           "by_rs": {k: {"n": sum(1 for x in judged if (x.get("rs") or "?") == k),
                         "cert": sum(1 for x in judged if (x.get("rs") or "?") == k
                                     and x["certified"])}
                     for k in sorted(set(x.get("rs") or "?" for x in judged))},
           "renderer": ("exam_single_v3" if a.mode == "exam"
                        else "gold_partial_envelope(cap8)"),
           # NOTE: Chinese report payload values kept verbatim (data, not log lines).
           "caveat": ("exam = 单渲染考试口径; gold = 有界包络(装置上界), 两列不得混用"
                      if a.mode == "gold" else "单渲染考试口径 (无枚举)"),
           "judge_md5": B.JUDGE_MD5, "wall_s": round(time.time() - t0, 1)}
    B.jdump(rep, os.path.join(B.OUT, "GEN44_%s_ARM_%s.json" % (up, a.mode.upper())))
    print(json.dumps({k: v for k, v in rep.items()}, ensure_ascii=False, indent=1))


def _flush(out, up, a):
    with open(os.path.join(B.OUT, "GEN44_%s_ROWS.jsonl.s%d" % (up, a.shard)),
              "w", encoding="utf-8") as f:
        for x in out:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()

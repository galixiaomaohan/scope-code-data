# -*- coding: utf-8 -*-
# gen36_probe_state.py -- routing-bottleneck probe (CPU): tail kind <=> device residual constant? what does the model input carry?
#   1) corpus + exam set: relation between the tail kind (zero/square/other) and the device residual (R_out constant rho)
#   2) model input channels: whether cache-row res/state/slots carry device state such as (ell/mult/R_in)
# Read-only; write scope is scratch/ccp/gen36/** (out/) only.
import argparse
import collections
import importlib.machinery
import importlib.util
import json
import os
import sys
from fractions import Fraction

sys.dont_write_bytecode = True

RES = os.environ.get("SCOPE_RESEARCH_ROOT", ".")
CCP = RES + "/scratch/ccp"
G36 = CCP + "/gen36"
G35, G34, G33, G32, G13 = (CCP + "/gen35", CCP + "/gen34", CCP + "/gen33",
                           CCP + "/gen32", CCP + "/gen13")

from math import isqrt                                              # noqa: E402


def is_perfect_square(n):
    if n < 0:
        return False
    r = isqrt(n)
    return r * r == n


def load(name, path):
    ld = importlib.machinery.SourceFileLoader(name, path)
    m = importlib.util.module_from_spec(importlib.util.spec_from_loader(name, ld))
    ld.exec_module(m)
    return m


def poly_const(dump, nv):
    """Polynomial dump -> constant term (Fraction). Keys are "i,j,k,..." exponent vectors."""
    if not dump:
        return None
    z = ",".join(["0"] * nv)
    v = dump.get(z)
    return Fraction(v) if v is not None else Fraction(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=G32 + "/DEEP_CORPUS_V1.jsonl")
    ap.add_argument("--exam-side", default=G34 + "/out/DEEP_EXAM_SIDECAR.jsonl")
    ap.add_argument("--cache", default=G33 + "/cache/deep_all.bin")
    ap.add_argument("--out", default=G36 + "/out/GEN36_STATE_PROBE.json")
    a = ap.parse_args()

    rep = {"schema": "gen36-state-probe/1"}

    # ---------- 1) tail kind <=> device residual constant ----------
    cls_conf = collections.Counter()
    rho_hist = collections.defaultdict(collections.Counter)
    n_cascade = 0
    for line in open(a.corpus, encoding="utf-8"):
        if not line.strip():
            continue
        d = json.loads(line)
        if d.get("family") != "gen32_ineq_cascade":
            continue
        n_cascade += 1
        m = d["meta"]
        nv = int(m["nv"])
        cls = m["cls"]
        # corpus steps[-1].args.rho = device residual constant (generator targs["rho"] = R.get(const))
        rho = Fraction(str((d["steps"][-1].get("args") or {}).get("rho", "0")))
        pred = ("zero" if rho == 0 else
                ("square" if is_perfect_square(int(rho)) else "other"))
        cls_conf["%s->%s" % (cls, pred)] += 1
        rho_hist[cls][str(rho)] += 1
    rep["A_corpus_tailcls_vs_rho"] = {"n": n_cascade,
                                      "map_confusion": dict(cls_conf.most_common()),
                                      "agree": sum(v for k, v in cls_conf.items()
                                                   if k.split("->")[0] == k.split("->")[1]),
                                      "rho_set_by_cls": {k: sorted(set(v))[:20]
                                                         for k, v in rho_hist.items()}}

    # ---------- 1b) exam side: same criterion (chain-walk result vs ground truth) ----------
    ex_rows = [json.loads(l) for l in open(a.exam_side, encoding="utf-8") if l.strip()]
    ec = [r for r in ex_rows if r.get("family") == "gen32_ineq_cascade"]
    econf = collections.Counter()
    ex_rho = {}
    for r in ec:
        sfa = r.get("sfa") or []
        tail_args = sfa[-1] if sfa else {}
        rho = Fraction(str(tail_args.get("rho", "0")))
        pred = ("zero" if rho == 0 else
                ("square" if is_perfect_square(int(rho)) else "other"))
        gold_op = (r.get("ops") or [None])[-1]
        gold_cls = gold_op.split("tail_")[-1] if gold_op else None
        econf["%s->%s" % (gold_cls, pred)] += 1
        ex_rho[r["name"]] = {"rho": str(rho), "cls_pred": pred, "cls_gold": gold_cls,
                             "cls_code": r.get("cls_code")}
    rep["A_exam_tailcls_vs_rho"] = {"n": len(ec), "map_confusion": dict(econf.most_common()),
                                    "agree": sum(v for k, v in econf.items()
                                                 if k.split("->")[0] == k.split("->")[1])}

    # ---------- 2) model input channels: cache row structure ----------
    CCF = load("g36_ccf", os.path.join(G13, "ccp_cachefull.py"))
    rows, hdr = CCF.read_cache(a.cache)
    seg = json.load(open(a.cache + ".segidx.json", encoding="utf-8"))
    rep["B_cache"] = {"n_rows": len(rows), "hdr": hdr["hdr"], "segkeys": list(seg.keys())[:10]}
    # find cascade rows (segidx / name mapping unavailable -> use statistics)
    n_res_present = 0
    n_steps_res = 0
    ex_row = None
    for i, R in enumerate(rows):
        rs = R.get("res") or []
        n_steps_res += len(rs)
        n_res_present += sum(1 for x in rs if x is not None)
        if ex_row is None and len(rs) >= 6 and any(x is not None for x in rs):
            ex_row = (i, R)
    rep["B_cache"]["n_step_frames"] = n_steps_res
    rep["B_cache"]["n_res_vectors_present"] = n_res_present
    rep["B_cache"]["res_channel_alive"] = bool(n_res_present > 0)
    if ex_row is not None:
        i, R = ex_row
        rep["B_cache"]["sample_row_with_res"] = {
            "i": i, "n_ids": len(R["ids"]), "labels": R["labels"][:12],
            "desc_head": R["desc"][:14], "res_head": [None if x is None else
                                                      [round(float(y), 4) for y in x]
                                                      for x in R["res"][:8]],
            "bsl_head": R["bsl"][:24], "state_pos": R.get("state_pos")}
    # bsl structure of one cascade row (n ~ 6..7 steps): slots = MS(24) per frame
    for i, R in enumerate(rows):
        nl = len(R["labels"])
        rs = R.get("res") or []
        if nl in (6, 7, 8) and len(rs) >= 5:
            rep["B_cache"]["sample_cascade_like"] = {
                "i": i, "n_labels": nl, "t4": R["t4"], "readout_only": R["readout_only"],
                "desc_head": R["desc"][:10], "feat_pos": R["feat_pos"],
                "res_all_none": all(x is None for x in rs),
                "bsl_len": len(R["bsl"]), "bsl_head": R["bsl"][:48],
                "n_state_pos": len(R.get("state_pos") or [])}
            break

    # ---------- 3) can visible features discriminate the tail kind? (coarse probe: constant term / block count / nv histogram) ----------
    feat = collections.defaultdict(lambda: collections.Counter())
    for line in open(a.corpus, encoding="utf-8"):
        if not line.strip():
            continue
        d = json.loads(line)
        if d.get("family") != "gen32_ineq_cascade":
            continue
        m = d["meta"]
        nv = int(m["nv"])
        dom = m.get("dom_dump")
        c0 = poly_const(dom, nv)
        rho = Fraction(str(m["rho"]))
        # visible quantities: dom constant term (visible in the problem text), block count, nv, D
        key = "c0_mod3=%d|nblk=%d|nv=%d" % (int(c0) % 3, m["n_blk"], nv)
        feat[key][m["cls"]] += 1
    # accuracy of the best coarse-feature rule (mode class per bucket) = a rough lower bound on the visible-feature ceiling
    tot, hit = 0, 0
    for k, c in feat.items():
        tot += sum(c.values())
        hit += max(c.values())
    rep["C_coarse_feature_ceiling"] = {
        "n": tot, "best_in_bucket": hit, "rate": round(hit / max(tot, 1), 6),
        "n_buckets": len(feat),
        "note": "可见粗特征 (dom 常数 mod 3 / 块数 / nv) 分桶取众数类的准确率; 仅作可见信息量的粗下界"}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=1)
    with open(G36 + "/out/GEN36_EXAM_TAILRHO.json", "w", encoding="utf-8") as f:
        json.dump(ex_rho, f, ensure_ascii=False, indent=1)
    print(json.dumps({k: rep[k] for k in ("A_corpus_tailcls_vs_rho", "A_exam_tailcls_vs_rho",
                                          "C_coarse_feature_ceiling")},
                     ensure_ascii=False, indent=1)[:4000])
    print(json.dumps(rep.get("B_cache", {}), ensure_ascii=False, indent=1)[:2500])


if __name__ == "__main__":
    main()

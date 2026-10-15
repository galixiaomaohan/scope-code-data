# ---- gen36 fork registry (literal/reverse dual-convention judging) ----
# This file is a **minimal-surface fork copy** of gen35_judge.py (md5 28238df3cb79eb53985b4191a38af7e4); parent untouched; judging logic unchanged.
# Differences: only [1] import gen36_common (write scope -> gen36/**).
# ---- gen35 fork registry ----
# -*- coding: utf-8 -*-
# gen35_judge.py -- node [5] **dual-convention judging**: full batch (already done inside each arm file)
#   + certified stratified literal re-judging (>=60) + 12 reverse-verification cases
#   (R1 terminal truncation / R2 cross-problem proof swap => expect rejection).
#   C5 rule: exception != disagreement (agree is defined only when both sides have a definite verdict).
#   Reverse pool: this arm's certified pool; if empty, follow the gen33 precedent and substitute the
#   **gold substrate** (compiler upper-bound pool) under the same convention.
# Writes only under scratch/ccp/gen35/**. Whole pipeline nice -n 19; frozen artifacts untouched (md5 measured before and after).
import argparse
import collections
import json
import os
import random
import sys
import time

import gen36_common as C

OUT = C.G36 + "/out"
LEAN = C.G36 + "/lean"


def log(msg, p=None):
    C.log(msg, p)


def load_rows(path):
    if path.endswith(".jsonl"):
        return C.jlload(path)
    d = json.load(open(path, encoding="utf-8"))
    return d["rows"] if isinstance(d, dict) else d


def _side(rec):
    if rec is None:
        return "not_executed", "no_record"
    if rec.get("exc"):
        return "exception", rec["exc"]
    if "certified" not in rec:
        return "not_executed", "missing_certified"
    lean = rec.get("lean")
    if isinstance(lean, dict) and lean.get("rc") in (124, 127):
        return "exception", "lean_infra_rc=%s" % lean.get("rc")
    return (("certified", rec.get("reasons") or []) if rec.get("certified")
            else ("rejected", rec.get("reasons") or []))


def literal_judge(items, wd, logp=None):
    """Literal convention (direct DC.judge_attempt calls; frozen artifact), compared case by case against batch verdicts."""
    import drct_common as DC
    os.makedirs(wd, exist_ok=True)
    recs = []
    for x in items:
        t0 = time.time()
        try:
            j = DC.judge_attempt(x["name"], x["head"], x["concl"], x["proof_text"], wd,
                                 "lit35_%s" % x.get("arm_tag", "arm"))
        except Exception as e:                                        # noqa: BLE001
            j = {"certified": None, "reasons": [], "lean": None, "n1n2": None,
                 "exc": "%s: %s" % (type(e).__name__, e)}
        lv, lw = _side(j)
        bv, bw = _side({"certified": x.get("certified"),
                        "reasons": x.get("judge_reasons")} if "certified" in x else None)
        if lv in ("certified", "rejected") and bv in ("certified", "rejected"):
            cls = "agree" if lv == bv else "disagree"
            agree = (lv == bv)
        elif "exception" in (lv, bv):
            cls, agree = "exception", None
        elif "not_executed" in (lv, bv):
            cls, agree = "not_executed", None
        else:
            cls, agree = "unknown", None
        recs.append({"name": x["name"], "family": x.get("family"), "arm_tag": x.get("arm_tag"),
                     "literal_verdict": lv, "batch_verdict": bv,
                     "literal_certified": (bool(j.get("certified")) if lv in
                                           ("certified", "rejected") else None),
                     "batch_certified": x.get("certified"),
                     "xc_class": cls, "agree": agree, "literal_reasons": j.get("reasons"),
                     "literal_lean": j.get("lean"), "literal_exc": j.get("exc"),
                     "wall_s": round(time.time() - t0, 2)})
        log("LITERAL %s literal=%s batch=%s class=%s agree=%s %.0fs"
            % (x["name"], lv, bv, cls, agree, recs[-1]["wall_s"]), logp)
    n_def = sum(1 for r in recs if r["agree"] is not None)
    rep = {"schema": "gen35-literal/1", "n": len(recs), "n_defined": n_def,
           "agree": sum(1 for r in recs if r["agree"] is True),
           "disagree": sum(1 for r in recs if r["agree"] is False),
           "exception": sum(1 for r in recs if r["xc_class"] == "exception"),
           "not_executed": sum(1 for r in recs if r["xc_class"] == "not_executed"),
           "literal_certified": sum(1 for r in recs if r["literal_certified"] is True),
           "batch_certified": sum(1 for r in recs if r["batch_certified"] is True),
           "by_family": {}, "records": recs}
    for r in recs:
        f = rep["by_family"].setdefault(r["family"] or "?", {"n": 0, "agree": 0,
                                                             "literal_cert": 0})
        f["n"] += 1
        f["agree"] += 1 if r["agree"] else 0
        f["literal_cert"] += 1 if r["literal_certified"] else 0
    return rep


def reverse_judge(pool, n=12, seed=20260923, logp=None):
    """R1 terminal truncation / R2 cross-problem proof swap => expect rejection (judging is non-trivial, self-evidenced)."""
    import drct_common as DC
    rnd = random.Random(seed)
    pool = sorted(pool, key=lambda x: x["name"])
    if len(pool) < 2:
        return {"n": 0, "note": "池不足 2, 反向核验不可做 (如实登记)", "records": []}
    picks = rnd.sample(pool, min(n, len(pool)))
    wd = os.path.join(LEAN, "reverse")
    os.makedirs(wd, exist_ok=True)
    recs = []

    def judge(name, head, concl, proof, tag):
        try:
            j = DC.judge_attempt(name, head, concl, proof, wd, tag)
            return bool(j.get("certified")), j.get("reasons"), j.get("lean")
        except Exception as e:                                        # noqa: BLE001
            return None, ["EXC:%s" % e], None

    for i, x in enumerate(picks):
        lines = [l for l in (x["proof_text"] or "").split("\n") if l.strip()]
        keep = max(1, int(len(lines) * 0.6))
        trunc = "\n".join(lines[:keep])
        c1, r1, _l1 = judge(x["name"], x["head"], x["concl"], trunc, "rev_trunc")
        y = picks[(i + 1) % len(picks)]
        c2, r2, _l2 = judge(x["name"], x["head"], x["concl"], y["proof_text"], "rev_swap")
        recs.append({"name": x["name"], "n_lines": len(lines), "n_keep": keep,
                     "R1_trunc_certified": c1, "R1_reasons": r1,
                     "R2_swap_from": y["name"], "R2_swap_certified": c2, "R2_reasons": r2})
        log("REVERSE %s R1_trunc=%s R2_swap=%s (src=%s)"
            % (x["name"], c1, c2, y["name"]), logp)
    return {"schema": "gen35-reverse/1", "n": len(recs),
            "R1_trunc_rejected": sum(1 for r in recs if r["R1_trunc_certified"] is False),
            "R2_swap_rejected": sum(1 for r in recs if r["R2_swap_certified"] is False),
            "R1_trunc_unexpectedly_certified": [r["name"] for r in recs
                                                if r["R1_trunc_certified"] is not False],
            "R2_swap_unexpectedly_certified": [r["name"] for r in recs
                                               if r["R2_swap_certified"] is not False],
            "records": recs}


def strat_sample(rows, n, seed=20260923):
    """Stratified (family) sampling; take everything if short."""
    rnd = random.Random(seed)
    by = collections.defaultdict(list)
    for r in rows:
        by[r.get("family") or "?"].append(r)
    fams = sorted(by)
    per = max(1, n // max(1, len(fams)))
    out = []
    for f in fams:
        pool = sorted(by[f], key=lambda x: x["name"])
        out.extend(rnd.sample(pool, min(per, len(pool))))
    if len(out) < n:
        rest = [r for r in rows if r not in out]
        out.extend(rnd.sample(rest, min(n - len(out), len(rest))))
    return out[:max(n, len(out))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="", help="rows jsonl (must contain name/head/concl/proof_text)")
    ap.add_argument("--src-tag", default="pure_plan")
    ap.add_argument("--pool", default="certified", choices=("certified", "all", "first_hit"))
    ap.add_argument("--n-literal", type=int, default=66)
    ap.add_argument("--n-reverse", type=int, default=12)
    ap.add_argument("--fallback-src", default="", help="gold-substrate source used when the pool is empty (same-convention substitute)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--log", default=C.G36 + "/logs/gen36_judge.log")
    a = ap.parse_args()
    log("== gen35 judge src=%s pool=%s n_literal=%d n_reverse=%d =="
        % (a.src, a.pool, a.n_literal, a.n_reverse), a.log)
    C.lean_gate(5, a.log)
    jm0 = C.md5f(C.DRCT + "/drct_common.py")
    assert jm0 == C.JUDGE_MD5_EXPECT, jm0
    rows = load_rows(a.src) if a.src else []
    for r in rows:
        r["arm_tag"] = a.src_tag
    sel = {"certified": [r for r in rows if r.get("certified")],
           "first_hit": [r for r in rows if r.get("first_hit")],
           "all": rows}[a.pool]
    rep = {"schema": "gen35-judge/1",
           "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "judge_md5_before": jm0, "src": a.src, "src_tag": a.src_tag,
           "src_n": len(rows), "pool": a.pool, "pool_n": len(sel),
           "judge_entry": "drct_common (冻结 7c31b58d): gen23_prescreen_batch (批量) / "
                          "DC.judge_attempt (字面, 反向)",
           "c5_rule": "异常 != 分歧; agree 仅在双侧均确定判决时定义"}
    lit_pool, lit_note = sel, "本臂池"
    if not sel and a.fallback_src:
        fb = load_rows(a.fallback_src)
        for r in fb:
            r["arm_tag"] = "gold_substrate"
        lit_pool = [r for r in fb if r.get("certified")]
        lit_note = ("本臂 certified 池=0 ⇒ 按 gen33 先例改用**金标底物**池 (编译器上界) "
                    "同式替代 (如实登记)")
    rep["literal_pool_note"] = lit_note
    rep["literal_pool_n"] = len(lit_pool)
    if lit_pool:
        lit = strat_sample(lit_pool, a.n_literal)
        rep["literal"] = literal_judge(lit, os.path.join(LEAN, "literal_%s" % a.src_tag), a.log)
        rep["reverse"] = reverse_judge(lit_pool, a.n_reverse, 20260923, a.log)
    else:
        rep["literal"] = {"n": 0, "note": "无可用池 (如实登记)"}
        rep["reverse"] = {"n": 0, "note": "无可用池 (如实登记)"}
    jm1 = C.md5f(C.DRCT + "/drct_common.py")
    rep["judge_md5_after"] = jm1
    rep["judge_unchanged"] = bool(jm0 == jm1)
    rep["out_md5"] = C.save_json(a.out, rep)
    print(json.dumps({"out": a.out, "md5": rep["out_md5"],
                      "literal_n": rep["literal"].get("n"),
                      "literal_agree": rep["literal"].get("agree"),
                      "literal_disagree": rep["literal"].get("disagree"),
                      "literal_certified": rep["literal"].get("literal_certified"),
                      "R1_rejected": rep["reverse"].get("R1_trunc_rejected"),
                      "R2_rejected": rep["reverse"].get("R2_swap_rejected")},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()

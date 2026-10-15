# ---- gen36 fork registry (judging/compiler payload port) ----
# This file is a **minimal-surface fork copy** of gen35_armrun.py (md5 bc22d1a69f2e02ade627b46db0b3aa1c); parent untouched.
# Differences: [1] import gen36_common; [2] LEAN work dir -> gen36/lean (write-scope discipline);
#   the compiler still imports gen35_compile (frozen read-only).
# ---- gen35 fork registry ----
# -*- coding: utf-8 -*-
# gen35_armrun.py -- shared executor for the four-arm comparison (compiler x judging)
#   Arm 2: gold chain x compile (compiler ceiling) . Arm 3: gen34 A-arm chain x compile (stock chain)
#   Arm 4: pure-planning arm chain x compile (node [4])
#   Also: AUD-1 zero-parameter chain mode baseline (corpus distribution -> exam set; no exam labels)
# Imports the frozen judging chain without modifying it; writes only under scratch/ccp/gen35/**.
import collections
import json
import os
import threading
import time

import gen36_common as C
import gen35_compile as CP

OUT = C.G35 + "/out"
LEAN = C.G36 + "/lean"


def load_exam():
    return C.jlload(C.EXAM_SIDECAR)


def load_gen34_chains(arm="slotA"):
    """gen34 chain walks (chunked EVAL artifacts) -> {ri: {walked, gold, exact_hit, first_hit, name}}"""
    out = {}
    for f in sorted(os.listdir(C.G34 + "/out")):
        pre = "GEN34_EVAL_%s_exam_deep_c" % arm
        if not (f.startswith(pre) and f.endswith(".json")):
            continue
        d = json.load(open(C.G34 + "/out/" + f, encoding="utf-8"))
        for r in d["rows"]:
            ch = r.get("chain") or {}
            out[int(r["ri"])] = {"name": r["name"], "walked": list(ch.get("walked") or []),
                                 "gold": list(ch.get("gold") or []),
                                 "exact_hit": bool(ch.get("exact_hit")),
                                 "first_hit": bool(ch.get("first_hit")),
                                 "overrun": bool(ch.get("overrun")),
                                 "terminated_end": bool(ch.get("terminated_end")),
                                 "arch_src": "gen34_%s" % arm}
    return out


def load_gen35_chains(path):
    """gen35 chain-walk artifacts (GEN35_EVAL_*.json or jsonl) -> {ri: {...}}"""
    out = {}
    if path.endswith(".json"):
        d = json.load(open(path, encoding="utf-8"))
        rows = d["rows"] if isinstance(d, dict) else d
    else:
        rows = C.jlload(path)
    for r in rows:
        ch = r.get("chain") or {}
        out[int(r["ri"])] = {"name": r["name"], "walked": list(ch.get("walked") or []),
                             "gold": list(ch.get("gold") or []),
                             "exact_hit": bool(ch.get("exact_hit")),
                             "first_hit": bool(ch.get("first_hit")),
                             "overrun": bool(ch.get("overrun")),
                             "terminated_end": bool(ch.get("terminated_end")),
                             "arch_src": r.get("arm") or "gen35"}
    return out


def compile_arm(recs, chains, tag=""):
    """Per row: chain -> compiler -> proof text + per-step trace. chains={ri: {...}}; missing chain => empty chain."""
    rows = []
    err_hist = collections.Counter()
    fam_stat = collections.defaultdict(lambda: {"n": 0, "exact": 0, "first": 0,
                                                "n_emit": 0, "n_mismatch": 0})
    for rec in recs:
        ri = int(rec["ri"])
        ch = chains.get(ri) or {"walked": [], "gold": rec.get("ops") or []}
        pv = CP.pv_of_exam(rec)
        res = CP.compile_chain(pv, ch["walked"], keep_recs=True)
        fam = rec.get("family")
        f = fam_stat[fam]
        f["n"] += 1
        f["exact"] += 1 if res["exact_hit"] else 0
        f["first"] += 1 if res["first_hit"] else 0
        f["n_emit"] += res["n_emit"]
        f["n_mismatch"] += res["n_mismatch"]
        for k, v in (res["err_hist"] or {}).items():
            err_hist[k] += v
        rows.append({"ri": ri, "name": rec["name"], "family": fam, "cls": rec.get("cls"),
                     "cls_code": rec.get("cls_code"), "tag": tag,
                     "head": rec["head"], "concl": rec["concl"],
                     "proof_text": res["proof_text"], "n_lines": res["n_lines"],
                     "proof_md5": res["proof_md5"],
                     "gold_ops": list(rec.get("ops") or []), "walked": list(ch["walked"]),
                     "first_hit": res["first_hit"], "exact_hit": res["exact_hit"],
                     "over_run": res["over_run"], "n_emit": res["n_emit"],
                     "n_mismatch": res["n_mismatch"], "err_hist": res["err_hist"],
                     "chain_src": ch.get("arch_src"),
                     "walk_exact_hit": ch.get("exact_hit"),
                     "walk_first_hit": ch.get("first_hit"),
                     "step_errs": [{"k": s["k"], "op": s["op"], "err": s.get("err")}
                                   for s in res["steps"] if s.get("err")]})
    stat = {"n": len(rows), "n_exact": sum(1 for r in rows if r["exact_hit"]),
            "n_first": sum(1 for r in rows if r["first_hit"]),
            "n_overrun": sum(1 for r in rows if r["over_run"]),
            "err_hist": dict(err_hist),
            "fam": {k: dict(v) for k, v in fam_stat.items()},
            "empty_proof": sum(1 for r in rows if not r["proof_text"].strip())}
    stat["exact_rate"] = round(stat["n_exact"] / max(1, stat["n"]), 6)
    stat["first_rate"] = round(stat["n_first"] / max(1, stat["n"]), 6)
    return rows, stat


def serial_batch(items, wd, chunk=60, tag="gen35"):
    """Batch judging (frozen drct_common semantics; serial + nice -n 19 comes from run_lean_full)."""
    import drct_common as DC
    import gen23_prescreen_batch as PBATCH
    jm = C.md5f(DC.__file__)
    if jm != C.JUDGE_MD5_EXPECT:
        raise SystemExit("FATAL 判卷件非冻结版 %s" % jm)
    lock = threading.Lock()
    orig = PBATCH.run_lean_full

    def ser(*a, **k):
        with lock:
            return orig(*a, **k)
    PBATCH.run_lean_full = ser
    try:
        out = PBATCH.judge_items_batch(items, wd, chunk=chunk, tag=tag, judge_md5=jm)
    finally:
        PBATCH.run_lean_full = orig
    return out, jm


def judge_rows(rows, tag, chunk=60, logp=None):
    wd = os.path.join(LEAN, "batch_%s" % tag)
    os.makedirs(wd, exist_ok=True)
    items = [{"name": r["name"], "head": r["head"], "concl": r["concl"],
              "proof_text": r["proof_text"]} for r in rows]
    t0 = time.time()
    jr, jm = serial_batch(items, wd, chunk=chunk, tag=tag)
    by = {x["name"]: x for x in jr}
    for r in rows:
        x = by.get(r["name"]) or {}
        r["certified"] = bool(x.get("certified"))
        r["judge_reasons"] = x.get("reasons")
        r["judge_md5"] = jm
    cert = sum(1 for r in rows if r["certified"])
    stat = {"n": len(rows), "certified": cert, "rate": round(cert / max(1, len(rows)), 6),
            "judge_md5": jm, "wall_s": round(time.time() - t0, 1),
            "reason_hist": dict(collections.Counter(
                x for r in rows if not r["certified"] for x in (r.get("judge_reasons") or []))),
            "fam": {}}
    for r in rows:
        f = stat["fam"].setdefault(r["family"], {"n": 0, "certified": 0, "exact": 0})
        f["n"] += 1
        f["certified"] += 1 if r["certified"] else 0
        f["exact"] += 1 if r["exact_hit"] else 0
    C.log("JUDGE_ARM %s certified=%d/%d (%.4f) wall=%.0fs"
          % (tag, cert, len(rows), stat["rate"], stat["wall_s"]), logp)
    return stat


def aud1_zero_param(exam_recs, corpus_path=None):
    """AUD-1 zero-parameter chain mode baseline: per-family mode chain from the training corpus,
    applied to the exam set => no exam labels used."""
    corpus = C.jlload(corpus_path or C.DEEP_CORPUS)
    mode = {}
    byfam = collections.defaultdict(collections.Counter)
    for r in corpus:
        byfam[r["family"]][tuple(r["chain_ops"])] += 1
    for fam, c in byfam.items():
        mode[fam] = list(c.most_common(1)[0][0])
    rows, per_pos = [], {}
    for rec in exam_recs:
        pred = mode.get(rec.get("family")) or []
        gold = list(rec.get("ops") or [])
        cmp_n = min(len(pred), len(gold))
        rows.append({"name": rec["name"], "family": rec.get("family"), "pred": pred,
                     "gold": gold, "exact": bool(pred == gold),
                     "first": bool(pred and gold and pred[0] == gold[0]),
                     "prefix": bool(cmp_n and pred[:cmp_n] == gold[:cmp_n]),
                     "n_gold": len(gold)})
        for k in range(cmp_n):
            d = per_pos.setdefault(k, [0, 0])
            d[1] += 1
            d[0] += 1 if pred[k] == gold[k] else 0
    return {"schema": "gen35-aud1/1",
            "note": "零参数 (族众数链, 取自训练语料分布) 应用于考卷; 不用考卷标签",
            "mode_by_family": {k: v for k, v in mode.items()},
            "n": len(rows), "n_exact": sum(1 for r in rows if r["exact"]),
            "exact_rate": round(sum(1 for r in rows if r["exact"]) / max(1, len(rows)), 6),
            "n_first": sum(1 for r in rows if r["first"]),
            "first_rate": round(sum(1 for r in rows if r["first"]) / max(1, len(rows)), 6),
            "n_prefix": sum(1 for r in rows if r["prefix"]),
            "prefix_rate": round(sum(1 for r in rows if r["prefix"]) / max(1, len(rows)), 6),
            "per_pos_acc": {str(k): round(v[0] / max(1, v[1]), 6) for k, v in per_pos.items()},
            "rows": rows}

# -*- coding: utf-8 -*-
# gen33_judge.py -- gen33 judging (frozen module drct_common 7c31b58d imported as-is, zero modification)
#
# Three conventions (V33_DESIGN §8):
#   batch   : batch full pass (multi-declaration three-file set main/n1/n2) -- frozen judging semantics
#             certified := lean ∧ ¬(n1_accepted ∧ ¬n1_vacuous) ∧ n2
#   literal : **literal-convention** re-judging of a certified stratified sample >= 60 (direct DC.judge_attempt)
#   reverse : 12 reverse cases -- (R1) terminal truncation, (R2) cross-problem proof swap (statement A with proof B)
#             ⇒ expected rejection (demonstrates judging is non-trivial; if it still certifies = judging is broken, must be reported)
#
# Discipline: pre-launch global count of our Lean jobs < 5 (path attribution: cmdline contains 'lake env lean' ∧
#   tcaboost or /.elan) · whole pipeline under nice -n 19 · frozen module untouched (md5 measured before and after)
#   · C5 rule inherited: exception != disagreement (agree is defined only when both sides give a definite verdict).
# Writes only scratch/ccp/gen33/**.
import argparse
import collections
import json
import os
import random
import time

import g33_common as C

G33 = C.G33
OUT = os.path.join(G33, "out")
LEAN = os.path.join(G33, "lean")


def log(msg, p=None):
    C.log(msg, p)


def load_evals(tag_prefixes):
    """Collect chunked EVAL artifacts for an arm (or arms) -> {name: {head, concl, body, arm_tag}}"""
    recs = {}
    for tag in tag_prefixes:
        for f in sorted(os.listdir(OUT)):
            if not (f.startswith("GEN33_EVAL_%s_exam_deep_c" % tag) and f.endswith(".json")):
                continue
            d = json.load(open(os.path.join(OUT, f), encoding="utf-8"))
            side = {}
            for line in open(os.path.join(OUT, "DEEP_EXAM_SIDECAR.jsonl"), encoding="utf-8"):
                if line.strip():
                    r = json.loads(line)
                    side[int(r["ri"])] = r
            for r in d["rows"]:
                s = side[r["ri"]]
                recs[r["name"]] = {"name": r["name"], "cls": r["cls"],
                                   "head": s["head"], "concl": s["concl"],
                                   "proof_text": (r.get("render") or {}).get("body") or "",
                                   "raw_text": (r.get("render") or {}).get("text") or "",
                                   "arm_tag": tag, "chain": r.get("chain"),
                                   "family": s.get("family"),
                                   "chain_len": s.get("chain_len")}
    return recs


def batch_judge(items, tag, chunk=60, logp=None):
    """Batch full pass (via gen33_eval.judge_texts => frozen gen23_prescreen_batch => drct_common)."""
    import gen33_eval as EV
    jm = C.md5f(os.path.join(C.DRCT, "drct_common.py"))
    if jm != C.JUDGE_MD5_EXPECT:
        raise SystemExit("FATAL 判卷件非冻结版 %s" % jm)
    jt = [{"name": x["name"], "head": x["head"], "concl": x["concl"],
           "proof_text": x["proof_text"]} for x in items]
    wd = os.path.join(LEAN, "batch_%s" % tag)
    os.makedirs(wd, exist_ok=True)
    return EV.judge_texts(jt, wd, chunk=chunk, tag="gen33_%s" % tag)


def literal_judge(items, trace, logp=None, n_workers=1):
    """Literal-convention re-judging (direct DC.judge_attempt; C5 rule: exception != disagreement)."""
    import drct_common as DC
    wd = os.path.join(LEAN, "literal")
    os.makedirs(wd, exist_ok=True)
    XC_DEF = ("certified", "rejected")

    def side(rec):
        if rec is None:
            return "not_executed", "no_record"
        if rec.get("exc"):
            return "exception", rec["exc"]
        if "certified" not in rec:
            return "not_executed", "missing_certified"
        lean = rec.get("lean")
        if isinstance(lean, dict) and lean.get("rc") in (124, 127):
            return "exception", "lean_infra_rc=%s" % lean.get("rc")
        return ("certified", rec.get("reasons") or []) if rec.get("certified") \
            else ("rejected", rec.get("reasons") or [])

    tr = {}
    for l in open(trace, encoding="utf-8"):
        if l.strip():
            r = json.loads(l)
            tr[r["name"]] = r
    recs = []
    for x in items:
        t0 = time.time()
        try:
            j = DC.judge_attempt(x["name"], x["head"], x["concl"], x["proof_text"], wd,
                                 "lit_%s" % x["arm_tag"])
        except Exception as e:                                        # noqa: BLE001
            j = {"certified": None, "reasons": [], "lean": None, "n1n2": None,
                 "exc": "%s: %s" % (type(e).__name__, e)}
        lv, lw = side(j)
        b = tr.get(x["name"])
        bv, bw = side(b)
        if lv in XC_DEF and bv in XC_DEF:
            cls = "agree" if lv == bv else "disagree"
            agree = (lv == bv)
        elif "exception" in (lv, bv):
            cls, agree = "exception", None
        elif "not_executed" in (lv, bv):
            cls, agree = "not_executed", None
        else:
            cls, agree = "unknown", None
        recs.append({"name": x["name"], "arm_tag": x["arm_tag"], "cls": x["cls"],
                     "chain_len": x["chain_len"],
                     "literal_verdict": lv, "batch_verdict": bv,
                     "literal_certified": (bool(j.get("certified")) if lv in XC_DEF
                                           else None),
                     "batch_certified": (bool(b.get("certified")) if bv in XC_DEF
                                         and b is not None else None),
                     "xc_class": cls, "agree": agree, "literal_reasons": j.get("reasons"),
                     "literal_lean": j.get("lean"), "literal_exc": j.get("exc"),
                     "wall_s": round(time.time() - t0, 2)})
        log("LITERAL %s literal=%s batch=%s class=%s agree=%s %.0fs"
            % (x["name"], lv, bv, cls, agree, recs[-1]["wall_s"]), logp)
    return recs


def reverse_judge(cert_pool, n=12, seed=20260923, logp=None):
    """Reverse check: (R1) terminal truncation, (R2) cross-problem proof swap => expected rejection."""
    import drct_common as DC
    rnd = random.Random(seed)
    pool = sorted(cert_pool, key=lambda x: x["name"])
    if len(pool) < 2:
        return {"n": 0, "note": "certified 池不足 2, 反向核验不可做 (如实登记)"}
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
        c1, r1, l1 = judge(x["name"], x["head"], x["concl"], trunc, "rev_trunc")
        y = picks[(i + 1) % len(picks)]
        c2, r2, l2 = judge(x["name"], x["head"], x["concl"], y["proof_text"], "rev_swap")
        recs.append({"name": x["name"], "n_lines": len(lines), "n_keep": keep,
                     "R1_trunc_certified": c1, "R1_reasons": r1,
                     "R2_swap_from": y["name"], "R2_swap_certified": c2, "R2_reasons": r2})
        log("REVERSE %s R1_trunc=%s R2_swap=%s (src=%s)"
            % (x["name"], c1, c2, y["name"]), logp)
    r1_no = sum(1 for r in recs if r["R1_trunc_certified"] is False)
    r2_no = sum(1 for r in recs if r["R2_swap_certified"] is False)
    return {"schema": "gen33-reverse/1", "n": len(recs),
            "R1_trunc_rejected": r1_no, "R2_swap_rejected": r2_no,
            "R1_trunc_unexpectedly_certified": [r["name"] for r in recs
                                                if r["R1_trunc_certified"] is not False],
            "R2_swap_unexpectedly_certified": [r["name"] for r in recs
                                               if r["R2_swap_certified"] is not False],
            "records": recs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True,
                    choices=("batch", "literal", "reverse", "all"))
    ap.add_argument("--tags", default="base,attn")
    ap.add_argument("--n-literal", type=int, default=66, help=">=60")
    ap.add_argument("--n-reverse", type=int, default=12)
    ap.add_argument("--chunk", type=int, default=60)
    ap.add_argument("--out", default="")
    ap.add_argument("--log", default=G33 + "/logs/gen33_judge.log")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.log), exist_ok=True)
    C.lean_gate(5, a.log)
    jm0 = C.md5f(os.path.join(C.DRCT, "drct_common.py"))
    assert jm0 == C.JUDGE_MD5_EXPECT, jm0
    tags = [t.strip() for t in a.tags.split(",") if t.strip()]
    rep = {"schema": "gen33-judge/1", "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "judge_md5_before": jm0, "tags": tags,
           "judge_entry": "drct_common (冻结 7c31b58d) via gen33_eval.judge_texts "
                          "(batch) / DC.judge_attempt (literal, reverse)",
           "c5_rule": "异常 != 分歧; agree 仅在双侧均确定判决时定义"}
    all_items = []
    per_arm = {}
    for tg in tags:
        recs = load_evals([tg])
        per_arm[tg] = recs
        all_items += list(recs.values())
        log("loaded %s: n=%d (with rendered body %d)" % (tg, len(recs),
                                          sum(1 for x in recs.values() if x["proof_text"])),
            a.log)
    if a.mode in ("batch", "all"):
        # Per-arm judging (clear trace attribution)
        traces = {}
        for tg in tags:
            items = list(per_arm[tg].values())
            if not items:
                continue
            jr = batch_judge(items, tg, chunk=a.chunk, logp=a.log)
            tp = os.path.join(OUT, "GEN33_PRESCREEN_TRACE_%s.jsonl" % tg)
            C.jlsave(jr["raw"], tp)
            traces[tg] = {"path": tp, "md5": C.md5f(tp), "n": len(jr["raw"]),
                          "certified": jr["certified"], "judge_md5": jr["judge_md5"],
                          "certified_rate": round(jr["certified"] / max(1, len(jr["raw"])),
                                                  6)}
            log("BATCH %s certified=%d/%d (%.4f) trace=%s"
                % (tg, jr["certified"], len(jr["raw"]),
                   traces[tg]["certified_rate"], tp), a.log)
        rep["batch"] = traces
    if a.mode in ("literal", "all"):
        lit = []
        for tg in tags:
            tp = os.path.join(OUT, "GEN33_PRESCREEN_TRACE_%s.jsonl" % tg)
            if not os.path.exists(tp):
                continue
            pool = sorted(per_arm[tg].values(), key=lambda x: (x["cls"] or "",
                                                              x["chain_len"] or 0, x["name"]))
            # Stratify by (cls, chain_len)
            strata = collections.defaultdict(list)
            for x in pool:
                strata["%s|%s" % (x["cls"], x["chain_len"])].append(x)
            rnd = random.Random(20260923)
            per = max(1, a.n_literal // max(1, len(strata)))
            samp = []
            for k in sorted(strata):
                samp += rnd.sample(strata[k], min(per, len(strata[k])))
            samp = samp[:max(a.n_literal, 60)]
            lit += literal_judge(samp, tp, logp=a.log)
        n_def = sum(1 for r in lit if r["agree"] is not None)
        rep["literal"] = {"n": len(lit), "n_per_stratum": None,
                          "n_agree": sum(1 for r in lit if r["agree"]),
                          "n_disagree": sum(1 for r in lit if r["xc_class"] == "disagree"),
                          "n_exception": sum(1 for r in lit if r["xc_class"] == "exception"),
                          "n_not_executed": sum(1 for r in lit
                                                if r["xc_class"] == "not_executed"),
                          "n_verdict_pairs": n_def,
                          "agree_rate": (round(sum(1 for r in lit if r["agree"])
                                              / max(1, n_def), 6) if n_def else None),
                          "literal_certified": sum(1 for r in lit
                                                   if r["literal_certified"] is True),
                          "records": lit}
    if a.mode in ("reverse", "all"):
        cert_pool = []
        for tg in tags:
            tp = os.path.join(OUT, "GEN33_PRESCREEN_TRACE_%s.jsonl" % tg)
            if not os.path.exists(tp):
                continue
            for l in open(tp, encoding="utf-8"):
                if l.strip():
                    r = json.loads(l)
                    if r.get("certified") and per_arm[tg].get(r["name"], {}).get("proof_text"):
                        x = dict(per_arm[tg][r["name"]])
                        cert_pool.append(x)
        rep["reverse"] = reverse_judge(cert_pool, n=a.n_reverse, logp=a.log)
    rep["judge_md5_after"] = C.md5f(os.path.join(C.DRCT, "drct_common.py"))
    rep["judge_unchanged"] = bool(rep["judge_md5_after"] == jm0 == C.JUDGE_MD5_EXPECT)
    outp = a.out or os.path.join(OUT, "GEN33_JUDGE_%s.json" % a.mode)
    rep["md5"] = C.save_json(outp, rep)
    log("JUDGE %s DONE md5=%s judge_unchanged=%s" % (a.mode, rep["md5"],
                                                   rep["judge_unchanged"]), a.log)
    C.heartbeat("judge_%s" % a.mode, "md5=%s" % rep["md5"])
    print("GEN33_JUDGE_RC=0 -> %s" % outp)
    print(json.dumps({k: v for k, v in rep.items()
                      if k in ("batch", "literal", "reverse", "judge_unchanged")},
                     ensure_ascii=False)[:1500])


if __name__ == "__main__":
    main()

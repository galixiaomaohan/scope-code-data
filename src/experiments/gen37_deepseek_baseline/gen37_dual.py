# -*- coding: utf-8 -*-
# gen37_dual.py -- gen37 external baseline [4]: dual-convention judging (certified stratified literal re-judging >=60 + reverse checks 12)
#   Literal re-judging: stratified sample from the certified pool, re-judge each case via DC.judge_attempt,
#           compare with the batch verdict (C5: exception != disagreement; agree is only defined when both
#           sides give a definite verdict).
#   Reverse checks: R1 terminal truncation (keep first 60% of lines) / R2 cross-problem proof swap
#           => expected not certified (the judge module self-certifies non-trivially).
#   Empty or small pool: registered by stratum (N3: not executed != scored zero).
# Discipline: writes only under scratch/ccp/gen37/**; whole run under nice -n 19; frozen module untouched
# (md5 measured before and after).
import argparse
import collections
import hashlib
import json
import os
import random
import sys
import time

RES = os.environ.get("SCOPE_RESEARCH_ROOT", ".")
G37 = RES + "/scratch/ccp/gen37"
DRCT = RES + "/scratch/drct"
JUDGE_MD5_EXPECT = "7c31b58da9d7228cee31b9be581fe5dd"


def md5f(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def app(msg, logp=None):
    line = "[%s] %s" % (time.strftime("%Y-%m-%dT%H:%M:%S%z"), msg)
    print(line, flush=True)
    if logp:
        with open(logp, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def strat_sample(rows, n, seed=20260923):
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


def _side(rec):
    if rec is None:
        return "not_executed", "no_record"
    if rec.get("exc"):
        return "exception", rec["exc"]
    if "certified" not in rec or rec.get("certified") is None:
        return "not_executed", "missing_certified"
    lean = rec.get("lean")
    if isinstance(lean, dict) and lean.get("rc") in (124, 127):
        return "exception", "lean_infra_rc=%s" % lean.get("rc")
    return (("certified", rec.get("reasons") or []) if rec.get("certified")
            else ("rejected", rec.get("reasons") or []))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default=G37 + "/out/GEN37_PROMPTS.jsonl")
    ap.add_argument("--attempts", default=G37 + "/out/GEN37_DSRL_attempts.jsonl")
    ap.add_argument("--records", default=G37 + "/out/GEN37_JUDGE_records.jsonl")
    ap.add_argument("--out", default=G37 + "/out/GEN37_DUAL.json")
    ap.add_argument("--n-literal", type=int, default=66)
    ap.add_argument("--n-reverse", type=int, default=12)
    ap.add_argument("--log", default=G37 + "/logs/gen37_dual.log")
    a = ap.parse_args()

    sys.path.insert(0, DRCT)
    import drct_common as DC
    jm0 = md5f(DRCT + "/drct_common.py")
    assert jm0 == JUDGE_MD5_EXPECT, "frozen judge module md5 mismatch: %s" % jm0
    with open(a.prompts, encoding="utf-8") as f:
        probs = {r["name"]: r for r in (json.loads(l) for l in f if l.strip())}
    with open(a.attempts, encoding="utf-8") as f:
        atts = {r["name"]: r for r in (json.loads(l) for l in f if l.strip())}
    with open(a.records, encoding="utf-8") as f:
        recs = [json.loads(l) for l in f if l.strip()]
    rows = []
    for r in recs:
        p, t = probs[r["problem"]], atts.get(r["problem"]) or {}
        rows.append({"name": r["problem"], "ri": r["ri"], "family": r["family"],
                     "cls": r["cls"], "head": p["head"], "concl": p["concl"],
                     "proof_text": DC.extract_proof(t.get("text") or ""),
                     "certified": r["certified"], "judge_reasons": r["reasons"],
                     "lean": r["lean"], "judge_wall_s": r["judge_wall_s"]})
    pool = [r for r in rows if r["certified"]]
    rep = {"schema": "gen37-dual/1", "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "judge_md5_before": jm0, "n_rows": len(rows), "pool": "certified",
           "pool_n": len(pool), "n_literal_req": a.n_literal, "n_reverse_req": a.n_reverse,
           "c5_rule": "exception != disagreement; agree is defined only when both sides give a definite verdict",
           "spec": "certified stratified literal re-judging >=60 + reverse 12 (task spec §6); if the pool is small, registered by stratum"}
    app("DUAL start rows=%d pool=%d" % (len(rows), len(pool)), a.log)
    # ---- literal re-judging ----
    wd = os.path.join(G37, "lean", "dual")
    os.makedirs(wd, exist_ok=True)
    if pool:
        sel = strat_sample(pool, a.n_literal)
        lit = []
        for x in sel:
            t0 = time.time()
            try:
                j = DC.judge_attempt(x["name"], x["head"], x["concl"], x["proof_text"], wd,
                                     "lit37_a0")
                exc = None
            except Exception as e:                                   # noqa: BLE001
                j = {"certified": None, "reasons": [], "lean": None}
                exc = "%s: %s" % (type(e).__name__, e)
            lv, _ = _side(j if not exc else {"exc": exc})
            bv, _ = _side({"certified": x["certified"], "reasons": x["judge_reasons"],
                           "lean": x["lean"]})
            if lv in ("certified", "rejected") and bv in ("certified", "rejected"):
                cls, agree = ("agree" if lv == bv else "disagree"), (lv == bv)
            elif "exception" in (lv, bv):
                cls, agree = "exception", None
            else:
                cls, agree = "not_executed", None
            lit.append({"name": x["name"], "family": x["family"], "literal_verdict": lv,
                        "batch_verdict": bv, "literal_certified": (bool(j.get("certified"))
                        if lv in ("certified", "rejected") else None),
                        "batch_certified": x["certified"], "xc_class": cls, "agree": agree,
                        "literal_reasons": j.get("reasons"), "literal_exc": exc,
                        "wall_s": round(time.time() - t0, 2)})
            app("LITERAL %s lit=%s batch=%s cls=%s agree=%s %.0fs"
                % (x["name"], lv, bv, cls, agree, lit[-1]["wall_s"]), a.log)
        ndef = sum(1 for r in lit if r["agree"] is not None)
        try:
            from collections import Counter
            pop = Counter()
            for r in lit:
                pop[r["family"]] += 1
        except Exception:                                            # noqa: BLE001
            pop = {}
        rep["literal"] = {
            "n": len(lit), "n_req": a.n_literal, "rounding": "stratified (family) sampling; take all if fewer",
            "n_defined": ndef, "agree": sum(1 for r in lit if r["agree"] is True),
            "disagree": sum(1 for r in lit if r["agree"] is False),
            "exception": sum(1 for r in lit if r["xc_class"] == "exception"),
            "not_executed": sum(1 for r in lit if r["xc_class"] == "not_executed"),
            "literal_certified": sum(1 for r in lit if r["literal_certified"] is True),
            "batch_certified": sum(1 for r in lit if r["batch_certified"] is True),
            "fam_pick_counts": dict(pop), "records": lit}
    else:
        rep["literal"] = {"n": 0, "note": "certified pool = 0: literal re-judging not possible (N3 not executed != scored zero; registered as-is)"}
        app("LITERAL skipped: pool=0", a.log)
    # ---- reverse checks ----
    if len(pool) >= 2:
        rnd = random.Random(20260923)
        picks = rnd.sample(sorted(pool, key=lambda x: x["name"]), min(a.n_reverse, len(pool)))
        rwd = os.path.join(G37, "lean", "reverse")
        os.makedirs(rwd, exist_ok=True)
        rev = []
        for i, x in enumerate(picks):
            lines = [l for l in (x["proof_text"] or "").split("\n") if l.strip()]
            keep = max(1, int(len(lines) * 0.6))
            trunc = "\n".join(lines[:keep])
            y = picks[(i + 1) % len(picks)]

            def jd(name, head, concl, proof, tag):
                try:
                    j = DC.judge_attempt(name, head, concl, proof, rwd, tag)
                    return bool(j.get("certified")), j.get("reasons"), None
                except Exception as e:                               # noqa: BLE001
                    return None, ["EXC:%s" % e], "%s" % e
            c1, r1, e1 = jd(x["name"], x["head"], x["concl"], trunc, "rev_trunc37")
            c2, r2, e2 = jd(x["name"], x["head"], x["concl"], y["proof_text"], "rev_swap37")
            rev.append({"name": x["name"], "n_lines": len(lines), "n_keep": keep,
                        "R1_trunc_certified": c1, "R1_reasons": r1, "R1_exc": e1,
                        "R2_swap_from": y["name"], "R2_swap_certified": c2, "R2_reasons": r2,
                        "R2_exc": e2})
            app("REVERSE %s R1=%s R2=%s (src=%s)" % (x["name"], c1, c2, y["name"]), a.log)
        rep["reverse"] = {
            "n": len(rev), "n_req": a.n_reverse,
            "R1_trunc_rejected": sum(1 for r in rev if r["R1_trunc_certified"] is False),
            "R2_swap_rejected": sum(1 for r in rev if r["R2_swap_certified"] is False),
            "R1_unexpectedly_certified": [r["name"] for r in rev if r["R1_trunc_certified"] is not False],
            "R2_unexpectedly_certified": [r["name"] for r in rev if r["R2_swap_certified"] is not False],
            "records": rev}
    else:
        rep["reverse"] = {"n": 0, "note": "pool < 2: reverse checks not possible (registered as-is)"}
        app("REVERSE skipped: pool<2", a.log)
    rep["judge_md5_after"] = md5f(DRCT + "/drct_common.py")
    rep["judge_unchanged"] = bool(rep["judge_md5_before"] == rep["judge_md5_after"])
    open(a.out, "w", encoding="utf-8").write(json.dumps(rep, ensure_ascii=False, indent=1))
    app("DUAL done lit=%s rev=%s unchanged=%s"
        % (rep["literal"].get("n"), rep["reverse"].get("n"), rep["judge_unchanged"]), a.log)
    print(json.dumps({"out": a.out, "literal": rep["literal"].get("n"),
                      "literal_agree": rep["literal"].get("agree"),
                      "literal_disagree": rep["literal"].get("disagree"),
                      "literal_certified": rep["literal"].get("literal_certified"),
                      "R1_rejected": rep["reverse"].get("R1_trunc_rejected"),
                      "R2_rejected": rep["reverse"].get("R2_swap_rejected"),
                      "md5": md5f(a.out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
# gen37_selftest.py -- gen37 CPU-side self-test [judging pipeline + statement convention] (zero GPU)
#   Case A: a gen35 M1_F certified proof (expect certified=True) — verifies the frozen module + statement
#           head/concl can be judged certified;
#   Case B: a gen35 M1_F rejected proof (expect certified=False) — verifies the rejection path;
#   Case C: prompt-file first problem's prompt vs the render_lean rendered statement, literal cross-check
#           (zero gap on the statement side).
# Reads existing files only; writes only under scratch/ccp/gen37/**.
import argparse
import hashlib
import json
import os
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=G37 + "/logs/gen37_selftest.log")
    a = ap.parse_args()
    sys.path.insert(0, DRCT)
    import drct_common as DC
    jm = md5f(DRCT + "/drct_common.py")
    assert jm == JUDGE_MD5_EXPECT, jm
    prows = {}
    with open(G37 + "/out/GEN37_PROMPTS.jsonl", encoding="utf-8") as f:
        for l in f:
            if l.strip():
                r = json.loads(l)
                prows[r["name"]] = r
    g35 = []
    with open(RES + "/scratch/ccp/gen35/out/GEN35_EVAL_PLAN_ROWS.jsonl", encoding="utf-8") as f:
        for l in f:
            if l.strip():
                g35.append(json.loads(l))
    wd = G37 + "/lean/selftest"
    os.makedirs(wd, exist_ok=True)
    out = {"schema": "gen37-selftest/1", "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "judge_md5": jm, "cases": []}
    for want in (True, False):
        x = next(r for r in g35 if r.get("certified") is want)
        p = prows[x["name"]]
        t0 = time.time()
        j = DC.judge_attempt(x["name"], p["head"], p["concl"], x["proof_text"], wd, "st37")
        rec = {"case": "A_certified_expect_true" if want else "B_rejected_expect_false",
               "name": x["name"], "gen35_certified": want, "judge_certified": j.get("certified"),
               "reasons": j.get("reasons"), "lean": j.get("lean"), "n1n2": j.get("n1n2"),
               "wall_s": round(time.time() - t0, 2)}
        rec["PASS"] = (j.get("certified") is want)
        out["cases"].append(rec)
        app("SELFTEST %s name=%s certified=%s (want %s) %.1fs"
            % (rec["case"], x["name"], j.get("certified"), want, rec["wall_s"]), a.log)
    # C: literal statement cross-check
    p0 = prows[g35[0]["name"]]
    rend = DC.render_lean(p0["name"], p0["head"], p0["concl"], "  <PROOF_BODY_SENTINEL>")
    stmt_head = ("%s %s := by" % (p0["head"], p0["concl"]))
    hit = stmt_head in rend
    out["cases"].append({"case": "C_statement_identity", "name": p0["name"],
                         "prompt_statement_equals_render": bool(hit),
                         "prompt_tail": p0["prompt"][-120:]})
    out["PASS_all"] = all(c.get("PASS", c.get("prompt_statement_equals_render"))
                          for c in out["cases"])
    open(G37 + "/out/GEN37_SELFTEST.json", "w", encoding="utf-8").write(
        json.dumps(out, ensure_ascii=False, indent=1))
    app("SELFTEST done PASS_all=%s" % out["PASS_all"], a.log)
    print(json.dumps(out, ensure_ascii=False)[:1200])


if __name__ == "__main__":
    main()

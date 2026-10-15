# -*- coding: utf-8 -*-
# ccp_gate_d.py -- course-chain proof check-up (CCP version of gate d from v2; advance team gen12, change 58)
#
# Reads the per-problem proof bodies in scratch/ccp/gen/ccp_<cls>_chains.jsonl, renders them
# (official header + statement + proof + #print axioms) and hands them to Lean:
# rc=0 AND no sorryAx AND no sorry dependency. **This is the only hard evidence that every
# line in the course really has a Lean proof**: generation time only re-verifies the
# certificate (arithmetic) independently; Lean is the final court (v2 lesson: certificate
# PASS != proof body compiles).
# Usage: python scratch/ccp/ccp_gate_d.py --cls ind [--max 400] [--all]
import argparse
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ccp_lean as L

RES = os.environ.get("SCOPE_RESEARCH_ROOT", ".")
CCP = os.path.join(RES, "scratch", "ccp")
GEN = os.path.join(CCP, "gen")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cls", required=True)
    ap.add_argument("--max", type=int, default=400)
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    cp = os.path.join(GEN, "ccp_%s_chains.jsonl" % a.cls)
    rows = [json.loads(ln) for ln in open(cp, encoding="utf-8")]
    if not a.all and len(rows) > a.max:
        step = max(1, len(rows) // a.max)
        samp = rows[::step][:a.max]
    else:
        samp = rows
    jobs = [(r["name"], r["stmt"], "\n".join(r["proof"]), "chain", "gated_" + a.cls)
            for r in samp]
    res = L.parallel_jobs(jobs)
    ok = [n for (n, _, _, _, _) in jobs if (res.get((n, "chain")) or {}).get("ok")]
    fails = [(n, (res.get((n, "chain")) or {}).get("head", "")[:300])
             for (n, _, _, _, _) in jobs if not (res.get((n, "chain")) or {}).get("ok")]
    rep = {"schema": "ccp-gate-d/1", "cls": a.cls, "n_course_problems": len(rows),
           "n_checked": len(jobs), "n_ok": len(ok), "n_fail": len(fails),
           "sampling": "all" if a.all or len(rows) <= a.max else "every %d-th up to %d" % (
               max(1, len(rows) // a.max), a.max),
           "fails": fails[:20],
           "all_ok": len(fails) == 0}
    op = os.path.join(GEN, "CCP_GATE_D_%s.json" % a.cls.upper())
    rep["md5"] = L.save_json(op, rep)
    print("[gate-d %s] %d/%d proof bodies rc=0 (course problems %d) → %s md5=%s" %
          (a.cls, len(ok), len(jobs), len(rows), op, rep["md5"]), flush=True)
    for n, h in fails[:5]:
        print("  [FAIL] %s %s" % (n, h.replace("\n", " ")[:200]), flush=True)
    return rep


if __name__ == "__main__":
    main()

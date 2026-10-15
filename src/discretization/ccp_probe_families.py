# -*- coding: utf-8 -*-
# ccp_probe_families.py -- problem-family feasibility probe (advance team gen12, constitution change 58)
# Purpose: before bulk generation, measure each candidate family for (1) full-chain proof body Lean rc=0 and
# (2) eight-tactic bare-tactic floor hardness.
# Discipline: writes only scratch/ccp/lean/probe/**; reads miniF2F only; results land in scratch/ccp/PROBE_FAMILIES.json.
import json
import os
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ccp_families as F
import ccp_lean as L

RES = os.environ.get("SCOPE_RESEARCH_ROOT", ".")
CCP = os.path.join(RES, "scratch", "ccp")
MASTER = 20260923
N_PER_FAM = int(sys.argv[1]) if len(sys.argv) > 1 else 3


def main():
    t0 = time.time()
    probs, errs = [], Counter()
    for cls in F.CLASSES:
        for family in F.PROBE_FAMS[cls]:
            got = 0
            seq = 0
            while got < N_PER_FAM and seq < N_PER_FAM * 60:
                seq += 1
                pr, err = F.make_problem(cls, family, seq, MASTER)
                if pr is None:
                    errs["%s/%s:%s" % (cls, family, err)] += 1
                    continue
                probs.append(pr)
                got += 1
    print("[probe] generated %d problems | failure reasons %s" % (len(probs), dict(errs)), flush=True)
    # listing print (human-reviewable: statement + chain + proof)
    for pr in probs:
        print("---- %s [%s] ops=%s" % (pr["name"], pr["family"], [s["op"] for s in pr["chain"]]))
        print("     STMT: %s" % pr["stmt"].replace("\n", " ⏎ "))
        print("     G1  : %s" % pr["g1"][:200])
        print("     PROOF: %s" % " ;; ".join(pr["proof"])[:220])
    # (1) full chain
    t1 = time.time()
    cres = L.chain_check([(p["name"], p["stmt"], "\n".join(p["proof"])) for p in probs],
                         subdir="probe_chain")
    chain_ok = {k: v["ok"] for k, v in cres.items()}
    print("[probe] full chain rc=0: %d/%d | %.0fs" % (sum(chain_ok.values()), len(probs),
                                                     time.time() - t1), flush=True)
    for p in probs:
        r = cres[p["name"]]
        if not r["ok"]:
            print("   [FAIL] %s rc=%s head=%s" % (p["name"], r["rc"], r["head"][:300]))
    # (1b) combined single probe (first | menu...) cross-checked against (1)
    t2 = time.time()
    first_proof = ("first\n" + "\n".join("  | (%s; dbg_trace \"CCPFLOOR %s\")" % (t, t)
                                         for t in L.MENU))
    fres = L.chain_check([(p["name"], p["stmt"], first_proof) for p in probs],
                         subdir="probe_first")
    print("[probe] combined single probe (first) rc=0: %d/%d | %.0fs" %
          (sum(1 for v in fres.values() if v["ok"]), len(probs), time.time() - t2), flush=True)
    # (2) eight-tactic floor
    t3 = time.time()
    fl = L.floor_probe([(p["name"], p["stmt"]) for p in probs], subdir="probe_floor")
    n_hard = sum(1 for v in fl.values() if not v["closers"])
    print("[probe] floor: %d/%d hard | %.0fs" % (n_hard, len(probs), time.time() - t3), flush=True)
    by_fam = {}
    for p in probs:
        f = p["family"]
        d = by_fam.setdefault(f, {"n": 0, "hard": 0, "chain_ok": 0, "first_ok": 0, "closers": Counter()})
        d["n"] += 1
        d["hard"] += 0 if fl[p["name"]]["closers"] else 1
        d["chain_ok"] += 1 if chain_ok[p["name"]] else 0
        d["first_ok"] += 1 if fres[p["name"]]["ok"] else 0
        for c in fl[p["name"]]["closers"]:
            d["closers"][c] += 1
    out = {"n_per_fam": N_PER_FAM, "master": MASTER, "n_problems": len(probs),
           "chain_ok": sum(chain_ok.values()), "first_ok": sum(1 for v in fres.values() if v["ok"]),
           "floor_hard": n_hard,
           "by_family": {k: {"n": v["n"], "hard": v["hard"], "chain_ok": v["chain_ok"],
                             "first_ok": v["first_ok"], "closers": dict(v["closers"])}
                         for k, v in by_fam.items()},
           "gen_errors": dict(errs),
           "chain_fail_head": {k: v["head"][:400] for k, v in cres.items() if not v["ok"]},
           "first_fail_head": {k: v["head"][:200] for k, v in fres.items()
                               if v["ok"] != chain_ok[k]},
           "total_secs": round(time.time() - t0, 1)}
    md5 = L.save_json(os.path.join(CCP, "PROBE_FAMILIES.json"), out)
    print("[probe] report → scratch/ccp/PROBE_FAMILIES.json md5=%s | total %.0fs" % (md5, time.time() - t0))
    print("[probe] by family: %s" % json.dumps({k: {kk: vv for kk, vv in v.items()}
                                                for k, v in out["by_family"].items()},
                                               ensure_ascii=False))


if __name__ == "__main__":
    main()

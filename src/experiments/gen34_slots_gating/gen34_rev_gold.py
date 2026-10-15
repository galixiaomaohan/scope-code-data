# -*- coding: utf-8 -*-
# gen34_rev_gold.py -- reverse check on the gold substrate, 12 cases (formula-for-formula
# identical to gen33 CTRL2.reverse_gold)
#
# Purpose: when the model's certified pool is smaller than 2 (the same pathology as gen33),
#   the reverse check is **infeasible on model outputs** => run the same-form reverse check
#   on the **gold substrate** (proof_text of corpus/exam rows, which itself must be
#   certified=True). The goal is unchanged = show the judge is **non-trivial**
#   (R1 terminal truncation must be rejected / R2 cross-problem proof swap must be rejected).
# The frozen drct_common is import-only, zero edits (measured before and after judging).
# Writes only scratch/ccp/gen34/**.
import argparse
import json
import os
import random
import sys
import time

import gen34_common as C

G34, G33 = C.G34, C.G33
OUT = os.path.join(G34, "out")
LEAN = os.path.join(G34, "lean")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--out", default=OUT + "/GEN34_JUDGE_reverse_gold.json")
    ap.add_argument("--log", default=G34 + "/logs/gen34.log")
    a = ap.parse_args()
    C.lean_gate(5, a.log)
    sys.path.insert(0, C.DRCT)
    import drct_common as DC
    jm0 = C.md5f(DC.__file__)
    assert jm0 == C.JUDGE_MD5_EXPECT, jm0
    C.log("REV-GOLD judge module frozen md5=%s" % jm0, a.log)
    items = {it["name"]: it for it in json.load(open(C.EXAM_CAND, encoding="utf-8"))["items"]}
    rows = [r for r in C.jlload(C.DEEP_CORPUS) if r["name"] in items]
    rows = rows[:164]                       # adjacent slice of the held-out segment
                                            # (a_held has 237 rows; take a large-enough pool)
    rnd = random.Random(20260923)
    pool = sorted(rows, key=lambda x: x["name"])
    picks = rnd.sample(pool, min(a.n, len(pool)))
    wd = os.path.join(LEAN, "revgold")
    os.makedirs(wd, exist_ok=True)
    recs = []

    def judge(name, head, concl, proof, tag):
        try:
            j = DC.judge_attempt(name, head, concl, proof, wd, tag)
            return bool(j.get("certified")), j.get("reasons"), j.get("lean")
        except Exception as e:                                       # noqa: BLE001
            return None, ["EXC:%s" % e], None
    t0 = time.time()
    for i, x in enumerate(picks):
        lines = [l for l in (x["proof_text"] or "").split("\n") if l.strip()]
        keep = max(1, int(len(lines) * 0.6))
        trunc = "\n".join(lines[:keep])
        g0, gr0, _ = judge(x["name"], x["head"], x["concl"], x["proof_text"], "g_sub")
        c1, r1, _ = judge(x["name"], x["head"], x["concl"], trunc, "rev_trunc")
        y = picks[(i + 1) % len(picks)]
        c2, r2, _ = judge(x["name"], x["head"], x["concl"], y["proof_text"], "rev_swap")
        recs.append({"name": x["name"], "n_lines": len(lines), "n_keep": keep,
                     "gold_certified": g0, "gold_reasons": gr0,
                     "R1_trunc_certified": c1, "R1_reasons": r1,
                     "R2_swap_from": y["name"], "R2_swap_certified": c2, "R2_reasons": r2,
                     "wall_s": round(time.time() - t0, 1)})
        C.log("REV-GOLD %s gold=%s R1_trunc=%s R2_swap=%s (src=%s)"
              % (x["name"], g0, c1, c2, y["name"]), a.log)
    rep = {"schema": "gen34-reverse-gold/1", "n": len(recs),
           "gold_certified": sum(1 for r in recs if r["gold_certified"] is True),
           "R1_trunc_rejected": sum(1 for r in recs if r["R1_trunc_certified"] is False),
           "R2_swap_rejected": sum(1 for r in recs if r["R2_swap_certified"] is False),
           "R1_trunc_unexpectedly_certified": [r["name"] for r in recs
                                               if r["R1_trunc_certified"] is not False],
           "R2_swap_unexpectedly_certified": [r["name"] for r in recs
                                              if r["R2_swap_certified"] is not False],
           "substrate": "gold corpus proofs (exam rows included; certified=True required)",
           "why": ("model certified pool smaller than 2 => reversing on 'model outputs' is "
                   "infeasible (registered as such); the same-form reverse check on the gold "
                   "substrate has an unchanged goal = show the judge is non-trivial"),
           "judge_md5_before": jm0, "records": recs}
    rep["judge_md5_after"] = C.md5f(DC.__file__)
    rep["judge_unchanged"] = bool(rep["judge_md5_after"] == jm0)
    rep["md5"] = C.save_json(a.out, rep)
    C.log("REV-GOLD DONE n=%d gold=%d R1_rej=%d R2_rej=%d judge_unchanged=%s"
          % (rep["n"], rep["gold_certified"], rep["R1_trunc_rejected"],
             rep["R2_swap_rejected"], rep["judge_unchanged"]), a.log)
    print("GEN34_REV_GOLD_RC=0 -> %s" % a.out)
    return rep


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
# ccp_diag.py -- bridge-rule / problem-family diagnostics (read-only print, writes no artifacts)
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ccp_bridge as BR
import ccp_families as F
import ccp_ops as OPS

cls = sys.argv[1] if len(sys.argv) > 1 else "ineq"
family = sys.argv[2] if len(sys.argv) > 2 else "ineq_diff2"
print("=== %s / %s ===" % (cls, family))
for seq in range(1, 4):
    pr, err = F.make_problem(cls, family, seq, 20260923)
    if pr is None:
        print(" [%d] GEN FAIL: %s" % (seq, err))
        # manual walk-through to localize the failure
        import random
        rng = random.Random(1)
        p = F.gen_params(cls, family, rng)
        print("     params=%s" % p)
        if cls == "ineq":
            try:
                nv = p["nvars"]
                S = F.poly_of_squares(nv, p["squares"], 0)
                zero = (0,) * nv
                S1 = {m: c for m, c in S.items() if m != zero}
                D = 3
                K = S.get(zero, 0) + p["rho"] + D
                LHS = dict(S1)
                LHS[zero] = LHS.get(zero, 0) + K
                txt = F.render_poly(LHS, p["vars"])
                stmt = "theorem T (%s : ℚ) :\n    %s ≥ %d" % (" ".join(p["vars"]), txt, D)
                print("     STMT: %s" % stmt.replace("\n", " / "))
                print("     BRIDGE: %s" % BR.bridge_ineq(stmt, p["vars"])["dom"])
                print("     PLAN  : %s" % {m: c for m, c in S.items()})
            except Exception:
                traceback.print_exc()
        if cls == "nt":
            try:
                m = p["m"]
                stmt = ("theorem T (a b : ℕ) (ha : %d ∣ a) (hb : %d ∣ b) :\n    %d ∣ a * b"
                        % (m, m, m * m))
                print("     STMT: %s" % stmt.replace("\n", " / "))
                print("     BRIDGE: %s" % BR.bridge_nt_prod(stmt))
            except Exception:
                traceback.print_exc()
        continue
    print(" [%d] %s" % (seq, pr["stmt"].replace("\n", " / ")))
    print("      ops=%s" % [s["op"] for s in pr["chain"]])
    print("      proof=%s" % " ;; ".join(pr["proof"]))

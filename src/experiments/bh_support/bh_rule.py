# -*- coding: utf-8 -*-
# bh_rule.py -- bridge-hard batch rule baseline arms (gen14; zero model / zero training / zero GPU)
#
# Four arms (all emit only "operator name / class / candidate index"; arguments always come from the
# device/bridge, consistent with the change-49 convention):
#   RB0    20-line lexical rules (statement regexes + constant-filled templates) -- the "key family"
#          isomorphic to gen12 lexrule
#   RBx    reasonable extended static rules: top-level parse + unfolding + **local** O(1) predicates +
#          template chains;
#          **forbidden**: executing chain steps / reading intermediate state / curriculum-core
#          computation (completing-the-square search / recurrence closed-form solving / dependency
#          closure and witness construction / polynomial square-root decision) -- convention in
#          BH_DESIGN.md (pre-registered)
#   RBs    state-aware shallow rules (may read device-computed intermediate objects; readings disclosed)
#   RBplay device self-play reference (= bh_env.play_program; upper bound = 1, disclosed per batch)
#
# Criterion: "solved" = this arm's operator sequence runs to a closed chain on the **device** and every
# step's certificate passes (bh_env.run_program).
import re
from fractions import Fraction

import bh_bridge as BR
import bh_env as ENV
import bh_ops as OPS
import bh_poly as P


# ================================================================ RB0: 20-line lexical rules
def rb0(cls, stmt):
    t = stmt
    if cls == "ind":
        if re.search(r"f \(k \+ 1\) = f k \+ \d+\)", t):
            return [{"op": "bh_ind_sum_d0"}, {"op": "bh_ind_close_poly"}]
        if re.search(r"f \(k \+ 1\) = \d+ \* f k [+-] \d+\)", t):
            return [{"op": "bh_ind_geo"}, {"op": "bh_ind_close_pow"}]
        return [{"op": "bh_ind_sum_d1"}, {"op": "bh_ind_close_poly"}]
    if cls == "ineq":
        return [{"op": None, "cand": 0}, {"op": None, "cand": 0}]
    hyp = re.findall(r"\((\w+) : [0-9+\-*^() ]+? \u2223 \w+\)", t)
    return ([{"op": None, "cand": i} for i in range(len(hyp))]
            + [{"op": None, "cand": 0}])


# ================================================================ RBx: reasonable extended static rules
def rbx(cls, stmt):
    """Top-level parse + unfolding + local predicates + template chains (forbidden: curriculum-core
    computation / reading intermediate state)."""
    if cls == "ind":
        try:
            lf = BR.bridge_ind_linform(stmt)
        except Exception:                                      # noqa: BLE001
            return [{"op": "bh_ind_sum_d1"}, {"op": "bh_ind_close_poly"}]
        if lf["kind"] == "additive":
            S = P.pscale(lf["d"], Fraction(-1) / lf["a"])
            deg = max([m[0] for m in S] or [0])
            deg = min(3, max(0, deg))
            return [{"op": "bh_ind_sum_d%d" % deg}, {"op": "bh_ind_close_poly"}]
        # Geometric family: class determination needs the fixed point solved (forbidden) => lexical default
        # (both declared forms in this batch contain '^ n' + the same factor)
        return [{"op": "bh_ind_geo"}, {"op": "bh_ind_close_pow"}]
    if cls == "ineq":
        # Completing-the-square search = curriculum-core computation (forbidden) => can only emit "1 block + default terminal class"
        return [{"op": None, "cand": 0}, {"op": None, "cand": 0}]
    # nt: dependency closure / witness construction (forbidden) => gen12-isomorphic: expand hypotheses one by one + close
    hyps = re.findall(r"\((\w+) : [0-9+\-*^() ]+? \u2223 (\w+)\)", stmt)
    return ([{"op": None, "cand": i} for i in range(len(hyps))]
            + [{"op": None, "cand": 0}])


# ================================================================ RBs: state-aware shallow rules
def _quad(R, vars_, v):
    i = vars_.index(v)
    q = Fraction(0)
    for mono, c in R.items():
        if mono[i] == 2 and sum(mono) == 2:
            q += c
    return q


def _is_sq(x):
    try:
        n = int(x)
    except Exception:                                          # noqa: BLE001
        return False
    if n <= 0:
        return False
    r = int(round(n ** 0.5))
    return r * r == n


def _match_block(pr, cs, R):
    """Shallow block choice: the univariate candidate whose residual quadratic coefficient == mult*a^2
    (local arithmetic match); otherwise the first viable one."""
    m = pr["meta"]
    vars_ = m["vars"]
    for i, c in enumerate(cs):
        a_ = c.get("args") or {}
        ell = a_.get("ell")
        if not ell or len(ell["coefs"]) != 1:
            continue
        v = list(ell["coefs"])[0]
        a = Fraction(ell["coefs"][v])
        mult = Fraction(a_.get("mult", 1))
        q = _quad(R, vars_, v)
        if q != 0 and mult * a * a == q:
            return i
    for i, c in enumerate(cs):
        if c.get("viable"):
            return i
    return 0


def rbs_program(cls, pr):
    """State-aware shallow-rule program (reads device-computed intermediate objects)."""
    if cls == "ind":
        prog = []
        for k in range(2):
            cs = pr["cands"][k]
            if k == 0:
                ops = [c["op"] for c in cs]
                pick = None
                for pref in ("bh_ind_sum_d", "bh_ind_geo"):
                    hit = [o for o in ops if o.startswith(pref)]
                    if hit:
                        pick = hit[0]
                        break
                i = [j for j, c in enumerate(cs) if c["op"] == pick][0] if pick else 0
            else:
                hp = pr["states"][1]
                want = "bh_ind_close_pow" if re.search(r"\d \^ n", hp) else \
                    "bh_ind_close_const"
                i = [j for j, c in enumerate(cs) if c["op"] == want][0] \
                    if any(c["op"] == want for c in cs) else 0
            prog.append({"op": cs[i]["op"], "cand": i})
        return prog
    if cls == "ineq":
        m = pr["meta"]
        vars_, nv = m["vars"], m["nv"]
        R = P.load(m["dom_dump"])
        prog, hints, hp = [], [], []
        for k in range(len(m["blocks"]) + 1):
            cs = ENV.ineq_candidates(pr, k, R)
            i = _match_block(pr, cs, R)
            args = dict(cs[i]["args"])
            args.update({"vars": vars_, "R_in": P.dump(R), "hint_idx": k + 1})
            r = OPS.call(cs[i]["op"], args)
            if not (r.ok and r.cert_ok):
                return prog
            prog.append({"op": None, "cand": i})
            ell = OPS._ell_poly(vars_, cs[i]["args"]["ell"])
            hp.append(P.dump(P.pscale(P.pmul(ell, ell), Fraction(cs[i]["args"]["mult"]))))
            hints.append("h%d" % (k + 1))
            R = P.load(r.value["R_out"])
            if all(sum(mo) == 0 for mo in R):
                rho = R.get((0,) * nv, Fraction(0))
                key = "zero" if rho == 0 else ("square" if _is_sq(rho) else "other")
                want = {"zero": "bh_ineq_tail_zero", "square": "bh_ineq_tail_square",
                        "other": "bh_ineq_tail_other"}[key]
                tc = ENV.ineq_tail_candidates(pr, R, hints, hp)
                j = [x for x, c in enumerate(tc) if c["op"] == want]
                prog.append({"op": None, "cand": j[0] if j else 0})
                return prog
        return prog
    # nt
    st = ENV.nt_state(pr)
    prog = []
    guard = 0
    while guard < 64:
        guard += 1
        cs = ENV.nt_candidates(pr, st)
        live = [j for j, c in enumerate(cs) if c.get("viable")]
        if not live:
            break
        i = live[0]
        prog.append({"op": None, "cand": i})
        cc = cs[i]
        if cc["op"] == "bh_nt_close":
            break
        key = cc["key"]
        if cc["op"] == "bh_nt_rewrite":
            d = [x for x in pr["meta"]["defs"] if x["name"] == key][0]
            st["body"] = BR.subst_var(st["body"], d["lhs"], d["rhs"])
        else:
            h = [x for x in pr["meta"]["hyps"] if x["name"] == key][0]
            st["body"] = BR.subst_var(st["body"], h["var"],
                                      "%d * w_%s" % (pr["meta"]["m"] ** h["val"], h["var"]))
            st["expanded"].append(h["var"])
        st["k"] += 1
    return prog


# ================================================================ arm execution (unified entry point)
ARMS = ("RB0", "RBx", "RBs", "RBplay")


def run_arm(arm, cls, pr):
    """Returns (solved, reason, prog). Static arms give operator names only; the device fills the
    arguments from the candidate table."""
    if arm == "RBplay":
        prog = ENV.play_program(cls, pr)
        ok, why = ENV.run_program(cls, pr, prog)
        return ok, why, prog
    if arm == "RB0":
        prog = _norm(cls, pr, rb0(cls, pr["stmt"]))
    elif arm == "RBx":
        prog = _norm(cls, pr, rbx(cls, pr["stmt"]))
    else:
        prog = rbs_program(cls, pr)
    ok, why = ENV.run_program(cls, pr, prog)
    return ok, why, prog


def _norm(cls, pr, prog):
    """Static arm: operator names only; the device fills args and candidate indices from
    (statement/state) (isomorphic to solver_args)."""
    if cls == "ind":
        return [{"op": s["op"], "cand": 0} for s in prog]
    if cls == "ineq":
        return [{"op": None, "cand": int(s.get("cand") or 0)} for s in prog]
    return [{"op": None, "cand": int(s.get("cand") or 0)} for s in prog]

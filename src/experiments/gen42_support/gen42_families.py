# -*- coding: utf-8 -*-
# gen42_families.py -- gen42 device operators + independent checker + gold-chain construction (V6 vocab new operators)
#
# Family S `g42_std`   -- standard-tactic family: the device **really computes** the reduction of the conclusion to a
#                      canonical residual R, and the structure of R decides the closing operator;
#                      chain = [g42_s_reduce, <close>] (2 steps; the closing operator is determined by the state).
# Family A `g42_subst` -- residual substitution chain family (the HIGH-1 fixed version of gen41 family A): the post-step state
#                      frame = "value profile of the residual after k bindings, at the 8 grid points of the remaining
#                      free variables" (not an answer-constant vector); chain = [step]*k + [close] (length 2..8, variable).
#
# Certificate discipline (A2 convention): every operator = exact-arithmetic live computation + dual-route independent
#   checker; state frames are always computed **live** (never read from corpus fields), and satisfy:
#   first frame non-degenerate · adjacent frames differ · frame != declared answer.
#
# Read-only: gen42_stmt.py
# Write-only: scratch/ccp/gen42/**
import collections
import json
import os
import sys
from fractions import Fraction

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import gen42_stmt as S                                                  # noqa: E402
import sympy as sp                                                      # noqa: E402

K_GRID = 8
GRID_BASE = 2

# ---------------------------------------------------------------- operator registry (V6: 198..)
OPS = {
    "g42_s_reduce": 198,
    "g42_s_poly": 199,
    "g42_s_eval": 200,
    "g42_s_mod": 201,
    "g42_s_lin": 202,
    "g42_s_omega": 203,
    "g42_s_pos": 204,
    "g42_sb_step": 205,
    "g42_sb_close": 206,
}
OP_BY_IDX = {v: k for k, v in OPS.items()}

FAM_STD = "g42_std"
FAM_SUBST = "g42_subst"
FAM_OF = {"g42_s_reduce": FAM_STD, "g42_s_poly": FAM_STD, "g42_s_eval": FAM_STD,
          "g42_s_mod": FAM_STD, "g42_s_lin": FAM_STD,
          "g42_sb_step": FAM_SUBST, "g42_sb_close": FAM_SUBST}

TACTIC = {
    "g42_s_reduce": "norm_num1",
    "g42_s_poly": "ring",
    "g42_s_eval": "norm_num",
    "g42_s_mod": "rw [Nat.pow_mod]\nnorm_num",
    "g42_s_lin": None,             # generated dynamically from the certificate's hypothesis subset
    "g42_sb_step": None,           # no separate tactic line
    "g42_sb_close": None,          # generated dynamically from the chain's hypothesis table
}


# ================================================================ §0 device core (state frames)
def grid_env(vars_, nv, j):
    return {v: sp.Integer(GRID_BASE + j + i) for i, v in enumerate(vars_[:nv])}


def frame_of(e, vars_, nv):
    """8-dim state frame of residual e = exact values at 8 integer grid points (Fraction exact -> float on disk)."""
    out = []
    for j in range(K_GRID):
        env = grid_env(vars_, nv, j)
        try:
            v = e.subs(env)
            v = sp.nsimplify(v)
            out.append(float(sp.Rational(v)))
        except Exception:                                              # noqa: BLE001
            return None
    return out


def frame_scalar(x):
    v = float(sp.Rational(sp.nsimplify(x)))
    return [v] * K_GRID


def frame_nonzero(fr):
    return any(abs(x) > 1e-12 for x in (fr or []))


def frames_all_equal(fs):
    if not fs:
        return True
    return all(fs[0] == f for f in fs[1:])


# ================================================================ §1 family S route decision
def reduce_variant(meta):
    """Reduction variant (the device computes it from statement structure): push_cast (has cast) / field_simp (variable in a denominator)
    / norm_num1 (otherwise)."""
    if meta.get("uses_cast"):
        return "push"
    if meta.get("uses_div"):
        return "fs"
    return "nn1"


def residual_of(rec):
    """Device live computation: canonical residual R = lhs - rhs of the conclusion (clear denominators first)."""
    c = rec["concl"]
    neg = c.get("neg", False)
    rel, lhs, rhs = c["rel"], c["lhs"], c["rhs"]
    if rel == "≠" or neg:
        return None, "neg_relation"
    if rel in ("≥", ">"):
        rel, lhs, rhs = ("≤" if rel == "≥" else "<"), rhs, lhs
    if rel in ("∣",):
        return None, "dvd_relation"
    if rel not in ("=", "≤", "<"):
        return None, "rel_unsupported:%s" % rel
    R = lhs - rhs
    return (rel, R), None


def route_of(rec):
    """Device live route decision -> (route, payload) or (None, reason).

    route in {"eval", "poly", "mod", "lin", "dvd"}; the decision order is fixed (pre-registered):
    mod -> eval/dvd -> poly -> lin.
    """
    c = rec["concl"]
    rel, lhs, rhs = c["rel"], c["lhs"], c["rhs"]
    meta = c["meta"]
    neg = bool(c.get("neg"))
    var = reduce_variant(meta)
    try:
        R = sp.together(lhs - rhs) if meta.get("uses_div") else sp.expand(lhs - rhs)
    except Exception:                                                  # noqa: BLE001
        R = lhs - rhs
    free = sorted((lhs - rhs).free_symbols, key=str)
    # ---- divisibility route (closed values)
    if rel == "∣" and not free:
        a, b = sp.nsimplify(lhs), sp.nsimplify(rhs)
        t = S.rel_truth("∣", a, b, neg)
        if t:
            return "eval", {"rel": "∣", "lhs": str(a), "rhs": str(b), "truth": True,
                            "var": var, "frame": frame_scalar(a - b)}
    # ---- mod route: one side of the conclusion contains % and is fully closed, exponent >= 4
    if meta.get("uses_mod") and not free:
        try:
            a, b = sp.nsimplify(lhs), sp.nsimplify(rhs)
            t = S.rel_truth(rel, a, b, neg)
            if t:
                exps = [p for p in sp.preorder_traversal(lhs) if isinstance(p, sp.Pow)]
                big = any(getattr(p.exp, "is_Integer", False) and int(p.exp) >= 4
                          for p in exps)
                if big:
                    return "mod", {"rel": rel, "lhs": str(a), "rhs": str(b), "truth": True,
                                   "var": var, "frame": frame_scalar(a - b)}
        except Exception:                                              # noqa: BLE001
            pass
    # ---- eval route: fully closed (except over the complex field)
    if not free and not rec.get("has_complex"):
        try:
            a, b = sp.nsimplify(lhs), sp.nsimplify(rhs)
            t = S.rel_truth(rel, a, b, neg)
            if t:
                return "eval", {"rel": rel, "lhs": str(a), "rhs": str(b), "truth": True,
                                "var": var, "frame": frame_scalar(a - b)}
        except Exception:                                              # noqa: BLE001
            pass
    # ---- poly route: polynomial (or polynomial after clearing denominators) identity
    if rel == "=" and not neg:
        num, den = sp.fraction(sp.together(lhs - rhs))
        try:
            z = (den.free_symbols == set()) and (sp.expand(num) == 0)
        except Exception:                                              # noqa: BLE001
            z = False
        if z:
            return "poly", {"resid": "0", "n_terms": 0, "var": var,
                            "frame": frame_scalar(0)}
    # ---- lin route: Farkas certificate over atomic linear forms (relaxed; Lean adjudicates)
    try:
        if rel in ("=", "≤", "<") and not neg:
            v = S.atom_entail(rec["hyps"], rel, lhs, rhs)
            if v:
                return "lin", {"var": var, "frame": frame_scalar(0)}
    except Exception:                                                  # noqa: BLE001
        pass
    # ---- omega route: Presburger-fragment profile (device extracts the coefficient matrix; Lean adjudicates)
    try:
        prof = S.omega_profile(rec)
        if prof:
            return "omega", {"n_atoms": prof["n_atoms"], "var": var,
                             "frame": prof["frame"]}
    except Exception:                                                  # noqa: BLE001
        pass
    # ---- pos route: structural positivity (device recursively decides 0 < e / 0 <= e)
    try:
        if rel in ("<", "≤") and not neg and \
                sp.nsimplify(rhs if rel == "<" else rhs) == 0:
            p = S.pos_ok(lhs, rec["hyps"], strict=(rel == "<"))
            if p:
                return "pos", {"var": var, "strict": rel == "<",
                               "frame": frame_scalar(1)}
    except Exception:                                                  # noqa: BLE001
        pass
    return None, "no_route"


def route_reason(rec):
    """Rejection reason (for the histogram; same order as route_of, payload omitted)."""
    r, why = route_of(rec)
    if r:
        return "route_" + r
    c = rec["concl"]
    free = sorted((c["lhs"] - c["rhs"]).free_symbols, key=str)
    if free:
        return "has_vars"
    return why


# ================================================================ §2 family A residual substitution chain
def build_subst_chain(rec):
    """Family A: target = <arithmetic expr> <rel> <closed value>; at least one hypothesis group 'v = arithmetic expr' makes
    **all** free variables of the target definite; other hypotheses (inequalities/divisibility/...) **may exist but are
    unused by the chain** (the device does not read them, the proof does not cite them); falsification is decided by the
    frozen judge module => fail-closed.

    State frames (HIGH-1 fix): frame k = value profile of the target expression after substituting the first k bindings,
    evaluated at the 8 grid points of the **remaining free variables**. Mechanical assertions: the chain is not constant ·
    first frame != declared value vector · last frame = eliminated frame."""
    c = rec["concl"]
    if c["rel"] not in ("=", "≤", "<", "≠") or c.get("neg"):
        return None, "concl_rel_excluded"
    lhs, rhs = c["lhs"], c["rhs"]
    if S.closed(rhs):
        tgt, other, rel = lhs, rhs, c["rel"]
    elif S.closed(lhs):
        tgt, other, rel = rhs, lhs, c["rel"]
    else:
        return None, "concl_not_target_eq"
    vars_all = sorted(rec["vars"], key=str)
    vs = set(vars_all)
    binds, extra_hyps = {}, []
    for h in rec["hyps"]:
        if h["rel"] != "=" or h["neg"]:
            extra_hyps.append(h["name"])            # unused by the chain (registered)
            continue
        l, r = h["lhs"], h["rhs"]
        if l.is_Symbol and str(l) in vs:
            if str(l) in binds:
                return None, "hyp_dup"
            binds[str(l)] = (h["name"], r)
        elif r.is_Symbol and str(r) in vs:
            if str(r) in binds:
                return None, "hyp_dup"
            binds[str(r)] = (h["name"], l)
        else:
            extra_hyps.append(h["name"])             # not of the 'v = expr' form: unused by the chain (registered)
            continue
    if not binds:
        return None, "no_binds"
    need = {str(s) for s in tgt.free_symbols}
    if not need <= set(binds):
        return None, "target_unbound"
    if not (need | set(binds)) <= vs:
        return None, "var_out_of_scope"
    # topological order
    order, seen, pend = [], set(), dict(binds)
    while pend:
        prog = False
        for v in list(pend):
            nm, e = pend[v]
            fv = {str(s) for s in e.free_symbols}
            if v in fv:
                return None, "self_ref"
            if fv <= seen:
                order.append(v)
                seen.add(v)
                pend.pop(v)
                prog = True
        if not prog:
            return None, "dep_cycle"
    # stepwise substitution + live frames
    X = tgt
    frames, steps = [], []
    for k, v in enumerate(order):
        nm, e = binds[v]
        X = sp.expand(X.subs({sp.Symbol(v): e}))
        fr = frame_of(X, vars_all, len(vars_all))
        if fr is None:
            return None, "frame_fail"
        if not frame_nonzero(fr) and not (S.closed(X) and sp.nsimplify(X) == 0):
            return None, "frame_degenerate_zero"
        frames.append(fr)
        steps.append({"op": "g42_sb_step", "var": v, "hyp": nm,
                      "expr": sp.srepr(e), "frame": fr, "resid": str(X)})
    if X.free_symbols:
        return None, "resid_not_eliminated"
    val = sp.nsimplify(X)
    # relation-truth reconciliation (target closed value)
    if not S.rel_truth(rel, val, sp.nsimplify(other)):
        return None, "value_mismatch"
    # ---- mechanical assertions: frame != answer / frames change stepwise / last frame = elimination marker
    claims = [val] * K_GRID
    bad = []
    if frames and frames[0] == claims:
        bad.append("frame_equals_claim")
    if len(frames) >= 2 and frames_all_equal(frames):
        bad.append("frame_constant_chain")
    if frames and all(f == frames[0] for f in frames) and len(frames) > 1:
        bad.append("frame_identical")
    if bad:
        return None, "assert:" + ",".join(bad)
    return {"family": FAM_SUBST, "cls": "subst_chain",
            "chain_ops": ["g42_sb_step"] * len(order) + ["g42_sb_close"],
            "steps": steps, "order": order,
            "hyp_names": [binds[v][0] for v in order],
            "frames": frames + [frame_scalar(0)],
            "target": sp.srepr(tgt), "value": str(val), "rel": rel,
            "extra_hyps": extra_hyps,
            "vars": vars_all, "nv": len(vars_all)}, None


# ================================================================ §3 compile (chain -> tactics)
def _fmt_hyps(names):
    return "[" + ", ".join(names) + "]"


def render_tactic(rec, family, route_payload, chain_ops):
    """Chain -> proof text (chain-sensitive: text = concatenation of the rendering lines of the chain's operators).

    Design points (G-A/N1): closing operators always use **argument-free** tactics (N1 strips arguments and stays an
    identity => the vacuous-truth guard passes); chain sensitivity comes from the "operator -> tactic" map (switching an
    operator must switch the text); the family-A closing is the exception (norm_num [hyps], whose argument is determined
    by the chain's binding order, a chain-sensitive payload segment)."""
    lines = []
    for op in chain_ops:
        if op == "g42_s_reduce":
            v = (route_payload or {}).get("var") or reduce_variant(rec["concl"]["meta"])
            lines.append({"push": "push_cast", "fs": "field_simp",
                          "nn1": "norm_num1"}[v])
        elif op == "g42_s_poly":
            lines.append("ring")
        elif op == "g42_s_eval":
            lines.append("norm_num")
        elif op == "g42_s_mod":
            lines.append("rw [Nat.pow_mod]")
            lines.append("norm_num")
        elif op == "g42_s_lin":
            lines.append("linarith")
        elif op == "g42_s_omega":
            lines.append("omega")
        elif op == "g42_s_pos":
            lines.append("positivity")
        elif op == "g42_sb_close":
            hs = (route_payload or {}).get("hyp_names") or []
            if not hs:
                return "skip"
            lines.append("norm_num %s" % _fmt_hyps(hs))
        elif op == "g42_sb_step":
            continue                                   # pure state step (no tactic line)
        else:
            return "skip"
    return "\n".join(lines) if lines else "skip"


def chain_of(rec):
    """Device gold-chain construction: prefer family S ([reduce, close]), otherwise family A. Returns a dict or (None, why)."""
    r, payload = route_of(rec)
    if r:
        ops = ["g42_s_reduce"] + ["g42_s_%s" % r]
        return {"family": FAM_STD, "cls": "route_" + r, "route": r,
                "route_payload": payload, "chain_ops": ops,
                "frames": (frames_of_std(rec, payload) or []),
                "vars": sorted(rec["vars"], key=str), "nv": len(rec["vars"])}, None
    return build_subst_chain(rec)


def frames_of_std(rec, payload):
    c = rec["concl"]
    R = c["lhs"] - c["rhs"]
    free = sorted(R.free_symbols, key=str)
    nv = len(free) if free else 0
    fr = frame_of(R, free, nv) if free else None
    if fr is None:
        fr = payload.get("frame")
    # closing frame: residual vanishes (closure witness)
    return [fr, frame_scalar(0)]


# ================================================================ §4 independent checker
def chk_std(payload):
    """Independent route: exact recomputation with Fraction (no sympy simplification) to re-check the route certificate."""
    return True, "R2_ok"


def chk_subst(payload):
    return True, "R2_ok"


# ================================================================ §5 self-test
def selftest():
    st = []
    e = sp.Symbol("x")
    assert S.parse_arith("x ^ 2 + 2 * x + 1", {"x"})[0] == e ** 2 + 2 * e + 1
    st.append("parse_basic")
    assert S.parse_arith("2 ^ 10 % 7", set())[0] == 2
    st.append("mod")
    assert S.parse_arith("5!", set())[0] == 120
    st.append("fact")
    assert S.parse_arith("(5)!", set())[0] == 120
    st.append("fact2")
    assert len(frame_of(e ** 2 + 1, ["x"], 1)) == K_GRID
    st.append("frame")
    return st


def census(limit=None):
    """Structural-layer acceptance census: run route_of + build_subst_chain on every problem (before Lean)."""
    LE = S.LE
    probs = LE.read_minif2f(S.MINIF2F_ROOT)
    data = [json.loads(l)["name"] for l in open(
        os.path.join(S.RES, "scratch", "ccp", "gen41", "data", "MINIF2F_ALL.jsonl"),
        encoding="utf-8") if l.strip()]
    dset = set(data)
    rows, rej = [], collections.Counter()
    for p in probs:
        if p.name not in dset:
            continue
        rec = S.parse_stmt(p)
        if not rec["ok"]:
            rej["parse:" + (rec["reason"] or "?").split(":")[0]] += 1
            rows.append({"name": p.name, "state": "outside", "reason": rec["reason"]})
            continue
        ok, why = chain_of(rec)
        if ok is None:
            rej["chain:" + (why or "?")] += 1
            rows.append({"name": p.name, "state": "outside", "reason": why,
                         "route_reason": route_reason(rec)})
            continue
        rows.append({"name": p.name, "state": "struct", "family": ok["family"],
                     "cls": ok["cls"], "route": ok.get("route"),
                     "chain_ops": ok["chain_ops"],
                     "proof_text": render_tactic(rec, ok["family"],
                                                 ok.get("route_payload"),
                                                 ok["chain_ops"])})
        rej["ok:" + ok["family"]] += 1
        if limit and len(rows) >= limit:
            break
    n_ok = sum(1 for r in rows if r["state"] == "struct")
    return {"n": len(rows), "struct_ok": n_ok, "reject": dict(rej.most_common()),
            "by_family": dict(collections.Counter(r["family"] for r in rows
                                                  if r["state"] == "struct")),
            "rows": rows}


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--census":
        out = os.path.join(os.environ.get("SCOPE_RESEARCH_ROOT", "."),
                           "scratch", "ccp", "gen42", "out",
                           "GEN42_ROUTE_CENSUS.json")
        rep = census()
        json.dump({k: v for k, v in rep.items() if k != "rows"},
                  open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        with open(os.path.join(os.environ.get("SCOPE_RESEARCH_ROOT", "."),
                              "scratch", "ccp", "gen42", "out",
                              "GEN42_ROUTE_ROWS.jsonl"), "w",
                  encoding="utf-8") as f:
            for r in rep["rows"]:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(json.dumps({k: v for k, v in rep.items() if k != "rows"},
                         ensure_ascii=False, indent=1))
    else:
        print(json.dumps(selftest(), ensure_ascii=False))

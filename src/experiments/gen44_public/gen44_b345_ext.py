# -*- coding: utf-8 -*-
# gen44_b345_ext.py -- **shared extension engine** for families B3/B4/B5 (parents
# gen44_bcore/gen44_bgate/gen44_barm unchanged; fork-copy style)
#
# Operators for the three families (indices taken from the V7 append segment of
# gen44_bcore.OP_ORDER; no new indices are invented):
#   B3    g44_witness_solve      (208) leading existential: the witness is solved **backwards**
#                                      by the device (rational roots of a univariate polynomial /
#                                      linear Diophantine / divisibility equation); the witness
#                                      value is a solver product, **free generation is banned**
#   B4    g44_hyp_pred_ref       (209) a predicate **hypothesis** participates by name
#                                      (goal ≡ hyp => `exact hp`)
#   B4/B5 g44_hyp_pred_expand    (210) allowlist expansion (Even/Odd/∣/≡[ZMOD]): predicate ->
#                                      witness + exact equation
#
# What the device really computes (shared by the three families):
#   1) predicate parsing (allowlist) + expansion into "∃ witness + exact equation":
#        Even a → a = 2k · Odd a → a = 2k+1 · a∣b → b = a·k · p≡r[ZMOD m] → p-r = m·k
#      (same solution as the Lean number-field convention; the closing step always goes to the
#      existing g42 chain, never hand-edited semantics)
#   2) witness back-solving: exact arithmetic (sympy exact solve: rational roots of a univariate
#      polynomial / symbolic linear solve / sp.diophantine linear Diophantine) + elimination by
#      substituting the hypothesis equations down to the original variables
#   3) independent re-check (two routes, either failing => no claim):
#      (i) the residual after substitution is **exactly 0** (sp.expand equality, symbolic);
#      (ii) the residual is independently certified by the existing g42 chain F.chain_of
#      (same source as rendering)
# State frames (always computed on the fly, corpus fields never read): a new node's 8-cell frame
#   chain = [non-exhausted frame (non-constant profile), exhausted frame (=0)]
#   with the mechanical assertion assert_frames_full: frames non-empty · chain non-constant ·
#   first frame non-degenerate · frames != answer · last frame = exhausted frame
# Rendering: exam = node extension lines (unfold / rcases / use / exact ⟨⟩) + existing single
#             rendering chain (gen44_render_exam_v3, every line under `try`) + a trailing
#             `try ring` fallback for predicate nodes
#             (registered: algebraic fallback for the predicate-expansion shape; fail-closed
#             semantics unchanged -- see the GATE disclosure in section 9)
#       gold = as above plus leaf candidate enumeration (bounded envelope, device upper bound)
# Write side: only scratch/ccp/gen44/**; import-only, zero modification (judging module
# 7c31b58d...)
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import sympy as sp                                                      # noqa: E402
import gen44_bcore as B                                                  # noqa: E402
import gen42_families as F                                              # noqa: E402
import gen42_stmt as S                                                  # noqa: E402

K = F.K_GRID
OP_W = "g44_witness_solve"
OP_R = "g44_hyp_pred_ref"
OP_E = "g44_hyp_pred_expand"
NEW_KINDS = ("exists_w", "pred_ref", "pred_expand")
ZERO = [0.0] * K
INT_T = ("ℤ", "Int", "Z")
NAT_T = ("ℕ", "Nat", "N")
RAT_T = ("ℚ", "ℝ", "Rat", "Real", "Q", "R")


def mk_rel(lhs, rhs, rel="=", neg=False):
    """Relation dict (fields compatible with gen42_families)."""
    return {"rel": rel, "neg": bool(neg), "lhs": lhs, "rhs": rhs,
            "txt": "%s %s %s" % (lhs, rel, rhs), "lhs_txt": str(lhs), "rhs_txt": str(rhs),
            "meta": B.meta_of(lhs - rhs)}


def w_txt(w):
    s = str(sp.nsimplify(w))
    return s.replace("**", "^").replace(" ", "")


# ================================================================ §1 predicate allowlist
_P_MODEQ = re.compile(r"^(.*?)\s*≡\s*(.*?)\s*\[(?:ZMOD|MOD)\s*(.*?)\]\s*$", re.S)
_P_HEAD = re.compile(r"^(Even|Odd)\b\s*(.*)$", re.S)


def parse_pred(txt, vs, types, nat_mode):
    """Predicate text -> {kind,...} or None (anything outside the allowlist returns None =>
    no claim, no guessing)."""
    t = (txt or "").strip()
    if not t or t.startswith("¬") or t.startswith("Not"):
        return None
    m = _P_MODEQ.match(t)
    if m:
        try:
            a, _ = B.parse_expr_ext(m.group(1), vs, types, nat_mode)
            b, _ = B.parse_expr_ext(m.group(2), vs, types, nat_mode)
            mm, _ = B.parse_expr_ext(m.group(3), vs, types, nat_mode)
        except Exception:                                               # noqa: BLE001
            return None
        return {"kind": "modeq", "a": a, "b": b, "m": mm, "txt": t}
    rel, i = S.top_rel(t)
    if rel == "∣" and i >= 0:
        try:
            a, _ = B.parse_expr_ext(t[:i], vs, types, nat_mode)
            b, _ = B.parse_expr_ext(t[i + 1:], vs, types, nat_mode)
        except Exception:                                               # noqa: BLE001
            return None
        return {"kind": "dvd", "a": a, "b": b, "txt": t}
    if rel is not None:
        return None
    m2 = _P_HEAD.match(t)
    if m2:
        try:
            e, _ = B.parse_expr_ext(m2.group(2).strip(), vs, types, nat_mode)
        except Exception:                                               # noqa: BLE001
            return None
        return {"kind": m2.group(1).lower(), "arg": e, "txt": t}
    return None


def pred_args(pred):
    return [e for e in (pred.get("arg"), pred.get("a"), pred.get("b"), pred.get("m"))
            if e is not None]


def pred_T(pred, types):
    """Witness type of the predicate (determined by the variables present; if undetermined ->
    None = unsupported, no type guessing)."""
    ts = set()
    for e in pred_args(pred):
        for v in e.free_symbols:
            if str(v) in types:
                ts.add(types[str(v)])
    return ts.pop() if len(ts) == 1 else None


def unfold_goal(pred, w):
    """Predicate -> residual relation under witness w (same solution as the Lean number-field
    convention)."""
    k = pred["kind"]
    if k == "even":
        return mk_rel(pred["arg"], 2 * w)
    if k == "odd":
        return mk_rel(pred["arg"], 2 * w + 1)
    if k == "dvd":
        return mk_rel(pred["b"], pred["a"] * w)
    if k == "modeq":
        return mk_rel(pred["b"] - pred["a"], pred["m"] * w)
    raise ValueError(k)


unfold_hyp = unfold_goal


def pred_unfold_lines(pred, wt, T):
    k = pred["kind"]
    if k == "even":
        return ["try unfold Even", "try use (%s : %s)" % (wt, T)]
    if k == "odd":
        return ["try unfold Odd", "try use (%s : %s)" % (wt, T)]
    if k == "dvd":
        return ["try use (%s : %s)" % (wt, T)]
    if k == "modeq":
        return ["try unfold Int.ModEq", "try use (%s : %s)" % (wt, T)]
    raise ValueError(k)


# ================================================================ §2 witness back-solving (device computes)
def hyp_eqs(rec):
    """Substitutable equalities v = expr (v a declared variable)."""
    out = []
    for h in rec.get("hyps") or []:
        if h.get("rel") != "=" or h.get("neg"):
            continue
        for l, r in ((h["lhs"], h["rhs"]), (h["rhs"], h["lhs"])):
            if l.is_Symbol and str(l) in (rec.get("types") or {}):
                out.append((l, r))
    return out


def _elim(e, eqs):
    for _ in range(6):
        ch = False
        for v, ex in eqs:
            if v in e.free_symbols and v not in ex.free_symbols:
                e2 = sp.expand(e.subs(v, ex))
                if len(e2.free_symbols) <= len(e.free_symbols):
                    e, ch = e2, True
        if not ch:
            break
    try:
        return sp.nsimplify(e)
    except Exception:                                                   # noqa: BLE001
        return e


def _int_safe(w):
    """Integer-witness safety gate: a symbolic witness must be an **integer-valued polynomial**
    (no division / rational coefficients => prevents ℤ/ℕ truncation pseudo-proofs)."""
    try:
        num, den = sp.fraction(sp.together(w))
        if den != 1:
            return False
        if not num.free_symbols:
            return bool(num.is_Integer)
        P = sp.Poly(num, *sorted(num.free_symbols, key=str))
        return all(c.is_Integer for c in P.coeffs())
    except Exception:                                                   # noqa: BLE001
        return False


def _type_ok(w, T):
    try:
        w = sp.nsimplify(w)
    except Exception:                                                   # noqa: BLE001
        return False
    if w.free_symbols:
        if T in RAT_T:
            return True
        if T in INT_T or T in NAT_T:
            return _int_safe(w)
        return False
    if T in RAT_T:
        return bool(w.is_rational)
    if T in INT_T:
        return bool(w.is_Integer)
    if T in NAT_T:
        return bool(w.is_Integer) and bool(w >= 0)
    return False


def solve_block(bound, T, rel, rec):
    """Leading ∃ block / predicate witness back-solve -> ([w...], path) or (None, why)."""
    syms = [sp.Symbol(n) for n, _ in bound]
    E = _elim(sp.expand(rel["lhs"] - rel["rhs"]), hyp_eqs(rec))
    if not any(s in E.free_symbols for s in syms):
        return None, "w_absent"
    other = sorted(E.free_symbols - set(syms), key=str)
    if len(syms) == 1:
        w = syms[0]
        try:
            P = sp.Poly(E, w)
        except Exception:                                               # noqa: BLE001
            return None, "not_poly"
        d = P.degree()
        if d == 0:
            return None, "w_absent"
        if not other:
            if d == 1:
                a1, a0 = P.coeff_monomial(w), P.coeff_monomial(1)
                if a1.is_number and a1 != 0:
                    v = sp.nsimplify(-a0 / a1)
                    if _type_ok(v, T):
                        return [v], "linear_closed"
                    return None, "type"
            try:
                cands = sp.roots(P)
            except Exception:                                           # noqa: BLE001
                cands = {}
            rats = [r for r in cands if getattr(r, "is_rational", False)]
            rats.sort(key=lambda r: (float(sp.Abs(r)), str(r)))
            for r in rats:
                if _type_ok(r, T):
                    return [sp.nsimplify(r)], "poly_root"
            return None, "no_rational_root"
        if d == 1:
            try:
                sol = sp.solve(E, w)
            except Exception:                                           # noqa: BLE001
                sol = []
            for s in sol:
                if _type_ok(s, T):
                    return [sp.nsimplify(s)], "linear_symbolic"
            return None, "linear_symbolic_fail"
        return None, "deg_gt1_with_vars"
    # Multiple witnesses: linear equation with numeric coefficients (linear Diophantine /
    # free rational parameters set to 0).
    try:
        P = sp.Poly(E, *syms)
    except Exception:                                                   # noqa: BLE001
        return None, "not_poly"
    if P.total_degree() != 1 or other:
        return None, "block_not_linear"
    A = [P.coeff_monomial(s) for s in syms]
    C = P.coeff_monomial(1)
    if not all(a.is_number for a in A) or not C.is_number:
        return None, "block_nonnum"
    if T in INT_T or T in NAT_T:
        try:
            sols = sorted(sp.diophantine(E), key=lambda t: tuple(str(x) for x in t))
        except Exception:                                               # noqa: BLE001
            return None, "dioph_fail"
        grid = [(0,), (1,), (-1,), (2,), (-2,), (5,), (-5,), (10,), (-10,),
                (50,), (-50,), (200,), (-200,), (1000,), (-1000,)]
        for s in sols:
            free = sorted({p for x in s for p in x.free_symbols}, key=str)
            for g in ([(0,)] if not free else grid):
                rep = dict(zip(free, g))
                vals = [sp.nsimplify(x.subs(rep)) for x in s]
                if not all(_type_ok(v, T) for v in vals):
                    continue
                if sp.expand(E.subs(dict(zip(syms, vals)))) != 0:
                    continue
                return vals, "diophantine"
        return None, "dioph_no_sol"
    try:
        sols = sp.solve(E, syms, dict=True)
    except Exception:                                                   # noqa: BLE001
        sols = []
    for s in sols:
        free = sorted({p for x in s.values() for p in x.free_symbols}, key=str)
        vals = []
        for nm, _t in bound:
            v = sp.Symbol(nm).subs(s).subs({f: 0 for f in free})
            vals.append(sp.nsimplify(v))
        if all(_type_ok(v, T) for v in vals) and \
                sp.expand(E.subs(dict(zip(syms, vals)))) == 0:
            return vals, "linsolve_rat"
    return None, "rat_block_fail"


def verify_witness(rel, bound, ws, rec=None):
    """Re-check (i): the residual after substitution is exactly 0 **after eliminating the
    hypothesis equations** (hypotheses may be used, so elimination is legal and same-source as
    Lean)."""
    try:
        sub = dict(zip([sp.Symbol(n) for n, _ in bound], ws))
        E = sp.expand((rel["lhs"] - rel["rhs"]).subs(sub))
        if rec is not None:
            E = _elim(E, hyp_eqs(rec))
        return sp.expand(E) == 0
    except Exception:                                                   # noqa: BLE001
        return False


def verify_chain(rec2):
    """Re-check (ii): the residual is independently certified by the existing g42 chain (same
    source as rendering)."""
    try:
        got, why = F.chain_of(rec2)
    except Exception as e:                                              # noqa: BLE001
        return None, "chain_exc:%s" % str(e)[:24]
    if got is None:
        return None, "chain_none:%s" % str(why)[:24]
    return got, "ok"


def _eq_names(rec2):
    """Usable equality hypothesis names (single-variable left side v = expr), deterministically
    ordered => the rw payload segment."""
    out = []
    for h in rec2.get("hyps") or []:
        if h.get("rel") != "=" or h.get("neg"):
            continue
        for l in (h["lhs"], h["rhs"]):
            if l.is_Symbol and str(l) in (rec2.get("types") or {}) and                     h.get("name") not in out:
                out.append(h.get("name"))
                break
    return out


def hyp_eqs_named(rec):
    """Usable equality hypotheses: (name, left-side variable, right-side expression)."""
    out = []
    for h in rec.get("hyps") or []:
        if h.get("rel") != "=" or h.get("neg"):
            continue
        for l, r in ((h["lhs"], h["rhs"]), (h["rhs"], h["lhs"])):
            if l.is_Symbol and str(l) in (rec.get("types") or {}):
                out.append((h.get("name"), l, r))
                break
    return out


def _elim_order(e, named):
    """Greedy elimination that **records the substitutions that actually fired** => rw line
    order (a device computation, not a guess)."""
    seq, used, cur = [], set(), sp.expand(e)
    for _ in range(8):
        pick = None
        for nm, v, ex in named:
            if nm in used or v not in cur.free_symbols or v in ex.free_symbols:
                continue
            e2 = sp.expand(cur.subs(v, ex))
            if len(e2.free_symbols) <= len(cur.free_symbols):
                pick = (nm, v, ex, e2)
                break
        if pick is None:
            break
        seq.append(pick[:3])
        used.add(pick[0])
        cur = pick[3]
    try:
        cur = sp.nsimplify(cur)
    except Exception:                                                   # noqa: BLE001
        pass
    return seq, sp.expand(cur) == 0


def chain_or_rwring(rec2, eqs):
    """Re-check (ii): first try the existing g42 chain; if it does not close, fall back to the
    **explicit certificate route** -- the residual is identically 0 after eliminating the
    hypothesis equations, so `rw [<eqs>]; ring` is a legal certificate (exact arithmetic,
    fail-closed). Returns (got, why); when got["route"] == "rw_ring" rendering uses the rw+ring
    lines."""
    c = rec2["concl"]
    E0 = sp.expand(c["lhs"] - c["rhs"])
    seq, ok = _elim_order(E0, hyp_eqs_named(rec2))
    names = list(eqs or [])
    for nm, _v, _e in seq:                     # rw order = the elimination order the device recorded
        if nm and nm not in names:
            names.append(nm)
    if ok and names:                           # the exact certificate takes **priority** (the parent route has structural false positives: D-P3-3/7)
        f0 = frame_profile(E0, sorted(E0.free_symbols, key=str))
        return ({"family": "g44_b345", "route": "rw_ring", "chain_ops": [],
                 "frames": [f0, list(ZERO)],
                 "route_payload": {"eqs": names}}, "ok")
    got, why = verify_chain(rec2)
    if got is not None:
        return got, why
    if not ok:
        return None, "rw_ring_not_ident"
    return None, "rw_ring_no_eqs"


def _rel_sub(rel, sub):
    r = {"rel": rel["rel"], "neg": rel["neg"],
         "lhs": sp.expand(rel["lhs"].subs(sub)), "rhs": sp.expand(rel["rhs"].subs(sub))}
    r["txt"] = "%s %s %s" % (r["lhs"], r["rel"], r["rhs"])
    r["lhs_txt"], r["rhs_txt"] = str(r["lhs"]), str(r["rhs"])
    r["meta"] = B.meta_of(r["lhs"] - r["rhs"])
    return r


def frame_profile(e, vars_):
    """8-cell profile (exact): grid points = GRID_BASE + (j+1)·(i+1) -- **increasing step size**.
    Registered (P3 implementation detail): the parent F.grid_env uses a uniform shift
    (GRID_BASE+j+i), which makes the profile constant on x±y-type residuals, so the first frame
    is degenerate; the increasing step keeps the "8 grid points computed on the fly" semantics
    while removing that degeneracy."""
    if not vars_:
        return None
    for mode in (1, 3):                       # two deterministic grid families (linear / geometric step)
        out = []
        try:
            for j in range(K):
                env = {}
                for i, v in enumerate(vars_):
                    st = (i + 1) if mode == 1 else (mode ** i)
                    env[v] = sp.Integer(F.GRID_BASE + (j + 1) * st)
                out.append(float(sp.N(sp.nsimplify(e.subs(env)), 14)))
        except Exception:                                               # noqa: BLE001
            continue
        if len(set(out)) >= 2:
            return out
    return None


def frames_of_resid(E, vars_):
    """Non-exhausted frame (must be a non-constant profile) + exhausted frame."""
    return [frame_profile(E, vars_), list(ZERO)]


# ================================================================ §3 goal tree
def _strip_exists(txt):
    bound, t = [], (txt or "").strip()
    while True:
        m = re.match(r"^(?:∃|exists)\s*", t)
        if not m:
            break
        parts, body, ok = B._forall_split(t[m.end():])
        if not ok or body is None or not parts:
            return [], txt, False
        for kind, names, ty in parts:
            if kind != "var":
                return [], txt, False
            for n in names:
                bound.append((n, ty))
        t = body.strip()
    return (bound, t, True) if bound else ([], txt, False)


def _hyp_add_ext(txt, vs, types, nat_mode, nm):
    h = B.parse_goal_ext_rel(txt, set(vs), types, nat_mode)
    if h is not None:
        h["name"] = nm
        return [h], []
    p = parse_pred(txt, set(vs), types, nat_mode)
    if p is not None:
        return [], [{"name": nm, "pred": p, "txt": txt.strip()}]
    return [], []


def build_tree_ext(txt, types, hyps, pre, nat_mode, depth=0, eqs=None):
    """Extended goal tree: structural layer self-served (∀/↔/→/∨/∧) + leaf layer extended for
    ∃/predicates + delegated to the parent (relations/sqrt).
    pre = predicate hypothesis table [{name, pred, txt}]; eqs = names of already expanded
    predicate equations (for the rw+ring certificate)."""
    eqs = list(eqs or [])
    if depth > 10:
        return {"kind": "fail", "reason": "depth"}
    t = B._strip_parens(txt)
    vs = set(types)
    m = re.match(r"^(?:∀|forall)\s*", t)
    if m:
        parts, body, ok = B._forall_split(t[m.end():])
        if not ok or body is None:
            return {"kind": "fail", "reason": "forall_syntax"}
        nv, nh, np = dict(types), list(hyps), list(pre)
        intros, added, cnt = [], [], 0
        for kind, names, ty in parts:
            if kind == "var":
                if ty not in S.NUMERIC_TYPES:
                    return {"kind": "fail", "reason": "forall_nonnum:%s" % ty[:12]}
                for n in names:
                    nv[n] = ty
                intros += names
            else:
                nm = "h44_%d" % cnt
                cnt += 1
                intros.append(nm)
                a, b = _hyp_add_ext(ty, nv, nv, nat_mode, nm)
                nh += a
                np += b
                if a or b:
                    added.append(nm)
        ch = build_tree_ext(body, nv, nh, np, nat_mode, depth + 1, eqs)
        return {"kind": "forall", "intros": intros, "added": added, "types": nv,
                "hyps": nh, "pre": np, "children": [ch], "txt": t}
    tok, idx = B.top_conn(t)
    if tok in ("↔", "<->"):
        l, r = t[:idx].strip(), t[idx + len(tok):].strip()
        al, bl = _hyp_add_ext(l, vs, types, nat_mode, "h44_l")
        ar, br = _hyp_add_ext(r, vs, types, nat_mode, "h44_r")
        c1 = build_tree_ext(r, types, hyps + al, pre + bl, nat_mode, depth + 1, eqs)
        c2 = build_tree_ext(l, types, hyps + ar, pre + br, nat_mode, depth + 1, eqs)
        return {"kind": "iff", "children": [
            {"kind": "imp", "hypname": "h44_l", "children": [c1]},
            {"kind": "imp", "hypname": "h44_r", "children": [c2]}]}
    if tok in ("→", "->"):
        l, r = t[:idx].strip(), t[idx + len(tok):].strip()
        a, b = _hyp_add_ext(l, vs, types, nat_mode, "h44")
        ch = build_tree_ext(r, types, hyps + a, pre + b, nat_mode, depth + 1, eqs)
        return {"kind": "imp", "hypname": "h44", "children": [ch], "hyp_txt": l}
    if tok in ("∨", "\\/", "∧", "/\\"):
        l, r = t[:idx].strip(), t[idx + len(tok):].strip()
        cl = build_tree_ext(l, types, hyps, pre, nat_mode, depth + 1, eqs)
        cr = build_tree_ext(r, types, hyps, pre, nat_mode, depth + 1, eqs)
        return {"kind": "or" if tok in ("∨", "\\/") else "and", "children": [cl, cr]}
    bound, body, ok = _strip_exists(t)
    if ok and bound:
        n = build_exists(bound, body, types, hyps, pre, nat_mode, depth, eqs)
        if n is not None:
            return n
    pg = parse_pred(t, vs, types, nat_mode)
    if pg is not None:
        n = build_pred_goal(pg, hyps, pre, types, nat_mode, depth, txt=t, eqs=eqs)
        if n is not None:
            return n
    return B.build_tree(t, types, hyps, nat_mode, depth)


def build_exists(bound, body, types, hyps, pre, nat_mode, depth, eqs=None):
    """Leading ∃ block -> exists_w node (witness back-solve + two-route re-check)."""
    TS = [ty for _n, ty in bound]
    if len(set(TS)) != 1 or TS[0] not in S.NUMERIC_TYPES:
        return None
    T = TS[0]
    nv = dict(types)
    for n, ty in bound:
        nv[n] = ty
    body_rel = B.parse_goal_ext_rel(body, set(nv), nv, nat_mode)
    if body_rel is None:
        return None
    rec0 = B._mk_rec("<exists>", nv, nv, hyps, body_rel, nat_mode)
    ws, path = solve_block(bound, T, body_rel, rec0)
    if ws is None or not verify_witness(body_rel, bound, ws, rec0):
        return None
    sub = dict(zip([sp.Symbol(n) for n, _ in bound], ws))
    rec2 = dict(rec0)
    rec2["concl"] = _rel_sub(body_rel, sub)
    got, why = chain_or_rwring(rec2, eqs)
    if got is None:
        return None
    child = B.build_sqrt_node(rec2)
    if child is None or child.get("got") is None:
        child = {"kind": "leaf", "rec": rec2, "got": got, "why": why}
    E = body_rel["lhs"] - body_rel["rhs"]
    vars_ = [sp.Symbol(n) for n, _ in bound] + \
        sorted(E.free_symbols - {sp.Symbol(n) for n, _ in bound}, key=str)
    f0 = frame_profile(E, vars_)
    f1 = [float(sp.N(sp.nsimplify(E.subs(sub))))]
    f1 = f1 * K if f1 and abs(f1[0]) < 1e-9 else None
    return {"kind": "exists_w", "side": "goal", "bound": bound, "ws": ws, "path": path,
            "rec": rec0, "rec2": rec2, "got": got, "why": why, "T": T, "ops": [OP_W],
            "children": [child], "frames": [f0, f1],
            "w_txt": [w_txt(x) for x in ws], "E": E}


def build_pred_goal(pred, hyps, pre, types, nat_mode, depth, txt=None, eqs=None):
    """Predicate goal: (1) goal≡hyp prefers ref (`exact hp`); (2) direct expansion with witness
    back-solve + chain; (3) expand the **predicate hypotheses** one by one (B4 entry) and judge
    again; if everything fails, yield to the parent / fail."""
    ref = _pred_ref(pred, pre)
    if ref is not None:
        return ref
    T = pred_T(pred, types)
    if T is not None:
        w = sp.Symbol("w44")
        rel = unfold_goal(pred, w)
        rec0 = B._mk_rec("<predg>", dict(types), dict(types), hyps, rel, nat_mode)
        ws, path = solve_block([("w44", T)], T, rel, rec0)
        if ws is not None and verify_witness(rel, [("w44", T)], ws, rec0):
            rec2 = dict(rec0)
            rec2["concl"] = _rel_sub(rel, {w: ws[0]})
            got, why = chain_or_rwring(rec2, eqs)
            if got is not None:
                child = {"kind": "leaf", "rec": rec2, "got": got, "why": why}
                E = rel["lhs"] - rel["rhs"]
                vars_ = [w] + sorted(E.free_symbols - {w}, key=str)
                return {"kind": "pred_expand", "side": "goal", "pred": pred, "rec": rec0,
                        "rec2": rec2, "got": got, "why": why, "T": T, "ops": [OP_E],
                        "children": [child], "frames": frames_of_resid(E, vars_),
                        "w": ws[0], "w_txt": w_txt(ws[0]), "path": path, "E": E}
    if txt and pre:
        for i, e in enumerate(pre):
            n = build_hyp_expand(txt, e["pred"], e["name"], types, hyps, pre, nat_mode,
                                 depth, tag=i, eqs=eqs)
            if n is not None and certifiable_ext(n):
                return n
    return None


def _pred_ref(pred, pre):
    """goal ≡ some predicate hypothesis => ref node (the hypothesis participates **by name**)."""
    for e in pre:
        p2 = e["pred"]
        if p2["kind"] != pred["kind"]:
            continue
        a1, a2 = pred_args(pred), pred_args(p2)
        if len(a1) != len(a2):
            continue
        if all(sp.expand(x - y) == 0 for x, y in zip(a1, a2)):
            args = sorted(pred_args(pred), key=lambda e: -len(e.free_symbols))
            f0 = None
            if args and args[0].free_symbols:
                f0 = frame_profile(args[0], sorted(args[0].free_symbols, key=str))
            return {"kind": "pred_ref", "side": "hyp", "pred": pred, "hypname": e["name"],
                    "ops": [OP_R], "children": [], "frames": [f0, list(ZERO)]}
    return None


def build_hyp_expand(txt, pred, hypp, types, hyps, pre, nat_mode, depth, tag=0, eqs=None):
    """Hypothesis-side expansion (B4): rcases the witness + equation => re-judge the subgoal
    under the extended hypotheses."""
    kname, hname = "k44_%d_%d" % (depth, tag), "hk44_%d_%d" % (depth, tag)  # nesting-unique
    ks = sp.Symbol(kname)
    eq = unfold_hyp(pred, ks)
    nh = list(hyps) + [dict(eq, name=hname)]
    npre = [e for e in pre if e["name"] != hypp]
    neqs = list(eqs or []) + [hname]
    ch = build_tree_ext(txt, types, nh, npre, nat_mode, depth + 1, neqs)
    E = eq["lhs"] - eq["rhs"]
    vars_ = [ks] + sorted(E.free_symbols - {ks}, key=str)
    return {"kind": "pred_expand", "side": "hyp", "pred": pred, "hypname": hypp,
            "fresh": kname, "eq_name": hname, "eq": eq, "children": [ch],
            "ops": [OP_E], "frames": frames_of_resid(E, vars_), "E": E,
            "pre_lines": ["try rcases %s with ⟨%s, %s⟩" % (hypp, kname, hname)]}


# ================================================================ §4 rendering
def _closer(got):
    return {"poly": "ring", "eval": "norm_num", "lin": "linarith", "omega": "omega",
            "mod": "norm_num", "pos": "positivity"}.get((got or {}).get("route"), "ring")


def _render_leafish(n, mode, ci=0):
    """exists_w / pred_expand / leaf / sqrt_red -> line chain (None = not renderable)."""
    k = n.get("kind")
    if k == "pred_ref":
        return ["try exact %s" % n["hypname"]]
    if k == "pred_expand" and n.get("side") == "hyp":
        tail = render_tree_ext(n["children"][0], mode, None, None)
        return None if tail is None else list(n["pre_lines"]) + list(tail)
    if k == "exists_w":
        pre = ["try use (%s : %s)" % (wt, n["T"]) for wt in n["w_txt"]]
        cn = n["children"][0]
        tail = _render_leafish(cn, mode, ci) if cn.get("kind") in \
            ("leaf", "sqrt_red") else render_tree_ext(cn, mode, None, None)
        return None if tail is None else pre + list(tail)
    if k == "pred_expand" and n.get("side") == "goal":
        wt, T = n["w_txt"], n["T"]
        pre = pred_unfold_lines(n["pred"], wt, T) + \
            ["try exact ⟨(%s : %s), by %s⟩" % (wt, T, _closer(n.get("got")))]
        cn = n["children"][0]
        tail = _render_leafish(cn, mode, ci)
        if tail is None:
            return None
        out = pre + list(tail)
        if mode == "exam" and "try ring" not in out:
            out = out + ["try ring"]
        return out
    if k in ("leaf", "sqrt_red"):
        if (n.get("got") or {}).get("route") == "rw_ring":
            names = ((n["got"].get("route_payload") or {}).get("eqs") or [])
            if not names:
                return None
            return (["try rw [%s]" % nm for nm in names] + ["try ring"])
        return B._leaf_lines(n, mode, ci)
    return None


def render_tree_ext(n, mode="exam", cimap=None, ctr=None):
    if ctr is None:
        ctr = [0]
    if cimap is None:
        cimap = {}
    k = n.get("kind")
    if k in ("exists_w", "pred_expand", "pred_ref"):
        if k in ("exists_w", "pred_expand"):
            i = ctr[0]
            ctr[0] += 1
            return _render_leafish(n, mode, cimap.get(i, 0))
        return _render_leafish(n, mode, 0)
    if k == "fail":
        return None
    if k == "forall":
        sub = render_tree_ext(n["children"][0], mode, cimap, ctr)
        return None if sub is None else ["try intro " + " ".join(n["intros"])] + sub
    if k == "imp":
        sub = render_tree_ext(n["children"][0], mode, cimap, ctr)
        return None if sub is None else ["try intro " + n["hypname"]] + sub
    if k in ("iff", "and"):
        lines = ["try constructor"]
        for c in n["children"]:
            sub = render_tree_ext(c, mode, cimap, ctr)
            if sub is None:
                return None
            lines += sub
        return lines
    if k == "or":
        for tag, c in (("left", n["children"][0]), ("right", n["children"][1])):
            if certifiable_ext(c):
                sub = render_tree_ext(c, mode, cimap, ctr)
                if sub is not None:
                    return ["try " + tag] + sub
        return None
    if k in ("leaf", "sqrt_red"):
        i = ctr[0]
        ctr[0] += 1
        return B._leaf_lines(n, mode, cimap.get(i, 0))
    return None


def certifiable_ext(n):
    k = n.get("kind")
    if k == "pred_ref":
        return bool(n.get("hypname"))
    if k == "exists_w":
        return n.get("got") is not None and bool(n["children"]) and \
            certifiable_ext(n["children"][0])
    if k == "pred_expand":
        if n.get("side") == "goal":
            return n.get("got") is not None and certifiable_ext(n["children"][0])
        return bool(n["children"]) and certifiable_ext(n["children"][0])
    if k in ("leaf", "sqrt_red"):
        g = n.get("got") or {}
        rel = ((n.get("rec") or {}).get("concl") or {}).get("rel")
        if rel == "∣" and g.get("route") not in ("eval", "mod"):
            return False          # D-P3-3: the parent route_of misjudges ∣ goals as omega-reachable (Lean rejects in practice)
    return B.certifiable(n)          # leaf/sqrt_red/connectives/quantifiers delegated to the parent


def slots_of_ext(n, out=None, deep=True):
    out = [] if out is None else out
    k = n.get("kind")
    if k in NEW_KINDS:
        out.append(n)
        if not deep:
            return out
    for c in n.get("children") or []:
        slots_of_ext(c, out, deep)
    return out


def render_multi_ext(n, mode="gold", cap=8):
    base = render_tree_ext(n, mode, {}, [0])
    if base is None:
        return []
    texts = ["\n".join(base)]
    if len(texts) >= cap:
        return texts
    for i, nd in enumerate(slots_of_ext(n, deep=True)):
        if nd["kind"] == "pred_ref":
            continue
        if nd["kind"] == "leaf":
            rec, got = nd.get("rec"), nd.get("got")
        else:
            rec, got = nd.get("rec2"), nd.get("got")
        if not got or rec is None:
            continue
        try:
            import gen44_render_gold as RG
            nc = len(RG.candidates(F, rec, got))
        except Exception:                                               # noqa: BLE001
            nc = 0
        for j in range(1, nc):
            t = render_tree_ext(n, mode, {i: j}, [0])
            if t is None:
                continue
            s = "\n".join(t)
            if s not in texts:
                texts.append(s)
            if len(texts) >= cap:
                return texts
    return texts


def tree_ops_ext(n):
    ops = []
    k = n.get("kind")
    if k in NEW_KINDS:
        ops += n.get("ops") or []
    if k == "forall":
        ops.append("g44_intro_binder")
    if k in ("iff", "and", "or"):
        ops.append("g44_goal_split")
    for c in n.get("children") or []:
        ops += tree_ops_ext(c)
    return ops


def tree_leaf_ops_ext(n):
    ops = []
    if n.get("kind") in ("leaf", "sqrt_red"):
        ops += list(((n.get("got") or {}).get("chain_ops") or []))
    for c in n.get("children") or []:
        ops += tree_leaf_ops_ext(c)
    return ops


def tree_frames_ext(n):
    fr = []
    if n.get("kind") in NEW_KINDS and n.get("frames"):
        fr += list(n["frames"])
    for c in n.get("children") or []:
        fr += tree_frames_ext(c)
    return fr


def leaf_frame_bad(n, out=None):
    out = [] if out is None else out
    if n.get("kind") in ("leaf", "sqrt_red"):
        fr = (n.get("got") or {}).get("frames") or []
        b = B.assert_frames(fr)
        if b:
            out.append((b, (n.get("rec") or {}).get("name", "?")))
    for c in n.get("children") or []:
        leaf_frame_bad(c, out)
    return out


def assert_frames_full(n):
    """Mechanical assertions (per new node): frames non-empty · chain non-constant · first frame
    non-degenerate · frames != answer · last frame = exhausted frame."""
    bad = []
    for i, nd in enumerate(slots_of_ext(n, deep=False)):
        fr = nd.get("frames")
        if not fr or any(x is None for x in fr):
            bad.append({"i": i, "kind": nd["kind"], "why": "frames_empty"})
            continue
        f0, fl = fr[0], fr[-1]
        if len({tuple(x) for x in fr}) < 2:
            bad.append({"i": i, "kind": nd["kind"], "why": "chain_constant"})
        if len(set(f0)) < 2:
            bad.append({"i": i, "kind": nd["kind"], "why": "first_frame_degenerate"})
        if fl != ZERO:
            bad.append({"i": i, "kind": nd["kind"], "why": "last_not_exhausted"})
        if f0 == ZERO:
            bad.append({"i": i, "kind": nd["kind"], "why": "frame_equals_answer"})
    return bad


# ================================================================ §5 statement parsing (fork)
def parse_stmt_b345(prob):
    full = S.norm_answer(prob.head)
    concl_txt = S.norm_answer(getattr(prob, "concl", "") or "")
    rest = re.sub(r"^theorem\s+", "", full).strip()
    m = re.match(r"^([A-Za-z0-9_.']+)\s*(.*)$", rest, re.S)
    binders = (m.group(2) if m else "").strip()
    if binders.endswith(":"):
        binders = binders[:-1].strip()
    vars_ty, hypgrp = {}, []
    for names, ty, br in S.binder_groups(binders):
        rel, _i = S.top_rel(ty)
        ht = ty.strip()[:1]
        nm0 = any(t in S.NAT_TYPES for t in vars_ty.values())
        if rel or ht in ("∀", "∃", "¬"):
            for n in names:
                hypgrp.append((n, ty))
        elif parse_pred(ty, set(vars_ty), vars_ty, nm0) is not None:
            for n in names:                       # predicate hypothesis (B4 entry): not a variable
                hypgrp.append((n, ty))
        else:
            for n in names:
                vars_ty[n] = ty
    rec = {"name": prob.name, "vars": vars_ty, "types": dict(vars_ty), "hyps": [],
           "pre": [], "extra_hyps": [], "ok": False, "reason": None,
           "concl_txt": concl_txt,
           "nat_mode": any(t in S.NAT_TYPES for t in vars_ty.values()),
           "has_complex": any(t in S.CPLX_TYPES for t in vars_ty.values())}
    if not all(t in S.NUMERIC_TYPES for t in vars_ty.values()):
        rec["reason"] = "nonnumeric_type"
        return rec
    vs = set(vars_ty)
    hyps, pre = [], []
    for gname, gty in hypgrp:
        parts = S.split_top_conj(gty)
        for pi, ptxt in enumerate(parts):
            hname = gname if len(parts) == 1 else "%s.%d" % (gname, pi)
            p = parse_pred(ptxt.strip(), vs, vars_ty, rec["nat_mode"])
            if p is not None:                    # predicate hypotheses take priority (B4 entry; ∣ no longer enters arithmetic hyps)
                pre.append({"name": hname, "pred": p, "txt": ptxt.strip()})
                continue
            h = B.parse_hyp_ext(ptxt.strip(), hname, vs, vars_ty, rec["nat_mode"])
            if h is None:
                rec["extra_hyps"].append(hname)
            else:
                hyps.append(h)
    rec["hyps"], rec["pre"] = hyps, pre
    tr = build_tree_ext(concl_txt, vars_ty, hyps, pre, rec["nat_mode"], 0)
    rec["tree"] = tr
    if tr.get("kind") == "fail":
        rec["reason"] = "goal:" + str(tr.get("reason"))
        return rec
    rec["ok"] = True
    return rec


# ================================================================ §6 payload-ness (witness-slot / name-slot damage)
def perturb_numeral(row, src="concl", delta=1):
    """Numeral-slot damage: add delta to the last integer constant (avoiding digits inside
    identifiers). Returns the modified row or None."""
    t = row.get(src) or ""
    ms = list(re.finditer(r"(?<![A-Za-z_0-9])\d+(?![0-9])", t))
    if not ms:
        return None
    m = ms[-1]
    nt = t[:m.start()] + str(int(m.group(0)) + delta) + t[m.end():]
    if nt == t:
        return None
    r2 = dict(row)
    r2[src] = nt
    return r2


def shift_witness(n, delta=1):
    """Witness-slot damage: add delta to the first witness value (the text must change; Lean must
    reject). Returns (new tree, position description) or (None, why)."""
    if n.get("kind") == "exists_w":
        m = dict(n)
        m["ws"] = [w + delta for w in n["ws"]]
        m["w_txt"] = [w_txt(w) for w in m["ws"]]
        return m, "exists_w:%s" % m["w_txt"]
    if n.get("kind") == "pred_expand" and n.get("side") == "goal":
        m = dict(n)
        m["w"] = n["w"] + delta
        m["w_txt"] = w_txt(m["w"])
        return m, "pred_expand:%s" % m["w_txt"]
    if n.get("kind") == "pred_expand" and n.get("side") == "hyp":
        m = dict(n)
        pl = list(n["pre_lines"])
        m["pre_lines"] = [pl[0].replace(n["fresh"], n["fresh"] + "_z")] + pl[1:]
        return m, "hyp_expand:rename"
    if n.get("kind") == "pred_ref":
        m = dict(n)
        m["hypname"] = n["hypname"] + "_z"
        return m, "pred_ref:rename"
    for i, c in enumerate(n.get("children") or []):
        r, why = shift_witness(c, delta)
        if r is not None:
            m = dict(n)
            ch = list(n["children"])
            ch[i] = r
            m["children"] = ch
            return m, why
    return None, "no_slot"


def first_witness(n):
    if n.get("kind") == "exists_w":
        return {"kind": "exists_w", "w": n["ws"][0], "w_txt": n["w_txt"][0], "T": n["T"]}
    if n.get("kind") == "pred_expand" and n.get("side") == "goal":
        return {"kind": "pred_expand", "w": n["w"], "w_txt": n["w_txt"], "T": n["T"]}
    for c in n.get("children") or []:
        r = first_witness(c)
        if r:
            return r
    return None


# ================================================================ §7 self-test
def selftest():
    st = []
    d = parse_stmt_b345(B.P("t1", "theorem t1 (n : ℤ) (h : Even n)", "Even n"))
    assert d["ok"] and d["tree"]["kind"] == "pred_ref", d.get("reason")
    st.append("b4_ref")
    d = parse_stmt_b345(B.P("t2", "theorem t2 (n : ℤ) (h : Even n)", "Odd (n + 1)"))
    assert d["ok"] and d["tree"]["kind"] == "pred_expand" and d["tree"]["side"] == "hyp", \
        d.get("reason")
    st.append("b4_expand")
    d = parse_stmt_b345(B.P("t3", "theorem t3 (n : ℤ) (h : n = 8)", "Even n"))
    assert d["ok"] and d["tree"]["kind"] == "pred_expand" and d["tree"]["side"] == "goal", \
        d.get("reason")
    assert d["tree"]["w_txt"] == "4", d["tree"]["w_txt"]
    st.append("b5_even")
    d = parse_stmt_b345(B.P("t4", "theorem t4 (a b : ℤ) (h : b = 3 * a)", "a ∣ b"))
    assert d["ok"] and d["tree"]["kind"] == "pred_expand" and d["tree"]["w_txt"] == "3", \
        d.get("reason")
    st.append("b5_dvd")
    d = parse_stmt_b345(B.P("t5", "", "∃ x : ℚ, x ^ 2 - 5 * x + 6 = 0"))
    assert d["ok"] and d["tree"]["kind"] == "exists_w" and d["tree"]["w_txt"] == ["2"], \
        (d.get("reason"), d.get("tree", {}).get("kind"))
    st.append("b3_polyroot")
    d = parse_stmt_b345(B.P("t6", "", "∃ x : ℤ, ∃ y : ℤ, 3 * x + 5 * y = 1"))
    assert d["ok"] and d["tree"]["kind"] == "exists_w" and len(d["tree"]["ws"]) == 2, \
        d.get("reason")
    st.append("b3_dio2")
    d = parse_stmt_b345(B.P("t7", "", "∃ k : ℤ, 35 = k * 7"))
    assert d["ok"] and d["tree"]["w_txt"] == ["5"], d.get("reason")
    st.append("b3_dvd_eq")
    d = parse_stmt_b345(B.P("t8", "theorem t8 (m p : ℤ) (h : p = 7 + 2 * m)",
                            "p ≡ 7 [ZMOD 2]"))
    assert d["ok"] and d["tree"]["kind"] == "pred_expand", d.get("reason")
    st.append("b5_modeq")
    for nm, hd, cc in (("s1", "", "∃ x : ℚ, x ^ 2 - 5 * x + 6 = 0"),
                       ("s2", "theorem s2 (n : ℤ) (h : Even n)", "Odd (n + 1)"),
                       ("s3", "theorem s3 (n : ℤ) (h : n = 8)", "Even n")):
        d = parse_stmt_b345(B.P(nm, hd, cc))
        assert d["ok"], (nm, d.get("reason"))
        txt = render_tree_ext(d["tree"], "exam")
        assert txt, nm
        assert not assert_frames_full(d["tree"]), (nm, assert_frames_full(d["tree"]))
    st.append("render+frames")
    return st


if __name__ == "__main__":
    print({"selftest": selftest(), "ops": {OP_W: B.OP_IDX[OP_W],
                                           OP_R: B.OP_IDX[OP_R], OP_E: B.OP_IDX[OP_E]}})

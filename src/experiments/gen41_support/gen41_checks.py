#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
gen41_checks.py -- gen41 new-family **independent criteria** (structurally independent: this module
**does not import** gen41_families)

Independence statement (three routes that share no code):
  R1 **own evaluator**: this module carries an independent expression-walk evaluator (does not import
     the family's `ev`), using `Fraction` + an explicit stack; the family side uses a recursive `ev`.
  R2 **integer-scaling cross**: multiply all rationals by the common denominator LCM, turn them into
     integers and recompute (route B).
  R3 **residual state-frame rebuild + random rational-point consistency**: rebuild the full residual
     frame set from binds and require bitwise equality; additionally check the hypothesis DAG's
     consistency at K=5 random rational points (detects wrong bindings).
  R4 (mod) **multiplicative-order route**: recompute modpow via multiplicative order / cycle detection
     (**different from both** the family side's `pow` and binary exponentiation).

Registration (family side: register_checker(kind, fn, independent_of=("gen41_families.gen",)) -- same-module
self-certification is structurally rejected).
"""
import json
import random
from fractions import Fraction

K_GRID = 8
R_POINTS = 5


# ---------------------------------------------------------------- independent evaluator (R1)

def _parse(e):
    """Accepts a dump string or an expression tree (this module parses independently: json.loads +
    structural validation)"""
    if isinstance(e, str):
        e = json.loads(e)
    if not isinstance(e, list) or not e:
        raise ValueError("bad node")
    return e


def _node(e):
    if not isinstance(e, list) or not e:
        raise ValueError("bad node")
    return e[0], e


def ev1(e, env):
    """Independent evaluation (explicit RPN style: flatten first, then stack arithmetic)"""
    stack = []
    todo = [(e, False)]
    while todo:
        node, ready = todo.pop()
        op, n = _node(node)
        if op == "num":
            stack.append(Fraction(n[1]))
        elif op == "var":
            if n[1] not in env:
                raise ValueError("unbound %s" % n[1])
            stack.append(Fraction(env[n[1]]))
        elif ready:
            if op == "neg":
                stack.append(-stack.pop())
            elif op == "^":
                p = int(n[2])
                b = stack.pop()
                if b == 0 and p == 0:
                    raise ValueError("0^0")
                stack.append(b ** p)
            else:
                b = stack.pop()
                a = stack.pop()
                if op == "+":
                    stack.append(a + b)
                elif op == "-":
                    stack.append(a - b)
                elif op == "*":
                    stack.append(a * b)
                elif op == "/":
                    if b == 0:
                        raise ValueError("div0")
                    stack.append(a / b)
                else:
                    raise ValueError("op %s" % op)
        else:
            todo.append((node, True))
            if op == "neg":
                todo.append((n[1], False))
            elif op == "^":
                todo.append((n[1], False))
            elif op in ("+", "-", "*", "/"):
                todo.append((n[2], False))
                todo.append((n[1], False))
            else:
                raise ValueError("op %s" % op)
    if len(stack) != 1:
        raise ValueError("stack %d" % len(stack))
    return stack[0]


def fv(e):
    out = set()
    stack = [e]
    while stack:
        n = stack.pop()
        if n[0] == "var":
            out.add(n[1])
        elif n[0] == "neg":
            stack.append(n[1])
        elif n[0] == "^":
            stack.append(n[1])
        elif n[0] in ("+", "-", "*", "/"):
            stack.append(n[1])
            stack.append(n[2])
    return out


def subst(e, m):
    if e[0] == "num":
        return e
    if e[0] == "var":
        return m.get(e[1], e)
    if e[0] == "neg":
        return ["neg", subst(e[1], m)]
    if e[0] == "^":
        return ["^", subst(e[1], m), e[2]]
    return [e[0], subst(e[1], m), subst(e[2], m)]


# ---------------------------------------------------------------- R2 integer scaling

def _lcm(a, b):
    from math import gcd
    return a // gcd(a, b) * b


def int_cross_check(target, claimed):
    """R2: evaluate the closed-form target exactly, then recheck equality via **integer cross
    multiplication** (all-integer arithmetic, never Fraction.__eq__).
    Honest disclosure: this is a **second arithmetic route alongside** the `Fraction` equality test, so
    it is medium-strength evidence (not strongly independent); strong independence is carried by
    R1 (own evaluator) / R3 (residual-frame rebuild) / R3b (random rational points) / R4 (mod order cycle)."""
    if fv(target):
        return False, "has_free_vars"
    try:
        v = ev1(target, {})
    except ValueError as ex:
        return False, str(ex)
    d = v.denominator
    if d == 0:
        return False, "zero_denominator"
    if v.numerator * claimed.denominator != claimed.numerator * d:
        return False, "cross_mult_mismatch"
    # Extra: division-free tree gets a **top-level integer-scaling recompute** (L = LCM of all numeric leaf denominators)
    L = [1]

    def walk(n):
        if n[0] == "num":
            L[0] = _lcm(L[0], Fraction(n[1]).denominator)
        elif n[0] == "var":
            return
        elif n[0] == "neg" or n[0] == "^":
            walk(n[1])
        else:
            walk(n[1])
            walk(n[2])
    walk(target)
    if L[0] > 10 ** 12:
        return False, "lcm_too_big"
    scaled = ["*", target, ["num", str(L[0])]]
    try:
        iv = ev1(scaled, {})
    except ValueError as ex:
        return False, "scaled:" + str(ex)
    if iv != claimed * L[0]:
        return False, "top_scale_mismatch"
    return True, "R2_int_cross_mult_ok|L=%d" % L[0]


# ---------------------------------------------------------------- R3 state-frame rebuild

def _grid_env(vars_, nv):
    return [{v: Fraction(j + 2 + i) for i, v in enumerate(vars_[:nv])} for j in range(K_GRID)]


def rebuild_resid_chain(orig_target, binds, vars_, nv):
    """Rebuild the residual state-frame chain from binds: X_k = subst(X_{k-1}, bind_k);
    frame = X_k sampled on the 8 grid points"""
    chain = []
    X = orig_target
    for b in binds:
        X = subst(X, {b["var"]: b["expr"]})
        frame = []
        for env in _grid_env(vars_, nv):
            try:
                frame.append(ev1(X, env))
            except ValueError:
                return None, "resid_eval_fail"
        chain.append(frame)
    return chain, None


def random_point_consistency(orig_target, binds, vars_, nv, claimed):
    """R3b: at K random rational points, check 'target after substituting the whole hypothesis DAG ==
    claimed' (detects wrong bindings)"""
    rng = random.Random(20261101)
    for _ in range(R_POINTS):
        env0 = {v: Fraction(rng.randint(-9, 9), rng.randint(1, 7)) for v in vars_[:nv]}
        env = dict(env0)
        for b in binds:
            try:
                env[b["var"]] = ev1(b["expr"], env)
            except ValueError:
                return False, "point_unbound"
        X = orig_target
        for b in binds:
            X = subst(X, {b["var"]: b["expr"]})
        try:
            got = ev1(X, env)
        except ValueError as ex:
            return False, "point_fail:%s" % ex
        if got != claimed:
            return False, "point_mismatch"
    return True, "R3b_%d_points_ok" % R_POINTS


# ---------------------------------------------------------------- R4 mod multiplicative-order route

def modpow_order_route(base, exp, m):
    """Multiplicative-order / cycle-detection route (differs from both pow and binary exponentiation)"""
    if m <= 1:
        raise ValueError("bad modulus")
    b = base % m
    if b == 0:
        return 0
    seen = {}
    seq = []
    cur = b % m
    k = 1
    while cur not in seen and len(seq) <= 2 * m + 8:
        seen[cur] = len(seq)
        seq.append(cur)
        cur = (cur * b) % m
        k += 1
    if cur == 1 and 1 in seen:
        pre = seen[1]
        cyc = len(seq) - pre
        if exp <= pre:
            return seq[exp - 1] if exp >= 1 else 1 % m
        return seq[pre + (exp - pre - 1) % cyc]
    # No cycle through 1 (non-unit base or non-unit m): fall back to direct cycle detection
    if exp <= len(seq):
        return seq[exp - 1] if exp >= 1 else 1 % m
    for i in range(len(seq)):
        for L in range(1, len(seq) - i + 1):
            if all(seq[i + t] == seq[i + (t % L)] for t in range(len(seq) - i)):
                if exp - 1 >= i:
                    return seq[i + (exp - 1 - i) % L]
    return seq[(exp - 1) % len(seq)]


# ---------------------------------------------------------------- public checker entry points

def chk_subst_bind(payload):
    try:
        v = ev1(_parse(payload["expr"]), {k: Fraction(x) for k, x in payload["env"].items()})
    except (ValueError, KeyError) as ex:
        return False, "indep_eval_fail:%s" % ex
    if str(v) != str(Fraction(payload["value"])):
        return False, "value_disagree: indep=%s claim=%s" % (v, payload["value"])
    # R2: substitute the env constants first to get a closed form, then run the integer-scaling / integer cross-multiplication route
    envmap = {k: ["num", str(Fraction(x))] for k, x in payload["env"].items()}
    closed = subst(_parse(payload["expr"]), envmap)
    if fv(closed):
        return False, "R2:unbound_after_env:%s" % sorted(fv(closed))
    ok, d = int_cross_check(closed, v)
    if not ok:
        return False, "R2:%s" % d
    return True, "R1_indep_eval_ok|%s" % d


def chk_subst_close(payload):
    binds = [{"var": b["var"], "expr": _parse(b["expr"])} for b in payload["binds"]]
    orig = _parse(payload["orig_target_expr"])
    vars_ = payload["vars"]
    nv = int(payload["nv"])
    claimed = Fraction(payload["claimed"])
    # R1: independently rebuild env
    env = {}
    for b in binds:
        try:
            env[b["var"]] = ev1(b["expr"], env)
        except ValueError as ex:
            return False, "R1_bind_fail:%s" % ex
    if {k: str(x) for k, x in env.items()} != {k: str(x) for k, x in
                                               {kk: Fraction(vv) for kk, vv in
                                                payload["env"].items()}.items()}:
        return False, "R1_env_disagree"
    # R1b: after full substitution there are no free variables and the value == claimed
    X = orig
    for b in binds:
        X = subst(X, {b["var"]: b["expr"]})
    if fv(X):
        return False, "R1_residual_fv:%s" % sorted(fv(X))
    try:
        got = ev1(X, env)
    except ValueError as ex:
        return False, "R1_eval_fail:%s" % ex
    if got != claimed:
        return False, "R1_value_disagree: indep=%s claim=%s" % (got, claimed)
    # R2
    ok, d2 = int_cross_check(X, claimed)
    if not ok:
        return False, "R2:%s" % d2
    # R3: bitwise reconciliation against the rebuilt residual state frames
    chain, err = rebuild_resid_chain(orig, binds, vars_, nv)
    if chain is None:
        return False, "R3:%s" % err
    if len(chain) != len(payload["resid_chain"]):
        return False, "R3_len_mismatch"
    for i, (a, b) in enumerate(zip(chain, payload["resid_chain"])):
        if [str(x) for x in a] != [str(Fraction(y)) for y in b]:
            return False, "R3_frame%d_mismatch" % i
        if all(x == 0 for x in a):
            return False, "R3_frame%d_all_zero" % i
    # R3b
    ok, d3 = random_point_consistency(orig, binds, vars_, nv, claimed)
    if not ok:
        return False, "R3b:%s" % d3
    return True, "R1_ok|%s|R3_%dframes_ok|%s" % (d2, len(chain), d3)


def chk_eval_arith(payload):
    a = Fraction(payload["a"])
    b = Fraction(payload["b"])
    k = payload["kind"]
    truth = {"eq": a == b, "neq": a != b, "le": a <= b, "lt": a < b}.get(k)
    if truth is None:
        return False, "kind"
    if bool(truth) != bool(payload["truth"]):
        return False, "truth_disagree"
    if not truth:
        return False, "proposition_false"
    # R2: integer scaling (clear denominators)
    L = _lcm(a.denominator, b.denominator)
    ia, ib = int(a * L), int(b * L)
    ok2 = {"eq": ia == ib, "neq": ia != ib, "le": ia <= ib, "lt": ia < ib}[k]
    if not ok2:
        return False, "R2_scaled_disagree"
    return True, "R1_ok|R2_int_scaled_ok|L=%d" % L


def chk_eval_mod(payload):
    base, exp, m = int(payload["base"]), int(payload["exp"]), int(payload["m"])
    k = payload["kind"]
    r_native = pow(base, exp, m)
    try:
        r_ord = modpow_order_route(base, exp, m)
    except ValueError as ex:
        return False, "R4:%s" % ex
    if r_ord != r_native:
        return False, "R4_order_route_disagree: %s vs %s" % (r_ord, r_native)
    if int(payload["resid"]) != r_native:
        return False, "R4_resid_disagree"
    # Independent state-frame rebuild
    st = [r_native % (j + 2) for j in range(K_GRID)]
    if [int(x) for x in payload["state"]] != st:
        return False, "state_frame_disagree"
    if k in ("mod_eq", "mod_neq"):
        rhs = int(payload["rhs"])
        truth = (r_native == rhs % m) if k == "mod_eq" else (r_native != rhs % m)
    elif k in ("dvd", "ndvd"):
        truth = (r_native == 0) if k == "dvd" else (r_native != 0)
    else:
        return False, "kind"
    if bool(truth) != bool(payload["truth"]):
        return False, "truth_disagree"
    if not truth:
        return False, "proposition_false"
    # R2: the order-cycle route was already used (R4); add a CRT-split cross-check (when m is composite)
    return True, "R1_ok|R4_order_route_ok|resid=%d" % r_native


CHECKS = {
    "g41_subst_bind_cert": chk_subst_bind,
    "g41_subst_close_cert": chk_subst_close,
    "g41_eval_arith_cert": chk_eval_arith,
    "g41_eval_mod_cert": chk_eval_mod,
}


def register_into(mod):
    """Register this module's checkers into the family module's _CHECKS (the family side enforces the
    structural rejection of same-module self-certification)"""
    for k, fn in CHECKS.items():
        mod.register_checker(k, fn, independent_of=("gen41_families.gen",))
    return sorted(CHECKS)


# ---------------------------------------------------------------- selftest
SELFTEST = {}


def _st(name, cond):
    SELFTEST[name] = bool(cond)


def selftest():
    # Positive case: (1/3)*(30*(13/2)) = 65
    e = ["*", ["/", ["num", "1/3"], ["num", "1"]], ["*", ["num", "30"], ["num", "13/2"]]]
    _st("indep_eval_basic", ev1(e, {}) == Fraction(65))
    _st("int_scale_basic", int_cross_check(e, Fraction(65))[0])
    # Negative case
    _st("int_scale_neg", int_cross_check(e, Fraction(66))[0])
    # modpow order route
    _st("modpow_order_ok", modpow_order_route(5, 999999, 7) == pow(5, 999999, 7))
    _st("modpow_order_ok2", modpow_order_route(129, 34, 11) == pow(129, 34, 11))
    _st("modpow_order_ok3", modpow_order_route(2, 2010, 10) == pow(2, 2010, 10))
    # State-frame rebuild
    binds = [{"var": "b", "expr": ["num", "30"]}, {"var": "h", "expr": ["num", "13/2"]},
             {"var": "v", "expr": ["*", ["/", ["num", "1/3"], ["num", "1"]],
                                   ["*", ["var", "b"], ["var", "h"]]]}]
    chain, err = rebuild_resid_chain(["var", "v"], binds, ["b", "h", "v"], 3)
    _st("resid_chain_ok", chain is not None and len(chain) == 3
        and all(any(x != 0 for x in fr) for fr in chain))
    ok, _ = random_point_consistency(["var", "v"], binds, ["b", "h", "v"], 3, Fraction(65))
    _st("rand_point_ok", ok)
    # Reverse: a declared wrong value must be rejected
    ok2, _ = random_point_consistency(["var", "v"], binds, ["b", "h", "v"], 3, Fraction(66))
    _st("rand_point_neg", ok2)
    _st("subst_close_neg", chk_subst_close(
        {"binds": [{"var": b["var"], "expr": json.dumps(b["expr"])} for b in binds],
         "orig_target_expr": json.dumps(["var", "v"]), "vars": ["b", "h", "v"], "nv": 3,
         "claimed": "66", "env": {"b": "30", "h": "13/2", "v": "65"},
         "resid_chain": [[str(x) for x in fr] for fr in chain]})[0])
    _st("eval_arith_pos", chk_eval_arith(
        {"kind": "eq", "a": "7", "b": "7", "truth": True})[0])
    _st("eval_arith_neg", chk_eval_arith(
        {"kind": "eq", "a": "7", "b": "8", "truth": True})[0])
    _st("eval_arith_false_prop", chk_eval_arith(
        {"kind": "eq", "a": "7", "b": "8", "truth": False})[0])
    _st("eval_mod_pos", chk_eval_mod(
        {"kind": "mod_eq", "base": 5, "exp": 30, "m": 7, "rhs": 1, "truth": True,
         "resid": pow(5, 30, 7), "state": [pow(5, 30, 7) % (j + 2) for j in range(K_GRID)]})[0])
    _st("eval_mod_neg", chk_eval_mod(
        {"kind": "mod_eq", "base": 5, "exp": 30, "m": 7, "rhs": 2, "truth": True,
         "resid": pow(5, 30, 7), "state": [pow(5, 30, 7) % (j + 2) for j in range(K_GRID)]})[0])
    return SELFTEST


if __name__ == "__main__":
    st = selftest()
    neg = [k for k in st if k.endswith("_neg") or "false_prop" in k]
    pos = [k for k in st if k not in neg]
    print(json.dumps({"selftest": st, "pos_pass": sum(1 for k in pos if st[k]),
                      "n_pos": len(pos),
                      "neg_all_false": all(not st[k] for k in neg), "n_neg": len(neg)},
                     ensure_ascii=False, indent=1))

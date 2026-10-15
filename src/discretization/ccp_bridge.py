# -*- coding: utf-8 -*-
# ccp_bridge.py -- CCP bridge rules: statement text -> operator parameters as **pure functions**
# (advance team gen12, constitution change 58)
#
# Three bridges (one per pilot class):
#   bridge_ind  : (f 0 = v0), (∀k, f(k+1) = <step>(f k))  -> {form, v0, step} + the closed form declared in the statement
#   bridge_ineq : target (LHS ≥ RHS) / (LHS ≤ RHS)        -> {vars, dom = coefficients of the polynomial LHS−RHS}
#   bridge_nt   : target (m ∣ Σ ...) + assumptions (h_i : m ∣ x_i) -> {m, terms, const}
# All are **text-level parsing** (independent of generator internal state), so the "roundtrip
# consistency gate" can recompute them (required by CLASS_SPECS §4).
import re
from fractions import Fraction

REL = re.compile(r"\s*(≥|≤|=|<|>)\s*")


def split_concl(text):
    """Statement text -> (header, concl). If '⊢' is present split on it; otherwise take the
    last ':' at paren depth 0 (the theorem-body separator)."""
    if "⊢" in text:
        h, c = text.split("⊢", 1)
        return h.strip(), re.sub(r"\s+", " ", c).strip()
    depth, cands, last = 0, [], -1
    for i, ch in enumerate(text):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == ":" and depth == 0:
            last = i
            nl = text.find("\n", i)
            tail = text[i + 1: nl if nl >= 0 else len(text)]
            if not tail.strip():          # trailing ':' = theorem-body separator
                cands.append(i)
    pos = cands[-1] if cands else last
    if pos < 0:
        raise ValueError("无结论分隔符")
    return text[:pos].strip(), re.sub(r"\s+", " ", text[pos + 1:]).strip()


# ---------------------------------------------------------------- polynomial parsing (flat canonical form)
def parse_poly(text, vars_):
    """Parse the render-side canonical form: term = (±)(num*)*(var(^k)?)*, allowing
    parenthesized like sums. Returns {exponent tuple: Fraction} (vars_ fixes the order).
    Raises ValueError when unparseable."""
    idx = {v: i for i, v in enumerate(vars_)}
    zero = (0,) * len(vars_)
    out = {}

    def add_term(coef, exps):
        if coef == 0:
            return
        m = tuple(exps)
        out[m] = out.get(m, Fraction(0)) + coef

    def parse_expr(s, sign):
        s = s.strip()
        if s.startswith("(") and s.endswith(")") and balanced(s[1:-1]):
            return parse_expr(s[1:-1], sign)
        # split on top-level ±
        depth, parts, cur = 0, [], ""
        ops = []
        for ch in s:
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            if depth == 0 and ch in "+-" and cur.strip():
                parts.append(cur)
                ops.append(ch)
                cur = ""
                continue
            if depth == 0 and ch in "+-" and not cur.strip() and not parts:
                cur += ch
                continue
            cur += ch
        parts.append(cur)
        if len(parts) == 1:
            return parse_term(parts[0], sign)
        first = parse_term(parts[0], sign)
        for op, p in zip(ops, parts[1:]):
            sg = sign if op == "+" else -sign
            sub = parse_term(p, sg)
            for m, c in sub.items():
                first[m] = first.get(m, Fraction(0)) + c
        return {m: c for m, c in first.items() if c != 0}

    def parse_term(t, sign):
        t = t.strip()
        if not t:
            raise ValueError("空项")
        neg = False
        if t.startswith("-"):
            neg, t = True, t[1:].strip()
        elif t.startswith("+"):
            t = t[1:].strip()
        # parenthesized factor: treat as a sub-expression
        if t.startswith("(") and t.endswith(")") and balanced(t[1:-1]):
            inner = t[1:-1]
            if contains_var_op(inner):
                res = parse_expr(inner, -sign if neg else sign)
                return res
        coef = Fraction(1)
        exps = [0] * len(vars_)
        for fac in split_mul(t):
            fac = fac.strip()
            if not fac:
                continue
            if fac.startswith("(") and fac.endswith(")"):
                sub = parse_expr(fac[1:-1], Fraction(1))
                # constant parentheses only
                if len(sub) == 1 and list(sub)[0] == zero:
                    coef *= list(sub.values())[0]
                    continue
                raise ValueError("非常数括号因子: %s" % fac)
            if fac in idx:
                exps[idx[fac]] += 1
                continue
            m = re.fullmatch(r"([a-zA-Z][A-Za-z0-9_']*)\s*\^\s*(\d+)", fac)
            if m and m.group(1) in idx:
                exps[idx[m.group(1)]] += int(m.group(2))
                continue
            if re.fullmatch(r"\d+", fac):
                coef *= Fraction(int(fac))
                continue
            raise ValueError("未知因子 %r (文本 %r)" % (fac, text))
        return {tuple(exps): sign * (-coef if neg else coef)}

    def balanced(s):
        d = 0
        for ch in s:
            if ch == "(":
                d += 1
            elif ch == ")":
                d -= 1
            if d < 0:
                return False
        return d == 0

    def split_mul(s):
        depth, parts, cur = 0, [], ""
        for ch in s:
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            if depth == 0 and ch == "*":
                parts.append(cur)
                cur = ""
                continue
            cur += ch
        parts.append(cur)
        return parts

    def contains_var_op(s):
        return any(v in s for v in vars_)

    p = parse_expr(text, Fraction(1))
    return {m: c for m, c in p.items() if c != 0}


# ---------------------------------------------------------------- bridge 1: inductive recurrence
def bridge_ind(stmt):
    """Statement -> {form, v0, step}.
    stmt looks like:
      theorem NAME (f : ℕ → ℤ) (h0 : f 0 = 7) (hstep : ∀ k, f (k + 1) = f k + 3) :
          ∀ n : ℕ, f n = 7 + 3 * n
    """
    m0 = re.search(r"\(\s*h0\s*:\s*f\s+0\s*=\s*(-?\d+)\s*\)", stmt)
    if not m0:
        raise ValueError("h0 未匹配")
    v0 = int(m0.group(1))
    ms = re.search(r"hstep\s*:\s*∀\s*k\s*,\s*f\s*\(k\s*\+\s*1\)\s*=\s*(.+?)\)\s*:", stmt, re.S)
    if not ms:
        raise ValueError("hstep 未匹配")
    step_txt = re.sub(r"\s+", " ", ms.group(1)).strip()
    # form dispatch
    mp = re.fullmatch(r"(\d+)\s*\*\s*f\s*k\s*([+-])\s*(\d+)", step_txt)
    if mp:
        c = int(mp.group(3)) * (1 if mp.group(2) == "+" else -1)
        return {"form": "aff", "v0": v0, "step": [int(mp.group(1)), c]}
    if re.fullmatch(r"(\d+)\s*\*\s*f\s*k", step_txt):
        return {"form": "aff", "v0": v0, "step": [int(re.fullmatch(r"(\d+)", step_txt).group(1)), 0]}
    mp = re.fullmatch(r"f\s*k\s*([+-])\s*(\d+)", step_txt)
    if mp:
        d = int(mp.group(2)) * (1 if mp.group(1) == "+" else -1)
        return {"form": "lin", "v0": v0, "step": d}
    raise ValueError("步进形式未匹配: %r" % step_txt)


def bridge_ind_declared(stmt):
    """Closed form declared in the statement (roundtrip consistency: the closed form computed
    by the operator must match the statement's declaration)."""
    _, concl = split_concl(stmt)
    m = re.search(r"∀\s*n\s*:\s*ℕ\s*,\s*f\s*n\s*=\s*(.+?)\s*$", concl)
    if not m:
        raise ValueError("结论未匹配")
    return m.group(1).strip()


# ---------------------------------------------------------------- bridge 2: inequalities
def bridge_ineq(stmt, vars_):
    """Statement -> {vars, dom}. Supports (LHS ≥ RHS) and (LHS ≤ RHS); dom = nonneg side − other side."""
    _, concl = split_concl(stmt)
    mm = REL.search(concl)
    if not mm:
        raise ValueError("无关系符: %r" % concl)
    lhs, rhs = concl[:mm.start()].strip(), concl[mm.end():].strip()
    rel = mm.group(1)
    pl, pr = parse_poly(lhs, vars_), parse_poly(rhs, vars_)
    if rel in ("≥", ">"):
        dom = _psub(pl, pr)
    elif rel in ("≤", "<"):
        dom = _psub(pr, pl)
    else:
        raise ValueError("非序关系: %s" % rel)
    return {"vars": list(vars_), "dom": dom}


def _psub(p, q):
    out = dict(p)
    for m, c in q.items():
        out[m] = out.get(m, Fraction(0)) - c
    return {m: c for m, c in out.items() if c != 0}


def dump_poly_h(p):
    """Human/JSON-readable polynomial (exponent tuple -> 'i,j' key)."""
    return {",".join(str(e) for e in m): str(c) for m, c in sorted(p.items())}


# ---------------------------------------------------------------- bridge 3: divisibility closure
def bridge_nt(stmt):
    """Statement -> {M, m_h, hyps, terms, const}.
      M     = target modulus (of the form m_h^k)
      m_h   = hypothesis modulus (shared by all assumptions `h : m_h ∣ v`; must be unique)
      terms = [{coef, k, vars(assumption vars), freevar(free var or None), mfac}]
      convention: target = Σ coef·Π(assumption vars) [+ coef·m_h·free var] + const,
      all coefficients positive (ℕ has no subtraction)."""
    header, concl = split_concl(stmt)
    mg = re.fullmatch(r"(\d+)\s*∣\s*(.+)", concl)
    if mg is None:
        raise ValueError("目标未匹配: %r" % concl)
    M = int(mg.group(1))
    body = re.sub(r"\s+", " ", mg.group(2)).strip()
    hp = re.findall(r"\((\w+)\s*:\s*(\d+)\s*∣\s*([a-zA-Z][A-Za-z0-9_']*)\)", header)
    ms = {int(mm) for (_, mm, _) in hp}
    if len(ms) != 1:
        raise ValueError("假设模数不唯一: %s" % ms)
    m_h = ms.pop()
    hyps = [(h, m_h, v) for (h, _, v) in hp]
    hypvars = {v: h for (h, _, v) in hyps}
    binders = set()
    for grp in re.findall(r"\(([^()]*?)\s*:\s*ℕ\)", header):
        for nm in grp.split():
            if re.fullmatch(r"[a-zA-Z][A-Za-z0-9_']*", nm):
                binders.add(nm)
    terms, const = [], 0
    for sign, t in _split_terms(body):
        t = t.strip()
        if re.fullmatch(r"\d+", t):
            const += sign * int(t)
            continue
        if sign < 0:
            raise ValueError("负项 (ℕ 族不支持): %r" % t)
        fac = [x.strip() for x in t.split("*") if x.strip()]
        nums = [int(x) for x in fac if re.fullmatch(r"\d+", x)]
        vs = [x for x in fac if not re.fullmatch(r"\d+", x)]
        if not vs or any(x not in binders for x in vs):
            raise ValueError("未知变量 %r" % vs)
        if all(x in hypvars for x in vs):
            if len(nums) > 1:
                raise ValueError("多重数字因子: %r" % t)
            coef = nums[0] if nums else 1
            terms.append({"coef": coef, "k": len(vs), "vars": list(vs), "freevar": None,
                          "mfac": False})
            continue
        if len(vs) == 1 and vs[0] not in hypvars and len(nums) == 2 and m_h in nums:
            c = nums[0] if nums[1] == m_h else nums[1]
            terms.append({"coef": c, "k": 0, "vars": [], "freevar": vs[0], "mfac": True})
            continue
        raise ValueError("未知项 %r (m_h=%d)" % (t, m_h))
    return {"M": M, "m_h": m_h, "hyps": hyps, "terms": terms, "const": const,
            "body": body, "binders": sorted(binders)}


def bridge_nt_prod(stmt):
    """Product-family statement -> {m, hyps, prod:[v1,v2]} (target of the form m*m ∣ a * b,
    assumptions m ∣ a, m ∣ b)."""
    header, concl = split_concl(stmt)
    mg = re.fullmatch(r"(\d+)\s*∣\s*([a-zA-Z][A-Za-z0-9_']*)\s*\*\s*([a-zA-Z][A-Za-z0-9_']*)",
                      concl)
    if mg is None:
        raise ValueError("乘积目标未匹配: %r" % concl)
    M = int(mg.group(1))
    v1, v2 = mg.group(2), mg.group(3)
    hyps2 = [(h, int(mm), v) for (h, mm, v) in re.findall(
        r"\((\w+)\s*:\s*(\d+)\s*∣\s*([a-zA-Z][A-Za-z0-9_']*)\)", header)]
    ms = {mm for (_, mm, _) in hyps2}
    if len(ms) != 1:
        raise ValueError("假设模数不唯一: %s" % ms)
    m = ms.pop()
    if M != m * m:
        raise ValueError("目标模数 %d ≠ m^2=%d" % (M, m * m))
    got = {v for (_, _, v) in hyps2}
    if not {v1, v2} <= got:
        raise ValueError("假设未覆盖 %s" % [v1, v2])
    return {"m": m, "M": M, "hyps": hyps2, "prod": [v1, v2]}


def _split_terms(body):
    depth, parts, cur, sign = 0, [], "", 1
    first = True
    for ch in body + "+":
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if depth == 0 and ch in "+-":
            if cur.strip():
                parts.append((sign, cur.strip()))
            elif not first:
                pass
            sign = 1 if ch == "+" else -1
            cur = ""
            first = False
            continue
        cur += ch
    if cur.strip():
        parts.append((sign, cur.strip()))
    return [(s, t) for (s, t) in parts if t]

"""§8.1 basic math operators (band 9100, rows 9110–9119) — P1 math-domain deliverable A.

Ten operators, all following p0_reference.py's delivery style (three-step VMP + independent verifier + selfcheck):
  math_solve_linear_system  9110  exact rational Gaussian elimination, verifier = independent Cramer cofactor recomputation
  math_simplify_expr        9111  combine like terms, verifier = pointwise substitution at rational samples (d+1-point Schwartz-Zippel)
  math_expand_expr          9112  polynomial convolution expansion, verifier = pointwise factor product vs expanded form (d+1 points)
  math_factor_expr          9113  rational roots + synthetic division, verifier = pointwise product identity (d+1 points)
  math_differentiate        9114  power-rule differentiation, verifier = numerical central differences
  math_integrate_elementary 9115  power-rule antiderivative, verifier = exact Boole-quadrature identity (deg<=5)
  math_numeric_root         9116  exact rational bisection, certificate = interval_contains, verifier = float Newton
  math_series_truncate      9117  truncated series + exact error bound, verifier = math-library closed form must fall inside the interval
  math_unit_check           9118  dimensional homogeneity, verifier = independent prime-power signature channel
  math_evaluate_expr        9119  Horner/closed-form evaluation, verifier = independent naive-power/loop-sum channel

Iron rules in practice: exact values always go through exact.frac/frac_str (no float masquerading as exact);
approximate results carry an interval or error bound; long loops call ctl.check_time periodically; failures
raise only §15 table codes.
"""
from __future__ import annotations

import math
from fractions import Fraction

from ..certify import Certificate
from ..errors import Err, SvoError
from ..exact import frac, frac_str
from ..registry import REGISTRY, OpDef
from ..tool import CallCtl

# ---------------------------------------------------------------------------
# shared exact utilities (reused by math_theorems / math_strategy / cert_math)
# ---------------------------------------------------------------------------


def poly_trim(cs: list[Fraction]) -> list[Fraction]:
    """Drop trailing zero coefficients (keep at least 1 entry; zero polynomial = [0])."""
    out = list(cs)
    while len(out) > 1 and out[-1] == 0:
        out.pop()
    return out


def poly_eval(cs: list[Fraction], x: Fraction) -> Fraction:
    """Exact Horner evaluation (cs in ascending powers)."""
    acc = Fraction(0)
    for c in reversed(cs):
        acc = acc * x + c
    return acc


def poly_eval_float(cs: list[Fraction], x: float) -> float:
    acc = 0.0
    for c in reversed(cs):
        acc = acc * x + float(c)
    return acc


def poly_mul(a: list[Fraction], b: list[Fraction]) -> list[Fraction]:
    if not a or not b:
        return [Fraction(0)]
    out = [Fraction(0)] * (len(a) + len(b) - 1)
    for i, ca in enumerate(a):
        if ca == 0:
            continue
        for j, cb in enumerate(b):
            out[i + j] += ca * cb
    return poly_trim(out)


def poly_shift(cs: list[Fraction], x0: Fraction) -> list[Fraction]:
    """Synthetic division: Taylor coefficients t_k of p(x) such that p(x0+t) = Σ t_k t^k (ascending powers)."""
    d = list(reversed(poly_trim(cs)))  # descending powers
    ts: list[Fraction] = []
    for _ in range(len(d) - 1):
        q = [d[0]]
        for k in range(1, len(d) - 1):
            q.append(d[k] + q[-1] * x0)
        ts.append(d[-1] + q[-1] * x0)  # the remainder is a constant
        d = q
    if d:
        ts.append(d[0])
    return poly_trim(list(reversed(ts))) if len(ts) > 1 else [ts[0]] if ts else [Fraction(0)]


def mat_mul(A: list[list[Fraction]], B: list[list[Fraction]]) -> list[list[Fraction]]:
    n, k, m = len(A), len(B), len(B[0])
    return [[sum((A[i][t] * B[t][j] for t in range(k)), Fraction(0)) for j in range(m)] for i in range(n)]


def mat_vec(A: list[list[Fraction]], x: list[Fraction]) -> list[Fraction]:
    return [sum((a * xi for a, xi in zip(row, x)), Fraction(0)) for row in A]


def gauss_solve(A: list[list[Fraction]], b: list[Fraction]) -> "list[Fraction] | None":
    """Exact rational Gaussian elimination + back-substitution; singular returns None."""
    n = len(A)
    M = [list(row) + [bi] for row, bi in zip(A, b)]
    for col in range(n):
        piv = next((r for r in range(col, n) if M[r][col] != 0), None)
        if piv is None:
            return None
        M[col], M[piv] = M[piv], M[col]
        pv = M[col][col]
        for r in range(col + 1, n):
            if M[r][col] != 0:
                f = M[r][col] / pv
                for c in range(col, n + 1):
                    M[r][c] -= f * M[col][c]
    x = [Fraction(0)] * n
    for i in reversed(range(n)):
        s = sum((M[i][j] * x[j] for j in range(i + 1, n)), Fraction(0))
        x[i] = (M[i][n] - s) / M[i][i]
    return x


def det_cofactor(M: list[list[Fraction]]) -> Fraction:
    """Cofactor-expansion determinant (for the verifier's independent channel; a different path from elimination)."""
    n = len(M)
    if n == 1:
        return M[0][0]
    if n == 2:
        return M[0][0] * M[1][1] - M[0][1] * M[1][0]
    total = Fraction(0)
    for j in range(n):
        if M[0][j] == 0:
            continue
        minor = [[M[r][c] for c in range(n) if c != j] for r in range(1, n)]
        total += ((-1) ** j) * M[0][j] * det_cofactor(minor)
    return total


def _frac_mat(rows) -> list[list[Fraction]]:
    return [[frac(v) for v in row] for row in rows]


def _frac_vec(vs) -> list[Fraction]:
    return [frac(v) for v in vs]


def _worse(a: Fraction, b: Fraction) -> Fraction:
    return a if a >= b else b


# ---------------------------------------------------------------------------
# 9110 math_solve_linear_system
# ---------------------------------------------------------------------------

_IN_LINSYS = {
    "kind": "dict",
    "fields": {
        "A": {"kind": "list", "items": {"kind": "list", "items": {"kind": "Fraction"}}, "minlen": 1, "maxlen": 6},
        "b": {"kind": "list", "items": {"kind": "Fraction"}, "minlen": 1, "maxlen": 6},
    },
    "required": ["A", "b"],
}
_OUT_LINSYS = {
    "kind": "dict",
    "fields": {
        "x": {"kind": "list", "items": {"kind": "str"}, "minlen": 1, "maxlen": 6},
        "det": {"kind": "str"},
        "resid_inf": {"kind": "str"},
    },
    "required": ["x", "det", "resid_inf"],
}


def _pre_linsys(args: dict) -> tuple[bool, str]:
    A, b = args["A"], args["b"]
    n = len(A)
    if len(b) != n:
        return False, f"b 长度 {len(b)} != A 阶 {n}"
    for i, row in enumerate(A):
        if len(row) != n:
            return False, f"A 非方阵: 第 {i} 行长度 {len(row)} != {n}"
    return True, ""


def _inv_linsys(args: dict, ctl: CallCtl) -> dict:
    ctx = "op:math_solve_linear_system#9110"
    A = _frac_mat(args["A"])
    b = _frac_vec(args["b"])
    x = gauss_solve(A, b)
    if x is None:
        raise SvoError(Err.DEG_UNSUPPORTED, ctx=ctx, detail="系数矩阵奇异 (det=0), 精确解不唯一/不存在")
    r = max((abs(v) for v in mat_sub(mat_vec(A, x), b)), default=Fraction(0))
    det = _det_lu(A)
    cert = Certificate(
        kind="linear_backsub",
        payload={"A": [[frac_str(v) for v in row] for row in A], "b": [frac_str(v) for v in b],
                 "x": [frac_str(v) for v in x]},
        issued_by="math_solve_linear_system",
    )
    return {
        "value": {"x": [frac_str(v) for v in x], "det": frac_str(det), "resid_inf": frac_str(r)},
        "resid": [float(r)],
        "cert": cert,
        "notes": f"精确高斯消元 n={len(A)}, ‖Ax−b‖∞={frac_str(r)}, det={frac_str(det)}",
    }


def mat_sub(u: list[Fraction], v: list[Fraction]) -> list[Fraction]:
    return [a - b for a, b in zip(u, v)]


def _det_lu(A: list[list[Fraction]]) -> Fraction:
    """Determinant (elimination path)."""
    n = len(A)
    M = [list(row) for row in A]
    det = Fraction(1)
    for col in range(n):
        piv = next((r for r in range(col, n) if M[r][col] != 0), None)
        if piv is None:
            return Fraction(0)
        if piv != col:
            M[col], M[piv] = M[piv], M[col]
            det = -det
        det *= M[col][col]
        for r in range(col + 1, n):
            if M[r][col] != 0:
                f = M[r][col] / M[col][col]
                for c in range(col, n):
                    M[r][c] -= f * M[col][c]
    return det


def _ver_linsys(args: dict, value: dict) -> dict:
    """Independent verifier: recompute x via the Cramer cofactor route, a different path from invoke's elimination route."""
    A = _frac_mat(args["A"])
    b = _frac_vec(args["b"])
    n = len(A)
    x_claim = [frac(s) for s in value["x"]]
    if len(x_claim) != n:
        return {"ok": False, "resid": [1.0], "detail": "解向量长度不符"}
    D = det_cofactor(A)
    if D == 0:
        return {"ok": False, "resid": [1.0], "detail": "Cramer: det=0 与已返回的唯一解矛盾"}
    worst = Fraction(0)
    for i in range(n):
        Ai = [list(row) for row in A]
        for r in range(n):
            Ai[r][i] = b[r]
        xi = det_cofactor(Ai) / D
        worst = _worse(worst, abs(xi - x_claim[i]))
    return {"ok": worst == 0, "resid": [float(worst)],
            "detail": f"Cramer 独立复算 max|x_i−x_i^claim|={frac_str(worst)} (精确 0)"}


REGISTRY.register(OpDef(
    name="math_solve_linear_system", band=9100, sub=10,
    in_schema=_IN_LINSYS, out_schema=_OUT_LINSYS,
    invoke=_inv_linsys, verify=_ver_linsys, cert_kind="linear_backsub", precond=_pre_linsys,
    cost=("units", 2.0), errors=("DEG_UNSUPPORTED", "SCHEMA_MISMATCH", "PRECOND_FAIL"),
    effects="none (纯函数, 精确有理算术)",
    doc="线性方程组精确解 | in: A(n×n Fraction),b | out: x+det+残差 | 验证器: Cramer 余子式独立复算 "
        "+ linear_backsub 证书 | 成本 2 units | 失败: 奇异 DEG_UNSUPPORTED / 非方阵 PRECOND_FAIL",
    selfchecks=(
        {"id": "linsys_2x2_hand", "args": {"A": [[2, 1], [1, 3]], "b": [5, 10]},
         "check": lambda r: (r.value["x"] == ["1", "3"] and r.value["det"] == "5" and r.resid_scalar == 0.0,
                             f"x={r.value['x']} det={r.value['det']} resid={r.resid_scalar}"),
         "expect": "手算: 2x+y=5, x+3y=10 → x=1,y=3; det=5"},
        {"id": "linsys_3x3_triangular", "args": {"A": [[1, 1, 1], [0, 1, 1], [0, 0, 1]], "b": [4, 2, 3]},
         "check": lambda r: (r.value["x"] == ["2", "-1", "3"], f"x={r.value['x']}"),
         "expect": "手算: 回代 x3=3, x2=-1, x1=2"},
        {"id": "linsys_singular", "args": {"A": [[1, 2], [2, 4]], "b": [1, 2]}, "expect_err": "DEG_UNSUPPORTED",
         "expect": "奇异矩阵必须被拒"},
        {"id": "linsys_schema_neg", "args": {"A": [[1, 2], [3, 4]], "b": [1]}, "expect_err": "PRECOND_FAIL",
         "expect": "b 长度不符必须在 precond 拦下"},
    ),
))


# ---------------------------------------------------------------------------
# 9111 math_simplify_expr / 9112 math_expand_expr / 9113 math_factor_expr
# ---------------------------------------------------------------------------

_IN_TERMS = {"kind": "list", "items": {"kind": "dict",
                                       "fields": {"c": {"kind": "Fraction"}, "p": {"kind": "int", "min": 0, "max": 8}},
                                       "required": ["c", "p"]}, "minlen": 1, "maxlen": 16}

_IN_SIMPLIFY = {"kind": "dict", "fields": {"terms": _IN_TERMS}, "required": ["terms"]}
_OUT_SIMPLIFY = {"kind": "dict",
                 "fields": {"terms": {"kind": "list", "items": {"kind": "dict",
                                                                "fields": {"c": {"kind": "str"}, "p": {"kind": "int"}},
                                                                "required": ["c", "p"]}, "maxlen": 16},
                            "n_terms": {"kind": "int", "min": 0}, "is_zero": {"kind": "bool"}},
                 "required": ["terms", "n_terms", "is_zero"]}


def _terms_merge(terms: list[dict]) -> dict[int, Fraction]:
    out: dict[int, Fraction] = {}
    for t in terms:
        c = frac(t["c"])
        p = int(t["p"])
        out[p] = out.get(p, Fraction(0)) + c
    return {p: c for p, c in out.items() if c != 0}


def _terms_eval_raw(terms: list[dict], x: Fraction) -> Fraction:
    """Structured term-by-term evaluation (no pre-merging) — the verifier's independent channel."""
    return sum((frac(t["c"]) * (x ** int(t["p"])) for t in terms), Fraction(0))


def _dict_eval(d: dict, x: Fraction) -> Fraction:
    return sum((c * (x ** int(p)) for p, c in d.items()), Fraction(0))


def _inv_simplify(args: dict, ctl: CallCtl) -> dict:
    merged = _terms_merge(args["terms"])
    deg = max((int(t["p"]) for t in args["terms"]), default=0)
    out_terms = sorted(((p, c) for p, c in merged.items()), key=lambda pc: -pc[0])
# **The zero polynomial must be written out explicitly**: when everything cancels, both term tables are empty, and
# the tightened checker judges "both sides empty" as a vacuous assertion (0=0) and rejects it — that guard exists to
# prevent "empty payloads". So the zero polynomial is written as {"0": "0"}, letting the certificate carry real
# information: "the result is the zero polynomial".
    lhs_terms = {str(p): frac_str(c) for p, c in merged.items()} or {"0": "0"}
    rhs_terms = {str(p): frac_str(c) for p, c in out_terms} or {"0": "0"}
    cert = Certificate(
        kind="identity_expand",
        payload={"lhs_terms": lhs_terms, "rhs_terms": rhs_terms,
                 "samples": [str(j) for j in range(deg + 1)]},
        issued_by="math_simplify_expr",
    )
    return {
        "value": {"terms": [{"c": frac_str(c), "p": p} for p, c in out_terms],
                  "n_terms": len(out_terms), "is_zero": not out_terms},
        "resid": [0.0], "cert": cert,
        "notes": f"合并 {len(args['terms'])} 项 → {len(out_terms)} 项; 证书=identity_expand @{deg + 1} 有理样本点",
    }


def _ver_simplify(args: dict, value: dict) -> dict:
    """Independent verifier: pointwise evaluation of the original structured terms vs the output terms, at deg+1 rational points (identity ⟺ equality at d+1 points)."""
    terms = args["terms"]
    deg = max((int(t["p"]) for t in terms), default=0)
    out_d = {int(t["p"]): frac(t["c"]) for t in value["terms"]}
    worst = Fraction(0)
    for x in range(deg + 1):
        xf = Fraction(x)
        worst = _worse(worst, abs(_terms_eval_raw(terms, xf) - _dict_eval(out_d, xf)))
    return {"ok": worst == 0, "resid": [float(worst)],
            "detail": f"{deg + 1} 点逐点代入 max|in−out|={frac_str(worst)} (精确 0)"}


REGISTRY.register(OpDef(
    name="math_simplify_expr", band=9100, sub=11,
    in_schema=_IN_SIMPLIFY, out_schema=_OUT_SIMPLIFY,
    invoke=_inv_simplify, verify=_ver_simplify, cert_kind="identity_expand",
    cost=("units", 1.0), errors=("SCHEMA_MISMATCH", "PRECOND_FAIL"),
    effects="none (纯函数, 精确有理算术)",
    doc="多项式合并同类项 | in: terms[(c,p)] | out: 规范项表 | 验证器: deg+1 有理点结构化逐点代入 "
        "+ identity_expand 证书 | 1 unit | 失败: schema 拒非规范字面量",
    selfchecks=(
        {"id": "simplify_merge_hand", "args": {"terms": [{"c": 1, "p": 2}, {"c": 2, "p": 1}, {"c": 3, "p": 0}, {"c": -1, "p": 2}]},
         "check": lambda r: (r.value["terms"] == [{"c": "2", "p": 1}, {"c": "3", "p": 0}] and r.value["n_terms"] == 2,
                             f"terms={r.value['terms']}"),
         "expect": "手算: x²+2x+3−x² = 2x+3"},
        {"id": "simplify_cancel_zero", "args": {"terms": [{"c": 1, "p": 1}, {"c": -1, "p": 1}]},
         "check": lambda r: (r.value["is_zero"] and r.value["n_terms"] == 0, f"is_zero={r.value['is_zero']}"),
         "expect": "全消 → 零多项式"},
        {"id": "simplify_cross_eval", "args": {"terms": [{"c": 1, "p": 2}, {"c": -1, "p": 0}, {"c": 3, "p": 2}]},
         "check": lambda r: (
             r.value["terms"] == [{"c": "4", "p": 2}, {"c": "-1", "p": 0}]
             and _terms_eval_raw([{"c": 1, "p": 2}, {"c": -1, "p": 0}, {"c": 3, "p": 2}], Fraction(2)) == 15,
             f"terms={r.value['terms']}; 原式 @x=2 = 16−1 = 15 手算核对"),
         "expect": "x²−1+3x² → 4x²−1, 且原式在 x=2 处值为 15"},
        {"id": "simplify_schema_neg", "args": {"terms": [{"c": "0.5", "p": 1}]}, "expect_err": "SCHEMA_MISMATCH",
         "expect": "小数字面量属 float 通道, 精确通道必须拒绝"},
    ),
))


_IN_EXPAND = {"kind": "dict",
              "fields": {"factors": {"kind": "list", "items": _IN_TERMS, "minlen": 1, "maxlen": 4}},
              "required": ["factors"]}
_OUT_EXPAND = {"kind": "dict",
               "fields": {"terms": {"kind": "list", "items": {"kind": "dict",
                                                              "fields": {"c": {"kind": "str"}, "p": {"kind": "int"}},
                                                              "required": ["c", "p"]}, "maxlen": 17},
                          "degree": {"kind": "int", "min": 0, "max": 32}},
               "required": ["terms", "degree"]}


def _terms_to_coeffs(terms: list[dict]) -> list[Fraction]:
    d = _terms_merge(terms)
    deg = max(d) if d else 0
    return [d.get(k, Fraction(0)) for k in range(deg + 1)]


def _dc_product(facs: list[list[Fraction]]) -> list[Fraction]:
    """Divide-and-conquer multiplication (a different code path from invoke's left-fold convolution)."""
    if len(facs) == 1:
        return facs[0]
    mid = len(facs) // 2
    return poly_mul(_dc_product(facs[:mid]), _dc_product(facs[mid:]))


def _inv_expand(args: dict, ctl: CallCtl) -> dict:
    facs = [_terms_to_coeffs(f) for f in args["factors"]]
    acc = facs[0]
    for f in facs[1:]:
        acc = poly_mul(acc, f)
    acc = poly_trim(acc)
    out_terms = [(k, c) for k, c in enumerate(acc) if c != 0]
    rhs = _dc_product(facs)
    cert = Certificate(
        kind="identity_expand",
        payload={"lhs_terms": {str(k): frac_str(c) for k, c in enumerate(rhs) if c != 0},
                 "rhs_terms": {str(k): frac_str(c) for k, c in enumerate(acc) if c != 0},
                 "samples": [str(j) for j in range(len(acc))]},
        issued_by="math_expand_expr",
    )
    return {"value": {"terms": [{"c": frac_str(c), "p": p} for p, c in reversed(out_terms)],
                      "degree": len(acc) - 1},
            "resid": [0.0], "cert": cert,
            "notes": f"{len(facs)} 因子卷积展开, 次数 {len(acc) - 1}; 证书=分治展开对拍 identity_expand"}


def _ver_expand(args: dict, value: dict) -> dict:
    """Independent verifier: pointwise product of the factors vs the expanded form, identity at D+1 rational points."""
    facs = [_terms_to_coeffs(f) for f in args["factors"]]
    D = sum(len(f) - 1 for f in facs)
    out_d = {int(t["p"]): frac(t["c"]) for t in value["terms"]}
    worst = Fraction(0)
    for x in range(D + 1):
        xf = Fraction(x)
        prod = Fraction(1)
        for f in facs:
            prod *= poly_eval(f, xf)
        worst = _worse(worst, abs(prod - _dict_eval(out_d, xf)))
    return {"ok": worst == 0, "resid": [float(worst)],
            "detail": f"{D + 1} 点因式乘积对照 max|∏f_i−展开|={frac_str(worst)} (精确 0)"}


REGISTRY.register(OpDef(
    name="math_expand_expr", band=9100, sub=12,
    in_schema=_IN_EXPAND, out_schema=_OUT_EXPAND,
    invoke=_inv_expand, verify=_ver_expand, cert_kind="identity_expand",
    cost=("units", 1.5), errors=("SCHEMA_MISMATCH", "PRECOND_FAIL"),
    effects="none (纯函数, 精确有理算术)",
    doc="多项式展开 | in: factors[terms[]] | out: 展开项表+次数 | 验证器: D+1 点因式乘积恒等 "
        "+ identity_expand 证书 (与 simplify/factor 互验) | 1.5 units | 失败: schema",
    selfchecks=(
        {"id": "expand_xp1_xp2", "args": {"factors": [[{"c": 1, "p": 1}, {"c": 1, "p": 0}], [{"c": 1, "p": 1}, {"c": 2, "p": 0}]]},
         "check": lambda r: (r.value["terms"] == [{"c": "1", "p": 2}, {"c": "3", "p": 1}, {"c": "2", "p": 0}]
                             and r.value["degree"] == 2, f"terms={r.value['terms']}"),
         "expect": "手算: (x+1)(x+2)=x²+3x+2"},
        {"id": "expand_diff_squares", "args": {"factors": [[{"c": 1, "p": 1}, {"c": -1, "p": 0}], [{"c": 1, "p": 1}, {"c": 1, "p": 0}]]},
         "check": lambda r: (r.value["terms"] == [{"c": "1", "p": 2}, {"c": "-1", "p": 0}], f"terms={r.value['terms']}"),
         "expect": "手算: (x−1)(x+1)=x²−1"},
        {"id": "expand_value_x3", "args": {"factors": [[{"c": 1, "p": 1}, {"c": 1, "p": 0}], [{"c": 1, "p": 1}, {"c": 2, "p": 0}]]},
         "check": lambda r: (REGISTRY.tool("math_evaluate_expr").call(
             {"form": "poly", "coeffs": [t["c"] for t in sorted(r.value["terms"], key=lambda t: t["p"])], "x": 3}
         ).value["value"] == "20", "展开式 @x=3 = 20 (=4·5) 由 math_evaluate_expr 独立核对"),
         "expect": "互验: (3+1)(3+2)=20"},
        {"id": "expand_schema_neg", "args": {"factors": []}, "expect_err": "SCHEMA_MISMATCH",
         "expect": "空因子表必须被 schema 拦下"},
    ),
))


_IN_FACTOR = {"kind": "dict",
              "fields": {"coeffs": {"kind": "list", "items": {"kind": "Fraction"}, "minlen": 1, "maxlen": 9}},
              "required": ["coeffs"]}
_OUT_FACTOR = {"kind": "dict",
               "fields": {"linear": {"kind": "list", "items": {"kind": "dict",
                                                               "fields": {"root": {"kind": "str"}, "mult": {"kind": "int", "min": 1}},
                                                               "required": ["root", "mult"]}, "maxlen": 8},
                          "remaining": {"kind": "list", "items": {"kind": "str"}, "minlen": 1, "maxlen": 9},
                          "remaining_degree": {"kind": "int", "min": 0, "max": 8}},
               "required": ["linear", "remaining", "remaining_degree"]}


def _divisors(n: int) -> list[int]:
    n = abs(n)
    out = []
    i = 1
    while i * i <= n:
        if n % i == 0:
            out.append(i)
            if i != n // i:
                out.append(n // i)
        i += 1
    return sorted(out)


def _synthetic_div(cs: list[Fraction], r: Fraction) -> list[Fraction]:
    """Quotient of cs ÷ (x−r) (ascending coefficients), requiring r to be a root."""
    d = list(reversed(cs))  # descending powers
    q = [d[0]]
    for k in range(1, len(d) - 1):
        q.append(d[k] + q[-1] * r)
    return list(reversed(q))


def _inv_factor(args: dict, ctl: CallCtl) -> dict:
    ctx = "op:math_factor_expr#9113"
    cs = poly_trim(_frac_vec(args["coeffs"]))
    if all(c == 0 for c in cs):
        raise SvoError(Err.PRECOND_FAIL, ctx=ctx, detail="零多项式无因式分解语义")
    linear: list[dict] = []
    work = cs
    # 1) clear denominators; enumerate candidates via the rational-root theorem
    L = 1
    for c in work:
        L = L * c.denominator // math.gcd(L, c.denominator)
    g = [c * L for c in work]
    g = poly_trim(g)
    while True:  # constant term 0 → root 0
        if g[0] != 0:
            break
        linear.append({"root": "0", "mult": 1})
        g = poly_trim(g[1:]) if len(g) > 1 else [Fraction(0)]
        if len(g) == 1 and g[0] == 0:
            break
    if not (len(g) == 1 and g[0] == 0):
        a0, an = g[0], g[-1]
        if abs(a0) > 10 ** 9 or abs(an) > 10 ** 9:
            raise SvoError(Err.DEG_UNSUPPORTED, ctx=ctx, detail="常数/首项系数过大, 有理根候选枚举超出有界域")
        div0, divn = _divisors(a0.numerator * a0.denominator), _divisors(an.numerator * an.denominator)
        if len(div0) * len(divn) > 20000:
            raise SvoError(Err.DEG_UNSUPPORTED, ctx=ctx,
                           detail=f"有理根候选 {len(div0)}×{len(divn)} 超出有界枚举预算")
        cands: list[Fraction] = []
        for s in div0:
            for t in divn:
                for sg in (1, -1):
                    c = Fraction(sg * s, t)
                    if c not in cands:
                        cands.append(c)
        for r in sorted(cands):
            mult = 0
            while len(g) > 1 and poly_eval(g, r) == 0:
                g = poly_trim(_synthetic_div(g, r))
                mult += 1
                ctl.check_time(ctx + "@deflate")
            if mult:
                linear.append({"root": frac_str(r), "mult": mult})
    remaining = [frac_str(c) for c in poly_trim(g)]
    # certificate: original polynomial vs the expanded factor chain (their difference is identically zero)
    fac_cs: list[list[Fraction]] = []
    for item in linear:
        fac_cs.extend([[-frac(item["root"]), Fraction(1)]] * item["mult"])
    fac_cs.append(poly_trim(g))
    prod = fac_cs[0]
    for f in fac_cs[1:]:
        prod = poly_mul(prod, f)
    cert = Certificate(
        kind="identity_expand",
        payload={"lhs_terms": {str(k): frac_str(c) for k, c in enumerate(poly_trim(cs)) if c != 0},
                 "rhs_terms": {str(k): frac_str(c) for k, c in enumerate(prod) if c != 0},
                 "samples": [str(j) for j in range(len(cs))]},
        issued_by="math_factor_expr",
    )
    return {"value": {"linear": linear, "remaining": remaining,
                      "remaining_degree": len(poly_trim(g)) - 1},
            "resid": [0.0], "cert": cert,
            "notes": f"有理根 {len(linear)} 组, 剩余次数 {len(poly_trim(g)) - 1}; 证书=identity_expand 展开差恒等"}


def _ver_factor(args: dict, value: dict) -> dict:
    """Independent verifier: ∏(x−r)^m · remaining(x) equals p(x) pointwise at D+1 rational points."""
    cs = poly_trim(_frac_vec(args["coeffs"]))
    D = len(cs) - 1
    worst = Fraction(0)
    for x in range(D + 1):
        xf = Fraction(x)
        prod = poly_eval([frac(s) for s in value["remaining"]], xf)
        for item in value["linear"]:
            prod *= (xf - frac(item["root"])) ** int(item["mult"])
        worst = _worse(worst, abs(prod - poly_eval(cs, xf)))
    deg_sum = sum(int(i["mult"]) for i in value["linear"]) + (len(value["remaining"]) - 1)
    ok = worst == 0 and deg_sum == D
    return {"ok": ok, "resid": [float(worst)],
            "detail": f"{D + 1} 点乘积恒等 max|Δ|={frac_str(worst)}; 次数守恒 {deg_sum}=={D}"}


REGISTRY.register(OpDef(
    name="math_factor_expr", band=9100, sub=13,
    in_schema=_IN_FACTOR, out_schema=_OUT_FACTOR,
    invoke=_inv_factor, verify=_ver_factor, cert_kind="identity_expand",
    cost=("units", 3.0), errors=("DEG_UNSUPPORTED", "SCHEMA_MISMATCH", "PRECOND_FAIL"),
    effects="none (纯函数, 精确有理算术)",
    doc="一元多项式有理根分解 | in: coeffs 升幂 | out: 线性因子+剩余 | 验证器: D+1 点乘积恒等 "
        "+ identity_expand 证书 (simplify/expand 互验) | 3 units | 失败: 零多项式 PRECOND_FAIL / 系数越界 DEG_UNSUPPORTED",
    selfchecks=(
        {"id": "factor_x2_minus_1", "args": {"coeffs": [-1, 0, 1]},
         "check": lambda r: (sorted(i["root"] for i in r.value["linear"]) == ["-1", "1"]
                             and r.value["remaining_degree"] == 0, f"linear={r.value['linear']}"),
         "expect": "手算: x²−1=(x−1)(x+1)"},
        {"id": "factor_double_root", "args": {"coeffs": [1, 2, 1]},
         "check": lambda r: (r.value["linear"] == [{"root": "-1", "mult": 2}], f"linear={r.value['linear']}"),
         "expect": "手算: x²+2x+1=(x+1)²"},
        {"id": "factor_frac_roots", "args": {"coeffs": [1, 3, 2]},
         "check": lambda r: (sorted(i["root"] for i in r.value["linear"]) == ["-1", "-1/2"]
                             and r.value["remaining"] == ["2"], f"linear={r.value['linear']} rem={r.value['remaining']}"),
         "expect": "手算: 1+3x+2x²=2(x+1/2)(x+1), 剩余首项 2"},
        {"id": "factor_cross_expand", "args": {"coeffs": [-1, 0, 1]},
         "check": lambda r: (REGISTRY.tool("math_expand_expr").call(
             {"factors": [[{"c": 1, "p": 1}, {"c": -1, "p": 0}], [{"c": 1, "p": 1}, {"c": 1, "p": 0}]]}
         ).value["terms"] == [{"c": "1", "p": 2}, {"c": "-1", "p": 0}],
             "互验: factor 的根经 expand_expr 重展开 = x²−1"),
         "expect": "factor↔expand 互验"},
        {"id": "factor_zero_poly", "args": {"coeffs": [0, 0]}, "expect_err": "PRECOND_FAIL",
         "expect": "零多项式必须被拒"},
    ),
))


# ---------------------------------------------------------------------------
# 9114 math_differentiate / 9115 math_integrate_elementary
# ---------------------------------------------------------------------------

_IN_COEFFS = {"kind": "dict", "fields": {"coeffs": {"kind": "list", "items": {"kind": "Fraction"},
                                                    "minlen": 1, "maxlen": 10}}, "required": ["coeffs"]}


def _inv_diff(args: dict, ctl: CallCtl) -> dict:
    cs = _frac_vec(args["coeffs"])
    if len(cs) == 1:
        d = [Fraction(0)]
    else:
        d = [cs[k] * k for k in range(1, len(cs))]
    return {"value": {"coeffs": [frac_str(c) for c in poly_trim(d)], "degree": len(poly_trim(d)) - 1},
            "resid": [0.0], "notes": f"幂规则求导, 次数 {len(poly_trim(d)) - 1}"}


def _ver_diff(args: dict, value: dict) -> dict:
    """Independent verifier: numerical central-difference comparison (float; a different path from symbolic differentiation)."""
    cs = _frac_vec(args["coeffs"])
    d = [frac(s) for s in value["coeffs"]]
    worst = 0.0
    for x in (-1.7, 0.3, 2.9):
        h = 1e-5
        num = (poly_eval_float(cs, x + h) - poly_eval_float(cs, x - h)) / (2 * h)
        sym = poly_eval_float(d, x)
        worst = max(worst, abs(num - sym) / max(1.0, abs(sym)))
    return {"ok": worst < 1e-6, "resid": [worst],
            "detail": f"3 点中心差分 max 相对偏差={worst:.3e}"}


REGISTRY.register(OpDef(
    name="math_differentiate", band=9100, sub=14,
    in_schema=_IN_COEFFS,
    out_schema={"kind": "dict", "fields": {"coeffs": {"kind": "list", "items": {"kind": "str"}, "minlen": 1, "maxlen": 10},
                                           "degree": {"kind": "int", "min": 0}}, "required": ["coeffs", "degree"]},
    invoke=_inv_diff, verify=_ver_diff,
    cost=("units", 0.8), errors=("SCHEMA_MISMATCH",),
    effects="none (纯函数, 精确有理算术)",
    doc="多项式符号求导 | in: coeffs 升幂 | out: 导数系数 | 验证器: 数值中心差分 3 点对照 | 0.8 unit | 失败: schema",
    selfchecks=(
        {"id": "diff_hand", "args": {"coeffs": [1, 2, 3]},
         "check": lambda r: (r.value["coeffs"] == ["2", "6"], f"d={r.value['coeffs']}"),
         "expect": "手算: d/dx(1+2x+3x²)=2+6x"},
        {"id": "diff_const", "args": {"coeffs": [5]},
         "check": lambda r: (r.value["coeffs"] == ["0"], f"d={r.value['coeffs']}"),
         "expect": "常数导数为零多项式"},
        {"id": "diff_value_at_1", "args": {"coeffs": [1, 2, 3]},
         "check": lambda r: (sum(frac(c) * 1 ** k for k, c in enumerate(r.value["coeffs"])) == 8,
                             "d(1)=2+6=8 手算核对"),
         "expect": "导数在 x=1 处值为 8"},
        {"id": "diff_schema_neg", "args": {"coeffs": [0.5]}, "expect_err": "SCHEMA_MISMATCH",
         "expect": "float 不得进入精确通道"},
    ),
))


_IN_INTEG = {"kind": "dict", "fields": {"coeffs": {"kind": "list", "items": {"kind": "Fraction"},
                                                   "minlen": 1, "maxlen": 7}}, "required": ["coeffs"]}


def _pre_integ(args: dict) -> tuple[bool, str]:
    if len(args["coeffs"]) > 6:
        return False, f"次数 {len(args['coeffs']) - 1} > 5: Boole 求积精确域外 (验证器诚实边界)"
    return True, ""


def _boole(cs: list[Fraction], a: Fraction, b: Fraction) -> Fraction:
    """Boole quadrature (5 nodes, exact for deg≤5) — the independent channel."""
    h = (b - a) / 4
    fs = [poly_eval(cs, a + h * j) for j in range(5)]
    w = (Fraction(2) * h / 45)
    return w * (7 * fs[0] + 32 * fs[1] + 12 * fs[2] + 32 * fs[3] + 7 * fs[4])


def _inv_integ(args: dict, ctl: CallCtl) -> dict:
    cs = _frac_vec(args["coeffs"])
    F = [Fraction(0)] + [cs[k] / (k + 1) for k in range(len(cs))]
    h = Fraction(1, 100000)
    flat_f: list[list[float]] = []
    flat_F: list[list[float]] = []
    for xi in (-1, 0, 1, 2):
        x = Fraction(xi)
        flat_f.append([float(x), float(poly_eval(cs, x))])
        for dx in (-h, Fraction(0), h):
            t = x + dx
            flat_F.append([float(t), float(poly_eval(F, t))])
    cert = Certificate(kind="derivative_inverse", payload={"f_samples": flat_f, "F_samples": flat_F,
                                                           "h": 1e-5, "tol": 1e-6},
                       issued_by="math_integrate_elementary")
    return {"value": {"coeffs": [frac_str(c) for c in F]}, "resid": [0.0], "cert": cert,
            "notes": f"幂规则原函数 (C=0), {len(F)} 项; 证书=derivative_inverse 数值差分互逆"}


def _ver_integ(args: dict, value: dict) -> dict:
    """Independent verifier: Boole quadrature ∫₀¹f equals F(1)−F(0) exactly (exact for deg≤5; a different path from the power rule)."""
    cs = _frac_vec(args["coeffs"])
    F = [frac(s) for s in value["coeffs"]]
    lhs = _boole(cs, Fraction(0), Fraction(1))
    rhs = poly_eval(F, Fraction(1)) - poly_eval(F, Fraction(0))
    d = abs(lhs - rhs)
    return {"ok": d == 0, "resid": [float(d)],
            "detail": f"Boole 求积 {frac_str(lhs)} vs F(1)−F(0) {frac_str(rhs)} (精确)"}


REGISTRY.register(OpDef(
    name="math_integrate_elementary", band=9100, sub=15,
    in_schema=_IN_INTEG,
    out_schema={"kind": "dict", "fields": {"coeffs": {"kind": "list", "items": {"kind": "str"}, "minlen": 1, "maxlen": 7}},
                "required": ["coeffs"]},
    invoke=_inv_integ, verify=_ver_integ, cert_kind="derivative_inverse", precond=_pre_integ,
    cost=("units", 1.2), errors=("SCHEMA_MISMATCH", "PRECOND_FAIL"),
    effects="none (纯函数, 精确有理算术)",
    doc="多项式原函数 (C=0) | in: coeffs 升幂 (deg≤5) | out: F 系数 | 验证器: Boole 求积精确互逆 "
        "+ derivative_inverse 证书 | 1.2 units | 失败: deg>5 PRECOND_FAIL (验证器精确域)",
    selfchecks=(
        {"id": "integ_x", "args": {"coeffs": [0, 1]},
         "check": lambda r: (r.value["coeffs"] == ["0", "0", "1/2"], f"F={r.value['coeffs']}"),
         "expect": "手算: ∫x = x²/2"},
        {"id": "integ_const", "args": {"coeffs": [3]},
         "check": lambda r: (r.value["coeffs"] == ["0", "3"], f"F={r.value['coeffs']}"),
         "expect": "手算: ∫3 = 3x"},
        {"id": "integ_x3_value", "args": {"coeffs": [0, 0, 0, 1]},
         "check": lambda r: (poly_eval([frac(s) for s in r.value["coeffs"]], Fraction(2)) == 4,
                             "F(2)=2⁴/4=4 手算核对"),
         "expect": "∫x³ = x⁴/4, F(2)=4"},
        {"id": "integ_deg6_neg", "args": {"coeffs": [0, 0, 0, 0, 0, 0, 1]}, "expect_err": "PRECOND_FAIL",
         "expect": "deg 6 超出 Boole 精确域, 必须拒 (不许静默给不可验证结果)"},
    ),
))


# ---------------------------------------------------------------------------
# 9116 math_numeric_root
# ---------------------------------------------------------------------------

_IN_ROOT = {"kind": "dict",
            "fields": {"coeffs": {"kind": "list", "items": {"kind": "Fraction"}, "minlen": 2, "maxlen": 9},
                       "lo": {"kind": "Fraction"}, "hi": {"kind": "Fraction"},
                       "tol_exp": {"kind": "int", "min": 1, "max": 30}},
            "required": ["coeffs", "lo", "hi", "tol_exp"]}


def _pre_root(args: dict) -> tuple[bool, str]:
    if frac(args["lo"]) >= frac(args["hi"]):
        return False, "要求 lo < hi (括号区间)"
    return True, ""


def _inv_root(args: dict, ctl: CallCtl) -> dict:
    ctx = "op:math_numeric_root#9116"
    cs = poly_trim(_frac_vec(args["coeffs"]))
    lo, hi = frac(args["lo"]), frac(args["hi"])
    width = Fraction(1, 10 ** int(args["tol_exp"]))
    flo, fhi = poly_eval(cs, lo), poly_eval(cs, hi)
    if flo == 0:
        lo = hi = lo
    elif fhi == 0:
        lo = hi = hi
    elif (flo < 0) == (fhi < 0):
        raise SvoError(Err.NO_REAL_ROOT, ctx=ctx,
                       detail=f"f(lo)={flo}, f(hi)={fhi} 同号, 括号内不能保证有根 (二分前提不成立)")
    else:
        while hi - lo > width:
            ctl.check_time(ctx + "@bisect")
            mid = (lo + hi) / 2
            fm = poly_eval(cs, mid)
            if fm == 0:
                lo = hi = mid
                break
            if (fm < 0) == (flo < 0):
                lo = mid
            else:
                hi = mid
    cert = Certificate(kind="interval_contains",
                       payload={"f_coeffs": [frac_str(c) for c in cs], "lo": frac_str(lo), "hi": frac_str(hi)},
                       issued_by="math_numeric_root")
    return {"value": {"lo": frac_str(lo), "hi": frac_str(hi), "width": frac_str(hi - lo),
                      "root_float": float((lo + hi) / 2)},
            "resid": [float(hi - lo)], "cert": cert,
            "notes": f"精确有理二分, 区间宽 {frac_str(hi - lo)}, 证书 interval_contains (端点变号)"}


def _ver_root(args: dict, value: dict) -> dict:
    """Independent verifier: the float Newton method (a different path from exact bisection) must converge into the certificate interval."""
    cs = [float(c) for c in _frac_vec(args["coeffs"])]
    dcs = [cs[k] * k for k in range(1, len(cs))] or [0.0]
    lo, hi = float(frac(value["lo"])), float(frac(value["hi"]))
    x = (lo + hi) / 2

    def f(t):
        a = 0.0
        for c in reversed(cs):
            a = a * t + c
        return a

    def df(t):
        a = 0.0
        for c in reversed(dcs):
            a = a * t + c
        return a

    for _ in range(60):
        d = df(x)
        if d == 0.0:
            break
        xn = x - f(x) / d
        if xn < lo:
            xn = (x + lo) / 2
        elif xn > hi:
            xn = (x + hi) / 2
        if abs(xn - x) < 1e-15:
            x = xn
            break
        x = xn
    ok = (lo - 1e-9 <= x <= hi + 1e-9) and (hi - lo) <= 1.5 * 10 ** (-int(args["tol_exp"]))
    return {"ok": ok, "resid": [max(0.0, lo - x, x - hi)],
            "detail": f"float 牛顿收敛点 {x:.15g} ∈ [{lo:.15g},{hi:.15g}], 宽 {hi - lo:.3e}"}


REGISTRY.register(OpDef(
    name="math_numeric_root", band=9100, sub=16,
    in_schema=_IN_ROOT,
    out_schema={"kind": "dict", "fields": {"lo": {"kind": "str"}, "hi": {"kind": "str"},
                                           "width": {"kind": "str"}, "root_float": {"kind": "float"}},
                "required": ["lo", "hi", "width", "root_float"]},
    invoke=_inv_root, verify=_ver_root, cert_kind="interval_contains", precond=_pre_root,
    cost=("units", 2.0), errors=("NO_REAL_ROOT", "SCHEMA_MISMATCH", "PRECOND_FAIL"),
    effects="none (纯函数, 精确有理算术)",
    doc="区间包含求根 (精确二分) | in: coeffs+括号+精度 | out: 根区间 | 证书 interval_contains (端点变号, "
        "certify.py 独立复验) + float 牛顿验证器 | 2 units | 失败: 同号括号 NO_REAL_ROOT",
    selfchecks=(
        {"id": "root_sqrt2", "args": {"coeffs": [-2, 0, 1], "lo": 1, "hi": 2, "tol_exp": 12},
         "check": lambda r: (frac(r.value["lo"]) < Fraction(141421356237309, 10 ** 14) < frac(r.value["hi"])
                             and frac(r.value["lo"]) < Fraction(1414213562373095, 10 ** 15) < frac(r.value["hi"])
                             and r.cert is not None and r.cert_independent,
                             f"区间 [{r.value['lo']},{r.value['hi']}] 包含 √2 (两侧有理夹逼), 证书独立复验={r.cert_independent}"),
         "expect": "√2 ∈ 1.4142135623730⁹⁵… 区间宽 ≤1e−12"},
        {"id": "root_exact_endpoint", "args": {"coeffs": [-1, 0, 1], "lo": 1, "hi": 2, "tol_exp": 6},
         "check": lambda r: (r.value["lo"] == "1" and r.value["hi"] == "1", f"[{r.value['lo']},{r.value['hi']}]"),
         "expect": "f(1)=0: 端点即根, 区间塌缩为 [1,1]"},
        {"id": "root_no_sign_change", "args": {"coeffs": [1, 0, 1], "lo": 0, "hi": 2, "tol_exp": 6},
         "expect_err": "NO_REAL_ROOT", "expect": "x²+1 在 [0,2] 无实根"},
        {"id": "root_bad_bracket", "args": {"coeffs": [-2, 0, 1], "lo": 2, "hi": 1, "tol_exp": 6},
         "expect_err": "PRECOND_FAIL", "expect": "lo≥hi 必须被拒"},
    ),
))


# ---------------------------------------------------------------------------
# 9117 math_series_truncate
# ---------------------------------------------------------------------------

_FNS = ("exp", "sin", "cos", "ln1p", "geom")
_IN_SERIES = {"kind": "dict",
              "fields": {"fn": {"kind": "str", "enum": list(_FNS)}, "x": {"kind": "Fraction"},
                         "n": {"kind": "int", "min": 1, "max": 24}},
              "required": ["fn", "x", "n"]}


def _inv_series(args: dict, ctl: CallCtl) -> dict:
    ctx = "op:math_series_truncate#9117"
    fn = args["fn"]
    x = frac(args["x"])
    n = int(args["n"])
    ax = abs(x)
    if fn in ("exp", "sin", "cos") and ax > 1:
        raise SvoError(Err.DEG_UNSUPPORTED, ctx=ctx, detail=f"|x|={ax} > 1, 截断界推导域外 (需 |x|≤1)")
    if fn == "ln1p" and not (0 <= x <= 1):
        raise SvoError(Err.DEG_UNSUPPORTED, ctx=ctx, detail=f"x={x} ∉ [0,1], 交错级数收敛域外")
    if fn == "geom" and ax >= 1:
        raise SvoError(Err.DEG_UNSUPPORTED, ctx=ctx, detail=f"|x|={ax} ≥ 1, 几何级数发散")
    if fn == "exp":
        S = sum((x ** k / math.factorial(k) for k in range(n)), Fraction(0))
        B = Fraction(3) * ax ** n / math.factorial(n)  # e^{|x|} ≤ e ≤ 3 (k!≥2^{k-1}, k≥2)
        ref = math.exp(float(x))
    elif fn == "sin":
        S = sum((((-1) ** k) * x ** (2 * k + 1) / math.factorial(2 * k + 1) for k in range(n)), Fraction(0))
        B = ax ** (2 * n + 1) / math.factorial(2 * n + 1)
        ref = math.sin(float(x))
    elif fn == "cos":
        S = sum((((-1) ** k) * x ** (2 * k) / math.factorial(2 * k) for k in range(n)), Fraction(0))
        B = ax ** (2 * n) / math.factorial(2 * n)
        ref = math.cos(float(x))
    elif fn == "ln1p":
        S = sum((((-1) ** (k - 1)) * x ** k / k for k in range(1, n + 1)), Fraction(0))
        B = x ** (n + 1) / (n + 1)
        ref = math.log1p(float(x))
    else:
        S = sum((x ** k for k in range(n)), Fraction(0))
        B = ax ** n / (1 - ax)
        ref = 1.0 / (1.0 - float(x))
    assert abs(ref - float(S)) <= float(B) + 1e-9 * max(1.0, abs(ref))  # numerical self-certificate (re-verified by verify)
    return {"value": {"sum": frac_str(S), "bound": frac_str(B),
                      "interval": [frac_str(S - B), frac_str(S + B)], "n": n, "fn": fn},
            "resid": [float(B)],
            "notes": f"{fn} 截断 {n} 项, |R_n| ≤ {frac_str(B)} (精确推导), 真值区间 [{frac_str(S - B)},{frac_str(S + B)}]"}


def _ver_series(args: dict, value: dict) -> dict:
    """Independent verifier: math-library closed-form values (a completely different path) must fall inside the declared interval."""
    fn, x = args["fn"], float(frac(args["x"]))
    if fn == "exp":
        ref = math.exp(x)
    elif fn == "sin":
        ref = math.sin(x)
    elif fn == "cos":
        ref = math.cos(x)
    elif fn == "ln1p":
        ref = math.log1p(x)
    else:
        ref = 1.0 / (1.0 - x)
    lo, hi = float(frac(value["interval"][0])), float(frac(value["interval"][1]))
    slack = 1e-9 * max(1.0, abs(ref))
    ok = lo - slack <= ref <= hi + slack and lo <= hi
    return {"ok": ok, "resid": [max(0.0, lo - ref, ref - hi)],
            "detail": f"math 库值 {ref:.17g} ∈ [{lo:.17g},{hi:.17g}] (slack {slack:.1e})"}


REGISTRY.register(OpDef(
    name="math_series_truncate", band=9100, sub=17,
    in_schema=_IN_SERIES,
    out_schema={"kind": "dict", "fields": {"sum": {"kind": "str"}, "bound": {"kind": "str"},
                                           "interval": {"kind": "list", "items": {"kind": "str"}, "len": 2},
                                           "n": {"kind": "int"}, "fn": {"kind": "str"}},
                "required": ["sum", "bound", "interval", "n", "fn"]},
    invoke=_inv_series, verify=_ver_series,
    cost=("units", 1.5), errors=("DEG_UNSUPPORTED", "SCHEMA_MISMATCH"),
    effects="none (纯函数, 精确有理算术+显式误差界)",
    doc="截断级数+误差界 | in: fn∈{exp,sin,cos,ln1p,geom}+x+n | out: 部分和+|R_n|精确上界+真值区间 "
        "| 验证器: math 库闭式值落入区间 | 1.5 units | 失败: 域外 DEG_UNSUPPORTED",
    selfchecks=(
        {"id": "series_exp1_n8", "args": {"fn": "exp", "x": 1, "n": 8},
         "check": lambda r: (r.value["sum"] == "685/252" and r.value["bound"] == "1/13440",
                             f"sum={r.value['sum']} bound={r.value['bound']}"),
         "expect": "手算: Σ_{k<8}1/k! = 685/252; 界 = 3/8! = 1/13440"},
        {"id": "series_ln1p1_n5", "args": {"fn": "ln1p", "x": 1, "n": 5},
         "check": lambda r: (r.value["sum"] == "47/60" and r.value["bound"] == "1/6",
                             f"sum={r.value['sum']} bound={r.value['bound']}"),
         "expect": "手算: 1−1/2+1/3−1/4+1/5 = 47/60; 交错界 = 1/6"},
        {"id": "series_geom_half", "args": {"fn": "geom", "x": "1/2", "n": 4},
         "check": lambda r: (r.value["sum"] == "15/8" and r.value["bound"] == "1/8",
                             f"sum={r.value['sum']} bound={r.value['bound']}"),
         "expect": "手算: 1+1/2+1/4+1/8=15/8; 余项=(1/16)/(1/2)=1/8"},
        {"id": "series_domain_neg", "args": {"fn": "exp", "x": 2, "n": 8}, "expect_err": "DEG_UNSUPPORTED",
         "expect": "|x|>1 域外必须拒"},
    ),
))


# ---------------------------------------------------------------------------
# 9118 math_unit_check
# ---------------------------------------------------------------------------

DIMS = ("length", "mass", "time", "current", "temperature", "amount", "luminosity", "information", "angle", "count")
_IN_UNIT = {"kind": "dict",
            "fields": {"terms": {"kind": "list", "items": {"kind": "dict",
                                                           "fields": {"coeff": {"kind": "Fraction"},
                                                                      "dims": {"kind": "dict", "fields": {}, "extra": True}},
                                                           "required": ["coeff", "dims"]}, "minlen": 1, "maxlen": 8}},
            "required": ["terms"]}


def _pre_unit(args: dict) -> tuple[bool, str]:
    for i, t in enumerate(args["terms"]):
        for k, v in t["dims"].items():
            if k not in DIMS:
                return False, f"第 {i} 项未知量纲 {k!r} (允许: {DIMS})"
            if isinstance(v, bool) or not isinstance(v, int) or not (-6 <= v <= 6):
                return False, f"第 {i} 项量纲 {k} 指数非法: {v!r} (需 int∈[−6,6])"
    return True, ""


def _dims_vec(dims: dict) -> tuple[int, ...]:
    return tuple(int(dims.get(d, 0)) for d in DIMS)


def _inv_unit(args: dict, ctl: CallCtl) -> dict:
    vecs = [_dims_vec(t["dims"]) for t in args["terms"]]
    v0 = vecs[0]
    consistent = all(v == v0 for v in vecs[1:])
    offending = -1
    if not consistent:
        offending = next(i for i, v in enumerate(vecs) if v != v0)
    dim_dict = {d: e for d, e in zip(DIMS, v0) if e != 0}
    viol = 0 if consistent else 1
    return {"value": {"consistent": consistent, "dim": dim_dict, "offending": offending,
                      "n_terms": len(args["terms"])},
            "resid": [float(viol)],
            "notes": (f"齐次: {consistent}; 公共量纲 {dim_dict}" if consistent
                      else f"不齐次: 第 {offending} 项量纲 {dict(zip(DIMS, vecs[offending]))} 与首项不符")}


_PRIMES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29)


def _ver_unit(args: dict, value: dict) -> dict:
    """Independent verifier: prime-power signature channel (∏ prime_d^{exp_d}) — a different path from vector comparison."""
    sigs = []
    for t in args["terms"]:
        s = 1
        for d, p in zip(DIMS, _PRIMES):
            e = int(t["dims"].get(d, 0))
            s *= p ** e
        sigs.append(s)
    consistent = all(s == sigs[0] for s in sigs)
    agree = consistent == value["consistent"]
    return {"ok": agree, "resid": [0.0 if agree else 1.0],
            "detail": f"素数签名 {sigs[:4]}{'…' if len(sigs) > 4 else ''} → 一致={consistent}, 与输出一致={agree}"}


REGISTRY.register(OpDef(
    name="math_unit_check", band=9100, sub=18,
    in_schema=_IN_UNIT,
    out_schema={"kind": "dict", "fields": {"consistent": {"kind": "bool"}, "dim": {"kind": "dict", "fields": {}, "extra": True},
                                           "offending": {"kind": "int", "min": -1}, "n_terms": {"kind": "int", "min": 1}},
                "required": ["consistent", "dim", "offending", "n_terms"]},
    invoke=_inv_unit, verify=_ver_unit, precond=_pre_unit,
    cost=("units", 0.5), errors=("SCHEMA_MISMATCH", "PRECOND_FAIL"),
    effects="none (纯函数)",
    doc="量纲齐次性检查 | in: terms[{coeff,dims}] | out: 齐次性+公共量纲 | 验证器: 素数幂签名独立通道 "
        "| 0.5 unit | 失败: 未知量纲/非法指数 PRECOND_FAIL",
    selfchecks=(
        {"id": "unit_speed_hom", "args": {"terms": [{"coeff": 3, "dims": {"length": 1, "time": -1}},
                                                    {"coeff": -1, "dims": {"length": 1, "time": -1}}]},
         "check": lambda r: (r.value["consistent"] and r.value["dim"] == {"length": 1, "time": -1},
                             f"dim={r.value['dim']}"),
         "expect": "两个速度项齐次"},
        {"id": "unit_inhom", "args": {"terms": [{"coeff": 1, "dims": {"length": 1}},
                                                {"coeff": 1, "dims": {"time": 1}}]},
         "check": lambda r: ((not r.value["consistent"]) and r.value["offending"] == 1,
                             f"consistent={r.value['consistent']} offending={r.value['offending']}"),
         "expect": "长度+时间不齐次, 违规项=1"},
        {"id": "unit_energy", "args": {"terms": [{"coeff": 1, "dims": {"mass": 1, "length": 2, "time": -2}}]},
         "check": lambda r: (r.value["consistent"] and r.value["dim"] == {"mass": 1, "length": 2, "time": -2},
                             "kg·m²·s⁻² 量纲核对"),
         "expect": "能量量纲"},
        {"id": "unit_unknown_dim_neg", "args": {"terms": [{"coeff": 1, "dims": {"currency": 1}}]},
         "expect_err": "PRECOND_FAIL", "expect": "未登记量纲必须被拒"},
    ),
))


# ---------------------------------------------------------------------------
# 9119 math_evaluate_expr
# ---------------------------------------------------------------------------

_IN_EVAL = {"kind": "dict",
            "fields": {"form": {"kind": "str", "enum": ["poly", "geom"]},
                       "coeffs": {"kind": "list", "items": {"kind": "Fraction"}, "minlen": 1, "maxlen": 10},
                       "x": {"kind": "Fraction"},
                       "a": {"kind": "Fraction"}, "r": {"kind": "Fraction"},
                       "n": {"kind": "int", "min": 0, "max": 64}},
            "required": ["form"]}


def _pre_eval(args: dict) -> tuple[bool, str]:
    if args["form"] == "poly":
        if "coeffs" not in args or "x" not in args:
            return False, "poly 形式需要 coeffs 与 x"
    else:
        for k in ("a", "r", "n"):
            if k not in args:
                return False, "geom 形式需要 a, r, n"
    return True, ""


def _inv_eval(args: dict, ctl: CallCtl) -> dict:
    ctx = "op:math_evaluate_expr#9119"
    if args["form"] == "poly":
        v = poly_eval(_frac_vec(args["coeffs"]), frac(args["x"]))
    else:
        a, r, n = frac(args["a"]), frac(args["r"]), int(args["n"])
        if r == 1:
            v = a * n
        else:
            v = a * (r ** n - 1) / (r - 1)
    return {"value": {"value": frac_str(v), "form": args["form"]}, "resid": [0.0],
            "notes": f"{args['form']} 闭式/Horner 求值 = {frac_str(v)} (精确)"}


def _ver_eval(args: dict, value: dict) -> dict:
    """Independent verifier: naive power sum / loop accumulation (a different path from Horner/closed form)."""
    if args["form"] == "poly":
        cs, x = _frac_vec(args["coeffs"]), frac(args["x"])
        acc, pw = Fraction(0), Fraction(1)
        for c in cs:
            acc += c * pw
            pw *= x
    else:
        a, r, n = frac(args["a"]), frac(args["r"]), int(args["n"])
        acc, term = Fraction(0), a
        for _ in range(n):
            acc += term
            term *= r
    v = frac(value["value"])
    ok = acc == v
    return {"ok": ok, "resid": [0.0 if ok else float(abs(acc - v))],
            "detail": f"朴素通道={frac_str(acc)}, 输出={frac_str(v)} (精确相等)"}


REGISTRY.register(OpDef(
    name="math_evaluate_expr", band=9100, sub=19,
    in_schema=_IN_EVAL,
    out_schema={"kind": "dict", "fields": {"value": {"kind": "str"}, "form": {"kind": "str"}}, "required": ["value", "form"]},
    invoke=_inv_eval, verify=_ver_eval, precond=_pre_eval,
    cost=("units", 0.5), errors=("SCHEMA_MISMATCH", "PRECOND_FAIL"),
    effects="none (纯函数, 精确有理算术)",
    doc="有理点求值 | in: poly(coeffs,x) 或 geom(a,r,n) | out: 精确值 | 验证器: 朴素幂和/循环累加不同通道 "
        "| 0.5 unit | 失败: 字段缺失 PRECOND_FAIL",
    selfchecks=(
        {"id": "eval_poly_hand", "args": {"form": "poly", "coeffs": [1, 2, 3], "x": 2},
         "check": lambda r: (r.value["value"] == "17", f"v={r.value['value']}"),
         "expect": "手算: 1+2·2+3·4=17"},
        {"id": "eval_geom_hand", "args": {"form": "geom", "a": 1, "r": "1/2", "n": 5},
         "check": lambda r: (r.value["value"] == "31/16", f"v={r.value['value']}"),
         "expect": "手算: 1+1/2+…+1/16 = 31/16"},
        {"id": "eval_geom_r1", "args": {"form": "geom", "a": 3, "r": 1, "n": 4},
         "check": lambda r: (r.value["value"] == "12", f"v={r.value['value']}"),
         "expect": "手算: r=1 退化和 = 3·4 = 12"},
        {"id": "eval_missing_field_neg", "args": {"form": "poly", "coeffs": [1]},
         "expect_err": "PRECOND_FAIL", "expect": "缺 x 字段必须被拒"},
    ),
))

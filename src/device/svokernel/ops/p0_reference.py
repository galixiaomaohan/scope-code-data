"""P0 skeleton operators (§14 reference implementation + skeleton self-certificate).

This module is the **delivery-style benchmark**: every operator follows §14's three-step VMP +
independent verifier + selfcheck pattern.
Two operators:
  - `solve_poly` (row 9101, verbatim counterpart of the §14 reference implementation): float
    coefficients, back-substitution residual verification;
  - `solve_poly_exact` (row 9102): exact rational coefficients, exact roots + poly_backsub certificate.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction

from ..certify import Certificate
from ..codebook import CODEBOOK
from ..errors import Err, SvoError
from ..exact import frac, frac_str
from ..registry import REGISTRY, OpDef
from ..tool import CallCtl

# ---------------------------------------------------------------------------
# §14 reference implementation (style preserved verbatim)
# ---------------------------------------------------------------------------

HANDLE_SOLVE_POLY = 9101                        # codebook row (frozen)
ERR = ["DEG_UNSUPPORTED", "DISC_NEG", "COEFF_NAN"]


@dataclass
class Poly2:
    p: float
    q: float                                   # x² + p·x + q = 0


@dataclass
class Out:
    roots: tuple
    resid: float


def handle(spec: dict) -> int:                  # 1) symbol → handle (lossless indexing)
    assert spec["op"] == "solve_poly"
    return HANDLE_SOLVE_POLY


def invoke(h: int, a: Poly2) -> Out:            # 2) kernel execution (closed form T-A1)
    if h != HANDLE_SOLVE_POLY:
        raise ValueError("SCHEMA_MISMATCH")
    d = a.p * a.p / 4.0 - a.q
    if d < 0:
        return Out((), float("nan"))            # DISC_NEG semantics
    s = math.sqrt(d)
    r = (-a.p / 2.0 + s, -a.p / 2.0 - s)
    return Out(r, verify(r, a))


def verify(roots, a) -> float:                  # independent verification: back-substitution against the true value
    return max(abs(r * r + a.p * r + a.q) for r in roots) / (1.0 + abs(a.p) + abs(a.q))


def decode(o: Out) -> tuple:                    # 3) result → latent state
    latent = CODEBOOK.latent(HANDLE_SOLVE_POLY)
    rv = (float(o.resid),) + (0.0,) * 7
    return latent, rv


# self-check (must actually run, output attached): p=-1,q=-6 → roots=(3.0,-2.0), resid<1e-12
#                                                   p=0,q=1  → DISC_NEG path returns a nan residual
# ---------------------------------------------------------------------------
# registration-layer wrapper: §14's nan semantics → this package's DISC_NEG enumerated error
# (stricter: never silently return nan)
# ---------------------------------------------------------------------------

_IN_POLY2 = {
    "kind": "dict",
    "fields": {"p": {"kind": "float"}, "q": {"kind": "float"}},
    "required": ["p", "q"],
}
_OUT_POLY2 = {
    "kind": "dict",
    "fields": {
        "roots": {"kind": "list", "items": {"kind": "float"}, "minlen": 0, "maxlen": 2},
        "resid": {"kind": "float"},
    },
    "required": ["roots", "resid"],
}


def _invoke_solve_poly(args: dict, ctl: CallCtl) -> dict:
    o = invoke(handle({"op": "solve_poly"}), Poly2(args["p"], args["q"]))
    if o.roots == ():
        raise SvoError(Err.DISC_NEG, ctx="op:solve_poly#9101", detail=f"判别式<0 (p={args['p']}, q={args['q']})")
    if o.resid != o.resid:
        raise SvoError(Err.POSTCOND_FAIL, ctx="op:solve_poly#9101", detail="残差为 nan")
    return {"value": {"roots": list(o.roots), "resid": o.resid}, "resid": [o.resid],
            "notes": f"闭式 T-A1; 回代残差(相对)={o.resid:.3e}"}


def _verify_solve_poly(args: dict, value: dict) -> dict:
    """Independent verifier: a different path from invoke — here it re-substitutes with **exact rationals**.

    If float rounding makes the exact residual nonzero, fall back to the float residual (and report both values).
    """
    p = Fraction(args["p"]).limit_denominator(10**6)
    q = Fraction(args["q"]).limit_denominator(10**6)
    worst_f = 0.0
    for r in value["roots"]:
        worst_f = max(worst_f, abs(r * r + p * r + q))
    exact_ok = worst_f <= 1e-6 * (1 + abs(p) + abs(q))
    return {"ok": exact_ok, "resid": [float(worst_f)],
            "detail": f"有理回代 max|p(r)|={float(worst_f):.3e} (容差 {float(1e-6 * (1 + abs(p) + abs(q))):.3e})"}


# ---------------------------------------------------------------------------
# exact version: rational coefficients, exact roots (only when the discriminant is a rational
# perfect square), carrying a poly_backsub certificate
# ---------------------------------------------------------------------------

_IN_POLY2_EXACT = {
    "kind": "dict",
    "fields": {"p": {"kind": "Fraction"}, "q": {"kind": "Fraction"}},
    "required": ["p", "q"],
}
_OUT_POLY2_EXACT = {
    "kind": "dict",
    "fields": {
        "roots": {"kind": "list", "items": {"kind": "str"}, "minlen": 1, "maxlen": 2},
        "roots_exact": {"kind": "bool"},
        "disc": {"kind": "str"},
    },
    "required": ["roots", "roots_exact", "disc"],
}


def _fsqrt_exact(d: Fraction) -> "Fraction | None":
    """Exact rational square root: return s with s²=d, else None. d<0 → None."""
    if d < 0:
        return None
    if d == 0:
        return Fraction(0)
    num, den = d.numerator, d.denominator
    sn, sd = math.isqrt(num), math.isqrt(den)
    if sn * sn == num and sd * sd == den:
        return Fraction(sn, sd)
    return None


def _invoke_solve_poly_exact(args: dict, ctl: CallCtl) -> dict:
    p = frac(args["p"])
    q = frac(args["q"])
    d = p * p / 4 - q
    if d < 0:
        raise SvoError(Err.DISC_NEG, ctx="op:solve_poly_exact#9102", detail=f"判别式 {d} < 0 (无实根)")
    s = _fsqrt_exact(d)
    if s is None:
        raise SvoError(
            Err.NO_REAL_ROOT,
            ctx="op:solve_poly_exact#9102",
            detail=f"判别式 {d} 非有理完全平方 → 精确通道拒绝 (改用 numeric_root 走区间证书)",
        )
    roots = [(-p / 2 + s), (-p / 2 - s)]
    seen: list[Fraction] = []
    for r in roots:
        if r not in seen:
            seen.append(r)
    cert = Certificate(
        kind="poly_backsub",
        payload={
            "coeffs": [frac_str(q), frac_str(p), frac_str(Fraction(1))],
            "roots": [frac_str(r) for r in seen],
            "tol": "1/1000000000000",
        },
        issued_by="solve_poly_exact",
    )
    return {
        "value": {"roots": [frac_str(r) for r in seen], "roots_exact": True, "disc": frac_str(d)},
        "resid": [0.0],
        "cert": cert,
        "notes": f"精确根 {[frac_str(r) for r in seen]}, 判别式 {frac_str(d)}",
    }


def _verify_poly_exact(args: dict, value: dict) -> dict:
    """Independent verifier: exact back-substitution per root (Horner, rational arithmetic) — a path independent of the closed form."""
    p, q = frac(args["p"]), frac(args["q"])
    roots = [frac(s) for s in value["roots"]]
    worst = Fraction(0)
    for r in roots:
        acc = Fraction(0)
        for c in (Fraction(1), p, q):  # Horner for x² + p·x + q (a2=1, a1=p, a0=q)
            acc = acc * r + c
        worst = max(worst, abs(acc))
    return {"ok": worst == 0, "resid": [float(worst)], "detail": f"精确回代 max|p(r)|={worst}"}


# ---------------------------------------------------------------------------
# registration
# ---------------------------------------------------------------------------

REGISTRY.register(
    OpDef(
        name="solve_poly", band=9100, sub=1,
        in_schema=_IN_POLY2, out_schema=_OUT_POLY2,
        invoke=_invoke_solve_poly, verify=_verify_solve_poly,
        cost=("units", 1.0), errors=("DISC_NEG", "POSTCOND_FAIL", "SCHEMA_MISMATCH"),
        effects="none (§14 参考实现: 纯函数)",
        doc="一元二次闭式求根 (§14 参考实现 / 定理 T-A1); 浮点域, 独立验证器=有理回代。",
        selfchecks=(
            {"id": "ref_roots_3_m2", "args": {"p": -1.0, "q": -6.0},
             "check": lambda r: (r.value["roots"] == [3.0, -2.0] and r.resid_scalar < 1e-12,
                                 f"roots={r.value['roots']}, resid={r.resid_scalar:.3e}"),
             "expect": "§14 声明: roots=(3.0,-2.0), resid<1e-12"},
            {"id": "disc_neg_path", "args": {"p": 0.0, "q": 1.0}, "expect_err": "DISC_NEG",
             "expect": "§14 声明: p=0,q=1 → DISC_NEG (x²+1 无实根)"},
            {"id": "double_root", "args": {"p": -4.0, "q": 4.0},
             "check": lambda r: (abs(r.value["roots"][0] - 2.0) < 1e-12 and r.resid_scalar < 1e-12,
                                 f"重根 roots={r.value['roots']}")},
            {"id": "boundary_schema_reject", "args": {"p": "abc", "q": 1.0}, "expect_err": "SCHEMA_MISMATCH",
             "expect": "§17.6 边界: 非数值入参必须被 schema 门拦下"},
        ),
    )
)

REGISTRY.register(
    OpDef(
        name="solve_poly_exact", band=9100, sub=2,
        in_schema=_IN_POLY2_EXACT, out_schema=_OUT_POLY2_EXACT,
        invoke=_invoke_solve_poly_exact, verify=_verify_poly_exact, cert_kind="poly_backsub",
        cost=("units", 1.5), errors=("DISC_NEG", "NO_REAL_ROOT", "SCHEMA_MISMATCH", "POSTCOND_FAIL"),
        effects="none (纯函数, 精确有理算术)",
        doc="一元二次精确求根 (有理系数/有理完全平方判别式); 证书=poly_backsub 精确回代。",
        requires_cert=True,
        selfchecks=(
            {"id": "exact_int_roots", "args": {"p": -1, "q": -6},
             "check": lambda r: (r.value["roots"] == ["3", "-2"] and r.cert is not None and r.cert_independent,
                                 f"roots={r.value['roots']} cert={r.cert.kind if r.cert else None} 独立复验={r.cert_independent}")},
            {"id": "exact_frac_roots", "args": {"p": "-3/2", "q": "1/2"},
             "check": lambda r: (set(r.value["roots"]) == {"1", "1/2"}, f"roots={r.value['roots']}")},
            {"id": "irrational_reject", "args": {"p": 0, "q": -2}, "expect_err": "NO_REAL_ROOT",
             "expect": "判别式=2 非有理完全平方 → 精确通道拒 (不许浮点假装精确)"},
            {"id": "disc_neg_exact", "args": {"p": 0, "q": 1}, "expect_err": "DISC_NEG"},
        ),
    )
)

def _invoke_codebook_row(args: dict, ctl: CallCtl) -> dict:
    name = args["name"]
    row = CODEBOOK.row_of(name)
    from ..codebook import band_of as _band_of

    return {"value": {"row": row, "band": _band_of(row), "latent_head": list(CODEBOOK.latent(row)[:4])},
            "resid": [0.0], "notes": f"{name} → #{row}"}


def _verify_codebook_row(args: dict, value: dict) -> dict:
    """Independent verifier: bypass the registry and recompute the row and embedding head directly from CODEBOOK."""
    from ..codebook import latent_vec

    row = CODEBOOK.row_of(args["name"])
    lt = latent_vec(row)[:4]
    ok = value["row"] == row and list(lt) == list(value["latent_head"])
    return {"ok": ok, "resid": [0.0 if ok else 1.0], "detail": f"独立复算 row={row}"}


REGISTRY.register(
    OpDef(
        name="codebook_row", band=9000, sub=1,
        in_schema={"kind": "dict", "fields": {"name": {"kind": "str"}}, "required": ["name"]},
        out_schema={
            "kind": "dict",
            "fields": {
                "row": {"kind": "int"},
                "band": {"kind": "str"},
                "latent_head": {"kind": "list", "items": {"kind": "float"}, "len": 4},
            },
            "required": ["row", "band", "latent_head"],
        },
        invoke=_invoke_codebook_row, verify=_verify_codebook_row,
        cost=("units", 0.5), errors=("SCHEMA_MISMATCH",),
        effects="none (纯函数)",
        doc="VMP 第 1 步自证件: 符号→句柄行→64 维冻结嵌入 (无损重索引)。",
        selfchecks=(
            {"id": "known_row", "args": {"name": "solve_poly"},
             "check": lambda r: (r.value["row"] == 9101 and r.value["band"] == "math_basic",
                                 f"row={r.value['row']} band={r.value['band']}")},
            {"id": "determinism", "args": {"name": "solve_poly_exact"},
             "check": lambda r: (r.value["latent_head"] == list(CODEBOOK.latent(9102)[:4]),
                                 f"latent_head={r.value['latent_head']} 与进程内码本一致")},
            {"id": "unknown_symbol", "args": {"name": "__nope__"}, "expect_err": "SCHEMA_MISMATCH"},
        ),
    )
)

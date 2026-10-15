# -*- coding: utf-8 -*-
"""V31 infrastructure operators (band 9000, rows 9002–9004) -- codebook self-witness /
algebraic-number arithmetic / ledger.

Three operators carry the "self-description" duty of V31 (same idea as the legacy
p0_reference.codebook_row):
  v31_codebook_row  9002  V31 codebook self-witness: symbol -> row -> 64-dim frozen embedding head + is-operator flag
  v31_qsqrtd_arith  9003  Q(√d) exact arithmetic (callable entry of the new type system + equality certificate)
  v31_census        9004  V31 ledger (band/layer/status censuses + anchor-chain verdict + frozen-layout assertions)

**Frozen layout (V31_FROZEN)**: these three self-checks hard-code the scale of V31. Anyone who
adds or removes an operator turns the census self-check red immediately -- that is the "layout
frozen" discipline we want: the head dimension is a model-side premise and must not drift
silently.
"""
from __future__ import annotations

from .v31registry import V31OpDef, V31REGISTRY, anchor_chain_audit, audit_static_v31

from svokernel.certify import Certificate
from svokernel.codebook import latent_vec
from svokernel.errors import Err, SvoError
from svokernel.exact import frac, frac_str
from svokernel.tool import CallCtl

from .algnum import QsqrtD, is_squarefree_int
from .v31codebook import (V31_CODEBOOK, V31_END_MARKER, V31_END_MARKER_ROW, v31_band_of,
                          v31_head_dim)

BAND_INFRA = 9000
_EXACT = {"kind": "int"}
_FR = {"kind": "Fraction"}

#: Frozen layout (settled in stage F; asserted item by item by the census self-check).
V31_FROZEN = {
    "total_ops": 154,          # 118 legacy refs + 10 L0 + 11 L1 + 8 L2 + 4 L3 + 3 infra
    "n_verified": 146,
    "n_stub": 8,               # three L1 + five L2
    "n_legacy": 118,
    "codebook_rows": 155,      # 154 operator rows + 1 END sentinel row
    "head_dim": 155,           # model-side routing head = operators + 1 (END slot) = codebook rows
    "checks_total": 656,
}


def _sdict(fields, req):
    return {"kind": "dict", "fields": fields, "required": req}


def _reg(**kw):
    kw.setdefault("status", "verified")
    kw.setdefault("layer", -1)      # -1 = V31 infrastructure
    return V31REGISTRY.register(V31OpDef(**kw))


# ===========================================================================
# 9002  v31_codebook_row
# ===========================================================================

_CR_IN = _sdict({"name": {"kind": "str"}}, ["name"])
_CR_OUT = _sdict({
    "name": {"kind": "str"}, "row": _EXACT, "band": {"kind": "str"},
    "latent_head": {"kind": "list", "items": {"kind": "float"}, "len": 4},
    "is_operator": {"kind": "bool"}, "is_end_marker": {"kind": "bool"},
    "identity_checked": {"kind": "bool"},
}, ["name", "row", "band", "latent_head", "is_operator", "is_end_marker", "identity_checked"])


def _invoke_cr(args, ctl):
    name = args["name"]
    row = V31_CODEBOOK.row_of(name)      # unregistered raises SCHEMA_MISMATCH (with ctx)
    band = v31_band_of(row)
    head = list(latent_vec(row)[:4])
    is_op = V31REGISTRY.has(name)
    is_end = (name == V31_END_MARKER)
    # consistency check: if it is also an operator, the registry handle must match the codebook row
    ident = True
    if is_op:
        ident = (V31REGISTRY.tool(name).handle == row)
    return {"value": {"name": name, "row": row, "band": band, "latent_head": head,
                      "is_operator": is_op, "is_end_marker": is_end,
                      "identity_checked": ident},
            "resid": [0.0 if ident else 1.0], "cost": 0.5,
            "notes": "%s → #%d (%s)%s" % (name, row, band,
                                          " [END 哨兵, 非算子]" if is_end else "")}


def _verify_cr(args, value):
    """Independent verifier: recompute the row/embedding head directly against the V31 codebook
    (never through the registry) and check the band-name window."""
    bad = []
    row = V31_CODEBOOK.row_of(args["name"])
    if row != value["row"]:
        bad.append("行号复算 %d ≠ %d" % (row, value["row"]))
    if list(latent_vec(row)[:4]) != list(value["latent_head"]):
        bad.append("嵌入头复算不符")
    if v31_band_of(row) != value["band"]:
        bad.append("段名复算不符")
    if value["is_operator"] != V31REGISTRY.has(args["name"]):
        bad.append("is_operator 与注册表不符")
    end_row_ok = (value["is_end_marker"] and row == V31_END_MARKER_ROW)
    if value["is_end_marker"] and not end_row_ok:
        bad.append("END 哨兵行号漂移")
    return {"ok": not bad, "resid": [float(len(bad))],
            "detail": "码本行/嵌入头/段名独立复算一致 (行 %d)" % row if not bad else "; ".join(bad)}


_reg(
    name="v31_codebook_row", band=BAND_INFRA, sub=2,
    in_schema=_CR_IN, out_schema=_CR_OUT,
    invoke=_invoke_cr, verify=_verify_cr,
    cost=("units", 0.5), errors=("SCHEMA_MISMATCH", "POSTCOND_FAIL"),
    effects="none (纯函数)",
    doc=("V31 codebook self-witness (VMP step 1): symbol -> row -> 64-dim frozen embedding head, "
         "marking whether the symbol is an operator or the END sentinel. The independent verifier "
         "recomputes row/embedding head/band name directly against the V31 codebook, never "
         "through the registry. The embedding function reuses the legacy latent_vec "
         "(deterministic derivation from the same GLOBAL_SEED) -- V31 did not invent a second "
         "vector film."),
    selfchecks=(
        {"id": "cr_nt_lte_row", "args": {"name": "nt_lte"},
         "check": lambda r: (r.value["row"] == 9901 and r.value["band"] == "nt_comp_L0"
                             and r.value["is_operator"] and r.value["identity_checked"]
                             and r.value["latent_head"] == list(latent_vec(9901)[:4]),
                             "nt_lte → #%d (%s)" % (r.value["row"], r.value["band"])),
         "expect": "L0 首算子冻结行 9901"},
        {"id": "cr_legacy_row_identical", "args": {"name": "solve_poly"},
         "check": lambda r: (r.value["row"] == 9101 and r.value["band"] == "math_basic",
                             "存量算子行号在 V31 码本中不变: #%d" % r.value["row"]),
         "expect": "锚链: 存量行号逐行复刻 (9101)"},
        {"id": "cr_end_marker", "args": {"name": "v31_end_marker"},
         "check": lambda r: (r.value["row"] == 9999 and r.value["is_end_marker"]
                             and not r.value["is_operator"],
                             "END 哨兵 #9999 (非算子, 只占头槽)"),
         "expect": "END 哨兵行 9999 不是算子"},
        {"id": "cr_unknown_reject", "args": {"name": "__nope__"},
         "expect_err": "SCHEMA_MISMATCH", "expect": "未登记符号被拒 (附 ctx 句柄)"},
        {"id": "cr_schema_reject", "args": {"name": 5},
         "expect_err": "SCHEMA_MISMATCH", "expect": "非字符串入参"},
    ),
)


# ===========================================================================
# 9003  v31_qsqrtd_arith
# ===========================================================================

_QA_IN = _sdict({
    "d": {"kind": "int", "min": 2},
    "op": {"kind": "str", "enum": ["add", "sub", "mul", "norm", "inv"]},
    "x1": _FR, "y1": _FR, "x2": _FR, "y2": _FR,
}, ["d", "op", "x1", "y1", "x2", "y2"])
_QA_OUT = _sdict({
    "d": _EXACT, "op": {"kind": "str"},
    "operand1": {"kind": "str"}, "operand2": {"kind": "str"}, "result": {"kind": "str"},
    "result_field": {"kind": "str", "enum": ["QsqrtD", "Q"]},
    "norm1": {"kind": "str"}, "norm_multiplicative": {"kind": "bool"},
    "roundtrip_ok": {"kind": "bool"},
}, ["d", "op", "operand1", "operand2", "result", "result_field", "norm1",
    "norm_multiplicative", "roundtrip_ok"])


def _qa_precond(args):
    d = args["d"]
    if d <= 1:
        return False, "d > 1"
    if not is_squarefree_int(d):
        return False, ("d=%d 非无平方 ⇒ Q(√d) 退化为 Q (或非二次域); 本类型显式拒绝"
                       "而不静默降级" % d)
    if args["op"] == "inv" and (frac(args["x1"]) == 0 and frac(args["y1"]) == 0):
        return False, "零元素不可逆 (范数为 0)"
    return True, ""


def _invoke_qa(args, ctl):
    d, op = args["d"], args["op"]
    x1, y1 = frac(args["x1"]), frac(args["y1"])
    x2, y2 = frac(args["x2"]), frac(args["y2"])
    A = QsqrtD(d, x1, y1)
    # note: norm/inv are unary, but the schema requires all four coefficients (unified binary
    # interface); for unary operations operand 2 is **ignored**, which is stated in doc and notes,
    # never silently.
    if op == "norm":
        r = A.norm()
        res_field = "Q"
        res_str = frac_str(r)
        # B5 fix: this used to be Fraction(0), but the module only imports frac/frac_str (no
        # fractions.Fraction) => op="norm" raised NameError inside the **generator** (an
        # out-of-table exception => RUNTIME_ERR).
        res_pair = (r, frac(0))
    elif op == "inv":
        B = A.inv()
        res_field = "QsqrtD"
        res_str = B.to_str()
        res_pair = (B.a, B.b)
    else:
        B = QsqrtD(d, x2, y2)
        C = {"add": lambda: A + B, "sub": lambda: A - B, "mul": lambda: A * B}[op]()
        res_field = "QsqrtD"
        res_str = C.to_str()
        res_pair = (C.a, C.b)
    n1 = A.norm()
    nmul = True
    if op == "mul":
        nmul = (A * QsqrtD(d, x2, y2)).norm() == n1 * QsqrtD(d, x2, y2).norm()
    rt = True
    if op == "inv":
        rt = (A * A.inv()) == QsqrtD(d, 1, 0)
    # B6 fix: the certificate's operands must follow the **arity** of op -- the checker
    # (certcheck_alg, independent implementation) requires exactly 1 operand for norm/inv and
    # exactly 2 for add/sub/mul. The original always gave 2, so unary certificates were rejected
    # by the checker with "inv 需要一个操作数". The schema layer keeps x2/y2 for the unified
    # interface (unary operations ignore them, explicitly noted), but the **certificate** must not
    # carry an ignored operand.
    ops_pairs = ([[frac_str(x1), frac_str(y1)]] if op in ("norm", "inv")
                 else [[frac_str(x1), frac_str(y1)], [frac_str(x2), frac_str(y2)]])
    cert = Certificate(kind="qsqrtd_arith",
                       payload={"d": d, "op": op,
                                "operands": ops_pairs,
                                "result": [frac_str(res_pair[0]), frac_str(res_pair[1])]},
                       issued_by="v31_qsqrtd_arith")
    return {"value": {"d": d, "op": op, "operand1": A.to_str(),
                      "operand2": QsqrtD(d, x2, y2).to_str() if op not in ("norm", "inv") else "(忽略)",
                      "result": res_str, "result_field": res_field, "norm1": frac_str(n1),
                      "norm_multiplicative": nmul, "roundtrip_ok": rt},
            "resid": [0.0], "cert": cert, "cost": 1.5,
            "notes": "%s in Q(√%d): %s" % (op, d, res_str)}


def _verify_qa(args, value):
    """Independent verifier: recompute with the **regular representation** (2×2 matrix
    [[a, bd],[b, a]]) -- a different path from expanding the formulas directly."""
    d = args["d"]
    x1, y1 = frac(args["x1"]), frac(args["y1"])
    x2, y2 = frac(args["x2"]), frac(args["y2"])
    bad = []

    def reg(a, b):
        return (a, b * d, b, a)     # (m00, m01, m10, m11)

    def mul(M, N):
        return (M[0] * N[0] + M[1] * N[2], M[0] * N[1] + M[1] * N[3],
                M[2] * N[0] + M[3] * N[2], M[2] * N[1] + M[3] * N[3])

    op = args["op"]
    if op == "add":
        want = (x1 + x2, y1 + y2)
    elif op == "sub":
        want = (x1 - x2, y1 - y2)
    elif op == "mul":
        M = mul(reg(x1, y1), reg(x2, y2))
        if M[1] != M[2] * d:
            bad.append("矩阵乘法结果不是 Q(√d) 的型 (表示不一致)")
        want = (M[0], M[2])
    elif op == "norm":
        want = (x1 * x1 - d * y1 * y1, frac(0))
    else:
        n = x1 * x1 - d * y1 * y1
        if n == 0:
            bad.append("零元素不可逆")
            want = None
        else:
            want = (x1 / n, -y1 / n)
    if want is not None:
        got = (frac(value["result"])) if value["result_field"] == "Q" else None
        if value["result_field"] == "Q":
            if got != want[0]:
                bad.append("范数复算 %s ≠ 声称 %s" % (want[0], got))
        else:
            # read-back is lenient: to_str emits a pure rational string for domain elements with
            # b=0 ("5/6"), because a+0√d is just a (see the algnum.to_str convention). The
            # comparison only uses the (a,b) pair, independent of d.
            from .algnum import parse_qsqrtd_lenient
            P = parse_qsqrtd_lenient(value["result"])
            if (P.a, P.b) != want:
                bad.append("正规表示复算 %r ≠ 声称 (%s, %s)" % (want, P.a, P.b))
    if op == "inv" and not value["roundtrip_ok"]:
        bad.append("逆元回乘 ≠ 1")
    return {"ok": not bad, "resid": [float(len(bad))],
            "detail": "正规表示 (2×2 矩阵) 复算一致 (op=%s, d=%d)" % (op, d)
                      if not bad else "; ".join(bad)}


_reg(
    name="v31_qsqrtd_arith", band=BAND_INFRA, sub=3,
    in_schema=_QA_IN, out_schema=_QA_OUT,
    invoke=_invoke_qa, verify=_verify_qa, cert_kind="qsqrtd_arith",
    cost=("units", 1.5),
    errors=("PRECOND_FAIL", "DEG_UNSUPPORTED", "SCHEMA_MISMATCH", "POSTCOND_FAIL"),
    precond=_qa_precond,
    effects="none (纯函数, 精确有理算术)",
    doc=("Callable entry of the new type system Q(√d): add/sub/mul/norm/inv in canonical form "
         "(a + b√d, a,b ∈ Q) with an equality certificate. Generation = direct field-element "
         "formulas; independent verifier = **regular representation** (embed a+b√d into the 2×2 "
         "matrix [[a, bd],[b, a]] and do matrix arithmetic) -- a different code path from direct "
         "expansion; certificate qsqrtd_arith is recomputed independently by certcheck_alg. "
         "**Interface note**: d must be square-free and > 1 (a square d would degenerate Q(√d) to "
         "Q and is rejected at the type layer); norm/inv are unary, and the unified interface "
         "keeps x2/y2 but **ignores** them (stated in the notes)."),
    selfchecks=(
        {"id": "qa_mul_sqrt2", "args": {"d": 2, "op": "mul", "x1": 1, "y1": 1, "x2": 3, "y2": -1},
         "check": lambda r: (r.value["result"] == "1+2√2" and r.value["norm_multiplicative"]
                             and r.cert_independent,
                             "(1+√2)(3−√2) = %s (范数乘性成立)" % r.value["result"]),
         "expect": "(1+√2)(3−√2) = 1 + 2√2"},
        {"id": "qa_norm_sqrt2", "args": {"d": 2, "op": "norm", "x1": 1, "y1": 1, "x2": 0, "y2": 0},
         "check": lambda r: (r.value["result"] == "-1" and r.value["result_field"] == "Q",
                             "N(1+√2) = 1 − 2 = %s" % r.value["result"]),
         "expect": "范数: N(1+√2) = −1 (模 1 单位)"},
        {"id": "qa_inv_sqrt2", "args": {"d": 2, "op": "inv", "x1": 1, "y1": 1, "x2": 0, "y2": 0},
         "check": lambda r: (r.value["result"] == "-1+1√2" and r.value["roundtrip_ok"],
                             "(1+√2)^-1 = %s = √2 − 1" % r.value["result"]),
         "expect": "逆元: (1+√2)^{-1} = √2 − 1"},
        {"id": "qa_add_rational", "args": {"d": 3, "op": "add", "x1": "1/2", "y1": 0, "x2": "1/3", "y2": 0},
         "check": lambda r: (r.value["result"] == "5/6", "有理元素相加: %s" % r.value["result"]),
         "expect": "b=0 的有理元素 (含分数系数)"},
        {"id": "qa_nonsquarefree_reject", "args": {"d": 4, "op": "mul", "x1": 1, "y1": 0, "x2": 1, "y2": 0},
         "expect_err": "PRECOND_FAIL", "expect": "d=4 完全平方 ⇒ 类型层拒绝"},
        {"id": "qa_inv_zero_reject", "args": {"d": 2, "op": "inv", "x1": 0, "y1": 0, "x2": 0, "y2": 0},
         "expect_err": "PRECOND_FAIL", "expect": "零元素不可逆"},
        {"id": "qa_bad_op_reject", "args": {"d": 2, "op": "pow", "x1": 1, "y1": 0, "x2": 1, "y2": 0},
         "expect_err": "SCHEMA_MISMATCH", "expect": "未登记 op 被 enum 拦下"},
        {"id": "qa_float_reject", "args": {"d": 2, "op": "mul", "x1": 0.5, "y1": 1, "x2": 1, "y2": 0},
         "expect_err": "SCHEMA_MISMATCH", "expect": "float 不得进精确通道"},
    ),
)


# ===========================================================================
# 9004  v31_census
# ===========================================================================

_CEN_IN = _sdict({"group_by": {"kind": "str", "enum": ["band", "status", "layer"]},
                  "with_audit": {"kind": "bool"}}, ["group_by", "with_audit"])
_CEN_OUT = _sdict({
    "group_by": {"kind": "str"},
    "counts": {"kind": "list", "minlen": 1, "maxlen": 32,
               "items": {"kind": "tuple", "items": [{"kind": "str"}, _EXACT]}},
    "total_ops": _EXACT, "n_verified": _EXACT, "n_stub": _EXACT, "n_legacy": _EXACT,
    "codebook_rows": _EXACT, "head_dim": _EXACT, "checksum": {"kind": "str"},
    "checks_total": _EXACT, "checks_stub": _EXACT,
    "layout_frozen_ok": {"kind": "bool"}, "static_problems": _EXACT,
    "anchor_ok": {"kind": "bool"}, "anchor_n_legacy_joined": _EXACT,
}, ["group_by", "counts", "total_ops", "n_verified", "n_stub", "n_legacy", "codebook_rows",
    "head_dim", "checksum", "checks_total", "checks_stub", "layout_frozen_ok",
    "static_problems", "anchor_ok", "anchor_n_legacy_joined"])


def _invoke_census(args, ctl):
    n_legacy = 0
    from svokernel.registry import REGISTRY as _LR
    for n in _LR.names():
        if V31REGISTRY.has(n):
            n_legacy += 1
    if args["group_by"] == "band":
        raw = V31REGISTRY.band_census()
    elif args["group_by"] == "status":
        raw = V31REGISTRY.status_census()
    else:
        raw = {str(k): v for k, v in V31REGISTRY.layer_census().items()}
    counts = sorted((str(k), int(v)) for k, v in raw.items())
    statuses = V31REGISTRY.status_census()
    checks = V31REGISTRY.checks_by_status()
    frozen_ok = (len(V31REGISTRY) == V31_FROZEN["total_ops"]
                 and statuses.get("verified", 0) == V31_FROZEN["n_verified"]
                 and statuses.get("stub", 0) == V31_FROZEN["n_stub"]
                 and n_legacy == V31_FROZEN["n_legacy"]
                 and V31_CODEBOOK.size() == V31_FROZEN["codebook_rows"]
                 and v31_head_dim() == V31_FROZEN["head_dim"])
    static = audit_static_v31()
    anchor = anchor_chain_audit() if args["with_audit"] else {"ok": False, "n_legacy_joined": 0}
    return {"value": {"group_by": args["group_by"], "counts": counts,
                      "total_ops": len(V31REGISTRY), "n_verified": statuses.get("verified", 0),
                      "n_stub": statuses.get("stub", 0), "n_legacy": n_legacy,
                      "codebook_rows": V31_CODEBOOK.size(), "head_dim": v31_head_dim(),
                      "checksum": V31_CODEBOOK.checksum(),
                      "checks_total": sum(checks.values()), "checks_stub": checks.get("stub", 0),
                      "layout_frozen_ok": frozen_ok, "static_problems": len(static),
                      "anchor_ok": bool(anchor["ok"]),
                      "anchor_n_legacy_joined": int(anchor.get("n_legacy_joined", 0))},
            "resid": [0.0 if (frozen_ok and not static) else 1.0], "cost": 5.0,
            "notes": "总数 %d (verified %d / stub %d), 码本 %d 行, 头维度 %d, 冻结一致=%s"
                     % (len(V31REGISTRY), statuses.get("verified", 0), statuses.get("stub", 0),
                        V31_CODEBOOK.size(), v31_head_dim(), frozen_ok)}


def _verify_census(args, value):
    """Independent verifier: recompute the counts **separately** from the codebook row table and
    from the registry name list (two paths cross-checked)."""
    bad = []
    n_ops = len(V31REGISTRY)
    if n_ops != value["total_ops"]:
        bad.append("算子数复算 %d ≠ %d" % (n_ops, value["total_ops"]))
    st = V31REGISTRY.status_census()
    if st.get("verified", 0) != value["n_verified"] or st.get("stub", 0) != value["n_stub"]:
        bad.append("状态计数不符")
    if st.get("verified", 0) + st.get("stub", 0) != n_ops:
        bad.append("verified + stub ≠ total (台账不闭合)")
    if V31_CODEBOOK.size() != value["codebook_rows"]:
        bad.append("码本行数不符")
    # head dimension = codebook rows (the convention where the END sentinel occupies one row)
    if not V31_CODEBOOK.has(V31_END_MARKER):
        bad.append("END 哨兵缺失 ⇒ 头维度口径不成立")
    elif V31_CODEBOOK.size() != value["head_dim"]:
        bad.append("head_dim ≠ 码本行数 (END 哨兵口径)")
    ck = V31REGISTRY.checks_by_status()
    if sum(ck.values()) != value["checks_total"]:
        bad.append("自检项总数不符")
    if ck.get("stub", 0) != value["checks_stub"]:
        bad.append("stub 自检项数不符")
    if bool(value["layout_frozen_ok"]) != (
            n_ops == V31_FROZEN["total_ops"]
            and st.get("verified") == V31_FROZEN["n_verified"]
            and st.get("stub") == V31_FROZEN["n_stub"]
            and V31_CODEBOOK.size() == V31_FROZEN["codebook_rows"]):
        bad.append("layout_frozen_ok 标志与冻结常量比对不符")
    # compare directly against the frozen constants (any layout drift must blow up)
    for k in ("total_ops", "n_verified", "n_stub", "codebook_rows", "head_dim"):
        if value[k] != V31_FROZEN[k]:
            bad.append("冻结布局漂移: %s 实得 %r, 冻结 %r" % (k, value[k], V31_FROZEN[k]))
    static = audit_static_v31()
    if len(static) != value["static_problems"]:
        bad.append("静态违规计数不符")
    return {"ok": not bad, "resid": [float(len(bad))],
            "detail": "台账闭合 + 头维度口径 + 冻结布局对拍 + 两路计数一致 通过" if not bad
                      else "; ".join(bad)}


_reg(
    name="v31_census", band=BAND_INFRA, sub=4,
    in_schema=_CEN_IN, out_schema=_CEN_OUT,
    invoke=_invoke_census, verify=_verify_census,
    cost=("units", 5.0),
    errors=("SCHEMA_MISMATCH", "POSTCOND_FAIL"),
    effects="none (纯函数; 只读 V31 注册表/码本, 不改动任何状态)",
    doc=("V31 ledger: census along the three dimensions band/status/layer, plus totals / status / "
         "codebook rows / **head dimension** / selfcheck count / static violations / anchor-chain "
         "verdict. **Head-dimension convention**: the V31 model-side routing head = operators + 1 "
         "(END slot) = codebook rows (the END sentinel occupies the real codebook row 9999) -- "
         "this is the model-side head-size premise, hard-coded in this selfcheck to prevent "
         "drift. Independent verifier = recompute the counts **separately** from the codebook and "
         "from the registry, and cross-check ledger closure (verified + stub = total) against the "
         "frozen constants V31_FROZEN; the static-audit (audit_static_v31) violation count is also "
         "recomputed and compared."),
    selfchecks=(
        {"id": "census_band", "args": {"group_by": "band", "with_audit": True},
         # B7 fix: the original criterion ended with "and r.cert_independent", but v31_census
         # **issues no certificate** (cert_kind unregistered, cert_independent is always None)
         # => that conjunct was always false. The ledger is an **account**, not a mathematical
         # proposition; its independence is carried by _verify_census' "codebook row table vs
         # registry name list" two-path recomputation + the item-by-item comparison against the
         # frozen constants V31_FROZEN, and must not be faked with the certificate mechanism of a
         # mathematical proposition.
         "check": lambda r: (r.value["total_ops"] == 154 and r.value["n_verified"] == 146
                             and r.value["n_stub"] == 8 and r.value["n_legacy"] == 118
                             and r.value["codebook_rows"] == 155 and r.value["head_dim"] == 155
                             and r.value["layout_frozen_ok"] and r.value["static_problems"] == 0
                             and r.value["anchor_ok"],
                             "总量 %d = verified %d + stub %d; 存量 %d; 码本 %d 行; 头 %d"
                             % (r.value["total_ops"], r.value["n_verified"], r.value["n_stub"],
                                r.value["n_legacy"], r.value["codebook_rows"], r.value["head_dim"])),
         "expect": "冻结布局: 154 算子 (146 verified + 8 stub, 含 118 存量), 码本/头 155"},
        {"id": "census_status", "args": {"group_by": "status", "with_audit": False},
         "check": lambda r: (dict(r.value["counts"]).get("verified") == 146
                             and dict(r.value["counts"]).get("stub") == 8,
                             "状态 census: %r" % (r.value["counts"],)),
         "expect": "状态维 census"},
        {"id": "census_layer", "args": {"group_by": "layer", "with_audit": False},
         "check": lambda r: (dict(r.value["counts"]).get("0") == 10
                             and dict(r.value["counts"]).get("1") == 11
                             and dict(r.value["counts"]).get("2") == 8
                             and dict(r.value["counts"]).get("3") == 4
                             and dict(r.value["counts"]).get("-2") == 118,
                             "层维 census: %r" % (r.value["counts"],)),
         "expect": "层维: L0=10, L1=11, L2=8, L3=4, 存量=-2, infra=-1"},
        {"id": "census_bad_group_reject", "args": {"group_by": "color", "with_audit": False},
         "expect_err": "SCHEMA_MISMATCH", "expect": "未登记分组维被 enum 拦下"},
    ),
)

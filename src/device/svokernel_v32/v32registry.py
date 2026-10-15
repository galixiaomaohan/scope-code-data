# -*- coding: utf-8 -*-
"""V32 parallel registry -- reference-style re-registration of the full v31 table + skeleton
meta-operators + anchor-chain audit.

Three responsibilities (the same as what v31 did to the legacy package):
1. **V32OpDef**: adds a `skeleton` flag (is it a skeleton meta-operator) on top of the v31
   bookkeeping fields (status/layer/stub);
2. **V32Registry**: a separate registry instance (attached to V32_CODEBOOK). The 154 v31 operators
   are merged **by reference**: the **same** invoke/verify/selfchecks function objects are reused
   with sub = row − band given explicitly => row numbers unchanged row by row;
3. **Anchor-chain audit** (anchor_chain_audit_v32): really run and check item by item that
   "118 legacy / 154 v31 / v31 codebook / 28+26 certificate checkers / v31 row replication" were
   not touched. The audit reports only what actually ran.
"""
from __future__ import annotations

from dataclasses import dataclass

from svokernel.certify import CHECKERS
from svokernel.codebook import CODEBOOK as LEGACY_CODEBOOK
from svokernel.errors import ERROR_TABLE, Err, SvoError
from svokernel.registry import REGISTRY as LEGACY_REGISTRY
from svokernel.tool import validate_opdef

from svokernel_v31 import V31REGISTRY
from svokernel_v31.v31codebook import V31_CODEBOOK, V31_END_MARKER
from svokernel_v31.v31registry import (LEGACY_ANCHOR, LEGACY_CERT_KINDS_FROZEN, V31OpDef,
                                       V31Registry)

from .v32codebook import (N_SKEL_OPS, V32_CODEBOOK, V32_END_MARKER, V32_END_MARKER_ROW,
                          V32_HEAD_DIM_FROZEN, V32_SKEL_BAND, V32_SCHEMA, v32_band_of,
                          v32_op_index_map)

__all__ = [
    "V32OpDef", "V32Registry", "V32REGISTRY", "V32_ANCHOR", "join_v31",
    "JOINED_V31", "audit_static_v32", "anchor_chain_audit_v32", "SKEL_CERT_KINDS",
]

#: v32 frozen anchors (taken verbatim from real v31 runs; any mismatch is red).
V32_ANCHOR = {
    "legacy_n_ops": 118,
    "legacy_codebook_rows": 118,
    "legacy_codebook_checksum": "bd233ec24e84deec",
    "legacy_cert_kinds": 28,
    "v31_n_ops": 154,
    "v31_codebook_rows": 155,
    "v31_prefix_ops": 154,
    "v31_head_dim": 155,
    "v32_codebook_rows": 161,
    "v32_head_dim": 161,
    "v32_n_skel_ops": N_SKEL_OPS,
}

#: v32 skeleton certificate kinds (the independent checkers register in certcheck_skel.py).
SKEL_CERT_KINDS = ("case_exhaustive", "induction_scheme", "stack_machine",
                   "leaf_residue_complete", "contradiction_witness", "skeleton_qed")

#: The 26 new v31 certificate kinds (frozen: they must still be registered in a v32 process, and
#: still inside a v31 checker module).
V31_CERT_KINDS_FROZEN = (
    "lte_valuation", "hensel_root", "pratt_tree", "miller_witness", "factor_product",
    "order_exhaustive", "jacobi_euler", "zsi_primitive", "arith_conv_table",
    "padic_valuation", "cf_convergent", "pell_backsub", "form_reduce_chain",
    "qexp_mul", "hecke_tp_expand", "ec_chord_backsub", "torsion_walk",
    "pointcount_small", "qsqrtd_arith", "frey_invariants", "psi_interval",
    "gelfond_apply", "thm_apply_witness", "lemma_instance", "equiv_truth_table",
    "lemma_validity",
)


@dataclass
class V32OpDef(V31OpDef):
    """v31 bookkeeping fields + the skeleton flag. Legacy/v31 semantics unchanged."""

    skeleton: bool = False

    def five_tuple(self, handle: int) -> dict:
        d = super().five_tuple(handle)
        d["skeleton"] = self.skeleton
        return d


class V32Registry(V31Registry):
    """V32 registry: separate codebook + v31 band-name resolution + skeleton ledger."""

    def __init__(self) -> None:
        super().__init__()
        self.cb = V32_CODEBOOK
        self._tools = {}

    def band_census(self) -> dict:
        out: dict = {}
        for n in self._ops:
            b = v32_band_of(self._tools[n].handle)
            out[b] = out.get(b, 0) + 1
        return dict(sorted(out.items()))

    def skeleton_census(self) -> dict:
        out = {"skeleton": 0, "non_skeleton": 0}
        for op in self._ops.values():
            out["skeleton" if getattr(op, "skeleton", False) else "non_skeleton"] += 1
        return out

    def index_map(self) -> dict:
        return v32_op_index_map()


#: The single V32 process-level registry.
V32REGISTRY = V32Registry()


# ---------------------------------------------------------------------------
# reference-style re-registration of the full v31 table (zero copy, zero modification)
# ---------------------------------------------------------------------------

def join_v31() -> list:
    """Merge the 154 v31 operators into the v32 registry **by reference**; returns the merged
    operator names (ordered by row).

    Fail-safe: an empty v31 registry is always rejected -- "merge 0 operators without an error" is
    the most dangerous silent failure mode.
    """
    if len(V31REGISTRY) == 0:
        raise SvoError(Err.SCHEMA_MISMATCH, ctx="v32:join_v31",
                       detail="v31 注册表为空 (svokernel_v31 未装载) —— 拒绝并入 0 个算子")
    if len(V31REGISTRY) != V32_ANCHOR["v31_n_ops"]:
        raise SvoError(Err.SCHEMA_MISMATCH, ctx="v32:join_v31",
                       detail="v31 注册表 %d 算子, 冻结 %d —— 拒绝在漂移的表上转注册"
                              % (len(V31REGISTRY), V32_ANCHOR["v31_n_ops"]))
    joined: list = []
    for name in V31REGISTRY.names():
        if V32REGISTRY.has(name):
            joined.append(name)
            continue
        op = V31REGISTRY._ops[name]
        row = V31REGISTRY.tool(name).handle
        sub = row - op.band
        if sub < 1:
            raise SvoError(Err.SCHEMA_MISMATCH, ctx="v32:join_v31",
                           detail="v31 行 %d 落在段基址 (band=%d) —— 与约定不符, 拒绝盲目转注册"
                                  % (row, op.band))
        new = V32OpDef(
            name=op.name, band=op.band, sub=sub,
            in_schema=op.in_schema, out_schema=op.out_schema,
            invoke=op.invoke, verify=op.verify, cert_kind=op.cert_kind,
            cost=op.cost, errors=tuple(op.errors), effects=op.effects,
            selfchecks=tuple(op.selfchecks), doc=op.doc,
            requires_cert=op.requires_cert, precond=op.precond,
            stmt=op.stmt, thm=op.thm,
            status=op.status, layer=op.layer,
            stub_reason=getattr(op, "stub_reason", ""),
            external_dep=getattr(op, "external_dep", ""),
            planned_cert=getattr(op, "planned_cert", ""),
            skeleton=False,
        )
        V32REGISTRY.register(new)
        joined.append(name)
    return joined


JOINED_V31 = join_v31()

#: Re-check: after re-registration the v31 row numbers must agree row by row (never continue with
#: wrong row numbers).
for _n in V31REGISTRY.names():
    _want, _got = V31REGISTRY.tool(_n).handle, V32_CODEBOOK.row_of(_n)
    if _want != _got:
        raise SvoError(Err.SCHEMA_MISMATCH, ctx="v32:join_v31@verify",
                       detail="转注册行号漂移: %s v31 #%d vs v32 #%d" % (_n, _want, _got))

for _n in LEGACY_REGISTRY.names():
    if V32REGISTRY.has(_n):
        _want, _got = LEGACY_REGISTRY.handle_of(_n), V32_CODEBOOK.row_of(_n)
        if _want != _got:
            raise SvoError(Err.SCHEMA_MISMATCH, ctx="v32:join_v31@verify-legacy",
                           detail="存量行号漂移: %s 存量 #%d vs v32 #%d" % (_n, _want, _got))


def _ack_end_marker() -> int:
    """Register the END sentinel row (not an operator): makes the v32 head dimension equal the
    codebook row count. Idempotent."""
    h = V32_CODEBOOK.allocate(V32_END_MARKER, 9990, sub=9)
    if h.row != V32_END_MARKER_ROW:
        raise SvoError(Err.SCHEMA_MISMATCH, ctx="v32:end_marker",
                       detail="END 哨兵行漂移: 期望 #%d, 实得 #%d" % (V32_END_MARKER_ROW, h.row))
    return h.row


_ack_end_marker()


# ---------------------------------------------------------------------------
# static audit (v32 edition: mirrors v31's audit_static_v31, adds three skeleton criteria)
# ---------------------------------------------------------------------------

def audit_static_v32() -> list:
    """Return the problem list (empty = no static violation found). Every criterion is
    machine-checkable; no subjective judgement."""
    problems: list = []

    def bad(code: str, op: str, detail: str) -> None:
        problems.append({"code": code, "op": op, "detail": detail})

    seen_rows: dict = {}
    for name in V32REGISTRY.names():
        op = V32REGISTRY._ops[name]
        tool = V32REGISTRY.tool(name)
        row = tool.handle

        if row in seen_rows:
            bad("ROW_DUP", name, "行 %d 与 %s 冲突" % (row, seen_rows[row]))
        seen_rows[row] = name

        if op.verify is None and op.cert_kind is None:
            bad("NO_VERIFIER", name, "§7.3 没有验证器不许交")
        if not op.selfchecks:
            bad("NO_SELFCHECK", name, "§7 注册流程要求 self-check")
        if not op.errors:
            bad("NO_ERROR_ENUM", name, "§7.6 必须枚举失败模式")
        for e in op.errors:
            if e not in ERROR_TABLE:
                bad("ERR_OUT_OF_TABLE", name, "错误码 %r 不在 §15 表内" % (e,))
        if op.cost[0] != "units" or not isinstance(op.cost[1], (int, float)) or op.cost[1] < 0:
            bad("BAD_COST", name, "成本标注非法 %r" % (op.cost,))
        if not op.effects:
            bad("NO_EFFECTS", name, "§7.2 副作用须显式")
        if not op.doc:
            bad("NO_DOC", name, "五元组文档缺失")

        if op.cert_kind is not None:
            if not CHECKERS.has(op.cert_kind):
                bad("CERT_UNCHECKED", name, "证书类型 %r 无独立检查器" % (op.cert_kind,))
            else:
                cmod = CHECKERS.module_of(op.cert_kind)
                gmod = getattr(op.invoke, "__module__", "")
                if cmod == gmod:
                    bad("SELF_CERT", name,
                        "检查器 %s 与生成器同处模块 %s —— §17.9 禁自证" % (op.cert_kind, cmod))
                elif _checker_imports_generator(CHECKERS.fn(op.cert_kind), gmod):
                    bad("SELF_CERT_IMPORT", name,
                        "检查器模块 %s 直接 import 了生成器模块 %s" % (cmod, gmod))

        ids: set = set()
        for sc in op.selfchecks:
            for k in ("id", "args"):
                if k not in sc:
                    bad("SELFCHECK_SCHEMA", name, "自检项缺字段 %r: %r" % (k, sc))
            cid = sc.get("id", "?")
            if cid in ids:
                bad("SELFCHECK_DUP", name, "自检 id 重复: %s" % cid)
            ids.add(cid)

        if op.status not in ("verified", "stub"):
            bad("V32_BAD_STATUS", name, "status 必须为 verified|stub, 得 %r" % (op.status,))
        if op.status == "stub":
            if not op.stub_reason:
                bad("V32_STUB_NO_REASON", name, "stub 必须写明具体缺口")
            if not op.external_dep:
                bad("V32_STUB_NO_DEP", name, "stub 必须写明外部依赖")
            if op.cert_kind is not None:
                bad("V32_STUB_WITH_CERT", name, "stub 不许声明 cert_kind")
            if not any(sc.get("expect_err") for sc in op.selfchecks):
                bad("V32_STUB_NO_ABSENCE_CHECK", name, "stub 自检必须有一条 expect_err")
            if op.band != 9940:
                bad("V32_STUB_BAND", name, "stub 必须落在隔离段 9940 (现 %d)" % op.band)
        else:
            if op.layer not in (0, 1, 2, 3, -1, -2, 4):
                bad("V32_BAD_LAYER", name, "layer 必须 ∈ {0,1,2,3,-1,-2,4}, 得 %r" % (op.layer,))
            if not op.verify:
                bad("V32_VERIFIED_NO_VERIFY", name, "verified 算子必须有独立验证器函数")

        # --- three V32 additions (skeleton-operator discipline) ---
        if getattr(op, "skeleton", False):
            if not (V32_SKEL_BAND <= row < V32_SKEL_BAND + 100):
                bad("V32_SKEL_BAND", name, "骨架算子必须落在段 %d (现行 %d)" % (V32_SKEL_BAND, row))
            if op.cert_kind not in SKEL_CERT_KINDS:
                bad("V32_SKEL_CERT", name,
                    "骨架算子 cert_kind=%r 不在骨架证书清单内" % (op.cert_kind,))
            if op.layer != 4:
                bad("V32_SKEL_LAYER", name, "骨架算子 layer 必须为 4 (结构层), 得 %r" % (op.layer,))
            if op.effects != "none (纯函数)":
                bad("V32_SKEL_EFFECTS", name, "骨架算子必须无副作用 (栈是值, 不是全局状态)")
        else:
            if op.layer == 4:
                bad("V32_NONSKEL_LAYER4", name, "非骨架算子不许使用 layer=4")

    return problems


def _checker_imports_generator(fn, generator_module: str) -> bool:
    """Whether the checker namespace directly holds the generator module object (i.e. imported it)
    -- direct dependencies only."""
    if fn is None or not generator_module:
        return False
    import types

    for v in getattr(fn, "__globals__", {}).values():
        if isinstance(v, types.ModuleType) and getattr(v, "__name__", "") == generator_module:
            return True
    return False


# ---------------------------------------------------------------------------
# anchor-chain audit (v32 edition): was anything in the 118 legacy + 154 v31 operators touched
# ---------------------------------------------------------------------------

def anchor_chain_audit_v32() -> dict:
    """Verify item by item that the legacy/v31 anchors were not touched by v32; returns
    {ok, checks, mismatches, ...}."""
    checks: list = []
    mismatches: list = []

    def ck(name: str, ok: bool, detail: str) -> None:
        checks.append({"check": name, "ok": bool(ok), "detail": detail})
        if not ok:
            mismatches.append({"check": name, "detail": detail})

    ck("legacy_registry_size", len(LEGACY_REGISTRY) == V32_ANCHOR["legacy_n_ops"],
       "存量注册表 %d 算子 (冻结 %d)" % (len(LEGACY_REGISTRY), V32_ANCHOR["legacy_n_ops"]))
    ck("legacy_codebook_checksum",
       LEGACY_CODEBOOK.checksum() == V32_ANCHOR["legacy_codebook_checksum"],
       "存量码本 checksum=%s (冻结 %s)" % (LEGACY_CODEBOOK.checksum(),
                                           V32_ANCHOR["legacy_codebook_checksum"]))
    ck("legacy_band_census", LEGACY_REGISTRY.band_census() == LEGACY_ANCHOR["bands"],
       "存量段分布 %r" % (LEGACY_REGISTRY.band_census(),))

    missing_kinds = [k for k in LEGACY_CERT_KINDS_FROZEN if not CHECKERS.has(k)]
    ck("legacy_cert_kinds_present", not missing_kinds,
       "缺失 %r" % (missing_kinds,) if missing_kinds else "28 个存量检查器全在册")

    missing31 = [k for k in V31_CERT_KINDS_FROZEN if not CHECKERS.has(k)]
    ck("v31_cert_kinds_present", not missing31,
       "缺失 %r" % (missing31,) if missing31 else "26 个 v31 检查器全在册")

    bad31 = [k for k in V31_CERT_KINDS_FROZEN
             if CHECKERS.has(k) and not CHECKERS.module_of(k).startswith("svokernel_v31")]
    ck("v31_cert_kind_modules", not bad31,
       "归属不符: %r" % (bad31,) if bad31 else "26 个 v31 检查器 module_of 均在 svokernel_v31.*")

    skel_bad = [k for k in SKEL_CERT_KINDS
                if CHECKERS.has(k) and CHECKERS.module_of(k) != "svokernel_v32.certcheck_skel"]
    ck("skel_cert_kind_modules", not skel_bad,
       "归属不符: %r" % (skel_bad,) if skel_bad else "6 个骨架检查器归属 svokernel_v32.certcheck_skel")

    n_v31 = len(V31REGISTRY)
    ck("v31_registry_size", n_v31 == V32_ANCHOR["v31_n_ops"],
       "v31 注册表 %d 算子 (冻结 %d)" % (n_v31, V32_ANCHOR["v31_n_ops"]))
    ck("v31_codebook_rows", V31_CODEBOOK.size() == V32_ANCHOR["v31_codebook_rows"],
       "v31 码本 %d 行 (冻结 %d)" % (V31_CODEBOOK.size(), V32_ANCHOR["v31_codebook_rows"]))

    # v31 rows replicated row by row in the v32 codebook (END sentinel included)
    row_bad = []
    for name in V31REGISTRY.names():
        if not V32_CODEBOOK.has(name):
            row_bad.append((name, "缺"))
            continue
        want = V31REGISTRY.tool(name).handle
        got = V32_CODEBOOK.row_of(name)
        if want != got:
            row_bad.append((name, want, got))
    ck("v31_rows_replicated", not row_bad,
       "行号不符 %r" % (row_bad[:5],) if row_bad else "v31 154 行逐行复刻")

    ck("v32_codebook_rows", V32_CODEBOOK.size() == V32_ANCHOR["v32_codebook_rows"],
       "v32 码本 %d 行 (冻结 %d = 155 v31 行 + 6 骨架行)" % (
           V32_CODEBOOK.size(), V32_ANCHOR["v32_codebook_rows"]))

    idx = v32_op_index_map()
    idx_bad = [(n, v) for n, v in idx.items() if n != V32_END_MARKER and n.startswith("skel_")
               and not (155 <= v <= 160)]
    end_ok = idx.get(V32_END_MARKER) == 154
    ck("v32_skel_indices", not idx_bad and end_ok,
       "骨架索引 %r (须 155..160); END=%r (须 154)" % (
           {n: idx[n] for n in idx if n.startswith("skel_")}, idx.get(V32_END_MARKER)))

    v31_head_ok = all(idx[n] == i for i, n in enumerate(
        [x for x, _ in sorted(V31_CODEBOOK.rows().items(), key=lambda kv: kv[1])]))
    ck("v31_index_prefix", v31_head_ok, "v31 行序在 v32 索引空间里是逐位不变的前缀")

    st = V32REGISTRY.status_census()
    ck("v32_status_census", st.get("verified", 0) + st.get("stub", 0) == len(V32REGISTRY),
       "v32 状态分布 %r" % (st,))

    return {
        "ok": not mismatches,
        "checks": checks,
        "mismatches": mismatches,
        "n_v31_joined": len(JOINED_V31),
        "n_ops": len(V32REGISTRY),
        "codebook_rows": V32_CODEBOOK.size(),
        "schema": V32_SCHEMA,
        "skel_cert_kinds_added": sorted(k for k in SKEL_CERT_KINDS if CHECKERS.has(k)),
    }

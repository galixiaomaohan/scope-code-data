# -*- coding: utf-8 -*-
"""V31 parallel registry + anchor-chain audit (zero-modification reference to the legacy set).

Three responsibilities:
1. **V31OpDef**: adds three bookkeeping fields (status/layer/stub documentation) on top of
   the legacy OpDef; legacy field semantics are untouched word for word -- so the TypedTool
   call pipeline (schema gate -> precond -> budget -> invoke -> output schema -> independent
   verifier -> certificate re-check -> accept) is reused as is; V31 has no second call path.
2. **V31Registry**: a separate registry instance (attached to V31_CODEBOOK). The legacy
   REGISTRY is only **read**; the 118 operators are merged by re-registration: same
   invoke/verify function objects + same row numbers, no code duplication, no modification
   of the legacy registry object.
3. **Anchor-chain audit** (anchor_chain_audit): verifies item by item that the legacy
   package was really not touched -- registry size / codebook checksum / names and modules
   of the 28 certificate checkers / row-number replication for the legacy operators.
   The audit reports only what actually ran; it does not claim "audit passed".
"""
from __future__ import annotations

from dataclasses import dataclass

from svokernel.certify import CHECKERS
from svokernel.codebook import CODEBOOK as LEGACY_CODEBOOK
from svokernel.errors import ERROR_TABLE, Err, SvoError
from svokernel.registry import REGISTRY as LEGACY_REGISTRY
from svokernel.registry import Registry
from svokernel.tool import OpDef, TypedTool, validate_opdef

from .v31codebook import V31_CODEBOOK, v31_band_of

__all__ = [
    "V31OpDef", "V31Registry", "V31REGISTRY", "LEGACY_ANCHOR",
    "LEGACY_CERT_KINDS_FROZEN", "audit_static_v31", "anchor_chain_audit",
]

#: Anchor constants (frozen legacy facts, taken verbatim from real runs; any mismatch is a red alert).
LEGACY_ANCHOR = {
    "n_ops": 118,
    "codebook_rows": 118,
    "codebook_checksum": "bd233ec24e84deec",
    "n_cert_kinds": 28,
    "bands": {
        "agent": 10, "code": 19, "core_cert": 5, "infra": 1, "logic": 18,
        "math_basic": 12, "math_strategy": 8, "math_theorem": 35, "se": 10,
    },
}

#: The 28 legacy certificate checkers (names frozen); each one is audited for presence and module.
LEGACY_CERT_KINDS_FROZEN = (
    "axiom_table", "bmc_bounded_ok", "bmc_counterexample", "cex_witness",
    "derivative_inverse", "divergence_witness", "drat_proof", "equivalence_pair",
    "identity_expand", "interval_contains", "interval_set_contains", "linear_backsub",
    "lp_dual", "matrix_identity", "mc_counterexample", "mc_exhaustive",
    "mod_congruence", "mutation_report", "nd_proof", "poly_backsub", "proof_kernel",
    "replay", "sat_assignment", "smt_prooflog", "test_vector", "tlc_exhaustive",
    "tlc_violation", "unify_witness",
)

#: Checker module ownership (only two legacy modules are not certify; used to verify that
#: the independence metadata was not rewritten).
LEGACY_CHECKER_MODULES = {"cex_witness": "svokernel.ndcore", "nd_proof": "svokernel.ndcore"}


@dataclass
class V31OpDef(OpDef):
    """Legacy OpDef + V31 bookkeeping fields (added fields; legacy semantics untouched).

    status     : "verified" | "stub"  -- a stub never masquerades as verified (hard honesty rule);
    layer      : 0/1/2/3 = the four operator layers; -1 = infrastructure (infra);
    stub_reason: the **concrete** gap of a stub (which machine / which formalization is
                 missing); non-empty forces status=stub;
    external_dep: external formalization/literature a stub depends on (for the honesty table
                 in the DESIGN document);
    planned_cert: certificate kind a stub plans to use (no independent checker yet => no
                 cert_kind is assigned, so it cannot look "certified");
    """

    status: str = "verified"
    layer: int = -1
    stub_reason: str = ""
    external_dep: str = ""
    planned_cert: str = ""

    def five_tuple(self, handle: int) -> dict:
        d = super().five_tuple(handle)
        d["status"] = self.status
        d["layer"] = self.layer
        if self.stub_reason:
            d["stub_reason"] = self.stub_reason
        if self.external_dep:
            d["external_dep"] = self.external_dep
        if self.planned_cert:
            d["planned_cert"] = self.planned_cert
        return d


class V31Registry(Registry):
    """V31 registry: separate codebook + band names via the V31 band table + a status/layer ledger."""

    def __init__(self) -> None:
        super().__init__(V31_CODEBOOK)

    def band_census(self) -> dict:
        out: dict = {}
        for n in self._ops:
            b = v31_band_of(self._tools[n].handle)
            out[b] = out.get(b, 0) + 1
        return dict(sorted(out.items()))

    def layer_census(self) -> dict:
        out: dict = {}
        for op in self._ops.values():
            out[op.layer] = out.get(op.layer, 0) + 1
        return dict(sorted(out.items()))

    def status_census(self) -> dict:
        out = {"verified": 0, "stub": 0}
        for op in self._ops.values():
            out[op.status] = out.get(op.status, 0) + 1
        return out

    def ledger(self) -> list:
        """Full operator ledger ordered by row (name/row/band/layer/status/verifier/certificate)."""
        rows = []
        for n in self.names():
            op = self._ops[n]
            t = self._tools[n]
            rows.append({
                "op": n, "handle": t.handle, "band": v31_band_of(t.handle),
                "layer": op.layer, "status": op.status,
                "verifier": ("cert:" + op.cert_kind) if op.cert_kind else (
                    op.verify.__name__ if op.verify else "无"),
                "cert_kind": op.cert_kind,
                "checks": len(op.selfchecks),
                "legacy": n in LEGACY_REGISTRY._ops,
                "stub_reason": op.stub_reason,
                "external_dep": op.external_dep,
            })
        return rows

    def checks_by_status(self) -> dict:
        out = {"verified": 0, "stub": 0}
        for op in self._ops.values():
            out[op.status] = out.get(op.status, 0) + len(op.selfchecks)
        return out


#: The single V31 process-level registry.
V31REGISTRY = V31Registry()


# ---------------------------------------------------------------------------
# Static audit (V31 edition; criteria copied from svokernel.selfcheck.audit_static
# + three V31 additions)
# ---------------------------------------------------------------------------


def audit_static_v31() -> list:
    """Return the problem list (empty = no static violation found). Every criterion is
    machine-checkable; no subjective judgement."""
    problems: list = []

    def bad(code: str, op: str, detail: str) -> None:
        problems.append({"code": code, "op": op, "detail": detail})

    seen_rows: dict = {}
    for name in V31REGISTRY.names():
        op = V31REGISTRY._ops[name]
        tool = V31REGISTRY.tool(name)
        row = tool.handle

        if row in seen_rows:
            bad("ROW_DUP", name, "行 %d 与 %s 冲突" % (row, seen_rows[row]))
        seen_rows[row] = name

        # §7 six admission rules
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

        # a certificate kind must have an independent checker, and checker module != generator
        # module (§17.9 forbids self-certification)
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

        # self-check structure
        ids: set = set()
        for sc in op.selfchecks:
            for k in ("id", "args"):
                if k not in sc:
                    bad("SELFCHECK_SCHEMA", name, "自检项缺字段 %r: %r" % (k, sc))
            cid = sc.get("id", "?")
            if cid in ids:
                bad("SELFCHECK_DUP", name, "自检 id 重复: %s" % cid)
            ids.add(cid)

        # --- V31 additional criteria ---
        if op.status not in ("verified", "stub"):
            bad("V31_BAD_STATUS", name, "status 必须为 verified|stub, 得 %r" % (op.status,))
        if op.status == "stub":
            # three hard stub constraints: state the gap, claim no certificate, and let a
            # self-check assert the explicit absence
            if not op.stub_reason:
                bad("V31_STUB_NO_REASON", name, "stub 必须写明具体缺口 (禁空泛标注)")
            if not op.external_dep:
                bad("V31_STUB_NO_DEP", name, "stub 必须写明外部依赖")
            if op.cert_kind is not None:
                bad("V31_STUB_WITH_CERT", name, "stub 不许声明 cert_kind (免得看起来有证书)")
            if not any(sc.get("expect_err") for sc in op.selfchecks):
                bad("V31_STUB_NO_ABSENCE_CHECK", name,
                    "stub 自检必须有一条断言显式缺席 (expect_err), 禁把 stub 当 verified 混过")
        else:
            # layer semantics: 0/1/2/3 = the four V31 layers; -1 = V31 infra; -2 = legacy
            # re-registered reference
            if op.layer not in (0, 1, 2, 3, -1, -2):
                bad("V31_BAD_LAYER", name, "layer 必须 ∈ {0,1,2,3,-1,-2}, 得 %r" % (op.layer,))
            if not op.verify:
                bad("V31_VERIFIED_NO_VERIFY", name, "verified 算子必须有独立验证器函数")
        if op.status == "stub" and op.band != 9940:
            bad("V31_STUB_BAND", name, "stub 必须落在隔离段 9940 (现 %d)" % op.band)

    return problems


def _checker_imports_generator(fn, generator_module: str) -> bool:
    """Whether the checker namespace directly holds the generator module object (i.e. imported
    it) -- direct dependencies only."""
    if fn is None or not generator_module:
        return False
    import types

    for v in getattr(fn, "__globals__", {}).values():
        if isinstance(v, types.ModuleType) and getattr(v, "__name__", "") == generator_module:
            return True
    return False


# ---------------------------------------------------------------------------
# Anchor-chain audit: was the legacy package really left untouched (compare by running,
# not by declaration)
# ---------------------------------------------------------------------------


def anchor_chain_audit() -> dict:
    """Verify item by item that no V31 code touched the legacy anchors; returns
    {ok, checks:[...], mismatches:[...]}.

    Criteria (all machine-traceable):
      1. the legacy REGISTRY still holds exactly 118 operators, and its name set equals the
         legacy subset inside V31;
      2. legacy CODEBOOK size/checksum unchanged (the checksum is the frozen fingerprint);
      3. legacy band_census equals the frozen constant band by band;
      4. the 28 legacy certificate checkers are still registered, with module_of matching the
         frozen ownership (V31 kinds are only appended);
      5. the 118 legacy operator names have **identical row numbers** in the V31 codebook
         (re-registration does not move bands).
    """
    checks: list = []
    mismatches: list = []

    def ck(name: str, ok: bool, detail: str) -> None:
        checks.append({"check": name, "ok": bool(ok), "detail": detail})
        if not ok:
            mismatches.append({"check": name, "detail": detail})

    n_legacy = len(LEGACY_REGISTRY)
    ck("legacy_registry_size", n_legacy == LEGACY_ANCHOR["n_ops"],
       "存量注册表 %d 算子 (冻结 %d)" % (n_legacy, LEGACY_ANCHOR["n_ops"]))

    ck("legacy_codebook_size", LEGACY_CODEBOOK.size() == LEGACY_ANCHOR["codebook_rows"],
       "存量码本 %d 行 (冻结 %d)" % (LEGACY_CODEBOOK.size(), LEGACY_ANCHOR["codebook_rows"]))

    ck("legacy_codebook_checksum", LEGACY_CODEBOOK.checksum() == LEGACY_ANCHOR["codebook_checksum"],
       "存量码本 checksum=%s (冻结 %s)" % (LEGACY_CODEBOOK.checksum(), LEGACY_ANCHOR["codebook_checksum"]))

    ck("legacy_band_census", LEGACY_REGISTRY.band_census() == LEGACY_ANCHOR["bands"],
       "存量段分布 %r" % (LEGACY_REGISTRY.band_census(),))

    missing_kinds = [k for k in LEGACY_CERT_KINDS_FROZEN if not CHECKERS.has(k)]
    ck("legacy_cert_kinds_present", not missing_kinds,
       "存量 28 检查器缺失: %r" % (missing_kinds,) if missing_kinds else "28 个存量检查器全在册")

    mod_bad = []
    for k in LEGACY_CERT_KINDS_FROZEN:
        if not CHECKERS.has(k):
            continue
        want = LEGACY_CHECKER_MODULES.get(k, "svokernel.certify")
        got = CHECKERS.module_of(k)
        if got != want:
            mod_bad.append((k, want, got))
    ck("legacy_cert_kind_modules", not mod_bad,
       "module_of 与冻结归属不符: %r" % (mod_bad,) if mod_bad else "归属未变 (27×svokernel.certify + 2×svokernel.ndcore 口径)")

    # row replication of the re-registered legacy operators (non-empty only after legacy_join)
    n_join = 0
    row_bad = []
    for name in LEGACY_REGISTRY.names():
        if not V31_CODEBOOK.has(name):
            continue
        n_join += 1
        want = LEGACY_REGISTRY.handle_of(name)
        got = V31_CODEBOOK.row_of(name)
        if want != got:
            row_bad.append((name, want, got))
    ck("legacy_rows_replicated", not row_bad and (n_join == 0 or n_join == n_legacy),
       "已转注册 %d 个; 行号不符 %r" % (n_join, row_bad[:5]))

    return {
        "ok": not mismatches,
        "checks": checks,
        "mismatches": mismatches,
        "n_legacy_joined": n_join,
        "v31_cert_kinds_added": sorted(k for k in CHECKERS.kinds() if k not in LEGACY_CERT_KINDS_FROZEN),
    }

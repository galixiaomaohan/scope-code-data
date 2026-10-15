"""Operator registry: codebook row allocation + five-tuple table + selfcheck execution (§7 registration flow).

Registration enforces the six §7 admission rules at runtime:
  1. typed IO — schema declarations are validated;
  2. explicit side effects — the effects field is mandatory;
  3. verifiable — at least one of verify or cert_kind, otherwise rejected (NO_CERTIFICATE);
  4. enumerable small combinations — compose_budget declares the ≤3-step composition enumeration cap (0 = not participating);
  5. cost annotation — cost=(unit, value);
  6. fail-safe — errors must be a subset of the §15 table.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .audit import AuditTrace, BudgetLedger
from .codebook import CODEBOOK, Codebook
from .errors import Err, SvoError
from .tool import CallCtl, OpDef, ToolResult, TypedTool, validate_opdef


@dataclass
class SelfCheckResult:
    op: str
    handle: int
    cid: str
    passed: bool
    detail: str
    err: str | None
    resid: float
    elapsed_ms: float

    def line(self) -> str:
        mark = "[check-PASS]" if self.passed else "[check-FAIL]"
        return (
            f"{mark} {self.op}#{self.handle} :: {self.cid} :: "
            f"err={self.err} resid={self.resid:.3e} t={self.elapsed_ms:.2f}ms :: {self.detail}"
        )


class Registry:
    def __init__(self, codebook: Codebook | None = None) -> None:
        self.cb = codebook or CODEBOOK
        self._ops: dict[str, OpDef] = {}
        self._tools: dict[str, TypedTool] = {}

    # -- registration -------------------------------------------------------
    def register(self, op: OpDef) -> int:
        if op.name in self._ops:
            raise SvoError(Err.SCHEMA_MISMATCH, ctx=f"op:{op.name}@register", detail="重复注册")
        validate_opdef(op)
        handle = self.cb.allocate(op.name, op.band, op.sub).row
        self._ops[op.name] = op
        self._tools[op.name] = TypedTool(op, handle)
        return handle

    def register_all(self, ops: "list[OpDef]") -> "list[int]":
        return [self.register(o) for o in ops]

    # -- lookup -------------------------------------------------------------
    def names(self) -> list[str]:
        return sorted(self._ops, key=lambda n: self._tools[n].handle)

    def has(self, name: str) -> bool:
        return name in self._ops

    def tool(self, name_or_handle: "str | int") -> TypedTool:
        if isinstance(name_or_handle, int):
            for t in self._tools.values():
                if t.handle == name_or_handle:
                    return t
            raise SvoError(Err.SCHEMA_MISMATCH, ctx="registry", detail=f"行号未登记: {name_or_handle}")
        if name_or_handle not in self._tools:
            raise SvoError(Err.SCHEMA_MISMATCH, ctx="registry", detail=f"算子未登记: {name_or_handle!r}")
        return self._tools[name_or_handle]

    def handle_of(self, name: str) -> int:
        return self.tool(name).handle

    def __len__(self) -> int:
        return len(self._ops)

    # -- tables and catalogs ------------------------------------------------
    def five_tuple_table(self) -> list[dict]:
        """§20 deliverable: per-operator five-tuple table."""
        return [self._ops[n].five_tuple(self._tools[n].handle) for n in self.names()]

    def band_census(self) -> dict[str, int]:
        from .codebook import band_of

        out: dict[str, int] = {}
        for n in self._ops:
            b = band_of(self._tools[n].handle)
            out[b] = out.get(b, 0) + 1
        return dict(sorted(out.items()))

    # -- selfcheck ----------------------------------------------------------
    def run_selfchecks(self, only: "str | None" = None, trace: AuditTrace | None = None) -> list[SelfCheckResult]:
        """Actually run every selfcheck item of every operator (skipping and faking are both forbidden).

        only supports prefix filtering: "math_*" / a single operator name.
        """
        import time

        out: list[SelfCheckResult] = []
        if only is None:
            names = self.names()
        else:
            pref = only.rstrip("*")
            names = [n for n in self.names() if n == only or n.startswith(pref)]
        for n in names:
            tool = self._tools[n]
            for sc in self._ops[n].selfchecks:
                trace_i = trace or AuditTrace(f"selfcheck:{n}")
                budget = BudgetLedger(total_units=1e9)
                ctl = CallCtl(trace=trace_i, budget=budget, timeout_s=sc.get("timeout_s", 60.0), seed=int(sc.get("seed", 0)))
                t0 = time.perf_counter()
                r = tool.call(sc["args"], ctl)
                dt = (time.perf_counter() - t0) * 1000.0
                passed, detail = self._judge(sc, r)
                out.append(
                    SelfCheckResult(
                        op=n, handle=tool.handle, cid=sc["id"], passed=passed,
                        detail=detail, err=r.err, resid=r.resid_scalar, elapsed_ms=dt,
                    )
                )
        return out

    @staticmethod
    def _judge(sc: dict, r: ToolResult) -> tuple[bool, str]:
        expect_err = sc.get("expect_err")
        if expect_err is not None:
            ok = (not r.ok) and r.err == expect_err
            return ok, f"期望错误码 {expect_err}, 实得 {r.err}" + (f" ({r.notes})" if r.notes else "")
        if not r.ok:
            return False, f"意外失败 {r.err}: {r.notes}"
        chk = sc.get("check")
        if chk is not None:
            ok, detail = chk(r)
            return bool(ok), detail
        rmax = sc.get("resid_max")
        if rmax is not None and not (r.resid_scalar <= float(rmax)):
            return False, f"残差 {r.resid_scalar:.3e} > 上限 {float(rmax):.3e}"
        return True, sc.get("expect", "通过默认判据 (ok + 可选残差上限)")

    def selfcheck_summary(self, results: "list[SelfCheckResult]") -> dict:
        n_pass = sum(1 for r in results if r.passed)
        return {
            "ops": len({r.op for r in results}),
            "checks": len(results),
            "passed": n_pass,
            "failed": len(results) - n_pass,
            "green": n_pass == len(results),
        }


#: Process-wide unique registry (operator modules self-register on import).
REGISTRY = Registry()


def op_registered(name: str, band: int, **kw) -> Callable[[Callable], Callable]:
    """Decorator: register an operator in one step.

    Usage:
        @op_registered("solve_poly", 9100, sub=1, in_schema=..., out_schema=...,
                       invoke=_inv, verify=_ver, cert_kind="poly_backsub",
                       errors=("DEG_UNSUPPORTED","DISC_NEG"), selfchecks=(...))
        def _noop(): ...
    """

    def deco(fn: Callable) -> Callable:
        op = OpDef(name=name, band=band, invoke=kw["invoke"], in_schema=kw["in_schema"],
                   out_schema=kw["out_schema"], verify=kw.get("verify"), cert_kind=kw.get("cert_kind"),
                   cost=kw.get("cost", ("units", 1.0)), errors=tuple(kw.get("errors", ())),
                   effects=kw.get("effects", "none (纯函数)"), selfchecks=tuple(kw.get("selfchecks", ())),
                   doc=kw.get("doc", ""), requires_cert=kw.get("requires_cert", False),
                   precond=kw.get("precond"), sub=kw.get("sub"),
                   stmt=kw.get("stmt", ""), thm=kw.get("thm", ""))
        REGISTRY.register(op)
        return fn

    return deco

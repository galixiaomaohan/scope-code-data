"""TypedTool base class (§11.1 implemented verbatim) + operator invocation pipeline (§11.2).

The pipeline's six gates (all recorded in the audit trace; failure returns a structured result, never silent):
    input schema check → precond gate → budget/deadline gate → invoke (out-of-table exceptions
    collapsed) → output schema check → independent verifier (postcond) → independent certificate
    re-verification → accept

Fail-safe (§7.6): `call()` **never raises**; it always returns ToolResult(err=<enumerated code>).
Callers that need exception semantics use `call_or_raise()`. This is not a style choice — the
navigation layer must backtrack across candidates by error code, and exceptions would turn the
backtracking path into a waterfall of try/except.

Deadline honesty statement: in-process preemption is impossible. `CallCtl.check_time()` is a
**cooperative** deadline — operators must call it explicitly inside long loops (every loop-style
operator in this package is wired in). Subprocess sandboxes (exec_sandbox / lean / drat) use real
SIGKILL-grade timeouts and are not subject to this limit.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from .audit import AuditTrace, BudgetLedger
from .certify import Certificate, verify_certificate
from .codebook import CODEBOOK, Handle, resid_summary, resid_vec
from .errors import Err, ERROR_TABLE, SvoError, guard
from .schema import check, schema_str

UNITS = "units"
"""Cost unit: abstract compute unit (1 unit ≈ one medium closed-form operation)."""


@dataclass
class CallCtl:
    """Per-call control plane: audit trace + budget + cooperative deadline + seed."""

    trace: AuditTrace
    budget: BudgetLedger | None = None
    timeout_s: float | None = None
    seed: int = 0
    _t_start: float = field(default_factory=time.perf_counter)

    def check_time(self, ctx: str) -> None:
        if self.timeout_s is None:
            return
        if time.perf_counter() - self._t_start > self.timeout_s:
            raise SvoError(
                Err.TIMEOUT,
                ctx=ctx,
                detail=f"协作式时限 {self.timeout_s}s 超 (已用 {time.perf_counter() - self._t_start:.3f}s)",
            )

    def charge(self, op: str, units: float) -> None:
        if self.budget is not None:
            self.budget.charge(op, units)


@dataclass
class ToolResult:
    ok: bool
    op: str
    handle: int
    value: Any = None
    resid: tuple[float, ...] = ()
    resid_scalar: float = float("nan")
    cert: Certificate | None = None
    cert_independent: bool | None = None
    err: str | None = None
    ctx: str = ""
    cost_units: float = 0.0
    elapsed_ms: float = 0.0
    notes: str = ""
    stages: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "op": self.op,
            "handle": self.handle,
            "value": self.value,
            "resid": list(self.resid),
            "resid_scalar": self.resid_scalar,
            "cert": None if self.cert is None else self.cert.kind,
            "cert_independent": self.cert_independent,
            "err": self.err,
            "ctx": self.ctx,
            "cost_units": self.cost_units,
            "elapsed_ms": round(self.elapsed_ms, 4),
            "notes": self.notes,
            "stages": list(self.stages),
        }


@dataclass
class OpDef:
    """Operator definition = five-tuple documentation (§7) + codebook row + implementation + independent verifier + selfchecks.

    Five-tuple: (input type, output type, verifier, cost, failure modes)
    Additional registrations (§8.2 theorem registry requires "statement|type|condition|verifier|code"):
      stmt — the statement (one-sentence form of the proposition);
      thm  — the theorem code (e.g. "T-A1"); conditions are hard-checked item by item via precond,
             never replaced by a textual declaration.
    """

    name: str
    band: int
    in_schema: dict
    out_schema: dict
    invoke: Callable[[dict, CallCtl], dict]
    verify: Callable[[dict, Any], dict] | None = None
    cert_kind: str | None = None
    cost: tuple[str, float] = (UNITS, 1.0)
    errors: tuple[str, ...] = ()
    effects: str = "none (纯函数)"
    selfchecks: tuple[dict, ...] = ()
    doc: str = ""
    requires_cert: bool = False
    precond: Callable[[dict], "tuple[bool, str]"] | None = None
    sub: int | None = None
    stmt: str = ""
    thm: str = ""

    def five_tuple(self, handle: int) -> dict:
        return {
            "op": self.name,
            "handle": handle,
            "band": self.band,
            "io": f"{schema_str(self.in_schema)} → {schema_str(self.out_schema)}",
            "verifier": ("cert:" + self.cert_kind) if self.cert_kind else (self.verify.__name__ if self.verify else "无"),
            "cost": f"{self.cost[1]} {self.cost[0]}",
            "errors": list(self.errors),
            "effects": self.effects,
            "thm": self.thm,
            "stmt": self.stmt,
        }


class TypedTool:
    """§11.1 base class: name / IO schemas / pre-post conditions / cost / error enum / call()."""

    def __init__(self, op: OpDef, handle: int) -> None:
        self.op = op
        self.name: str = op.name
        self.handle: int = handle
        self.input_schema: dict = op.in_schema
        self.output_schema: dict = op.out_schema
        self.cost: tuple[str, float] = op.cost
        self.error_enum: list[str] = list(op.errors)

        def precond(args: dict) -> bool:
            ok, _ = check(self.input_schema, args)
            if not ok:
                return False
            if self.op.precond is not None:
                ok2, _ = self.op.precond(args)
                return ok2
            return True

        def postcond(args: dict, out: Any) -> bool:
            """(input, output) → bool: output passes schema + independent verifier passes (§7.3)."""
            ok, _ = check(self.output_schema, out)
            if not ok:
                return False
            if self.op.verify is None:
                return True  # certificate-only operators: certificate re-verification happens separately in the pipeline
            r = self.op.verify(args, out)
            return bool(r.get("ok", False))

        self.precond = precond
        self.postcond = postcond

    # -- three VMP steps (§4) ----------------------------------------------
    def handle_of(self, name: str | None = None) -> Handle:
        return Handle(self.handle, name or self.name)

    def latent(self) -> tuple[float, ...]:
        """handle → 64-dim frozen embedding (vector-membrane entry)."""
        return CODEBOOK.latent(self.handle)

    # -- invocation ---------------------------------------------------------
    def call(self, args: dict, ctl: CallCtl | None = None) -> ToolResult:
        trace = ctl.trace if ctl else AuditTrace(f"tool:{self.name}")
        ctl = ctl or CallCtl(trace=trace)
        ctx = f"op:{self.name}#{self.handle}"
        t0 = time.perf_counter()
        stages: list[str] = []

        # gate 1: input schema + precond
        stages.append("precond")
        ok, msg = check(self.input_schema, args)
        if not ok:
            trace.add(self.name, self.handle, "precond", False, err=Err.SCHEMA_MISMATCH.value, detail=msg)
            return self._fail(Err.SCHEMA_MISMATCH, ctx, msg, stages, t0)
        if self.op.precond is not None:
            pok, pmsg = self.op.precond(args)
            if not pok:
                trace.add(self.name, self.handle, "precond", False, err=Err.PRECOND_FAIL.value, detail=pmsg)
                return self._fail(Err.PRECOND_FAIL, ctx, pmsg, stages, t0)
        trace.add(self.name, self.handle, "precond", True, detail="schema+precond 通过")

        # gate 2: budget / deadline
        stages.append("budget")
        units = float(self.cost[1])
        try:
            ctl.check_time(ctx)
            ctl.charge(self.name, units)
        except SvoError as e:
            trace.add(self.name, self.handle, "budget", False, err=e.code.value, detail=e.detail)
            return self._fail(e.code, ctx, e.detail, stages, t0)

        # gate 3: invoke (out-of-table exceptions collapsed)
        stages.append("invoke")
        try:
            with guard(f"{ctx}@invoke"):
                raw = self.op.invoke(args, ctl)
        except SvoError as e:
            trace.add(self.name, self.handle, "invoke", False, err=e.code.value, detail=e.detail)
            return self._fail(e.code, ctx, e.detail, stages, t0, units=units)
        if not isinstance(raw, dict):
            trace.add(self.name, self.handle, "invoke", False, err=Err.POSTCOND_FAIL.value, detail="invoke 未返回 dict")
            return self._fail(Err.POSTCOND_FAIL, ctx, "invoke 未返回 dict (契约违反)", stages, t0, units=units)

        value = raw.get("value")
        trace.add(self.name, self.handle, "invoke", True, detail="核执行完成")

        # gate 4: output schema
        ok, msg = check(self.output_schema, value)
        if not ok:
            trace.add(self.name, self.handle, "postcond", False, err=Err.POSTCOND_FAIL.value, detail=f"输出违反 out_schema: {msg}")
            return self._fail(Err.POSTCOND_FAIL, ctx, f"输出违反 out_schema: {msg}", stages, t0, units=units)

        # gate 5: independent verifier (postcond)
        stages.append("postcond")
        resids: list[float] = [float(x) for x in raw.get("resid", [])]
        if self.op.verify is not None:
            try:
                with guard(f"{ctx}@verify"):
                    vr = self.op.verify(args, value)
            except SvoError as e:
                trace.add(self.name, self.handle, "postcond", False, err=e.code.value, detail=f"验证器抛错: {e.detail}")
                return self._fail(e.code, ctx, f"验证器抛错: {e.detail}", stages, t0, units=units)
            resids = [float(x) for x in vr.get("resid", resids)] or resids
            if not vr.get("ok", False):
                rs = resid_summary(resid_vec(resids))
                trace.add(self.name, self.handle, "postcond", False, resid=rs, err=Err.POSTCOND_FAIL.value, detail=vr.get("detail", ""))
                return self._fail(Err.POSTCOND_FAIL, ctx, f"独立验证器否决: {vr.get('detail', '')}", stages, t0, units=units, resid=rs)
            trace.add(self.name, self.handle, "postcond", True, detail=vr.get("detail", ""))

        # gate 6: independent certificate re-verification
        stages.append("cert")
        cert = raw.get("cert")
        if isinstance(cert, dict):
            cert = Certificate(**cert)
        cert_ind: bool | None = None
        if cert is not None or self.op.requires_cert:
            try:
                okc, cresid, cdetail = verify_certificate(cert, ctx, require=self.op.requires_cert)
                cert_ind = bool(okc)
                resids = list(resids) + list(cresid)
                trace.add(self.name, self.handle, "cert", okc, detail=cdetail)
            except SvoError as e:
                trace.add(self.name, self.handle, "cert", False, err=e.code.value, detail=e.detail)
                return self._fail(e.code, ctx, e.detail, stages, t0, units=units)

        rv = resid_vec(resids)
        rs = resid_summary(rv)
        cost = units + float(raw.get("cost", 0.0))
        stages.append("accept")
        trace.add(self.name, self.handle, "accept", True, resid=rs, cost_units=cost, detail=raw.get("notes", ""))
        return ToolResult(
            ok=True, op=self.name, handle=self.handle, value=value, resid=rv, resid_scalar=rs,
            cert=cert, cert_independent=cert_ind, err=None, ctx=ctx, cost_units=cost,
            elapsed_ms=(time.perf_counter() - t0) * 1000.0, notes=raw.get("notes", ""), stages=tuple(stages),
        )

    def call_or_raise(self, args: dict, ctl: CallCtl | None = None) -> ToolResult:
        r = self.call(args, ctl)
        if not r.ok:
            raise SvoError(r.err or Err.RUNTIME_ERR, ctx=r.ctx, detail=r.notes)
        return r

    # -- decode (§4 step 3) -------------------------------------------------
    @staticmethod
    def decode(result: ToolResult) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """Kernel result → (latent-state vector, verification residual vector).

        Numeric results → scalar payload vector; symbolic results → handle (the caller holds the
        handle reference). The residual vector flows back unchanged for AIC aggregation on the
        scoreboard.
        """
        return CODEBOOK.latent(result.handle), result.resid

    def _fail(self, code: Err, ctx: str, detail: str, stages: list[str], t0: float, units: float = 0.0, resid: float = float("nan")) -> ToolResult:
        return ToolResult(
            ok=False, op=self.name, handle=self.handle, value=None, resid=resid_vec([resid]),
            resid_scalar=resid, err=code.value, ctx=ctx, cost_units=units,
            elapsed_ms=(time.perf_counter() - t0) * 1000.0, notes=detail, stages=tuple(stages),
        )


def validate_opdef(op: OpDef) -> None:
    """Static validation before registration (§7 six admission rules + §7.3: no verifier, no entry)."""
    ctx = f"op:{op.name}@register"
    if not op.name or not isinstance(op.name, str):
        raise SvoError(Err.SCHEMA_MISMATCH, ctx=ctx, detail="算子名非法")
    if op.verify is None and op.cert_kind is None:
        raise SvoError(Err.NO_CERTIFICATE, ctx=ctx, detail="既无独立验证器又无证书类型 —— §7.3 没有验证器不许交")
    for e in op.errors:
        if e not in ERROR_TABLE:
            raise SvoError(Err.SCHEMA_MISMATCH, ctx=ctx, detail=f"失败模式 {e!r} 不在 §15 错误表内")
    if not op.selfchecks:
        raise SvoError(Err.SCHEMA_MISMATCH, ctx=ctx, detail="无自检项 —— §7 注册流程要求 self-check 通过")
    if op.cost[0] != UNITS or not isinstance(op.cost[1], (int, float)) or op.cost[1] < 0:
        raise SvoError(Err.SCHEMA_MISMATCH, ctx=ctx, detail=f"成本标注非法: {op.cost!r}")
    from .schema import check_declaration

    check_declaration(op.in_schema, ctx + ".in")
    check_declaration(op.out_schema, ctx + ".out")

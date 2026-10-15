"""Audit trace and deterministic replay (§11.7 combined trust anchor).

- **AuditTrace**: an OpenTelemetry-style event stream — every invoke is recorded (time/cost/residual/error).
  Recording is **unconditional**: the success, failure and exception paths all write events.
- **ReplayLog**: the call sequence plus the seed. Deterministic replay = rerun the same sequence and
  compare event fingerprints. The fingerprint consumes only **structural fields** (normalized
  op/args/err/resid), never wall-clock time — otherwise replay would never compare equal.
- **BudgetLedger**: the global budget ledger, interfacing the calibration tiers (the base of §11.5
  budget_governor).

Honesty boundary: determinism here comes from (a) operators being pure functions, (b) no network,
(c) no unseeded randomness. Any operator that introduces unseeded randomness necessarily fails
replay — that is not a bug but a designed visibility property.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any

from .errors import Err, SvoError


@dataclass
class Event:
    seq: int
    wall_ms: float
    op: str
    handle: int
    stage: str  # 'precond' | 'invoke' | 'postcond' | 'cert' | 'accept'
    ok: bool
    resid: float | None = None
    err: str | None = None
    cost_units: float = 0.0
    detail: str = ""

    def to_dict(self) -> dict:
        return {
            "seq": self.seq,
            "t_ms": round(self.wall_ms, 3),
            "op": self.op,
            "handle": self.handle,
            "stage": self.stage,
            "ok": self.ok,
            "resid": None if self.resid is None else float(self.resid),
            "err": self.err,
            "cost": float(self.cost_units),
            "detail": self.detail,
        }


class AuditTrace:
    def __init__(self, name: str = "trace") -> None:
        self.name = name
        self.events: list[Event] = []
        self._t0 = time.perf_counter()

    def add(
        self,
        op: str,
        handle: int,
        stage: str,
        ok: bool,
        resid: float | None = None,
        err: str | None = None,
        cost_units: float = 0.0,
        detail: str = "",
    ) -> Event:
        ev = Event(
            seq=len(self.events),
            wall_ms=(time.perf_counter() - self._t0) * 1000.0,
            op=op,
            handle=handle,
            stage=stage,
            ok=ok,
            resid=resid,
            err=err,
            cost_units=cost_units,
            detail=detail,
        )
        self.events.append(ev)
        return ev

    def violations(self) -> list[Event]:
        return [e for e in self.events if not e.ok]

    def struct_fingerprint(self) -> str:
        """Structural fingerprint (no wall clock): used for replay comparison."""
        blob = json.dumps(
            [
                {
                    "op": e.op,
                    "stage": e.stage,
                    "ok": e.ok,
                    "err": e.err,
                    "resid": None if e.resid is None else round(float(e.resid), 12),
                }
                for e in self.events
            ],
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

    def to_jsonl(self) -> str:
        return "\n".join(json.dumps(e.to_dict(), ensure_ascii=False) for e in self.events)

    def summary(self) -> dict:
        ok = sum(1 for e in self.events if e.ok)
        return {
            "events": len(self.events),
            "ok": ok,
            "violations": len(self.events) - ok,
            "cost_units": round(sum(e.cost_units for e in self.events), 6),
            "fingerprint": self.struct_fingerprint(),
            "errors": sorted({e.err for e in self.events if e.err}),
        }


class BudgetLedger:
    """Global budget ledger (§11.5). Overspending raises BUDGET_EXHAUSTED and never silently continues."""

    def __init__(self, total_units: float) -> None:
        self.total = float(total_units)
        self.spent = 0.0
        self.entries: list[tuple[str, float]] = []

    def remaining(self) -> float:
        return self.total - self.spent

    def charge(self, op: str, units: float) -> None:
        u = float(units)
        if u < 0:
            raise SvoError(Err.SCHEMA_MISMATCH, ctx="budget", detail=f"负成本 {u}")
        if self.spent + u > self.total:
            raise SvoError(
                Err.BUDGET_EXHAUSTED,
                ctx=f"budget@{op}",
                detail=f"需 {u}, 余 {self.remaining():.6f} (总 {self.total})",
            )
        self.spent += u
        self.entries.append((op, u))

    def snapshot(self) -> dict:
        return {"total": self.total, "spent": round(self.spent, 6), "remaining": round(self.remaining(), 6)}


@dataclass
class ReplayLog:
    """Call-sequence record: (op, args, seed) → replayable."""

    calls: list[dict] = field(default_factory=list)
    seed: int = 0

    def record(self, op: str, args: dict, fingerprint: str) -> None:
        self.calls.append({"op": op, "args": args, "fingerprint": fingerprint})

    def to_json(self) -> str:
        return json.dumps({"seed": self.seed, "calls": self.calls}, ensure_ascii=False, sort_keys=True, default=str)

    @staticmethod
    def from_json(blob: str) -> "ReplayLog":
        d = json.loads(blob)
        rl = ReplayLog(seed=int(d["seed"]))
        rl.calls = list(d["calls"])
        return rl

    def diff(self, other: "ReplayLog") -> list[str]:
        """Compare the call sequences of two runs; return difference descriptions (empty = identical)."""
        out: list[str] = []
        if self.seed != other.seed:
            out.append(f"seed 不同: {self.seed} vs {other.seed}")
        if len(self.calls) != len(other.calls):
            out.append(f"调用数不同: {len(self.calls)} vs {len(other.calls)}")
        for i, (a, b) in enumerate(zip(self.calls, other.calls)):
            if a != b:
                out.append(f"第 {i} 次调用不同: {a} vs {b}")
        return out


def replay_equal(blob_a: str, blob_b: str) -> tuple[bool, list[str]]:
    """Independent decision of deterministic replay (the implementation behind the certificate checker 'replay')."""
    d = ReplayLog.diff(ReplayLog.from_json(blob_a), ReplayLog.from_json(blob_b))
    return not d, d


def canon_args(args: Any) -> Any:
    """Argument canonicalization (for fingerprints and replay comparison): sort dict keys, tuple→list, fixed float precision."""
    if isinstance(args, dict):
        return {k: canon_args(v) for k, v in sorted(args.items(), key=lambda kv: str(kv[0]))}
    if isinstance(args, (list, tuple)):
        return [canon_args(v) for v in args]
    if isinstance(args, float):
        return round(args, 12)
    return args

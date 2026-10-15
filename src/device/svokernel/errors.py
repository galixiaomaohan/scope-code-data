"""SVO §15 master error enumeration.

Iron rule: operators may raise only errors from this table, with a context handle (ctx).
Out-of-table exceptions are collapsed at the invoke boundary by guard() into RUNTIME_ERR,
keeping the original exception text (never a silent wrong value, never a silently swallowed
exception).

Counter-example (forbidden): `raise ValueError('bad')` escaping an operator to the caller —
the caller cannot programmatically backtrack. Correct form:
`raise SvoError(Err.DEG_UNSUPPORTED, ctx=..., detail=...)`.
"""
from __future__ import annotations

import contextlib
from enum import Enum
from typing import Iterator


class Err(str, Enum):
    """§15 full enumeration (26 entries, frozen; additions require a version change, not in-place table edits)."""

    # resource class
    TIMEOUT = "TIMEOUT"
    OOM = "OOM"
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"
    # execution class
    COMPILE_ERR = "COMPILE_ERR"
    RUNTIME_ERR = "RUNTIME_ERR"
    # interface-contract class
    SCHEMA_MISMATCH = "SCHEMA_MISMATCH"
    PRECOND_FAIL = "PRECOND_FAIL"
    POSTCOND_FAIL = "POSTCOND_FAIL"
    SPEC_AMBIGUOUS = "SPEC_AMBIGUOUS"
    # logic/proof class
    UNIFY_FAIL = "UNIFY_FAIL"
    NO_CERTIFICATE = "NO_CERTIFICATE"
    DRAT_CHECK_FAIL = "DRAT_CHECK_FAIL"
    # math class
    DEG_UNSUPPORTED = "DEG_UNSUPPORTED"
    DISC_NEG = "DISC_NEG"
    NO_REAL_ROOT = "NO_REAL_ROOT"
    UNIDENTIFIABLE = "UNIDENTIFIABLE"
    BOUNDED_VIOLATION = "BOUNDED_VIOLATION"
    MODEL_CHECK_FAIL = "MODEL_CHECK_FAIL"
    # code-domain class
    REGRESS_FAIL = "REGRESS_FAIL"
    MUTATION_SURVIVE = "MUTATION_SURVIVE"
    API_INCOMPAT = "API_INCOMPAT"
    DEP_CONFLICT = "DEP_CONFLICT"
    # external-kernel class
    LEAN_ELAB_ERR = "LEAN_ELAB_ERR"
    LEAN_TIMEOUT = "LEAN_TIMEOUT"
    MATHLIB_MISS = "MATHLIB_MISS"
    # bridge class
    MCP_BRIDGE_ERR = "MCP_BRIDGE_ERR"


ERROR_TABLE: tuple[str, ...] = tuple(e.value for e in Err)
_ERRORS: frozenset[str] = frozenset(ERROR_TABLE)


class SvoError(Exception):
    """The only exception type allowed to escape.

    code   : one of Err (out-of-table strings are rejected outright — the error table must not be
             extended in place)
    ctx    : context handle, recommended form "op:<name>#<handle>" or "op:<name>@<stage>"
    detail : human-readable detail; must not be empty, and must not smuggle in a silently-passing
             semantics
    """

    __slots__ = ("code", "ctx", "detail")

    def __init__(self, code: "Err | str", ctx: str = "", detail: str = "") -> None:
        if isinstance(code, Err):
            self.code = code
        else:
            s = str(code)
            if s not in _ERRORS:
                raise AssertionError(
                    f"非枚举错误码 {s!r}; §15 只许抛表内错误 (表: {len(_ERRORS)} 项)"
                )
            self.code = Err(s)
        self.ctx = ctx
        self.detail = detail
        parts = [self.code.value]
        if ctx:
            parts.append(f"[{ctx}]")
        if detail:
            parts.append(f": {detail}")
        super().__init__("".join(parts))

    def to_dict(self) -> dict:
        return {"error": self.code.value, "ctx": self.ctx, "detail": self.detail}


@contextlib.contextmanager
def guard(ctx: str, on: tuple[type[BaseException], ...] = ()) -> Iterator[None]:
    """Invoke-boundary guard: collapse out-of-table exceptions into RUNTIME_ERR, keeping the original text.

    Exception types inside `on` map to more specific codes, e.g. on=((TimeoutError, Err.TIMEOUT),).
    Usage:
        with guard("op:solve_poly@invoke", ((ZeroDivisionError, Err.DEG_UNSUPPORTED),)):
            ...
    """
    mapping: dict[type[BaseException], Err] = {}
    for spec in on:
        exc_t, code_t = spec  # type: ignore[misc]
        mapping[exc_t] = code_t
    try:
        yield
    except SvoError:
        raise
    except BaseException as e:  # noqa: BLE001 - the single collapse point
        for exc_t, code_t in mapping.items():
            if isinstance(e, exc_t):
                raise SvoError(code_t, ctx=ctx, detail=f"{type(e).__name__}: {e}") from e
        raise SvoError(
            Err.RUNTIME_ERR, ctx=ctx, detail=f"表外异常 {type(e).__name__}: {e}"
        ) from e


def is_enum_error(code: str) -> bool:
    return code in _ERRORS

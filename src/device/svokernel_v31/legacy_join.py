# -*- coding: utf-8 -*-
"""Reference-style re-registration of the 118 legacy operators (zero copy, zero modification).

Method: read the legacy REGISTRY OpDefs one by one and build a V31OpDef pointing at the
**same** invoke / verify / precond function objects and the same selfchecks, with the band
unchanged and sub = row - band given explicitly -- so the 118 legacy operators keep **exactly
the same row numbers** in the V31 codebook (anchor-chain audit criterion 5 checks this).

Why sub is given explicitly instead of letting the codebook queue them: queueing depends on
registration order; any order change would shift row numbers and misalign all 118 old slots
of the routing head. Explicit row numbers make the row a **declaration**, not a side effect.

Impact on historical numbers: zero. The legacy REGISTRY / CODEBOOK / CHECKERS are only read;
the V31 registry is a new object; the 583 legacy self-checks and the 5 checksum numbers are
unchanged (stage A evidence in STATUS.md).
"""
from __future__ import annotations

# Legacy operators must be loaded **first**: the 118 operators self-register when the
# svokernel.ops modules are imported. Without the import LEGACY_REGISTRY is empty -- and the
# re-registration would "successfully" merge 0 operators, turning all 5 anchor-chain audit
# criteria red (anchor_ok=False; this package was silently hit by that trap once).
# This import only makes explicit what the legacy package would do anyway: import only,
# never write/modify/copy legacy files.
import svokernel.ops as _legacy_ops  # noqa: F401  (import side effect = legacy operator self-registration)
from svokernel.errors import Err, SvoError
from svokernel.registry import REGISTRY as LEGACY_REGISTRY

from .v31codebook import V31_CODEBOOK
from .v31registry import V31OpDef, V31REGISTRY

__all__ = ["join_legacy", "JOINED"]


def join_legacy() -> list:
    """Re-register every legacy operator; returns the merged names (ordered by row).
    Idempotent (a repeated call returns the already-merged set)."""
    # Fail-safe: an empty table is always rejected -- "merge 0 operators without an error"
    # is the most dangerous silent failure mode of this package.
    if len(LEGACY_REGISTRY) == 0:
        raise SvoError(Err.SCHEMA_MISMATCH, ctx="v31:legacy_join",
                       detail="存量注册表为空 (svokernel.ops 未装载) —— 拒绝并入 0 个算子")
    joined: list = []
    for name in LEGACY_REGISTRY.names():
        if V31REGISTRY.has(name):
            joined.append(name)
            continue
        op = LEGACY_REGISTRY._ops[name]
        row = LEGACY_REGISTRY.handle_of(name)
        sub = row - op.band
        if sub < 1:
            # legacy convention: no row sits on a band base (checked: 118/118 satisfy sub>=1)
            raise SvoError(Err.SCHEMA_MISMATCH, ctx="v31:legacy_join",
                           detail="存量行 %d 落在段基址 (band=%d) —— 与冻结约定不符, 拒绝盲目转注册"
                                  % (row, op.band))
        new = V31OpDef(
            name=op.name, band=op.band, sub=sub,
            in_schema=op.in_schema, out_schema=op.out_schema,
            invoke=op.invoke, verify=op.verify, cert_kind=op.cert_kind,
            cost=op.cost, errors=tuple(op.errors), effects=op.effects,
            selfchecks=tuple(op.selfchecks), doc=op.doc,
            requires_cert=op.requires_cert, precond=op.precond,
            stmt=op.stmt, thm=op.thm,
            status="verified", layer=-2,   # -2 = legacy re-registered reference (not a new V31 layer)
        )
        V31REGISTRY.register(new)
        joined.append(name)
    return joined


#: The join happens at import time (idempotent); __init__ and tests use it directly.
JOINED = join_legacy()

# Re-check: after re-registration both tables must agree row by row -- a mismatch raises
# (never continue with wrong row numbers).
for _n in LEGACY_REGISTRY.names():
    _want, _got = LEGACY_REGISTRY.handle_of(_n), V31_CODEBOOK.row_of(_n)
    if _want != _got:
        raise SvoError(Err.SCHEMA_MISMATCH, ctx="v31:legacy_join@verify",
                       detail="转注册行号漂移: %s 存量 #%d vs V31 #%d" % (_n, _want, _got))

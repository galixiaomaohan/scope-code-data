# -*- coding: utf-8 -*-
"""svokernel_v32 -- V32 proof-skeleton meta-operator set (reference-style inclusion of the full
v31 table + 6 structural-layer operators + independent certificate checkers).

Relationship to v31 / legacy (hard boundaries):
  - **Zero modification**: this package never writes or import-side-effects the files of legacy
    `svokernel/` and `svokernel_v31/`; their registries/codebooks/certificate checkers are only
    **read**. v31's 792 selfchecks and codebook checksum are unchanged in any V32 process
    (anchor_chain_audit_v32 re-checks them item by item by really running);
  - **Reference-style inclusion**: the 154 v31 operators point at the same invoke/verify/selfcheck
    function objects in the V32 registry, with row numbers replicated row by row
    (v32registry.join_v31) -- exactly what v31 did for the 118 legacy operators;
  - **The only shared mutable global** is still `svokernel.certify.CHECKERS`: the 6 V32 skeleton
    certificate checkers are **appended** through the legacy mechanism (adding keys, not changing
    values); the names and module_of of the 28 legacy + 26 v31 kinds are re-checked by the
    anchor-chain audit.

Load manifest (allowlist style, same discipline as v31):
  certificate-checker modules first (they register kinds), then the operator modules (they register
  operators).
"""
from __future__ import annotations

__version__ = "32.0.0"
SCHEMA_VERSION = "svo-32-skel/1"

from .v32codebook import (  # noqa: F401
    N_SKEL_OPS, V32_BANDS, V32_CODEBOOK, V32_END_MARKER, V32_END_MARKER_ROW,
    V32_HEAD_DIM_FROZEN, V32_NEW_BANDS, V32_SCHEMA, V32_SKEL_BAND, V32Codebook,
    v31_prefix_rows, v32_action_ops, v32_band_of, v32_op_index_map,
)
from .v32registry import (  # noqa: F401
    JOINED_V31, SKEL_CERT_KINDS, V31_CERT_KINDS_FROZEN, V32_ANCHOR, V32OpDef, V32Registry,
    V32REGISTRY, anchor_chain_audit_v32, audit_static_v32, join_v31,
)

from svokernel.errors import Err, SvoError  # noqa: E402

#: Load manifest: (module name, description). A missing module only records MISSING, it does not
#: blow up -- but it does under strict.
MANIFEST = (
    ("certcheck_skel", "骨架证书独立检查器 (6 类: case_exhaustive/induction_scheme/"
                       "stack_machine/leaf_residue_complete/contradiction_witness/skeleton_qed)"),
    ("ops_skel", "证明骨架元算子 (band 9950, 行 9951..9956, layer=4)"),
)

LOADED: list = []
MISSING: list = []


def load_all(strict: bool = False):
    """Load MANIFEST; return (loaded, missing). Only "module file absent" is swallowed; an error
    raised inside a module must propagate."""
    import importlib

    global LOADED, MISSING
    LOADED, MISSING = [], []
    for mod, _desc in MANIFEST:
        try:
            importlib.import_module("." + mod, __name__)
            LOADED.append(mod)
        except ModuleNotFoundError as e:
            if e.name and e.name.split(".")[-1] != mod:
                raise
            MISSING.append(mod)
            if strict:
                raise
    return LOADED, MISSING


def describe() -> str:
    lines = ["V32 loaded %d/%d modules:" % (len(LOADED), len(MANIFEST))]
    for mod, desc in MANIFEST:
        lines.append("  [%s] %-16s %s" % ("OK " if mod in LOADED else "MISS", mod, desc))
    return "\n".join(lines)


load_all()

#: Skeleton operator names (ordered by row)
SKEL_OPS = tuple(n for n in V32REGISTRY.names() if getattr(V32REGISTRY._ops[n], "skeleton", False))

__all__ = [
    "__version__", "SCHEMA_VERSION", "MANIFEST", "LOADED", "MISSING", "load_all", "describe",
    "SKEL_OPS", "SKEL_CERT_KINDS", "V31_CERT_KINDS_FROZEN", "V32_ANCHOR", "V32_BANDS",
    "V32_CODEBOOK", "V32_END_MARKER", "V32_END_MARKER_ROW", "V32_HEAD_DIM_FROZEN",
    "V32_NEW_BANDS", "V32_SCHEMA", "V32_SKEL_BAND", "V32Codebook", "V32OpDef", "V32Registry",
    "V32REGISTRY", "JOINED_V31", "anchor_chain_audit_v32", "audit_static_v32", "join_v31",
    "N_SKEL_OPS", "v31_prefix_rows", "v32_action_ops", "v32_band_of", "v32_op_index_map",
    "Err", "SvoError",
]

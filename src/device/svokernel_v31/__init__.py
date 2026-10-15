# -*- coding: utf-8 -*-
"""svokernel_v31 -- V31 superoperator set (parallel registry + four new operator layers
+ independent certificate checkers).

Relationship to svokernel (hard boundaries, see STATUS.md anchor-chain section):
  - **Zero modification**: this package never writes or import-side-effects the legacy
    files; legacy REGISTRY/CODEBOOK/CHECKERS are read-only here. The 583 self-checks of
    the 118 legacy operators and the legacy codebook checksum are unchanged in any V31
    process;
  - **Reference-style inclusion**: the 118 legacy operators point at the same invoke/verify
    function objects in the V31 registry, with row numbers replicated row by row
    (legacy_join.py);
  - **The only shared mutable global** is svokernel.certify.CHECKERS: V31 certificate
    checkers are **appended** through the legacy mechanism (the register_checker
    decorator). That adds keys, never changes values -- the names and module_of of the
    28 legacy kinds are re-checked item by item by anchor_chain_audit().

Load manifest (allowlist style, same discipline as svokernel.ops.MANIFEST: loading
whatever happens to sit in the directory is forbidden):
  certificate-checker modules are loaded first (they register kinds), then the operator
  modules (they register operators), and finally the END sentinel row is acked.
"""
from __future__ import annotations

__version__ = "31.0.0"
SCHEMA_VERSION = "svo-31"

from .v31codebook import (  # noqa: F401
    V31_BANDS, V31_NEW_BANDS, V31_CODEBOOK, V31Codebook, V31_END_MARKER,
    V31_END_MARKER_ROW, V31_INFRA_BAND, V31_STUB_BAND, v31_band_of,
    v31_head_dim, v31_action_ops, EMB_DIM, RESID_DIM, GLOBAL_SEED, latent_vec,
)
from .v31registry import (  # noqa: F401
    V31OpDef, V31Registry, V31REGISTRY, LEGACY_ANCHOR, LEGACY_CERT_KINDS_FROZEN,
    audit_static_v31, anchor_chain_audit,
)
from .legacy_join import join_legacy, JOINED  # noqa: F401

from svokernel.errors import Err, SvoError  # noqa: E402

#: Load manifest: (module name, description). A missing module only records MISSING,
#: it does not blow up -- but it does under strict.
MANIFEST = (
    ("certcheck_nt", "L0 竞赛数论证书独立检查器 (10 类)"),
    ("certcheck_alg", "L1 结构性数论证书独立检查器 (9 类)"),
    ("certcheck_res", "L2 研究级证书独立检查器 (3 类)"),
    ("certcheck_goal", "L3 调用机制证书独立检查器 (4 类)"),
    ("ops_infra", "V31 基础设施算子 (码本自证件 / 代数数算术 / census)"),
    ("ops_nt_l0", "第 0 层竞赛数论 (band 9900, 10 算子)"),
    ("ops_alg_l1", "第 1 层结构性数论 (band 9910 实现 + 9940 stub)"),
    ("ops_res_l2", "第 2 层研究级 (band 9920 实现 + 9940 stub)"),
    ("ops_goal_l3", "第 3 层调用机制 (band 9930, 4 算子)"),
)

LOADED: list = []
MISSING: list = []


def load_all(strict: bool = False):
    """Load MANIFEST; return (loaded, missing). Only "module file absent" is swallowed;
    an error raised inside a module must propagate."""
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
    _ack_end_marker()
    return LOADED, MISSING


def _ack_end_marker() -> int:
    """Register the END sentinel row (not an operator): makes the V31 head dimension
    equal the codebook row count. Idempotent."""
    h = V31_CODEBOOK.allocate(V31_END_MARKER, 9990, sub=9)
    if h.row != V31_END_MARKER_ROW:
        raise SvoError(Err.SCHEMA_MISMATCH, ctx="v31:end_marker",
                       detail="END 哨兵行漂移: 期望 #%d, 实得 #%d" % (V31_END_MARKER_ROW, h.row))
    return h.row


def describe() -> str:
    lines = ["V31 loaded %d/%d modules:" % (len(LOADED), len(MANIFEST))]
    for mod, desc in MANIFEST:
        lines.append("  [%s] %-16s %s" % ("OK " if mod in LOADED else "MISS", mod, desc))
    return "\n".join(lines)


load_all()

__all__ = [
    "__version__", "SCHEMA_VERSION", "MANIFEST", "LOADED", "MISSING", "load_all", "describe",
    "V31_BANDS", "V31_NEW_BANDS", "V31_CODEBOOK", "V31Codebook", "V31_END_MARKER",
    "V31_END_MARKER_ROW", "V31_INFRA_BAND", "V31_STUB_BAND", "v31_band_of",
    "v31_head_dim", "v31_action_ops", "EMB_DIM", "RESID_DIM", "GLOBAL_SEED", "latent_vec",
    "V31OpDef", "V31Registry", "V31REGISTRY", "LEGACY_ANCHOR", "LEGACY_CERT_KINDS_FROZEN",
    "audit_static_v31", "anchor_chain_audit", "join_legacy", "JOINED", "Err", "SvoError",
]

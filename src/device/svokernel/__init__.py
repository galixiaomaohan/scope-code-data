"""SVO vectorized super-operator system — kernel package (VMP + operator registry + certificates + audit).

Four public entry points only:
    from svokernel import REGISTRY, CODEBOOK, Err, SvoError, CallCtl, AuditTrace, BudgetLedger
    from svokernel import ops                      # importing registers all operators
    t = REGISTRY.tool("solve_poly"); r = t.call({...})
    r.ok, r.value, r.resid_scalar, r.cert, r.err

Design constraints (package-wide; auditable item by item):
  - Standard library only (no numpy/torch): the exactness claim does not depend on third-party numerical libraries;
  - No in-process network calls (mock_search consumes a fixed corpus only; lean/drat use subprocesses but all inputs are local);
  - Every failure = a §15 enumerated error with a ctx handle; the invoke boundary collapses out-of-table exceptions;
  - Verifiers are independent of the generation path (no self-certification), enforced at registration time by certify.CHECKERS.
"""
from __future__ import annotations

__version__ = "1.0.0"
SCHEMA_VERSION = "svo-1"

from .audit import AuditTrace, BudgetLedger, ReplayLog, replay_equal
from .certify import CHECKERS, Certificate, verify_certificate
from .codebook import CODEBOOK, EMB_DIM, RESID_DIM, Codebook, Handle, band_of, latent_vec, resid_vec
from .errors import ERROR_TABLE, Err, SvoError, guard
from .logiccore import check_proof, check_sat_assignment, resolve, unify
from .ndcore import check_nd, falsifies as logic_falsifies
from .registry import REGISTRY, OpDef, Registry, SelfCheckResult, op_registered
from .schema import check as schema_check, validate as schema_validate
from .score import AIC_C, EPS_FLOOR, Acceptance, aic_score, accept, undetermined_check
from .tool import CallCtl, ToolResult, TypedTool

__all__ = [
    "__version__", "SCHEMA_VERSION",
    "Err", "SvoError", "guard", "ERROR_TABLE",
    "Codebook", "Handle", "CODEBOOK", "EMB_DIM", "RESID_DIM", "band_of", "latent_vec", "resid_vec",
    "Certificate", "CHECKERS", "verify_certificate",
    "AuditTrace", "BudgetLedger", "ReplayLog", "replay_equal",
    "aic_score", "accept", "Acceptance", "AIC_C", "EPS_FLOOR", "undetermined_check",
    "schema_check", "schema_validate",
    "check_proof", "check_sat_assignment", "resolve", "unify",
    "check_nd", "logic_falsifies",
    "OpDef", "Registry", "REGISTRY", "TypedTool", "ToolResult", "CallCtl", "SelfCheckResult", "op_registered",
]

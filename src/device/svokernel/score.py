"""Acceptance criteria and AIC scoring (§6 last paragraph + §17.4).

All scoring is AIC-type: `n_eq·ln(rss+ε) + c·dof`.
ε = the numerical floor (declared on record); c ≈ 2 (standard AIC; may be overridden via
--aic-c, but every override must be registered with the experiment).

**Why linear penalties are banned** (§17.4, empirically on controlled domains): a linear
`λ·dof` loses to wrong low-dof answers on near-degenerate problems — in that regime the
residual drop per added dof is far faster than the linear compensation. The log-residual
term `n_eq·ln(rss+ε)` diverges to −∞ as rss→0, so it naturally punishes "extra dof that
buys no residual drop".

Misuse warning (§17.2 upfront): this score only ranks among multiple solutions; it does
**not** decide underdetermination. Problems with n_eq ≤ dof must be screened by the caller
first (hard constraint n_eq > dof), otherwise AIC gives the same score to infinitely many
zero-residual solutions — and then "the minimum score" is arbitrary.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .errors import Err, SvoError

EPS_FLOOR = 1e-12
"""Numerical floor ε (declared on record: any residual below it counts as the floor, to keep ln(0) out)."""

AIC_C = 2.0
"""Dof penalty coefficient c."""


def aic_score(n_eq: int, rss: float, dof: int, c: float = AIC_C, eps: float = EPS_FLOOR) -> float:
    """§6: n_eq·ln(rss+ε) + c·dof. rss is the residual sum of squares (may be a relative-residual sum of squares)."""
    if n_eq <= 0:
        raise SvoError(Err.SCHEMA_MISMATCH, ctx="score.aic", detail=f"n_eq 须 >0, 得 {n_eq}")
    if dof < 0:
        raise SvoError(Err.SCHEMA_MISMATCH, ctx="score.aic", detail=f"dof 须 ≥0, 得 {dof}")
    r = float(rss)
    if r != r or r == float("inf"):
        return float("inf")
    return n_eq * math.log(max(r, 0.0) + eps) + c * float(dof)


@dataclass
class Acceptance:
    """Acceptance record of one candidate (§6: residual ≤ ε and AIC dominance → accepted)."""

    accepted: bool
    resid: float
    score: float
    eps: float
    reason: str

    def to_dict(self) -> dict:
        return {
            "accepted": self.accepted,
            "resid": self.resid,
            "score": self.score,
            "eps": self.eps,
            "reason": self.reason,
        }


def accept(
    resid: float,
    score: float,
    eps: float,
    best_score: float | None = None,
    dof: int = 0,
) -> Acceptance:
    """Acceptance: residual ≤ ε and (no rival or better AIC score).

    `best_score=None` means the first candidate (no rival). Returns a structured record and
    never raises — an acceptance miss is **normal control flow** (backtrack to another
    candidate), not an error.
    """
    if resid != resid:
        return Acceptance(False, resid, score, eps, "残差为 nan")
    if resid > eps:
        return Acceptance(False, resid, score, eps, f"残差 {resid:.3e} > ε {eps:.3e}")
    if best_score is not None and not (score < best_score):
        return Acceptance(False, resid, score, eps, f"AIC {score:.6f} 未优于 {best_score:.6f}")
    return Acceptance(True, resid, score, eps, "验收通过")


def undetermined_check(n_eq: int, dof: int, ctx: str) -> None:
    """§17.2: equation count > dof must be argued — here only the hard check;

    the strictness argument lives in the operator documentation. The lower limit
    n_eq == dof + 1 is permitted (exactly determined + 1); n_eq <= dof raises UNIDENTIFIABLE.
    """
    if n_eq <= dof:
        raise SvoError(
            Err.UNIDENTIFIABLE,
            ctx=ctx,
            detail=f"欠定: 方程数 n_eq={n_eq} ≤ 自由度 dof={dof}; 禁在此区宣称唯一解",
        )

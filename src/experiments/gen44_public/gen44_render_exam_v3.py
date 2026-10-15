# -*- coding: utf-8 -*-
# gen44_render_exam_v3.py -- **model exam-convention** renderer v3 (deterministic single
# rendering; the only usable one for the [8] final test)
#
# Version history (all registered, nothing deleted):
#   v1 = the original gen42 renderer (`gen42_families.render_tactic` + `gen42_payload.REDUCE_LINE`):
#        normalization variant = {has ↑: push_cast; has /: field_simp; otherwise norm_num1};
#        the closing line had no `try`.
#        Defects: (1) the normalization step already closed the goal, so the last step reported
#                 `No goals to be solved` (a rendering-side false kill);
#               (2) field_simp was selected while there was nothing to do => `field_simp made no
#                 progress` (a rendering-side false kill).
#   v2 = gen44_render_exam.py (md5 d19bbf41eddfefb2486b401df06f2b80):
#        FIX-A selects the normalization variant by whether the residual denominator contains free
#        variables + FIX-B wraps only the closing line in `try`.
#        **Measured defect (registered)**: `sympy.together` **cancels algebraic factors**, removing
#        variable denominators, so the code wrongly concludes field_simp is unnecessary and
#        produces **2 regressions** on the miniF2F 86 rows (`(I/2)^2 = -(1/4)` in
#        `mathd_algebra_302`; `12/(x*x)*(x^4/(14*x))*(35/(3*x)) = 10` in `mathd_algebra_441`, whose
#        denominator has no numeric variables after cancellation).
#   v3 = this file (adopted): **variant returns to the v1 rule (syntax level: has ↑ / has / )**
#        plus **every line wrapped in `try`**.
#        Semantics = "attempt the reductions step by step; closing counts as soon as one attempt
#        converges" -- consistent with the device operator semantics ("each step advances the
#        state, the terminal state must close"); **fail-closed**: if every line fails the goal
#        remains, Lean reports unsolved goals and the row does not pass. **No candidate
#        enumeration** (one chain -> one text).
#        N1 semantics: `strip_tactic_args` does not touch `try`, so stripping arguments is the
#        identity and the vacuity guard is skipped.
import re

_TRY = "try "


def exam_variant_v1(meta):
    """v1 syntax-level rule (deterministic)."""
    if meta.get("uses_cast"):
        return "push_cast"
    if meta.get("uses_div"):
        return "field_simp"
    return "norm_num1"


_CLOSE = {"g42_s_poly": "ring", "g42_s_eval": "norm_num", "g42_s_lin": "linarith",
          "g42_s_omega": "omega", "g42_s_pos": "positivity"}


def render_exam_v3(F, rec, got, chain_ops=None):
    """Deterministic single rendering (v3): every line wrapped in `try`."""
    ops = list(chain_ops if chain_ops is not None else got["chain_ops"])
    if got["family"] == F.FAM_SUBST:
        pl = dict(got.get("route_payload") or {})
        pl.update(got)
        t = F.render_tactic(rec, got["family"], pl, ops)
        # The family-A closing carries hypothesis arguments (a chain-sensitive payload segment) and
        # must not be wrapped in `try` (that would mask argument errors) => keep as is.
        return t
    lines = []
    for op in ops:
        if op == "g42_s_reduce":
            lines.append(_TRY + exam_variant_v1(rec["concl"]["meta"]))
        elif op == "g42_s_mod":
            lines.append(_TRY + "rw [Nat.pow_mod]")
            lines.append(_TRY + "norm_num")
        elif op in _CLOSE:
            lines.append(_TRY + _CLOSE[op])
        else:
            return "skip"
    return "\n".join(lines) if lines else "skip"

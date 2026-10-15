"""audit_closure.py -- **regression verification** of the adversarial-audit findings
(attack replay -> must now be rejected).

This is not "run the selfchecks again": it replays the **attack payloads** that auditors C
(certificate-forgery red team), A (triple-gate bypass), B (sampling contract) and D (numeric
honesty) actually used, and asserts item by item that they are now rejected or no longer produce
false conclusions. Each item cites the source auditor and the original attack form.

How to run (server):
    PYTHONHASHSEED=0 /usr/bin/python3 tests/audit_closure.py

**Honesty statement**: this file covers only the specific attack surfaces pointed out during the
audit. It is not a proof that "the certificate system has no holes" -- new attack surfaces may
still exist; its purpose is to prevent **already-fixed issues from regressing**.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fractions import Fraction  # noqa: E402

from svokernel import ops  # noqa: F401,E402
from svokernel.certify import CHECKERS, Certificate  # noqa: E402
from svokernel.registry import REGISTRY  # noqa: E402

RESULTS: list = []


def expect_rejected(kind: str, payload: dict, who: str, desc: str) -> None:
    """The attack payload must be rejected by the checker."""
    ok, resid, detail = CHECKERS.check(Certificate(kind=kind, payload=payload))
    RESULTS.append((not ok, f"[{who}] {kind}: {desc}", f"ok={ok} resid={resid} {detail[:110]}"))


def expect_accepted(kind: str, payload: dict, who: str, desc: str) -> None:
    """False-negative fix: a legitimate certificate must pass."""
    ok, resid, detail = CHECKERS.check(Certificate(kind=kind, payload=payload))
    RESULTS.append((ok, f"[{who}] {kind}: {desc}", f"ok={ok} resid={resid} {detail[:110]}"))


def expect_call_err(tool: str, args: dict, err: str, who: str, desc: str) -> None:
    """The operator call must fail with the given error code (no false conclusion)."""
    r = REGISTRY.tool(tool).call(args)
    good = (not r.ok) and r.err == err
    RESULTS.append((good, f"[{who}] {tool}: {desc}", f"ok={r.ok} err={r.err} {r.notes[:100]}"))


def expect_call_ok(tool: str, args: dict, pred, who: str, desc: str) -> None:
    r = REGISTRY.tool(tool).call(args)
    good = r.ok and pred(r)
    RESULTS.append((good, f"[{who}] {tool}: {desc}", f"ok={r.ok} err={r.err} {r.notes[:110]}"))


# ===========================================================================
# C-01 poly_backsub: forged tolerance tol=1e9 + fake roots (the "roots" 5, 7 of x²+1)
# ===========================================================================
expect_rejected("poly_backsub",
                {"coeffs": ["1", "0", "1"], "roots": ["5", "7"], "tol": "1000000000"},
                "C", "伪造容差 tol 不再被采信")

# ===========================================================================
# C-02 drat_proof: satisfiable CNF + empty proof
# ===========================================================================
expect_rejected("drat_proof", {"cnf": [[1, 2]], "proof": []},
                "C", "空证明不再算反证 (要求真导出空子句)")

# ===========================================================================
# C-03/C-04 mutation_report / test_vector: self-consistent numbers only
# ===========================================================================
expect_rejected("mutation_report", {"total": 1, "survived": 0},
                "C", "自报计数不再算证据 (须带 program/tests/mutants)")
expect_rejected("test_vector",
                {"cases": [{"args": [1], "expect": 5, "got": 5}], "held_out": True},
                "C", "自报 got 不再算证据 (须带 program, 由检查器执行)")

# ===========================================================================
# C-05 mc_counterexample: forged trace without trans
# ===========================================================================
expect_rejected("mc_counterexample",
                {"path": [{"from": "s0", "to": "bad"}], "bad": "bad", "claim": "reachable"},
                "C", "无 trans 的轨迹不再被采信")

# ===========================================================================
# C-06/C-07 tlc_exhaustive / bmc_bounded_ok: blank certificates carrying only counts
# ===========================================================================
expect_rejected("tlc_exhaustive", {"n_states": 7},
                "C", "无 spec 的计数不再算穷举证书")
expect_rejected("bmc_bounded_ok", {"k": 10, "n_paths": 50},
                "C", "无 program 的计数不再算界内证书")

# ===========================================================================
# C-08/C-09 equivalence_pair / divergence_witness: forged behavior pair (both directions)
# ===========================================================================
_cases_fake = {"cases": [{"args": [2, 3], "expect": 5, "got": 5}], "held_out": True}
expect_rejected("equivalence_pair", _cases_fake, "C", "伪造等价 (只报行为对)")
expect_rejected("divergence_witness",
                {"cases": [{"args": [1, 1], "expect": 1, "got": 2}], "held_out": True},
                "C", "伪造不等价 (只报行为对)")

# ===========================================================================
# C-10 axiom_table: self-extended allowlist / empty payload
# ===========================================================================
expect_rejected("axiom_table",
                {"axioms": ["Classical.choice", "funext", "LEM"],
                 "allowed": ["Classical.choice", "funext", "LEM"]},
                "C", "自扩白名单不再通过 (白名单由检查器持有)")
expect_rejected("axiom_table", {}, "C", "空 payload 不再通过")

# ===========================================================================
# C-11 replay: two self-reported fingerprints
# ===========================================================================
expect_rejected("replay", {"run_a": "abc", "run_b": "abc"},
                "C", "自报指纹不再算证据 (须带 calls, 由检查器重放)")

# ===========================================================================
# B/C-12 identity_expand: undersampled + default samples masking a nonzero polynomial
# ===========================================================================
expect_rejected("identity_expand",
                {"diff_terms": {"1": "-2", "2": "-1", "3": "-2", "4": "1"},
                 "samples": ["0", "1", "-1", "2"]},
                "B/C", "样本数 < 次数+1 不再通过 (Schwartz-Zippel 契约)")
expect_rejected("identity_expand", {"diff_terms": {}, "samples": ["0", "1"]},
                "C", "两侧皆空的空真断言不再通过")

# ===========================================================================
# C-13/C-14 linear_backsub / lp_dual: degenerate inputs
# ===========================================================================
expect_rejected("linear_backsub", {"A": [], "b": [], "x": []},
                "C", "空方程组不再让任意 x 通过")
expect_rejected("lp_dual", {"c": [], "A": [], "b": [], "x": [], "y": [], "value": "0"},
                "C", "空维 LP 不再通过")
expect_rejected("lp_dual",
                {"c": ["1"], "A": [[1, 2]], "b": ["1"], "x": ["1"], "y": ["1"], "value": "1"},
                "C", "齿矩阵不再被静默截断后通过")

# ===========================================================================
# C-15 unify_witness: the string-σ **false negative** (a legitimate certificate must pass)
# ===========================================================================
# The term language comes from logiccore.parse_term: variables are written **?x** (prefix ?);
# symbols without ? are constants. The **keys** of σ match the internal variable names of the term
# (`V("?x")` stores "x" internally), so the key is written "x"; variables in values are written
# "?y".
expect_accepted("unify_witness", {"a": "f(?x)", "b": "f(g(?y))", "sigma": {"x": "g(?y)"}},
                "C", "字符串 σ 的合法合一不再被误拒 (假阴性修复)")

# ===========================================================================
# A-01 triple gate gate-3 bypass: with check_box=False a surviving mutant must not be
# silently dropped
# ===========================================================================
# The construction requires three things at once:
#   (1) the original program **really fails** (otherwise the "no patch needed" premise triggers
#       and gate 3 cannot be exercised);
#   (2) a **one-step-reachable** fix exists (literal 3→2);
#   (3) the patched mutants contain **an equivalent item the test set cannot kill** (change the
#       unobserved v1 from 7 to 8 / delete it), used to distinguish "behavior box enabled" from
#       "not enabled".
_tests_weak = [{"args": [1], "expect": 2}, {"args": [2], "expect": 4}]
_buggy = [{"t": "assign", "var": "v0", "e": {"t": "bin", "op": "*", "l": {"t": "var", "n": "a0"},
                                             "r": {"t": "lit", "v": 3}}},
          {"t": "assign", "var": "v1", "e": {"t": "lit", "v": 7}}]
expect_call_err("rep_patch_gen",
                {"program": _buggy, "tests": _tests_weak, "max_candidates": 32,
                 "check_box": False},
                "MUTATION_SURVIVE", "A",
                "check_box=False 不再让门 3 空转 (存活即失败, 不产出假'100%杀死'证书)")
expect_call_ok("rep_patch_gen",
               {"program": _buggy, "tests": _tests_weak, "max_candidates": 32, "check_box": True},
               lambda r: r.value["gate3_mutation"] and r.cert is not None
               and r.cert_independent is True, "A",
               "check_box=True 时补丁仍能过门 3 (证书由检查器复算存活数)")

# ===========================================================================
# D-01 kappa_ub false upper bound: auditor D's counterexample (κ_ub must be >= the true κ)
# ===========================================================================
_A = [["1/10", 0], ["3/10", 0], [0, "1/100"]]
# true κ₂ = σ_max/σ_min; AᵀA = diag(1/10, 1/10000) → σ_max=√(1/10), σ_min=1/100
_true_kappa = Fraction(1, 100) and float((Fraction(1, 10)) ** 0.5 / Fraction(1, 100))
expect_call_ok("cert_lstsq", {"A": _A, "y": [1, 0, 0]},
               lambda r: float(Fraction(r.value["kappa_ub"])) >= _true_kappa,
               "D", f"kappa_ub 不再小于真实 κ₂ ({_true_kappa:.6f})")

# ===========================================================================
# D-02 cert_lp infeasible instance: must not sign "certified optimal"
# ===========================================================================
expect_call_err("cert_lp", {"c": [1], "A": [[1]], "b": [-1]},
                "DEG_UNSUPPORTED", "D", "不可行实例 (b<0) 不再返回'认证最优'")
expect_call_err("cert_lp", {"c": [1], "A": [[-1], [1]], "b": [-2, 1]},
                "DEG_UNSUPPORTED", "D", "不可行实例 (含负 b) 不再返回捏造的 x 与最优值")

# ===========================================================================
# D-03/D-04 Ival: the exp degenerate branch no longer gives a fake interval; sin/cos no
# longer crash
# ===========================================================================
def _ival_checks():
    from svokernel.ops.cert_math import Ival
    out = []
    # e^100: the declared interval must contain the true value (use a high-precision reference)
    from decimal import Decimal, getcontext
    getcontext().prec = 60
    iv = Ival(100, 100).exp()
    true_val = Decimal(100).exp()
    lo, hi = Decimal(iv.lo.numerator) / Decimal(iv.lo.denominator), \
        Decimal(iv.hi.numerator) / Decimal(iv.hi.denominator)
    out.append((lo <= true_val <= hi, "D", "Ival.exp 退化分支 (e^100)",
                f"declared=[{float(lo):.6e},{float(hi):.6e}] true={float(true_val):.6e}"))
    # sin([2,3]): must not raise TypeError, and the interval must contain the true range of sin
    import math as _m
    try:
        s = Ival(2, 3).sin()
        # [2,3] contains π (3.14159? no) → sin is monotonically decreasing on 2..3 (decreasing
        # after π/2≈1.5708)
        lo_true = min(_m.sin(2.0), _m.sin(3.0))
        hi_true = max(_m.sin(2.0), _m.sin(3.0))
        ok = float(s.lo) <= lo_true + 1e-12 and float(s.hi) >= hi_true - 1e-12
        out.append((ok, "D", "Ival.sin([2,3]) 不崩溃且包含真值",
                    f"[{float(s.lo):.9f},{float(s.hi):.9f}] true∈[{lo_true:.9f},{hi_true:.9f}]"))
    except Exception as e:  # noqa: BLE001
        out.append((False, "D", "Ival.sin([2,3]) 不崩溃", f"{type(e).__name__}: {e}"))
    try:
        c = Ival("1/10", "1/2").cos()
        lo_true = min(_m.cos(0.1), _m.cos(0.5))
        hi_true = max(_m.cos(0.1), _m.cos(0.5))
        ok = float(c.lo) <= lo_true + 1e-12 and float(c.hi) >= hi_true - 1e-12
        out.append((ok, "D", "Ival.cos([0.1,0.5]) 包含真值",
                    f"[{float(c.lo):.9f},{float(c.hi):.9f}]"))
    except Exception as e:  # noqa: BLE001
        out.append((False, "D", "Ival.cos([0.1,0.5]) 不崩溃", f"{type(e).__name__}: {e}"))
    return out


for _ok, _who, _desc, _detail in _ival_checks():
    RESULTS.append((_ok, f"[{_who}] {_desc}", _detail))


def main() -> int:
    bad = 0
    for ok, title, detail in RESULTS:
        mark = "[CLOSED]" if ok else "[OPEN  ]"
        if not ok:
            bad += 1
        print(f"{mark} {title}\n          {detail}")
    print("=" * 100)
    print(f"audit-closure regression: {len(RESULTS)} assertions, passed {len(RESULTS) - bad}, open {bad}")
    print("Note: this regression covers the attack surfaces the audit already pointed out; it does not prove 'the certificate system has no holes', it only guards fixed issues against regression.")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

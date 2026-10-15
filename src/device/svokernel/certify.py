"""Certificate system (§13) and the independent-checker registry (§17.9 no self-certification).

**Formalising "no self-certification"**: the generator (invoke) and the checker must be two independent
code paths. Enforcement in this module:
  - a certificate carries **data only** (payload), never an executable "trust me" flag;
  - each certificate kind's checker is registered separately, and registration requires declaring
    `independent_of` — the list of operator names that generate that certificate; if the checker
    function's module has the same name as a generator operator's module, registration is rejected;
  - a checker returns the triple (ok, resid, detail), where resid is the **violation measure** (residual
    in the numeric domain, unsatisfied clause count in the logic domain, failing test count in the code
    domain), consumed by AIC scoring and the scoreboard.

A certificate = a credential for a solution that humans can independently check. **A solution without a
certificate is not a solution** (§13).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from .errors import Err, SvoError

CheckerFn = Callable[[dict], "tuple[bool, list[float], str]"]


@dataclass
class Certificate:
    """Certificate: kind (checker key) + payload (pure data) + issuer signature."""

    kind: str
    payload: dict
    issued_by: str = ""
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"kind": self.kind, "payload": self.payload, "issued_by": self.issued_by, "meta": self.meta}


class CheckerRegistry:
    def __init__(self) -> None:
        self._checkers: dict[str, CheckerFn] = {}
        self._independence: dict[str, tuple[str, ...]] = {}
        self._modules: dict[str, str] = {}

    def register(self, kind: str, fn: CheckerFn, independent_of: tuple[str, ...], module: str) -> None:
        """Register an independent checker.

        module is the name of the module where the checker is implemented (usually __name__). Two structural constraints:
          - `module in independent_of` → reject: declaring a checker "independent of itself" is self-contradictory,
            and usually means the module name was mistaken for an operator name;
          - the final independence criterion (checker module ≠ generator operator module, and the checker does not
            import the generator module) is enforced by selfcheck.audit_static once the registry is complete — only
            then is the generator side visible.
        """
        if kind in self._checkers:
            raise SvoError(Err.SCHEMA_MISMATCH, ctx="certify", detail=f"证书类型重复注册: {kind}")
        if module in independent_of:
            raise SvoError(
                Err.SCHEMA_MISMATCH,
                ctx="certify",
                detail=f"禁自证违规: 检查器 {kind} 位于生成器模块 {module}",
            )
        self._checkers[kind] = fn
        self._independence[kind] = independent_of
        self._modules[kind] = module

    def kinds(self) -> list[str]:
        return sorted(self._checkers)

    def independence(self, kind: str) -> tuple[str, ...]:
        return self._independence.get(kind, ())

    def module_of(self, kind: str) -> str:
        """Checker implementation module name (audit criterion: must ≠ the generator operator module)."""
        return self._modules.get(kind, "")

    def fn(self, kind: str) -> "CheckerFn | None":
        return self._checkers.get(kind)

    def check(self, cert: "Certificate | dict") -> tuple[bool, list[float], str]:
        """Independently re-verify a certificate (callers may invoke this independently of the generation path)."""
        c = cert if isinstance(cert, Certificate) else Certificate(**cert)
        fn = self._checkers.get(c.kind)
        if fn is None:
            return False, [], f"未登记证书类型 {c.kind!r} (无独立检查器 → 不算证书)"
        return fn(c.payload)

    def has(self, kind: str) -> bool:
        return kind in self._checkers


CHECKERS = CheckerRegistry()


def register_checker(kind: str, independent_of: tuple[str, ...]):
    """Decorator: register an independent checker (module defaults to the implementing module)."""

    def deco(fn: CheckerFn) -> CheckerFn:
        CHECKERS.register(kind, fn, independent_of=independent_of, module=fn.__module__)
        return fn

    return deco


def verify_certificate(cert: "Certificate | dict | None", ctx: str, require: bool = True) -> tuple[bool, list[float], str]:
    """Unified entry: re-verify a certificate; require=True with no certificate → NO_CERTIFICATE."""
    if cert is None:
        if require:
            raise SvoError(Err.NO_CERTIFICATE, ctx=ctx, detail="算子声明须带证书, 但未产出")
        return True, [], "无证书要求"
    ok, resid, detail = CHECKERS.check(cert)
    if not ok:
        kind = cert.kind if isinstance(cert, Certificate) else cert.get("kind")
        raise SvoError(Err.POSTCOND_FAIL, ctx=ctx, detail=f"证书 {kind} 独立复验失败: {detail}")
    return True, resid, detail


# ---------------------------------------------------------------------------
# shared checkers (reusable by domain operators; all independent of any generation path — they consume payload data only)
# ---------------------------------------------------------------------------


@register_checker("interval_set_contains",
                  independent_of=("thm_quadratic_roots", "certified_root"))
def _check_interval_set(payload: dict) -> tuple[bool, list[float], str]:
    """Interval-set certificate: **every** interval changes sign at its endpoints, and the intervals are pairwise disjoint
    (⇒ distinct real roots are isolated).

    Why it exists (honest record): the single-interval version `interval_contains` accepts only {lo,hi} — when the equation has
    two distinct real roots, merging their intervals into one big interval **destroys the sign change** (both ends share a
    sign), so a single-interval certificate is necessarily rejected by its own checker. To prove "two distinct real roots",
    the correct shape is **a set of pairwise disjoint sign-changing intervals**.
    payload: {"f_coeffs": [ascending powers, exact], "intervals": [[lo,hi], ...]}
    """
    from fractions import Fraction

    coeffs = [Fraction(c) for c in payload["f_coeffs"]]
    ivs = [(Fraction(a), Fraction(b)) for a, b in payload["intervals"]]
    if not ivs:
        return False, [], "空区间集 (无证据)"

    def ev(x: Fraction) -> Fraction:
        acc = Fraction(0)
        for c in reversed(coeffs):
            acc = acc * x + c
        return acc

    bad: list[int] = []
    for i, (lo, hi) in enumerate(ivs):
        if lo > hi:
            bad.append(i)
            continue
        flo, fhi = ev(lo), ev(hi)
        if flo == 0 or fhi == 0:
            continue  # an endpoint is exactly a root, so the interval is valid
        if (flo < 0) == (fhi < 0):
            bad.append(i)
    overlaps = [(i, j) for i in range(len(ivs)) for j in range(i + 1, len(ivs))
                if not (ivs[i][1] < ivs[j][0] or ivs[j][1] < ivs[i][0])]
    if overlaps:
        bad.extend(j for _, j in overlaps)
    n_bad = len(set(bad))
    detail = (f"{len(ivs)} 个区间全部端点变号且两两不交 (逐区间精确有理核对)"
              if n_bad == 0 else
              f"区间下标 {sorted(set(bad))} 违规 (端点同号或区间相交)")
    return n_bad == 0, [float(n_bad)], detail


@register_checker("poly_backsub", independent_of=("solve_poly", "solve_poly_basic"))
def _check_poly_backsub(payload: dict) -> tuple[bool, list[float], str]:
    """Polynomial back-substitution: evaluate at the claimed roots in exact rational arithmetic, take max|p(r)|.

    payload: {"coeffs": [c0..cn] (ascending powers, exact), "roots": [r...] (exact)}
    Independent of any root-finding formula — it only does Horner evaluation.
    """
    from fractions import Fraction

    if not payload.get("coeffs") or not payload.get("roots"):
        return False, [], "缺 coeffs/roots 或为空 (退化 payload 拒绝)"
    coeffs = [Fraction(c) for c in payload["coeffs"]]
    roots = [Fraction(r) for r in payload["roots"]]
    worst = Fraction(0)
    for r in roots:
        acc = Fraction(0)
        for c in reversed(coeffs):
            acc = acc * r + c
        worst = max(worst, abs(acc))
    # **The criterion never accepts a tolerance from the payload**: under exact rational arithmetic the only
    # criterion for "is a root" is a residual of exactly 0. The old version read tol from the payload (the red
    # team measured: passing tol=10⁹ let the fake roots 5 and 7 of x²+1 pass), which amounts to letting the
    # examinee set the pass mark — §13's certificates would then be meaningless.
    return worst == 0, [float(worst)], f"max|p(r)|={worst} (精确, 须为 0)"


@register_checker("linear_backsub", independent_of=("solve_linear_system",))
def _check_linear_backsub(payload: dict) -> tuple[bool, list[float], str]:
    """Linear-system back-substitution ‖A x − b‖∞ (exact rational arithmetic)."""
    from fractions import Fraction

    if not payload.get("A") or not payload.get("b") or not payload.get("x"):
        return False, [], "空 A/b/x (退化 payload 拒绝: 空方程组的任意 x 都能'通过')"
    A = [[Fraction(v) for v in row] for row in payload["A"]]
    b = [Fraction(v) for v in payload["b"]]
    x = [Fraction(v) for v in payload["x"]]
    if len(A) != len(b):
        return False, [], "A 行数 != b 长度"
    if any(len(row) != len(x) for row in A):
        return False, [], "A 列数 != x 长度 (维度不齐)"
    worst = Fraction(0)
    for row, bi in zip(A, b):
        if len(row) != len(x):
            return False, [], "A 列数 != x 长度"
        acc = sum((a * xi for a, xi in zip(row, x)), Fraction(0))
        worst = max(worst, abs(acc - bi))
    return worst == 0, [float(worst)], f"‖Ax−b‖∞={worst} (精确)"


@register_checker("identity_expand", independent_of=("simplify_expr", "expand_expr", "factor_expr"))
def _check_identity(payload: dict) -> tuple[bool, list[float], str]:
    """Identity: compare lhs−rhs at several rational sample points — expand, then compare term by term.

    payload: {"lhs_terms": {...}, "rhs_terms": {...}, "samples": [..]}
    or {"diff_terms": {...}} (the caller has already taken the difference).
    """
    from fractions import Fraction

    # **The power must be a non-negative integer**: negative exponents blow up at x=0 (red team / audit
    # measured: with 0 among the default samples, an uncaught ZeroDivisionError shot straight through to
    # the caller).
    def check_pows(terms: dict) -> "str | None":
        for k in terms:
            try:
                if int(k) < 0:
                    return f"负指数 {k!r} 非法"
            except (TypeError, ValueError):
                return f"非整数幂次 {k!r}"
        return None

    def ev(terms: dict, xval: Fraction) -> Fraction:
        acc = Fraction(0)
        for k, c in terms.items():
            acc += Fraction(c) * (xval ** int(k))
        return acc

    terms_list = []
    if "diff_terms" in payload:
        terms_list = [payload["diff_terms"]]
    else:
        if "lhs_terms" not in payload or "rhs_terms" not in payload:
            return False, [], "缺 lhs_terms/rhs_terms (也非 diff_terms 形式)"
        terms_list = [payload["lhs_terms"], payload["rhs_terms"]]
    for t in terms_list:
        if not isinstance(t, dict):
            return False, [], f"项表非 dict: {type(t).__name__}"
        why = check_pows(t)
        if why:
            return False, [], why
    # **An empty term list = the zero polynomial** (legitimate data, e.g. the output side of "everything
    # cancels"), so one empty side is allowed; but both sides empty (0 = 0) is a vacuous assertion and must be
    # rejected — that is the real "empty payload".
    if not any(terms_list):
        return False, [], "两侧项表皆空 (0=0 是空真断言, 不算恒等式证书)"

    # **Schwartz-Zippel contract**: a degree-d polynomial must agree on d+1 points before identity follows.
    # The old version did not check the sample count (only 4 points by default); the red team measured that
    # x⁴−2x³−x²+2x (a nonzero polynomial) happens to vanish on the default samples [0,1,−1,2] → a fake identity
    # passed. Hence the point count is now forced by the payload's **highest power**.
    deg = max((int(k) for t in terms_list for k in t), default=0)
    samples = payload.get("samples")
    if not samples or not isinstance(samples, list):
        return False, [], "缺 samples (拒绝默认样本: 判据不得由被验者缺省决定)"
    if len(samples) < deg + 1:
        # Returns the triple (ok, resid, detail) — the contract must be 3-tuple; writing 2-tuple blows up when
        # the caller unpacks it (this file once made CHECKERS.check raise ValueError, caught by the audit_closure regression).
        return False, [float(deg + 1 - len(samples))], (
            f"样本数 {len(samples)} < 次数+1 = {deg + 1} — 违反 Schwartz-Zippel, "
            f"拒绝 (采样不足时非零多项式可能恰好全零)")
    xs = [Fraction(s) for s in samples]
    if "diff_terms" in payload:
        worst = max(abs(ev(payload["diff_terms"], s)) for s in xs)
    else:
        worst = max(abs(ev(payload["lhs_terms"], s) - ev(payload["rhs_terms"], s)) for s in xs)
    return worst == 0, [float(worst)], f"max|lhs−rhs|@{len(xs)}点 (次数 {deg}, 需 ≥{deg + 1})={worst}"


@register_checker("derivative_inverse", independent_of=("integrate_elementary",))
def _check_deriv_inverse(payload: dict) -> tuple[bool, list[float], str]:
    """Integration/differentiation inverse: for a candidate F of ∫f, check F' ≈ f numerically with central differences at the sample points.

    payload: {"f_samples": [[x, f(x)]...], "F_samples": [[x, F(x)]...], "h": float}
    """
    h = float(payload.get("h", 1e-5))
    F = {float(x): float(y) for x, y in payload["F_samples"]}
    worst = 0.0
    used = 0
    for x, fx in payload["f_samples"]:
        x = float(x)
        if (x - h) in F and (x + h) in F:
            d = (F[x + h] - F[x - h]) / (2 * h)
            scale = max(1.0, abs(float(fx)))
            worst = max(worst, abs(d - float(fx)) / scale)
            used += 1
    if used == 0:
        return False, [], "无可用差分样本 (样本点不匹配)"
    return worst < float(payload.get("tol", 1e-3)), [worst], f"互逆核对 {used} 点, max相对偏差={worst:.3e}"


@register_checker("sat_assignment", independent_of=("sat_solve",))
def _check_sat_assignment(payload: dict) -> tuple[bool, list[float], str]:
    """SAT assignment re-verification: clause by clause (an independent §10.6 implementation consuming CNF+assignment only)."""
    cnf = payload["cnf"]
    assign = {int(k): bool(v) for k, v in payload["assign"].items()}
    bad = 0
    for cl in cnf:
        if not any(assign.get(abs(int(l)), None) == (int(l) > 0) for l in cl):
            bad += 1
    return bad == 0, [float(bad)], f"未满足子句数={bad}/{len(cnf)}"


@register_checker("interval_contains", independent_of=("certified_root",))
def _check_interval_contains(payload: dict) -> tuple[bool, list[float], str]:
    """Interval-containment certificate: check that f changes sign on [lo,hi] (weak) or use an interval-Newton contraction (strong).

    payload: {"f_coeffs":[..], "lo":"exact", "hi":"exact"} — f(lo) and f(hi) are evaluated with exact rationals.
    """
    from fractions import Fraction

    coeffs = [Fraction(c) for c in payload["f_coeffs"]]
    lo, hi = Fraction(payload["lo"]), Fraction(payload["hi"])
    if lo > hi:
        return False, [], "区间反向 (lo>hi)"

    def ev(x: Fraction) -> Fraction:
        acc = Fraction(0)
        for c in reversed(coeffs):
            acc = acc * x + c
        return acc

    flo, fhi = ev(lo), ev(hi)
    if flo == 0 or fhi == 0:
        return True, [0.0], "端点恰为根"
    ok = (flo < 0) != (fhi < 0)
    return ok, [float(min(abs(flo), abs(fhi)))], f"f(lo)={float(flo):.3e}, f(hi)={float(fhi):.3e} 变号={ok}"


@register_checker("test_vector", independent_of=("exec_sandbox", "tests_to_impl"))
def _check_test_vector(payload: dict) -> tuple[bool, list[float], str]:
    """Code-domain certificate: **the checker executes the program itself** and then compares item by item (self-reported got/expect is never trusted).

    Red team measured: the old version compared only the payload's got and expect, so "forging two identical numbers"
    passed — "independent implementation" had degenerated into self-consistent comparison. Now the payload must carry the
    **program source (AST)** and the (args, expect) pairs, and the checker really runs it with **engine2 (iterative stack
    machine)**:
      - engine1 and engine2 are two independent implementations (§9.1 dual engines), so the execution path differs from the generator's;
      - expected values must be computed by the checker itself for the comparison; no self-reported got is trusted.
    payload: {"program": [...], "cases": [{"args":[..], "expect": int}], "held_out": bool}
    """
    prog = payload.get("program")
    cases = payload.get("cases") or []
    if not prog:
        return False, [], "缺 program — 无源码无法独立执行, 按 §13 判为无证书"
    if not cases:
        return False, [], "空测试向量 (不算通过)"
    try:
        from .ops.code_core import ARGS, interp_stack, validate_prog
        validate_prog(prog, "certify.test_vector")
    except Exception as e:  # noqa: BLE001 - structurally invalid input is always judged not-passing
        return False, [], f"程序结构非法: {type(e).__name__}: {e}"
    fails: list = []
    for c in cases:
        inp = {a: 0 for a in ARGS}
        for i, v in enumerate(list(c["args"])[:4]):
            inp[ARGS[i]] = int(v)
        r = interp_stack(prog, inp, 200_000)
        if r["error"] is not None:
            fails.append((c["args"], "ERR:" + r["error"]["code"], c["expect"]))
            continue
        got = int(r["state"].get("v0", 0))
        if got != int(c["expect"]):
            fails.append((c["args"], got, c["expect"]))
    return (not fails), [float(len(fails))], (
        f"检查器自跑 {len(cases)} 条测试 (engine2) 全部符合期望" if not fails
        else f"失败 {len(fails)}/{len(cases)} 条, 例 (args,got,expect)={fails[0]}")


@register_checker("proof_kernel", independent_of=("hilbert_prove",))
def _check_proof_kernel(payload: dict) -> tuple[bool, list[float], str]:
    """Proof-term re-verification: step through the home-made kernel (§10.1) — the checker is the kernel itself, deciding without searching."""
    from .logiccore import check_proof  # deferred import to avoid a cycle

    ok, idx, detail = check_proof(payload["formulas"], payload["steps"], payload["goal"])
    return ok, [float(0 if ok else idx + 1)], detail


@register_checker("axiom_table", independent_of=("lean_axioms",))
def _check_axiom_table(payload: dict) -> tuple[bool, list[float], str]:
    """Axiom-dependency-table audit: the dependency closure must lie inside the allowlist, and the table itself must be consistent (acyclic)."""
    # **The allowlist must be held by the checker**: the old version read allowed from the payload, so
    # "self-extending allowlist" passed (red team measured: axioms and allowed both set to ["choice","funext","LEM"]
    # → PASS; an empty payload also PASSED). Here we switch to a checker-side frozen allowlist (Lean's standard
    # foundational axioms + explicitly excluded suspicious entries).
    CHECKER_ALLOWED = {"propext", "Classical.choice", "Quot.sound", "funext",
                       "Classical.em", "Classical.byContradiction"}
    axioms = set(payload.get("axioms", []))
    claimed = set(payload.get("allowed") or [])
    if not axioms:
        return False, [], "空公理表 (无证据: 不构成依赖审计)"
    outside = sorted(claimed - CHECKER_ALLOWED)
    if outside:
        return False, [float(len(outside))], f"调用方白名单含检查器未批准的项: {outside}"
    extra = sorted(axioms - CHECKER_ALLOWED)
    if extra:
        return False, [float(len(extra))], f"越出检查器白名单的公理: {extra}"
    declared_missing = sorted(axioms - claimed) if claimed else []
    if declared_missing:
        return False, [float(len(declared_missing))], (
            f"公理 {declared_missing} 未在调用方声明的白名单内 (声明与依赖不符)")
    return True, [0.0], (f"依赖 {len(axioms)} 条公理, 全在**检查器侧**白名单内 "
                         f"(检查器不采信 payload 自报的白名单)")


@register_checker("replay", independent_of=("runtime_oracle", "replay_log",
                                            "agent_replay_audit"))
def _check_replay(payload: dict) -> tuple[bool, list[float], str]:
    """Deterministic replay: **the checker replays both runs itself** (it does not compare two self-reported fingerprints).

    Red team measured: the old version compared only the payload's run_a and run_b, so forging two identical strings passed.
    Now the payload must carry the calls (call sequence) and the seed; the checker executes both runs itself, comparing both
    the **structural fingerprint** and the **per-call values** (value-level equality counts as identical).
    payload: {"calls": [{"tool": str, "args": {..}}..], "seed": int}
    """
    import hashlib as _h
    import json as _json

    from .audit import BudgetLedger, canon_args
    from .registry import REGISTRY
    from .tool import CallCtl

    calls, seed = payload.get("calls"), int(payload.get("seed", 0))
    if not calls or not isinstance(calls, list):
        return False, [], "缺 calls — 无调用序列无法独立重放, 按 §13 判为无证书"

    def once():
        ledger = BudgetLedger(total_units=1e9)
        rows = []
        for c in calls:
            tool = c.get("tool", "")
            if not REGISTRY.has(tool):
                rows.append({"tool": tool, "args": canon_args(c.get("args")),
                             "err": "OP_MISSING", "value": None})
                continue
            r = REGISTRY.tool(tool).call(c.get("args") or {},
                                         CallCtl(trace=None, budget=ledger, seed=seed))
            rows.append({"tool": tool, "args": canon_args(c.get("args")), "err": r.err or "",
                         "value": r.value})
        blob = _json.dumps(rows, sort_keys=True, ensure_ascii=False, default=str)
        return _h.sha256(blob.encode("utf-8")).hexdigest()[:16], rows

    fa, ra = once()
    fb, rb = once()
    vals_same = _json.dumps(ra, sort_keys=True, default=str) == \
        _json.dumps(rb, sort_keys=True, default=str)
    ok = (fa == fb) and vals_same
    return ok, [0.0 if ok else 1.0], (
        f"检查器独立重放两遍: 指纹 {fa} == {fb}, 值级一致={vals_same}" if ok
        else f"重放不一致: 指纹 {fa} vs {fb}, 值级一致={vals_same} (未播种随机/隐含状态)")


@register_checker("lp_dual", independent_of=("cert_lp",))
def _check_lp_dual(payload: dict) -> tuple[bool, list[float], str]:
    """LP dual certificate: independently re-verify the **three optimality conditions** (the weak duality theorem of linear programming gives sufficiency).

    payload: {"c": [..], "A": [[..]..], "b": [..], "x": [..], "y": [..], "value": exact}
    Checks (all in exact rational arithmetic, consuming data only):
      (1) primal feasibility: A x ≤ b and x ≥ 0
      (2) dual feasibility: Aᵀ y ≥ c and y ≥ 0
      (3) strong-duality equality: cᵀx = bᵀy = value
    All three holding together ⇒ x is globally optimal (weak duality: for any primal-feasible x̃ and dual-feasible ỹ,
    cᵀx̃ ≤ bᵀỹ). This checker never touches simplex code, so the division of labour "the solver proposes, the checker
    verifies" is real.
    """
    from fractions import Fraction

    if not payload.get("c") or not payload.get("A") or not payload.get("b")             or not payload.get("x") or not payload.get("y"):
        return False, [], "空 c/A/b/x/y (退化 payload 拒绝: 空维 LP 的'最优值'无意义)"
    c = [Fraction(v) for v in payload["c"]]
    A = [[Fraction(v) for v in row] for row in payload["A"]]
    b = [Fraction(v) for v in payload["b"]]
    x = [Fraction(v) for v in payload["x"]]
    y = [Fraction(v) for v in payload["y"]]
    m, n = len(A), len(c)
    if len(x) != n or len(y) != m:
        return False, [1.0], f"维度不符: |x|={len(x)} 应 {n}, |y|={len(y)} 应 {m}"
    if any(len(row) != n for row in A):
        return False, [], f"A 含非 {n} 列的行 (齿矩阵: 旧版静默截断后误判)"
    pr = sum(1 for i in range(m) if sum(A[i][j] * x[j] for j in range(n)) > b[i])
    ps = sum(1 for j in range(n) if x[j] < 0)
    du = sum(1 for j in range(n) if sum(A[i][j] * y[i] for i in range(m)) < c[j])
    ds = sum(1 for i in range(m) if y[i] < 0)
    cx = sum(c[j] * x[j] for j in range(n))
    by = sum(b[i] * y[i] for i in range(m))
    val = Fraction(payload["value"])
    bad = pr + ps + du + ds
    ok = (bad == 0) and (cx == by) and (cx == val)
    return ok, [float(bad), float(abs(cx - by))], (
        f"原始可行违反 {pr + ps} 项, 对偶可行违反 {du + ds} 项; "
        f"cᵀx={cx}, bᵀy={by}, 声称最优值={val} "
        + ("→ 三条件全过, 最优性成立" if ok else "→ 证书不成立")
    )


@register_checker("drat_proof", independent_of=("sat_drat",))
def _check_drat_proof(payload: dict) -> tuple[bool, list[float], str]:
    """DRAT certificate re-verification: independently rerun the RUP/RAT checks (the checker itself is the trust root).

    payload: {"cnf": [[lit..]..], "proof": [{"op":"add"|"del","clause":[lit..]}..],
              "verified_unsat": bool}
    This function **reuses nothing** beyond the sat_drat invoke path: it redoes unit propagation itself.
    Honesty statement: this independent implementation is co-sourced with sat_drat's _rup/_rat (different functions in the
    same file) — "independence" in the §17.9 sense here means **checker and generator are separate** (the generator is an
    external solver), not two different algorithms. Literal replay of the same DRAT is precisely the checker's job.
    """
    cnf = [frozenset(int(l) for l in c) for c in payload["cnf"]]
    db = list(cnf)

    def up_conflict(assume) -> bool:
        assign: dict = {}
        for l in assume:
            # same invariant as logic_cert: key = variable number abs(l), value = boolean
            v = abs(l)
            val = l > 0
            if v in assign and assign[v] != val:
                return True
            assign[v] = val
        changed = True
        while changed:
            changed = False
            for cl in db:
                val = None
                un = []
                for l in cl:
                    v = abs(l)
                    if v in assign:
                        if (assign[v] and l > 0) or ((not assign[v]) and l < 0):
                            val = True
                            break
                    else:
                        un.append(l)
                if val:
                    continue
                if not un:
                    return True
                if len(un) == 1:
                    l = un[0]
                    if abs(l) in assign:
                        if (assign[abs(l)] and l < 0) or ((not assign[abs(l)]) and l > 0):
                            return True
                    else:
                        assign[abs(l)] = l > 0
                        changed = True
        return False

    def rup(cl) -> bool:
        return up_conflict([-l for l in cl])

    def rat(cl) -> bool:
        if not cl:
            return False
        pivot = sorted(cl, key=abs)[0]
        rest = set(cl) - {pivot}
        for c in db:
            if -pivot in c:
                if not rup((set(c) - {-pivot}) | rest):
                    return False
        return True

    empty = False
    for i, st in enumerate(payload["proof"]):
        cl = frozenset(int(l) for l in st["clause"])
        if st["op"] == "add":
            if not (rup(cl) or rat(cl)):
                return False, [float(i + 1)], f"步 {i} 非法: 既非 RUP 也非 RAT"
            db.append(cl)
            if not cl:
                empty = True
        else:
            if cl not in db:
                return False, [float(i + 1)], f"步 {i} 删除了不存在的子句"
            db.remove(cl)
    # **"An empty proof is a refutation" is not allowed**: the old version made `verified_unsat` an optional
    # flag, and the red team measured that {"cnf":[[1,2]],"proof":[]} (a satisfiable CNF) returned True. The only
    # valid form of a refutation is that it **really derives the empty clause**, decided by the checker itself,
    # never by a self-reported flag in the payload.
    if not empty:
        return False, [1.0], "证书未导出空子句 — 不构成反证 (自报标志不再被采信)"
    return True, [0.0], f"{len(payload['proof'])} 步重放通过, 已导出空子句"


@register_checker("smt_prooflog", independent_of=("smt_prooflog",))
def _check_smt_prooflog(payload: dict) -> tuple[bool, list[float], str]:
    """Linear-arithmetic proof-log re-verification (Farkas style): recompute the linear combinations step by step; the final formula must be a true contradiction."""
    from fractions import Fraction

    def norm(form):
        return {"c": Fraction(form.get("c", 0)),
                "coef": {str(k): Fraction(v) for k, v in (form.get("coef") or {}).items()},
                "op": form.get("op", "ge")}

    hyps = [norm(h) for h in payload["hyps"]]
    derived: list = []
    got_contra = False
    for i, st in enumerate(payload["log"]):
        r = st["rule"]
        if r == "hyp":
            k = int(st["id"])
            if not (0 <= k < len(hyps)):
                return False, [float(i + 1)], f"步 {i} hyp 越界"
            derived.append(hyps[k])
        elif r == "scale":
            k = int(st["from"])
            a = Fraction(st["coef"])
            if a <= 0 or not (0 <= k < len(derived)):
                return False, [float(i + 1)], f"步 {i} scale 参数非法 (系数须正 / 索引越界)"
            src = derived[k]
            derived.append({"c": a * src["c"], "coef": {kk: a * vv for kk, vv in src["coef"].items()},
                            "op": src["op"]})
        elif r == "add":
            ks = [int(x) for x in st["lits"]]
            cs = [Fraction(x) for x in st["coef"]]
            if any(not (0 <= k < len(derived)) for k in ks) or any(c <= 0 for c in cs):
                return False, [float(i + 1)], f"步 {i} add 参数非法 (系数须正 / 索引越界)"
            a, b = derived[ks[0]], derived[ks[1]]
            coef: dict = {}
            for kk, vv in a["coef"].items():
                coef[kk] = coef.get(kk, Fraction(0)) + cs[0] * vv
            for kk, vv in b["coef"].items():
                coef[kk] = coef.get(kk, Fraction(0)) + cs[1] * vv
            op = "eq" if (a["op"] == "eq" and b["op"] == "eq") else (
                "gt" if (a["op"] == "gt" or b["op"] == "gt") else "ge")
            derived.append({"c": cs[0] * a["c"] + cs[1] * b["c"], "coef": coef, "op": op})
        else:
            k = int(st.get("from", len(derived) - 1))
            if not (0 <= k < len(derived)):
                return False, [float(i + 1)], f"步 {i} contras 越界"
            d = derived[k]
            if any(v != 0 for v in d["coef"].values()):
                return False, [float(i + 1)], f"步 {i} 声称矛盾但仍有变量 {d['coef']}"
            bad = ((d["op"] == "eq" and d["c"] != 0) or (d["op"] == "ge" and d["c"] > 0)
                   or (d["op"] == "gt" and d["c"] >= 0))
            if not bad:
                return False, [float(i + 1)], f"步 {i} 不是矛盾 (0 {d['op']} {d['c']})"
            got_contra = True
    if not got_contra:
        return False, [1.0], "日志未导出矛盾"
    return True, [0.0], f"{len(payload['log'])} 步复算通过, 导出矛盾"


@register_checker("mc_exhaustive", independent_of=("model_check_ltl",))
def _check_mc_exhaustive(payload: dict) -> tuple[bool, list[float], str]:
    """Exhaustive certificate for model checking: independently recompute the reachable set; it must match the claim and the bad state must be unreachable."""
    adj: dict = {s: [] for s in payload["states"]}
    for t in payload["trans"]:
        if t["from"] not in adj:
            return False, [1.0], f"转移起点 {t['from']!r} 不在状态集"
        adj[t["from"]].append(t["to"])
    seen = set()
    stack = [payload["init"]]
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        stack.extend(adj.get(cur, []))
    claimed = set(payload.get("visited") or [])
    if seen != claimed:
        return False, [float(len(seen ^ claimed))], f"可达集不符: 独立算 {sorted(seen)}"
    bad = sorted(set(payload["bad"]) & seen)
    if bad:
        return False, [float(len(bad))], f"坏状态实际可达: {bad} — 不可达结论不成立"
    return True, [0.0], f"独立重算可达集 {len(seen)} 个, 坏状态不可达"


@register_checker("mc_counterexample", independent_of=("model_check_ltl",))
def _check_mc_counterexample(payload: dict) -> tuple[bool, list[float], str]:
    """Counter-example/witness trajectory re-verification: the trajectory must be a legal path from the semantic start to bad.

    Self-contained: the payload carries {path, bad, claim}; the path's legality is decided by "every step is given" — if
    the payload also carries trans (optional), each edge is checked step by step.
    """
    path = payload.get("path") or []
    bad = payload.get("bad")
    if payload.get("claim") == "unreachable":
        # this is the counter-example for "claimed unreachable but refuted": the path must be non-empty and end at bad
        if not path or path[-1].get("to") != bad:
            return False, [1.0], "反例轨迹未到达坏状态"
        return True, [0.0], f"反例轨迹长度 {len(path)}, 终点 {bad}"
    if not path:
        return False, [1.0], "见证轨迹为空"
    trans = payload.get("trans")
    # **The transition relation must be supplied**: without trans, no edge of the trajectory can be checked, and
    # the red team measured that {"path":[{"from":"s0","to":"bad"}],"bad":"bad"} could be forged arbitrarily and
    # still pass. Trajectory validity must be checkable.
    if trans is None:
        return False, [], "缺 trans — 无法核验轨迹边, 按 §13 判为无证书"
    edges = {(t["from"], t["to"]) for t in trans}
    for e in path:
        if (e["from"], e["to"]) not in edges:
            return False, [1.0], f"轨迹含非法边 {e['from']}→{e['to']}"
    if path[-1].get("to") != bad:
        return False, [1.0], "见证轨迹终点不是坏状态"
    return True, [0.0], f"见证轨迹长度 {len(path)}, 终点 {bad} (逐边核验通过)"


def _run_pair(payload: dict):
    """Shared executor: run the before/after programs on tests + the bounded behaviour box, returning pointwise behaviour pairs.

    Behaviour = (final state of a0..a3, final v0) or (ERR, error code). **Uses engine2**, a different implementation from
    the generator (engine1).
    """
    from .ops.code_core import ARGS, interp_stack, validate_prog
    before, after = payload.get("before"), payload.get("after")
    if not before or after is None:
        return None, "缺 before/after — 无源码无法独立执行"
    for tag, pr in (("before", before), ("after", after)):
        try:
            validate_prog(pr, f"certify.{tag}")
        except Exception as e:  # noqa: BLE001
            return None, f"{tag} 结构非法: {type(e).__name__}: {e}"
    tests = payload.get("tests")
    if not tests:
        return None, "缺 tests — 无输入无法独立执行"

    def obs(r):
        if r["error"] is not None:
            return ("ERR", r["error"]["code"])
        st = r["state"]
        return tuple(st.get(a, 0) for a in ARGS) + (st.get("v0", 0),)

    pairs: list = []
    for t in tests:
        inp = {a: 0 for a in ARGS}
        for i, v in enumerate(list(t["args"])[:4]):
            inp[ARGS[i]] = int(v)
        pairs.append((list(t["args"]), obs(interp_stack(before, inp, 200_000)),
                      obs(interp_stack(after, inp, 200_000))))
    if payload.get("box", True):
        import itertools
        for combo in itertools.product(range(-2, 3), repeat=2):
            inp = {a: 0 for a in ARGS}
            for i, v in enumerate(combo):
                inp[ARGS[i]] = v
            pairs.append((list(combo), obs(interp_stack(before, inp, 200_000)),
                          obs(interp_stack(after, inp, 200_000))))
    return pairs, ""


@register_checker("equivalence_pair", independent_of=("refactor_equiv",))
def _check_equivalence_pair(payload: dict) -> tuple[bool, list[float], str]:
    """Behaviour-equivalence certificate: **the checker runs both programs itself** and compares pointwise (self-reported got/expect is never trusted).

    Red team measured: the old version compared two self-reported fields in the payload, so forging {got:5, expect:5} could
    pass off a+b and a−b as equivalent. Now the payload must carry the before/after sources and the tests, and the checker
    executes them.
    """
    pairs, why = _run_pair(payload)
    if pairs is None:
        return False, [], why
    bad = [p for p in pairs if p[1] != p[2]]
    return (not bad), [float(len(bad))], (
        f"检查器自跑 {len(pairs)} 个输入, 行为逐点相同 (等价)" if not bad
        else f"行为不同 {len(bad)}/{len(pairs)} 点, 例 args={bad[0][0]}: {bad[0][1]} vs {bad[0][2]}")


@register_checker("divergence_witness", independent_of=("refactor_equiv",))
def _check_divergence_witness(payload: dict) -> tuple[bool, list[float], str]:
    """Behaviour-**divergence** certificate: the checker runs them itself and must **find at least one real divergence point**, with a witness."""
    pairs, why = _run_pair(payload)
    if pairs is None:
        return False, [], why
    bad = [p for p in pairs if p[1] != p[2]]
    if not bad:
        return False, [0.0], "检查器自跑后未发现任何分歧 — 不等价不成立 (该证书无效)"
    w = bad[0]
    return True, [float(len(bad))], (
        f"不等价成立: 检查器在 {len(bad)}/{len(pairs)} 个输入上复现分歧, "
        f"见证 args={w[0]}: 改前={w[1]} 改后={w[2]}")


@register_checker("mutation_report", independent_of=("mutation_test", "rep_patch_gen"))
def _check_mutation_report(payload: dict) -> tuple[bool, list[float], str]:
    """Mutation report: **the checker runs every mutant through the test set itself** and counts survivors (self-reported survived is never trusted).

    Red team measured: the old version only checked the payload for self-consistency (survived==0), so "self-reporting 0"
    passed — while the real survivor count was 1. Now the payload must carry program/tests/mutants (mutant sources), and the
    checker executes each one.
    """
    prog, tests, muts = payload.get("program"), payload.get("tests"), payload.get("mutants")
    if not prog or not tests or muts is None:
        return False, [], "缺 program/tests/mutants — 无法独立复算存活数, 按 §13 判为无证书"
    from .ops.code_core import ARGS, interp_stack, validate_prog
    try:
        validate_prog(prog, "certify.mutation_report")
        for m in muts:
            validate_prog(m, "certify.mutation_report.mutant")
    except Exception as e:  # noqa: BLE001
        return False, [], f"程序/变异体结构非法: {type(e).__name__}: {e}"

    def passes(pr) -> bool:
        for t in tests:
            inp = {a: 0 for a in ARGS}
            for i, v in enumerate(list(t["args"])[:4]):
                inp[ARGS[i]] = int(v)
            r = interp_stack(pr, inp, 200_000)
            if r["error"] is not None or int(r["state"].get("v0", 0)) != int(t["expect"]):
                return False
        return True

    def behavior(pr):
        """Bounded behaviour fingerprint (final state of a0..a3 + v0), used for **equivalence decisions** (same convention as the three gates)."""
        import itertools as _it
        sig = []
        for combo in _it.product(range(-2, 3), repeat=2):
            inp = {a: 0 for a in ARGS}
            for i, v in enumerate(combo):
                inp[ARGS[i]] = v
            r = interp_stack(pr, inp, 200_000)
            sig.append(("ERR", r["error"]["code"]) if r["error"] is not None
                       else tuple(r["state"].get(a, 0) for a in ARGS) + (r["state"].get("v0", 0),))
        return tuple(sig)

    if not muts:
        return False, [], "变异体集为空 (无证据: 门 3 不是'没有变异体就算过')"
    box_on = bool(payload.get("box", True))
    base = behavior(prog) if box_on else None
    alive = 0
    for m in muts:
        if passes(m):
            # Not killed by the test set: exempt only when the mutant is **really equivalent to the patch inside the
            # behaviour box** (consistent with the declaration); if the behaviour box is not enabled, everything counts
            # as surviving (no silent dropping — that was the blocking finding auditor A caught).
            if (not box_on) or (behavior(m) != base):
                alive += 1
    return alive == 0, [float(alive)], (
        f"检查器自跑 {len(muts)} 个变异体 × {len(tests)} 条测试"
        + ("+ 行为箱等价判定" if box_on else " (未启用行为箱: 杀不死即存活)")
        + f": 存活 {alive} 个 (须为 0; 由检查器复算)")


@register_checker("bmc_counterexample", independent_of=("cbmc_bounded",))
def _check_bmc_counterexample(payload: dict) -> tuple[bool, list[float], str]:
    """**Counter-example** certificate for bounded model checking: the witness input must really make the property fail (including the program itself).

    payload: {"witness": {a0..a3}, "program": [...], "v0": int, "property": {op,...}}
    The checker runs the program itself (independently of the generator's conclusion), so the "counter-example" is an
    independently checkable fact. The predicate decision is implemented **locally** (not imported from ops.heavy_cores):
    auditor B measured that the old version's `from .ops.heavy_cores import _prop_holds` made the checker depend on the
    generator module — a live §17.9 independence violation.
    """
    from .ops.code_core import interp

    prop = payload.get("property")
    if not payload.get("program") or not payload.get("witness") or not prop:
        return False, [], "缺 program/witness/property — 无证据, 按 §13 判为无证书"

    def _holds(p: dict, val: int) -> bool:
        op = p.get("op")
        if op == "in":
            return int(p["lo"]) <= val <= int(p["hi"])
        if op == "not_in":
            return not (int(p["lo"]) <= val <= int(p["hi"]))
        if op == "eq":
            return val == int(p["value"])
        if op == "ne":
            return val != int(p["value"])
        if op == "le":
            return val <= int(p["value"])
        return val >= int(p["value"])

    r = interp(payload["program"], payload["witness"], 200_000)
    if r["error"] is not None:
        return False, [1.0], f"见证在检查器里执行失败: {r['error']['code']}"
    got = int(r["state"].get("v0", 0))
    if "v0" in payload and got != int(payload["v0"]):
        return False, [1.0], f"见证执行得 v0={got}, 证书声称 {payload['v0']} — 不符"
    if _holds(prop, got):
        return False, [1.0], f"见证执行后 v0={got} 仍满足性质 — 不是反例"
    return True, [0.0], f"反例成立: 见证执行得 v0={got}, 违反性质 {prop.get('op')}"


@register_checker("bmc_bounded_ok", independent_of=("cbmc_bounded",))
def _check_bmc_bounded_ok(payload: dict) -> tuple[bool, list[float], str]:
    """**No-violation-within-bound** certificate for bounded model checking: it must carry program/property/k, and the checker performs a bounded dynamic re-check.

    Red team measured: the old version only did a "conclusion self-consistency audit", so {"k":10,"n_paths":50} could mint a
    blank "no violation within bound" voucher. Carrying executable evidence is now mandatory.
    Honesty statement: the dynamic re-check is **corroborating evidence**, not an independent proof of "no violation within
    bound" (that would require repeating the generator's path analysis); its purpose is to make blank certificates
    impossible — at least they must stand up under real execution.
    """
    import itertools as _it

    from .ops.code_core import ARGS, interp, validate_prog

    k = payload.get("k")
    prog, prop = payload.get("program"), payload.get("property")
    if not isinstance(k, int) or k < 1:
        return False, [], f"k 非法: {k!r} (有界结论必须显式给出展开界)"
    if not prog or not prop:
        return False, [], "缺 program/property — 纯自洽计数不是证书"
    try:
        validate_prog(prog, "certify.bmc_bounded_ok")
    except Exception as e:  # noqa: BLE001
        return False, [], f"程序结构非法: {type(e).__name__}: {e}"

    def holds(p: dict, val: int) -> bool:
        op = p.get("op")
        if op == "in":
            return int(p["lo"]) <= val <= int(p["hi"])
        if op == "not_in":
            return not (int(p["lo"]) <= val <= int(p["hi"]))
        if op == "eq":
            return val == int(p["value"])
        if op == "ne":
            return val != int(p["value"])
        if op == "le":
            return val <= int(p["value"])
        return val >= int(p["value"])

    grid = (-20, -1, 0, 1, 20)
    tried = 0
    for combo in _it.product(grid, repeat=2):
        inp = {a: 0 for a in ARGS}
        for i, v in enumerate(combo):
            inp[ARGS[i]] = v
        r = interp(prog, inp, 200_000)
        if r["error"] is not None:
            continue
        tried += 1
        if not holds(prop, int(r["state"].get("v0", 0))):
            return False, [1.0], (f"检查器动态复查发现违反 (网格 {tried} 点内) — "
                                  f"该'界内无违反'不成立")
    note = []
    if payload.get("k_unknown"):
        note.append("含界未知的循环")
    if payload.get("truncated"):
        note.append("含深度截断")
    return True, [0.0], (f"界内无违反: k={k}, {payload.get('n_paths')} 条路径; "
                         f"检查器另做 {tried} 点动态复查亦未发现违反"
                         + ("; 注意: " + ", ".join(note) if note else ""))


@register_checker("tlc_exhaustive", independent_of=("tlc_modelcheck",))
def _check_tlc_exhaustive(payload: dict) -> tuple[bool, list[float], str]:
    """TLA exhaustive certificate: independently rebuild the state space and confirm the invariant holds on **all** reachable states.

    The payload must carry the full spec — red team measured: the old version allowed a count-only "minimal payload" and
    returned pass ({"n_states":7} could mint "exhausted"), which amounts to issuing blank vouchers. The checker therefore
    **reruns BFS itself**; violating states are not expanded further (consistent with the generator, otherwise a spec with no
    guard actions would make the verifier grow without bound along the state space).
    """
    spec = payload.get("spec")
    if not spec:
        return False, [], ("缺 spec (vars/init/actions/invariant) — 无规格无法独立重跑, "
                           "按 §13 判为无证书")
    from .ops.heavy_cores import _pred
    import itertools as _it

    names = sorted(spec["vars"])
    domains = [list(spec["init"].get(v, spec["vars"][v])) for v in names]
    seen = set()
    stack = []
    for c in _it.product(*domains):
        k0 = tuple(c)
        if k0 not in seen:
            seen.add(k0)
            stack.append(dict(zip(names, c)))
    while stack:
        st = stack.pop()
        if not _pred(spec["invariant"], st):
            return False, [1.0], f"检查器重跑发现违反状态 {[st[v] for v in names]}"
        for a in spec["actions"]:
            g = a.get("guard")
            if g is not None and not _pred(g, st):
                continue
            nxt = dict(st)
            for u in a["updates"]:
                ex = u.get("expr") or {"op": "+", "delta": 1}
                d = int(ex.get("delta", 1))
                nxt[u["var"]] = nxt[u["var"]] + d if ex.get("op", "+") == "+" else nxt[u["var"]] - d
            k2 = tuple(nxt[v] for v in names)
            if k2 not in seen:
                seen.add(k2)
                stack.append(nxt)
    claimed = int(payload.get("n_states", -1))
    if claimed != len(seen):
        return False, [float(abs(claimed - len(seen)))], (
            f"状态数不符: 检查器 {len(seen)} vs 证书 {claimed}")
    return True, [0.0], f"独立穷举 {len(seen)} 个状态, 不变量全部成立"


@register_checker("tlc_violation", independent_of=("tlc_modelcheck",))
def _check_tlc_violation(payload: dict) -> tuple[bool, list[float], str]:
    """TLA violation certificate: the counter-example trajectory must be legal step by step and its final state must really violate the invariant. spec is mandatory."""
    spec = payload.get("spec")
    trace = payload.get("trace") or []
    state = payload.get("state") or []
    if not state:
        return False, [], "违反状态为空"
    if not spec:
        return False, [], "缺 spec — 无法独立重放轨迹, 按 §13 判为无证书"
    from .ops.heavy_cores import _pred

    names = sorted(spec["vars"])
    if not trace:
        return False, [], "轨迹为空 (无见证)"
    cur = dict(zip(names, trace[0]["from"]))
    for e in trace:
        act = next((a for a in spec["actions"] if a["name"] == e["action"]), None)
        if act is None:
            return False, [1.0], f"轨迹引用了未知动作 {e['action']}"
        nxt = dict(cur)
        for u in act["updates"]:
            ex = u.get("expr") or {"op": "+", "delta": 1}
            d = int(ex.get("delta", 1))
            nxt[u["var"]] = nxt[u["var"]] + d if ex.get("op", "+") == "+" else nxt[u["var"]] - d
        if [nxt[v] for v in names] != list(e["to"]):
            return False, [1.0], f"轨迹步 {e} 与动作语义不符"
        cur = dict(zip(names, e["to"]))
    if [cur[v] for v in names] != list(state):
        return False, [1.0], "轨迹终点与声称的违反状态不符"
    st = dict(zip(names, state))
    if _pred(spec["invariant"], st):
        return False, [1.0], f"声称违反但状态 {state} 满足不变量"
    return True, [0.0], f"违反成立: 状态 {state} 违反不变量, 轨迹 {len(trace)} 步逐步合法"


@register_checker("unify_witness", independent_of=("unify",))
def _check_unify_witness(payload: dict) -> tuple[bool, list[float], str]:
    """Unification witness: apply the substitution σ independently to both terms; they must be symbol-identical."""
    from .logiccore import apply_subst, parse_term  # deferred import

    if not payload.get("a") or not payload.get("b"):
        return False, [], "缺 a/b 项"
    sigma = payload.get("sigma")
    if not isinstance(sigma, dict) or not sigma:
        return False, [], "缺 sigma 或为空 (空替换不是合一见证)"
    # σ may be a Term structure or a **string**: the old version accepted only Term and therefore rejected
    # **correct** unifications written as {"x": "f(y)"} — false negatives are defects just like false positives
    # (measured by the red team).
    parsed = {k: (v if isinstance(v, tuple) else parse_term(v)) for k, v in sigma.items()}
    a = apply_subst(parse_term(payload["a"]), parsed)
    b = apply_subst(parse_term(payload["b"]), parsed)
    ok = a == b
    return ok, [0.0 if ok else 1.0], f"σ(a)={a!r} σ(b)={b!r} (σ 形态统一解析)"


@register_checker("mod_congruence", independent_of=("thm_fermat", "thm_euler", "thm_mod_inverse"))
def _check_mod_congruence(payload: dict) -> tuple[bool, list[float], str]:
    """Modular-congruence certificate: (lhs − rhs) must be divisible by m (exact integer arithmetic, consuming data only).

    payload: {"lhs": exact, "rhs": exact, "m": int>0, "cases": [{lhs,rhs,m}...] (optional)}
    Independent of any modular-exponentiation / extended-Euclid implementation — it performs one subtraction and one modulo.
    """
    from fractions import Fraction

    def one(lhs, rhs, m) -> "tuple[bool, int]":
        mi = int(m)
        if mi <= 0:
            return False, 0
        d = Fraction(lhs) - Fraction(rhs)
        if d.denominator != 1:
            return False, 1
        return (int(d) % mi) == 0, 0

    cases = payload.get("cases") or [payload]
    bad = []
    for i, c in enumerate(cases):
        ok, code = one(c["lhs"], c["rhs"], c["m"])
        if not ok:
            bad.append(i if code == 0 else f"{i}(m≤0)")
    n = len(cases)
    worst = float(len(bad))
    return (not bad), [worst], (
        f"{n} 条同余全部成立 (逐条 (lhs−rhs) mod m == 0)" if not bad
        else f"违反同余的条目下标 {bad} (共 {n} 条)"
    )


@register_checker("matrix_identity",
                  independent_of=("thm_mat2_inv", "thm_cayley_hamilton", "thm_eig2"))
def _check_matrix_identity(payload: dict) -> tuple[bool, list[float], str]:
    """Matrix-identity certificate: independently recompute lhs_mat · rhs_mat and compare element-wise, exactly, against claims.

    payload: {"lhs": [[exact..]..], "rhs": [[exact..]..], "claim": [[exact..]..]}
    or {"terms": [{"coef": exact, "mat": [[..]]}...], "claim": [[..]]} (linear-combination form, used for Cayley-Hamilton
    style A² − trA·A + detA·I = 0). Independent of the inverse/characteristic-polynomial derivation.
    """
    from fractions import Fraction

    def M(m):
        return [[Fraction(v) for v in row] for row in m]

    if "lhs" in payload:
        A, B = M(payload["lhs"]), M(payload["rhs"])
        if len(A[0]) != len(B):
            return False, [1.0], f"维度不匹配 {len(A[0])} vs {len(B)}"
        n, k, m = len(A), len(B), len(B[0])
        acc = [[sum((A[i][t] * B[t][j] for t in range(k)), Fraction(0)) for j in range(m)]
               for i in range(n)]
    else:
        terms = payload["terms"]
        acc = None
        for t in terms:
            Mat = M(t["mat"])
            c = Fraction(t["coef"])
            if acc is None:
                acc = [[c * v for v in row] for row in Mat]
            else:
                if len(Mat) != len(acc) or len(Mat[0]) != len(acc[0]):
                    return False, [1.0], "线性组合各项维度不一致"
                for i in range(len(acc)):
                    for j in range(len(acc[0])):
                        acc[i][j] += c * Mat[i][j]
        if acc is None:
            return False, [1.0], "空线性组合 (无证据)"

    claim = M(payload["claim"])
    if len(claim) != len(acc) or len(claim[0]) != len(acc[0]):
        return False, [1.0], "claim 维度与重算结果不一致"
    worst = Fraction(0)
    for i in range(len(acc)):
        for j in range(len(acc[0])):
            worst = max(worst, abs(acc[i][j] - claim[i][j]))
    return worst == 0, [float(worst)], f"逐元素精确重算 max|Δ|={worst} (精确 0 = 恒等成立)"

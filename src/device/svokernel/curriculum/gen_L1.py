"""§8.4 L1 math exam generator (curriculum/gen_L1.py).

Four families, all **deterministically generated + answers known + machine-verifiable**:

  forward  forward: quadratic coefficients → roots          check: thm_quadratic_roots / solve_poly_exact
  reverse  reverse: roots → monic quadratic                 check: math_expand_expr  (with an **identifiability argument**)
  mixed    mixed: ≤2-step composition (solve → Vieta sum/product)  check: thm_vieta + thm_quadratic_roots
  template template-instantiated contest problems (AM-GM / inclusion-exclusion / convergence test)
           check: thm_amgm / thm_inclusion_exclusion / thm_ratio_test

Design constraints (brief §8.4 + §17.3):
  1. **Determinism**: only random.Random(seed); same (n, seed, families) → bit-identical exam set;
  2. **Answers known**: each problem ships its answer, and the answer is recomputed by an operator
     **independent of the generation path**;
  3. **Explicit identifiability argument**: inverse problems must provide an identifiability string —
     why the unique solution can be identified. Problems that cannot be argued **never enter the set**
     (the generator does not produce them in the first place);
  4. **Machine-verifiability must be measured**: verify_all() actually runs each problem's check spec
     and returns pass/fail counts. Claiming 100% verifiable without running is forbidden.

Honesty boundary (item by item):
  - the identifiability argument of the reverse family is **complete** within the family: a monic
    quadratic is uniquely determined by its roots (Vieta), so "roots → polynomial" has a unique
    solution, i.e. it is identifiable; without the monic constraint the solution set is a whole scaling
    orbit and **not identifiable** (§17.3 asks exactly for this kind of argument). This family therefore
    writes monicity into the problem statement and states in the identifiability field that "dropping
    monicity makes it unidentifiable".
  - the "contest problems" of the template family are template instantiations, not real contest
    problems; grading relies on the checking operators, never on human review.
  - each problem declares only **machine-checkable** properties; goals like "elegant proof" are excluded.
"""
from __future__ import annotations

import random
from fractions import Fraction

from ..errors import Err, SvoError
from ..exact import frac_str

MAX_FAMILY = ("forward", "reverse", "mixed", "template")


def _fs(x) -> str:
    return frac_str(Fraction(x))


def _gen_forward(rng: random.Random, idx: int) -> dict:
    """Forward: integer roots of x² + p x + q. Answer = the two roots (exact)."""
    r1 = rng.randint(-12, 12)
    r2 = rng.randint(-12, 12)
    p = -(r1 + r2)
    q = r1 * r2
    # A repeated root appears once — same measurement convention as thm_quadratic_roots (single-element
    # root list when r1==r2); without dedup the fwd family reports "expected two roots, got one" on
    # repeated-root problems (8 problems observed in practice).
    roots = sorted({r1, r2})
    return {
        "id": f"L1-fwd-{idx:05d}",
        "family": "forward",
        "statement": f"解方程 x² + ({p})x + ({q}) = 0, 给出全部实根",
        "answer": {"roots": [str(r) for r in roots], "p": p, "q": q},
        "check": {"op": "thm_quadratic_roots", "args": {"coeffs": [q, p, 1]},
                  "assert": {"roots": [str(r) for r in roots], "mode": "exact_roots"}},
        "identifiability": "正向: 判别式=D>0 时实根集唯一; 本题按构造 D=(r1−r2)²≥0 且为完全平方",
    }


def _gen_reverse(rng: random.Random, idx: int) -> dict:
    """Reverse: two roots → the **monic** quadratic.

    Identifiability: monicity pins the solution set from "a whole scaling orbit" to a single point —
    if we only required "roots r1,r2", then λ(x−r1)(x−r2) is a solution for every λ≠0, which is not
    identifiable. Hence the statement explicitly requires monic.
    """
    r1 = rng.randint(-9, 9)
    r2 = rng.randint(-9, 9)
    p = -(r1 + r2)
    q = r1 * r2
    return {
        "id": f"L1-rev-{idx:05d}",
        "family": "reverse",
        "statement": f"求以 {r1} 与 {r2} 为根的首一二次多项式 x² + px + q",
        "answer": {"p": str(p), "q": str(q)},
        "check": {"op": "math_expand_expr",
                  "args": {"factors": [[{"c": -r1, "p": 0}, {"c": 1, "p": 1}],
                                       [{"c": -r2, "p": 0}, {"c": 1, "p": 1}]]},
                  # The assertion must match the operator's output convention: math_expand_expr
                  # **omits zero-coefficient terms** (without skipping them when p=0 or q=0, the
                  # assertion compares 3 terms against 2 — the rev family failed on this in practice).
                  "assert": {"terms": [t for t in ({"c": "1", "p": 2},
                                                   {"c": str(p), "p": 1},
                                                   {"c": str(q), "p": 0}) if t["c"] != "0"]}},
        "identifiability": ("首一约束下解唯一 (韦达: p=−(r1+r2), q=r1r2 ⇒ p,q 被根唯一确定); "
                            "**去掉首一性则不可辨识** —— λ(x−r1)(x−r2) 对任意 λ≠0 同根, "
                            "解集是一维伸缩轨道。故题面必须写'首一'"),
    }


def _gen_mixed(rng: random.Random, idx: int) -> dict:
    """Mixed (2-step composition): solve the quadratic, then check sum and product with Vieta. Answer = (roots, sum, product)."""
    r1 = rng.randint(-10, 10)
    r2 = rng.randint(-10, 10)
    p = -(r1 + r2)
    q = r1 * r2
    roots = sorted([r1, r2])
    return {
        "id": f"L1-mix-{idx:05d}",
        "family": "mixed",
        "statement": f"解 x² + ({p})x + ({q}) = 0, 并用韦达定理核对两根之和与积",
        "answer": {"roots": [str(r) for r in roots], "sum": str(r1 + r2), "prod": str(r1 * r2),
                   "coeffs": [q, p, 1]},
        "check": {"op": "thm_vieta", "args": {"coeffs": [q, p, 1], "roots": [r1, r2]},
                  "assert": {"matches": True, "sum_claim": str(r1 + r2), "prod_claim": str(r1 * r2)}},
        "identifiability": "两步组合: 第 1 步 (求根) 由正向族保证唯一; 第 2 步 (韦达) 是恒等式核对, 不引入歧义",
    }


def _gen_template(rng: random.Random, idx: int) -> dict:
    """Template-instantiated contest problem: one of three (AM-GM inequality / inclusion-exclusion count / convergence test)."""
    kind = rng.choice(["amgm", "incl_excl", "ratio"])
    if kind == "amgm":
        xs = [rng.randint(1, 9) for _ in range(rng.randint(2, 4))]
        s = sum(xs)
        prod = 1
        for x in xs:
            prod *= x
        n = len(xs)
        # Denominator-free form: (Σx)ⁿ ≥ nⁿ∏x, gap = (Σx)ⁿ − nⁿ∏x
        gap = s ** n - (n ** n) * prod
        return {
            "id": f"L1-tpl-{idx:05d}",
            "family": "template",
            "statement": f"证明 (并给出差值): 对 x={xs}, 有 (Σx)^{n} ≥ {n}^{n}·∏x",
            "answer": {"sum": str(s), "prod": str(prod), "gap": str(gap), "n": n},
            "check": {"op": "thm_amgm", "args": {"xs": xs},
                      "assert": {"am": _fs(Fraction(s, n)), "prod": str(prod), "gap": str(gap)}},
            "identifiability": "模板实例化: 差值 (Σx)ⁿ−nⁿ∏x 是 x 的整式, 唯一确定 (无自由参数)",
        }
    if kind == "incl_excl":
        A = sorted(rng.sample(range(1, 12), rng.randint(1, 5)))
        B = sorted(rng.sample(range(1, 12), rng.randint(1, 5)))
        C = sorted(rng.sample(range(1, 12), rng.randint(1, 5)))
        U = len(set(A) | set(B) | set(C))
        return {
            "id": f"L1-tpl-{idx:05d}",
            "family": "template",
            "statement": f"求 |A∪B∪C|: A={A}, B={B}, C={C}",
            "answer": {"union": U},
            "check": {"op": "thm_inclusion_exclusion", "args": {"A": A, "B": B, "C": C},
                      "assert": {"union": U}},
            "identifiability": "有限集合并集基数唯一确定 (集合语义, 重复元素不计)",
        }
    r = Fraction(rng.randint(1, 5), rng.randint(2, 6))
    return {
        "id": f"L1-tpl-{idx:05d}",
        "family": "template",
        "statement": f"判定级数 aₙ = {_fs(r)}ⁿ 的收敛性 (比值判别法)",
        "answer": {"L": _fs(r), "verdict": "converges" if r < 1 else ("diverges" if r > 1 else "undecided")},
        "check": {"op": "thm_ratio_test", "args": {"c": 1, "r": _fs(r), "P": [1], "Q": [1]},
                  "assert": {"L": _fs(r), "verdict": "converges" if r < 1 else ("diverges" if r > 1 else "undecided")}},
        "identifiability": "几何族 aₙ=rⁿ 的比值极限 L=|r| 唯一; L=1 时判别法失效 → 答案显式写 undecided",
    }


_GENS = {"forward": _gen_forward, "reverse": _gen_reverse, "mixed": _gen_mixed,
         "template": _gen_template}

#: Default order: equal-weight rotation across families (keeps the exam set balanced across families).
_CYCLE = ("forward", "reverse", "mixed", "template")


def generate(n: int, seed: int = 20260920, families=None) -> "list[dict]":
    """Generate n problems. Same (n, seed, families) → bit-identical (only random.Random(seed) is consumed)."""
    if n < 0:
        raise SvoError(Err.SCHEMA_MISMATCH, ctx="gen_L1.generate", detail=f"n={n} 为负")
    fams = list(families) if families else list(_CYCLE)
    for f_ in fams:
        if f_ not in _GENS:
            raise SvoError(Err.SCHEMA_MISMATCH, ctx="gen_L1.generate",
                           detail=f"未知族 {f_!r} (允许 {tuple(_GENS)})")
    rng = random.Random(seed)
    out: list[dict] = []
    for i in range(n):
        fam = fams[i % len(fams)]
        item = _GENS[fam](rng, i)
        if not item.get("identifiability"):
            # Source-level gate: a problem without an identifiability argument never enters the set (§17.3)
            raise SvoError(Err.UNIDENTIFIABLE, ctx="gen_L1.generate",
                           detail=f"题 {item['id']} 缺可辨识性论证 — 不得入集")
        out.append(item)
    return out


def verify_one(item: dict) -> "tuple[bool, str]":
    """Actually run one problem's check spec: call the registry operator and assert its output matches `answer`."""
    from ..registry import REGISTRY

    spec = item["check"]
    op = spec["op"]
    if not REGISTRY.has(op):
        return False, f"校验算子 {op!r} 未注册"
    r = REGISTRY.tool(op).call(spec["args"])
    if not r.ok:
        return False, f"{op} 调用失败: {r.err} {r.notes}"
    bad = []
    for k, want in spec["assert"].items():
        got = r.value.get(k)
        if got != want:
            bad.append(f"{k}: 期望 {want!r}, 实得 {got!r}")
    return (not bad), ("校验通过: " + op if not bad else f"{op} 断言不符: {bad}")


def verify_all(items: "list[dict]", limit: "int | None" = None) -> dict:
    """Actually run the verification of all problems; return counts and failure details."""
    n = len(items) if limit is None else min(limit, len(items))
    failures: list[dict] = []
    passed = 0
    for it in items[:n]:
        ok, detail = verify_one(it)
        if ok:
            passed += 1
        else:
            failures.append({"id": it["id"], "detail": detail})
    return {"total": n, "passed": passed, "failed": n - passed,
            "rate": (passed / n) if n else 1.0, "failures": failures[:20]}


def manifest(items: "list[dict]") -> dict:
    """Exam-set summary: problem count / per-family distribution / fingerprint (for reproduction checks)."""
    from . import fingerprint

    fams: dict = {}
    for it in items:
        fams[it["family"]] = fams.get(it["family"], 0) + 1
    return {"n": len(items), "by_family": dict(sorted(fams.items())),
            "fingerprint": fingerprint(items)}


#: Selfcheck spec (lets the selfcheck module run a small batch with no arguments)
SELFCHECK_N = 40

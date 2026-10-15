"""§9.7 code exam generator (curriculum/gen_SE.py).

Three **machine-verifiable** task families (the three listed in brief §9.7):

  defect_fix     defect injection: mutate a defect out of a correct program; the patch must pass the
                 three gates (only "one-step reversible" classes are injected: arith_flip/boundary_pm1/
                 cond_negate; the inverses of lit_set and stmt_delete are outside the patch search space
                 → not injected; their difficulty is published in the §16.7 benchmark's "full-scope
                 measurement convention")
  refactor       refactoring equivalence: given before/after, decide "behavior-preserving equivalence"
  spec_impl      specification implementation: given spec + tests, pick the correct implementation
                 among candidates

**Honesty boundary of defect types (must be spelled out item by item, because the brief's list does not
match this language's capabilities)**: brief §9.7 lists five classes "off-by-one / null reference / type /
logic / resource leak". This language (§9.1's bounded AST) has **no** references, type system or heap —
therefore:
  - **off-by-one (off_by_one)**: representable ✓ → injected via literal ±1
  - **wrong constant (wrong_const)**: representable ✓ → injected by setting a literal to 0/1
  - **wrong operator (wrong_op)**: representable ✓ → injected by flipping an arithmetic/comparison operator
  - **condition negation (cond_negate)**: representable ✓ → comparison operator negated (counts as "logic" class)
  - **statement deletion (stmt_drop)**: representable ✓ (counts as "logic" class)
  - **null reference (null_deref)**: **not representable** — the language has no references/pointers, so the
    state "dereference a null pointer" does not exist
  - **type error (type_error)**: **not representable** — the language is single-type integer, with no type
    annotations to violate
  - **resource leak (resource_leak)**: **not representable** — the language has no heap allocation or file
    handles, so there is no resource to leak
The latter three are **not generated**. This is not laziness: generating a "null-reference defect" that
cannot be defined in this language's semantics would amount to fabricating problems. Covering those three
classes requires first adding reference/type/resource models to the language (an architectural extension of
§9.1), which is a task-specification-level decision — already registered in the delivery report's dissent
section.

**Determinism**: only random.Random(seed); same (n, seed, families) → bit-identical exam set.
**Every problem ships its answer and truth label**: the defect family records the injected defect class
(ground truth) for top-k hit-rate evaluation.
"""
from __future__ import annotations

import random

from ..errors import Err, SvoError
from ..ops.code_core import interp
from ..ops.code_repair import _patch_candidates, _mk_inputs

#: Representable defect classes (mapping to the brief's list is in the module docstring)
REPRESENTABLE = ("off_by_one", "wrong_const", "wrong_op", "cond_negate", "stmt_drop")
#: Defect classes listed in the brief but **not representable** in this language (not generated, declared explicitly)
NOT_REPRESENTABLE = ("null_deref", "type_error", "resource_leak")

_MUT2CLASS = {"boundary_pm1": "off_by_one", "lit_set": "wrong_const",
              "arith_flip": "wrong_op", "cond_negate": "cond_negate",
              "stmt_delete": "stmt_drop"}

#: Seed program templates (correct) + test input sets
_TEMPLATES: tuple[tuple, ...] = (
    ([{"t": "assign", "var": "v0", "e": {"t": "bin", "op": "+", "l": {"t": "var", "n": "a0"},
                                        "r": {"t": "var", "n": "a1"}}}],
     [[i, j] for i in range(0, 5) for j in range(0, 5)]),
    ([{"t": "assign", "var": "v0", "e": {"t": "bin", "op": "*", "l": {"t": "var", "n": "a0"},
                                        "r": {"t": "lit", "v": 2}}}],
     [[i] for i in range(0, 8)]),
    ([{"t": "assign", "var": "v0", "e": {"t": "bin", "op": "-", "l": {"t": "var", "n": "a0"},
                                        "r": {"t": "lit", "v": 3}}}],
     [[i] for i in range(0, 9)]),
    ([{"t": "if", "c": {"t": "bin", "op": "<", "l": {"t": "var", "n": "a0"},
                        "r": {"t": "lit", "v": 0}},
       "th": [{"t": "assign", "var": "v0", "e": {"t": "lit", "v": -1}}],
       "el": [{"t": "assign", "var": "v0", "e": {"t": "lit", "v": 1}}]}],
     [[i] for i in range(-5, 6)]),
)


def _tests_for(good, inputs) -> list:
    out = []
    for inp in inputs:
        r = interp(good, _mk_inputs(list(inp)), 200_000)
        if r["error"] is not None:
            continue
        out.append({"args": list(inp), "expect": int(r["state"].get("v0", 0))})
    return out


def _gen_defect(rng: random.Random, idx: int) -> dict:
    """Defect-injection problem: inject only from **reversible classes** (defect and fix both inside the patch search space); truth = defect class."""
    good, inputs = _TEMPLATES[rng.randrange(len(_TEMPLATES))]
    tests = _tests_for(good, inputs)
    if len(tests) < 4:
        raise SvoError(Err.PRECOND_FAIL, ctx="gen_SE", detail="模板测试不足 4 条 — 拒绝产题")
    # **Inject only "one-step reversible" defect classes**: the defect and its fix must both lie in the
    # patch search space, otherwise the problem itself is unsolvable (the benchmark measures problems, not
    # operators). Reversible classes = arith_flip / boundary_pm1 / cond_negate; lit_set (e.g. 10→0, whose
    # inverse 0→10 is not in the space) and stmt_delete (its inverse must rebuild statements) are **not
    # injected**. The true repair difficulty of these two classes is measured and published by the §16.7
    # benchmark's "full-scope measurement convention" (measured lit_set 0/11, stmt_delete 0/6) instead of
    # being disguised as solvable here — auditor F noted that at n=200, 187/200 failures came from these two
    # classes mixed in (all 13 cases unsolvable).
    _REVERSIBLE = ("arith_flip", "boundary_pm1", "cond_negate")
    cands = [c for c in _patch_candidates(good, 64) if c[0] in _REVERSIBLE]
    if not cands:
        raise SvoError(Err.PRECOND_FAIL, ctx="gen_SE", detail="模板无可注入的缺陷 — 拒绝产题")
    kind, desc, buggy = cands[rng.randrange(len(cands))]
    cls = _MUT2CLASS[kind]
    return {"id": f"SE-def-{idx:05d}", "family": "defect_fix", "defect_class": cls,
            "mutant_kind": kind, "mutant_desc": desc,
            "buggy": buggy, "tests": tests,
            "statement": f"修复缺陷 ({cls}): 使全部 {len(tests)} 条测试通过, 且补丁需过三重门",
            "check": {"op": "rep_patch_gen",
                      "args": {"program": buggy, "tests": tests, "max_candidates": 32,
                               "check_box": True},
                      "assert": {"gate1_train": True, "gate2_heldout": True,
                                 "gate3_mutation": True}},
            "identifiability": ("缺陷可辨识性: 测试集在**注入点**的邻域上区分正确与错误程序 "
                                f"(变异类 {kind} 在 {len(tests)} 条测试上被杀死); "
                                "补丁的唯一性不作保证 (可能有多个等价修复), 故验收看三重门而非看补丁形式")}


def _gen_refactor(rng: random.Random, idx: int) -> dict:
    """Refactoring-equivalence problem: half equivalent, half not (the non-equivalent cases test that the decider does not always answer "equivalent")."""
    good, inputs = _TEMPLATES[rng.randrange(len(_TEMPLATES))]
    tests = _tests_for(good, inputs)
    equal_case = bool(rng.getrandbits(1))
    if equal_case:
        # An equivalent refactoring must **really preserve behavior**. Two constructions:
        #   (a) the template is a single assignment → extract the subexpression into v1, then assign
        #       to v0 (v1 is never observed, so behavior is unchanged);
        #   (b) any other template (with if/loop) → only **append** a dead assignment to the
        #       unobserved slot v1 (the old version used (a) for all templates, so the if template
        #       became a semantically different program while still marked "expected equivalent";
        #       half of the 10/12 failures came from here).
        if len(good) == 1 and good[0]["t"] == "assign":
            inner = good[0]["e"]
            after = [{"t": "assign", "var": "v1", "e": inner},
                     {"t": "assign", "var": "v0", "e": {"t": "var", "n": "v1"}}]
        else:
            after = list(good) + [{"t": "assign", "var": "v1", "e": {"t": "lit", "v": 3}}]
        expect_equiv = True
    else:
        # Non-equivalent case: **overwrite v0** after the correct program (with a0+7), which changes the
        # final output for every template: v0's value in the original program cannot be identically a0+7
        # (a0 ranges over values other than 0 and 7).
        after = list(good) + [{"t": "assign", "var": "v0",
                               "e": {"t": "bin", "op": "+", "l": {"t": "var", "n": "a0"},
                                     "r": {"t": "lit", "v": 7}}}]
        expect_equiv = False
    return {"id": f"SE-ref-{idx:05d}", "family": "refactor", "expect_equivalent": expect_equiv,
            "before": good, "after": after,
            "tests": tests,
            "statement": "判定改前/改后是否**行为保持等价** (测试集 + 有界行为箱)",
            "check": {"op": "refactor_equiv",
                      "args": {"before": good, "after": after, "tests": tests, "box": True},
                      "assert": {"equivalent": expect_equiv}},
            "identifiability": ("等价性的可辨识性: 在**测试集 + 行为箱 (arg∈[−2,2])** 上比对"
                                "可观测行为 (输入槽终态, v0, 错误码); 范围写进 scope 字段 —— "
                                "箱外的行为差异不在本判据覆盖内 (这是操作定义, 不是逻辑等价)")}


def _gen_spec_impl(rng: random.Random, idx: int) -> dict:
    """Specification-implementation problem: given spec and tests, exactly one candidate is correct → pick it."""
    good, inputs = _TEMPLATES[rng.randrange(len(_TEMPLATES))]
    tests = _tests_for(good, inputs)
    wrongs = [c for c in _patch_candidates(good, 64) if c[0] in _MUT2CLASS]
    if not wrongs:
        raise SvoError(Err.PRECOND_FAIL, ctx="gen_SE", detail="无可作干扰项的变异 — 拒绝产题")
    _k, _d, wrong = wrongs[rng.randrange(len(wrongs))]
    cands = [wrong, good]
    rng.shuffle(cands)
    truth = cands.index(good)
    return {"id": f"SE-spec-{idx:05d}", "family": "spec_impl", "truth_index": truth,
            "signature": {"fn": "f", "arity": 2,
                          "params": [{"name": "a0", "type": "int", "lo": -6, "hi": 6},
                                     {"name": "a1", "type": "int", "lo": -6, "hi": 6}],
                          "output": {"type": "int", "lo": -100, "hi": 100}},
            "tests": tests, "candidates": cands,
            "statement": "从候选中选出满足规格 (通过全部测试) 的实现",
            "check": {"op": "tests_to_impl",
                      "args": {"signature": {"fn": "f", "arity": 2,
                                             "params": [{"name": "a0", "type": "int", "lo": -6, "hi": 6},
                                                        {"name": "a1", "type": "int", "lo": -6, "hi": 6}],
                                             "output": {"type": "int", "lo": -100, "hi": 100}},
                               "tests": tests, "candidates": cands, "held_out": 6},
                      "assert": {"gate1_tests": True, "gate2_heldout": True,
                                 "gate3_mutation": True}},
            "identifiability": (f"规格可辨识性: 干扰项与正确实现在**给定测试**上可区分 "
                                f"(干扰项被测试杀死), 且 held-out 补点进一步约束; "
                                f"真值下标随题给出 (truth_index={truth}) 供评分")}


_GENS = {"defect_fix": _gen_defect, "refactor": _gen_refactor, "spec_impl": _gen_spec_impl}
_CYCLE = ("defect_fix", "refactor", "spec_impl")


def generate(n: int, seed: int = 20260920, families=None) -> "list[dict]":
    if n < 0:
        raise SvoError(Err.SCHEMA_MISMATCH, ctx="gen_SE.generate", detail=f"n={n} 为负")
    fams = list(families) if families else list(_CYCLE)
    for f_ in fams:
        if f_ not in _GENS:
            raise SvoError(Err.SCHEMA_MISMATCH, ctx="gen_SE.generate", detail=f"未知族 {f_!r}")
    rng = random.Random(seed)
    out: list = []
    for i in range(n):
        item = _GENS[fams[i % len(fams)]](rng, i)
        if not item.get("identifiability"):
            raise SvoError(Err.UNIDENTIFIABLE, ctx="gen_SE.generate",
                           detail=f"题 {item['id']} 缺可辨识性论证 — 不得入集")
        out.append(item)
    return out


def verify_one(item: dict) -> "tuple[bool, str]":
    """Actually run the problem's acceptance operator and compare assertions field by field (including ground truths such as expect_equivalent / truth_index)."""
    from ..registry import REGISTRY

    spec = item["check"]
    if not REGISTRY.has(spec["op"]):
        return False, f"验收算子 {spec['op']!r} 未注册"
    r = REGISTRY.tool(spec["op"]).call(spec["args"])
    if not r.ok:
        return False, f"{spec['op']} 调用失败: {r.err} {r.notes[:120]}"
    bad = [f"{k}: 期望 {v!r} 实得 {r.value.get(k)!r}" for k, v in spec["assert"].items()
           if r.value.get(k) != v]
    if bad:
        return False, f"{spec['op']} 断言不符: {bad}"
    if item["family"] == "spec_impl":
        # The selected item must be the candidate pointed at by the truth index (the old version wrote
        # `impl not in ([cand],)`, wrapping one extra list → never equal, so all 4/4 spec_impl problems
        # failed spuriously)
        if r.value.get("impl") != item["candidates"][item["truth_index"]]:
            return False, "tests_to_impl 选中的实现不是真值下标所指的候选"
    return True, f"校验通过: {spec['op']}"


def verify_all(items: "list[dict]", limit: "int | None" = None) -> dict:
    n = len(items) if limit is None else min(limit, len(items))
    failures: list = []
    passed = 0
    for it in items[:n]:
        ok, detail = verify_one(it)
        if ok:
            passed += 1
        else:
            failures.append({"id": it["id"], "family": it["family"], "detail": detail})
    by_family: dict = {}
    for it in items[:n]:
        by_family.setdefault(it["family"], [0, 0])
        by_family[it["family"]][1] += 1
    for f_ in by_family:
        by_family[f_] = {"total": by_family[f_][1]}
    return {"total": n, "passed": passed, "failed": n - passed,
            "rate": (passed / n) if n else 1.0, "failures": failures[:20],
            "not_representable_defect_classes": list(NOT_REPRESENTABLE)}


def manifest(items: "list[dict]") -> dict:
    from . import fingerprint
    fams: dict = {}
    classes: dict = {}
    for it in items:
        fams[it["family"]] = fams.get(it["family"], 0) + 1
        if it.get("defect_class"):
            classes[it["defect_class"]] = classes.get(it["defect_class"], 0) + 1
    return {"n": len(items), "by_family": dict(sorted(fams.items())),
            "by_defect_class": dict(sorted(classes.items())),
            "representable": list(REPRESENTABLE), "not_representable": list(NOT_REPRESENTABLE),
            "fingerprint": fingerprint(items)}

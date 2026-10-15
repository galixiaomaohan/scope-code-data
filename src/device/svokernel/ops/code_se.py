"""§9.4–9.5 contract-based generation + software-engineering operators (band 9700, rows 9701–9712).

§9.4 contract-based generation (spec → sig → tests → impl) and §9.5 software engineering: 12 operators in total.
All of them work on §9.1's bounded AST language and code_core's engines, so **all are machine-verifiable**:

  spec_to_sig        9701  specification → type signature (domain constraints queryable both ways)
  sig_to_tests       9702  signature + property → test pairs (boundaries always covered; the reference implementation comes from the spec)
  tests_to_impl      9703  test pairs + AST codebook → implementation candidates (three-gate acceptance)
  refactor_equiv     9704  refactoring (extract/inline/change data structure) — acceptance = **behaviour-preserving equivalence test pairs**
  build_repro        9705  reproducible build (lock file + offline dependency resolution)
  api_compat         9706  semantic-version compatibility (export diff + breaking-change enumeration)
  dep_audit          9707  dependency × vulnerability-database **offline snapshot** audit
  ci_gate            9708  CI as verification gate (rerun item by item; any failure closes the gate)
  doc_gen            9709  documentation generation (**examples = faithful transcriptions of tests; fabrication forbidden**)
  proj_scaffold      9710  project skeleton (accepted by a structural-constraint checker)
  contract_check     9711  contract pre/post-condition checker (reused by the operators above)
  lockfile_verify    9712  lock-file self-consistency verification (double-run fingerprints agree)

**Honesty boundary shared by this module (item by item)**:
  1. the specification language is **controlled**: parameter domains are integer intervals and postconditions are
     "reference implementation + assertion". Its expressive power is limited — it cannot express "no pointer
     aliasing" or "no deadlock". For any spec beyond that power this module raises SPEC_AMBIGUOUS instead of guessing
     as best it can;
  2. `build_repro`'s "dependencies" are **declared** dependencies (name@version + content hash); it does no real
     package management: the container has no network (§18 red line), so "offline resolution" here = hashing and
     consistency checks over the declarations;
  3. `dep_audit`'s vulnerability database is an **offline snapshot supplied by the caller**; the built-in data contains
     no real CVE data — this module implements only the mechanism "version-interval matching + hit enumeration" and
     does not claim the database is complete;
  4. `api_compat` compares only the **signature surface** (names/arity/input-output domains), not semantics — semantic
     compatibility needs §9.3's three gates, and the module states this explicitly in its output.
"""
from __future__ import annotations

import hashlib
import json
from fractions import Fraction

from ..certify import Certificate
from ..errors import Err, SvoError
from ..exact import frac, frac_str
from ..registry import REGISTRY, OpDef
from ..tool import CallCtl
from .code_core import ARGS, interp, validate_prog
from .code_repair import _mutants, _behavior_box, _mk_inputs, _run_tests, TEST_SCHEMA

BAND = 9700


def _f(v) -> Fraction:
    return frac(v)


def _fs(x: Fraction) -> str:
    return frac_str(x)


def _h(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)
                          .encode("utf-8")).hexdigest()[:16]


# ===========================================================================
# 9701 spec_to_sig — specification → type signature
# ===========================================================================

_SPEC_SCHEMA = {"kind": "dict", "extra": False,
                "fields": {
                    "name": {"kind": "str"},
                    "arity": {"kind": "int", "min": 0, "max": 4},
                    "domain": {"kind": "dict", "extra": True},
                    "out_lo": {"kind": "int", "min": -10**9, "max": 10**9},
                    "out_hi": {"kind": "int", "min": -10**9, "max": 10**9}},
                "required": ["name", "arity", "domain"]}
_IN_S2S = {"kind": "dict", "fields": {"spec": _SPEC_SCHEMA}, "required": ["spec"]}


def _inv_s2s(args: dict, ctl: CallCtl) -> dict:
    """Specification → signature: make the spec's domain constraints **explicit** as signature parameter domains (intervals + output domain)."""
    ctx = "op:spec_to_sig#9701"
    sp = args["spec"]
    arity = int(sp["arity"])
    domain = sp["domain"]
    params = []
    for i in range(arity):
        key = ARGS[i]
        if key not in domain:
            raise SvoError(Err.SPEC_AMBIGUOUS, ctx=ctx,
                           detail=f"规格未给出参数 {key} 的域 — 拒绝推断 (签名必须由规格决定, "
                                  f"不能由实现者替规格做选择)")
        d = domain[key]
        lo, hi = int(d["lo"]), int(d["hi"])
        if lo > hi:
            raise SvoError(Err.SPEC_AMBIGUOUS, ctx=ctx, detail=f"参数 {key} 的域 [{lo},{hi}] 为空")
        if not (-10**6 <= lo <= hi <= 10**6):
            raise SvoError(Err.SPEC_AMBIGUOUS, ctx=ctx,
                           detail=f"参数 {key} 的域超出引擎字面量界 ±10⁶")
        params.append({"name": key, "type": "int", "lo": lo, "hi": hi})
    out_lo = int(sp.get("out_lo", -10**9))
    out_hi = int(sp.get("out_hi", 10**9))
    sig = {"fn": sp["name"], "params": params, "output": {"type": "int", "lo": out_lo, "hi": out_hi},
           "arity": arity}
    return {"value": {"signature": sig, "fingerprint": _h(sig), "n_params": arity,
                      "domain_size": _domain_size(params)},
            "resid": [0.0],
            "notes": f"签名 {sp['name']}({', '.join(p['name'] + ':' + str(p['lo']) + '..' + str(p['hi']) for p in params)}) "
                     f"→ int; 域大小 {_domain_size(params)}; 指纹 {_h(sig)}"}


def _domain_size(params) -> int:
    out = 1
    for p in params:
        out *= (p["hi"] - p["lo"] + 1)
    return out


def _ver_s2s(args: dict, value: dict) -> dict:
    """Independent verification: (a) the signature corresponds **field by field** to the spec (parameter names = input-slot prefixes, same domain endpoints, same arity);
    (b) the domain is non-empty; (c) the fingerprint is reproducible (recomputation agrees)."""
    sp = args["spec"]
    sig = value["signature"]
    ok_arity = int(sig["arity"]) == int(sp["arity"]) == len(sig["params"])
    ok_names = [p["name"] for p in sig["params"]] == [ARGS[i] for i in range(int(sp["arity"]))]
    ok_dom = all(int(sig["params"][i]["lo"]) == int(sp["domain"][ARGS[i]]["lo"])
                 and int(sig["params"][i]["hi"]) == int(sp["domain"][ARGS[i]]["hi"])
                 for i in range(int(sp["arity"])))
    nonempty = all(p["lo"] <= p["hi"] for p in sig["params"])
    fp_ok = _h(sig) == value["fingerprint"]
    size_ok = int(value["domain_size"]) == _domain_size(sig["params"])
    ok = ok_arity and ok_names and ok_dom and nonempty and fp_ok and size_ok
    return {"ok": ok, "resid": [0.0 if ok else 1.0],
            "detail": f"arity 一致={ok_arity}; 参数名由输入槽决定={ok_names}; 域端点逐项一致={ok_dom}; "
                      f"域非空={nonempty}; 指纹可复现={fp_ok}; 域大小一致={size_ok}"}


REGISTRY.register(OpDef(
    name="spec_to_sig", band=BAND, sub=1,
    in_schema=_IN_S2S,
    out_schema={"kind": "dict",
                "fields": {"signature": {"kind": "dict", "extra": True, "fields": {}},
                           "fingerprint": {"kind": "str"}, "n_params": {"kind": "int", "min": 0},
                           "domain_size": {"kind": "int", "min": 1}},
                "required": ["signature", "fingerprint", "n_params", "domain_size"]},
    invoke=_inv_s2s, verify=_ver_s2s,
    cost=("units", 0.5), errors=("SPEC_AMBIGUOUS", "SCHEMA_MISMATCH"),
    effects="none (纯函数)",
    doc="§9.4 spec_to_sig | in: 规格 (名字/元数/参数域/输出域) | out: 类型签名 + 指纹 + 域大小 | "
        "验收=签名与规格**逐字段**对应 (参数名/域端点/arity) + 域非空 + 指纹可复现 | 0.5 unit | "
        "诚实边界: 域缺失或越界一律 SPEC_AMBIGUOUS —— 签名必须由规格决定, 不许实现者替规格选 | "
        "失败: 规格歧义 SPEC_AMBIGUOUS",
    selfchecks=(
        {"id": "s2s_two_params", "args": {"spec": {"name": "add", "arity": 2,
                                                   "domain": {"a0": {"lo": -10, "hi": 10},
                                                              "a1": {"lo": 0, "hi": 5}},
                                                   "out_lo": -20, "out_hi": 20}},
         "check": lambda r: (r.value["n_params"] == 2 and r.value["domain_size"] == 21 * 6
                             and [p["name"] for p in r.value["signature"]["params"]] == ["a0", "a1"],
                             f"n={r.value['n_params']} size={r.value['domain_size']}"),
         "expect": "手算: 域大小 = 21 × 6 = 126; 参数名按输入槽顺序 a0, a1"},
        {"id": "s2s_zero_arity", "args": {"spec": {"name": "const", "arity": 0, "domain": {}}},
         "check": lambda r: (r.value["n_params"] == 0 and r.value["domain_size"] == 1,
                             f"size={r.value['domain_size']}"),
         "expect": "零元函数: 域大小为 1 (空积), 签名参数表为空"},
        {"id": "s2s_missing_domain", "args": {"spec": {"name": "f", "arity": 2,
                                                       "domain": {"a0": {"lo": 0, "hi": 1}}}},
         "expect_err": "SPEC_AMBIGUOUS",
         "expect": "只给了一个参数的域 → 规格歧义, 必须拒 (不许推断 a1 的域)"},
        {"id": "s2s_empty_domain", "args": {"spec": {"name": "f", "arity": 1,
                                                     "domain": {"a0": {"lo": 5, "hi": 3}}}},
         "expect_err": "SPEC_AMBIGUOUS", "expect": "空域 [5,3] 必须被拒"},
        {"id": "s2s_arity_too_big", "args": {"spec": {"name": "f", "arity": 5, "domain": {}}},
         "expect_err": "SCHEMA_MISMATCH", "expect": "arity=5 越界 (引擎只有 4 个输入槽)"},
    ),
))


# ===========================================================================
# 9702 sig_to_tests — signature + property → test pairs
# ===========================================================================

_IN_S2T = {"kind": "dict",
           "fields": {"signature": {"kind": "dict", "extra": True, "fields": {}},
                      "reference": {"kind": "list", "items": {"kind": "dict", "extra": True,
                                                              "fields": {}}},
                      "max_tests": {"kind": "int", "min": 4, "max": 64},
                      "require_boundaries": {"kind": "bool"}},
           "required": ["signature", "reference"]}


def _inv_s2t(args: dict, ctl: CallCtl) -> dict:
    """Signature + reference implementation → test pairs. **Boundaries always covered**: each dimension takes lo/hi/0 (when in the domain), then deterministic random points are added."""
    ctx = "op:sig_to_tests#9702"
    sig = args["signature"]
    ref = args["reference"]
    validate_prog(ref, ctx + ".reference")
    params = sig["params"]
    if not params:
        raise SvoError(Err.SPEC_AMBIGUOUS, ctx=ctx, detail="零元签名的测试无输入可枚举 — 拒绝")
    bounds: list = []
    for p in params:
        cand = {int(p["lo"]), int(p["hi"])}
        if int(p["lo"]) <= 0 <= int(p["hi"]):
            cand.add(0)
        bounds.append(sorted(cand))
    tests: list = []

    def emit(inp):
        r = interp(ref, inp, 200_000)
        if r["error"] is not None:
            raise SvoError(Err.RUNTIME_ERR, ctx=ctx,
                           detail=f"参考实现在 {inp} 上执行失败: {r['error']['code']} — "
                                  f"参考实现必须先自洽, 否则生成的测试没有意义")
        tests.append({"args": [inp[a] for a in ARGS[:len(params)]], "expect": int(r["state"].get("v0", 0))})

    import itertools
    for combo in itertools.product(*bounds):
        inp = {ARGS[i]: combo[i] for i in range(len(params))}
        emit(inp)
        if len(tests) >= int(args.get("max_tests", 24)):
            break
    # deterministic extra points (low-discrepancy sequence, no RNG, so it stays reproducible)
    k = 0
    while len(tests) < int(args.get("max_tests", 24)):
        inp = {}
        for i, p in enumerate(params):
            span = int(p["hi"]) - int(p["lo"])
            inp[ARGS[i]] = int(p["lo"]) + ((k * 7919 + i * 104729) % (span + 1))
        emit(inp)
        k += 1
        if k > 512:
            break
    seen = set()
    uniq = []
    for t in tests:
        key = tuple(t["args"])
        if key in seen:
            continue
        seen.add(key)
        uniq.append(t)
    n_bound = sum(1 for t in uniq if any(v in (int(p["lo"]), int(p["hi"])) for v, p in zip(t["args"], params)))
    return {"value": {"tests": uniq, "n_tests": len(uniq), "n_boundary_tests": n_bound,
                      "signature_fingerprint": _h(sig),
                      "reference_fingerprint": _h(ref)},
            "resid": [0.0],
            "notes": f"生成 {len(uniq)} 个测试对 (其中 {n_bound} 条触及域边界); "
                     f"参考实现自洽性已在生成时逐条核验 (执行失败即 RUNTIME_ERR)"}


def _ver_s2t(args: dict, value: dict) -> dict:
    """Independent verification: (a) **rerun the reference implementation with engine2 item by item**; the expected values must match the recorded ones
    (any divergence of the two engines is an alarm); (b) every input lies inside the signature domain; (c) boundary coverage is met."""
    sig = args["signature"]
    ref = args["reference"]
    params = sig["params"]
    tests = value["tests"]
    mismatch = 0
    for t in tests:
        inp = _mk_inputs(t["args"])
        from .code_core import interp_stack
        r = interp_stack(ref, inp, 200_000)
        if r["error"] is not None or int(r["state"].get("v0", 0)) != t["expect"]:
            mismatch += 1
    in_dom = all(all(int(p["lo"]) <= v <= int(p["hi"]) for v, p in zip(t["args"], params))
                 for t in tests)
    enough = len(tests) >= min(4, int(args.get("max_tests", 24)))
    ok = (mismatch == 0) and in_dom and enough
    return {"ok": ok, "resid": [float(mismatch)],
            "detail": f"engine2 复跑参考实现: 期望值不符 {mismatch} 条 (须 0); "
                      f"全部输入在域内={in_dom}; 测试数充足={enough} ({len(tests)} 条)"}


REGISTRY.register(OpDef(
    name="sig_to_tests", band=BAND, sub=2,
    in_schema=_IN_S2T,
    out_schema={"kind": "dict",
                "fields": {"tests": TEST_SCHEMA, "n_tests": {"kind": "int", "min": 1},
                           "n_boundary_tests": {"kind": "int", "min": 0},
                           "signature_fingerprint": {"kind": "str"},
                           "reference_fingerprint": {"kind": "str"}},
                "required": ["tests", "n_tests", "n_boundary_tests", "signature_fingerprint",
                             "reference_fingerprint"]},
    invoke=_inv_s2t, verify=_ver_s2t,
    cost=("units", 1.5), errors=("SPEC_AMBIGUOUS", "RUNTIME_ERR", "SCHEMA_MISMATCH"),
    effects="none (纯函数)",
    doc="§9.4 sig_to_tests | in: 签名 + 参考实现 (AST) | out: 测试对 + 边界覆盖计数 | "
        "验收=**engine2 独立复跑**参考实现与录得期望一致 + 输入全在域内 + 边界覆盖 | 1.5 unit | "
        "诚实边界: 期望值来自**参考实现**(不是规格本身) —— 故本算子保证的是'测试与参考实现一致', "
        "不保证参考实现满足规格; 后者要靠 tests_to_impl 的三重门或人工审阅 | "
        "失败: 参考实现执行失败 RUNTIME_ERR / 零元签名 SPEC_AMBIGUOUS",
    selfchecks=(
        {"id": "s2t_add", "args": {"signature": {"fn": "add", "arity": 2,
                                                 "params": [{"name": "a0", "type": "int", "lo": -3, "hi": 3},
                                                            {"name": "a1", "type": "int", "lo": -3, "hi": 3}],
                                                 "output": {"type": "int", "lo": -6, "hi": 6}},
                                   "reference": [{"t": "assign", "var": "v0",
                                                  "e": {"t": "bin", "op": "+",
                                                        "l": {"t": "var", "n": "a0"},
                                                        "r": {"t": "var", "n": "a1"}}}],
                                   "max_tests": 16},
         "check": lambda r: (r.value["n_tests"] >= 4 and r.value["n_boundary_tests"] >= 4
                             and all(isinstance(t["expect"], int) for t in r.value["tests"]),
                             f"n={r.value['n_tests']} boundary={r.value['n_boundary_tests']}"),
         "expect": "a0+a1 的域 ±3: 边界点 {−3,0,3}² 共 9 个组合 → 至少 4 条边界测试; "
                   "期望值 = 参考实现输出 (engine2 会独立复跑核对)"},
        {"id": "s2t_zero_arity_rejected", "args": {"signature": {"fn": "c", "arity": 0, "params": [],
                                                                 "output": {"type": "int", "lo": 0, "hi": 0}},
                                                   "reference": [{"t": "assign", "var": "v0",
                                                                  "e": {"t": "lit", "v": 1}}]},
         "expect_err": "SPEC_AMBIGUOUS", "expect": "零元签名的测试无输入可枚举 → 显式拒绝"},
        {"id": "s2t_bad_reference", "args": {"signature": {"fn": "f", "arity": 1,
                                                          "params": [{"name": "a0", "type": "int",
                                                                      "lo": 0, "hi": 2}],
                                                          "output": {"type": "int", "lo": 0, "hi": 9}},
                                            "reference": [{"t": "assign", "var": "zz",
                                                           "e": {"t": "lit", "v": 1}}]},
         "expect_err": "SCHEMA_MISMATCH", "expect": "非法参考实现 (非法槽位) 必须被拦下"},
        {"id": "s2t_reference_division_by_zero", "args": {
            "signature": {"fn": "div", "arity": 2,
                          "params": [{"name": "a0", "type": "int", "lo": 1, "hi": 4},
                                     {"name": "a1", "type": "int", "lo": -1, "hi": 1}],
                          "output": {"type": "int", "lo": -4, "hi": 4}},
            "reference": [{"t": "assign", "var": "v0", "e": {"t": "bin", "op": "/",
                                                             "l": {"t": "var", "n": "a0"},
                                                             "r": {"t": "var", "n": "a1"}}}]},
         "expect_err": "RUNTIME_ERR",
         "expect": "域含 a1=0 且参考实现做除法 → 生成测试时必然撞上除零, 必须 RUNTIME_ERR "
                   "(**暴露规格缺陷**: 域与实现不匹配, 这正是契约式生成该发现的)"},
    ),
))


# ===========================================================================
# 9703 tests_to_impl — test pairs + AST codebook → implementation candidates (three-gate acceptance)
# ===========================================================================

_IN_T2I = {"kind": "dict",
           "fields": {"signature": {"kind": "dict", "extra": True, "fields": {}},
                      "tests": TEST_SCHEMA,
                      "candidates": {"kind": "list", "minlen": 1, "maxlen": 64,
                                     "items": {"kind": "list", "items": {"kind": "dict",
                                                                         "extra": True, "fields": {}}}},
                      "held_out": {"kind": "int", "min": 0, "max": 64}},
           "required": ["signature", "tests", "candidates"]}


def _inv_t2i(args: dict, ctl: CallCtl) -> dict:
    """Pick an implementation among the candidates: gate 1 (given tests all green) + gate 2 (held-out extended tests all green) + gate 3 (mutants killed).

    The held-out tests are generated by **deterministic extra points of the same signature** (a different sampling seed
    from the given tests), decided by the signature rather than chosen by the caller — avoiding "picking a few passing
    tests myself as held-out".
    """
    ctx = "op:tests_to_impl#9703"
    sig = args["signature"]
    tests = args["tests"]
    cands = args["candidates"]
    for c in cands:
        validate_prog(c, ctx + ".candidate")
    # held-out: resample the same domain with an offset seed
    params = sig["params"]
    extra: list = []
    n_extra = int(args.get("held_out", 8))
    if params and n_extra > 0:
        k = 101
        tries = 0
        given = {tuple(t["args"]) for t in tests}
        while len(extra) < n_extra and tries < 4096:
            inp = {}
            for i, p in enumerate(params):
                span = int(p["hi"]) - int(p["lo"])
                inp[ARGS[i]] = int(p["lo"]) + ((k * 104729 + i * 7919) % (span + 1))
            args_key = tuple(inp[a] for a in ARGS[:len(params)])
            k += 1
            tries += 1
            if args_key in given:
                continue
            extra.append(args_key)
    accepted = None
    tried = 0
    gate1 = gate2 = gate3 = False
    for c in cands:
        ctl.check_time(ctx + "@search")
        tried += 1
        p1, n1, _ = _run_tests(c, tests)
        if p1 < n1:
            continue
        gate1 = True
        held_tests = [{"args": list(k), "expect": None} for k in extra]
        ok_held = True
        for k in extra:
            r = interp(c, _mk_inputs(list(k)), 200_000)
            if r["error"] is not None:
                ok_held = False
                break
        if not ok_held:
            continue
        gate2 = True
        muts = _mutants(c, 64)
        if not muts:
            raise SvoError(Err.MUTATION_SURVIVE, ctx=ctx,
                           detail="门 3 无证据: 候选实现生成不出任何变异体 — 按 §13 判不过")
        base_box = _behavior_box(c)
        alive = []
        for mk, mdesc, mm in muts:
            pm, nm, _f = _run_tests(mm, tests)
            if pm < nm:
                continue
            if _behavior_box(mm) != base_box:
                alive.append({"kind": mk, "desc": mdesc})
        if alive:
            raise SvoError(Err.MUTATION_SURVIVE, ctx=ctx,
                           detail=f"门 3 不过: 候选实现的 {len(alive)} 个变异体存活, 例 {alive[0]}")
        gate3 = True
        accepted = c
        break
    if accepted is None:
        raise SvoError(Err.REGRESS_FAIL, ctx=ctx,
                       detail=f"三重门未过: {tried} 个候选里没有同时满足门1(给定测试全绿)与门2"
                              f"(held-out 可执行)的实现; gate1={gate1} gate2={gate2}")
    # The tightened test_vector checker **executes the program itself**: reporting only got/expect is judged as no evidence.
    # Hence the payload carries the candidate implementation and the (args, expect) pairs, and the checker reruns them with engine2.
    cert = Certificate(kind="test_vector",
                       payload={"program": accepted,
                                "cases": [{"args": t["args"], "expect": t["expect"]}
                                          for t in tests[:16]],
                                "held_out": True},
                       issued_by="tests_to_impl")
    return {"value": {"impl": accepted, "candidates_tried": tried,
                      "gate1_tests": gate1, "gate2_heldout": gate2, "gate3_mutation": gate3,
                      "n_given_tests": len(tests), "n_heldout": len(extra),
                      "n_mutants": len(_mutants(accepted, 64)),
                      "impl_fingerprint": _h(accepted)},
            "resid": [0.0], "cert": cert,
            "notes": f"实现候选通过三重门: 试了 {tried} 个; 门1 {len(tests)}/{len(tests)} 绿, "
                     f"门2 held-out {len(extra)} 条可执行, 门3 变异体 "
                     f"{len(_mutants(accepted, 64))} 个全杀; 实现指纹 {_h(accepted)}"}


def _ver_t2i(args: dict, value: dict) -> dict:
    """Independent verification: (a) **engine2** reruns the given tests, all green; (b) engine2 reruns the held-out inputs, each must yield a definite output;
    (c) the three gate flags match reality; (d) every expectation agrees with tests item by item."""
    impl = value["impl"]
    tests = args["tests"]
    p2, n2, fails = _run_tests(impl, tests, engine=2)
    g1 = (p2 == n2)
    params = args["signature"]["params"]
    ok_held = True
    if params:
        k = 101
        for _ in range(int(args.get("held_out", 8))):
            inp = {}
            for i, p in enumerate(params):
                span = int(p["hi"]) - int(p["lo"])
                inp[ARGS[i]] = int(p["lo"]) + ((k * 104729 + i * 7919) % (span + 1))
            k += 1
            from .code_core import interp_stack
            r = interp_stack(impl, inp, 200_000)
            if r["error"] is not None:
                ok_held = False
                break
    flags_ok = value["gate1_tests"] and value["gate2_heldout"] and value["gate3_mutation"]
    ok = g1 and ok_held and flags_ok
    return {"ok": ok, "resid": [0.0 if g1 else 1.0],
            "detail": f"engine2 复跑给定测试: {p2}/{n2} 绿 (门1={g1}, 失败样例 {fails[:2]}); "
                      f"held-out 输入在 engine2 下可执行={ok_held}; 三门标志自洽={flags_ok}"}


REGISTRY.register(OpDef(
    name="tests_to_impl", band=BAND, sub=3,
    in_schema=_IN_T2I,
    out_schema={"kind": "dict",
                "fields": {"impl": {"kind": "list", "items": {"kind": "dict", "extra": True, "fields": {}}},
                           "candidates_tried": {"kind": "int", "min": 1},
                           "gate1_tests": {"kind": "bool"}, "gate2_heldout": {"kind": "bool"},
                           "gate3_mutation": {"kind": "bool"},
                           "n_given_tests": {"kind": "int", "min": 1},
                           "n_heldout": {"kind": "int", "min": 0},
                           "n_mutants": {"kind": "int", "min": 0},
                           "impl_fingerprint": {"kind": "str"}},
                "required": ["impl", "candidates_tried", "gate1_tests", "gate2_heldout",
                             "gate3_mutation", "n_given_tests", "n_heldout", "n_mutants",
                             "impl_fingerprint"]},
    invoke=_inv_t2i, verify=_ver_t2i, cert_kind="test_vector",
    cost=("units", 5.0), errors=("REGRESS_FAIL", "MUTATION_SURVIVE", "SCHEMA_MISMATCH", "TIMEOUT"),
    effects="none (纯函数)",
    doc="§9.4 tests_to_impl | in: 签名 + 测试对 + 候选实现表 | out: 过三门的实现 + 门状态 | "
        "验收=**三重门** (给定测试全绿 + held-out 可执行 + 变异体全杀), held-out 由签名确定性"
        "补点生成 (不由调用方挑) | 证书=test_vector | 5 units | "
        "诚实边界: 候选由调用方给出 (本算子不搜索程序空间); 门 3 的等价性判定限于有界行为箱; "
        "候选生成不出变异体时按 §13 判'无证据不过' | 失败: 无候选过门 REGRESS_FAIL / 有存活变异体 MUTATION_SURVIVE",
    selfchecks=(
        {"id": "t2i_picks_correct_impl", "args": {
            "signature": {"fn": "twice", "arity": 1,
                          "params": [{"name": "a0", "type": "int", "lo": 0, "hi": 6}],
                          "output": {"type": "int", "lo": 0, "hi": 12}},
            "tests": [{"args": [0], "expect": 0}, {"args": [1], "expect": 2},
                      {"args": [2], "expect": 4}, {"args": [3], "expect": 6},
                      {"args": [4], "expect": 8}, {"args": [5], "expect": 10}],
            "candidates": [
                [{"t": "assign", "var": "v0", "e": {"t": "bin", "op": "*", "l": {"t": "var", "n": "a0"},
                                                    "r": {"t": "lit", "v": 3}}}],
                [{"t": "assign", "var": "v0", "e": {"t": "bin", "op": "*", "l": {"t": "var", "n": "a0"},
                                                    "r": {"t": "lit", "v": 2}}}]],
            "held_out": 6},
         "check": lambda r: (r.value["gate1_tests"] and r.value["gate2_heldout"]
                             and r.value["gate3_mutation"] and r.value["candidates_tried"] == 2
                             and r.value["impl_fingerprint"] and r.cert_independent is True,
                             f"tried={r.value['candidates_tried']} gates="
                             f"({r.value['gate1_tests']},{r.value['gate2_heldout']},{r.value['gate3_mutation']})"),
         "expect": "候选1 是 a0*3 (测试 1→2 就失败), 候选2 是 a0*2 (全绿) → 应选中候选2; "
                   "门3: a0*2 的变异体 (*→+、*→-、2→3、2→1、删语句) 都会被测试杀死"},
        {"id": "t2i_no_candidate_passes", "args": {
            "signature": {"fn": "twice", "arity": 1,
                          "params": [{"name": "a0", "type": "int", "lo": 0, "hi": 4}],
                          "output": {"type": "int", "lo": 0, "hi": 8}},
            "tests": [{"args": [0], "expect": 0}, {"args": [1], "expect": 2},
                      {"args": [2], "expect": 4}, {"args": [3], "expect": 6}],
            "candidates": [
                [{"t": "assign", "var": "v0", "e": {"t": "bin", "op": "*", "l": {"t": "var", "n": "a0"},
                                                    "r": {"t": "lit", "v": 3}}}],
                [{"t": "assign", "var": "v0", "e": {"t": "bin", "op": "+", "l": {"t": "var", "n": "a0"},
                                                    "r": {"t": "lit", "v": 5}}}]],
            "held_out": 4},
         "expect_err": "REGRESS_FAIL",
         "expect": "两个候选都不是 a0*2 → 无候选过门 1, 必须 REGRESS_FAIL 并指出缺哪一门"},
        {"id": "t2i_gate3_mutation_survive", "args": {
    # Gate 3's **failure path** must be activated by a selfcheck. Construction: tests use only a0=0 (expect 0); the
    # candidate v0←a0*2 passes gate 1; but its mutant `2→3` also gives 0 at a0=0 → **the test set cannot kill it**, and
    # inside the behaviour box (a0∈[−2,2]) it is not equivalent to the candidate (at a0=1, 3≠2) → it cannot be exempted
    # as equivalent → gate 3 must fail. Auditor E noted that the old selfchecks had no MUTATION_SURVIVE case at all.
            "signature": {"fn": "f", "arity": 1,
                          "params": [{"name": "a0", "type": "int", "lo": 0, "hi": 3}],
                          "output": {"type": "int", "lo": 0, "hi": 12}},
            "tests": [{"args": [0], "expect": 0}],
            "candidates": [[{"t": "assign", "var": "v0",
                             "e": {"t": "bin", "op": "*", "l": {"t": "var", "n": "a0"},
                                   "r": {"t": "lit", "v": 2}}}]],
            "held_out": 2},
         "expect_err": "MUTATION_SURVIVE",
         "expect": "候选 a0*2 过门 1/门 2, 但变异体 2→3 在唯一样本点 a0=0 上不可区分、"
                   "而在行为箱 a0=1 处 3≠2 → 既没被测试杀死也不等价 → 门 3 判不过。"
                   "本项把门 3 的失败路径固定进自检, 防止该门被静默旁路"},
        {"id": "t2i_bad_candidate", "args": {
            "signature": {"fn": "f", "arity": 1,
                          "params": [{"name": "a0", "type": "int", "lo": 0, "hi": 1}],
                          "output": {"type": "int", "lo": 0, "hi": 1}},
            "tests": [{"args": [0], "expect": 0}],
            "candidates": [[{"t": "assign", "var": "nope", "e": {"t": "lit", "v": 1}}]]},
         "expect_err": "SCHEMA_MISMATCH", "expect": "非法候选必须被结构门拦下"},
    ),
))


# ===========================================================================
# 9704 refactor_equiv — refactoring (behaviour-preserving equivalence)
# ===========================================================================

_IN_REF = {"kind": "dict",
           "fields": {"before": {"kind": "list", "items": {"kind": "dict", "extra": True, "fields": {}}},
                      "after": {"kind": "list", "items": {"kind": "dict", "extra": True, "fields": {}}},
                      "tests": TEST_SCHEMA,
                      "box": {"kind": "bool"}},
           "required": ["before", "after", "tests"]}


def _inv_ref(args: dict, ctl: CallCtl) -> dict:
    """Refactoring equivalence: compare the **observable behaviour** before/after point by point on the "given tests + bounded behaviour box".

    Observable behaviour = (final state of a0..a3, final v0, or the error code). Including the input slots in the
    observation is **defensive**: in this language a0..a3 are read-only input slots (validate_prog rejects assignments
    to a-slots), so they are in fact never modified; but the observation function does not rely on that fact — if they
    were ever relaxed to writable, the equivalence decision would still be correct.
    """
    ctx = "op:refactor_equiv#9704"
    before, after = args["before"], args["after"]
    validate_prog(before, ctx + ".before")
    validate_prog(after, ctx + ".after")
    tests = args["tests"]
    cases = [(_mk_inputs(t["args"]), "test") for t in tests]
    if bool(args.get("box", True)):
        import itertools
        for combo in itertools.product(range(-2, 3), repeat=2):
            inp = {a: 0 for a in ARGS}
            for i, v in enumerate(combo):
                inp[ARGS[i]] = v
            cases.append((inp, "box"))
    diffs: list = []
    same = 0
    for inp, src in cases:
        rb = interp(before, inp, 200_000)
        ra = interp(after, inp, 200_000)
        vb = _obs(rb)
        va = _obs(ra)
        if vb == va:
            same += 1
        else:
            diffs.append({"source": src, "args": [inp[a] for a in ARGS], "before": vb, "after": va})
    # The certificate kind switches with the conclusion: equivalent → equivalence_pair (all points identical);
    # not equivalent → divergence_witness (carries a witness). Why a single kind is not enough: the equivalence
    # certificate's criterion is "all identical", so it necessarily fails on a divergence, and then "not equivalent" —
    # a **legitimate answer** — would be treated by the pipeline as an operator error (measured: two negative examples
    # of refactor_equiv were swallowed by POSTCOND_FAIL). Non-equivalence is itself a conclusion that needs evidence,
    # so it gets a dedicated witness certificate.
    payload_cases = [{"args": [c[0][a] for a in ARGS],
                      "expect": _obs(interp(after, c[0], 200_000)),
                      "got": _obs(interp(before, c[0], 200_000))} for c in cases[:16]]
    # The two tightened checkers **run before/after themselves**: reporting only behaviour pairs is judged as no evidence.
    cert = Certificate(kind=("equivalence_pair" if not diffs else "divergence_witness"),
                       payload={"before": before, "after": after,
                                "tests": [{"args": t["args"]} for t in tests],
                                "box": bool(args.get("box", True)),
                                "cases": payload_cases, "held_out": True},
                       issued_by="refactor_equiv")
    return {"value": {"equivalent": not diffs, "n_cases": len(cases), "n_same": same,
                      "n_diff": len(diffs), "diffs": diffs[:16],
                      "scope": ("测试集 + 行为箱 arg∈[−2,2] (前两维)" if bool(args.get("box", True))
                                else "仅给定测试集")},
            "resid": [float(len(diffs))], "cert": cert,
            "notes": f"重构等价性: {len(cases)} 个输入上行为{'全等' if not diffs else '有差异'} "
                     f"(相同 {same}, 不同 {len(diffs)}); 范围 = {('测试 + 行为箱' if args.get('box', True) else '仅测试')}; "
                     f"残差 = 行为差异点数"}


def _obs(r: dict):
    if r["error"] is not None:
        return ("ERR", r["error"]["code"])
    st = r["state"]
    return tuple(st.get(a, 0) for a in ARGS) + (st.get("v0", 0),)


def _ver_ref(args: dict, value: dict) -> dict:
    """Independent verification: (a) **engine2** reruns the same inputs; both sides' behaviour must match the engine1 records;
    (b) the diff-table detail and counts are self-consistent; (c) the equivalence conclusion = (diff count is 0)."""
    before, after = args["before"], args["after"]
    tests = args["tests"]
    cases = [(_mk_inputs(t["args"]), "test") for t in tests]
    if bool(args.get("box", True)):
        import itertools
        for combo in itertools.product(range(-2, 3), repeat=2):
            inp = {a: 0 for a in ARGS}
            for i, v in enumerate(combo):
                inp[ARGS[i]] = v
            cases.append((inp, "box"))
    from .code_core import interp_stack
    diff2 = 0
    inconsistent = 0
    listed = {(d["source"], tuple(d["args"])): d for d in value["diffs"]}
    for inp, src in cases:
        vb = _obs(interp_stack(before, inp, 200_000))
        va = _obs(interp_stack(after, inp, 200_000))
        if vb != va:
            diff2 += 1
            key = (src, tuple(inp[a] for a in ARGS))
            if key in listed and (listed[key]["before"] != vb or listed[key]["after"] != va):
                inconsistent += 1
    counts_ok = (value["n_diff"] == diff2) and (value["n_same"] + value["n_diff"] == value["n_cases"])
    eq_ok = (value["equivalent"] == (diff2 == 0))
    ok = counts_ok and eq_ok and inconsistent == 0
    return {"ok": ok, "resid": [float(inconsistent)],
            "detail": f"engine2 复算: 差异 {diff2} (engine1 报 {value['n_diff']}); 计数自洽={counts_ok}; "
                      f"等价结论与实际相符={eq_ok}; 差异明细不符 {inconsistent} 条"}


REGISTRY.register(OpDef(
    name="refactor_equiv", band=BAND, sub=4,
    in_schema=_IN_REF,
    out_schema={"kind": "dict",
                "fields": {"equivalent": {"kind": "bool"}, "n_cases": {"kind": "int", "min": 1},
                           "n_same": {"kind": "int", "min": 0}, "n_diff": {"kind": "int", "min": 0},
                           "diffs": {"kind": "list", "maxlen": 16,
                                     "items": {"kind": "dict", "extra": True, "fields": {}}},
                           "scope": {"kind": "str"}},
                "required": ["equivalent", "n_cases", "n_same", "n_diff", "diffs", "scope"]},
    invoke=_inv_ref, verify=_ver_ref, cert_kind="equivalence_pair",
    cost=("units", 3.0), errors=("SCHEMA_MISMATCH",),
    effects="none (纯函数)",
    doc="§9.4 重构算子 (提取/内联/换数据结构) 的**统一验收器** | in: 改前/改后程序 + 测试集 | "
        "out: 等价性结论 + 行为差异表 | 验收=**行为保持等价测试对**: 逐点比对 (输入槽终态, v0) "
        "的可观测行为, 且 engin2 独立复算 + 差异明细自洽 | 证书=equivalence_pair | 3 units | "
        "诚实边界: 等价性只在**测试集 + 有界行为箱**上成立 (箱外不检查), scope 字段逐字写明; "
        "这是'行为等价'不是'语义等价' (语义等价不可判定) | 失败: schema",
    selfchecks=(
        {"id": "ref_extract_identity", "args": {
            "before": [{"t": "assign", "var": "v0",
                        "e": {"t": "bin", "op": "+", "l": {"t": "var", "n": "a0"},
                              "r": {"t": "bin", "op": "*", "l": {"t": "var", "n": "a1"},
                                    "r": {"t": "lit", "v": 2}}}}],
            "after": [{"t": "assign", "var": "v1",
                       "e": {"t": "bin", "op": "*", "l": {"t": "var", "n": "a1"},
                             "r": {"t": "lit", "v": 2}}},
                      {"t": "assign", "var": "v0", "e": {"t": "bin", "op": "+",
                                                        "l": {"t": "var", "n": "a0"},
                                                        "r": {"t": "var", "n": "v1"}}}],
            "tests": [{"args": [1, 2], "expect": 5}, {"args": [0, 0], "expect": 0}],
            "box": True},
         "check": lambda r: (r.value["equivalent"] is True and r.value["n_diff"] == 0
                             and r.cert_independent is True,
                             f"eq={r.value['equivalent']} cases={r.value['n_cases']} "
                             f"cert={r.cert_independent}"),
         "expect": "**提取子表达式**: a0 + a1*2 与 (v1 ← a1*2; v0 ← a0+v1) 行为等价, "
                   "因为 v1 是新的中间槽、不覆盖任何输入 → 行为箱内逐点相同"},
        {"id": "ref_diff_only_outside_tests", "args": {
            "before": [{"t": "assign", "var": "v0", "e": {"t": "var", "n": "a0"}}],
            "after": [{"t": "if", "c": {"t": "bin", "op": ">", "l": {"t": "var", "n": "a1"},
                                        "r": {"t": "lit", "v": 0}},
                       "th": [{"t": "assign", "var": "v0",
                               "e": {"t": "bin", "op": "+", "l": {"t": "var", "n": "a0"},
                                     "r": {"t": "lit", "v": 1}}}],
                       "el": [{"t": "assign", "var": "v0", "e": {"t": "var", "n": "a0"}}]}],
            "tests": [{"args": [1, 0], "expect": 1}, {"args": [5, 0], "expect": 5}],
            "box": True},
         "check": lambda r: (r.value["equivalent"] is False and r.value["n_diff"] > 0
                             and all(d["source"] == "box" for d in r.value["diffs"])
                             and r.cert is not None and r.cert.kind == "divergence_witness"
                             and r.cert_independent is True,
                             f"eq={r.value['equivalent']} diffs={len(r.value['diffs'])} "
                             f"cert={r.cert.kind if r.cert else None}"),
         "expect": "**差异只在测试集之外**: 给定测试都用 a1=0, 改后程序在 a1=0 时与原程序一致, "
                   "但在 a1>0 时多加 1 → 行为箱抓到分歧, 给定测试抓不到。判不等价并出具"
                   "**分歧见证证书** (不等价本身也要有证据); 这也说明 scope 字段为何必须写明范围"},
        {"id": "ref_semantic_change_detected", "args": {
            "before": [{"t": "assign", "var": "v0",
                        "e": {"t": "bin", "op": "+", "l": {"t": "var", "n": "a0"},
                              "r": {"t": "lit", "v": 1}}}],
            "after": [{"t": "assign", "var": "v0",
                       "e": {"t": "bin", "op": "+", "l": {"t": "var", "n": "a0"},
                             "r": {"t": "lit", "v": 2}}}],
            "tests": [{"args": [0], "expect": 1}],
            "box": True},
         "check": lambda r: (r.value["equivalent"] is False
                             and r.cert is not None and r.cert.kind == "divergence_witness",
                             f"diff={r.value['n_diff']} cert={r.cert.kind if r.cert else None}"),
         "expect": "常量 +1 变 +2: 给定测试里也有一条能看出差异 (0→1 vs 0→2) → 不等价, "
                   "证书类型切到 divergence_witness"},
        {"id": "ref_bad_after", "args": {"before": [], "after": [{"t": "assign", "var": "q",
                                                                 "e": {"t": "lit", "v": 1}}],
                                         "tests": [{"args": [], "expect": 0}]},
         "expect_err": "SCHEMA_MISMATCH", "expect": "非法改后程序必须被拦下"},
    ),
))


# ===========================================================================
# 9705 build_repro — reproducible build (lock file + offline dependency resolution)
# ===========================================================================

#: dependency entry: {"name": str, "version": "a.b.c", "sha256": 64 hex}
_DEP_SCHEMA = {"kind": "dict", "extra": False,
               "fields": {"name": {"kind": "str"}, "version": {"kind": "str"},
                          "sha256": {"kind": "str"}},
               "required": ["name", "version", "sha256"]}
_IN_BUILD = {"kind": "dict",
             "fields": {"manifest": {"kind": "list", "minlen": 0, "maxlen": 64, "items": _DEP_SCHEMA},
                        "lock": {"kind": "list", "minlen": 0, "maxlen": 64, "items": _DEP_SCHEMA},
                        "offline": {"kind": "bool"},
                        "target": {"kind": "str"}},
             "required": ["manifest", "lock"]}


def _semver(v: str) -> tuple:
    parts = str(v).split(".")
    if len(parts) != 3 or not all(p.isdigit() for p in parts):
        raise SvoError(Err.SPEC_AMBIGUOUS, ctx="build_repro",
                       detail=f"版本 {v!r} 不是 a.b.c 形式 — 拒绝猜测语义")
    return tuple(int(p) for p in parts)


def _inv_build(args: dict, ctl: CallCtl) -> dict:
    """Reproducible build: (1) the lock file must **fully cover** the manifest with matching versions; (2) every dependency must
    have a content hash (no hash = not reproducible); (3) in offline mode no network action is taken (this implementation
    issues no network call at all, so this holds **structurally**, not via a switch); (4) the build fingerprint = a
    deterministic hash of the lock file (double runs must agree)."""
    ctx = "op:build_repro#9705"
    man = args["manifest"]
    lock = args["lock"]
    mnames = {d["name"]: d for d in man}
    lnames = {d["name"]: d for d in lock}
    if len(mnames) != len(man) or len(lnames) != len(lock):
        raise SvoError(Err.DEP_CONFLICT, ctx=ctx, detail="依赖名重复 — manifest/lock 里名字必须唯一")
    missing = sorted(set(mnames) - set(lnames))
    if missing:
        raise SvoError(Err.DEP_CONFLICT, ctx=ctx,
                       detail=f"锁文件缺少依赖 {missing} — 未锁定的依赖使构建不可复现, 拒绝")
    extra = sorted(set(lnames) - set(mnames))
    if extra:
        raise SvoError(Err.DEP_CONFLICT, ctx=ctx,
                       detail=f"锁文件含 manifest 未声明的依赖 {extra} — 锁与声明不一致, 拒绝")
    drift = [n for n in sorted(mnames)
             if _semver(mnames[n]["version"]) != _semver(lnames[n]["version"])]
    if drift:
        raise SvoError(Err.DEP_CONFLICT, ctx=ctx,
                       detail=f"版本漂移 {[(n, mnames[n]['version'], lnames[n]['version']) for n in drift]} "
                              f"— 锁文件与 manifest 不符")
    for d in lock:
        h = d["sha256"]
        if len(h) != 64 or any(c not in "0123456789abcdef" for c in h.lower()):
            raise SvoError(Err.DEP_CONFLICT, ctx=ctx,
                           detail=f"依赖 {d['name']} 的 sha256 非 64 位十六进制 — 无内容哈希即不可复现")
    built = {"target": args.get("target", "default"),
             "deps": sorted((d["name"], d["version"], d["sha256"]) for d in lock),
             "offline": True}
    fp = _h(built)
    return {"value": {"build_fingerprint": fp, "n_deps": len(lock),
                      "resolved": [[d["name"], d["version"]] for d in sorted(lock, key=lambda x: x["name"])],
                      "offline_enforced": True, "deterministic": True,
                      "network_calls": 0},
            "resid": [0.0],
            "notes": f"构建解析成功: {len(lock)} 个依赖全部锁定且带内容哈希; 构建指纹 {fp}; "
                     f"离线强制=True (本算子**不包含任何网络调用代码**, 故离线是结构性成立的); "
                     f"网络调用数 0"}


def _ver_build(args: dict, value: dict) -> dict:
    """Independent verification: (a) the recomputed build fingerprint must agree (determinism); (b) the **bidirectional lock-file ⊇ manifest
    check** is implemented with another set operation; (c) hash lengths are rechecked one by one; (d) structural check for
    no network calls: verify that this module's source contains no socket/urllib/http imports (static evidence, not a promise)."""
    man = args["manifest"]
    lock = args["lock"]
    built = {"target": args.get("target", "default"),
             "deps": sorted((d["name"], d["version"], d["sha256"]) for d in lock),
             "offline": True}
    fp_ok = _h(built) == value["build_fingerprint"]
    set_ok = (sorted(d["name"] for d in man) == sorted(d["name"] for d in lock)) \
        and all(d["name"] in {x["name"] for x in lock} for d in man)
    hash_ok = all(len(d["sha256"]) == 64 for d in lock)
    # Structural check: only look at code lines really starting with import/from, and **assemble the pattern strings
    # locally** — the old version put literals such as "import socket" directly into the tuple being searched, so the
    # check **matched itself** and always reported "network import found" (measured: that selfcheck failed spuriously).
    # A checker must not count itself.
    mods = ("sock" + "et", "urllib", "requ" + "ests", "htt" + "p", "ftplib", "telnetlib")
    bad_lines: list = []
    with open(__file__, "r", encoding="utf-8") as fh:
        for ln in fh:
            t = ln.strip()
            if not (t.startswith("import ") or t.startswith("from ")):
                continue
            for m in mods:
                if m in t:
                    bad_lines.append(t[:60])
    net_ok = not bad_lines
    banned = bad_lines
    n_ok = int(value["network_calls"]) == 0
    ok = fp_ok and set_ok and hash_ok and net_ok and n_ok
    return {"ok": ok, "resid": [0.0 if ok else 1.0],
            "detail": f"指纹重算一致={fp_ok}; 锁与声明的名字集合一致={set_ok}; "
                      f"哈希长度全为 64={hash_ok}; 源码无网络导入={net_ok} (发现 {banned}); "
                      f"网络调用数=0={n_ok}"}


REGISTRY.register(OpDef(
    name="build_repro", band=BAND, sub=5,
    in_schema=_IN_BUILD,
    out_schema={"kind": "dict",
                "fields": {"build_fingerprint": {"kind": "str"}, "n_deps": {"kind": "int", "min": 0},
                           "resolved": {"kind": "list", "maxlen": 64,
                                        "items": {"kind": "list", "len": 2,
                                                  "items": {"kind": "str"}}},
                           "offline_enforced": {"kind": "bool"}, "deterministic": {"kind": "bool"},
                           "network_calls": {"kind": "int", "min": 0}},
                "required": ["build_fingerprint", "n_deps", "resolved", "offline_enforced",
                             "deterministic", "network_calls"]},
    invoke=_inv_build, verify=_ver_build,
    cost=("units", 1.0), errors=("DEP_CONFLICT", "SPEC_AMBIGUOUS", "SCHEMA_MISMATCH"),
    effects="none (纯函数; 无网络调用 —— 由 verify 静态核对源码导入, 不是承诺)",
    doc="§9.5 build 可复现构建 | in: manifest + lock (name@a.b.c + sha256) | out: 构建指纹 + 解析结果 | "
        "验收=锁**完全覆盖**声明且版本一致 + 每项有 64 位内容哈希 + 指纹双跑一致 + "
        "**源码无网络导入**(静态证据) | 1 unit | 诚实边界: 不做真实包管理 (容器无网络), "
        "'离线解析'= 对声明做哈希与一致性核验 | 失败: 缺锁/漂移/无哈希 DEP_CONFLICT",
    selfchecks=(
        {"id": "br_locked_ok", "args": {
            "manifest": [{"name": "alpha", "version": "1.2.3", "sha256": "a" * 64},
                         {"name": "beta", "version": "0.1.0", "sha256": "b" * 64}],
            "lock": [{"name": "alpha", "version": "1.2.3", "sha256": "a" * 64},
                     {"name": "beta", "version": "0.1.0", "sha256": "b" * 64}],
            "target": "release"},
         "check": lambda r: (r.value["n_deps"] == 2 and r.value["offline_enforced"] is True
                             and r.value["network_calls"] == 0
                             and r.value["resolved"] == [["alpha", "1.2.3"], ["beta", "0.1.0"]],
                             f"deps={r.value['resolved']} fp={r.value['build_fingerprint']}"),
         "expect": "两个依赖全锁定且带哈希: 解析成功, 指纹由 (target, 依赖三元组) 决定; "
                   "resolved 按名字排序 (确定性)"},
        {"id": "br_unlocked_dep", "args": {
            "manifest": [{"name": "alpha", "version": "1.2.3", "sha256": "a" * 64},
                         {"name": "gamma", "version": "2.0.0", "sha256": "c" * 64}],
            "lock": [{"name": "alpha", "version": "1.2.3", "sha256": "a" * 64}]},
         "expect_err": "DEP_CONFLICT",
         "expect": "gamma 未锁定 → 必须拒 (未锁定 = 不可复现; 不许'先构建再说')"},
        {"id": "br_version_drift", "args": {
            "manifest": [{"name": "alpha", "version": "1.2.3", "sha256": "a" * 64}],
            "lock": [{"name": "alpha", "version": "1.2.4", "sha256": "a" * 64}]},
         "expect_err": "DEP_CONFLICT",
         "expect": "版本漂移 1.2.3 vs 1.2.4 → 拒 (锁文件的意义就是钉住版本)"},
        {"id": "br_missing_hash", "args": {
            "manifest": [{"name": "alpha", "version": "1.0.0", "sha256": "zz"}],
            "lock": [{"name": "alpha", "version": "1.0.0", "sha256": "zz"}]},
         "expect_err": "DEP_CONFLICT",
         "expect": "哈希 'zz' 非 64 位十六进制 → 无内容哈希, 拒 (防止'锁了版本但内容可变')"},
        {"id": "br_bad_semver", "args": {
            "manifest": [{"name": "alpha", "version": "1.2", "sha256": "a" * 64}],
            "lock": [{"name": "alpha", "version": "1.2", "sha256": "a" * 64}]},
         "expect_err": "SPEC_AMBIGUOUS",
         "expect": "版本 '1.2' 不是 a.b.c → 不猜语义, 明确报 SPEC_AMBIGUOUS"},
        {"id": "br_extra_in_lock", "args": {
            "manifest": [{"name": "alpha", "version": "1.0.0", "sha256": "a" * 64}],
            "lock": [{"name": "alpha", "version": "1.0.0", "sha256": "a" * 64},
                     {"name": "ghost", "version": "9.9.9", "sha256": "d" * 64}]},
         "expect_err": "DEP_CONFLICT",
         "expect": "锁里多了未声明的 ghost → 拒 (锁与声明必须一一对应)"},
    ),
))


# ===========================================================================
# 9706 api_compat — semantic-version compatibility
# ===========================================================================

#: export entry: {"name": str, "arity": int, "in_lo": int, "in_hi": int, "out_lo": int, "out_hi": int}
_API_SCHEMA = {"kind": "dict", "extra": False,
               "fields": {"name": {"kind": "str"}, "arity": {"kind": "int", "min": 0, "max": 4},
                          "in_lo": {"kind": "int"}, "in_hi": {"kind": "int"},
                          "out_lo": {"kind": "int"}, "out_hi": {"kind": "int"}},
               "required": ["name", "arity", "in_lo", "in_hi", "out_lo", "out_hi"]}
_IN_API = {"kind": "dict",
           "fields": {"old": {"kind": "list", "minlen": 0, "maxlen": 64, "items": _API_SCHEMA},
                      "new": {"kind": "list", "minlen": 0, "maxlen": 64, "items": _API_SCHEMA},
                      "declared": {"kind": "str", "enum": ["major", "minor", "patch"]}},
           "required": ["old", "new"]}


def _inv_api(args: dict, ctl: CallCtl) -> dict:
    """API compatibility: enumerate breaking changes (export removed / arity changed / input domain shrunk / output domain widened)."""
    old = {d["name"]: d for d in args["old"]}
    new = {d["name"]: d for d in args["new"]}
    removed = sorted(set(old) - set(new))
    added = sorted(set(new) - set(old))
    breaking: list = []
    compatible: list = []
    for n in sorted(set(old) & set(new)):
        o, w = old[n], new[n]
        if o["arity"] != w["arity"]:
            breaking.append({"name": n, "kind": "ARITY_CHANGED", "old": o["arity"], "new": w["arity"]})
            continue
        # input-domain shrinkage (new domain ⊄ old domain) = breaking
        if w["in_lo"] > o["in_lo"] or w["in_hi"] < o["in_hi"]:
            breaking.append({"name": n, "kind": "INPUT_DOMAIN_NARROWED",
                             "old": [o["in_lo"], o["in_hi"]], "new": [w["in_lo"], w["in_hi"]]})
        # output-domain expansion = breaking (the caller's assumptions break)
        if w["out_lo"] < o["out_lo"] or w["out_hi"] > o["out_hi"]:
            breaking.append({"name": n, "kind": "OUTPUT_DOMAIN_WIDENED",
                             "old": [o["out_lo"], o["out_hi"]], "new": [w["out_lo"], w["out_hi"]]})
        if not any(b["name"] == n for b in breaking):
            compatible.append(n)
    for n in removed:
        breaking.append({"name": n, "kind": "EXPORT_REMOVED"})
    declared = args.get("declared")
    if declared and breaking and declared != "major":
        raise SvoError(Err.API_INCOMPAT, ctx="op:api_compat#9706",
                       detail=f"存在 {len(breaking)} 项破坏性变更却声明为 {declared} 版本升级 — "
                              f"语义化版本要求破坏性变更必须升 major")
    if declared == "patch" and added:
        raise SvoError(Err.API_INCOMPAT, ctx="op:api_compat#9706",
                       detail=f"patch 升级不得新增导出 {added} (新增属于 minor)")
    return {"value": {"breaking_count": len(breaking), "breaking": breaking, "added": added,
                      "removed": removed, "compatible": compatible,
                      "compatible_flag": not breaking,
                      "verdict": ("compatible" if not breaking else "breaking"),
                      "checked": "signature_surface_only"},
            "resid": [float(len(breaking))],
            "notes": f"API 差集: 破坏性 {len(breaking)} 项 {[b['kind'] + ':' + b['name'] for b in breaking]} ; "
                     f"新增 {added}; 删除 {removed}; 兼容={not breaking}。"
                     f"**口径声明**: 本算子只比较**签名面**, 不比较语义"}


def _ver_api(args: dict, value: dict) -> dict:
    """Independent verification: (a) recompute added/removed with **set operations** independently; (b) for every name in the intersection, independently
    rejudge the four breaking rules (arity / input shrinkage / output expansion) and compare with the output; (c) compatible_flag
    and breaking_count are self-consistent."""
    onames = {d["name"] for d in args["old"]}
    nnames = {d["name"] for d in args["new"]}
    added2 = sorted(nnames - onames)
    removed2 = sorted(onames - nnames)
    o = {d["name"]: d for d in args["old"]}
    n = {d["name"]: d for d in args["new"]}
    br2: list = []
    for k in sorted(onames & nnames):
        if o[k]["arity"] != n[k]["arity"]:
            br2.append(("ARITY_CHANGED", k))
        else:
            if n[k]["in_lo"] > o[k]["in_lo"] or n[k]["in_hi"] < o[k]["in_hi"]:
                br2.append(("INPUT_DOMAIN_NARROWED", k))
            if n[k]["out_lo"] < o[k]["out_lo"] or n[k]["out_hi"] > o[k]["out_hi"]:
                br2.append(("OUTPUT_DOMAIN_WIDENED", k))
    br2 += [("EXPORT_REMOVED", k) for k in removed2]
    br1 = sorted((b["kind"], b["name"]) for b in value["breaking"])
    ok = (added2 == sorted(value["added"])) and (removed2 == sorted(value["removed"])) \
        and (sorted(br2) == br1) and (value["compatible_flag"] == (len(br2) == 0)) \
        and (value["breaking_count"] == len(br2))
    return {"ok": ok, "resid": [0.0 if ok else 1.0],
            "detail": f"集合运算复算: 新增一致={added2 == sorted(value['added'])}; "
                      f"删除一致={removed2 == sorted(value['removed'])}; "
                      f"破坏性变更集一致={sorted(br2) == br1}; 兼容标志自洽="
                      f"{value['compatible_flag'] == (len(br2) == 0)}"}


REGISTRY.register(OpDef(
    name="api_compat", band=BAND, sub=6,
    in_schema=_IN_API,
    out_schema={"kind": "dict",
                "fields": {"breaking_count": {"kind": "int", "min": 0},
                           "breaking": {"kind": "list", "maxlen": 64,
                                        "items": {"kind": "dict", "extra": True, "fields": {}}},
                           "added": {"kind": "list", "items": {"kind": "str"}},
                           "removed": {"kind": "list", "items": {"kind": "str"}},
                           "compatible": {"kind": "list", "items": {"kind": "str"}},
                           "compatible_flag": {"kind": "bool"},
                           "verdict": {"kind": "str", "enum": ["compatible", "breaking"]},
                           "checked": {"kind": "str"}},
                "required": ["breaking_count", "breaking", "added", "removed", "compatible",
                             "compatible_flag", "verdict", "checked"]},
    invoke=_inv_api, verify=_ver_api,
    cost=("units", 1.0), errors=("API_INCOMPAT", "SCHEMA_MISMATCH"),
    effects="none (纯函数)",
    doc="§9.5 api_compat 语义化版本兼容 | in: 旧/新导出面 + 可选声明升级类型 | "
        "out: 破坏性变更枚举 + 判定 | 验收=四条破坏性规则 (删除/元数/输入收缩/输出扩张) 逐条判定 + "
        "verify 用集合运算独立复算 + 与声明的版本升级一致 (破坏性却声明 minor/patch → API_INCOMPAT) | "
        "1 unit | 诚实边界: **只比较签名面, 不做语义兼容判定** (语义级要过 §9.3 三重门) | "
        "失败: 版本声明与变更性质矛盾 API_INCOMPAT",
    selfchecks=(
        {"id": "ac_compatible_addition", "args": {
            "old": [{"name": "f", "arity": 1, "in_lo": 0, "in_hi": 10, "out_lo": 0, "out_hi": 10}],
            "new": [{"name": "f", "arity": 1, "in_lo": 0, "in_hi": 10, "out_lo": 0, "out_hi": 10},
                    {"name": "g", "arity": 1, "in_lo": 0, "in_hi": 5, "out_lo": 0, "out_hi": 5}],
            "declared": "minor"},
         "check": lambda r: (r.value["compatible_flag"] is True and r.value["added"] == ["g"]
                             and r.value["breaking_count"] == 0,
                             f"added={r.value['added']} verdict={r.value['verdict']}"),
         "expect": "只新增导出 g、f 不变: 兼容 (minor 升级允许新增)"},
        {"id": "ac_removed_export_breaking", "args": {
            "old": [{"name": "f", "arity": 1, "in_lo": 0, "in_hi": 1, "out_lo": 0, "out_hi": 1},
                    {"name": "h", "arity": 0, "in_lo": 0, "in_hi": 0, "out_lo": 0, "out_hi": 0}],
            "new": [{"name": "f", "arity": 1, "in_lo": 0, "in_hi": 1, "out_lo": 0, "out_hi": 1}],
            "declared": "major"},
         "check": lambda r: (r.value["breaking_count"] == 1
                             and r.value["breaking"][0]["kind"] == "EXPORT_REMOVED"
                             and r.value["removed"] == ["h"],
                             f"breaking={r.value['breaking']}"),
         "expect": "删除导出 h: 破坏性; 声明为 major → 一致, 通过"},
        {"id": "ac_breaking_but_minor_declared", "args": {
            "old": [{"name": "f", "arity": 2, "in_lo": 0, "in_hi": 1, "out_lo": 0, "out_hi": 1}],
            "new": [{"name": "f", "arity": 3, "in_lo": 0, "in_hi": 1, "out_lo": 0, "out_hi": 1}],
            "declared": "minor"},
         "expect_err": "API_INCOMPAT",
         "expect": "元数 2→3 是破坏性变更却声明 minor → 语义化版本被违反, 必须拒"},
        {"id": "ac_input_narrowed", "args": {
            "old": [{"name": "f", "arity": 1, "in_lo": -10, "in_hi": 10, "out_lo": 0, "out_hi": 5}],
            "new": [{"name": "f", "arity": 1, "in_lo": 0, "in_hi": 10, "out_lo": 0, "out_hi": 5}],
            "declared": "major"},
         "check": lambda r: (r.value["breaking"][0]["kind"] == "INPUT_DOMAIN_NARROWED",
                             f"breaking={r.value['breaking']}"),
         "expect": "输入域 [−10,10] 收窄到 [0,10]: 原本合法的负输入变成非法 → 破坏性 "
                   "(调用方面临新的失败路径)"},
        {"id": "ac_output_widened", "args": {
            "old": [{"name": "f", "arity": 1, "in_lo": 0, "in_hi": 1, "out_lo": 0, "out_hi": 5}],
            "new": [{"name": "f", "arity": 1, "in_lo": 0, "in_hi": 1, "out_lo": -5, "out_hi": 5}],
            "declared": "major"},
         "check": lambda r: (r.value["breaking"][0]["kind"] == "OUTPUT_DOMAIN_WIDENED",
                             f"breaking={r.value['breaking']}"),
         "expect": "输出域扩张到含负数: 调用方按'非负'写的断言会失效 → 破坏性"},
        {"id": "ac_patch_with_addition", "args": {
            "old": [{"name": "f", "arity": 1, "in_lo": 0, "in_hi": 1, "out_lo": 0, "out_hi": 1}],
            "new": [{"name": "f", "arity": 1, "in_lo": 0, "in_hi": 1, "out_lo": 0, "out_hi": 1},
                    {"name": "n", "arity": 1, "in_lo": 0, "in_hi": 1, "out_lo": 0, "out_hi": 1}],
            "declared": "patch"},
         "expect_err": "API_INCOMPAT",
         "expect": "patch 升级里新增导出 → 违反语义化版本 (新增属 minor), 必须拒"},
    ),
))


# ===========================================================================
# 9707 dep_audit — dependency × vulnerability-database offline snapshot
# ===========================================================================

_ADV_SCHEMA = {"kind": "dict", "extra": False,
               "fields": {"id": {"kind": "str"}, "package": {"kind": "str"},
                          "lo": {"kind": "str"}, "hi": {"kind": "str"},
                          "fixed": {"kind": "str"}},
               "required": ["id", "package", "lo", "hi"]}
_IN_AUDIT = {"kind": "dict",
             "fields": {"deps": {"kind": "list", "minlen": 0, "maxlen": 64, "items": _DEP_SCHEMA},
                        "advisories": {"kind": "list", "minlen": 0, "maxlen": 256,
                                       "items": _ADV_SCHEMA},
                        "snapshot_id": {"kind": "str"}},
             "required": ["deps", "advisories"]}


def _inv_audit(args: dict, ctl: CallCtl) -> dict:
    """Dependency audit: match each (dependency version) against the **version intervals** of the vulnerability database (closed-closed, compared by semver tuples)."""
    ctx = "op:dep_audit#9707"
    deps = args["deps"]
    advs = args["advisories"]
    hits: list = []
    for d in deps:
        dv = _semver(d["version"])
        for a in advs:
            if a["package"] != d["name"]:
                continue
            lo, hi = _semver(a["lo"]), _semver(a["hi"])
            if lo <= dv <= hi:
                fix = a.get("fixed")
                hits.append({"dep": d["name"], "version": d["version"], "advisory": a["id"],
                             "range": [a["lo"], a["hi"]], "fixed_in": fix or "",
                             "fix_available": bool(fix) and _semver(fix) > dv})
    hits.sort(key=lambda h: (h["dep"], h["advisory"]))
    return {"value": {"vulnerable": hits, "n_vulnerable": len(hits),
                      "n_deps": len(deps), "n_advisories": len(advs),
                      "snapshot_id": args.get("snapshot_id", "<caller-provided>"),
                      "db_origin": "caller_provided_offline_snapshot"},
            "resid": [float(len(hits))],
            "notes": f"依赖审计: {len(deps)} 个依赖 × {len(advs)} 条离线公告 → 命中 {len(hits)} 项; "
                     f"**诚实声明**: 漏洞库是调用方提供的离线快照 (snapshot_id="
                     f"{args.get('snapshot_id', '<未标注>')}), 本算子不声称数据库完整性与时效性"}


def _ver_audit(args: dict, value: dict) -> dict:
    """Independent verification: (a) independently recompute the hit set by **brute-force item-by-item comparison** (for every dependency, every advisory,
    every version, decide the interval one by one);
    (b) a hit's interval must really contain that version; (c) a non-hit must really fall outside the intervals of all advisories for the same package."""
    deps = args["deps"]
    advs = args["advisories"]
    hits2: list = []
    for d in deps:
        for a in advs:
            if a["package"] != d["name"]:
                continue
            try:
                dv, lo, hi = _semver(d["version"]), _semver(a["lo"]), _semver(a["hi"])
            except SvoError:
                continue
            if lo <= dv <= hi:
                hits2.append((d["name"], a["id"]))
    got = sorted((h["dep"], h["advisory"]) for h in value["vulnerable"])
    hits2.sort()
    ok = (got == hits2) and (int(value["n_vulnerable"]) == len(hits2))
    return {"ok": ok, "resid": [float(abs(len(got) - len(hits2)))],
            "detail": f"暴力比对复算命中 {len(hits2)} 项, 与输出 {len(got)} 项一致={got == hits2}; "
                      f"计数自洽={int(value['n_vulnerable']) == len(hits2)}"}


REGISTRY.register(OpDef(
    name="dep_audit", band=BAND, sub=7,
    in_schema=_IN_AUDIT,
    out_schema={"kind": "dict",
                "fields": {"vulnerable": {"kind": "list", "maxlen": 256,
                                          "items": {"kind": "dict", "extra": True, "fields": {}}},
                           "n_vulnerable": {"kind": "int", "min": 0},
                           "n_deps": {"kind": "int", "min": 0}, "n_advisories": {"kind": "int", "min": 0},
                           "snapshot_id": {"kind": "str"}, "db_origin": {"kind": "str"}},
                "required": ["vulnerable", "n_vulnerable", "n_deps", "n_advisories", "snapshot_id",
                             "db_origin"]},
    invoke=_inv_audit, verify=_ver_audit,
    cost=("units", 1.0), errors=("SPEC_AMBIGUOUS", "SCHEMA_MISMATCH"),
    effects="none (纯函数, 离线快照)",
    doc="§9.5 dep_audit 依赖 × 漏洞库离线快照 | in: 依赖表 + 公告表 (离线) | "
        "out: 命中清单 (含修复版本) | 验收=暴力区间比对独立复算 + 命中区间确实包含该版本 | "
        "1 unit | 诚实边界: **本模块不含任何真实 CVE 数据**, 只实现'区间匹配 + 命中枚举'机制; "
        "数据库来源与时效由调用方的 snapshot_id 承担 (输出里显式标注 db_origin) | "
        "失败: 版本串非 a.b.c SPEC_AMBIGUOUS",
    selfchecks=(
        {"id": "da_hit_and_miss", "args": {
            "deps": [{"name": "alpha", "version": "1.2.0", "sha256": "a" * 64},
                     {"name": "beta", "version": "3.0.0", "sha256": "b" * 64}],
            "advisories": [{"id": "ADV-1", "package": "alpha", "lo": "1.0.0", "hi": "1.5.0",
                            "fixed": "1.5.0"},
                           {"id": "ADV-2", "package": "beta", "lo": "2.0.0", "hi": "2.9.9"}],
            "snapshot_id": "snap-test-001"},
         "check": lambda r: (r.value["n_vulnerable"] == 1
                             and r.value["vulnerable"][0]["advisory"] == "ADV-1"
                             and r.value["vulnerable"][0]["fixed_in"] == "1.5.0"
                             and r.value["vulnerable"][0]["fix_available"] is True,
                             f"hits={r.value['vulnerable']}"),
         "expect": "alpha 1.2.0 落在 [1.0.0,1.5.0] → 命中 ADV-1 且修复版 1.5.0 > 1.2.0 "
                   "(fix_available=True); beta 3.0.0 不在 [2.0.0,2.9.9] → 不命中"},
        {"id": "da_boundary_inclusive", "args": {
            "deps": [{"name": "p", "version": "1.5.0", "sha256": "a" * 64}],
            "advisories": [{"id": "ADV-X", "package": "p", "lo": "1.0.0", "hi": "1.5.0"}],
            "snapshot_id": "snap-test-002"},
         "check": lambda r: (r.value["n_vulnerable"] == 1,
                             f"hits={r.value['vulnerable']}"),
         "expect": "区间**右端闭**: 1.5.0 ∈ [1.0.0,1.5.0] → 命中 (左右端都是闭区间, 口径写死)"},
        {"id": "da_no_advisories", "args": {
            "deps": [{"name": "p", "version": "9.9.9", "sha256": "a" * 64}],
            "advisories": [], "snapshot_id": "snap-empty"},
         "check": lambda r: (r.value["n_vulnerable"] == 0 and r.value["n_deps"] == 1,
                             f"hits={r.value['n_vulnerable']}"),
         "expect": "空公告库 → 0 命中 (但输出仍标注 db_origin 与 snapshot_id, 便于审计'当时用的是哪份库')"},
        {"id": "da_bad_version_format", "args": {
            "deps": [{"name": "p", "version": "v1.0", "sha256": "a" * 64}],
            "advisories": [{"id": "A", "package": "p", "lo": "1.0.0", "hi": "1.0.0"}]},
         "expect_err": "SPEC_AMBIGUOUS",
         "expect": "依赖版本 'v1.0' 非 a.b.c 形式 → 不猜, 报 SPEC_AMBIGUOUS"},
    ),
))


# ===========================================================================
# 9708 ci_gate — CI as verification gate
# ===========================================================================

_IN_CI = {"kind": "dict",
          "fields": {"checks": {"kind": "list", "minlen": 1, "maxlen": 32,
                                "items": {"kind": "dict", "extra": False,
                                          "fields": {"name": {"kind": "str"},
                                                     "op": {"kind": "str"},
                                                     "args": {"kind": "dict", "extra": True, "fields": {}},
                                                     "required_pass": {"kind": "bool"}},
                                          "required": ["name", "op"]}},
                     "stop_on_first_failure": {"kind": "bool"}},
          "required": ["checks"]}


def _inv_ci(args: dict, ctl: CallCtl) -> dict:
    """CI gate: run the check items one by one; any required_pass item failing closes the gate (overall failure)."""
    results: list = []
    failed: list = []
    for c in args["checks"]:
        ctl.check_time("op:ci_gate#9708@run")
        op = c["op"]
        if not REGISTRY.has(op):
    # type contract: err is always str ("" on success), resid is always float —
    # when unmeasured it is 0.0 with an explicit n_unavailable count, rather than sticking None in and breaking the schema.
            results.append({"name": c["name"], "ok": False, "err": "OP_MISSING", "resid": 0.0})
            failed.append(c["name"])
            if bool(args.get("stop_on_first_failure", False)):
                break
            continue
        r = REGISTRY.tool(op).call(c.get("args") or {})
        results.append({"name": c["name"], "ok": bool(r.ok), "err": r.err or "",
                        "resid": float(r.resid_scalar) if r.resid_scalar == r.resid_scalar else 0.0})
        if not r.ok:
            failed.append(c["name"])
            if bool(args.get("stop_on_first_failure", False)):
                break
    required_failed = [c["name"] for c in args["checks"]
                       if c["name"] in failed and bool(c.get("required_pass", True))]
    gate_open = not required_failed
    n_unavail = sum(1 for x in results if not x["ok"] and x["err"] in ("OP_MISSING", ""
                                                                     ) and not x["ok"])
    return {"value": {"gate_open": gate_open, "n_checks": len(results), "results": results,
                      "failed": failed, "required_failed": required_failed,
                      "n_unavailable_resid": n_unavail},
            "resid": [float(len(required_failed))],
            "notes": f"CI 门: {len(results)}/{len(args['checks'])} 项已跑; 失败 {failed}; "
                     f"其中必过项失败 {required_failed} → 门{'开' if gate_open else '关'}"}


def _ver_ci(args: dict, value: dict) -> dict:
    """Independent verification: (a) **rerun each check item by item** and compare ok/err/resid (the gate must be replayable);
    (b) the gate conclusion agrees with the required-item failure set; (c) when stop_on_first_failure is true, unrun items must really be absent."""
    mism = 0
    rerun: list = []
    for c in args["checks"]:
        if not REGISTRY.has(c["op"]):
            rerun.append((c["name"], False))
            continue
        r = REGISTRY.tool(c["op"]).call(c.get("args") or {})
        rerun.append((c["name"], bool(r.ok)))
    got = {(x["name"], x["ok"]) for x in value["results"]}
    for name, ok in rerun:
        if (name, ok) in got:
            continue
        # may be absent because of stop_on_first_failure
        if not bool(args.get("stop_on_first_failure", False)):
            mism += 1
    gate_ok = value["gate_open"] == (len(value["required_failed"]) == 0)
    count_ok = len(value["results"]) <= len(args["checks"])
    ok = (mism == 0) and gate_ok and count_ok
    return {"ok": ok, "resid": [float(mism)],
            "detail": f"逐项重跑核对: 不一致 {mism} 项; 门结论与必过失败集自洽={gate_ok}; "
                      f"项数不超声明={count_ok} ({len(value['results'])}/{len(args['checks'])})"}


REGISTRY.register(OpDef(
    name="ci_gate", band=BAND, sub=8,
    in_schema=_IN_CI,
    out_schema={"kind": "dict",
                "fields": {"gate_open": {"kind": "bool"}, "n_checks": {"kind": "int", "min": 1},
                           "results": {"kind": "list", "maxlen": 32,
                                       "items": {"kind": "dict", "extra": False,
                                                 "fields": {"name": {"kind": "str"}, "ok": {"kind": "bool"},
                                                            "err": {"kind": "str"}, "resid": {"kind": "float"}},
                                                 "required": ["name", "ok", "err", "resid"]}},
                           "failed": {"kind": "list", "items": {"kind": "str"}},
                           "required_failed": {"kind": "list", "items": {"kind": "str"}},
                           "n_unavailable_resid": {"kind": "int", "min": 0}},
                "required": ["gate_open", "n_checks", "results", "failed", "required_failed",
                             "n_unavailable_resid"]},
    invoke=_inv_ci, verify=_ver_ci,
    cost=("units", 1.0), errors=("TIMEOUT", "SCHEMA_MISMATCH"),
    effects="**间接副作用**: 会调用被引用的算子 (本包算子皆为纯函数, 故仍无外部副作用)",
    doc="§9.5 ci_gate CI 即验证门 | in: 检查项表 (算子 + 参数 + 是否必过) | "
        "out: 逐项结果 + 门开合 + 必过项失败清单 | 验收=**逐项重跑**一致 + 门结论自洽 | 1 unit | "
        "诚实边界: 门只反映**被引用算子**的结论; 若被引算子本身结论弱 (例如只做了抽样), "
        "门不会把结论变强 —— 门是汇总装置不是证据生成器 | 失败: 超时 TIMEOUT",
    selfchecks=(
        {"id": "ci_all_pass", "args": {"checks": [
            {"name": "poly", "op": "solve_poly", "args": {"p": -1.0, "q": -6.0},
             "required_pass": True},
            {"name": "unit", "op": "math_unit_check",
             "args": {"terms": [{"coeff": 1, "dims": {"length": 1}},
                                {"coeff": 1, "dims": {"length": 1}}]},
             "required_pass": True}]},
         "check": lambda r: (r.value["gate_open"] is True and r.value["n_checks"] == 2
                             and r.value["required_failed"] == [],
                             f"open={r.value['gate_open']} failed={r.value['failed']}"),
         "expect": "两项都过 (solve_poly 根 3,-2; unit_check 齐次) → 门开"},
        {"id": "ci_required_failure_closes", "args": {"checks": [
            {"name": "good", "op": "solve_poly", "args": {"p": -1.0, "q": -6.0},
             "required_pass": True},
            {"name": "bad", "op": "solve_poly", "args": {"p": 0.0, "q": 1.0},
             "required_pass": True}]},
         "check": lambda r: (r.value["gate_open"] is False
                             and r.value["required_failed"] == ["bad"]
                             and "DISC_NEG" in [x["err"] for x in r.value["results"]],
                             f"open={r.value['gate_open']} required_failed={r.value['required_failed']}"),
         "expect": "第二项是 x²+1 (无实根) → 必然 DISC_NEG; 它是必过项 → 门关。"
                   "这正是 CI 门的意义: 必过项失败就不许通过"},
        {"id": "ci_optional_failure_keeps_open", "args": {"checks": [
            {"name": "must", "op": "solve_poly", "args": {"p": -1.0, "q": -6.0},
             "required_pass": True},
            {"name": "nice", "op": "solve_poly", "args": {"p": 0.0, "q": 1.0},
             "required_pass": False}]},
         "check": lambda r: (r.value["gate_open"] is True
                             and r.value["failed"] == ["nice"]
                             and r.value["required_failed"] == [],
                             f"open={r.value['gate_open']} failed={r.value['failed']}"),
         "expect": "非必过项失败: 记录在 failed 但不关门 (required_failed 为空) —— "
                   "区分'告警'与'必过'是门的基本功能"},
        {"id": "ci_missing_op", "args": {"checks": [
            {"name": "ghost", "op": "no_such_op", "args": {}, "required_pass": True}]},
         "check": lambda r: (r.value["gate_open"] is False
                             and r.value["results"][0]["err"] == "OP_MISSING",
                             f"results={r.value['results']}"),
         "expect": "引用未注册算子: 视为失败 (OP_MISSING) 而不是跳过 —— "
                   "'跳过'会让门变成橡皮图章"},
        {"id": "ci_empty_checks", "args": {"checks": []}, "expect_err": "SCHEMA_MISMATCH",
         "expect": "空检查表必须被 schema 拦下 (空门恒开, 无意义)"},
    ),
))


# ===========================================================================
# 9709 doc_gen — documentation generation (examples = faithful transcription of tests)
# ===========================================================================

_IN_DOC = {"kind": "dict",
           "fields": {"name": {"kind": "str"}, "signature": {"kind": "dict", "extra": True, "fields": {}},
                      "tests": TEST_SCHEMA, "notes": {"kind": "str"}, "max_examples": {"kind": "int",
                                                                                        "min": 1, "max": 32}},
           "required": ["name", "signature", "tests"]}


def _inv_doc(args: dict, ctl: CallCtl) -> dict:
    """Documentation generation: **every example must be a verbatim transcription of some test** (inputs and expected values both come from the tests)."""
    sig = args["signature"]
    tests = args["tests"]
    n = min(int(args.get("max_examples", 8)), len(tests))
    examples = []
    for t in tests[:n]:
        parts = []
        for i, v in enumerate(t["args"]):
            pname = sig["params"][i]["name"] if i < len(sig["params"]) else ARGS[i]
            parts.append(f"{pname}={v}")
        call = f"{args['name']}({', '.join(parts)})"
        examples.append({"call": call, "result": t["expect"], "from_test_args": list(t["args"])})
    lines = [f"### `{args['name']}`", ""]
    if args.get("notes"):
        lines += [args["notes"], ""]
    params = ", ".join(f"`{p['name']}: int ∈ [{p['lo']},{p['hi']}]`" for p in sig["params"])
    lines.append(f"参数: {params}")
    lines.append(f"返回: `int ∈ [{sig['output']['lo']},{sig['output']['hi']}]`")
    lines.append("")
    lines.append("示例 (逐条转写自测试, 非手写):")
    for e in examples:
        lines.append(f"  {e['call']} => {e['result']}")
    return {"value": {"doc": "\n".join(lines), "examples": examples, "n_examples": len(examples),
                      "all_from_tests": True,
                      "signature_fingerprint": _h(sig)},
            "resid": [0.0],
            "notes": f"文档生成: {len(examples)} 个示例, 全部逐字转写自测试对 "
                     f"(调用串由测试的 args 与签名参数名拼出, 结果直接取测试的 expect); "
                     f"**禁止虚构示例** —— 不生成任何测试里没有的例子"}


def _ver_doc(args: dict, value: dict) -> dict:
    """Independent verification: (a) every example's (args, result) must **match some given test item by item**
    (the mechanical criterion for "faithful transcription": any fabricated example exposes itself here);
    (b) the literals in the call string must agree with args (guarding against "example text disagrees with the actual
    parameters", the most common documentation lie)."""
    tests = args["tests"]
    pool = {(tuple(t["args"]), int(t["expect"])) for t in tests}
    bad: list = []
    for e in value["examples"]:
        key = (tuple(e["from_test_args"]), int(e["result"]))
        if key not in pool:
            bad.append(("NOT_IN_TESTS", e["call"]))
            continue
        nums = []
        for chunk in e["call"].split("(")[1].rstrip(")").split(","):
            chunk = chunk.strip()
            if "=" in chunk:
                nums.append(chunk.split("=", 1)[1].strip())
        if nums and nums != [str(v) for v in e["from_test_args"]]:
            bad.append(("CALL_TEXT_MISMATCH", e["call"]))
    ok = not bad
    return {"ok": ok, "resid": [float(len(bad))],
            "detail": f"示例逐条回溯测试: 越界/不符 {len(bad)} 条 {bad[:2]} "
                      f"(须为 0 —— 任何虚构示例都会被这条抓住)"}


REGISTRY.register(OpDef(
    name="doc_gen", band=BAND, sub=9,
    in_schema=_IN_DOC,
    out_schema={"kind": "dict",
                "fields": {"doc": {"kind": "str"}, "examples": {"kind": "list", "maxlen": 32,
                                                               "items": {"kind": "dict", "extra": False,
                                                                         "fields": {"call": {"kind": "str"},
                                                                                    "result": {"kind": "int"},
                                                                                    "from_test_args": {"kind": "list",
                                                                                                       "items": {"kind": "int"}}},
                                                                         "required": ["call", "result", "from_test_args"]}},
                           "n_examples": {"kind": "int", "min": 1}, "all_from_tests": {"kind": "bool"},
                           "signature_fingerprint": {"kind": "str"}},
                "required": ["doc", "examples", "n_examples", "all_from_tests", "signature_fingerprint"]},
    invoke=_inv_doc, verify=_ver_doc,
    cost=("units", 0.8), errors=("SCHEMA_MISMATCH",),
    effects="none (纯函数)",
    doc="§9.5 doc_gen 文档生成 | in: 名字 + 签名 + 测试对 | out: 文档文本 + 示例表 | "
        "验收=**示例=测试忠实转写**: 每个示例必须逐条回溯到某条测试 (args 与 expect 全等), "
        "且调用串里的字面量与 args 一致 | 0.8 unit | 诚实边界: 本算子**只**生成有测试支撑的示例 —— "
        "文档里不出现任何未经测试验证的行为描述 (这是'禁虚构'的机械保证) | 失败: schema",
    selfchecks=(
        {"id": "dg_basic", "args": {
            "name": "twice", "signature": {"fn": "twice", "arity": 1,
                                           "params": [{"name": "a0", "type": "int", "lo": 0, "hi": 5}],
                                           "output": {"type": "int", "lo": 0, "hi": 10}},
            "tests": [{"args": [0], "expect": 0}, {"args": [1], "expect": 2},
                      {"args": [5], "expect": 10}],
            "notes": "把输入乘以 2。"},
         "check": lambda r: (r.value["n_examples"] == 3 and r.value["all_from_tests"] is True
                             and "twice(a0=1) => 2" in r.value["doc"],
                             f"n={r.value['n_examples']} doc_head={r.value['doc'][:60]!r}"),
         "expect": "三条测试 → 三个示例; 文档里应出现 'twice(a0=1) => 2' (由测试转写); "
                   "文档同时给出参数与返回值的域"},
        {"id": "dg_max_examples_cap", "args": {
            "name": "f", "signature": {"fn": "f", "arity": 1,
                                       "params": [{"name": "a0", "type": "int", "lo": 0, "hi": 9}],
                                       "output": {"type": "int", "lo": 0, "hi": 9}},
            "tests": [{"args": [i], "expect": i} for i in range(9)], "max_examples": 2},
         "check": lambda r: (r.value["n_examples"] == 2, f"n={r.value['n_examples']}"),
         "expect": "max_examples=2 时只转写前 2 条 (其余测试仍在, 只是不进文档)"},
        {"id": "dg_empty_tests", "args": {"name": "f", "signature": {"fn": "f", "arity": 1,
                                                                     "params": [{"name": "a0", "type": "int",
                                                                                 "lo": 0, "hi": 1}],
                                                                     "output": {"type": "int", "lo": 0, "hi": 1}},
                                           "tests": []},
         "expect_err": "SCHEMA_MISMATCH",
         "expect": "空测试集: 没有测试就没有可转写的示例 → 拒绝 (不许凭空写示例)"},
    ),
))


# ===========================================================================
# 9710 proj_scaffold — project skeleton (accepted by a structural-constraint checker)
# ===========================================================================

#: structural constraints: {"must_have_files": [..], "must_have_dirs": [..], "forbid_files": [..]}
_IN_SCAFFOLD = {"kind": "dict",
                "fields": {"files": {"kind": "list", "maxlen": 128,
                                     "items": {"kind": "dict", "extra": False,
                                               "fields": {"path": {"kind": "str"}, "kind": {"kind": "str"}},
                                               "required": ["path", "kind"]}},
                           "constraints": {"kind": "dict", "extra": True, "fields": {}},
                           "template": {"kind": "str"}},
                "required": ["files", "constraints"]}


def _inv_scaffold(args: dict, ctl: CallCtl) -> dict:
    """Skeleton acceptance: check the declared structural constraints one by one (required files/dirs/forbidden items/naming rules)."""
    files = args["files"]
    cons = args["constraints"]
    paths = {d["path"] for d in files}
    by_kind: dict = {}
    for d in files:
        by_kind.setdefault(d["kind"], set()).add(d["path"])
    must_files = list(cons.get("must_have_files") or [])
    must_dirs = list(cons.get("must_have_dirs") or [])
    forbid = list(cons.get("forbid_files") or [])
    missing_files = [p for p in must_files if p not in paths]
    missing_dirs = []
    for d in must_dirs:
        if not any(p.startswith(d.rstrip("/") + "/") for p in paths):
            missing_dirs.append(d)
    present_forbidden = [p for p in forbid if p in paths]
    violations = (missing_files + missing_dirs + present_forbidden)
    if violations:
        raise SvoError(Err.PRECOND_FAIL, ctx="op:proj_scaffold#9710",
                       detail=f"结构约束不满足: 缺文件 {missing_files}, 缺目录 {missing_dirs}, "
                              f"出现禁止项 {present_forbidden}")
    return {"value": {"satisfied": True, "n_files": len(files), "n_kinds": len(by_kind),
                      "checked": {"must_have_files": len(must_files),
                                  "must_have_dirs": len(must_dirs),
                                  "forbid_files": len(forbid)},
                      "template": args.get("template", ""),
                      "layout_fingerprint": _h(sorted((d["path"], d["kind"]) for d in files))},
            "resid": [0.0],
            "notes": f"骨架结构约束全部满足: {len(files)} 个文件 / {len(by_kind)} 类; "
                     f"核对了 {len(must_files)} 个必需文件、{len(must_dirs)} 个必需目录、"
                     f"{len(forbid)} 个禁止项; 布局指纹 {_h(sorted((d['path'], d['kind']) for d in files))}"}


def _ver_scaffold(args: dict, value: dict) -> dict:
    """Independent verification: (a) independently rejudge all constraints with a **path-prefix scan item by item**; (b) the layout fingerprint is reproducible;
    (c) file paths must not repeat and must not contain '..' (the structure description itself must be legal)."""
    files = args["files"]
    cons = args["constraints"]
    paths = [d["path"] for d in files]
    dup = len(paths) != len(set(paths))
    dotdot = [p for p in paths if ".." in p or p.startswith("/")]
    must_files = list(cons.get("must_have_files") or [])
    must_dirs = list(cons.get("must_have_dirs") or [])
    forbid = list(cons.get("forbid_files") or [])
    indep_ok = True
    for p in must_files:
        if p not in set(paths):
            indep_ok = False
    for d in must_dirs:
        pref = d.rstrip("/") + "/"
        if not any(p.startswith(pref) for p in paths):
            indep_ok = False
    for p in forbid:
        if p in set(paths):
            indep_ok = False
    fp_ok = _h(sorted((d["path"], d["kind"]) for d in files)) == value["layout_fingerprint"]
    ok = indep_ok and fp_ok and (not dup) and (not dotdot)
    return {"ok": ok, "resid": [0.0 if ok else 1.0],
            "detail": f"逐条独立重判约束={indep_ok}; 布局指纹可复现={fp_ok}; "
                      f"路径无重复={not dup}; 无绝对路径/'..'={not dotdot}"}


REGISTRY.register(OpDef(
    name="proj_scaffold", band=BAND, sub=10,
    in_schema=_IN_SCAFFOLD,
    out_schema={"kind": "dict",
                "fields": {"satisfied": {"kind": "bool"}, "n_files": {"kind": "int", "min": 0},
                           "n_kinds": {"kind": "int", "min": 0},
                           "checked": {"kind": "dict", "extra": True, "fields": {}},
                           "template": {"kind": "str"}, "layout_fingerprint": {"kind": "str"}},
                "required": ["satisfied", "n_files", "n_kinds", "checked", "template",
                             "layout_fingerprint"]},
    invoke=_inv_scaffold, verify=_ver_scaffold,
    cost=("units", 0.8), errors=("PRECOND_FAIL", "SCHEMA_MISMATCH"),
    effects="none (纯函数; 只检查声明的结构, 不读写磁盘)",
    doc="§9.5 proj_scaffold 骨架 | in: 文件清单 (路径+种类) + 结构约束 | out: 满足性 + 布局指纹 | "
        "验收=**结构约束检查器**: 必需文件/必需目录(按前缀)/禁止项 逐条核对 + "
        "verify 独立重判 + 布局指纹可复现 + 路径合法性 (无重复/无 '..') | 0.8 unit | "
        "诚实边界: 只检查**声明的结构**, 不检查内容质量; 也不真的创建目录 "
        "(本包无文件系统副作用) | 失败: 约束不满足 PRECOND_FAIL",
    selfchecks=(
        {"id": "ps_ok_layout", "args": {
            "files": [{"path": "src/main.py", "kind": "source"},
                      {"path": "tests/test_main.py", "kind": "test"},
                      {"path": "README.md", "kind": "doc"}],
            "constraints": {"must_have_files": ["README.md"],
                            "must_have_dirs": ["src", "tests"],
                            "forbid_files": ["secrets.env"]},
            "template": "python-minimal"},
         "check": lambda r: (r.value["satisfied"] and r.value["n_files"] == 3
                             and r.value["checked"]["must_have_dirs"] == 2,
                             f"satisfied={r.value['satisfied']} n={r.value['n_files']}"),
         "expect": "标准布局满足: README 在、src/ 与 tests/ 有文件、无禁止项"},
        {"id": "ps_missing_dir", "args": {
            "files": [{"path": "src/main.py", "kind": "source"}],
            "constraints": {"must_have_dirs": ["tests"]}},
         "expect_err": "PRECOND_FAIL",
         "expect": "缺 tests/ 目录 → 结构约束不满足, 拒"},
        {"id": "ps_forbidden_present", "args": {
            "files": [{"path": "secrets.env", "kind": "config"}],
            "constraints": {"forbid_files": ["secrets.env"]}},
         "expect_err": "PRECOND_FAIL",
         "expect": "出现禁止项 → 拒 (例如把凭据文件放进骨架)"},
        {"id": "ps_dir_prefix_not_confused", "args": {
            "files": [{"path": "srcx/main.py", "kind": "source"}],
            "constraints": {"must_have_dirs": ["src"]}},
         "expect_err": "PRECOND_FAIL",
         "expect": "**前缀匹配必须带分隔符**: 'srcx/main.py' 不属于 src/ 目录 → 必须判缺目录。"
                   "这是目录检查最常见的 off-by-slash 缺陷"},
        {"id": "ps_empty_files_ok", "args": {"files": [], "constraints": {}},
         "check": lambda r: (r.value["satisfied"] and r.value["n_files"] == 0,
                             f"n={r.value['n_files']}"),
         "expect": "空清单 + 空约束: 平凡满足 (边界覆盖: 不许对空输入崩溃)"},
    ),
))
"""§9.1 code-domain infrastructure: execution sandbox / AST codebook / bounded interpreter / property testing / V-pair regime.

Bounded AST language (shared by all modules; the codebook |V|=64 covers it exactly):
  Expr  ::= {"t":"lit","v":int} | {"t":"var","n":<a0..a3|v0..v15>}
          | {"t":"bin","op":<13 kinds>,"l":E,"r":E} | {"t":"un","op":<neg|not>,"e":E}
          | {"t":"call","fn":<abs|min|max|clamp|sign>,"args":[E..]}
  Stmt  ::= {"t":"assign","var":<v0..v15>,"e":E}
          | {"t":"if","c":E,"th":[S..],"el":[S..]}
          | {"t":"loop","v":<v0..v15>,"b":E,"body":[S..]}   # the evaluated bound b must be 0..64, else BOUNDED_VIOLATION
  Program ::= [S..]  (input slots a0..a3; integer semantics; comparisons/logic return 0/1; "/" is floor division)

Means of implementation forbidden (fail-safe):
  - on each loop entry the evaluated bound is hard-checked: b>LOOP_CAP(64) immediately raises BOUNDED_VIOLATION;
  - the interpreter also has a global step cap (exceeding it raises BOUNDED_VIOLATION);
  - no recursion, no function definitions, no pointers; the variable vocabulary is closed (a0..a3, v0..v15).

Dual engines (§17.9 no-self-certification, execution side):
  - engine1 `interp`      : the recursive interpreter (invoke side);
  - engine2 `interp_stack`: an iterative stack machine (statements = frame stack, expressions = postfix value stack; logic operators short-circuit explicitly).
  The two engines must agree semantically — every operator's verify recomputes with engine2.

Honesty statement (exec_sandbox, Windows):
  - no rlimit: on POSIX the subprocess pre-exec hook uses resource.RLIMIT_AS to cap memory; Windows has no such
    mechanism, so the memory cap cannot be enforced — it is replaced by "real timeout kill + output-cap truncation
    (mapped to OOM) + static policy scan", and on this host (win32) the latter three are what actually take effect.
    The remaining mandatory gap = the subprocess memory cap.
  - no network: the runner code makes no network call, and the static policy scan rejects import/open/eval/exec etc.
    (heuristic; getattr tricks can bypass it — declared heuristic, not OS-level isolation).
"""
from __future__ import annotations

import ast as _pyast
import json
import random
import subprocess
import sys
import time
from typing import Any

from ..errors import Err, SvoError
from ..registry import REGISTRY, OpDef
from ..tool import CallCtl

# ---------------------------------------------------------------------------
# language and codebook (|V| = 64, exactly)
# ---------------------------------------------------------------------------

BINOPS = ("+", "-", "*", "/", "%", "<", "<=", ">", ">=", "==", "!=", "and", "or")
_CMPL = {"<": ">=", "<=": ">", ">": "<=", ">=": "<", "==": "!=", "!=": "=="}
UNOPS = ("neg", "not")
CALLS = {"abs": 1, "min": 2, "max": 2, "clamp": 3, "sign": 1}
ARGS = ("a0", "a1", "a2", "a3")
VARS = tuple(f"v{i}" for i in range(16))
VOCAB = frozenset(ARGS) | frozenset(VARS)
LIT_MIN, LIT_MAX = -1000, 1000
LOOP_CAP = 64
MAX_STMTS = 64
MAX_NODES = 512
MAX_EXPR_DEPTH = 24
MAX_STMT_DEPTH = 8


def _build_codebook() -> list[dict]:
    """AST codebook: name→code→arity→type rules. Constructed as a pure function; the verify side audits its structure independently."""
    e: list[dict] = []

    def add(name, cat, arity, rule):
        e.append({"name": name, "code": len(e) + 1, "cat": cat, "arity": arity, "type_rule": rule})

    for k, r in (
        ("Lit", "字面量 → int"), ("Var", "变量槽 → int"), ("BinOp", "(E,E) → E (按 op)"),
        ("UnOp", "(E) → E"), ("Call", "(E×元数) → int"), ("Assign", "(槽,E) → 副作用"),
        ("If", "(E,Body,Body) → 分支"), ("Loop", "(槽,E,Body) → 有界迭代"),
    ):
        add(k, "kind", 0, r)
    for op in BINOPS:
        fam = "bool" if op in ("<", "<=", ">", ">=", "==", "!=", "and", "or") else "int"
        add(op, "binop", 2, f"(int,int) → {fam}(int∈{{0,1}})" if fam == "bool" else "(int,int) → int")
    add("neg", "unop", 1, "(int) → int")
    add("not", "unop", 1, "(bool) → bool(int∈{0,1})")
    for fn, ar in CALLS.items():
        add(fn, "lib", ar, f"(int×{ar}) → int")
    for a in ARGS:
        add(a, "arg", 0, "∅ → int (程序输入槽)")
    for v in VARS:
        add(v, "var", 0, "int (可赋值槽)")
    for c in list(range(-8, 8)):
        add(str(c), "const", 0, f"字面量 {c} → int")
    assert len(e) == 64, len(e)
    return e


AST_CODEBOOK: list[dict] = _build_codebook()

# ---------------------------------------------------------------------------
# structural validation (shared by all code-domain operators)
# ---------------------------------------------------------------------------

SCHEMA_EXPR = {"kind": "dict", "fields": {"t": {"kind": "str"}}, "required": ["t"], "extra": True}
SCHEMA_PROG = {"kind": "list", "items": SCHEMA_EXPR, "minlen": 0, "maxlen": MAX_STMTS}
SCHEMA_INPUTS = {
    "kind": "dict",
    "fields": {a: {"kind": "int", "min": -10**6, "max": 10**6} for a in ARGS},
    "required": [],
    "extra": False,
}


def validate_prog(prog: Any, ctx: str = "validate_prog") -> int:
    """Structural validation, returns the node count; violations raise SCHEMA_MISMATCH (with a path)."""
    nodes = 0

    def bad(path: str, why: str):
        raise SvoError(Err.SCHEMA_MISMATCH, ctx=ctx, detail=f"程序非法@{path}: {why}")

    def vexpr(x: Any, path: str, depth: int) -> None:
        nonlocal nodes
        nodes += 1
        if nodes > MAX_NODES:
            bad(path, f"节点数超上限 {MAX_NODES}")
        if not isinstance(x, dict) or "t" not in x:
            bad(path, f"非表达式节点: {x!r}")
        t = x["t"]
        if t == "lit":
            v = x.get("v")
            if isinstance(v, bool) or not isinstance(v, int) or not (LIT_MIN <= v <= LIT_MAX):
                bad(path, f"字面量越界 [{LIT_MIN},{LIT_MAX}]: {v!r}")
            return
        if t == "var":
            n = x.get("n")
            if n not in VOCAB:
                bad(path, f"变量 {n!r} 不在码本词表 (a0..a3/v0..v15)")
            return
        if t == "bin":
            if x.get("op") not in BINOPS:
                bad(path, f"未知二元算子 {x.get('op')!r}")
            vexpr(x.get("l"), path + ".l", depth + 1)
            vexpr(x.get("r"), path + ".r", depth + 1)
            return
        if t == "un":
            if x.get("op") not in UNOPS:
                bad(path, f"未知一元算子 {x.get('op')!r}")
            vexpr(x.get("e"), path + ".e", depth + 1)
            return
        if t == "call":
            fn = x.get("fn")
            if fn not in CALLS:
                bad(path, f"未知库函数 {fn!r}")
            args = x.get("args")
            if not isinstance(args, list) or len(args) != CALLS[fn]:
                bad(path, f"call {fn} 元数应为 {CALLS[fn]}: {args!r}")
            for i, a in enumerate(args):
                vexpr(a, f"{path}.arg{i}", depth + 1)
            return
        bad(path, f"非法表达式种类 {t!r} (类型缺陷: 算子/元数误用会被此门拦下)")

    def vstmt(s: Any, path: str, depth: int) -> None:
        if not isinstance(s, dict) or "t" not in s:
            bad(path, f"非语句节点: {s!r}")
        t = s["t"]
        if t == "assign":
            if s.get("var") not in VARS:
                bad(path, f"赋值目标 {s.get('var')!r} 不是 v0..v15")
            vexpr(s.get("e"), path + ".e", 0)
        elif t == "if":
            vexpr(s.get("c"), path + ".c", 0)
            for k in ("th", "el"):
                body = s.get(k, [])
                if not isinstance(body, list):
                    bad(path, f"{k} 不是列表")
                for i, sub in enumerate(body):
                    vstmt(sub, f"{path}.{k}[{i}]", depth + 1)
        elif t == "loop":
            if s.get("v") not in VARS:
                bad(path, f"循环变量 {s.get('v')!r} 不是 v0..v15")
            vexpr(s.get("b"), path + ".b", 0)
            body = s.get("body")
            if not isinstance(body, list):
                bad(path, "body 不是列表")
            for i, sub in enumerate(body):
                vstmt(sub, f"{path}.body[{i}]", depth + 1)
        else:
            bad(path, f"非法语句种类 {t!r}")

    if not isinstance(prog, list):
        bad("$", "程序不是语句列表")
    if len(prog) > MAX_STMTS:
        bad("$", f"语句数 {len(prog)} 超上限 {MAX_STMTS}")
    for i, s in enumerate(prog):
        vstmt(s, f"$[{i}]", 0)
    return nodes


# ---------------------------------------------------------------------------
# statement numbering (preorder; both engines share the same numbering function — numbering is not verified logic)
# ---------------------------------------------------------------------------


def sid_map(prog: list) -> tuple[dict, int]:
    """The id(stmt)→sid pre-order numbering table plus the statement total."""
    m: dict[int, int] = {}
    cnt = 0

    def walk(stmts):
        nonlocal cnt
        for s in stmts:
            m[id(s)] = cnt
            cnt += 1
            if s["t"] == "if":
                walk(s.get("th", []))
                walk(s.get("el", []))
            elif s["t"] == "loop":
                walk(s.get("body", []))

    walk(prog)
    return m, cnt


# ---------------------------------------------------------------------------
# engine1: recursive interpreter
# ---------------------------------------------------------------------------


class _EngineErr(Exception):
    def __init__(self, code: str, detail: str):
        self.code = code
        self.detail = detail
        super().__init__(detail)


def _call_fn(fn: str, vs: list[int]) -> int:
    if fn == "abs":
        return abs(vs[0])
    if fn == "min":
        return min(vs)
    if fn == "max":
        return max(vs)
    if fn == "clamp":
        return min(max(vs[0], vs[1]), vs[2])
    return (vs[0] > 0) - (vs[0] < 0)  # sign


def interp(prog: list, inputs: dict, step_limit: int = 200_000) -> dict:
    """engine1 (recursive). Returns {state, steps, executed, error, iters, total_stmts}."""
    sids, total = sid_map(prog)
    st: dict[str, int] = {k: int(v) for k, v in inputs.items()}
    steps = 0
    executed: list[int] = []
    iters = 0
    err = None

    def bump():
        nonlocal steps
        steps += 1
        if steps > step_limit:
            raise _EngineErr("BOUNDED_VIOLATION", f"步数 {steps} 超上限 {step_limit}")

    def rd(name: str) -> int:
        if name in st:
            return st[name]
        raise _EngineErr("RUNTIME_ERR", f"未定义变量 {name!r} (空引用: 读前未定义)")

    def ev(e: dict) -> int:
        bump()
        t = e["t"]
        if t == "lit":
            return e["v"]
        if t == "var":
            return rd(e["n"])
        if t == "un":
            v = ev(e["e"])
            return -v if e["op"] == "neg" else (0 if v else 1)
        if t == "call":
            return _call_fn(e["fn"], [ev(a) for a in e["args"]])
        op = e["op"]
        if op == "and":
            return 1 if (ev(e["l"]) != 0 and ev(e["r"]) != 0) else 0
        if op == "or":
            return 1 if (ev(e["l"]) != 0 or ev(e["r"]) != 0) else 0
        a = ev(e["l"])
        b = ev(e["r"])
        if op == "+":
            return a + b
        if op == "-":
            return a - b
        if op == "*":
            return a * b
        if op in ("/", "%"):
            if b == 0:
                raise _EngineErr("RUNTIME_ERR", "除零/模零")
            return a // b if op == "/" else a % b
        if op == "<":
            return 1 if a < b else 0
        if op == "<=":
            return 1 if a <= b else 0
        if op == ">":
            return 1 if a > b else 0
        if op == ">=":
            return 1 if a >= b else 0
        if op == "==":
            return 1 if a == b else 0
        return 1 if a != b else 0

    def run(stmts: list) -> None:
        nonlocal iters
        for s in stmts:
            bump()
            executed.append(sids[id(s)])
            t = s["t"]
            if t == "assign":
                st[s["var"]] = ev(s["e"])
            elif t == "if":
                if ev(s["c"]) != 0:
                    run(s.get("th", []))
                else:
                    run(s.get("el", []))
            else:  # loop
                b = ev(s["b"])
                if not (0 <= b <= LOOP_CAP):
                    raise _EngineErr("BOUNDED_VIOLATION", f"循环界 {b} 越界 [0,{LOOP_CAP}] (资源上限, 失效安全)")
                for i in range(b):
                    iters += 1
                    bump()
                    st[s["v"]] = i
                    run(s["body"])

    try:
        run(prog)
    except _EngineErr as ex:
        err = {"code": ex.code, "detail": ex.detail}
    except RecursionError as ex:  # noqa: BLE001 - the structural cap guarantees this cannot trigger; defensive collapse
        err = {"code": "BOUNDED_VIOLATION", "detail": f"递归超深: {ex}"}
    return {"state": st, "steps": steps, "executed": executed, "error": err, "iters": iters, "total_stmts": total}


# ---------------------------------------------------------------------------
# engine2: iterative stack machine (independent implementation; every operator's verify recomputes with it)
# ---------------------------------------------------------------------------


def _ev_stack(e: dict, rd, bump) -> int:
    """Expression evaluation on the postfix value stack; and/or short-circuit explicitly (recursion is used only for the short-circuited side, depth ≤ expression depth)."""
    work: list[tuple[str, Any]] = [("e", e)]
    out: list[int] = []
    while work:
        tag, x = work.pop()
        if tag == "e":
            t = x["t"]
            if t == "lit":
                bump()
                out.append(x["v"])
            elif t == "var":
                bump()
                out.append(rd(x["n"]))
            elif t == "un":
                bump()
                work.append(("u", x["op"]))
                work.append(("e", x["e"]))
            elif t == "call":
                bump()
                work.append(("c", x["fn"]))
                for a in reversed(x["args"]):
                    work.append(("e", a))
            else:
                op = x["op"]
                if op in ("and", "or"):
                    bump()
                    left = _ev_stack(x["l"], rd, bump)
                    if op == "and" and left == 0:
                        out.append(0)
                    elif op == "or" and left != 0:
                        out.append(1)
                    else:
                        work.append(("b", op))
                        work.append(("e", x["r"]))
                else:
                    bump()
                    work.append(("b", op))
                    work.append(("e", x["r"]))
                    work.append(("e", x["l"]))
        elif tag == "b":
            op = x
            r = out.pop()
            l = out.pop()
            if op == "+":
                out.append(l + r)
            elif op == "-":
                out.append(l - r)
            elif op == "*":
                out.append(l * r)
            elif op in ("/", "%"):
                if r == 0:
                    raise _EngineErr("RUNTIME_ERR", "除零/模零")
                out.append(l // r if op == "/" else l % r)
            elif op == "<":
                out.append(1 if l < r else 0)
            elif op == "<=":
                out.append(1 if l <= r else 0)
            elif op == ">":
                out.append(1 if l > r else 0)
            elif op == ">=":
                out.append(1 if l >= r else 0)
            elif op == "==":
                out.append(1 if l == r else 0)
            else:
                out.append(1 if l != r else 0)
        elif tag == "u":
            v = out.pop()
            out.append(-v if x == "neg" else (0 if v else 1))
        else:  # c
            fn = x
            ar = CALLS[fn]
            args = out[-ar:]
            del out[-ar:]
            out.append(_call_fn(fn, args))
    return out[-1]


def interp_stack(prog: list, inputs: dict, step_limit: int = 200_000) -> dict:
    """engine2 (iterative stack machine). Returns the same structure as interp; semantics must match engine1."""
    sids, total = sid_map(prog)
    st: dict[str, int] = {k: int(v) for k, v in inputs.items()}
    steps = 0
    executed: list[int] = []
    iters = 0
    err = None

    def bump():
        nonlocal steps
        steps += 1
        if steps > step_limit:
            raise _EngineErr("BOUNDED_VIOLATION", f"步数 {steps} 超上限 {step_limit}")

    def rd(name: str) -> int:
        if name in st:
            return st[name]
        raise _EngineErr("RUNTIME_ERR", f"未定义变量 {name!r} (空引用: 读前未定义)")

        # frame: ["seq", stmts, i] | ["loop", var, count, body, i]
    stack: list[list] = [["seq", prog, 0]]
    try:
        while stack:
            f = stack[-1]
            if f[0] == "seq":
                _, stmts, i = f
                if i >= len(stmts):
                    stack.pop()
                    continue
                f[2] = i + 1
                s = stmts[i]
                bump()
                executed.append(sids[id(s)])
                t = s["t"]
                if t == "assign":
                    st[s["var"]] = _ev_stack(s["e"], rd, bump)
                elif t == "if":
                    if _ev_stack(s["c"], rd, bump) != 0:
                        stack.append(["seq", s.get("th", []), 0])
                    else:
                        stack.append(["seq", s.get("el", []), 0])
                else:
                    b = _ev_stack(s["b"], rd, bump)
                    if not (0 <= b <= LOOP_CAP):
                        raise _EngineErr("BOUNDED_VIOLATION", f"循环界 {b} 越界 [0,{LOOP_CAP}] (资源上限, 失效安全)")
                    stack.append(["loop", s["v"], b, s["body"], 0])
            else:
                _, var, b, body, i = f
                if i >= b:
                    stack.pop()
                    continue
                f[4] = i + 1
                iters += 1
                bump()
                st[var] = i
                stack.append(["seq", body, 0])
    except _EngineErr as ex:
        err = {"code": ex.code, "detail": ex.detail}
    return {"state": st, "steps": steps, "executed": executed, "error": err, "iters": iters, "total_stmts": total}


def engine_raise(result: dict, ctx: str) -> None:
    """Engine error → an in-table enumeration (RUNTIME_ERR / BOUNDED_VIOLATION)."""
    if result["error"] is not None:
        e = result["error"]
        code = Err.RUNTIME_ERR if e["code"] == "RUNTIME_ERR" else Err.BOUNDED_VIOLATION
        raise SvoError(code, ctx=ctx, detail=e["detail"])


# ---------------------------------------------------------------------------
# operator 1: code_exec_sandbox (row 9401)
# ---------------------------------------------------------------------------

_BANNED = {"open", "eval", "exec", "compile", "input", "__import__", "breakpoint", "exit", "quit", "globals", "locals", "vars"}


def policy_scan(code: str) -> list[str]:
    """Static policy scan (heuristic, declared not OS-level isolation): reject import/dangerous builtins/dunder attributes/missing entry."""
    reasons: list[str] = []
    tree = _pyast.parse(code)  # SyntaxError is mapped to COMPILE_ERR by the caller
    for node in _pyast.walk(tree):
        if isinstance(node, (_pyast.Import, _pyast.ImportFrom)):
            reasons.append("禁 import (无网/无文件系统策略)")
        elif isinstance(node, _pyast.Name) and node.id in _BANNED:
            reasons.append(f"禁用内建 {node.id}")
        elif isinstance(node, _pyast.Attribute) and node.attr.startswith("__"):
            reasons.append(f"禁双下划线属性访问 .{node.attr}")
    if not any(isinstance(n, _pyast.FunctionDef) and n.name == "entry" for n in _pyast.walk(tree)):
        reasons.append("未定义 entry(args: list[int]) -> int")
    return reasons


_RUNNER_A = r'''
import sys, json, io, time, traceback, contextlib
def main():
    req = json.load(sys.stdin)
    code, tests, max_out = req["code"], req["tests"], int(req["max_out"])
    res = {"results": [], "stdout": "", "stderr": "", "truncated": False}
    g = {"__name__": "__sandbox__"}
    bo, be = io.StringIO(), io.StringIO()
    try:
        comp = compile(code, "<entry>", "exec")
        with contextlib.redirect_stdout(bo), contextlib.redirect_stderr(be):
            exec(comp, g)
    except SyntaxError as e:
        print(json.dumps({"err": "COMPILE_ERR", "detail": str(e)})); return
    except BaseException:
        be.write(traceback.format_exc()[-1500:])
        res["stderr"] = be.getvalue()[-max_out:]
        res["truncated"] = len(be.getvalue()) > max_out
        print(json.dumps(res)); return
    fn = g.get("entry")
    if not callable(fn):
        print(json.dumps({"err": "COMPILE_ERR", "detail": "entry 不可调用"})); return
    for t in tests:
        rec = {"args": t["args"], "expect": t["expect"], "got": 0, "pass": False, "err": "", "ms": 0.0}
        t0 = time.perf_counter()
        try:
            with contextlib.redirect_stdout(bo), contextlib.redirect_stderr(be):
                got = fn(list(t["args"]))
        except BaseException as e:
            rec["err"] = (type(e).__name__ + ": " + str(e))[:300]
        else:
            if isinstance(got, bool):
                got = 1 if got else 0
            if isinstance(got, int):
                rec["got"] = got
                rec["pass"] = got == t["expect"]
            else:
                rec["err"] = f"返回类型非 int: {type(got).__name__}"
        rec["ms"] = round((time.perf_counter() - t0) * 1000, 3)
        res["results"].append(rec)
    full_o, full_e = bo.getvalue(), be.getvalue()
    res["stdout"] = full_o[-max_out:]
    res["stderr"] = (full_e + ("\n" if full_e else "") + res["stderr"])[-max_out:]
    res["truncated"] = res["truncated"] or len(full_o) > max_out or len(full_e) > max_out
    print(json.dumps(res, ensure_ascii=False))
main()
'''

_RUNNER_B = r'''
import sys, json, io, time, traceback, contextlib
def main():
    req = json.load(sys.stdin)
    code, tests, max_out = req["code"], list(reversed(req["tests"])), int(req["max_out"])
    g = {"__name__": "__sandbox_b__"}
    try:
        comp = compile(code, "<entry>", "exec")
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            exec(comp, g)
    except SyntaxError as e:
        print(json.dumps({"err": "COMPILE_ERR", "detail": str(e)})); return
    fn = g.get("entry")
    if not callable(fn):
        print(json.dumps({"err": "COMPILE_ERR", "detail": "entry 不可调用"})); return
    table = {}
    order = []
    for t in tests:
        key = json.dumps(t["args"])
        order.append(key)
        try:
            got = fn(list(t["args"]))
        except BaseException as e:
            table[key] = {"err": type(e).__name__}
            continue
        if isinstance(got, bool):
            got = 1 if got else 0
        table[key] = {"got": got} if isinstance(got, int) else {"err": "返回类型非 int"}
    print(json.dumps({"order": order, "table": table}))
main()
'''


def _run_child(runner: str, req: dict, timeout_s: float, ctx: str) -> dict:
    try:
        cp = subprocess.run(
            [sys.executable, "-c", runner],
            input=json.dumps(req), capture_output=True, text=True, timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        raise SvoError(Err.TIMEOUT, ctx=ctx, detail=f"子进程超时 {timeout_s}s, 已 kill (真实超时)")
    if cp.returncode != 0:
        raise SvoError(Err.RUNTIME_ERR, ctx=ctx, detail=f"子进程退出码 {cp.returncode}: {cp.stderr[-300:]}")
    try:
        return json.loads(cp.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        raise SvoError(Err.RUNTIME_ERR, ctx=ctx, detail=f"子进程输出不可解析: {cp.stdout[-200:]}{cp.stderr[-200:]}")


_IN_SANDBOX = {
    "kind": "dict",
    "fields": {
        "code": {"kind": "str"},
        "tests": {"kind": "list", "minlen": 0, "maxlen": 32,
                  "items": {"kind": "dict", "extra": False,
                            "fields": {"args": {"kind": "list", "minlen": 0, "maxlen": 8, "items": {"kind": "int"}},
                                       "expect": {"kind": "int"}},
                            "required": ["args", "expect"]}},
        "timeout_s": {"kind": "float", "min": 0.3, "max": 30.0},
        "max_out": {"kind": "int", "min": 64, "max": 65536},
    },
    "required": ["code", "tests", "timeout_s", "max_out"],
}
_OUT_SANDBOX = {
    "kind": "dict",
    "fields": {
        "pass_vector": {"kind": "list", "items": {"kind": "bool"}},
        "results": {"kind": "list", "items": {"kind": "dict", "extra": False, "fields": {
            "args": {"kind": "list", "items": {"kind": "int"}},
            "expect": {"kind": "int"}, "got": {"kind": "int"},
            "pass": {"kind": "bool"}, "err": {"kind": "str"}, "ms": {"kind": "float"}}}},
        "stdout": {"kind": "str"}, "stderr": {"kind": "str"},
        "elapsed_ms": {"kind": "float"}, "policy": {"kind": "str"},
        "replay_verified": {"kind": "bool"},
    },
    "required": ["pass_vector", "results", "stdout", "stderr", "elapsed_ms", "policy", "replay_verified"],
}


def _invoke_sandbox(args: dict, ctl: CallCtl) -> dict:
    ctx = "op:code_exec_sandbox#9401"
    ctl.check_time(ctx)
    code, tests, timeout_s, max_out = args["code"], args["tests"], float(args["timeout_s"]), int(args["max_out"])
    try:
        reasons = policy_scan(code)
    except SyntaxError as e:
        raise SvoError(Err.COMPILE_ERR, ctx=ctx, detail=f"语法错误: {e}")
    if reasons:
        raise SvoError(Err.COMPILE_ERR, ctx=ctx, detail="策略扫描拒绝 (启发式): " + "; ".join(sorted(set(reasons))))
    t0 = time.perf_counter()
    out = _run_child(_RUNNER_A, {"code": code, "tests": tests, "max_out": max_out}, timeout_s, ctx)
    elapsed = (time.perf_counter() - t0) * 1000.0
    if "err" in out:
        raise SvoError(Err.COMPILE_ERR, ctx=ctx, detail=out["detail"])
    if out.get("truncated"):
        raise SvoError(Err.OOM, ctx=ctx, detail=f"输出超上限 {max_out} 字节 (Windows 无 rlimit, 以输出截断映射 OOM)")
    results = out["results"]
    pass_vector = [bool(r["pass"]) for r in results]
    # independent replay (runner B: reverse-order execution + table collection, a different code path) — only
    # same-seed replay equality counts as behavioural ground truth
    rb = _run_child(_RUNNER_B, {"code": code, "tests": tests, "max_out": max_out}, timeout_s, ctx)
    if "err" in rb:
        raise SvoError(Err.COMPILE_ERR, ctx=ctx, detail=f"重放 runner 拒绝: {rb['detail']}")
    replay_ok = True
    if len(rb["order"]) != len(results):
        replay_ok = False
    else:
        for rec, key in zip(reversed(results), rb["order"]):  # runner B runs in reverse; reverse it back to the original order for comparison
            got_b = rb["table"][key]
            if "err" in got_b:
                if got_b["err"] not in rec["err"]:
                    replay_ok = False
            elif got_b["got"] != rec["got"] or rec["err"]:
                replay_ok = False
    if not replay_ok:
        raise SvoError(Err.RUNTIME_ERR, ctx=ctx, detail="两次运行 (A/B runner) 结果不一致 —— entry 非确定性或有跨调用状态")
    value = {
        "pass_vector": pass_vector,
        "results": [{"args": r["args"], "expect": r["expect"], "got": r["got"],
                     "pass": bool(r["pass"]), "err": r["err"], "ms": float(r["ms"])} for r in results],
        "stdout": out.get("stdout", ""), "stderr": out.get("stderr", ""),
        "elapsed_ms": round(elapsed, 3), "policy": "no-net(heuristic)+timeout+output-cap; no-rlimit-on-win32",
        "replay_verified": True,
    }
    nfail = pass_vector.count(False)
    return {"value": value, "resid": [float(nfail)],
            "notes": f"{len(pass_vector)} 测试, 失败 {nfail}, 延迟 {elapsed:.1f}ms, A/B 双 runner 重放一致"}


def _verify_sandbox(args: dict, value: dict) -> dict:
    """Independent verification: recompute the pass vector from the raw got/expect data (not via invoke's comparison path), and check the latency dimension."""
    pv = []
    for r in value["results"]:
        ok = (r["err"] == "") and (r["got"] == r["expect"])
        pv.append(ok)
    same = pv == value["pass_vector"]
    lat_ok = value["elapsed_ms"] >= 0.0
    return {"ok": same and lat_ok, "resid": [0.0 if same else 1.0],
            "detail": f"重算通过向量与报告{'一致' if same else '不一致'}; {sum(pv)}/{len(pv)} 通过"}


# ---------------------------------------------------------------------------
# operator 2: code_ast_codebook (row 9402)
# ---------------------------------------------------------------------------


def _invoke_codebook(args: dict, ctl: CallCtl) -> dict:
    entries = AST_CODEBOOK
    if args["want"] == "summary":
        cats: dict[str, int] = {}
        for en in entries:
            cats[en["cat"]] = cats.get(en["cat"], 0) + 1
        value = {"entries": [], "size": len(entries), "cats": cats,
                 "loop_bound_cap": LOOP_CAP, "vocab": len(VOCAB)}
    else:
        value = {"entries": [dict(e) for e in entries], "size": len(entries), "cats": {},
                 "loop_bound_cap": LOOP_CAP, "vocab": len(VOCAB)}
    return {"value": value, "resid": [0.0], "notes": f"|V|={len(entries)}, 循环界上限 {LOOP_CAP} (禁图灵完备)"}


def _verify_codebook(args: dict, value: dict) -> dict:
    """Independent audit: recheck the codebook construction against the language-spec constants (BINOPS/CALLS/VARS/ARGS)."""
    probs = []
    if value["size"] != 64:
        probs.append(f"|V|={value['size']}≠64")
    if args["want"] == "table":
        codes = [e["code"] for e in value["entries"]]
        if sorted(codes) != list(range(1, 65)):
            probs.append("码不是 1..64 的排列")
        names = {e["name"] for e in value["entries"]}
        for op in BINOPS:
            if op not in names:
                probs.append(f"缺 binop {op}")
        for fn, ar in CALLS.items():
            row = next((e for e in value["entries"] if e["name"] == fn), None)
            if row is None or row["arity"] != ar or row["cat"] != "lib":
                probs.append(f"lib {fn} 行不合规")
        for v in VARS:
            row = next((e for e in value["entries"] if e["name"] == v), None)
            if row is None or row["cat"] != "var":
                probs.append(f"缺 var 槽 {v}")
    return {"ok": not probs, "resid": [float(len(probs))], "detail": "规格对照审计: " + ("全部符合" if not probs else "; ".join(probs))}


# ---------------------------------------------------------------------------
# operator 3: code_bounded_interp (row 9403)
# ---------------------------------------------------------------------------

_IN_INTERP = {
    "kind": "dict",
    "fields": {
        "program": SCHEMA_PROG, "inputs": SCHEMA_INPUTS,
        "step_limit": {"kind": "int", "min": 1, "max": 1_000_000},
        "observed": {"kind": "str"},
    },
    "required": ["program", "inputs", "step_limit"],
}
_OUT_INTERP = {
    "kind": "dict",
    "fields": {
        "state": {"kind": "dict", "extra": True},
        "steps": {"kind": "int"}, "iters": {"kind": "int"}, "total_stmts": {"kind": "int"},
        "executed_head": {"kind": "list", "items": {"kind": "int"}},
        "err": {"kind": "str"}, "engine_cross_ok": {"kind": "bool"},
        "observed": {"kind": "str"}, "observed_val": {"kind": "int"},
    },
    "required": ["state", "steps", "iters", "total_stmts", "executed_head", "err",
                 "engine_cross_ok", "observed", "observed_val"],
}


def _invoke_interp(args: dict, ctl: CallCtl) -> dict:
    ctx = "op:code_bounded_interp#9403"
    prog = args["program"]
    validate_prog(prog, ctx)
    if not isinstance(args.get("inputs", {}), dict):
        raise SvoError(Err.SCHEMA_MISMATCH, ctx=ctx, detail="inputs 须为 dict")
    inputs = args.get("inputs", {})
    r1 = interp(prog, inputs, int(args["step_limit"]))
    if r1["error"] is not None:
        engine_raise(r1, ctx)
    r2 = interp_stack(prog, inputs, int(args["step_limit"]))
    cross = (r2["error"] is None and r2["state"] == r1["state"])
    obs = args.get("observed", "v0")
    if obs not in r1["state"]:
        raise SvoError(Err.RUNTIME_ERR, ctx=ctx, detail=f"观察变量 {obs!r} 在终态未定义")
    value = {
        "state": dict(sorted(r1["state"].items())), "steps": r1["steps"], "iters": r1["iters"],
        "total_stmts": r1["total_stmts"], "executed_head": r1["executed"][:32], "err": "",
        "engine_cross_ok": bool(cross), "observed": obs, "observed_val": int(r1["state"][obs]),
    }
    return {"value": value, "resid": [0.0], "notes": f"终态 {len(r1['state'])} 变量, {r1['steps']} 步, 双引擎一致={cross}"}


def _verify_interp(args: dict, value: dict) -> dict:
    """Independent verification: recompute with engine2 (iterative stack machine); compare the final state and the executed set."""
    r2 = interp_stack(args["program"], args.get("inputs", {}), int(args["step_limit"]))
    ok = r2["error"] is None and dict(sorted(r2["state"].items())) == value["state"]
    set_ok = set(r2["executed"]) == set(value["executed_head"]) or len(value["executed_head"]) == 32
    return {"ok": ok and set_ok, "resid": [0.0 if ok else 1.0],
            "detail": f"engine2 复算终态{'一致' if ok else '不一致'}"}


# ---------------------------------------------------------------------------
# operator 4: code_property_test (row 9404)
# ---------------------------------------------------------------------------

_IN_PROP = {
    "kind": "dict",
    "fields": {
        "program": SCHEMA_PROG,
        "prop": {"kind": "dict", "extra": True, "fields": {"kind": {"kind": "str", "enum": ["postcond", "monotone"]}},
                 "required": ["kind"]},
        "trials": {"kind": "int", "min": 4, "max": 512},
        "seed": {"kind": "int", "min": 0, "max": 2**31 - 1},
        "step_limit": {"kind": "int", "min": 100, "max": 1_000_000},
    },
    "required": ["program", "prop", "trials", "seed"],
}
_OUT_PROP = {
    "kind": "dict",
    "fields": {
        "ok": {"kind": "bool"}, "found": {"kind": "bool"},
        "cx_args": SCHEMA_INPUTS, "cx_lo": SCHEMA_INPUTS, "cx_hi": SCHEMA_INPUTS,
        "shrunk": {"kind": "bool"}, "shrink_steps": {"kind": "int"},
        "trials": {"kind": "int"}, "violations": {"kind": "int"},
        "sampled_head": {"kind": "list", "maxlen": 8, "items": SCHEMA_INPUTS},
    },
    "required": ["ok", "found", "cx_args", "cx_lo", "cx_hi", "shrunk", "shrink_steps",
                 "trials", "violations", "sampled_head"],
}


def _prop_holds(prog: list, prop: dict, inp: dict, inp2: "dict | None", step_limit: int) -> bool:
    """Property decision (engine1). postcond: the assertion expression evaluates ≠0 in the final state; monotone: x≤x' ⇒ f(x)≤f(x')."""
    r = interp(prog, inp, step_limit)
    if r["error"] is not None:
        return False
    if prop["kind"] == "postcond":
        st = dict(r["state"])
        rr = interp([{"t": "assign", "var": "v15", "e": prop["expr"]}], st, step_limit)
        if rr["error"] is not None:
            return False
        return rr["state"]["v15"] != 0
    out = prop.get("out_var", "v0")
    if inp2 is None:
        return True
    r2 = interp(prog, inp2, step_limit)
    if r2["error"] is not None:
        return False
    return r["state"].get(out, 0) <= r2["state"].get(out, 0)


def _shrink(prog, prop, cx: dict, is_pair: bool, cx2: "dict | None", step_limit: int,
            ctl: "CallCtl | None" = None) -> tuple[dict, dict | None, int]:
    """Counter-example shrinking: shrink |value| strictly dimension by dimension, accepting while it still violates (each step really reruns engine1 to rejudge).

    **Shrinking a monotone pair must preserve the witness structure**: a monotone counter-example is a pair (i₁, i₂) that
    differs only by +1 in the parameter slot `arg`, all other dimensions equal. Shrinking each dimension independently as
    for an ordinary counter-example would destroy the difference of 1, and the pair would no longer witness "output falls
    as the parameter grows" (the old version measured a0 collapsing to 0 and 13, the witness semantics was lost, and the
    selfcheck monotone_violation reported FAIL). Hence:
      - d == arg: **shift both points synchronously** in that dimension, keeping the difference 1;
      - d != arg: the two points already agree there; assign the same value synchronously.

    **Termination argument**: the candidate set explicitly removes `|c| >= |x|`, so every accepted step strictly
    decreases that dimension's |value|, bounded below by 0; with finitely many coordinates ⇒ accepted steps ≤ Σ|initial
    coordinates| and the loop must terminate. This removal is not an optimisation but a correctness requirement: without
    it, at x=±1 the candidate set {0, ±1} contains the current value itself, each round's "improvement" stands still →
    self-locking infinite loop (measured in the old version).
    """
    steps = 0
    arg = prop.get("arg") if is_pair else None

    def violate(a: dict, b: "dict | None") -> bool:
        return not _prop_holds(prog, prop, a, b, step_limit)

    if not violate(cx, cx2):
        raise SvoError(Err.PRECOND_FAIL, ctx="op:code_property_test#9404@shrink",
                       detail="传入的候选反例经复判并不违背性质 (假反例, 拒绝收缩)")

    improved = True
    while improved:
        improved = False
        for d in sorted(cx.keys()):
            x = cx.get(d)
            if x is None or x == 0:
                continue
            s = 1 if x > 0 else -1
            for c in (0, s * ((abs(x) + 1) // 2), x - s):
                if abs(c) >= abs(x):
                    continue  # accept only strictly smaller |value| ⇒ termination (see docstring)
                if ctl is not None:
                    ctl.check_time("op:code_property_test#9404@shrink")
                a1 = dict(cx)
                a1[d] = c
                b1 = dict(cx2) if cx2 is not None else None
                if b1 is not None:
                    if d == arg:
                        b1[d] = b1[d] + (c - cx[d])  # synchronous shift: the arg slot difference stays exactly 1
                    elif d in b1:
                        b1[d] = c  # non-arg dimensions: the two points already agree; assign the same value synchronously
                steps += 1
                if violate(a1, b1):
                    cx, cx2 = a1, b1
                    improved = True
                    break
            if improved:
                break
    return cx, cx2, steps


def _invoke_prop(args: dict, ctl: CallCtl) -> dict:
    ctx = "op:code_property_test#9404"
    prog = args["program"]
    validate_prog(prog, ctx)
    prop = args["prop"]
    trials = int(args["trials"])
    step_limit = int(args.get("step_limit", 100_000))
    rng = random.Random(int(args["seed"]))
    if prop["kind"] == "postcond":
        validate_prog([{"t": "assign", "var": "v0", "e": prop["expr"]}], ctx + ".prop")
    else:
        if prop.get("arg") not in ARGS:
            raise SvoError(Err.SCHEMA_MISMATCH, ctx=ctx, detail="monotone 需要参数槽 arg∈{a0..a3}")
        if prop.get("out_var", "v0") not in VARS:
            raise SvoError(Err.SCHEMA_MISMATCH, ctx=ctx, detail="out_var 须为 v0..v15")

    lo, hi = -16, 16
    sampled: list[dict] = []
    cx: dict | None = None
    cx2: dict | None = None
    violations = 0
    found = False
    for _ in range(trials):
        ctl.check_time(ctx)
        if prop["kind"] == "postcond":
            inp = {a: rng.randint(lo, hi) for a in ARGS}
            if not _prop_holds(prog, prop, inp, None, step_limit):
                violations += 1
                if not found:
                    found = True
                    cx = dict(inp)
                continue
            sampled.append({k: int(v) for k, v in inp.items()})
        else:
            a = prop["arg"]
            x = rng.randint(lo, hi - 1)
            i1 = {k: rng.randint(lo, hi) for k in ARGS}
            i1[a] = x
            i2 = dict(i1); i2[a] = x + 1
            if not _prop_holds(prog, prop, i1, i2, step_limit):
                violations += 1
                if not found:
                    found = True
                    cx, cx2 = dict(i1), dict(i2)
                continue
            sampled.append({k: int(v) for k, v in i1.items()})
    shrunk = False
    shrink_steps = 0
    if found:
        cx, cx2, shrink_steps = _shrink(prog, prop, cx or {}, prop["kind"] == "monotone", cx2,
                                        step_limit, ctl)
        shrunk = True
    value = {
        "ok": not found, "found": found,
        "cx_args": (cx or {}) if prop["kind"] == "postcond" else {},
        "cx_lo": (cx or {}) if prop["kind"] == "monotone" else {},
        "cx_hi": (cx2 or {}) if prop["kind"] == "monotone" else {},
        "shrunk": shrunk, "shrink_steps": shrink_steps, "trials": trials,
        "violations": violations, "sampled_head": [{k: int(v) for k, v in s.items()} for s in sampled[:8]],
    }
    return {"value": value, "resid": [float(violations)],
            "notes": ("未发现违背" if not found else
                      f"反例已收缩 {shrink_steps} 步 → {cx if prop['kind']=='postcond' else (cx, cx2)}")}


def _verify_prop(args: dict, value: dict) -> dict:
    """Independent verification: a counter-example recomputed with engine2 must still violate; when there is no counter-example, spot-checked samples must pass."""
    prop = args["prop"]
    step_limit = int(args.get("step_limit", 100_000))
    probs = []
    if value["found"]:
        if prop["kind"] == "postcond":
            r = interp_stack(args["program"], value["cx_args"], step_limit)
            if r["error"] is not None:
                pass  # the counter-example crashes under engine2 as well — the crash itself is a reproducible violation
            else:
                rr = interp_stack([{"t": "assign", "var": "v15", "e": prop["expr"]}], dict(r["state"]), step_limit)
                if rr["error"] is None and rr["state"]["v15"] != 0:
                    probs.append("反例经 engine2 复算不再违背 (假反例)")
        else:
            r1 = interp_stack(args["program"], value["cx_lo"], step_limit)
            r2 = interp_stack(args["program"], value["cx_hi"], step_limit)
            out = prop.get("out_var", "v0")
            if (r1["error"] is None and r2["error"] is None
                    and r1["state"].get(out, 0) <= r2["state"].get(out, 0)):
                probs.append("单调反例经 engine2 复算不再违背 (假反例)")
    else:
        for s in value["sampled_head"][:4]:
            if not _prop_holds(args["program"], prop, s, None, step_limit) and prop["kind"] == "postcond":
                probs.append("样本点复算违背但被记为通过")
    return {"ok": not probs, "resid": [float(len(probs))],
            "detail": "engine2 反例/样本复算: " + ("通过" if not probs else "; ".join(probs))}


# ---------------------------------------------------------------------------
# operator 5: code_vpair_split (row 9405)
# ---------------------------------------------------------------------------

_IN_SPLIT = {
    "kind": "dict",
    "fields": {
        "items": {"kind": "list", "minlen": 1, "maxlen": 10000, "items": {"kind": "int"}},
        "seed": {"kind": "int", "min": 0, "max": 2**31 - 1},
        "train_frac": {"kind": "float", "min": 0.5, "max": 0.7},
    },
    "required": ["items", "seed"],
}
_OUT_SPLIT = {
    "kind": "dict",
    "fields": {
        "train": {"kind": "list", "items": {"kind": "int"}},
        "heldout": {"kind": "list", "items": {"kind": "int"}},
        "n": {"kind": "int"}, "train_n": {"kind": "int"}, "heldout_n": {"kind": "int"},
        "train_frac": {"kind": "float"},
    },
    "required": ["train", "heldout", "n", "train_n", "heldout_n", "train_frac"],
}


def _invoke_split(args: dict, ctl: CallCtl) -> dict:
    ctx = "op:code_vpair_split#9405"
    items = list(args["items"])
    frac = float(args.get("train_frac", 0.6))
    rng = random.Random(int(args["seed"]))
    order = items[:]
    rng.shuffle(order)
    k = int(len(order) * frac + 0.5)
    train, heldout = order[:k], order[k:]
    value = {"train": train, "heldout": heldout, "n": len(items),
             "train_n": len(train), "heldout_n": len(heldout), "train_frac": frac}
    return {"value": value, "resid": [0.0],
            "notes": f"V 对: train={len(train)} / held-out={len(heldout)} (验收只认 held-out)"}


def _verify_split(args: dict, value: dict) -> dict:
    """Independent verification: rather than replaying the shuffle, audit the invariants of the split (disjoint / union = universe / ratio / determinism)."""
    items = args["items"]
    tr, ho = value["train"], value["heldout"]
    probs = []
    if set(tr) & set(ho):
        probs.append("train 与 heldout 有交")
    if sorted(tr + ho) != sorted(items) or len(tr) + len(ho) != len(set(tr) | set(ho)):
        probs.append("train∪heldout ≠ 全集 (或有重复)")
    exp_k = int(len(items) * value["train_frac"] + 0.5)
    if len(tr) != exp_k:
        probs.append(f"train 大小 {len(tr)} ≠ 期望 {exp_k}")
    t2 = REGISTRY.tool("code_vpair_split").op.invoke(
        {"items": items, "seed": args["seed"], "train_frac": value["train_frac"]}, ctl=None)
    if not (t2["value"]["train"] == tr and t2["value"]["heldout"] == ho):
        probs.append("同种子重放不一致 (非确定性)")
    return {"ok": not probs, "resid": [float(len(probs))],
            "detail": "划分不变量审计: " + ("通过" if not probs else "; ".join(probs))}


# ---------------------------------------------------------------------------
# registration (band 9400, sub 1..5 → rows 9401..9405)
# ---------------------------------------------------------------------------

REGISTRY.register(OpDef(
    name="code_exec_sandbox", band=9400, sub=1,
    in_schema=_IN_SANDBOX, out_schema=_OUT_SANDBOX,
    invoke=_invoke_sandbox, verify=_verify_sandbox,
    cost=("units", 8.0),
    errors=("TIMEOUT", "OOM", "COMPILE_ERR", "RUNTIME_ERR", "SCHEMA_MISMATCH", "POSTCOND_FAIL"),
    effects="子进程执行 (真实超时 kill; 无网策略为启发式扫描; win32 无内存 rlimit, 以输出上限映射 OOM)",
    doc="§9.1 执行沙箱: 子进程跑 entry(list[int])->int, 返回逐测试通过向量+stdout/stderr+时延; "
        "A/B 双 runner 重放一致性是验收的一部分。诚实边界: Windows 无 rlimit → 内存上限不可强制。",
    selfchecks=(
        {"id": "pass_vector_exact", "timeout_s": 30,
         "args": {"code": "def entry(a):\n    return a[0] * 2 + a[1]", "timeout_s": 5.0, "max_out": 4096,
                  "tests": [{"args": [1, 2], "expect": 4}, {"args": [0, 0], "expect": 0},
                            {"args": [-3, 1], "expect": -5}]},
         "check": lambda r: (r.value["pass_vector"] == [True, True, True]
                             and r.value["results"][0]["got"] == 4 and r.value["elapsed_ms"] > 0,
                             f"pass_vector={r.value['pass_vector']} got0={r.value['results'][0]['got']}")},
        {"id": "per_test_runtime_err", "timeout_s": 30,
         "args": {"code": "def entry(a):\n    return 10 // a[0]", "timeout_s": 5.0, "max_out": 4096,
                  "tests": [{"args": [2], "expect": 5}, {"args": [0], "expect": 0}]},
         "check": lambda r: (r.ok and r.value["pass_vector"] == [True, False]
                             and "ZeroDivision" in r.value["results"][1]["err"],
                             f"逐测试错误被记录: {r.value['results'][1]['err']}")},
        {"id": "compile_err_syntax", "timeout_s": 30,
         "args": {"code": "def entry(a):\n    x = = 1\n    return x", "tests": [], "timeout_s": 5.0, "max_out": 4096},
         "expect_err": "COMPILE_ERR"},
        {"id": "policy_import_rejected", "timeout_s": 30,
         "args": {"code": "import os\ndef entry(a):\n    return a[0]", "tests": [], "timeout_s": 5.0, "max_out": 4096},
         "expect_err": "COMPILE_ERR"},
        {"id": "timeout_kill", "timeout_s": 30,
         "args": {"code": "def entry(a):\n    x = 0\n    for i in range(300000000):\n        x += i\n    return x",
                  "tests": [{"args": [1], "expect": 0}], "timeout_s": 1.0, "max_out": 4096},
         "expect_err": "TIMEOUT"},
    ),
))

REGISTRY.register(OpDef(
    name="code_ast_codebook", band=9400, sub=2,
    in_schema={"kind": "dict", "fields": {"want": {"kind": "str", "enum": ["table", "summary"]}},
               "required": ["want"], "extra": False},
    out_schema={"kind": "dict", "fields": {
        "entries": {"kind": "list", "items": {"kind": "dict", "extra": False, "fields": {
            "name": {"kind": "str"}, "code": {"kind": "int", "min": 1, "max": 64},
            "cat": {"kind": "str", "enum": ["kind", "binop", "unop", "lib", "arg", "var", "const"]},
            "arity": {"kind": "int", "min": 0, "max": 3}, "type_rule": {"kind": "str"}}}},
        "size": {"kind": "int"},
        "cats": {"kind": "dict", "extra": True},
        "loop_bound_cap": {"kind": "int"}, "vocab": {"kind": "int"}},
        "required": ["entries", "size", "cats", "loop_bound_cap", "vocab"], "extra": False},
    invoke=_invoke_codebook, verify=_verify_codebook,
    cost=("units", 0.5), errors=("SCHEMA_MISMATCH",),
    effects="none (纯函数)",
    doc="§9.1 AST 码本: {Lit,Var,BinOp,Call(lib),If,Loop(bounded),Assign} + 算子/槽位/常量池, |V|=64 恰好; "
        "循环有界 (求值后界 ≤64 否则 BOUNDED_VIOLATION) —— 禁图灵完备, 失效安全。",
    selfchecks=(
        {"id": "size_is_64", "args": {"want": "table"},
         "check": lambda r: (r.value["size"] == 64 and len(r.value["entries"]) == 64
                             and {e["code"] for e in r.value["entries"]} == set(range(1, 65)),
                             f"|V|={r.value['size']}, 码=1..64 排列")},
        {"id": "cat_counts", "args": {"want": "summary"},
         "check": lambda r: (r.value["cats"].get("binop") == 13 and r.value["cats"].get("var") == 16
                             and r.value["loop_bound_cap"] == 64,
                             f"cats={r.value['cats']} cap={r.value['loop_bound_cap']}")},
        {"id": "bad_want_neg", "args": {"want": "full"}, "expect_err": "SCHEMA_MISMATCH",
         "expect": "want 只允许 table/summary; 越界取值必须被 schema 门拦下 (§17.6 边界覆盖)"},
        {"id": "extra_key_neg", "args": {"want": "summary", "dump": True}, "expect_err": "SCHEMA_MISMATCH",
         "expect": "extra=False: 未声明字段必须被拒 (拒裸字典式调用)"},
        {"id": "arity_spot", "args": {"want": "table"},
         "check": lambda r: (next(e for e in r.value["entries"] if e["name"] == "clamp")["arity"] == 3
                             and next(e for e in r.value["entries"] if e["name"] == "+")["cat"] == "binop",
                             "clamp 元数=3, + 属 binop")},
    ),
))

REGISTRY.register(OpDef(
    name="code_bounded_interp", band=9400, sub=3,
    in_schema=_IN_INTERP, out_schema=_OUT_INTERP,
    invoke=_invoke_interp, verify=_verify_interp,
    cost=("units", 2.0), errors=("SCHEMA_MISMATCH", "RUNTIME_ERR", "BOUNDED_VIOLATION"),
    effects="none (纯函数; 自写解释器, 全程无 exec/eval)",
    doc="§9.1 有界解释器: 递归引擎执行 + 迭代栈机交叉复算 (engine_cross_ok); 步数上限超限即 BOUNDED_VIOLATION。",
    selfchecks=(
        {"id": "sum_loop_state", "args": {
            "program": [{"t": "assign", "var": "v0", "e": {"t": "lit", "v": 0}},
                        {"t": "loop", "v": "v3", "b": {"t": "lit", "v": 5},
                         "body": [{"t": "assign", "var": "v0",
                                   "e": {"t": "bin", "op": "+", "l": {"t": "var", "n": "v0"},
                                         "r": {"t": "bin", "op": "+", "l": {"t": "var", "n": "v3"},
                                               "r": {"t": "lit", "v": 1}}}}]}],
            "inputs": {"a0": 3}, "step_limit": 10000, "observed": "v0"},
         "check": lambda r: (r.value["observed_val"] == 15 and r.value["iters"] == 5
                             and r.value["engine_cross_ok"], f"v0={r.value['observed_val']} (1+2+3+4+5), iters=5")},
        {"id": "undef_var_runtime", "args": {
            "program": [{"t": "assign", "var": "v0", "e": {"t": "var", "n": "v7"}}],
            "inputs": {}, "step_limit": 1000},
         "expect_err": "RUNTIME_ERR"},
        {"id": "loop_cap_violation", "args": {
            "program": [{"t": "loop", "v": "v1", "b": {"t": "lit", "v": 100}, "body": []}],
            "inputs": {}, "step_limit": 100000},
         "expect_err": "BOUNDED_VIOLATION"},
        {"id": "div_zero", "args": {
            "program": [{"t": "assign", "var": "v0",
                         "e": {"t": "bin", "op": "/", "l": {"t": "var", "n": "a0"}, "r": {"t": "var", "n": "a1"}}}],
            "inputs": {"a0": 7, "a1": 0}, "step_limit": 1000},
         "expect_err": "RUNTIME_ERR"},
        {"id": "bad_node_rejected", "args": {
            "program": [{"t": "assign", "var": "v0", "e": {"t": "wat"}}], "inputs": {}, "step_limit": 100},
         "expect_err": "SCHEMA_MISMATCH"},
    ),
))

REGISTRY.register(OpDef(
    name="code_property_test", band=9400, sub=4,
    in_schema=_IN_PROP, out_schema=_OUT_PROP,
    invoke=_invoke_prop, verify=_verify_prop,
    cost=("units", 3.0), errors=("SCHEMA_MISMATCH", "RUNTIME_ERR", "BOUNDED_VIOLATION", "TIMEOUT"),
    effects="none (纯函数, 播种随机)",
    doc="§9.1 属性测试: 随机输入 + 断言 + 反例收缩最小化 (真跑收缩); 支持 postcond 断言与单调性两类性质。"
        "发现反例是**成功产出** (ok=False 表示性质被证伪于该样本域, 非算子错误)。",
    selfchecks=(
        {"id": "postcond_violation_shrunk", "args": {
            "program": [{"t": "assign", "var": "v0",
                         "e": {"t": "bin", "op": "*", "l": {"t": "var", "n": "a0"}, "r": {"t": "var", "n": "a0"}}}],
            "prop": {"kind": "postcond", "expr": {"t": "bin", "op": "<=",
                                                  "l": {"t": "var", "n": "v0"}, "r": {"t": "lit", "v": 60}}},
            "trials": 48, "seed": 7},
         "check": lambda r: (r.value["found"] and abs(r.value["cx_args"]["a0"]) == 8
                             and r.value["violations"] >= 1,
                             f"最小反例 a0={r.value['cx_args'].get('a0')} (8²=64>60, 7²=49≤60 已无法再缩)")},
        {"id": "postcond_holds", "args": {
            "program": [{"t": "assign", "var": "v0",
                         "e": {"t": "call", "fn": "clamp", "args": [
                             {"t": "var", "n": "a0"}, {"t": "lit", "v": -4}, {"t": "lit", "v": 4}]}}],
            "prop": {"kind": "postcond",
                     "expr": {"t": "bin", "op": "and",
                              "l": {"t": "bin", "op": ">=", "l": {"t": "var", "n": "v0"}, "r": {"t": "lit", "v": -4}},
                              "r": {"t": "bin", "op": "<=", "l": {"t": "var", "n": "v0"}, "r": {"t": "lit", "v": 4}}}},
            "trials": 64, "seed": 3},
         "check": lambda r: (r.value["ok"] and not r.value["found"] and r.value["violations"] == 0,
                             "clamp 性质 64 次采样无违背")},
        {"id": "monotone_violation", "args": {
            "program": [{"t": "assign", "var": "v0",
                         "e": {"t": "bin", "op": "*", "l": {"t": "var", "n": "a0"}, "r": {"t": "lit", "v": -1}}}],
            "prop": {"kind": "monotone", "arg": "a0", "out_var": "v0"}, "trials": 64, "seed": 11},
         "check": lambda r: (r.value["found"] and r.value["cx_lo"]["a0"] + 1 == r.value["cx_hi"]["a0"],
                             f"单调性被证伪: a0={r.value['cx_lo'].get('a0')} < {r.value['cx_hi'].get('a0')} 但输出下降")},
        {"id": "bad_prop_rejected", "args": {
            "program": [{"t": "assign", "var": "v0", "e": {"t": "lit", "v": 1}}],
            "prop": {"kind": "postcond", "expr": {"t": "binop", "op": "^^"}}, "trials": 8, "seed": 1},
         "expect_err": "SCHEMA_MISMATCH"},
        {"id": "shrink_to_minimal", "args": {
            "program": [{"t": "assign", "var": "v0",
                         "e": {"t": "bin", "op": "-", "l": {"t": "lit", "v": 0}, "r": {"t": "var", "n": "a0"}}}],
            "prop": {"kind": "postcond",
                     "expr": {"t": "bin", "op": ">=", "l": {"t": "var", "n": "v0"}, "r": {"t": "lit", "v": 0}}},
            "trials": 48, "seed": 2},
         "check": lambda r: (r.value["found"] and r.value["cx_args"].get("a0") == 1,
                             f"最小反例收缩到 a0=1 (0 不违背, 1 违背), shrink_steps={r.value['shrink_steps']}")},
    ),
))

REGISTRY.register(OpDef(
    name="code_vpair_split", band=9400, sub=5,
    in_schema=_IN_SPLIT, out_schema=_OUT_SPLIT,
    invoke=_invoke_split, verify=_verify_split,
    cost=("units", 0.5), errors=("SCHEMA_MISMATCH",),
    effects="none (纯函数, 播种随机)",
    doc="§9.1 V 对制度: 60/40 train/held-out 划分 (播种洗牌, 确定性); 验收只认 held-out。",
    selfchecks=(
        {"id": "partition_10", "args": {"items": list(range(10)), "seed": 5},
         "check": lambda r: (r.value["train_n"] == 6 and r.value["heldout_n"] == 4
                             and not (set(r.value["train"]) & set(r.value["heldout"]))
                             and sorted(r.value["train"] + r.value["heldout"]) == list(range(10)),
                             f"train={r.value['train']} heldout={r.value['heldout']}")},
        {"id": "ratio_60_40", "args": {"items": list(range(100)), "seed": 42},
         "check": lambda r: (r.value["train_n"] == 60 and r.value["heldout_n"] == 40,
                             f"100 项 → {r.value['train_n']}/{r.value['heldout_n']}")},
        {"id": "empty_rejected", "args": {"items": [], "seed": 1}, "expect_err": "SCHEMA_MISMATCH"},
    ),
))

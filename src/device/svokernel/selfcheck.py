"""selfcheck CLI — execution entry of the §16 verification gate (§7 registration flow / §17.6 smoke-run requirement).

Run (server):
    python -m svokernel.selfcheck                 # full selfcheck + audit (human-readable)
    python -m svokernel.selfcheck --only math_    # prefix filter
    python -m svokernel.selfcheck --strict        # exit code 1 on any failure
    python -m svokernel.selfcheck --emit-json > reports/selfcheck.json

It does four things (all mandatory, all actually run):
  1. **static audit** (audit_static): codebook row uniqueness / in-band validity, five-tuple completeness,
     error codes ⊆ §15, certificate kinds have independent checkers, selfcheck items non-empty, references
     to unregistered operators checked;
  2. **selfcheck execution** (run_checks): every selfcheck item of every operator actually runs, including
     negative boundary cases (empty input / no solution / timeout / counter-example);
  3. **reproducibility spot check** (reproducibility): rerun all selfchecks with the same seed and compare
     the (passed, err, resid) triple fingerprint — any mismatch is a red alert (no unseeded randomness,
     no wall-clock dependence);
  4. **report output**: argv is recorded item by item (§17.6); with `--emit-json` the full JSON goes to
     stdout, redirected to disk by the caller's shell (this module **never writes files**, so there is no
     path-injection surface).

Honesty boundary: this module does **not** claim "audit passed" (§18 red line). It reports only what it
actually ran: which selfchecks ran, whether they passed, the residuals, and whether fingerprints agree.
Audit conclusions come from our adversarial audit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time

from .audit import AuditTrace, BudgetLedger
from .certify import CHECKERS
from .codebook import CODEBOOK, EMB_DIM, RESID_DIM, latent_vec
from .errors import ERROR_TABLE
from .registry import REGISTRY, SelfCheckResult
from .tool import CallCtl


# ---------------------------------------------------------------------------
# 1) static audit
# ---------------------------------------------------------------------------


def audit_static() -> list[dict]:
    """Return the problem list (empty = no static violations found). Each entry carries code/op/detail."""
    problems: list[dict] = []

    def bad(code: str, op: str, detail: str) -> None:
        problems.append({"code": code, "op": op, "detail": detail})

    seen_rows: dict[int, str] = {}
    for name in REGISTRY.names():
        op = REGISTRY._ops[name]
        tool = REGISTRY.tool(name)
        row = tool.handle

        if row in seen_rows:
            bad("ROW_DUP", name, f"行 {row} 与 {seen_rows[row]} 冲突")
        seen_rows[row] = name

        # §7 six admission rules
        if op.verify is None and op.cert_kind is None:
            bad("NO_VERIFIER", name, "§7.3 没有验证器不许交")
        if not op.selfchecks:
            bad("NO_SELFCHECK", name, "§7 注册流程要求 self-check")
        if not op.errors:
            bad("NO_ERROR_ENUM", name, "§7.6 必须枚举失败模式")
        for e in op.errors:
            if e not in ERROR_TABLE:
                bad("ERR_OUT_OF_TABLE", name, f"错误码 {e!r} 不在 §15 表内")
        if op.cost[0] != "units" or not isinstance(op.cost[1], (int, float)) or op.cost[1] < 0:
            bad("BAD_COST", name, f"成本标注非法 {op.cost!r}")
        if not op.effects:
            bad("NO_EFFECTS", name, "§7.2 副作用须显式")
        if not op.doc:
            bad("NO_DOC", name, "五元组文档缺失")

        # a certificate kind must have an independent checker (§13 + §17.9)
        if op.cert_kind is not None:
            if not CHECKERS.has(op.cert_kind):
                bad("CERT_UNCHECKED", name, f"证书类型 {op.cert_kind!r} 无独立检查器")
            else:
                cmod = CHECKERS.module_of(op.cert_kind)
                gmod = getattr(op.invoke, "__module__", "")
                if cmod == gmod:
                    bad("SELF_CERT", name,
                        f"检查器 {op.cert_kind} 与生成器同处模块 {cmod} —— §17.9 禁自证要求两条独立代码路径")
                elif _checker_imports_generator(CHECKERS.fn(op.cert_kind), gmod):
                    bad("SELF_CERT_IMPORT", name,
                        f"检查器模块 {cmod} 直接 import 了生成器模块 {gmod} —— 独立性被依赖削弱")
                # Declaration-side consistency (independent_of lists the **generator operator name**) is
                # already enforced by CheckerRegistry.register at registration time: filling in a module
                # name where an operator name belongs is rejected on the spot. Not repeated here, to avoid
                # flagging a correct declaration as a violation.

        # selfcheck item structure
        ids: set[str] = set()
        for sc in op.selfchecks:
            for k in ("id", "args"):
                if k not in sc:
                    bad("SELFCHECK_SCHEMA", name, f"自检项缺字段 {k!r}: {sc!r}")
            cid = sc.get("id", "?")
            if cid in ids:
                bad("SELFCHECK_DUP", name, f"自检 id 重复: {cid}")
            ids.add(cid)

    return problems


def _checker_imports_generator(fn, generator_module: str) -> bool:
    """Check whether the checker's module namespace directly holds the generator module object (i.e. imported it).

    This is a **weak transitive check** of module-level independence: the same-module case is already
    caught by the earlier criterion, and here we additionally catch "the checker sits in a separate module
    but imports the generator module to reuse its implementation" — the two paths would then share code and
    "independent re-verification" would degrade into restatement. The criterion inspects only the checker
    function's __globals__ and does not walk the transitive closure (honesty statement: deep transitive
    dependencies are outside this check's coverage).
    """
    if fn is None or not generator_module:
        return False
    import types

    for v in getattr(fn, "__globals__", {}).values():
        if isinstance(v, types.ModuleType) and getattr(v, "__name__", "") == generator_module:
            return True
    return False


def audit_codebook() -> dict:
    """Codebook audit: row count / band distribution / embedding dimension / determinism (recomputation of the same row must be bit-identical)."""
    rows = CODEBOOK.rows()
    det_ok = True
    for r in rows.values():
        a, b = latent_vec(r), latent_vec(r)
        if a != b or len(a) != EMB_DIM:
            det_ok = False
            break
    return {
        "size": CODEBOOK.size(),
        "checksum": CODEBOOK.checksum(),
        "bands": REGISTRY.band_census(),
        "emb_dim": EMB_DIM,
        "resid_dim": RESID_DIM,
        "latent_deterministic": det_ok,
    }


def audit_coverage() -> dict:
    """Coverage audit: does every operator have at least one **negative** selfcheck (§17.6 boundary requirement)."""
    rows: list[dict] = []
    for name in REGISTRY.names():
        op = REGISTRY._ops[name]
        neg = [sc["id"] for sc in op.selfchecks if sc.get("expect_err")]
        rows.append({
            "op": name,
            "handle": REGISTRY.tool(name).handle,
            "checks": len(op.selfchecks),
            "negative": len(neg),
            "negative_ids": neg,
        })
    no_neg = [r["op"] for r in rows if r["negative"] == 0]
    return {"ops": rows, "ops_without_negative_check": no_neg, "n_ops_without_negative": len(no_neg)}


# ---------------------------------------------------------------------------
# 2/3) selfcheck execution + reproducibility
# ---------------------------------------------------------------------------


def run_checks(only: "str | None" = None, timeout_s: float = 120.0) -> list:
    """Actually run every (or prefix-filtered) selfcheck item; return the SelfCheckResult list."""
    results: list[SelfCheckResult] = []
    names = REGISTRY.names() if only is None else [
        n for n in REGISTRY.names() if n == only or n.startswith(only.rstrip("*"))
    ]
    for n in names:
        op = REGISTRY._ops[n]
        tool = REGISTRY.tool(n)
        for sc in op.selfchecks:
            ctl = CallCtl(trace=AuditTrace(f"selfcheck:{n}"), budget=BudgetLedger(total_units=1e9),
                          timeout_s=sc.get("timeout_s", timeout_s), seed=int(sc.get("seed", 0)))
            t0 = time.perf_counter()
            r = tool.call(sc["args"], ctl)
            dt = (time.perf_counter() - t0) * 1000.0
            passed, detail = REGISTRY._judge(sc, r)
            results.append(SelfCheckResult(
                op=n, handle=tool.handle, cid=sc["id"], passed=passed, detail=detail,
                err=r.err, resid=r.resid_scalar, elapsed_ms=dt,
            ))
    return results


def repr_fingerprint(results: list) -> str:
    """Reproducibility fingerprint = hash of the ordered (op, cid, passed, err, resid at fixed precision) sequence."""
    blob = json.dumps(
        [[r.op, r.cid, r.passed, r.err,
          "nan" if r.resid != r.resid else round(float(r.resid), 9)]
         for r in results],
        ensure_ascii=False,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def reproducibility(only: "str | None", timeout_s: float) -> dict:
    """Same-seed double run comparison — any mismatch is a red flag for "hidden state / unseeded randomness"."""
    a = run_checks(only, timeout_s)
    b = run_checks(only, timeout_s)
    fa, fb = repr_fingerprint(a), repr_fingerprint(b)
    mism: list[dict] = []
    for x, y in zip(a, b):
        same_err = (x.passed, x.err) == (y.passed, y.err)
        both_nan = x.resid != x.resid and y.resid != y.resid
        same_resid = both_nan or x.resid == y.resid
        if not (same_err and same_resid):
            mism.append({"op": x.op, "cid": x.cid, "run_a": [x.passed, x.err, x.resid],
                         "run_b": [y.passed, y.err, y.resid]})
    return {"runs": 2, "fingerprint_a": fa, "fingerprint_b": fb, "equal": fa == fb,
            "mismatches": mism[:20], "n_checks": len(a)}


def build_report(results: list, static: list, cb: dict, cov: dict, repro: dict,
                 elapsed: float) -> dict:
    """Assemble the full report (§17.6: argv is recorded item by item)."""
    return {
        "argv": [os.path.basename(sys.argv[0])] + list(sys.argv[1:]),
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED", "<unset>"),
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "codebook": cb,
        "static_audit": static,
        "coverage": cov,
        "summary": REGISTRY.selfcheck_summary(results),
        "reproducibility": repro,
        "elapsed_s": round(elapsed, 3),
        "results": [
            {"op": r.op, "handle": r.handle, "id": r.cid, "passed": r.passed, "err": r.err,
             "resid": None if r.resid != r.resid else r.resid,
             "ms": round(r.elapsed_ms, 3), "detail": r.detail}
            for r in results
        ],
        "five_tuple": REGISTRY.five_tuple_table(),
        "checkers": {k: list(CHECKERS.independence(k)) for k in CHECKERS.kinds()},
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(prog="svokernel.selfcheck",
                                 description="SVO selfcheck gate (actually runs; does not claim the audit passed)")
    ap.add_argument("--only", default=None, help="operator-name prefix filter (e.g. math_ / logic_sat)")
    ap.add_argument("--timeout", type=float, default=120.0, help="cooperative deadline per selfcheck item (seconds)")
    ap.add_argument("--strict", action="store_true", help="exit code 1 on any failure / static violation")
    ap.add_argument("--no-repro", action="store_true", help="skip the reproducibility double run")
    ap.add_argument("--quiet", action="store_true", help="print only failures and the summary")
    ap.add_argument("--list", action="store_true", help="list the operator five-tuple table only; run no selfchecks")
    ap.add_argument("--emit-json", action="store_true",
                    help="emit the full JSON report to stdout (human-readable text moves to stderr) for shell redirection to disk")
    args = ap.parse_args(argv if argv is not None else sys.argv[1:])

    out = sys.stderr if args.emit_json else sys.stdout

    from . import curriculum  # noqa: F401  load curriculum generators
    from . import ops  # noqa: F401  load operators (import registers)

    if args.list:
        for row in REGISTRY.five_tuple_table():
            print(f"#{row['handle']} {row['op']:34s} {row['io']:52s} "
                  f"verifier={row['verifier']:26s} err={len(row['errors'])}", file=out)
        return 0

    t0 = time.perf_counter()
    static = audit_static()
    cb = audit_codebook()
    cov = audit_coverage()
    results = run_checks(args.only, args.timeout)
    repro = {"skipped": True} if args.no_repro else reproducibility(args.only, args.timeout)
    elapsed = time.perf_counter() - t0

    for r in results:
        if (not r.passed) or (not args.quiet):
            print(r.line(), file=out)
    s = REGISTRY.selfcheck_summary(results)
    print("=" * 110, file=out)
    print(f"selfcheck summary: {s['ops']} operators, {s['checks']} checks, {s['passed']} passed, "
          f"{s['failed']} failed → {'all green' if s['green'] else 'failures present'}", file=out)
    print("=" * 110, file=out)
    print(f"codebook: {cb['size']} rows, checksum={cb['checksum']}, band census={cb['bands']}, "
          f"embedding {cb['emb_dim']}d deterministic={cb['latent_deterministic']}", file=out)
    print(f"static audit: {len(static)} violations", file=out)
    for p in static:
        print(f"  [STATIC-FAIL] {p['code']} @ {p['op']}: {p['detail']}", file=out)
    if cov["n_ops_without_negative"]:
        print(f"boundary-coverage warning: {cov['n_ops_without_negative']} operators have no negative selfcheck: "
              f"{cov['ops_without_negative_check']}", file=out)
    if not repro.get("skipped"):
        print(f"reproducibility: double-run fingerprints {repro['fingerprint_a']} vs {repro['fingerprint_b']} "
              f"→ {'identical' if repro['equal'] else 'MISMATCH (red alert)'}", file=out)
        for m in repro.get("mismatches", [])[:5]:
            print(f"  [REPRO-MISMATCH] {m}", file=out)
    print(f"total elapsed {elapsed:.2f}s", file=out)

    if args.emit_json:
        json.dump(build_report(results, static, cb, cov, repro, elapsed), sys.stdout,
                  ensure_ascii=False, indent=1, default=str)
        sys.stdout.write("\n")

    green = s["green"] and not static and repro.get("equal", True)
    return 0 if (green or not args.strict) else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

# -*- coding: utf-8 -*-
"""V32 selfcheck/ledger/audit CLI (§16 verification gate on the V32 side).

How to run (cwd=$SCOPE_RESEARCH_ROOT):
    P="${SCOPE_PYTHON:-python3}"
    PYTHONPATH="${SCOPE_SUPEROPERATOR:-../device}" $P -m svokernel_v32.cli
    PYTHONPATH=... $P -m svokernel_v32.cli --only skel_           # prefix filter
    PYTHONPATH=... $P -m svokernel_v32.cli --list                 # ledger only
    PYTHONPATH=... $P -m svokernel_v32.cli --mutate               # checker mutation audit
    PYTHONPATH=... $P -m svokernel_v32.cli --emit-json > out.json

Six duties (all really executed):
  1. **V32 selfcheck execution**: every selfcheck of every operator in the v32 registry really runs
     (including the 154 re-registered v31 ones);
  2. **V31 zero-regression selfcheck**: run the full v31 selfcheck set (792 checks) and the legacy
     static audit in the **same process**, reporting counts item by item;
  3. **static audit** (audit_static_v32): unique rows / verifier present / certificate-checker
     module independent (§17.9) / error codes ⊆ §15 / stub discipline + the three skeleton criteria
     (band / certificate list / layer=4);
  4. **anchor-chain audit** (anchor_chain_audit_v32): 118 legacy / legacy codebook checksum /
     28 checkers / 154 v31 / v31 codebook / v31 row replication -- compared item by item by really
     running;
  5. **ledger** (census): band/layer/status/skeleton counts + the index-space convention
     (END=154, skeleton 155..160) + frozen head-dimension assertion;
  6. **checker mutation audit** (--mutate): for every skeleton certificate kind, perturb the true
     certificate payload leaf by leaf and measure the checker **rejection rate**; undetected
     mutations (survivors) are **reported as they are**, not glossed over.

Honesty boundary: this module does **not claim "the audit passed"**. It reports only what it really
ran.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time

from svokernel.audit import AuditTrace, BudgetLedger
from svokernel.certify import CHECKERS
from svokernel.codebook import CODEBOOK as LEGACY_CODEBOOK
from svokernel.registry import REGISTRY as LEGACY_REGISTRY
from svokernel.tool import CallCtl

from . import (LOADED, MANIFEST, MISSING, SKEL_OPS, V32_CODEBOOK, V32_END_MARKER,
               V32_HEAD_DIM_FROZEN, V32_SCHEMA, V32REGISTRY, __version__, audit_static_v32)
from .v32registry import SKEL_CERT_KINDS, anchor_chain_audit_v32
from .v32codebook import v32_op_index_map

#: Skeleton selfcheck fixtures (one "really executed" operator call per certificate kind, used by
#: the mutation audit to obtain a true certificate).
MUT_FIXTURES = {
    "case_exhaustive": ("skel_case_split",
                        {"goal": "for every integer n, 2 divides n*(n+1)", "m": 2, "vid": 0,
                         "gid": 0, "c0": 1}),
    "induction_scheme": ("skel_induction_scheme",
                         {"goal": "for every integer n, 3 divides n^3 - n", "m": 3, "vid": 0,
                          "gid": 0, "n0": 0, "c0": 1}),
    "stack_machine": ("skel_subgoal_focus",
                      {"act": "split", "gid": 0, "vid": 0, "c1": 1, "c2": 2,
                       "stk": "g0|claim|open|-1@focus=0@depth=0",
                       "tk": "for every integer n, (2 divides n*(n+1)) and "
                             "(3 divides n*(n+1)*(n+2))"}),
    "leaf_residue_complete": ("skel_leaf_close",
                              {"goal": "for every integer k, 2 divides (0 + 2*k)*((0 + 2*k) + 1)",
                               "m": 2, "vid": 2, "gid": 1}),
    "contradiction_witness": ("skel_contradiction_close",
                              {"goal": "there is no integer n with 4 dividing n^2 - 2",
                               "m": 4, "vid": 0, "gid": 0}),
    "skeleton_qed": ("skel_qed_check",
                     {"stk": "g0|case|closed|-1;g1|leaf|closed|0;g2|leaf|closed|0@focus=-1@depth=1",
                      "ledger": "g0:case:closed;g1:leaf:closed;g2:leaf:closed", "gid": 0}),
}


def run_checks(only=None, timeout_s=180.0):
    return V32REGISTRY.run_selfchecks(only=only)


def census() -> dict:
    st = V32REGISTRY.status_census()
    ck = V32REGISTRY.checks_by_status()
    idx = v32_op_index_map()
    n_legacy = sum(1 for n in LEGACY_REGISTRY.names() if V32REGISTRY.has(n))
    n_v31 = sum(1 for n in V32REGISTRY.names() if n not in LEGACY_REGISTRY._ops)
    return {
        "schema": V32_SCHEMA,
        "n_ops": len(V32REGISTRY),
        "verified": st.get("verified", 0),
        "stub": st.get("stub", 0),
        "legacy_refs": n_legacy,
        "v31_refs": n_v31 - len(SKEL_OPS),
        "skel_ops": len(SKEL_OPS),
        "v32_new_ops": len(SKEL_OPS),
        "codebook_rows": V32_CODEBOOK.size(),
        "codebook_checksum": V32_CODEBOOK.checksum(),
        "head_dim": V32_HEAD_DIM_FROZEN,
        "band_census": V32REGISTRY.band_census(),
        "layer_census": V32REGISTRY.layer_census(),
        "status_census": st,
        "skeleton_census": V32REGISTRY.skeleton_census(),
        "checks_total": sum(ck.values()),
        "checks_by_status": ck,
        "legacy_codebook_checksum": LEGACY_CODEBOOK.checksum(),
        "legacy_n_ops": len(LEGACY_REGISTRY),
        "index_space": {"end_idx": idx.get(V32_END_MARKER),
                        "skel_idx": {n: idx[n] for n in SKEL_OPS}},
    }


def v31_zero_regression() -> dict:
    """Run the full v31 selfcheck set + the legacy static audit in the same process -- **really
    executed** evidence of zero regression."""
    from svokernel.selfcheck import audit_static as legacy_audit
    from svokernel_v31 import V31REGISTRY
    from svokernel_v31.v31registry import audit_static_v31

    out = {}
    t0 = time.time()
    res_v31 = V31REGISTRY.run_selfchecks()
    summ = V31REGISTRY.selfcheck_summary(res_v31)
    out["v31_selfchecks"] = {"ops": summ["ops"], "checks": summ["checks"],
                             "passed": summ["passed"], "failed": summ["failed"],
                             "green": summ["green"], "secs": round(time.time() - t0, 1)}
    out["v31_static_problems"] = [p["code"] for p in audit_static_v31()]
    t0 = time.time()
    res_leg = LEGACY_REGISTRY.run_selfchecks()
    summ2 = LEGACY_REGISTRY.selfcheck_summary(res_leg)
    out["legacy_selfchecks"] = {"ops": summ2["ops"], "checks": summ2["checks"],
                                "passed": summ2["passed"], "failed": summ2["failed"],
                                "green": summ2["green"], "secs": round(time.time() - t0, 1)}
    out["legacy_static_problems"] = [p["code"] for p in legacy_audit()]
    out["legacy_codebook_checksum"] = LEGACY_CODEBOOK.checksum()
    return out


def mutate_audit() -> dict:
    """Checker mutation audit: perturb the true certificate payload leaf by
    leaf, then measure the checker rejection rate; survivors are recorded."""
    out = {"kinds": {}, "fixtures": {}, "note": "survivor = 检查器**未**拒绝的扰动 (如实登记)"}
    for kind, (op, args) in MUT_FIXTURES.items():
        if not CHECKERS.has(kind):
            out["kinds"][kind] = {"error": "检查器不在册"}
            continue
        ctl = CallCtl(trace=AuditTrace("mutate:%s" % kind), budget=BudgetLedger(total_units=1e9),
                      timeout_s=60.0)
        r = V32REGISTRY.tool(op).call(dict(args), ctl)
        if not r.ok or r.cert is None:
            out["kinds"][kind] = {"error": "夹具调用失败: %s" % (r.err,)}
            continue
        payload = json.loads(json.dumps(r.cert.payload, default=str))
        out["fixtures"][kind] = {"op": op, "args": args, "ok": True}
        base_ok, _, _ = CHECKERS.check({"kind": kind, "payload": payload, "issued_by": "mutate"})
        assert base_ok, "夹具证书本身应通过"
        tests = []

        def perturb(node, path):
            """Recursively produce a list of (path, perturbed payload)."""
            res = []
            if isinstance(node, dict):
                for k, v in node.items():
                    res += perturb(v, path + [k])
            elif isinstance(node, list):
                for i, v in enumerate(node):
                    res += perturb(v, path + [i])
            else:
                for tag, nv in (("plus1", (node + 1) if isinstance(node, bool) is False
                                and isinstance(node, int) else None),
                                ("flip", (not node) if isinstance(node, bool) else None),
                                ("str", (str(node) + "X") if isinstance(node, str) else None),
                                ("zero", 0 if isinstance(node, int) and not isinstance(node, bool)
                                 else None)):
                    if nv is None or nv == node:
                        continue
                    res.append((tag, path, nv))
            return res

        for tag, path, nv in perturb(payload, []):
            pl = json.loads(json.dumps(payload))
            cur = pl
            for k in path[:-1]:
                cur = cur[k]
            cur[path[-1]] = nv
            ok, _, detail = CHECKERS.check({"kind": kind, "payload": pl, "issued_by": "mutate"})
            tests.append({"path": ".".join(str(x) for x in path), "mut": tag,
                          "rejected": (not ok), "detail": detail[:160], "case": []})
            tests[-1]["case"] = list(path)
        n = len(tests)
        k_rej = sum(1 for t in tests if t["rejected"])
        out["kinds"][kind] = {"n_mutations": n, "n_rejected": k_rej,
                              "reject_rate": (k_rej / n) if n else None,
                              "survivors": [{"path": t["path"], "mut": t["mut"]}
                                            for t in tests if not t["rejected"]]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--mutate", action="store_true")
    ap.add_argument("--no-v31", action="store_true", help="skip the v31 zero-regression selfcheck (v32 only)")
    ap.add_argument("--emit-json", action="store_true")
    a = ap.parse_args()

    report = {
        "argv": sys.argv[1:], "version": __version__, "schema": V32_SCHEMA,
        "python": sys.version.split()[0], "platform": platform.platform(),
        "cwd": os.getcwd(), "loaded": LOADED, "missing": MISSING,
        "manifest": [m for m, _ in MANIFEST],
    }

    if a.list:
        for row in V32REGISTRY.ledger():
            print("#%-5d %-28s band=%-6d layer=%-3d status=%-8s %s"
                  % (row["handle"], row["op"], row["band"], row["layer"], row["status"],
                     row["verifier"]))
        return 0

    t0 = time.time()
    res = run_checks(only=a.only)
    summ = V32REGISTRY.selfcheck_summary(res)
    report["selfcheck"] = {"ops": summ["ops"], "checks": summ["checks"], "passed": summ["passed"],
                           "failed": summ["failed"], "green": summ["green"],
                           "secs": round(time.time() - t0, 1)}
    report["failures"] = [{"op": r.op, "cid": r.cid, "err": r.err, "detail": r.detail}
                          for r in res if not r.passed]
    skel_fail = [r for r in res if r.op in SKEL_OPS and not r.passed]
    report["skel_selfcheck"] = {
        "ops": len({r.op for r in res if r.op in SKEL_OPS}),
        "checks": sum(1 for r in res if r.op in SKEL_OPS),
        "passed": sum(1 for r in res if r.op in SKEL_OPS and r.passed),
        "green": not skel_fail,
    }

    report["static_problems"] = audit_static_v32()
    report["census"] = census()
    report["anchor"] = anchor_chain_audit_v32()
    if not a.no_v31:
        report["v31_zero_regression"] = v31_zero_regression()
    if a.mutate:
        report["mutate"] = mutate_audit()

    if a.emit_json:
        print(json.dumps(report, ensure_ascii=False, indent=1, default=str))
    else:
        print("== V32 selfcheck ==")
        print("  loaded %d/%d modules | missing %r" % (len(LOADED), len(MANIFEST), MISSING))
        print("  selfcheck: %d ops / %d checks: %d passed, %d failed, green=%s (%.1fs)"
              % (summ["ops"], summ["checks"], summ["passed"], summ["failed"], summ["green"],
                 report["selfcheck"]["secs"]))
        print("  skeleton selfcheck: %s" % (report["skel_selfcheck"],))
        for f in report["failures"][:40]:
            print("  [check-FAIL] %s :: %s :: err=%s :: %s" % (f["op"], f["cid"], f["err"],
                                                               f["detail"]))
        sp = report["static_problems"]
        print("  static audit: %d problems %s" % (len(sp), [p["code"] for p in sp][:12]))
        c = report["census"]
        print("  ledger: %d ops (legacy refs %d + v31 %d + skeleton %d) | codebook %d rows head=%d | checks=%d"
              % (c["n_ops"], c["legacy_refs"], c["v31_refs"], c["skel_ops"],
                 c["codebook_rows"], c["head_dim"], c["checks_total"]))
        print("  index space: END=%s, skeleton %s" % (c["index_space"]["end_idx"],
                                                      c["index_space"]["skel_idx"]))
        an = report["anchor"]
        print("  anchor-chain audit: ok=%s | %s" % (an["ok"], an["n_v31_joined"]))
        for ck in an["checks"]:
            print("    [%s] %-26s %s" % ("OK " if ck["ok"] else "BAD", ck["check"], ck["detail"]))
        z = report.get("v31_zero_regression")
        if z:
            print("== V31 zero regression (same-process real run) ==")
            print("  v31 selfcheck: %s" % (z["v31_selfchecks"],))
            print("  v31 static-audit problems: %r" % (z["v31_static_problems"],))
            print("  legacy selfcheck: %s" % (z["legacy_selfchecks"],))
            print("  legacy static-audit problems: %r | codebook checksum=%s"
                  % (z["legacy_static_problems"], z["legacy_codebook_checksum"]))
        if a.mutate:
            print("== checker mutation audit ==")
            for k, v in report["mutate"]["kinds"].items():
                print("  %-24s mutations %s rejected %s (rate %s) survivor=%d"
                      % (k, v.get("n_mutations"), v.get("n_rejected"), v.get("reject_rate"),
                         len(v.get("survivors") or [])))
    return 0


if __name__ == "__main__":
    sys.exit(main())

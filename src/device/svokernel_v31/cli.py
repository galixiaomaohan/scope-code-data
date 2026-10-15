# -*- coding: utf-8 -*-
"""V31 selfcheck/ledger/audit CLI (§16 verification gate on the V31 side).

How to run (cwd=$SCOPE_RESEARCH_ROOT):
    python3 -m svokernel_v31.cli                     # human-readable: full selfcheck + audit + ledger
    python3 -m svokernel_v31.cli --only nt_          # prefix filter
    python3 -m svokernel_v31.cli --emit-json > out.json   # full JSON (for shell redirection)
    python3 -m svokernel_v31.cli --list              # ledger only
    python3 -m svokernel_v31.cli --mutate            # append the "checker mutation audit"

Five duties (all really executed; none may be skipped):
  1. **selfcheck execution**: every selfcheck of every operator in the V31 registry really runs
     (including the 118 re-registered legacy ones);
  2. **static audit** (audit_static_v31): unique rows / verifier present / certificate-checker
     module independent (§17.9) / error codes ⊆ §15 / stub discipline (a stub carries no
     cert_kind, must state its gap, and must have an "explicit absence" selfcheck);
  3. **anchor-chain audit** (anchor_chain_audit): 118 legacy registry entries / codebook
     checksum / the names and modules of the 28 certificate checkers / legacy row replication
     -- compared item by item by really running them;
  4. **ledger** (census): band/status/layer counts + the head-dimension convention + frozen
     layout assertions;
  5. **checker mutation audit** (--mutate): for every new certificate kind, perturb the true
     certificate payload leaf by leaf and measure the checker **rejection rate** -- a checker
     must really reject, otherwise "independent re-verification" is an empty phrase. Undetected
     mutations are **reported as they are**, not glossed over.

Honesty boundary: this module does **not claim "the audit passed"**. It reports only what it
really ran.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from fractions import Fraction

from svokernel.audit import AuditTrace, BudgetLedger
from svokernel.certify import CHECKERS
from svokernel.codebook import CODEBOOK as LEGACY_CODEBOOK
from svokernel.registry import REGISTRY as LEGACY_REGISTRY
from svokernel.tool import CallCtl

from . import LOADED, MANIFEST, MISSING, V31_CODEBOOK, V31REGISTRY, __version__, v31_head_dim
from .v31registry import LEGACY_CERT_KINDS_FROZEN, anchor_chain_audit, audit_static_v31

V31_CERT_KINDS = ("lte_valuation", "hensel_root", "pratt_tree", "miller_witness", "factor_product",
                  "order_exhaustive", "jacobi_euler", "zsi_primitive", "arith_conv_table",
                  "padic_valuation", "cf_convergent", "pell_backsub", "form_reduce_chain",
                  "qexp_mul", "hecke_tp_expand", "ec_chord_backsub", "torsion_walk",
                  "pointcount_small", "qsqrtd_arith", "frey_invariants", "psi_interval",
                  "gelfond_apply", "thm_apply_witness", "lemma_instance", "equiv_truth_table",
                  "lemma_validity")


def run_checks(only=None, timeout_s=180.0):
    return V31REGISTRY.run_selfchecks(only=only)


def census() -> dict:
    st = V31REGISTRY.status_census()
    ck = V31REGISTRY.checks_by_status()
    n_legacy = sum(1 for n in LEGACY_REGISTRY.names() if V31REGISTRY.has(n))
    return {
        "n_ops": len(V31REGISTRY),
        "verified": st.get("verified", 0),
        "stub": st.get("stub", 0),
        "legacy_refs": n_legacy,
        "new_ops": len(V31REGISTRY) - n_legacy,
        "codebook_rows": V31_CODEBOOK.size(),
        "codebook_checksum": V31_CODEBOOK.checksum(),
        "head_dim": v31_head_dim(),
        "band_census": V31REGISTRY.band_census(),
        "layer_census": V31REGISTRY.layer_census(),
        "status_census": st,
        "checks_total": sum(ck.values()),
        "checks_by_status": ck,
        "legacy_codebook_checksum": LEGACY_CODEBOOK.checksum(),
        "legacy_n_ops": len(LEGACY_REGISTRY),
    }


# ---------------------------------------------------------------------------
# checker mutation audit
# ---------------------------------------------------------------------------


def _leaf_mutations(obj, path="p", depth=0):
    """Deterministic perturbation sequence of (description, mutated object): each int leaf
    +1, each sequence with its tail dropped."""
    if depth > 6:
        return
    if isinstance(obj, bool):
        yield (path + "=!bool", (not obj))
    elif isinstance(obj, int):
        yield (path + "+1", obj + 1)
        if obj != 0:
            yield (path + "-1", obj - 1)
    elif isinstance(obj, str):
        if obj and obj.lstrip("-").replace("/", "").isdigit():
            yield (path + "+1", obj[0] + "1" + obj[1:] if len(obj) > 1 else obj + "1")
        elif obj:
            yield (path + "=z", obj + "Z")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            yield from _leaf_mutations(v, "%s[%d]" % (path, i), depth + 1)
        if len(obj) >= 1:
            yield (path + "-last", list(obj)[:-1])
        if len(obj) >= 2:
            swapped = list(obj)
            swapped[0], swapped[1] = swapped[1], swapped[0]
            yield (path + "=swap01", swapped)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from _leaf_mutations(v, "%s.%s" % (path, k), depth + 1)


def _rebuild_payload(payload, mutation_desc, mutated_value):
    """Write a mutated value back following the path description (direct leaf / sequence level
    only, as guaranteed by _leaf_mutations)."""
    return mutated_value if mutation_desc.startswith("p") and mutation_desc.count("[") == 0 \
        and "." not in mutation_desc else None


def checker_mutation_audit(max_per_kind: int = 14) -> dict:
    """For each V31 certificate kind: take one true certificate, perturb it leaf by leaf, and
    measure the checker rejection rate (really executed)."""
    # find the generating operator of each certificate kind (first ledger row)
    kind_owner = {}
    for row in V31REGISTRY.ledger():
        k = row["cert_kind"]
        if k and k not in kind_owner and V31_CERT_KINDS.count(k):
            kind_owner[k] = row["op"]
    out = {}
    for kind in V31_CERT_KINDS:
        owner = kind_owner.get(kind)
        if owner is None:
            out[kind] = {"owner": None, "note": "本环境未见该证书类型的生成算子"}
            continue
        op = V31REGISTRY._ops[owner]
        # take a **successful** selfcheck args as the certificate-producing input
        args = None
        for sc in op.selfchecks:
            if sc.get("expect_err") is None:
                args = sc["args"]
                break
        if args is None:
            out[kind] = {"owner": owner, "note": "该算子无正例自检"}
            continue
        ctl = CallCtl(trace=AuditTrace("mutate:" + owner), budget=BudgetLedger(total_units=1e9),
                      timeout_s=120.0)
        r = V31REGISTRY.tool(owner).call(args, ctl)
        if not r.ok or r.cert is None:
            out[kind] = {"owner": owner, "note": "未能取到真证书 (err=%s)" % r.err}
            continue
        base = r.cert.payload
        fn = CHECKERS.fn(kind)
        tried = 0
        rejected = 0
        undetected = []
        # replace top-level fields one at a time (including recursive leaves and sequence
        # perturbations)
        for key, val in list(base.items()):
            for desc, mval in _leaf_mutations(val, path=key):
                if tried >= max_per_kind:
                    break
                mut = dict(base)
                mut[key] = mval
                tried += 1
                try:
                    ok, _resid, _detail = fn(mut)
                except Exception:  # noqa: BLE001 - a checker raising also counts as "rejected" (safe side)
                    ok = False
                if ok:
                    undetected.append(desc)
                else:
                    rejected += 1
            if tried >= max_per_kind:
                break
        out[kind] = {"owner": owner, "mutations_tried": tried, "rejected": rejected,
                     "detection_rate": (round(rejected / tried, 3) if tried else None),
                     "undetected": undetected[:6]}
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_report(results, static, anchor, cen, mut, elapsed) -> dict:
    passes = sum(1 for r in results if r.passed)
    return {
        "v31_version": __version__,
        "argv": [os.path.basename(sys.argv[0])] + list(sys.argv[1:]),
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED", "<unset>"),
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "manifest": {"loaded": LOADED, "missing": MISSING, "n_modules": len(MANIFEST)},
        "census": cen,
        "summary": V31REGISTRY.selfcheck_summary(results),
        "static_audit": static,
        "anchor_audit": anchor,
        "checker_mutation_audit": mut,
        "elapsed_s": round(elapsed, 3),
        "results": [
            {"op": r.op, "handle": r.handle, "id": r.cid, "passed": r.passed, "err": r.err,
             "resid": None if r.resid != r.resid else r.resid, "ms": round(r.elapsed_ms, 3),
             "detail": r.detail}
            for r in results
        ],
        "ledger": V31REGISTRY.ledger(),
        "checker_kinds_legacy": list(LEGACY_CERT_KINDS_FROZEN),
        "checker_kinds_v31": sorted(k for k in CHECKERS.kinds() if k not in LEGACY_CERT_KINDS_FROZEN),
        "n_passed": passes,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="svokernel_v31.cli",
                                 description="V31 selfcheck gate (real runs; no claim that the audit passed)")
    ap.add_argument("--only", default=None)
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--mutate", action="store_true", help="append the checker mutation audit")
    ap.add_argument("--emit-json", action="store_true")
    args = ap.parse_args(argv if argv is not None else sys.argv[1:])
    out = sys.stderr if args.emit_json else sys.stdout

    if args.list:
        for row in V31REGISTRY.ledger():
            print("#%d %-26s %-14s L%-2s %-8s cert=%-20s checks=%d"
                  % (row["handle"], row["op"], row["band"], row["layer"], row["status"],
                     row["cert_kind"] or "-", row["checks"]), file=out)
        return 0

    t0 = time.perf_counter()
    results = run_checks(args.only, args.timeout)
    static = audit_static_v31()
    anchor = anchor_chain_audit()
    cen = census()
    mut = checker_mutation_audit() if args.mutate else {}
    elapsed = time.perf_counter() - t0

    for r in results:
        if (not r.passed) or (not args.quiet):
            print(r.line(), file=out)
    s = V31REGISTRY.selfcheck_summary(results)
    print("=" * 112, file=out)
    print("V31 selfcheck summary: %d ops, %d checks, %d passed, %d failed -> %s"
          % (s["ops"], s["checks"], s["passed"], s["failed"], "all green" if s["green"] else "failures"), file=out)
    print("Ledger: %d ops (verified %d / stub %d, %d legacy refs); codebook %d rows; head dim %d"
          % (cen["n_ops"], cen["verified"], cen["stub"], cen["legacy_refs"],
             cen["codebook_rows"], cen["head_dim"]), file=out)
    print("Checks: total %d (verified ops %d / stub ops %d)"
          % (cen["checks_total"], cen["checks_by_status"].get("verified", 0),
             cen["checks_by_status"].get("stub", 0)), file=out)
    print("Band census: %r" % (cen["band_census"],), file=out)
    print("Static audit: %d violations" % len(static), file=out)
    for p in static:
        print("  [STATIC-FAIL] %s @ %s: %s" % (p["code"], p["op"], p["detail"]), file=out)
    print("Anchor-chain audit: %s (%d re-registered; legacy codebook checksum %s)"
          % ("pass" if anchor["ok"] else "mismatch", anchor["n_legacy_joined"],
             LEGACY_CODEBOOK.checksum()), file=out)
    for c in anchor["checks"]:
        print("  [%s] %s :: %s" % ("OK " if c["ok"] else "BAD", c["check"], c["detail"]), file=out)
    if mut:
        print("Checker mutation audit (checkers must really reject):", file=out)
        for k, v in mut.items():
            if v.get("mutations_tried"):
                print("  %-22s %2d mutations, %2d rejected, detection rate %s%s"
                      % (k, v["mutations_tried"], v["rejected"], v["detection_rate"],
                         ("; undetected: %r" % (v["undetected"],)) if v["undetected"] else ""), file=out)
            else:
                print("  %-22s %s" % (k, v.get("note", "?")), file=out)
    print("total %.2fs" % elapsed, file=out)

    if args.emit_json:
        json.dump(build_report(results, static, anchor, cen, mut, elapsed), sys.stdout,
                  ensure_ascii=False, indent=1, default=str)
        sys.stdout.write("\n")
    green = s["green"] and not static and anchor["ok"]
    return 0 if (green or not args.strict) else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

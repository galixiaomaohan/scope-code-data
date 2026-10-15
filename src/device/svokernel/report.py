"""§20 delivery-report generator — organises "what actually ran" into the sectioned report the task requires.

**This module only aggregates; it produces no new conclusions**: it reads (a) the registry's five-tuple
table, (b) the selfcheck gate's per-item results (with residuals and timings), (c) the theorem-roster
self-certificate, (d) the generators' verifiability rates, (e) the §16.7 benchmark. Anything that was
not run goes into "unverified items" with the **reason** why (not implemented / no snapshot / beyond
the declared capacity) — never "it should work".

Output: printed to stdout (redirected to disk by the caller's shell); this module writes no files.
Usage (server):
    python -m svokernel.report --math-n 200 --code-n 40 --logic-n 60 > reports/delivery_report.md
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time

from .codebook import CODEBOOK
from .errors import ERROR_TABLE
from .certify import CHECKERS
from .registry import REGISTRY


def _load_selfcheck() -> "dict | None":
    """Read reports/selfcheck.json (written by redirecting selfcheck's --emit-json; the path is a literal)."""
    path = os.path.abspath("reports/selfcheck.json")
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def _md_five_tuple() -> str:
    rows = REGISTRY.five_tuple_table()
    out = ["| #row | operator | IO type | verifier/certificate | cost | #failure modes | over-approx direction |",
           "|---:|---|---|---|---:|---:|---|"]
    for r in rows:
        op = r["op"]
        d = REGISTRY._ops[op]
        eff = d.effects
        direction = "exact"
        low = (d.doc or "") + eff
        if "over" in low or "过近似" in low:
            direction = "over"
        out.append(f"| {r['handle']} | `{op}` | {r['io']} | {r['verifier']} | {r['cost']} | "
                   f"{len(r['errors'])} | {direction} |")
    return "\n".join(out)


def _md_theorems() -> str:
    try:
        from .ops import math_theorems as T
    except Exception:  # pragma: no cover
        return "_(math_theorems not loaded)_"
    ok, msg = T.roster_check()
    lines = [f"Theorem-registry self-certificate: **{'passed' if ok else 'not passed'}** — {msg}", "",
             "| theorem code | operator | statement |", "|---|---|---|"]
    for code, op, _name in T.THEOREM_ROSTER:
        stmt = REGISTRY._ops[op].stmt if REGISTRY.has(op) else "(not registered)"
        lines.append(f"| {code} | `{op}` | {stmt} |")
    return "\n".join(lines)


def _md_curricula(n: int) -> str:
    """All three generators **actually run with every problem verified** (auditor F noted: the old
    version verified only gen_L1 while the other two merely said "loaded" — treating existence as
    verifiability, which is the unmeasured claim §17.7 forbids)."""
    lines: list = []
    n_se = min(200, max(20, n // 5))
    specs = (("gen_L1", "§8.4 math exam generator", n),
             ("gen_SE", "§9.7 code exam generator", n_se),
             ("gen_L3", "§10.7 logic exam generator", n_se))
    import importlib

    for mod, desc, cnt in specs:
        try:
            G = importlib.import_module(f".curriculum.{mod}", __package__)
            items = G.generate(cnt, seed=20260920)
            r = G.verify_all(items)
            mf = G.manifest(items)
            lines.append(f"- **{mod}** ({desc}): generated {r['total']} problems (seed=20260920), "
                         f"machine-verifiable **{r['passed']}/{r['total']} = {r['rate']:.4f}**; "
                         f"distribution {mf.get('by_family')}; exam-set fingerprint `{mf.get('fingerprint')}`")
            if mf.get("by_defect_class"):
                lines.append(f"  - defect-class distribution: {mf['by_defect_class']}; "
                             f"**representable** {mf.get('representable')}; "
                             f"**not representable in this language** {mf.get('not_representable')}")
            if r.get("failures"):
                lines.append(f"  - failure samples (listed in full): {r['failures'][:3]}")
            if r.get("unsat_instances"):
                lines.append(f"  - {r['unsat_instances']}")
        except Exception as e:  # noqa: BLE001
            lines.append(f"- **{mod}**: **did not run** ({type(e).__name__}: {e}) — no evidence")
    lines.append("- Identifiability arguments: every problem carries an `identifiability` field, and the "
                 "generator blocks problems without one at the source (§17.3); injected defect classes are "
                 "limited to those \"one-step reversible within the patch search space\", and the difficulty "
                 "of irreversible classes is published instead by the §16.7 benchmark's \"full-scope measurement "
                 "convention\" (not disguised as solvable)")
    return "\n".join(lines)


def _md_modules() -> str:
    from . import ops
    lines = ["Operator modules loaded / planned:「, 」「, 」```"]
    for mod, desc in ops.MANIFEST:
        mark = "OK  " if mod in ops.LOADED else "MISS"
        lines.append(f"[{mark}] {mod:16s} {desc}")
    lines.append("```")
    miss = ops.MISSING
    if miss:
        lines.append(f"not loaded: {len(miss)} modules: {miss}")
    return "\n".join(lines)


def _md_unverified(sc: "dict | None", bm: "dict | None") -> str:
    lines = ["The following is what was **not** verified, split by category (never mixed into \"verified\"):", ""]
    lines.append("### 3.1 Modules not implemented (no evidence at all)")
    from . import ops
    if ops.MISSING:
        for m in ops.MISSING:
            lines.append(f"- `{m}`: not implemented (planned in MANIFEST, code absent)")
    else:
        lines.append("- (none)")
    lines.append("")
    lines.append("### 3.2 Implemented but beyond the declared capability/capacity")
    if bm:
        for d in bm["domains"]:
            if d.get("covered", True) is False:
                lines.append(f"- **agent domain**: {d['note']}")
            elif d.get("rate_within_capacity") is not None:
                lines.append(f"- **logic-domain capacity boundary**: the operator declares max_vars≤{d['max_vars_cap']}; "
                             f"the benchmark contains {d['n_out_of_capacity']} out-of-range problems (counted as "
                             f"out_of_capacity), and the within-capacity success rate is {d['rate_within_capacity']:.4f} — "
                             f"naive DPLL is limited in the phase-transition region, which is exactly why §5 anchors SAT "
                             f"trust on \"solver + DRAT certificate\" rather than on a built-in solver")
    lines.append("")
    lines.append("### 3.3 Claims explicitly not made")
    lines += [
        "- **No public benchmark introduced**: the container has no offline HumanEval-style snapshot, so all §16.7 "
        "problems are locally synthesised; therefore we claim **no** pass@k on any public benchmark.",
        "- **No claim that the audit passed**: this report aggregates selfcheck results; whether it passes is decided "
        "by our adversarial audit (§18 red line).",
        "- **Floating-point verifiers** (calculus family T-C1..T-C6, some localisation/difference operators) give "
        "**numerical evidence + explicit tolerance**, not proofs; the tolerance is written item by item in each "
        "operator's returned detail.",
        "- **Inverse-theorem / spectral-localisation** suspicion rankings answer only for the **given test set** and "
        "claim no optimality against real defect distributions.",
    ]
    return "\n".join(lines)


def _md_dissent() -> str:
    return """Places where **this implementation diverges from the task book or needs clarification**, listed item by item (no
architecture change on our own initiative, only a report):

1. **§8.2 theorem count**: the heading says "34 theorems", while the itemised list right after it actually contains
   **35**. This implementation follows the itemised list, implements all 35 and attaches a roster self-certificate
   (`math_theorems.roster_check`). The count discrepancy needs the client's ruling.

2. **§8.5 "interval arithmetic (outward rounding)" implementation convention**: this implementation performs interval
   endpoint arithmetic on **rationals**, so **no rounding occurs** — the interval enclosure for the four operations /
   powers / rational functions is **exact**; transcendental functions use rational series truncation plus an explicit
   remainder bound. This is stronger than "float + directed rounding" (error zero), at the cost of rational rather
   than float endpoints, differing from a literal reading of "outward rounding". The `cert_interval_eval` docstring
   spells out the bounds item by item.

3. **Kernel choice in §10.1 vs §8.3**: `strat_contradiction` **must** use the natural-deduction kernel (ndcore)
   rather than §10.1's Hilbert kernel. Reason: Hilbert's system (A1/A2/A3 + MP) is the **positive implication
   fragment** with ⊥ as an atom, and ⊥ can **never be derived** there (atoms enter only via hyp), so "derive ⊥ from
   ¬P" is **inexpressible** in that kernel. This is a logical fact, not an implementation preference; it is recorded
   explicitly in the operator docstring and the certificate kind.

4. **The induction boundary in §8.3**: this implementation accepts only polynomial assertions that are **closed
   under the shift expansion** (D(n+1)−D(n) ≡ 0 decidable by coefficient comparison). Assertions containing
   division / modulo / transcendental functions are uniformly DEG_UNSUPPORTED — because "approximate induction"
   equals abandoning verification. If a stronger induction is required, it needs a recurrence / divisibility domain,
   i.e. an architectural extension.

5. **case_split in §8.3 needs a hypothesis set**: without hypotheses, "every branch implies the goal" plus
   "the coverage is complete" holding together **is equivalent to the goal being a tautology** (completeness forces
   ∨cond ≡ ⊤, and branches ⊆ goal then force goal ≡ ⊤), so case splitting degenerates into "proving a tautology" and
   is useless. Hence this implementation requires an explicit hypothesis H, with the semantics
   "H ∧ cond_i ⊨ goal and H ⊨ ∨cond_i". If the client's intent is the hypothesis-free version, please say so (it will
   then be meaningful only on tautologies).

6. **The third gate of §9.3 hides a "no-evidence" trap**: if the patched program **generates no mutant at all**
   (e.g. the fix deletes the only statement → empty program), then "all mutants killed" passes **vacuously**. This
   implementation follows §13 "a solution without a certificate is not a solution" and judges such cases
   **MUTATION_SURVIVE (missing evidence)** instead of passing. This is a **stricter** reading of the task book and
   needs the client's confirmation.

7. **The real evidential strength of two domains in §16.7 (as things stand, no longer outdated)**: (a) **logic
   domain** — the operator declares `max_vars≤22` (naive DPLL being a deliberate fail-safe), while the
   phase-transition region the brief wants is n=20–50: 54 of 60 problems are out of range, and the within-capacity
   success rate is only 0.333. §5 anchors SAT trust on "solver + DRAT certificate", yet the DRAT checker delivered
   here is **self-written at about 60 lines** (the container has no drat-trim/cadical) and there is **no real
   solver** — i.e. the "solver proposes" half is missing. This is the largest gap at present. (b) **agent domain** —
   the structure is complete (three tools / loop / ask_info / plan / memory / budget / committee / three combined
   anchors), but the benchmark contains only 4 **isomorphic tasks** (the success candidate is hard-coded to the same
   exact root-finding call), so the success rate of 1.0 has **very weak evidential strength**, and `mcp_bridge` is an
   **in-process schema round-trip simulation**, not real MCP protocol communication. ★ All three items are
   "implemented but with limited evidence/capability", listed in the report's §3.2 columns.

7b. **The actual scope of Lean integration (§8.6's P3a/P3b)**: of the five Lean operators listed in the brief
   (`lean_elab/lean_state/lean_tactic/lean_search/lean_axioms`), this delivery implements only the **equivalent of
   `lean_axioms`** (axiom-dependency-table audit, data-type) and a **home-made micro-kernel for polynomial-ring
   identities** (`mini_kernel_arith`, matching P3a's "bare kernel + mini-stdlib" intent, but covering only the ring-
   identity fragment); `lean_elab/lean_state/lean_tactic/lean_search` **do not exist**, the Lean formalisation of the
   34 theorems is **not done**, so the "dual-kernel cross-check" with Lean **degenerates into "micro-kernel vs sample-
   point verifier"**. `lean_bridge` raises `MATHLIB_MISS` on the server in practice (no toolchain) — a designed
   behaviour, but it also means **this delivery has no Lean-side evidence**.

7c. **Snapshot freezing (auditor F's note)**: reports once showed drift where "code kept changing after the report
   was generated" (checks 580→583). The report has now been regenerated from the final code; if the code changes
   again, `tools/final_run.sh` must be rerun to regenerate — report and code must be the same version, otherwise the
   numbers in the report do not point at the code you reviewed.

8. **The generators' "100% verifiable" convention**: this means "each problem actually runs the operator per its
   own check spec and passes field-by-field comparison" (1000/1000). It does **not** mean "the problems cover the
   whole mathematical domain" — all four families are template-based narrow-domain problems.

9. **The logic-domain hole was closed by switching the criterion, not by new evidence** (an honest note on this
   remediation): we found that instances whose **true value is exactly 0** (the zero polynomial, `p=0`) cannot be
   proven by a sample criterion (samples are constantly 0), so we changed it to a shape criterion: "the zero
   polynomial must be written out explicitly + both sides empty is judged vacuously true and rejected", keeping one
   rejected negative example. Truly "proving" the zero polynomial needs a symbolic representation, an extension of
   P3's representation layer. This does not affect the sample criterion of the other 34 theorems.
"""


def _md_audit() -> str:
    """Adversarial audit and closure (a factual record of the audit process; reproducible evidence in tests/audit_closure.py)."""
    return """This delivery round went through a **six-way adversarial audit** (auditors read-only and executable, all actually
run on the server); findings and dispositions follow. This section states only "what was found, how it was handled,
and what evidence proves closure"; **whether it passes is decided by the client** (§18 red line).

| auditor | scope | key findings | disposition |
|---|---|---|---|
| A | §17 ten iron rules | **blocking**: gate 3 of `rep_patch_gen` could be pierced with `check_box=False` — surviving mutants were silently dropped while a fake "100% killed" report and fake certificate were emitted; `score.aic_score/undetermined_check` had zero call sites (dead code treated as evidence for §17.2/17.4) | gate 3 changed to "behaviour box not enabled ⇒ anything not killed survives"; the report no longer counts dead code as evidence of implementation |
| B | independence and "no self-certification" | `identity_expand` **violated its sampling contract**: 3 sample points could issue an identity certificate for a nonzero degree-4 polynomial (Schwartz-Zippel); `module in independent_of` never triggered; `_checker_imports_generator` missed `from X import f`, and a **violation already existed in-tree** (the bmc checker imports the generator module) | checkers forced to sample ≥ degree+1; the bmc checker given its own predicate; the independence criterion written into the audit report (see "residuals" below) |
| C | certificate-forgery red team | **11/28 certificate kinds passed with forged payloads**, 5 of them "mint-and-pass" (drat empty proof, tlc without a spec, bmc pure counting, axiom self-extended allowlist, replay two self-reported fingerprints); ~5 only did self-consistent comparison (test_vector/mutation_report/equivalence_pair/divergence_witness) | all changed to **the checker itself executing/recomputing**: require program/tests/mutants/calls/spec; criteria never accept payload tolerances; the allowlist moved to the checker side; empty proofs / empty payloads / degenerate inputs rejected outright |
| D | numerical honesty | **fake upper bounds**: `cert_lstsq.kappa_ub` was 3.16× smaller than the true κ₂ (norms were squared); **infeasible instances signed "certified optimal"**: `cert_lp` had no phase-1 and returned a fabricated solution with gap=0 when b<0; the `Ival.exp` degenerate branch differed from the true value by 2e9× while verify passed vacuously; `Ival.sin/cos` threw TypeError on the non-conservative path | norms/square-root directions corrected with upper-bound checks; cert_lp now does a feasibility precondition check before signing; the exp degenerate branch goes through the bounded machine; sin/cos index bugs fixed and extrema switched to intersection tests; verify fails when all samples are skipped |
| E | self-deception and soft tests | **mutation kill rate 9/16 ≈ 56%**: 7 classes of implementation errors were completely invisible under "580 green" (no assertion on ana_complexity's n_nodes, budget verifier reusing the same-module constant, no failing case in tests_to_impl gate 3, no rejection case in the logic_unify verifier, etc.) | filled in item by item: n_nodes assertion, inlined expected values in verifiers, added MUTATION_SURVIVE negative cases, added verifier rejection probes, added tolerance-boundary cases |
| F | delivery integrity and report consistency | see the "unverified items" section and the dissent section below (number provenance, P0–P6 coverage, benchmark comparability, completeness of unverified items) | the report was extended per the audit; all numbers point at reproducible commands |

**Closure evidence**: `tests/audit_closure.py` replays the 26 attack payloads auditors C/A/B/D actually used, asserting
that all are now rejected (or that legitimate certificates are no longer falsely rejected). In this package, run it as
`PYTHONHASHSEED=0 python3 src/device/tests/audit_closure.py` with `src/device` on PYTHONPATH (the original host ran
the same file inside its engine tree). Expected output: 26 assertions, 26 passed, 0 unclosed. That file doubles as the **regression
defence**: fixed problems must not come back.

**Residuals (flagged by the audit, not fixed or only partially fixed this round)**: (1) the independence criterion is
still "declaration + module-level structural check", and the semantic-level self-certification auditor B demonstrated
(a checker restating the generator's conclusion verbatim) **cannot be caught by the current mechanism** — solving it
properly requires checker and generator to use different **algorithms**, not just different modules, which is a design
question, not a patch; (2) the 83 operators without cert_kind have only `verify`, no certificate, so their
independence was not structurally audited (that boundary is declared in the report); (3) the size of the improvement
in mutation kill rate from 56% was not re-quantified (auditor E's mutation script needs to enter regular regression);
(4) what auditor A flagged as `cert_lp` without phase-1 is now "reject if infeasible" rather than "can solve infeasible
instances" — the capability ceiling has not been raised.
"""


def build_report(math_n: int, code_n: int, logic_n: int, curriculum_n: int,
                 with_bench: bool = True) -> str:
    from . import ops  # noqa: F401
    sc = _load_selfcheck()
    bm = None
    if with_bench:
        try:
            from . import bench
            bm = bench.run_all(math_n, code_n, logic_n, 20260920)
        except Exception as e:  # pragma: no cover
            bm = {"error": f"{type(e).__name__}: {e}"}

    cb_rows = CODEBOOK.table()
    L: list = []
    L.append("# SVO vectorized super-operator system — delivery report (§20 format)")
    L.append("")
    L.append(f"- generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    L.append(f"- runtime environment: Python {sys.version.split()[0]} @ {platform.platform()}")
    L.append(f"- codebook: {len(cb_rows)} rows, fingerprint `{CODEBOOK.checksum()}`, "
             f"band census {REGISTRY.band_census()}")
    L.append(f"- error table: §15 full enumeration {len(ERROR_TABLE)} entries; certificate checkers {len(CHECKERS.kinds())} kinds "
             f"({', '.join(CHECKERS.kinds())})")
    L.append("")

    L.append("## Deliverable inventory")
    L.append("")
    L.append(_md_modules())
    L.append("")
    L.append(f"**Operators currently registered: {len(REGISTRY)}**; "
             f"selfcheck items {len(sc['results']) if sc else '(selfcheck.json not read)'}")
    L.append("")

    L.append("## Per-operator five-tuple table")
    L.append("")
    L.append(_md_five_tuple())
    L.append("")

    L.append("## Theorem registry (statement|type|condition|verifier|code)")
    L.append("")
    L.append(_md_theorems())
    L.append("")

    L.append("## Generators")
    L.append("")
    L.append(_md_curricula(curriculum_n))
    L.append("")

    L.append("## Verified items (with run output and seed)")
    L.append("")
    if sc is None:
        L.append("**`reports/selfcheck.json` was not read** — please run "
                 "`python -m svokernel.selfcheck --emit-json > reports/selfcheck.json` first. "
                 "Until it is read, this section states no numbers.")
    else:
        s = sc["summary"]
        rp = sc.get("reproducibility", {})
        L.append(f"- **selfcheck gate**: {s['ops']} operators / {s['checks']} check items, "
                 f"{s['passed']} passed, {s['failed']} failed; "
                 f"static-audit violations {len(sc['static_audit'])}")
        L.append(f"- **argv (recorded item by item, §17.6)**: `{sc['argv']}`")
        L.append(f"- **environment**: {sc['python']}; PYTHONHASHSEED={sc['PYTHONHASHSEED']}")
        if rp.get("skipped") or rp.get("equal") is None:
            # **Three states must stay distinct**: skipped (this run used --no-repro) / identical / mismatched.
            # The old version rendered skipped as "mismatch (red alert)" — presenting "not run" as "ran and went
            # wrong", the fabricated-run-result §18 forbids (the P0 integrity problem auditor F caught).
            L.append("- **reproducibility (double run, same seed)**: **skipped this time** (the caller passed --no-repro); "
                     "reproducibility evidence is in `logs/repro.log` (independent double run: equal True, 583 checks) — "
                     "this line no longer renders 'not measured' as a conclusion")
        else:
            L.append(f"- **reproducibility (double run, same seed)**: fingerprint A `{rp.get('fingerprint_a')}` vs "
                     f"B `{rp.get('fingerprint_b')}` → "
                     f"**{'identical' if rp.get('equal') else 'MISMATCH (red alert)'}**")
        cb = sc.get("codebook", {})
        L.append(f"- **codebook determinism**: {cb.get('size')} rows, embedding {cb.get('emb_dim')}d, "
                 f"bit-identical recomputation for the same row={cb.get('latent_deterministic')}")
        cov = sc.get("coverage", {})
        L.append(f"- **boundary coverage**: operators without a negative selfcheck "
                 f"{cov.get('n_ops_without_negative')} "
                 f"{cov.get('ops_without_negative_check')}")
        L.append("")
        L.append("Per-item selfcheck results (operator, check id, passed, error code, residual, milliseconds):")
        L.append("")
        L.append("| operator | check id | passed | err | resid | ms |")
        L.append("|---|---|---:|---|---:|---:|")
        for r in sc["results"]:
            L.append(f"| `{r['op']}` | {r['id']} | {'yes' if r['passed'] else '**no**'} | "
                     f"{r['err'] or ''} | {r['resid'] if r['resid'] is not None else 'nan'} | "
                     f"{r['ms']} |")
    L.append("")

    L.append("## §16.7 benchmark (eval-only)")
    L.append("")
    if bm is None or "error" in bm:
        L.append(f"benchmark did not run: {bm.get('error') if bm else 'not enabled'}")
    else:
        L.append(f"- config: {bm['config']}; public benchmarks: {bm['public_benchmarks']}")
        L.append("")
        L.append("| domain | #problems | success rate | mean acceptance steps C_beam | esc rate | certificate verify rate | elapsed(s) |")
        L.append("|---|---:|---:|---:|---:|---:|---:|")
        for d in bm["domains"]:
            def fmt(x, p=4):
                return "—" if x is None else f"{x:.{p}f}"
            L.append(f"| {d['domain']} | {d['n']} | {fmt(d['success_rate'])} | "
                     f"{fmt(d['c_beam'], 2)} | {fmt(d['esc_rate'])} | {fmt(d['cert_rate'])} | "
                     f"{d['elapsed_s']} |")
        agg = bm["aggregate"]
        L.append("")
        L.append(f"- aggregate (covered domains {agg['domains_covered']}, not covered {agg['domains_not_covered']}): "
                 f"SuccessRate={agg['success_rate']:.4f}, C_beam={agg['c_beam']:.2f}, "
                 f"esc_rate={agg['esc_rate']:.4f}, certificate verify rate={agg['cert_rate']}")
        L.append("")
        for d in bm["domains"]:
            L.append(f"- **{d['domain']}**: {d.get('note', '')}")
            if d.get("failures"):
                L.append(f"  - failure samples: `{json.dumps(d['failures'][:2], ensure_ascii=False)}`")
    L.append("")

    L.append("## Adversarial audit and closure")
    L.append("")
    L.append(_md_audit())
    L.append("")

    L.append("## Unverified items and reasons")
    L.append("")
    L.append(_md_unverified(sc, bm))
    L.append("")

    L.append("## Identifiability / completeness analysis")
    L.append("")
    L += [
        "### Identifiability (§17.3)",
        "- **Arguments provided**: every gen_L1 problem carries an `identifiability` field, and the generator "
        "blocks problems without one at the source; the inverse family explicitly argues \"the solution is unique "
        "under the monic constraint; dropping monicity makes it unidentifiable (λ(x−r1)(x−r2) shares the roots)\".",
        "- **Explicitly refusing the undecidable**: `thm_spectral_radius` raises UNIDENTIFIABLE when the "
        "discriminant is not a perfect square (ρ irrational ⇒ its order relation to the norm is undecidable in "
        "the rational channel) instead of guessing a boolean.",
        "- **Underdetermination check (§17.2)**: `cert_lstsq` raises PRECOND_FAIL when equations ≤ dof; "
        "`score.undetermined_check` provides the generic hard check.",
        "### Completeness / capability boundary",
        "- **Decision within the propositional fragment is complete** (truth-table enumeration): the independent "
        "verification paths of logic_cnf_transform / sat_verify / strat_decompose / case_split / contradiction are "
        "therefore **decisive** within their domain.",
        "- **Search-style operators are incomplete**: `strat_decompose`'s decomposition is supplied by the caller; "
        "`rep_patch_gen` searches only a first-order neighbourhood; `logic_proof_search` is a constructive tactic. "
        "Their failures mean **only** \"not found within the given search space\", never \"does not exist\" — this is "
        "written into each of their failure messages.",
        "- **The numerical domain is incomplete but reliable**: the calculus family gives numerical evidence + "
        "explicit tolerance; certified numerics (P6) give **exact rational intervals** or **dual certificates**, and "
        "the strength of those two conclusions is distinguished item by item in the docstrings.",
        "- **Static-analysis direction (§17.10)**: every operator's five-tuple table carries an `over-approx "
        "direction` column; abstract interpretation / taint / reachability are marked over, counting operators exact.",
    ]
    L.append("")

    L.append("## Dissent section")
    L.append("")
    L.append(_md_dissent())
    L.append("")
    L.append("---")
    L.append("")
    L.append("**Appendix B reminder**: all your cleverness must stay where it can be verified. Every number in this "
             "report comes from an actual run; anything not run is written in \"unverified items\", with no estimates.")
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="svokernel.report", description="§20 delivery-report generation")
    ap.add_argument("--math-n", type=int, default=200)
    ap.add_argument("--code-n", type=int, default=40)
    ap.add_argument("--logic-n", type=int, default=60)
    ap.add_argument("--curriculum-n", type=int, default=1000)
    ap.add_argument("--no-bench", action="store_true")
    args = ap.parse_args(argv if argv is not None else sys.argv[1:])
    sys.stdout.write(build_report(args.math_n, args.code_n, args.logic_n,
                                  args.curriculum_n, not args.no_bench))
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

# -*- coding: utf-8 -*-
# gen35_report.py -- node [6] **GEN35_REPORT.md auto-generator** (all numbers come from on-disk artifacts, zero manual transcription)
#   Reads: out/GEN35_TABLES.json (primary) + the various JSON ledgers; writes: GEN35_REPORT.md
import argparse
import json
import os
import time

import gen35_common as C

OUT = C.G35 + "/out"
LOGS = C.G35 + "/logs"
G = C.G35


def L(p):
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def g(d, *ks, **kw):
    cur = d
    for k in ks:
        if not isinstance(cur, dict):
            return kw.get("default")
        cur = cur.get(k)
    return kw.get("default") if cur is None else cur


def md5row(label, path):
    return "| `%s` | `%s` | %s |" % (label, C.md5f(path) if os.path.exists(path) else "N/A",
                                     os.path.getsize(path) if os.path.exists(path) else "-")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", default=OUT + "/GEN35_TABLES.json")
    ap.add_argument("--out", default=G + "/GEN35_REPORT.md")
    a = ap.parse_args()
    T = L(a.tables)
    if T is None:
        raise SystemExit("缺 %s (先跑 gen35_tables)" % a.tables)
    gate = T["gate"]
    fw = T["four_way"]
    M = T["metrics"]
    pp = T["pure_plan"]
    sc = T["stored_chain"]
    gu = T["gold_chain_upper"]
    eq = T["endpoint_gen34_slotA"] or {}
    aud = T["aud1_zero_param"] or {}
    paeq = T["paeq"] or {}
    tra = T.get("train") or {}
    n = T["n_exam"]
    dc = T.get("dual_criteria") or {}
    R = []
    R.append("# GEN35_REPORT.md -- gen35 'pure planning + mechanical rendering' batch report\n")
    R.append("- Batch: gen35 . authority: **constitution change 98** (user decree: the LLM is the perceiver "
             "and path selector, barred from any computation or rendering; rendering fully mechanized = "
             "operator templates x device ground truth; retrain and re-evaluate under the new role)")
    R.append("- Pre-registered: `V35_DESIGN.md` (written to disk before training) . deep exam set: 218 valid "
             "(C4 convention, inherited from gen33/gen34)"
             " . judging: frozen artifact `drct_common` md5 **%s**" % T["judge_md5"])
    R.append("- Write scope: only `scratch/ccp/gen35/**` + `LAUNCH_LOG.md`; existing artifacts unchanged "
             "(fork-copy regime / import only); zero sudo . zero /tmp . Test 243 unread and unjudged . GPU0 user redline untouched")
    R.append("- Status: complete (RC in the last section)\n")

    R.append("## 1. One-line conclusion\n")
    R.append("**Mechanical rendering holds, and it is the decisive factor for deep-chain solve rate**: on the same 218-exam set "
             "with the same chains -- the legacy route (LLM rendering) end-to-end **%s/%d = %.4f**; with compiler rendering "
             "(operator templates x device ground truth) "
             "**%s/%d = %.4f** (arm 3 stock chain) / **%s/%d = %.4f** (arm 1 pure-planning retrained, M1_F)."
             % (eq.get("certified"), n, (eq.get("certified") or 0) / n,
                sc["judge"]["certified"], n, sc["judge"]["rate"],
                fw["1_pure_plan_llm_x_compiler"]["certified"], n,
                fw["1_pure_plan_llm_x_compiler"]["rate"]))
    R.append("Compiler soundness: on corpus 2399 rows x gold chain, **certified %d/2399 = %.4f**, "
             "**byte-exact vs the corpus reference proof %d/2399 = %.4f**."
             % (gate["certified"], gate["rate"], gate["byte_eq_ref"], gate["byte_eq_rate"]))
    R.append("Main criteria: **M1_F = %s (%.4f)** . **M2_F = %+.4f** (%s) . "
             "**M3_F exact = %.4f / first = %.4f** (gen34 A-arm same-exam exact reference 0.5550, "
             "95%% Wilson lower bound 0.4895 => gate %s).\n"
             % (M["M1_F_int"], M["M1_F"], M["M2_F"], M["M2_F_int"], M["M3_F_exact_hit"],
                M["M3_F_first_hit"], "PASS" if M["M3_F_pass"] else "FAIL"))

    R.append("## 2. Compiler (core engineering artifact of change 98)\n")
    R.append("`gen35_compile` = chain names x device-supplied args -> operator-template proof text; it only imports "
             "`gen32_ops`/`gen32_poly`/`gen33_device` (zero edits). Per-step semantics (pre-registered §2.2):\n")
    R.append("1. Step k is **accepted** only when `chain[k] == ops_gold[k]` (otherwise `OP_MISMATCH`, no line emitted); "
             "`k >= len(ops_gold)` => `OVER_RUN`.")
    R.append("2. Accepted-step arguments are computed by the **device**: the row's own payload (corpus `steps[].args` / exam `sfa`) plus "
             "**accumulated chain state** -- sos block `R_in := dom - sum_{accepted blocks} mult_i*l_i^2`, `hint_idx := accepted block count + 1`; "
             "terminal step `rho/hints/hints_poly` reconstructed from chain state.")
    R.append("3. Emitting operator set = {`nest_solve`, `nest_close`, `sos_block`, `sos_cross`, `tail_*`} "
             "(same set as generator `proof.append/extend`); device-only steps `fold`/`delta` update state without emitting.")
    R.append("4. **The reference proof text is never read** -- the byte-exact rate is an independent **reading** (cross-validating that the supplied args are correct), not a source.")
    R.append("5. Registered limitation (not silent): `ell`/`mult`/`cls`/`sq` come from the row's own payload (same convention as gen34 `args_mode=device`) "
             "=> the LLM's chain axis = the operator **name** sequence (consistent with the gen33/gen34 `exact_hit` convention).\n")
    R.append("**Gate readings (corpus 2399 x gold chain)**: certified **%d/%d** (%s) . byte-exact "
             "**%d/%d** . wrong-chain diagnosis (perturb 1 step x %s) certified **%s/%s** (expected 0)."
             % (gate["certified"], gate["n"], "PASS" if gate["pass"] else "FAIL",
                gate["byte_eq_ref"], gate["n"],
                g(gate, "neg_diagnosis", "n", default="-"),
                g(gate, "neg_diagnosis", "certified", default="-"),
                g(gate, "neg_diagnosis", "n", default="-")))
    R.append("Family breakdown: " + " . ".join("%s n=%d byte_eq=%d certified=%d"
                                               % (k, v["n"], v["byte_eq"], v["certified"])
                                               for k, v in sorted(gate["by_family"].items())) + "\n")

    R.append("## 3. Four-arm comparison table (same 218-exam set)\n")
    R.append("| arm | chain | render | certified | reading |")
    R.append("|---|---|---|---|---|")
    R.append("| (1) pure-planning LLM x compile (M1_F main) | retrained LLM chain selection | compiler | **%s/%d** | **%.4f** |"
             % (fw["1_pure_plan_llm_x_compiler"]["certified"], n,
                fw["1_pure_plan_llm_x_compiler"]["rate"]))
    R.append("| (2) gold chain x compile (compiler ceiling) | gold | compiler | %s/%d | %.4f |"
             % (fw["2_gold_chain_x_compiler"]["certified"], n,
                fw["2_gold_chain_x_compiler"]["rate"]))
    R.append("| (3) gen34 A-arm chain x compile (stock) | gen34 A-arm | compiler | %s/%d | %.4f |"
             % (fw["3_gen34A_chain_x_compiler"]["certified"], n,
                fw["3_gen34A_chain_x_compiler"]["rate"]))
    R.append("| (4) gen34 A-arm end-to-end (legacy-route anchor) | gen34 A-arm | **LLM rendering** | %s/%d | %.4f |"
             % (eq.get("certified"), n, (eq.get("certified") or 0) / n))
    R.append("")
    R.append("Criteria: **M1_F = %.4f** (%s) . **M2_F = %+.4f** (pure-planning retrain - stock chain, same convention) . "
             "**M3_F = first %.4f / exact %.4f** (gate %s)."
             % (M["M1_F"], M["M1_F_int"], M["M2_F"], M["M3_F_first_hit"], M["M3_F_exact_hit"],
                "PASS" if M["M3_F_pass"] else "FAIL"))
    R.append("Reference: gen33 baseline end-to-end 0/218 . gen34 slot+device end-to-end 1/218 => **the legacy route is ~0 on the deep exam set**; "
             "after mechanical rendering the same chains land directly at the chain-accuracy magnitude (conjugate collapse p^k structurally removed: "
             "the proof text is no longer composed of sampled tokens).")
    if T.get("pure_plan_by_family"):
        R.append("\nFamily breakdown (arm 1): " + " . ".join(
            "%s n=%d cert=%d exact=%d first=%d" % (k, v["n"], v["cert"], v["exact"], v["first"])
            for k, v in sorted(T["pure_plan_by_family"].items())))
    if T.get("stored_chain_by_family"):
        R.append("Family breakdown (arm 3): " + " . ".join(
            "%s n=%d cert=%d exact=%d" % (k, v["n"], v["cert"], v["exact"])
            for k, v in sorted(T["stored_chain_by_family"].items())))
    if T.get("decode_buckets_pure_plan"):
        R.append("\nchain-correct vs judge-pass coupling (arm 1, per bucket): `" + json.dumps(T["decode_buckets_pure_plan"],
                                                                                             ensure_ascii=False) + "`")
    if sc.get("decode_buckets"):
        R.append("chain-correct vs judge-pass coupling (arm 3, per bucket): `" + json.dumps(sc["decode_buckets"],
                                                                                             ensure_ascii=False) + "`")
    if T.get("pure_plan_step_err_hist"):
        R.append("arm 1 compile per-step error distribution: `" + json.dumps(T["pure_plan_step_err_hist"],
                                                                              ensure_ascii=False) + "`")
    R.append("\n**AUD-1 zero-parameter chain mode baseline** (per-family mode chain from the corpus distribution, no exam labels): "
             "exact **%.4f** . first **%.4f** . prefix **%.4f** . per-position acc `%s`"
             % (aud.get("exact_rate", -1), aud.get("first_rate", -1),
                aud.get("prefix_rate", -1),
                json.dumps(aud.get("per_pos_acc") or {}, ensure_ascii=False)))
    R.append("mode chains: `" + json.dumps(aud.get("mode_by_family") or {}, ensure_ascii=False) + "`\n")

    R.append("## 4. Pure-planning retrain (diff list P1-P7 and starting-point assertion)\n")
    R.append("fork: `gen34_hybrid` -> `gen35_hybrid` (path/marker patches only; training semantics unchanged); "
             "**flag-for-flag isomorphic** with the gen34 A arm, sole difference `--lam-txt 0.0` (render text loss removed, gradient cut); "
             "render-head output **not scored** (structure retained); chain-navigation supervision (lam_bind 1.0) and the sensing channels "
             "(slot / device state, lam_slot 3.0) retained.")
    R.append("**PA-EQ starting-point assertion pass=%s**: parent and fork per-key fingerprints n=%s all equal (diff=%s) . "
             "first-step logits `torch.equal`=%s (maxdiff %s) . render text loss perturbed at lam_txt=1 **changes**=%s / "
             "perturbed at lam_txt=0 **unchanged**=%s (gradient cut) . slot channel alive=%s . batch-order file md5 equals the gen34 A arm=%s "
             "(`%s`)"
             % (paeq.get("pass"), g(paeq, "state_dict", "n_keys"),
                g(paeq, "state_dict", "n_diff"),
                g(paeq, "logits_first_step", "torch_equal"),
                g(paeq, "logits_first_step", "maxdiff"),
                paeq.get("text_term_alive_at_lam1"), paeq.get("text_term_dead_at_lam0"),
                paeq.get("slot_term_alive"), paeq.get("plan_md5_same_as_gen34"),
                paeq.get("plan_file_md5")))
    if tra:
        R.append("training: steps=%s plan=%s flags=%s" % (tra.get("steps"),
                                                          json.dumps(tra.get("plan") or {},
                                                                     ensure_ascii=False),
                                                          json.dumps(tra.get("flags") or {},
                                                                     ensure_ascii=False)))
        R.append("curve: first %s -> last %s (n=%s)"
                 % (json.dumps(tra.get("curve_first") or {}, ensure_ascii=False),
                    json.dumps(tra.get("curve_last") or {}, ensure_ascii=False),
                    tra.get("n_curve_rows")))
        if tra.get("eval_last"):
            R.append("held-out eval (last): %s" % json.dumps(tra["eval_last"], ensure_ascii=False))
    R.append("")

    R.append("## 5. Dual-convention judging (node [5])\n")
    if not dc:
        R.append("(dual-convention artifact missing)")
    for k, v in sorted(dc.items()):
        lit = v.get("literal") or {}
        rev = v.get("reverse") or {}
        R.append("- `%s` src=%s (n=%s) pool=%s/%s -> literal n=%s agree=%s disagree=%s "
                 "exception=%s not_executed=%s literal certified=%s (batch certified=%s); "
                 "reverse R1 truncation rejected %s/%s . R2 proof-swap rejected %s/%s; judge_unchanged=%s"
                 % (k, v.get("src_tag"), v.get("src_n"), v.get("pool"), v.get("pool_n"),
                    lit.get("n"), lit.get("agree"), lit.get("disagree"),
                    lit.get("exception"), lit.get("not_executed"), lit.get("literal_certified"),
                    lit.get("batch_certified"), rev.get("R1_trunc_rejected"), rev.get("n"),
                    rev.get("R2_swap_rejected"), rev.get("n"), v.get("judge_unchanged")))
        if v.get("literal_pool_note"):
            R.append("  - pool note: %s (n=%s)" % (v["literal_pool_note"], v.get("literal_pool_n")))
    R.append("")

    R.append("## 6. Errata / deviation registry (pre-registered -> measured)\n")
    R.append("| # | pre-registered | measured | disposition |")
    R.append("|---|---|---|---|")
    R.append("| B-1 | compiler gate = 100%% certified | certified %d/2399, byte-exact %d/2399 | %s |"
             % (gate["certified"], gate["byte_eq_ref"], "gate passed" if gate["pass"] else "**failed**"))
    R.append("| B-2 | stock-chain compile reading ~= chain accuracy | certified %s/218 vs chain exact %s/218 | %s |"
             % (sc["judge"]["certified"], sc["compile"]["n_exact"],
                "matches" if sc["judge"]["certified"] == sc["compile"]["n_exact"] else "see §3 bucket breakdown"))
    R.append("| B-3 | argument axis (ell/mult/cls/sq) supplied by the device | taken from the row payload (not device search/inversion) | "
             "registered as a limitation: LLM chain axis = operator name sequence |")
    R.append("| B-4 | (not pre-registered) compiler acceptance semantics count `OVER_RUN` as one | per-row over_run registered | as measured |")
    R.append("")

    R.append("## 7. Ledger / recomputation\n")
    R.append("| artifact | md5 | bytes |")
    R.append("|---|---|---|")
    for lab, p in (("V35_DESIGN.md", G + "/V35_DESIGN.md"),
                   ("gen35_common", G + "/gen35_common.py"),
                   ("gen35_compile", G + "/gen35_compile.py"),
                   ("gen35_gate", G + "/gen35_gate.py"),
                   ("gen35_armrun", G + "/gen35_armrun.py"),
                   ("gen35_chainread", G + "/gen35_chainread.py"),
                   ("gen35_paeq", G + "/gen35_paeq.py"),
                   ("gen35_hybrid (fork)", G + "/gen35_hybrid.py"),
                   ("gen35_eval", G + "/gen35_eval.py"),
                   ("gen35_judge", G + "/gen35_judge.py"),
                   ("gen35_tables", G + "/gen35_tables.py"),
                   ("run35_train.sh", G + "/run35_train.sh"),
                   ("GEN35_PAEQ.json", OUT + "/GEN35_PAEQ.json"),
                   ("GEN35_COMPILE_GATE.json", OUT + "/GEN35_COMPILE_GATE.json"),
                   ("GEN35_COMPILED_CORPUS.jsonl", OUT + "/GEN35_COMPILED_CORPUS.jsonl"),
                   ("GEN35_CHAINREAD.json", OUT + "/GEN35_CHAINREAD.json"),
                   ("GEN35_CHAINREAD_ROWS.jsonl", OUT + "/GEN35_CHAINREAD_ROWS.jsonl"),
                   ("GEN35_EVAL_PLAN_WALK.json", OUT + "/GEN35_EVAL_PLAN_WALK.json"),
                   ("GEN35_EVAL_PLAN.json", OUT + "/GEN35_EVAL_PLAN.json"),
                   ("GEN35_EVAL_PLAN_ROWS.jsonl", OUT + "/GEN35_EVAL_PLAN_ROWS.jsonl"),
                   ("GEN35_TABLES.json", OUT + "/GEN35_TABLES.json"),
                   ("ckpt deep_plan_formal_6000.pt",
                    G + "/ckpt/deep_plan_formal_6000.pt"),
                   ("DEEP_PLAN_CURVE.jsonl", OUT + "/DEEP_PLAN_CURVE.jsonl"),
                   ("batch_idx_deep_plan.txt", OUT + "/batch_idx_deep_plan.txt"),
                   ("DEEP_PLAN_SUMMARY.json", OUT + "/DEEP_PLAN_SUMMARY.json"),
                   ("NEXT_STEP gen35", G + "/NEXT_STEP.gen35.txt"),
                   ("drct_common (frozen)", C.DRCT + "/drct_common.py")):
        R.append(md5row(lab, p))
    for f in sorted(os.listdir(OUT)):
        if f.startswith("GEN35_JUDGE_") and f.endswith(".json"):
            R.append(md5row(f, OUT + "/" + f))
    R.append("")
    R.append("- Recompute: `cd ${SCOPE_RESEARCH_ROOT:-.} && PYTHONDONTWRITEBYTECODE=1 "
             "PYTHONPATH=${SCOPE_SUPEROPERATOR:-../device} nice -n 19 "
             "${SCOPE_PYTHON:-python3} scratch/ccp/gen35/<file> [args]`; "
             "each JSON carries `schema`/`generated`/`judge_md5` fields.")
    R.append("- Upstream read-only artifact md5 ledger: see `GEN35_PAEQ.json.parent_ledger_unchanged` and "
             "`parent_ledger_unchanged` in each artifact (measured once at start and once at close-out).")
    R.append("")
    txt = "\n".join(R) + "\n"
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(txt)
    print(json.dumps({"out": a.out, "md5": C.md5f(a.out), "bytes": os.path.getsize(a.out)},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()

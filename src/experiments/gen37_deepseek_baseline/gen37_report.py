# -*- coding: utf-8 -*-
# gen37_report.py -- gen37 external baseline [5]: comparison table + final report (reads existing files; writes gen37/** only)
#   Rows: 1) DeepSeek-Prover-V1.5-RL 7B int8 greedy (gen37 M1_H) 2) our 135M orchestration (gen35 M1_F)
#       3) gold chain x compiler upper bound (gen35) 4) gen34 end-to-end old route 5) same-model external
#       precedent UN-030 (DRC 243 exam, different exam)
#   All convention differences disclosed (whether LLM rendering / token definition / hardware / parallelism).
import argparse
import hashlib
import json
import os
import statistics
import time

RES = os.environ.get("SCOPE_RESEARCH_ROOT", ".")
G37 = RES + "/scratch/ccp/gen37"
G35 = RES + "/scratch/ccp/gen35"
DRCT = RES + "/scratch/drct"
JUDGE_MD5_EXPECT = "7c31b58da9d7228cee31b9be581fe5dd"


def md5f(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def jload(p, default=None):
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except Exception:                                                # noqa: BLE001
        return default


def jlload(p):
    out = []
    try:
        with open(p, encoding="utf-8") as f:
            for l in f:
                if l.strip():
                    out.append(json.loads(l))
    except Exception:                                                # noqa: BLE001
        pass
    return out


def md_table(head, rows):
    s = "| " + " | ".join(head) + " |\n|" + "|".join(["---"] * len(head)) + "|\n"
    for r in rows:
        s += "| " + " | ".join(str(x) for x in r) + " |\n"
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=G37 + "/GEN37_REPORT.md")
    ap.add_argument("--cmp", default=G37 + "/out/GEN37_COMPARE.json")
    a = ap.parse_args()

    build = jload(G37 + "/out/GEN37_BUILD.json", {}) or {}
    res = jload(G37 + "/out/GEN37_DSRL_resource.json", {}) or {}
    judge = jload(G37 + "/out/GEN37_JUDGE_summary.json", {}) or {}
    dual = jload(G37 + "/out/GEN37_DUAL.json", {}) or {}
    recs = jlload(G37 + "/out/GEN37_JUDGE_records.jsonl")
    atts = {r["name"]: r for r in jlload(G37 + "/out/GEN37_DSRL_attempts.jsonl")}
    p35p = G35 + "/out/GEN35_EVAL_PLAN.json"
    p35w = G35 + "/out/GEN35_EVAL_PLAN_WALK.json"
    p35r = G35 + "/out/GEN35_EVAL_PLAN_ROWS.jsonl"
    g35p, g35w = jload(p35p, {}) or {}, jload(p35w, {}) or {}
    g35rows = jlload(p35r)
    exam = []
    try:
        with open(RES + "/scratch/ccp/gen33/out/DEEP_EXAM_SIDECAR.jsonl", encoding="utf-8") as f:
            exam = [json.loads(l) for l in f if l.strip()]
    except Exception:                                                # noqa: BLE001
        pass
    cl_sum = sum(x.get("chain_len") or 0 for x in exam)
    dp_sum = sum(x.get("decision_points") or 0 for x in exam)
    n_exam = len(exam) or 218
    jm = md5f(DRCT + "/drct_common.py")
    lines = []
    W = lines.append
    W("# gen37 external baseline report — DeepSeek-Prover-V1.5-RL(7B, int8) x 218 deep exam vs our 135M orchestration architecture")
    W("")
    W("- generated: %s | judge module: `scratch/drct/drct_common.py` md5 **%s** (frozen 7c31b58d match=%s)"
      % (time.strftime("%Y-%m-%dT%H:%M:%S%z"), jm, jm == JUDGE_MD5_EXPECT))
    W("- exam: 218 valid deep-exam problems after gen33 C4 (`DEEP_EXAM_SIDECAR.jsonl` md5 %s); same exam and same judge module as gen35 M1_F"
      % (build.get("exam_md5", "?"),))
    W("- external arm convention: official quick_start completion-form zero-shot prompt (E3 official header) + int8 + **greedy** + "
      "max_new=%s; one attempt per problem (attempt=0)" % (res.get("sampling", {}).get("max_new_tokens_cap", "?")))
    W("")
    # ---------------- main reading ----------------
    nc = judge.get("certified", 0)
    nn = judge.get("n", 0) or 1
    eff = judge.get("efficiency", {})
    W("## 0. Main reading (pre-registered M1_H)")
    W("")
    W("- **M1_H = DeepSeek certified = %d/%d = %.4f%%**" % (nc, nn, 100.0 * nc / nn))
    W("- raw Lean accepted (before N1/N2) = %s; judging exceptions = %s; empty proofs = %s"
      % (judge.get("lean_raw_accepted"), judge.get("exceptions"), judge.get("proof_empty")))
    W("- rejection reason histogram: `%s`" % json.dumps(judge.get("reason_hist", {}), ensure_ascii=False))
    W("")
    W("## 1. Efficiency trio (requested metering; full 218-problem exam)")
    W("")
    t1 = res.get("T1_tokens", {})
    t2 = res.get("T2_wall", {})
    W(md_table(["quantity", "value", "convention"], [
        ["prompt_tokens total/per problem", "%s / %s" % (t1.get("input_sum"), t1.get("input_per_q")),
         "header+statement tokenizer length (model tokenizer)"],
        ["completion_tokens total/per problem", "%s / %s" % (t1.get("output_sum"), t1.get("output_per_q")),
         "generated token count (incl. residual after stop; untruncated count)"],
        ["total wall (sum of per-problem generate)", "%s s" % t2.get("decode_wall_sum_s"), "pure model decode time"],
        ["total wall (batch incl. tokenize/write)", "%s s" % t2.get("batch_wall_s"), "end-to-end batch time"],
        ["per-problem time (decode)", "%s s" % t2.get("secs_per_q_decode"), "decode_wall_sum / n"],
        ["**throughput tok/s (generation)**", "%s" % t2.get("toks_per_s_output"), "completion_tokens / decode_wall_sum"],
        ["throughput tok/s (batch)", "%s" % t2.get("toks_per_s_batch"), "completion / batch_wall"],
        ["problems/hour", "%s" % t2.get("problems_per_h"), ""],
        ["GPU peak VRAM", "%s MiB" % res.get("T3_vram", {}).get("nvidia_smi_peak_mib"),
         res.get("T3_vram", {}).get("device")],
        ["judging wall (batch, 32 jobs parallel)",
         "%s s" % eff.get("judge_wall_batch_s"), "Lean judging (CPU; not counted in LLM time)"],
    ]))
    W("")
    # ---------------- stratification ----------------
    W("## 2. DeepSeek stratified readings (certified)")
    W("")
    rows = []
    for fam, d in sorted((judge.get("by_family") or {}).items()):
        rows.append([fam, d["n"], d["certified"], "%.2f%%" % (100.0 * d["certified"] / max(1, d["n"]))])
    for cls, d in sorted((judge.get("by_cls") or {}).items()):
        rows.append(["cls=%s" % cls, d["n"], d["certified"],
                     "%.2f%%" % (100.0 * d["certified"] / max(1, d["n"]))])
    W(md_table(["stratum", "n", "certified", "rate"], rows))
    W("")
    # ---------------- dual convention ----------------
    lit, rev = dual.get("literal", {}) or {}, dual.get("reverse", {}) or {}
    W("## 3. Dual-convention judging (certified stratified literal re-judging + reverse checks)")
    W("")
    W("- literal re-judging: n=%s (required >=60; pool=%s) · agree=%s · disagree=%s · exception=%s · "
      "not_executed=%s · literal_certified=%s · batch_certified=%s"
      % (lit.get("n"), dual.get("pool_n"), lit.get("agree"), lit.get("disagree"),
         lit.get("exception"), lit.get("not_executed"), lit.get("literal_certified"),
         lit.get("batch_certified")))
    W("- reverse checks: n=%s · R1 terminal truncation rejected %s/%s · R2 cross-problem swap rejected %s/%s"
      % (rev.get("n"), rev.get("R1_trunc_rejected"), rev.get("n"),
         rev.get("R2_swap_rejected"), rev.get("n")))
    W("- judge module identical before/after: %s" % dual.get("judge_unchanged"))
    W("")
    # ---------------- comparison table ----------------
    our_n = g35p.get("judge", {}).get("n") or len(g35rows) or 218
    our_c = g35p.get("judge", {}).get("certified")
    our_rate = g35p.get("judge", {}).get("rate")
    our_walk = g35w.get("wall_s")
    our_jw = g35p.get("judge", {}).get("wall_s")
    our_lines = sum(v.get("n_emit", 0) for v in (g35p.get("compile", {}).get("fam", {}) or {}).values())
    pchars = [len(r.get("proof_text") or "") for r in g35rows]
    W("## 4. Comparison table (final deliverable of the task spec)")
    W("")
    heads = ["contestant / architecture", "certified (n=%d)" % nn, "rate", "LLM generated tokens per problem",
             "LLM input tokens per problem", "time per problem (LLM)", "hardware (same-generation 12GB)", "rendering source"]
    rows = [
        ["**DeepSeek-Prover-V1.5-RL 7B** int8 greedy (gen37 M1_H; external baseline)",
         "%d/%d" % (nc, nn), "%.4f%%" % (100.0 * nc / nn),
         "%s (batch; decode convention %s)"
         % (t1.get("output_per_q"), res.get("sampling", {}).get("max_new_tokens_cap")),
         "%s" % t1.get("input_per_q"),
         "%s s (decode) / %s s (batch)" % (t2.get("secs_per_q_decode"), t2.get("secs_per_q_batch")),
         "RTX 4070 SUPER 12282MiB (GPU2 exclusive); peak %s MiB"
         % res.get("T3_vram", {}).get("nvidia_smi_peak_mib"),
         "LLM free text (full proof generation)"],
        ["**our 135M orchestration architecture** (gen35 M1_F pure planning x compiler; same exam, same judge module)",
         "%s/%s" % (our_c, our_n), "%.4f%%" % (100.0 * (our_rate or 0)),
         "0 free text (operator-selection decisions %.2f/problem = chain steps %d/%d; one argmax per step over the "
         "legal operator space, step budget WALK_BUDGET=12 (gen34_eval.py:151); task-spec registered convention "
         "12-30 operator-selection tokens/problem)"
         % (cl_sum / max(1, n_exam), cl_sum, n_exam),
         "frame-state input (not free-text tokens); chain decision points %.2f/problem" % (dp_sum / max(1, n_exam)),
         "%.2f s (chain walk wall %s s/%d)" % ((our_walk or 0) / max(1, our_n), our_walk, our_n),
         "RTX 4070 Ti 12282MiB (GPU1); 135.5M fp32",
         "**compiler rendering (0 LLM tokens)**; %d lines emitted (%.2f/problem), proof median %s chars"
         % (our_lines, our_lines / max(1, n_exam),
            int(statistics.median(pchars)) if pchars else "?")],
        ["[reference] gold chain x compiler upper bound (gen35)",
         "218/218", "100.0000%", "0", "0", "compiler CPU (judging wall %s s/batch)" % our_jw,
         "same as above", "compiler (upper-bound reference)"],
        ["[reference] gen34 end-to-end old route (LLM-rendered text)",
         "1/218", "0.4587%", "full proof text", "statement", "—", "RTX 4070 Ti", "LLM free text"],
        ["[precedent, different exam] UN-030: same model int8 greedy x DRC 243 exam (raw 78/243)",
         "59/243", "24.2798%", "—", "—", "3858.8 s/243 problems = 15.9 s/problem",
         "RTX 4070 Ti (12282MiB)", "LLM free text"],
    ]
    W(md_table(heads, rows))
    W("")
    W("### 4b. Convention-difference disclosure (honest list; where the two sides are not comparable)")
    W("")
    W("1. **LLM output nature differs**: DeepSeek generates **the full proof text** (free tokens); our 135M only does "
      "**discrete operator selection** (one argmax per step), and the proof text is rendered by the compiler from "
      "operator templates x device ground truth => our rendered tokens = 0, no sampling-error accumulation "
      "(proved in gen35: chain correct <=> judged passed, 117 exact <=> 117 certified).")
    W("2. **input token definition differs**: DeepSeek input = official header + statement Lean text; our input = "
      "device frame state (not free-text tokens); the statement text is not fed to the LLM.")
    W("3. **hardware/batch convention**: both sides are local single-GPU 12GB-class 40-series (DeepSeek: GPU2 "
      "RTX 4070 SUPER / ours: GPU1 RTX 4070 Ti), both batch=1, greedy. The two sides were **not run at the same "
      "time on the same card**; same-generation, same-magnitude convention.")
    W("4. **judging time listed separately**: Lean judging (frozen module) is on the CPU side and is not counted in "
      "either side's LLM time; the DeepSeek judging batch was a single full 218-problem run; our 117/121 readings "
      "come from the same judge module in gen35.")
    W("5. **prompt header**: this distribution's E3 official header (`import MiniF2F.ProblemImports`; not the HF "
      "quick_start `import Mathlib/Aesop` header — the registered adjudication for this statements-only miniF2F "
      "distribution), same form as UN-030; that choice may underestimate DeepSeek (header out of training "
      "distribution), recorded.")
    W("6. **attempt count**: DeepSeek this batch = 1 attempt per problem (greedy, pass@1 convention); our M1_F = 1 chain walk.")
    W("")
    # ---------------- conclusion ----------------
    W("## 5. Honest conclusion (written from measurements)")
    W("")
    dsq = float(t1.get("output_per_q") or 0)
    ourq = cl_sum / max(1, n_exam)          # ours: operator decisions (chain steps) per problem
    r_tok = (dsq / ourq) if ourq else 0.0
    r_size = 7.55e9 / 1.355e8
    if nc / nn >= (our_rate or 0):
        verdict = ("**DeepSeek scores high (certified %.2f%% >= our %.2f%%)** => important calibration: this deep-exam "
                   "task is **not hard** for a 7B proof-specialized model (same-exam certified is not below "
                   "\"135M chain selection + compiler rendering\"); our advantage should be positioned on the "
                   "**cost/auditability/size** axes: LLM generated tokens per problem %.0f (DeepSeek) vs "
                   "%.2f operator decisions (ours, 0 free-text tokens) = x%.0f; size 135.5M vs 7.55B = x%.0f."
                   % (100.0 * nc / nn, 100.0 * (our_rate or 0), dsq, ourq, r_tok, r_size))
    else:
        verdict = ("**DeepSeek does not beat ours (certified %.2f%% < our %.2f%%)** => external corroboration of the "
                   "device-grounded + mechanized rendering route: the 7B proof-specialized model did not win against "
                   "\"135M chain selection + compiler rendering\" on this synthetic-family deep exam; the cost gap "
                   "holds as well (LLM generated tokens per problem %.0f vs %.2f operator decisions = x%.0f; "
                   "size 135.5M vs 7.55B = x%.0f)."
                   % (100.0 * nc / nn, 100.0 * (our_rate or 0), dsq, ourq, r_tok, r_size))
    W(verdict)
    W("")
    W("### 5b. Stratified reading (family)")
    W("")
    for fam, d in sorted((judge.get("by_family") or {}).items()):
        o = next((v for k, v in (g35p.get("judge", {}).get("fam", {}) or {}).items() if k == fam), {})
        W("- %s: DeepSeek %d/%d vs ours %s/%s"
          % (fam, d["certified"], d["n"], o.get("certified", "?"), o.get("n", "?")))
    W("")
    # ---------------- key artifacts ----------------
    W("## 6. Key artifact md5")
    W("")
    for k, p in (("prompt file", G37 + "/out/GEN37_PROMPTS.jsonl"),
                 ("build metadata", G37 + "/out/GEN37_BUILD.json"),
                 ("attempts", G37 + "/out/GEN37_DSRL_attempts.jsonl"),
                 ("resource", G37 + "/out/GEN37_DSRL_resource.json"),
                 ("judge records", G37 + "/out/GEN37_JUDGE_records.jsonl"),
                 ("judge summary", G37 + "/out/GEN37_JUDGE_summary.json"),
                 ("dual file", G37 + "/out/GEN37_DUAL.json"),
                 ("comparison file", a.cmp),
                 ("frozen judge module", DRCT + "/drct_common.py")):
        if os.path.exists(p):
            W("- %s: `%s` **%s**" % (k, os.path.basename(p), md5f(p)))
    W("")
    cmp_doc = {"schema": "gen37-compare/1", "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
               "M1_H": judge.get("M1_H"), "deepseek": {"certified": nc, "n": nn,
                                                       "rate": nc / nn if nn else None,
                                                       "out_tokens_per_q": t1.get("output_per_q"),
                                                       "in_tokens_per_q": t1.get("input_per_q"),
                                                       "secs_per_q": t2.get("secs_per_q_decode"),
                                                       "toks_per_s": t2.get("toks_per_s_output")},
               "ours_135m": {"certified": our_c, "n": our_n, "rate": our_rate,
                             "walk_wall_s": our_walk, "secs_per_q": (our_walk or 0) / max(1, our_n),
                             "chain_steps_per_q": cl_sum / max(1, n_exam),
                             "render_tok_per_q": 0, "emit_lines": our_lines},
               "judge_wall": {"deepseek_batch_s": eff.get("judge_wall_batch_s"),
                              "ours_batch_s": our_jw},
               "dual": {"literal_n": lit.get("n"), "literal_agree": lit.get("agree"),
                        "R1_rejected": rev.get("R1_trunc_rejected"),
                        "R2_rejected": rev.get("R2_swap_rejected")},
               "oath": {"judge_md5": jm, "judge_frozen_ok": jm == JUDGE_MD5_EXPECT,
                        "single_card_only": "GPU2", "other_gpus": "GPU0 (other-user red line) / GPU1 (reserved) zero occupancy"}}
    os.makedirs(os.path.dirname(a.cmp), exist_ok=True)
    with open(a.cmp, "w", encoding="utf-8") as f:
        json.dump(cmp_doc, f, ensure_ascii=False, indent=1)
    W("")
    W(">`GEN37_COMPARE.json` machine-readable file: %s **%s**" % (os.path.basename(a.cmp), md5f(a.cmp)))
    with open(a.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(json.dumps({"report": a.out, "md5": md5f(a.out), "cmp_md5": md5f(a.cmp),
                      "M1_H": judge.get("M1_H")}, ensure_ascii=False))


if __name__ == "__main__":
    main()

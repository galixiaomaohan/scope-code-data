# -*- coding: utf-8 -*-
# gen23_exam.py -- gen23 (A1 build batch) T3: exam-set assembly EXAM_V1 (P3 composite paper)
#
# Constitution change 71 (user ruling "execute as recommended", all of P1-P7) + A_DEVICE_DESIGN sections 7/10 P3 + 11.1-3 (three corrections)
#   * Main criterion set = gen14 bridge-hard problems where **RB0 and RBx and RBs all score zero solves**
#     (three-arm readings taken from BH_HARDGATE_CERT_FULL 6e4dcaa2, not re-run); nt is demoted entirely to
#     the disclosure column because RBs=1.000 -- executed as-is and reported as measured, nothing padded.
#   * Target per class: main criterion set >= 30; shortfalls reported as-is.
#   * 3 real (miniF2F) ineq problems = the ANCHOR_V2 (42a9d403) ineq anchor set, reference column (marked sample-limited).
#   * Every exam text passes N1/N2 (same gate on both sides = the same frozen judge module drct_common 7c31b58d):
#     model-side / reference-side texts come from the T1 prescreen trace (A_PRESCREEN_TRACE); rule-arm side
#     texts = tactic lines recovered by **device execution** of the three arm programs (recorded via OPS.call),
#     uniformly converted to inline form and passed through the same batch judge
#     (gen23_prescreen_batch.judge_items_batch).
#   * Test 243 is never read (zero references anywhere; problem sources = synthetic curriculum + miniF2F Valid anchor set).
#
# Phases: stage1 (zero Lean) / stage2 (Lean batch) / stage3 (assembly) / all
import argparse
import collections
import hashlib
import json
import os
import sys
import time

RES = os.environ.get("SCOPE_RESEARCH_ROOT", ".")
CCP = os.path.join(RES, "scratch", "ccp")
G23 = os.path.join(CCP, "gen23")
OUT = os.path.join(G23, "out")
LEANWD = os.path.join(G23, "lean")
BHOUT = os.path.join(CCP, "bh", "out")
PB = os.path.join(OUT, "A_CORPUS_BUILD.jsonl")
PT = os.path.join(OUT, "A_PRESCREEN_TRACE.jsonl")
PC = os.path.join(OUT, "A_CORPUS_V1.jsonl")
PS = os.path.join(OUT, "A_CORPUS_V1_STATS.json")
PEB = os.path.join(OUT, "A_EXAM_BUILD.json")
PET = os.path.join(OUT, "A_EXAM_RULE_TEXTS.jsonl")
PEJ = os.path.join(OUT, "A_EXAM_RULE_JUDGE.json")
PEX = os.path.join(OUT, "EXAM_V1.json")
for _p in (os.path.join(RES, "scratch", "drct"), os.path.join(CCP), os.path.join(CCP, "bh"),
           os.path.join(CCP, "gen17")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import drct_common as DC                                             # noqa: E402
import bh_rule as RULE                                               # noqa: E402
import bh_env as ENV                                                 # noqa: E402
import bh_ops as OPS                                                 # noqa: E402
import bh_families as FAM                                            # noqa: E402
import gen23_prescreen_batch as PBATCH                               # noqa: E402

MASTER_SEED = 20260924                                               # same as BH_HARDGATE_CERT_FULL.master_seed

CLASSES = ("ind", "ineq", "nt")
CERT = os.path.join(BHOUT, "BH_HARDGATE_CERT_FULL.json")
CERT_MD5_EXPECT = "6e4dcaa23bd7821a7164b3ced7d42665"
ANCHOR = os.path.join(CCP, "ANCHOR_V2.json")
ANCHOR_MD5_EXPECT = "42a9d4037142af2b2a6928f7646e8dc3"
LOG = os.path.join(G23, "logs", "gen23_exam.log")
os.makedirs(os.path.dirname(LOG), exist_ok=True)


def md5f(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def log(msg):
    line = "[%s] %s" % (time.strftime("%Y-%m-%dT%H:%M:%S%z"), msg)
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_chains(cls):
    out = {}
    p = os.path.join(CCP, "bh", "gen", "bh_%s_chains.jsonl" % cls)
    for l in open(p, encoding="utf-8"):
        if l.strip():
            c = json.loads(l)
            out.setdefault(c["name"], c)
    return out


def arm_text(cls, pr, arm):
    """Arm program -> text (tactic lines recovered from device execution; semantics = the device's own
    run_program)."""
    rec = []
    orig = OPS.call

    def _rec(op, args):
        r = orig(op, args)
        rec.append((op, args, r))
        return r
    OPS.call = _rec
    try:
        ok, why, prog = RULE.run_arm(arm, cls, pr)
    finally:
        OPS.call = orig
    lines = []
    for (_op, _a, r) in rec:
        t = (getattr(r, "tactic", "") or "").strip()
        if t:
            lines.extend([x for x in t.split("\n")])
    return ok, why, prog, "\n".join(lines), len(rec)


def stage1(families=None, limit_per_class=None):
    t0 = time.time()
    cerr = md5f(CERT)
    aerr = md5f(ANCHOR)
    if cerr != CERT_MD5_EXPECT or aerr != ANCHOR_MD5_EXPECT:
        log("WARN upstream md5 differs from the registered value: cert=%s(expected %s) anchor=%s(expected %s)"
            % (cerr, CERT_MD5_EXPECT, aerr, ANCHOR_MD5_EXPECT))
    cert = json.load(open(CERT, encoding="utf-8"))
    hp = cert["hard_problems"]
    corpus = {}
    for l in open(PB, encoding="utf-8"):
        if l.strip():
            it = json.loads(l)
            if it.get("layer") == "L1" and it.get("status") == "ok":
                corpus[it["name"]] = it
    zero_by_cls = collections.Counter()
    withref_by_cls = collections.Counter()
    items = []
    reasons_hist = collections.defaultdict(collections.Counter)
    for p in hp:
        arms = p["arms"]
        if arms["RB0"] or arms["RBx"] or arms["RBs"]:
            continue
        zero_by_cls[p["cls"]] += 1
        for a in ("RB0", "RBx", "RBs"):
            reasons_hist[p["cls"]][p.get("%s_why" % a.lower()) or "?"] += 1
        it = corpus.get(p["name"])
        if it is None:
            continue
        withref_by_cls[p["cls"]] += 1
        items.append({"name": p["name"], "cls": p["cls"], "family": p["family"],
                      "n_steps": p["n_steps"], "chain_ops": p["chain_ops"],
                      "arms": {a: arms[a] for a in ("RB0", "RBx", "RBs", "RBplay")},
                      "arm_why": {a: p.get("%s_why" % a.lower()) for a in ("RB0", "RBx", "RBs")},
                      "corpus_id": it["id"], "text_md5": it["text_md5"],
                      "stmt_md5": it["stmt_md5"], "presentation": it["presentation"],
                      "stmt_state": it["stmt_state"], "head": it["head"], "concl": it["concl"],
                      "proof_text": it["proof_text"]})
    if limit_per_class:
        cut = []
        for c in CLASSES:
            cut += [x for x in items if x["cls"] == c][:limit_per_class]
        items = cut
    # ---- Rule-arm side texts (recovered by device execution; pr = purely deterministic replay of
    # F.make_problem with the same seed as the certificate)
    chains = {c: load_chains(c) for c in CLASSES}
    rule_recs = []
    arm_hist = collections.Counter()
    regen_mismatch = []
    for it in items:
        seq = int(it["name"].rsplit("_", 1)[-1])
        pr, err = FAM.make_problem(it["cls"], it["family"], seq, MASTER_SEED)
        ch = chains[it["cls"]].get(it["name"]) or {}
        if pr is None:
            regen_mismatch.append({"name": it["name"], "err": str(err)})
            it["rule_side"] = {"status": "regen_fail:%s" % err}
            continue
        if pr.get("stmt") != ch.get("stmt"):
            regen_mismatch.append({"name": it["name"], "err": "stmt_mismatch",
                                   "regen": (pr.get("stmt") or "")[:120],
                                   "chain": (ch.get("stmt") or "")[:120]})
        rs = {}
        for arm in ("RB0", "RBx", "RBs"):
            ok, why, prog, text, n_call = arm_text(it["cls"], pr, arm)
            inline, err2 = inline_of(text)
            rs[arm] = {"solved": bool(ok), "why": why, "n_prog": len(prog or []),
                       "n_calls": n_call, "text_md5": hashlib.md5(
                           (inline if inline else text).encode("utf-8")).hexdigest(),
                       "presentation": ("inline_rewrite" if inline else "chain_text(%s)" % err2),
                       "has_text": bool(text.strip())}
            arm_hist["%s|%s|%s" % (it["cls"], arm, ("solved" if ok else why.split(":")[0]))] += 1
            rule_recs.append({"id": "%s|%s" % (it["corpus_id"], arm),
                              "name": it["name"], "cls": it["cls"], "arm": arm,
                              "solved": bool(ok), "why": why, "text": inline if inline else text,
                              "presentation": rs[arm]["presentation"],
                              "text_md5": rs[arm]["text_md5"]})
        it["rule_side"] = rs
    # ---- 3 real (miniF2F) problems (ANCHOR_V2 ineq anchor set, reference column)
    anc = json.load(open(ANCHOR, encoding="utf-8"))
    zt = []
    for x in anc["classes"]["ineq"]:
        zt.append({"name": x["name"], "cls": "ineq", "head": x["head"], "concl": x["concl"],
                   "tag": x.get("tag"), "stmt_hash": x.get("stmt_hash"),
                   "rulearm_certified": (x.get("rulearm") or {}).get("certified"),
                   "rulearm_reason": (x.get("rulearm") or {}).get("reason"),
                   "device_passthrough": x.get("device_passthrough"),
                   "sample_limit": "Valid(256) 桥可解析池 11 题; 规则臂过滤后锚定 v2 = 0/3/0 (ind/ineq/nt)",
                   "source": "ANCHOR_V2.json classes.ineq (md5 %s)" % aerr})
    meta = {"schema": "ccp-a-exam-build/1", "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "cert": {"path": CERT, "md5": cerr, "expected": CERT_MD5_EXPECT},
            "anchor": {"path": ANCHOR, "md5": aerr, "expected": ANCHOR_MD5_EXPECT},
            "criterion": "主判据集 = hard 层 RB0 ∧ RBx ∧ RBs 全零解出 (三臂证书 BH_HARDGATE_CERT_FULL)",
            "n_hard": len(hp), "n_zero_by_cls": dict(zero_by_cls),
            "n_zero_with_reference_text_by_cls": dict(withref_by_cls),
            "n_main_items": len(items), "n_zhenti_ref": len(zt),
            "arm_failure_reason_hist": {c: dict(reasons_hist[c]) for c in reasons_hist},
            "rule_arm_status_hist": dict(arm_hist),
            "regen": {"master_seed": MASTER_SEED, "method": "F.make_problem 纯确定性重放 (同证书 seed)",
                      "n_mismatch": len(regen_mismatch), "mismatch_sample": regen_mismatch[:10]},
            "items": items, "zhenti_ref": zt, "wall_s": round(time.time() - t0, 2)}
    json.dump(meta, open(PEB, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    with open(PET, "w", encoding="utf-8") as f:
        for r in rule_recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log("STAGE1 zero=%s withref=%s main=%d zhenti=%d | rule texts=%d | %s md5=%s wall=%.1fs"
        % (dict(zero_by_cls), dict(withref_by_cls), len(items), len(zt), len(rule_recs),
           PEB, md5f(PEB), meta["wall_s"]))
    print(json.dumps({"n_zero_by_cls": dict(zero_by_cls),
                      "n_zero_with_reference_text_by_cls": dict(withref_by_cls),
                      "arm_failure_reason_hist": {c: dict(reasons_hist[c]) for c in reasons_hist},
                      "rule_arm_status_hist": dict(arm_hist)}, ensure_ascii=False, indent=1))


def inline_of(text):
    sys.path.insert(0, G23)
    import gen23_corpus as CO
    return CO.rewrite_inline(text)


def stage2(chunk=300):
    """Rule-arm texts go through the same frozen judge module (batch executor; deduplicated by text md5
    before judging, results broadcast back to every row)."""
    recs = [json.loads(l) for l in open(PET, encoding="utf-8") if l.strip()]
    uniq = {}
    for r in recs:
        t = r.get("text") or ""
        if not t.strip():
            continue
        h = hashlib.md5(t.encode("utf-8")).hexdigest()
        if h not in uniq:
            uniq[h] = {"id": "ruletxt|%s" % h, "name": "rt_%s" % h[:12], "head": None,
                       "concl": None, "proof_text": t, "text_md5": h,
                       "srcs": [{"name": r["name"], "arm": r["arm"], "cls": r["cls"]}]}
        else:
            uniq[h]["srcs"].append({"name": r["name"], "arm": r["arm"], "cls": r["cls"]})
    jobs = []
    peb_items = {x["name"]: x for x in json.load(open(PEB, encoding="utf-8"))["items"]}
    for h, u in uniq.items():
        src = u["srcs"][0]
        it = peb_items.get(src["name"])
        if not it:
            continue
        jobs.append({"id": u["id"], "name": "rt_%s" % h[:12], "head": it["head"],
                     "concl": it["concl"], "proof_text": u["proof_text"], "text_md5": h,
                     "srcs": u["srcs"], "cls": src["cls"]})
    log("STAGE2 unique rule-arm texts %d (raw %d) chunk=%d" % (len(jobs), len(recs), chunk))
    res = PBATCH.judge_items_batch(jobs, os.path.join(LEANWD, "exam_rule"), chunk=chunk,
                                   tag="examrule")
    by_md5 = {}
    for r, j in zip([j for j in jobs], res):
        by_md5[j["id"].split("|", 1)[1]] = {"certified": j["certified"], "reasons": j["reasons"],
                                            "lean": j["lean"], "n1n2": j["n1n2"],
                                            "axioms": j.get("axioms"),
                                            "n_srcs": len(r["srcs"]),
                                            "srcs_sample": r["srcs"][:6]}
    out = {"schema": "ccp-a-exam-rulejudge/1", "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "judge_md5": md5f(DC.__file__), "n_texts": len(jobs),
           "n_certified": sum(1 for v in by_md5.values() if v["certified"]),
           "by_md5": by_md5}
    json.dump(out, open(PEJ, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    log("STAGE2 DONE texts=%d certified=%d -> %s md5=%s"
        % (len(jobs), out["n_certified"], PEJ, md5f(PEJ)))


def stage3():
    peb = json.load(open(PEB, encoding="utf-8"))
    pej = json.load(open(PEJ, encoding="utf-8")) if os.path.exists(PEJ) else {"by_md5": {}}
    trace = {}
    for l in open(PT, encoding="utf-8"):
        if l.strip():
            r = json.loads(l)
            trace[r["id"]] = r
    corpus = {}
    if os.path.exists(PC):
        for l in open(PC, encoding="utf-8"):
            if l.strip():
                c = json.loads(l)
                corpus[c["id"]] = c
    by_hm = pej.get("by_md5") or {}
    items_out = []
    ps_gt = collections.Counter()
    ps_rule = collections.Counter()
    flagged = []
    for it in peb["items"]:
        tr = trace.get(it["corpus_id"]) or {}
        gt_ps = {"certified": tr.get("certified"), "reasons": tr.get("reasons"),
                 "runner": tr.get("runner"), "judge_md5": tr.get("judge_md5")}
        ps_gt["%s|%s" % (it["cls"], "cert" if tr.get("certified") else
                         ("rej:" + (tr.get("reasons") or ["?"])[0]))] += 1
        rs = it.get("rule_side") or {}
        rs_out = {}
        for arm in ("RB0", "RBx", "RBs"):
            v = (rs.get(arm) or {}) if isinstance(rs, dict) else {}
            if not v:
                rs_out[arm] = {"status": "no_record"}
                continue
            h = v.get("text_md5")
            j = by_hm.get(h) or {}
            cert = j.get("certified")
            rs_out[arm] = {"solved": v.get("solved"), "why": v.get("why"),
                           "n_prog": v.get("n_prog"), "has_text": v.get("has_text"),
                           "text_md5": h, "presentation": v.get("presentation"),
                           "judged": (h in by_hm), "certified": cert,
                           "reasons": j.get("reasons")}
            ps_rule["%s|%s|%s" % (it["cls"], arm, "cert" if cert else
                                  ("notjudged" if v.get("has_text") else "notext"))] += 1
            if cert:
                flagged.append({"name": it["name"], "arm": arm, "text_md5": h})
        items_out.append({
            "id": it["corpus_id"], "name": it["name"], "cls": it["cls"], "family": it["family"],
            "n_steps": it["n_steps"], "chain_ops": it["chain_ops"],
            "why_included": "hard 层三臂 (RB0/RBx/RBs) 全零解出 (证书 BH_HARDGATE_CERT_FULL); "
                            "且参考证明文本在册 (语料 L1, 预筛通过=%s)" % gt_ps["certified"],
            "three_arm_certificate": {"source": "BH_HARDGATE_CERT_FULL.json (md5 %s)" % peb["cert"]["md5"],
                                      "arms": it["arms"], "why": it["arm_why"]},
            "reference_text": {"text_md5": it["text_md5"], "presentation": it["presentation"],
                               "stmt_md5": it["stmt_md5"]},
            "prescreen_gt": gt_ps, "gt_prescreen_pass": bool(tr.get("certified")),
            "training_exclusion": {
                "corpus_id": it["corpus_id"],
                "corpus_split": (corpus.get(it["corpus_id"]) or {}).get("split"),
                "eligible_train": (corpus.get(it["corpus_id"]) or {}).get("eligible_train"),
                "in_train_set": bool((corpus.get(it["corpus_id"]) or {}).get("eligible_train")),
                "field": "A_CORPUS_V1.jsonl.split (exam_excluded) / eligible_train=false"},
            "rule_side_prescreen": rs_out})
    # Real (miniF2F) reference column
    zt = []
    for x in peb["zhenti_ref"]:
        zt.append({**x, "why_included": "真题锚定列 (参考): Valid(256) 桥可解析 ∧ 规则臂不可解; "
                                        "n=3 < 30 ⇒ sample-limited, 不作主判据",
                   "prescreen_gt": {"certified": None, "note": "装置无该题正解文本 (真题非课程生成); "
                                                               "模型侧文本将在 A3 运行时经同一 judge_attempt 判定"},
                   "prescreen_rule": {"certified": x["rulearm_certified"],
                                      "reason": x["rulearm_reason"]}})
    # Disclosure column: nt (RBs=1.000 solves everything) and the ineq problems only RBs solves
    cert = json.load(open(CERT, encoding="utf-8"))
    disc = collections.Counter()
    for p in cert["hard_problems"]:
        a = p["arms"]
        if p["cls"] == "nt":
            disc["nt|RBs_solved"] += 1 if a["RBs"] else 0
            disc["nt|total"] += 1
        elif p["cls"] == "ineq" and a["RBs"] and not a["RB0"] and not a["RBx"]:
            disc["ineq|RBs_solved_only"] += 1
        elif p["cls"] == "ind" and a["RBs"]:
            disc["ind|RBs_solved"] += 1
    ex = {"schema": "ccp-a-exam-v1/1", "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
          "kind": "组合卷 (P3 裁定; 变更 71)",
          "criterion": "M1_A[类] = certified_model_A>0 ∧ certified_model_A>certified_rule ∧ "
                       "敏感性门 PASS ∧ 装置可判性门 PASS; 0 vs 0 禁读作能力结论",
          "sources": {"hardgate_cert": {"path": CERT, "md5": peb["cert"]["md5"]},
                      "anchor_v2": {"path": ANCHOR, "md5": peb["anchor"]["md5"]},
                      "corpus_build": {"path": PB, "md5": md5f(PB)},
                      "corpus_prescreen_trace": {"path": PT, "md5": md5f(PT)},
                      "rule_judge": {"path": PEJ, "md5": md5f(PEJ) if os.path.exists(PEJ) else None},
                      "judge_md5": md5f(DC.__file__),
                      "judge_entry": "drct_common.judge_attempt + N1/N2 (冻结判件, P7 零改动)"},
          "parts": {"main_criteria": {"n_by_cls": dict(collections.Counter(x["cls"] for x in items_out)),
                                      "n_by_cls_gt_prescreened": dict(collections.Counter(
                                          x["cls"] for x in items_out if x["gt_prescreen_pass"])),
                                      "target_per_class": 30,
                                      "target_met": {c: (sum(1 for x in items_out if x["cls"] == c) >= 30)
                                                     for c in CLASSES},
                                      "clean_subset_note": "gt_prescreen_pass=True 的子集 (参考文本自身过 N1/N2); "
                                                           "主判据集含 GT 预筛未过者 55 条 (其参考文本被判 N1 装饰), "
                                                           "读数时两者并列 (主集 / 净集)",
                                      "items": items_out},
                    "zhenti_reference": {"n": len(zt), "sample_limited": True, "items": zt},
                    "disclosure": {"nt_hard_layer": "RBs 解出 1176/1176 ⇒ 全部降披露列 (不进主判据集)",
                                   "counts": dict(disc)}},
          "n_zero_by_cls": peb["n_zero_by_cls"],
          "n_zero_with_reference_text_by_cls": peb["n_zero_with_reference_text_by_cls"],
          "arm_failure_reason_hist": peb["arm_failure_reason_hist"],
          "prescreen": {"gt_side_hist": dict(ps_gt), "rule_side_hist": dict(ps_rule),
                        "rule_texts_judged": pej.get("n_texts"),
                        "rule_texts_certified": pej.get("n_certified"),
                        "flagged_false_achieve": flagged,
                        "term": "双侧同门: GT/正解侧取自 T1 全量预筛留痕; 规则臂侧文本经装置执行取回后"
                                "转内联形, 过同一批量判卷器 (judge_md5 双侧一致)"},
          "compliance": {"test243": "未读未评未引用 (题源=合成课程 + Valid(256) 锚定子集)",
                         "probes": "未读", "tmp": "无 /tmp 写入", "sudo": "无",
                         "gpu": "零 GPU (仅 CPU Lean, nice -n 19)",
                         "write_face": "scratch/ccp/gen23/**"},
          "training_exclusion": {
              "policy": "主代 CRIT 修正令: EXAM_V1 任何部分的题在**语料层**硬排除 (split=exam_excluded), "
                        "训练集 split=='train' 且 eligible_train=true (单一事实源 = A_CORPUS_V1.jsonl)",
              "corpus_stats_md5": md5f(PS) if os.path.exists(PS) else None,
              "leak_gate": "见 out/A_EXAM_LEAK_CHECK.json (n_exam_in_train 必须 = 0)"},
          "wall_note": "stage1 零 Lean; stage2 批量判卷 (冻结判件三件套语义)"}
    json.dump(ex, open(PEX, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    log("STAGE3 EXAM_V1 %s md5=%s main=%d by_cls=%s zhenti=%d flagged=%d"
        % (PEX, md5f(PEX), len(items_out),
           dict(collections.Counter(x["cls"] for x in items_out)), len(zt), len(flagged)))


def verify():
    """Hard gate: the corpus train set and the exam set are **disjoint** (recomputable; checked by the
    main agent)."""
    corpus = {}
    for l in open(PC, encoding="utf-8"):
        if l.strip():
            c = json.loads(l)
            corpus[c["id"]] = c
    peb = json.load(open(PEB, encoding="utf-8"))
    main = {x["name"] for x in peb["items"]}
    zhenti = {x["name"] for x in peb["zhenti_ref"]}
    cert = json.load(open(CERT, encoding="utf-8"))
    nt_hard = {p["name"] for p in cert["hard_problems"] if p["cls"] == "nt"}
    ineq_rbs = {p["name"] for p in cert["hard_problems"] if p["cls"] == "ineq"
                and p["arms"]["RBs"] and not p["arms"]["RB0"] and not p["arms"]["RBx"]}
    train = {c["name"] for c in corpus.values() if c.get("eligible_train")}
    held = {c["name"] for c in corpus.values() if c.get("split") == "synth_held"}
    excl = {c["name"] for c in corpus.values() if c.get("split") == "exam_excluded"}
    disc = {c["name"] for c in corpus.values() if c.get("split") == "exam_disclosure"}
    all_exam = main | zhenti | nt_hard | ineq_rbs
    res = {"schema": "ccp-a-exam-leak-check/1",
           "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "n_exam_main": len(main), "n_zhenti": len(zhenti),
           "n_disclosure_nt_hard": len(nt_hard), "n_disclosure_ineq_rbs_only": len(ineq_rbs),
           "n_corpus_train": len(train), "n_corpus_synth_held": len(held),
           "n_corpus_exam_excluded": len(excl), "n_corpus_exam_disclosure": len(disc),
           "n_exam_names_in_corpus_train": len((main | zhenti | nt_hard | ineq_rbs) & train),
           "n_main_in_train": len(main & train), "n_main_in_held": len(main & held),
           "n_main_in_excluded": len(main & excl),
           "n_nt_hard_in_train": len(nt_hard & train),
           "n_ineq_rbs_in_train": len(ineq_rbs & train),
           "leak_names_sample": sorted(list(((main | zhenti | nt_hard | ineq_rbs) & train)))[:20],
           "GATE": "PASS" if not ((main | zhenti | nt_hard | ineq_rbs) & train) else "FAIL",
           "repro_cmd": "cd ${SCOPE_RESEARCH_ROOT:-.}/scratch/ccp/gen23 && "
                        "nice -n 19 ${SCOPE_PYTHON:-python3} gen23_exam.py "
                        "--phase verify",
           "corpus_md5": md5f(PC), "exam_md5": md5f(PEX) if os.path.exists(PEX) else None,
           "note": "train 集 = 语料 eligible_train=true (split=='train'); 考题 = 主判据集 ∪ 真题 ∪ "
                   "披露列 (nt hard / ineq RBs-only)"}
    out = os.path.join(OUT, "A_EXAM_LEAK_CHECK.json")
    json.dump(res, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    res["file_md5"] = md5f(out)
    log("VERIFY GATE=%s | exam_main=%d in_train=%d | corpus_train=%d excl=%d disc=%d | %s"
        % (res["GATE"], len(main), res["n_exam_names_in_corpus_train"], len(train), len(excl),
           len(disc), out))
    print(json.dumps({k: res[k] for k in ("GATE", "n_exam_main", "n_corpus_train",
                                          "n_corpus_synth_held", "n_corpus_exam_excluded",
                                          "n_corpus_exam_disclosure",
                                          "n_exam_names_in_corpus_train")}, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True,
                    choices=("stage1", "stage2", "stage3", "verify", "all"))
    ap.add_argument("--chunk", type=int, default=300)
    ap.add_argument("--limit-per-class", type=int, default=None)
    a = ap.parse_args()
    if a.phase in ("stage1", "all"):
        stage1(limit_per_class=a.limit_per_class)
    if a.phase in ("stage2", "all"):
        stage2(chunk=a.chunk)
    if a.phase in ("stage3", "all"):
        stage3()
    if a.phase in ("verify", "all"):
        verify()


if __name__ == "__main__":
    main()

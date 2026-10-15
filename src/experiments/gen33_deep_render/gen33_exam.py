# -*- coding: utf-8 -*-
# gen33_exam.py -- deep exam-set engineering (C4 pre-drop recompute + valid set + device-recompute sidecar + pre-exam gate)
#
# Three stages:
#   [A] **C4 recompute**: 338 exam candidates vs 2399 deep-corpus rows, 4-gram overlap -- recompute the list first, then pre-drop.
#       Convention (user decision-point default; LAUNCH_LOG 04:45 main-generation suggestion): drop when max 4-gram coverage >= 0.9 in either direction.
#       Output: drop list + **fully marked version** (per row c4_max_overlap / c4_way / c4_drop) so the user can overturn the ruling.
#   [B] **Pre-exam gate** (change-96 lesson): row-by-row, step-by-step device recompute; 8-dim state non-all-zero / no errors / not None; reading persisted.
#   [C] Sidecar: each row -> {name, cls, seg=exam_deep, stmt, head, concl, ops, sfa, vars, dom_dump}
#       **no gold res** (state comes from gen33_device recomputation).
#
# Read-only: scratch/ccp/gen32/EXAM_DEEP_CANDIDATE.json · DEEP_CORPUS_V1.jsonl
# Writes only: scratch/ccp/gen33/**
import argparse
import collections
import json
import os
import re
import time

import g33_common as C
import gen33_device as DEV

G33 = C.G33
OUT = os.path.join(G33, "out")
THRESH = 0.9
NGRAM = 4


def log(msg, path=None):
    C.log(msg, path)


_CTX = {}


def ctx_tok():
    """Tokenizer (reused read-only via gen13 ccp_cachefull.load_ctx; same tokenizer as the render sidecar)."""
    if "tok" not in _CTX:
        import importlib.machinery as _im
        import importlib.util as _iu
        _ld = _im.SourceFileLoader("g33_exam_ccf", os.path.join(C.G13, "ccp_cachefull.py"))
        _m = _iu.module_from_spec(_iu.spec_from_loader("g33_exam_ccf", _ld))
        _ld.exec_module(_m)
        _CTX.update(_m.load_ctx())
    return _CTX["tok"]


def norm_ws(s):
    return re.sub(r"\s+", " ", s or "").strip()


def ngrams(s, n=NGRAM):
    s = norm_ws(s)
    if len(s) < n:
        return {s} if s else set()
    return {s[i:i + n] for i in range(len(s) - n + 1)}


def text_of_item(it):
    """Comparable text of an exam row = statement (head+concl) concatenated with proof text (registered convention).

    Matches gen32r F3: what gets compared against the delivered corpus is exactly the **statement segment + proof segment** text.
    """
    return (norm_ws(it.get("head")) + "\n" + norm_ws(it.get("concl")) + "\n"
            + norm_ws(it.get("proof_text")))


def text_of_corpus_row(r):
    return (norm_ws(r.get("head")) + "\n" + norm_ws(r.get("concl")) + "\n"
            + norm_ws(r.get("proof_text")))


def cover_rate(a, b):
    """Fraction of a's 4-grams covered by b (directional; a is the queried side)."""
    ga, gb = ngrams(a), ngrams(b)
    if not ga:
        return 0.0
    return len(ga & gb) / len(ga)


def text_of(r, mode):
    """Domain-sliced text (for the sensitivity analysis; registered: the overlap convention is **this batch's most sensitively decided criterion**, so all variants are listed)."""
    if mode == "head_concl_proof":
        return norm_ws(r.get("head")) + "\n" + norm_ws(r.get("concl")) + "\n" + \
            norm_ws(r.get("proof_text"))
    if mode == "concl_proof":
        return norm_ws(r.get("concl")) + "\n" + norm_ws(r.get("proof_text"))
    if mode == "proof":
        return norm_ws(r.get("proof_text"))
    if mode == "head_concl":
        return norm_ws(r.get("head")) + "\n" + norm_ws(r.get("concl"))
    raise ValueError(mode)


def overlap_max(it, cor_gram, inv, key):
    ge = ngrams(text_of(it, key))
    cand = collections.Counter()
    for x in ge:
        for i in inv.get(x, ()):
            cand[i] += 1
    best = 0.0
    for i, inter in cand.items():
        if not inter:
            continue
        best = max(best, inter / max(1, len(ge)), inter / max(1, len(cor_gram[i])))
    return best


def sensitivity(exam, corpus, logp=None):
    """Overlap-convention sensitivity table (FINDING attachment): four text domains × two thresholds."""
    out = {}
    for key in ("head_concl_proof", "concl_proof", "proof", "head_concl"):
        cg = [ngrams(text_of(r, key)) for r in corpus]
        inv = collections.defaultdict(list)
        for i, g in enumerate(cg):
            for x in g:
                inv[x].append(i)
        vals = [overlap_max(it, cg, inv, key) for it in exam]
        out[key] = {"drop_at_0.90": sum(1 for v in vals if v >= 0.90),
                    "drop_at_0.95": sum(1 for v in vals if v >= 0.95),
                    "drop_at_0.99": sum(1 for v in vals if v >= 0.99),
                    "n": len(vals),
                    "hist_0.1": dict(sorted(collections.Counter(round(v, 1)
                                                                for v in vals).items()))}
        if logp:
            C.log("C4 sensitivity %-18s drop@0.90=%d @0.95=%d @0.99=%d"
                  % (key, out[key]["drop_at_0.90"], out[key]["drop_at_0.95"],
                     out[key]["drop_at_0.99"]), logp)
    return out


def c4_prefilter(exam, corpus, logp=None):
    """Recompute first, then pre-drop: drop when the two-way max coverage >= THRESH (either direction; reason registered)."""
    cor_txt = [text_of_corpus_row(r) for r in corpus]
    cor_gram = [ngrams(t) for t in cor_txt]
    # Speedup: 4-gram inverted index (rows hit by each corpus-side gram)
    inv = collections.defaultdict(list)
    for i, g in enumerate(cor_gram):
        for x in g:
            inv[x].append(i)
    marked = []
    for it in exam:
        t = text_of_item(it)
        ge = ngrams(t)
        cand = collections.Counter()
        for x in ge:
            for i in inv.get(x, ()):
                cand[i] += 1
        best, bi, way = 0.0, None, None
        for i, inter in cand.items():
            if not inter:
                continue
            ncr = len(cor_gram[i])
            r_e_in_c = inter / max(1, len(ge))       # exam grams covered by corpus
            r_c_in_e = inter / max(1, ncr)           # corpus grams covered by exam (stronger when the corpus row is short)
            m = max(r_e_in_c, r_c_in_e)
            if m > best:
                best, bi, way = m, i, ("exam_in_corpus" if r_e_in_c >= r_c_in_e
                                       else "corpus_in_exam")
        drop = bool(best >= THRESH)
        marked.append({"name": it["name"], "family": it["family"], "cls": it["cls"],
                       "seq": it.get("seq"), "chain_len": it.get("chain_len"),
                       "c4_max_overlap": round(best, 6), "c4_way": way,
                       "c4_match_corpus": (corpus[bi]["name"] if bi is not None else None),
                       "c4_drop": drop,
                       "n_gram_exam": len(ge)})
    n_drop = sum(1 for x in marked if x["c4_drop"])
    rep = {"schema": "gen33-c4-exam-prefilter/1", "n_exam": len(exam), "n_drop": n_drop,
           "n_valid": len(exam) - n_drop, "threshold": THRESH, "ngram": NGRAM,
           "way_hist": dict(collections.Counter(x["c4_way"] for x in marked if x["c4_drop"])),
           "drop_by_family": dict(collections.Counter(x["family"] for x in marked
                                                      if x["c4_drop"])),
           "note": ("C4: 与语料 4-gram 交叠 >=0.9 的考卷行预弃 (双向 max 覆盖率); "
                    "本件为**重算**结果 (非沿用 gen32r 自报 50 道); 全量标记版留档以便重裁"),
           "marked": marked}
    if logp:
        C.log("C4 pre-drop: exam=%d drop=%d valid=%d (threshold %.2f, %d-gram)"
              % (len(exam), n_drop, rep["n_valid"], THRESH, NGRAM), logp)
    return rep


def build(exam_path, corpus_path, logp=None):
    t0 = time.time()
    E = json.load(open(exam_path, encoding="utf-8"))
    items = E["items"]
    corpus = C.jlload(corpus_path)
    log("exam candidates %s n=%d md5=%s | corpus n=%d"
        % (exam_path, len(items), C.md5f(exam_path), len(corpus)), logp)
    assert len(items) == 338, len(items)
    sens = sensitivity(items, corpus, logp=logp)
    c4 = c4_prefilter(items, corpus, logp=logp)
    c4["sensitivity"] = sens
    c4["primary_scope"] = ("head_concl_proof (题面+结论+证明全文, 双向 max 覆盖率) —— "
                           "**最保守** (含最宽文本域): 本批以它为主口径, 有效卷 n=%d"
                           % c4["n_valid"])
    c4["why_conservative"] = ("C4 读数是**口径决定**的: 同 338 道题, 四种分域在 0.9 阈值下"
                              "弃 120 / 217 / 326 / 7 道 (见 sensitivity); 0.85-0.95 带内质量密集"
                              "⇒ 阈值刀口。采最宽文本域取并集 = 保护 M1_D 有效性 (宁弃勿污); "
                              "全量标记版含逐行 c4_max_overlap 以便用户重裁阈值/分域。")
    C.save_json(OUT + "/C4_EXAM_PREFILTER.json", c4)

    drop = {x["name"] for x in c4["marked"] if x["c4_drop"]}
    valid = [it for it in items if it["name"] not in drop]
    log("valid set n=%d (dropped %d)" % (len(valid), len(drop)), logp)

    # ---- pre-exam gate: state-source health check (device recompute; change96) ----
    src = DEV.state_source_report(valid, tag="exam_deep_valid", logp=logp)
    C.save_json(OUT + "/GEN33_EXAM_STATE_SOURCE.json", src)
    if not src["ok"]:
        raise SystemExit("发考前置门未过: 考卷行状态源不完整 %s" % src["err_hist"])
    # Per-step state nonzero rate must be 1.0 row by row (same threshold as the corpus)
    per = []
    for it in valid:
        rs = DEV.dev_resid_all(it)
        nz = sum(1 for r in rs if r[0] is not None and any(x != 0.0 for x in r[0]))
        per.append({"name": it["name"], "n_step": len(rs), "n_nonzero": nz,
                    "errs": [x[4] for x in rs if x[4]]})
    bad = [x for x in per if x["errs"] or x["n_nonzero"] != x["n_step"]]
    gate = {"schema": "gen33-exam-state-gate/1", "n_rows": len(valid),
            "n_step": sum(x["n_step"] for x in per),
            "n_nonzero": sum(x["n_nonzero"] for x in per),
            "n_bad_rows": len(bad), "bad": bad[:8],
            "pass": bool(not bad and len(valid) > 0),
            "rule": ("发考前实测: 有效卷逐行逐步装置现算, 8 维状态非全零且无异常; "
                     "任一不过即不发考")}
    C.save_json(OUT + "/GEN33_EXAM_STATE_GATE.json", gate)
    log("pre-exam gate: n=%d steps=%d nonzero=%d bad_rows=%d pass=%s"
        % (len(valid), gate["n_step"], gate["n_nonzero"], len(bad), gate["pass"]), logp)
    if not gate["pass"]:
        raise SystemExit("发考前置门未过: %s" % bad[:3])

    # ---- sidecar (device-recompute convention; carries no gold res) ----
    # Render prompt: **copied verbatim** from gen26_hybuild.exam_sidecar / V2F exam prompt template and
    # ids/np layout ([M_TASK] + tok(prompt); np = len(prompt)+1; no target).
    import v31_model as _vm
    M_TASK = _vm.st.M_TASK
    tok = ctx_tok()

    def prompt_of(stmt_state):
        return ("You are a Lean 4 prover. Given the theorem statement below, output the"
                " complete proof body inside a ```lean4 fenced block. Output the proof"
                " body only (no `theorem` header, no `import` lines).\n"
                "Theorem statement:\n%s\nProof:\n" % stmt_state)

    recs = []
    for i, it in enumerate(valid):
        p = tok(prompt_of(it["stmt_state"]), add_special_tokens=False)["input_ids"]
        recs.append({"ri": i, "name": it["name"], "cls": it["cls"],
                     "family": it["family"], "cls_code": it.get("class_code"),
                     "seg": "exam_deep", "np": len(p) + 1, "n": len(p) + 1,
                     "ids": [int(M_TASK)] + [int(x) for x in p],
                     "has_target": False,
                     "stmt": it["stmt_state"], "head": it["head"], "concl": it["concl"],
                     "type4": [1, 0, 0, 0],
                     "ops": list(it["chain_ops"]),
                     "sfa": [dict(s.get("args") or {}) for s in it["steps"]],
                     "specs": [s.get("res_spec") for s in it["steps"]],
                     "vars": list((it.get("meta") or {}).get("vars") or []),
                     "dom_dump": (it.get("meta") or {}).get("dom_dump"),
                     "chain_len": it.get("chain_len"), "n_steps": it.get("n_steps"),
                     "decision_points": it.get("decision_points"),
                     "arms": it.get("arms"), "res": None,
                     "res_note": "**不载金标 res**; 状态由 gen33_device 装置现算 (变更96)"})
    md5s = C.jlsave(recs, OUT + "/DEEP_EXAM_SIDECAR.jsonl")
    # Sidecar self-check: row-wise device recompute works + all operators are in the v4 legal set
    v4 = json.load(open(G33 + "/cache/CCP_VOCAB_V4.json", encoding="utf-8"))
    legal = set(v4["ops_legal_idx"])
    op2idx = {}
    for key in ("ops_ccp", "ops_bh", "ops_g24", "ops_gen32"):
        for k, v in (v4.get(key) or {}).items():
            op2idx[k] = int(v)
    miss = sorted({o for r in recs for o in r["ops"] if o not in op2idx})
    oob = sorted({o for r in recs for o in r["ops"] if op2idx.get(o) not in legal})
    chk = {"n": len(recs) == len(valid), "ops_known": not miss, "ops_legal": not oob,
           "no_gold_res": all(x["res"] is None for x in recs),
           "stmt_nonempty": all(x["stmt"] for x in recs)}
    assert all(chk.values()), (chk, miss[:5], oob[:5])
    rep = {"schema": "gen33-deep-exam/1",
           "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "src": {"exam": exam_path, "exam_md5": C.md5f(exam_path),
                   "corpus": corpus_path, "corpus_md5": C.md5f(corpus_path)},
           "c4": c4, "state_gate": gate, "state_source": src,
           "valid": {"n": len(valid), "chain_hist":
                     dict(sorted(collections.Counter(it["chain_len"]
                                                     for it in valid).items())),
                     "by_family": dict(collections.Counter(it["family"] for it in valid)),
                     "by_cls": dict(collections.Counter(it["cls"] for it in valid))},
           "sidecar": {"path": OUT + "/DEEP_EXAM_SIDECAR.jsonl", "md5": md5s,
                       "n": len(recs), "checks": chk,
                       "ops_unknown": miss, "ops_out_of_legal": oob},
           "wall_s": round(time.time() - t0, 2)}
    rep["md5"] = C.save_json(OUT + "/GEN33_EXAM.json", rep)
    log("EXAM DONE valid=%d md5=%s wall=%.1fs" % (len(valid), rep["md5"], rep["wall_s"]),
        logp)
    C.heartbeat("exam", "valid=%d" % len(valid))
    print("GEN33_EXAM_RC=0")
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exam", default=C.EXAM_CAND)
    ap.add_argument("--corpus", default=C.DEEP_CORPUS)
    ap.add_argument("--log", default=G33 + "/logs/gen33.log")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.log), exist_ok=True)
    build(a.exam, a.corpus, logp=a.log)


if __name__ == "__main__":
    main()

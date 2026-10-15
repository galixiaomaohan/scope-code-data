# -*- coding: utf-8 -*-
# gen23_corpus.py -- gen23 (A1 data batch) T1: build A corpus v1 + full N1/N2 pre-screening + axiom allowlist
#
# Basis (each item traceable):
#   * A_DEVICE_DESIGN.md (md5 2bd76424..., incl. §11 post-review corrections) §2.1 self-consistency
#     gate / §2.2 four-layer material / §2.3 viable stripping / §2.4 inline re-render / §2.5
#     pre-registered dedup / §6.5 held-out rule / §9 pre-registration / §11.3 five pre-screening
#     reinforcements
#   * GEN21_AUDIT.md (md5 4b58f521...) §3 ruling two (pre-screening necessary but not sufficient) /
#     T4 measurement convention (dedup 4622->3126, A-convention tokens)
#   * Judge-module freeze order (change 71 P7): scratch/drct/drct_common.py zero modification
#     (md5 7c31b58d...); all judging goes through the existing judge_attempt + N1/N2 -- this script
#     only imports, never copies or rewrites judging semantics.
#
# Phases (--phase):
#   build      : build candidate corpus (L0/L1/L2 render + viable stripping + pre-registered dedup
#                + held-out split) -> out/A_CORPUS_BUILD.jsonl
#   prescreen  : per-item judge_attempt (same frozen judge module) -> out/A_PRESCREEN_TRACE.jsonl
#                (append, resumable)
#   merge      : build + prescreen -> out/A_CORPUS_V1.jsonl (final corpus, with pre-screen verdicts)
#   axioms     : static tactic/axiom allowlist scan + batch Lean #print axioms -> out/A_AXIOM_WHITELIST.json
#   stats      : summary -> out/A_CORPUS_V1_STATS.json (real-tokenizer tokens / class distribution /
#                dedup before-after / pre-screen rejection breakdown)
#
# Write surface: scratch/ccp/gen23/** only. Zero GPU; all Lean runs at nice -n 19 with concurrency
# <= 5 (yielding to in-flight gen22/gen24 batches).
import argparse
import collections
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

RES = os.environ.get("SCOPE_RESEARCH_ROOT", ".")
CCP = os.path.join(RES, "scratch", "ccp")
G23 = os.path.join(CCP, "gen23")
OUT = os.path.join(G23, "out")
LEANWD = os.path.join(G23, "lean")
os.makedirs(OUT, exist_ok=True)
os.makedirs(LEANWD, exist_ok=True)
for _p in (os.path.join(RES, "scratch", "drct"), os.path.join(CCP, "gen17"), CCP,
           os.path.join(CCP, "gen13"), os.path.join(CCP, "gen16"),
           os.path.join(CCP, "bh"),
           os.environ.get("SCOPE_SUPEROPERATOR", os.path.join(RES, os.pardir, "device"))):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import drct_common as DC                                             # noqa: E402
import ccp_plan_render as PR                                         # noqa: E402
import ccp_bridge as CB                                              # noqa: E402
SPLIT = CB.split_concl                          # accepts both state form and theorem form (ccp_bridge)

CLASSES = ("ind", "ineq", "nt")
SCHEMA_BUILD = "ccp-a-corpus-build/1"
SCHEMA_TRACE = "ccp-a-prescreen-trace/1"
SCHEMA_CORPUS = "ccp-a-corpus-v1/1"
SCHEMA_AX = "ccp-a-axiom-whitelist/1"
SCHEMA_STATS = "ccp-a-corpus-stats/1"
TOKENIZER = os.path.join(RES, "models", "SmolLM2-135M")
PB = os.path.join(OUT, "A_CORPUS_BUILD.jsonl")
PT = os.path.join(OUT, "A_PRESCREEN_TRACE.jsonl")
PC = os.path.join(OUT, "A_CORPUS_V1.jsonl")
PA = os.path.join(OUT, "A_AXIOM_WHITELIST.json")
PS = os.path.join(OUT, "A_CORPUS_V1_STATS.json")
LOG = os.path.join(G23, "logs", "gen23_corpus.log")
os.makedirs(os.path.dirname(LOG), exist_ok=True)
_LK = threading.Lock()
WS = re.compile(r"\s+")

# ---------------------------------------------------------------- training protocol prefix (recorded, §2.1)
PROMPT_PREFIX = ("You are a Lean 4 prover. Given the theorem statement below, output the complete "
                 "proof body inside a ```lean4 fenced block. Output the proof body only "
                 "(no `theorem` header, no `import` lines).\nTheorem statement:\n")
PROMPT_SUFFIX = "\nProof:\n"
FENCE_OPEN = "```lean4\n"
FENCE_CLOSE = "\n```"


def prompt_of(stmt_state):
    return PROMPT_PREFIX + stmt_state + PROMPT_SUFFIX


def target_of(proof_text):
    return FENCE_OPEN + proof_text + FENCE_CLOSE


# ---------------------------------------------------------------- utilities
def md5f(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def md5s(s):
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def log(msg):
    line = "[%s] %s" % (time.strftime("%Y-%m-%dT%H:%M:%S%z"), msg)
    with _LK:
        print(line, flush=True)
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def norm(s):
    return WS.sub(" ", str(s or "")).strip()


def jdump(obj, p, indent=1):
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)
    return md5f(p)


def load_rows(path):
    seen = {}
    for ln in open(path, encoding="utf-8"):
        if ln.strip():
            r = json.loads(ln)
            seen.setdefault(r["ccp_name"], r)
    return seen


def load_chains(path):
    out = {}
    for ln in open(path, encoding="utf-8"):
        if ln.strip():
            c = json.loads(ln)
            out.setdefault(c["name"], c)
    return out


# ---------------------------------------------------------------- inline re-render transform (byte-identical to gen21_bh_ineq_fix.rewrite_v2, validated 0/6->6/6 and 698/698 for the whole family)
RE_HAVE_SQ = re.compile(r"^have\s+\w+\s*:\s*\(0:ℚ\)\s*≤\s*(\d+)\s*\*\s*(\(.+\))\s*\^\s*2\s*:=\s*by positivity\s*$")
RE_HAVE_CONST = re.compile(r"^have\s+\w+\s*:\s*\(0:ℚ\)\s*≤\s*\((\d+)\s*:\s*ℚ\)\s*:=\s*by norm_num\s*$")
RE_HAVE_SQPOW = re.compile(r"^have\s+\w+\s*:\s*\(0:ℚ\)\s*≤\s*\((\d+)\s*:\s*ℚ\)\s*\^\s*2\s*:=\s*sq_nonneg\s*\((\d+\s*:\s*ℚ)\)\s*$")
RE_CLOSER = re.compile(r"^nlinarith\s*\[([^\]]*)\]\s*$")


def rewrite_inline(proof_text):
    """Reuse the validated gen21 transform (same source): drop the have lines and inline the
    evidence into the nlinarith brackets; on failure returns (None, reason)."""
    lines = [ln for ln in proof_text.split("\n") if ln.strip()]
    if not lines:
        return None, "empty"
    m = RE_CLOSER.match(lines[-1].strip())
    if not m:
        return None, "no_bracket_closer"
    names = [s.strip() for s in m.group(1).split(",") if s.strip()]
    hmap = {}
    for ln in lines[:-1]:
        s = ln.strip()
        mm = re.match(r"^have\s+(\w+)\s*:", s)
        hname = mm.group(1) if mm else None
        for pat, fn in ((RE_HAVE_SQ, lambda r: ("sq_nonneg %s" % r.group(2) if r.group(1) == "1"
                                                else "mul_nonneg (by norm_num : (0:ℚ) ≤ %s) (sq_nonneg %s)"
                                                % (r.group(1), r.group(2)))),
                        (RE_HAVE_CONST, lambda r: "(by norm_num : (0:ℚ) ≤ %s)" % r.group(1)),
                        (RE_HAVE_SQPOW, lambda r: "sq_nonneg (%s)" % r.group(2))):
            r = pat.match(s)
            if r:
                hmap[hname] = fn(r)
                break
        else:
            return None, "unparsed_have:%s" % s[:60]
    if any(n not in hmap for n in names):
        return None, "unresolved_name"
    return ("nlinarith [%s]" % ", ".join(hmap[n] for n in names)), None


# ---------------------------------------------------------------- phase build
def build():
    t0 = time.time()
    items = []
    src_stats = {}
    stmt_consistency = []
    for src, base, pre, layer in (("gen12", os.path.join(CCP, "gen"), "ccp_%s", "L0"),
                                  ("gen14", os.path.join(CCP, "bh", "gen"), "bh_%s", "L1")):
        for cls in CLASSES:
            rp = os.path.join(base, pre % cls + "_rows.jsonl")
            cp = os.path.join(base, pre % cls + "_chains.jsonl")
            rows, chains = load_rows(rp), load_chains(cp)
            n_elig = 0
            for name, r in rows.items():
                if src == "gen12":
                    if not (r.get("accepted") and not r.get("readout_only")):
                        continue
                    if not PR.label_shape_ok(cls, r):
                        continue
                    if r.get("ccp_rowkind") != "pos" or name not in chains:
                        continue
                else:
                    if r.get("bh_layer") != "hard" or r.get("ccp_rowkind") != "pos":
                        continue
                    if name not in chains:
                        continue
                n_elig += 1
                ch = chains[name]
                if src == "gen12":
                    item = PR.item_of_row(cls, r)
                    head, concl = item["head"], item["concl"]
                    stmt_state = norm(r.get("stmt") or "")
                    plan, err = PR.gt_plan(cls, r, concl)
                    if err:
                        items.append({"id": "%s|%s" % (src, name), "src": src, "layer": layer,
                                      "cls": cls, "name": name, "status": "gt_plan_err",
                                      "err": err, "family": r.get("ccp_family")})
                        continue
                    res = PR.render_plan(cls, name, head, concl, PR.stmt_theorem(head, concl), plan)
                    if res["refused"]:
                        items.append({"id": "%s|%s" % (src, name), "src": src, "layer": layer,
                                      "cls": cls, "name": name, "status": "render_refused",
                                      "err": res["refused"], "family": r.get("ccp_family")})
                        continue
                    text = "\n".join(res["lines"])
                    pres = "gen12_render_inline"
                else:
                    stmt_raw = ch.get("stmt") or ""
                    hdr, concl = SPLIT(stmt_raw)                    # hdr includes "theorem <name> <binders>"
                    head = hdr + " :"
                    row_stmt = norm(r.get("stmt") or "")
                    bip = hdr
                    if bip.startswith("theorem "):
                        parts = bip.split(" ", 2)
                        bip = parts[2] if len(parts) > 2 else ""
                    recon = norm("%s ⊢ %s" % (bip, concl))
                    stmt_state = row_stmt
                    stmt_consistency.append({"id": "%s|%s" % (src, name),
                                             "ok": (row_stmt == recon),
                                             "row": row_stmt[:160], "chain_recon": recon[:160]})
                    text = "\n".join(ch.get("proof") or [])
                    new, err = rewrite_inline(text)
                    if new:
                        text, pres = new, "gen14_inline_rewrite_v2"
                    else:
                        pres = "gen14_chain_text(%s)" % err
                items.append({
                    "id": "%s|%s" % (src, name), "src": src, "layer": layer, "cls": cls,
                    "name": name, "family": r.get("ccp_family"), "rowkind": "pos",
                    "status": "ok", "head": head, "concl": concl, "stmt_state": stmt_state,
                    "proof_text": text, "n_lines": len([x for x in text.split("\n") if x.strip()]),
                    "presentation": pres, "text_md5": md5s(text), "stmt_md5": md5s(stmt_state),
                    "chain_ops": list(r.get("chain_ops") or []), "target_arm": "proof",
                    "sf_n": len(r.get("sf_args") or []),
                    "src_row_md5": md5s(json.dumps(r, ensure_ascii=False, default=str)),
                    "src_chain_md5": md5s(json.dumps(ch, ensure_ascii=False, default=str)),
                })
            src_stats["%s_%s" % (src, cls)] = {"rows": len(rows), "chains": len(chains), "n_eligible": n_elig,
                                               "rows_md5": md5f(rp), "chains_md5": md5f(cp), "path_rows": rp}
            log("BUILD %s %s eligible=%d rows=%d" % (src, cls, n_elig, len(rows)))
    # ---- L2 negatives: gen12 rowkind == neg (endfail disabled, per the §11.5 erratum)
    l2 = []
    for cls in CLASSES:
        rp = os.path.join(CCP, "gen", "ccp_%s_rows.jsonl" % cls)
        cp = os.path.join(CCP, "gen", "ccp_%s_chains.jsonl" % cls)
        rows, chains = load_rows(rp), load_chains(cp)
        n = 0
        for name, r in rows.items():
            if r.get("ccp_rowkind") != "neg":
                continue
            ch = chains.get(name) or {}
            text = "\n".join(ch.get("proof") or [])
            hdr, concl = SPLIT(r.get("stmt") or "")
            head = "theorem %s %s :" % (name, hdr)
            l2.append({"id": "gen12|%s" % name, "src": "gen12", "layer": "L2", "cls": cls,
                       "name": name, "family": r.get("ccp_family"), "rowkind": "neg",
                       "status": "ok", "head": head, "concl": concl,
                       "stmt_state": norm(r.get("stmt") or ""), "proof_text": text,
                       "n_lines": len([x for x in text.split("\n") if x.strip()]),
                       "presentation": "gen12_neg_chain_text",
                       "text_md5": md5s(text), "stmt_md5": md5s(norm(r.get("stmt") or "")),
                       "chain_ops": list(r.get("chain_ops") or []), "target_arm": "stop_refuse",
                       "fail_op": (r.get("program") or [None])[-1],
                       "src_row_md5": md5s(json.dumps(r, ensure_ascii=False, default=str)),
                       "src_chain_md5": md5s(json.dumps(ch, ensure_ascii=False, default=str))})
            n += 1
        src_stats["gen12_%s_neg" % cls] = {"n": n}
        log("BUILD L2 neg %s n=%d" % (cls, n))
    items += l2

    # ---- pre-registered dedup (§2.5): R1 globally unique after whitespace normalization of the
    # problem statement; R2 4-gram two-way containment >= 0.9
    # information order (who is kept): longer proof -> longer chain -> L1 first -> ascending name
    # (rule fixed before execution)
    def info_key(it):
        return (-len(it.get("proof_text") or ""), -len(it.get("chain_ops") or []),
                0 if it.get("layer") == "L1" else 1, it.get("name") or "")

    order = sorted([it for it in items if it.get("status") == "ok"
                    and it.get("layer") in ("L0", "L1")], key=info_key)
    neg_items = [it for it in items if it.get("status") == "ok" and it.get("layer") == "L2"]
    for it in neg_items:
        it["dedup"] = {"kept": True, "note": "L2 负例不参与正例去重 (独立目标臂)"}
    dropped = []
    kept, seen_stmt = [], {}
    n_uniq_src_cls = collections.Counter()
    for it in order:
        key = norm(it["stmt_state"])
        n_uniq_src_cls["%s|%s|n" % (it["src"], it["cls"])] += 1
        if key in seen_stmt:
            it["dedup"] = {"r1_exact_dup_of": seen_stmt[key]["id"], "kept": False}
            dropped.append(it)
            continue
        seen_stmt[key] = it
        kept.append(it)
        n_uniq_src_cls["%s|%s|u" % (it["src"], it["cls"])] += 1
    n_after_r1 = len(kept)
    log("DEDUP R1: %d -> %d (drop %d)" % (len(order), n_after_r1, len(dropped)))

    def grams(txt, k=4):
        t = norm(txt).split()
        return {tuple(t[i:i + k]) for i in range(max(0, len(t) - k + 1))}

    gm = {}
    idx = collections.defaultdict(list)
    for it in kept:
        g = grams(it["stmt_state"])
        gm[it["id"]] = g
        for x in g:
            idx[x].append(it["id"])
    big = {x: len(v) for x, v in idx.items() if len(v) > 1000}
    cand = collections.Counter()
    for x, ids in idx.items():
        if len(ids) > 1000:                       # ultra-frequent gram not enumerated (recorded in stats)
            continue
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                cand[(ids[i], ids[j])] += 1
    pairs_ge = []
    pairs_ge_min = 0
    for (a, b), n in cand.items():
        ga, gb = gm[a], gm[b]
        if not ga or not gb:
            continue
        ca, cb = n / float(len(ga)), n / float(len(gb))
        if max(ca, cb) >= 0.9:
            pairs_ge.append((a, b, round(ca, 4), round(cb, 4)))
            if min(ca, cb) >= 0.9:
                pairs_ge_min += 1
    pair_by_id = collections.defaultdict(list)
    for a, b, ca, cb in pairs_ge:
        pair_by_id[a].append((b, ca, cb))
        pair_by_id[b].append((a, cb, ca))
    drop2 = set()
    by_id = {it["id"]: it for it in items}
    for it in sorted(kept, key=info_key):
        if it["id"] in drop2:
            continue
        for (o, c1, c2) in pair_by_id.get(it["id"], []):
            if o in drop2 or o == it["id"]:
                continue
            drop2.add(o)
            o2 = by_id.get(o)
            if o2 is not None:
                o2["dedup"] = {"r2_near_dup_of": it["id"], "r2_containment": [c1, c2],
                               "kept": False}
    kept2 = [it for it in kept if it["id"] not in drop2]
    for it in kept2:
        it.setdefault("dedup", {"kept": True})
    log("DEDUP R2: %d -> %d (drop %d) | pairs_ge0.9=%d | skipped_common_grams=%d"
        % (n_after_r1, len(kept2), len(drop2), len(pairs_ge), len(big)))
    # Non-selected items (gt_plan_err / render_refused / neg / dedup-dropped) are all kept in the
    # build file for audit
    rest = [it for it in items if it.get("status") != "ok" or it.get("layer") == "L2"]
    for it in rest:
        if it.get("layer") == "L2":
            continue
        it.setdefault("dedup", {"kept": False, "reason": it.get("status")})
    # ---- held-out split (thermometer set): md5(name)%10==7 (same rule as gen17 §6.5 / bh_course)
    for it in items:
        if it.get("status") == "ok":
            h = int(hashlib.md5(it["name"].encode("utf-8")).hexdigest()[:8], 16) % 10
            it["split"] = ("synth_held" if h == 7 else "train") if it.get("layer") in ("L0", "L1") \
                else "l2_neg"
            it["prompt"] = prompt_of(it["stmt_state"])
            it["target"] = target_of(it["proof_text"])
    with open(PB, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    meta = {"schema": SCHEMA_BUILD, "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "phase": "build", "n_items_total": len(items),
            "n_ok": sum(1 for it in items if it.get("status") == "ok"),
            "n_by_status": dict(collections.Counter(it.get("status") for it in items)),
            "n_by_layer": dict(collections.Counter(it.get("layer") for it in items if it.get("status") == "ok")),
            "n_by_layer_cls": dict(collections.Counter("%s|%s|%s" % (it["layer"], it["cls"], it.get("status"))
                                                       for it in items)),
            "dedup": {"r1_in": len(order), "r1_out": n_after_r1, "r1_dropped": len(dropped),
                      "r2_out": len(kept2), "r2_dropped": len(drop2),
                      "r2_eligible_pairs_ge0.9": len(pairs_ge),
                      "r2_pairs_min_ge0.9_sensitivity": pairs_ge_min,
                      "r1_unique_by_src_cls": dict(n_uniq_src_cls),
                      "r2_common_grams_skipped": len(big),
                      "r2_common_gram_hist_gt1000": {"|".join(k): v for k, v in
                                                     sorted(big.items(), key=lambda kv: -kv[1])[:10]},
                      "rule_r1": "题面白空格归一后全局唯一; 保信息量大者 (证明更长>链更长>L1>name)",
                      "rule_r2": "4-gram 双向包含率 max(|A∩B|/|A|,|A∩B|/|B|) >= 0.9 判近重复; 保信息量大者",
                      "r2_interpretation_note": "设计 §2.5 措辞 '4-gram 双向包含率 < 0.9' 有歧义; 本批采"
                                                " '任一方向 >=0.9 即近重复' (主口径), 并报 '两方向均 >=0.9' "
                                                "的保守敏感性计数 (r2_pairs_min_ge0.9_sensitivity)"},
            "src_stats": src_stats,
            "stmt_state_consistency": {"n_checked": len(stmt_consistency),
                                       "n_mismatch": sum(1 for x in stmt_consistency if not x["ok"]),
                                       "mismatch_sample": [x for x in stmt_consistency
                                                           if not x["ok"]][:10],
                                       "note": "gen14 题面 = 行 stmt (状态形, 与审计口径 R1 一致); "
                                               "与链 stmt 反推状态形逐条对拍"},
            "judge_md5": {"drct_common": md5f(DC.__file__), "ccp_plan_render": md5f(PR.__file__),
                          "ccp_bridge": md5f(CB.__file__)},
            "build_md5_self": None}
    meta["build_md5_self"] = md5f(PB)
    jdump(meta, os.path.join(OUT, "A_CORPUS_BUILD_META.json"))
    log("BUILD DONE items=%d ok=%d | %s md5=%s | wall=%.1fs"
        % (len(items), meta["n_ok"], PB, meta["build_md5_self"], time.time() - t0))


# ---------------------------------------------------------------- phase recut (main-agent CRIT correction order: exam problems moved out of the training set)
def recut():
    """Hard-exclude any EXAM_V1 problem **at the corpus level** (split = exam_excluded /
    exam_disclosure) to prevent train-exam leakage. Zero Lean; only split/eligible_train change;
    texts and pre-screening are untouched."""
    t0 = time.time()
    cert = json.load(open(os.path.join(CCP, "bh", "out", "BH_HARDGATE_CERT_FULL.json"),
                          encoding="utf-8"))
    exam_build = os.path.join(OUT, "A_EXAM_BUILD.json")
    peb = json.load(open(exam_build, encoding="utf-8"))
    main_names = set(x["name"] for x in peb["items"])
    nt_hard = set(p["name"] for p in cert["hard_problems"] if p["cls"] == "nt")
    ineq_rbs_only = set(p["name"] for p in cert["hard_problems"]
                        if p["cls"] == "ineq" and p["arms"]["RBs"]
                        and not p["arms"]["RB0"] and not p["arms"]["RBx"])
    zhenti = set(x["name"] for x in peb["zhenti_ref"])
    before_md5 = md5f(PB)
    items = [json.loads(l) for l in open(PB, encoding="utf-8") if l.strip()]
    hist = collections.Counter()
    for it in items:
        nm = it.get("name")
        role = None
        if nm in main_names:
            role, sp = "main_criteria", "exam_excluded"
        elif nm in zhenti:
            role, sp = "zhenti_ref", "exam_excluded"
        elif nm in nt_hard or nm in ineq_rbs_only:
            role, sp = "disclosure_layer", "exam_disclosure"
        if role:
            it["exam_role"] = role
            it["split"] = sp
            it["exam_exclusion"] = {
                "reason": "主代 CRIT 修正令: EXAM_V1 任何部分的题移出训练集 (防训练-考试泄漏)",
                "source": "BH_HARDGATE_CERT_FULL.json (md5 %s) + A_EXAM_BUILD.json" % md5f(
                    os.path.join(CCP, "bh", "out", "BH_HARDGATE_CERT_FULL.json")),
                "role": role}
        hist["%s|%s" % (it.get("layer"), it.get("split"))] += 1
    with open(PB, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    meta_p = os.path.join(OUT, "A_CORPUS_BUILD_META.json")
    meta = json.load(open(meta_p, encoding="utf-8"))
    meta.setdefault("recut", []).append(
        {"ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "why": "主代 CRIT 修正令 (考题移出训练集)",
         "before_md5": before_md5, "after_md5": md5f(PB),
         "n_main_criteria_names": len(main_names), "n_nt_hard_names": len(nt_hard),
         "n_ineq_rbs_only_names": len(ineq_rbs_only), "n_zhenti": len(zhenti),
         "split_hist": dict(hist)})
    meta["build_md5_self"] = md5f(PB)
    jdump(meta, meta_p)
    log("RECUT before=%s after=%s | %s | wall=%.1fs"
        % (before_md5, meta["recut"][-1]["after_md5"], dict(hist), time.time() - t0))


# ---------------------------------------------------------------- phase prescreen
def prescreen(workers=5, limit=None, freeze_md5=None):
    t0 = time.time()
    items = [json.loads(l) for l in open(PB, encoding="utf-8") if l.strip()]
    items = [it for it in items if it.get("status") == "ok"]
    done = set()
    if os.path.exists(PT):
        for l in open(PT, encoding="utf-8"):
            if l.strip():
                try:
                    done.add(json.loads(l)["id"])
                except Exception:                                     # noqa: BLE001
                    pass
    todo = [it for it in items if it["id"] not in done]
    if limit:
        todo = todo[:limit]
    log("PRESCREEN todo=%d (done=%d total=%d) workers=%d" % (len(todo), len(done), len(items), workers))
    judge_md5 = md5f(DC.__file__)
    if freeze_md5 and judge_md5 != freeze_md5:
        log("PRESCREEN ABORT: judge md5 drift %s != %s" % (judge_md5, freeze_md5))
        raise SystemExit(3)
    wl = threading.Lock()

    def one(it):
        wd = os.path.join(LEANWD, "corpus")
        os.makedirs(wd, exist_ok=True)
        t = time.time()
        try:
            j = DC.judge_attempt(it["name"], it["head"], it["concl"], it["proof_text"], wd,
                                 "corpus_%s" % it["layer"])
            rec = {"certified": bool(j.get("certified")), "reasons": j.get("reasons") or [],
                   "lean": j.get("lean"), "n1n2": j.get("n1n2")}
        except Exception as e:                                        # noqa: BLE001
            rec = {"certified": False, "reasons": ["judge_exc:%s:%s" % (type(e).__name__, e)],
                   "lean": None, "n1n2": None}
        rec["wall_s"] = round(time.time() - t, 2)
        out = {"schema": SCHEMA_TRACE, "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
               "id": it["id"], "name": it["name"], "cls": it["cls"], "src": it["src"],
               "layer": it["layer"], "rowkind": it["rowkind"], "target_arm": it["target_arm"],
               "presentation": it["presentation"], "text_md5": it["text_md5"],
               "stmt_md5": it["stmt_md5"], "judge_md5": judge_md5,
               "judge_entry": "drct_common.judge_attempt + N1/N2 (冻结判卷件, 零改动)"}
        out.update(rec)
        return out

    n = 0
    with open(PT, "a", encoding="utf-8") as f:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(one, it): it for it in todo}
            for fu in as_completed(futs):
                rec = fu.result()
                with wl:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    f.flush()
                    n += 1
                    if n % 25 == 0:
                        log("PRESCREEN %d/%d cert=%s %s %s" % (n, len(todo), rec["certified"],
                                                               rec["id"], rec["reasons"]))
    log("PRESCREEN DONE n=%d wall=%.1fs trace=%s md5=%s" % (n, time.time() - t0, PT, md5f(PT)))


# ---------------------------------------------------------------- phase merge
def merge():
    items = [json.loads(l) for l in open(PB, encoding="utf-8") if l.strip()]
    tr = {}
    for l in open(PT, encoding="utf-8"):
        if l.strip():
            r = json.loads(l)
            tr[r["id"]] = r
    n_ok = n_cert = 0
    out = []
    for it in items:
        if it.get("status") != "ok":
            continue
        n_ok += 1
        r = tr.get(it["id"]) or {}
        rec = dict(it)
        rec["prescreen"] = {"certified": bool(r.get("certified")), "reasons": r.get("reasons") or [],
                            "lean_rc": (r.get("lean") or {}).get("rc"),
                            "lean_accepted": (r.get("lean") or {}).get("accepted"),
                            "sorry": (r.get("lean") or {}).get("sorry"),
                            "n1_vacuous": (r.get("n1n2") or {}).get("n1_vacuous"),
                            "n1_accepted": (r.get("n1n2") or {}).get("n1_accepted"),
                            "n2_accepted": (r.get("n1n2") or {}).get("n2_accepted"),
                            "judge_md5": r.get("judge_md5")}
        ok_base = bool(rec.get("dedup", {}).get("kept")) and bool(r.get("certified")) and \
            rec.get("layer") in ("L0", "L1")
        # eligible = usable corpus (dedup-kept and pre-screen certified and split in
        # {train, synth_held}); eligible_train = the subset with split=="train" => SFT training
        # stream (hard gate: exam problems are split=exam_excluded/exam_disclosure)
        rec["eligible"] = bool(ok_base and rec.get("split") in ("train", "synth_held"))
        rec["eligible_train"] = bool(ok_base and rec.get("split") == "train")
        rec["in_train_set"] = rec["eligible_train"]
        if rec.get("layer") == "L2":
            # negative usability = **pre-screen rejection** (the text really does not close);
            # a neg row label does not imply non-closure (same source as the endfail erratum)
            rec["l2_usable"] = (not bool(r.get("certified")))
        if r.get("certified"):
            n_cert += 1
        out.append(rec)
    with open(PC, "w", encoding="utf-8") as f:
        for rec in out:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    log("MERGE n_ok=%d n_certified=%d eligible_train=%d -> %s md5=%s"
        % (n_ok, n_cert, sum(1 for r in out if r["eligible_train"]), PC, md5f(PC)))


# ---------------------------------------------------------------- phase axioms
# Static allowlist: tactic heads actually present in the corpus (data-driven freeze) + banned-token blocklist
BAN_TOKENS = ("sorry", "admit", "native_decide", "axiom ", "unsafe", "Lean.trustCompiler",
              "by_contra!", "Classical.choice ", "exact?")
AXIOM_WHITELIST = ("propext", "Classical.choice", "Quot.sound")


def batch_lean(path, timeout=3600.0):
    """**Evidence-gathering** Lean run with the same env/cwd as DC.run_lean (captures the full
    stdout; judge module untouched)."""
    env = dict(os.environ)
    env["PATH"] = DC._ELAN_BIN + os.pathsep + env.get("PATH", "")
    t0 = time.time()
    try:
        cp = subprocess.run(["nice", "-n", "19", "lake", "env", "lean", path],
                            capture_output=True, text=True, timeout=timeout, cwd=DC.MINIF2F, env=env)
        out = (cp.stdout or "") + (cp.stderr or "")
        return {"rc": cp.returncode, "out": out, "secs": round(time.time() - t0, 2)}
    except subprocess.TimeoutExpired:
        return {"rc": 124, "out": "LEAN_TIMEOUT", "secs": round(time.time() - t0, 2)}


def axioms(chunk=200):
    t0 = time.time()
    items = [json.loads(l) for l in open(PB, encoding="utf-8") if l.strip()]
    items = [it for it in items if it.get("status") == "ok"]
    heads = collections.Counter()
    static_ban = []
    for it in items:
        for ln in it["proof_text"].split("\n"):
            s = ln.strip()
            if not s:
                continue
            heads[s.split()[0]] += 1
        low = it["proof_text"]
        hit = [b for b in BAN_TOKENS if b in low]
        if hit:
            static_ban.append({"id": it["id"], "hits": hit})
    whitelist = sorted(heads)
    non_wl = [it["id"] for it in items
              if any(ln.strip() and ln.strip().split()[0] not in whitelist
                     for ln in it["proof_text"].split("\n"))]
    # ---- batch Lean #print axioms (same as judge-module render_lean: same official header + same declaration shape)
    ax = {}
    chunks = [items[i:i + chunk] for i in range(0, len(items), chunk)]
    wd = os.path.join(LEANWD, "axioms")
    os.makedirs(wd, exist_ok=True)
    for ci, ch in enumerate(chunks):
        p = os.path.join(wd, "ax_chunk_%03d.lean" % ci)
        with open(p, "w", encoding="utf-8") as f:
            for i, it in enumerate(ch):
                blk = DC.render_lean(it["name"], it["head"], it["concl"], it["proof_text"])
                if i > 0:                      # strip the repeated import header after the first block (Lean forbids mid-file import)
                    blk = "\n".join(blk.split("\n")[3:])
                f.write(blk)
                f.write("\n")
        r = batch_lean(p)
        # parse: 'name' depends on axioms: [a, b] / 'name' does not depend on any axioms
        for m in re.finditer(r"'([A-Za-z0-9_.]+)' (?:depends on axioms: \[([^\]]*)\]"
                             r"|does not depend on any axioms)", r["out"]):
            nm, axs = m.group(1), m.group(2)
            ax[nm] = [] if axs is None else [x.strip() for x in axs.split(",")]
        log("AXIOMS chunk %d/%d rc=%s secs=%s parsed=%d" % (ci + 1, len(chunks), r["rc"],
                                                            r["secs"], len(ax)))
    bad = {it["id"]: ax.get(it["name"]) for it in items
           if ax.get(it["name"]) is None or any(a not in AXIOM_WHITELIST for a in ax[it["name"]])}
    # cross-check against the axioms field in the pre-screen trace (two independent paths through the same judge module)
    trace_ax, ax_mismatch = {}, []
    if os.path.exists(PT):
        for l in open(PT, encoding="utf-8"):
            if l.strip():
                r = json.loads(l)
                trace_ax[r["id"]] = r.get("axioms")
        for it in items:
            if it["id"] in trace_ax and sorted(trace_ax[it["id"]] or []) != sorted(ax.get(it["name"]) or []):
                ax_mismatch.append({"id": it["id"], "trace": trace_ax[it["id"]],
                                    "axioms_phase": ax.get(it["name"])})
    out = {"schema": SCHEMA_AX, "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "n_items": len(items), "n_chunks": len(chunks), "chunk_size": chunk,
           "tactic_head_hist": dict(heads.most_common()),
           "tactic_whitelist_frozen": whitelist,
           "n_tactic_head_outside_whitelist": len(non_wl),
           "static_ban_tokens": list(BAN_TOKENS), "n_static_ban_hits": len(static_ban),
           "static_ban_records": static_ban[:50],
           "axiom_whitelist": list(AXIOM_WHITELIST),
           "n_axioms_parsed": len(ax),
           "n_axiom_violations": len(bad), "axiom_violations_sample": dict(list(bad.items())[:50]),
           "n_trace_crosschecked": len([it for it in items if it["id"] in trace_ax]),
           "n_trace_axiom_mismatch": len(ax_mismatch), "trace_axiom_mismatch_sample": ax_mismatch[:20],
           "axiom_hist": dict(collections.Counter(",".join(sorted(v)) for v in ax.values())),
           "wall_s": round(time.time() - t0, 2)}
    out["file_md5"] = jdump(out, PA)
    log("AXIOMS DONE violations=%d heads=%d %s md5=%s" % (len(bad), len(heads), PA, out["file_md5"]))


# ---------------------------------------------------------------- phase stats
def stats():
    t0 = time.time()
    items = [json.loads(l) for l in open(PC, encoding="utf-8") if l.strip()]
    tk = None
    try:
        from transformers import AutoTokenizer
        tk = AutoTokenizer.from_pretrained(TOKENIZER)
        tok_note = "SmolLM2-135M 真分词器 (models/SmolLM2-135M; add_special_tokens=False)"
    except Exception as e:                                            # noqa: BLE001
        tok_note = "代理口径 char/4 (tokenizer 不可用: %s)" % type(e).__name__

    def nt(s):
        return len(tk(s, add_special_tokens=False)["input_ids"]) if tk else max(1, int(len(s) / 4))

    def pct(v, q):
        if not v:
            return None
        s = sorted(v)
        return s[min(len(s) - 1, int(q * len(s)))]

    out = {"schema": SCHEMA_STATS, "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "tokenizer": tok_note, "classes": {}, "dedup": {}, "prescreen": {},
           "splits": {}, "sources": {}, "wall_s": None,
           "corpus": {"path": PC, "md5": md5f(PC), "n_records": len(items)},
           "build_meta_md5": None}
    tm = os.path.join(OUT, "A_CORPUS_BUILD_META.json")
    if os.path.exists(tm):
        bm = json.load(open(tm, encoding="utf-8"))
        out["build_meta_md5"] = md5f(tm)
        out["dedup"] = bm["dedup"]
        out["sources"] = bm["src_stats"]
        out["n_by_status"] = bm["n_by_status"]
    # per-class tokens/length (eligible_train only)
    tot = collections.Counter()
    for cls in CLASSES:
        sub = [r for r in items if r["cls"] == cls and r.get("eligible_train")]
        pt = [nt(r["prompt"]) for r in sub]
        tt = [nt(r["target"]) for r in sub]
        out["classes"][cls] = {
            "n_eligible_train": len(sub),
            "n_layer_L0": sum(1 for r in sub if r["layer"] == "L0"),
            "n_layer_L1": sum(1 for r in sub if r["layer"] == "L1"),
            "n_split_train": sum(1 for r in sub if r["split"] == "train"),
            "n_split_synth_held": sum(1 for r in sub if r["split"] == "synth_held"),
            "prompt_tokens": {"p50": pct(pt, .5), "p90": pct(pt, .9), "max": max(pt) if pt else None,
                              "sum": sum(pt)},
            "target_tokens": {"p50": pct(tt, .5), "p90": pct(tt, .9), "max": max(tt) if tt else None,
                              "sum": sum(tt)},
            "n_lines": {"p50": pct([r["n_lines"] for r in sub], .5),
                        "max": max([r["n_lines"] for r in sub] or [0])},
        }
        tot["prompt"] += sum(pt)
        tot["target"] += sum(tt)
        tot["n"] += len(sub)
    out["totals"] = {"n_eligible_train": tot["n"], "prompt_tokens_sum": tot["prompt"],
                     "target_tokens_sum": tot["target"],
                     "one_epoch_tokens": tot["prompt"] + tot["target"]}
    # pre-screen breakdown
    ps = collections.Counter()
    rej = collections.Counter()
    for r in items:
        p = r.get("prescreen") or {}
        key = "%s|%s|%s" % (r["layer"], r["cls"], r["rowkind"])
        ps[key + "|cert=%s" % p.get("certified")] += 1
        if not p.get("certified"):
            for x in (p.get("reasons") or ["?"]):
                rej[key + "|" + x.split(":")[0]] += 1
    out["prescreen"] = {"by_class_cert": dict(ps), "reject_reasons": dict(rej),
                        "judge_md5": (items[0].get("prescreen") or {}).get("judge_md5") if items else None}
    for sp in ("train", "synth_held", "l2_neg"):
        out["splits"][sp] = sum(1 for r in items if r.get("split") == sp and r.get("status") == "ok")
    # viable leakage self-check (the training presentation surface must not carry viable/candidate markers)
    leak = 0
    for r in items:
        blob = json.dumps(r, ensure_ascii=False)
        if "viable" in blob or "bh_cands" in blob or "bh_resid" in blob:
            leak += 1
    out["viable_leak_check"] = {"n_records_with_marker": leak,
                               "note": "训练呈现面字段 = id/src/layer/cls/family/rowkind/head/concl/"
                                       "stmt_state/prompt/target/presentation/split/dedup/prescreen"}
    out["axiom_file_md5"] = md5f(PA) if os.path.exists(PA) else None
    out["wall_s"] = round(time.time() - t0, 2)
    out["file_md5"] = jdump(out, PS)
    log("STATS n_eligible=%d epoch_tokens=%d file=%s md5=%s"
        % (out["totals"]["n_eligible_train"], out["totals"]["one_epoch_tokens"], PS, out["file_md5"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True,
                    choices=("build", "recut", "prescreen", "merge", "axioms", "stats", "all"))
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--chunk", type=int, default=200)
    ap.add_argument("--freeze-judge-md5", default="7c31b58da9d7228cee31b9be581fe5dd")
    a = ap.parse_args()
    if a.phase in ("build", "all"):
        build()
    if a.phase == "recut":
        recut()
    if a.phase == "prescreen":
        prescreen(workers=a.workers, limit=a.limit, freeze_md5=a.freeze_judge_md5)
    if a.phase in ("merge", "all"):
        merge()
    if a.phase in ("axioms", "all"):
        axioms(chunk=a.chunk)
    if a.phase in ("stats", "all"):
        stats()


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
# gen44_bgate.py -- shared gate implementation for families B (B1/B2/B6): structural census /
#                  synthetic corpus / stratified check / payload tests / three-channel leakage /
#                  GATE assembly   (writes only scratch/ccp/gen44/**)
#
# Measurement convention:
#   * structural layer = device computation only, zero Lean: parse_stmt_ext -> goal tree ->
#     leaf certificate (F.chain_of)
#   * end-to-end = real LWB rows x exam single rendering (gen44_render_exam_v3) x frozen judging
#     module (nice -19)  => see gen44_barm.py
#   * synthetic corpus = random variable names + random coefficients; **no LWB statement or
#     name may appear** (mechanical assertion)
import argparse
import collections
import glob
import json
import os
import random
import re
import sys
import time

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
if _PKG_DIR not in sys.path:
    sys.path.insert(0, _PKG_DIR)
import gen44_bcore as B                                                  # noqa: E402

OUT = B.OUT
FAM_MODS = {}


def fam_module(fam):
    if fam not in FAM_MODS:
        FAM_MODS[fam] = __import__("gen44_fam_%s" % fam)
    return FAM_MODS[fam]


def G_claim(fam, d):
    try:
        return bool(fam_module(fam).claim(d["tree"]))
    except Exception:                                                   # noqa: BLE001
        return False


# ================================================================ §1 structural census
def census(fam, limit=None, rows=None, lwb=None):
    cen = rows if rows is not None else B.load_census_rows()
    lwb = lwb if lwb is not None else B.load_lwb()
    pop = B.fam_rows(fam, cen)
    if limit:
        pop = pop[:limit]
    st = collections.Counter()
    why = collections.Counter()
    kd = collections.Counter()
    recs = []
    t0 = time.time()
    for i, r in enumerate(pop):
        rec = lwb.get(r["idx"])
        if rec is None:
            st["missing"] += 1
            continue
        row = {"idx": r["idx"], "name": r["name"], "sub": r.get("sub"), "rs": r.get("rs")}
        if len(rec.get("concl") or "") > 2500 or len(rec.get("head") or "") > 2500:
            st["oversize_skipped"] += 1
            row.update({"state": "oversize_skipped"})
            recs.append(row)
            continue
        try:
            with B.time_guard(15.0):
                d = B.parse_stmt_ext(B.P(rec["name"], rec["head"], rec["concl"],
                                         rec.get("subset"), rec["idx"]))
                ok = d["ok"]
                claim = G_claim(fam, d) if ok else False
                cert = B.certifiable(d["tree"]) if (ok and claim) else False
                wy = leaf_whys(d["tree"]) if (ok and claim and not cert) else []
        except B.TimeoutErr:
            st["row_timeout"] += 1
            row.update({"state": "row_timeout"})
            recs.append(row)
            continue
        except Exception as ex:                                         # noqa: BLE001
            st["row_error"] += 1
            row.update({"state": "row_error", "reason": str(ex)[:60]})
            recs.append(row)
            continue
        if not ok:
            k = "parse_" + str(d["reason"]).split(":")[0]
            st[k] += 1
            row.update({"state": "unparsed", "reason": str(d["reason"])[:60]})
        else:
            tr = d["tree"]
            row["kind"] = tr["kind"]
            kd[tr["kind"]] += 1
            if not claim:
                st["parsed_not_claim"] += 1
                row["state"] = "not_claim"
            else:
                st["claim"] += 1
                if cert:
                    st["certified_struct"] += 1
                    row["state"] = "certifiable"
                    row["ops"] = B.tree_ops(tr) + B.tree_leaf_ops(tr)
                    row["chain_len"] = len(B.tree_leaf_ops(tr))
                else:
                    st["claim_uncert"] += 1
                    row["state"] = "claim_uncert"
                    row["why"] = wy[:3]
                    for w in wy:
                        why[w] += 1
        recs.append(row)
        if (i + 1) % 2000 == 0:
            print("[census %s %d/%d] %s %.1fs" % (fam, i + 1, len(pop), dict(st),
                                                  time.time() - t0), flush=True)
    rep = {"schema": "gen44-b-census/1", "fam": fam, "n_pop": len(pop),
           "counts": dict(st), "tree_kinds": dict(kd), "leaf_why": dict(why.most_common(20)),
           "wall_s": round(time.time() - t0, 1)}
    return rep, recs


def leaf_whys(node, out=None):
    out = [] if out is None else out
    k = node.get("kind")
    if k in ("leaf", "sqrt_red"):
        if node.get("got") is None:
            out.append(("sqrt_red:" if k == "sqrt_red" else "leaf:") + str(node.get("why")))
    elif k == "fail":
        out.append("fail:" + str(node.get("reason")))
    for c in node.get("children") or []:
        leaf_whys(c, out)
    return out


# ================================================================ §2 synthetic corpus (+ mechanical assertions)
def gen_corpus(fam, n, seed=20260926):
    mod = fam_module(fam)
    rng = random.Random(seed)
    rows, tries = [], 0
    while len(rows) < n and tries < n * 60:
        tries += 1
        r = mod.corpus_row(rng, len(rows))
        if r is None:
            continue
        d = B.parse_stmt_ext(B.P(r["name"], r["head"], r["concl"], None, None))
        if not d["ok"]:
            continue
        tr = d["tree"]
        if not mod.claim(tr) or not B.certifiable(tr):
            continue
        txt = B.render_tree(tr, "exam")
        if not txt:
            continue
        r["proof"] = "\n".join(txt)
        r["chain_ops"] = B.tree_ops(tr) + B.tree_leaf_ops(tr)
        r["chain_len"] = len(B.tree_leaf_ops(tr))
        r["kind"] = tr["kind"]
        r["frames"] = B.tree_frames(tr)[:2]
        r["const_vec"] = [int(x) for x in re.findall(r"\b\d+\b", r["concl"])][:16]
        r["stmt"] = "%s %s" % (r["head"], r["concl"])
        r["head"] = r["head"].rstrip() + " :"        # LWB head convention (trailing colon; the judging module joins <head> <concl>)
        rows.append(r)
    rep = {"schema": "gen44-b-gen/1", "fam": fam, "n": len(rows), "target": n,
           "tries": tries, "seed": seed,
           "text_hist": dict(collections.Counter(r["proof"] for r in rows).most_common(20)),
           "chainlen_hist": dict(collections.Counter(r["chain_len"] for r in rows)),
           "kind_hist": dict(collections.Counter(r["kind"] for r in rows)),
           "ops_hist": dict(collections.Counter(o for r in rows for o in r["chain_ops"]))}
    return rows, rep


# ================================================================ §3 payload-ness (single-slot damage)
def payload_tests(fam, rows, limit=240):
    """Single-slot chain damage: replace the last leaf operator with a sibling => the rendered
       text must change (n_slot_bad expected 0).
       Numeral-slot damage (meaningful for B6 only): perturb the radicand away from a perfect
       square => the certificate must change or disappear.
       B1/B2 chains are **structural** (tree shape + operator order) => the numeral axis is N/A
       (registered, not a failure)."""
    mod = fam_module(fam)
    slot_n = slot_bad = digit_n = digit_bad = hyp_n = hyp_same = 0
    slot_ex = []
    for r in rows[:limit]:
        dmg = mod.damage_ops(r["chain_ops"])
        if dmg is not None:
            slot_n += 1
            d2 = B.parse_stmt_ext(B.P(r["name"], r["head"], r["concl"], None, None))
            node = _force_ops(d2["tree"], dmg) if d2["ok"] else None
            txt = B.render_tree(node, "exam") if node else None
            if txt is None or "\n".join(txt) == r["proof"]:
                slot_bad += 1
                if len(slot_ex) < 5:
                    slot_ex.append({"name": r["name"], "dmg": dmg})
        r3 = mod.perturb_digit(r)
        if r3 is not None:
            digit_n += 1
            d3 = B.parse_stmt_ext(B.P(r3["name"], r3["head"], r3["concl"], None, None))
            t3 = (B.render_tree(d3["tree"], "exam")
                  if d3["ok"] and B.certifiable(d3["tree"]) else None)
            if t3 is not None and "\n".join(t3) == r["proof"]:
                digit_bad += 1
        # Hypothesis-dependence reading (not a failure gate): drop the first "hypothesis-like"
        # binder group => the certificate should change or disappear.
        hd = _drop_first_hyp(r)
        if hd is not None:
            hyp_n += 1
            d4 = B.parse_stmt_ext(B.P(hd["name"], hd["head"], hd["concl"], None, None))
            t4 = (B.render_tree(d4["tree"], "exam")
                  if d4["ok"] and B.certifiable(d4["tree"]) else None)
            if t4 is not None and "\n".join(t4) == r["proof"]:
                hyp_same += 1
    return {"schema": "gen44-b-payload/1", "fam": fam,
            "n_slot": slot_n, "slot_bad": slot_bad, "slot_examples": slot_ex[:5],
            "n_digit": digit_n, "digit_bad": digit_bad,
            # NOTE: Chinese report payload values kept verbatim (data, not log lines).
            "digit_axis": ("radicand 完全平方扰动 (B6)" if fam == "b6"
                           else "N/A: 链为结构型(树形+算子序), 数字不进编译文本 (登记)"),
            "hyp_drop": {"n": hyp_n, "same_cert": hyp_same,
                         "note": "删首条假设后证书仍逐字相同 => 该证书与假设无关 (读数, 非判废门)"},
            "verdict": "PASS" if (slot_bad == 0 and digit_bad == 0) else "FAIL"}


def _drop_first_hyp(row):
    """Drop the first parenthesized binder group containing a relation symbol (hypothesis-like)
    from head, leaving everything else unchanged."""
    h = row.get("head") or ""
    m = re.match(r"^(theorem\s+[A-Za-z0-9_']+)(.*)$", h)
    if not m:
        return None
    pre, rest = m.group(1), m.group(2)
    for grp in re.finditer(r"\(([^()]*)\)", rest):
        inner = grp.group(1)
        if re.search(r"[≤<>=]", inner) and ":" in inner:
            new_rest = rest[:grp.start()] + rest[grp.end():]
            return {"name": row["name"], "head": pre + new_rest, "concl": row["concl"]}
    return None


def _force_ops(node, ops):
    """Replace the **leaf chain** operators of the tree's first leaf with ops (single-slot damage
    rendering) -- used only by the payload tests.
    Registered device defect D-B1: the first version also pushed tree-level operators (g44_*)
    into the leaf chain, so RE3 returned skip and the test reported a false FAIL; the fix keeps
    only the g42_* operators the leaf renderer can render."""
    leaf_ops = [o for o in ops if o.startswith("g42_")]
    if not leaf_ops:
        return None
    k = node.get("kind")
    if k in ("leaf", "sqrt_red"):
        g = dict(node.get("got") or {})
        g["chain_ops"] = leaf_ops
        nn = dict(node)
        nn["got"] = g
        return nn
    for i, c in enumerate(node.get("children") or []):
        r = _force_ops(c, ops)
        if r is not None:
            nn = dict(node)
            ch = list(node["children"])
            ch[i] = r
            nn["children"] = ch
            return nn
    return None


# ================================================================ §4 stratified check (Lean empirical)
def check_corpus(fam, rows, sample=48, shard=0, nshard=1, maxj=5, seed=20260926):
    """Stratified sampling (strata over the proof text) -> real Lean judging (nice -19 +
    N1/N2)."""
    by_text = collections.defaultdict(list)
    for r in rows:
        by_text[r["proof"]].append(r)
    keys = sorted(by_text)
    rng = random.Random(seed)
    per = max(1, sample // max(1, len(keys)))
    sel = []
    for k in keys:
        g = by_text[k]
        rng.shuffle(g)
        sel += g[:per]
    if len(sel) > sample:                       # D-B6: >=1 row per stratum blows the total up => cap at sample
        rng.shuffle(sel)
        sel = sel[:sample]
    sel = sel[shard::nshard]
    wd = os.path.join(B.G44, "lean", "bchk_%s" % fam)
    res = []
    t0 = time.time()
    for i, r in enumerate(sel):
        jr = B.judge_nice(r["name"], r["head"], r["concl"], r["proof"], wd,
                          "bc%s" % fam, timeout=300.0, maxj=maxj)
        res.append({"name": r["name"], "proof_md5": B.md5s(r["proof"]),
                    "certified": bool(jr["certified"]), "reasons": jr["reasons"],
                    "lean": jr["lean"], "n1n2": jr["n1n2"]})
        if (i + 1) % 5 == 0:
            print("[chk %s %d/%d] cert=%d %.0fs" % (
                fam, i + 1, len(sel), sum(1 for x in res if x["certified"]),
                time.time() - t0), flush=True)
    nc = sum(1 for x in res if x["certified"])
    n2ok = sum(1 for x in res if (x.get("n1n2") or {}).get("n2_accepted"))
    n1v = sum(1 for x in res if (x.get("n1n2") or {}).get("n1_vacuous"))
    return {"schema": "gen44-b-check/1", "fam": fam, "n_sample": len(sel),
            "n_certified": nc, "rate": round(nc / max(1, len(sel)), 4),
            "wilson95": B.wilson(nc, len(sel)), "n2_accepted": n2ok, "n1_vacuous": n1v,
            "strat_keys": len(keys), "rows": res,
            "reasons_hist": dict(collections.Counter(
                r["reasons"][0] if r["reasons"] else "ok" for r in res)),
            "wall_s": round(time.time() - t0, 1)}


# ================================================================ §5 three-channel leakage
# Gate functions: norm_statement/toks/grams/gram_min_cover/const_fp are copied verbatim from
# scratch/ccp/gen42/gen42_leak.py (itself copied verbatim from gen41_build.py; parent unchanged)
# so the gate semantics stay comparable.
# Tightened (from gen42 D13 / this batch's ERRATUM): (1) 4-gram uses **token 4-gram +
#   min-cover>=0.9** (not any character overlap); (2) a constant vector counts as a hit only when
#   distinct>=5 and total>=8; (3) screening uses the union of all corpus grams (fast), while the
#   hit criterion is still re-checked problem by problem.
CONST_RE = re.compile(r"(?<![A-Za-z0-9_'.])\d+(?:\s*/\s*\d+)?(?![A-Za-z0-9_'.])")


def norm_statement(s):
    return re.sub(r"\s+", " ", s).strip()


def toks(s):
    return re.findall(r"[A-Za-z0-9_]+|[^\sA-Za-z0-9_]", s)


def grams(s, k=4):
    t = toks(s)
    return {tuple(t[i:i + k]) for i in range(max(0, len(t) - k + 1))}


def gram_min_cover(a, b):
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / min(len(a), len(b))


def const_fp(s):
    c = collections.Counter()
    for m in CONST_RE.findall(s):
        v = m.replace(" ", "")
        try:
            from fractions import Fraction
            c[str(Fraction(v))] += 1
        except Exception:                                               # noqa: BLE001
            pass
    return c


def leak(fam, corpus, pool_limit=None):
    """Corpus x full LWB: P1 name / P2 normalized statement / P3 head segment / G2b token
       4-gram (min-cover>=0.9) / G3c constant vector (tightened: distinct>=5 and total>=8) plus
       a positive control."""
    lwb = B.load_lwb()
    idxs = sorted(lwb)
    if pool_limit:
        idxs = idxs[:pool_limit]

    def stmt_of(rec):
        return norm_statement((rec.get("head") or "") + " " + (rec.get("concl") or ""))

    names, stmts, heads = set(), set(), set()
    U = set()
    n_pool = 0
    for i in idxs:
        r = lwb[i]
        nm, st = r.get("name"), stmt_of(r)
        names.add(nm)
        stmts.add(st)
        heads.add(norm_statement((r.get("head") or "").split(":", 1)[0]))
        for g in grams(st):
            U.add(hash(g))
        n_pool += 1
    hits = {"P1_name": [], "P2_stmt": [], "P3_head": [], "G2b_4gram": [],
            "G3c_const_vec": []}
    n_screen = 0
    cand_idx = []
    for r in corpus:
        st = stmt_of(r)
        if r.get("name") in names:
            hits["P1_name"].append(r["name"])
        if st in stmts:
            hits["P2_stmt"].append(r["name"])
        if norm_statement((r.get("head") or "").split(":", 1)[0]) in heads:
            hits["P3_head"].append(r["name"])
        g = grams(st)
        if g and len([1 for x in g if hash(x) in U]) / len(g) >= 0.9:
            n_screen += 1
            cand_idx.append((r["name"], g))
        cs = set(const_fp(st))
        if len(cs) >= 5 and sum(const_fp(st).values()) >= 8:
            hits["G3c_const_vec"].append({"name": r["name"], "consts": sorted(cs)[:10]})
    # Per-problem re-check (candidate rows x the whole pool, min-cover>=0.9 problem by problem).
    for nm, g in cand_idx:
        for i in idxs:
            mg = grams(stmt_of(lwb[i]))
            if gram_min_cover(g, mg) >= 0.9:
                hits["G2b_4gram"].append([nm, lwb[i].get("name")])
                break
    pos_ok = False
    if corpus and idxs:
        tgt = lwb[idxs[0]]
        ts = stmt_of(tgt)
        pos_ok = (ts in stmts) and any(
            gram_min_cover(grams(ts), grams(stmt_of(lwb[i]))) >= 0.9 for i in idxs[:50])
    total = sum(len(v) for v in hits.values())
    return {"schema": "gen44-b-leak/1", "fam": fam, "tag": "gen44_%s_corpus" % fam,
            "n_corpus": len(corpus), "n_pool": n_pool,
            "hits": {k: len(v) for k, v in hits.items()},
            "hit_examples": {k: (v[:6] if k != "G2b_4gram" else v[:3])
                             for k, v in hits.items()},
            "n_screen_candidates": n_screen,
            "positive_control_pass": bool(pos_ok), "gate_total_hits": total,
            "verdict": "PASS" if (total == 0 and pos_ok) else "FAIL",
            # NOTE: Chinese report payload value kept verbatim (data, not a log line).
            "note": ("闸函数逐字复制自 gen42_leak.py (父件零改); 4-gram = token 4-gram min-cover"
                     ">=0.9 (候选行逐题复核); 常量向量加严 distinct>=5 且总数>=8; "
                     "池 = LWB 全库 139826 行的 head+concl")}


# ================================================================ §6 GATE assembly
def assemble(fam):
    g = {"schema": "gen44-b-gate/1", "fam": fam, "created": B.now_iso()}
    mod = fam_module(fam)
    g["spec"] = mod.SPEC
    g["ops"] = {k: B.OP_IDX[k] for k in mod.SPEC["ops"]}
    for tag, pat in (("census", "GEN44_%s_CENSUS.json" % fam.upper()),
                     ("gen", "GEN44_%s_CORPUS.json" % fam.upper()),
                     ("check", "GEN44_%s_CHECK.json" % fam.upper()),
                     ("payload", "GEN44_%s_PAYLOAD.json" % fam.upper()),
                     ("leak", "GEN44_%s_LEAK.json" % fam.upper()),
                     ("arm_exam", "GEN44_%s_ARM_EXAM.json" % fam.upper()),
                     ("arm_gold", "GEN44_%s_ARM_GOLD.json" % fam.upper())):
        p = os.path.join(OUT, pat)
        g[tag] = json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None
    g["md5"] = {k: B.md5f(os.path.join(_PKG_DIR, k)) for k in
                ("gen44_bcore.py", "gen44_bgate.py", "gen44_barm.py",
                 "gen44_fam_%s.py" % fam, "gen44_gen_%s.py" % fam,
                 "gen44_check_%s.py" % fam, "gen44_payload_%s.py" % fam,
                 "gen44_render_exam_v3.py")}
    g["judge_md5"] = B.JUDGE_MD5
    g["judge_md5_live"] = B.md5f(os.path.join(B.RES, "scratch", "drct", "drct_common.py"))
    g["verdict"] = _verdict(g)
    return g


def _verdict(g):
    bad = []
    if g.get("judge_md5") != g.get("judge_md5_live"):
        bad.append("judge_md5")
    if g["md5"].get("gen44_render_exam_v3.py") != B.EXAM_MD5:
        bad.append("exam_render_md5")
    for k in ("census", "gen", "check", "payload", "leak", "arm_exam"):
        if not g.get(k):
            bad.append("missing:" + k)
    if g.get("leak") and g["leak"].get("verdict") != "PASS":
        bad.append("leak")
    if g.get("payload") and g["payload"].get("verdict") != "PASS":
        bad.append("payload")
    return {"status": "PENDING" if bad else "COMPLETE", "gaps": bad}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fam", required=True, choices=("b1", "b2", "b6"))
    ap.add_argument("--stage", required=True,
                    choices=("census", "gen", "leak", "payload", "check", "assemble",
                             "selftest"))
    ap.add_argument("--n", type=int, default=800)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sample", type=int, default=48)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--max-lean-jobs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=20260926)
    a = ap.parse_args()
    fam = a.fam
    up = fam.upper()
    if a.stage == "selftest":
        print(json.dumps({"selftest": fam_module(fam).selftest(),
                          "core": B.selftest()}, ensure_ascii=False, indent=1))
        return
    if a.stage == "census":
        rep, recs = census(fam, limit=(a.limit or None))
        B.jdump(rep, os.path.join(OUT, "GEN44_%s_CENSUS.json" % up))
        with open(os.path.join(OUT, "GEN44_%s_CENSUS.jsonl" % up), "w",
                  encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(json.dumps(rep, ensure_ascii=False, indent=1))
    elif a.stage == "gen":
        rows, rep = gen_corpus(fam, a.n, a.seed)
        with open(os.path.join(OUT, "GEN44_%s_CORPUS.jsonl" % up), "w",
                  encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        B.jdump(rep, os.path.join(OUT, "GEN44_%s_CORPUS.json" % up))
        print(json.dumps({k: v for k, v in rep.items() if k != "text_hist"},
                         ensure_ascii=False, indent=1))
    elif a.stage == "leak":
        rows = [json.loads(l) for l in open(os.path.join(
            OUT, "GEN44_%s_CORPUS.jsonl" % up), encoding="utf-8") if l.strip()]
        rep = leak(fam, rows, pool_limit=(a.limit or None))
        B.jdump(rep, os.path.join(OUT, "GEN44_%s_LEAK.json" % up))
        print(json.dumps(rep, ensure_ascii=False, indent=1))
    elif a.stage == "payload":
        rows = [json.loads(l) for l in open(os.path.join(
            OUT, "GEN44_%s_CORPUS.jsonl" % up), encoding="utf-8") if l.strip()]
        rep = payload_tests(fam, rows)
        rep["schema"] = "gen44-b-payload/1"
        rep["fam"] = fam
        B.jdump(rep, os.path.join(OUT, "GEN44_%s_PAYLOAD.json" % up))
        print(json.dumps(rep, ensure_ascii=False, indent=1)[:1500])
    elif a.stage == "check":
        rows = [json.loads(l) for l in open(os.path.join(
            OUT, "GEN44_%s_CORPUS.jsonl" % up), encoding="utf-8") if l.strip()]
        rep = check_corpus(fam, rows, sample=a.sample, shard=a.shard, nshard=a.nshard,
                           maxj=a.max_lean_jobs, seed=a.seed)
        B.jdump(rep, os.path.join(OUT, "GEN44_%s_CHECK.json.s%d" % (up, a.shard)))
        print(json.dumps({k: v for k, v in rep.items() if k != "rows"},
                         ensure_ascii=False, indent=1))
    elif a.stage == "assemble":
        # Merge the check shards.
        rows, meta = [], None
        for p in sorted(glob.glob(os.path.join(OUT, "GEN44_%s_CHECK.json.s*" % up))):
            d = json.load(open(p, encoding="utf-8"))
            rows += d["rows"]
            meta = {k: v for k, v in d.items() if k != "rows"}
        if meta is not None:
            nc = sum(1 for r in rows if r["certified"])
            meta["rows"] = rows
            meta["n_sample"] = len(rows)
            meta["n_certified"] = nc
            meta["rate"] = round(nc / max(1, len(rows)), 4)
            meta["wilson95"] = B.wilson(nc, len(rows))
            B.jdump(meta, os.path.join(OUT, "GEN44_%s_CHECK.json" % up))
        g = assemble(fam)
        B.jdump(g, os.path.join(OUT, "GEN44_%s_GATE.json" % up))
        print(json.dumps({k: g[k] for k in ("fam", "verdict", "spec")},
                         ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()

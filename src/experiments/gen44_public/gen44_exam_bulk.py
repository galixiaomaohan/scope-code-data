# -*- coding: utf-8 -*-
# gen44_exam_bulk.py -- [S8] full-library final-exam **bulk orchestrator** (arm slot S8F, decisive arm)
#
# Measurement convention (ERRATUM6 section D.5 clause 1, two columns; S6R handoff section 5.2; this arm's task sheet):
#   Column 1 = family-gate static "structurally certifiable n / claim pool"   (device convention: read each family's GEN44_B*_GATE.json directly)
#   Column 2 = exam "end-to-end certified n / n actually judged"              (judging convention: this orchestrator + frozen judge module, measured)
#   * Exam volumes: A1 (2050, model chain walk + exam single-rendering v3) / B1-B9 (device-tree exam single-rendering)
#   * (2) exam_parseable_no_family / (3) exam_unparseable are **registered as separate columns, never scored 0**
#   * render_skip (device has no certificate / model chain not renderable) **listed separately, never scored 0**; unjudged != 0
#
# Zero-change discipline (clause by clause against the task sheet):
#   * imports the exam single-rendering v3 body `gen44_render_exam_v3.py` (zero change) -- A1 volume rendering
#   * **never** imports the gold enumeration renderer `gen44_render_gold.py` (no such import anywhere in this file)
#   * model chain walk uses `gen44_eval` (18505cf1... on the experiment host before
#     the package comment translation; re-frozen to c80c7f8a... in this package):
#     this file imports it and calls
#     phase_walk; the only runtime override = `E34._payload_of` pointed, inside this file, at this file's function
#     (A1 per-step states come from the sidecar `frames`, same convention as training-cache steps[k].frame; the
#     override applies only to seg=exam_bulk_va, other segs fall back to the gen44_eval implementation) -- same
#     form as gen44_eval's existing treatment of gen34_eval, recorded.
#   * judge module `scratch/drct/drct_common.py` (7c31b58d...) imported zero-change only (rejudge stage; md5 verified before and after)
#   * judging engine = `gen44_batchconfirm.py` (d5a02f1d... host value before the
#     package comment translation; current digest in MANIFEST_SHA256.txt); this file only **calls** it
#
# Usage:
#   PY="${SCOPE_PYTHON:-python3}"; G44=$R/scratch/ccp/gen44
#   $PY $G44/gen44_exam_bulk.py --stage items_va       # -> out/GEN44_S8_ITEMS_VA.jsonl
#   bash $G44/gen44_s8_run.sh walk                     # GPU chain walk (gen44_eval phase_walk)
#   $PY $G44/gen44_exam_bulk.py --stage manifest_va    # -> out/GEN44_S8_MANIFEST_VA.jsonl
#   $PY $G44/gen44_exam_bulk.py --stage manifest_vb    # -> out/GEN44_S8_MANIFEST_VB.jsonl
#   $PY $G44/gen44_exam_bulk.py --stage classify       # -> out/GEN44_S8_CLASSIFICATION.json
#   $PY $G44/gen44_exam_bulk.py --stage summary        # summarize shard judging -> GEN44_S8_MAIN_SUMMARY.json
#   $PY $G44/gen44_exam_bulk.py --stage rejudge        # stratified literal re-judging of certified >=60
# Write scope: scratch/ccp/gen44/** only. No sudo, no /tmp.
import argparse
import collections
import hashlib
import json
import os
import random
import re
import sys
import time

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
RES = os.environ.get("SCOPE_RESEARCH_ROOT", ".")
CCP = os.path.join(RES, "scratch", "ccp")
G42 = os.path.join(CCP, "gen42")
G34 = os.path.join(CCP, "gen34")
# the gen44 tree on the experiment host (out/data/ckpt live there, not beside
# this file in the package layout)
G44 = os.path.join(CCP, "gen44")
OUT = os.path.join(G44, "out")
DATA = os.path.join(G44, "data")
for _p in (_PKG_DIR, G42, G34):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import gen42_stmt as S                                                  # noqa: E402
import gen42_families as F                                              # noqa: E402
import gen44_render_exam_v3 as RE3                                      # noqa: E402
import gen44_bcore as B                                                 # noqa: E402

try:                                                                    # D-B789-2 guard
    sys.set_int_max_str_digits(1000000)
except Exception:                                                       # noqa: BLE001
    pass

FAMS = ["b%d" % i for i in range(1, 10)]
B_BASE = ("b1", "b2", "b6")
B_EXT = ("b3", "b4", "b5")
CKPT = os.path.join(G44, "ckpt", "g44_v7_formal_s1_a2wt1_6000.pt")
CKPT_MD5 = "8d7c81f0d69322cce1e4eda2b4660fdf"
ITEMS_VA = os.path.join(OUT, "GEN44_S8_ITEMS_VA.jsonl")
WALK_VA = os.path.join(OUT, "GEN44_S8_WALK_VA.json")
MAN_VA = os.path.join(OUT, "GEN44_S8_MANIFEST_VA.jsonl")
MAN_VB = os.path.join(OUT, "GEN44_S8_MANIFEST_VB.jsonl")
CLASS_JSON = os.path.join(OUT, "GEN44_S8_CLASSIFICATION.json")
C6_PATH = os.path.join(OUT, "GEN44_C6_EXCLUDED.jsonl")


# ================================================================ shared helpers
def md5f(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def jlsave(rows, p):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return md5f(p)


def jldump(o, p):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(o, f, ensure_ascii=False, indent=1)
    return md5f(p)


def wilson(k, n, z=1.959964):
    if n <= 0:
        return [0.0, 0.0]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]


class P(object):
    __slots__ = ("name", "head", "concl", "split", "idx")

    def __init__(self, name, head, concl, split=None, idx=None):
        self.name, self.head, self.concl, self.split, self.idx = name, head, concl, split, idx


_LWB = None


def load_lwb():
    global _LWB
    if _LWB is None:
        d = {}
        with open(os.path.join(DATA, "LWB_ALL.jsonl"), encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    d[int(r["idx"])] = r
        _LWB = d
    return _LWB


def load_c6():
    s = set()
    if os.path.exists(C6_PATH):
        for l in open(C6_PATH, encoding="utf-8"):
            if l.strip():
                d = json.loads(l)
                k = d.get("idx", d.get("key"))
                if k is not None:
                    s.add(int(k))
    return s


def a1_idx():
    p = os.path.join(OUT, "GEN44_CENSUS_ROWS_V3.jsonl")
    out = []
    for l in open(p, encoding="utf-8"):
        if not l.strip():
            continue
        r = json.loads(l)
        if r.get("st") == "state1":
            out.append(int(r["idx"]))
    return sorted(out)


def census_all_st():
    p = os.path.join(OUT, "GEN44_CENSUS_ROWS_V3.jsonl")
    out = {}
    for l in open(p, encoding="utf-8"):
        if not l.strip():
            continue
        r = json.loads(l)
        out[int(r["idx"])] = r.get("st")
    return out


def fam_census(fam):
    p = os.path.join(OUT, "GEN44_%s_CENSUS.jsonl" % fam.upper())
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def binders_of(rec, head):
    """binder / hypothesis preamble (same form as gen42_examsid.binders_of)."""
    b = (rec or {}).get("binders")
    if b:
        return b
    h = (head or "").strip()
    m = re.match(r"^\s*theorem\s+[A-Za-z0-9_'.]+\s*(.*)$", h, re.S)
    t = (m.group(1) if m else h).strip()
    return t[:-1].strip() if t.endswith(":") else t


# ================================================================ [1] V-A1 walk sidecar
def stage_items_va(a):
    lwb = load_lwb()
    c6 = load_c6()
    idxs = a1_idx()
    rows, why = [], collections.Counter()
    for i, idx in enumerate(idxs):
        rec0 = lwb.get(int(idx))
        if rec0 is None:
            why["missing"] += 1
            continue
        head = S.norm_answer(rec0["head"])
        concl = S.norm_answer(rec0["concl"])
        prob = P(rec0["name"], rec0["head"], rec0["concl"], rec0.get("subset"), rec0["idx"])
        try:
            d = S.parse_stmt(prob)
        except Exception as e:                                         # noqa: BLE001
            why["parse_exc"] += 1
            rows.append({"idx": idx, "name": rec0["name"], "in_domain": False,
                         "state": "outside_parse", "reason": repr(e)[:60]})
            continue
        if not d["ok"]:
            why["parse_reject"] += 1
            rows.append({"idx": idx, "name": rec0["name"], "in_domain": False,
                         "state": "outside_parse", "reason": str(d.get("reason"))[:60]})
            continue
        got, w2 = F.chain_of(d)
        if got is None:
            why["no_chain"] += 1
            rows.append({"idx": idx, "name": rec0["name"], "in_domain": False,
                         "state": "outside_route", "reason": str(w2)[:60]})
            continue
        binder = binders_of(d, head)
        stmt = ("%s \u22a2 %s" % (binder, concl)) if binder else ("\u22a2 %s" % concl)
        ops = list(got["chain_ops"])
        if got["family"] == F.FAM_SUBST:
            steps = got.get("steps") or []
            sfa = [{k: str(v) for k, v in s.items() if k != "frame"} for s in steps]
        else:
            pl = got.get("route_payload") or {}
            sfa = [{"op": ops[0], "var": str(pl.get("var") or "nn1")},
                   {"op": ops[1] if len(ops) > 1 else None}]
        row = {"ri": int(idx), "name": rec0["name"], "sub": rec0.get("subset"),
               "head": head, "concl": concl, "stmt": stmt,
               "type4": [1, 0, 0, 0], "seg": "exam_bulk_va", "n": len(stmt) + 1, "np": 1,
               "family": got["family"], "cls": got.get("cls"), "route": got.get("route"),
               "ops": ops, "sfa": sfa,
               "frames": [list(map(float, x)) if x is not None else None
                          for x in (got.get("frames") or [])],
               "in_domain": True, "state": "struct",
               "c6_excluded": int(idx) in c6}
        rows.append(row)
    md5 = jlsave(rows, ITEMS_VA)
    ind = [r for r in rows if r.get("in_domain")]
    rep = {"schema": "gen44-s8-items-va/1", "created": now(), "n": len(rows),
           "n_in_domain": len(ind), "why": dict(why), "n_c6": sum(1 for r in rows if r.get("c6_excluded")),
           "route_hist": dict(collections.Counter(r.get("route") for r in ind)),
           "family_hist": dict(collections.Counter(r.get("family") for r in ind)),
           "chainlen_hist": dict(collections.Counter(len(r.get("ops") or []) for r in ind)),
           "sfa_len_hist": dict(collections.Counter(len(r.get("sfa") or []) for r in ind)),
           "frames_len_hist": dict(collections.Counter(len(r.get("frames") or []) for r in ind)),
           "items_md5": md5, "items_path": ITEMS_VA}
    jldump(rep, os.path.join(OUT, "GEN44_S8_ITEMS_VA_META.json"))
    print(json.dumps(rep, ensure_ascii=False, indent=1)[:2500], flush=True)
    return 0


# ================================================================ [2] GPU chain walk
def stage_walk_va(a):
    import gen44_eval as E44
    import gen34_eval as E34
    assert md5f(CKPT) == CKPT_MD5, "终测 ckpt md5 不符"
    assert md5f(os.path.join(_PKG_DIR, "gen44_eval.py")) == a.expect_eval_md5, "gen44_eval md5 mismatch against the pinned (re-frozen) version"

    def _payload_bulk(side_rec):
        if side_rec.get("seg") == "exam_bulk_va":
            fr = side_rec.get("frames") or []
            return {"res": [list(x) if x is not None else None for x in fr],
                    "sfa": list(side_rec.get("sfa") or []), "dev_errs": [],
                    "res_src": "sidecar_device_frames(A1; 与训练缓存 steps[k].frame 同口径)"}
        return E44._ORIG_E34_PAYLOAD_OF(side_rec)

    E34._payload_of = _payload_bulk
    ns = argparse.Namespace()
    ns.vocab = os.path.join(G44, "cache", "CCP_VOCAB_V7.json")
    ns.cache = os.path.join(G44, "cache", "GEN44_V7_ALL.bin")
    ns.side = os.path.join(G44, "cache", "GEN44_V7_SIDECAR.jsonl")
    ns.exam_side = a.items
    ns.allow = os.path.join(G44, "cache", "ccp_allow_v7.bin")
    ns.lat31 = "cppcuda/cache/lat_tab_v31.bin"
    ns.init = "ckpt/sla_v31f.pt"
    ns.ckpt = CKPT
    ns.seg = "exam_bulk_va"
    n_items = 0
    for l in open(a.items, encoding="utf-8"):
        if l.strip() and json.loads(l).get("seg") == "exam_bulk_va":
            n_items += 1
    assert n_items > 0, "侧车无 seg=exam_bulk_va 行: %s" % a.items
    ns.n = a.n if a.n > 0 else n_items
    ns.seed = 20260926
    ns.slot_side = os.path.join(G44, "cache", "GEN44_V7_SLOT_SIDECAR.jsonl")
    ns.dev = a.dev
    ns.log = a.log
    ns.walk_out = WALK_VA
    ns.state_shift = 0
    E44.phase_walk(ns, a.log)
    return 0


# ================================================================ [3] V-A1 render -> manifest
def stage_manifest_va(a):
    """A1 volume manifest rendering.
    use_gold=0 (default): model-walk ops (task-sheet main convention: model chain walk -> render -> judge)
    use_gold=1          : device-chain ops (reference arm: device chain x exam single-rendering v3; reported
                          alongside the A2 gold enumeration convention)"""
    use_gold = int(getattr(a, "use_gold", 0))
    globals()["MAN_VA"] = (os.path.join(OUT, "GEN44_S8_MANIFEST_VA_DEV.jsonl") if use_gold
                           else MAN_VA)
    w = json.load(open(WALK_VA, encoding="utf-8"))
    lwb = load_lwb()
    rows, st = [], collections.Counter()
    for r in w["rows"]:
        idx = int(r["ri"])
        rec0 = lwb.get(idx)
        walked = list((r.get("chain") or {}).get("walked") or [])
        ent = {"key": str(idx), "name": r["name"], "idx": idx, "vol": "A1",
               "R": None, "sub": (rec0 or {}).get("subset"),
               "ops_walked": walked, "ops_gold": list((r.get("chain") or {}).get("gold") or []),
               "walk_exact_hit": bool((r.get("chain") or {}).get("exact_hit")),
               "walk_first_hit": bool((r.get("chain") or {}).get("first_hit")),
               "walk_terminated_end": bool((r.get("chain") or {}).get("terminated_end"))}
        if rec0 is None:
            ent["state"] = "missing"
            rows.append(ent)
            st["missing"] += 1
            continue
        prob = P(rec0["name"], rec0["head"], rec0["concl"], rec0.get("subset"), idx)
        d = S.parse_stmt(prob)
        if not d["ok"]:
            ent["state"] = "parse_reject"
            rows.append(ent)
            st["parse_reject"] += 1
            continue
        got, _w = F.chain_of(d)
        ent["R"] = got.get("route") if got else None
        if use_gold:
            walked = list((got or {}).get("chain_ops") or [])
            ent["ops_used"] = walked
        fam = (F.FAM_STD if (walked and walked[0] != "g42_sb_close") else F.FAM_SUBST)
        got2 = {"family": fam, "chain_ops": walked, "route": (got or {}).get("route"),
                "cls": (got or {}).get("cls"),
                "route_payload": ((got or {}).get("route_payload") or {})}
        try:
            txt = RE3.render_exam_v3(F, d, got2, walked)
        except Exception as e:                                         # noqa: BLE001
            ent["state"] = "render_exc"
            ent["reason"] = repr(e)[:80]
            rows.append(ent)
            st["render_exc"] += 1
            continue
        if not walked or txt == "skip" or not str(txt).strip():
            ent["state"] = "render_skip"
            ent["render_skip_why"] = ("no_walked" if not walked else "v3_skip")
            rows.append(ent)
            st["render_skip"] += 1
            continue
        ent.update({"state": "rendered", "head": rec0["head"], "concl": rec0["concl"],
                    "proofs": [txt], "proof_md5": hashlib.md5(txt.encode()).hexdigest(),
                    "n_lines": len(txt.split("\n")),
                    "renderer": "exam_single_v3", "render_family": fam})
        rows.append(ent)
        st["rendered"] += 1
    md5 = jlsave(rows, MAN_VA)
    jrows = [r for r in rows if r["state"] == "rendered"]
    jpath = MAN_VA.replace(".jsonl", "_JUDGE.jsonl")
    jmd5 = jlsave(jrows, jpath)
    rep = {"schema": "gen44-s8-manifest-va/1", "created": now(), "n": len(rows),
           "arm": ("device_chain_v3 (参照臂: 装置链 x 考试单渲染 v3)" if use_gold
                   else "model_walk_v3 (任务书主口径: 模型走链 x 考试单渲染 v3)"),
           "state_hist": dict(st), "manifest_md5": md5, "manifest_path": MAN_VA,
           "walk_table": w.get("table"), "ckpt_md5": w.get("ckpt_md5"),
           "judge_manifest": {"path": jpath, "md5": jmd5, "n": len(jrows),
                              "note": "只含 state=rendered 行 (render_skip/missing 未判, 单列不作 0)"},
           "by_route": {k: {"n": sum(1 for r in rows if (r.get("R") or "?") == k),
                            "rendered": sum(1 for r in rows if (r.get("R") or "?") == k
                                            and r.get("state") == "rendered")}
                        for k in sorted(set(r.get("R") or "?" for r in rows))},
           "renderer_md5": md5f(os.path.join(_PKG_DIR, "gen44_render_exam_v3.py"))}
    jldump(rep, os.path.join(OUT, "GEN44_S8_MANIFEST_VA%s_META.json"
                             % ("_DEV" if use_gold else "")))
    print(json.dumps(rep, ensure_ascii=False, indent=1)[:2500], flush=True)
    return 0


# ================================================================ [4] V-B family exam-set manifest
_FAMMOD = {}


def fam_io(fam):
    """Family API binding (same dispatch as gen44_cachebuild_v7.Fam; zero-change import)."""
    if fam in _FAMMOD:
        return _FAMMOD[fam]
    if fam in B_BASE:
        mod = B
        io = {"parse": B.parse_stmt_ext, "render": lambda tr: B.render_tree(tr, "exam"),
              "claim": (lambda tr, d=None: bool(__import__("gen44_fam_%s" % fam).claim(tr))),
              "ops": B.tree_ops, "lops": B.tree_leaf_ops}
    elif fam in B_EXT:
        import gen44_b345_ext as E
        mod = E
        fmod = __import__("gen44_fam_%s" % fam)
        io = {"parse": E.parse_stmt_b345,
              "render": lambda tr: E.render_tree_ext(tr, "exam"),
              "claim": (lambda tr, d=None: bool(fmod.claim(tr, d))),
              "ops": E.tree_ops_ext, "lops": E.tree_leaf_ops_ext}
    else:
        import gen44_fam_b789 as M
        fmod = __import__("gen44_fam_%s" % fam)
        io = {"parse": (lambda pr: M.parse_stmt_ext789(pr, fam)),
              "render": lambda tr: M.render_tree(tr, "exam"),
              "claim": (lambda tr, d=None: bool(fmod.claim(tr))),
              "ops": M.tree_ops, "lops": M.tree_leaf_ops}
    _FAMMOD[fam] = io
    return io


def stage_manifest_vb(a):
    lwb = load_lwb()
    c6 = load_c6()
    rows, byfam = [], {}
    for fam in FAMS:
        io = fam_io(fam)
        cen = fam_census(fam)
        pool = [r for r in cen if r.get("state") in ("certifiable", "claim_uncert")]
        cert = [r for r in cen if r.get("state") == "certifiable"]
        fst = collections.Counter()
        fr = []
        for r in cert:
            idx = int(r["idx"])
            ent = {"key": str(idx), "name": r["name"], "idx": idx, "vol": fam.upper(),
                   "fam": fam, "struct_state": "certifiable", "kind": r.get("kind"),
                   "rs": r.get("rs"), "sub": r.get("sub"),
                   "ops_struct": list(r.get("ops") or []),
                   "c6_excluded": idx in c6}
            rec0 = lwb.get(idx)
            if rec0 is None:
                ent["state"] = "missing"
                fst["missing"] += 1
                fr.append(ent)
                continue
            prob = P(rec0["name"], rec0["head"], rec0["concl"], rec0.get("subset"), idx)
            try:
                d = io["parse"](prob)
            except Exception as e:                                     # noqa: BLE001
                ent["state"] = "parse_exc"
                ent["reason"] = repr(e)[:70]
                fst["parse_exc"] += 1
                fr.append(ent)
                continue
            if not d.get("ok"):
                ent["state"] = "parse_reject"
                ent["reason"] = str(d.get("reason"))[:70]
                fst["parse_reject"] += 1
                fr.append(ent)
                continue
            tr = d["tree"]
            try:
                ok_claim = io["claim"](tr, d)
            except Exception as e:                                     # noqa: BLE001
                ok_claim = False
                ent["claim_exc"] = repr(e)[:60]
            if not ok_claim:
                ent["state"] = "claim_lost"
                fst["claim_lost"] += 1
                fr.append(ent)
                continue
            try:
                lines = io["render"](tr)
            except Exception as e:                                     # noqa: BLE001
                ent["state"] = "render_exc"
                ent["reason"] = repr(e)[:70]
                fst["render_exc"] += 1
                fr.append(ent)
                continue
            if not lines:
                ent["state"] = "render_skip"
                fst["render_skip"] += 1
                fr.append(ent)
                continue
            txt = "\n".join(lines)
            ent.update({"state": "rendered", "head": rec0["head"], "concl": rec0["concl"],
                        "proofs": [txt], "proof_md5": hashlib.md5(txt.encode()).hexdigest(),
                        "n_lines": len(lines), "renderer": "family_exam_single",
                        "ops_render": [str(x) for x in (io["ops"](tr) or [])] +
                                      [str(x) for x in (io["lops"](tr) or [])]})
            fst["rendered"] += 1
            fr.append(ent)
        rows += fr
        byfam[fam] = {"n_pop": len(cen), "n_claim_pool": len(pool), "n_certifiable": len(cert),
                      "state_hist": dict(fst),
                      "n_c6_in_pool": sum(1 for r in pool if int(r["idx"]) in c6),
                      "n_c6_in_cert": sum(1 for r in cert if int(r["idx"]) in c6)}
        print("[vb %s] %s" % (fam, json.dumps(byfam[fam], ensure_ascii=False)), flush=True)
    md5 = jlsave(rows, MAN_VB)
    jrows = [r for r in rows if r["state"] == "rendered"]
    jpath = MAN_VB.replace(".jsonl", "_JUDGE.jsonl")
    jmd5 = jlsave(jrows, jpath)
    st = collections.Counter(r["state"] for r in rows)
    rep = {"schema": "gen44-s8-manifest-vb/1", "created": now(), "n": len(rows),
           "state_hist": dict(st), "by_fam": byfam, "manifest_md5": md5,
           "manifest_path": MAN_VB, "n_c6": sum(1 for r in rows if r.get("c6_excluded")),
           "unjudged_pools": {
               "claim_uncert_unrenderable": sum(v["n_claim_pool"] - v["n_certifiable"]
                                                for v in byfam.values()),
               "note": "claim 池内非 certifiable 行 (装置无证书) 无法渲染 => 未判, 单列不作 0 分"},
           "judge_manifest": {"path": jpath, "md5": jmd5, "n": len(jrows)},
           "renderer": "族考试单渲染 (bcore/b345_ext/fam_b789 .render_*, mode=exam; 零改 import)"}
    jldump(rep, os.path.join(OUT, "GEN44_S8_MANIFEST_VB_META.json"))
    print(json.dumps({k: rep[k] for k in ("n", "state_hist", "n_c6")},
                     ensure_ascii=False, indent=1), flush=True)
    return 0


# ================================================================ [5] three-way classification registry
def stage_classify(a):
    st = census_all_st()
    c6 = load_c6()
    N = len(st)
    va = set(a1_idx())
    vb_pool, vb_cert = set(), set()
    for fam in FAMS:
        for r in fam_census(fam):
            if r.get("state") in ("certifiable", "claim_uncert"):
                vb_pool.add(int(r["idx"]))
            if r.get("state") == "certifiable":
                vb_cert.add(int(r["idx"]))
    one = va | vb_pool
    state2 = {i for i, s in st.items() if s == "state2"}
    state3 = {i for i, s in st.items() if s == "state3"}
    err = {i for i, s in st.items() if s not in ("state1", "state2", "state3")}
    two = ((state2 | (state3 - one)) | err) - err      # (2) and (3) are mutually exclusive (err rows listed separately)
    all_idx = set(st.keys())
    rep = {"schema": "gen44-s8-classify/1", "created": now(), "N_library": N,
           "classes": {
               "exam_in_domain": {"n": len(one), "rule":
                                  "state1(A1 路由行) ∪ B1-B9 claim 池 (装置可解析且有族可编链)"},
               "exam_parseable_no_family": {"n": len(two - c6), "rule":
                                            "state2 ∪ (state3 − ①) ∪ census error 行; 不作 0 分"},
               "exam_unparseable": {"n": len(err - c6), "rule": "census error 行 (解析层失败); 不作 0 分"}},
           "counts": {"A1_state1": len(va), "B_claim_pool": len(vb_pool),
                      "B_certifiable": len(vb_cert), "state2": len(state2),
                      "state3": len(state3), "error": len(err)},
           "c6": {"n": len(c6), "in_A1": len(c6 & va), "in_B_pool": len(c6 & vb_pool),
                  "in_B_cert": len(c6 & vb_cert), "in_state2": len(c6 & state2),
                  "in_state3": len(c6 & state3), "in_err": len(c6 & err)},
           "denominator": {"value": N - len(c6), "doc_value": 132375,
                           "delta_vs_doc": (N - len(c6)) - 132375,
                           "note": "139826 − 7448 = 132378 (文书 132375, 差 3 行随报告披露)"},
           "judged_volume": {"A1_rendered": None, "B_cert_rendered": None,
                             "note": "由 manifest_va / manifest_vb 阶段回填"},
           "render_skip_policy": "① 内不可渲染行 (claim_uncert 无装置证书 / 模型链不可渲染) 单列, 不作 0 分",
           "not_counted_as_zero": ["exam_parseable_no_family", "exam_unparseable",
                                   "render_skip"]}
    # Backfill the actually-judged counts (when the manifests exist)
    for p, key in ((os.path.join(OUT, "GEN44_S8_MANIFEST_VA_META.json"), "A1_rendered"),
                   (os.path.join(OUT, "GEN44_S8_MANIFEST_VB_META.json"), "B_cert_rendered")):
        if os.path.exists(p):
            d = json.load(open(p, encoding="utf-8"))
            rep["judged_volume"][key] = (d.get("state_hist") or {}).get("rendered")
    jldump(rep, CLASS_JSON)
    print(json.dumps(rep, ensure_ascii=False, indent=1)[:3000], flush=True)
    return 0


# ================================================================ [6] summary
def load_shards(prefix, nshard):
    rows, shards = [], []
    for i in range(nshard):
        p = os.path.join(OUT, "%s_s%d.json" % (prefix, i))
        if not os.path.exists(p):
            continue
        try:
            d = json.load(open(p, encoding="utf-8"))
        except Exception as e:                                          # noqa: BLE001
            print("!! shard %d read failed: %r" % (i, e), flush=True)
            continue
        rows += d.get("rows") or []
        shards.append({"shard": i, "n": d.get("n_rows"), "n_cert": d.get("n_certified"),
                       "wall_s": d.get("wall_s"), "md5": md5f(p), "judge_md5": d.get("judge_md5")})
    return rows, shards


def man_key_map():
    """manifest (VAM/VAD/VB) -> {key: {vol, fam, proof, proof_md5}} (merged for summary/rejudge)."""
    m = {}
    for p, vol in ((MAN_VA, "A1-M"),
                   (os.path.join(OUT, "GEN44_S8_MANIFEST_VA_DEV.jsonl"), "A1"),
                   (MAN_VB, None)):
        if not os.path.exists(p):
            continue
        for l in open(p, encoding="utf-8"):
            if not l.strip():
                continue
            d = json.loads(l)
            pr = (d.get("proofs") or [])
            m[str(d.get("key"))] = {"vol": d.get("vol") or vol or "?",
                                    "fam": d.get("fam"), "R": d.get("R"),
                                    "idx": d.get("idx"), "head": d.get("head"),
                                    "concl": d.get("concl"),
                                    "proof": pr[0] if pr else None,
                                    "proof_md5": d.get("proof_md5")}
    return m


def parse_vols(spec):
    out = []
    for part in str(spec).split(","):
        if not part.strip():
            continue
        a_, b_ = part.split(":")
        out.append((a_.strip(), int(b_)))
    return out


def stage_summary(a):
    km = man_key_map()
    rows, shards = [], []
    vols = parse_vols(a.vols)
    for prefix, ns in vols:
        r2, s2 = load_shards(prefix, ns)
        for x in r2:
            k = km.get(str(x.get("key"))) or {}
            x["_vol"] = k.get("vol") or "?"
            x["_fam"] = k.get("fam")
        rows += r2
        shards += [dict(x, prefix=prefix) for x in s2]
    judged = [r for r in rows if r.get("certified") is not None]
    cert = sum(1 for r in judged if r["certified"])
    # Per-volume / per-family table
    def bucket(r):
        if (r.get("_vol") or "") == "A1":
            return "A1"
        return (r.get("_fam") or r.get("_vol") or "?").upper()
    by = collections.defaultdict(lambda: {"n": 0, "cert": 0, "n1_void": 0, "n2_fail": 0})
    reasons = collections.Counter()
    for r in judged:
        b = by[bucket(r)]
        b["n"] += 1
        b["cert"] += 1 if r["certified"] else 0
        for rs in (r.get("reasons") or []):
            reasons[rs] += 1
            if rs == "bare_closure_after_strip":
                b["n1_void"] += 1
            if rs == "autoimplicit_dependent":
                b["n2_fail"] += 1
    # Column 1 (family-gate static reading, cited)
    col1 = {}
    for fam in FAMS:
        p = os.path.join(OUT, "GEN44_%s_GATE.json" % fam.upper())
        if os.path.exists(p):
            try:
                g = json.load(open(p, encoding="utf-8"))
                cnt = ((g.get("census") or {}).get("counts") or {})
                col1[fam] = {"gate_md5": md5f(p),
                             "n_claim_pool": cnt.get("claim"),
                             "n_certifiable": cnt.get("certified_struct"),
                             "n_pop": (g.get("census") or {}).get("n_pop"),
                             "src": "GEN44_%s_GATE.json" % fam.upper()}
            except Exception:                                           # noqa: BLE001
                pass
    # Per-arm table (by shard prefix; authoritative convention: A1-M model walk vs A1 device chain vs B volumes)
    by_arm = {}
    for prefix, ns in vols:
        r2, _s2 = load_shards(prefix, ns)
        jj = [x for x in r2 if x.get("certified") is not None]
        cc = sum(1 for x in jj if x["certified"])
        by_arm[prefix] = {"n_judged": len(jj), "n_certified": cc,
                          "rate": round(cc / max(1, len(jj)), 4),
                          "wilson95": wilson(cc, len(jj))}
    rep = {"schema": "gen44-s8-main-summary/1", "created": now(),
           "judge_caliper": {"n_judged": len(judged), "n_certified": cert,
                             "rate": round(cert / max(1, len(judged)), 4),
                             "wilson95": wilson(cert, len(judged))},
           "by_vol_fam": {k: dict(v, rate=round(v["cert"] / max(1, v["n"]), 4),
                                  wilson95=wilson(v["cert"], v["n"]))
                          for k, v in sorted(by.items())},
           "reasons_hist": dict(reasons),
           "column1_static": col1,
           "by_arm": by_arm,
           "shards": shards,
           "vols": [{"prefix": p, "nshard": n} for p, n in vols],
           "n_shards_done": len(shards),
           "scope": ("判卷口径 = 考试端到端 certified/实判卷数; 第一栏静态读数见 column1_static "
                     "(引用族门件); A1-M = 模型走链 v3 卷, A1 = 装置链 v3 参照臂, B* = 族树单渲染")}
    jldump(rep, os.path.join(OUT, "GEN44_S8_MAIN_SUMMARY.json"))
    print(json.dumps({k: rep[k] for k in ("judge_caliper", "by_vol_fam", "reasons_hist",
                                          "n_shards_done")}, ensure_ascii=False, indent=1),
          flush=True)
    return 0


# ================================================================ [7] literal re-judging
def stage_rejudge(a):
    import importlib.machinery
    import importlib.util
    jp = os.path.join(RES, "scratch", "drct", "drct_common.py")
    ld = importlib.machinery.SourceFileLoader("drct_common_rj", jp)
    DC = importlib.util.module_from_spec(importlib.util.spec_from_loader("drct_common_rj", ld))
    ld.exec_module(DC)
    jm0 = md5f(jp)
    km = man_key_map()
    rows = []
    for prefix, ns in parse_vols(a.vols):
        r2, _s = load_shards(prefix, ns)
        rows += r2
    rec = [r for r in rows if r.get("certified")]
    if int(getattr(a, "only_bn1", 0)):
        # [S8F addition] literal re-judging subset: 1 = certified rows with batch_n==1; 2 = **all judged rows** (full-pool reference convention)
        _sc = int(getattr(a, "only_bn1"))
        if _sc == 1:
            rec = [r for r in rec if r.get("main_batch_n") == 1]
        else:
            rec = [r for r in rows if r.get("certified") is not None]
        rec = sorted(rec, key=lambda x: str(x.get("key")))[int(getattr(a, "rshard", 0))::
                                                           int(getattr(a, "rnshard", 1))]
        km2 = man_key_map()
        rows2 = []
        for r in rec:
            k = km2.get(str(r.get("key"))) or {}
            r["_vol"] = k.get("vol")
            r["_fam"] = k.get("fam")
            r["_proof"] = k.get("proof")
            r["_proof_md5_ref"] = k.get("proof_md5")
            rows2.append(r)
        _run_literal(rows2, a, km2, fname="GEN44_S8_REJUDGE_%s%s.json" % (
            "BN1" if _sc == 1 else "LITALL",
            ("_s%d" % int(getattr(a, "rshard", 0))) if int(getattr(a, "rnshard", 1)) > 1 else ""))
        return 0
    # Stratify by volume/family + main_batch_n, then sample evenly
    strata = collections.defaultdict(list)
    for r in rec:
        k = km.get(str(r.get("key"))) or {}
        r["_vol"] = k.get("vol")
        r["_fam"] = k.get("fam")
        r["_proof"] = k.get("proof")
        r["_proof_md5_ref"] = k.get("proof_md5")
        b = (r.get("_vol") or "?") if (r.get("_vol") or "").startswith("A1") else (r.get("_fam") or "?")
        strata[b].append(r)
    rng = random.Random(a.seed)
    picked = []
    keys = sorted(strata)
    per = max(1, -(-a.n // max(1, len(keys))))          # ceil: guarantee total >= n (small strata contribute all they have)
    for k in keys:
        lst = sorted(strata[k], key=lambda x: str(x.get("idx")))
        rng.shuffle(lst)
        picked += lst[:per]
    picked = picked[:a.n] if a.n else picked
    _run_literal(picked, a, km, keys=keys, fname="GEN44_S8_REJUDGE.json")
    return 0


def _run_literal(picked, a, km, keys=None, fname="GEN44_S8_REJUDGE.json"):
    """Per-row literal re-judging (single-theorem files; same rendered-text semantics as the original
    judging) -> report written to disk."""
    import importlib.machinery
    import importlib.util
    jp = os.path.join(RES, "scratch", "drct", "drct_common.py")
    ld = importlib.machinery.SourceFileLoader("drct_common_rj2", jp)
    DC = importlib.util.module_from_spec(importlib.util.spec_from_loader("drct_common_rj2", ld))
    ld.exec_module(DC)
    jm0 = md5f(jp)
    wd = os.path.join(G44, "lean", "s8_rejudge")
    os.makedirs(wd, exist_ok=True)
    out, t0 = [], time.time()
    agree = 0
    for i, r in enumerate(picked):
        ent = {"key": r.get("key"), "name": r.get("name"), "vol": r.get("_vol"),
               "fam": r.get("_fam"), "idx": r.get("idx"),
               "batch_certified": bool(r.get("certified")),
               "batch_main_batch_n": r.get("main_batch_n"),
               "proof_md5": r.get("proof_md5")}
        # Per-row literal re-judging (single-theorem file; same rendered-text semantics as the original judging)
        proof = r.get("_proof")
        if proof is None:
            ent["state"] = "no_proof_text"
            out.append(ent)
            continue
        ent["proof_md5_same_as_manifest"] = (hashlib.md5(proof.encode()).hexdigest()
                                            == r.get("_proof_md5_ref"))
        jr = DC.judge_attempt(r["name"], km.get(str(r.get("key")), {}).get("head") or r.get("head"),
                              km.get(str(r.get("key")), {}).get("concl") or r.get("concl"),
                              proof, wd, "s8rj", timeout=a.timeout)
        ent["lit_certified"] = bool(jr["certified"])
        ent["lit_reasons"] = jr["reasons"]
        ent["lit_lean"] = jr["lean"]
        ent["agree"] = (bool(jr["certified"]) == bool(r["certified"]))
        agree += 1 if ent["agree"] else 0
        out.append(ent)
        if (i + 1) % 10 == 0:
            print("[rejudge %d/%d] agree=%d %.0fs" % (i + 1, len(picked), agree,
                                                      time.time() - t0), flush=True)
    jm1 = md5f(jp)
    reps = {k: sum(1 for x in picked if (x.get("vol") or "") == k
                   or (x.get("fam") or "") == k) for k in (keys or [])}
    rep = {"schema": "gen44-s8-rejudge/1", "created": now(), "n_picked": len(picked),
           "n_agree": agree, "criterion": "certified 分层字面复判 (逐行单 theorem) 一致率",
           "n_lit_certified": sum(1 for x in out if x.get("lit_certified")),
           "n_batch_certified": sum(1 for x in out if x.get("batch_certified")),
           "strata": reps,
           "judge": {"md5_pre": jm0, "md5_post": jm1, "ok": jm0 == jm1},
           "note": "proof 文本取自批判 manifest (同文本同判卷件); 本条只验**字面口径**一致性",
           "wall_s": round(time.time() - t0, 1), "rows": out}
    jldump(rep, os.path.join(OUT, fname))
    print(json.dumps({k: rep[k] for k in ("n_picked", "n_agree", "n_lit_certified",
                                          "judge", "wall_s")}, ensure_ascii=False), flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=("items_va", "walk_va", "manifest_va", "manifest_vb",
                             "classify", "summary", "rejudge"))
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--dev", default="cuda:0")
    ap.add_argument("--items", default=ITEMS_VA)
    ap.add_argument("--log", default=os.path.join(G44, "logs", "gen44_s8_bulk.log"))
    ap.add_argument("--vols", default="GEN44_S8_MAIN_VAM:1,GEN44_S8_MAIN_VAD:8,"
                                      "GEN44_S8_MAIN_VB:8",
                    help="summary/rejudge: comma-separated prefix:nshard list")
    ap.add_argument("--seed", type=int, default=20260926)
    ap.add_argument("--only-bn1", type=int, default=0, dest="only_bn1",
                    help="rejudge: 1 = literal re-judging of all certified rows with batch_n==1")
    ap.add_argument("--rshard", type=int, default=0)
    ap.add_argument("--rnshard", type=int, default=1)
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--use-gold", type=int, default=0, dest="use_gold",
                    help="manifest_va: 1 = render with the device chain (reference arm)")
    ap.add_argument("--expect-eval-md5", dest="expect_eval_md5",
                    default="c80c7f8ac66a5e261257d65c6c18ba4b")  # re-frozen for the package (2026-10-11)
    a = ap.parse_args()
    os.makedirs(os.path.join(G44, "logs"), exist_ok=True)
    fn = {"items_va": stage_items_va, "walk_va": stage_walk_va,
          "manifest_va": stage_manifest_va, "manifest_vb": stage_manifest_vb,
          "classify": stage_classify, "summary": stage_summary,
          "rejudge": stage_rejudge}[a.stage]
    return fn(a)


if __name__ == "__main__":
    sys.exit(main())

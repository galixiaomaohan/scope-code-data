# -*- coding: utf-8 -*-
# gen44_batchconfirm.py -- [S7.5] N1/N2 **batch confirmation** + three gates (V44_DESIGN_ERRATUM6 section B)
#
# Measurement convention (ERRATUM6 section B): N1 and N2 confirmation changed from
#   "one theorem per file" to "K rows per Lean file + automatic fallback on batch failure" --
#   isomorphic to the already-validated main `--batch-size` path; every row inside a batch is still
#   an independent theorem with its own verdict location. **Judging semantics unchanged**:
#     main  = rc==0 and no sorry (per-row semantics; batch rc==0 and no sorry => each row alone would also be accepted)
#     N1    = e0a strip-tactic-args skeleton re-judging: still accepted after stripping and the strip is **not
#             the identity** => bare_closure_after_strip => rejected
#     N2    = original text + set_option autoImplicit false + relaxedAutoImplicit false fails to recompile => rejected
#     certified = main and not N1-rejected and N2
#   * Fallback implementation, recorded: batch failure uses **recursive halving** to locate the bad row
#     (converges to the **same per-row verdict** as a per-row fallback; semantics equivalent, bad-row
#     localization exact to the row; a purely per-row fallback is actually slower at K=8 / 28% failure
#     density). Device-checked by the KD gate G-FALL.
#   * Frozen judge module `scratch/drct/drct_common.py` (md5 7c31b58da9d7228cee31b9be581fe5dd)
#     is **imported zero-change**; this file only reads it (run_lean / render_lean / strip_tactic_args /
#     _normalize_indent / OFFICIAL_HEADER); md5 verified before and after judging.
# Write scope: scratch/ccp/gen44/** only (out/GEN44_S7_* · lean/batchconfirm_*/**).
import argparse
import hashlib
import importlib.machinery
import importlib.util
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
RES = os.environ.get("SCOPE_RESEARCH_ROOT", ".")
# the gen44 tree on the experiment host; the package copy of this module sits
# in src/experiments/gen44_public/ and serves imports only
G44 = os.path.join(RES, "scratch", "ccp", "gen44")
OUT = os.path.join(G44, "out")
LEAN = os.path.join(G44, "lean")
JUDGE_PATH = os.path.join(RES, "scratch", "drct", "drct_common.py")
JUDGE_MD5 = "7c31b58da9d7228cee31b9be581fe5dd"
if _PKG_DIR not in sys.path:
    sys.path.insert(0, _PKG_DIR)


def md5f(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def now_iso():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _load(name, path):
    ld = importlib.machinery.SourceFileLoader(name, path)
    sp = importlib.util.spec_from_loader(name, ld)
    m = importlib.util.module_from_spec(sp)
    ld.exec_module(m)
    return m


DC = _load("drct_common_bc", JUDGE_PATH)
assert md5f(JUDGE_PATH) == JUDGE_MD5, "判卷件 md5 不符 (判前)"
import gen44_bcore as B                                                  # noqa: E402
_LOCK = threading.Lock()
_INFLIGHT = {"n": 0}


def our_lean_jobs():
    """Global count of our real Lean jobs: prefer the P4 scheduler's exact comm convention;
    fall back to this file's own comm convention."""
    try:
        import gen44_p4_scheduler as SCH
        return int(SCH.our_lean_jobs())
    except Exception:                                                   # noqa: BLE001
        pass
    try:
        import subprocess
        out = subprocess.run(["ps", "-eo", "comm,cmd", "--no-headers"],
                             capture_output=True, text=True).stdout
        n = 0
        for l in out.split("\n"):
            p = l.split(None, 1)
            if len(p) < 2:
                continue
            comm, cmd = p[0].strip(), p[1]
            if comm not in ("lake", "lean"):
                continue
            if ("tcaboost" in cmd) or ("elan" in cmd):
                n += 1
        return n
    except Exception:                                                   # noqa: BLE001
        return -1


def prep_proof(text):
    """Same preprocessing as judge_attempt: drop fence lines and leading/trailing newlines."""
    p = (text or "").strip("\n")
    fl = re.compile(r"\s*`{3,}[a-zA-Z0-9_-]*\s*")
    return "\n".join(ln for ln in p.split("\n") if not fl.fullmatch(ln))


# ================================================================ section 1: file construction (same shape for all three states)
def body_main(row):
    return DC._normalize_indent(row["proof"])


def body_n1(row):
    sk = "\n".join(DC.strip_tactic_args(ln) for ln in row["proof"].split("\n"))
    return DC._normalize_indent(sk), (sk.strip() == row["proof"].strip())


def body_n2(row):
    return DC._normalize_indent(row["proof"])


def write_file(rows, kind, path):
    L = list(DC.OFFICIAL_HEADER)
    if kind == "n2":
        L += ["set_option autoImplicit false", "set_option relaxedAutoImplicit false"]
    L.append("")
    for r in rows:
        b = body_main(r) if kind in ("main", "n2") else body_n1(r)[0]
        L.append("%s %s := by" % (r["head"], r["concl"]))
        if b:
            L.append(b)
        L.append("#print axioms %s" % r["name"])
        L.append("")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    return path


def gate_wait(maxj, gate_wait_s):
    waited = 0.0
    while True:
        with _LOCK:
            mine = _INFLIGHT["n"]
        j = our_lean_jobs()
        if mine < maxj and (j < 0 or j <= maxj):
            return waited
        if waited >= gate_wait_s:
            return -1.0
        time.sleep(2.0)
        waited += 2.0


def run_file(path, timeout, maxj, gate_wait_s, nice=True):
    w = gate_wait(maxj, gate_wait_s)
    if w < 0:
        return {"rc": -9, "accepted": False, "sorry": False, "secs": 0.0,
                "out_head": "GATE_STOP", "gate_stop": True}, -1.0
    with _LOCK:
        _INFLIGHT["n"] += 1
    try:
        r = DC.run_lean(path, timeout, nice=nice)
    finally:
        with _LOCK:
            _INFLIGHT["n"] -= 1
    return r, w


# ================================================================ section 2: batched judging (halving fallback)
def phase(rows, kind, K, wd, tag, timeout, maxj, gate_wait_s, workers):
    """Returns {key: {rc,accepted,sorry,secs,batch_n,gate_wait_s,files}}."""
    res = {}
    lvl = [rows[i:i + K] for i in range(0, len(rows), K)] if rows else []
    depth = 0
    n_files = 0
    t0 = time.time()
    ex = ThreadPoolExecutor(max_workers=max(1, workers))

    def judge_chunk(ch, path):
        r, w = run_file(path, timeout, maxj, gate_wait_s)
        return ch, r, w

    while lvl:
        jobs, paths = [], []
        for ci, ch in enumerate(lvl):
            p = os.path.join(wd, "%s_d%d_c%d.lean" % (tag, depth, ci))
            write_file(ch, kind, p)
            paths.append(p)
        futs = [ex.submit(judge_chunk, ch, p) for ch, p in zip(lvl, paths)]
        nxt = []
        for fu in futs:
            ch, r, w = fu.result()
            n_files += 1
            if r.get("accepted"):
                for x in ch:
                    res[x["key"]] = {"rc": r["rc"], "accepted": True, "sorry": r["sorry"],
                                     "secs": round(r["secs"] / max(1, len(ch)), 2),
                                     "batch_n": len(ch), "gate_wait_s": round(w, 1),
                                     "files": 1}
                continue
            if len(ch) == 1:
                x = ch[0]
                res[x["key"]] = {"rc": r["rc"], "accepted": False, "sorry": r["sorry"],
                                 "secs": r["secs"], "batch_n": 1,
                                 "gate_wait_s": round(w, 1),
                                 "out_head": (r.get("out_head") or "")[:400],
                                 "gate_stop": bool(r.get("gate_stop")), "files": 1}
                continue
            # Halving fallback (equivalent to per-row; locates the bad row exactly)
            h = len(ch) // 2
            nxt.append(ch[:h])
            nxt.append(ch[h:])
        lvl = nxt
        depth += 1
    ex.shutdown()
    return {"res": res, "n_files": n_files, "wall_s": round(time.time() - t0, 1),
            "max_depth": depth}


def confirm(rows, K, wd, tag, timeout, maxj, gate_wait_s, workers):
    """Three-state batched judging -> per-row reading (same verdict as the frozen judge_attempt)."""
    t0 = time.time()
    ph_main = phase(rows, "main", K, wd, tag + "_main", timeout, maxj, gate_wait_s, workers)
    acc = [r for r in rows if ph_main["res"].get(r["key"], {}).get("accepted")]
    ph_n1 = phase(acc, "n1", K, wd, tag + "_n1", timeout, maxj, gate_wait_s, workers)
    ph_n2 = phase(acc, "n2", K, wd, tag + "_n2", timeout, maxj, gate_wait_s, workers)
    out = []
    for r in rows:
        m = ph_main["res"].get(r["key"]) or {"accepted": False, "rc": None, "sorry": False,
                                            "secs": 0.0, "batch_n": None}
        rec = {"key": r["key"], "name": r["name"], "sub": r.get("sub"), "R": r.get("R"),
               "idx": r.get("idx"), "ref_certified": r.get("ref_certified"),
               "ref_state": r.get("ref_state"),
               "lean": {k: m.get(k) for k in ("rc", "accepted", "sorry", "secs")},
               "main_batch_n": m.get("batch_n"), "certified": False, "reasons": [],
               "n1n2": None, "proof_md5": hashlib.md5(r["proof"].encode()).hexdigest()}
        if not m.get("accepted"):
            rec["reasons"] = ["lean_reject" + ("_sorry" if m.get("sorry") else "")]
            if m.get("gate_stop"):
                rec["reasons"] = ["gate_stop"]
            out.append(rec)
            continue
        n1 = ph_n1["res"].get(r["key"]) or {"accepted": False, "secs": 0.0}
        n2 = ph_n2["res"].get(r["key"]) or {"accepted": False, "secs": 0.0}
        sk, vac = body_n1(r)
        rec["n1n2"] = {"n1_accepted": bool(n1.get("accepted")),
                       "n2_accepted": bool(n2.get("accepted")),
                       "n1_secs": n1.get("secs"), "n2_secs": n2.get("secs"),
                       "n1_vacuous": bool(vac),
                       "n1_note": ("N1 不适用(无实参可剥: 剥参恒等)" if vac else "N1 已判"),
                       "n1_batch_n": n1.get("batch_n"), "n2_batch_n": n2.get("batch_n")}
        reasons = []
        if n1.get("accepted") and not vac:
            reasons.append("bare_closure_after_strip")
        if not n2.get("accepted"):
            reasons.append("autoimplicit_dependent")
        rec["certified"] = not reasons
        rec["reasons"] = reasons
        out.append(rec)
    rep = {"schema": "gen44-batchconfirm/1", "created": now_iso(), "K": K,
           "n_rows": len(rows), "n_main_acc": len(acc),
           "n_certified": sum(1 for x in out if x["certified"]),
           "n_judged_reject": sum(1 for x in out if not x["certified"]),
           "phase": {"main": {k: ph_main[k] for k in ("n_files", "wall_s", "max_depth")},
                     "n1": {k: ph_n1[k] for k in ("n_files", "wall_s", "max_depth")},
                     "n2": {k: ph_n2[k] for k in ("n_files", "wall_s", "max_depth")}},
           "workers": workers, "max_lean_jobs": maxj, "timeout": timeout,
           "judge_md5": JUDGE_MD5, "wall_s": round(time.time() - t0, 1),
           "engine": "N1/N2 批量 (折半回退); 语义 = drct_common.judge_attempt 同判",
           "rows": out}
    return rep


# ================================================================ section 3: gates
def cmp_ref(rows):
    agree = dis = 0
    dis_rows = []
    n_pos = n_neg = 0
    for r in rows:
        if "ref_certified" not in r:
            continue
        if r["ref_certified"]:
            n_pos += 1
        else:
            n_neg += 1
        if bool(r["certified"]) == bool(r["ref_certified"]):
            agree += 1
        else:
            dis += 1
            if len(dis_rows) < 10:
                dis_rows.append({"key": r["key"], "name": r["name"],
                                 "mine": r["certified"], "ref": r["ref_certified"],
                                 "reasons": r["reasons"]})
    return {"n_ref": n_pos + n_neg, "n_ref_cert": n_pos, "n_ref_rej": n_neg,
            "agree": agree, "n_disagree": dis, "disagree_rows": dis_rows}


def load_manifest(path, limit=0, keys=None):
    rows = []
    for l in open(path, encoding="utf-8"):
        if not l.strip():
            continue
        d = json.loads(l)
        pr = d.get("proofs") or []
        rows.append({"key": str(d.get("key") or d.get("name")), "name": d["name"],
                     "head": d["head"], "concl": d["concl"],
                     "proof": prep_proof(pr[0] if pr else ""),
                     "ref_certified": d.get("ref_certified"),
                     "ref_state": d.get("ref_state"), "sub": d.get("sub"),
                     "R": d.get("R"), "idx": d.get("idx")})
    if keys:
        ks = set(keys)
        rows = [r for r in rows if r["key"] in ks]
    if limit:
        rows = rows[:limit]
    return rows


def stage_run(a):
    jm0 = md5f(JUDGE_PATH)
    assert jm0 == JUDGE_MD5, ("判前 md5 不符", jm0)
    rows = load_manifest(a.manifest, a.limit)
    n_all = len(rows)
    _nsh = int(getattr(a, "nshard", 1))
    _sh = int(getattr(a, "shard", 0))
    if _nsh > 1:                                    # [S8F] sharding (stride)
        rows = rows[_sh::_nsh]
    wd = os.path.join(LEAN, "batchconfirm_%s" % a.tag)
    rep = confirm(rows, a.K, wd, a.tag, a.timeout, a.max_lean_jobs, a.gate_wait_s,
                  a.workers)
    rep["manifest"] = {"path": a.manifest, "md5": md5f(a.manifest)}
    rep["shard"] = {"shard": _sh, "nshard": _nsh, "n_manifest_rows": n_all,
                    "n_shard_rows": len(rows)}
    rep["ref_cmp"] = cmp_ref(rep["rows"])
    jm1 = md5f(JUDGE_PATH)
    rep["judge"] = {"md5_pre": jm0, "md5_post": jm1, "ok": jm0 == jm1 == JUDGE_MD5}
    a2 = a.manifest
    if _nsh > 1:
        _verdict = "N/A (shard %d/%d)" % (_sh, _nsh)
    elif rep["ref_cmp"]["n_ref"] == 0:
        _verdict = "N/A (manifest 无 ref 列)"
    else:
        _verdict = "PASS" if (rep["ref_cmp"]["n_disagree"] == 0
                              and rep["ref_cmp"]["n_ref"] > 0
                              and rep["ref_cmp"]["n_ref_cert"] > 0
                              and rep["ref_cmp"]["n_ref_rej"] > 0) else "FAIL"
    rep["gate_G-AGREE"] = {
        "verdict": _verdict,
        "n_ref": rep["ref_cmp"]["n_ref"], "agree": rep["ref_cmp"]["agree"],
        "two_states_present": bool(rep["ref_cmp"]["n_ref_cert"] > 0
                                  and rep["ref_cmp"]["n_ref_rej"] > 0),
        "manifest": a2}
    rep["gate_G-MD5"] = {"verdict": "PASS" if rep["judge"]["ok"] else "FAIL",
                         "judge_md5": jm0, "pre": jm0, "post": jm1}
    B.jdump(rep, a.out)
    print(json.dumps({k: rep[k] for k in ("n_rows", "n_main_acc", "n_certified",
                                          "n_judged_reject", "wall_s", "phase")},
                     ensure_ascii=False), flush=True)
    print("SHARD %d/%d manifest_rows=%d shard_rows=%d wall=%.1fs out=%s"
          % (_sh, _nsh, n_all, len(rows), rep["wall_s"], a.out), flush=True)
    print(json.dumps({"G-AGREE": rep["gate_G-AGREE"], "G-MD5": rep["gate_G-MD5"],
                      "ref_cmp": {k: rep["ref_cmp"][k] for k in
                                  ("n_ref", "n_ref_cert", "n_ref_rej", "agree",
                                   "n_disagree")}}, ensure_ascii=False), flush=True)
    print("out -> %s md5 %s" % (a.out, md5f(a.out)), flush=True)
    return rep


def stage_fall(a):
    """G-FALL: run a **clean baseline** first, then a batch with one bad row -> the halving fallback
    must locate the bad row exactly and leave every other row bitwise identical to the baseline."""
    jm0 = md5f(JUDGE_PATH)
    assert jm0 == JUDGE_MD5, ("判前 md5 不符", jm0)
    rows = load_manifest(a.manifest, a.limit)
    if a.fall_only_cert:
        rows = [r for r in rows if r.get("ref_certified")]
    rows = rows[:a.fall_n]
    if not rows:
        raise SystemExit("空清单")
    wd = os.path.join(LEAN, "batchconfirm_%s" % a.tag)
    base = confirm(rows, a.K, wd, a.tag + "_clean", a.timeout, a.max_lean_jobs,
                   a.gate_wait_s, a.workers)
    n_base_cert = sum(1 for x in base["rows"] if x["certified"])
    bad_i = a.fall_bad % len(rows)
    bad_rows = [dict(x) for x in rows]
    bad_rows[bad_i]["proof"] = a.fall_bad_proof
    bad_rows[bad_i]["_orig_proof"] = rows[bad_i]["proof"]
    bad_rows[bad_i]["key"] = rows[bad_i]["key"]
    rep = confirm(bad_rows, a.K, wd, a.tag + "_bad", a.timeout, a.max_lean_jobs,
                  a.gate_wait_s, a.workers)
    bad = rep["rows"][bad_i]
    mism = []
    for i, (x, y) in enumerate(zip(base["rows"], rep["rows"])):
        if i == bad_i:
            continue
        same = (bool(x["certified"]) == bool(y["certified"]))
        if x["n1n2"] and y["n1n2"]:
            same = same and (x["n1n2"]["n1_accepted"] == y["n1n2"]["n1_accepted"]
                             and x["n1n2"]["n2_accepted"] == y["n1n2"]["n2_accepted"])
        if not same:
            mism.append({"i": i, "name": x["name"], "clean": x["certified"],
                         "bad_run": y["certified"]})
    ok_loc = bool((not bad["certified"]) and not (bad["lean"] or {}).get("accepted")
                  and not bad["reasons"] == [])
    others = [x for i, x in enumerate(rep["rows"]) if i != bad_i]
    ok_rest = (not mism) and all(x["certified"] for x in others)
    jm1 = md5f(JUDGE_PATH)
    out = {"schema": "gen44-batchconfirm-fall/1", "created": now_iso(),
           "K": a.K, "n_rows": len(rows), "bad_idx": bad_i,
           "bad_name": bad["name"], "bad_proof": a.fall_bad_proof,
           "bad_row": {k: bad[k] for k in ("name", "certified", "reasons", "lean",
                                           "main_batch_n")},
           "clean_baseline": {"n_certified": n_base_cert, "n_rows": len(base["rows"]),
                              "phase": base["phase"], "wall_s": base["wall_s"]},
           "bad_run_phase": rep["phase"],
           "n_others": len(others), "n_others_certified": sum(1 for x in others
                                                             if x["certified"]),
           "n_others_mismatch_vs_clean": len(mism), "mismatch_rows": mism,
           "bad_located": bool(ok_loc), "others_unpolluted": bool(ok_rest),
           "bad_main_calls": bad.get("main_batch_n"),
           "wall_s": round(base["wall_s"] + rep["wall_s"], 1),
           "judge": {"md5_pre": jm0, "md5_post": jm1, "ok": jm0 == jm1 == JUDGE_MD5}}
    out["gate_G-FALL"] = {"verdict": "PASS" if (ok_loc and ok_rest
                                               and out["judge"]["ok"]) else "FAIL",
                          "bad_located": bool(ok_loc),
                          "others_unpolluted": bool(ok_rest),
                          "criteria": "坏行: main 判拒 (rc/理由非空); 其余行: 与干净基线逐行同判"}
    B.jdump(out, a.out)
    print(json.dumps(out["gate_G-FALL"], ensure_ascii=False), flush=True)
    print(json.dumps({"bad_name": bad["name"], "bad_reasons": bad["reasons"],
                      "bad_main_batch_n": bad["main_batch_n"],
                      "clean_cert": n_base_cert, "bad_run_cert": sum(
                          1 for x in rep["rows"] if x["certified"]),
                      "n_others": out["n_others"], "mism": len(mism),
                      "wall_s": out["wall_s"]}, ensure_ascii=False), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="run", choices=("run", "fall"))
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--max-lean-jobs", type=int, default=3)
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--gate-wait-s", type=float, default=900.0)
    ap.add_argument("--tag", default="a2_300")
    ap.add_argument("--shard", type=int, default=0,
                    help="[S8F] shard index (stride slice; default 0")
    ap.add_argument("--nshard", type=int, default=1,
                    help="[S8F] shard count (1 = no sharding; semantics identical to before)")
    ap.add_argument("--out", default="")
    ap.add_argument("--fall-n", type=int, default=8)
    ap.add_argument("--fall-only-cert", type=int, default=1)
    ap.add_argument("--fall-bad", type=int, default=3)
    ap.add_argument("--fall-bad-proof", default="try norm_num1\ntry linarith_typo")
    a = ap.parse_args()
    if not a.out:
        a.out = os.path.join(OUT, "GEN44_S7_%s_%s.json" % (a.stage.upper(), a.tag))
    if int(getattr(a, "nshard", 1)) > 1 and "_s%d" % int(a.shard) not in a.out:
        a.out = a.out.replace(".json", "_s%d.json" % int(a.shard))
    if a.stage == "run":
        stage_run(a)
    else:
        stage_fall(a)


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
# gen36_cachebuild.py -- node [2a] training cache **byte-patch variant** (V36_DESIGN §3 D2 / gates G-2, G-3)
#
# Semantics (defined in gen36_state.py, single state source): frame k (k <= n-2) carries the **post-step
#   device residual** res := evals(R_out(k)) = evals(R_in - mult*l^2)  (device arithmetic, not a copy);
#   the terminal frame (k = n-1) keeps the parent value (read only by the END decision; registered).
# Rationale (node [0]): the terminal-step decision needs the residual constant rho after all block actions;
#   the parent frame carries the pre-step residual => one step late.
# Method: **locate the res float blocks byte-exactly and rewrite in place** (all other bytes unchanged) => verifiable, rollback-safe.
#   Gate G-2: zero-change patch (--dry-run 1) output md5 == source md5; read_cache round-trip consistent; parse cross-check.
#   Gate G-3: cascade rows (a) last-block R_out is constant and == corpus meta rho (b) k<=n-3 frames == parent k+1 frames
#           (device-state chain cross-check: "shift" and "device recomputation" are two independent paths, same value)
#           (c) non-cascade families byte-identical (--scope cascade).
# Reads only: gen33 cache/sidecar, gen32 corpus; writes only under scratch/ccp/gen36/**.
import argparse
import importlib.machinery
import importlib.util
import json
import os
import struct
import sys
from fractions import Fraction

import gen36_common as C

G36, G33, G32 = C.G36, C.G33, C.G32
OUT = G36 + "/out"
CACHE_IN = C.CACHE_G35
CACHE_OUT = C.CACHE_A2
MAGIC = b"CPPCD5\x00\x00"


def load(name, path):
    ld = importlib.machinery.SourceFileLoader(name, path)
    m = importlib.util.module_from_spec(importlib.util.spec_from_loader(name, ld))
    ld.exec_module(m)
    return m


def parse_res_blocks(b):
    """Locate res blocks row by row (offset, nr, [(has, float_offset)]) -- layout verbatim from gen13 write_cache."""
    assert b[:8] == MAGIC, b[:8]
    off = 8
    hdr = list(struct.unpack_from("<16i", b, off))
    off += 64
    n_rows, nT4 = hdr[0], hdr[1]
    off += 16 * nT4                      # t4 table

    def ri():
        nonlocal off
        v = struct.unpack_from("<i", b, off)[0]
        off += 4
        return v

    def rf():
        nonlocal off
        v = struct.unpack_from("<f", b, off)[0]
        off += 4
        return v

    blocks = []
    for _ in range(n_rows):
        nid = ri(); ri(); ri()
        off += 4 * nid
        rf(); rf(); rf()
        nd = ri(); off += 8 * nd
        blk = {"nr": 0, "vecs": [], "labels": None, "nid": nid}
        nr = ri()
        blk["nr"] = nr
        for _s in range(nr):
            has = b[off]
            off += 1
            if has:
                blk["vecs"].append((1, off))
                off += 32
            else:
                blk["vecs"].append((0, off))
        nl = ri()
        blk["labels"] = [struct.unpack_from("<i", b, off + 4 * i)[0] for i in range(nl)]
        off += 4 * nl
        for _k in ("feat_pos", "bsl", "state_pos"):
            n = ri()
            off += 4 * n
        blocks.append(blk)
    assert off == len(b), ("解析余量", off, len(b))
    return hdr, blocks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-in", default=CACHE_IN)
    ap.add_argument("--cache-out", default=CACHE_OUT)
    ap.add_argument("--render-side", default=G33 + "/out/DEEP_RENDER_SIDECAR.jsonl")
    ap.add_argument("--corpus", default=C.DEEP_CORPUS)
    ap.add_argument("--scope", default="cascade", choices=("cascade", "all"))
    ap.add_argument("--dry-run", type=int, default=0, help="1 = zero-change patch (byte gate G-2)")
    ap.add_argument("--report", default=OUT + "/GEN36_CACHE_GATE.json")
    ap.add_argument("--log", default=G36 + "/logs/gen36_cachebuild.log")
    a = ap.parse_args()
    os.makedirs(G36 + "/cache", exist_ok=True)
    C.log("== gen36 cache byte patch (scope=%s dry_run=%d) ==" % (a.scope, a.dry_run), a.log)
    led = C.parent_ledger()
    assert led["drct_common"]["md5"] == C.JUDGE_MD5_EXPECT, led["drct_common"]
    C.log("parent ledger n=%d judge artifact md5=%s" % (len(led), led["drct_common"]["md5"]), a.log)

    md5_in = C.md5f(a.cache_in)
    b = bytearray(open(a.cache_in, "rb").read())
    hdr, blocks = parse_res_blocks(b)
    C.log("source %s md5=%s bytes=%d rows=%d nT4=%d" %
          (a.cache_in, md5_in, len(b), hdr[0], hdr[1]), a.log)

    # ---- gate G-2a: parse cursor vs frozen reader read_cache, bitwise cross-check ----
    CCF = load("g36_ccf2", os.path.join(C.G13, "ccp_cachefull.py"))
    crows, _meta = CCF.read_cache(a.cache_in)
    assert len(crows) == len(blocks), (len(crows), len(blocks))
    n_cmp, n_bad = 0, 0
    for R, blk in zip(crows, blocks):
        assert len(R["res"]) == blk["nr"], "nr 失配"
        for x, (has, off_) in zip(R["res"], blk["vecs"]):
            assert bool(has) == bool(x is not None), "has 失配"
            if x is not None:
                got = list(struct.unpack_from("<8f", b, off_))
                n_cmp += 1
                if got != [float(y) for y in x]:
                    n_bad += 1
        assert R["labels"] == blk["labels"], "labels 失配"
    C.log("G-2a parse cross-check: %d res vectors, bad=%d (labels/pos equal row by row)" % (n_cmp, n_bad), a.log)
    assert n_bad == 0

    # ---- corpus side: ri -> name -> family / args / rho ----
    side = C.jlload(a.render_side)
    fam_of, args_of, rho_of = {}, {}, {}
    for line in open(a.corpus, encoding="utf-8"):
        if not line.strip():
            continue
        d = json.loads(line)
        nm = d["name"]
        fam_of[nm] = d.get("family")
        args_of[nm] = [(s.get("args") or {}) for s in (d.get("steps") or [])]
        if d.get("family") == "gen32_ineq_cascade":
            rho_of[nm] = str(Fraction(str((d["meta"] or {}).get("rho", "0"))))
    assert len(side) == len(blocks), (len(side), len(blocks))
    tgt = []
    for s in side:
        ri = int(s["ri"])
        fam = fam_of.get(s["name"])
        if a.scope == "all" or fam == "gen32_ineq_cascade":
            tgt.append((ri, s["name"], fam))
    C.log("target rows %d / %d (scope=%s; non-target families **byte-identical**)" % (len(tgt), len(side), a.scope),
          a.log)

    # ---- rewrite: frame k carries the post-step device residual (gen36_state is the single state source) ----
    import gen36_state as S
    n_shift, n_hold, n_none, n_err = 0, 0, 0, 0
    xc_ok, xc_tot, xc_bad = 0, 0, []            # gate G-3b: shift == device recomputation
    sem_ok, sem_tot, sem_bad = 0, 0, []         # gate G-3a: after the last block = constant rho
    rho_ag, rho_tot = 0, 0                      # device-computed rho vs corpus meta rho
    for ri, name, fam in tgt:
        blk = blocks[ri]
        vecs = blk["vecs"]
        n = len(vecs)
        old = [list(struct.unpack_from("<8f", b, o)) if h else None for h, o in vecs]
        if a.dry_run:
            new = list(old)
        else:
            new, diag = S.shifted_res_frames(args_of.get(name) or [], old[n - 1])
            if diag["errs"]:
                n_err += 1
                if len(sem_bad) < 5:
                    sem_bad.append({"name": name, "errs": diag["errs"][:2]})
        for k in range(n):
            h, o = vecs[k]
            if not h:
                n_none += 1
                continue
            if not a.dry_run and new[k] is not None and new[k] != old[k]:
                struct.pack_into("<8f", b, o, *new[k])
                n_shift += 1
            elif new[k] == old[k]:
                n_hold += 1
        if fam == "gen32_ineq_cascade" and not a.dry_run and n >= 2 and not diag["errs"]:
            # G-3b: k <= n-3 frames == parent k+1 frames (state-chain cross-check)
            for k in range(n - 2):
                xc_tot += 1
                if new[k] == old[k + 1]:
                    xc_ok += 1
                elif len(xc_bad) < 3:
                    xc_bad.append({"name": name, "k": k})
            # G-3a: post-last-block residual = constant and == corpus meta rho
            sem_tot += 1
            rho_meta = Fraction(rho_of.get(name, "0"))
            v = list(struct.unpack_from("<8f", b, vecs[n - 2][1]))
            is_const_ok = diag["last_const"] and all(abs(x - float(rho_meta)) < 1e-6 for x in v)
            sem_ok += 1 if is_const_ok else 0
            if not is_const_ok and len(sem_bad) < 5:
                sem_bad.append({"name": name, "rho_meta": str(rho_meta), "res_n2": v,
                                "rho_dev": diag["rho_dev"]})
            if diag["rho_dev"] is not None:
                rho_tot += 1
                if Fraction(diag["rho_dev"]) == rho_meta:
                    rho_ag += 1

    os.makedirs(os.path.dirname(a.cache_out), exist_ok=True)
    with open(a.cache_out, "wb") as f:
        f.write(bytes(b))
    md5_out = C.md5f(a.cache_out)

    # ---- gate G-2b: round-trip read consistent + non-target rows byte-identical ----
    rrows, _ = CCF.read_cache(a.cache_out)
    want_fn = (lambda ri, old_r: old_r) if a.dry_run else None
    n_rt_bad = 0
    for ri, name, fam in tgt:
        old_r = crows[ri]["res"]
        new_r = rrows[ri]["res"]
        args_l = args_of.get(name) or []
        if a.dry_run:
            want = old_r
        else:
            want, _d = S.shifted_res_frames(args_l, old_r[len(old_r) - 1])
        wf = [None if x is None else [float(y) for y in x] for x in want]
        gf = [None if x is None else [float(y) for y in x] for x in new_r]
        if wf != gf:
            n_rt_bad += 1
    n_other_diff = 0
    if a.scope == "cascade":
        tgt_set = {ri for ri, _n, _f in tgt}
        for ri in range(len(blocks)):
            if ri in tgt_set:
                continue
            if crows[ri]["res"] != rrows[ri]["res"]:
                n_other_diff += 1
    seg = json.load(open(a.cache_in + ".segidx.json", encoding="utf-8"))
    seg_src = seg.get("cache")
    seg["cache"] = a.cache_out
    seg["gen36_patch"] = {"src_cache": a.cache_in, "src_md5": md5_in,
                          "mode": "res_post_action(%s)" % ("dry_run" if a.dry_run else a.scope),
                          "note": ("帧 k (k<=n-2) 载 evals(R_out(k)) = 步后装置残量; "
                                   "尾帧保持父件值; 仅 res 浮点块改写")}
    with open(a.cache_out + ".segidx.json", "w", encoding="utf-8") as f:
        json.dump(seg, f, ensure_ascii=False, indent=1)

    rep = {"schema": "gen36-cache-gate/1", "scope": a.scope, "dry_run": int(a.dry_run),
           "cache_in": {"path": a.cache_in, "md5": md5_in, "bytes": os.path.getsize(a.cache_in)},
           "cache_out": {"path": a.cache_out, "md5": md5_out,
                         "bytes": os.path.getsize(a.cache_out)},
           "hdr": hdr, "n_rows": len(blocks),
           "G2a_parse_xcheck": {"n_res_cmp": n_cmp, "bad": n_bad},
           "G2_byte_gate_identical": bool(a.dry_run and md5_out == md5_in),
           "G2b_roundtrip_read_ok": bool(n_rt_bad == 0), "n_rt_bad": n_rt_bad,
           "G3a_last_const_eq_rho": {"n": sem_tot, "ok": sem_ok,
                                     "rate": round(sem_ok / max(1, sem_tot), 6), "bad": sem_bad},
           "G3b_shift_eq_device_chain": {"n": xc_tot, "ok": xc_ok,
                                         "rate": round(xc_ok / max(1, xc_tot), 6), "bad": xc_bad},
           "G3c_nontarget_rows_changed": n_other_diff,
           "rho_dev_vs_meta": {"n": rho_tot, "agree": rho_ag},
           "n_target_rows": len(tgt), "n_shifted_vectors": n_shift,
           "n_hold_vectors": n_hold, "n_none_vectors": n_none, "n_rows_dev_err": n_err,
           "segidx_src_cache": seg_src,
           "parent_ledger_unchanged": {k: v["md5"] for k, v in led.items()}}
    rep["md5"] = C.save_json(a.report, rep)
    C.log("cache patch: md5 %s -> %s | shifted=%d held=%d none=%d err=%d | G-3a %d/%d . G-3b %d/%d "
          "| non-target rows changed=%d" % (md5_in[:8], md5_out[:8], n_shift, n_hold, n_none, n_err,
                                            sem_ok, sem_tot, xc_ok, xc_tot, n_other_diff), a.log)
    print(json.dumps({"out": a.cache_out, "md5_in": md5_in, "md5_out": md5_out,
                      "dry_run_identical": bool(a.dry_run and md5_out == md5_in),
                      "G3a": rep["G3a_last_const_eq_rho"], "G3b": rep["G3b_shift_eq_device_chain"],
                      "G2b_rt_ok": rep["G2b_roundtrip_read_ok"],
                      "n_shifted": n_shift, "n_other_diff": n_other_diff,
                      "rho_agree": rep["rho_dev_vs_meta"], "report_md5": rep["md5"]},
                     ensure_ascii=False))
    return rep


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
# gen33_build.py -- deep corpus -> hybrid-arm trainable cache + render-segment sidecar (gen33; interface adaptation layer)
#
# Positioning (V33_DESIGN §4): **interface adaptation layer**. Adapts gen32 deep-corpus rows (each row carries
#   chain_ops gold + proof_text + steps[].args + steps[].res true-value states) into the CPPCD5 cache consumed
#   directly by the gen13 chain training module, and attaches the render-segment sidecar (prompt+proof_text token
#   sequence).
#
# Differences vs gen26_hybuild (registered):
#   * No gen12 curriculum row to join: deep-corpus steps[].op / args already live in-row (self-contained), and
#     type4 takes the CCP curriculum's only type4 = [1,0,0,0] (D-4; ccp_rollout.py registered the same).
#   * Value space 352 / operator space 194 (v4 ctx): opmap includes the nine gen32_* operators; slot table uses v4 slot_schema_v4_new.
#   * State frame (KIND_PST) takes "ok" (every deep-chain step is a success step; of the ST_IDX 4 states, ok), terminal frame "end" (D-3).
#   * **res frames carry the true numeric state** (8-dim integer-lattice exact values; semantic change this generation).
#   * The sidecar deliberately carries **no res** (device-recompute convention): only ops + sfa(args) + vars/dom_dump
#     (what recomputation needs), rebuilt on the fly by gen33_device -- train and exam share one source = device state (change96).
#
# Read-only: scratch/ccp/gen32/DEEP_CORPUS_V1.jsonl · gen26/out/CCP_VOCAB_V3.json · gen13/**
# Writes only: scratch/ccp/gen33/**
import argparse
import collections
import hashlib
import importlib.machinery
import importlib.util
import json
import os
import sys
import time

import g33_common as C

RES = C.RES
CCP = C.CCP
G33 = C.G33
G13 = C.G13
OUT = os.path.join(G33, "out")
os.makedirs(OUT, exist_ok=True)
if G13 not in sys.path:
    sys.path.insert(0, G13)

from gen33s_vocab4 import load_ctx4                                   # noqa: E402
import gen33_device as DEV                                            # noqa: E402

_ld = importlib.machinery.SourceFileLoader("g33_cachefull",
                                          os.path.join(G13, "ccp_cachefull.py"))
CCF = importlib.util.module_from_spec(importlib.util.spec_from_loader("g33_cachefull", _ld))
_ld.exec_module(CCF)

CLASSES = ("ind", "ineq")
VOC4 = G33 + "/cache/CCP_VOCAB_V4.json"
PROMPT_TMPL = ("You are a Lean 4 prover. Given the theorem statement below, output the"
               " complete proof body inside a ```lean4 fenced block. Output the proof body"
               " only (no `theorem` header, no `import` lines).\n"
               "Theorem statement:\n%s\nProof:\n")
TYPE4 = [1, 0, 0, 0]


def log(msg, path=None):
    C.log(msg, path)


def to_src_row(row):
    """Deep-corpus row -> input row shape for ccp_cachefull.build (frame-construction convention follows gen13 bit-for-bit)."""
    steps, sfa = [], []
    for s in row["steps"]:
        steps.append({"op": s["op"], "ok": True, "err": None,
                      "res": [float(x) for x in s["res"]], "state": "ok"})
        sfa.append(dict(s["args"]))
    return {"stmt": row["stmt_state"], "type4": list(TYPE4),
            "program": list(row["chain_ops"]), "steps": steps, "sf_args": sfa,
            "episode": None, "accepted": True, "readout_only": False,
            "name": row["name"], "cls": row["cls"]}


def to_src_row_dev(row):
    """Same row shape, but res replaced by **device-recomputed** values (side (b) of the frame-level cross-check)."""
    steps, sfa = [], []
    for k, s in enumerate(row["steps"]):
        res, _ex, _sp, _op, err = DEV.dev_resid(row, k)
        if err:
            raise RuntimeError("装置现算失败 %s[%d]: %s" % (row["name"], k, err))
        steps.append({"op": s["op"], "ok": True, "err": None, "res": res, "state": "ok"})
        sfa.append(dict(s["args"]))
    return {"stmt": row["stmt_state"], "type4": list(TYPE4),
            "program": list(row["chain_ops"]), "steps": steps, "sf_args": sfa,
            "episode": None, "accepted": True, "readout_only": False,
            "name": row["name"], "cls": row["cls"]}


def build_render_sidecar(tok, rows, segs, logp=None):
    """Render-segment sidecar: ids = [M_TASK] + tok(prompt) + tok(target); loss start np = len(prompt)+1."""
    import v31_model as vm
    M_TASK = vm.st.M_TASK
    out, ntok = [], collections.Counter()
    for i, r in enumerate(rows):
        stmt = r["stmt_state"]
        p = tok(PROMPT_TMPL % stmt, add_special_tokens=False)["input_ids"]
        t = tok("```lean4\n" + r["proof_text"] + "\n```",
                add_special_tokens=False)["input_ids"]
        assert t, ("空 target token", r["name"])
        ids = [int(M_TASK)] + [int(x) for x in p] + [int(x) for x in t]
        out.append({"ri": i, "name": r["name"], "cls": r["cls"],
                    "seg": "a_train" if i < segs["ntr"] else "a_held",
                    "np": len(p) + 1, "n": len(ids), "ids": ids,
                    "stmt": stmt, "head": r["head"], "concl": r["concl"],
                    "type4": list(TYPE4),
                    "ops": list(r["chain_ops"]),
                    "sfa": [dict(s["args"]) for s in r["steps"]],
                    "vars": list((r.get("meta") or {}).get("vars") or []),
                    "dom_dump": (r.get("meta") or {}).get("dom_dump"),
                    "family": r["family"], "chain_len": r["chain_len"],
                    "res": None,
                    "res_note": "侧车**不载金标 res**; 状态由 gen33_device 装置现算 (变更96)"})
        ntok["n"] += len(ids)
        ntok["max"] = max(ntok["max"], len(ids))
    return out, dict(ntok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=C.DEEP_CORPUS)
    ap.add_argument("--vocab", default=VOC4)
    ap.add_argument("--allow", default=G33 + "/cache/ccp_allow_v4.bin")
    ap.add_argument("--out-cache", default=G33 + "/cache/deep_all.bin")
    ap.add_argument("--out-render", default=OUT + "/DEEP_RENDER_SIDECAR.jsonl")
    ap.add_argument("--out-report", default=OUT + "/GEN33_BUILD.json")
    ap.add_argument("--log", default=G33 + "/logs/gen33.log")
    ap.add_argument("--frame-n", type=int, default=36, help="sampled rows for the frame-level cross-check (>=30)")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.log), exist_ok=True)
    t0 = time.time()
    log("== gen33 deep-corpus interface adaptation (cache + render sidecar + device verification) ==", a.log)
    led = C.parent_ledger()
    log("parent ledger: %s" % json.dumps({k: v["md5"] for k, v in led.items()},
                                         ensure_ascii=False), a.log)
    assert led["drct_common"]["md5"] == C.JUDGE_MD5_EXPECT

    rows = C.jlload(a.corpus)
    log("corpus %s rows=%d md5=%s" % (a.corpus, len(rows), C.md5f(a.corpus)), a.log)
    assert len(rows) == 2399, len(rows)
    ch = collections.Counter(r["chain_len"] for r in rows)
    fam = collections.Counter(r["family"] for r in rows)
    clsh = collections.Counter(r["cls"] for r in rows)
    n_step = sum(r["n_steps"] for r in rows)
    n_nz = sum(r["resid_nonzero"] for r in rows)
    log("check: chain_len=%s family=%s cls=%s n_steps=%d resid_nonzero=%d/%d"
        % (dict(sorted(ch.items())), dict(fam), dict(clsh), n_step, n_nz, n_step), a.log)
    assert n_step == 15791 and n_nz == 15791, (n_step, n_nz)
    assert set(clsh) == set(CLASSES), dict(clsh)

    # ---- split: md5(name)[:8] % 10 == 7 -> held (registered identically in gen13) ----
    train = [r for r in rows
             if int(hashlib.md5(r["name"].encode("utf-8")).hexdigest()[:8], 16) % 10 != 7]
    held = [r for r in rows
            if int(hashlib.md5(r["name"].encode("utf-8")).hexdigest()[:8], 16) % 10 == 7]
    ordered = train + held
    ntr, nheld = len(train), len(held)
    log("split: train=%d held=%d (rule: md5(name)[:8]%%10==7)" % (ntr, nheld), a.log)
    assert ntr + nheld == len(rows)
    bcls_tr = collections.Counter(r["cls"] for r in train)
    assert set(bcls_tr) == set(CLASSES), ("训练段缺类 (交错混喂要求每类非空)", dict(bcls_tr))

    # ---- device verification (change-97 gate 1): full recompute vs gold ----
    dv = DEV.verify_against_gold(rows, logp=a.log)
    C.save_json(OUT + "/GEN33_DEVICE_VS_GOLD.json",
                {"schema": "gen33-device-vs-gold-report/1", **dv,
                 "corpus": a.corpus, "corpus_md5": C.md5f(a.corpus),
                 "entry": "gen33_device.dev_resid (装置现算, 不读行内 res)"})
    if not dv["bitwise_equal"]:
        raise SystemExit("变更97 门(1) 未过: 装置现算 != 金标 res (%s)" % dv)

    ctx = load_ctx4(a.vocab, logp=a.log)
    vocab = ctx["vocab"]
    log("ctx4: NVALUES=%d names_head=%d per_ops=%d" % (ctx["NVALUES"], len(ctx["names_head"]),
                                                      len(ctx["per"])), a.log)

    # ---- build cache ----
    src_rows = [to_src_row(r) for r in ordered]
    recs, t4set, stats = CCF.build(ctx, src_rows)
    assert len(recs) == len(ordered), (len(recs), len(ordered))
    segidx = {"cache": a.out_cache, "schema_rev": "ccpcd5/1",
              "ops": {"base": 161, "ccp": vocab["ops_ccp"], "bh": vocab["ops_bh"],
                      "g24": vocab["ops_g24"], "gen32": vocab["ops_gen32"],
                      "end": ctx["END"], "head": vocab["head"],
                      "n_vals": vocab["n_vals_ext"]},
              "segments": {"a_train": [0, ntr], "a_held": [ntr, ntr + nheld]},
              "held_rows": list(range(ntr, ntr + nheld)),
              "held_rule": "md5(name)[:8] % 10 == 7 (gen13 登记同式; 从深语料切出)",
              "n_held": nheld, "vocab_md5": C.md5f(a.vocab),
              "corpus_md5": C.md5f(a.corpus), "corpus_path": a.corpus,
              "name_space_note": ("段名沿用父件 a_train/a_held (复用父件判读逻辑零改); "
                                  "语义 = 深语料训练段/留出段")}
    md5c = CCF.write_cache(a.out_cache, recs, t4set, ctx, segidx)
    back, hdr = CCF.read_cache(a.out_cache)
    v4 = json.load(open(a.vocab, encoding="utf-8"))
    chk = {"n_rows": len(back) == len(recs),
           "labels_feat": all(len(b["labels"]) == len(b["feat_pos"]) + 1 for b in back),
           "labels_range": all(max(b["labels"] + [0]) < v4["head"] for b in back),
           "labels_has_new_ops": all(any(161 <= x < 194 for x in b["labels"]) for b in back),
           "bsl_len": all(len(b["bsl"]) == len(b["labels"]) * ctx["MS"] for b in back),
           "ids": all(b["ids"] == r["ids"] for b, r in zip(back, recs)),
           "bsl_range": all((max(b["bsl"]) < v4["n_vals_ext"]) if b["bsl"] else True
                            for b in back),
           "desc": all(b["desc"] == r["desc"] for b, r in zip(back, recs)),
           "res_load": all(len(b["res"]) == len(r["res"]) for b, r in zip(back, recs)),
           "hdr": (hdr["hdr"][12] == v4["head"] and hdr["hdr"][10] == ctx["END"]
                   and hdr["hdr"][11] == 161 and hdr["hdr"][14] == v4["n_vals_ext"]
                   and hdr["hdr"][13] == ctx["MS"])}
    # nonzero rate of res frames (read-back cache; cross-checked against all 15791 corpus states)
    n_res = n_res_nz = 0
    for b in back:
        for k, rv in enumerate(b["res"]):
            if rv is None:
                continue
            n_res += 1
            if any(x != 0.0 for x in rv):
                n_res_nz += 1
    chk["res_frames_nonzero"] = bool(n_res == n_step and n_res_nz == n_step)
    roundtrip = all(chk.values())
    assert roundtrip, chk
    log("cache %s md5=%s n=%d (train %d / held %d) res_frames=%d/%d nonzero roundtrip=%s"
        % (a.out_cache, md5c, len(recs), ntr, nheld, n_res_nz, n_res, roundtrip), a.log)

    # ---- frame-level cross-check (change-97 gate 2): device-recomputed frames vs cache frames ----
    import random
    rnd = random.Random(20260923)
    strata = collections.defaultdict(list)
    for r in rows:
        strata["%s|%d" % (r["family"], r["chain_len"])].append(r)
    sample = []
    per = max(1, a.frame_n // max(1, len(strata)))
    for k in sorted(strata):
        pool = sorted(strata[k], key=lambda x: x["name"])
        sample += rnd.sample(pool, min(per, len(pool)))
    sample = sample[:max(a.frame_n, 30)]
    fd_bad, fd_n = [], 0
    for r in sample:
        rg, _, _ = CCF.build(ctx, [to_src_row(r)])
        rd, _, _ = CCF.build(ctx, [to_src_row_dev(r)])
        fd_n += 1
        dg = [tuple(x) for x in rg[0]["desc"]]
        dd = [tuple(x) for x in rd[0]["desc"]]
        diff = sum(1 for x, y in zip(dg, dd) if x != y) + abs(len(dg) - len(dd))
        if diff or rg[0]["bsl"] != rd[0]["bsl"] or rg[0]["labels"] != rd[0]["labels"]:
            fd_bad.append({"name": r["name"], "desc_diff": diff,
                           "bsl_eq": rg[0]["bsl"] == rd[0]["bsl"],
                           "labels_eq": rg[0]["labels"] == rd[0]["labels"]})
    frame_gate = {"schema": "gen33-frame-consistency/1", "n_sample": fd_n,
                  "strata": sorted(strata), "n_bad": len(fd_bad), "bad": fd_bad[:8],
                  "desc_diff_total": sum(x["desc_diff"] for x in fd_bad),
                  "pass": bool(not fd_bad and fd_n >= 30),
                  "rule": ("同 ctx 下 (a) 载荷=行内 res 与 (b) 载荷=装置现算 res 两次构帧, "
                           "帧序列 desc 逐位比对; desc_diff=0 且 bsl/labels 相等才过")}
    C.save_json(OUT + "/GEN33_FRAME_CONSISTENCY.json", frame_gate)
    log("change-97 frame-consistency gate: n=%d bad=%d pass=%s" % (fd_n, len(fd_bad), frame_gate["pass"]),
        a.log)
    if not frame_gate["pass"]:
        raise SystemExit("变更97 门(2) 未过: 帧级对拍有差异 %s" % fd_bad[:3])

    # ---- render sidecar ----
    side, rtok = build_render_sidecar(ctx["tok"], ordered, {"ntr": ntr}, logp=a.log)
    md5s = C.jlsave(side, a.out_render)
    rback = [json.loads(l) for l in open(a.out_render, encoding="utf-8") if l.strip()]
    chk_r = {"n": len(rback) == len(side),
             "ids": all(x["ids"] == y["ids"] for x, y in zip(rback, side)),
             "np": all(0 < x["np"] < x["n"] for x in rback),
             "seg": all((x["seg"] == "a_train") == (x["ri"] < ntr) for x in rback),
             "no_gold_res": all(x["res"] is None for x in rback)}
    assert all(chk_r.values()), chk_r
    log("sidecar %s md5=%s n=%d tok(sum/max)=%s/%s" % (a.out_render, md5s, len(side),
                                                     rtok.get("n"), rtok.get("max")), a.log)

    Ls = sorted(stats["L"])
    sup_unk = collections.Counter()
    for b in back:
        for k, op in enumerate(b["labels"]):
            if k * ctx["MS"] >= len(b["bsl"]):
                break
            for j in range(ctx["MS"]):
                if b["bsl"][k * ctx["MS"] + j] == ctx["UNK"]:
                    sup_unk[k] += 1
    rep = {"schema": "gen33-deep-build/1",
           "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "law": {"用户令": "现在应该开始直接训练模型，并批卷",
                   "预注册": "scratch/ccp/gen33/V33_DESIGN.md"},
           "parent_ledger": led,
           "corpus": {"path": a.corpus, "md5": C.md5f(a.corpus), "n_rows": len(rows),
                      "chain_hist": dict(sorted(ch.items())), "by_family": dict(fam),
                      "by_cls": dict(clsh), "n_steps": n_step,
                      "resid_nonzero": "%d/%d" % (n_nz, n_step)},
           "split": {"rule": "md5(name)[:8] % 10 == 7 -> a_held",
                     "a_train": ntr, "a_held": nheld,
                     "train_by_cls": dict(bcls_tr), "held_by_cls":
                     dict(collections.Counter(r["cls"] for r in held))},
           "vocab": {"path": a.vocab, "md5": C.md5f(a.vocab), "head": v4["head"],
                     "n_vals": v4["n_vals_ext"], "END": ctx["END"], "MS": ctx["MS"]},
           "cache": {"path": a.out_cache, "md5": md5c, "segments": segidx["segments"],
                     "roundtrip_ok": bool(roundtrip), "roundtrip_checks": chk,
                     "L": {"max": Ls[-1], "p50": Ls[len(Ls) // 2],
                           "mean": round(sum(Ls) / len(Ls), 1)},
                     "labels_max": max(len(r["labels"]) for r in recs),
                     "t4": [list(x) for x in t4set],
                     "slot_sup_rate": round(stats["slots_sup"] / max(1, stats["slots_all"]), 4),
                     "unk_fallback": stats["unk_fallback"],
                     "oob_fallback": stats["oob_fallback"],
                     "frames": stats["frames"],
                     "slot_unk_by_frame": dict(sup_unk.most_common(8))},
           "device": {"verify_vs_gold": dv, "frame_gate": frame_gate,
                      "entry": "gen33_device (装置现算, 禁止读金标链状态)"},
           "render": {"path": a.out_render, "md5": md5s, "n": len(side),
                      "token_sum": rtok.get("n"), "token_max": rtok.get("max"),
                      "layout": "[M_TASK] + tok(prompt) + tok(target); 损失仅 target 段",
                      "roundtrip_checks": chk_r,
                      "prompt_tmpl_md5": C.md5s(PROMPT_TMPL)},
           "wall_s": round(time.time() - t0, 2)}
    rep["md5"] = C.save_json(a.out_report, rep)
    log("BUILD DONE md5=%s wall=%.1fs" % (rep["md5"], rep["wall_s"]), a.log)
    C.heartbeat("build", "cache_md5=%s" % md5c)
    print("GEN33_BUILD_RC=0")
    return rep


if __name__ == "__main__":
    main()

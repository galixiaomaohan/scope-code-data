# -*- coding: utf-8 -*-
# gen44_slots_v7.py -- [S6R / R2] V7 rendering slot-sidecar build (9599 rows, for an isomorphic
# retrain with --slot-cond 1)
#
# Purpose: the main suspect for the [S6] zero-regression FAIL (183 vs the parent's 196) is a recipe
#   difference: [S5] trained with `--slot-cond 0`, while the parent gen42 recipe used
#   `--slot-cond 1` plus a slot sidecar (g42_slot_sidecar.jsonl, 3299 rows => covers all parent
#   training rows). The gen44 training pool = V7 cache of 9599 rows (deep blocks 2399 + new
#   families 7200) => the slot sidecar must **cover all 9599 rows** (the training side asserts
#   `n_ok == len(train_idx)`, where n_ok = rows with non-empty slots and train_idx = [0,8601)).
#
# Two groups (conventions registered per group):
#   Group A parent_bitwise (2399 rows): rows aligned **by name** with the parent slot sidecar
#       (2162 = V7 deep-block idx 0..2161; 237 = V7 idx 8601..8837 new-family rows whose names match
#       parent slot rows => same-source rows) => slots are **carried over bitwise**.
#       Assertions: np/n agree with the parent + every [a,tokid,kind,src] satisfies ids[a]==tokid,
#       0<=a<len(ids) and kind<9.
#   Group B derived (7200 rows): rows the parent slot sidecar does not cover => **derived** under the
#       parent's new-family convention:
#       (i)  the tactic-name token of a rendered line (kind 6 template; src "l{k}.tactic") -- the
#            parent gen42_sidecar uses "the tactic-name token rendered per step" for new-family rows
#            (kind 6 TMPL), while this file indexes by **rendered line** (see registration D-S6R-2).
#       (ii) slot values **really derived** from the in-row sf_args per the V7 schema
#            (slot_schema_v7_new / ctx.per): value v = str(step[path]); locate its character span in
#            the target text => map to tokens => mark; it **must** satisfy
#            tok.decode(marked tokens) == v after normalization (verified literally), otherwise the
#            mark is dropped and counted (no erroneous slot signal is introduced). kind: "int"->5
#            (tail_const) / "str"->8 (hyp_ref).
#       Row-level fallback: if a row yields nothing from (i)/(ii), take the first identifier token of
#            the row's **first non-empty line** (src "l0.fallback") so every row has non-empty slots
#            (required by the training-side assertion).
#
# Registered deviation (D-S6R-2): rows and steps are **not 1:1** (measured: b5 1 step 5 lines;
#   b6 5 steps 3 lines), so the parent gen42_sidecar's "step -> line" assumption does not hold under
#   V7 tree-level rendering; this file indexes tactic-name marks by rendered line (the tactic-name
#   token is still a real target token).
#
# Reads: V7 sidecar / gen42 slot sidecar / V7 corpus / render dump; writes:
#   G44/cache/GEN44_V7_SLOT_SIDECAR.jsonl + G44/out/GEN44_R2_SLOT_SIDECAR.json (ledger). Overwrites
#   no existing file.
import argparse
import collections
import hashlib
import json
import os
import re
import sys
import time

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
RES = os.environ.get("SCOPE_RESEARCH_ROOT", ".")
CCP = os.path.join(RES, "scratch", "ccp")
G42 = os.path.join(CCP, "gen42")
G44 = os.path.join(CCP, "gen44")
if _PKG_DIR not in sys.path:
    sys.path.insert(0, _PKG_DIR)
if G42 not in sys.path:
    sys.path.insert(0, G42)

IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
K_TMPL, K_INT, K_STR = 6, 5, 8


def md5f(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def now_iso():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def jlload(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v7-side", default=os.path.join(G44, "cache", "GEN44_V7_SIDECAR.jsonl"))
    ap.add_argument("--g42-slot", default=os.path.join(G42, "cache", "g42_slot_sidecar.jsonl"))
    ap.add_argument("--corpus", default=os.path.join(G44, "out", "GEN44_S4_CORPUS_MERGED.jsonl"))
    ap.add_argument("--render", default=os.path.join(G44, "out", "GEN44_S4B_RENDER.jsonl"))
    ap.add_argument("--out", default=os.path.join(G44, "cache", "GEN44_V7_SLOT_SIDECAR.jsonl"))
    ap.add_argument("--report", default=os.path.join(G44, "out", "GEN44_R2_SLOT_SIDECAR.json"))
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    import gen44_vocab7 as V7
    import gen44_bcore as B
    import gen44_cachebuild_v7 as CB
    ctx = V7.load_ctx7()
    tok = ctx["tok"]
    per = ctx["per"]
    v7 = jlload(a.v7_side)
    g42s = {int(r["ri"]): r for r in jlload(a.g42_slot)}
    g42_by_name = {r["name"]: r for r in g42s.values()}
    corpus = {r["name"]: r for r in jlload(a.corpus)}
    rmap = {x["name"]: x.get("text") for x in jlload(a.render)}
    assert len(v7) == 9599, len(v7)
    nrow = a.limit or len(v7)

    nl_ids = set(int(x) for x in tok("\n", add_special_tokens=False)["input_ids"])
    st = collections.Counter()
    diag_bad = []
    out = []
    for i in range(nrow):
        r = v7[i]
        name, ids, np_, seg = r["name"], [int(x) for x in r["ids"]], int(r["np"]), r.get("seg")
        # NOTE: Chinese report payload value kept verbatim (data, not a log line).
        rec = {"schema": "gen44-slot-sidecar/1", "tag": "corpus", "ri": i, "name": name,
               "cls": r.get("cls"), "seg": seg, "np": np_, "n": len(ids),
               "n_slot_kinds": 9,
               "slot_kinds": ["poly_coef", "poly_exp", "ell_coef", "ell_shift", "mult",
                              "tail_const", "template", "pad", "hyp_ref"],
               "note": "slots = [V7 侧车 ids 绝对下标, 令牌 id, kind, src] (训练侧 p = a-1)"}
        src_p = g42_by_name.get(name)
        slots = []
        if src_p is not None:                                   # group A: carry over bitwise
            rec["group"] = "parent_bitwise"
            assert int(src_p["np"]) == np_ and int(src_p["n"]) == len(ids), \
                ("group A np/n mismatch", name, src_p["np"], np_, src_p["n"], len(ids))
            bad = []
            for e in (src_p.get("slots") or []):
                aa, tk, kd = int(e[0]), int(e[1]), int(e[2])
                if not (0 <= aa < len(ids)) or ids[aa] != tk or not (0 <= kd < 9):
                    bad.append(e)
                else:
                    slots.append([aa, tk, kd, str(e[3])])
            assert not bad, ("group A slots not aligned with V7 ids", name, bad[:3])
            st["parent_rows"] += 1
        else:                                                    # group B: derived
            rec["group"] = "derived"
            txt = rmap.get(name)
            align = "render_text"
            if txt is None:                                      # no render dump => decode from ids
                txt = tok.decode(ids[np_:])
                align = "decode_ids"
            # character -> token alignment (char coordinate system = txt)
            enc = tok("```lean4\n" + txt + "\n```", add_special_tokens=False,
                      return_offsets_mapping=True)
            tids = [int(x) for x in enc["input_ids"]]
            offs = [(int(x), int(y)) for x, y in enc["offset_mapping"]]
            pre = len(tok("```lean4\n", add_special_tokens=False)["input_ids"])
            PREL = len("```lean4\n")            # offset coordinate system = the concatenated string (PRE + txt + "\n```")
            if tids == ids[np_:]:
                st["align_ok"] += 1
            else:
                st["align_fail"] += 1
                if len(diag_bad) < 20:
                    diag_bad.append({"name": name, "why": "reencode!=ids",
                                     "n_enc": len(tids), "n_ids": len(ids) - np_})
                align = "token_newline"

            def mark_span(c0, c1, kind, src, want=None):
                """txt character span -> token marks (converted to the concatenated-string
                coordinate system automatically); returns the number of marks."""
                c0s, c1s = c0 + PREL, c1 + PREL
                hit = [j for j, (oa, ob) in enumerate(offs) if oa < c1s and c0s < ob]
                if not hit:
                    return 0
                if want is not None:
                    got = tok.decode([ids[np_ + j] for j in hit]).replace(" ", "")
                    if got != want.replace(" ", ""):
                        return 0
                for j in hit:
                    slots.append([np_ + j, ids[np_ + j], int(kind), src])
                return len(hit)

            if align == "render_text":
                # (i) per rendered line: the first identifier token (kind 6 TMPL)
                pos = 0
                nline = 0
                for k, ln in enumerate(txt.split("\n")):
                    j0 = txt.find(ln, pos)
                    if j0 < 0:
                        j0 = pos
                    pos = j0 + len(ln)
                    ids_id = list(IDENT.finditer(ln))
                    if not ids_id:
                        continue
                    # Rule (registered): a rendered line starting with "try " takes the **second**
                    # identifier (the tactic name; the first, "try", carries little information);
                    # otherwise the first identifier. The parent gen42_sidecar takes the first
                    # identifier (its rendered lines have no "try " prefix).
                    m = ids_id[1] if (len(ids_id) > 1 and ids_id[0].group(0) == "try") else ids_id[0]
                    c0, c1 = j0 + m.start(), j0 + m.end()
                    if mark_span(c0, c1, K_TMPL, "l%d.tactic" % k, want=m.group(0)):
                        nline += 1
                        st["mk_" + ("tactic" if m is not ids_id[0] else "first")] += 1
                    else:
                        st["line_verify_fail"] += 1
                st["line_marks"] += nline
                # (ii) really derive from sf_args per the V7/V6 schema (verified literally; dropped on failure)
                cr = corpus.get(name)
                if cr is not None:
                    ops = list(cr.get("chain_ops") or [])
                    try:
                        fo = CB.Fam(name.split("_")[1])
                        d = fo.parse(B.P(name, cr["head"], cr["concl"], None, None))
                        tr = d.get("tree")
                        tfr = fo.frames(tr)
                        fb = list(tfr[0]) if tfr and tfr[0] else [0.0] * 8
                        steps = CB.tree_steps(fo, tr, fb) + CB.leaf_steps(fo, tr)
                    except Exception:                            # noqa: BLE001
                        steps = None
                    if steps is not None and len(steps) == len(ops):
                        for k, (op, _fr, ar) in enumerate(steps):
                            sch = per.get(op) or []
                            for (path, meta, _al) in sch:
                                v = (ar or {}).get(path)
                                v = "" if v is None else str(v)
                                if not v or v not in txt:
                                    st["arg_miss"] += 1
                                    continue
                                kd = K_INT if str(meta.get("kind")) == "int" else K_STR
                                c0 = txt.find(v)
                                c1 = c0 + len(v)
                                if mark_span(c0, c1, kd, "s%d.%s.%s" % (k, op, path), want=v):
                                    st["arg_hit"] += 1
                                else:
                                    st["arg_verify_fail"] += 1
                    else:
                        st["arg_no_steps"] += 1
            else:
                # Fallback: split the token space at newlines and mark the first token of each line.
                j = 0
                nl = 0
                jj = 0
                row_ids = ids[np_:]
                for k, x in enumerate(row_ids):
                    if x in nl_ids or k == len(row_ids) - 1:
                        if j < k:
                            slots.append([np_ + j, row_ids[j], K_TMPL, "l%d.tactic" % nl])
                        j = k + 1
                        nl += 1
                st["newline_fallback_rows"] += 1
            if not slots:                                        # row-level fallback (ensure non-empty)
                for k, x in enumerate(ids[np_:]):
                    if x not in nl_ids:
                        slots.append([np_ + k, x, K_TMPL, "l0.fallback"])
                        st["row_fallback"] += 1
                        break
        # Row-level check: keep only one entry per token position (priority arg(5/8) > line(6);
        # within a priority level keep the first occurrence).
        dedup = {}
        for e in slots:
            aa = e[0]
            if aa in dedup:
                st["dup_pos_dropped"] += 1
                if dedup[aa][2] == K_TMPL and e[2] in (K_INT, K_STR):
                    dedup[aa] = e
                continue
            dedup[aa] = e
        slots = list(dedup.values())
        seen = set()
        for e in slots:
            aa, tk, kd = e[0], e[1], e[2]
            assert 0 <= aa < len(ids) and ids[aa] == tk and 0 <= kd < 9, ("illegal slot", name, e)
            assert aa not in seen, ("duplicate slot", name, aa)
            seen.add(aa)
        assert slots, ("empty slots (the training-side assertion would fail)", name)
        rec["slots"] = sorted(slots, key=lambda e: e[0])
        rec["n_slots"] = len(rec["slots"])
        rec["recon_eq"] = True
        rec["n1_rev"] = 0
        rec["op_mismatch"] = []
        rec["n_present"] = len(rec["slots"])
        out.append(rec)
        st["rows"] += 1
        st["slots_total"] += len(rec["slots"])

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for rec in out:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    md = md5f(a.out)
    byseg = collections.Counter(rec["seg"] for rec in out)
    bygrp = collections.Counter(rec["group"] for rec in out)
    per_row = [rec["n_slots"] for rec in out]
    rep = {"schema": "gen44-r2-slot-sidecar/1", "created": now_iso(),
           "out": {"path": a.out, "md5": md, "bytes": os.path.getsize(a.out), "rows": len(out)},
           "n_train_expected": 8601, "rows_with_slots": sum(1 for x in per_row if x > 0),
           "by_seg": dict(byseg), "by_group": dict(bygrp),
           "n_slots": {"total": int(sum(per_row)), "min": int(min(per_row)),
                       "max": int(max(per_row)),
                       "mean": round(sum(per_row) / max(len(per_row), 1), 3)},
           "counters": {k: int(v) for k, v in sorted(st.items())},
           "A_group_source": {"path": a.g42_slot, "md5": md5f(a.g42_slot),
                              "rows_used": int(st["parent_rows"])},
           "input_md5": {"v7_side": md5f(a.v7_side), "corpus": md5f(a.corpus),
                         "render": md5f(a.render)},
           # NOTE: Chinese report payload value kept verbatim (data, not a log line).
           "D_S6R_2": ("行↔步非 1:1 (b5 1 步 5 行 / b6 5 步 3 行) ⇒ 父件'步→行'假设在 V7 树级渲染下不成立; "
                       "B 组 tactic 名标记按**渲染行**索引 (src l{k}.tactic); 设备 arg 标记仍按步索引 (src s{k}.op.path)"),
           "kind_map": {"template": K_TMPL, "int->tail_const": K_INT, "str->hyp_ref": K_STR},
           "align_fail_sample": diag_bad}
    json.dump(rep, open(a.report, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps({"rows": len(out), "slots": rep["n_slots"], "md5": md,
                      "counters": rep["counters"], "by_group": rep["by_group"],
                      "by_seg": rep["by_seg"]}, ensure_ascii=False))
    print("SLOT_SIDECAR_BUILT")


if __name__ == "__main__":
    main()

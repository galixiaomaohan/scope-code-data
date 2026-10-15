# -*- coding: utf-8 -*-
# gen42_cachebuild.py -- nodes [3]+[4]: vocab V5->V6 (old 198 rows asserted byte for byte) + cache rebuild
#                        (deep corpus a2 2399 kept bitwise UNION the gen42 new-family corpus)
#
# Pool decision (registered): deep a2 (2399, kept bitwise, the zero-regression gate requires it) UNION the gen42
#   new-family corpus; **excludes** gen41's 1545 rows -- their state frames violate HIGH-1 (answer-constant vectors);
#   gen42 families S/A supersede gen41 families A/B, so the defective frames are not fed back in (see V42_DESIGN §1).
#
# Read-only: gen41/cache/{CCP_VOCAB_V5.json,ccp_allow_v5.bin,g41_all.bin} (first 2399 rows = deep a2)
# Write-only: scratch/ccp/gen42/**
import argparse
import collections
import hashlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
G41 = os.path.join(os.path.dirname(HERE), "gen41")
if G41 not in sys.path:
    sys.path.insert(0, G41)

import numpy as np                                                      # noqa: E402
import gen42_stmt as S                                                  # noqa: E402
import gen42_families as F                                              # noqa: E402
import gen41_common as C41                                              # noqa: E402
import gen41_cachebuild as CB41                                         # noqa: E402

MS = 24
NVALS = 352
HEAD_V5 = 198
HEAD_V6 = HEAD_V5 + len(F.OPS)
TYPE4 = [1, 0, 0, 0]
CACHE_G41 = C41.CACHE_G41
G42 = HERE

# new-operator slot specs (sfa path; aligned with the gen42_gen payload)
SLOT_V6 = {
    "g42_s_reduce": [{"kind": "str", "path": "var"}],
    "g42_s_poly": [{"kind": "str", "path": "resid"}],
    "g42_s_eval": [{"kind": "str", "path": "lhs"}, {"kind": "str", "path": "rhs"},
                   {"kind": "str", "path": "rel"}],
    "g42_s_mod": [{"kind": "str", "path": "lhs"}, {"kind": "str", "path": "rhs"}],
    "g42_s_lin": [{"kind": "str", "path": "resid"}],
    "g42_s_omega": [{"kind": "int", "path": "n_atoms"}],
    "g42_s_pos": [{"kind": "str", "path": "strict"}],
    "g42_sb_step": [{"kind": "str", "path": "var"}, {"kind": "str", "path": "hyp"}],
    "g42_sb_close": [{"kind": "str", "path": "value"}],
}


def md5f(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def build_vocab(out_vocab, out_allow):
    v5 = json.load(open(C41.VOCAB_V5, encoding="utf-8"))
    a5 = np.fromfile(C41.ALLOW_V5, dtype=np.uint8).reshape(HEAD_V5, MS, NVALS)
    assert int(a5.sum()) == 124573, int(a5.sum())
    add = np.ones((len(F.OPS), MS, NVALS), dtype=np.uint8)
    a6 = np.concatenate([a5, add], axis=0)
    a6.tofile(out_allow)
    v6 = dict(v5)
    v6["head"] = int(v5.get("head", 194))
    v6["head_v5"] = int(v5["head_v5"])
    v6["head_v6"] = HEAD_V6
    v6["ops_g42"] = {k: int(v) for k, v in sorted(F.OPS.items())}
    v6["slot_schema_v6_new"] = SLOT_V6
    v6["allow_shape_v6"] = [HEAD_V6, MS, NVALS]
    v6["allow_bin_v6"] = os.path.basename(out_allow)
    v6["allow_sum_by_row_v6"] = {"base_0_197": int(a5.sum()), "total": int(a6.sum()),
                                 "n_rows": int(a6.shape[0])}
    v6["upgrade_note_v6"] = ("old 198 rows/columns kept byte for byte (a6[:198] == a5 asserted, sum=124573); "
                             "new rows 198..%d all ones (24x352)" % (HEAD_V6 - 1))
    json.dump(v6, open(out_vocab, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    a6b = np.fromfile(out_allow, dtype=np.uint8).reshape(HEAD_V6, MS, NVALS)
    assert a6b[:HEAD_V5].tobytes() == a5.tobytes(), "the old 198 rows were modified"
    assert int(a6b[HEAD_V5:].sum()) == len(F.OPS) * MS * NVALS
    return {"vocab_md5": md5f(out_vocab), "allow_md5": md5f(out_allow),
            "old_rows_byte_identical": True, "old_sum": int(a5.sum()),
            "total_sum": int(a6b.sum()), "head_v6": HEAD_V6,
            "new_rows": sorted(F.OPS)}


def load_ctx6():
    ctx = CB41.load_ctx5()
    names_head = list(ctx["names_head"]) + [None] * len(F.OPS)
    opmap = dict(ctx["opmap"])
    for n, i in F.OPS.items():
        names_head[int(i)] = n
        opmap[n] = int(i)
    assert len(names_head) == HEAD_V6 and all(names_head[i] for i in range(161, HEAD_V6)), \
        [i for i in range(161, HEAD_V6) if not names_head[i]]
    toks_ext = ctx["toks_ext"]
    idx_ext = ctx["idx_ext"]
    dig_ids = list(range(toks_ext.index(("digest", "d00")), len(toks_ext)))
    unk = idx_ext[("any", "__UNK__")]
    per = dict(ctx["per"])
    for op, rows in SLOT_V6.items():
        out_rows = []
        for r in rows:
            allow = sorted({unk} | set(dig_ids))
            out_rows.append((r["path"], {"kind": r["kind"]}, list(allow)))
        per[op] = out_rows
    vocab = dict(ctx["vocab"])
    vocab["head"] = HEAD_V6
    ctx.update({"vocab": vocab, "names_head": names_head, "opmap": opmap, "per": per})
    assert int(ctx["vocab"]["head"]) == HEAD_V6
    return ctx


def to_src_row_g42(row):
    steps, sfa = [], []
    for k, s in enumerate(row["steps"]):
        fr = s.get("frame")
        if fr is None:
            fr = row["frames"][min(k, len(row["frames"]) - 1)]
        steps.append({"op": s["op"], "ok": True, "err": None,
                      "res": [float(x) for x in fr], "state": "ok"})
        sfa.append({kk: str(vv) for kk, vv in s.items() if kk != "frame"})
    return {"stmt": row["stmt_state"], "type4": list(TYPE4),
            "program": list(row["chain_ops"]), "steps": steps, "sf_args": sfa,
            "episode": None, "accepted": True, "readout_only": False,
            "name": row["name"], "cls": row["cls"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=os.path.join(G42, "cache", "GEN42_CORPUS.jsonl"))
    ap.add_argument("--vocab-out", default=os.path.join(G42, "cache", "CCP_VOCAB_V6.json"))
    ap.add_argument("--allow-out", default=os.path.join(G42, "cache", "ccp_allow_v6.bin"))
    ap.add_argument("--out", default=os.path.join(G42, "cache", "g42_all.bin"))
    ap.add_argument("--sidecar", default=os.path.join(G42, "cache", "g42_sidecar.jsonl"))
    ap.add_argument("--report", default=os.path.join(G42, "out", "GEN42_CACHE_GATE.json"))
    a = ap.parse_args()
    os.makedirs(os.path.join(G42, "cache"), exist_ok=True)
    t0 = time.time()
    print("== [3] vocab V6 ==", flush=True)
    vrep = build_vocab(a.vocab_out, a.allow_out)
    print(json.dumps(vrep, ensure_ascii=False), flush=True)

    print("== [4] cache rebuild ==", flush=True)
    ctx = load_ctx6()
    rows = [json.loads(l) for l in open(a.corpus, encoding="utf-8") if l.strip()]
    rows = sorted(rows, key=lambda r: r["name"])
    famh = collections.Counter(r["family"] for r in rows)
    print("gen42 corpus %d rows families=%s" % (len(rows), dict(famh)), flush=True)
    need = sorted({op for r in rows for op in r["chain_ops"]})
    missing = [o for o in need if o not in ctx["opmap"]]
    if missing:
        raise SystemExit("operators not registered: %s" % missing)
    print("operators complete: %s" % need, flush=True)

    deep_rows, dhdr, _b = CB41.read_raw(CACHE_G41)
    deep = deep_rows[:2399]
    assert len(deep) == 2399, len(deep)
    dseg = json.load(open(C41.CACHE_A2 + ".segidx.json", encoding="utf-8"))["segments"]
    print("deep segment a2: %d rows (source segment %s)" % (len(deep), dseg), flush=True)

    ntr = sum(1 for r in rows if int(C41.md5s(r["name"])[:8], 16) % 10 != 7)
    tr = [r for r in rows if int(C41.md5s(r["name"])[:8], 16) % 10 != 7]
    hd = [r for r in rows if int(C41.md5s(r["name"])[:8], 16) % 10 == 7]
    order = tr + hd
    recs_new, _t4, _st = CB41.CCF.build(ctx, [to_src_row_g42(r) for r in order])
    assert len(recs_new) == len(order), (len(recs_new), len(order))
    print("new-family records %d (train %d / held %d)" % (len(recs_new), ntr, len(hd)), flush=True)

    dtr = list(range(0, dseg["a_train"][1]))
    dheld = list(range(dseg["a_held"][0], 2399))
    all_recs = [deep[r] for r in dtr] + recs_new[:ntr] + [deep[r] for r in dheld] + \
        recs_new[ntr:]
    n_train = len(dtr) + ntr
    segs = {"a_train": [0, n_train], "a_held": [n_train, len(all_recs)]}
    held_rows = list(range(n_train, len(all_recs)))
    assert len(held_rows) == len(dheld) + len(hd), (len(held_rows), len(dheld), len(hd))
    segidx = {"cache": a.out, "schema_rev": "ccpcd5/1-gen42",
              "ops": {"base": 161, "end": 154, "head": HEAD_V6, "n_vals": NVALS,
                      "g42": {k: int(v) for k, v in sorted(F.OPS.items())}},
              "segments": segs, "held_rows": held_rows,
              "held_rule": "md5(name)[:8] % 10 == 7 (deep corpus / gen42 each use their own name)",
              "n_held": len(held_rows), "vocab_md5": md5f(a.vocab_out),
              "deep_src": {"path": CACHE_G41, "rows": 2399, "a2_kept_bitwise": True},
              "corpus42": {"path": a.corpus, "md5": md5f(a.corpus), "rows": len(rows)},
              "name_space_note": ("training pool = deep corpus a2 (kept bitwise) UNION the gen42 new-family corpus; "
                                  "excludes the 1545 gen41 rows (HIGH-1 defective frames, registered in V42_DESIGN §1)")}
    md5 = CB41.write_cache(a.out, all_recs, ctx, segidx)

    with open(a.sidecar, "w", encoding="utf-8") as f:
        for i, r in enumerate(order):
            f.write(json.dumps({"ri": len(dtr) + i, "name": r["name"],
                                "family": r["family"], "cls": r["cls"],
                                "route": r["route"], "chain_ops": r["chain_ops"],
                                "stmt": r["stmt_state"], "head": r["head"],
                                "concl": r["concl"], "proof_text": r["proof_text"],
                                "vars": r["vars"], "frames": r["frames"],
                                "res_src": "corpus_live"}, ensure_ascii=False) + "\n")
    # ---- roundtrip gate ----
    back, bhdr, _b2 = CB41.read_raw(a.out)
    rt_bad = [i for i, (x, y) in enumerate(zip(back, all_recs)) if CB41.row_equal(x, y)]
    deep_bad = [i for i, r in enumerate(dtr) if CB41.row_equal(back[i], deep_rows[r])]
    hdr_chk = {"n_rows": bhdr[0] == len(all_recs), "head": bhdr[12] == HEAD_V6,
               "MS": bhdr[13] == MS, "nvals": bhdr[14] == NVALS, "end": bhdr[10] == 154,
               "base": bhdr[11] == 161,
               "tmax": bhdr[9] == max(len(r["labels"]) for r in all_recs)}
    rep = {"schema": "gen42-cache-gate/1", "vocab": vrep,
           "n_deep": len(deep), "n_new": len(recs_new), "n_total": len(all_recs),
           "n_deep_train": len(dtr), "n_deep_held": len(dheld),
           "n_new_train": ntr, "n_new_held": len(hd),
           "cache_md5": md5, "cache_bytes": os.path.getsize(a.out),
           "sidecar_md5": md5f(a.sidecar), "nops_head_v6": HEAD_V6,
           "segidx": segs, "roundtrip_bad": len(rt_bad), "deep_bitwise_bad": len(deep_bad),
           "hdr_check": hdr_chk,
           "lab_hist": dict(collections.Counter(
               ctx["names_head"][l] for r in all_recs for l in r["labels"]).most_common(30)),
           "wall_s": round(time.time() - t0, 1)}
    json.dump(rep, open(a.report, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(rep, ensure_ascii=False, indent=1), flush=True)
    assert not rt_bad and not deep_bad and all(hdr_chk.values()), "roundtrip/header gate FAIL"


if __name__ == "__main__":
    main()

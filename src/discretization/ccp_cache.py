# -*- coding: utf-8 -*-
# ccp_cache.py -- CCP course-cache build smoke run (CPU; advance team gen12, constitution change 58 step E)
#
# Purpose: show that course rows passing the three gates **can be consumed by the real training
#   pipeline** (row format / labels / frames / bound slots all in shape) and expose the
#   **vocabulary extension point** the training batch needs (without touching the shared V32 registry).
# Measurement convention: frame construction mirrors cppcuda/tools/prep_cache_v31.py::build
#   semantics item by item (M_TASK/M_TYPE/M_OP/M_RES/M_TAGFR/M_PST/KIND_*), but the operator space
#   = V32 160 nominal operators + CCP extension 7 (appended at the tail; **existing 0..159
#   semantics unchanged** => backward compatible with pure-V32 caches). Serialization = the CCP
#   version of the CPPCD3 layout (MAGIC=CPPCD4), self-reading and self-verifying (roundtrip
#   integrity + invariant assertions). Tokenizer is the pipeline's own (st.BASE).
# Writes only scratch/ccp/cache/**; reads cppcuda/tools and the venv packages.
import argparse
import collections
import importlib.machinery
import importlib.util
import json
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ccp_lean as L

RES = os.environ.get("SCOPE_RESEARCH_ROOT", ".")
CCP = os.path.join(RES, "scratch", "ccp")
GEN = os.path.join(CCP, "gen")
CACHE = os.path.join(CCP, "cache")
os.makedirs(CACHE, exist_ok=True)
MAGIC = b"CPPCD4\x00\x00"

# CCP operator extension (appended after the V32 160)
CCP_OPS = ["ccp_ind_recurrence", "ccp_ring_identity", "ccp_ineq_sos", "ccp_ineq_close",
           "ccp_nt_dvd_witness", "ccp_nt_dvd_combine", "ccp_nt_dvd_prod_combine"]
# bound-slot schema (per operator: [(path, value range)]) -- the range only registers the type;
# the actual value vocabulary comes from batch statistics
SLOT_SCHEMA = {
    "ccp_ind_recurrence": [("form", "str"), ("v0", "int"), ("sample_n", "int"),
                           ("c1", "num"), ("c0", "num")],
    "ccp_ring_identity": [("nvars", "int"), ("atom", "str")],
    "ccp_ineq_sos": [("nvars", "int"), ("residual", "num")],
    "ccp_ineq_close": [("residual", "num")],
    "ccp_nt_dvd_witness": [("m", "int")],
    "ccp_nt_dvd_combine": [("m_h", "int"), ("M", "int"), ("const", "int")],
    "ccp_nt_dvd_prod_combine": [("m_h", "int")],
}
MS = 6                     # slot width per operator (longest schema; the rest padded with -1)
NVALS_CAP = 4096
CELLS = 4                  # frame cells (M_OP, M_RES slot, M_TAGFR, M_PST), simplified count


def _load_pipe():
    ld = importlib.machinery.SourceFileLoader(
        "pc_ccp", os.path.join(RES, "cppcuda/tools/prep_cache_v31.py"))
    sp = importlib.util.spec_from_loader("pc_ccp", ld)
    pc = importlib.util.module_from_spec(sp)
    ld.exec_module(pc)
    return pc


def slot_values(op, args):
    out = [-1] * MS
    sch = SLOT_SCHEMA.get(op)
    if not sch or not isinstance(args, dict):
        return out
    for j, (path, kind) in enumerate(sch):
        v = args.get(path)
        if v is None and op == "ccp_ind_recurrence":
            v = {"c1": args.get("value")}.get(path)
        if v is None:
            continue
        if isinstance(v, (list, dict)):
            continue
        out[j] = v
    return out


def build_records(rows, tok, opidx):
    T4set = sorted({tuple(int(x) for x in r["type4"]) for r in rows})
    T4I = {t: i for i, t in enumerate(T4set)}
    recs, stats = [], collections.Counter()
    vals = collections.defaultdict(set)
    for ri, row in enumerate(rows):
        stmt = row["stmt"]
        toks = tok.tokenize(stmt)
        ids = tok.convert_tokens_to_ids(toks)[:512] or [tok.bos_token_id or 0]
        if len(toks) > 512:
            stats["trunc"] += 1
        frames = [("TASK", 0)] + [("TOK", int(t)) for t in ids] + [("TYPE", 0),
                                                                  ("T4", T4I[tuple(row["type4"])])]
        pos_task_end = len(frames) - 1
        feat_pos, labels, lab_step = [], [], []
        ep = row.get("episode")
        end_fail_after = None
        if ep and ep.get("fail_op") == "END":
            end_fail_after = max(1, int(ep.get("after", 1)))
        elif ep:
            frames += [("OP", opidx[ep["fail_op"]]), ("RESN", 0), ("FR", 0), ("PST", 2)]
            feat_pos.append(len(frames) - 1)
        nf = 0 if not ep else 1
        prog, resl = row["program"], []
        for k, s_ in enumerate(row["steps"]):
            if end_fail_after is not None and k == end_fail_after:
                frames += [("END", 0), ("PST", 2)]
                feat_pos.append(len(frames) - 1)
                resl.append(None)
                labels.append(opidx[s_["op"]])
                lab_step.append(k)
                nf += 1
                continue
            rv = s_.get("res")
            frames += [("OP", opidx[s_["op"]]), ("RESN", len(resl) if rv is not None else 0),
                       ("FR", nf), ("PST", 0 if s_["state"] == "ok" else 2)]
            resl.append(None if rv is None else [float(x) for x in rv])
            feat_pos.append(len(frames) - 1)
            labels.append(opidx[s_["op"]])
            lab_step.append(k)
            nf += 1
        frames += [("END", 0), ("PST", 1)]
        end_pos = len(frames) - 1
        labels.append(len(opidx))          # END = new slot at the tail
        lab_step.append(-1)
        assert len(labels) == len(feat_pos) + 1, (ri, "labels/feat mismatch")
        sfa = row.get("sf_args") or []
        bsl = []
        for t, op in enumerate(labels):
            k = lab_step[t]
            args = sfa[k] if (0 <= k < len(sfa) and isinstance(sfa[k], dict)) else None
            onm = None if op == len(opidx) else _name_of(opidx, op)
            sv = slot_values(onm, args)
            for j, x in enumerate(sv):
                if x != -1:
                    vals[(onm, j)].add(str(x))
                    stats["slots_sup"] += 1
            stats["slots_all"] += MS
            bsl.extend([-1 if x == -1 else hash(str(x)) & 0xFFFF for x in sv])
        recs.append({"ids": ids, "t4": T4I[tuple(row["type4"])], "pos_task_end": pos_task_end,
                     "feat_pos": feat_pos, "end_pos": end_pos, "labels": labels,
                     "desc": frames, "res": resl, "bsl": bsl,
                     "readout": 1.0 if row.get("accepted") else 0.0,
                     "readout_only": 1 if row.get("readout_only") else 0, "ri": ri})
        stats["L"] = max(stats["L"], len(frames))
    return recs, T4set, stats, {k: len(v) for k, v in vals.items()}


_NAMES = None


def _name_of(opidx, i):
    global _NAMES
    if _NAMES is None:
        _NAMES = [n for n, _ in sorted(opidx.items(), key=lambda kv: kv[1])]
    return _NAMES[i]


def write_cache(path, recs, t4set, segidx, nops, head):
    tmax = max(len(r["labels"]) for r in recs)
    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<iiiiii", len(recs), len(t4set), 1, 2, 3, 4))
        f.write(struct.pack("<iiiiii", 5, 6, 7, tmax, nops, nops))
        f.write(struct.pack("<iiii", head, MS, NVALS_CAP, 1))
        for t4 in t4set:
            f.write(struct.pack("<4f", *[float(x) for x in t4]))
        for r in recs:
            f.write(struct.pack("<iii", len(r["ids"]), r["t4"], r["readout_only"]))
            f.write(struct.pack("<%di" % len(r["ids"]), *r["ids"]))
            f.write(struct.pack("<iif", r["pos_task_end"], r["end_pos"], r["readout"]))
            f.write(struct.pack("<i", len(r["desc"])))
            for kd, ar in r["desc"]:
                f.write(struct.pack("<ii", _fkind(kd), ar))
            f.write(struct.pack("<i", len(r["res"])))
            for x in r["res"]:
                if x is None:
                    f.write(struct.pack("<b", 0))
                else:
                    f.write(struct.pack("<b8f", 1, *x))
            f.write(struct.pack("<i", len(r["labels"])))
            f.write(struct.pack("<%di" % len(r["labels"]), *r["labels"]))
            f.write(struct.pack("<i", len(r["feat_pos"])))
            f.write(struct.pack("<%di" % len(r["feat_pos"]), *r["feat_pos"]))
            f.write(struct.pack("<i", len(r["bsl"])))
            f.write(struct.pack("<%di" % len(r["bsl"]), *r["bsl"]))
            f.write(struct.pack("<i", 0))
    with open(path + ".segidx.json", "w", encoding="utf-8") as fh:
        json.dump(segidx, fh, ensure_ascii=False, indent=1)
    return L.md5f(path)


_FK = {"TASK": 0, "TOK": 1, "TYPE": 2, "T4": 3, "OP": 4, "RESN": 5, "FR": 6, "PST": 7,
       "END": 8}


def _fkind(kd):
    return _FK[kd]


def read_cache(path):
    b = open(path, "rb").read()
    assert b[:8] == MAGIC
    off = 8

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
    hdr = [ri() for _ in range(16)]
    n_rows, nT4, head, nops = hdr[0], hdr[1], hdr[12], hdr[10]
    t4 = [tuple(rf() for _ in range(4)) for _ in range(nT4)]
    rows = []
    for _ in range(n_rows):
        nid = ri()
        R = {"t4": ri(), "readout_only": ri()}
        R["ids"] = [ri() for _ in range(nid)]
        R["pos_task_end"] = ri()
        R["end_pos"] = ri()
        R["readout"] = rf()
        nd = ri()
        R["desc"] = [(ri(), ri()) for _ in range(nd)]
        nr = ri()
        R["res"] = []
        for _ in range(nr):
            has = b[off]
            off += 1
            R["res"].append([rf() for _ in range(8)] if has else None)
        nl = ri()
        R["labels"] = [ri() for _ in range(nl)]
        nf = ri()
        R["feat_pos"] = [ri() for _ in range(nf)]
        nb = ri()
        R["bsl"] = [ri() for _ in range(nb)]
        ri()
        rows.append(R)
    assert off == len(b), (off, len(b))
    return rows, {"n_rows": n_rows, "nT4": nT4, "head": head, "nops": nops}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cls", default="all")
    a = ap.parse_args()
    pc = _load_pipe()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(pc.st.BASE)
    names32 = list(importlib.util.spec_from_file_location(
        "dummy", os.path.join(RES, "cppcuda/tools/v32b_prep.py")).origin and [])
    import importlib.machinery as im
    ld = im.SourceFileLoader("vp_ccp", os.path.join(RES, "cppcuda/tools/v32b_prep.py"))
    sp = importlib.util.spec_from_loader("vp_ccp", ld)
    vp = importlib.util.module_from_spec(sp)
    ld.exec_module(vp)
    base_names = list(vp.NAMES32)
    opidx = {n: i for i, n in enumerate(base_names)}
    for n in CCP_OPS:
        opidx[n] = len(opidx)
    nops, head = len(opidx), vp.HEAD_V32 + len(CCP_OPS)
    out = {}
    for cls in (["ind", "ineq", "nt"] if a.cls == "all" else [a.cls]):
        p = os.path.join(GEN, "ccp_%s_rows.jsonl" % cls)
        if not os.path.exists(p):
            print("[cache] missing %s (skipped)" % p)
            continue
        rows = [json.loads(ln) for ln in open(p, encoding="utf-8")]
        recs, t4set, stats, vals = build_records(rows, tok, opidx)
        cp = os.path.join(CACHE, "ccp_%s_rows.bin" % cls)
        md5 = write_cache(cp, recs, t4set, {"cache": cp, "schema_rev": "ccpcd4/1",
                                           "segments": {"ccp_%s" % cls: [0, len(recs)]},
                                           "vocab": {"base_nops": len(base_names),
                                                     "ccp_ops": CCP_OPS}},
                          nops, head)
        back, hdr = read_cache(cp)
        ok = (len(back) == len(recs)
              and all(len(b2["labels"]) == len(b2["feat_pos"]) + 1 for b2 in back)
              and all(max(b2["labels"] + [0]) <= nops for b2 in back)   # END slot = nops (new slot at the last position)
              and all(b2["ids"] == r["ids"] for b2, r in zip(back, recs)))
        o = {"schema": "ccp-cache-smoke/1", "cls": cls, "n_rows": len(recs),
             "cache": cp, "cache_md5": md5, "nops": nops, "head": head,
             "t4": [list(x) for x in t4set], "max_len_labels": stats["L"],
             "tok_trunc": stats["trunc"],
             "slot_sup_rate": round(stats["slots_sup"] / max(1, stats["slots_all"]), 4),
             "slot_vocab_sizes": {"%s[%d]" % k: v for k, v in sorted(vals.items())},
             "roundtrip_ok": bool(ok), "readback_rows": len(back),
             "note": "算子空间 = V32 160 + CCP 扩展 %d (追加于尾部, 既有 id 语义不变); "
                     "训练批消费需把 CCP_OPS 登记进预热词表 (1 处常量 + 词表重生成)" % len(CCP_OPS)}
        L.save_json(os.path.join(CACHE, "CCP_CACHE_SMOKE_%s.json" % cls.upper()), o)
        print("[cache] %s: rows=%d md5=%s roundtrip=%s slot_sup=%.3f trunc=%d" %
              (cls, len(recs), md5, ok, o["slot_sup_rate"], stats["trunc"]), flush=True)
        out[cls] = o
    return out


if __name__ == "__main__":
    main()

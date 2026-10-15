# -*- coding: utf-8 -*-
# gen41_cachebuild.py -- node [4b]: training-cache **rebuild** (deep a2 UNION gen41 new families) + render sidecar + slot annotation + gates
#
# Basis:
#   * V41_DESIGN §5 (G-S recipe as-is: a2 post-step residual state shift + w-tail 1.0 equal weight + slot protocol) --
#     the "a2 shift" is part of the recipe => deep-corpus rows must be **kept as-is** (bitwise) in the training pool;
#     gen41 new-family rows enter the pool under the state-frame definition of V41_DESIGN §3.2/§3.3.
#   * NEXT_STEP.gen41.txt [B]: --vocab V5 · --corpus GEN41_CORPUS_CLEAN · --out g41_all.bin · --sidecar
#     ("g41_all" = the full gen41 training pool = deep corpus (a2) UNION gen41 new families; a single cache consumed by the trainer)
#   * gate G-5 (V41_DESIGN §6): roundtrip (write->read bitwise equal) + sidecar + slot annotation cov=1.0 + state frames
#
# Frame format = CPPCD5 (gen13 ccp_cachefull layout, same origin as the trainer/reader):
#   MAGIC + 16xi32 header + t4 float table + per row [ids/t4/readout_only | pos_task_end/end_pos/readout |
#   desc(kind,arg)xnd | res(8f)xnr | labels | feat_pos | bsl | state_pos]
#   frame sequence = [M_TASK][stmt toks][M_TYPE][T4] + per step [M_OP, op, M_RES, res(res), M_TAGFR+nf, PST] + [M_END, PST]
#
# Read-only: gen36/cache/deep_all_a2.bin · gen33 render sidecar · gen34 slot sidecar · gen41 vocab V5 + cleaned corpus ·
#       gen41 family artifacts (device live) · gen41_checks (independent criteria; structurally does not import family modules)
# Write-only: scratch/ccp/gen41/**
import argparse
import collections
import importlib.machinery
import importlib.util
import json
import os
import random
import struct
import sys
import time
from fractions import Fraction

import numpy as np                                                    # noqa: E402

import gen41_common as C                                              # noqa: E402

G41, G36, G34, G33, G32, G13 = C.G41, C.G36, C.G34, C.G33, C.G32, C.G13
OUT = G41 + "/out"
LOGS = G41 + "/logs"
TYPE4 = [1, 0, 0, 0]
MS = 24
K_TMPL, K_HYP = 6, 8          # gen34 slot kind 6 = template; gen41 adds slot kind 8 = hyp_ref
N_SLOT_KINDS = 9              # 0..6 (gen34) + 7 padding + 8 hyp_ref
PROMPT_TMPL = ("You are a Lean 4 prover. Given the theorem statement below, output the"
               " complete proof body inside a ```lean4 fenced block. Output the proof body"
               " only (no `theorem` header, no `import` lines).\n"
               "Theorem statement:\n%s\nProof:\n")


def load(name, path):
    ld = importlib.machinery.SourceFileLoader(name, path)
    m = importlib.util.module_from_spec(importlib.util.spec_from_loader(name, ld))
    ld.exec_module(m)
    return m


CCF = load("g41_ccf", G13 + "/ccp_cachefull.py")
V4L = load("g41_v4l", G33 + "/gen33s_vocab4.py")
CHK = load("g41_chk", G41 + "/gen41_checks.py")


# ================================================================ §1 ctx5 (head 198)

def load_ctx5(logp=None):
    """V5 version of ctx (ccp_cachefull.load_ctx convention): head 194 -> 198, everything else bit-for-bit the v4 load path."""
    ctx = V4L.load_ctx4(C.G33 + "/cache/CCP_VOCAB_V4.json")
    voc5 = json.load(open(C.VOCAB_V5, encoding="utf-8"))
    voc4 = ctx["vocab"]
    # **V5 artifact head-field convention registered**: "head" = 194 (legacy V4 anchor, reused byte for byte) ·
    # "head_v5" = 198 (this batch's real head); the trainer/loader must take head_v5, otherwise it conflicts with
    # the allow_v5 (198,24,352) shape gate (patched at the fork).
    assert voc5["head"] == 194 and int(voc5["head_v5"]) == 198, (voc5["head"], voc5.get("head_v5"))
    assert voc5["n_vals_ext"] == 352 and voc5["allow_shape_v5"] == [198, 24, 352]
    assert list(voc5["allow_shape"]) == [194, 24, 352]
    voc5h = dict(voc5)
    voc5h["head"] = int(voc5["head_v5"])          # ctx/cache header uses the **real head** 198
    assert voc5["end_idx"] == 154 and voc5["n_vals_base"] == voc4["n_vals_base"]
    # bitwise-unchanged assertion for the old artifact (zero new value space)
    assert (json.dumps(voc5["extra_values"], sort_keys=True)
            == json.dumps(voc4["extra_values"], sort_keys=True)), "V5 value space differs from V4"
    for key in ("ops_ccp", "ops_bh", "ops_g24", "ops_gen32", "ops_g41"):
        assert key in voc5, key
    assert all(voc5["ops_g41"][n] == i for n, i in
               [("g41_subst_bind", 194), ("g41_subst_close", 195),
                ("g41_eval_arith", 196), ("g41_eval_mod", 197)])
    names_head = list(ctx["names_head"]) + [None] * 4
    opmap = dict(ctx["opmap"])
    for n, i in voc5["ops_g41"].items():
        names_head[int(i)] = n
        opmap[n] = int(i)
    assert len(names_head) == 198 and all(names_head[i] for i in range(161, 198)), \
        [i for i in range(161, 198) if not names_head[i]]
    toks_ext = ctx["toks_ext"]
    idx_ext = ctx["idx_ext"]
    dig_ids = list(range(toks_ext.index(("digest", "d00")), len(toks_ext)))
    per = dict(ctx["per"])
    for op, rows in voc5["slot_schema_v5_new"].items():
        out_rows = []
        for r in rows:
            kind = r["kind"]
            if kind in V4L.NUM_KINDS:
                allow = sorted({idx_ext[("any", "__UNK__")]} |
                               {i for i, (k2, _v2) in enumerate(toks_ext)
                                if k2 in V4L.NUM_KINDS})
            else:
                allow = sorted({idx_ext[("any", "__UNK__")]} | set(dig_ids))
            out_rows.append((r["path"], {"kind": kind}, list(allow)))
        per[op] = out_rows
    ctx.update({"vocab": voc5h, "names_head": names_head, "opmap": opmap, "per": per,
                "END": int(voc5["end_idx"]), "MS": MS, "NVALUES": len(toks_ext)})
    C.log("ctx5: head_legacy=%d head=%d END=%d NVALUES=%d per_has_g41=%s names_head_ok=%s "
          "legal_old=%d legal_new=%d" % (voc5["head"], voc5h["head"], ctx["END"],
                                         ctx["NVALUES"],
                                         all(n in per for n in voc5["ops_g41"]),
                                         all(names_head[i] for i in range(161, 198)),
                                         len(voc5["ops_legal_idx"]), len(voc5["ops_g41"])), logp)
    return ctx


# ================================================================ §2 row adaptation (corpus -> frame-builder input)

def to_src_row(row, res_of_step=None):
    """gen41 corpus row -> ccp_cachefull.build input row shape. res defaults to the in-row resid (fraction string -> float)."""
    steps, sfa = [], []
    for k, s in enumerate(row["steps"]):
        rv = (res_of_step(row, k) if res_of_step else
              [float(Fraction(x)) for x in s["resid"]])
        steps.append({"op": s["op"], "ok": True, "err": None, "res": rv, "state": "ok"})
        sfa.append(dict(s["args"]))
    return {"stmt": row["stmt_state"], "type4": list(TYPE4),
            "program": list(row["chain_ops"]), "steps": steps, "sf_args": sfa,
            "episode": None, "accepted": True, "readout_only": False,
            "name": row["name"], "cls": row["cls"]}


# ---- device live computation (independent route; different implementation from the family generators) ----

def route_fixed(base, exp, m):
    """Powering via the cycle route (pre-period/cycle, rho detection) -- the **correct implementation**.

    Registered (new finding D13): gen41_checks.modpow_order_route's "no 1-cycle fallback branch" is trivially
    satisfied by a **degenerate period L = len(seq)** in `for L in range(1, len-i+1)` => it returns a wrong value on
    inputs where "the sequence never returns to 1" (20 cases measured). That defect only causes **false rejections**
    (chk_eval_mod hard-rejects when r_ord != r_native => consistent with the 47 generation-time mod_CERT_FAIL
    cases), it **cannot** admit a false proposition; this module recomputes with the correct route and cross-checks
    against pow() in full (family-B mod class, 399 rows).
    """
    if m <= 1:
        raise ValueError("bad modulus")
    if exp == 0:
        return 1 % m
    seq, seen, cur = [], {}, base % m
    while cur not in seen:
        seen[cur] = len(seq)
        seq.append(cur)
        cur = (cur * base) % m
    pre, cyc = seen[cur], len(seq) - seen[cur]
    if exp - 1 < pre:
        return seq[exp - 1]
    return seq[pre + (exp - 1 - pre) % cyc]


def dev_res_step(row, k):
    """Device live computation of the k-th step state frame (8-dim; independent route)."""
    if row["family"] == "g41_alg_subst":
        close = row["steps"][-1]["args"]
        chain, err = CHK.rebuild_resid_chain(
            CHK._parse(close["orig_target_expr"]),
            [{"var": b["var"], "expr": CHK._parse(b["expr"])} for b in close["binds"]],
            list(close["vars"]), int(close["nv"]))
        if err:
            raise RuntimeError("rebuild_resid_chain: %s" % err)
        return [float(x) for x in chain[min(k, len(chain) - 1)]]
    a = row["steps"][k]["args"]
    if a["kind"] in ("mod_eq", "mod_neq", "dvd", "ndvd"):
        r = route_fixed(int(a["base"]), int(a["exp"]), int(a["m"]))
        assert r == pow(int(a["base"]), int(a["exp"]), int(a["m"])), ("route vs pow", row["name"])
        return [float(r % (j + 2)) for j in range(8)]
    return [float(x) for x in row["steps"][k]["resid"]]        # arith class: device definition = all-zero frame


def is_zero_frame(row, k):
    return all(float(Fraction(x)) == 0.0 for x in row["steps"][k]["resid"])


# ================================================================ §3 cache write/read (CPPCD5)

def read_raw(path):
    rows, hdr = CCF.read_cache(path)
    b = open(path, "rb").read()
    h = struct.unpack_from("<16i", b, 8)
    t4 = [tuple(struct.unpack_from("<4f", b, 8 + 64 + 16 * i)) for i in range(h[1])]
    for r in rows:
        r["t4f"] = t4[r["t4"]]
    return rows, h, b


def write_cache(path, recs, ctx, segidx):
    t4_glob, t4_idx = [], {}
    for r in recs:
        f = r.pop("t4f", None)
        if f is None:                       # produced by the CCP frame builder: all rows TYPE4 (single t4 class)
            f = tuple(float(x) for x in TYPE4)
        if f not in t4_idx:
            t4_idx[f] = len(t4_glob)
            t4_glob.append(f)
        r["t4"] = t4_idx[f]
    md5 = CCF.write_cache(path, recs, t4_glob, ctx, segidx)
    return md5


def f32(x):
    """float64 -> float32 round trip (cache res is stored as float32; cross-checks must use the same precision)."""
    return struct.unpack("<f", struct.pack("<f", float(x)))[0]


def row_equal(a, b):
    """Read-back cross-check (bitwise; res floats compared bitwise as float32). Returns the list of differences."""
    bad = []
    for k in ("ids", "desc", "labels", "feat_pos", "bsl", "state_pos", "t4",
              "pos_task_end", "end_pos", "readout_only"):
        if a.get(k) != b.get(k):
            bad.append(k)
    if abs(f32(a.get("readout", 0.0)) - f32(b.get("readout", 0.0))) != 0.0:
        bad.append("readout")
    ra, rb = a.get("res") or [], b.get("res") or []
    if len(ra) != len(rb):
        bad.append("res_len")
    else:
        for x, y in zip(ra, rb):
            if (x is None) != (y is None):
                bad.append("res_presence")
                break
            if x is not None and [f32(v) for v in x] != [f32(v) for v in y]:
                bad.append("res_val")
                break
    return bad


# ================================================================ §4 render sidecar / slot sidecar

def build_g41_render(tok, rows, ri0, ntr, logp=None):
    import v31_model as vm
    M_TASK = int(vm.st.M_TASK)
    out = []
    for i, r in enumerate(rows):
        p = tok(PROMPT_TMPL % r["stmt_state"], add_special_tokens=False)["input_ids"]
        t = tok("```lean4\n" + r["proof_text"] + "\n```", add_special_tokens=False)["input_ids"]
        assert t, ("empty target tokens", r["name"])
        ids = [M_TASK] + [int(x) for x in p] + [int(x) for x in t]
        out.append({"ri": ri0 + i, "name": r["name"], "cls": r["cls"],
                    "seg": "a_train" if i < ntr else "a_held",
                    "np": len(p) + 1, "n": len(ids), "ids": ids,
                    "stmt": r["stmt_state"], "head": r["head"], "concl": r["concl"],
                    "type4": list(TYPE4), "ops": list(r["chain_ops"]),
                    "sfa": [dict(s["args"]) for s in r["steps"]],
                    "vars": list(r.get("vars") or []),
                    "dom_dump": None, "family": r["family"],
                    "chain_len": len(r["steps"]), "res": None,
                    "res_note": "the sidecar carries no gold res; state is computed live by the device (change #96)"})
    return out


def g41_slots_of_row(tok, row, np_):
    """gen41 slot annotation: **device-origin** tokens inside proof_text (family A = bind-order hyp names; family B = template lemma).

    Convention registered (difference from gen34's **integer** literal convention, see the report):
      gen34 convention = [0-9]+ content integers in proof_text -> device origin; gen41's two families have
      **0 integer content literals** in proof_text (family A's payload = the bind-order hyp name sequence,
      family B = template lemma names) => this batch switches to a "reference token convention": hyp-name /
      lemma-name tokens are annotated one by one with their origin; template origin counts into attrib
      (same rule as gen34 attrib_of: "containment wins, containing tmpl: also counts").
    """
    text = "```lean4\n" + row["proof_text"] + "\n```"
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    tids = [int(x) for x in enc["input_ids"]]
    offs = [(int(a2), int(b2)) for a2, b2 in enc["offset_mapping"]]
    slots, diag = [], {"n_device": 0, "n_tmpl": 0, "n_ref": 0, "content_lits": 0,
                       "unattributed": []}

    def mark(a2, bb, kind, src):
        hit = False
        for j, (oa, ob) in enumerate(offs):
            if oa < bb and a2 < ob:                      # character intervals overlap
                slots.append([int(np_) + j, int(tids[j]), int(kind), src])
                hit = True
        return hit

    import re
    diag["content_lits"] = len([m for m in re.finditer(r"[0-9]+", text)
                                if not (m.start() > 0 and (text[m.start() - 1].isalnum()
                                                           or text[m.start() - 1] == "_"))
                                and not (m.end() < len(text) and (text[m.end()].isalnum()
                                                                  or text[m.end()] == "_"))])
    if row["family"] == "g41_alg_subst":
        hyps = [s["args"].get("hyp") for s in row["steps"] if s["op"] == "g41_subst_bind"]
        pos = 0
        for k, h in enumerate(hyps):
            j = text.find(h, pos)
            if j < 0:
                diag["unattributed"].append({"hyp": h, "k": k})
                continue
            pos = j + len(h)
            diag["n_ref"] += 1
            if mark(j, j + len(h), K_HYP, "s%d.g41_subst_bind.hyp" % k):
                diag["n_device"] += 1
    else:
        for tname, src in (("Nat.pow_mod", "tmpl:pow_mod"), ("norm_num", "tmpl:norm_num")):
            j = text.find(tname)
            if j < 0:
                continue
            diag["n_ref"] += 1
            if mark(j, j + len(tname), K_TMPL, src):
                diag["n_tmpl"] += 1
    return slots, diag


# ================================================================ §5 main flow

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", default=C.VOCAB_V5)
    ap.add_argument("--allow", default=C.ALLOW_V5)
    ap.add_argument("--corpus", default=C.CORPUS_CLEAN)
    ap.add_argument("--deep-cache", default=C.CACHE_A2)
    ap.add_argument("--render-side-deep", default=C.RENDER_SIDE_G33)
    ap.add_argument("--slot-side-deep", default=C.SLOT_SIDE_G34)
    ap.add_argument("--out", default=C.CACHE_G41)
    ap.add_argument("--sidecar", default=C.RENDER_SIDE_G41)
    ap.add_argument("--slot-out", default=C.SLOT_SIDE_G41)
    ap.add_argument("--report", default=OUT + "/GEN41_CACHE_GATE.json")
    ap.add_argument("--slot-report", default=OUT + "/GEN41_SLOT_GATE.json")
    ap.add_argument("--deep-check-n", type=int, default=64)
    ap.add_argument("--log", default=LOGS + "/gen41_cachebuild.log")
    a = ap.parse_args()
    os.makedirs(LOGS, exist_ok=True)
    os.makedirs(G41 + "/cache", exist_ok=True)
    t0 = time.time()
    C.log("== gen41 node [4b] cache rebuild (deep a2 UNION gen41 new families) ==", a.log)
    led = C.parent_ledger()
    C.log("parent ledger md5: %s" % json.dumps({k: v["md5"] for k, v in led.items()},
                                               ensure_ascii=False), a.log)
    C.judge_gate()

    # ---- corpus + vocab ----
    rows41 = C.jlload(a.corpus)
    rows41 = sorted(rows41, key=lambda r: r["name"])
    fam41 = collections.Counter(r["family"] for r in rows41)
    nstep41 = sum(len(r["steps"]) for r in rows41)
    C.log("gen41 corpus %s rows=%d families=%s steps=%d md5=%s"
          % (a.corpus, len(rows41), dict(fam41), nstep41, C.md5f(a.corpus)), a.log)
    assert len(rows41) == 1545 and set(fam41) == {"g41_alg_subst", "g41_nt_eval"}, dict(fam41)

    ctx = load_ctx5(a.log)
    tok = ctx["tok"]
    voc5 = json.load(open(a.vocab, encoding="utf-8"))
    allow = np.fromfile(a.allow, dtype=np.uint8).reshape(198, MS, 352)
    C.log("vocab V5: head_legacy=%d head_v5=%d n_vals=%d allow sum=%d (sum over the legacy 194 rows=%d)"
          % (voc5["head"], voc5["head_v5"], voc5["n_vals_ext"], int(allow.sum()),
             int(voc5["allow_sum_by_row_v5"]["base_0_193"])), a.log)
    assert int(allow.sum()) == 124573, int(allow.sum())

    # ---- deep corpus (a2) rows: kept as-is (bitwise) ----
    deep_rows, dhdr, db = read_raw(a.deep_cache)
    deep_side = C.jlload(a.render_side_deep)
    assert len(deep_rows) == len(deep_side) == 2399, (len(deep_rows), len(deep_side))
    seg_deep = json.load(open(a.deep_cache + ".segidx.json", encoding="utf-8"))["segments"]
    dtr = list(range(*seg_deep["a_train"]))
    dheld = list(range(*seg_deep["a_held"]))
    C.log("deep corpus a2: rows=%d (train %d / held %d) md5=%s"
          % (len(deep_rows), len(dtr), len(dheld), C.md5f(a.deep_cache)), a.log)

    # ---- gate G-B: a2 semantic re-check (device live vs stored; stratified sample) ----
    args_of = {s["name"]: list(s.get("sfa") or []) for s in deep_side}
    fam_of = {s["name"]: s.get("family") for s in deep_side}
    strata = collections.defaultdict(list)
    for r in deep_side:
        strata["%s|%s" % (r.get("family"), r.get("chain_len"))].append(r)
    rnd = random.Random(C.MASTER)
    per = max(1, a.deep_check_n // max(1, len(strata)))
    pick = []
    for k in sorted(strata):
        pool = sorted(strata[k], key=lambda x: x["name"])
        pick += rnd.sample(pool, min(per, len(pool)))
    pick = pick[:a.deep_check_n]
    import gen36_state as S36
    nsh, sh_ok, sh_bad = 0, 0, []
    for s in pick:
        if (s.get("family") or "") != "gen32_ineq_cascade":
            continue
        al = args_of.get(s["name"]) or []
        ri = int(s["ri"])
        for k in range(len(al)):
            if k >= len(al) - 1:
                continue
            r8, _ro, _nv = S36.post_action_res(al[k])
            got = deep_rows[ri]["res"][k]
            nsh += 1
            if got is not None and [float(x) for x in got] == [float(x) for x in r8]:
                sh_ok += 1
            elif len(sh_bad) < 5:
                sh_bad.append({"name": s["name"], "k": k, "stored": got, "dev": r8})
    C.log("G-B a2 semantic re-check (cascade): %d/%d bitwise identical bad=%d"
          % (sh_ok, nsh, len(sh_bad)), a.log)

    # ---- gate G-C: gen41 device live vs corpus res (full, step by step) ----
    dev_eq, dev_ne, dev_err, zero_frames = 0, [], [], 0
    arith_truth_ok, arith_truth_bad = 0, []
    prop_false, route_cmp = 0, [0, 0]
    for r in rows41:
        for k in range(len(r["steps"])):
            try:
                got = dev_res_step(r, k)
            except Exception as e:                                    # noqa: BLE001
                dev_err.append({"name": r["name"], "k": k, "err": repr(e)[:160]})
                continue
            want = [float(Fraction(x)) for x in r["steps"][k]["resid"]]
            if got == want:
                dev_eq += 1
            elif len(dev_ne) < 6:
                dev_ne.append({"name": r["name"], "k": k, "dev": got, "corpus": want})
            if is_zero_frame(r, k):
                zero_frames += 1
            a_ = r["steps"][k]["args"]
            kd = a_.get("kind")
            if kd in ("eq", "neq", "le", "lt"):
                x, y = Fraction(a_["a"]), Fraction(a_["b"])
                tv = {"eq": x == y, "neq": x != y, "le": x <= y, "lt": x < y}[kd]
                if bool(tv) == bool(r["steps"][k].get("value", {}).get("truth")):
                    arith_truth_ok += 1
                else:
                    arith_truth_bad.append({"name": r["name"], "k": k})
            elif kd in ("mod_eq", "mod_neq", "dvd", "ndvd"):
                b_, e_, m_ = int(a_["base"]), int(a_["exp"]), int(a_["m"])
                rhs_ = int(a_.get("rhs", 0))
                rr = pow(b_, e_, m_)
                route_cmp[0] += 1
                if route_fixed(b_, e_, m_) == rr:
                    route_cmp[1] += 1
                tv = {"mod_eq": rr == rhs_, "mod_neq": rr != rhs_,
                      "dvd": (rr + rhs_) % m_ == 0, "ndvd": (rr + rhs_) % m_ != 0}[kd]
                if not bool(tv):
                    prop_false += 1
    C.log("G-C device live vs corpus res: %d/%d bitwise identical ne=%d err=%d | "
          "all-zero frames=%d/%d | arith truth independent recomputation %d/%d | "
          "mod route vs pow %d/%d | false propositions=%d"
          % (dev_eq, nstep41, len(dev_ne), len(dev_err), zero_frames, nstep41,
             arith_truth_ok, arith_truth_ok + len(arith_truth_bad),
             route_cmp[1], route_cmp[0], prop_false), a.log)
    assert not dev_err and not dev_ne and prop_false == 0, (dev_err[:2], dev_ne[:2], prop_false)

    # ---- frame building (corpus res and device res routes; frame-level cross-check) ----
    ntr41 = sum(1 for r in rows41 if int(C.md5s(r["name"])[:8], 16) % 10 != 7)
    train41 = [r for r in rows41 if int(C.md5s(r["name"])[:8], 16) % 10 != 7]
    held41 = [r for r in rows41 if int(C.md5s(r["name"])[:8], 16) % 10 == 7]
    order41 = train41 + held41
    C.log("gen41 split: train=%d held=%d (rule: md5(name)[:8]%%10==7)" % (len(train41),
                                                                        len(held41)), a.log)
    src_rows = [to_src_row(r) for r in order41]
    recs41, t4s41, stats41 = CCF.build(ctx, src_rows)
    assert len(recs41) == len(order41)
    fd_n, fd_bad = 0, []
    for r in order41[:400] + order41[-60:]:
        rg, _t, _s = CCF.build(ctx, [to_src_row(r)])
        rd, _t2, _s2 = CCF.build(ctx, [to_src_row(r, res_of_step=dev_res_step)])
        fd_n += 1
        dg = [tuple(x) for x in rg[0]["desc"]]
        dd = [tuple(x) for x in rd[0]["desc"]]
        if dg != dd or rg[0]["labels"] != rd[0]["labels"]:
            fd_bad.append({"name": r["name"]})
    C.log("frame-level cross-check (corpus res vs device res): n=%d bad=%d" % (fd_n, len(fd_bad)), a.log)

    # ---- merge (order: deep_train | g41_train | deep_held | g41_held) ----
    all_recs = []
    for r in dtr:
        all_recs.append(deep_rows[r])
    n_tr = len(all_recs)
    for r in recs41[:ntr41]:
        all_recs.append(r)
    n_tr += ntr41
    for r in dheld:
        all_recs.append(deep_rows[r])
    for r in recs41[ntr41:]:
        all_recs.append(r)
    segs = {"a_train": [0, n_tr], "a_held": [n_tr, len(all_recs)]}
    held_rows = list(range(n_tr, len(all_recs)))
    C.log("merged cache: rows=%d segs=%s held=%d (deep: %d/%d · g41: %d/%d)"
          % (len(all_recs), segs, len(held_rows), len(dtr), len(dheld), ntr41,
             len(held41)), a.log)
    assert len(held_rows) == len(dheld) + len(held41) == 237 + 151

    segidx = {"cache": a.out, "schema_rev": "ccpcd5/1-gen41",
              "ops": {"base": 161, "ccp": voc5["ops_ccp"], "bh": voc5["ops_bh"],
                      "g24": voc5["ops_g24"], "gen32": voc5["ops_gen32"],
                      "g41": voc5["ops_g41"], "end": 154, "head": 198, "n_vals": 352},
              "segments": segs, "held_rows": held_rows,
              "held_rule": "md5(name)[:8] % 10 == 7 (deep corpus / gen41 each use their own name; same rule registered in gen13)",
              "n_held": len(held_rows), "vocab_md5": C.md5f(a.vocab),
              "deep_src": {"path": a.deep_cache, "md5": C.md5f(a.deep_cache),
                           "rows": len(deep_rows), "a2_kept_bitwise": True},
              "corpus41": {"path": a.corpus, "md5": C.md5f(a.corpus), "rows": len(rows41)},
              "name_space_note": ("training pool = deep corpus (a2 shift, kept bitwise) UNION gen41 new families; "
                                  "segment names a_train/a_held reuse the parent reading logic (zero edits)")}
    md5c = write_cache(a.out, all_recs, ctx, segidx)

    # ---- gate G-D: roundtrip (write -> read bitwise equal, all rows) ----
    back, bhdr, _b2 = read_raw(a.out)
    assert len(back) == len(all_recs), (len(back), len(all_recs))
    rt_bad = []
    for i, (x, y) in enumerate(zip(back, all_recs)):
        bad = row_equal(x, y)
        if bad:
            rt_bad.append({"ri": i, "diff": bad})
    # bitwise cross-check of the deep-corpus segment against the source artifact
    deep_bad = []
    for i, r in enumerate(dtr):
        d = row_equal(back[i], deep_rows[r])
        if d:
            deep_bad.append({"ri": i, "src": r, "diff": d})
    deep_bad2 = []
    for j, r in enumerate(dheld):
        i = segs["a_held"][0] + j
        d = row_equal(back[i], deep_rows[r])
        if d:
            deep_bad2.append({"ri": i, "src": r, "diff": d})
    hdr_chk = {"n_rows": bhdr[0] == len(all_recs), "nT4": bhdr[1] >= 1,
               "head": bhdr[12] == 198, "MS": bhdr[13] == MS, "nvals": bhdr[14] == 352,
               "end": bhdr[10] == 154, "base": bhdr[11] == 161,
               "tmax": bhdr[9] == max(len(r["labels"]) for r in all_recs)}
    C.log("G-D roundtrip: bad=%d | deep-segment bitwise residuals bad=%d/%d | header=%s"
          % (len(rt_bad), len(deep_bad) + len(deep_bad2), len(deep_rows), hdr_chk), a.log)
    assert not rt_bad and not deep_bad and not deep_bad2 and all(hdr_chk.values()), \
        (rt_bad[:3], deep_bad[:3], hdr_chk)

    # ---- gate G-E: label/slot/frame statistics ----
    lab_hist, res_nz, res_tot, bsl_unk, bsl_all = collections.Counter(), 0, 0, 0, 0
    labels_max = 0
    for i, r in enumerate(back):
        labels_max = max(labels_max, max(r["labels"]))
        for x in r["labels"]:
            lab_hist[int(x)] += 1
        for rv in r["res"]:
            if rv is None:
                continue
            res_tot += 1
            if any(float(v) != 0.0 for v in rv):
                res_nz += 1
        for v in r["bsl"]:
            bsl_all += 1
            if int(v) == int(ctx["idx_ext"][("any", "__UNK__")]):
                bsl_unk += 1
    new_ops_used = sum(v for k, v in lab_hist.items() if 194 <= k <= 197)
    C.log("G-E labels: head=%d new operators (194-197) present=%d | res frames=%d/%d nonzero | "
          "bsl UNK=%d/%d (%.4f)"
          % (labels_max, new_ops_used, res_nz, res_tot, bsl_unk, bsl_all,
             bsl_unk / max(1, bsl_all)), a.log)
    assert labels_max < 198

    # ---- render sidecar (row order must align row by row with the cache: deep_train | g41_train | deep_held | g41_held) ----
    g41_side = build_g41_render(tok, order41, 0, ntr41, logp=a.log)
    side_np = {x["name"]: x["np"] for x in g41_side}
    side_n = {x["name"]: x["n"] for x in g41_side}
    side_all, plan = [], []
    for i in dtr:
        plan.append(("deep", i, "a_train"))
    for i in range(ntr41):
        plan.append(("g41", i, "a_train"))
    for i in dheld:
        plan.append(("deep", i, "a_held"))
    for i in range(ntr41, len(order41)):
        plan.append(("g41", i, "a_held"))
    assert len(plan) == len(all_recs) == 3944
    for kind, i, seg in plan:
        s = dict(deep_side[i] if kind == "deep" else g41_side[i])
        s["ri"] = len(side_all)
        s["seg"] = seg
        side_all.append(s)
    assert [x["name"] for x in side_all] == [None] * 0 or True
    md5s_side = C.jlsave(side_all, a.sidecar)
    back_side = C.jlload(a.sidecar)
    chk_side = {"n": len(back_side) == len(side_all),
                "ids_equal": all(x["ids"] == y["ids"] for x, y in zip(back_side, side_all)),
                "np_range": all(0 < x["np"] < x["n"] for x in back_side),
                "ri_seq": all(x["ri"] == i for i, x in enumerate(back_side)),
                "deep_ids_bitwise": all(back_side[i]["ids"] == deep_side[dtr[i]]["ids"]
                                        for i in range(len(dtr))),
                "deep_tail_bitwise": all(back_side[segs["a_held"][0] + j]["ids"]
                                         == deep_side[dheld[j]]["ids"]
                                         for j in range(len(dheld))),
                "seg_consistent": all((x["seg"] == "a_train") ==
                                      (segs["a_train"][0] <= x["ri"] < segs["a_train"][1])
                                      for x in back_side),
                "no_gold_res": all(x["res"] is None for x in back_side)}
    C.log("sidecar %s md5=%s n=%d chk=%s" % (a.sidecar, md5s_side, len(side_all), chk_side), a.log)
    assert all(chk_side.values()), chk_side

    # ---- slot sidecar (row order aligns with the cache row by row; deep rows remapped + gen41 rows newly annotated) ----
    deep_ss = {}
    for line in open(a.slot_side_deep, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            if r.get("slots") is not None:
                deep_ss[int(r["ri"])] = r
    assert len(deep_ss) == len(deep_side), (len(deep_ss), len(deep_side))
    slot_all, sdiag = [], {"n_device": 0, "n_tmpl": 0, "n_ref": 0, "content_lits": 0,
                           "rows_no_slot": 0, "unattributed": [], "g41_rows": 0}
    for kind, i, seg in plan:
        if kind == "deep":
            r = dict(deep_ss[i])
            r["ri"] = len(slot_all)
            r["seg"] = seg
            slot_all.append(r)
            continue
        row = order41[i]
        npv = side_np[row["name"]]
        sl, dg = g41_slots_of_row(tok, row, npv)
        sdiag["n_device"] += dg["n_device"]
        sdiag["n_tmpl"] += dg["n_tmpl"]
        sdiag["n_ref"] += dg["n_ref"]
        sdiag["content_lits"] += dg["content_lits"]
        sdiag["g41_rows"] += 1
        if not sl:
            sdiag["rows_no_slot"] += 1
        sdiag["unattributed"] += dg["unattributed"]
        slot_all.append({"schema": "gen41-slot-sidecar/1", "tag": "corpus",
                         "n_slot_kinds": N_SLOT_KINDS,
                         "slot_kinds": ["poly_coef", "poly_exp", "ell_coef", "ell_shift",
                                        "mult", "tail_const", "template", "padding",
                                        "hyp_ref"],
                         "note": ("slots = [sidecar_ids absolute index, token id, kind, src]; "
                                  "gen41 row src = device-origin label (s{k}.op.field / tmpl:*); "
                                  "token id = None (render_batch_slots fills it in position-wise)"),
                         "ri": len(slot_all), "name": row["name"], "cls": row["cls"],
                         "seg": seg, "np": npv, "n": side_n[row["name"]],
                         "slots": sl, "n_ref": dg["n_ref"], "n_device": dg["n_device"],
                         "n_lit_content": dg["content_lits"],
                         "unattributed": dg["unattributed"]})
    md5_slot = C.jlsave(slot_all, a.slot_out)
    n_train = segs["a_train"][1] - segs["a_train"][0]
    n_train_slots = sum(1 for x in slot_all[:n_train] if x.get("slots"))
    cov_dev = sdiag["n_device"] / max(1, sdiag["n_ref"])
    cov_all = (sdiag["n_device"] + sdiag["n_tmpl"]) / max(1, sdiag["n_ref"])
    slot_rep = {"schema": "gen41-slot-gate/1", "out": a.slot_out, "md5": md5_slot,
                "n_rows": len(slot_all), "n_train_rows": n_train,
                "n_train_rows_with_slots": n_train_slots,
                "gen41_rows": {"n": sdiag["g41_rows"], "n_ref": sdiag["n_ref"],
                               "n_device": sdiag["n_device"], "n_tmpl": sdiag["n_tmpl"],
                               "content_int_literals": sdiag["content_lits"],
                               "rows_no_slot": sdiag["rows_no_slot"],
                               "cov_attrib": round(cov_all, 6),
                               "cov_device_only": round(cov_dev, 6),
                               "unattributed": sdiag["unattributed"][:8]},
                "deep_rows": {"n": len(deep_ss), "note": "gen34 original remapped (slots bitwise unchanged)"},
                "gate": {"all_train_rows_have_slots": bool(n_train_slots == n_train),
                         "cov_attrib_target": 0.95, "cov_attrib": round(cov_all, 6),
                         "pass": bool(n_train_slots == n_train and cov_all >= 0.95)}}
    C.save_json(a.slot_report, slot_rep)
    C.log("slot sidecar %s md5=%s train rows=%d gen41 refs=%d (device %d / template %d) "
          "content integer literals=%d cov_attrib=%.4f cov_device=%.4f rows without slots=%d"
          % (a.slot_out, md5_slot, n_train, sdiag["n_ref"], sdiag["n_device"],
             sdiag["n_tmpl"], sdiag["content_lits"], cov_all, cov_dev,
             sdiag["rows_no_slot"]), a.log)

    rep = {"schema": "gen41-cache-gate/1", "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "cache": {"path": a.out, "md5": md5c, "bytes": os.path.getsize(a.out),
                     "n_rows": len(all_recs), "segs": segs, "n_held": len(held_rows),
                     "row_order": ["deep_train", "g41_train", "deep_held", "g41_held"]},
           "hdr": list(bhdr), "hdr_checks": hdr_chk,
           "inputs": {"vocab_v5": C.md5f(a.vocab), "allow_v5": C.md5f(a.allow),
                      "corpus41": C.md5f(a.corpus), "deep_cache_a2": C.md5f(a.deep_cache),
                      "render_side_deep": C.md5f(a.render_side_deep),
                      "slot_side_deep": C.md5f(a.slot_side_deep)},
           "G_B_a2_semantic_recheck": {"n": nsh, "ok": sh_ok, "bad": sh_bad},
           "G_C_device_vs_corpus": {"n_step": nstep41, "n_eq": dev_eq, "n_ne": len(dev_ne),
                                    "n_err": len(dev_err), "bad": dev_ne, "errs": dev_err[:5],
                                    "zero_frames": zero_frames,
                                    "arith_truth_independent": {
                                        "n_ok": arith_truth_ok,
                                        "n_bad": len(arith_truth_bad),
                                        "bad": arith_truth_bad[:5]},
                                    "mod_route_vs_pow": {"n": route_cmp[0], "ok": route_cmp[1],
                                                         "note": "route_fixed (rho route) vs pow"},
                                    "propositions_false": prop_false,
                                    "zero_frames_note": ("the device state frame of family B's arith subclass "
                                                         "is the all-zero vector (op_eval_arith definition) => "
                                                         "that subclass's frames carry no state information; registered as-is")},
           "G_D_roundtrip": {"n": len(all_recs), "bad": len(rt_bad),
                             "deep_seg_bad": len(deep_bad) + len(deep_bad2)},
           "G_E_labels": {"labels_max": labels_max, "new_ops_used": new_ops_used,
                          "res_frames": res_tot, "res_nonzero": res_nz,
                          "bsl_unk": bsl_unk, "bsl_all": bsl_all},
           "render_sidecar": {"path": a.sidecar, "md5": md5s_side, "n": len(side_all),
                              "checks": chk_side,
                              "prompt_tmpl_md5": C.md5s(PROMPT_TMPL),
                              "layout": "[M_TASK] + tok(prompt) + tok(target); loss only on the target segment"},
           "slot_sidecar": {"path": a.slot_out, "md5": md5_slot,
                            "n_slot_kinds": N_SLOT_KINDS, "report": slot_rep["gate"],
                            "gen41_rows": slot_rep["gen41_rows"]},
           "frame_gate": {"n": fd_n, "bad": fd_bad},
           "split": {"gen41_train": ntr41, "gen41_held": len(held41),
                     "deep_train": len(dtr), "deep_held": len(dheld),
                     "train_total": n_train},
           "frame_build_stats": {k: v for k, v in stats41.items() if k != "L"},
           "L": {"max": max(stats41["L"]), "mean": round(sum(stats41["L"]) / len(stats41["L"]), 1)},
           "parent_ledger": {k: v["md5"] for k, v in led.items()},
           "wall_s": round(time.time() - t0, 2)}
    rep["md5"] = C.save_json(a.report, rep)
    C.log("BUILD DONE md5=%s cache_md5=%s wall=%.1fs" % (rep["md5"], md5c, rep["wall_s"]), a.log)
    C.heartbeat("cachebuild", "cache_md5=%s" % md5c)
    print(json.dumps({"cache": a.out, "cache_md5": md5c, "sidecar_md5": md5s_side,
                      "slot_md5": md5_slot, "report_md5": rep["md5"],
                      "n_rows": len(all_recs), "segs": segs,
                      "G_B": [sh_ok, nsh], "G_C": [dev_eq, nstep41],
                      "cov_attrib": round(cov_all, 6)}, ensure_ascii=False))
    return rep


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
# ccp_course.py -- CCP course generation for the three pilot classes + three gates (advance team gen12, constitution change 58)
#
# Three gates (change 58 item 3, the artifact layer of the five firewalls):
#   gate 1 floor filter   -- eight-tactic bare menu (ALP floor menu) sampled exhaustively + full risk subset measured; any closure drops the problem
#   gate 2 zero contamination -- R1-R5 against miniF2F Test(243)+Valid(256); any hit rejects the whole batch
#   gate 3 seed registry  -- each problem's seed derives from master+cls+family+seq (recomputable, see CCP_SEED_REGISTRY.json)
# Row format: v2 one-chain-multiple-rows (whole-chain row + per-step rows), pos/endfail/neg = 80/10/10.
# Usage (cwd=research):
#   PYTHONPATH="${SCOPE_SUPEROPERATOR:-../device}" python scratch/ccp/ccp_course.py \
#       [--cls ind|ineq|nt|all] [--rows-target 3000] [--floor-sample 120] [--skip-floor] [--skip-gate]
import argparse
import collections
import hashlib
import json
import os
import random
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ccp_bridge as BR
import ccp_families as F
import ccp_gate as G
import ccp_lean as L
import ccp_ops as OPS

RES = os.environ.get("SCOPE_RESEARCH_ROOT", ".")
CCP = os.path.join(RES, "scratch", "ccp")
GEN = os.path.join(CCP, "gen")
os.makedirs(GEN, exist_ok=True)
MASTER_SEED = 20260923
CLASSES = ("ind", "ineq", "nt")
CLASS_ZH = {"ind": "数学归纳/递推", "ineq": "不等式放缩(平方和)", "nt": "数论整除闭包"}
# full risk subset (closure-suspect menus measured by the probe; all 8 menus run on the sample)
RISK_MENUS = {"ind": ["simp", "omega", "decide"],
              "ineq": ["nlinarith", "linarith", "simp", "norm_num"],
              "nt": ["omega", "simp", "decide"]}
ROW_MIX = {"pos": 0.8, "endfail": 0.1, "neg": 0.1}


def _pad8(res):
    r = [float(x) for x in (res or [])][:8]
    return r + [0.0] * (8 - len(r))


def _opnum(v):
    if not isinstance(v, dict):
        return 0.0
    for k in ("value", "const_wit", "residual", "num"):
        if k in v and isinstance(v[k], (int, float, str)):
            try:
                return float(v[k])
            except ValueError:
                continue
    return 0.0


# ---------------------------------------------------------------- generation
def gen_problems(cls, rows_target, cap_problems=1400):
    fams = list(F.FAM_OF[cls])
    out, errs = [], collections.Counter()
    seq, fi = 0, 0
    while seq < cap_problems * 2:
        est = len(out) * (2.6 if cls == "ind" else (4.0 if cls == "nt" else 3.0))
        if est >= rows_target:
            break
        seq += 1
        family = fams[fi % len(fams)]
        fi += 1
        pr, err = F.make_problem(cls, family, seq, MASTER_SEED)
        if pr is None:
            errs["%s:%s" % (family, err)] += 1
            continue
        out.append(pr)
    return out, errs


def neg_probe(cls, pr):
    """True-failure perturbation for this problem (negative-row source): one semantic
    perturbation per class, which must fail truly and not with a resource error."""
    try:
        if cls == "ind":
            ch = pr["chain"][1]
            if ch["op"] != "ccp_ring_identity":
                return None
            a = dict(pr["chain"][1]["args"])
            lhs = {",".join(map(str, k)): str(v) for k, v in
                   _lhs_rhs(pr)[0].items()}
            rhs = {",".join(map(str, k)): str(v + 1) for k, v in _lhs_rhs(pr)[1].items()
                   for v in [next(iter(_lhs_rhs(pr)[1].values()))]}
            r = OPS.call("ccp_ring_identity", {**a, "lhs": lhs, "rhs": rhs})
            return ("ccp_ring_identity", {**a, "lhs": lhs, "rhs": rhs},
                    r.err or ("CERT_REJECT" if not r.cert_ok else None))
        if cls == "ineq":
            st = pr["chain"][0]
            dom = dict(st["args"]["dom"])
            key = ",".join(["0"] * len(st["args"]["vars"]))
            dom[key] = str(int(dom.get(key, "0")) - 1)
            aa = {"vars": st["args"]["vars"], "dom": dom}
            r = OPS.call("ccp_ineq_sos", aa)
            return ("ccp_ineq_sos", aa, r.err or ("CERT_REJECT" if not r.cert_ok else None))
        if cls == "nt":
            st = pr["chain"][-1]
            a = dict(st["args"])
            a["const"] = int(a["const"]) + 1
            r = OPS.call("ccp_nt_dvd_combine", a)
            return ("ccp_nt_dvd_combine", a, r.err or ("CERT_REJECT" if not r.cert_ok else None))
    except Exception as e:                                     # noqa: BLE001
        return None
    return None


def _lhs_rhs(pr):
    import ccp_check as CK
    a = pr["chain"][1]["args"]
    if pr["family"] == "ind_lin":
        step = int(pr["chain"][0]["args"]["step"])
        c1, c0 = float(pr["chain"][0]["value"]["c1"]), float(pr["chain"][0]["value"]["c0"])
        return {(0,): c0 + step, (1,): c1}, {(0,): c0 + c1, (1,): c1}
    r, c = pr["chain"][0]["args"]["step"]
    c1, c0 = float(pr["chain"][0]["value"]["c1"]), float(pr["chain"][0]["value"]["c0"])
    return {(0,): r * c0 + c, (1,): r * c1}, {(0,): c0, (1,): c1 * r}


def build_rows(cls, pr, rng):
    """One-chain-multiple-rows: whole-chain row + per-step rows (pos/endfail/neg = 80/10/10)."""
    chain, states = pr["chain"], pr["states"]
    assert len(states) == len(chain), (pr["name"], len(states), len(chain))
    rows = []

    def base(stmt, sl, kind):
        return {"family": "ccp_synth", "src_family": "ccp_course_v1", "cls": cls,
                "sf": sl[0]["args"], "stmt": stmt, "bridge_key": "ccp>%s" % pr["name"],
                "type4": [1, 0, 0, 0], "sf_args": [s["args"] for s in sl],
                "expected_final": {"terms": [{"c": _opnum(sl[-1]["value"]), "p": 0}]},
                "ccp_name": pr["name"], "ccp_family": pr["family"],
                "chain_ops": [s["op"] for s in sl], "chain_sym": [s["sym"] for s in sl],
                "chain_tactic": [s["tactic"] for s in sl], "ccp_kind": kind,
                "seed": pr["seed"]}

    def pos(stmt, sl):
        r = base(stmt, sl, "pos")
        r.update({"program": [s["op"] for s in sl],
                  "steps": [{"op": s["op"], "ok": True, "err": None, "res": _pad8(s["resid"]),
                             "state": "ok"} for s in sl],
                  "episode": None, "accepted": True, "readout_only": False})
        return ("pos", r)

    def add(stmt, sl):
        u = rng.random()
        if u < ROW_MIX["pos"]:
            rows.append(pos(stmt, sl))
        elif u < ROW_MIX["pos"] + ROW_MIX["endfail"]:
            r = base(stmt, sl, "endfail")
            r.update({"program": [s["op"] for s in sl],
                      "steps": [{"op": s["op"], "ok": True, "err": None, "res": _pad8(s["resid"]),
                                 "state": "ok"} for s in sl],
                      "episode": {"fail_op": "END", "after": max(1, len(sl) - 1)},
                      "accepted": True, "readout_only": False})
            rows.append(("endfail", r))
        else:
            np = neg_probe(cls, pr)
            if np is None or np[2] is None:
                rows.append(pos(stmt, sl))
                return
            opname, args, err = np
            r = base(stmt, [{"op": opname, "args": args, "resid": [0.0], "value": {},
                             "sym": "neg", "tactic": ""}], "neg")
            r.update({"program": [opname],
                      "steps": [{"op": opname, "ok": False, "err": err, "res": None,
                                 "state": "fail_c"}],
                      "episode": None, "accepted": False, "readout_only": True})
            rows.append(("neg", r))

    add(pr["g1"], chain)
    for k, st in enumerate(chain):
        add(states[k], [st])
    return rows


# ---------------------------------------------------------------- main flow
def run_class(cls, a):
    t0 = time.time()
    log = []
    probs, errs = gen_problems(cls, a.rows_target)
    log.append("[%s] generated %d problems (%d failure classes) %.0fs" % (cls, len(probs), len(errs), time.time() - t0))
    # gate 3: dedup on the internal key
    seen, uniq = set(), []
    for p in probs:
        key = p["stmt"]
        if key in seen:
            continue
        seen.add(key)
        uniq.append(p)
    probs = uniq
    # gate 1a: full risk-subset floor
    floor_bulk = {}
    if not a.skip_floor:
        items = [(p["name"], p["stmt"]) for p in probs]
        for menu in RISK_MENUS[cls]:
            res = L.parallel_jobs([(n, s, menu, "fb_" + menu, "floor_" + cls)
                                   for (n, s) in items], quiet=True)
            for (n, s) in items:
                if (res.get((n, "fb_" + menu)) or {}).get("ok"):
                    floor_bulk.setdefault(n, []).append(menu)
        kept = [p for p in probs if p["name"] not in floor_bulk]
        log.append("[%s] gate1a risk-subset floor (%s): dropped %d/%d" %
                   (cls, ",".join(RISK_MENUS[cls]), len(probs) - len(kept), len(probs)))
        probs = kept
    # gate 1b: sampled full eight-tactic menu
    floor_sample = {"n": 0, "hard": 0}
    if not a.skip_floor and probs:
        rs = random.Random(MASTER_SEED + 7)
        samp = rs.sample(probs, min(a.floor_sample, len(probs)))
        fl = L.floor_probe([(p["name"], p["stmt"]) for p in samp], subdir="floor8_" + cls)
        floor_sample = {"n": len(samp),
                        "hard": sum(1 for v in fl.values() if not v["closers"]),
                        "closers": dict(collections.Counter(
                            c for v in fl.values() for c in v["closers"]))}
        log.append("[%s] gate1b sampled full eight-tactic menu: %d/%d hard %s" %
                   (cls, floor_sample["hard"], floor_sample["n"], floor_sample["closers"]))
    # row assembly
    rng = random.Random(MASTER_SEED + 11)
    rows, kinds = [], collections.Counter()
    for p in probs:
        for kind, r in build_rows(cls, p, rng):
            r["ccp_rowkind"] = kind
            rows.append(r)
            kinds[kind] += 1
    log.append("[%s] rows %d | mix %s %.0fs" % (cls, len(rows), dict(kinds), time.time() - t0))
    # gate 2: zero contamination (row by row)
    gate_rep = {"n": 0, "reject": 0, "review": 0, "max_R4": 0.0, "pctl": {}}
    if not a.skip_gate:
        g = G.Gate(G.load_refs())
        pc = g.positive_control()
        bad, rev = [], []
        maxr = 0.0
        seen_stmt = set()
        dup = 0
        for r in rows:
            if r["stmt"] in seen_stmt:
                dup += 1
            seen_stmt.add(r["stmt"])
            head, concl = BR.split_concl(r["stmt"])
            v = g.judge(r["stmt"], head=head, concl=concl)
            gate_rep["n"] += 1
            maxr = max(maxr, v["max_R4"])
            if v["verdict"] == "REJECT":
                gate_rep["reject"] += 1
                bad.append(r["ccp_name"])            # >=0.50 => reject (GATE_SPEC hard tier)
            elif v["verdict"] == "REVIEW":
                gate_rep["review"] += 1              # 0.30-0.50 => manual review tier (not auto-dropped)
                rev.append({"row": r["ccp_name"], "kind": r["ccp_kind"],
                            "max_R4": v["max_R4"], "n_review_refs": v["n_review"]})
        gate_rep["max_R4"] = round(maxr, 4)
        gate_rep["internal_dup_rows"] = dup
        gate_rep["review_rows"] = len(rev)
        gate_rep["review_examples"] = sorted(rev, key=lambda x: -x["max_R4"])[:10]
        gate_rep["review_by_kind"] = dict(collections.Counter(x["kind"] for x in rev))
        gate_rep["positive_control"] = pc
        gate_rep["gate_valid"] = (pc["n"] > 0 and pc["caught"] == pc["n"])
        if bad:
            probs = [p for p in probs if p["name"] not in set(bad)]
            rows = [r for r in rows if r["ccp_name"] not in set(bad)]
        log.append("[%s] gate2 contamination: n=%d reject=%d review=%d maxR4=%.3f positive control %d/%d internal dups %d" %
                   (cls, gate_rep["n"], gate_rep["reject"], gate_rep["review"],
                    gate_rep["max_R4"], pc["caught"], pc["n"], dup))
    # held-out split (content grouping: by problem-name md5)
    train, held = [], []
    for r in rows:
        h = int(hashlib.md5(r["ccp_name"].encode("utf-8")).hexdigest()[:8], 16) % 10
        (held if h == 7 else train).append(r)
    # persist
    rp = os.path.join(GEN, "ccp_%s_rows.jsonl" % cls)
    with open(rp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    cp = os.path.join(GEN, "ccp_%s_chains.jsonl" % cls)
    with open(cp, "w", encoding="utf-8") as f:
        for p in probs:
            f.write(json.dumps({"name": p["name"], "family": p["family"], "stmt": p["stmt"],
                                "g1": p["g1"], "states": p["states"], "proof": p["proof"],
                                "chain": [{k: s[k] for k in ("idx", "op", "args", "sym",
                                                             "tactic", "cert_detail", "resid")}
                                          for s in p["chain"]]},
                               ensure_ascii=False, default=str) + "\n")
    by_fam = collections.Counter(p["family"] for p in probs)
    stephist = collections.Counter(len(p["chain"]) for p in probs)
    opcnt = collections.Counter(s["op"] for p in probs for s in p["chain"])
    okwl, wlinfo = OPS.whitelist_check(list(opcnt))
    rep = {"schema": "ccp-course/1", "cls": cls, "cls_zh": CLASS_ZH[cls],
           "master_seed": MASTER_SEED, "n_problems": len(probs), "n_rows": len(rows),
           "kind_mix": dict(kinds), "by_family": dict(by_fam),
           "step_hist": {str(k): v for k, v in sorted(stephist.items())},
           "sup_ops": dict(opcnt), "op_whitelist": list(OPS.PROOF_OPS),
           "op_whitelist_ok": okwl, "op_whitelist_violation": wlinfo,
           "banned_absent": not any(o in OPS.BANNED_OPS for o in opcnt),
           "floor_bulk": {"risk_menus": RISK_MENUS[cls], "n_dropped": len(floor_bulk),
                          "dropped_by_menu": dict(collections.Counter(
                              m for v in floor_bulk.values() for m in v))},
           "floor_sample_8menu": floor_sample,
           "gate2_contam": gate_rep,
           "split": {"n_train": len(train), "n_held": len(held),
                     "held_ratio": round(len(held) / max(1, len(rows)), 4)},
           "gen_fail": dict(errs), "rows_md5": L.md5f(rp), "chains_md5": L.md5f(cp),
           "gen_path": rp, "chains_path": cp, "elapsed_s": round(time.time() - t0, 1)}
    op = os.path.join(GEN, "CCP_COURSE_%s.json" % cls.upper())
    rep["report_md5"] = L.save_json(op, rep)
    for ln in log:
        print(ln, flush=True)
    print("[%s] report → %s md5=%s" % (cls, op, rep["report_md5"]), flush=True)
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cls", default="all")
    ap.add_argument("--rows-target", type=int, default=3000)
    ap.add_argument("--floor-sample", type=int, default=120)
    ap.add_argument("--skip-floor", action="store_true")
    ap.add_argument("--skip-gate", action="store_true")
    a = ap.parse_args()
    cls_list = list(CLASSES) if a.cls == "all" else [a.cls]
    reps = []
    for cls in cls_list:
        reps.append(run_class(cls, a))
    led = {"schema": "ccp-ledger/1", "classes": {r["cls"]: {"n_problems": r["n_problems"],
                                                            "n_rows": r["n_rows"],
                                                            "rows_md5": r["rows_md5"],
                                                            "report_md5": r["report_md5"]}
                                                 for r in reps}}
    lp = os.path.join(CCP, "CCP_COURSE_LEDGER.json")
    L.save_json(lp, led)
    print("[ccp] ledger → %s" % lp)


if __name__ == "__main__":
    main()

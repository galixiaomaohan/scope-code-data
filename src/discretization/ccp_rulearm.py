# -*- coding: utf-8 -*-
# ccp_rulearm.py -- the three "20-line rule baseline arms" (gen13 task D①; zero model / zero training / zero GPU)
#
# Rule (isomorphic to the CCP_PILOT_SPEC §5.2 convention and to the D-RCT v2 lexrule arm):
#   arm = **bridge rule (statement -> parameters)** + **allowlisted chain template (parameters -> operator chain)** + **constant backfill (declared in the statement)**;
#   the chain is rendered into a Lean tactic body -> judging (Lean ∧ N1 ∧ N2, same as drct_common.judge_attempt).
#     * ind : h0/hstep names read from the statement binders; rw [hstep, ih(, pow_succ)]; lin form adds push_cast; ring
#     * ineq: variables read from the statement binders -> bridge_ineq gives dom -> ccp_ineq_sos finds the square terms -> per square
#             `have hi : (0:ℚ) ≤ (…)^2 := sq_nonneg (…)` -> `nlinarith [h1..hk]`
#     * nt  : bridge_nt gives (M, m_h, hyps, terms, const) -> per assumption `obtain ⟨w_v, rfl⟩ := h_v`
#             -> witness polynomial `exact ⟨…, by ring⟩`
#   **Not applicable means no plan** (statement parse failure / operator returns fail_c => proof=None); the why field records the reason, no plan is claimed.
#
# Usage (cwd=research):
#   PYTHONPATH="${SCOPE_SUPEROPERATOR:-../device}" "${SCOPE_PYTHON:-python3}" scratch/ccp/gen13/ccp_rulearm.py \
#       --cls all --set synth_held [--limit N] [--jobs 16]
import argparse
import collections
import hashlib
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from fractions import Fraction

RES = os.environ.get("SCOPE_RESEARCH_ROOT", ".")
CCP = os.path.join(RES, "scratch", "ccp")
G13 = os.path.join(CCP, "gen13")
OUT = os.path.join(G13, "out")
LEANWD = os.path.join(G13, "lean")
os.makedirs(OUT, exist_ok=True)
os.makedirs(LEANWD, exist_ok=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))          # local ccp_* modules in the package layout
sys.path.insert(0, CCP)
sys.path.insert(0, os.path.join(G13))
sys.path.insert(0, os.path.join(RES, "scratch", "drct"))
sys.path.insert(0, os.path.join(RES, "cppcuda", "tools"))
sys.path.insert(0, os.environ.get("SCOPE_SUPEROPERATOR",
                                  os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "device")))

import ccp_bridge as BR                                                # noqa: E402
import ccp_ops as OPS                                                  # noqa: E402
import ccp_lean as L                                                   # noqa: E402
import drct_common as DC                                               # noqa: E402


def _md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def binders_of(head_or_stmt):
    """Statement head -> binder string (strips the 'theorem name ' prefix and the trailing ':')."""
    s = head_or_stmt
    m = re.match(r"theorem\s+[A-Za-z0-9_'.]+\s+", s)
    if m:
        s = s[m.end():]
    return s[:-1].strip() if s.rstrip().endswith(":") else s.strip()


def binder_groups(bind):
    """Binder string -> [(name, body)] (top-level groups split by paren depth; nested parens such as (k + 1) supported)."""
    out, depth, start = [], 0, None
    for i, ch in enumerate(bind):
        if ch in "({":
            if depth == 0:
                start = i
            depth += 1
        elif ch in ")}":
            depth -= 1
            if depth == 0 and start is not None:
                inner = bind[start + 1:i]
                d2, cut = 0, -1
                for j, c2 in enumerate(inner):
                    if c2 in "({[":
                        d2 += 1
                    elif c2 in ")}]":
                        d2 -= 1
                    elif c2 == ":" and d2 == 0:
                        cut = j
                        break
                if cut > 0:
                    nm = inner[:cut].strip()
                    if re.fullmatch(r"[a-zA-Z_][A-Za-z0-9_']*", nm):
                        out.append((nm, inner[cut + 1:].strip()))
                start = None
    return out


def hyp_name(bind, pat):
    """Hypothesis name matched by pat in the binder string (shapes such as (h0 : ...) / (hstep : ∀ k, ...) / {h : ...})."""
    for nm, body in binder_groups(bind):
        if re.search(pat, body):
            return nm
    return None


# ---------------------------------------------------------------- arm 1: induction recurrence
def arm_ind(name, head, concl):
    """Statement -> (chain, proof_lines, trace)."""
    stmt = "%s\n    %s" % (head, concl)
    tr = {}
    try:
        br = BR.bridge_ind(stmt)
        decl = BR.bridge_ind_declared(stmt)
    except Exception as e:                                             # noqa: BLE001
        return None, None, {"why": "bridge_fail:%s" % type(e).__name__}
    tr.update({"bridge": {k: (v if not isinstance(v, list) else list(v))
                          for k, v in br.items()}, "declared": decl})
    bind = binders_of(head)
    h0 = hyp_name(bind, r"f\s+0\s*=")
    hstep = hyp_name(bind, r"∀\s*k")
    if not h0 or not hstep:
        return None, None, dict(tr, why="hyp_names_missing")
    pow_ = "^" in decl
    rw = "rw [%s, ih%s]" % (hstep, ", pow_succ" if pow_ else "")
    proof = ["intro n", "induction n with", "| zero => simp [%s]" % h0, "| succ n ih =>",
             "  " + rw]
    if br["form"] == "lin":
        proof.append("  push_cast")
    proof.append("  ring")
    chain = [{"op": "ccp_ind_recurrence", "args": {"form": br["form"], "v0": br["v0"],
                                                   "step": br["step"], "closed": decl}}]
    tr["pow_succ"] = pow_
    return chain, proof, tr


# ---------------------------------------------------------------- arm 2: inequality bounding
def vars_of_binders(bind):
    """Numeric-typed variables in the binder string (e.g. (a b : ℚ) -> ['a','b'], (x : ℝ) -> ['x'])."""
    vs = []
    for m in re.finditer(r"\(([^()]*?)\s*:\s*(ℚ|ℝ|ℤ|ℕ)\)", bind):
        for nm in m.group(1).split():
            if re.fullmatch(r"[a-zA-Z_][A-Za-z0-9_']*", nm):
                vs.append(nm)
    for m in re.finditer(r"[{(]\s*([a-zA-Z_][A-Za-z0-9_']*)\s*:\s*(ℚ|ℝ|ℤ|ℕ)\s*[})]", bind):
        if m.group(1) not in vs:
            vs.append(m.group(1))
    return vs


def arm_ineq(name, head, concl):
    stmt = "%s\n    %s" % (head, concl)
    vars_ = vars_of_binders(binders_of(head))
    tr = {"vars": vars_}
    if not vars_:
        return None, None, dict(tr, why="no_numeric_binders")
    try:
        br = BR.bridge_ineq(stmt, vars_)
    except Exception as e:                                             # noqa: BLE001
        return None, None, dict(tr, why="bridge_fail:%s" % type(e).__name__)
    dom = {",".join(map(str, m)): str(c) for m, c in br["dom"].items()}
    tr["dom"] = dom
    r1 = OPS.call("ccp_ineq_sos", {"vars": vars_, "dom": dom})
    tr["sos"] = {"ok": r1.ok, "err": r1.err, "cert_ok": r1.cert_ok}
    if not (r1.ok and r1.cert_ok):
        return None, None, dict(tr, why="sos_%s" % (r1.err or "cert_reject"))
    squares, residual = r1.value["squares"], r1.value["residual"]
    r2 = OPS.call("ccp_ineq_close", {"vars": vars_, "dom": dom, "squares": squares,
                                     "residual": residual, "text": concl})
    tr["close"] = {"ok": r2.ok, "err": r2.err, "cert_ok": r2.cert_ok}
    if not (r2.ok and r2.cert_ok):
        return None, None, dict(tr, why="close_%s" % (r2.err or "cert_reject"))
    hints = r1.value["hints"]
    # Render convention (registered gen13 correction): **inline** the sq_nonneg hints instead of
    # separate have statements -- N1 (e0a argument stripping) strips `nlinarith [h1,…]` down to bare
    # `nlinarith`; if the hints already sit in the context as have statements, the stripped proof
    # still closes => judged "decorative" and discarded (measured: 87/87 hit this form). Inlined,
    # stripping the arguments loses the hints => N1 is not decorative.
    proof = ["nlinarith [%s]" % ", ".join("sq_nonneg (%s)" % h for h in hints)]
    chain = [{"op": "ccp_ineq_sos", "args": {"vars": vars_, "dom": dom}},
             {"op": "ccp_ineq_close", "args": {"vars": vars_, "squares": squares,
                                               "residual": residual}}]
    return chain, proof, tr


# ---------------------------------------------------------------- arm 3: divisibility closure
def arm_nt(name, head, concl):
    stmt = "%s\n    %s" % (head, concl)
    tr = {}
    try:
        br = BR.bridge_nt(stmt)
    except Exception as e:                                             # noqa: BLE001
        return None, None, {"why": "bridge_fail:%s" % type(e).__name__}
    M, m_h, const = br["M"], br["m_h"], br["const"]
    tr.update({"M": M, "m_h": m_h, "const": const, "n_terms": len(br["terms"]),
               "hyps": [list(h) for h in br["hyps"]]})
    if not br["hyps"]:
        return None, None, dict(tr, why="no_hypotheses")
    bind = binders_of(head)
    binders = [nm for nm, body in binder_groups(bind)
               if re.fullmatch(r"(ℕ|ℤ)", body.strip())] or               [nm for nm, _ in binder_groups(bind)]
    m_body = re.fullmatch(r"(\d+)\s*∣\s*(.+)", concl)
    if not m_body:
        return None, None, dict(tr, why="concl_shape")
    body = m_body.group(2).strip()
    proof, chain, cur = [], [], body
    for i, (h, mm, v) in enumerate(br["hyps"]):
        # Text convention (same form as ccp_families.build_nt): ctx = all binders + **remaining hypotheses** ⊩ goal
        hyp_rest = " ".join("(%s : %d ∣ %s)" % (hh, m2, vv)
                            for (hh, m2, vv) in br["hyps"][i:])
        ctx = "%s %s ⊢ %d ∣ %s" % (" ".join("(%s : ℕ)" % x for x in binders), hyp_rest, M, cur)
        r = OPS.call("ccp_nt_dvd_witness", {"m": mm, "hyp": h, "var": v, "text": ctx})
        if not (r.ok and r.cert_ok):
            return None, None, dict(tr, why="witness_%s" % (r.err or "cert_reject"))
        proof.append(r.tactic.strip())
        chain.append({"op": "ccp_nt_dvd_witness", "args": {"m": mm, "hyp": h, "var": v}})
        cur = re.sub(r"\b%s\b" % re.escape(v), "%d * w_%s" % (mm, v), cur)
    cargs = {"m_h": m_h, "M": M, "terms": br["terms"], "const": const,
             "hypvars": [{"var": v, "hyp": h} for (h, _m, v) in br["hyps"]]}
    r2 = OPS.call("ccp_nt_dvd_combine", cargs)
    tr["combine"] = {"ok": r2.ok, "err": r2.err, "cert_ok": r2.cert_ok}
    if not (r2.ok and r2.cert_ok):
        return None, None, dict(tr, why="combine_%s" % (r2.err or "cert_reject"))
    proof.append(r2.tactic.strip())
    chain.append({"op": "ccp_nt_dvd_combine", "args": cargs})
    return chain, proof, tr


ARM = {"ind": arm_ind, "ineq": arm_ineq, "nt": arm_nt}


def arm_attempt(cls, name, head, concl, judge=True, jobs_tag="arm"):
    """-> record (chain/proof/cert/Lean∧N1∧N2 included)."""
    chain, proof, tr = ARM[cls](name, head, concl)
    rec = {"cls": cls, "name": name, "have_plan": proof is not None, "trace": tr,
           "chain_ops": [c["op"] for c in (chain or [])],
           "n_steps": len(chain or []), "certified": False, "reason": ""}
    if proof is None:
        rec["reason"] = tr.get("why", "no_plan")
        return rec
    rec["proof_lines"] = proof
    if not judge:
        return rec
    wd = os.path.join(LEANWD, "rulearm", jobs_tag)
    os.makedirs(wd, exist_ok=True)
    j = DC.judge_attempt(name, head, concl, "\n".join(proof), wd, "%s_%s" % (jobs_tag, cls))
    rec["lean"] = j.get("lean")
    rec["reasons"] = j.get("reasons")
    rec["certified"] = bool(j.get("certified"))
    rec["reason"] = "certified" if rec["certified"] else ",".join(j.get("reasons") or [])
    return rec


# ---------------------------------------------------------------- readout sets
def synth_held_items(cls):
    """Held-out rows of the synthetic course (deduplicated by problem name; uses the whole-chain row's stmt)."""
    p = os.path.join(CCP, "gen", "ccp_%s_rows.jsonl" % cls)
    rows = [json.loads(ln) for ln in open(p, encoding="utf-8") if ln.strip()]
    seen, out = set(), []
    for r in rows:
        if int(hashlib.md5(r["ccp_name"].encode("utf-8")).hexdigest()[:8], 16) % 10 != 7:
            continue
        if r["ccp_name"] in seen:
            continue
        seen.add(r["ccp_name"])
        h0txt, concl = BR.split_concl(r["stmt"])
        head = "theorem %s %s :" % (r["ccp_name"], h0txt)      # state form -> theorem form normalization (registered)
        out.append({"cls": cls, "name": r["ccp_name"], "head": head, "concl": concl,
                    "stmt": r["stmt"], "family": r.get("ccp_family"),
                    "rowkind": r.get("ccp_rowkind"), "head_note": "state→theorem 归一",
                    "gold_ops": r.get("chain_ops")})
    return out


def anchor_items():
    p = os.path.join(G13, "CCP_ANCHOR.json")
    if not os.path.exists(p):
        return {}
    doc = json.load(open(p, encoding="utf-8"))
    out = collections.defaultdict(list)
    for cls, items in doc["classes"].items():
        for it in items:
            out[cls].append({"cls": cls, "name": it["name"], "head": it["head"],
                             "concl": it["concl"], "stmt": it.get("stmt", ""),
                             "src": "valid_anchor"})
    return out


def run_set(items, jobs, tag, judge=True):
    recs = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=jobs) as ex:
        futs = {ex.submit(arm_attempt, it["cls"], it["name"], it["head"], it["concl"],
                          judge, tag): it for it in items}
        done = 0
        for f in as_completed(futs):
            recs.append(f.result())
            done += 1
            if done % 25 == 0:
                print("[rulearm] %d/%d (%.0fs)" % (done, len(items), time.time() - t0),
                      flush=True)
    return recs


def summarize(recs):
    per = collections.defaultdict(lambda: {"n": 0, "have_plan": 0, "certified": 0,
                                          "why": collections.Counter()})
    for r in recs:
        d = per[r["cls"]]
        d["n"] += 1
        d["have_plan"] += int(r["have_plan"])
        d["certified"] += int(r["certified"])
        d["why"][r["reason"]] += 1
    out = {}
    for cls, d in per.items():
        out[cls] = {"n": d["n"], "have_plan": d["have_plan"], "certified": d["certified"],
                    "cert_rate": round(d["certified"] / max(1, d["n"]), 4),
                    "plan_rate": round(d["have_plan"] / max(1, d["n"]), 4),
                    "why": dict(d["why"])}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cls", default="all")
    ap.add_argument("--set", default="synth_held", choices=["synth_held", "anchor", "both"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--no-judge", action="store_true")
    a = ap.parse_args()
    clss = ["ind", "ineq", "nt"] if a.cls == "all" else [a.cls]
    rep = {"schema": "ccp-rulearm/1", "argv": " ".join(sys.argv[1:]),
           "arm": "桥规则 + 白名单链模板 + 常量回填 (零模型/零训练/零 GPU)",
           "judge": "Lean ∧ N1 ∧ N2 (drct_common.judge_attempt)",
           "judge_md5": _md5(os.path.join(RES, "scratch", "drct", "drct_common.py")),
           "sets": {}, "generated": time.strftime("%Y-%m-%dT%H:%M:%S")}
    for setname in (["synth_held", "anchor"] if a.set == "both" else [a.set]):
        items = []
        if setname == "synth_held":
            for cls in clss:
                items += synth_held_items(cls)
        else:
            anc = anchor_items()
            for cls in clss:
                items += anc.get(cls, [])
        if a.limit:
            items = items[:a.limit]
        if not items:
            print("[rulearm] %s: empty set (skipped)" % setname)
            continue
        recs = run_set(items, a.jobs, setname, judge=not a.no_judge)
        summ = summarize(recs)
        rep["sets"][setname] = {"summary": summ, "records": recs}
        print("[rulearm] == %s ==" % setname, flush=True)
        for cls, s in summ.items():
            print("   %-5s n=%d plan=%d certified=%d rate=%.3f why=%s"
                  % (cls, s["n"], s["have_plan"], s["certified"], s["cert_rate"],
                     json.dumps(s["why"], ensure_ascii=False)[:160]), flush=True)
    p = os.path.join(OUT, "RULEARM_%s.json" % a.set)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=1, default=str)
    print("[rulearm] → %s md5=%s" % (p, _md5(p)))
    print("CCP_RULEARM_RC=0")


if __name__ == "__main__":
    main()

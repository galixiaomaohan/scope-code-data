# -*- coding: utf-8 -*-
# gen44_accept.py -- Arm A (empirical accept arm): structural accept rows -> frozen judging module
# certification (certified) + statement-level compilability layer (A0).
#
# Measurement convention (V44_DESIGN sections 1/3):
#   A2 = judge_attempt triple (Lean accepts AND N1 non-decorative AND N2 passes) => certified
#   A0 = the statement itself compiles (render_lean(proof="sorry") + run_lean); used to separate
#        mathlib version drift (I11) into its own column.
#   Attribution arm = for A2 rejects, append a skeleton (proof="sorry") compile, to distinguish
#        stmt_not_compilable / proof_rejected.
#
# Red lines: Lean under `nice -n 19` (drct_common.run_lean(nice=True)) · measure our own Lean
#   concurrency <5 before launch and at every step · this process is serial (concurrency is
#   controlled by shard) · never modify the judging module (import only) · write only under
#   scratch/ccp/gen44/**.
import argparse
import collections
import hashlib
import importlib.machinery as _im
import importlib.util as _iu
import json
import os
import random
import re
import subprocess
import sys
import time
from datetime import datetime, timezone

RES = os.environ.get("SCOPE_RESEARCH_ROOT", ".")
# HERE is the gen44 tree of the experiment host (data/lean/out subdirs there,
# not next to this file); sys.path below still gets the package-local dir for
# sibling imports.
HERE = os.path.join(RES, "scratch", "ccp", "gen44")
_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
G42 = os.path.join(RES, "scratch", "ccp", "gen42")
sys.path.insert(0, _PKG_DIR)
sys.path.insert(0, G42)

import gen42_stmt as S                                                  # noqa: E402
import gen42_families as F                                              # noqa: E402


def _load(name, path):
    ld = _im.SourceFileLoader(name, path)
    m = _iu.module_from_spec(_iu.spec_from_loader(name, ld))
    sys.modules[name] = m
    ld.exec_module(m)
    return m


DC = _load("drct_common_g44", os.path.join(RES, "scratch", "drct", "drct_common.py"))
JUDGE_MD5 = "7c31b58da9d7228cee31b9be581fe5dd"


MAXJ = [5]


def our_lean_jobs():
    # Count locally running Lean workers (allowlist: one job per GPU is enforced elsewhere).
    try:
        ps = subprocess.run(
            ["bash", "-lc",
             "ps -eo pid,cmd --no-headers | grep -F 'lake env lean' "
             "| grep -Ev grep | grep -E 'tcaboost|elan'"],
            capture_output=True, text=True).stdout
    except Exception:                                                   # noqa: BLE001
        return -1
    return len([l for l in ps.split("\n") if l.strip()])


class P(object):
    __slots__ = ("name", "head", "concl", "split", "idx")

    def __init__(self, name, head, concl, split, idx):
        self.name, self.head, self.concl, self.split, self.idx = name, head, concl, split, idx


STMT_HEADER = list(DC.OFFICIAL_HEADER)


def stmt_compile(name, head, concl, workdir, tag, timeout=240.0):
    """A0 statement-level compilability: test only whether `theorem … := by sorry` passes
    syntax/type checking.

    Not DC.render_lean: it appends `#print axioms`, so sorry always hits sorryAx and accepted is
    always False. Instead render with the same header and judge by rc/errors; the call still goes
    through DC.run_lean (same cwd/PATH/nice convention as the judging module).
    Criterion: rc == 0 and no `error:` in the output (a sorry warning is not an error)."""
    L = list(STMT_HEADER)
    L += ["", "%s %s := by" % (head, concl), "  sorry", ""]
    p = DC.safe_path(os.path.join(workdir, "g44c_%s_%s.lean" % (
        tag, re.sub(r"[^A-Za-z0-9_.-]", "_", name))))
    open(p, "w", encoding="utf-8").write("\n".join(L) + "\n")
    r = DC.run_lean(p, timeout, nice=True)
    out = (r.get("out_head") or "")
    bad = ("error:" in out) or ("unsolved goals" in out)
    r["compiles"] = bool(r["rc"] == 0 and not bad)
    r["err_head"] = out[:600]
    return r


def load_lwb(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def build(rec):
    """Rebuild the device chain plus rendered text from an LWB row. Returns dict or None."""
    d = S.parse_stmt(P(rec["name"], rec["head"], rec["concl"], rec["subset"], rec["idx"]))
    if not d["ok"]:
        return None
    got, why = F.chain_of(d)
    if got is None:
        return None
    # D12 fix (ERRATUM4): for family A, hyp_names sits at the top level of chain_of's return value,
    # not inside route_payload. Passing only route_payload made family A render "skip", so those
    # rows were misrecorded as rebuild_fail (untested). Merge the payloads.
    pl = dict(got.get("route_payload") or {})
    pl.update(got)
    pt = F.render_tactic(d, got["family"], pl, got["chain_ops"])
    if pt == "skip":
        return None
    return {"rec": d, "got": got, "text": pt}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--struct", default=os.path.join(HERE, "out", "GEN44_STRUCT.jsonl"))
    ap.add_argument("--lwb", default=os.path.join(HERE, "data", "LWB_ALL.jsonl"))
    ap.add_argument("--out", default=os.path.join(HERE, "out", "GEN44_ACCEPT.json"))
    ap.add_argument("--rows", default=os.path.join(HERE, "out", "GEN44_ACCEPT_ROWS.jsonl"))
    ap.add_argument("--lean-dir", default=os.path.join(HERE, "lean", "accept"))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--head-n", type=int, default=0,
                    help=">0: run only the first n rows (in idx order)")
    ap.add_argument("--only-route", default="", help="run a single route only (e.g. g42_subst)")
    ap.add_argument("--strat-n", type=int, default=0,
                    help=">0: draw n rows stratified by route")
    ap.add_argument("--seed", type=int, default=4400)
    ap.add_argument("--timeout", type=float, default=240.0)
    ap.add_argument("--max-lean-jobs", type=int, default=5)
    ap.add_argument("--n1n2", action="store_true", default=True)
    ap.add_argument("--no-lean", action="store_true")
    ap.add_argument("--skip-compile", action="store_true",
                    help="skip the skeleton attribution arm")
    ap.add_argument("--attr-max", type=int, default=0,
                    help="max skeleton-compile attribution runs per launch; 0 = unlimited")
    a = ap.parse_args()
    MAXJ[0] = a.max_lean_jobs
    n_attr = 0
    os.makedirs(a.lean_dir, exist_ok=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)

    rows = [json.loads(l) for l in open(a.struct, encoding="utf-8") if l.strip()]
    by = {r["idx"]: r for r in load_lwb(a.lwb)}
    order = sorted(rows, key=lambda r: r["idx"])
    if a.strat_n > 0:
        rng = random.Random(a.seed)
        strata = collections.defaultdict(list)
        for r in order:
            strata[r.get("R") or "?"].append(r)
        tot = sum(len(v) for v in strata.values())
        sel = []
        for k, v in sorted(strata.items()):
            n = max(1, int(round(a.strat_n * len(v) / max(1, tot))))
            sel += rng.sample(v, min(n, len(v)))
        order = sorted(sel, key=lambda r: r["idx"])
    if a.only_route:
        order = [r for r in order if (r.get("R") or "") == a.only_route]
    if a.head_n > 0:
        order = order[:a.head_n]
    todo = [r for i, r in enumerate(order) if i % a.nshard == a.shard]
    print("accept arm over %d struct rows (shard %d/%d, lean=%s)"
          % (len(todo), a.shard, a.nshard, not a.no_lean), flush=True)

    out_rows = []
    t0 = time.time()
    for k, sr in enumerate(todo):
        rec = by.get(sr["idx"])
        ent = {"idx": sr["idx"], "name": sr["name"], "sub": sr["sub"], "R": sr.get("R"),
               "ops": sr.get("ops")}
        if rec is None:
            ent["state"] = "lwb_missing"
            out_rows.append(ent)
            continue
        b = build(rec)
        if b is None:
            ent["state"] = "rebuild_fail"
            out_rows.append(ent)
            continue
        ent["proof_md5"] = hashlib.md5(b["text"].encode()).hexdigest()
        ent["route_recheck"] = b["got"].get("route")
        if a.no_lean:
            ent["state"] = "struct_only"
            out_rows.append(ent)
            continue
        j = our_lean_jobs()
        if j >= MAXJ[0] or j < 0:
            print("GATE_STOP lean_jobs=%s at idx=%s" % (j, sr["idx"]), flush=True)
            break
        head = S.norm_answer(rec["head"])
        concl = S.norm_answer(rec["concl"])
        v = DC.judge_attempt(rec["name"], head, concl, b["text"], a.lean_dir,
                             "g44a%d" % a.shard, a.timeout)
        ent["certified"] = bool(v.get("certified"))
        ent["reasons"] = v.get("reasons")
        ent["lean"] = v.get("lean")
        ent["n1n2"] = v.get("n1n2")
        ent["state"] = "certified" if ent["certified"] else "judged_reject"
        # Attribution arm: on reject, compile the skeleton to separate a non-compilable statement
        # (I11 mathlib drift) from a rejected proof.
        if not ent["certified"] and not a.skip_compile and n_attr < a.attr_max:
            r = stmt_compile(rec["name"], head, concl, a.lean_dir,
                             "s%d" % a.shard, a.timeout)
            n_attr += 1
            ent["stmt_compiles"] = bool(r["compiles"])
            ent["stmt_lean"] = {kk: r[kk] for kk in ("rc", "accepted", "secs", "compiles")}
            if not r["compiles"]:
                ent["reasons"] = (ent["reasons"] or []) + ["stmt_not_compilable"]
                ent["stmt_err"] = (r.get("err_head") or "")[:300]
        out_rows.append(ent)
        if (k + 1) % 10 == 0:
            nc = sum(1 for r in out_rows if r.get("certified"))
            ns = sum(1 for r in out_rows if r["state"] in ("certified", "judged_reject"))
            print("[%d/%d] struct=%d certified=%d %.1fs"
                  % (k + 1, len(todo), ns, nc, time.time() - t0), flush=True)
            with open(a.rows + ".s%d" % a.shard, "w", encoding="utf-8") as f:
                for r in out_rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(a.rows + ".s%d" % a.shard, "w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    judged = [r for r in out_rows if r["state"] in ("certified", "judged_reject")]
    ncert = sum(1 for r in out_rows if r.get("certified"))
    nstmt_fail = sum(1 for r in out_rows if r.get("stmt_compiles") is False)
    rep = {
        "schema": "gen44-accept/1",
        "created": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "shard": a.shard, "nshard": a.nshard,
        "n_target": len(rows), "n_todo": len(todo), "n_run": len(out_rows),
        "n_judged": len(judged), "n_certified": ncert,
        "cert_rate_of_judged": round(ncert / max(1, len(judged)), 4),
        "stmt_not_compilable": nstmt_fail,
        "route_hist": dict(collections.Counter(r.get("R") for r in out_rows
                                               if r.get("R")).most_common()),
        "cert_by_route": dict(collections.Counter(
            r.get("R") for r in out_rows if r.get("certified")).most_common()),
        "reject_hist": dict(collections.Counter(
            (r.get("reasons") or ["?"])[0] for r in out_rows).most_common()),
        "judge_md5_expected": JUDGE_MD5,
        "wall_s": round(time.time() - t0, 1),
    }
    json.dump(rep, open(a.out + ".s%d" % a.shard, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(json.dumps(rep, ensure_ascii=False, indent=1), flush=True)


if __name__ == "__main__":
    main()

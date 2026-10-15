# -*- coding: utf-8 -*-
# gen23_prescreen_batch.py -- gen23 T1 (2) **batch executor** for the full N1/N2 prescreen + literal-judge cross-check
#
# Motivation (recorded): literal DC.judge_attempt measured at ~31s per item (3 Lean runs, Mathlib import
#   dominates) => the full 4842 items take about 8.5h wall clock. Batching merges the same three-pass
#   semantics (main / N1 stripped-args / N2 strict) into **one multi-declaration file each**, paying the
#   Lean import once.
#
# Semantic equivalence (asserted per item, not approximate):
#   * each declaration block = verbatim DC.render_lean(name, head, concl, proof) from the frozen judge
#     module (official header + #print axioms); an N1 block = the same rendering after per-line
#     DC.strip_tactic_args; an N2 block = the same rendering with strict=True;
#   * per-item verdict = "no error in the declaration's line range and the name's #print axioms was
#     emitted and no sorryAx" ≡ the per-item form of judge_attempt's (rc==0 and not sorry and not NOT_RUN);
#   * certified = accepted and not (n1_accepted and not n1_vacuous) and n2_accepted  (verbatim the same
#     reasons logic as DC.certify);
#   * declarations inside a batch are mutually independent (a Lean error does not abort later
#     declarations; no cross-declaration dependency);
#   * cross-check: --crosscheck re-judges a registered stratified sample one by one with the **literal
#     DC.judge_attempt** and reports the agreement rate.
#
# Write scope: scratch/ccp/gen23/** only. nice -n 19; zero GPU.
import argparse
import collections
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import time

RES = os.environ.get("SCOPE_RESEARCH_ROOT", ".")
CCP = os.path.join(RES, "scratch", "ccp")
G23 = os.path.join(CCP, "gen23")
OUT = os.path.join(G23, "out")
LEANWD = os.path.join(G23, "lean")
PB = os.path.join(OUT, "A_CORPUS_BUILD.jsonl")
PT = os.path.join(OUT, "A_PRESCREEN_TRACE.jsonl")
PX = os.path.join(OUT, "A_PRESCREEN_CROSSCHECK.json")
for _p in (os.path.join(RES, "scratch", "drct"),):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import drct_common as DC                                             # noqa: E402
from concurrent.futures import ThreadPoolExecutor, as_completed      # noqa: E402

LOG = os.path.join(G23, "logs", "gen23_prescreen.log")
os.makedirs(os.path.dirname(LOG), exist_ok=True)
RE_ERR = re.compile(r"^(.*?):(\d+):(\d+): (error|warning): (.*)$", re.M)
RE_AX = re.compile(r"'([A-Za-z0-9_.]+)' (?:depends on axioms: \[([^\]]*)\]"
                   r"|does not depend on any axioms)")


def md5f(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def log(msg):
    line = "[%s] %s" % (time.strftime("%Y-%m-%dT%H:%M:%S%z"), msg)
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run_lean_full(path, timeout=7200.0):
    path = os.path.abspath(path)          # relative path + cwd=miniF2F cannot find the file (recorded: one debug run misreported notrun)
    env = dict(os.environ)
    env["PATH"] = DC._ELAN_BIN + os.pathsep + env.get("PATH", "")
    t0 = time.time()
    try:
        cp = subprocess.run(["nice", "-n", "19", "lake", "env", "lean", path],
                            capture_output=True, text=True, timeout=timeout, cwd=DC.MINIF2F, env=env)
        return {"rc": cp.returncode, "out": (cp.stdout or "") + (cp.stderr or ""),
                "secs": round(time.time() - t0, 2)}
    except subprocess.TimeoutExpired:
        return {"rc": 124, "out": "LEAN_TIMEOUT", "secs": round(time.time() - t0, 2)}


def ranges_from_file(path):
    """Derive each declaration's line range **from the written file itself** (no arithmetic
    accumulation, so line-number drift is impossible).
    A block spans '^theorem <name>' to the first following '#print axioms' line."""
    lines = open(path, encoding="utf-8").read().split("\n")
    rng, i = {}, 0
    while i < len(lines):
        m = re.match(r"^theorem (\S+)", lines[i])
        if m:
            nm = m.group(1)
            j = i
            while j < len(lines) and not lines[j].strip().startswith("#print axioms"):
                j += 1
            rng.setdefault(nm, (i + 1, min(j + 1, len(lines))))
            i = j + 1
        else:
            i += 1
    return rng


def parse_pass(out, ranges):
    """ranges: {name: (line_start, line_end)} -> {name: {accepted, err, axioms, ax_present}}
    accepted criterion (equivalent to DC.certify's Lean-level criterion): the declaration emits its
    #print axioms line **by name** and the axiom list **does not contain sorryAx** (a failed proof
    always lands in sorryAx) and its line range has no error."""
    errs = collections.defaultdict(list)
    for m in RE_ERR.finditer(out):
        ln = int(m.group(2))
        for nm, (a, b) in ranges.items():
            if a <= ln <= b:
                if m.group(4) == "error":
                    errs[nm].append("%s:%s: %s" % (m.group(4), ln, m.group(5)[:160]))
                break
    ax = {}
    for m in RE_AX.finditer(out):
        ax[m.group(1)] = [] if m.group(2) is None else [x.strip() for x in m.group(2).split(",")]
    res = {}
    for nm, (a, b) in ranges.items():
        e = errs.get(nm) or []
        axs = ax.get(nm)
        sorry = bool(axs) and any(x == "sorryAx" for x in axs)
        res[nm] = {"accepted": bool(nm in ax and not sorry and not e), "err": e[:3],
                   "axioms": axs, "ax_present": nm in ax, "sorry_ax": sorry}
    return res


HEADER_N = 3          # render_lean prefix = [import line, open scoped line, blank line]


def block_of(it, mode, first=False):
    """Declaration block in the frozen judge module's own form; non-first blocks in a batch drop the
    repeated import header (Lean forbids imports mid-file)."""
    if mode == "main":
        blk = DC.render_lean(it["name"], it["head"], it["concl"], it["proof_text"])
    elif mode == "n1":
        sk = it.get("_skel") or "\n".join(DC.strip_tactic_args(x)
                                          for x in it["proof_text"].split("\n"))
        blk = DC.render_lean(it["name"], it["head"], it["concl"], sk)
    else:
        blk = DC.render_lean(it["name"], it["head"], it["concl"], it["proof_text"], strict=True)
    if not first:
        lines = blk.split("\n")
        blk = "\n".join(lines[HEADER_N:])
    return blk


def judge_items_batch(items, wd, chunk=300, tag="batch", judge_md5=None):
    """General batch judging (the frozen judge module's three-part semantics). items:
    [{name, head, concl, proof_text, **meta}]
    Returns [{ip, certified, reasons, lean, n1n2, axioms, lean_file, lean_secs}] (ip = input index)."""
    os.makedirs(wd, exist_ok=True)
    judge_md5 = judge_md5 or md5f(DC.__file__)
    out_recs = []
    for ci in range(0, len(items), chunk):
        group = items[ci:ci + chunk]
        for it in group:
            it["_n1_vacuous"] = (("\n".join(DC.strip_tactic_args(x)
                                            for x in it["proof_text"].split("\n"))).strip()
                                 == it["proof_text"].strip())
            it["_skel"] = "\n".join(DC.strip_tactic_args(x) for x in it["proof_text"].split("\n"))
        txt, rng = blocks_of(group, "main")
        p = os.path.join(wd, "%s_main_%04d.lean" % (tag, ci // chunk))
        open(p, "w", encoding="utf-8").write(txt)
        rm = run_lean_full(p)
        pm = parse_pass(rm["out"], ranges_from_file(p))
        notrun = any(k in rm["out"] for k in DC.NOT_RUN)
        sub = [it for it in group if not it["_n1_vacuous"]]
        pn = {}
        r1 = {"secs": 0.0}
        if sub:
            parts = [block_of(it, "n1", first=(i == 0)) for i, it in enumerate(sub)]
            p1 = os.path.join(wd, "%s_n1_%04d.lean" % (tag, ci // chunk))
            open(p1, "w", encoding="utf-8").write("\n".join(parts))
            r1 = run_lean_full(p1)
            pn = parse_pass(r1["out"], ranges_from_file(p1))
        parts = [block_of(it, "n2", first=(i == 0)) for i, it in enumerate(group)]
        p2 = os.path.join(wd, "%s_n2_%04d.lean" % (tag, ci // chunk))
        open(p2, "w", encoding="utf-8").write("\n".join(parts))
        r2 = run_lean_full(p2)
        pn2 = parse_pass(r2["out"], ranges_from_file(p2))
        for gi, it in enumerate(group):
            nm = it["name"]
            m = pm.get(nm) or {}
            a1 = pn.get(nm) or {}
            a2 = pn2.get(nm) or {}
            reasons = []
            if not m.get("accepted"):
                reasons.append("lean_reject" + ("_sorry" if m.get("sorry_ax") else "")
                               + ("" if m.get("ax_present") else "_no_ax")
                               + ("_notrun" if notrun else ""))
            if a1.get("accepted") and not it["_n1_vacuous"]:
                reasons.append("bare_closure_after_strip")
            if not a2.get("accepted"):
                reasons.append("autoimplicit_dependent")
            if any(x == "sorryAx" for x in (m.get("axioms") or [])):
                reasons.append("sorry_ax")
            out_recs.append({"ip": ci + gi, "id": it.get("id"), "name": nm,
                             "certified": not reasons, "reasons": reasons,
                             "lean": {"rc": 0 if m.get("accepted") else 1,
                                      "accepted": bool(m.get("accepted")),
                                      "sorry": any(x == "sorryAx" for x in (m.get("axioms") or [])),
                                      "err": (m.get("err") or [])[:2]},
                             "n1n2": {"n1_accepted": bool(a1.get("accepted")),
                                      "n2_accepted": bool(a2.get("accepted")),
                                      "n1_vacuous": bool(it["_n1_vacuous"])},
                             "axioms": m.get("axioms"), "judge_md5": judge_md5,
                             "lean_file": os.path.basename(p),
                             "lean_secs": [rm["secs"], r1.get("secs"), r2["secs"]]})
        log("JUDGE_BATCH tag=%s chunk %d/%d n=%d main=%ss n2=%ss"
            % (tag, ci // chunk + 1, (len(items) + chunk - 1) // chunk, len(group),
               rm["secs"], r2["secs"]))
    return out_recs


def blocks_of(items, mode):
    """mode: main | n1 | n2 -- generate per-item declaration blocks in the frozen judge module's own
    form; returns (text, ranges)."""
    parts, ranges, ln = [], {}, 0
    for i, it in enumerate(items):
        blk = block_of(it, mode, first=(i == 0))
        lines = blk.split("\n")
        ranges[it["name"]] = (ln + 1, ln + len(lines))
        ln += len(lines) + 1
        parts.append(blk)
    return "\n".join(parts), ranges


def prescreen_batch(chunk=150, only=None, resume=True):
    """Full-corpus prescreen -> A_PRESCREEN_TRACE.jsonl (append, resumable); judging logic =
    judge_items_batch."""
    t0 = time.time()
    items = [json.loads(l) for l in open(PB, encoding="utf-8") if l.strip()]
    items = [it for it in items if it.get("status") == "ok"]
    if only:
        items = [it for it in items if re.match(only, it["id"])]
    done = set()
    if resume and os.path.exists(PT):
        for l in open(PT, encoding="utf-8"):
            if l.strip():
                try:
                    done.add(json.loads(l)["id"])
                except Exception:                                     # noqa: BLE001
                    pass
    todo = [it for it in items if it["id"] not in done]
    judge_md5 = md5f(DC.__file__)
    n1_skip = 0
    for it in todo:
        _sk = "\n".join(DC.strip_tactic_args(x) for x in it["proof_text"].split("\n"))
        if _sk.strip() == it["proof_text"].strip():
            n1_skip += 1
    log("BATCH todo=%d (done=%d) chunk=%d | N1 vacuous skipped=%d | judge_md5=%s"
        % (len(todo), len(done), chunk, n1_skip, judge_md5))
    wd = os.path.join(LEANWD, "batch")
    res = judge_items_batch(todo, wd, chunk=chunk, tag="corpus", judge_md5=judge_md5)
    with open(PT, "a", encoding="utf-8") as fo:
        for it, j in zip(todo, res):
            rec = {"schema": "ccp-a-prescreen-trace/1",
                   "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "id": it["id"], "name": it["name"],
                   "cls": it["cls"], "src": it["src"], "layer": it["layer"],
                   "rowkind": it["rowkind"], "target_arm": it["target_arm"],
                   "presentation": it["presentation"], "text_md5": it["text_md5"],
                   "stmt_md5": it["stmt_md5"], "judge_md5": judge_md5,
                   "judge_entry": "drct_common.judge_attempt + N1/N2 (冻结判卷件, 零改动)",
                   "runner": "batch_v1 (逐项语义=judge_attempt 三件套; 见脚本头注)",
                   "certified": j["certified"], "reasons": j["reasons"], "lean": j["lean"],
                   "n1n2": j["n1n2"], "axioms": j.get("axioms"),
                   "lean_file": j["lean_file"], "lean_secs": j["lean_secs"]}
            fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
    log("BATCH DONE written=%d wall=%.1fs trace=%s md5=%s"
        % (len(res), time.time() - t0, PT, md5f(PT)))


def crosscheck(n_per_stratum=25, seed=20260923, workers=5):
    """Literally re-judge a stratified sample with DC.judge_attempt (item-by-item comparison against the
    batch executor)."""
    items = [json.loads(l) for l in open(PB, encoding="utf-8") if l.strip()]
    items = [it for it in items if it.get("status") == "ok"]
    strata = collections.defaultdict(list)
    for it in items:
        strata["%s|%s" % (it["layer"], it["cls"])].append(it)
    rnd = random.Random(seed)
    sample = []
    for k in sorted(strata):
        pool = sorted(strata[k], key=lambda x: x["name"])
        sample += rnd.sample(pool, min(n_per_stratum, len(pool)))
    batch = {}
    for l in open(PT, encoding="utf-8"):
        if l.strip():
            r = json.loads(l)
            batch[r["id"]] = r
    log("CROSSCHECK n=%d strata=%s" % (len(sample), sorted(strata)))
    wd = os.path.join(LEANWD, "crosscheck")
    os.makedirs(wd, exist_ok=True)

    def one(it):
        t = time.time()
        j = DC.judge_attempt(it["name"], it["head"], it["concl"], it["proof_text"], wd, "xc_%s" % it["layer"])
        b = batch.get(it["id"]) or {}
        return {"id": it["id"], "layer": it["layer"], "cls": it["cls"],
                "literal": {"certified": bool(j.get("certified")), "reasons": j.get("reasons"),
                            "lean": j.get("lean"), "n1n2": j.get("n1n2")},
                "batch": {"certified": b.get("certified"), "reasons": b.get("reasons"),
                          "n1n2": b.get("n1n2")},
                "agree": bool(j.get("certified")) == bool(b.get("certified")),
                "wall_s": round(time.time() - t, 2)}

    res = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for fu in as_completed([ex.submit(one, it) for it in sample]):
            r = fu.result()
            res.append(r)
            log("XC %s literal=%s batch=%s agree=%s %.1fs" % (r["id"], r["literal"]["certified"],
                                                             r["batch"]["certified"], r["agree"], r["wall_s"]))
    out = {"schema": "ccp-a-prescreen-crosscheck/1", "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "seed": seed, "n_per_stratum": n_per_stratum, "n": len(res),
           "n_agree": sum(1 for r in res if r["agree"]),
           "judge_md5": md5f(DC.__file__), "records": sorted(res, key=lambda r: r["id"])}
    json.dump(out, open(PX, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    log("CROSSCHECK DONE n=%d agree=%d md5=%s" % (len(res), out["n_agree"], md5f(PX)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=("batch", "crosscheck"))
    ap.add_argument("--chunk", type=int, default=150)
    ap.add_argument("--only", default=None)
    ap.add_argument("--n-per-stratum", type=int, default=25)
    ap.add_argument("--workers", type=int, default=5)
    a = ap.parse_args()
    if a.mode == "batch":
        prescreen_batch(chunk=a.chunk, only=a.only)
    else:
        crosscheck(n_per_stratum=a.n_per_stratum, workers=a.workers)


if __name__ == "__main__":
    main()

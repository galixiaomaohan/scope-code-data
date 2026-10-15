# -*- coding: utf-8 -*-
# gen37_judge.py -- gen37 external baseline [3]: generated text -> frozen Lean batch judging (certified, same convention)
#
# Judging = scratch/drct/drct_common.py (frozen 7c31b58d, import only, zero edits):
#   proof = DC.extract_proof(text) (protocol §8 form) -> DC.judge_attempt (real Lean judging + N1 e0a + N2).
# Metering: the judging summary merges gen37 per-problem token/wall readings (in/out tokens, wall_s) into the efficiency trio.
# Discipline: writes only under scratch/ccp/gen37/**; whole run under nice -n 19 (pre-launch: our Lean jobs < 5); zero sudo / zero /tmp.
import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

RES = os.environ.get("SCOPE_RESEARCH_ROOT", ".")
G37 = RES + "/scratch/ccp/gen37"
DRCT = RES + "/scratch/drct"
JUDGE_MD5_EXPECT = "7c31b58da9d7228cee31b9be581fe5dd"
WD = G37 + "/lean/main"

_G = {}


def md5f(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def app(msg, logp=None):
    line = "[%s] %s" % (time.strftime("%Y-%m-%dT%H:%M:%S%z"), msg)
    print(line, flush=True)
    if logp:
        with open(logp, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def n_lean_jobs():
    import subprocess
    try:
        ps = subprocess.run(["bash", "-lc",
                             "ps -eo pid,cmd --no-headers | grep -F 'lake env lean' "
                             "| grep -Ev 'grep' | grep -E 'tcaboost|\\.elan'"],
                            capture_output=True, text=True).stdout
    except Exception:                                                # noqa: BLE001
        return -1
    return len([l for l in ps.split("\n") if l.strip()])


def _init(wd, tag):
    sys.path.insert(0, DRCT)
    import drct_common as DC
    _G["DC"], _G["wd"], _G["tag"] = DC, wd, tag


def _one(rec):
    DC, wd, tag = _G["DC"], _G["wd"], _G["tag"]
    t0 = time.time()
    try:
        j = DC.judge_attempt(rec["name"], rec["head"], rec["concl"], rec["proof"], wd, tag)
        exc = None
    except Exception as e:                                           # noqa: BLE001
        j = {"certified": None, "reasons": [], "lean": None, "n1n2": None}
        exc = "%s: %s" % (type(e).__name__, e)
    return {"problem": rec["name"], "ri": rec["ri"], "family": rec["family"],
            "cls": rec["cls"], "cls_code": rec["cls_code"],
            "certified": j.get("certified"), "reasons": j.get("reasons"),
            "lean": j.get("lean"), "n1n2": j.get("n1n2"), "exc": exc,
            "judge_wall_s": round(time.time() - t0, 2),
            "in_tokens": rec["in_tokens"], "out_tokens": rec["out_tokens"],
            "gen_wall_s": rec["wall_s"], "stop_hit": rec.get("stop_hit"),
            "proof_md5": hashlib.md5((rec["proof"] or "").encode("utf-8")).hexdigest(),
            "proof_empty": (not (rec["proof"] or "").strip()),
            "proof_lines": len([l for l in (rec["proof"] or "").split("\n") if l.strip()]),
            "raw_text_md5": rec.get("text_md5")}


def pct(x, n):
    return round(100.0 * x / max(1, n), 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default=G37 + "/out/GEN37_PROMPTS.jsonl")
    ap.add_argument("--attempts", default=G37 + "/out/GEN37_DSRL_attempts.jsonl")
    ap.add_argument("--out", default=G37 + "/out/GEN37_JUDGE")
    ap.add_argument("--tag", default="g37_a0")
    ap.add_argument("--jobs", type=int, default=32)
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--log", default=G37 + "/logs/gen37_judge.log")
    a = ap.parse_args()

    sys.path.insert(0, DRCT)
    import drct_common as DC
    jm0 = md5f(DRCT + "/drct_common.py")
    assert jm0 == JUDGE_MD5_EXPECT, "frozen judge module md5 mismatch: %s" % jm0
    nl = n_lean_jobs()
    assert nl < 5, "pre-launch gate: our Lean jobs %d >= 5" % nl
    for d in (WD, os.path.join(G37, "lean", "dual"), os.path.join(G37, "lean", "reverse")):
        os.makedirs(d, exist_ok=True)
    with open(a.prompts, encoding="utf-8") as f:
        probs = {r["name"]: r for r in (json.loads(l) for l in f if l.strip())}
    att = []
    with open(a.attempts, encoding="utf-8") as f:
        for l in f:
            if l.strip():
                att.append(json.loads(l))
    if a.limit:
        att = att[:a.limit]
    miss = [r["name"] for r in att if r["name"] not in probs]
    assert not miss, "attempts contain problem names outside the exam: %s" % miss[:3]
    todo = []
    for r in att:
        p = probs[r["name"]]
        proof = DC.extract_proof(r.get("text") or "")
        todo.append({"name": r["name"], "ri": r["ri"], "family": r["family"], "cls": r["cls"],
                     "cls_code": r["cls_code"], "head": p["head"], "concl": p["concl"],
                     "proof": proof, "in_tokens": r["in_tokens"], "out_tokens": r["out_tokens"],
                     "wall_s": r["wall_s"], "stop_hit": r.get("stop_hit"),
                     "text_md5": r.get("text_md5")})
    app("JUDGE start n=%d jobs=%d judge=%s lean_ours=%d" % (len(todo), a.jobs, jm0[:8], nl), a.log)
    recs = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.jobs, initializer=_init,
                             initargs=(WD, a.tag)) as ex:
        futs = [ex.submit(_one, r) for r in todo]
        for i, fu in enumerate(as_completed(futs), 1):
            recs.append(fu.result())
            if i % 20 == 0:
                app("judge %d/%d" % (i, len(todo)), a.log)
    order = {r["name"]: i for i, r in enumerate(todo)}
    recs.sort(key=lambda r: order.get(r["problem"], 1 << 30))
    with open(a.out + "_records.jsonl", "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n = len(recs)
    cert = [r for r in recs if r["certified"] is True]
    exc = [r for r in recs if r["exc"]]
    rh = {}
    for r in recs:
        for x in (r["reasons"] or []):
            rh[x] = rh.get(x, 0) + 1
    byfam, bycls = {}, {}
    for r in recs:
        for d, k in ((byfam, r["family"]), (bycls, r["cls"])):
            e = d.setdefault(k, {"n": 0, "certified": 0})
            e["n"] += 1
            e["certified"] += 1 if r["certified"] else 0
    si = sum(r["in_tokens"] for r in recs)
    so = sum(r["out_tokens"] for r in recs)
    swg = sum(r["gen_wall_s"] for r in recs)
    swj = sum(r["judge_wall_s"] for r in recs)
    summ = {
        "schema": "gen37-judge/1", "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "tag": a.tag, "n": n, "denominator": "n (the full 218-deep-exam set; no convergence issue measured: one attempt per problem)",
        "certified": len(cert), "certified_rate": round(len(cert) / max(1, n), 6),
        "certified_pct": pct(len(cert), n),
        "lean_raw_accepted": sum(1 for r in recs if (r["lean"] or {}).get("accepted")),
        "reason_hist": rh, "exceptions": len(exc), "exception_names": [r["problem"] for r in exc],
        "proof_empty": sum(1 for r in recs if r["proof_empty"]),
        "by_family": byfam, "by_cls": bycls,
        "M1_H": "%d/%d = %.4f%%" % (len(cert), n, pct(len(cert), n)),
        "efficiency": {
            "prompt_tokens_sum": si, "completion_tokens_sum": so,
            "prompt_tokens_per_q": round(si / max(1, n), 2),
            "completion_tokens_per_q": round(so / max(1, n), 2),
            "gen_wall_sum_s": round(swg, 1), "gen_secs_per_q": round(swg / max(1, n), 2),
            "judge_wall_sum_s": round(swj, 1), "judge_wall_batch_s": round(time.time() - t0, 1),
            "toks_per_s_output": round(so / max(0.1, swg), 2),
            "note": ("judge_wall_sum = sum of per-problem judging wall clock (parallel CPU Lean); "
                     "judge_wall_batch = batch wall clock (jobs=%d)" % a.jobs)},
        "judge_md5_before": jm0, "judge_md5_after": md5f(DRCT + "/drct_common.py"),
        "judge_entry": "drct_common.judge_attempt (frozen 7c31b58d; proof=DC.extract_proof(text))",
        "certified_names": [r["problem"] for r in cert],
    }
    summ["judge_unchanged"] = bool(summ["judge_md5_before"] == summ["judge_md5_after"])
    open(a.out + "_summary.json", "w", encoding="utf-8").write(
        json.dumps(summ, ensure_ascii=False, indent=1))
    app("JUDGE done certified=%d/%d (%.4f%%) reasons=%s batch=%.0fs"
        % (len(cert), n, pct(len(cert), n), rh, time.time() - t0), a.log)
    print(json.dumps({"M1_H": summ["M1_H"], "certified": len(cert), "n": n,
                      "reasons": rh, "exceptions": len(exc),
                      "completion_tokens_per_q": summ["efficiency"]["completion_tokens_per_q"],
                      "toks_per_s_output": summ["efficiency"]["toks_per_s_output"]},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()

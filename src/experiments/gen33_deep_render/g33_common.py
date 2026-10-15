# -*- coding: utf-8 -*-
# g33_common.py -- gen33 (deep-generation V3.5) shared utilities
#
# Discipline: write surface is ONLY $R/scratch/ccp/gen33/** + LAUNCH_LOG.md; all existing artifacts read-only.
#   Zero sudo; zero /tmp; Test 243 sealed; drct_common 7c31b58d frozen read-only; GPU0 user redline absolutely off-limits.
#   Every output path passes the safe_out() project-root + write-surface assertion.
import collections
import hashlib
import json
import os
import sys
import time

sys.dont_write_bytecode = True

RES = os.environ.get("SCOPE_RESEARCH_ROOT", ".")
CCP = RES + "/scratch/ccp"
G33 = CCP + "/gen33"
G32 = CCP + "/gen32"
G30 = CCP + "/gen30"
G29 = CCP + "/gen29"
G28 = CCP + "/gen28"
G26 = CCP + "/gen26"
G13 = CCP + "/gen13"
BH = CCP + "/bh"
DRCT = RES + "/scratch/drct"
LAUNCH_LOG = CCP + "/LAUNCH_LOG.md"
HEARTBEAT = CCP + "/HEARTBEAT.gen33"

MASTER = 20261005                 # gen32 master (corpus source batch; this batch consumes only, no new problems)
JUDGE_MD5_EXPECT = "7c31b58da9d7228cee31b9be581fe5dd"
DEEP_CORPUS = G32 + "/DEEP_CORPUS_V1.jsonl"
EXAM_CAND = G32 + "/EXAM_DEEP_CANDIDATE.json"
VOCAB_V3 = G26 + "/out/CCP_VOCAB_V3.json"
ALLOW_V3 = G26 + "/cache/ccp_allow_v3.bin"

for _p in (G33, G32, BH, CCP, DRCT, os.path.join(RES, "cppcuda", "tools"),
           os.path.join(RES, "skel32"),
           os.environ.get("SCOPE_SUPEROPERATOR", "../device")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def safe_out(p, extra=()):
    """Project-root + write-surface assertion (redline: writes only under gen33/** or explicit exemptions)."""
    rp = os.path.realpath(p)
    allowed = [os.path.realpath(G33)] + [os.path.realpath(e) for e in extra]
    if not any(rp == a or rp.startswith(a + os.sep) for a in allowed):
        raise ValueError("写面越界: %r" % p)
    return rp


def md5f(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def md5s(s):
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def log(msg, path=None):
    line = "[%s] %s" % (time.strftime("%Y-%m-%dT%H:%M:%S%z"), msg)
    print(line, flush=True)
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def heartbeat(tag, extra=""):
    line = "GEN33_HEARTBEAT %s %s %s\n" % (time.strftime("%Y-%m-%dT%H:%M:%S%z"), tag, extra)
    try:
        with open(HEARTBEAT, "a", encoding="utf-8") as f:
            f.write(line)
    except Exception:                                                # noqa: BLE001
        pass


def launch_log(line):
    """LAUNCH_LOG append-only (the only exempt write surface)."""
    with open(LAUNCH_LOG, "a", encoding="utf-8") as f:
        f.write(line.rstrip("\n") + "\n")


def jlload(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def jlsave(rows, p):
    safe_out(p)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return md5f(p)


def save_json(p, obj):
    safe_out(p)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
    return md5f(p)


def canon(v):
    return json.dumps(v, sort_keys=True, ensure_ascii=False, default=str)


def norm(s):
    import re
    return re.sub(r"\s+", " ", s or "").strip()


def parent_ledger():
    """Parent-artifact md5 ledger (measured once at start and once at finish; the two readings must match)."""
    out = {}
    for k, p in (("gen30_hybrid", G30 + "/gen30_hybrid.py"),
                 ("gen30_eval", G30 + "/gen30_eval.py"),
                 ("gen30_planattn", G30 + "/gen30_planattn.py"),
                 ("gen26_hybuild", G26 + "/gen26_hybuild.py"),
                 ("gen26s_vocab3", G26 + "/gen26s_vocab3.py"),
                 ("gen13_ccp_cachefull", G13 + "/ccp_cachefull.py"),
                 ("drct_common", DRCT + "/drct_common.py"),
                 ("vocab_v3", VOCAB_V3), ("allow_v3", ALLOW_V3),
                 ("deep_corpus", DEEP_CORPUS), ("exam_cand", EXAM_CAND)):
        out[k] = {"path": p, "md5": md5f(p) if os.path.exists(p) else None,
                  "bytes": os.path.getsize(p) if os.path.exists(p) else None}
    return out


def judge_gate():
    """Frozen judging-module md5 gate (import only, zero modification)."""
    import drct_common as DC
    jm = md5f(DC.__file__)
    if jm != JUDGE_MD5_EXPECT:
        raise SystemExit("FATAL 判卷件非冻结版: md5=%s 期望=%s" % (jm, JUDGE_MD5_EXPECT))
    return jm


def gpu_gate(dev, logp=None):
    """Single-GPU exclusivity gate (free > 4500MiB). GPU0 redline: structurally refused."""
    import subprocess
    idx = dev.split(":")[1] if ":" in dev else "0"
    if idx == "0":
        raise SystemExit("GPU0 用户红线: 本批绝对禁用 GPU0 (dev=%s)" % dev)
    q = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,memory.total",
                        "--format=csv,noheader,nounits", "-i", idx],
                       capture_output=True, text=True).stdout.strip()
    used, total = [int(x) for x in q.split(",")[1:3]]
    free = total - used
    log("GPU GATE dev=%s used=%dMiB free=%dMiB (require >4500MiB)" % (dev, used, free), logp)
    if free <= 4500:
        raise SystemExit("GPU 独占门未过: free=%dMiB" % free)
    return {"dev": dev, "used": used, "total": total, "free": free}


def n_lean_jobs():
    """Count of our Lean jobs (pre-launch gate: <5). Attribution: cmdline contains 'lake env lean' and project root/.elan."""
    import subprocess
    try:
        ps = subprocess.run(["bash", "-lc",
                             "ps -eo pid,cmd --no-headers | grep -F 'lake env lean' "
                             "| grep -Ev 'grep' | grep -E 'tcaboost|/.elan'"],
                            capture_output=True, text=True).stdout
    except Exception:                                                # noqa: BLE001
        return -1, ""
    lines = [l for l in ps.split("\n") if l.strip()]
    return len(lines), "\n".join(lines[:5])


def lean_gate(limit=5, logp=None):
    n, det = n_lean_jobs()
    log("LEAN GATE our_jobs=%d (require <%d)" % (n, limit), logp)
    if n >= limit:
        raise SystemExit("Lean 并发门未过: %d >= %d" % (n, limit))
    return n


def counts_hist(xs):
    return dict(collections.Counter(xs).most_common())

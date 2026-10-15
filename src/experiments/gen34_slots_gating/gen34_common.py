# -*- coding: utf-8 -*-
# gen34_common.py -- shared helpers for gen34 (integer-rendering push)
#
# Discipline: write surface is **only** $R/scratch/ccp/gen34/** + LAUNCH_LOG.md; every
#       pre-existing artifact is read-only. Zero sudo; zero /tmp; Test 243 sealed;
#       drct_common 7c31b58d frozen read-only; GPU0 user red line absolutely banned.
#       All on-disk paths go through safe_out() project root + write-surface assertion.
import collections
import hashlib
import json
import os
import sys
import time

sys.dont_write_bytecode = True

RES = os.environ.get("SCOPE_RESEARCH_ROOT", ".")
CCP = RES + "/scratch/ccp"
G34 = CCP + "/gen34"
G33 = CCP + "/gen33"
G32 = CCP + "/gen32"
G30 = CCP + "/gen30"
G26 = CCP + "/gen26"
G13 = CCP + "/gen13"
BH = CCP + "/bh"
DRCT = RES + "/scratch/drct"
LAUNCH_LOG = CCP + "/LAUNCH_LOG.md"
HEARTBEAT = CCP + "/HEARTBEAT.gen34"

MASTER = 20261005               # seed of the gen32 corpus batch consumed
                               # here (training itself uses 20260923)
JUDGE_MD5_EXPECT = "7c31b58da9d7228cee31b9be581fe5dd"
DEEP_CORPUS = G32 + "/DEEP_CORPUS_V1.jsonl"
EXAM_CAND = G32 + "/EXAM_DEEP_CANDIDATE.json"

for _p in (G34, G33, G32, BH, CCP, DRCT, G13, os.path.join(RES, "cppcuda", "tools"),
           os.path.join(RES, "skel32"),
           os.environ.get("SCOPE_SUPEROPERATOR", os.path.join(RES, os.pardir, "superoperator"))):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def safe_out(p, extra=()):
    """Project root + write-surface assertion (red line: gen34/** and explicit exemptions only)."""
    rp = os.path.realpath(p)
    allowed = [os.path.realpath(G34)] + [os.path.realpath(e) for e in extra]
    if not any(rp == a or rp.startswith(a + os.sep) for a in allowed):
        raise ValueError("write-surface violation: %r" % p)
    return rp


def md5f(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def md5s(s):
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def crlf_count(p):
    """Binary CRLF count (fork-ledger line-ending convention; a CRLF parent read in
    text mode by this batch => LF-normalized)."""
    with open(p, "rb") as f:
        b = f.read()
    return b.count(bytes([13, 10]))


def log(msg, path=None):
    line = "[%s] %s" % (time.strftime("%Y-%m-%dT%H:%M:%S%z"), msg)
    print(line, flush=True)
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def heartbeat(tag, extra=""):
    line = "GEN34_HEARTBEAT %s %s %s\n" % (time.strftime("%Y-%m-%dT%H:%M:%S%z"), tag, extra)
    try:
        with open(HEARTBEAT, "a", encoding="utf-8") as f:
            f.write(line)
    except Exception:                                                # noqa: BLE001
        pass


def launch_log(line):
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


def parent_ledger():
    """Parent-file md5 ledger (measured once at start and once at close; must match)."""
    pr = [("gen33_hybrid", G33 + "/gen33_hybrid.py"),
          ("gen33_eval", G33 + "/gen33_eval.py"),
          ("gen33_planattn", G33 + "/gen33_planattn.py"),
          ("gen33_judge", G33 + "/gen33_judge.py"),
          ("gen33_device", G33 + "/gen33_device.py"),
          ("gen32_families", G32 + "/gen32_families.py"),
          ("gen32_ops", G32 + "/gen32_ops.py"),
          ("gen32_corpus", G32 + "/gen32_corpus.py"),
          ("bh_poly", BH + "/bh_poly.py"),
          ("g33_common", G33 + "/g33_common.py"),
          ("gen13_v32ccp_train", G13 + "/v32ccp_train.py"),
          ("gen13_ccp_cachefull", G13 + "/ccp_cachefull.py"),
          ("drct_common", DRCT + "/drct_common.py"),
          ("deep_corpus", DEEP_CORPUS), ("exam_cand", EXAM_CAND),
          ("deep_all_bin", G33 + "/cache/deep_all.bin"),
          ("render_sidecar", G33 + "/out/DEEP_RENDER_SIDECAR.jsonl"),
          ("exam_sidecar", G33 + "/out/DEEP_EXAM_SIDECAR.jsonl"),
          ("c4_prefilter", G33 + "/out/C4_EXAM_PREFILTER.json"),
          ("ckpt_base_gen33", G33 + "/ckpt/deep_base_formal_6000.pt"),
          ("vocab_v4", G33 + "/cache/CCP_VOCAB_V4.json"),
          ("allow_v4", G33 + "/cache/ccp_allow_v4.bin")]
    out = {}
    for k, p in pr:
        out[k] = {"path": p, "md5": md5f(p) if os.path.exists(p) else None,
                  "bytes": os.path.getsize(p) if os.path.exists(p) else None}
    return out


def judge_gate():
    import drct_common as DC
    jm = md5f(DC.__file__)
    if jm != JUDGE_MD5_EXPECT:
        raise SystemExit("FATAL judge module is not the frozen version: md5=%s expected=%s"
                         % (jm, JUDGE_MD5_EXPECT))
    return jm


def gpu_gate(dev, logp=None):
    """Exclusive single-GPU gate (free > 4500MiB). GPU0 user red line: structurally rejected."""
    import subprocess
    idx = dev.split(":")[1] if ":" in dev else "0"
    if idx == "0":
        raise SystemExit("GPU0 user red line: GPU0 is absolutely banned in this batch (dev=%s)"
                         % dev)
    q = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,memory.total",
                        "--format=csv,noheader,nounits", "-i", idx],
                       capture_output=True, text=True).stdout.strip()
    used, total = [int(x) for x in q.split(",")[1:3]]
    free = total - used
    log("GPU GATE dev=%s used=%dMiB free=%dMiB (require >4500MiB)" % (dev, used, free), logp)
    if free <= 4500:
        raise SystemExit("GPU exclusive gate failed: free=%dMiB" % free)
    return {"dev": dev, "used": used, "total": total, "free": free}


def n_lean_jobs():
    """Count of our own Lean jobs (pre-launch gate: <5)."""
    import subprocess
    try:
        ps = subprocess.run(["bash", "-lc",
                             "ps -eo pid,cmd --no-headers | grep -F 'lake env lean' "
                             "| grep -Ev 'grep' | grep -E 'tcaboost|\\.elan'"],
                            capture_output=True, text=True).stdout
    except Exception:                                                # noqa: BLE001
        return -1, ""
    lines = [l for l in ps.split("\n") if l.strip()]
    return len(lines), "\n".join(lines[:5])


def lean_gate(limit=5, logp=None):
    n, det = n_lean_jobs()
    log("LEAN GATE our jobs=%d (require <%d)" % (n, limit), logp)
    if n >= limit:
        raise SystemExit("Lean concurrency gate failed: %d >= %d" % (n, limit))
    return n


def counts_hist(xs):
    return dict(collections.Counter(xs).most_common())

# -*- coding: utf-8 -*-
# gen35_common.py -- shared helpers for gen35 (pure planning + mechanical rendering)
#
# Discipline: write scope is **only** $R/scratch/ccp/gen35/** + LAUNCH_LOG.md; every pre-existing
#       artifact is read-only (fork/import pattern). No sudo; no /tmp; Test 243 sealed;
#       drct_common 7c31b58d frozen read-only; GPU0 user redline strictly forbidden.
#       All output paths go through safe_out() (project root + write-scope assertion).
import collections
import hashlib
import json
import os
import sys
import time

sys.dont_write_bytecode = True

RES = os.environ.get("SCOPE_RESEARCH_ROOT", ".")
CCP = RES + "/scratch/ccp"
G35 = CCP + "/gen35"
G34 = CCP + "/gen34"
G33 = CCP + "/gen33"
G32 = CCP + "/gen32"
G30 = CCP + "/gen30"
G26 = CCP + "/gen26"
G13 = CCP + "/gen13"
BH = CCP + "/bh"
DRCT = RES + "/scratch/drct"
LAUNCH_LOG = CCP + "/LAUNCH_LOG.md"
HEARTBEAT = CCP + "/HEARTBEAT.gen35"

MASTER = 20261005               # seed of the gen32 corpus batch consumed
                               # here (training itself uses 20260923)
JUDGE_MD5_EXPECT = "7c31b58da9d7228cee31b9be581fe5dd"
DEEP_CORPUS = G32 + "/DEEP_CORPUS_V1.jsonl"
EXAM_CAND = G32 + "/EXAM_DEEP_CANDIDATE.json"
EXAM_SIDECAR = G33 + "/out/DEEP_EXAM_SIDECAR.jsonl"

for _p in (G35, G34, G33, G32, BH, CCP, DRCT, G13, G26, G30, G30,
           os.path.join(CCP, "gen23"), os.path.join(CCP, "gen24"),
           os.path.join(CCP, "bh", "gen"),
           os.path.join(RES, "cppcuda", "tools"),
           os.path.join(RES, "skel32"),
           os.environ.get("SCOPE_SUPEROPERATOR", "../device")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def safe_out(p, extra=()):
    """Project root + write-scope assertion (redline: write scope is gen35/** plus explicit exemptions)."""
    rp = os.path.realpath(p)
    allowed = [os.path.realpath(G35)] + [os.path.realpath(e) for e in extra]
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
    if isinstance(s, str):
        s = s.encode("utf-8")
    return hashlib.md5(s).hexdigest()


def log(msg, path=None):
    line = "[%s] %s" % (time.strftime("%Y-%m-%dT%H:%M:%S%z"), msg)
    print(line, flush=True)
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def heartbeat(tag, extra=""):
    line = "GEN35_HEARTBEAT %s %s %s\n" % (time.strftime("%Y-%m-%dT%H:%M:%S%z"), tag, extra)
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
    """Parent-artifact md5 ledger (measured at start and at close-out; values must match)."""
    pr = [("gen34_hybrid", G34 + "/gen34_hybrid.py"),
          ("gen34_eval", G34 + "/gen34_eval.py"),
          ("gen34_planattn", G34 + "/gen34_planattn.py"),
          ("gen34_judge", G34 + "/gen34_judge.py"),
          ("gen33_device", G33 + "/gen33_device.py"),
          ("gen33_judge", G33 + "/gen33_judge.py"),
          ("gen32_families", G32 + "/gen32_families.py"),
          ("gen32_ops", G32 + "/gen32_ops.py"),
          ("gen32_corpus", G32 + "/gen32_corpus.py"),
          ("gen32_poly", G32 + "/gen32_poly.py"),
          ("gen32_check", G32 + "/gen32_check.py"),
          ("bh_poly", BH + "/bh_poly.py"),
          ("g33_common", G33 + "/g33_common.py"),
          ("gen13_v32ccp_train", G13 + "/v32ccp_train.py"),
          ("gen13_ccp_cachefull", G13 + "/ccp_cachefull.py"),
          ("drct_common", DRCT + "/drct_common.py"),
          ("deep_corpus", DEEP_CORPUS), ("exam_cand", EXAM_CAND),
          ("exam_sidecar", EXAM_SIDECAR),
          ("deep_all_bin", G33 + "/cache/deep_all.bin"),
          ("render_sidecar", G33 + "/out/DEEP_RENDER_SIDECAR.jsonl"),
          ("slot_sidecar_corpus", G34 + "/out/DEEP_SLOT_SIDECAR.jsonl"),
          ("slot_sidecar_exam", G34 + "/out/DEEP_EXAM_SLOT_SIDECAR.jsonl"),
          ("ckpt_base_gen33", G33 + "/ckpt/deep_base_formal_6000.pt"),
          ("ckpt_slotA_gen34", G34 + "/ckpt/deep_slotA_formal_6000.pt"),
          ("vocab_v4", G33 + "/cache/CCP_VOCAB_V4.json"),
          ("allow_v4", G33 + "/cache/ccp_allow_v4.bin"),
          ("tables_gen34", G34 + "/out/GEN34_TABLES.json")]
    out = {}
    for k, p in pr:
        out[k] = {"path": p, "md5": md5f(p) if os.path.exists(p) else None,
                  "bytes": os.path.getsize(p) if os.path.exists(p) else None}
    return out


def judge_gate():
    import drct_common as DC
    jm = md5f(DC.__file__)
    if jm != JUDGE_MD5_EXPECT:
        raise SystemExit("FATAL 判卷件非冻结版: md5=%s 期望=%s" % (jm, JUDGE_MD5_EXPECT))
    return jm


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
    log("LEAN GATE our_jobs=%d (require <%d)" % (n, limit), logp)
    if n >= limit:
        raise SystemExit("Lean 并发门未过: %d >= %d" % (n, limit))
    return n


def gpu_gate(dev, logp=None):
    """Single-GPU exclusive gate (free > 4500MiB). GPU0 user redline: structurally rejected."""
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


def counts_hist(xs):
    return dict(collections.Counter(xs).most_common())


def wilson_lb(k, n, z=1.959964):
    """95% Wilson lower bound for a binomial rate (used by the M3_F significance gate)."""
    if n <= 0:
        return 0.0
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return max(0.0, (c - h) / d)

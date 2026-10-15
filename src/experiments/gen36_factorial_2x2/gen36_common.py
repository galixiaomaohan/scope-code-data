# -*- coding: utf-8 -*-
# gen36_common.py -- shared helpers for gen36 (terminal-step routing push) (forked from gen35_common.py; parent untouched)
#
# Discipline: write scope is **only** $R/scratch/ccp/gen36/** + LAUNCH_LOG.md; every pre-existing artifact is
#       read-only (fork/import pattern). No sudo; no /tmp; Test 243 sealed; drct_common 7c31b58d frozen read-only.
#       GPU allowlist: **cuda:2 only**; GPU0 user redline / GPU1 reserved zone => structurally rejected (see gpu_gate).
import collections
import hashlib
import json
import os
import sys
import time

sys.dont_write_bytecode = True

RES = os.environ.get("SCOPE_RESEARCH_ROOT", ".")
CCP = RES + "/scratch/ccp"
G36 = CCP + "/gen36"
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
HEARTBEAT = CCP + "/HEARTBEAT.gen36"

MASTER = 20261005               # seed of the gen32 corpus batch this run
                               # consumes (not the training seed; training
                               # uses 20260923 via the launcher)
JUDGE_MD5_EXPECT = "7c31b58da9d7228cee31b9be581fe5dd"
DEEP_CORPUS = G32 + "/DEEP_CORPUS_V1.jsonl"
EXAM_CAND = G32 + "/EXAM_DEEP_CANDIDATE.json"
EXAM_SIDECAR = G33 + "/out/DEEP_EXAM_SIDECAR.jsonl"
CACHE_G35 = G33 + "/cache/deep_all.bin"
CACHE_A2 = G36 + "/cache/deep_all_a2.bin"           # post-step state-shift variant (V36_DESIGN D2)
PLAN_MD5_EXPECT = "89de9e657b1986c6acf3cdf944e721ca"
GPU_ALLOW = ("2",)                                   # allowlist

for _p in (G36, G35, G34, G33, G32, BH, CCP, DRCT, G13, G26, G30, G30,
           os.path.join(CCP, "gen23"), os.path.join(CCP, "gen24"),
           os.path.join(CCP, "bh", "gen"),
           os.path.join(RES, "cppcuda", "tools"),
           os.path.join(RES, "skel32"),
           os.environ.get("SCOPE_SUPEROPERATOR", "../device")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def safe_out(p, extra=()):
    """Project root + write-scope assertion (redline: write scope is gen36/** plus explicit exemptions)."""
    rp = os.path.realpath(p)
    allowed = [os.path.realpath(G36)] + [os.path.realpath(e) for e in extra]
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
    line = "GEN36_HEARTBEAT %s %s %s\n" % (time.strftime("%Y-%m-%dT%H:%M:%S%z"), tag, extra)
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
    """Parent-artifact md5 ledger (measured once at start and once at close-out; values must match). All gen35 files are read-only parents for this batch."""
    pr = [("gen35_hybrid", G35 + "/gen35_hybrid.py"),
          ("gen35_eval", G35 + "/gen35_eval.py"),
          ("gen35_common", G35 + "/gen35_common.py"),
          ("gen35_compile", G35 + "/gen35_compile.py"),
          ("gen35_armrun", G35 + "/gen35_armrun.py"),
          ("gen35_judge", G35 + "/gen35_judge.py"),
          ("gen35_tables", G35 + "/gen35_tables.py"),
          ("gen35_gate", G35 + "/gen35_gate.py"),
          ("gen35_paeq", G35 + "/gen35_paeq.py"),
          ("gen34_hybrid", G34 + "/gen34_hybrid.py"),
          ("gen34_eval", G34 + "/gen34_eval.py"),
          ("gen33_device", G33 + "/gen33_device.py"),
          ("gen32_ops", G32 + "/gen32_ops.py"),
          ("gen32_poly", G32 + "/gen32_poly.py"),
          ("gen32_families", G32 + "/gen32_families.py"),
          ("gen32_corpus", G32 + "/gen32_corpus.py"),
          ("gen13_ccp_cachefull", G13 + "/ccp_cachefull.py"),
          ("gen13_v32ccp_train", G13 + "/v32ccp_train.py"),
          ("drct_common", DRCT + "/drct_common.py"),
          ("deep_corpus", DEEP_CORPUS), ("exam_cand", EXAM_CAND),
          ("exam_sidecar", EXAM_SIDECAR),
          ("cache_g35", CACHE_G35),
          ("render_sidecar", G33 + "/out/DEEP_RENDER_SIDECAR.jsonl"),
          ("slot_sidecar_corpus", G34 + "/out/DEEP_SLOT_SIDECAR.jsonl"),
          ("slot_sidecar_exam", G34 + "/out/DEEP_EXAM_SLOT_SIDECAR.jsonl"),
          ("vocab_v4", G33 + "/cache/CCP_VOCAB_V4.json"),
          ("allow_v4", G33 + "/cache/ccp_allow_v4.bin"),
          ("ckpt_gen35_plan", G35 + "/ckpt/deep_plan_formal_6000.pt"),
          ("walk_gen35", G35 + "/out/GEN35_EVAL_PLAN_WALK.json"),
          ("tables_gen35", G35 + "/out/GEN35_TABLES.json")]
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


def gpu_gate(dev, logp=None, require_free=4500):
    """GPU allowlist gate (V36_DESIGN G-1): **cuda:2 only**.

    GPU0 = user redline (structurally rejected, not "usable when idle"); GPU1 = user reserved zone (rejected too).
    If env CUDA_VISIBLE_DEVICES is set, only "2" is allowed (otherwise reject: the visible set changes cuda:N semantics).
    """
    import subprocess
    has_cvd = "CUDA_VISIBLE_DEVICES" in os.environ
    env_cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if has_cvd and env_cvd.strip() != "2":
        raise SystemExit("CUDA_VISIBLE_DEVICES 白名单未过: %r (只允许**未设**或 '2'; "
                         "空串会遮蔽全部 GPU)" % env_cvd)
    idx = dev.split(":")[1] if ":" in dev else "0"
    if idx == "0":
        raise SystemExit("GPU0 用户红线: 本批绝对禁用 GPU0 (dev=%s)" % dev)
    if idx not in GPU_ALLOW:
        raise SystemExit("GPU 白名单外: dev=%s 允许=%s (GPU1 用户保留区同样禁用)"
                         % (dev, GPU_ALLOW))
    q = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,memory.total",
                        "--format=csv,noheader,nounits", "-i", idx],
                       capture_output=True, text=True).stdout.strip()
    used, total = [int(x) for x in q.split(",")[1:3]]
    free = total - used
    log("GPU GATE dev=%s used=%dMiB free=%dMiB (require >%dMiB)" % (dev, used, free, require_free),
        logp)
    if free <= require_free:
        raise SystemExit("GPU 独占门未过: free=%dMiB" % free)
    return {"dev": dev, "used": used, "total": total, "free": free, "env_cvd": env_cvd}


def tail_op_idx(vocab_path):
    """Terminal-step operator label indices (read from the vocabulary; the three gen32-family tail operators)."""
    v = json.load(open(vocab_path, encoding="utf-8"))
    d = v.get("ops_gen32") or {}
    return sorted(int(i) for n, i in d.items() if "tail" in n)


def counts_hist(xs):
    return dict(collections.Counter(xs).most_common())


def wilson_lb(k, n, z=1.959964):
    """95% Wilson lower bound for a binomial rate (used by the significance gate)."""
    if n <= 0:
        return 0.0
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return max(0.0, (c - h) / d)

# -*- coding: utf-8 -*-
# g32_common.py -- gen32 (layer-III long-chain deep-thinking problem-family generator batch) shared module
#
# Discipline: write scope **only** $R/scratch/ccp/gen32/** + LAUNCH_LOG.md; every existing artifact is read-only.
#             Zero GPU; zero sudo; zero /tmp; Test 243 sealed; drct_common 7c31b58d frozen read-only.
#             Every output path passes the safe_out() project-root assertion.
import collections
import hashlib
import json
import os
import re
import sys
import time

sys.dont_write_bytecode = True

RES = os.environ.get("SCOPE_RESEARCH_ROOT", ".")
CCP = RES + "/scratch/ccp"
G32 = CCP + "/gen32"
G23O = CCP + "/gen23/out"
G28 = CCP + "/gen28"
G29 = CCP + "/gen29"
BH = CCP + "/bh"
DRCT = RES + "/scratch/drct"
MINIF2F = os.environ.get("SCOPE_MINIF2F", os.path.join(RES, "lean", "miniF2F-main"))
LAUNCH_LOG = CCP + "/LAUNCH_LOG.md"
HEARTBEAT = CCP + "/HEARTBEAT.gen32"

for _p in (BH, CCP, DRCT, G32):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---- New seeds for this batch (disjoint from all history; see G32_SEED_REGISTRY.json)
MASTER = 20261005
SEQ_BASE = 200001            # corpus seq (structurally disjoint from bh bridge 100001+ / exam sets 1..600)
DEEP_SEQ_BASE = 300001       # deep exam-set candidate-layer seq (separate problem-name space)
WS = re.compile(r"\s+")
THEOREM = "theorem "

# The 11 bh_* operators of the exam main criterion (verbatim from gen29 g29_common; this batch uses them only
# for the "relation to the exam operator space" disclosure)
EXAM_OPS11 = ["bh_ind_close_const", "bh_ind_close_sq2", "bh_ind_close_sq4",
              "bh_ind_geo", "bh_ind_sum_d1", "bh_ind_sum_d3",
              "bh_ineq_block", "bh_ineq_cross", "bh_ineq_tail_other",
              "bh_ineq_tail_square", "bh_ineq_tail_zero"]


def safe_out(p, extra=()):
    """Project-root + write-scope assertion (red line: write scope is gen32/** and explicitly
    exempted files only)."""
    rp = os.path.realpath(p)
    allowed = [os.path.realpath(G32)] + [os.path.realpath(e) for e in extra]
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


def norm(s):
    return WS.sub(" ", s or "").strip()


def jlload(p):
    with open(p, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def jlsave(rows, p):
    safe_out(p)
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, p)
    return md5f(p)


def save_json(p, obj):
    safe_out(p)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1, default=str)
    os.replace(tmp, p)
    return md5f(p)


def log(msg, path=None):
    line = "[%s] %s" % (time.strftime("%Y-%m-%dT%H:%M:%S%z"), msg)
    print(line, flush=True)
    if path:
        safe_out(path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def heartbeat(tag, extra=""):
    """Heartbeat (one of the three obligations): append a globals line."""
    try:
        with open(HEARTBEAT, "a", encoding="utf-8") as f:
            f.write("GEN32_HEARTBEAT %s %s %s\n"
                    % (time.strftime("%Y-%m-%dT%H:%M:%S%z"), tag, extra))
    except Exception:                                              # noqa: BLE001
        pass


# ---------------------------------------------------------------- seeds (md5 paradigm)
def seeded(cls, family, seq, master=None):
    m = MASTER if master is None else master
    return int(hashlib.md5(("gen32|%d|%s|%s|%d" % (m, cls, family, seq)).encode("utf-8")
                           ).hexdigest()[:8], 16)


# ---------------------------------------------------------------- Lean judging (nice19)
def lean_proc_count():
    """Global count of our Lean jobs (lean+lake processes; same convention as bh_lean)."""
    import subprocess
    try:
        cp = subprocess.run(["ps", "-eo", "comm="], capture_output=True, text=True)
        return sum(1 for ln in cp.stdout.splitlines() if ln.strip() in ("lean", "lake"))
    except Exception:                                              # noqa: BLE001
        return 0


def run_lean(path, timeout=5400.0, nice=True, cwd=None):
    """nice -n 19 lake env lean <abs path>; cwd = the miniF2F root (same convention as drct_common)."""
    import subprocess
    path = os.path.abspath(path)
    argv = (["nice", "-n", "19"] if nice else []) + ["lake", "env", "lean", path]
    env = dict(os.environ)
    env["PATH"] = os.path.expanduser("~/.elan/bin") + os.pathsep + env.get("PATH", "")
    t0 = time.time()
    try:
        cp = subprocess.run(argv, capture_output=True, text=True, timeout=timeout,
                            cwd=cwd or MINIF2F, env=env)
        out = (cp.stdout or "") + (cp.stderr or "")
        rc = cp.returncode
    except subprocess.TimeoutExpired:
        return {"rc": 124, "out": "LEAN_TIMEOUT", "secs": round(time.time() - t0, 2)}
    except FileNotFoundError as e:
        return {"rc": 127, "out": "LEAN_BIN_MISSING: %s" % e, "secs": round(time.time() - t0, 2)}
    return {"rc": rc, "out": out, "secs": round(time.time() - t0, 2)}


# Python 3.9 compatibility
import subprocess                                                   # noqa: E402

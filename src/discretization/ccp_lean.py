# -*- coding: utf-8 -*-
# ccp_lean.py -- CCP Lean instrumentation (advance team gen12, constitution change 58)
# Self-contained (never writes scratch/drct, does not import drct_gen's write path);
# writes only under scratch/ccp/lean/**.
# Measurement convention: official miniF2F header + statement + proof body + #print axioms;
# pass requires rc==0 AND no sorryAx AND no not-run.
import json
import os
import re
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor

RES = os.environ.get("SCOPE_RESEARCH_ROOT", ".")
CCP = os.path.join(RES, "scratch", "ccp")
MINIF2F = os.environ.get("SCOPE_MINIF2F", os.path.join(RES, "lean", "miniF2F-main"))
LEAND = os.path.join(CCP, "lean")
os.makedirs(LEAND, exist_ok=True)
_ELAN_BIN = os.path.expanduser("~/.elan/bin")

OFFICIAL_HEADER = ["import MiniF2F.ProblemImports",
                   "open scoped Real Nat Topology Polynomial"]
NOT_RUN = ("does not exist", "no such file or directory", "LEAN_BIN_MISSING",
           "no default toolchain configured", "cannot execute")
TIMEOUT = 900.0
MENU = ["simp", "norm_num", "decide", "linarith", "nlinarith", "omega", "ring", "rfl"]
MY_HARD_CAP = 48
TOTAL_CAP = 80   # assumes the original 192-core host (~21% load); still yields to other tracks via lean_proc_count


def safe_path(p, root=RES):
    rp = os.path.realpath(p)
    if not (rp == root or rp.startswith(root + os.sep)):
        raise ValueError("path escapes sandbox: %r" % (p,))
    return rp


def indent_proof(proof):
    """Indent the proof body to a 2-space base (first line after ':= by' included);
    blank lines preserved."""
    lines = [ln.rstrip() for ln in str(proof).split("\n")]
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    if not lines:
        return []
    base = len(lines[0]) - len(lines[0].lstrip(" "))
    out = []
    for ln in lines:
        if not ln.strip():
            out.append("")
        else:
            out.append("  " + ln[base:])
    return out


def render(stmt, proof):
    """stmt excludes ':= by'; proof is the tactic body (multi-line)."""
    L = list(OFFICIAL_HEADER)
    L.append("")
    L.append(stmt + " := by")
    L.extend(indent_proof(proof))
    L.append("")
    L.append("#print axioms %s" % stmt.split()[1])
    return "\n".join(L) + "\n"


def run_lean_nice(path):
    argv = ["nice", "-n", "19", "lake", "env", "lean", path]
    env = dict(os.environ)
    env["PATH"] = _ELAN_BIN + os.pathsep + env.get("PATH", "")
    t0 = time.time()
    try:
        cp = subprocess.run(argv, capture_output=True, text=True, timeout=TIMEOUT,
                            cwd=MINIF2F, env=env)
        out = (cp.stdout or "") + (cp.stderr or "")
        rc = cp.returncode
    except subprocess.TimeoutExpired:
        return {"rc": 124, "ok": False, "sorry": False, "secs": round(time.time() - t0, 2),
                "head": "TIMEOUT"}
    except FileNotFoundError as e:
        return {"rc": 127, "ok": False, "sorry": False, "secs": round(time.time() - t0, 2),
                "head": "NOBIN %s" % e}
    ok = (rc == 0) and (not any(k in out for k in NOT_RUN))
    sorry = ("sorryAx" in out) or ("declaration uses 'sorry'" in out)
    ax = ""
    m = re.search(r"depends on axioms: \[([^\]]*)\]", out)
    if m:
        ax = m.group(1)
    return {"rc": rc, "ok": bool(ok), "sorry": bool(sorry),
            "secs": round(time.time() - t0, 2), "axioms": ax, "head": out[:400]}


def lean_proc_count():
    try:
        cp = subprocess.run(["ps", "-eo", "comm="], capture_output=True, text=True)
        return sum(1 for ln in cp.stdout.splitlines() if ln.strip() in ("lean", "lake"))
    except Exception:
        return 0


def avail_gb():
    try:
        with open("/proc/meminfo") as f:
            for ln in f:
                if ln.startswith("MemAvailable:"):
                    return int(ln.split()[1]) / 1048576.0
    except Exception:
        pass
    return 64.0


_busy_lock = threading.Lock()
_busy = 0


def one_job(name, stmt, proof, tag, subdir="ccp"):
    global _busy
    wd = os.path.join(LEAND, subdir)
    os.makedirs(wd, exist_ok=True)
    fname = "%s_%s.lean" % (re.sub(r"[^A-Za-z0-9_.-]", "_", name),
                            re.sub(r"[^A-Za-z0-9_.-]", "_", tag))
    fpath = safe_path(os.path.join(wd, fname))
    with open(fpath, "w", encoding="utf-8") as f:
        f.write(render(stmt, proof))
    with _busy_lock:
        _busy += 1
    try:
        r = run_lean_nice(fpath)
    finally:
        with _busy_lock:
            _busy -= 1
    r["path"] = fpath
    return r


def parallel_jobs(jobs, log_every=400, quiet=False):
    """jobs: [(name, stmt, proof, tag, subdir)] -> {(name, tag): result}. Yielding parallelism."""
    out = {}
    idx, t0 = 0, time.time()
    while idx < len(jobs):
        ext = max(0, lean_proc_count() - _busy)
        by_yield = TOTAL_CAP - ext
        by_mem = int((avail_gb() - 16) // 3)
        budget = max(2, min(MY_HARD_CAP, by_yield, by_mem))
        chunk = jobs[idx: idx + budget]
        idx += len(chunk)
        with ThreadPoolExecutor(max_workers=budget) as ex:
            futs = {ex.submit(one_job, n, s, p, g, d): (n, g) for (n, s, p, g, d) in chunk}
            for fu, key in futs.items():
                out[key] = fu.result()
        if not quiet and (idx % log_every < len(chunk)):
            print("[lean] %d/%d | ext=%d budget=%d mem=%.0fG | %.0fs" %
                  (idx, len(jobs), lean_proc_count(), budget, avail_gb(), time.time() - t0),
                  flush=True)
    return out


def floor_probe(items, subdir="ccp_floor", quiet=True):
    """items: [(name, stmt)]; try each of the eight menu tactics bare.
    Returns {name: {"closers":[...], "detail":{t:rec}}}."""
    jobs = []
    for (name, stmt) in items:
        for t in MENU:
            jobs.append((name, stmt, t, "floor_" + t, subdir))
    res = parallel_jobs(jobs, quiet=quiet)
    out = {}
    for (name, stmt) in items:
        det = {}
        for t in MENU:
            r = res.get((name, "floor_" + t)) or {}
            det[t] = {"ok": bool(r.get("ok")), "rc": r.get("rc"), "secs": r.get("secs"),
                      "head": (r.get("head") or "")[:200]}
        out[name] = {"closers": [t for t in MENU if det[t]["ok"]], "detail": det}
    return out


def chain_check(items, subdir="ccp_cert", quiet=True):
    """items: [(name, stmt, proof)]; full-chain proof check (rc==0 and no sorry)."""
    jobs = [(name, stmt, proof, "chain", subdir) for (name, stmt, proof) in items]
    res = parallel_jobs(jobs, quiet=quiet)
    out = {}
    for (name, stmt, proof) in items:
        r = res.get((name, "chain")) or {}
        out[name] = {"ok": bool(r.get("ok")), "rc": r.get("rc"), "sorry": r.get("sorry"),
                     "secs": r.get("secs"), "head": (r.get("head") or "")[:400]}
    return out


def md5f(path):
    import hashlib
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save_json(path, obj):
    with open(safe_path(path), "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1, default=str)
    return md5f(path)

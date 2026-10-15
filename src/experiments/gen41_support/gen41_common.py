# -*- coding: utf-8 -*-
# gen41_common.py -- gen41 "miniF2F domain expansion" shared module (forked from gen36_common.py; parent untouched)
#
# Discipline: write surface is **only** $R/scratch/ccp/gen41/** (+ log gen41/LAUNCH_LOG_APPEND_gen41.txt); existing
#       files are read-only (fork-and-copy / import only). Zero sudo; zero /tmp; Test243 timing seal (single T8 unseal);
#       drct_common md5 7c31b58d frozen read-only.
# GPU: **one job per GPU (constitution #104)**. Launch condition = target card has **no job of ours** + free > 6000 MiB;
#       no card => wait and write a wait record (never share a card with another job).
import collections
import hashlib
import json
import os
import sys
import time

sys.dont_write_bytecode = True

RES = os.environ.get("SCOPE_RESEARCH_ROOT", ".")
CCP = RES + "/scratch/ccp"
G41 = CCP + "/gen41"
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
LAUNCH_LOG = G41 + "/LAUNCH_LOG_APPEND_gen41.txt"      # do not write LAUNCH_LOG.md (red line)
HEARTBEAT = G41 + "/HEARTBEAT.gen41"
WAIT_LOG = G41 + "/out/GEN41_GPU_WAIT.log"

MASTER = 20260923
JUDGE_MD5_EXPECT = "7c31b58da9d7228cee31b9be581fe5dd"
DEEP_CORPUS = G32 + "/DEEP_CORPUS_V1.jsonl"
EXAM_SIDECAR = G34 + "/out/DEEP_EXAM_SIDECAR.jsonl"
CACHE_A2 = G36 + "/cache/deep_all_a2.bin"              # gen36 G-S training cache (a2 shift, read-only input)
RENDER_SIDE_G33 = G33 + "/out/DEEP_RENDER_SIDECAR.jsonl"
SLOT_SIDE_G34 = G34 + "/out/DEEP_SLOT_SIDECAR.jsonl"
VOCAB_V5 = G41 + "/cache/CCP_VOCAB_V5.json"
ALLOW_V5 = G41 + "/cache/ccp_allow_v5.bin"
CORPUS_CLEAN = G41 + "/cache/GEN41_CORPUS_CLEAN.jsonl"
CACHE_G41 = G41 + "/cache/g41_all.bin"
RENDER_SIDE_G41 = G41 + "/cache/g41_sidecar.jsonl"
SLOT_SIDE_G41 = G41 + "/cache/g41_slot_sidecar.jsonl"
CKPT_GS_PARENT = G36 + "/ckpt/deep_plan_formal_a2wt1_6000.pt"    # gen36 G-S arm (191/218)
LAT31 = "cppcuda/cache/lat_tab_v31.bin"
HEAD_V5, NVALS_V5, END_V5 = 198, 352, 154
NSLOT = 24

for _p in (G41, G36, G35, G34, G33, G32, BH, CCP, DRCT, G13, G26, G30,
           os.path.join(CCP, "gen23"), os.path.join(CCP, "gen24"),
           os.path.join(CCP, "bh", "gen"),
           os.path.join(RES, "cppcuda", "tools"),
           os.path.join(RES, "skel32"),
           os.environ.get("SCOPE_SUPEROPERATOR", os.path.join(RES, "superoperator"))):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def safe_out(p, extra=()):
    """Write-surface assertion (red line: writes only under gen41/**)."""
    rp = os.path.realpath(p)
    allowed = [os.path.realpath(G41)] + [os.path.realpath(e) for e in extra]
    if not any(rp == a or rp.startswith(a + os.sep) for a in allowed):
        raise ValueError("write outside the allowed surface: %r" % p)
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
    line = "GEN41_HEARTBEAT %s %s %s\n" % (time.strftime("%Y-%m-%dT%H:%M:%S%z"), tag, extra)
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
    """md5 ledger of this batch's **parents/inputs** (measured once at start and once at close; must match)."""
    pr = [("gen36_hybrid", G36 + "/gen36_hybrid.py"),
          ("gen36_eval", G36 + "/gen36_eval.py"),
          ("gen36_armrun", G36 + "/gen36_armrun.py"),
          ("gen36_judge", G36 + "/gen36_judge.py"),
          ("gen36_common", G36 + "/gen36_common.py"),
          ("gen36_tables", G36 + "/gen36_tables.py"),
          ("gen36_state", G36 + "/gen36_state.py"),
          ("gen36_cache_a2", CACHE_A2),
          ("gen34_slots", G34 + "/gen34_slots.py"),
          ("gen34_hybrid", G34 + "/gen34_hybrid.py"),
          ("gen34_eval", G34 + "/gen34_eval.py"),
          ("gen33_device", G33 + "/gen33_device.py"),
          ("gen33_build", G33 + "/gen33_build.py"),
          ("gen33_vocab4", G33 + "/gen33s_vocab4.py"),
          ("gen33_render_side", RENDER_SIDE_G33),
          ("gen34_slot_side", SLOT_SIDE_G34),
          ("gen32_ops", G32 + "/gen32_ops.py"),
          ("gen32_poly", G32 + "/gen32_poly.py"),
          ("gen32_corpus", G32 + "/gen32_corpus.py"),
          ("deep_corpus", DEEP_CORPUS),
          ("exam_sidecar", EXAM_SIDECAR),
          ("gen13_ccp_cachefull", G13 + "/ccp_cachefull.py"),
          ("gen13_v32ccp_train", G13 + "/v32ccp_train.py"),
          ("drct_common", DRCT + "/drct_common.py"),
          ("vocab_v5", VOCAB_V5), ("allow_v5", ALLOW_V5),
          ("corpus_clean_g41", CORPUS_CLEAN),
          ("vocab_v4", G33 + "/cache/CCP_VOCAB_V4.json"),
          ("ckpt_gs_parent", CKPT_GS_PARENT)]
    out = {}
    for k, p in pr:
        out[k] = {"path": p, "md5": md5f(p) if os.path.exists(p) else None,
                  "bytes": os.path.getsize(p) if os.path.exists(p) else None}
    return out


def judge_gate():
    import drct_common as DC
    jm = md5f(DC.__file__)
    if jm != JUDGE_MD5_EXPECT:
        raise SystemExit("FATAL judge module is not the frozen version: md5=%s expected=%s" % (jm, JUDGE_MD5_EXPECT))
    return jm


def n_lean_jobs():
    """Our Lean job count (pre-launch gate: <5)."""
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
    log("LEAN GATE ours=%d (required <%d)" % (n, limit), logp)
    if n >= limit:
        raise SystemExit("Lean concurrency gate not passed: %d >= %d" % (n, limit))
    return n


def gpu_probe():
    """(per card: index, used, total, free, our compute-process pids). Ours = venv python compute processes."""
    import subprocess
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,memory.total",
                          "--format=csv,noheader,nounits"], capture_output=True,
                         text=True).stdout.strip()
    cards = {}
    for ln in out.split("\n"):
        if not ln.strip():
            continue
        i, u, t = [x.strip() for x in ln.split(",")]
        cards[int(i)] = {"index": int(i), "used": int(u), "total": int(t),
                         "free": int(t) - int(u), "pids": []}
    apps = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory",
                           "--format=csv,noheader,nounits"], capture_output=True,
                          text=True).stdout.strip()
    uuid2idx = {}
    out2 = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
                          capture_output=True, text=True).stdout.strip()
    for ln in out2.split("\n"):
        if ln.strip():
            i, uu = [x.strip() for x in ln.split(",")]
            uuid2idx[uu] = int(i)
    for ln in apps.split("\n"):
        if not ln.strip():
            continue
        uu, pid, mem = [x.strip() for x in ln.split(",")]
        cards.setdefault(uuid2idx.get(uu, -1), {"pids": []})
        cards[uuid2idx.get(uu, -1)].setdefault("pids", []).append(
            {"pid": int(pid), "mem": int(mem)})
    # process ownership: "ours" = processes launched by the experiment host's
    # venv interpreter. Hosts that run the code outside that venv should
    # adjust this marker; the free-memory gate below applies regardless.
    ps = subprocess.run(["bash", "-lc",
                         "ps -eo pid,args --no-headers | grep -F 'venv/bin/python'"
                         " | grep -Ev 'grep'"], capture_output=True, text=True).stdout
    ours = {}
    for ln in ps.split("\n"):
        ln = ln.strip()
        if not ln:
            continue
        pid = int(ln.split(None, 1)[0])
        ours[pid] = ln
    for i, c in cards.items():
        for p in c.get("pids", []):
            p["ours"] = p["pid"] in ours
            if p["pid"] in ours:
                p["cmd"] = ours[p["pid"]][:180]
    return cards


def gpu_gate(dev, logp=None, require_free=6000, wait_log=True, note="", ours_big_mib=1000):
    """Launch gate (**constitution #104 + change #108**): target card free > require_free and **no substantive job of ours**.

    Change #108 (user order 2026-09-25): all cards usable, no card pinned; one job per GPU stays in force.
    A compute process of ours using >= ours_big_mib (default 1 GiB) counts as a "job"; smaller residual CUDA
    contexts (historical context of another batch on our card, <2% VRAM) do not -- same criterion as change #108's
    "no compute app of ours".
    Not satisfied => write a wait record and raise SystemExit (no hard run).
    """
    idx = int(str(dev).split(":")[1]) if ":" in str(dev) else 0
    cards = gpu_probe()
    c = cards.get(idx)
    if c is None:
        raise SystemExit("GPU probe failed: dev=%s cards=%s" % (dev, sorted(cards)))
    mine = [p for p in c.get("pids", []) if p.get("ours")]
    mine_big = [p for p in mine if int(p.get("mem") or 0) >= ours_big_mib]
    others = [p for p in c.get("pids", []) if not p.get("ours")]
    line = ("[%s] GPU GATE dev=%s free=%dMiB (need >%d) ours=%d (substantive jobs >=%dMiB: %d) "
            "other-side compute procs=%d %s"
            % (time.strftime("%Y-%m-%dT%H:%M:%S%z"), dev, c["free"], require_free,
               len(mine), ours_big_mib, len(mine_big), len(others), note))
    if logp:
        log(line, logp)
    if wait_log:
        try:
            with open(WAIT_LOG, "a", encoding="utf-8") as f:
                f.write(line + " | " + json.dumps(c, ensure_ascii=False) + "\n")
        except Exception:                                            # noqa: BLE001
            pass
    if mine_big:
        raise SystemExit("launch gate not passed: target card %s has a substantive job of ours %s" % (dev, mine_big))
    if c["free"] <= require_free:
        raise SystemExit("launch gate not passed: %s free=%dMiB <= %dMiB" % (dev, c["free"], require_free))
    return {"dev": dev, "free": c["free"], "used": c["used"], "total": c["total"],
            "residual_ctx_mib": [p.get("mem") for p in mine],
            "others": [p["pid"] for p in others]}


def counts_hist(xs):
    return dict(collections.Counter(xs).most_common())


def wilson_lb(k, n, z=1.959964):
    if n <= 0:
        return 0.0
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return max(0.0, (c - h) / d)

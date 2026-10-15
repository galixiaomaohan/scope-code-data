# -*- coding: utf-8 -*-
# gen44_p4_scheduler.py -- [gen44/P4] full-corpus final-test judging scheduler (T1)
#
# Purpose: judging infrastructure for the [S8] full-corpus final test of 132375 problems --
#       chunking (configurable chunk size) + resumable runs (one JSONL per chunk, reconcilable
#       chunk by chunk) + Lean concurrency gate (global own-side count measured before dispatch) +
#       long-tail monitoring (configurable per-problem timeout; timed-out problems listed
#       separately without blocking the queue) + progress heartbeat (observable by tail).
#
# Discipline (consistent with NEXT_STEP §1/§4):
#   * Judge module scratch/drct/drct_common.py (md5 7c31b58da9d7228cee31b9be581fe5dd) is
#     **imported with zero modification** -- judging semantics = the original DC.judge_attempt;
#     this file does not copy its logic.
#   * Lean nice -n 19: at startup this file re-nices itself to 19 (inherited by children),
#     verified by measurement and persisted as nice_self.
#   * Concurrency gate (two conditions, both `ps`-measured before dispatch): (1) self-limit =
#     this process's inflight < --max-lean-jobs; (2) yield = global own-side `lake env lean`
#     count <= --yield-over (0=off). If either fails it **waits** (gate_stop recorded only after
#     --gate-wait-s); no silent skip. P4 stress test: --max-lean-jobs 2 --yield-over 3
#     => own-side total on the box <= 4 < 5 (three-arm sharing discipline); main-arm exclusive
#     final test: --max-lean-jobs 5 --yield-over 0.
#   * Zero GPU; zero sudo; no /tmp; write surface only scratch/ccp/gen44/** (out/GEN44_P4_*
#     and lean/p4sched_*).
#
# ---------------------------------------------------------------- manifest format
# Input = rendered proof texts awaiting judging from the final test (JSONL, one problem per
# line; missing fields get defaults):
#   {"key": "<globally unique; defaults to name>",
#    "name": "lean_workbook_18",            # Lean theorem name (required)
#    "head": "theorem lean_workbook_18 :",  # required; same shape as LWB_ALL.jsonl, **trailing colon included**
#    "concl": "3 ^ 2 + 4 ^ 2 = 5 ^ 2",      # required; conclusion (excluding ':= by')
#    "proofs": ["norm_num1\nnorm_num"],     # proof texts to judge (one render = 1 entry; candidates tried in order until certified)
#    "proof_text": "...",                   # equivalent alias of proofs (single-candidate shorthand)
#    "cand_idx": 0, "n_cand": 1,            # optional (trace/reconciliation)
#    "ref_certified": true, "ref_state": "certified",   # optional: reference reading (stress-test reconciliation)
#    "R": "eval", "sub": "lean_workbook"}   # optional (per-route statistics)
#
# ---------------------------------------------------------------- usage
#   PY=${SCOPE_PYTHON:-python3}; G44=${SCOPE_RESEARCH_ROOT:-.}/scratch/ccp/gen44
#   # (1) dry run (zero Lean): chunk layout + pending count only
#   $PY $G44/gen44_p4_scheduler.py run --manifest M.jsonl --run-id gt_v1 --dry-run
#   # (2) full run (main-arm full volume: --max-lean-jobs per the recorded value; P4 stress test self-limits to 2)
#   nohup $PY $G44/gen44_p4_scheduler.py run --manifest M.jsonl --run-id gt_v1 \
#       --chunk-size 256 --workers 5 --max-lean-jobs 5 --timeout 240 > logs/p4_gt_v1.log 2>&1 &
#   # (3) resume = re-issue the original command (finished chunks skipped; a partial chunk only fills missing rows)
#   # (4) reconcile (zero-diff criterion: missing=0 and dup=0 and zero disagreement with ref readings)
#   $PY $G44/gen44_p4_scheduler.py reconcile --manifest M.jsonl --run-id gt_v1
#   # (5) progress snapshot / heartbeat tail
#   $PY $G44/gen44_p4_scheduler.py status --run-id gt_v1
#   tail -f $G44/out/GEN44_P4_SCHED_gt_v1/HEARTBEAT.txt
import argparse
import collections
import hashlib
import importlib.machinery as _im
import importlib.util as _iu
import json
import os
import signal
import subprocess
import sys
import threading
import time

RES = os.environ.get("SCOPE_RESEARCH_ROOT", ".")
# the gen44 tree on the experiment host (out/logs live there, not beside this
# file in the package layout)
G44 = os.path.join(RES, "scratch", "ccp", "gen44")
OUT = os.path.join(G44, "out")
LOGS = os.path.join(G44, "logs")
JUDGE_PATH = os.path.join(RES, "scratch", "drct", "drct_common.py")
JUDGE_MD5 = "7c31b58da9d7228cee31b9be581fe5dd"
# States treated as "incomplete" on resume (must be re-judged; any other state is final, skipped)
RETRY_STATES = ("gate_stop", "error", "aborted", "timeout")


def _load(name, path):
    ld = _im.SourceFileLoader(name, path)
    m = _iu.module_from_spec(_iu.spec_from_loader(name, ld))
    sys.modules[name] = m
    ld.exec_module(m)
    return m


DC = _load("drct_common_p4sched", JUDGE_PATH)
STOP = threading.Event()
LOCK = threading.Lock()
# Rows this process is currently running Lean for (one judge call per row = at most one Lean
# child) -- the real object of "--max-lean-jobs"
MY_INFLIGHT = {"n": 0}


# ================================================================ shared helpers
def now_str():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def md5s(s):
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def md5f(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def wilson(k, n, z=1.96):
    if n <= 0:
        return [0.0, 0.0]
    p = k / n
    dd = 1 + z * z / n
    c = (p + z * z / (2 * n)) / dd
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / dd
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]


def our_lean_jobs():
    """Measured before dispatch: global count of **real** own-side Lean processes.
    **P4 correction**: the inherited convention (`grep -F 'lake env lean'`) also counts
    Python/bash wrapper processes whose cmdline contains that string (measured 2-4 phantoms,
    which deadlocks the concurrency gate). This file counts exactly via the `comm` column
    (executable name = lake/lean) and only accepts own-side paths."""
    try:
        ps = subprocess.run(
            ["bash", "-lc",
             "ps -eo comm,cmd --no-headers | grep -E '^(lake|lean)[[:space:]]' "
             "| grep -E 'tcaboost|elan' | grep -Ev grep"],
            capture_output=True, text=True).stdout
    except Exception:                                                   # noqa: BLE001
        return -1
    return len([l for l in ps.split("\n") if l.strip()])


def count_lean_jobs_legacy():
    """Legacy convention (kept only for cross-checks/reports): cmdline match."""
    try:
        ps = subprocess.run(
            ["bash", "-lc",
             "ps -eo pid,cmd --no-headers | grep -F 'lake env lean' | grep -Ev grep "
             "| grep -E 'tcaboost|elan'"],
            capture_output=True, text=True).stdout
    except Exception:                                                   # noqa: BLE001
        return -1
    return len([l for l in ps.split("\n") if l.strip()])


def renice_self(target=19):
    """Re-nice self to target (inherited by children) -- the process-level implementation of
    'Lean nice -n 19'."""
    cur_txt = "?"
    try:
        cur = os.nice(0)
        cur_txt = str(cur)
        if target > cur:
            os.nice(target - cur)
        return {"nice_before": cur_txt, "nice_after": str(os.nice(0)),
                "target": target, "ok": os.nice(0) == target}
    except Exception as e:                                              # noqa: BLE001
        return {"nice_before": cur_txt, "nice_after": None, "target": target,
                "ok": False, "err": str(e)}


def run_dirs(run_id):
    d = os.path.join(OUT, "GEN44_P4_SCHED_%s" % run_id)
    return {"run": d, "chunks": d, "hb": os.path.join(d, "HEARTBEAT.txt"),
            "hbj": os.path.join(d, "HEARTBEAT.json"),
            "longtail": os.path.join(d, "LONGTAIL.jsonl"),
            "timeouts": os.path.join(d, "TIMEOUT.jsonl"),
            "status": os.path.join(d, "STATUS.json"),
            "meta": os.path.join(d, "RUN_META.json"),
            "summary": os.path.join(OUT, "GEN44_P4_SCHED_%s_SUMMARY.json" % run_id)}


# ================================================================ manifest / chunking
def load_manifest(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if not r.get("name") or not r.get("concl"):
                raise ValueError("manifest 第 %d 行缺 name/concl" % ln)
            r.setdefault("key", r["name"])
            if "proofs" not in r:
                r["proofs"] = [r["proof_text"]] if r.get("proof_text") else []
            if not isinstance(r["proofs"], list):
                r["proofs"] = [r["proofs"]]
            rows.append(r)
    return rows


def n_chunks(n_rows, chunk_size):
    return (n_rows + chunk_size - 1) // chunk_size


def chunk_row_range(ci, chunk_size, n_rows):
    a = ci * chunk_size
    return a, min(n_rows, a + chunk_size)


def read_chunk_rows(path):
    """Read back existing chunk rows; tolerate a half line left by a hard kill (only at the
    tail) -> (rows, n_bad_tail)."""
    rows, bad = [], 0
    if not os.path.exists(path):
        return rows, bad
    with open(path, encoding="utf-8") as f:
        lines = f.read().split("\n")
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except Exception:                                              # noqa: BLE001
            if i >= len(lines) - 2:
                bad += 1                                               # tail half line: tolerated
            else:
                raise ValueError("chunk %s 第 %d 行损坏 (非末尾)" % (path, i + 1))
    return rows, bad


def chunk_state(ci, chunk_size, n_rows, dirs):
    """-> dict(rows_path, done_path, n_expected, n_done, md5_ok, status)"""
    a, b = chunk_row_range(ci, chunk_size, n_rows)
    rp = os.path.join(dirs["chunks"], "chunk_%05d.jsonl" % ci)
    dp = os.path.join(dirs["chunks"], "chunk_%05d.done.json" % ci)
    rows, bad = read_chunk_rows(rp)
    st = {"ci": ci, "a": a, "b": b, "n_expected": b - a, "rows_path": rp,
          "done_path": dp, "n_written": len(rows), "n_bad_tail": bad,
          "keys": [r.get("key") for r in rows], "md5_ok": False, "done": False,
          "done_rec": None}
    if os.path.exists(dp):
        try:
            d = json.load(open(dp, encoding="utf-8"))
        except Exception:                                              # noqa: BLE001
            d = None
        st["done_rec"] = d
        if d and d.get("n_rows") == st["n_written"] and \
                d.get("rows_md5") == (md5f(rp) if os.path.exists(rp) else None):
            st["done"], st["md5_ok"] = True, True
    return st


def finalize_chunk(st, dirs, wall_s, t_start):
    rp = st["rows_path"]
    m = md5f(rp) if os.path.exists(rp) else None
    n = st["n_written"]
    rec = {"ci": st["ci"], "n_expected": st["n_expected"], "n_rows": n,
           "rows_md5": m, "status": "done", "iso": now_str(),
           "a": st["a"], "b": st["b"], "wall_s": round(wall_s, 2),
           "chunk_started_iso": t_start}
    with open(st["done_path"], "w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=1)
    return rec


# ================================================================ single-row judging
def judge_one(row, ctx):
    """Judge one row; returns the result entry (gen44_accept_fix entry fields + scheduler fields)."""
    t0 = time.time()
    ent = {"key": row.get("key"), "name": row["name"], "R": row.get("R"),
           "sub": row.get("sub"), "idx": row.get("idx"),
           "ref_certified": row.get("ref_certified"),
           "ref_state": row.get("ref_state"),
           "gate_wait_s": 0.0, "n_cand": len(row["proofs"]), "cand_idx": None,
           "certified": False, "engine": "drct_common.judge_attempt",
           "ts": now_str()}
    texts = [t for t in row["proofs"] if t and t.strip()]
    if not texts:
        ent["state"] = "empty_proof"
        ent["reasons"] = ["empty_proof"]
        ent["wall_s"] = round(time.time() - t0, 2)
        return ent
    for ci, text in enumerate(texts):
        # ---- concurrency gate (two conditions, both against the global own-side
        # `lake env lean` count measured **before dispatch**):
        #   (1) self-limit: this process's inflight < --max-lean-jobs (P4 stress test 2)
        #   (2) yield: global count <= --yield-over (0=off; includes own side; P4 stress test 3
        #       => box total <=4 <5)
        # While gated it waits (capped by --gate-wait-s; gate_stop recorded on timeout).
        waited = 0.0
        while True:
            if STOP.is_set():
                ent["state"], ent["reasons"] = "aborted", ["aborted_before_dispatch"]
                ent["wall_s"] = round(time.time() - t0, 2)
                return ent
            j = our_lean_jobs()
            with LOCK:
                mine = MY_INFLIGHT["n"]
            if mine < ctx["max_lean"] and (ctx["yield_over"] <= 0 or j < 0
                                           or j <= ctx["yield_over"]):
                break
            if waited >= ctx["gate_wait_s"]:
                ent["state"] = "gate_stop"
                ent["reasons"] = ["gate_stop_jobs=%s,mine=%s" % (j, mine)]
                ent["gate_lean_jobs"] = j
                ent["wall_s"] = round(time.time() - t0, 2)
                ent["gate_wait_s"] = round(waited, 1)
                return ent
            time.sleep(2.0)
            waited += 2.0
        ent["gate_wait_s"] = round(ent["gate_wait_s"] + waited, 1)
        tg = time.time()
        with LOCK:
            MY_INFLIGHT["n"] += 1
        try:
            v = DC.judge_attempt(row["name"], row["head"], row["concl"], text,
                                 ctx["lean_dir"], ctx["tag"], ctx["timeout"])
        finally:
            with LOCK:
                MY_INFLIGHT["n"] -= 1
        ent["judge_s"] = ent.get("judge_s", 0.0) + round(time.time() - tg, 2)
        ent["cand_idx"] = ci
        ent["proof_md5"] = md5s(text)
        ent["certified"] = bool(v.get("certified"))
        ent["reasons"] = v.get("reasons")
        ent["lean"] = v.get("lean")
        ent["n1n2"] = v.get("n1n2")
        tmo = bool((v.get("lean") or {}).get("rc") == 124)
        if v.get("n1n2"):
            tmo = tmo or (max(v["n1n2"].get("n1_secs") or 0.0,
                              v["n1n2"].get("n2_secs") or 0.0) >= ctx["timeout"] * 0.99)
        ent["timeout_hit"] = tmo
        ent["state"] = ("certified" if ent["certified"] else
                        ("timeout" if tmo else "judged_reject"))
        if ent["certified"]:
            break
    ent["wall_s"] = round(time.time() - t0, 2)
    return ent


def _dispatch(fn, ctx):
    """Unified concurrency-gate + inflight-accounting wrapper: returns ("ok", result, wait_s) /
    ("gate_stop", (j, mine, wait_s), wait_s) / ("aborted", None, wait_s)."""
    waited = 0.0
    while True:
        if STOP.is_set():
            return ("aborted", None, waited)
        j = our_lean_jobs()
        with LOCK:
            mine = MY_INFLIGHT["n"]
        if mine < ctx["max_lean"] and (ctx["yield_over"] <= 0 or j < 0
                                       or j <= ctx["yield_over"]):
            break
        if waited >= ctx["gate_wait_s"]:
            return ("gate_stop", (j, mine, waited), waited)
        time.sleep(2.0)
        waited += 2.0
    with LOCK:
        MY_INFLIGHT["n"] += 1
    try:
        return ("ok", fn(), waited)
    finally:
        with LOCK:
            MY_INFLIGHT["n"] -= 1


def judge_batch(grp, ctx):
    """**Batch main judging** (--batch-size K>1): K rows rendered into one Lean file, so a single
    `lake env lean` call main-judges K rows.
    Semantics preserved: batch rc==0 and no sorry => each row alone would also be accepted
    (same header, same statement) => equivalent; a failed batch => **per-row fallback to
    judge_one** (fail-closed, judged exactly as in single-row mode).
    Certification still needs the N1+N2 triplet (per row, frozen module DC.certify) -- batching
    only absorbs the bulk of the fixed overhead (~9.7s per call)."""
    rows = [r for _i, r in grp]
    names = [r["name"] for r in rows]
    if (len(set(names)) != len(names)
            or any(not (r["proofs"] and r["proofs"][0].strip()) for r in rows)):
        return [judge_one(r, dict(ctx, tag=ctx["tag"] + "_f%d" % i))
                for i, r in enumerate(rows)]
    L = list(DC.OFFICIAL_HEADER) + [""]
    for r in rows:
        body = DC._normalize_indent(r["proofs"][0])
        L.append("%s %s := by" % (r["head"], r["concl"]))
        if body:
            L.append(body)
        L.append("#print axioms %s" % r["name"])
        L.append("")
    f = os.path.join(ctx["lean_dir"], "batch_%s_%d.lean" % (ctx["tag"], grp[0][0]))
    with open(f, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    st, res, waited = _dispatch(lambda: DC.run_lean(f, ctx["timeout"]), ctx)
    out = []
    if st == "ok" and res.get("accepted"):
        for i, (_ri, row) in enumerate(grp):
            ent = {"key": row.get("key"), "name": row["name"], "R": row.get("R"),
                   "sub": row.get("sub"), "idx": row.get("idx"),
                   "ref_certified": row.get("ref_certified"),
                   "ref_state": row.get("ref_state"),
                   "gate_wait_s": round(waited if i == 0 else 0.0, 1),
                   "n_cand": len(row["proofs"]), "cand_idx": 0,
                   "engine": "drct_common.certify (主判=批判)", "batch_hit": True,
                   "batch_n": len(rows), "proof_md5": md5s(row["proofs"][0]),
                   "lean": {k: res[k] for k in ("rc", "accepted", "sorry", "secs")},
                   "certified": False, "reasons": [], "n1n2": None,
                   "ts": now_str()}
            tg0 = time.time()
            st2, res2, w2 = _dispatch(
                lambda row=row: DC.certify(row["name"], row["head"], row["concl"],
                                           row["proofs"][0], ctx["lean_dir"],
                                           ctx["tag"] + "_b", ctx["timeout"]), ctx)
            if st2 != "ok":
                ent["state"] = "gate_stop" if st2 == "gate_stop" else "aborted"
                ent["reasons"] = ["batch_certify_%s" % st2]
                ent["wall_s"] = round(time.time() - tg0, 2)
                out.append(ent)
                continue
            ok, reasons, detail = res2
            ent["certified"] = bool(ok)
            ent["reasons"] = reasons
            ent["n1n2"] = {"n1_accepted": detail["n1"]["accepted"],
                           "n2_accepted": detail["n2"]["accepted"],
                           "n1_secs": detail["n1"]["secs"], "n2_secs": detail["n2"]["secs"],
                           "n1_vacuous": detail.get("n1_vacuous"),
                           "n1_note": detail.get("n1_note")}
            tmo = max(detail["n1"]["secs"], detail["n2"]["secs"]) >= ctx["timeout"] * 0.99
            ent["timeout_hit"] = bool(tmo)
            ent["state"] = ("certified" if ent["certified"]
                            else ("timeout" if tmo else "judged_reject"))
            ent["wall_s"] = round(time.time() - tg0 + (waited / max(1, len(rows))), 2)
            out.append(ent)
        return out
    # ---- batch judging did not pass: per-row fallback (covering gate_stop / aborted too)
    if st == "gate_stop":
        return [judge_one(r, dict(ctx, tag=ctx["tag"] + "_r%d" % i))
                for i, r in enumerate(rows)]
    return [judge_one(r, dict(ctx, tag=ctx["tag"] + "_r%d" % i))
            for i, r in enumerate(rows)]


# ================================================================ heartbeat
class Heartbeat(object):
    def __init__(self, dirs, ctx, init_done=0, init_cert=0, init_tmo=0, init_lt=0,
                 init_wall=0.0):
        self.d = dirs
        self.ctx = ctx
        self.lock = threading.Lock()
        self.t0 = time.time()
        self.n_done = init_done
        self.init_done = init_done
        self.n_cert = init_cert
        self.n_tmo = init_tmo
        self.n_gate = 0
        self.n_lt = init_lt
        self.sum_judge = 0.0
        self.sum_wall = init_wall
        self.init_wall = init_wall
        self.cur = {}
        self.n_total = ctx["n_pending"]
        self.last_hb = 0.0

    def start_row(self, wid, key, ci, ri):
        with self.lock:
            self.cur[wid] = {"key": key, "ci": ci, "ri": ri, "t0": time.time()}

    def end_row(self, wid, ent):
        with self.lock:
            self.cur.pop(wid, None)
            self.n_done += 1
            self.n_cert += 1 if ent.get("certified") else 0
            self.n_tmo += 1 if ent.get("timeout_hit") else 0
            self.n_gate += 1 if ent.get("state") == "gate_stop" else 0
            self.sum_wall += ent.get("wall_s") or 0.0
            self.sum_judge += ent.get("judge_s") or 0.0
            if (ent.get("wall_s") or 0.0) >= self.ctx["longtail_s"]:
                self.n_lt += 1

    def line(self, force=False):
        now = time.time()
        if not force and now - self.last_hb < 30.0:
            return
        self.last_hb = now
        el = now - self.t0
        rate = ((self.n_done - self.init_done) / el * 60.0) if el > 0 else 0.0
        eta = (self.n_total - self.n_done) / rate if rate > 0 else -1.0
        with self.lock:
            cur = " ".join("w%s:%s(%.0fs)" % (k, v["key"], now - v["t0"])
                           for k, v in sorted(self.cur.items()))
            done, cert, tmo, gate, lt = (self.n_done, self.n_cert, self.n_tmo,
                                         self.n_gate, self.n_lt)
            sj, sw = self.sum_judge, self.sum_wall
        conc = ((sw - self.init_wall) / el) if el > 0 else 0.0
        s = ("[%s] t=%.0fs rows=%d/%d cert=%d tmo=%d gate=%d lt=%d conc=%.2f/%d "
             "rate=%.2f/min eta=%s inflight=[%s]\n"
             % (now_str(), el, done, self.n_total, cert, tmo, gate, lt, conc,
                self.ctx["workers"],
                rate, ("%.1fmin" % eta) if eta >= 0 else "-", cur))
        with open(self.d["hb"], "a", encoding="utf-8") as f:
            f.write(s)
            f.flush()
        snap = {"iso": now_str(), "elapsed_s": round(el, 1), "n_total": self.n_total,
                "n_done": done, "n_certified": cert, "n_timeout": tmo,
                "n_gate_stop": gate, "n_longtail": lt,
                "rate_rows_per_min": round(rate, 3), "eta_min": (round(eta, 1)
                                                                 if eta >= 0 else None),
                "concurrency_achieved": round(conc, 3), "workers": self.ctx["workers"],
                "sum_judge_s": round(sj, 1), "sum_row_wall_s": round(sw, 1),
                "inflight": sorted(self.cur.keys())}
        with open(self.d["hbj"], "w", encoding="utf-8") as f:
            json.dump(snap, f, ensure_ascii=False, indent=1)
        return s


# ================================================================ main run
def cmd_run(a):
    rows = load_manifest(a.manifest)
    n_all = len(rows)
    if a.limit:
        rows = rows[:a.limit]
    dirs = run_dirs(a.run_id)
    os.makedirs(dirs["chunks"], exist_ok=True)
    man_md5 = md5f(a.manifest)
    judge_md5 = md5f(JUDGE_PATH)
    if judge_md5 != JUDGE_MD5:
        print("!! judge md5 mismatch: %s != %s" % (judge_md5, JUDGE_MD5), flush=True)
        return 3
    if os.path.exists(dirs["meta"]) and not a.dry_run:
        m0 = json.load(open(dirs["meta"], encoding="utf-8"))
        if m0.get("manifest_md5") != man_md5 and not a.allow_manifest_change:
            print("!! manifest md5 differs from existing RUN_META (%s != %s); "
                  "pass --allow-manifest-change to switch manifests"
                  % (man_md5, m0.get("manifest_md5")), flush=True)
            return 4
    lean_dir = os.path.join(G44, "lean", "p4sched_%s" % a.run_id)
    os.makedirs(lean_dir, exist_ok=True)
    nch = n_chunks(len(rows), a.chunk_size)
    states = [chunk_state(ci, a.chunk_size, len(rows), dirs) for ci in range(nch)]
    n_done_rows = sum(min(s["n_written"], s["n_expected"]) for s in states if s["done"])
    n_partial_rows = sum(s["n_written"] for s in states if not s["done"] and s["n_written"])
    n_pending = (len(rows) - n_done_rows - n_partial_rows +
                 sum(s["n_bad_tail"] for s in states if not s["done"]))
    ctx = {"max_lean": a.max_lean_jobs, "yield_over": a.yield_over,
           "gate_wait_s": a.gate_wait_s, "timeout": a.timeout, "workers": a.workers,
           "longtail_s": a.longtail_s, "lean_dir": lean_dir, "n_pending": n_pending,
           "batch": max(1, a.batch_size)}
    nice = renice_self(a.nice) if a.nice > 0 else {"ok": None, "target": a.nice}
    if a.dry_run:
        print(json.dumps({"dry_run": True, "manifest": a.manifest, "n_rows": len(rows),
                          "chunk_size": a.chunk_size, "n_chunks": nch,
                          "n_chunk_done": sum(1 for s in states if s["done"]),
                          "n_chunk_partial": sum(1 for s in states
                                                 if not s["done"] and s["n_written"]),
                          "n_rows_done": n_done_rows, "n_rows_partial": n_partial_rows,
                          "n_rows_pending": n_pending, "manifest_md5": man_md5,
                          "judge_md5": judge_md5, "nice": nice,
                          "global_lean_jobs_now": our_lean_jobs(),
                          "workers": a.workers, "max_lean_jobs": a.max_lean_jobs},
                         ensure_ascii=False, indent=1), flush=True)
        return 0
    meta = {"schema": "gen44-p4-sched/1", "run_id": a.run_id, "manifest": a.manifest,
            "manifest_md5": man_md5, "n_rows": len(rows), "n_rows_manifest": n_all,
            "limit": a.limit, "chunk_size": a.chunk_size, "n_chunks": nch,
            "workers": a.workers, "max_lean_jobs": a.max_lean_jobs,
            "yield_over": a.yield_over, "timeout": a.timeout,
            "gate_wait_s": a.gate_wait_s, "longtail_s": a.longtail_s,
            "judge_md5": judge_md5, "judge_md5_expected": JUDGE_MD5,
            "nice": nice, "lean_dir": lean_dir,
            "created_iso": now_str(), "pid": os.getpid()}
    if not os.path.exists(dirs["meta"]):
        with open(dirs["meta"], "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=1)
    with open(dirs["hb"], "a", encoding="utf-8") as f:
        f.write("[%s] RUN start run_id=%s rows=%d chunks=%d(size=%d) workers=%d "
                "max_lean=%d yield_over=%d timeout=%.0fs nice=%s judge_md5=%s\n"
                % (now_str(), a.run_id, len(rows), nch, a.chunk_size, a.workers,
                   a.max_lean_jobs, a.yield_over, a.timeout, nice.get("nice_after"),
                   judge_md5))
    print(json.dumps(meta, ensure_ascii=False, indent=1)[:2000], flush=True)
    if nice.get("ok") is False and a.nice > 0:
        print("!! renice failed: %s" % nice, flush=True)

    def _sig(signum, _fr):
        STOP.set()
        with open(dirs["hb"], "a", encoding="utf-8") as f:
            f.write("[%s] SIGNAL %s -> graceful stop (exits once the current row finishes)\n"
                    % (now_str(), signum))
    for s in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(s, _sig)
        except Exception:                                              # noqa: BLE001
            pass

    pre_rows = []
    for ci in range(nch):
        rr, _b = read_chunk_rows(os.path.join(dirs["chunks"], "chunk_%05d.jsonl" % ci))
        pre_rows += rr
    init = {"n": len(pre_rows), "cert": sum(1 for r in pre_rows if r.get("certified")),
            "tmo": sum(1 for r in pre_rows if r.get("timeout_hit")),
            "lt": sum(1 for r in pre_rows if (r.get("wall_s") or 0) >= ctx["longtail_s"]),
            "wall": sum(r.get("wall_s") or 0 for r in pre_rows)}
    hb = Heartbeat(dirs, ctx, init_done=init["n"], init_cert=init["cert"],
                   init_tmo=init["tmo"], init_lt=init["lt"], init_wall=init["wall"])
    t0 = time.time()
    summary = collections.Counter(r.get("state") or "?" for r in pre_rows)
    chunk_recs = []
    for st in states:
        if STOP.is_set():
            break
        if st["done"]:
            continue
        rows_c = rows[st["a"]:st["b"]]
        exist, _bad = read_chunk_rows(st["rows_path"])
        # Unresolved states (gate_stop/error/aborted/timeout) do not count as done: re-judge
        # (rewrite the chunk without the old rows first, to avoid duplicate keys)
        redo = [r for r in exist if (r.get("state") in RETRY_STATES)]
        if redo:
            keep = [r for r in exist if r.get("state") not in RETRY_STATES]
            with open(st["rows_path"], "w", encoding="utf-8") as fw:
                for r in keep:
                    fw.write(json.dumps(r, ensure_ascii=False) + "\n")
            with open(dirs["hb"], "a", encoding="utf-8") as f:
                f.write("[%s] CHUNK %d: re-judging %d unresolved rows (states=%s)\n"
                        % (now_str(), st["ci"],
                           len(redo), sorted(set(r.get("state") for r in redo))))
            exist = keep
        done_keys = {r.get("key") for r in exist}
        todo = [(st["a"] + i, r) for i, r in enumerate(rows_c)
                if r.get("key") not in done_keys]
        q = collections.deque(todo)
        wlc = threading.Lock()
        n_written = [len(exist)]
        fh = open(st["rows_path"], "a", encoding="utf-8")
        t_c0 = now_str()
        t_chunk0 = time.time()

        def worker(wid):
            while True:
                with wlc:
                    if not q or STOP.is_set():
                        return
                    grp = []
                    while q and len(grp) < ctx["batch"]:
                        grp.append(q.popleft())
                ri0, row0 = grp[0]
                hb.start_row(wid, row0.get("key"), st["ci"], ri0)
                t_row = time.time()
                ctxw = dict(ctx, tag="p4s%d_w%d" % (st["ci"], wid))
                try:
                    if len(grp) == 1 or ctx["batch"] <= 1:
                        ents = [judge_one(row0, ctxw)]
                    else:
                        ents = judge_batch(grp, ctxw)
                except Exception as e:                                  # noqa: BLE001
                    ents = [{"key": r.get("key"), "name": r.get("name"), "state": "error",
                             "certified": False,
                             "reasons": ["scheduler_exc:%s" % type(e).__name__],
                             "err": str(e)[:400],
                             "wall_s": round(time.time() - t_row, 2)}
                            for _i, r in grp]
                for (ri, row), ent in zip(grp, ents):
                    ent["chunk"] = st["ci"]
                    ent["row_i"] = ri
                    ent["worker"] = wid
                    line = json.dumps(ent, ensure_ascii=False) + "\n"
                    with wlc:
                        fh.write(line)
                        fh.flush()
                        n_written[0] += 1
                    hb.end_row(wid, ent)
                    hb.line()
                    if (ent.get("wall_s") or 0.0) >= ctx["longtail_s"]:
                        with open(dirs["longtail"], "a", encoding="utf-8") as f:
                            f.write(json.dumps({"key": ent.get("key"),
                                                "name": ent.get("name"),
                                                "wall_s": ent["wall_s"],
                                                "judge_s": ent.get("judge_s"),
                                                "lean_secs": (ent.get("lean") or {}).get("secs"),
                                                "state": ent.get("state"),
                                                "ts": ent["ts"]}, ensure_ascii=False) + "\n")
                    if ent.get("timeout_hit") or ent.get("state") == "timeout":
                        with open(dirs["timeouts"], "a", encoding="utf-8") as f:
                            f.write(json.dumps({"key": ent.get("key"),
                                                "name": ent.get("name"),
                                                "state": ent.get("state"),
                                                "lean": ent.get("lean"),
                                                "n1n2": ent.get("n1n2"),
                                                "wall_s": ent.get("wall_s"),
                                                "ts": ent["ts"]}, ensure_ascii=False) + "\n")
                    summary[ent.get("state") or "? "] += 1

        def ticker():
            """Periodic heartbeat (every 30s): inflight rows stay visible to a main-arm tail even
            when a row is long outstanding (gate wait / long tail)."""
            while not STOP.is_set():
                hb.line(force=True)
                time.sleep(30.0)

        ths = [threading.Thread(target=worker, args=(w,), daemon=True)
               for w in range(a.workers)]
        tk = threading.Thread(target=ticker, daemon=True)
        tk.start()
        for t in ths:
            t.start()
        for t in ths:
            t.join()
        fh.close()
        st["n_written"] = n_written[0]
        if not STOP.is_set() and st["n_written"] >= st["n_expected"]:
            rec = finalize_chunk(st, dirs, time.time() - t_chunk0, t_c0)
            st["done"], st["done_rec"], st["n_written"] = True, rec, rec["n_rows"]
            chunk_recs.append(rec)
            with open(dirs["hb"], "a", encoding="utf-8") as f:
                f.write("[%s] CHUNK %d/%d done rows=%d wall=%.1fs\n"
                        % (now_str(), st["ci"] + 1, nch, rec["n_rows"], rec["wall_s"]))
        else:
            with open(dirs["hb"], "a", encoding="utf-8") as f:
                f.write("[%s] CHUNK %d/%d INTERRUPTED rows=%d/%d (not marked done; resumable)\n"
                        % (now_str(), st["ci"] + 1, nch, st["n_written"], st["n_expected"]))
            break
    hb.line(force=True)
    wall = time.time() - t0
    # ---- summary (covers rows added by this run + pre-existing rows, re-read in full from
    # disk so it matches reconcile)
    all_rows, per_chunk = [], []
    for ci in range(nch):
        rr, bad = read_chunk_rows(os.path.join(dirs["chunks"], "chunk_%05d.jsonl" % ci))
        per_chunk.append({"ci": ci, "n": len(rr), "bad_tail": bad})
        all_rows += rr
    judged = [r for r in all_rows if r.get("state") in ("certified", "judged_reject")]
    ncert = sum(1 for r in judged if r.get("certified"))
    ref = [r for r in all_rows if r.get("ref_certified") is not None]
    rep = {"schema": "gen44-p4-sched-summary/1", "run_id": a.run_id,
           "iso": now_str(), "wall_s": round(wall, 1),
           "n_rows_manifest": n_all, "n_rows_target": len(rows),
           "n_chunks": nch, "n_chunks_done": sum(1 for s in states if s["done"]),
           "chunk_size": a.chunk_size, "workers": a.workers,
           "max_lean_jobs": a.max_lean_jobs, "yield_over": a.yield_over,
           "timeout_s": a.timeout, "longtail_s": a.longtail_s, "nice": nice,
           "batch_size": max(1, a.batch_size),
           "stopped": bool(STOP.is_set()),
           "n_rows_this_run": hb.n_done - init["n"],
           "n_rows_written": len(all_rows), "n_judged": len(judged),
           "n_certified": ncert,
           "rate_certified": round(ncert / max(1, len(judged)), 4),
           "wilson95": wilson(ncert, len(judged)),
           "state_hist": dict(summary),
           "n_timeout": sum(1 for r in all_rows if r.get("timeout_hit")),
           "n_gate_stop": sum(1 for r in all_rows if r.get("state") == "gate_stop"),
           "n_longtail": sum(1 for r in all_rows
                             if (r.get("wall_s") or 0) >= a.longtail_s),
           "rows_per_min": round((hb.n_done - init["n"]) / max(1.0, wall) * 60.0, 3),
           "sum_row_wall_s": round(sum(r.get("wall_s") or 0 for r in all_rows), 1),
           "sum_judge_s": round(sum(r.get("judge_s") or 0 for r in all_rows), 1),
           "sum_gate_wait_s": round(sum(r.get("gate_wait_s") or 0 for r in all_rows), 1),
           "concurrency_achieved": round((hb.sum_wall - init["wall"])
                                         / max(1e-9, wall) / a.workers, 4),
           "by_route": {k: {"n": sum(1 for x in judged if (x.get("R") or "?") == k),
                            "cert": sum(1 for x in judged if (x.get("R") or "?") == k
                                        and x.get("certified"))}
                        for k in sorted(set(x.get("R") or "?" for x in judged))},
           "by_subset": {k: {"n": sum(1 for x in judged if (x.get("sub") or "?") == k),
                             "cert": sum(1 for x in judged if (x.get("sub") or "?") == k
                                         and x.get("certified"))}
                         for k in sorted(set(x.get("sub") or "?" for x in judged))},
           "ref_cmp": {"n_ref": len(ref),
                       "agree": sum(1 for r in ref
                                    if bool(r.get("certified")) == bool(r.get("ref_certified"))),
                       "disagree_keys": [r.get("key") for r in ref
                                         if bool(r.get("certified"))
                                         != bool(r.get("ref_certified"))][:50]},
           "per_chunk": per_chunk, "chunks_finalized_this_run": chunk_recs,
           "judge_md5": judge_md5, "judge_md5_expected": JUDGE_MD5,
           "manifest_md5": man_md5, "pid": os.getpid()}
    with open(dirs["summary"], "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=1)
    with open(dirs["status"], "w", encoding="utf-8") as f:
        json.dump({"iso": now_str(), "stopped": bool(STOP.is_set()),
                   "n_chunks_done": rep["n_chunks_done"], "n_chunks": nch,
                   "n_rows_written": len(all_rows), "n_rows_target": len(rows),
                   "n_certified": ncert, "n_judged": len(judged),
                   "rows_per_min": rep["rows_per_min"],
                   "concurrency_achieved": rep["concurrency_achieved"],
                   "ref_agree": rep["ref_cmp"]["agree"], "ref_n": rep["ref_cmp"]["n_ref"]},
                  f, ensure_ascii=False, indent=1)
    print(json.dumps({k: rep[k] for k in ("run_id", "wall_s", "n_rows_written",
                                          "n_judged", "n_certified", "rate_certified",
                                          "rows_per_min", "concurrency_achieved",
                                          "state_hist", "n_timeout", "n_gate_stop",
                                          "n_longtail", "ref_cmp", "stopped",
                                          "n_chunks_done")},
                     ensure_ascii=False, indent=1), flush=True)
    return 0 if not STOP.is_set() else 130


# ================================================================ reconcile / status
def cmd_reconcile(a):
    rows = load_manifest(a.manifest)
    if a.limit:
        rows = rows[:a.limit]
    dirs = run_dirs(a.run_id)
    nch = n_chunks(len(rows), a.chunk_size)
    man_keys = [r.get("key") for r in rows]
    ref_of = {r.get("key"): r.get("ref_certified") for r in rows}
    seen, all_rows, chunks = collections.Counter(), [], []
    for ci in range(nch):
        st = chunk_state(ci, a.chunk_size, len(rows), dirs)
        rr, bad = read_chunk_rows(st["rows_path"])
        chunks.append({"ci": ci, "n_expected": st["n_expected"], "n_written": st["n_written"],
                       "bad_tail": bad, "done": st["done"], "md5_ok": st["md5_ok"]})
        all_rows += rr
        seen.update(r.get("key") for r in rr)
    judged = [r for r in all_rows if r.get("state") in ("certified", "judged_reject")]
    ncert = sum(1 for r in judged if r.get("certified"))
    dup = [k for k, v in seen.items() if v > 1]
    missing = [k for k in man_keys if seen.get(k, 0) == 0]
    extra = [k for k in seen if k not in set(man_keys)]
    ref = [r for r in all_rows if ref_of.get(r.get("key")) is not None]
    dis = [r.get("key") for r in ref
           if bool(r.get("certified")) != bool(ref_of[r.get("key")])]
    zero_diff = (not missing) and (not dup) and (not extra) and (not dis)
    rep = {"schema": "gen44-p4-reconcile/1", "run_id": a.run_id, "iso": now_str(),
           "manifest": a.manifest, "manifest_md5": md5f(a.manifest),
           "n_manifest": len(rows), "n_chunks": nch, "chunk_size": a.chunk_size,
           "n_chunks_done": sum(1 for c in chunks if c["done"]),
           "n_rows_written": len(all_rows), "n_unique_keys": len(seen),
           "n_duplicates": len(dup), "duplicate_keys": dup[:50],
           "n_missing": len(missing), "missing_keys": missing[:50],
           "n_extra": len(extra), "extra_keys": extra[:50],
           "n_bad_tail": sum(c["bad_tail"] for c in chunks),
           "n_judged": len(judged), "n_certified": ncert,
           "rate_certified": round(ncert / max(1, len(judged)), 4),
           "wilson95": wilson(ncert, len(judged)),
           "state_hist": dict(collections.Counter(r.get("state") for r in all_rows)),
           "ref_cmp": {"n_ref": len(ref), "agree": len(ref) - len(dis),
                       "n_disagree": len(dis), "disagree_keys": dis[:50]},
           "by_route": {k: {"n": sum(1 for x in judged if (x.get("R") or "?") == k),
                            "cert": sum(1 for x in judged if (x.get("R") or "?") == k
                                        and x.get("certified"))}
                        for k in sorted(set(x.get("R") or "?" for x in judged))},
           "chunks": chunks, "zero_diff": zero_diff}
    p = os.path.join(OUT, "GEN44_P4_RECONCILE_%s.json" % a.run_id)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=1)
    print(json.dumps({k: rep[k] for k in ("run_id", "n_manifest", "n_rows_written",
                                          "n_unique_keys", "n_missing", "n_duplicates",
                                          "n_extra", "n_bad_tail", "n_certified",
                                          "n_judged", "state_hist", "ref_cmp",
                                          "zero_diff")},
                     ensure_ascii=False, indent=1), flush=True)
    print("reconcile -> %s" % p, flush=True)
    return 0 if zero_diff else 1


def cmd_status(a):
    dirs = run_dirs(a.run_id)
    out = {"run_id": a.run_id}
    for k in ("meta", "status", "hbj"):
        if os.path.exists(dirs[k]):
            out[os.path.basename(dirs[k])] = json.load(open(dirs[k], encoding="utf-8"))
    if os.path.exists(dirs["hb"]):
        with open(dirs["hb"], encoding="utf-8") as f:
            out["heartbeat_tail"] = f.read().split("\n")[-6:]
    print(json.dumps(out, ensure_ascii=False, indent=1), flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "reconcile", "status"):
        p = sub.add_parser(name)
        p.add_argument("--manifest", default="")
        p.add_argument("--run-id", required=True)
        p.add_argument("--chunk-size", type=int, default=256)
        p.add_argument("--limit", type=int, default=0)
        if name == "run":
            p.add_argument("--workers", type=int, default=0)
            p.add_argument("--max-lean-jobs", type=int, default=5,
                           help="self-limit on this process's inflight Lean jobs "
                                "(P4 stress test 2; main-arm final test per the recorded value)")
            p.add_argument("--yield-over", type=int, default=0,
                           help="yield line: wait unconditionally while the global own-side lean "
                                "count > this value (0=off; 3 for shared-box stress tests)")
            p.add_argument("--timeout", type=float, default=240.0)
            p.add_argument("--gate-wait-s", type=float, default=900.0)
            p.add_argument("--longtail-s", type=float, default=90.0)
            p.add_argument("--batch-size", type=int, default=1,
                           help="K>1: render K rows into the **same Lean file** and main-judge "
                                "them in one call (absorbs the ~9.7s per-call fixed Mathlib load "
                                "overhead); a failed batch falls back per row, semantics unchanged")
            p.add_argument("--nice", type=int, default=19)
            p.add_argument("--dry-run", action="store_true")
            p.add_argument("--allow-manifest-change", action="store_true")
    a = ap.parse_args()
    if a.cmd == "status":
        return cmd_status(a)
    if not a.manifest:
        print("!! --manifest required", flush=True)
        return 2
    if a.cmd == "reconcile":
        return cmd_reconcile(a)
    if not a.workers:
        a.workers = max(1, a.max_lean_jobs)
    return cmd_run(a)


if __name__ == "__main__":
    sys.exit(main())

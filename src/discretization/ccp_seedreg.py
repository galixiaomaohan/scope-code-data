# -*- coding: utf-8 -*-
# ccp_seedreg.py -- generator seed and template-version registry (change 58 item 3: falsifiability)
# Anyone can use this table to recompute this batch of course problems (seed formula + template
# file md5s + commands), so no verbal account of the batch is needed.
#
# gen15 revision (disposition of CCP_GEN12_AUDIT.md F-02; registry level, course entities untouched):
#   1) rerunning this script refreshes template_files_md5 -- the gen12 registered version had 3/7
#      stale entries (ccp_families / ccp_course / ccp_gate were modified after the 10:45
#      registration) and families.nt still held the control group (pre-fix snapshot); the current
#      on-disk version is the one that reproduces the final course.
#   2) adds generation_versions: per class the **generation time** (artifact mtime, local timezone)
#      + **code hashes at that time** (installed version, backed by the full-regeneration
#      cross-check) + artifact md5 (cross-checked against the value self-loaded from
#      gen/CCP_COURSE_<CLS>.json).
#   3) adds --recompute-check (executable): **read-only** full-regeneration cross-check (seed
#      formula line by line + per-problem name sequence + row/chain byte-for-byte), writing
#      scratch/ccp/CCP_RECOMPUTE_CHECK.json and echoing the summary back into this table (evidence).
#      **Read-only guarantee**: only calls ccp_course's generation functions to rebuild in memory,
#      never run_class => writes neither gen/ nor cache/.
import argparse
import collections
import hashlib
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ccp_families as F
import ccp_lean as L

RES = os.environ.get("SCOPE_RESEARCH_ROOT", ".")
CCP = os.path.join(RES, "scratch", "ccp")
GEN = os.path.join(CCP, "gen")
MASTER = 20260923
CLASSES = ("ind", "ineq", "nt")
ROWS_TARGET = 3000
TEMPLATES = ["ccp_families.py", "ccp_ops.py", "ccp_check.py", "ccp_bridge.py",
             "ccp_course.py", "ccp_gate.py", "ccp_lean.py"]
RISK = {"ind": ["simp", "omega", "decide"],
        "ineq": ["nlinarith", "linarith", "simp", "norm_num"],
        "nt": ["omega", "simp", "decide"]}
SELF = "ccp_seedreg.py"


def _md5s(files):
    return {f: L.md5f(os.path.join(CCP, f)) for f in files
            if os.path.exists(os.path.join(CCP, f))}


def _iso(path):
    return time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(os.path.getmtime(path)))


def generation_versions(tmpl):
    """Per class: generation time + code hashes at that time + artifact md5 (cross-checked
    against the artifact's self-declared rows_md5)."""
    bundle = hashlib.md5(("|".join("%s:%s" % (k, tmpl[k]) for k in sorted(tmpl)))
                         .encode("utf-8")).hexdigest()
    out = {"code_files_md5_at_registration": tmpl,
           "code_bundle_md5": bundle,
           "basis": "现装版代码 (gen15 登记时刻); 该版本由 --recompute-check 全量重生成对拍背书 —— "
                    "gen12 登记时值 (10:45) 中 ccp_families/ccp_course/ccp_gate 三件早于终版课程, "
                    "不可用于复算; 见 recompute_check_evidence",
           "classes": {}}
    for cls in CLASSES:
        rp = os.path.join(GEN, "ccp_%s_rows.jsonl" % cls)
        cp = os.path.join(GEN, "ccp_%s_chains.jsonl" % cls)
        jp = os.path.join(GEN, "CCP_COURSE_%s.json" % cls.upper())
        rec = {"rows_mtime": _iso(rp), "chains_mtime": _iso(cp),
               "course_json_mtime": _iso(jp),
               "rows_md5": L.md5f(rp), "chains_md5": L.md5f(cp),
               "n_rows": sum(1 for _ in open(rp, encoding="utf-8")),
               "n_problems": sum(1 for _ in open(cp, encoding="utf-8"))}
        j = json.load(open(jp, encoding="utf-8"))
        rec["course_declared_rows_md5"] = j.get("rows_md5")
        rec["course_declared_chains_md5"] = j.get("chains_md5")
        rec["rows_md5_matches_course_json"] = (rec["rows_md5"] == j.get("rows_md5"))
        rec["chains_md5_matches_course_json"] = (rec["chains_md5"] == j.get("chains_md5"))
        rec["floor_n_dropped"] = j.get("floor_bulk", {}).get("n_dropped")
        out["classes"][cls] = rec
    return out


def _disk_chains(cls):
    p = os.path.join(GEN, "ccp_%s_chains.jsonl" % cls)
    return [json.loads(ln) for ln in open(p, encoding="utf-8")]


def _disk_rows(cls):
    p = os.path.join(GEN, "ccp_%s_rows.jsonl" % cls)
    return [json.loads(ln) for ln in open(p, encoding="utf-8")]


def _chains_line(p):
    return json.dumps({"name": p["name"], "family": p["family"], "stmt": p["stmt"],
                       "g1": p["g1"], "states": p["states"], "proof": p["proof"],
                       "chain": [{k: s[k] for k in ("idx", "op", "args", "sym",
                                                    "tactic", "cert_detail", "resid")}
                                 for s in p["chain"]]},
                      ensure_ascii=False, default=str)


def recompute_check():
    """Read-only full-regeneration cross-check (the on-disk state is ground truth). Returns the
    evidence dict; the caller persists it."""
    import ccp_course as C
    ev = {"schema": "ccp-recompute-check/1",
          "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
          "master_seed": MASTER, "rows_target": ROWS_TARGET,
          "method": "只读: 用现装代码全量重生成 (gen_problems + 同构去重), 与现盘 gen/ 逐字节对拍: "
                    "(1) 行内 seed 逐行按 _cseed(master,cls,family,seq) 复算; (2) 题级 name 序列与 stmt; "
                    "(3) 行级 JSONL 序列化逐字节 + md5; (4) 链级 JSONL 逐字节 + md5。"
                    "地板剔除步骤不重跑 Lean (以现盘题名集为准), 差集须与该类声明 n_dropped 相等。",
          "classes": {}, "all_ok": True, "writes": "none (本检查不写 gen/ 与 cache/)"}
    for cls in CLASSES:
        t0 = time.time()
        probs, errs = C.gen_problems(cls, ROWS_TARGET)
        seen, uniq = set(), []
        for p in probs:
            if p["stmt"] in seen:
                continue
            seen.add(p["stmt"])
            uniq.append(p)
        disk_ch = _disk_chains(cls)
        disk_names = [d["name"] for d in disk_ch]
        dset = set(disk_names)
        kept = [p for p in uniq if p["name"] in dset]
        # recompute the seed formula line by line (rows carry seed / ccp_name / ccp_family / cls)
        seed_n = seed_ok = 0
        for r in _disk_rows(cls):
            seq = int(r["ccp_name"].rsplit("_", 1)[1])
            fam = r["ccp_family"]
            seed_n += 1
            if (F._cseed(MASTER, r["cls"], fam, seq) == r["seed"]
                    and r["ccp_name"] == "ccp_%s_%s_%06d" % (r["cls"], fam.split("_", 1)[1], seq)):
                seed_ok += 1
        # row-level regeneration (same shape as run_class: same rng seed, on-disk problem order)
        rng = random.Random(C.MASTER_SEED + 11)
        rows = []
        for p in kept:
            for kind, r in C.build_rows(cls, p, rng):
                r["ccp_rowkind"] = kind
                rows.append(r)
        rows_txt = "".join(json.dumps(r, ensure_ascii=False, default=str) + "\n" for r in rows)
        disk_rows_txt = open(os.path.join(GEN, "ccp_%s_rows.jsonl" % cls),
                             encoding="utf-8").read()
        ch_txt = "".join(_chains_line(p) + "\n" for p in kept)
        disk_ch_txt = open(os.path.join(GEN, "ccp_%s_chains.jsonl" % cls),
                           encoding="utf-8").read()
        jp = os.path.join(GEN, "CCP_COURSE_%s.json" % cls.upper())
        j = json.load(open(jp, encoding="utf-8"))
        dropped = len(uniq) - len(kept)
        rec = {"n_problems_regen": len(uniq), "n_problems_on_disk": len(disk_names),
               "regen_superset_of_disk": dset <= {p["name"] for p in uniq},
               "dropped_not_on_disk": dropped,
               "floor_n_dropped_declared": j.get("floor_bulk", {}).get("n_dropped"),
               "dropped_explained_by_floor": dropped == j.get("floor_bulk", {}).get("n_dropped"),
               "order_ok": [p["name"] for p in kept] == disk_names,
               "stmt_ok": [p["stmt"] for p in kept] == [d["stmt"] for d in disk_ch],
               "seed_rows_checked": seed_n, "seed_rows_ok": seed_ok,
               "rows_byte_identical": rows_txt == disk_rows_txt,
               "rows_md5_regen": hashlib.md5(rows_txt.encode("utf-8")).hexdigest(),
               "rows_md5_disk": L.md5f(os.path.join(GEN, "ccp_%s_rows.jsonl" % cls)),
               "rows_md5_course_declared": j.get("rows_md5"),
               "chains_byte_identical": ch_txt == disk_ch_txt,
               "chains_md5_regen": hashlib.md5(ch_txt.encode("utf-8")).hexdigest(),
               "chains_md5_disk": L.md5f(os.path.join(GEN, "ccp_%s_chains.jsonl" % cls)),
               "gen_fail": dict(errs), "secs": round(time.time() - t0, 1)}
        rec["ok"] = bool(dset <= {p["name"] for p in uniq}
                         and rec["dropped_explained_by_floor"] and rec["order_ok"]
                         and rec["stmt_ok"] and seed_n == seed_ok
                         and rec["rows_byte_identical"] and rec["chains_byte_identical"]
                         and rec["rows_md5_disk"] == rec["rows_md5_course_declared"])
        ev["classes"][cls] = rec
        ev["all_ok"] = ev["all_ok"] and rec["ok"]
        print("[recompute %s] ok=%s | problems %d/%d (dropped %d = declared %s) | seeds %d/%d | rows byte %s | chains byte %s | %.0fs"
              % (cls, rec["ok"], len(uniq), len(disk_names), dropped,
                 rec["floor_n_dropped_declared"], seed_ok, seed_n,
                 rec["rows_byte_identical"], rec["chains_byte_identical"], rec["secs"]), flush=True)
    return ev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recompute-check", action="store_true",
                    help="read-only full-regeneration cross-check (on-disk state is ground truth); evidence goes to CCP_RECOMPUTE_CHECK.json and is echoed back into this table")
    a = ap.parse_args()
    tmpl = _md5s(TEMPLATES)
    reg = {
        "schema": "ccp-seed-registry/2",
        "revision": "gen15 (F-02): template_files_md5 刷新 (gen12 登记版 3/7 陈旧) + families.nt 刷新 "
                    "(对照组不入课程) + generation_versions (每类生成时刻/当时代码哈希/产物 md5) "
                    "+ recompute_check 可执行 (--recompute-check, 只读)",
        "master_seed": MASTER,
        "seed_formula": "per-problem: int(md5('ccp|<master>|<cls>|<family>|<seq>')[:8], 16); "
                        "seq 从 1 起, 按 FAM_OF[cls] 轮转族; 同 (master, cls, family, seq) 必得同题",
        "master_constants": {"ccp_families.MASTER_SEED(调用方)": MASTER,
                             "ccp_course.MASTER_SEED": 20260923,
                             "leak_avoidance": "与测试集种子 20260922 / SFT 90212026 / drct v2 92320260 不相交"},
        "families": {c: list(F.FAM_OF[c]) for c in F.CLASSES},
        "probe_families": {c: list(F.PROBE_FAMS[c]) for c in F.CLASSES},
        "row_mix": {"pos": 0.8, "endfail": 0.1, "neg": 0.1,
                    "note": "镜像 drct_curriculum_v2; endfail=episode{'fail_op':'END','after':k}; "
                            "neg=语义扰动真败 (CERT_REJECT/CONST_NOT_DIVISIBLE/RESIDUAL_NEGATIVE)"},
        "floor_risk_menus": RISK,
        "floor_8menu_sample": {"n_per_class": 100, "menus": ["simp", "norm_num", "decide", "linarith",
                                                            "nlinarith", "omega", "ring", "rfl"]},
        "leftout_split": "按 ccp_name 的 md5 前 8 位取模 10 == 7 的行入留出 (内容分组, 约 10%)",
        "template_files_md5": tmpl,
        "generation_versions": generation_versions(tmpl),
        "registry_generator": {"file": SELF, "md5": L.md5f(os.path.join(CCP, SELF))},
        "recompute_cmd": "cd $SCOPE_RESEARCH_ROOT && "
                         "PYTHONPATH=$SCOPE_SUPEROPERATOR $SCOPE_PYTHON "
                         "scratch/ccp/ccp_course.py --cls <ind|ineq|nt> --rows-target 3000 "
                         "[--skip-floor --skip-gate]",
        "recompute_check": "重跑后 gen/ccp_<cls>_rows.jsonl 的 md5 应与本批一致 "
                           "(地板/闸开关不影响生成确定性)",
        "recompute_check_cmd": "cd $SCOPE_RESEARCH_ROOT && "
                               "PYTHONPATH=$SCOPE_SUPEROPERATOR $SCOPE_PYTHON "
                               "scratch/ccp/ccp_seedreg.py --recompute-check  "
                               "(只读; 证据 = scratch/ccp/CCP_RECOMPUTE_CHECK.json)",
        "banned_ops": list(__import__("ccp_ops").BANNED_OPS),
        "proof_ops_whitelist": list(__import__("ccp_ops").PROOF_OPS),
    }
    if a.recompute_check:
        ev = recompute_check()
        ep = os.path.join(CCP, "CCP_RECOMPUTE_CHECK.json")
        emd5 = L.save_json(ep, ev)
        reg["recompute_check_evidence"] = {"path": ep, "md5": emd5, "ts": ev["ts"],
                                           "all_ok": ev["all_ok"],
                                           "classes_ok": {c: ev["classes"][c]["ok"] for c in CLASSES}}
        print("[seedreg] recompute_check all_ok=%s → %s md5=%s" % (ev["all_ok"], ep, emd5))
    p = os.path.join(CCP, "CCP_SEED_REGISTRY.json")
    md5 = L.save_json(p, reg)
    print("[seedreg] %s md5=%s schema=%s | templates=%d | gen_versions=%d"
          % (p, md5, reg["schema"], len(reg["template_files_md5"]),
             len(reg["generation_versions"]["classes"])))


if __name__ == "__main__":
    main()

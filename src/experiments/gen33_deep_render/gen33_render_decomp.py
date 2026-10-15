# -*- coding: utf-8 -*-
# gen33_render_decomp.py -- **decomposition reading** of the render axis (mechanism attribution for M1_D=0; post-hoc diagnostic, not a pre-registered criterion)
#
# Motivation: teacher-forced render_tok_acc reaches 0.9637 on the held-out segment while judging gives certified = 0 ⇒
#   **token-level accuracy is a misleading proxy**: proof-skeleton tokens (intro/induction/push_cast/ring...) dominate,
#   while the **integer-coefficient tokens** that carry the mathematical content are a small share, and one wrong one invalidates the whole proof.
# This module decomposes each problem along three axes:
#   S skeleton identical (integers -> placeholder, then char-by-char comparison)
#   N integer-position hits (integer sequences in statement/proof aligned position-wise)
#   L line-count ratio (render lines / gold lines)
# Inputs: GEN33_EVAL_<tag>_exam_deep_c*.json (render body) + EXAM_DEEP_CANDIDATE.json (gold proof)
#       + DEEP_RENDER_SIDECAR.jsonl / per-row gold (a_held control)
# Writes only scratch/ccp/gen33/**.
import argparse
import collections
import glob
import json
import os
import re

import g33_common as C

G33 = C.G33
OUT = os.path.join(G33, "out")
NUM = re.compile(r"-?\d+")


def skel(s):
    return NUM.sub("#", re.sub(r"\s+", " ", s or "").strip())


def nums(s):
    return NUM.findall(s or "")


def lines(s):
    return [l for l in (s or "").split("\n") if l.strip()]


def decomp(rows, gold_of):
    n = len(rows)
    n_skel = n_exact = 0
    num_hit = num_tot = 0
    n_short = n_long = 0
    have_g = have_h = 0
    by_cls = collections.defaultdict(lambda: {"n": 0, "skel": 0, "exact": 0,
                                              "num_hit": 0, "num_tot": 0})
    for r in rows:
        g = gold_of(r)
        if g is None:
            continue
        h = r["render"]["body"] if "render" in r else r["body"]
        c = r.get("cls")
        d = by_cls[c]
        d["n"] += 1
        if skel(g) == skel(h):
            n_skel += 1
            d["skel"] += 1
        if re.sub(r"\s+", " ", g).strip() == re.sub(r"\s+", " ", h).strip():
            n_exact += 1
            d["exact"] += 1
        a, b = nums(g), nums(h)
        m = min(len(a), len(b))
        hit = sum(1 for i in range(m) if a[i] == b[i])
        num_hit += hit
        num_tot += max(len(a), len(b))
        d["num_hit"] += hit
        d["num_tot"] += max(len(a), len(b))
        lg, lh = lines(g), lines(h)
        if len(lh) < len(lg) * 0.8:
            n_short += 1
        if len(lh) > len(lg) * 1.2:
            n_long += 1
        have_g += sum(1 for l in lg if l.strip().startswith("have"))
        have_h += sum(1 for l in lh if l.strip().startswith("have"))
    return {"n": n, "skeleton_identical": n_skel,
            "skeleton_rate": round(n_skel / max(1, n), 6),
            "full_string_exact": n_exact,
            "int_position_hit": num_hit, "int_position_tot": num_tot,
            "int_position_rate": round(num_hit / max(1, num_tot), 6),
            "render_shorter_than_gold": n_short, "render_longer_than_gold": n_long,
            "have_lines_gold_total": have_g, "have_lines_render_total": have_h,
            "by_cls": {k: dict(v, skeleton_rate=round(v["skel"] / max(1, v["n"]), 6),
                               int_position_rate=round(v["num_hit"] / max(1, v["num_tot"]), 6))
                       for k, v in sorted(by_cls.items())}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default="base,attn")
    ap.add_argument("--out", default=OUT + "/GEN33_RENDER_DECOMP.json")
    ap.add_argument("--log", default=G33 + "/logs/gen33.log")
    a = ap.parse_args()
    E = {x["name"]: x for x in json.load(open(C.EXAM_CAND, encoding="utf-8"))["items"]}
    rep = {"schema": "gen33-render-decomp/1",
           "note": ("事后诊断件 (非预注册判据): 分解 M1_D=0 的机理。"
                    "teacher-forced token acc 高 ≠ 证明可判: 骨架 token 占绝大多数, "
                    "整数系数 token 才是内容轴。"),
           "exam": {}}
    for t in [x.strip() for x in a.tags.split(",") if x.strip()]:
        rows = []
        for f in sorted(glob.glob(OUT + "/GEN33_EVAL_%s_exam_deep_c*.json" % t)):
            rows += json.load(open(f, encoding="utf-8"))["rows"]
        rep["exam"][t] = decomp(rows, lambda r: (E.get(r["name"]) or {}).get("proof_text"))
    ctrl = OUT + "/GEN33_EVAL_base_a_held_n20.json"
    if os.path.exists(ctrl):
        d = json.load(open(ctrl, encoding="utf-8"))
        gold = {r["name"]: r["proof_text"]
                for r in C.jlload(C.DEEP_CORPUS)}
        rows = [{"name": r["name"], "cls": r["cls"],
                 "body": (r.get("render") or {}).get("body") or ""} for r in d["rows"]]
        ch = json.load(open(OUT + "/GEN33_CTRL_HELD.json", encoding="utf-8"))
        rep["ctrl_a_held_n20"] = {
            "decomp": decomp(rows, lambda r: gold.get(r["name"])),
            "judge": {"n": ch["n"], "certified": ch["certified"], "reasons": ch["reasons"]},
            "note": ("同分布控制 (非考题留出段 20 行, 基线臂, device 口径): "
                     "用于区分『考题迁移失败』与『渲染通道本身失败』")}
    rep["md5"] = C.save_json(a.out, rep)
    C.log("RENDER_DECOMP: exam base skel=%s int=%s | attn skel=%s int=%s | ctrl_held certified=%s/%s"
          % (rep["exam"]["base"]["skeleton_rate"], rep["exam"]["base"]["int_position_rate"],
             rep["exam"].get("attn", {}).get("skeleton_rate"),
             rep["exam"].get("attn", {}).get("int_position_rate"),
             rep.get("ctrl_a_held_n20", {}).get("judge", {}).get("certified"),
             rep.get("ctrl_a_held_n20", {}).get("judge", {}).get("n")), a.log)
    print("GEN33_RENDER_DECOMP_RC=0 -> %s" % a.out)
    return rep


if __name__ == "__main__":
    main()

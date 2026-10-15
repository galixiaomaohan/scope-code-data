#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
gen44_extract_lwb.py -- node [0] Lean-Workbook acquisition + per-row registration + md5 ledger

Responsibilities (zero GPU / zero Lean / CPU):
  1. Read data/raw/lean_workbook.json (the official internlm/Lean-Workbook main artifact,
     140214 rows).
  2. Canonical convention (following gen41's handling of miniF2F; the criteria are hard-coded
     here, never hand-tuned per problem):
       C1 head matches `^theorem\s+NAME`
       C2 both `:=` and `by` present (this corpus always uses `:=  by sorry`)
       C3 single theorem: `theorem` appears once and `:= by` appears once in the block
       C4 the proof body is exactly `sorry` (no leftover proof)
       C5 a top-level colon exists => head/concl split succeeded (head convention as in
          probes/lean_exam.py.txt: head = "theorem NAME <binders> :" including the colon;
          concl = the conclusion text as-is)
       C6 non-empty binders + concl
     Any failure => the row is registered in LWB_EXCLUDED.jsonl (never silently dropped).
  3. Uniqueness check: name unique + normalized statement body (theorem name removed, whitespace
     squeezed) unique; duplicate rows are **all kept**, tagged with dup_group (first-seen idx of
     the group), and duplicate-group statistics are registered in the ledger.
  4. Per-row md5 (formal_statement) + per-artifact md5.
  5. Secondary artifact wkbk_1009.parquet registered as well (step-level file; counts only, not
     part of the main registry).
  6. Structural statistics (no semantic reading): subset counts / tag histogram / binder shapes /
     top-level operator histogram of the conclusion (purely lexical).

Write side: only scratch/ccp/gen44/**. Existing files unchanged; read-only, no imports.
Usage:
  cd $R && PYTHONDONTWRITEBYTECODE=1 $PY gen44_extract_lwb.py --all
"""
import argparse
import collections
import hashlib
import json
import os
import re
import unicodedata
from datetime import datetime, timezone

RES = os.environ.get("SCOPE_RESEARCH_ROOT", ".")
DATA = os.path.join(RES, "scratch", "ccp", "gen44", "data")
RAW = os.path.join(DATA, "raw")
SCHEMA = "gen44-lwb/1"
SRC_URL = "https://hf-mirror.com/datasets/internlm/Lean-Workbook"
SRC_REPO = "internlm/Lean-Workbook"
SRC_SHA = "2e066e310b2c6d2c27616927ae131f82901c8f1c"

_WS = re.compile(r"\s+")
_DOC_RE = re.compile(r"/--.*?-/", re.S)
_NAME_RE = re.compile(r"\s*theorem\s+([A-Za-z0-9_.'\u00c0-\u024f]+)")
_BY_RE = re.compile(r":=\s*by\b")
_SORRY_ONLY = re.compile(r"\Asorry\s*\Z")


def md5f(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def md5s(s):
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def unicode_norm(s):
    s = unicodedata.normalize("NFC", s)
    for a, b in (("\u00a0", " "), ("\u2009", " "), ("\u200a", " "), ("\u200b", "")):
        s = s.replace(a, b)
    return s


def norm_ws(s):
    return _WS.sub(" ", s or "").strip()


def strip_lean_comments(s):
    return _DOC_RE.sub(" ", s or "")


def split_top_colon(s):
    """Find the top-level colon (not inside brackets, and not part of `:=`). Returns -1 when
    not found. The convention is copied verbatim from probes/lean_exam.py.txt::_split_top_colon."""
    depth, i = 0, 0
    while i < len(s):
        ch = s[i]
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif ch == ":" and depth == 0:
            nxt = s[i + 1] if i + 1 < len(s) else ""
            if nxt == "=":
                i += 1
            else:
                return i
        i += 1
    return -1


def top_rel(s):
    """Outermost relation symbol (bracket depth 0); returns (rel, idx) or (None, -1).
    Lexical level."""
    rels = ("≠", "≤", "≥", "=", "<", ">", "∣", "↔", "→", "∧", "∨")
    d = 0
    for i, ch in enumerate(s):
        if ch in "([{":
            d += 1
        elif ch in ")]}":
            d -= 1
        elif d == 0 and ch in rels:
            return ch, i
    return None, -1


def canonicalize(rec, idx):
    """-> (row, None) or (None, reason)."""
    fs = rec.get("formal_statement") or ""
    if not isinstance(fs, str) or not fs.strip():
        return None, "empty_statement"
    m = _NAME_RE.match(fs)
    if not m:
        return None, "no_theorem_head"
    name = m.group(1)
    if fs.count("theorem") != 1:
        return None, "multi_theorem"
    # `:=` + by
    bym = _BY_RE.search(fs)
    if not bym:
        return None, "no_by"
    if len(re.findall(r":=\s*by", fs)) != 1:
        return None, "multi_by"
    head_part = fs[m.end():bym.start()]
    proof_txt = fs[bym.end():].strip()
    if not _SORRY_ONLY.match(proof_txt):
        return None, "proof_not_sorry:%s" % proof_txt[:20]
    ci = split_top_colon(head_part)
    if ci < 0:
        return None, "no_top_colon"
    binders = head_part[:ci].strip()
    concl = norm_ws(head_part[ci + 1:])
    if not concl:
        return None, "empty_concl"
    head = "theorem %s %s :" % (name, binders)
    head = norm_ws(head)
    body_key = norm_ws("%s | %s" % (binders, concl))
    return {
        "schema": SCHEMA, "idx": idx, "name": name,
        "subset": rec.get("split"),
        "head": head, "binders": binders, "concl": concl,
        "stmt_md5": md5s(fs),
        "body_key_md5": md5s(body_key),
        "n_binder_chars": len(binders), "n_concl_chars": len(concl),
        "answer": rec.get("answer") or "",
        "tags": rec.get("tags") or [],
        "n_proof_steps": len(rec.get("proof") or []),
        "nl_len": len(rec.get("natural_language_statement") or ""),
        "nl": rec.get("natural_language_statement") or "",
    }, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--raw", default=os.path.join(RAW, "lean_workbook.json"))
    ap.add_argument("--parquet", default=os.path.join(RAW, "wkbk_1009.parquet"))
    ap.add_argument("--outdir", default=DATA)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    t0 = datetime.now(timezone.utc).astimezone()
    src_md5 = md5f(a.raw)
    d = json.load(open(a.raw, encoding="utf-8"))
    n_raw = len(d)

    rows, excl = [], []
    for i, rec in enumerate(d):
        row, why = canonicalize(rec, i)
        if row is None:
            excl.append({"idx": i,
                         "name": (re.match(r"\s*theorem\s+(\S+)",
                                           rec.get("formal_statement") or "") or
                                  [None, None])[1],
                         "reason": why,
                         "stmt_head": norm_ws((rec.get("formal_statement") or "")[:160])})
            continue
        rows.append(row)

    # ---- uniqueness check (name / body_key)
    nc = collections.Counter(r["name"] for r in rows)
    bc = collections.Counter(r["body_key_md5"] for r in rows)
    name_dups = {k: v for k, v in nc.items() if v > 1}
    body_dups = {k: v for k, v in bc.items() if v > 1}
    first_of = {}
    for r in rows:
        k = r["body_key_md5"]
        r["dup_group"] = first_of.get(k)
        if k not in first_of:
            first_of[k] = r["idx"]
    n_dup_rows = sum(bc[k] - 1 for k in body_dups)

    # ---- structural statistics (no semantic reading; purely lexical)
    rel_hist = collections.Counter()
    for r in rows:
        rel, _ = top_rel(r["concl"])
        rel_hist[rel or "__NONE__"] += 1
    tag_hist = collections.Counter(t for r in rows for t in r["tags"])
    sub_hist = collections.Counter(r["subset"] for r in rows)

    # ---- write out: main registry
    allp = os.path.join(a.outdir, "LWB_ALL.jsonl")
    with open(allp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    exp = os.path.join(a.outdir, "LWB_EXCLUDED.jsonl")
    with open(exp, "w", encoding="utf-8") as f:
        for r in excl:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ---- secondary artifact registration (step-level parquet; counts only)
    pq_meta = {"present": os.path.exists(a.parquet)}
    if pq_meta["present"]:
        pq_meta["md5"] = md5f(a.parquet)
        try:
            import pyarrow.parquet as pq                     # noqa: E402
            t = pq.ParquetFile(a.parquet).read()
            dd = t.to_pylist()
            ids = set(x["id"] for x in dd)
            names = set(r["name"] for r in rows)
            pq_meta.update({
                "rows": len(dd),
                "columns": list(t.schema.names),
                "distinct_ids": len(ids),
                "distinct_formal_statement": len(set(x["formal_statement"] for x in dd)),
                "status_hist": dict(collections.Counter(x["status"] for x in dd)),
                "ids_outside_main": len(ids - names),
                # NOTE: Chinese report payload value kept verbatim (data, not a log line).
                "role": "步级副件 (state_before/state_after/tactic); 非 statement 划分; 不入主册",
            })
        except Exception as e:                               # noqa: BLE001
            pq_meta["error"] = repr(e)

    ledger = {
        "schema": "gen44-lwb-ledger/1",
        "created": t0.isoformat(timespec="seconds"),
        "source": {"repo": SRC_REPO, "url": SRC_URL, "sha": SRC_SHA,
                   "hf_endpoint": "https://hf-mirror.com",
                   "license": "apache-2.0",
                   # NOTE: Chinese report payload value kept verbatim (data, not a log line).
                   "note": "官方分 split = lean_workbook / lean_workbook_plus (非 train/valid); 全量入册"},
        "files": [{"path": "raw/lean_workbook.json", "bytes": os.path.getsize(a.raw),
                   "md5": src_md5, "etag_sha256": "7f17892be5748b54d4a98860950f8ffde86b7ee534e8bcbe42ac15b1c8c3f945"},
                  {"path": "raw/wkbk_1009.parquet",
                   "bytes": (os.path.getsize(a.parquet) if os.path.exists(a.parquet) else None),
                   "md5": (md5f(a.parquet) if os.path.exists(a.parquet) else None),
                   "etag_sha256": "5351b8449fc73c36339ab718bc268ad4d43ebaa77c121fea6c008d04b82ac74b"}],
        "n_raw": n_raw,
        "n_canonical": len(rows),
        "n_excluded": len(excl),
        "exclude_hist": dict(collections.Counter(e["reason"].split(":")[0] for e in excl)),
        "uniqueness": {
            "name_distinct": len(nc), "name_dup_names": len(name_dups),
            "body_distinct": len(bc), "body_dup_groups": len(body_dups),
            "body_dup_rows": n_dup_rows,
            "max_group_size": (max(bc.values()) if bc else 0),
            # NOTE: Chinese report payload value kept verbatim (data, not a log line).
            "policy": "全量保留 (全量入册); dup_group = 同组首见 idx; 普查总数口径 = 全量 (副列出 dedup 数)",
        },
        "subset_hist": dict(sub_hist),
        "tag_hist": dict(tag_hist.most_common()),
        "concl_top_rel_hist": dict(rel_hist.most_common()),
        "parquet_secondary": pq_meta,
        "products": {"LWB_ALL.jsonl": {"n": len(rows), "md5": md5f(allp)},
                     "LWB_EXCLUDED.jsonl": {"n": len(excl), "md5": md5f(exp)}},
        "write_face": "scratch/ccp/gen44/**",
    }
    lp = os.path.join(a.outdir, "LWB_LEDGER.json")
    json.dump(ledger, open(lp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps({k: ledger[k] for k in
                      ("n_raw", "n_canonical", "n_excluded", "exclude_hist",
                       "subset_hist", "uniqueness", "parquet_secondary")},
                     ensure_ascii=False, indent=1))
    print("products:", json.dumps(ledger["products"], ensure_ascii=False))
    print("concl_top_rel_hist:", json.dumps(ledger["concl_top_rel_hist"],
                                            ensure_ascii=False))


if __name__ == "__main__":
    main()

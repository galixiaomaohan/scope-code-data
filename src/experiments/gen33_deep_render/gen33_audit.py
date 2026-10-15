# -*- coding: utf-8 -*-
# gen33_audit.py -- AUD-1 reading discipline (institutionalized for the main generation; re-implements gen26r_baseline B1/B2/B3)
#   * B1 chain diversity (whole set + train segment + per class)
#   * B2 render-target template collapse (per-class mode share)
#   * B3 zero-parameter per-class per-position mode baseline (train statistics -> eval slice; render and chain axes)
#   * B4 paired two-arm table (same eval slice, per-column deltas)
# Rule: B1/B2/B3 follow gen30_audit_baseline.py line-for-line (objects swapped to deep corpus/deep curves).
# Read-only inputs; writes only under scratch/ccp/gen33/out/**.
import argparse
import collections
import json
import os

import g33_common as C

G33 = C.G33
OUT = os.path.join(G33, "out")


def load_side(p):
    side = {}
    for line in open(p, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            side[int(r["ri"])] = r
    return side


def load_curve(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def build(side_path, curve_attn, curve_base, logp=None):
    side = load_side(side_path)
    train = [i for i in side if side[i]["seg"] == "a_train"]
    held = [i for i in side if side[i]["seg"] == "a_held"]
    rep = {"schema": "gen33-audit-baseline/1", "side": side_path,
           "side_md5": C.md5f(side_path), "n_train": len(train), "n_held": len(held)}

    # ---- B1 chain diversity ----
    for tag, keys in (("all", sorted(side)), ("train", train)):
        rep["B1_chain_distinct_%s" % tag] = len(set(tuple(side[i]["ops"]) for i in keys))
        rep["B1_chain_distinct_by_cls_%s" % tag] = len(
            {(side[i]["cls"], tuple(side[i]["ops"])) for i in keys})
    ch = collections.Counter(tuple(side[i]["ops"]) for i in train)
    rep["B1_chain_hist_train"] = {"+".join(k): v for k, v in ch.most_common()}
    rep["B1_note"] = ("深语料链种类数 = 语料级性质 (与模型无关); "
                      "停手线对『模型走出链的种类数』另列 (考段)")

    # ---- B2 render-target template collapse ----
    tgt = collections.defaultdict(collections.Counter)
    for i in train:
        s = side[i]
        tgt[s["cls"]][tuple(s["ids"][s["np"]:])] += 1
    rep["B2_target_template_train"] = {
        c: {"n_rows": sum(cnt.values()), "n_distinct_targets": len(cnt),
            "mode_share": round(max(cnt.values()) / sum(cnt.values()), 4),
            "mode_tokens": len(cnt.most_common(1)[0][0])}
        for c, cnt in sorted(tgt.items())}

    # ---- B3 zero-parameter per-class per-position mode baseline (render axis + chain axis) ----
    pos_mode = collections.defaultdict(collections.Counter)
    for i in train:
        s = side[i]
        for off, tok in enumerate(s["ids"][s["np"]:]):
            pos_mode[(s["cls"], off)][tok] += 1
    mode = {k: v.most_common(1)[0][0] for k, v in pos_mode.items()}

    op_mode = collections.defaultdict(collections.Counter)
    for i in train:
        s = side[i]
        for k, op in enumerate(s["ops"]):
            op_mode[(s["cls"], k)][op] += 1
    omode = {k: v.most_common(1)[0][0] for k, v in op_mode.items()}

    def slice_baseline(idx):
        hit = tot = 0
        for i in idx:
            s = side[i]
            for off, tok in enumerate(s["ids"][s["np"]:]):
                m = mode.get((s["cls"], off))
                if m is None:
                    continue
                tot += 1
                hit += int(m == tok)
        return {"hit": hit, "tot": tot, "acc": round(hit / max(tot, 1), 6)}

    def slice_chain_baseline(idx):
        hit = tot = pos_hit = pos_tot = 0
        for i in idx:
            s = side[i]
            pred = [omode.get((s["cls"], k)) for k in range(len(s["ops"]))]
            if all(p is not None for p in pred):
                tot += 1
                hit += int(pred == list(s["ops"]))
            for k, op in enumerate(s["ops"]):
                m = omode.get((s["cls"], k))
                if m is None:
                    continue
                pos_tot += 1
                pos_hit += int(m == op)
        return {"exact_hit": hit, "n": tot,
                "exact_rate": round(hit / max(tot, 1), 6),
                "pos_acc": {True: round(pos_hit / max(pos_tot, 1), 6), False: None}[True],
                "pos_hit": pos_hit, "pos_tot": pos_tot}

    rep["B3_render_mode_baseline_all"] = slice_baseline(sorted(side))
    rep["B3_render_mode_baseline_held"] = slice_baseline(held)
    rep["B3_chain_mode_baseline_train"] = slice_chain_baseline(train)
    rep["B3_chain_mode_baseline_held"] = slice_chain_baseline(held)
    rep["B3_rule"] = ("训练同类 target/链 逐位置众数 (零参数); 越界跳过 "
                      "(gen30_audit_baseline.B3 同式)")

    # ---- B4 paired two-arm table (same eval slice) ----
    def curve_head(p):
        if not os.path.exists(p):
            return None
        cur = load_curve(p)
        ev = [r for r in cur if r.get("eval_idx")]
        if not ev:
            return None
        last = ev[-1]
        return {"path": p, "md5": C.md5f(p), "steps": len(cur), "step": last["step"],
                "eval_idx": last["eval_idx"], "eval": last.get("eval"),
                "last": {k: last.get(k) for k in ("loss", "chain", "slot", "text", "gn",
                                                  "L", "pa_delta_max", "ps_delta_max")},
                "min": {k: min(r[k] for r in cur) for k in ("loss", "chain", "slot", "text")}}

    ca, cb = curve_head(curve_attn), curve_head(curve_base)
    rep["curves"] = {"attn": ca, "base": cb}
    if ca and cb:
        same = ca["eval_idx"] == cb["eval_idx"]
        rep["B4_slice_identity"] = {"same_eval_idx": bool(same), "step_attn": ca["step"],
                                    "step_base": cb["step"], "n": len(ca["eval_idx"] or [])}
        mb = slice_baseline(ca["eval_idx"])
        rep["B4_mode_baseline"] = dict(mb)
        rows = {}
        for k in ("sel_top1", "slot_top1", "slot_top1_nonunk", "render_tok_acc",
                  "n_frames", "n_slots", "n_render_tok"):
            va, vb = (ca["eval"] or {}).get(k), (cb["eval"] or {}).get(k)
            rows[k] = {"attn": va, "base": vb,
                       "delta": (round(va - vb, 6) if isinstance(va, (int, float))
                                 and isinstance(vb, (int, float)) else None)}
        for k in ("loss", "chain", "slot", "text"):
            va, vb = (ca["last"] or {}).get(k), (cb["last"] or {}).get(k)
            rows["last_" + k] = {"attn": va, "base": vb,
                                 "delta": (round(va - vb, 6) if isinstance(va, (int, float))
                                           and isinstance(vb, (int, float)) else None)}
        acc = (ca["eval"] or {}).get("render_tok_acc")
        if acc is not None:
            rows["render_tok_acc_vs_mode_baseline"] = {
                "model": acc, "baseline": mb["acc"], "tot_tokens": mb["tot"],
                "delta_pp": round((acc - mb["acc"]) * 100, 3),
                "delta_tokens": int(round((acc - mb["acc"]) * mb["tot"]))}
        rep["paired_table_train_side"] = rows
        rep["AUD1_discipline"] = {
            "mode_baseline_same_slice": mb["acc"],
            "attn_render_acc": acc,
            "base_render_acc": (cb["eval"] or {}).get("render_tok_acc"),
            "chain_mode_exact_rate_held": rep["B3_chain_mode_baseline_held"]["exact_rate"],
            "note": "AUD-1: 报读数必同报零参数众数基线 (pp+token, 同切片) 与链种类数"}
    if logp:
        C.log("AUD1 mode_baseline=%s chain_mode_held_exact=%s chain_distinct_train=%s"
              % (rep.get("B3_render_mode_baseline_held", {}).get("acc"),
                 rep["B3_chain_mode_baseline_held"]["exact_rate"],
                 rep.get("B1_chain_distinct_train")), logp)
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--side", default=OUT + "/DEEP_RENDER_SIDECAR.jsonl")
    ap.add_argument("--curve-attn", default=OUT + "/DEEP_ATTN_CURVE.jsonl")
    ap.add_argument("--curve-base", default=OUT + "/DEEP_BASE_CURVE.jsonl")
    ap.add_argument("--out", default=OUT + "/GEN33_AUDIT_BASELINE.json")
    ap.add_argument("--log", default=G33 + "/logs/gen33.log")
    a = ap.parse_args()
    rep = build(a.side, a.curve_attn, a.curve_base, logp=a.log)
    rep["md5"] = C.save_json(a.out, rep)
    print("GEN33_AUDIT_RC=0 -> %s md5=%s" % (a.out, rep["md5"]))
    return rep


if __name__ == "__main__":
    main()

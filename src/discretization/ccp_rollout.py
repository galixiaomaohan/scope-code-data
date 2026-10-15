# -*- coding: utf-8 -*-
# ccp_rollout.py -- per-frame operator-sequence decoding with the CCP model + three readouts (gen13 task E①)
#
# Convention (registered):
#   * input = the statement (same surface form as the training side: "<binders> ⊢ <concl>") + T4 = [1,0,0,0] (the only type4 in the CCP course).
#   * decoding = greedy / multi-sample: the decision-point t indices match training (idxH = last statement frame, idxS = previous frame / last statement frame);
#     the stepping head's argmax -> operator id; a ccp_* operator hit calls the **certificate machine** (ccp_ops.call) for the result frame,
#     appends (M_OP/op, M_RES/result, M_TAGFR+nf, PST=ok) and continues; an END(154) hit stops; any other id is recorded as
#     out_of_domain and stops (no such operator is claimed).
#   * **argument-provenance disclosure (same discipline as the D-RCT lexrule arm)**: this problem family's operator arguments are derived by the
#     **certificate machine from the statement** (ind: statement closed form; ineq: bridge_ineq->dom->square terms; nt: bridge_nt); the model contributes the **operator order**.
#     The certified readout therefore measures "model operator-order decisions + device rendering" as a whole, not the discriminative power of the bound slots;
#     bound slots get a separate diagnostic readout (bind_slot_acc: model-predicted value vs certificate-machine-derived value).
#   * plan acceptance = the operator order matches that class's **registered template** item by item (see PLAN_PATTERN); a mismatch means no plan (the why field records it).
#   * judging = Lean ∧ N1 ∧ N2 (drct_common.judge_attempt; the independent judging suffix is in the summary output name).
import argparse
import collections
import hashlib
import json
import os
import re
import struct
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

RES = os.environ.get("SCOPE_RESEARCH_ROOT", ".")
CCP = os.path.join(RES, "scratch", "ccp")
G13 = os.path.join(CCP, "gen13")
OUT = os.path.join(G13, "out")
LEANWD = os.path.join(G13, "lean")
os.makedirs(OUT, exist_ok=True)
os.makedirs(LEANWD, exist_ok=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))          # local ccp_* modules in the package layout
sys.path.insert(0, CCP)
sys.path.insert(0, G13)
sys.path.insert(0, os.path.join(RES, "scratch", "drct"))
sys.path.insert(0, os.path.join(RES, "cppcuda", "tools"))
sys.path.insert(0, os.path.join(RES, "skel32"))
sys.path.insert(0, os.environ.get("SCOPE_SUPEROPERATOR",
                                  os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "device")))

import numpy as np                                                      # noqa: E402
import torch                                                            # noqa: E402

import ccp_bridge as BR                                                # noqa: E402
import ccp_ops as OPS                                                  # noqa: E402
import ccp_rulearm as RA                                               # noqa: E402

OP_IND, OP_RING = 161, 162
OP_SOS, OP_CLOSE = 163, 164
OP_WIT, OP_COMB, OP_PROD = 165, 166, 167
END_ID = 154
M_TASK, M_TYPE, M_OP, M_RES, M_END, M_TAGFR = 730, 731, 732, 733, 734, 740
ST_IDX = {"ok": 0, "fail_c": 1, "fail_r": 2, "end": 3}
KIND_EMB, KIND_OP, KIND_RES, KIND_PST, KIND_WIN, KIND_PSZ = 0, 1, 2, 3, 4, 5
UNK_ID = 0


def _md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class ShimRow(object):
    __slots__ = ("ids", "t4", "readout_only", "pos_task_end", "end_pos", "readout", "desc",
                 "resHas", "res", "labels", "feat_pos", "bsl", "state_pos", "src", "stmt_txt")


class ShimCache(object):
    """Single-row pseudo cache (BatchCpu32/assemble read only these fields; same semantics as a CPPCD5 row)."""

    def __init__(self, t4=(1.0, 0.0, 0.0, 0.0)):
        self.rows = []
        self.t4 = [tuple(t4)]
        self.nT4 = 1
        self.HEAD, self.NVALS, self.ENDIDX = 168, 0, END_ID
        self.segidx = None

    def seg_of(self, i):
        return None


def build_row(tok, stmt, max_tok=512):
    R = ShimRow()
    R.stmt_txt = stmt
    ids = tok.convert_tokens_to_ids(tok.tokenize(stmt))[:max_tok] or [tok.bos_token_id or 0]
    R.ids = ids
    R.t4 = 0
    R.readout_only = 0
    R.desc = [(KIND_EMB, M_TASK)] + [(KIND_EMB, int(t)) for t in ids] + \
             [(KIND_EMB, M_TYPE), (KIND_WIN, 0)]
    R.pos_task_end = len(R.desc) - 1
    R.feat_pos, R.res, R.resHas = [], [], []
    R.labels = [END_ID]          # placeholder (Talloc >= 1; the logits for decision point t=0 come from idxS = last statement frame)
    R.bsl = [-1] * 24
    R.readout, R.state_pos = 0.0, []
    R.end_pos = R.pos_task_end
    R.src = None
    return R


class Model(object):
    def __init__(self, ckpt, dev="cuda:0", vocab=None, base=None):
        # gen13-era trainer module; lives only in the experiment-tree gen13
        # directory (see README.md section 7, "Not shipped")
        import v32ccp_train as T
        voc = json.load(open(vocab or os.path.join(G13, "CCP_VOCAB.json"), encoding="utf-8"))
        self.head = voc["head"]
        self.nvals = voc["n_vals_ext"]
        self.parents = {int(k): v["parent_idx"] for k, v in voc["init_map"].items()}
        self.allow = torch.from_numpy(np.fromfile(
            os.path.join(G13, "ccp_allow_ext.bin"), dtype=np.uint8).reshape(
            self.head, 24, self.nvals).copy())
        self.dev = torch.device(dev)
        m = T.CCPModel(base=base, lat_tab_path="cppcuda/cache/lat_tab_v31.bin",
                       head=self.head, nvals=self.nvals, parents=self.parents)
        sd = torch.load(ckpt, map_location="cpu", weights_only=True)
        missing, unexpected = m.load_state_dict(sd, strict=False)
        assert not [k for k in missing if not k.startswith("lm.lm_head")], missing[:5]
        assert not unexpected, list(unexpected)[:5]
        self.model = m.to(self.dev).eval()
        self.vocab_id2val = self._id2val(vocab or os.path.join(G13, "CCP_VOCAB.json"))
        self.tok = None
        self.ckpt = ckpt

    @staticmethod
    def _id2val(vocab_path):
        """Value-vocabulary id -> (kind, value) reverse lookup (151 base values + 88 appended values)."""
        import ccp_vocab as V
        tr, L32, O32, NAMES_IDX, toks_base, per = V.load_base(verbose=False)
        voc = json.load(open(vocab_path, encoding="utf-8"))
        toks = list(toks_base) + [None] * voc["n_extra_values"]
        for ev in voc["extra_values"]:
            toks[ev["idx"]] = (ev["kind"], ev.get("typed_value", ev["value"]))
        return toks

    @torch.no_grad()
    def step_logits(self, C, R, dev=None):
        # gen13-era trainer module (host-tree only; README.md section 7)
        import v32ccp_train as T
        L32 = T.L32
        dev = dev or self.dev
        bc = L32.BatchCpu32().build(C, [0], 0, 0, 0)
        bc.rowIds = [0]
        cm, cosc, sinc = self.model.consts(bc.L, dev)
        mk = self.model.mask_of(cm, bc.B, bc.L, bc.lenB, dev)
        seq = self.model.assemble(bc.B, bc.L, bc, C, dev)
        H = self.model.forward_body(seq, mk, cosc, sinc)
        lg, zb, _t = self.model.heads(H, bc, C, dev, self.allow)
        T = bc.Talloc
        t = len(R.feat_pos)                     # current decision point (= training-side labels index)
        sl = lg.view(bc.B, T, -1)[0, t].float()
        vals = zb.view(bc.B, T, 24, self.nvals)[0, t].float()
        return sl.cpu(), vals.cpu()

    def rollout(self, stmt, cls, n_attempts=1, greedy=True, seed=20260923, max_steps=8,
                temperature=1.0, top_p=0.95):
        """-> per-attempt records (ops sequence / bound-value diagnostics / why)."""
        C = ShimCache()
        R = build_row(self.tok, stmt)
        C.rows = [R]
        attempts = []
        for k in range(n_attempts):
            hs = int(hashlib.md5(stmt.encode("utf-8")).hexdigest()[:8], 16) % 997
            gen = torch.Generator().manual_seed(seed + 1000 * hs + k)
            R2 = build_row(self.tok, stmt)
            C.rows = [R2]
            ops, binds, why, pred = [], [], "", []
            for _ in range(max_steps):
                sl, vals = self.step_logits(C, R2)
                if greedy:
                    op = int(sl.argmax())
                else:
                    p = torch.softmax(sl / temperature, -1)
                    idx = torch.argsort(p, descending=True)
                    cum = torch.cumsum(p[idx], 0)
                    keep = idx[cum <= top_p]
                    keep = keep if len(keep) else idx[:1]
                    pi = p[keep] / p[keep].sum()
                    op = int(keep[torch.multinomial(pi, 1, generator=gen)])
                if op == END_ID:
                    pred.append("END")
                    why = "end"
                    break
                name = OPNAME.get(op)
                pred.append(name or ("op%d" % op))
                if name is None:
                    why = "out_of_domain:op%d" % op
                    break
                # bound-slot diagnostic (argmax inside the allow mask)
                al = self.allow[op].bool()
                sv = []
                for j in range(24):
                    row = vals[j].masked_fill(~al[j], -1e9)
                    sv.append(int(row.argmax()) if bool(al[j].any()) else -1)
                binds.append({"op": name, "slots": sv})
                args = arm_args(cls, name, R2, sv, self.vocab_id2val)
                if args is None:
                    why = "args_unavailable:%s" % name
                    break
                r = OPS.call(name, args)
                ops.append(name)
                if not r.ok:
                    why = "op_fail:%s" % (r.err or "?")
                    break
                # append the result frame (same form as the cache side)
                nf = len([1 for (kd, _a) in R2.desc if kd == KIND_OP])
                rv = r.resid
                resv = None if rv is None else ([float(x) for x in rv] + [0.0] * 8)[:8]
                R2.desc += [(KIND_EMB, M_OP), (KIND_OP, op), (KIND_EMB, M_RES),
                            (KIND_RES, len(R2.res)) if resv is not None else (KIND_PSZ, 0),
                            (KIND_EMB, M_TAGFR + nf), (KIND_PST, ST_IDX["ok"])]
                R2.res.append(resv)
                R2.resHas.append(1 if resv is not None else 0)
                R2.feat_pos.append(len(R2.desc) - 1)
                R2.labels.append(op)
                R2.bsl += [-1] * 24
            else:
                why = "max_steps"
            attempts.append({"attempt": k + 1, "ops": ops, "pred_ops": pred,
                             "binds": binds, "why": why, "n_steps": len(ops)})
        return attempts


OPNAME = {}

# ---------------------------------------------------------------- arguments = certificate machine derived from the statement
def arm_args(cls, opname, R, sv, id2val):
    """Derive the operator arguments from the statement (same convention as the D-RCT device; returns None when unavailable)."""
    stmt_full = stmt_of(R)
    if opname == "ccp_ind_recurrence":
        try:
            br = BR.bridge_ind(stmt_full)
            return {"form": br["form"], "v0": br["v0"], "step": br["step"], "sample_n": 3}
        except Exception:                                              # noqa: BLE001
            return None
    if opname == "ccp_ring_identity":
        return {"nvars": 1, "atom": "n", "side": 4, "lhs": {"0": "1"}, "rhs": {"0": "1"}}
    if opname == "ccp_ineq_sos":
        vars_ = RA.vars_of_binders(RA.binders_of(head_of(R)))
        try:
            br = BR.bridge_ineq(stmt_full, vars_)
        except Exception:                                              # noqa: BLE001
            return None
        return {"vars": vars_, "dom": {",".join(map(str, m)): str(c)
                                       for m, c in br["dom"].items()}}
    if opname == "ccp_ineq_close":
        vars_ = RA.vars_of_binders(RA.binders_of(head_of(R)))
        try:
            br = BR.bridge_ineq(stmt_full, vars_)
        except Exception:                                              # noqa: BLE001
            return None
        dom = {",".join(map(str, m)): str(c) for m, c in br["dom"].items()}
        r1 = OPS.call("ccp_ineq_sos", {"vars": vars_, "dom": dom})
        if not (r1.ok and r1.cert_ok):
            return None
        return {"vars": vars_, "squares": r1.value["squares"],
                "residual": r1.value["residual"], "text": concl_of(R)}
    if opname == "ccp_nt_dvd_witness":
        try:
            br = BR.bridge_nt(stmt_full)
        except Exception:                                              # noqa: BLE001
            return None
        k = len([a for a in br["hyps"]])
        i = len([1 for (kd, _a) in R.desc if kd == KIND_OP])
        if i >= k:
            return None
        h, mm, v = br["hyps"][i]
        return {"m": mm, "hyp": h, "var": v,
                "text": "%s ⊢ %d ∣ %s" % (RA.binders_of(head_of(R)), br["M"], concl_of(R))}
    if opname == "ccp_nt_dvd_combine":
        try:
            br = BR.bridge_nt(stmt_full)
        except Exception:                                              # noqa: BLE001
            return None
        return {"m_h": br["m_h"], "M": br["M"], "terms": br["terms"], "const": br["const"],
                "hypvars": [{"var": v, "hyp": h} for (h, _m, v) in br["hyps"]]}
    return None


def stmt_of(R):
    return R.stmt_txt


def head_of(R):
    return stmt_of(R).split("⊢")[0].strip()


def concl_of(R):
    s = stmt_of(R)
    return s.split("⊢")[1].strip() if "⊢" in s else s


_TOK = None


# ---------------------------------------------------------------- plan template
def plan_pattern(cls, h):
    """Registered template: cls -> expected operator order (h = number of hypotheses, used by nt only)."""
    if cls == "ind":
        return ["ccp_ind_recurrence", "ccp_ring_identity"]
    if cls == "ineq":
        return ["ccp_ineq_sos", "ccp_ineq_close"]
    if cls == "nt":
        return ["ccp_nt_dvd_witness"] * h + ["ccp_nt_dvd_combine"]
    raise KeyError(cls)


def n_hyps_of(head):
    return len(re.findall(r"\([^()]*:\s*\d+\s*∣\s*[^()]*\)", head))


# ---------------------------------------------------------------- readouts (judging included)
def eval_item(m, item, n_attempts=1, greedy=True, judge=True):
    cls = item["cls"]
    head, concl = item["head"], item["concl"]
    stmt = "%s ⊢ %s" % (RA.binders_of(head), concl) if "⊢" not in item.get("stmt", "") else \
        item["stmt"]
    if item.get("stmt", "").count("⊢") != 1:
        stmt = "%s ⊢ %s" % (RA.binders_of(head), concl)
    rec = {"cls": cls, "name": item["name"], "stratum": item.get("tag"),
           "n_attempts": n_attempts, "mode": "greedy" if greedy else "p8",
           "certified": False, "attempts": []}
    pat = plan_pattern(cls, n_hyps_of(head))
    rec["expected_ops"] = pat
    att = m.rollout(stmt, cls, n_attempts=n_attempts, greedy=greedy)
    best = None
    for a in att:
        ops = a["ops"]
        plan_ok = (ops == pat) or (cls == "nt" and ops[-1:] == ["ccp_nt_dvd_combine"]
                                   and set(ops[:-1]) == {"ccp_nt_dvd_witness"})
        rec["attempts"].append({"attempt": a["attempt"], "ops": ops,
                                "pred_ops": a.get("pred_ops"), "why": a["why"],
                                "plan_ok": bool(plan_ok),
                                "first_pred_hit": bool(a.get("pred_ops")) and
                                a["pred_ops"][0] == pat[0],
                                "bind_slot_sample": a["binds"][:2]})
        if plan_ok and best is None:
            best = a
    rec["plan_ok"] = bool(best is not None)
    if not (best and judge):
        rec["reason"] = "no_plan" if not best else "no_judge"
        return rec
    # render: this class's template (the proof body is produced by the certificate machine / templates; arguments come from the same source as arm_args)
    ch, proof, tr = RA.ARM[cls](item["name"], head, concl)
    if proof is None:
        rec["reason"] = "render_%s" % tr.get("why")
        return rec
    wd = os.path.join(LEANWD, "rollout")
    os.makedirs(wd, exist_ok=True)
    j = RA.DC.judge_attempt(item["name"], head, concl, "\n".join(proof), wd,
                            "rollout_%s" % cls)
    rec["certified"] = bool(j.get("certified"))
    rec["reasons"] = j.get("reasons")
    rec["lean"] = j.get("lean")
    rec["reason"] = "certified" if rec["certified"] else ",".join(j.get("reasons") or [])
    rec["proof_lines"] = proof
    return rec


def main():
    global _TOK
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dev", default="cuda:2")
    ap.add_argument("--sets", default="synth_held,anchor")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--n-attempts", type=int, default=1)
    ap.add_argument("--p8", action="store_true", help="multi-sample (T=1.0 top_p=0.95, 8 attempts)")
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--jobs", type=int, default=1, help="number of parallel judging threads (GPU forward shared)")
    ap.add_argument("--base", default=None, help="body base (default = st.BASE = SmolLM2-135M)")
    ap.add_argument("--tokenizer", default="models/SmolLM2-135M")
    ap.add_argument("--tag", default="ccpB135")
    a = ap.parse_args()
    from transformers import AutoTokenizer
    _TOK = AutoTokenizer.from_pretrained(a.tokenizer)
    m = Model(a.ckpt, a.dev, base=a.base)
    m.tok = _TOK
    for op, j in OPNAME.items():
        assert j
    items = []
    for s in a.sets.split(","):
        if s == "synth_held":
            for cls in ("ind", "ineq", "nt"):
                items += RA.synth_held_items(cls)
        elif s == "anchor":
            anc = RA.anchor_items()
            for cls in ("ind", "ineq", "nt"):
                items += anc.get(cls, [])
    if a.limit:
        items = items[:a.limit]
    print("[rollout] ckpt=%s sets=%s n_items=%d attempts=%d mode=%s"
          % (a.ckpt, a.sets, len(items), a.n_attempts, "p8" if a.p8 else "greedy"), flush=True)
    recs, t0 = [], time.time()
    n_att = 8 if a.p8 else a.n_attempts
    if a.jobs and a.jobs > 1:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=a.jobs) as ex:
            futs = [ex.submit(eval_item, m, it, n_att, not a.p8, not a.no_judge)
                    for it in items]
            done = 0
            for fu in as_completed(futs):
                recs.append(fu.result())
                done += 1
                if done % 20 == 0:
                    print("[rollout] %d/%d (%.0fs)" % (done, len(items), time.time() - t0),
                          flush=True)
    else:
        for i, it in enumerate(items):
            recs.append(eval_item(m, it, n_attempts=n_att, greedy=not a.p8,
                                  judge=not a.no_judge))
            if (i + 1) % 20 == 0:
                print("[rollout] %d/%d (%.0fs)" % (i + 1, len(items), time.time() - t0),
                      flush=True)
    summ = collections.defaultdict(lambda: {"n": 0, "plan_ok": 0, "certified": 0,
                                           "first_pred_hit": 0,
                                           "why": collections.Counter()})
    for r in recs:
        d = summ[(r["cls"], r["mode"])]
        d["n"] += 1
        d["plan_ok"] += int(r["plan_ok"])
        d["certified"] += int(r["certified"])
        d["first_pred_hit"] += int(bool(r["attempts"]) and
                                   r["attempts"][0].get("first_pred_hit"))
        d["why"][r["reason"]] += 1
    rep = {"schema": "ccp-rollout/1", "argv": " ".join(sys.argv[1:]),
           "ckpt": a.ckpt, "ckpt_md5": _md5(a.ckpt), "sets": a.sets,
           "plan_pattern": {"ind": "ccp_ind_recurrence→ccp_ring_identity",
                            "ineq": "ccp_ineq_sos→ccp_ineq_close",
                            "nt": "witness×h→combine"},
           "arg_disclosure": "算子实参由证书机器从题面导出 (bridge/ops); 模型贡献 = 算子序; "
                             "绑定槽另作诊断读数",
           "summary": {"%s|%s" % k: {"n": v["n"], "plan_ok": v["plan_ok"],
                                     "certified": v["certified"],
                                     "plan_rate": round(v["plan_ok"] / max(1, v["n"]), 4),
                                     "cert_rate": round(v["certified"] / max(1, v["n"]), 4),
                                     "first_pred_hit": v["first_pred_hit"],
                                     "first_pred_rate": round(v["first_pred_hit"] /
                                                              max(1, v["n"]), 4),
                                     "why": dict(v["why"])}
                       for k, v in summ.items()},
           "records": recs, "wall_s": round(time.time() - t0, 1)}
    p = os.path.join(OUT, "ROLLOUT_%s_%s.json" % (a.tag, a.sets.replace(",", "-")))
    with open(p, "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=1, default=str)
    for k, v in rep["summary"].items():
        print("[rollout] %-14s n=%d plan=%d certified=%d why=%s"
              % (k, v["n"], v["plan_ok"], v["certified"],
                 json.dumps(v["why"], ensure_ascii=False)[:120]), flush=True)
    print("[rollout] → %s md5=%s" % (p, _md5(p)))
    print("CCP_ROLLOUT_RC=0")


for _i, _n in enumerate(["ccp_ind_recurrence", "ccp_ring_identity", "ccp_ineq_sos",
                         "ccp_ineq_close", "ccp_nt_dvd_witness", "ccp_nt_dvd_combine",
                         "ccp_nt_dvd_prod_combine"]):
    OPNAME[161 + _i] = _n

if __name__ == "__main__":
    main()

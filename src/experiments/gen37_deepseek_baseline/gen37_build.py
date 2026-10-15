# -*- coding: utf-8 -*-
# gen37_build.py -- gen37 external baseline [1]: exam loading -> DeepSeek-Prover-V1.5 official prompt file (CPU, zero GPU)
#
# Measurement conventions (recorded):
#   * exam = 218 valid deep-exam problems after gen33 C4: scratch/ccp/gen33/out/DEEP_EXAM_SIDECAR.jsonl
#     (fields head=theorem header / concl=conclusion; no gold proof, judging relies on the frozen Lean device).
#   * prompt = DeepSeek-Prover-V1.5 quick_start completion form (zero-shot), header = this distribution's E3
#     official header (xcmp_prompts.OFFICIAL_HEADER_TXT). Same origin as the frozen judge module
#     drct_common.render_lean => prompt statement == judged rendered statement (zero gap on the statement side).
#   * disclosure: the HF quick_start form is "import Mathlib\nimport Aesop\n\nset_option maxHeartbeats 0\n\n
#     open BigOperators Real Nat Topology Rat"; this distribution's miniF2F is statements-only, so the E3
#     adjudication uses the MiniF2F.ProblemImports header (same form as the existing external arm UN-030,
#     cross-arm comparability takes priority).
# Writes only under scratch/ccp/gen37/**.
import argparse
import hashlib
import json
import os
import time

RES = os.environ.get("SCOPE_RESEARCH_ROOT", ".")
G37 = RES + "/scratch/ccp/gen37"
EXAM = RES + "/scratch/ccp/gen33/out/DEEP_EXAM_SIDECAR.jsonl"
MODEL = RES + "/models/DeepSeek-Prover-V1.5-RL"
DRCT = RES + "/scratch/drct"
JUDGE_MD5_EXPECT = "7c31b58da9d7228cee31b9be581fe5dd"

OFFICIAL_HEADER_TXT = "import MiniF2F.ProblemImports\nopen scoped Real Nat Topology Polynomial"
DS_HEADER = ("Complete the following Lean 4 code:\n```lean4\n" + OFFICIAL_HEADER_TXT + "\n\n")
N_EXAM_EXPECT = 218


def md5f(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def app(msg, logp=None):
    line = "[%s] %s" % (time.strftime("%Y-%m-%dT%H:%M:%S%z"), msg)
    print(line, flush=True)
    if logp:
        with open(logp, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def ds_prompt(head, concl):
    """Same form as xcmp_prompts.ds_prompt: header + 'theorem ... concl := by' + newline."""
    return DS_HEADER + "%s %s := by\n" % (head, concl)


def load_exam(path=EXAM):
    rows = []
    with open(path, encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            if line.strip():
                r = json.loads(line)
                r["_ln"] = ln
                rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exam", default=EXAM)
    ap.add_argument("--out", default=G37 + "/out/GEN37_PROMPTS.jsonl")
    ap.add_argument("--log", default=G37 + "/logs/gen37_build.log")
    ap.add_argument("--tok-check", action="store_true",
                    help="load tokenizer on CPU to verify prompt lengths (zero GPU)")
    a = ap.parse_args()

    jm = md5f(DRCT + "/drct_common.py")
    assert jm == JUDGE_MD5_EXPECT, "frozen judge module md5 mismatch: %s" % jm
    rows = load_exam(a.exam)
    assert len(rows) == N_EXAM_EXPECT, "exam rows %d != %d" % (len(rows), N_EXAM_EXPECT)
    names = [r["name"] for r in rows]
    assert len(set(names)) == len(names), "duplicate problem names"
    out = []
    for r in rows:
        head, concl = r["head"], r["concl"]
        assert head.strip().startswith("theorem "), r["name"]
        assert head.strip().endswith(":"), (r["name"], head[-60:])
        assert concl.strip(), r["name"]
        pr = ds_prompt(head, concl)
        assert pr.endswith(":= by\n")
        out.append({"ri": r["ri"], "name": r["name"], "cls": r["cls"],
                    "cls_code": r["cls_code"], "family": r["family"], "np": r["np"],
                    "chain_len": r.get("chain_len"), "n_steps": r.get("n_steps"),
                    "decision_points": r.get("decision_points"),
                    "head": head, "concl": concl, "prompt": pr,
                    "prompt_md5": hashlib.md5(pr.encode("utf-8")).hexdigest()})
    with open(a.out, "w", encoding="utf-8") as f:
        for o in out:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    fam = {}
    for o in out:
        d = fam.setdefault(o["family"], {"n": 0, "cls": o["cls"]})
        d["n"] += 1
    meta = {
        "schema": "gen37-build/1", "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "exam": a.exam, "exam_md5": md5f(a.exam), "n_exam": len(out),
        "exam_lineage_note": "218 valid deep-exam problems after gen33 C4 (same exam and same judge module as gen35 M1_F)",
        "judge_md5": jm, "judge_path": DRCT + "/drct_common.py",
        "model_dir": MODEL, "n_prompts": len(out),
        "prompt_form": "DeepSeek-Prover-V1.5 quick_start completion form (zero-shot)",
        "header": OFFICIAL_HEADER_TXT,
        "header_disclosure": ("E3 official header (local miniF2F statements-only adjudication); the HF original "
                              "form adds import Mathlib/Aesop + set_option maxHeartbeats 0 + open BigOperators "
                              "Real Nat Topology Rat — difference recorded, same form as UN-030"),
        "prompt_min_chars": min(len(o["prompt"]) for o in out),
        "prompt_max_chars": max(len(o["prompt"]) for o in out),
        "by_family": fam,
        "prompts_md5": md5f(a.out),
    }
    if a.tok_check:
        mt = md5f(MODEL + "/tokenizer.json")
        from transformers import AutoTokenizer
        tk = AutoTokenizer.from_pretrained(MODEL)
        lens = [len(tk(o["prompt"]).input_ids) for o in out]
        meta["tok_check"] = {"tokenizer_json_md5": mt, "n": len(lens),
                            "prompt_tokens_min": min(lens), "prompt_tokens_max": max(lens),
                            "prompt_tokens_sum": sum(lens),
                            "ctx_4096_headroom_min": 4096 - max(lens)}
    mj = G37 + "/out/GEN37_BUILD.json"
    open(mj, "w", encoding="utf-8").write(json.dumps(meta, ensure_ascii=False, indent=1))
    app("BUILD ok n=%d prompts_md5=%s judge=%s" % (len(out), meta["prompts_md5"][:8], jm[:8]),
        a.log)
    if a.tok_check:
        app("TOK %s" % json.dumps(meta["tok_check"]), a.log)
    print(json.dumps({"out": a.out, "meta": mj, "n": len(out),
                      "prompts_md5": meta["prompts_md5"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()

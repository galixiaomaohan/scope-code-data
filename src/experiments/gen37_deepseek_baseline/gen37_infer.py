# -*- coding: utf-8 -*-
# gen37_infer.py -- gen37 external baseline [2]: DeepSeek-Prover-V1.5-RL (int8) x 218 deep exam, greedy
#
# Convention reuse (recorded): xcmp_player.py (gen9-era; same-origin external arm as UN-030)
#   * loading: AutoModelForCausalLM.from_pretrained(load_in_8bit=True, device_map={"": dev});
#           fallback BitsAndBytesConfig(load_in_8bit) or --cpu-offload (device_map=auto, slow, no quantization).
#   * sampling: greedy (do_sample=False); max_new = min(--max-new, ctx - in_tok - 8).
#   * statement-boundary truncation: stop_strings (same table as xcmp); a post-hoc truncation at the
#     same boundary keeps equivalence (disclosed).
#   * metering: per-problem in_tokens/out_tokens/wall_s; aggregate tok/s (generated tokens / decode wall).
# Discipline: writes only under scratch/ccp/gen37/**; all three GPUs allowed (change #104: one job per GPU;
#       the 95/100 reserved-region regime ended); pre-launch gate = frozen judge md5 + our Lean jobs < 5 +
#       target GPU free >= --min-free; zero sudo / zero /tmp.
import argparse
import hashlib
import json
import os
import subprocess
import sys
import threading
import time

RES = os.environ.get("SCOPE_RESEARCH_ROOT", ".")
G37 = RES + "/scratch/ccp/gen37"
DRCT = RES + "/scratch/drct"
MODEL = RES + "/models/DeepSeek-Prover-V1.5-RL"
JUDGE_MD5_EXPECT = "7c31b58da9d7228cee31b9be581fe5dd"
STOP_STRINGS = ["\ntheorem ", "\nTheorem", "\n### ", "\nComplete the following"]
TAG = "g37_dsrl_int8_greedy"
FORBIDDEN_DEVS = ()  # change #104: all three GPUs allowed; the real gate is the free>=min-free assert (unchanged)


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


def n_lean_jobs():
    """Our Lean job count (pre-launch gate: <5). Same form as gen35_common.n_lean_jobs."""
    try:
        ps = subprocess.run(["bash", "-lc",
                             "ps -eo pid,cmd --no-headers | grep -F 'lake env lean' "
                             "| grep -Ev 'grep' | grep -E 'tcaboost|\\.elan'"],
                            capture_output=True, text=True).stdout
    except Exception:                                                # noqa: BLE001
        return -1
    return len([l for l in ps.split("\n") if l.strip()])


def gpu_q(idx, field="memory.free"):
    q = subprocess.run(["nvidia-smi", "--query-gpu=" + field, "--format=csv,noheader,nounits",
                        "-i", str(idx)], capture_output=True, text=True, timeout=10)
    return int(q.stdout.strip().split("\n")[0])


class VramSampler(threading.Thread):
    def __init__(self, idx):
        super().__init__(daemon=True)
        self.idx, self.peak, self.stop_flag = idx, 0, False

    def run(self):
        while not self.stop_flag:
            try:
                self.peak = max(self.peak, gpu_q(self.idx, "memory.used"))
            except Exception:                                        # noqa: BLE001
                pass
            time.sleep(1.0)


def load_model(path, dev, logp, cpu_offload=False):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tk = AutoTokenizer.from_pretrained(path)
    if tk.pad_token is None:
        tk.pad_token = tk.eos_token
    if cpu_offload:
        model = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype="bfloat16", device_map="auto", max_memory={0: "0GiB", 1: "0GiB",
                                                                        2: "7GiB", "cpu": "120GiB"},
            trust_remote_code=True, offload_folder=G37 + "/cache/offload")
        return tk, model, "cpu_offload(device_map=auto, bf16, no quantization)"
    try:
        model = AutoModelForCausalLM.from_pretrained(
            path, load_in_8bit=True, device_map={"": dev}, trust_remote_code=True)
        mode = "load_in_8bit=True (same form as UN-030)"
    except Exception as e:                                           # noqa: BLE001
        app("legacy int8 entry failed (%s: %s) -> BitsAndBytesConfig" % (type(e).__name__, e), logp)
        from transformers import BitsAndBytesConfig
        model = AutoModelForCausalLM.from_pretrained(
            path, quantization_config=BitsAndBytesConfig(load_in_8bit=True),
            device_map={"": dev}, trust_remote_code=True)
        mode = "BitsAndBytesConfig(load_in_8bit)"
    model.eval()
    return tk, model, mode


def truncate_at_stop(text):
    cut = len(text)
    for s in STOP_STRINGS:
        i = text.find(s)
        if i != -1:
            cut = min(cut, i)
    return text[:cut], (cut < len(text))


def gen_once(tk, model, prompt, dev, max_new):
    import torch
    enc = tk(prompt, return_tensors="pt", return_attention_mask=True).to(dev)
    in_tok = int(enc.input_ids.shape[1])
    kw = dict(max_new_tokens=max_new, do_sample=False, pad_token_id=tk.pad_token_id,
              return_dict_in_generate=True)
    t0 = time.time()
    try:
        with torch.no_grad():
            out = model.generate(**enc, stop_strings=STOP_STRINGS, tokenizer=tk, **kw)
    except TypeError:
        with torch.no_grad():
            out = model.generate(**enc, **kw)
    wall = time.time() - t0
    gen_ids = out.sequences[:, in_tok:]
    text = tk.decode(gen_ids[0], skip_special_tokens=True)
    return {"in_tokens": in_tok, "out_tokens": int(gen_ids.shape[1]), "text": text,
            "wall_s": round(wall, 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default=G37 + "/out/GEN37_PROMPTS.jsonl")
    ap.add_argument("--out", default=G37 + "/out/GEN37_DSRL")
    ap.add_argument("--dev", default="cuda:2")
    ap.add_argument("--max-new", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--min-free", type=int, default=11400)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dry", action="store_true", help="gate only + print plan (zero GPU)")
    ap.add_argument("--cpu-offload", action="store_true")
    ap.add_argument("--log", default=G37 + "/logs/gen37_infer.log")
    a = ap.parse_args()

    idx = a.dev.split(":")[-1]
    if idx in FORBIDDEN_DEVS:
        raise SystemExit("structural refusal: GPU%s is in a reserved region (dev=%s)" % (idx, a.dev))
    jm = md5f(DRCT + "/drct_common.py")
    assert jm == JUDGE_MD5_EXPECT, "frozen judge module md5 mismatch: %s" % jm
    nl = n_lean_jobs()
    app("GATE judge=%s | lean_ours=%d (<5) | dev=%s free=%dMiB (>=%d)"
        % (jm[:8], nl, a.dev, gpu_q(idx), a.min_free), a.log)
    assert nl < 5, "pre-launch gate: our Lean jobs %d >= 5" % nl
    with open(a.prompts, encoding="utf-8") as f:
        prows = [json.loads(l) for l in f if l.strip()]
    if a.limit:
        prows = prows[:a.limit]
    att = a.out + "_attempts.jsonl"
    done = set()
    if a.resume and os.path.exists(att):
        with open(att, encoding="utf-8") as f:
            for l in f:
                if l.strip():
                    done.add(json.loads(l)["name"])
    todo = [p for p in prows if p["name"] not in done]
    app("PLAN n_prompts=%d done=%d todo=%d max_new=%d greedy=1 stop=%r"
        % (len(prows), len(done), len(todo), a.max_new, STOP_STRINGS), a.log)
    if a.dry:
        app("DRY ok (zero GPU, model not loaded); first 300 chars of the first prompt:\n%s"
            % prows[0]["prompt"][:300], a.log)
        return
    free = gpu_q(idx)
    assert free >= a.min_free, "GPU gate not passed: free=%dMiB < %d" % (free, a.min_free)
    t_import = time.time()
    import torch
    app("torch %s cuda=%s avail=%s" % (torch.__version__, torch.version.cuda,
                                       torch.cuda.is_available()), a.log)
    t_load = time.time()
    tk, model, lmode = load_model(MODEL, a.dev, a.log, a.cpu_offload)
    app("model loaded %.0fs | %s | dev=%s | peak_alloc=%dMiB"
        % (time.time() - t_load, lmode, a.dev,
           int(torch.cuda.max_memory_allocated() / 1048576)), a.log)
    vs = VramSampler(int(idx))
    vs.start()
    ctx = int(getattr(model.config, "max_position_embeddings", 4096) or 4096)
    fout = open(att, "a", encoding="utf-8")
    t0 = time.time()
    tot_in = tot_out = 0
    for i, p in enumerate(todo):
        max_new = min(a.max_new, max(64, ctx - len(tk(p["prompt"]).input_ids) - 8))
        r = gen_once(tk, model, p["prompt"], a.dev, max_new)
        text, stop_hit = truncate_at_stop(r["text"])
        rec = {"ri": p["ri"], "name": p["name"], "attempt": 0, "cls": p["cls"],
               "cls_code": p["cls_code"], "family": p["family"],
               "in_tokens": r["in_tokens"], "out_tokens": r["out_tokens"],
               "wall_s": r["wall_s"], "max_new": max_new, "stop_hit": bool(stop_hit),
               "text": text, "text_full_len": len(r["text"]),
               "text_md5": hashlib.md5(r["text"].encode("utf-8")).hexdigest(),
               "prompt_md5": p["prompt_md5"], "tag": TAG,
               "model": "DeepSeek-Prover-V1.5-RL", "quant": lmode, "dev": a.dev}
        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fout.flush()
        tot_in += r["in_tokens"]
        tot_out += r["out_tokens"]
        if (i + 1) % 10 == 0 or i == len(todo) - 1:
            el = time.time() - t0
            app("RUN %d/%d | in=%d out=%d | %.0fs | %.2f tok/s(out) | last wall=%.1fs"
                % (i + 1, len(todo), tot_in, tot_out, el, tot_out / max(0.1, el),
                   r["wall_s"]), a.log)
    fout.close()
    vs.stop_flag = True
    time.sleep(1.2)
    wall_total = round(time.time() - t0, 1)
    allrows = []
    with open(att, encoding="utf-8") as f:
        for l in f:
            if l.strip():
                allrows.append(json.loads(l))
    si = sum(r["in_tokens"] for r in allrows)
    so = sum(r["out_tokens"] for r in allrows)
    sw = sum(r["wall_s"] for r in allrows)
    n = len(allrows)
    res = {
        "schema": "gen37-infer/1", "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "player": "gen37", "arm": TAG, "model": "models/DeepSeek-Prover-V1.5-RL",
        "model_dir": MODEL, "model_load_mode": lmode, "n_problems": n,
        "ctx": ctx, "dev": a.dev, "sampling": {"greedy": True, "do_sample": False,
                                              "max_new_tokens_cap": a.max_new},
        "T1_tokens": {"input_sum": si, "output_sum": so,
                      "input_per_q": round(si / max(1, n), 2),
                      "output_per_q": round(so / max(1, n), 2)},
        "T2_wall": {"decode_wall_sum_s": round(sw, 1),
                    "batch_wall_s": wall_total,
                    "secs_per_q_decode": round(sw / max(1, n), 2),
                    "secs_per_q_batch": round(wall_total / max(1, n), 2),
                    "toks_per_s_output": round(so / max(0.1, sw), 2),
                    "toks_per_s_batch": round(so / max(0.1, wall_total), 2),
                    "problems_per_h": round(n / max(0.1, wall_total / 3600), 2)},
        "T3_vram": {"nvidia_smi_peak_mib": vs.peak,
                    "device": "RTX 4070 SUPER 12282MiB (GPU%s)" % idx},
        "attempts_file": att,
        "judge_md5_at_launch": jm,
        "stop_strings": STOP_STRINGS,
        "resource_note": ("decode_wall_sum = sum of per-problem generate wall clock (pure model time); "
                          "batch_wall includes tokenize/write; the primary tok/s reading is output/decode_wall"),
    }
    rp = a.out + "_resource.json"
    open(rp, "w", encoding="utf-8").write(json.dumps(res, ensure_ascii=False, indent=1))
    app("DONE %s | n=%d wall=%.0fs in=%d out=%d %.2f tok/s smi_peak=%dMiB"
        % (rp, n, wall_total, si, so, so / max(0.1, sw), vs.peak), a.log)
    print(json.dumps(res, ensure_ascii=False))


if __name__ == "__main__":
    main()

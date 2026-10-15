
---

## 5. 判卷收尾（终表交付 + certified 明细）— judged under concurrency guard

**判卷纪律实测与处置（红线合规登记）**：首次启动判卷时实测**全局** `lake env lean` = **7 进程**，经 `ps` 归因**全部属并行批 gen29**（`gen29/g29_prescreen.txt` 判 `bridge_g00X_main_*.lean`，约 4 作业），非本批。为守"并发≤5"，本批采取两项措施并**让位**：
1. **串行化**：`gen26_eval.judge_texts` 加锁，使本批 Lean 调用**同时只有 1 个作业**（md5 由 `9fe8258127c7b58d0cbabe69ee55b7fc` 前的旧版 → 现版已登记）；
2. **守门启动**：`logs/judge_guard.sh`（PID 417144，日志 `logs/L0_PROBE_JUDGE_GUARD.log`）轮询全局 `lake env lean` 计数，**仅在 ≤3 时**才启动串行判卷（本批 +1 ⇒ 全局 ≤4 ≤ 5）；等待上限 480 次 × 30s = 4h，超时写 `GUARD_TIMEOUT_NO_SLOT`。

**终表（五列 × 三臂；certified = 待判卷回填）**

| 臂 | seg | n | **certified** | 首算子 | 整链 | 槽列 | 一致性 | 渲染 tok |
|---|---|---|---|---|---|---|---|---|
| **hybrid**（probe ckpt） | exam_v1 | 30 | **待回填** | **结构不可判·词表零交** | **结构不可判·词表零交** | 不可评 | 1.00 | 48.0 |
| **naked**（零训练） | exam_v1 | 30 | **待回填** | 不适用 | 不适用 | — | 0.00（空列口径） | 160.0 |
| **rule**（零训练，装置战术行） | exam_v1 | 30 | **待回填** | 不适用 | 不适用 | — | 0.00（空列口径） | 25.0 |
| hybrid | exam_zhenti | 3 | **待回填** | 结构不可判·词表零交 | 结构不可判·词表零交 | 不可评 | 0.50 | 94.0 |

> 链两栏"结构不可判·词表零交" = R1-甲实测（训练 6 `ccp_*` ∩ 考卷 11 `bh_*` = ∅，可评 0/1083），**非能力读数**。

**AUD-1 对照（渲染列，硬性）**：模型 **1.000**（2282 tok，训练内切片）vs **零参数逐类逐位置众数基线 0.913234**（2084/2282）⇒ **+8.677 pp ≈ +198 tok**；**本版 held = 训练内温度计，非泛化读数**（V2F 全为 train 行、name-hash 切分在本语料命中 0 行）。链列须标注**链种类数 = 3（逐类恒定）⇒ 等价 3 类分类**。

> **【头条·硬性披露（原文照登）】L0 探路版·语料覆盖 11.7%（1097/9366）·读数定性 = 管线 shaken + 趋势信号·不作 M1_A 主读数**

**certified 可复算命令（冻结件 `drct_common` 7c31b58d 零改；nice -n 19；串行 1 作业）**：
```bash
P=/home/data/ZHC/tcaboost/venv/bin/python; export PYTHONPATH=/home/data/ZHC/tcaboost/superoperator
cd /home/data/ZHC/tcaboost/research
nice -n 19 $P scratch/ccp/gen26/gen26_eval.py --arm hybrid --seg exam_v1 --args-mode bridge \
  --ckpt scratch/ccp/gen26/ckpt/hybrid_probe_l0_f935e143_3000.pt --n 30 --judge 1 \
  --out scratch/ccp/gen26/out/L0_PROBE_JUDGE_hybrid_exam_v1.json
nice -n 19 $P scratch/ccp/gen26/gen26_eval.py --arm naked --seg exam_v1 --n 30 --judge 1 \
  --out scratch/ccp/gen26/out/L0_PROBE_JUDGE_naked_exam_v1.json
nice -n 19 $P scratch/ccp/gen26/gen26_eval.py --arm rule  --seg exam_v1 --n 30 --judge 1 \
  --out scratch/ccp/gen26/out/L0_PROBE_JUDGE_rule_exam_v1.json
nice -n 19 $P scratch/ccp/gen26/gen26_eval.py --arm hybrid --seg exam_zhenti --args-mode bridge \
  --ckpt scratch/ccp/gen26/ckpt/hybrid_probe_l0_f935e143_3000.pt --n 3 --judge 1 \
  --out scratch/ccp/gen26/out/L0_PROBE_JUDGE_hybrid_zhenti.json
```
判卷件（新件，命名 `L0_PROBE_JUDGE_*`）：`out/L0_PROBE_JUDGE_{hybrid,naked,rule}_exam_v1.json` · `out/L0_PROBE_JUDGE_hybrid_zhenti.json` · 日志 `logs/L0_PROBE_JUDGE.log`（含逐题 `reasons`/`n1n2`/`axioms`/`lean_file`）· 守门日志 `logs/L0_PROBE_JUDGE_GUARD.log`。
判卷工作目录（Lean 文件落盘）：`scratch/ccp/gen26/lean/judge_{hybrid,naked,rule}/`。
**红线**：判卷期间零改冻结件；certified 结果**先交主代亲验**（抽题复算），本版定性（管线 shaken+趋势）不变，**不作 M1_A 结论**。

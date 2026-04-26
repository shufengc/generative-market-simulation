# Project Status Report

**Date:** April 23, 2026
**Repo:** `shufengc/generative-market-simulation`

---

## 1. Team Members and Branch Status

### Shufeng Chen — `shufeng` branch (merged to main)

**Contributions:** DDPM baseline, DDPM improved (v-prediction, Student-t forward process), 6-phase ablation study, integration, demo, README.


| Phase       | What Was Done                                                                            | Key Finding                                          |
| ----------- | ---------------------------------------------------------------------------------------- | ---------------------------------------------------- |
| Phase 1     | 7-variant ablation (baseline, DiT, vpred, sigmoid, crossattn, etc.), 3 seeds, 200 epochs | Sigmoid wins weighted composite; vpred wins MMD/W1   |
| Phase 2     | Temporal attention, heterogeneous noise, auxiliary SF loss                               | aux_sf_loss best; sigmoid suppresses vpred fat tails |
| Phase 3     | 128ch / 400ep fair comparison vs NormFlow                                                | vpred without sigmoid is the correct DDPM config     |
| Phase 4     | 64ch parameter-fair test (2.3M params)                                                   | vpred 4.3/6 vs baseline 1.7/6 at lower capacity      |
| Phase 5     | vpred + Student-t (df=5), wavelet, ACF guidance                                          | vpred + Student-t matches NormFlow at 5/6 SF         |
| **Phase 6** | **Re-baseline under unified settings** (stride=1, new eval, 3 seeds)                     | **vpred 5.0/6, Student-t 5.0/6 with MMD 0.006**      |


**Status:** Phase 6 results merged to main. All DDPM work uses unified settings.

---

### Yixuan Ye — `yixuan` branch + `experiment/timegan-3seed-colab` + `archive/timegan-experiments`

**Contributions:** Evaluation framework rewrite (`stylized_facts.py`), data pipeline fixes (stride=1, Yahoo conditioning), TimeGAN tuning, TimeGAN 3-seed rebaseline, regime conditioning (FRED + GJR-GARCH for SF3).


| Work Item                                                               | Branch                           | Status                                 |
| ----------------------------------------------------------------------- | -------------------------------- | -------------------------------------- |
| New stylized facts eval (Hill, GARCH gamma, Hurst R/S, eigenvalue, MAA) | `yixuan` (merged to main)        | Done                                   |
| Data pipeline stride=1 fix                                              | `yixuan` (merged to main)        | Done                                   |
| TimeGAN tuning (BCE+GP fix, CuDNN workaround, G/D balancing)            | `yixuan`                         | Done but not merged                    |
| Aux-loss TimeGAN experiments (8 iterations)                             | `archive/timegan-experiments`    | Archived — no improvement over vanilla |
| **TimeGAN 3-seed rebaseline (unified settings)**                        | `experiment/timegan-3seed-colab` | **Done today (April 23)**              |


**TimeGAN 3-seed results (unified settings, stride=1, 100 epochs, seeds 42/123/456):**


| Metric               | Result                          |
| -------------------- | ------------------------------- |
| Stylized Facts       | **4/6** (all 3 seeds identical) |
| MMD                  | 0.110 +/- 0.025                 |
| Discriminative Score | 1.00                            |


TimeGAN passes fat tails, vol clustering, leverage, and long memory, but fails cross-asset correlation and no-autocorrelation. Discriminative score of 1.00 means individual samples are easily distinguishable from real data despite matching structural statistics.

**Open PR:** [experiment/timegan-3seed-colab](https://github.com/shufengc/generative-market-simulation/pull/new/experiment/timegan-3seed-colab) — needs review and merge.

---

### Yizheng Lin — `yizheng` branch

**Contributions:** VAE improvements (bidirectional GRU, Student-t output, free-bits + cyclical KL, auxiliary moment-matching, low-rank market factor emission), data pipeline (`run_pipeline.py` fixes, FRED API key), unified stride experiment runner.


| Work Item                                                                       | Status         |
| ------------------------------------------------------------------------------- | -------------- |
| VAE improved (conditional decoding, teacher-forcing schedule, Student-t output) | Done on branch |
| VAE pipeline runner + cross-model analysis outputs                              | Done on branch |
| Unified stride experiment runner                                                | Done on branch |
| PR #8 merged to main (basic VAE + pipeline fixes)                               | Merged         |


**VAE improved results (from Yizheng's `vae_cross_model` runner):**


| Model        | SF  | MMD    | Disc |
| ------------ | --- | ------ | ---- |
| VAE improved | 5/6 | 0.0003 | 1.00 |
| VAE original | 4/6 | 0.42   | 1.00 |


The improved VAE achieves remarkable MMD (0.0003), but discriminative score remains 1.00. These results need verification: it is unclear whether they used the unified `stylized_facts.py` from main or a separate evaluation path. The 3 newest commits on `yizheng` have not been merged to main.

---

### Yuxia Meng (marsdoctorm) — `yuxia/normflow-cross-model` + `yuxia/ddpm-improved-enhancement`

**Contributions:** NormFlow (enhanced RealNVP with ActNorm, residual coupling, multi-scale), cross-model analysis framework, DDPM improved enhancements (Min-SNR weighting, warmup LR, best model restore).


| Work Item                                               | Branch                                             | Status         |
| ------------------------------------------------------- | -------------------------------------------------- | -------------- |
| Enhanced NormFlow + cross-model analysis                | `yuxia/normflow-cross-model` (merged via PR #1/#2) | Done           |
| DDPM Min-SNR weighting + warmup LR + best-model restore | `yuxia/ddpm-improved-enhancement`                  | **Not merged** |


**NormFlow results (from cross-model analysis, April 16):**


| Metric               | Result |
| -------------------- | ------ |
| Stylized Facts       | 5/6    |
| MMD                  | 0.052  |
| Discriminative Score | 0.54   |


**WARNING:** These NormFlow results were generated with `--quick` mode (20 epochs) and the **old evaluation framework** (pre-unification). The NormFlow analysis file itself states: *"Results are from --quick mode (20 epochs)."* These numbers are **not directly comparable** to DDPM Phase 6 or TimeGAN 3-seed results. NormFlow needs a full re-run under unified settings.

Additionally, on `origin/yixuan`, Yixuan stripped the NormFlow code back to basic RealNVP (removing ActNorm, residual coupling, multi-scale). The version of NormFlow on main still has Yuxia's enhanced architecture, but the `yixuan` branch version is simpler. This needs to be reconciled before the final comparison.

---

### Kevin Sun — No branch, no commits

**Kevin has not pushed any code to the repository.** There are zero commits from any author matching "Kevin" or "Sun" across all branches. The GARCH model (`src/models/garch.py`) exists in the codebase but was created as part of the initial scaffold, not by Kevin.

The GARCH results shown in the cross-model comparison (4/6 SF, MMD 0.28) come from Yuxia's cross-model analysis run, using the pre-existing `garch.py` code.

**GARCH is the TA-designated baseline.** Per TA Gaurav's feedback, GARCH should serve as the baseline for comparison. Someone needs to re-run GARCH under unified settings (stride=1, new eval, 3 seeds) if Kevin does not do it.

---

## 2. Unified Settings Audit

The team agreed on April 15 to unify all experiments under:

- **stride=1** (~5,293 training windows)
- **Yixuan's new `stylized_facts.py`** (Hill, GARCH gamma, Hurst R/S, eigenvalue, MAA+Ljung-Box)
- **3 seeds** (42, 123, 456) for statistical reliability

### Which Results Are Trustworthy?


| Model                           | Stride=1 | New Eval          | 3-Seed           | Verdict                |
| ------------------------------- | -------- | ----------------- | ---------------- | ---------------------- |
| DDPM vpred (Phase 6)            | Yes      | Yes               | Yes              | **Trustworthy**        |
| DDPM vpred+Student-t (Phase 6)  | Yes      | Yes               | Yes              | **Trustworthy**        |
| TimeGAN 3-seed (April 23)       | Yes      | Yes               | Yes              | **Trustworthy**        |
| VAE improved (Yizheng)          | Likely   | Likely            | No (1 run)       | **Needs verification** |
| NormFlow (Yuxia, April 16)      | Likely   | **No (old eval)** | No (1 run, 20ep) | **Not trustworthy**    |
| GARCH (from cross-model)        | Likely   | **No (old eval)** | No (1 run)       | **Not trustworthy**    |
| VAE original (from cross-model) | Likely   | **No (old eval)** | No (1 run)       | **Not trustworthy**    |


### Verified Cross-Model Comparison (Unified Settings Only)


| Model                    | SF      | MMD       | Disc | Seeds | Notes                 |
| ------------------------ | ------- | --------- | ---- | ----- | --------------------- |
| **DDPM vpred+Student-t** | **5/6** | **0.006** | 0.85 | 3     | Best overall MMD + SF |
| **DDPM vpred**           | **5/6** | 0.037     | 0.93 | 3     | Strong baseline       |
| **TimeGAN (WGAN-GP)**    | 4/6     | 0.110     | 1.00 | 3     | Structural baseline   |


All three models fail **SF6 (no raw autocorrelation)**. DDPM also fails SF6; TimeGAN additionally fails SF5 (cross-asset correlation).

---

## 3. TA Guidance (from Gaurav)

> *"Maybe prioritize DDPM as your primary model and GARCH as the baseline, then add TimeGAN/VAE/Flows as time allows. That way you have a strong core result no matter what. Don't skip the leverage effect test during validation — it's the one most models struggle with and it'll make your results more convincing."*

**Implications:**

- DDPM is the headline model. Our Phase 6 results (5/6 SF, MMD 0.006 with Student-t) are strong.
- GARCH must be properly re-run as the designated baseline for the paper.
- TimeGAN 3-seed results are now available as a comparison point.
- The leverage effect (SF3) passes for DDPM — this is good and should be highlighted.
- SF6 (no autocorrelation) is the remaining failure for DDPM; this should be discussed as a known limitation.

---

## 4. What Has Been Accomplished

### Done

- DDPM baseline and improved model with v-prediction, Student-t noise, classifier-free guidance
- 6-phase ablation study for DDPM, culminating in Phase 6 re-baseline
- Unified evaluation framework (6 stylized facts with rigorous statistical tests)
- Data pipeline with stride=1 windowing, Yahoo Finance + FRED conditioning
- TimeGAN implementation and 3-seed rebaseline under unified settings
- NormFlow (enhanced RealNVP) implementation and cross-model analysis framework
- VAE improved with Student-t output, moment-matching, and low-rank market factors
- GARCH baseline model (DCC-GARCH with correlated innovations)
- Interactive demo (FastAPI + Chart.js)
- Cross-model analysis tooling (bootstrap CI, pairwise significance, radar charts)

### Not Done

- NormFlow re-run under unified settings (stride=1, new eval, 3 seeds, full epochs)
- GARCH re-run under unified settings (3 seeds)
- VAE improved verification under unified settings (3 seeds, confirm eval framework)
- DDPM SF6 investigation (autocorrelation removal)
- Merge Yuxia's DDPM enhancements (Min-SNR, warmup LR) and test impact
- Merge Yizheng's latest VAE commits (3 new commits not on main)
- Final paper / report writing
- Kevin's contributions (GARCH baseline, visualization)

---

## 5. DDPM 优化方案（Phase 7 计划）

### 当前状态

DDPM vpred+Student-t 在统一设置下达到 **5/6 SF**，唯一失败的是 **SF6（无自相关性）**。

SF6 的判定条件是两个都必须满足：

1. MAA (lags 1-20 的平均绝对自相关) < 0.05
2. Ljung-Box p_min > 0.05

当前结果：MAA 在 0.018-0.029 之间（看起来通过了 MAA），但 Ljung-Box p_min = 0.0（强烈拒绝"无自相关"假设），说明合成数据在短 lag 上存在系统性的自相关。

### 优化方向一：训练改进（降低学到的自相关）


| 方案                           | 原理                                                                                       | 负责人     | 优先级                                        |
| ---------------------------- | ---------------------------------------------------------------------------------------- | ------- | ------------------------------------------ |
| **Min-SNR-gamma 加权**         | 平衡不同时间步的训练权重，避免高 SNR 步骤主导训练（Hang et al. 2023）。已有代码在 `yuxia/ddpm-improved-enhancement` 分支 | Yuxia   | 高                                          |
| **Warmup LR + cosine decay** | 前 10 个 epoch 线性 warmup，避免早期训练不稳定。同样已在 Yuxia 分支                                           | Yuxia   | 高                                          |
| **ACF 正则化损失**                | 在 `p_losses` 中加入原始 returns 的 ACF 惩罚项（不是绝对值 ACF）。当前 `use_aux_sf_loss` 只惩罚 kurtosis 和      | returns | 的 ACF，没有直接针对原始 returns 的 lag-1 至 lag-5 自相关 |
| **更长训练 (800 epochs)**        | Phase 3 已表明 400→800 epochs 对 aux_sf 配置有改善。更长训练可能让模型更好地消除自相关                              | 任意      | 中                                          |


### 优化方向二：采样改进（生成时去自相关）


| 方案                  | 原理                                                                          | 负责人     | 优先级              |
| ------------------- | --------------------------------------------------------------------------- | ------- | ---------------- |
| **DDIM eta 调参**     | 当前 DDIM 是完全确定性的（eta=0）。增加少量随机性（eta=0.1-0.3）可能打破采样过程中的自相关结构                  | Shufeng | 高                |
| **更多 DDIM 步数**      | 当前 50 步。增加到 100-200 步可能改善生成质量                                               | 任意      | 低                |
| **ACF guidance 调优** | `use_acf_guidance` 已存在但在 Phase 6 实验中未启用。可以改为针对原始 returns 的 ACF guidance（不仅是 | returns | ），在采样过程中直接引导去自相关 |


### 优化方向三：后处理（生成后去自相关）


| 方案                     | 原理                                                                                      | 负责人 | 优先级 |
| ---------------------- | --------------------------------------------------------------------------------------- | --- | --- |
| **AR(1) 残差校正**         | 对生成的 returns 做 AR(1) 拟合，减去自相关成分。简单且不影响其他 SF                                             | 任意  | 中   |
| **时间维度随机 shuffle（轻量）** | 对每条生成路径的 returns 做微小的随机扰动，打破短 lag 自相关。但需要小心不破坏 vol clustering (SF2) 和 long memory (SF4) | 任意  | 低   |


### 建议执行顺序

```
第一轮（快速实验，2-3小时）:
  1. 合并 Yuxia 的 Min-SNR + warmup LR → 跑一次 vpred+Student-t 400ep 看效果
  2. 在 p_losses 中加入 raw returns ACF 正则化 → 跑一次看 MAA 和 Ljung-Box 变化
  3. 试不同 DDIM eta (0.1, 0.2, 0.3) → 不用重新训练，直接用已有 checkpoint 生成

第二轮（如果第一轮不够）:
  4. 启用并调优 ACF guidance（修改 _acf_guidance_step 针对 raw returns）
  5. 尝试 800 epoch 训练
  6. AR(1) 后处理

目标: MAA < 0.05 且 Ljung-Box p > 0.05 → 达到 6/6 SF
```

---

## 6. 各组员进度同步（中文）

### Shufeng

**已完成：**

- DDPM baseline 和 improved 模型（v-prediction、Student-t 前向过程、classifier-free guidance）
- 6 阶段消融实验（Phase 1-6），Phase 6 在统一设置下得到 5/6 SF、MMD 0.006
- README 更新（团队角色、引用文献），已合并到 main
- 项目集成、Demo（FastAPI + Chart.js）

**待做：**

- DDPM Phase 7 优化（SF6 自相关问题）
- 合并 Yuxia 的 Min-SNR 代码并测试
- 最终论文撰写

---

### 叶逸轩（Yixuan）

**已完成：**

- 评估框架重写：`stylized_facts.py`（Hill 估计量、GARCH gamma、Hurst R/S、特征值、MAA+Ljung-Box）
- 数据管线修复（stride=1、Yahoo conditioning）
- TimeGAN 调优（BCE+GP 修复、CuDNN workaround、G/D 平衡）
- Aux-loss TimeGAN 8 轮实验（已归档，效果不如原版）
- **TimeGAN 3-seed 重跑（4月23日完成）：4/6 SF、MMD 0.11、Disc 1.00**

**待做：**

- `experiment/timegan-3seed-colab` PR 合并
- 协助 NormFlow 在统一设置下重跑（stride=1、新 eval、400 epochs、3 seeds）
- 确认 yixuan 分支上简化的 NormFlow 与 main 上 Yuxia 增强版的取舍

---

### Yizheng

**已完成：**

- VAE 改进（双向 GRU、Student-t 输出、free-bits + cyclical KL、moment-matching 正则、low-rank market factor emission）
- 数据管线修复（`run_pipeline.py`、FRED API key）
- VAE pipeline runner + cross-model analysis 输出
- PR #8 合并到 main

**VAE improved 结果：5/6 SF、MMD 0.0003、Disc 1.00**（单次运行，待 3-seed 验证）

**待做：**

- 确认 VAE improved 是否使用了统一的 `stylized_facts.py`
- 补 3-seed 验证运行
- 将 yizheng 分支最新 3 个 commit（conditional decoding、teacher-forcing、stride runner）开 PR 合并到 main
- 如 Kevin 无法完成，协助跑 GARCH 3-seed

---

### Yuxia

**已完成：**

- NormFlow（增强 RealNVP：ActNorm、residual coupling、multi-scale）实现
- Cross-model analysis 框架（bootstrap CI、pairwise significance、radar chart）
- DDPM 训练增强（Min-SNR 加权、warmup LR、best model restore）在 `yuxia/ddpm-improved-enhancement`

**注意：** NormFlow 当前结果（5/6 SF、MMD 0.052、Disc 0.54）是用 `--quick`（20 epochs）+ 旧 eval 跑的，**不可直接与 Phase 6 比较**

**待做：**

- NormFlow 在统一设置下重跑（stride=1、新 eval、400 epochs、3 seeds）
- `yuxia/ddpm-improved-enhancement` 合并到 main 并验证效果
- 协助 DDPM Phase 7 优化（Min-SNR 部分）

---

### Kevin

**当前状态：无任何代码提交。** 仓库中没有任何来自 Kevin 的 commit。

GARCH 模型（`src/models/garch.py`）已存在于代码库中，但由初始 scaffold 创建。GARCH 是 **TA 指定的 baseline**，必须在统一设置下（stride=1、新 eval、seeds 42/123/456）完成 3-seed 运行。

**待做：**

- GARCH 3-seed 重跑并提交结果
- 可视化工作（proposal 中分配的任务）

---

## 7. 整体优先级排序


| 优先级 | 任务                              | 负责人                 | 截止          |
| --- | ------------------------------- | ------------------- | ----------- |
| P0  | DDPM Phase 7 优化（SF6）            | Shufeng + Yuxia     | 尽快          |
| P0  | GARCH 3-seed 重跑（TA 指定 baseline） | Kevin（或 Yizheng 兜底） | 尽快          |
| P1  | NormFlow 统一设置重跑                 | Yuxia + Yixuan      | 本周内         |
| P1  | VAE improved 3-seed 验证          | Yizheng             | 本周内         |
| P1  | TimeGAN PR 合并                   | Shufeng review      | 今天          |
| P2  | 论文撰写                            | 全组                  | 根据 deadline |


---

## 8. Recommended Next Steps (English Summary)

### Priority 1: DDPM Refinement — Target 6/6 SF

See Section 5 above for the detailed Phase 7 plan. Key actions:

1. Merge Yuxia's Min-SNR + warmup LR from `yuxia/ddpm-improved-enhancement`
2. Add raw-returns ACF regularization loss in `p_losses`
3. Tune DDIM eta (0.1-0.3) at inference time
4. Highlight SF3 (leverage effect) in the paper — TA specifically called this out

### Priority 2: Baseline Re-runs

- GARCH 3-seed under unified settings (Kevin's responsibility, Yizheng backup)
- NormFlow full re-run (stride=1, new eval, 400 epochs, 3 seeds)

### Priority 3: VAE Verification

- Confirm VAE improved results used unified eval, run 3 seeds
- Merge Yizheng's 3 pending commits

### Priority 4: Paper Writing

Paper structure: DDPM vpred+Student-t headline, GARCH baseline, TimeGAN/VAE/NormFlow comparisons, sigmoid-vpred discovery as ablation insight, SF6 as known limitation

---

## 9. Branch Merge Status


| Branch                            | Merged to Main?            | Action Needed                 |
| --------------------------------- | -------------------------- | ----------------------------- |
| `shufeng`                         | Yes (Phase 6)              | None                          |
| `yizheng`                         | Partially (PR #8)          | 3 new commits pending         |
| `yixuan`                          | Partially (eval framework) | TimeGAN fixes not merged      |
| `experiment/timegan-3seed-colab`  | No                         | Merge (adds ANALYSIS.md only) |
| `yuxia/normflow-cross-model`      | Yes                        | None                          |
| `yuxia/ddpm-improved-enhancement` | No                         | Review + merge after testing  |
| `archive/timegan-experiments`     | No (archived)              | No action needed              |
| `merge-to-main`                   | Superseded                 | Can be deleted                |



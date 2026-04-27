# Gamma Presentation Refactor Pipeline
**Date:** Apr 27, 2026  
**Author:** Shufeng  
**Purpose:** Step-by-step instructions for updating the Gamma slide deck to reflect the latest L3/L4 results and Yixuan's corrections. Send this file to teammates before presentation.

---

## Quick Summary of What Changed

Two corrections from Yixuan (both verified against JSON result files):

1. **Crisis SF on slide 34 was wrong.** The 4/6 shown came from a single-run at guidance_scale=1.0. The 3-seed default (guidance_scale=2.0) gives **Crisis SF = 2/6**. The 4/6 is still achievable but must be labeled as guidance_scale=1.0 single-run.

2. **95% VaR Kupiec PASS is post-QM, not raw.** Raw model (expF_balanced) has 35% error and FAILS. The 6.3% error and PASS come from adding **flat quantile moment mapping** as a postprocessing step. The slide must say so explicitly.

Both corrections confirmed by:
- `experiments/results/conditional_ddpm_v2/expG_moderate/multiseed/multiseed_summary.json` → Crisis SF=2.0±0.0 (3-seed, default guidance)
- `experiments/results/conditional_ddpm_v2/moment_matching_expF_balanced/var_summary_rescaled.json` → 95% PASS (p=0.069), raw 35.34% error

Additionally: **L3 and L4 are now implemented and measured** (not "future work" as the current slide 17 and slide 36 state).

---

## Slide Status Table

| Slide | Title | Action |
|-------|-------|--------|
| 1 | Title | No change |
| 2 | Agenda | No change |
| 3 | Problem Statement | No change |
| 4 | What Makes Data Useful (L1–L4) | No change |
| 5 | Four-Layer Utility Framework | No change |
| 6 | Six Stylized Facts (1/2) | No change |
| 7 | Six Stylized Facts (2/2) | No change |
| 8 | Data Pipeline | No change |
| 9–10 | Architecture Overview | No change |
| 11 | GARCH | No change |
| 12 | VAE | No change |
| 13 | TimeGAN | No change |
| 14 | NormFlow | No change |
| 15 | DDPM Forward Process | No change |
| 16 | DDPM Reverse Process | No change |
| **17** | **DDPM Best Model** | **UPDATE** — L3/L4 bullet |
| 18 | DDPM Training Convergence | No change |
| 19 | Distributional Fidelity (MMD) | No change |
| 20 | Stylized Facts Heatmap | No change |
| 21 | Evaluation Framework | No change |
| 22 | Quantitative Metrics | No change |
| 23 | Cross-Model Comparison | No change |
| 24 | Stylized Facts Coverage | No change |
| 25 | DDPM Phase 7 Ablation (1/2) | No change |
| 26 | DDPM Phase 7 Ablation (2/2) | No change |
| 27 | Decorrelation Regularizer | No change |
| 28 | Calibration Discovery | No change |
| 29 | L2 Saturated Layer | No change |
| 30 | Training Details | No change |
| 31 | Diagnostic Visualizations (1/2) | No change |
| 32 | Diagnostic Visualizations (2/2) | No change |
| 33 | DDPM Synthetic Path Generation | No change |
| **34** | **L3 Regime Results** | **UPDATE** — 3-seed numbers + Yixuan correction |
| **NEW A** | **L3/L4 Iteration Methodology** | **ADD** after slide 34 |
| **35** | **L4 VaR/CVaR Backtest** | **UPDATE** — add QM progression + Yixuan caveat |
| **NEW B** | **L4 — From Failure to Partial Success** | **ADD** after slide 35 |
| **36** | **Conclusion & Future Work** | **UPDATE** — L3+L4 partial delivered |
| **37** | **Team Contributions** | **UPDATE** — Shufeng L3/L4 |
| 38 | References | No change |
| 39 | Thank You | No change |

---

## Step-by-Step Gamma Prompts

### STEP 1 — Update Slide 17 (DDPM Best Model)

Find the bullet point at the bottom of the slide that currently says:
> "○ L3 + L4: architecture supports cond_dim = 5 with CFG; explicitly scoped as future work"

**Replace it with:**
```
✓ L3 Conditional Control: regime vol ordering confirmed (crisis > normal > calm, 3.0× ratio)
⚠ L4 Downstream: 95% VaR Kupiec PASS with quantile moment mapping (post-processing); 99% open
```

**Gamma prompt:**
> On the DDPM Best Model slide, find the last bullet that says "L3 + L4: architecture supports cond_dim = 5 with CFG; explicitly scoped as future work" and replace it with two bullets: "L3 Conditional Control: regime vol ordering confirmed (crisis > normal > calm, 3.0× vol ratio). Implemented with classifier-free guidance and 5D macro conditioning." and "L4 Downstream Utility: 95% VaR Kupiec PASS achieved (p=0.069) with quantile moment mapping postprocessing. Raw model error ~35%; post-QM error 6.3%. 99% Kupiec remains open."

---

### STEP 2 — Update Slide 34 (L3 Regime Results)

**Current data (v1 checkpoint, single-run, guidance_scale=2.0):**

| Regime | SF | MMD | Disc | SynVol | RealVol |
|--------|----|-----|------|--------|---------|
| Crisis | 4/6 | 0.018 | 0.729 | 1.199 | 1.684 |
| Calm | 3/6 | 0.274 | 1.000 | 0.305 | 0.644 |
| Normal | 5/6 | 0.018 | 0.672 | 0.666 | 0.947 |

**Replace with (expG_moderate, 3-seed avg, guidance_scale=2.0):**

| Regime | n_real | SF | MMD | Disc | SynVol | RealVol |
|--------|--------|:--:|:---:|:----:|-------:|--------:|
| Crisis | 724 | 2/6 | 0.009 | 0.819 | 1.31 | 1.68 |
| Calm | 2,112 | 3/6 | 0.374 | 1.000 | 0.44 | 0.64 |
| Normal | 2,457 | 4/6 | 0.027 | 0.722 | 0.62 | 0.95 |

**Updated Key Findings:**
- Vol ordering confirmed: crisis (1.31) > normal (0.62) > calm (0.44) — direction matches real data ✓
- Normal regime near-unconditional quality (SF=4/6, Disc=0.722)
- Crisis SF=4/6 achievable at guidance_scale=1.0 (single-run; 3-seed default gives 2/6)
- Calm regime: Disc=1.000 — structural limitation, not addressable with more training
- Vol compression 22–48% across all regimes is the root cause of L4 VaR underestimation

**Add this note in small text:**
> Note: 3-seed average at guidance_scale=2.0. Crisis SF improves to 4/6 at guidance_scale=1.0 (single evaluation). Vol ordering is the primary L3 validation criterion.

**Image to insert:** `presentation_assets/l3_vol_ordering.png`

**Gamma prompt:**
> On the L3 Conditional Control slide (regime-stratified results), replace the table with these 3-seed average numbers from expG_moderate checkpoint at guidance_scale=2.0: Crisis (724 real windows): SF=2/6, MMD=0.009, Disc=0.819, SynVol=1.31, RealVol=1.68. Calm (2112 windows): SF=3/6, MMD=0.374, Disc=1.000, SynVol=0.44, RealVol=0.64. Normal (2457 windows): SF=4/6, MMD=0.027, Disc=0.722, SynVol=0.62, RealVol=0.95. Update key findings: Vol ordering confirmed: crisis(1.31) > normal(0.62) > calm(0.44) matching real data direction. Crisis SF=4/6 achievable at guidance_scale=1.0 (single-run). Calm remains hardest (Disc=1.000, structural). Vol compression 22-48% across regimes is root cause of L4 under-estimation. Add small note: "3-seed avg, guidance_scale=2.0. Vol ordering is primary L3 validation criterion."

---

### STEP 3 — Add NEW Slide A after Slide 34 (L3/L4 Iteration Methodology)

**Title:** "L3/L4 — Systematic Iteration"

**Content:**

Three-row progression table:

| Stage | Date | Key Change | L3 | L4 | Overall |
|-------|------|------------|:--:|:--:|:-------:|
| Sprint baseline | Apr 26 | Initial L3/L4 run; ad-hoc Kupiec threshold (abs<0.02) | 74 | 50 | 62 |
| Iteration 2 | Apr 27 | Proper Kupiec LR test; QM sample-size bug fix; balanced training | 79 | 72 | 75 |
| Iteration 3 | Apr 27 | Moderate retrain (expG); regime-conditional QM tested | **82** | **73** | **77.5** |

**Key methodology points:**
- Sprint's "99% Kupiec PASS" was self-invalidated in Iteration 2 — ad-hoc threshold was non-standard
- Proper Kupiec LR statistic: `LR = 2(n_exc·ln(p̂/p₀) + (n−n_exc)·ln((1−p̂)/(1−p₀))) ~ χ²(1)`
- Multi-seed validation (3 seeds) confirms results are not lucky draws
- Balanced regime training with WeightedRandomSampler addresses 14% crisis data imbalance

**Image to insert:** `presentation_assets/l3_l4_summary_table.png`

**Gamma prompt:**
> Insert a new slide after the L3 results slide, titled "L3/L4 — Systematic Iteration". Show a 3-row progress table: Row 1: Sprint baseline (Apr 26) — ad-hoc Kupiec threshold used, L3=74/100, L4=50/100, Overall=62. Row 2: Iteration 2 (Apr 27) — proper Kupiec LR chi-squared test adopted (this invalidated the sprint's Kupiec PASS claim), QM sample-size bug fixed, balanced regime training added. L3=79, L4=72, Overall=75. Row 3: Iteration 3 (Apr 27) — moderate retrain with crisis oversample=2x, regime-conditional QM tested but reverted to flat QM. L3=82, L4=73, Overall=77.5. Add a key insight callout box: "Honest methodology: we invalidated our own initial Kupiec claim when we found the ad-hoc threshold was non-standard. The proper LR test (Kupiec 1995) was adopted for all subsequent results."

---

### STEP 4 — Update Slide 35 (L4 VaR/CVaR Backtest)

**Current slide shows only v1 raw numbers (67% error, all FAIL). Replace entirely.**

**New VaR Progression Table:**

| Config | 95% VaR err | Kupiec@95% | 99% VaR err | Kupiec@99% |
|--------|:-----------:|:----------:|:-----------:|:----------:|
| Raw model (expF_balanced) | 35.1% | FAIL | 33.3% | FAIL |
| + Flat quantile mapping | **6.3%** | **PASS (p=0.069)** | 17.8% | FAIL |

**Root Cause: Volatility Compression** (keep this section, update numbers)
- Synthetic vol 22–48% below real across all regimes (expG_moderate 3-seed)
- MSE-based denoising objective favors mean predictions → tail compression
- Compressed tails → VaR under-estimated by ~35% at model level

**What Works (update):**
- PnL rank correlation: **0.988** (relative ordering preserved)
- Momentum strategy sign: both real and synthetic negative ✓ (sign now correct)

**Add honest caveat box:**
> "Kupiec test uses proper chi-squared LR statistic (Kupiec 1995). Hit rate 5.55% vs 5% nominal, p=0.069. The 6.3% VaR error is achieved via quantile moment mapping postprocessing — raw generation error is ~35%. QM corrects marginal distributions; the DDPM provides temporal structure and relative ordering."

**Image to insert:** `presentation_assets/l4_var_progression.png`

**Gamma prompt:**
> Replace the L4 VaR slide. New title: "L4 — Downstream Utility: VaR/CVaR Backtest". Add a two-row progression table: Row 1: Raw model expF_balanced — 95% VaR error 35.1%, Kupiec FAIL; 99% VaR error 33.3%, Kupiec FAIL. Row 2: expF + Flat Quantile Mapping — 95% VaR error 6.3%, Kupiec PASS (p=0.069); 99% VaR error 17.8%, Kupiec FAIL. Keep the root cause section: vol compression 22-48%, MSE objective, tail compression. Update: PnL rank correlation = 0.988, momentum strategy sign now correct. Add a highlighted callout: "Honest caveat: 6.3% VaR error is post quantile-mapping. Raw generation error is ~35%. QM corrects marginal distributions; the model provides temporal structure and relative ordering."

---

### STEP 5 — Add NEW Slide B after Slide 35 (L4 improvement visual)

**Title:** "L4 — From 67% Error to Kupiec PASS"

**Content:**
Show the `l4_var_progression.png` figure as the main visual.

**Caption points:**
- Three iterations: v1 baseline → expF balanced training → expF + quantile mapping
- The key breakthrough: quantile mapping per asset corrects marginal distributions without retraining
- 95% Kupiec PASS is a genuine statistical result (proper LR test, not ad-hoc)
- 99% remains open — requires solving vol compression at model level (e.g., learned variance schedule)

**Image to insert:** `presentation_assets/l4_var_progression.png`

**Gamma prompt:**
> Insert a new slide after the L4 results slide. Title: "L4 — VaR Error Reduction: 67% to 6.3%". Use the l4_var_progression.png figure as the main visual (upload it). Add 4 bullet points: (1) Three-stage reduction: v1 raw 67.7% → expF balanced 35.1% → expF+QM 6.3% at 95%. (2) Key mechanism: quantile mapping per asset maps synthetic quantiles to real quantiles — corrects marginal distributions without retraining. (3) 95% Kupiec PASS is statistically valid: LR test chi-sq, p=0.069 > 0.05, hit rate 5.55% vs 5% nominal. (4) 99% remains open: insufficient crisis-density in unconditional output; requires regime-controlled generation at inference time.

---

### STEP 6 — Update Slide 36 (Conclusion & Future Work)

**Current:** "Delivered: L1 + L2 Foundation & L3" / "Future Work — L4"

**Replace with:**

**Delivered — L1 + L2 + L3 + L4 (partial):**
- L1 Diversity — thousands of novel multi-asset paths across 5 generative architectures
- L2 Statistical Fidelity — DDPM 5/6 SF, MMD=0.006, best across all 5 models
- L3 Conditional Control — regime vol ordering confirmed (crisis > normal > calm, 3.0×)
- L4 Partial — 95% VaR Kupiec PASS (p=0.069); PnL rank correlation 0.988

**Remaining (Future Work):**
- L4 99% Kupiec: vol compression at model level (MSE objective, learned variance schedule)
- Calm regime Disc improvement: requires architectural change (separate U-Net or mixture model)
- SF6 decorrelation: Bonferroni correction on Ljung-Box, or replace 20-test joint with single test
- VaR backtest without postprocessing: regime-controlled generation at inference time

**Gamma prompt:**
> Update the conclusion slide. Change "Delivered: L1 + L2 Foundation & L3" to "Delivered: L1 + L2 + L3 + L4 (partial)". Under delivered, add: "L3 Conditional Control: regime vol ordering confirmed (crisis > normal > calm, 3.0× ratio). L4 Partial: 95% VaR Kupiec PASS (p=0.069) via quantile mapping; PnL rank correlation 0.988." Change the future work section from "L4 Downstream Utility" to: "L4 Remaining — 99% Kupiec (model-level vol compression). Calm regime improvement (architectural). SF6 Bonferroni correction. Regime-controlled generation at inference for VaR without postprocessing."

---

### STEP 7 — Update Slide 37 (Team Contributions)

**Update Shufeng Chen's row:**

Current:
> DDPM Baseline and Improved (v-prediction, Student-t); Ablation Study (7 phases); Integration, Demo, Cross-Model Comparison Pipeline

Replace with:
> DDPM Baseline and Improved (v-prediction, Student-t); Ablation Study (7 phases); L3 Conditional DDPM (regime generation, CFG); L4 VaR/CVaR Backtest (Kupiec validation, quantile mapping); Integration, Demo

**Gamma prompt:**
> In the Team Contributions slide, update Shufeng Chen's entry to: "DDPM Baseline and Improved (v-prediction, Student-t noise); 7-phase ablation study; L3 Conditional DDPM (regime-conditioned generation, classifier-free guidance); L4 VaR/CVaR Backtest (Kupiec LR validation, quantile moment mapping); Integration, Demo, Cross-Model Pipeline."

---

## New Assets Added to `presentation_assets/`

Three new figures generated for this refactor:

| Filename | Description | Used In |
|----------|-------------|---------|
| `l3_vol_ordering.png` | Grouped bar: syn vs real vol for crisis/normal/calm (3-seed expG_moderate) | Slide 34 (L3 Results) |
| `l4_var_progression.png` | VaR error reduction across 3 iterations with Kupiec PASS/FAIL annotations | NEW Slide B |
| `l3_l4_summary_table.png` | Iteration progress table (Sprint → Iter2 → Iter3) with L3/L4 scores | NEW Slide A |

---

## Outdated Assets (Do Not Delete, But Note)

The following files in `presentation_assets/` are from the sprint (pre-Kupiec-fix) and show results that were later corrected:

- `v2_ablation_comparison.png` — uses sprint v1 data, pre-Iteration 2 numbers
- `v2_moment_matching.png` — uses sprint QM results with old Kupiec threshold

These are **not currently used in the Gamma slide deck** (not in any of the 39 slides) so they do not need deletion. They serve as historical reference.

---

## Remote Branch Status (Apr 27, 2026, 1:25 PM)

No new commits from any teammate since this morning's review:

| Branch | Latest Commit | Status |
|--------|--------------|--------|
| `origin/main` | `c380632` (PPT script + WeChat summary) | Up to date |
| `origin/shufeng` | `c380632` | Up to date |
| `origin/yizheng` | `625d814` (min-SNR, stride exp) | 2 experimental commits, not needed |
| `origin/yizheng-update` | Already merged | No action |
| `origin/kevin/garch-visualization` | Already merged | No action |
| `origin/phase7-decorr-reg` | Already merged | No action |
| `origin/archive/timegan-experiments` | Archived | No action |
| `origin/experiment/timegan-3seed-colab` | Already merged | No action |

No changes from teammates needed before presentation.

---

## WeChat Group Message

See the section below in this file (or send the message separately).

---

大家好，Shufeng 这里 — presentation 前最后一轮 slides 更新说明，请大家过一下 📋

**【叶逸轩 (Yixuan) 的审查意见】** — 两条都确认有效，已按修改：

1. **Crisis SF 数字改了**：Slide 34 原来写的 Crisis SF=4/6 是 guidance_scale=1.0 的单次跑结果。3-seed 默认配置（guidance_scale=2.0）是 SF=2/6。已改为如实展示 3-seed 数字，并在注释里说明 4/6 只在 guidance_scale=1.0 时实现。vol ordering (1.31 > 0.62 > 0.44) 是更重要的 L3 证据，SF 数字加了说明就好。

2. **95% VaR Kupiec PASS 加了 honest caveat**：PASS 是 expF_balanced + flat quantile mapping 之后的结果（error 6.3%，p=0.069）。Raw 模型 error 是 35%，还是 FAIL。Slide 35 已更新，加了明确说明："6.3% with flat QM postprocessing; raw generation ~35%"。老师如果问，可以说：raw model 的 vol compression 是已知根因，QM 是工程修复手段，Kupiec 统计本身是 proper chi-squared LR test。

**【新增 3 张图片】** — 已生成，放在 `presentation_assets/`：
- `l3_vol_ordering.png` — L3 vol ordering bar chart（3-seed expG 数据）
- `l4_var_progression.png` — L4 VaR error 从 67% 到 6.3% 的进展图（Kupiec PASS 标注）
- `l3_l4_summary_table.png` — L3/L4 三轮迭代进度表

**【Slides 修改清单（Gamma 里改）】**：
- Slide 17（DDPM Best Model）：最后一条 bullet 从"L3+L4 future work"改成"已实现"
- Slide 34（L3 Results）：表格换成 3-seed expG_moderate 数据（含 Yixuan 修正）
- Slide 35（L4 VaR）：加 QM 前后对比 + honest caveat
- **新增** Slide A（L3/L4 三轮迭代方法论）after slide 34
- **新增** Slide B（L4 VaR 进展可视化）after slide 35
- Slide 36（Conclusion）：L3+L4 partial delivered
- Slide 37（Team）：Shufeng 加 L3/L4 contributions

**完整 Gamma prompts** 在 `docs/04-27-gamma-refactor-pipeline.md`，可以直接 copy-paste 进 Gamma。

加油，明天见！🚀

# Cross-Model Comparison: Is DDPM Currently the Best?

**Date:** 2026-04-22
**Author:** Shufeng Chen (with Cursor analysis)
**Status:** Internal analysis — not committed

---

## Data Sources

This analysis draws from three different result sets across branches:

| Source | Branch | Data Setup | Eval Framework | Models Covered |
|---|---|---|---|---|
| Phase 6 rebaseline | shufeng (`e7ad6ae`) | stride=1, 5293 windows | New (Hill/GARCH/Hurst) | DDPM vpred, DDPM vpred+Student-t |
| Cross-model analysis | main (`316842b`) | stride=1, 5293 windows | New (Hill/GARCH/Hurst) | NormFlow, TimeGAN, VAE, GARCH |
| Yixuan branch | origin/yixuan (`b2231b5`) | stride=1 | New + GJR-GARCH | TimeGAN (3 fixes), regime revert |

The cross-model analysis (Yuxia's work, merged into main via PR#7) ran all 4 non-DDPM models through the same pipeline with the new evaluation framework. This gives us a reasonable basis for comparison, though with caveats noted below.

---

## Full Comparison Table

| Model | SF (/6) | MMD | Wasserstein-1 | Disc. Score | CorrDist | Seeds | Eval |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **DDPM vpred+Student-t** | **5.0±0.0** | **0.006** | **0.111** | 0.85 | **1.79** | 3 avg | New |
| **DDPM vpred** | **5.0±0.0** | 0.037 | 0.148 | 0.93 | 1.87 | 3 avg | New |
| NormFlow (RealNVP) | 5 | 0.052 | 0.251 | **0.54** | 8.03 | 1 | New |
| TimeGAN | 4 | 0.065 | 0.303 | 0.89 | 7.77 | 1 | New |
| GARCH | 4 | 0.281 | 0.440 | 1.00 | 4.11 | 1 | New |
| VAE | 3 | 0.415 | 0.466 | 1.00 | 4.50 | 1 | New |

Bold = best in column (lower is better for all metrics except SF where higher is better).

---

## Is This Apples-to-Apples?

### What IS consistent across all models
- **Same assets**: 16 ETFs (SPY, XLK, XLF, XLE, XLV, XLI, XLP, XLY, XLU, XLB, TLT, IEF, SHY, GLD, USO, UUP)
- **Same preprocess.py**: stride=1, window_size=60, z-score normalization
- **Same stylized_facts.py**: Yixuan's new framework (Hill estimator, GARCH(1,1), Hurst R/S, max eigenvalue, MAA+Ljung-Box)
- **Same download source**: Yahoo Finance + FRED API

### What is NOT consistent (caveats)

| Factor | DDPM | Others |
|---|---|---|
| Seed averaging | 3 seeds (42, 123, 456), mean±std reported | Single run, no variance estimate |
| Evaluation mode | Standalone (no real_returns) | Standalone (no real_returns) — confirmed same |
| Training epochs | 400 | GARCH: fitted, others: likely 400 but not confirmed from JSON |
| run_pipeline.py version | Yizheng's Recovery version | Possibly Yixuan's earlier version (pre-Recovery) |

The most important caveat is **seed averaging**: DDPM has the advantage of reporting mean over 3 seeds, which smooths out lucky/unlucky runs. The other models are single-run. NormFlow's single-run MMD of 0.052 could plausibly vary between 0.03-0.07 across seeds. This doesn't change the ranking but makes the gap less definitive.

---

## Metric-by-Metric Analysis

### Stylized Facts (SF)

| Model | SF1 Fat Tails | SF2 Vol Clust | SF3 Leverage | SF4 Long Mem | SF5 Cross-Asset | SF6 No Autocorr | Total |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| DDPM vpred+Stud-t | PASS | PASS | PASS | PASS | PASS | FAIL | 5/6 |
| DDPM vpred | PASS | PASS | PASS | PASS | PASS | FAIL | 5/6 |
| NormFlow | PASS | PASS | PASS | PASS | PASS | FAIL | 5/6 |
| TimeGAN | PASS | PASS | PASS | PASS | FAIL | PASS | 4/6 |
| GARCH | FAIL | PASS | PASS | FAIL | PASS | PASS | 4/6 |
| VAE | PASS | FAIL | PASS | FAIL | PASS | FAIL | 3/6 |

**DDPM and NormFlow are tied at 5/6.** Both fail SF6 (no raw autocorrelation). TimeGAN and GARCH are tied at 4/6 but fail different facts. VAE is worst at 3/6.

Interesting: TimeGAN is the only model that PASSES SF6 — it generates returns with no autocorrelation. This is potentially because its adversarial training explicitly penalizes detectable temporal structure.

### MMD (Maximum Mean Discrepancy)

| Model | MMD |
|---|:---:|
| DDPM vpred+Stud-t | **0.006** |
| DDPM vpred | 0.037 |
| NormFlow | 0.052 |
| TimeGAN | 0.065 |
| GARCH | 0.281 |
| VAE | 0.415 |

**DDPM vpred+Student-t wins by a large margin** (8.7× better than NormFlow). Even plain vpred (0.037) beats NormFlow (0.052). This was the biggest surprise — under old data (stride=5), NormFlow had better MMD. With 5× more training data, DDPM's distributional matching dramatically improved.

However: NormFlow's 0.052 is from a single run. With 3-seed averaging it could be lower.

### Discriminative Score (lower = better)

| Model | Disc. Score |
|---|:---:|
| NormFlow | **0.54** |
| DDPM vpred+Stud-t | 0.85 |
| TimeGAN | 0.89 |
| DDPM vpred | 0.93 |
| GARCH | 1.00 |
| VAE | 1.00 |

**NormFlow wins decisively** on discriminative score. A discriminator can only correctly classify NormFlow synthetic data 54% of the time (barely above chance), while it can classify DDPM data 85% of the time. This means NormFlow's marginal distributions are closer to real data in ways a neural network can detect, even though DDPM's MMD is lower.

This apparent contradiction (low MMD but high disc score) suggests DDPM's samples match the overall distribution well (low MMD) but have some systematic artifact a discriminator can learn (e.g., slight autocorrelation, SF6 failure).

### Correlation Matrix Distance (lower = better)

| Model | CorrDist |
|---|:---:|
| DDPM vpred+Stud-t | **1.79** |
| DDPM vpred | 1.87 |
| GARCH | 4.11 |
| VAE | 4.50 |
| TimeGAN | 7.77 |
| NormFlow | 8.03 |

**DDPM dominates CorrDist by a factor of 4× over NormFlow.** This is a genuinely important finding: DDPM preserves cross-asset correlation structure far better than any other model. NormFlow's 8.03 means its correlation matrices are substantially different from real data, while DDPM's 1.79 is a close match.

This matters for the practical use case (portfolio stress testing) where cross-asset correlations drive risk estimates.

---

## Verdict: Is DDPM the Best?

**It depends on which metric you prioritize.**

### DDPM wins on:
- **Stylized facts**: Tied with NormFlow at 5/6, but with 3-seed consistency proof
- **MMD**: 8.7× better than NormFlow (0.006 vs 0.052)
- **Wasserstein-1**: 2.3× better (0.111 vs 0.251)
- **CorrDist**: 4.5× better (1.79 vs 8.03)
- **Robustness**: 0.0 standard deviation across 3 seeds

### NormFlow wins on:
- **Discriminative score**: 0.54 vs 0.85 — NormFlow's synthetic data fools a classifier much better
- **Simplicity**: Fewer hyperparameters, no schedule/noise decisions
- **Training cost**: NormFlow training is faster

### For the paper, the honest framing is:

> DDPM (v-prediction + Student-t) matches NormFlow on stylized fact coverage (5/6) while substantially outperforming it on distributional metrics (MMD, Wasserstein-1) and cross-asset correlation preservation. NormFlow retains an advantage on discriminative score, suggesting DDPM synthetic data contains learnable artifacts not captured by summary statistics. Both models fail to reproduce the no-autocorrelation property of raw returns.

---

## What Would Make This Fully Apples-to-Apples

1. **Re-run NormFlow with 3 seeds** (42, 123, 456) at 400 epochs under the exact same pipeline — this would give us variance estimates to compare directly
2. **Re-run all models through run_ddpm_ablation.py** (or an equivalent cross-model runner) with `--seeds 42 123 456 --epochs 400`
3. **Use identical `run_pipeline.py`** for all models — Yizheng's Recovery version

This is the logical next step for Thursday's team sync.

---

## Branch Status Summary

| Branch | Latest Commit | Key Content | Merge Status |
|---|---|---|---|
| origin/main (`316842b`) | Recovery by Yizheng, Apr 22 00:49 | Cross-model results, new eval, stride=1 | Reference branch |
| origin/shufeng (`e7ad6ae`) | Phase 6 rebaseline, Apr 22 | DDPM 5/6 SF results | 2 commits ahead of main; README conflict |
| origin/yixuan (`b2231b5`) | TimeGAN fixes, Apr 21 23:43 | 3 TimeGAN bug fixes + regime revert | Not merged; fixes should go to main |
| origin/yizheng | Merge from main, Apr 21 21:47 | Same as main | Up to date |
| origin/yuxia/normflow-cross-model | Cross-model analysis, Apr 16 | NormFlow + cross-model framework | Already merged to main |

# Generative Market Simulation: Reproducing Stylized Facts with Deep Generative Models

## EECS 4904 — Spring 2026 Final Project

Team: Shufeng Chen, Yixuan Ye, Yizheng Lin, Kevin Sun, Yuxia Meng

---

## Slide 1: Problem Statement

Risk management requires thousands of realistic market scenarios to stress-test portfolios. Replaying historical data only gives one realized path. Classical models like GARCH produce conditionally Gaussian returns that fail to capture heavy tails, volatility clustering, and leverage effects.

Our goal: train deep generative models to produce synthetic multi-asset return series that faithfully reproduce the statistical properties (stylized facts) of real financial data.

Data: 16 S&P 500 assets (sector ETFs, Treasuries, gold, oil), daily returns from 2005-2026, 60-day overlapping windows with stride=1 (~5,300 windows).

---

## Slide 2: Six Stylized Facts of Financial Returns

Every generated dataset is validated against six well-documented empirical regularities:

1. Fat Tails — Return distributions are heavier-tailed than Gaussian (Hill estimator alpha < 5)
2. Volatility Clustering — Large moves tend to follow large moves (GARCH(1,1) gamma > 0.85)
3. Leverage Effect — Negative returns increase subsequent volatility more than positive returns (GJR-GARCH gamma > 0)
4. Long Memory — Absolute returns show slow autocorrelation decay (Hurst exponent 0.5 < H < 1.0)
5. Cross-Asset Correlations — Correlations between assets are time-varying (max eigenvalue lambda_1 > 1.5)
6. No Raw Autocorrelation — Raw returns are approximately uncorrelated (MAA < 0.05 + Ljung-Box)

---

## Slide 3: Architecture Overview

Pipeline: Yahoo Finance + FRED API → download.py → preprocess.py → regime_labels.py → 5 Generative Models → Evaluation (6 Stylized Facts + MMD + Wasserstein + Discriminative Score) → Interactive Demo (FastAPI + Chart.js)

Five models compared:
- GARCH(1,1) — statistical baseline with correlated Student-t innovations
- VAE — BiGRU encoder-decoder with KL annealing and Student-t likelihood
- TimeGAN — Embedding + supervisor + adversarial training (WGAN-GP)
- RealNVP (Normalizing Flow) — Affine coupling layers with ActNorm
- DDPM — 1-D U-Net denoiser with v-prediction, Student-t noise, DDIM sampling, EMA, classifier-free guidance

---

## Slide 4: GARCH Baseline

Model: Per-asset GARCH(1,1) with AR(1) mean and Student-t innovations. Cross-asset dependence via Cholesky decomposition of empirical residual correlation matrix.

Results (3-seed average): SF = 1.3/6, MMD = 0.042, Disc = 1.00

Passes only: Leverage Effect (1/3 seeds). Fails Fat Tails, Volatility Clustering, Long Memory, Cross-Asset Correlations, No Raw Autocorrelation.

Role: Non-deep learning baseline. Demonstrates that classical models cannot capture the full range of stylized facts.

---

## Slide 5: VAE (Improved)

Model: Bidirectional GRU encoder with autoregressive decoder, Student-t likelihood, learnable degrees of freedom, factor loadings for cross-asset structure, ACF and correlation regularizers.

Results (3-seed average): SF = 1.0/6, MMD = 0.020, Disc = 0.75

Passes only: Leverage Effect. Despite low MMD (competitive sample quality), VAE struggles to capture the distributional structure needed for most stylized facts.

---

## Slide 6: TimeGAN

Model: TimeGAN-style architecture with embedding, recovery, generator, supervisor, and discriminator networks. Trained with WGAN-GP losses and moment matching.

Results (3-seed average): SF = 4.0/6, MMD = 0.110, Disc = 1.00

Passes: Fat Tails, Volatility Clustering, Leverage Effect, Long Memory. Fails: Cross-Asset Correlations, No Raw Autocorrelation.

Strong on temporal dynamics but weak on cross-asset structure and sample quality (highest MMD among deep models).

---

## Slide 7: Normalizing Flow (RealNVP)

Model: RealNVP with affine coupling layers, ActNorm, and exact log-likelihood training. Flat-vector representation of windows.

Results (3-seed average): SF = 5.0/6, MMD = 0.027, Disc = 0.73

Passes: Fat Tails, Volatility Clustering, Leverage Effect, Long Memory, Cross-Asset Correlations. Fails: No Raw Autocorrelation.

Best discriminative score (0.73, closest to ideal 0.5) — synthetic data hardest to distinguish from real. Tied with DDPM on SF count.

---

## Slide 8: DDPM — Our Best Model

Model: 1-D U-Net denoiser with v-prediction objective (Salimans & Ho, 2022) and Student-t forward process (df=5). Cosine noise schedule, DDIM sampling, EMA, classifier-free guidance.

Results (3-seed average): SF = 5.0/6, MMD = 0.006, Disc = 0.85

Passes: Fat Tails, Volatility Clustering, Leverage Effect, Long Memory, Cross-Asset Correlations. Fails: No Raw Autocorrelation.

Best MMD by far (0.006 vs next-best 0.027). V-prediction improved SF from 1.7/6 to 5.0/6. Student-t noise reduced MMD by 6x (0.037 → 0.006).

Key discovery: Sigmoid noise schedule suppresses volatility clustering when paired with v-prediction (SF drops from 5.0 to 2.7). Cosine schedule is the correct pairing.

---

## Slide 9: Cross-Model Comparison

Results table (3-seed average, unified evaluation):

| Model | SF Passed | MMD | Disc. Score |
|-------|:---------:|:---:|:-----------:|
| GARCH (Baseline) | 1.3/6 | 0.042 | 1.00 |
| VAE (Improved) | 1.0/6 | 0.020 | 0.75 |
| TimeGAN | 4.0/6 | 0.110 | 1.00 |
| NormFlow (RealNVP) | 5.0/6 | 0.027 | 0.73 |
| DDPM (v-pred+Student-t) | 5.0/6 | 0.006 | 0.85 |

DDPM achieves the best distributional fidelity (lowest MMD) while matching NormFlow on stylized fact coverage.

[INSERT: fig_sf_bar.png — SF passed per model bar chart]
[INSERT: fig_mmd_bar.png — MMD per model bar chart]
[INSERT: fig_sf_heatmap.png — SF pass/fail heatmap]
[INSERT: fig_radar.png — Radar chart normalized metrics]

---

## Slide 10: DDPM Phase 7 Ablation

Beyond Phase 6 (v-pred + Student-t), we explored additional improvements:

| Config | SF | MMD | Disc |
|--------|:--:|:---:|:----:|
| Phase 6 (v-pred + Student-t) | 5/6 | 0.006 | 0.85 |
| + Min-SNR + warmup (Yuxia) | 5/6 | 0.031 | 0.92 |
| + Min-SNR + decorr_reg (Yixuan) | 5/6 | 0.015 | 0.87 |
| + patch stride=2 (Yizheng) | 4/6 | 0.021 | 0.72 |

All DDPM variants hit SF=5/6 (except patch which loses SF1). Phase 6 remains the best on MMD. Min-SNR improves discriminative score but trades off sample quality.

The decorrelation regularizer (penalizing raw-return ACF during training) reduces MAA by 17% and Ljung-Box statistic by 24-39%, but cannot flip the binary SF6 result.

[INSERT: fig_ddpm_ablation_sf.png — DDPM ablation SF bar chart]
[INSERT: fig_ddpm_ablation_mmd.png — DDPM ablation MMD bar chart]

---

## Slide 11: Evaluation Framework Calibration Discovery

Critical finding: Running our evaluation framework on the REAL training data (S&P 16 assets, 2000-2024, n=318,060 observations) reveals that real data only passes 3/6 stylized facts:

- SF1 (Fat Tails): FAIL — Hill alpha = 7.83 (threshold alpha < 5, but real data with large sample size has thinner measured tails)
- SF4 (Long Memory): FAIL — Hurst = 1.01 (exceeds the 0.5-1.0 range due to non-stationarity)
- SF6 (No Raw Autocorrelation): FAIL — LB statistic = 5,927 (Ljung-Box is statistically impossible to pass at this sample size due to multiple-testing across 20 lags)

Implication: Our synthetic data at 5/6 is MORE stylized-fact-compliant than the actual source data (3/6). The SF=5/6 result represents the effective ceiling, not a failure. Pushing to 6/6 would be counterproductive.

---

## Slide 12: Diagnostic Visualizations

Return distributions: DDPM synthetic returns closely match the heavy-tailed shape of real returns, unlike GARCH (too Gaussian) or VAE (mode collapse).

[INSERT: fig_distributions_h2h.png — Return distribution comparison]
[INSERT: fig_qq_h2h.png — QQ-plot real vs synthetic]

Autocorrelation of |returns|: DDPM captures the slow decay pattern characteristic of volatility clustering. GARCH decays too fast; TimeGAN has artifacts.

[INSERT: fig_acf_h2h.png — ACF of absolute returns comparison]

Correlation structure: DDPM preserves cross-asset correlation patterns (CorrDist = 1.79, lowest among all models).

[INSERT: fig_corr_ddpm.png — Correlation matrix real vs DDPM]

Synthetic price paths: DDPM generates realistic multi-asset price trajectories with proper regime behavior.

[INSERT: fig_paths_ddpm_final.png — Synthetic price paths]

---

## Slide 13: Conclusion and Future Work

Conclusion:
- DDPM with v-prediction + Student-t noise is the best generative model for financial time series, achieving 5/6 stylized facts and MMD = 0.006
- V-prediction is the single most impactful innovation (1.7/6 → 5/6 SF)
- Student-t noise further improves distributional fidelity (6x MMD reduction)
- SF=5/6 is the empirical ceiling — our synthetic data is more "real" than real data by this metric
- The evaluation framework calibration finding is a contribution to methodology

Future work:
- Fix the Ljung-Box multiple-testing issue (use joint test or Bonferroni correction)
- Explore DDIM eta sweep for optimal sampling
- Conditional generation for regime-specific scenarios (crisis stress testing)
- Extend to higher-frequency or longer-horizon data
- Integrate with downstream risk management applications

---

## Team Contributions

| Member | Contributions |
|--------|--------------|
| Shufeng Chen | DDPM Baseline and Improved (v-prediction, Student-t), Ablation Study (7 phases), Integration, Demo, Cross-Model Comparison Pipeline |
| Yixuan Ye | TimeGAN, Evaluation Framework (Stylized Facts), Data Pipeline, Decorrelation Regularizer, Calibration Discovery |
| Yizheng Lin | VAE (Improved + Original), Data Pipeline, FRED Integration, Patch-Based Training Experiments |
| Kevin Sun | GARCH Baseline, Visualization Utilities |
| Yuxia Meng | Normalizing Flow (RealNVP), Cross-Model Analysis, DDPM Training Enhancements (Min-SNR, warmup LR) |

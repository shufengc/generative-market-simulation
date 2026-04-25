# GARCH Rebaseline Analysis

**Date:** 2026-04-24  
**Script:** `experiments/run_garch_rebaseline.py`  
**Model:** Kevin Sun's improved `GARCHModel` (AR(1) mean, EGARCH fallback, Student-t innovations, DCC correlation)

## Setup

| Setting | Value |
|---------|-------|
| Data | stride=1, seq_len=60, 5,293 windows, 16 assets, 20 years |
| Epochs | N/A (GARCH is fitted, not gradient-trained) |
| Seeds | 42, 123, 456 |
| Generated samples | 1,000 per seed |
| Evaluation | Unified 6-SF framework + MMD + W1 + Disc + CorrDist |
| Hardware | CPU (Apple M4 Pro) -- GARCH uses `arch` library, no GPU |

## Results Summary

| Seed | SF Passed | MMD | Wasserstein-1D | Discriminative | CorrDist | Train Time |
|------|:---------:|:---:|:--------------:|:--------------:|:--------:|:----------:|
| 42 | 1/6 | 0.0380 | 3.504 | 1.000 | 2.960 | 16.6s |
| 123 | 2/6 | 0.0313 | 3.428 | 1.000 | 2.899 | 16.5s |
| 456 | 1/6 | 0.0554 | 3.747 | 1.000 | 3.057 | 16.6s |
| **Mean ± SD** | **1.3 ± 0.5** | **0.042 ± 0.010** | **3.560 ± 0.136** | **1.000 ± 0.000** | **2.972 ± 0.065** | |

## Per-SF Breakdown (Seed 42 representative)

| Stylized Fact | GARCH | DDPM vpred+Student-t |
|---------------|:-----:|:--------------------:|
| SF1: Fat Tails (Hill α) | FAIL (α=3.40 vs real 7.84) | PASS |
| SF2: Volatility Clustering | **PASS** | PASS |
| SF3: Leverage Effect | FAIL | PASS |
| SF4: Long Memory (Hurst) | FAIL (H=0.60 vs real 1.01) | PASS |
| SF5: Cross-Asset Correlations | FAIL (λ₁ gap=27%) | PASS |
| SF6: No Raw Autocorrelation | FAIL (LB p=0.0) | FAIL |

## Key Observations

1. **GARCH is a weak generative model** for our 6-SF benchmark -- passes only 1-2/6 across seeds. This is expected: GARCH was designed to model volatility dynamics, not to reproduce the full joint distribution of financial returns.

2. **Discriminative score = 1.000 in all seeds** -- a trivial discriminator perfectly separates GARCH samples from real data. This confirms GARCH generates obviously synthetic distributions.

3. **Only Volatility Clustering passes consistently** -- this makes intuitive sense since GARCH is explicitly designed to model heteroskedastic variance (ARCH effects). It does exactly what it was built for.

4. **Fat Tails fail despite Student-t innovations** -- the Hill estimator measures the tail index of the marginal distribution. GARCH's AR(1) + Student-t produces heavy tails in log-return space, but the windowed joint distribution (what the Hill estimator sees on flattened windows) shows α=3.4 vs real α=7.8. The KS test p=0.0 confirms distributional mismatch.

5. **Cross-asset correlations miss by ~27%** -- the static Cholesky decomposition of empirical residual correlations captures mean correlation but not time-varying correlation structure. This is expected from a scalar-DCC-style baseline.

6. **Wasserstein-1D is extremely high (3.5)** -- reflects the distributional mismatch in marginal distributions. Compare to DDPM: W1≈0.9.

## Role in Presentation

GARCH serves as our **baseline model** -- it establishes a lower bound on performance. The presentation narrative is:
- GARCH (1.3/6 SF): Simple statistical baseline; fast but limited
- TimeGAN (2.7/6 SF): GAN-based; better but discriminator collapses
- NormFlow (5.0/6 SF): Strong but complex and slower  
- VAE improved (TBD/6 SF): Variational autoencoder approach
- **DDPM vpred+Student-t (5/6 SF, Disc=0.85)**: Best model -- strong SF coverage AND realistic discriminative score

The GARCH baseline makes the DDPM improvement look compelling because GARCH fails 4-5 of the 6 tests while DDPM passes 5/6.

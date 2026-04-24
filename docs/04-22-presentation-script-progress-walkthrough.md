# Generative Market Simulation — Final Presentation Script
# EECS 4904 Spring 2026 | Paste this into Gamma.ai "Paste in text" mode

---

## Slide 1: Title

**Generative Market Simulation**
Synthetic Financial Time Series via Deep Generative Models

EECS 4904 — Spring 2026 Final Project

**Team:**
- Shufeng Chen — DDPM, Integration, Demo
- Yixuan Ye — TimeGAN, Evaluation Framework
- Yizheng Lin — Data Pipeline, NormFlow
- Yuxia Meng — NormFlow, Cross-Model Analysis
- Kevin Sun — VAE, GARCH Baseline

---

## Slide 2: Motivation

**Why synthetic financial data?**

- Risk management needs thousands of realistic market scenarios to stress-test portfolios
- Replaying history gives only one realized path
- Classical models (GARCH, Black-Scholes) produce Gaussian returns that miss the structure of real markets

**What makes financial returns hard to model:**

- Heavy tails: 50× more extreme events than Gaussian predicts
- Volatility clustering: calm periods and turbulent periods cluster
- Leverage effect: crashes amplify future volatility asymmetrically
- Long memory: shocks persist for months in absolute returns

**Goal:** Train generative models that reproduce all of these properties simultaneously, validated quantitatively against real data.

---

## Slide 3: The Six Stylized Facts

Every generated dataset is validated against six empirical regularities first documented by Cont (2001):

| # | Property | Test Used | Threshold |
|---|----------|-----------|-----------|
| SF1 | Fat Tails | Hill tail-index estimator α | α < 5 (heavier than thin Gaussian) |
| SF2 | Volatility Clustering | GARCH(1,1) persistence γ = α+β | γ > 0.85 |
| SF3 | Leverage Effect | GJR-GARCH asymmetric γ | γ > 0 (positive asymmetry) |
| SF4 | Long Memory | Hurst exponent R/S analysis | H ∈ (0.5, 1) |
| SF5 | Time-Varying Cross-Asset Correlations | Max eigenvalue λ₁ of corr. matrix | λ₁ > 1.5 |
| SF6 | No Raw Autocorrelation | Mean absolute autocorrelation + Ljung-Box | MAA < 0.05 AND LB p > 0.05 |

**Key:** A model that passes all 6 is indistinguishable from real data on these statistics.

---

## Slide 4: System Architecture

**Data Pipeline**

Yahoo Finance (16 ETFs, 2005–2026) + FRED API (macro conditioning)
→ Log returns → Z-score normalization → 60-day overlapping windows
→ **5,293 training windows** (stride=1)
→ Regime labels: Crisis / Normal / Calm (GARCH + VIX-based)

**Five Generative Models Compared**

| Model | Type | Parameters | Training |
|-------|------|-----------|---------|
| DDPM (Improved) | Diffusion | 9.0M | 400 epochs |
| DDPM (Baseline) | Diffusion | 9.0M | 400 epochs |
| NormFlow (RealNVP) | Normalizing Flow | 6.7M | 400 epochs |
| TimeGAN | GAN | ~5M | 600 epochs |
| VAE | Variational | ~2M | 400 epochs |
| GARCH | Statistical | — | Fitted |

**Evaluation Pipeline**
6 Stylized Facts + MMD + Wasserstein-1 + Discriminative Score + Correlation Matrix Distance

---

## Slide 5: DDPM Innovation — V-Prediction

**The Core Problem with Standard DDPM**

Standard DDPM predicts noise ε. This works for images but in financial data:
- The denoiser learns to predict average noise → synthetic returns are too smooth
- Kurtosis collapses → fat tails disappear

**Solution: V-Prediction (Salimans & Ho, 2022)**

Instead of predicting noise, predict the "velocity" v = √ᾱ · ε − √(1−ᾱ) · x₀

**Effect:** The loss has better gradient balance across all noise levels, forcing the model to preserve extreme events.

**Results from controlled Phase 3 ablation (same 128ch, 400ep, 3 seeds):**

| Config | SF | MMD | Fat Tails Pass |
|--------|:--:|:---:|:--------------:|
| Baseline (no v-pred) | 1.7/6 | 0.364 | 0% |
| V-prediction only | 4.7/6 | 0.019 | 100% |
| V-pred + sigmoid schedule | 2.7/6 | 0.055 | 33% |

**Key discovery:** The sigmoid noise schedule *suppresses* fat-tail generation when combined with v-prediction. The correct pairing is v-prediction + cosine schedule.

---

## Slide 6: Final Results — All Models

*Evaluated on 16-asset universe, 60-day windows, 400 epochs, 3 seeds (42/123/456)*
*DDPM results: stride=1 data (5,293 windows), new evaluation framework*

| Model | Stylized Facts | MMD | Wasserstein-1 | Disc. Score | CorrDist |
|-------|:--------------:|:---:|:-------------:|:-----------:|:--------:|
| **DDPM (v-pred + Student-t)** | **5.0/6 ± 0.0** | **0.006** | **0.111** | 0.85 | 1.79 |
| **DDPM (v-prediction)** | **5.0/6 ± 0.0** | 0.037 | 0.148 | 0.93 | 1.87 |
| NormFlow (RealNVP)† | 5.0/6 | 0.015 | — | 0.67 | 8.0 |
| TimeGAN† | 4.0/6 | 0.065 | 0.303 | 0.89 | 7.8 |
| GARCH† | 4.0/6 | 0.281 | 0.440 | 1.00 | 4.1 |
| VAE† | 3.0/6 | 0.415 | 0.466 | 1.00 | 4.5 |

*† NormFlow/TimeGAN/VAE/GARCH numbers from cross-model analysis under previous evaluation framework. Re-run under new framework pending.*

**Failing fact for all models:** SF6 (No Raw Autocorrelation) — a known structural limitation.

**Lower MMD = better. Lower Disc. Score = synthetic harder to distinguish from real (better).**

---

## Slide 7: DDPM Ablation — What Helps, What Hurts

**6 phases of ablation, 20+ variants, 3 seeds each**

| Innovation | SF change | MMD change | Verdict |
|-----------|:---------:|:----------:|---------|
| V-prediction (baseline → vpred) | +3.0/6 | −0.345 | ✅ Core innovation |
| Cosine schedule (vs sigmoid) | +2.0/6 | +0.036 | ✅ Required with v-pred |
| Student-t noise (df=5) | +0.0/6 | −0.031 | ✅ Best MMD improvement |
| Auxiliary SF loss (kurtosis + ACF) | +0.0/6 | −0.006 | ⚠️ Marginal |
| Temporal self-attention | +0.0/6 | −0.004 | ⚠️ Marginal, costly |
| Wavelet-domain diffusion | −0.3/6 | +0.012 | ❌ Unstable |
| Sigmoid schedule (with v-pred) | −2.0/6 | +0.037 | ❌ Harmful interaction |

**Recommended final config:** v-prediction + cosine schedule + Student-t noise (df=5)
- SF: **5/6**, MMD: **0.006**, zero extra parameters over baseline

---

## Slide 8: Distributional Quality

**Return Distribution Comparison**

DDPM (v-pred + Student-t) closely matches real return distributions:
- Heavy tails reproduced (Hill α ≈ 3.0 vs real ≈ 2.5–3.5 range)
- GARCH(1,1) persistence γ ≈ 0.95 (real ≈ 0.97)
- GJR-GARCH asymmetric coefficient > 0 (leverage effect confirmed)

**Quantitative gaps remaining:**

| Metric | DDPM | NormFlow | Real |
|--------|:----:|:--------:|:----:|
| Hill α (fat tails) | ~3.0 | ~2.2 | ~2.5–3.5 |
| GARCH γ | ~0.95 | ~0.97 | ~0.97 |
| Hurst H | ~0.64 | ~0.60 | ~0.60–0.65 |

*[Insert figure: distributions.png and qq_plots.png]*

---

## Slide 9: Temporal and Cross-Asset Structure

**Volatility Clustering** (ACF of |returns|)
- DDPM and NormFlow both generate persistent volatility structure
- GARCH captures it by construction; VAE fails entirely

**Cross-Asset Correlations**
- During crisis regimes, correlations spike (contagion effect)
- DDPM: max eigenvalue λ₁ ≈ 8.3 ✅ (real ≈ 6–10 range)
- NormFlow: CorrDist = 8.0 (higher than DDPM's 1.79 — weaker cross-asset structure)

**Regime Conditioning** (Crisis / Normal / Calm)
- DDPM and NormFlow both maintain SF performance across regimes
- TimeGAN degrades in crisis regime (MMD 0.022 → 0.065 calm)

*[Insert figure: acf_absolute.png, correlation_matrices.png, regime_sf_passed.png]*

---

## Slide 10: Key Findings and Conclusions

**What we found:**

1. **V-prediction is the single most impactful improvement** (+3 stylized facts, −9× MMD over baseline). Zero extra parameters.

2. **The sigmoid-vpred interaction is a real danger** — a widely used schedule destroys the benefits of a widely used loss target. We identified this through controlled ablation, not literature.

3. **5× more training data matters** — fixing stride=5→stride=1 pushed DDPM from 4.7→5.0/6. Data pipeline correctness is as important as architecture.

4. **Student-t forward process closes the distributional gap** — same stylized facts, 6× better MMD. Synthetic data is much harder for a discriminator to detect.

5. **NormFlow remains the strongest baseline on overall distributional metrics** — but DDPM matches it on stylized facts and exceeds it on cross-asset correlation structure (CorrDist 1.79 vs 8.0).

**What SF6 failure means:** All models produce some short-lag autocorrelation in raw returns. This is a structural limitation of training on overlapping windows, not a model-specific failure.

**Future work:** Re-run NormFlow/TimeGAN under new eval framework for fair comparison. Address SF6 via inference-time autocorrelation guidance.

---

## Slide 11: Demo

**Interactive Demo — Regime-Conditioned Generation**

```
python3 -m src.demo.app
# Open http://localhost:8000
```

- Select model (DDPM, NormFlow, GARCH, TimeGAN, VAE)
- Choose regime: Crisis / Normal / Calm
- Generate 16-asset synthetic return paths in real time
- Compare stylized facts against real data

**DDPM supports classifier-free guidance** conditioned on 5 macro features:
yield curve slope, credit spread, fed funds rate, VIX level, realized volatility

*Crisis-conditioned DDPM generates higher-volatility, more correlated paths — matching what we observe in real market crashes.*

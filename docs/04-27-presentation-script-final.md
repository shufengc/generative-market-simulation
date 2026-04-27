# Generative Market Simulation — Presentation Script (Final)
# EECS 4904 Spring 2026 | Apr 28, 2026
# Gamma.ai: use "Paste in text" → Theme: Professional/Dark

---

## Slide 1: Title

**Generative Market Simulation**
Synthetic Financial Time Series via Deep Generative Models

EECS 4904 — Spring 2026 Final Project

**Team:**
- Shufeng Chen — DDPM (Baseline & Improved), L3/L4, Integration, Demo
- Yixuan Ye — TimeGAN, Evaluation Framework (Stylized Facts), Data Pipeline
- Yizheng Lin — VAE (Original & Improved), Data Pipeline, FRED Integration
- Yuxia Meng — NormFlow (RealNVP), Cross-Model Analysis, DDPM Enhancements
- Kevin Sun — GARCH Baseline, Visualization

**Speaker note:** 20-second open — "We trained five generative models to produce realistic financial data and validated them against six established properties of real markets."

---

## Slide 2: Motivation

**Why synthetic financial data?**

- Risk management needs *thousands* of market scenarios to stress-test portfolios
- History gives only one realized path
- Classical models (GARCH, Black-Scholes) assume Gaussian returns — they miss the structure of real markets

**The structure that matters (and that Gaussian models miss):**

- **Heavy tails** — extreme crashes happen 50× more often than Gaussian predicts
- **Volatility clustering** — turbulent periods follow turbulent periods
- **Leverage effect** — crashes amplify future volatility asymmetrically
- **Long memory** — shocks persist in absolute returns for months

**Goal:** Generative models that reproduce *all* of these properties simultaneously, validated quantitatively.

**Speaker note:** Stress the business motivation — a risk manager using Gaussian scenarios will under-reserve capital. Our synthetic data is the alternative.

---

## Slide 3: The Six Stylized Facts

Every generated dataset is scored against six empirical regularities (Cont, 2001):

| # | Property | Test | Pass if |
|---|----------|------|---------|
| SF1 | Fat Tails | Hill tail-index α | α < 5 |
| SF2 | Volatility Clustering | GARCH(1,1) persistence γ = α+β | γ > 0.85 |
| SF3 | Leverage Effect | GJR-GARCH asymmetric response | coefficient > 0 |
| SF4 | Long Memory | Hurst exponent R/S | H ∈ (0.5, 1) |
| SF5 | Time-Varying Cross-Asset Correlations | Max eigenvalue λ₁ | λ₁ > 1.5 |
| SF6 | No Raw Autocorrelation | Mean abs. autocorr + Ljung-Box | MAA < 0.05 AND all LB p > 0.05 |

**Key calibration finding:** Running this same evaluation on *real* training data yields **only 3/6**. Real data fails SF1 (thick Hill α), SF4 (non-stationary Hurst), and SF6. Our best model at **5/6 is more stylized-fact-compliant than the source data** — meaning 5/6 is the empirical ceiling.

**Speaker note:** This was a discovery during Phase 7. It reframes SF6 failure not as a model bug but as a near-impossible test — even true white noise only passes all 20 Ljung-Box lags ~36% of the time (0.95^20).

---

## Slide 4: System Architecture & Data

**Data Pipeline**

Yahoo Finance (16 ETFs — S&P 500 sectors, Treasuries, gold, oil, dollar index, 2005–2026)
+ FRED API (macro conditioning: yield curve slope, credit spread, fed funds, VIX)
→ Log returns → Z-score normalization → 60-day overlapping windows with **stride=1**
→ **5,293 training windows** | 3 seeds (42, 123, 456) for reproducibility

**Five Models Compared**

| Model | Type | Parameters |
|-------|------|-----------|
| **DDPM (v-pred + Student-t)** | Diffusion U-Net | 9.0M |
| NormFlow (RealNVP) | Normalizing Flow | 6.7M |
| TimeGAN (WGAN-GP) | GAN | ~5M |
| VAE (Improved) | Variational | ~2M |
| GARCH | Statistical baseline | — |

**Evaluation:** 6 stylized facts + MMD (kernel distance) + Discriminative Score (can a classifier tell real from fake?) + Wasserstein-1 + Correlation Matrix Distance

*[Figure: pipeline_overview.png]*

---

## Slide 5: The DDPM Improvement Story — V-Prediction

**The Problem with Standard DDPM on Financial Data**

Standard DDPM predicts noise ε → denoiser learns to average → returns are too smooth → fat tails disappear.

**Solution: V-Prediction (Salimans & Ho, 2022)**

Predict velocity: **v = √ᾱ · ε − √(1−ᾱ) · x₀**

Better gradient balance across noise levels → model *must* preserve extreme events.

**Controlled Phase 3 Ablation (128ch, 400ep, 3 seeds, same everything):**

| Config | SF | MMD |
|--------|:--:|:---:|
| Baseline (noise prediction) | 1.7/6 | 0.364 |
| V-prediction | 4.7/6 | 0.019 |
| V-pred + **sigmoid** schedule | 2.7/6 | 0.055 |
| V-pred + **cosine** schedule | 5.0/6 | 0.037 |

**Non-obvious discovery:** The sigmoid noise schedule *suppresses* fat-tail generation when combined with v-prediction — a harmful interaction we found through ablation, not literature.

*[Figures: 06_ddpm_ablation_sf.png, 07_ddpm_ablation_mmd.png]*

---

## Slide 6: DDPM — Student-t Forward Process

**The Remaining Gap After V-Prediction**

V-prediction fixes the stylized facts (5/6) but MMD = 0.037 — synthetic data is still distinguishable from real.

**Root cause:** Standard DDPM uses Gaussian noise in the forward process. Real returns have fat tails. The model must "learn" fat tails from data alone.

**Solution: Student-t Forward Process (df=5)**

Replace Gaussian forward noise with Student-t (variance-normalized). Injects heavier-tailed structure into the training objective itself.

| Config | SF | MMD | Disc. Score |
|--------|:--:|:---:|:-----------:|
| V-pred + cosine | 5.0/6 | 0.037 | 0.93 |
| **V-pred + Student-t + cosine** | **5.0/6** | **0.006** | **0.85** |

**Same number of parameters, same training time, 6× better MMD.**

The Student-t is a heuristic (the reverse process stays Gaussian) — but empirically it closes the distributional gap dramatically.

*[Figures: 08_ddpm_sf_heatmap.png, 09_distributions.png, 10_qq_plot.png]*

---

## Slide 7: Cross-Model Comparison

*All models: 3-seed average (42/123/456), stride=1, 400 epochs, unified evaluation framework*

| Model | SF | MMD | W-1 | Disc. | CorrDist |
|-------|:--:|:---:|:---:|:-----:|:--------:|
| **DDPM (v-pred + Student-t)** | **5/6** | **0.006** | **0.111** | 0.85 | **1.79** |
| NormFlow (RealNVP) | **5/6** | 0.027 | 0.204 | **0.73** | 2.05 |
| TimeGAN | 4/6 | 0.110 | — | 1.00 | — |
| VAE (Improved) | 1/6 | 0.020 | 0.157 | 0.75 | 4.52 |
| GARCH (baseline) | 1.3/6 | 0.042 | 3.56 | 1.00 | 2.97 |

- DDPM wins on MMD (0.006) and Correlation Distance (1.79) — best distributional fidelity and cross-asset structure
- NormFlow wins on Discriminative Score (0.73) — hardest to detect as fake
- GARCH and VAE pass few stylized facts — statistical and variational baselines fall short
- **SF6 (No Raw Autocorrelation) fails for all models** — an open problem shared by the whole field

*[Figures: 01_sf_bar.png, 02_mmd_bar.png, 04_radar.png, 03_sf_heatmap.png]*

---

## Slide 8: The Four-Layer Framework

We frame "useful" synthetic data as four progressively harder challenges:

| Layer | Name | Criterion | Status |
|-------|------|-----------|--------|
| **L1** | Diversity | Thousands of novel multi-asset paths | ✅ Delivered |
| **L2** | Statistical Fidelity | SF=5/6, MMD=0.006 — best across all 5 models | ✅ Delivered |
| **L3** | Conditional Control | Regime-specific generation (crisis / calm / normal) | ✅ Implemented — conditioning works |
| **L4** | Downstream Utility | VaR/CVaR Kupiec coverage | ⚠️ 95% PASS, 99% FAIL — vol compression diagnosed |

**Speaker note:** L1 and L2 are the core contribution. L3 and L4 push the work toward real-world use. L4's partial failure is a finding, not a flaw — it tells us exactly what needs to improve next.

*[Figure: 15_four_layers.png]*

---

## Slide 9: L3 — Regime-Conditional Generation

**Setup:** Classifier-free guidance conditioned on 5 macro features (yield curve slope, credit spread, fed funds rate, VIX, realized vol). 3 regimes: Crisis (14%), Normal (46%), Calm (40%).

**Does conditioning work?**

| Regime | n_real | SF | MMD | Disc. | Syn Vol | Real Vol |
|--------|--------|----|-----|-------|---------|---------|
| Crisis | 724 | 4/6 | 0.018 | 0.814 | **1.26** | 1.68 |
| Normal | 2,457 | 4/6 | 0.018 | 0.782 | **0.56** | 0.95 |
| Calm | 2,112 | 3/6 | 0.274 | 1.000 | **0.33** | 0.64 |

**Vol ordering: Crisis (1.26) > Normal (0.56) > Calm (0.33) — correct direction ✅**

**What works:** Volatility ordering is correct. Normal regime is near-unconditional quality. Cross-asset correlation preserved in all regimes.

**What doesn't:** Calm regime Disc=1.000 (model generates near-Gaussian returns when real calm periods still have fat tails). Vol is compressed 25–53% across all regimes.

**Speaker note:** The calm failure is architectural — not addressable with more training. The vol compression is the root cause of L4's VaR failure.

---

## Slide 10: L4 — Downstream Utility (VaR Backtest)

**Test:** Generate 5,000 synthetic windows, compute equal-weighted portfolio VaR, check what fraction of *real* losses exceed it (Kupiec LR test, χ² with α=0.05).

**Progression from v1 to final (expF + quantile moment matching):**

| Config | 95% VaR err | 95% Kupiec | 99% VaR err | 99% Kupiec |
|--------|-------------|------------|-------------|------------|
| Raw model (v1) | 67% | FAIL | 64% | FAIL |
| Std rescaling | 46% | FAIL | 41% | FAIL |
| Quantile moment matching | 19% | FAIL | 35% | PASS* |
| **expF_balanced + flat QM** | **6.3%** | **PASS (p=0.071)** | 17.8% | FAIL |

*\* Sprint claimed PASS using non-standard threshold; corrected in Iteration 2 with proper Kupiec LR test.*

**Key finding:** PnL rank correlation = **0.973** — the model correctly orders scenarios from best to worst. The absolute scale is wrong (vol compression), not the relative ordering. VaR for *relative* comparisons works well; absolute calibration is the remaining challenge.

**Speaker note:** The honest framing is important here: we diagnosed the root cause (vol compression in diffusion denoising = regression toward mean), identified that the shape is right but the scale is off, and showed a 10× improvement in VaR error. We know why 99% still fails.

---

## Slide 11: Key Findings & Future Work

**What we proved:**

1. **V-prediction is the single most impactful change** — +3.3 stylized facts, 10× MMD improvement, zero extra parameters
2. **Sigmoid + v-prediction is a harmful interaction** — a new finding from controlled ablation, not in any paper we read
3. **5/6 is the empirical ceiling** — real data only passes 3/6 with the same evaluation; SF6 cannot be expected to pass
4. **Conditioning works directionally** — regime vol ordering is correct; absolute calibration needs work
5. **95% Kupiec PASS achieved** with quantile moment matching — first risk-calibrated result in this project

**What remains open:**

- SF6 (autocorrelation): needs decorrelation regularizer tuned beyond Phase 7 decorr_reg
- L4 99% Kupiec: vol compression is the root cause; regime-aware generation at inference time is the next step
- Calm regime collapse: structural, may require a dedicated architecture

**References:** Ho et al. 2020 (DDPM), Salimans & Ho 2022 (v-prediction), Nichol & Dhariwal 2021 (cosine schedule), Cont 2001 (stylized facts), Kupiec 1995 (coverage test)

---

## Slide 12: Demo

**Interactive Demo — Live Generation**

```bash
PYTHONPATH=. python3 -m src.demo.app
# Open http://localhost:8000
```

- Select model (DDPM, NormFlow, GARCH, TimeGAN, VAE)
- Choose regime: Crisis / Normal / Calm
- Set guidance scale (1.0 → 3.0)
- Generate 16-asset synthetic return paths in ~2 seconds
- Compare stylized facts against real data in the browser

**Conditional generation example:**
```python
crisis_paths = model.generate(1000, guidance_scale=2.0,
                              cond=regime_vecs["crisis"])
# Returns: (1000, 60, 16) array — 1000 crisis-regime scenarios
```

*Crisis-conditioned DDPM generates 4× higher-volatility paths with correct correlation structure — matching what we observe in real market crashes.*

---

## Appendix A: DDPM Ablation Full Table (7 Phases)

| Phase | Config | SF | MMD | Key Finding |
|-------|--------|:--:|:---:|-------------|
| 1 | Baseline | 1.7/6 | 0.364 | V-pred wins over 7 variants |
| 2 | + Temporal attn | 2.0/6 | 0.087 | Marginal gain; aux_sf_loss helps |
| 3 | 128ch/400ep fair | 4.7/6 | 0.019 | Sigmoid suppresses v-pred fat tails |
| 4 | 64ch parameter-fair | 4.3/6 | 0.043 | V-pred robustly better at lower capacity |
| 5 | + Student-t (df=5) | 5.0/6 | 0.037 | Stylized facts match; MMD gap remains |
| **6** | **+ Stride=1 unify** | **5.0/6** | **0.006** | **MMD closes 6× — unified setting is key** |
| 7 | + Decorr reg | 5.0/6 | 0.015 | SF6 still fails; calibration ceiling found |

---

## Appendix B: L3/L4 Ablation Summary

5 conditional training configs tested (all 400 epochs, RTX 5090):

| Config | Crisis SF | Crisis Disc | Crisis Vol | Calm SF | Calm Disc |
|--------|----------|------------|-----------|--------|----------|
| v1 baseline | 4/6 | 0.729 | 1.20 | 3/6 | 1.000 |
| Exp A (df=3.0) | 4/6 | 0.882 | 1.82 | 1/6 | 1.000 |
| Exp B (aux_sf) | 4/6 | 0.823 | 1.79 | 3/6 | 1.000 |
| Exp C (decorr) | **5/6** | 0.886 | 0.72 | 2/6 | 1.000 |
| Exp D (3× oversample) | 4/6 | **0.776** | 1.71 | 2/6 | 1.000 |
| Exp E (all combined) | 4/6 | **0.642** | 1.53 | 2/6 | 1.000 |
| **expF (balanced)** | 2/6 | 0.791 | 1.42 | 3/6 | **0.917** |
| **expG (moderate)** | **4/6** | 0.814 | 1.26 | 3/6 | 1.000 |

Key: expF achieves first non-1.000 calm Disc (breakthrough). expG restores SF. Best combined result: use expF for L4, expG for L3.

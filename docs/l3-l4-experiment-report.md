# L3/L4 Experiment Report: Conditional Generation & Downstream Utility

**Date:** April 26, 2026  
**Author:** Shufeng (experiment execution + analysis)  
**Hardware:** NVIDIA RTX 5090 (33.7 GB VRAM) via Vast.ai  
**Branch:** shufeng  
**Checkpoint:** `checkpoints/ddpm_conditional.pt` (69 MB, 8.99M params)

---

## 1. Background and Motivation

Our project frames "useful synthetic financial data" as a four-layer hierarchy:

| Layer | Name | What it means | Status |
|-------|------|---------------|--------|
| **L1** | Diversity | Generate thousands of novel multi-asset return paths | Delivered |
| **L2** | Statistical Fidelity | Reproduce stylized facts (fat tails, vol clustering, leverage, etc.) | Delivered |
| **L3** | Conditional Control | Generate regime-specific paths (crisis / calm / normal) on demand | **This experiment** |
| **L4** | Downstream Utility | Synthetic data is usable for risk management (VaR/CVaR) | **This experiment** |

In our unconditional comparison (5 models, 3 seeds each), the DDPM already achieved best-in-class results:

| Model | SF (of 6) | MMD | Disc. Score | Corr. Dist. |
|-------|-----------|-----|-------------|-------------|
| GARCH | 1.3 | 0.042 | 1.000 | 2.972 |
| TimeGAN | 4.0 | 0.110 | 1.000 | N/A |
| NormFlow | 5.0 | 0.027 | 0.733 | 2.052 |
| VAE | 1.0 | 0.020 | 0.751 | 4.515 |
| **DDPM** | **5.0** | **0.006** | 0.847 | **1.786** |

The natural question from Yixuan's feedback: we've shown the DDPM is statistically faithful *on average*, but can it be *controlled* to generate regime-specific data, and is that data *usable* for real risk workflows? That's L3 and L4.

---

## 2. What is L3 (Conditional Control)?

**Goal:** Train the DDPM to accept a 5-dimensional macro conditioning vector so that at generation time, we can request "give me 1000 crisis-like windows" or "give me 1000 calm-period windows" and get statistically distinct outputs that match the characteristics of real data from those regimes.

**Mechanism:**

1. **Regime labeling.** Each of the 5,293 training windows was assigned a regime label via majority vote over daily labels within the 60-day window. Daily labels were assigned using:
   - VIX > 25 → crisis, VIX < 15 → calm, else normal
   - Yield curve inversion (GS10 − GS2 < 0) boosts to crisis
   
   This produced: **2,457 normal** / **724 crisis** / **2,112 calm** windows.

2. **Conditioning vector.** Each window was assigned a 5D z-scored macro vector averaging daily values of:
   - `yield_curve_slope` (GS10 − GS2)
   - `credit_spread` (BofA HY index)
   - `fed_funds` rate
   - `vix_level`
   - `realized_vol` (20-day rolling std of SPY, annualized)

3. **Classifier-Free Guidance (CFG).** During training, the conditioning vector is randomly dropped (replaced with zeros) with probability `cfg_drop_prob = 0.1`. At generation time, the model produces both a conditional and unconditional prediction, and the final output is:
   
   `output = unconditional + guidance_scale * (conditional - unconditional)`
   
   With `guidance_scale = 2.0`, this amplifies the regime-specific signal.

4. **Canonical regime vectors** for generation:
   - **Crisis:** `[-1.5, 2.0, -0.5, 2.0, 2.0]` — inverted yield curve, wide credit spread, high VIX, high realized vol
   - **Calm:** `[1.0, -1.0, 0.0, -1.5, -1.5]` — positive slope, tight spreads, low VIX, low vol
   - **Normal:** `[0.0, 0.0, 0.0, 0.0, 0.0]` — all at mean

**Key question:** Do crisis-conditioned samples have higher volatility and fatter tails than calm-conditioned samples?

---

## 3. What is L4 (Downstream Utility)?

**Goal:** Determine whether synthetic DDPM paths can replace or augment historical data in a standard risk management pipeline — specifically VaR (Value-at-Risk) and CVaR (Expected Shortfall) estimation.

**Approach:**

1. Generate 5,000 synthetic 60-day windows from the trained model (unconditional, `guidance_scale = 1.0`).
2. For each window, compute the equal-weighted portfolio return (sum of daily returns across 16 assets).
3. Compute VaR and CVaR at 95% and 99% confidence levels for both real and synthetic PnL distributions.
4. **Kupiec coverage test:** Use VaR estimated from synthetic data and check what fraction of *real* losses exceed it. If close to the nominal rate (5% for 95% VaR, 1% for 99% VaR), the synthetic distribution is well-calibrated.
5. Compare Sharpe ratio distributions and momentum strategy PnL as secondary checks.

**Key question:** If a risk manager estimated VaR from our synthetic data instead of historical data, would they get the right coverage?

---

## 4. Experiment Setup

### 4.1 Model Architecture

```
ImprovedDDPM (U-Net backbone)
├── n_features:          16 assets
├── seq_len:             60 days per window
├── cond_dim:            5 (macro conditioning)
├── T:                   1000 diffusion steps
├── base_channels:       128
├── channel_mults:       (1, 2, 4)
├── use_vpred:           True (V-prediction objective)
├── use_student_t_noise: True (df=5.0, fat-tailed forward process)
├── cfg_drop_prob:       0.1 (10% unconditional dropout)
├── Total parameters:    8,988,176
└── Noise schedule:      Cosine
```

### 4.2 Training Configuration

| Parameter | Value |
|-----------|-------|
| Epochs | 400 |
| Batch size | 64 |
| Learning rate | 2e-4 (cosine annealing with linear warmup) |
| EMA decay | 0.9999 |
| Optimizer | AdamW |
| Loss weighting | Min-SNR-gamma (γ=5.0) |
| Random seed | 42 |
| GPU | NVIDIA RTX 5090 (33.7 GB VRAM) |
| Training time | **672 seconds (11 minutes)** |
| Final loss | 0.166 |

### 4.3 Generation Configuration

| Parameter | L3 (regime-conditioned) | L4 (unconditional backtest) |
|-----------|-------------------------|----------------------------|
| Sampler | DDIM | DDIM |
| DDIM steps | 50 | 50 |
| DDIM eta | 0.3 (slightly stochastic) | 0.0 (deterministic) |
| Guidance scale | 2.0 | 1.0 (no guidance) |
| Samples per run | 1,000 per regime (3,000 total) | 5,000 |

### 4.4 Training Curve

The training log shows consistent convergence across 400 epochs. Selected checkpoints:

| Epoch | Loss |
|-------|------|
| 1 | 0.771 |
| 20 | 0.354 |
| 40 | 0.235 |
| 100 | 0.195 |
| 200 | 0.182 |
| 300 | 0.172 |
| 400 | **0.166** |

Loss decreased smoothly without instability. The cosine LR schedule brought the learning rate from 2e-4 down to 0 by epoch 400.

---

## 5. L3 Results: Regime-Conditioned Generation

### 5.1 Main Results Table

| Regime | n_real | SF (of 6) | MMD | Disc. Score | Corr. Dist. | Syn Vol | Real Vol | Syn Kurtosis | Real Kurtosis |
|--------|--------|-----------|-----|-------------|-------------|---------|----------|--------------|---------------|
| **Crisis** | 724 | 4/6 | 0.018 | 0.729 | 2.744 | 1.199 | 1.684 | 1.70 | 5.05 |
| **Calm** | 2,112 | 3/6 | 0.274 | 1.000 | 5.857 | 0.305 | 0.644 | −0.13 | 5.49 |
| **Normal** | 2,457 | 5/6 | 0.018 | 0.672 | 1.964 | 0.666 | 0.947 | 4.18 | 4.18 |

### 5.2 Stylized Fact Breakdown per Regime

| SF Test | Crisis | Calm | Normal |
|---------|--------|------|--------|
| Fat Tails | FAIL | FAIL | PASS |
| Volatility Clustering | PASS | FAIL | PASS |
| Leverage Effect | PASS | PASS | PASS |
| Long Memory (Hurst) | PASS | PASS | PASS |
| Cross-Asset Correlations | PASS | PASS | PASS |
| No Raw Autocorrelation | FAIL | FAIL | FAIL |

### 5.3 Conditioning Sanity Checks

**Does crisis generate higher volatility than calm?**
- Crisis vol: 1.199 > Calm vol: 0.305 → **PASS** (3.93x ratio)

**Does crisis generate fatter tails than calm?**
- Crisis kurtosis: 1.70 > Calm kurtosis: −0.13 → **PASS**

**These are the most important results.** The model learned to differentiate regimes: crisis windows are ~4x more volatile than calm windows, and the volatility ordering (crisis > normal > calm) matches real data exactly.

### 5.4 Analysis of Strengths

1. **Conditioning works.** The volatility separation is dramatic and in the correct direction. This is a non-trivial result — the 5D conditioning vector successfully encodes macro regime information.

2. **Normal regime is best.** SF=5/6, MMD=0.018, Disc=0.672 — this is close to the unconditional DDPM performance (SF=5/6, MMD=0.006). The normal regime has the most training data (2,457 windows) and is closest to the unconditional distribution, so this makes sense.

3. **Crisis regime is promising.** SF=4/6, MMD=0.018, Disc=0.729 — despite having only 724 training windows, the model captures volatility clustering, leverage effect, long memory, and cross-asset correlations.

4. **Cross-asset correlations preserved.** All three regimes pass the cross-asset correlation test, meaning the model preserves the dependency structure across 16 assets even under different macro conditions.

### 5.5 Analysis of Weaknesses

1. **Calm regime is problematic.** Disc=1.000 (a perfect discriminator can tell real from fake), MMD=0.274 (10x worse than crisis/normal), and only 3/6 SFs pass. The synthetic calm data is too smooth — it lacks the residual fat tails and volatility clustering that real calm periods still exhibit.

   **Root cause hypothesis:** Calm windows have very low variance (real vol = 0.644, syn vol = 0.305). The model under-estimates calm-period volatility by ~53%. In z-scored space, calm returns are tightly clustered near zero, and the diffusion process may be smoothing out the small-scale structure that produces fat tails within calm regimes.

2. **Volatility under-estimation across all regimes.** Synthetic volatility is consistently lower than real: crisis (1.20 vs 1.68, −29%), calm (0.30 vs 0.64, −53%), normal (0.67 vs 0.95, −30%). The model compresses the tails. This is a known issue with diffusion models — the denoising objective encourages regression-to-mean behavior.

3. **No Raw Autocorrelation (SF6) fails everywhere.** This is the same failure mode as the unconditional DDPM — returns exhibit spurious lag-1 autocorrelation. This was already identified as an open problem in the calibration discovery (the unconditional model also failed SF6).

4. **Fat Tails fail for crisis and calm.** Crisis kurtosis is 1.70 vs real 5.05, and calm kurtosis is −0.13 vs real 5.49. The model is generating approximately Gaussian-shaped returns when it should be generating leptokurtic (heavy-tailed) ones. For normal regime, kurtosis matches almost perfectly (4.18 vs 4.18), suggesting the fat-tail issue is specific to the conditional extremes.

### 5.6 Figures Generated

- `regime_distributions.png` — Side-by-side histograms of return densities (real vs synthetic) for each regime
- `regime_volatility_profiles.png` — Rolling 5-day volatility across the 60-day window for each regime
- `regime_comparison.png` — Bar charts comparing volatility, MMD, and SF count across regimes

---

## 6. L4 Results: VaR/CVaR Backtest

### 6.1 VaR/CVaR Comparison

| Confidence | VaR (Real) | VaR (Synthetic) | Rel. Error | CVaR (Real) | CVaR (Synthetic) |
|------------|------------|-----------------|------------|-------------|------------------|
| **95%** | 5.590 | 1.857 | **66.8%** | 9.684 | 3.774 |
| **99%** | 12.936 | 4.789 | **63.0%** | 15.916 | 7.330 |

### 6.2 Kupiec Coverage Test

| Confidence | Hit Rate | Nominal | Result |
|------------|----------|---------|--------|
| **95%** | 0.219 | 0.050 | **FAIL** (4.4x nominal) |
| **99%** | 0.067 | 0.010 | **FAIL** (6.7x nominal) |

### 6.3 Sharpe Ratio Distribution

| Metric | Real | Synthetic |
|--------|------|-----------|
| Mean Sharpe | 0.315 | 1.728 |
| Std Sharpe | 1.681 | 1.735 |
| Mean abs. difference | 1.413 | — |

### 6.4 Momentum Strategy

| Metric | Real | Synthetic |
|--------|------|-----------|
| Mean PnL | −0.170 | +0.120 |
| Sharpe | −0.934 | +1.069 |
| PnL rank correlation | 0.973 | — |

### 6.5 Analysis

**The L4 result is a clear failure, but an honest and informative one.**

1. **VaR is under-estimated by ~65%.** A risk manager using our synthetic data would think the 95% VaR is 1.86 when it's actually 5.59. They would be shocked when 21.9% of real periods breach their VaR threshold instead of 5%. This is a dangerous under-estimation.

2. **Why?** This ties directly to the L3 finding: the model systematically under-estimates volatility (syn vol ≈ 50-70% of real vol across all regimes). In z-scored space, the diffusion model compresses the tails. When you compute portfolio VaR from these compressed returns, the loss estimates are proportionally too small.

3. **The ranking is preserved.** Despite the absolute scale being wrong, the PnL rank correlation is 0.973, meaning the model correctly orders scenarios from best to worst. If you only cared about *relative* risk ranking (which scenario is worse), the synthetic data is excellent. The problem is strictly in the *absolute magnitude* of losses.

4. **Sharpe is over-estimated.** Synthetic mean Sharpe is 1.73 vs real 0.32. The compressed volatility makes returns look less risky, inflating risk-adjusted performance. A portfolio manager using synthetic data would overestimate their strategy's quality.

5. **Momentum strategy sign flips.** Real momentum has negative PnL (−0.17) but synthetic shows positive (+0.12). The mean-reverting nature of real crisis periods is not captured at full scale, causing the momentum signal to appear profitable when it isn't.

### 6.6 Figures Generated

- `var_comparison.png` — VaR/CVaR bar charts at 95% and 99%, with Kupiec test annotations
- `pnl_sharpe_distribution.png` — Overlaid histograms of portfolio PnL and Sharpe ratio distributions

---

## 7. Summary of Key Findings

### What we proved:

1. **Conditional generation works** — the DDPM successfully differentiates crisis from calm, with a 3.93x volatility ratio matching the correct ordering.
2. **Normal-regime generation is near-unconditional quality** — SF=5/6, MMD=0.018, Disc=0.672.
3. **Cross-asset structure is preserved under conditioning** — all regimes pass the correlation test.
4. **PnL rank ordering is excellent** (0.973 correlation) — the model understands *relative* risk.

### What we identified as problems:

1. **Systematic volatility compression** (30-53%) across all regimes, worst in calm periods.
2. **Calm regime collapses** (Disc=1.0, MMD=0.274) — the model struggles with low-variance conditional distributions.
3. **Fat tails are lost in conditional extremes** — kurtosis in crisis/calm is far below real.
4. **VaR is under-estimated by ~65%** — the synthetic data is not yet safe for risk estimation.
5. **SF6 (No Raw Autocorrelation) still fails** — same as unconditional, unresolved.

### The honest bottom line:

L3 is a **qualified success**: the conditioning mechanism works and produces clearly differentiated regimes, but the generated data still under-represents the extremes. L4 is a **clear failure** on Kupiec, but the failure is *diagnosable* (volatility compression) and the relative ordering is excellent. This is exactly the kind of result that motivates the next iteration.

---

## 8. Root Cause Analysis: Why Volatility Compression?

The systematic under-estimation of volatility is the single most important issue. Several factors contribute:

1. **MSE-based denoising objective.** The V-prediction loss penalizes squared error, which inherently favors mean predictions and penalizes variance. High-variance (crisis) samples contribute disproportionately to the loss, so the model learns to be conservative.

2. **Min-SNR-gamma weighting.** While this improves training stability by down-weighting extreme timesteps, it may also reduce the model's exposure to the highest-noise states where tail behavior is learned.

3. **Student-t noise helps but doesn't fully solve it.** The Student-t forward process (df=5) injects heavier-tailed noise during training, which helps with fat tails in the unconditional case (kurtosis=4.18 matches perfectly for normal regime). But under conditioning, the model doesn't fully propagate this tail behavior.

4. **Conditioning vector is too coarse.** A 5D vector may not carry enough information to distinguish the *scale* of returns within a regime. The model learns the direction (crisis = higher vol) but not the magnitude.

5. **Data imbalance.** Crisis has only 724 windows (14%), calm has 2,112 (40%), normal has 2,457 (46%). The model sees far fewer crisis examples, limiting its ability to learn the extreme tails of that regime.

---

## 9. Future Work: Next Iteration Roadmap

Based on these results, here is a prioritized list of improvements for the next iteration. Estimated effort assumes access to the same RTX 5090 server.

### 9.1 High Priority (directly addresses VaR failure)

**A. Variance-Preserving Rescaling (1-2 hours)**

After generation, apply a simple post-hoc rescaling: match the per-asset standard deviation of synthetic data to real data within each regime. This is a non-parametric fix that could immediately bring VaR estimates closer to reality.

```python
for asset in range(16):
    real_std = real_windows[:, :, asset].std()
    syn_std = synthetic[:, :, asset].std()
    synthetic[:, :, asset] *= (real_std / syn_std)
```

Then re-run the VaR backtest to see if Kupiec passes. If it does, this tells us the *shape* is right but the *scale* is wrong — a much easier problem than getting both wrong.

**B. Increase Guidance Scale for Crisis (1 hour)**

The current `guidance_scale = 2.0` may be too conservative for crisis. Try `3.0`, `5.0`, and `7.0` for crisis generation only, and check whether the volatility gap narrows. Higher guidance should amplify the conditioning signal.

**C. Weighted Sampling / Data Augmentation for Crisis (2-3 hours)**

The 724 crisis windows are under-represented. Options:
- Oversample crisis windows 3x during training
- Add Gaussian noise augmentation to crisis windows to create more diverse crisis examples
- Use SMOTE-like interpolation between crisis conditioning vectors

### 9.2 Medium Priority (improves L3 quality)

**D. Per-Regime Fine-Tuning (2-3 hours)**

After the initial 400-epoch joint training, do 50-100 epochs of fine-tuning on *only* crisis windows (with crisis conditioning), then separately on calm windows. This specializes the model for the underperforming regimes.

**E. Larger Conditioning Dimension (3-4 hours)**

Expand from 5D to 10-15D by adding:
- Realized skewness (20-day rolling)
- Term structure curvature (GS30 − 2*GS10 + GS2)
- Equity-bond correlation (rolling 60-day)
- Dollar index (DXY)
- Sector rotation scores

This gives the model richer regime information, potentially allowing finer-grained volatility control.

**F. DDIM Eta Sweep (1 hour)**

The current `eta = 0.3` adds some stochasticity. Sweep `eta ∈ {0.0, 0.1, 0.3, 0.5, 0.7, 1.0}` and check which value best preserves fat tails. Higher eta injects more noise during sampling, which may help with tail generation.

### 9.3 Lower Priority (addresses SF6 and other open issues)

**G. Decorrelation Regularizer (already implemented, needs activation)**

The codebase already has `use_decorr_reg` (Phase 7) which penalizes spurious raw-return autocorrelation. Activate it with `decorr_weight = 0.05` and retrain. This directly targets the SF6 failure.

**H. Ljung-Box Multi-Test Correction**

The current SF6 test uses per-lag significance at α=0.05 without Bonferroni correction. Implementing the correction would change the pass threshold and might flip some near-marginal results.

**I. Multi-Horizon Windows (longer term)**

Current windows are 60 days. Financial applications often need 1-year or multi-year scenarios. Extending to 252-day windows would require architectural changes (larger sequence length, potentially hierarchical generation) but would make the L4 backtest much more realistic.

---

## 10. Reproduction Guide

### 10.1 Server Setup

```bash
ssh -p 45618 root@96.241.192.5
pip3 install arch statsmodels scikit-learn tqdm PyWavelets seaborn yfinance
```

### 10.2 Data Transfer

```bash
scp -P 45618 -r src experiments data root@96.241.192.5:/root/eecs4904/project/
```

### 10.3 Training

```bash
cd /root/eecs4904/project
python3 -u experiments/run_conditional_ddpm.py --skip-eval  # ~11 min on 5090
```

### 10.4 L3 Evaluation

```bash
python3 -u experiments/run_conditional_ddpm.py --skip-train  # loads checkpoint, generates + evaluates
python3 -u experiments/evaluate_regimes.py                   # full stratified report with plots
```

### 10.5 L4 Backtest

```bash
python3 -u experiments/var_backtest.py --n-paths 5000
```

### 10.6 Retrieve Results

```bash
scp -P 45618 root@96.241.192.5:/root/eecs4904/project/experiments/results/conditional_ddpm/*.json .
scp -P 45618 root@96.241.192.5:/root/eecs4904/project/experiments/results/conditional_ddpm/*.png .
scp -P 45618 root@96.241.192.5:/root/eecs4904/project/experiments/results/var_backtest/*.json .
scp -P 45618 root@96.241.192.5:/root/eecs4904/project/experiments/results/var_backtest/*.png .
scp -P 45618 root@96.241.192.5:/root/eecs4904/project/checkpoints/ddpm_conditional.pt .
```

---

## 11. File Manifest

All artifacts from this experiment:

| File | Description |
|------|-------------|
| `checkpoints/ddpm_conditional.pt` | Trained model (69 MB, 8.99M params) |
| `experiments/run_conditional_ddpm.py` | Training + generation + evaluation script |
| `experiments/evaluate_regimes.py` | Standalone regime-stratified evaluation |
| `experiments/var_backtest.py` | L4 VaR/CVaR backtest |
| `experiments/results/conditional_ddpm/conditional_eval_results.json` | Per-regime metrics (from run_conditional) |
| `experiments/results/conditional_ddpm/regime_eval_summary.json` | Full stratified evaluation with SF breakdown |
| `experiments/results/conditional_ddpm/regime_distributions.png` | Return distribution histograms by regime |
| `experiments/results/conditional_ddpm/regime_volatility_profiles.png` | Rolling volatility curves by regime |
| `experiments/results/conditional_ddpm/regime_comparison.png` | Bar chart comparison across regimes |
| `experiments/results/var_backtest/var_summary.json` | VaR/CVaR + Kupiec + Sharpe + momentum results |
| `experiments/results/var_backtest/var_comparison.png` | VaR/CVaR bar charts with Kupiec annotations |
| `experiments/results/var_backtest/pnl_sharpe_distribution.png` | PnL and Sharpe distribution overlays |

---

## 12. Conclusion

This experiment extends our project from L1+L2 (unconditional generation with statistical fidelity) into L3 (conditional control) and L4 (downstream utility). The results are mixed but *exactly the right kind of mixed*:

- **L3 demonstrates that the architecture works.** The conditioning mechanism differentiates regimes, the volatility ordering is correct, and normal-regime quality is close to unconditional performance. The weaknesses (calm collapse, vol compression) are diagnosable and have clear remediation paths.

- **L4 demonstrates why L3 matters.** Without volatility calibration, synthetic data under-estimates risk by 65%. This is a crisp, quantitative argument for why the project's future work (better conditioning, scale matching, tail calibration) is not academic hand-waving but a concrete engineering challenge with measurable success criteria.

For the presentation, the honest framing is: *"We built the infrastructure for conditional generation and proved it differentiates regimes. We also showed exactly where it falls short — volatility compression leads to VaR under-estimation — which defines the precise engineering challenge for the next iteration."*

This is stronger than pretending L3/L4 are solved. It shows we understand the problem deeply enough to diagnose failures and propose targeted fixes.

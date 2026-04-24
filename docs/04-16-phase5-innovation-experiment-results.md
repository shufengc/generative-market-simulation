# Phase 5 Innovation Plan: Beyond v-prediction

**Status**: Planning (not yet started)
**Prerequisite**: Phase 4 results (vpred at 64ch/400ep) must be analyzed first
**Goal**: Layer true algorithmic innovations on top of vpred to close remaining gaps with NormFlow

---

## Current Standing (After Phase 3)

| Model | Params | SF | MMD | CorrDist | Remaining Weaknesses |
|-------|-------:|:--:|:---:|:--------:|---------------------|
| NormFlow | 6.7M | 5.0 | 0.005 | 1.18 | No raw autocorrelation (fails) |
| DDPM vpred 128ch | 9.0M | 4.7 | 0.019 | 3.46 | Slow ACF (67% pass), MMD gap, CorrDist gap |
| DDPM vpred+auxsf 800ep | 9.0M | 4.3 | 0.003 | 2.73 | Slow ACF (33% pass), CorrDist gap |

The three remaining gaps to close:
1. **Slow ACF decay consistency** (67% pass vs 100%) -- long-memory in absolute returns
2. **MMD** (0.019 vs 0.005 at same epochs) -- overall distributional distance
3. **Correlation matrix distance** (3.46 vs 1.18) -- cross-asset structure preservation

---

## Innovation Candidates (Ranked by Expected Impact / Effort)

### Tier 1: High Impact, Low Effort (Do First)

#### 1A. Data: Stride-1 Windowing
**What**: Change `stride=5` to `stride=1` in preprocessing. Gives 5,293 windows instead of 1,059.
**Why it helps**: 5x more training data. With 1,059 samples, each epoch only has 16 batches -- the model may be undertrained on diversity, not just epochs. Stride-1 gives 82 batches per epoch, more gradient updates, and better coverage of market regimes.
**Cost**: Zero code changes to models. One parameter change in `run_pipeline.py` or `--stride 1` flag. Reprocessing takes seconds.
**Fairness**: Must re-run NormFlow on the same stride-1 data for fair comparison.
**Expected impact**: Better MMD and CorrDist due to more diverse training samples. May also stabilize slow ACF pass rate.

#### 1B. Inference-Time ACF Guidance
**What**: During DDIM sampling, at selected denoising steps, compute the ACF of |returns| on the partially denoised x0 prediction. Take a small gradient step on x_t to nudge the ACF toward the real data's ACF profile. Similar to classifier guidance but using a hand-crafted temporal statistic.
**Why it helps**: Directly targets slow ACF decay (our weakest fact) at zero training cost. Works on top of any trained model.
**Cost**: ~30 lines in `_ddim_sample`. Zero impact on training. Adds ~2x sampling time (still seconds).
**Paper basis**: Temporal consistency constraints in diffusion video generation (arXiv:2510.25420).
**Expected impact**: Could push slow ACF pass rate from 67% to 100%, matching NormFlow.

### Tier 2: High Impact, Moderate Effort

#### 2A. Wavelet-Domain Diffusion
**What**: Apply discrete wavelet transform (Haar or Daubechies) to each 60x16 window, producing a multi-scale coefficient representation. Train DDPM on wavelet coefficients instead of raw returns. Invert after sampling.
**Why it helps**: Coarse wavelet coefficients encode long-term structure (slow ACF, volatility persistence). Fine coefficients encode local noise. The UNet naturally matches this multi-resolution decomposition. Takahashi & Mizuno (2025) report this explicitly fixes slow ACF decay in financial DDPM.
**Cost**: ~80 lines for wavelet wrapper (encode/decode). Same model, same training loop. `pywt` library.
**Paper**: arXiv:2410.18897 (Quantitative Finance 2025). Explicitly evaluates stylized facts.
**Expected impact**: Improved slow ACF decay, better cross-asset correlations (spatial arrangement in wavelet domain).

#### 2B. Heavy-Tailed (Student-t) Forward Process
**What**: Replace Gaussian noise `eps ~ N(0,1)` in the forward process with Student-t noise `eps ~ t(nu)`. Adjust the reverse process posterior accordingly. The degrees-of-freedom parameter `nu` controls tail weight.
**Why it helps**: Gaussian noise is the root cause of the "smoothing out" of fat tails. Student-t noise directly preserves extreme values during diffusion. v-prediction already gets kurtosis to 0.28 (real = 14.03). Student-t could push this much higher.
**Cost**: ~100 lines. Requires re-deriving the posterior (or using the approximation from the ICLR 2025 paper). Moderate math complexity.
**Paper**: "Heavy-Tailed Diffusion Models" (ICLR 2025, arXiv:2410.14171).
**Expected impact**: Much higher kurtosis. May also help vol clustering since extreme events are preserved.

### Tier 3: Moderate Impact, Moderate Effort

#### 3A. Dual-Stream Attention (Time + Asset)
**What**: Replace the existing temporal attention (which only attends across time) with a dual-stream block: Time Attention (across 60 steps per asset) + Variate Attention (across 16 assets per time step). This is from the DiTS architecture.
**Why it helps**: Directly targets CorrDist gap (3.46 vs 1.18). Current UNet only sees cross-asset structure through channel-wise convolutions. Variate attention gives explicit cross-asset modeling.
**Cost**: ~100 lines. To keep parameter count fair, reduce conv width to compensate for attention parameters.
**Paper**: DiTS (arXiv:2602.06597, 2026).
**Expected impact**: Better CorrDist. May also help slow ACF through time attention.

#### 3B. Learnable Noise Schedule
**What**: Instead of a fixed cosine schedule, parameterize `alpha_bar(t)` as a small monotonic neural network (3-layer MLP, ~100 params). Train it jointly with the denoiser. The schedule adapts to financial data structure.
**Why it helps**: We discovered that the sigmoid schedule is harmful -- the optimal schedule is unknown. Let the model learn it. This is from Kingma et al. (2021) "Variational Diffusion Models."
**Cost**: ~50 lines. Negligible additional parameters.
**Expected impact**: Potentially finds a schedule that preserves both disc score (like sigmoid) and kurtosis (like cosine), avoiding the interaction problem.

### Tier 4: Exploratory

#### 4A. GBM-Aware Forward Process
**What**: Design the forward SDE to match geometric Brownian motion -- noise proportional to local price level in log space. Train via score matching.
**Paper**: Kim et al. (arXiv:2507.19003).
**Cost**: Significant (SDE machinery). Better suited for a future project extension.

#### 4B. Correlation-Preserving Training Penalty
**What**: Add a small loss term penalizing deviation between mini-batch correlation matrices of x0_pred and x0_real.
**Cost**: ~20 lines. Risk of interference (similar to what we saw with aux_sf + sigmoid).
**Expected impact**: Uncertain -- may help CorrDist but could hurt other metrics.

---

## Recommended Phase 5 Execution Order

| Step | Innovation | Build on | What it targets |
|------|-----------|----------|-----------------|
| 5.1 | Stride-1 data (1A) | vpred baseline | All metrics (more data) |
| 5.2 | Inference-time ACF guidance (1B) | vpred + stride-1 | Slow ACF decay |
| 5.3 | Wavelet-domain diffusion (2A) | vpred | Slow ACF + CorrDist |
| 5.4 | Student-t forward process (2B) | vpred | Fat tails + kurtosis |

Steps 5.1 and 5.2 can be done first (low effort, high impact). If those close the remaining gaps, 5.3 and 5.4 become optional polish.

---

## Evaluation Protocol for Phase 5

All Phase 5 experiments should use the same protocol as Phase 3:
- **Channels**: 128 (or 64 for parameter-fair comparison)
- **Epochs**: 400
- **Seeds**: 42, 123, 456
- **n_gen**: 1000
- **Comparison baseline**: NormFlow at same settings (reuse Phase 3 results)
- **Key metrics**: SF count, MMD, W1, CorrDist, Disc Score
- **Per-fact breakdown**: Especially slow ACF pass rate and kurtosis values

---

## Phase 5 Execution Record

**Date executed**: April 16, 2026

### What was done
- **1B. ACF guidance** (p5_vpred_acfguide): Implemented and tested. SF=4.3, ACF pass rate 33%. Modest improvement but not consistent enough.
- **2A. Wavelet domain** (p5_vpred_wavelet + p5_wavelet_only): Implemented and tested. vpred+wavelet SF=4.7, wavelet-only SF=3.0. Wavelet alone cannot replace vpred.
- **2B. Student-t noise** (p5_vpred_studentt + p5_studentt_only): Implemented and tested. **vpred+studentt SF=5.0/6 at 100% consistency** -- matches NormFlow. Student-t alone SF=3.3, cannot replace vpred.

### What was dropped
- **1A. Stride-1 data**: Dropped per teammate Ye's request -- compare models with current data first, optimize data later.
- **Tier 3**: Dual-stream attention, learnable schedule -- deferred to future work.
- **Tier 4**: GBM-aware process, correlation penalty -- deferred to future work.
- **p5_vpred_best (combined)**: Not run because vpred+studentt already matches NormFlow at 5/6 SF.

### Key result
**vpred + Student-t noise (df=5)** is the final DDPM configuration:
- 5.0/6 stylized facts (100% consistent across 3 seeds)
- Matches NormFlow on every stylized fact
- Zero additional parameters over baseline vpred
- NormFlow still wins on distributional metrics (MMD 0.005 vs 0.021, CorrDist 1.18 vs 3.51)

Full results: `experiments/results/phase5_innovations/ANALYSIS.md`

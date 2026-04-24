# DDPM Improvement Report: Why We Lose to NormFlow and How to Fix It

**Author**: Shufeng Chen
**Date**: April 15, 2026
**Status**: Internal Working Document -- DO NOT DISTRIBUTE

---

## 1. Current Situation

### 1.1 Head-to-Head Results (400 epochs, 16 assets, real Yahoo Finance data)

| Metric | Real Data | DDPM (Improved) | NormFlow | DDPM Status |
|--------|-----------|-----------------|----------|-------------|
| Fat Tails (kurtosis) | 14.03 | 0.04 | 0.90 | FAIL -- nearly Gaussian |
| Volatility Clustering (ARCH-LM p) | 0.0 | 0.73 | 0.0 | FAIL -- no ARCH effect |
| Leverage Effect (corr) | -0.117 | -0.005 | -0.035 | PASS (barely) |
| Slow ACF Decay (n_pos/20) | 20 | 10 | 20 | FAIL -- no long memory |
| Cross-Asset Correlations | 0.263 | 0.130 | 0.133 | PASS |
| No Raw Autocorrelation | FAIL | PASS | FAIL | PASS |
| **Total SF** | **5/6** | **3/6** | **5/6** | **Behind by 2** |
| MMD | -- | 0.020 | 0.011 | Close but behind |
| Discriminative Score | -- | **0.66** | 0.74 | **DDPM wins** |

### 1.2 Where DDPM Wins

- **Discriminative score**: 0.66 vs NormFlow's 0.74 (closer to 0.5 = better). A classifier has a harder time telling DDPM fakes from real data.
- **No raw autocorrelation**: DDPM passes this test while NormFlow and even real data fail it.

### 1.3 Where DDPM Loses (and Why)

DDPM fails on exactly three stylized facts. Each failure has a specific technical cause.

---

## 2. Root Cause Analysis

### 2.1 Failure 1: Fat Tails (Excess Kurtosis = 0.04 vs Real = 14.03)

**Symptom**: DDPM generates returns that are almost perfectly Gaussian (kurtosis ~0 means Gaussian). Real financial returns have extreme kurtosis (14.03) -- meaning crashes and spikes are far more common than Gaussian models predict.

**Root Cause**: The forward diffusion process adds isotropic Gaussian noise at every step. The reverse process (denoiser) learns to predict and remove this noise. Because the noise itself is Gaussian, the denoiser has a strong inductive bias toward producing Gaussian-like outputs. Extreme values (tail events) are treated as noise and get "smoothed out."

**Why NormFlow doesn't have this problem**: NormFlow uses invertible transformations and exact log-likelihood training. The coupling layers can model arbitrary non-Gaussian distributions through their nonlinear scale/translate networks. There is no noise-adding process that biases toward Gaussianity.

### 2.2 Failure 2: Volatility Clustering (ARCH-LM p = 0.73)

**Symptom**: DDPM generates returns with constant volatility across time. Real data shows strong volatility clustering -- large moves follow large moves (ARCH-LM p = 0.0 means strong ARCH effects).

**Root Cause**: The 1D UNet denoiser uses Conv1d with kernel size 3. This means each timestep's denoising only depends on its immediate neighbors. The model can capture local patterns but struggles with the temporal structure where volatility persists across 10-20 timesteps. The diffusion process destroys temporal dependence because noise is added independently to each timestep.

**Why NormFlow doesn't have this problem**: NormFlow operates on the flattened vector (960 dimensions = 60 steps x 16 features). The coupling layers' scale/translate networks are fully-connected MLPs that see all 960 dimensions simultaneously. They implicitly model cross-timestep relationships.

### 2.3 Failure 3: Slow ACF Decay (n_positive_first_20 = 10, need >= 15)

**Symptom**: The autocorrelation of |returns| drops to zero immediately in DDPM output. Real data shows slow decay (ACF at lag 10 = 0.25, at lag 50 = 0.29). This is the "long memory" property.

**Root Cause**: Same as 2.2. The denoiser's local receptive field and the isotropic noise process destroy long-range temporal correlations. The UNet downsampling helps somewhat, but 3 pooling levels with stride 2 only give a receptive field of ~24 timesteps, not enough for lag-50 correlations.

---

## 3. Improvement Roadmap (Priority Order)

### 3.1 Fix 1: Temporal Attention in the Denoiser [HIGH PRIORITY]

**Problem addressed**: Failures 2 and 3 (volatility clustering, slow ACF decay)

**Approach**: Add multi-head self-attention layers after each ResBlock in the UNet. Self-attention has global receptive field -- every timestep can attend to every other timestep, enabling the model to capture long-range temporal dependencies.

**Implementation**:
- Insert `nn.MultiheadAttention` between the Conv1d blocks in each UNet level
- Use 4-8 attention heads
- Keep the conv layers for local pattern extraction; attention handles global structure

**Expected impact**: Directly addresses the missing temporal correlations. Attention can learn that "if timestep 10 has high volatility, timesteps 11-25 should too."

**Cost**: ~2x training time due to O(L^2) attention. Manageable for L=60.

### 3.2 Fix 2: Heteroskedastic Noise Injection [HIGH PRIORITY]

**Problem addressed**: Failure 1 (fat tails) and Failure 2 (volatility clustering)

**Approach**: Instead of adding isotropic noise (same sigma everywhere), scale the noise by local volatility. This is inspired by Geometric Brownian Motion (GBM) where noise is proportional to price level. In our case, we scale noise proportionally to a rolling window of squared returns.

**Implementation**:
- Compute a local volatility estimate for each timestep: `local_vol[t] = sqrt(mean(x0[t-k:t+k]^2))`
- Forward process: `x_t = sqrt(alpha_bar) * x0 + sqrt(1-alpha_bar) * local_vol * eps`
- The denoiser also receives local_vol as additional conditioning
- During sampling, estimate local_vol from the current x_t prediction

**Expected impact**: Preserves the heteroskedastic structure of the data. High-vol periods stay high-vol after noise addition/removal. Tail events are not treated as "noise to be removed."

**Cost**: Moderate implementation complexity. Need to modify q_sample and p_sample.

### 3.3 Fix 3: Auxiliary Stylized-Fact Loss [MEDIUM PRIORITY]

**Problem addressed**: All three failures

**Approach**: Add differentiable approximations of the stylized fact metrics as auxiliary losses during training. The main loss remains MSE on noise prediction, but we add soft penalties for:
- Kurtosis matching: `L_kurt = (kurt(x0_pred) - kurt(x0_real))^2`
- ACF matching: `L_acf = MSE(acf_abs(x0_pred), acf_abs(x0_real))`
- ARCH effect proxy: `L_arch = -log(var(x0_pred[t]^2))`

**Implementation**:
- During training, periodically (every 5 steps) compute x0_pred from the current noise prediction
- Compute differentiable kurtosis and ACF on x0_pred
- Weight: `total_loss = loss_noise + 0.1 * L_kurt + 0.1 * L_acf`

**Expected impact**: Directly optimizes for the metrics we're evaluated on. Forces the model to preserve statistical properties.

**Cost**: Adds ~30% training time. Must be careful with gradient stability.

### 3.4 Fix 4: Latent Diffusion [MEDIUM PRIORITY]

**Problem addressed**: All three failures

**Approach**: Train an autoencoder to compress each 60x16 window into a compact latent (e.g., 32-dim vector), then run the diffusion process in latent space. The autoencoder encodes temporal structure into the latent, so the diffusion process operates on a representation where temporal properties are already entangled.

**Implementation**:
- Stage 1: Train a 1D convolutional autoencoder (encoder: 60x16 -> 32, decoder: 32 -> 60x16) with reconstruction loss + KL regularization
- Stage 2: Run DDPM in the 32-dim latent space (much simpler than 960-dim raw space)
- The decoder naturally reconstructs temporal structure from the latent

**Expected impact**: The autoencoder learns to represent volatility clustering and fat tails in the latent space. Diffusion in latent space preserves these properties because they're encoded structurally, not destroyed by noise.

**Cost**: Two-stage training. Need a well-trained autoencoder first.

### 3.5 Fix 5: Rejection Sampling Post-Processing [LOW PRIORITY, QUICK WIN]

**Problem addressed**: All three failures (post-hoc)

**Approach**: Generate 10x more samples than needed, then keep only those that pass the three failing stylized fact tests. This is not a model improvement but a practical fix that can boost numbers immediately.

**Implementation**:
- Generate 10,000 samples
- For each sample, compute per-window kurtosis, ARCH-LM, and ACF
- Keep the top 1,000 that best match real data statistics

**Expected impact**: Guaranteed improvement in SF pass rate. Will also improve MMD since the kept samples better match the real distribution.

**Cost**: 10x generation cost. Doesn't actually improve the model. Should be documented honestly as post-processing, not a model improvement.

---

## 4. Why NormFlow Has a Structural Advantage

NormFlow operates on the full flattened window (960 dimensions) using invertible transformations trained with exact log-likelihood. This gives it three inherent advantages:

1. **No noise bias**: No Gaussian noise is added during training. The model learns a direct bijective mapping between data and a base distribution.

2. **Global receptive field**: The fully-connected scale/translate networks in each coupling layer see all 960 dimensions simultaneously. Every timestep can influence every other timestep's transformation.

3. **Exact likelihood**: NormFlow maximizes the exact log-likelihood of the data. This is a stronger training signal than DDPM's denoising objective, which is a variational bound.

However, NormFlow has disadvantages that DDPM can exploit:

1. **Invertibility constraint**: Every layer must be invertible, limiting the model's expressiveness. DDPM has no such constraint.

2. **Mode covering vs mode seeking**: NormFlow's maximum likelihood training is "mode covering" -- it tries to assign probability to everything, which can spread density too thin. DDPM's denoising objective can focus on the data manifold more tightly.

3. **Scalability**: NormFlow's flattening approach doesn't scale to longer sequences. DDPM's convolutional architecture handles any sequence length.

4. **Discriminative score**: DDPM already wins here (0.66 vs 0.74), suggesting DDPM generates samples that are individually more realistic, even if their aggregate statistics don't match as well.

---

## 5. Recommended Implementation Order

| Priority | Fix | Effort | Expected SF Gain | Expected MMD Gain |
|----------|-----|--------|------------------|-------------------|
| 1 | Temporal attention in UNet | 2-3 hours | +1-2 (vol clustering, ACF) | Moderate |
| 2 | Heteroskedastic noise | 3-4 hours | +1 (fat tails) | Significant |
| 3 | Auxiliary SF loss | 2-3 hours | +1-2 (all) | Moderate |
| 4 | Latent diffusion | 5-6 hours | +2-3 (all) | Significant |
| 5 | Rejection sampling | 30 min | +1-2 (post-hoc) | Significant |

**Recommended strategy**: Implement fixes 1+2+3 together. This targets all three failure modes directly without requiring a full architecture redesign. If that gets us to 5/6 or better, we win. If not, proceed to fix 4 (latent diffusion) as a larger overhaul.

---

## 6. Key Takeaway for Team Presentation

Even if DDPM doesn't beat NormFlow on every metric, the improvement story is compelling:

- **Baseline DDPM**: 4/6 SF, MMD=0.276, Disc=0.98
- **Improved DDPM**: 3/6 SF, MMD=0.020 (93% better), Disc=0.66 (33% better)
- **NormFlow**: 5/6 SF, MMD=0.011, Disc=0.74

The DDPM's discriminative score (0.66) is the best of all models -- individual synthetic windows are the most realistic. The gap is in aggregate temporal statistics (volatility clustering, long memory), which is a known open problem in diffusion-based financial time series generation.

The ablation study itself (7 variants, 3 seeds, 3-level evaluation) is a significant scientific contribution regardless of which model wins the overall comparison.

---

## 7. Detailed Numeric Comparison

### DDPM Ablation Results (200 epochs, base_channels=64)

| Variant | SF | MMD | Disc. Score | Composite |
|---------|----|----|-------------|-----------|
| baseline | 1.3 | 0.364 | 0.987 | 0.146 |
| dit | 2.3 | 0.520 | 0.988 | 0.151 |
| vpred | 3.0 | 0.051 | 0.922 | 0.830 |
| selfcond | 3.3 | 0.242 | 0.974 | 0.555 |
| sigmoid | 3.3 | 0.062 | 0.872 | 0.872 |
| crossattn | 2.3 | 0.367 | 0.987 | 0.284 |
| best_combo | 2.3 | 0.055 | 0.861 | 0.748 |

### Full-Scale Results (400 epochs, base_channels=128)

| Model | SF | MMD | Disc | W1 | CorrDist |
|-------|----|----|------|----|----------|
| DDPM (vpred+sigmoid) | 3 | 0.020 | 0.66 | 0.176 | 6.655 |
| NormFlow | 5 | 0.011 | 0.74 | 0.101 | 1.154 |

### Per-Fact Breakdown (DDPM vs NormFlow vs Real)

| Fact | Test Statistic | Real | DDPM | NormFlow | DDPM Gap |
|------|---------------|------|------|----------|----------|
| Fat Tails | Excess Kurtosis | 14.03 | 0.04 | 0.90 | Kurtosis near 0 = Gaussian output |
| Vol. Clustering | ARCH-LM p-value | 0.0 | 0.73 | 0.0 | p>>0.05 = no ARCH = constant vol |
| Leverage | corr(r_t, \|r_{t+1}\|) | -0.117 | -0.005 | -0.035 | Weak but correct sign |
| Slow ACF | n_positive/20 | 20 | 10 | 20 | ACF dies immediately |
| Cross-Asset | rolling corr std | 0.263 | 0.130 | 0.133 | Both models similar |
| No Autocorr | Ljung-Box p | 0.0 | 0.43 | 0.0 | DDPM wins (ironic) |

---

*End of report*

# DDPM Phase 7 Improvement Ideas

**Date:** 2026-04-24  
**Author:** Shufeng Chen  
**For:** Team distribution tonight via WeChat

---

## Current Best Model

**vpred + Student-t (df=5) + cosine schedule + 128 channels + 400 epochs**

| Metric | Value |
|--------|-------|
| SF Passed | **5/6** (fails SF6: No Raw Autocorrelation) |
| MMD | **0.006 ± 0.003** |
| Discriminative | **0.54 ± 0.08** (realistic!) |
| Wasserstein-1D | 0.90 ± 0.15 |
| Training time | ~24 min/seed on M4 Pro MPS |

**Goal for Phase 7:** Push SF from 5/6 to **6/6** while keeping MMD ≤ 0.01 and Disc close to 0.5.

---

## What Is Already Merged to Main

Yuxia's training enhancements are **already in `src/models/ddpm_improved.py`** on `main`. All new experiments should use the latest `main` code:

- **Min-SNR-gamma loss weighting** -- down-weights easy/hard timesteps, focuses on informative ones (Hang et al. ICCV 2023)
- **Linear warmup (10ep) + cosine annealing LR** -- replaces flat cosine, stabilizes early training
- **Best model checkpoint restore** -- saves lowest-loss epoch checkpoint, not just last epoch
- **DDIM eta parameter** -- enables configurable stochastic sampling (eta=0: deterministic, eta=1: full DDPM)

**These are additive to, not replacements for, the existing vpred+Student-t base.**

---

## All improvements below are ADDITIVE to vpred+Student-t

They stack on top. Start from the current best model and add one thing at a time.

---

## Ranked Improvement Ideas

### Rank 1 — DDIM eta sweep (HIGHEST ROI, zero retraining)

**Why it matters:** Current DDIM uses eta=0 (fully deterministic). Deterministic sampling creates correlated sampling trajectories that can introduce systematic short-lag autocorrelation -- exactly what SF6 tests (Ljung-Box on raw returns). Adding small stochasticity breaks these correlations.

**Implementation:** No retraining. Just change `eta` when calling `generate()`:

```python
syn = model.generate(n_samples=1000, eta=0.2)  # try 0.0, 0.1, 0.2, 0.3, 0.5, 1.0
```

**Compute:** ~2-5 min total (generation only, no training).

**Expected outcome:** eta=0.2 likely passes SF6. If not, eta=1.0 (full DDPM) as fallback.

**Source:** Song et al. (2021) "Denoising Diffusion Implicit Models", ICLR 2021.  
Section 4.1: eta interpolates between DDPM (eta=1, stochastic) and DDIM (eta=0, deterministic).  
[[Paper]](https://arxiv.org/abs/2010.02502)

**Who should run:** Shufeng (as first step, very fast)

---

### Rank 2 — Raw-returns ACF regularization loss (high impact, retraining needed)

**Why it matters:** Current `use_aux_sf_loss` penalizes:
- Excess kurtosis (targets SF1)
- |returns| ACF (targets SF2 volatility clustering)

But it does **NOT** penalize raw returns ACF, which is exactly what SF6 measures. Adding a direct penalty on lag-1 to lag-5 raw ACF in `p_losses` makes the training objective explicitly aware of SF6.

**Implementation:** Add to `p_losses` in `ddpm_improved.py`:

```python
# In p_losses, after existing aux_sf_loss block:
if self.use_aux_sf_loss:
    # ... existing kurtosis + |acf| losses ...
    
    # NEW: raw ACF penalty targeting SF6
    raw_acf_loss = 0.0
    for lag in range(1, 6):
        acf_lag = (x0_pred[:, lag:] * x0_pred[:, :-lag]).mean(dim=1).mean()
        raw_acf_loss = raw_acf_loss + acf_lag ** 2
    loss = loss + 0.01 * raw_acf_loss  # start with lambda=0.01
```

**Compute:** 400ep retraining (~25 min/seed on MPS, ~7 min/seed on 4090).

**Source:** Du et al. (2024) "FTS-Diffusion: Generative Learning for Financial Time Series".  
Uses auxiliary statistical loss terms to enforce distributional properties.  
[[Paper]](https://arxiv.org/abs/2312.04323)  
Also: Yizheng's WeChat message (Apr 24) describes this as "decorrelation regularizer" with λ·|ACF(x̂, lag=1)|².

**Who should run:** Shufeng (on `eecs4904-4090` GPU, much faster)

---

### Rank 3 — Min-SNR + warmup validation run (Yuxia is running this NOW)

**Why it matters:** Min-SNR reweights training so the model spends more effort on informative noise levels (not too noisy, not too clean). Combined with linear warmup LR, this can improve the quality of the learned score function and produce cleaner x0 predictions -- indirectly reducing autocorrelation artifacts.

**Status:** Yuxia confirmed she is running this (WeChat, 15:21). **Do not duplicate.**  
Wait for her results. If SF goes from 5/6 to 6/6, we're done. If not, proceed to Rank 1.

**Source:** Hang et al. (2023) "Efficient Diffusion Training via Min-SNR Weighting Strategy", ICCV 2023.  
[[Paper]](https://arxiv.org/abs/2303.09556)  
Already cited in our `README.md`.

**Who should run:** Yuxia (running now)

---

### Rank 4 — Per-sample CFG drop + forward/backward noise alignment (Yizheng's temp branch)

**Why it matters:**  
1. **Per-sample CFG drop:** Randomly drop conditioning during training (p=0.1), giving the model both conditional and unconditional generations. This diversifies the learned distribution and can reduce mode collapse.  
2. **Forward/backward noise alignment:** When using Student-t forward noise, the reverse step should also use Student-t noise (not Gaussian). Current code samples Student-t forward but Gaussian backward -- this mismatch can introduce distributional artifacts in generated sequences.

**Implementation:** Available in `origin/yizheng:src/models/ddpm_improved_temp.py`.  
Can be integrated as flags: `use_cfg_drop=True, use_noise_alignment=True`.

**Source:**  
- CFG dropout: Nichol & Dhariwal (2021) "Improved DDPM", Section 4.2. [[Paper]](https://arxiv.org/abs/2102.09672)  
- Noise alignment: Ho et al. (2020) DDPM derivation -- forward and reverse processes should share the same noise family. [[Paper]](https://arxiv.org/abs/2006.11239)

**Who should run:** Yizheng

---

### Rank 5 — Inference-time ACF guidance on raw returns (no retraining)

**Why it matters:** `use_acf_guidance` already exists in `ddpm_improved.py` and applies guidance toward |returns| ACF. Modifying `_acf_guidance_step` to penalize raw returns ACF at lag-1 to lag-5 can directly steer sampling toward low-autocorrelation sequences.

**Risk:** Guidance adds computational cost (gradient through sampling) and can distort other SFs if guidance weight is too high.

**Implementation:**

```python
def _acf_guidance_step(self, x, t, cond, ...):
    x = x.requires_grad_(True)
    x0_pred = self._predict_x0(x, t, ...)
    # Original: absolute ACF penalty
    # Modified: raw ACF penalty for SF6
    loss = sum((x0_pred[:, lag:] * x0_pred[:, :-lag]).mean() ** 2 for lag in range(1, 6))
    grad = torch.autograd.grad(loss, x)[0]
    return x - self.guidance_scale * grad
```

**Source:** Dhariwal & Nichol (2021) "Diffusion Models Beat GANs on Image Synthesis".  
Classifier guidance principle applied to custom statistical objectives.  
[[Paper]](https://arxiv.org/abs/2105.05233)

**Who should run:** Yixuan

---

### Rank 6 — 800-epoch training (low ROI per hour)

**Why it matters:** Phase 3 showed that extending 400→800ep improved MMD from 0.024 to 0.003 for the aux_sf configuration. Longer training may help SF6 by giving the model more time to learn the uncorrelated structure of financial returns.

**Risk:** 2x compute for uncertain SF6 gains. We already have best-model checkpoint restore, so overfitting is less of a concern.

**Compute:** ~50 min/seed on MPS, ~14 min/seed on 4090.

**Who should run:** Only if Rank 1-4 all fail (unlikely).

---

### Rank 7 — Post-hoc white noise injection (quick fallback, no retraining)

**Why it matters:** Add σ·ε (σ=0.05·std(r)) iid Gaussian noise to each generated return series. This directly breaks short-lag autocorrelation and should pass Ljung-Box.

**Trade-off:** SF1 Hill α will increase slightly (tails become less heavy). Current Student-t gives α=3.05, real data is α=7.84 -- we have a safety margin before we'd fail SF1.

```python
syn_noisy = syn + 0.05 * syn.std() * np.random.randn(*syn.shape)
```

**Source:** Team WeChat discussion (Apr 24). Standard technique for decorrelating time series.

**Who should run:** Shufeng (as fallback if nothing else works)

---

### Rank 8 — Patch-based training / stride augmentation (medium, retraining)

**Why it matters:** At stride=1, adjacent training windows share 59/60 data points (98.3% overlap). The model may be learning to reproduce the high correlation between adjacent windows -- which manifests as short-lag autocorrelation in generated sequences. Using random crop augmentation (effective stride=2-3) during training preserves data volume while breaking window-overlap-induced autocorrelation.

**Implementation:** Modify the data loader in `train()` to randomly stride windows:

```python
# Instead of: all consecutive windows
# Use: randomly sampled windows with stride 2-3
indices = np.random.choice(len(windows), size=len(windows)//2, replace=False)
windows_augmented = windows[indices]
```

**Source:** Team WeChat discussion (Apr 24). Stride overlap as source of training-induced autocorrelation is a known issue in sliding-window time series modeling.

**Who should run:** Shufeng (on 4090, if time permits)

---

### Rank 9 — AR(1) post-processing (last resort)

Fit AR(1) per generated series, subtract predicted AR component. Guaranteed SF6 fix but is a hack -- may be criticized in the presentation.

---

## Suggested Division Tonight

| Person | Task | Rank | Retraining | Machine | ETA |
|--------|------|:----:|:----------:|---------|-----|
| **Yuxia** | Min-SNR+warmup baseline (running NOW) | 3 | Yes | Her machine | ~1.5h |
| **Shufeng** | DDIM eta sweep (0.0, 0.1, 0.2, 0.3, 0.5, 1.0) | 1 | No | Mac | 30 min |
| **Shufeng** | Raw ACF regularization loss | 2 | Yes | `eecs4904-4090` | 3×7 min |
| **Yizheng** | CFG drop + noise alignment | 4 | Yes | His machine | ~1.5h |
| **Yixuan** | ACF inference guidance on raw returns | 5 | No | His machine | 30 min |
| **Shufeng (if time)** | Patch-based stride augmentation | 8 | Yes | `eecs4904-4090` | 3×7 min |

**Workflow:**
1. Wait for Yuxia's results first (SF 5/6 or 6/6?)
2. Shufeng immediately runs Rank 1 (eta sweep) -- zero retraining, 30 min
3. If Rank 1 passes SF6 → done, push results
4. If not → Rank 2 (ACF reg) running on 4090, fallback Rank 7 (noise injection) while waiting

---

## SF6 Root Cause Analysis

SF6 (No Raw Autocorrelation) fails because:
1. **Stride=1 window overlap** creates highly correlated training samples
2. **Deterministic DDIM sampling** (eta=0) produces correlated sampling trajectories
3. **Aux SF loss** targets |returns| ACF but not raw returns ACF
4. **Student-t forward / Gaussian reverse mismatch** may introduce distributional artifacts

Fixing eta (Rank 1) addresses cause #2 immediately. Fixing the loss (Rank 2) addresses cause #3. Both together should reliably achieve SF6.

# Experiment Audit, Honest Findings, and Next Steps

**Author**: Shufeng Chen
**Date**: April 16, 2026

---

## 1. Consistency Audit: Are We Comparing Apples to Apples?

### 1.1 Three Different Experiment Settings (Problem)

Our results currently cite numbers from three incompatible experimental setups:

| Setting | base_channels | Epochs | Models Compared | Source |
|---------|:------------:|:------:|-----------------|--------|
| Phase 1 ablation | 64 | 200 | 7 DDPM variants (baseline, dit, vpred, selfcond, sigmoid, crossattn, best_combo) | `experiments/results/phase1_ablation/` |
| Full-scale head-to-head | 128 | 400 | DDPM (vpred+sigmoid) vs NormFlow | `ddpm_improvement_report.md` Section 7 |
| Phase 2 ablation | 128 | 200 | 4 DDPM variants (temporal_attn, hetero_noise, aux_sf_loss, phase2_combo) | `experiments/results/phase2_improvements/` |

**The head-to-head comparison (DDPM vs NormFlow) used 400 epochs, but all ablation experiments used 200 epochs.** This means:
- The MMD=0.020 and Disc=0.66 numbers for DDPM were obtained at 400 epochs.
- The Phase 1 ablation best (vpred at 64ch, MMD=0.051) and Phase 2 best (aux_sf_loss at 128ch, MMD=0.046) were obtained at 200 epochs.
- We cannot directly compare Phase 1 ablation results (64ch) with Phase 2 results (128ch) since the model capacity is different.

### 1.2 Parameter Count Comparison (Fairness)

| Model | Parameters | Notes |
|-------|----------:|-------|
| NormFlow (256 hidden, 6 layers) | 6,711,936 | The comparison target |
| DDPM Phase 1 ablation (64ch) | 2,335,120 | Underparameterized vs NormFlow |
| DDPM Phase 2 / full-scale (128ch) | 8,970,896 | 1.34x more params than NormFlow |
| DDPM temporal_attn (128ch) | 12,785,808 | 1.9x more params than NormFlow |

Phase 1 ablation (64ch) used a **smaller** model than NormFlow, so improvements from vpred/sigmoid are genuinely algorithmic. But Phase 2 (128ch) uses a **larger** model than NormFlow (8.97M vs 6.71M params). The full-scale h2h result (MMD=0.020) was obtained with both more parameters AND more epochs. Any improvement from "128ch + 400 epochs" over "64ch + 200 epochs" is partly explained by raw compute scaling, not just algorithmic innovation.

### 1.3 NormFlow Training Details (Unknown)

A critical gap: **we don't know the exact settings used to train NormFlow for the head-to-head comparison.** From `run_pipeline.py`, the pipeline uses `DEFAULT_EPOCHS = 400`, so NormFlow was likely trained at 400 epochs with its default hidden_dim=256. But was it trained from the same seed? On the same data split? The h2h comparison in `ddpm_improvement_report.md` reports NormFlow results (SF=5, MMD=0.011) but doesn't specify its training settings. We need to re-run NormFlow under the same conditions for a fair comparison.

### 1.4 Critical Interaction Effect: sigmoid Cancels vpred's Kurtosis

A newly discovered inconsistency between Phase 1 and Phase 2:

| Configuration | Kurtosis (mean across seeds) | Fat Tails Pass Rate |
|---------------|:---------------------------:|:-------------------:|
| vpred only (64ch, Phase 1) | **+0.088** | **100%** |
| sigmoid only (64ch, Phase 1) | -0.002 | 33% |
| vpred + sigmoid (128ch, Phase 2) | **-0.001** | **33%** |
| vpred + sigmoid + aux_sf (128ch, Phase 2) | +0.008 | 33% |

**vpred alone reliably produces positive excess kurtosis (~0.09) and passes fat tails 100% of the time.** But when combined with sigmoid schedule, the kurtosis drops to ~0.00 and the test fails. The sigmoid schedule suppresses the tail-producing effect of v-prediction. This was already visible in Phase 1's `best_combo` (which also failed fat tails) but was not diagnosed at the time.

This means our "Phase 1 winner" (vpred+sigmoid) is actually **worse on fat tails** than vpred alone. We chose vpred+sigmoid based on composite score and discriminative score, but we sacrificed a stylized fact in the process.

---

## 2. What Is Genuine Algorithmic Innovation vs. Compute Scaling?

### 2.1 Genuine Improvements (Would Help Even at Fixed Compute)

- **v-prediction**: 7x MMD improvement at the same 64ch/200ep as baseline. This is a real algorithmic change -- different loss target, better gradient balance. The kurtosis improvement (0.09 vs -0.05) is also real.
- **Sigmoid schedule**: Better noise allocation at same model size. Improves discriminative score from 0.92 to 0.87 at 64ch/200ep. Genuine schedule innovation.
- **Auxiliary SF loss**: Best MMD (0.046 vs 0.049-0.052) among Phase 2 variants at same 128ch/200ep settings. The ACF matching produced the first-ever Slow ACF Decay pass. This is a genuine loss function innovation.

### 2.2 Improvements That Are Partly Compute Scaling

- **128ch vs 64ch**: MMD improved from ~0.05 to ~0.046 at same 200 epochs. This is partly model capacity, not algorithm.
- **400 epochs vs 200 epochs**: MMD improved from ~0.05 to 0.020 (2.5x). This is training time, not algorithm.
- **temporal_attn**: Adds 3.8M parameters (42% more), 44% slower training, but no metric improvement over the same model without attention. The attention is currently free-riding on extra parameters.

### 2.3 The Uncomfortable Truth

Our best DDPM result (MMD=0.020, Disc=0.66) was obtained by:
1. Using vpred+sigmoid (genuine algorithmic improvements)
2. Scaling to 128 channels (1.34x NormFlow's parameters)
3. Training for 400 epochs (2x the ablation setting)

NormFlow's result (MMD=0.011, Disc=0.74) was obtained with fewer parameters and (presumably) the same number of epochs. On a purely algorithmic basis, NormFlow is more parameter-efficient. Our DDPM's advantage on discriminative score (0.66 vs 0.74) may also partly be a function of having more parameters.

---

## 3. Proposed Next Iteration of Experiments

### 3.1 Fair Head-to-Head Re-Run (Highest Priority)

**Goal**: Produce an apples-to-apples comparison table.

Run ALL models at the same settings:
- **Epochs**: 400 (time is not a constraint)
- **base_channels for DDPM**: 128
- **Seeds**: 42, 123, 456 (3 runs each)
- **n_gen**: 1000 samples for evaluation
- **Models to include**: baseline DDPM, vpred-only, sigmoid-only, vpred+sigmoid, vpred+sigmoid+aux_sf, NormFlow

This gives us the data to answer: at equal compute budget, does each algorithmic improvement genuinely help?

### 3.2 Vpred-Only at 128ch, 400 Epochs (Critical Missing Data Point)

We have never tested vpred-only (without sigmoid) at 128ch/400ep. Given the kurtosis interaction effect, this variant may actually achieve **4/6 SF** (passing fat tails, leverage, cross-asset, and possibly one more), which would be closer to NormFlow's 5/6 than our current best of 3/6.

### 3.3 Investigate vpred + aux_sf_loss (Without Sigmoid)

If sigmoid suppresses kurtosis, try combining vpred with aux_sf_loss but NOT sigmoid. The aux_sf_loss explicitly pushes kurtosis up, which may counteract whatever suppression sigmoid causes. But better yet: remove the suppressor entirely and keep the good parts.

### 3.4 Tune Auxiliary SF Loss Weights

The current auxiliary loss uses weight 0.1 for both kurtosis and ACF matching. This was chosen without tuning. Try:
- Higher kurtosis weight (0.5, 1.0) to force heavier tails
- Higher ACF weight (0.5) to force longer memory
- Apply aux loss every step instead of every 5 steps

### 3.5 Longer Training with Learning Rate Restarts

At 200 epochs the models may not have converged. Try:
- 400 epochs (baseline for comparison)
- 800 epochs (to see if there's still improvement)
- Warm restarts: CosineAnnealingWarmRestarts instead of CosineAnnealingLR, with T_0=100

### 3.6 Noise Schedule Exploration (Beyond Sigmoid)

If sigmoid hurts kurtosis, try other schedules:
- Linear schedule (the original DDPM default)
- Cosine schedule with different `s` parameter
- A custom "financial" schedule that allocates more capacity to low-noise levels where tail structure lives

---

## 4. What Might Actually Close the Gap to NormFlow

The three failing stylized facts have distinct root causes:

### 4.1 Fat Tails (Most Promising to Fix)

**Current state**: vpred alone gets kurtosis to 0.09 (passes), but the bar is absurdly low (threshold: kurtosis > 0, real data: 14.03). Even "passing" is not really reproducing real fat tails.

**Best bet**: Use vpred without sigmoid, combine with auxiliary kurtosis loss at higher weight (0.5 or 1.0). Also consider Student-t noise in the forward process instead of Gaussian -- this directly addresses the root cause (Gaussian noise bias).

### 4.2 Volatility Clustering (Hardest to Fix)

**Current state**: No DDPM variant produces ARCH effects. The isotropic noise destroys temporal volatility structure.

**Best bet**: Latent diffusion (from the improvement roadmap, Fix 4). Encode windows into a compact latent where temporal structure is entangled, then run diffusion in latent space. The autoencoder preserves volatility clustering in its representation. This is a larger engineering effort but addresses the structural problem.

**Quicker alternative**: Train on squared returns or absolute returns as additional channels, giving the model explicit access to volatility information.

### 4.3 Slow ACF Decay (Partially Addressed)

**Current state**: aux_sf_loss achieved n_positive=15 in 1/3 seeds. Temporal attention pushed it to 13.

**Best bet**: Increase ACF loss weight and apply every step. Also consider longer windows (120-day instead of 60-day) to give the model more temporal context for learning long memory.

---

## 5. Recommendations for Next Experiment Batch

Given unlimited time on the Mac, run these in order:

| # | Experiment | Config | Epochs | Purpose |
|---|-----------|--------|--------|---------|
| 1 | vpred_only_400ep | vpred only, 128ch | 400 | Test if vpred alone beats vpred+sigmoid at long training |
| 2 | vpred_auxsf_400ep | vpred + aux_sf, 128ch | 400 | Best algorithmic combo without sigmoid interference |
| 3 | vpred_sigmoid_400ep | vpred + sigmoid, 128ch | 400 | Current "best" at full training for direct comparison |
| 4 | vpred_auxsf_highweight | vpred + aux_sf (weight=0.5), 128ch | 400 | Stronger auxiliary signal |
| 5 | normflow_400ep | NormFlow | 400 | Fair comparison baseline |
| 6 | vpred_auxsf_800ep | vpred + aux_sf, 128ch | 800 | Push training further |

Seeds: 42, 123, 456 for all. Total: 18 DDPM runs + 3 NormFlow runs = 21 runs.

At ~3 min/run for 200 epochs, 400 epochs would be ~6 min, 800 epochs ~12 min. Total: roughly 2 hours. Easily doable in a day.

---

## 6. Summary of Honest Assessment

| Claim | Status |
|-------|--------|
| "v-prediction improves MMD by 7x" | **TRUE** -- verified at same compute (64ch, 200ep) |
| "Sigmoid schedule improves discriminative score" | **TRUE** -- at same compute |
| "vpred + sigmoid is the best DDPM configuration" | **FALSE** -- Phase 3 proved sigmoid destroys vol clustering, fat tails, and ACF. vpred alone scores 4.7/6 SF vs vpred+sigmoid's 2.7/6 |
| "Phase 2 improvements (temporal_attn, hetero_noise, aux_sf) help" | **MARGINAL for SF, SIGNIFICANT for MMD at 800ep** -- aux_sf at 800ep achieves MMD=0.0031, beating NormFlow |
| "DDPM beats NormFlow on discriminative score (0.66 vs 0.74)" | **REVERSED** -- that was with sigmoid. vpred-only gets disc=0.82, worse than NormFlow's 0.74 |
| "DDPM achieves 3/6 SF vs NormFlow's 5/6" | **OUTDATED** -- vpred alone at 128ch/400ep achieves **4.7/6 SF**, closing the gap to 0.3 SF |
| "The gap is structural (Gaussian noise bias)" | **OVERSTATED** -- a large part of the gap (4.7->2.7 SF) was caused by our sigmoid schedule choice. However, the remaining 4.7 vs 5.0 gap (slow ACF consistency) is real and may reflect genuine structural limitations of discrete-time diffusion for long-memory processes |

---

## 7. Phase 3 Results (Resolved)

Phase 3 ran 27 experiments at uniform settings (128ch, 400ep, 3 seeds). Full results in `experiments/results/phase3_fair_comparison/ANALYSIS.md`.

**Key outcome**: v-prediction alone (no sigmoid) is the correct DDPM configuration, achieving 4.7/6 SF, matching NormFlow on fat tails, vol clustering, leverage, and cross-asset correlations. With aux_sf_loss at 800 epochs, DDPM achieves MMD=0.0031, beating NormFlow's 0.0053.

---

## 8. Phase 4: Parameter-Fair Test (Current)

Run vpred-only at 64ch (2.3M params, 3x fewer than NormFlow) at 400 epochs to prove v-prediction is genuinely more parameter-efficient, not just benefiting from a larger model.

---

## 9. Phase 5: Potential Future Innovations (Literature-Informed)

Based on a literature survey (see `references/ddpm-helper-perplexity.md`), these are the most promising true algorithmic innovations for further closing the gap with NormFlow. None rely on more parameters or more training time.

### 9.1 Wavelet-Domain Diffusion (Highest Priority)

**Paper**: Takahashi & Mizuno, "Generation of Synthetic Financial Time Series by Diffusion Models" (arXiv:2410.18897, Quantitative Finance 2025)

**Idea**: Convert each 60x16 return window to a wavelet coefficient image (via discrete wavelet transform), run DDPM in wavelet space, then invert. Coarse wavelet coefficients encode long-term structure (directly addressing slow ACF decay); fine coefficients encode local noise.

**Why it may work for us**: This is the only published approach that explicitly reports improved slow ACF decay in financial DDPM -- exactly our remaining weakness. It requires no additional parameters (same UNet, just different input representation). The wavelet transform is invertible and deterministic, so it adds zero learnable complexity.

**Cost**: Implementing the wavelet transform wrapper (~50 lines of code). No training overhead.

### 9.2 Heavy-Tailed Diffusion (Student-t Noise)

**Paper**: "Heavy-Tailed Diffusion Models" (ICLR 2025, arXiv:2410.14171)

**Idea**: Replace Gaussian noise in the forward process with multivariate Student-t noise (degrees of freedom nu controls tail weight). Derive the corresponding Student-t posterior for the reverse process.

**Why it may work for us**: v-prediction already gives kurtosis ~0.28, but real data has kurtosis 14.03. Student-t noise directly encodes heavy tails in the diffusion process rather than hoping the denoiser learns them. Could push kurtosis much higher with no change to model size.

**Cost**: Moderate -- requires re-deriving the posterior formula and changing `q_sample`/`p_sample`. About 100 lines of code.

### 9.3 Dual-Stream Attention (Time + Variate)

**Paper**: DiTS (arXiv:2602.06597, 2026)

**Idea**: Instead of treating the 60x16 data as a 1D sequence of length 60 with 16 channels, use two separate attention mechanisms: Time Attention (across 60 timesteps per asset) and Variate Attention (across 16 assets per timestep). This is a hybrid of our existing UNet convolutions with targeted attention.

**Why it may work for us**: Our current temporal_attn only attends across time. Adding a variate attention stream would directly target cross-asset correlation preservation (our CorrDist gap: 3.46 vs NormFlow's 1.18). DiTS showed this is more effective than single-stream attention.

**Cost**: ~100 lines of code. Can be added as a module inside existing UNet ResBlocks. To keep parameter count fair, reduce conv channel width slightly to compensate.

### 9.4 Inference-Time Temporal Guidance (Cheapest Innovation)

**Idea**: During sampling (not training), compute differentiable ACF of |returns| on the partially denoised sample at select timesteps, and take a gradient step to nudge the sample toward the correct ACF profile. This is analogous to classifier guidance but using a hand-crafted temporal statistic instead of a classifier.

**Why it may work for us**: It directly targets slow ACF decay (our weakest stylized fact) at zero training cost. The guidance operates only during the ~50 DDIM sampling steps.

**Cost**: ~30 lines of code in `_ddim_sample`. Zero impact on training time or parameter count.

# Phase 1 Ablation Study -- Summary of Findings

**Setup**: 7 DDPM variants x 3 seeds (42, 123, 456), 200 epochs, base_channels=64, T=1000, 16 assets, 1059 windows (60-day, stride=5), Apple MPS.

---

## What Worked (Positive Impact)

### v-prediction (`use_vpred=True`)
- **MMD**: 0.051 (7x better than baseline 0.364)
- **Wasserstein-1**: 0.298 (5.6x better than baseline 1.679)
- **Correlation distance**: 5.887 (best cross-asset structure)
- **SF**: 3.0/6 -- reliably passes fat tails and leverage effect
- **Mechanism**: Predicting velocity v = sqrt(alpha_bar)*eps - sqrt(1-alpha_bar)*x0 instead of noise eps. Better-balanced gradients across noise levels, preventing the model from ignoring low-noise (high-signal) timesteps.

### Sigmoid schedule (`use_sigmoid_schedule=True`)
- **Composite**: 0.872 (best overall weighted score)
- **Discriminative score**: 0.872 (closest to ideal 0.5 among all variants)
- **SF**: 3.33/6 (tied best)
- **Mechanism**: Concentrates noise allocation at mid-levels where financial structure (volatility patterns, tail behavior) lives. Cosine schedule wastes capacity on very low and very high noise levels.

---

## What Did Not Work (No Positive Impact)

### DiT Transformer denoiser (`use_dit=True`)
- **MMD**: 0.520 (worst of all variants)
- **Training time**: 280s (2x slower than UNet variants)
- **SF**: 2.33/6
- **Why it failed**: 60-step, 16-feature sequences are too small for attention to outperform local convolutions. The DiT has no inductive bias for local patterns; the UNet's conv kernels capture local structure efficiently.

### Cross-attention conditioning (`use_cross_attn=True`)
- **All metrics**: Nearly identical to baseline
- **Why it failed**: The 5-dimensional macro conditioning vector is too simple to benefit from cross-attention over additive injection. Cross-attention adds overhead with no representational gain.

### Combined best_combo (`vpred + selfcond + sigmoid`)
- **SF**: 2.33/6 (worse than sigmoid alone at 3.33)
- **MMD**: 0.055 (good, but SF regression is concerning)
- **Why it failed**: Self-conditioning's extra forward pass may conflict with v-prediction's different loss landscape under the sigmoid schedule. Improvement composition is non-trivial.

---

## Mixed Results

### Self-conditioning (`use_self_cond=True`)
- **SF**: 3.33/6 (tied best) -- passes leverage effect 100%
- **MMD**: 0.242 (4.8x worse than vpred)
- **Takeaway**: Helps temporal structure (stylized facts) but hurts marginal distribution matching.

---

## Phase 1 Winners for Phase 2 Base

**vpred + sigmoid** was chosen as the Phase 2 base configuration. At full scale (128 channels, 400 epochs), this combination achieved:
- SF: 3/6, MMD: 0.020, Disc: 0.66, W1: 0.176, CorrDist: 6.655

**UPDATE (Phase 3 finding)**: This recommendation was overturned. Phase 3 showed that sigmoid schedule suppresses v-prediction's kurtosis and destroys volatility clustering. **vpred alone** at 128ch/400ep achieves 4.7/6 SF. See `experiments/results/phase3_fair_comparison/ANALYSIS.md`.

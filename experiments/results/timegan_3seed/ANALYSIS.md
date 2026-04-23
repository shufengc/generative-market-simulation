# TimeGAN 3-Seed Rebaseline

**Date:** 2026-04-23
**Branch:** experiment/timegan-3seed-colab
**Data:** stride=1 → 5,299 windows (same as phase6_rebaseline)
**Eval framework:** `src/evaluation/stylized_facts.py` (same as DDPM rebaseline)
**Model:** TimeGAN (WGAN-GP, main branch version)
**Training:** epochs=100 (Phase 1: 60 AE / Phase 2: 30 Supervisor / Phase 3: 60 Joint)
**Seeds:** 42, 123, 456
**Hardware:** Colab (CPU fallback due to cuDNN double-backward limitation in WGAN-GP)

---

## Summary

| Seed | SF Passed | MMD    | Disc  |
|------|:---------:|:------:|:-----:|
| 42   | **4/6**   | 0.0839 | 1.00  |
| 123  | **4/6**   | 0.1433 | 1.00  |
| 456  | **4/6**   | 0.1036 | 1.00  |
| **Mean ± Std** | **4.00 ± 0.00** | 0.110 ± 0.025 | 1.00 ± 0.00 |

## Training Dynamics

Joint phase G/D loss (epoch 20 / 40 / 60):
- seed 42:  G=24.4 / 30.5 / 32.5,  D=-8.7 / -11.8 / -12.8
- seed 123: G=23.9 / 30.0 / 31.9,  D=-8.4 / -11.6 / -12.7
- seed 456: G=24.1 / 30.3 / 32.2,  D=-8.4 / -11.6 / -12.6

G-loss grows moderately from ~24 to ~32 over 60 Joint epochs. Earlier 600-epoch
experiments showed runaway divergence (G → 85 by epoch 240). The 100-epoch
budget is near-optimal for this architecture and dataset.

## Comparison to Other Models (Reference)

| Model                              | SF  | MMD    | Disc  |
|------------------------------------|:---:|:------:|:-----:|
| DDPM vpred+Student-t (phase6)      | 5/6 | 0.006  | 0.85  |
| NormFlow (old data, phase 3)       | 5/6 | 0.005  | N/A   |
| **TimeGAN (this run)**             | 4/6 | 0.110  | 1.00  |

## Interpretation

TimeGAN passes 4/6 stylized facts with **zero seed variance** — the structural
properties it learns (fat tails, vol clustering, leverage, long memory) are
highly reproducible across random initializations.

However:
- **MMD is ~18× higher than DDPM** — the marginal return distribution deviates
  from real data at the per-sample level.
- **Discriminative score = 1.00** — a simple classifier distinguishes real vs
  synthetic perfectly. Structural statistics match, but individual samples
  look wrong in shape/scale.

This is consistent with the known GAN trade-off: adversarial training optimizes
for statistics the discriminator detects in latent space, not for marginal
distributional fidelity.

## Recommendation

For the final paper, report TimeGAN as a **structural baseline**
(SF=4/6, MMD=0.11, Disc=1.00). DDPM vpred+Student-t (5/6, MMD=0.006, Disc=0.85)
remains the headline model. The large gap in Disc and MMD between TimeGAN and
DDPM constitutes a useful illustration of model-family trade-offs.

## Artifacts

- `seed{42,123,456}/timegan.pt` — trained checkpoints
- `seed{42,123,456}/generated.npy` — 500 synthetic windows each, shape (500, 60, 16)

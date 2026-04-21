# Phase 2 DDPM Improvement -- Ablation Analysis

## Experiment Setup

- **Base configuration**: vpred + sigmoid schedule (Phase 1 winners)
- **base_channels**: 128 (up from 64 in Phase 1 ablation)
- **Epochs**: 200 | **T**: 1000 | **batch_size**: 64 | **lr**: 2e-4
- **Seeds**: 42, 123, 456 (3 runs per variant)
- **Device**: Apple MPS (Mac GPU)
- **Data**: 1059 windows of 60-day x 16-asset normalized log returns

## Variants Tested

| ID | Description | New Mechanism |
|----|-------------|---------------|
| temporal_attn | Multi-head self-attention after each ResBlock | Global temporal receptive field |
| hetero_noise | Volatility-scaled noise injection | Preserves heteroskedastic structure |
| aux_sf_loss | Auxiliary kurtosis + ACF matching losses | Directly optimizes stylized facts |
| phase2_combo | All three combined | Full Phase 2 stack |

## Results Summary

| Model | SF (mean+-std) | MMD | Disc. Score | W1 | CorrDist | Train Time |
|-------|----------------|-----|-------------|----|---------:|------------|
| temporal_attn | 2.7+-0.5 | 0.0491 | 0.63 | 0.2852 | 7.948 | 200s |
| hetero_noise | 2.7+-0.5 | 0.0493 | 0.64 | 0.2869 | 7.968 | 138s |
| **aux_sf_loss** | **3.0+-0.0** | **0.0461** | **0.63** | **0.2727** | **7.898** | 140s |
| phase2_combo | 3.0+-0.0 | 0.0524 | 0.67 | 0.2984 | 8.014 | 201s |

## Comparison with Phase 1 Winner and NormFlow

| Model | SF | MMD | Disc | W1 | CorrDist | Notes |
|-------|---:|----:|-----:|---:|---------:|-------|
| Phase 1 (vpred+sigmoid, 128ch, 400ep) | 3 | 0.020 | 0.66 | 0.176 | 6.655 | 400 epochs, direct reference |
| Phase 1 ablation (vpred, 64ch, 200ep) | 3 | 0.051 | 0.92 | 0.298 | 5.887 | Comparable setup |
| **aux_sf_loss** (128ch, 200ep) | **3** | **0.046** | **0.63** | **0.273** | **7.898** | Best Phase 2 |
| phase2_combo (128ch, 200ep) | 3 | 0.052 | 0.67 | 0.298 | 8.014 | All improvements |
| NormFlow | 5 | 0.011 | 0.74 | 0.101 | 1.154 | Target to match |

## Per-Fact Pass Rates

| Model | Fat Tails | Vol Clustering | Leverage | Slow ACF | Cross-Asset | No Autocorr |
|-------|:---------:|:--------------:|:--------:|:--------:|:-----------:|:-----------:|
| temporal_attn | 33% | 0% | 100% | 0% | 100% | 67% |
| hetero_noise | 33% | 0% | 100% | 0% | 100% | 67% |
| aux_sf_loss | 33% | 0% | 100% | **33%** | 100% | 67% |
| phase2_combo | 33% | 0% | 100% | 0% | 100% | **100%** |

## Detailed Findings

### 1. Fat Tails: Still Failing

Excess kurtosis across all variants hovers near 0 (-0.01 to +0.03), essentially Gaussian. Real data has kurtosis = 14.03. Neither heteroskedastic noise nor auxiliary SF loss produces the heavy tails needed.

**Root cause persists**: The isotropic Gaussian noise in the diffusion forward process has a strong smoothing effect on tail events. The heteroskedastic noise modification scales noise by local volatility but does not fundamentally change the Gaussian nature of the noise distribution. The auxiliary loss pushes kurtosis up slightly but the MSE denoising objective dominates.

### 2. Volatility Clustering: Still Failing

ARCH-LM p-values range from 0.06 to 0.73 across all variants (need < 0.05 to pass). No variant consistently produces ARCH effects.

**Assessment**: Temporal attention was expected to capture long-range volatility patterns, but at 200 epochs with 128 channels, the attention layers have not learned to produce volatility clustering. The problem may require more training or a fundamentally different approach (e.g., latent diffusion where temporal structure is encoded in the latent space).

### 3. Slow ACF Decay: Partial Improvement

- aux_sf_loss achieved n_positive=15 in one seed (456), passing this test for the first time in any DDPM variant
- temporal_attn showed n_positive up to 13 (from baseline of 10), indicating the attention is partially capturing long memory
- The ACF matching loss in aux_sf_loss directly optimizes for this property

### 4. Discriminative Score: Improved

All Phase 2 variants achieve discriminative scores of 0.60-0.70, with the best being **0.60 (aux_sf_loss, seed 456)** -- closer to the ideal 0.50 than any previous DDPM variant. This confirms that the generated samples are individually realistic even if aggregate statistics lag.

### 5. Combination (phase2_combo) Does Not Help

Combining all three Phase 2 improvements produces MMD of 0.052 (worse than aux_sf_loss alone at 0.046). This mirrors the Phase 1 finding that combining improvements causes interference. The combo is slower (201s vs 140s) with no benefit.

## Key Takeaways

1. **aux_sf_loss is the best Phase 2 improvement**: Best MMD (0.046), best W1 (0.273), most consistent SF count (3.0 in all seeds), and the only variant to pass the Slow ACF Decay test (in 1 of 3 seeds).

2. **The three hardest stylized facts remain unsolved**: Fat tails, volatility clustering, and slow ACF decay are structurally difficult for diffusion models. The Gaussian noise process creates a fundamental inductive bias that these modifications alone cannot overcome.

3. **The gap to NormFlow is structural**: NormFlow's advantages (no noise bias, global receptive field via flattened MLPs, exact likelihood) give it an inherent edge on temporal statistics. Our DDPM wins on discriminative score (0.63 vs 0.74) but cannot match NormFlow's stylized fact coverage (3 vs 5).

4. **200 epochs at 128 channels gives comparable MMD to 200 epochs at 64 channels**: The Phase 2 results (MMD ~0.046-0.052) are in the same range as Phase 1 vpred at 64ch (0.051). The full-scale 400-epoch run at 128ch achieved MMD=0.020, suggesting that **epoch count matters more than channel width** for distributional matching.

## Critical Discovery: Sigmoid Schedule Suppresses v-prediction's Kurtosis

Cross-referencing Phase 1 and Phase 2 results reveals an interaction effect that was not diagnosed earlier:

| Configuration | Mean Kurtosis | Fat Tails Pass Rate |
|---------------|:------------:|:-------------------:|
| vpred only (64ch, Phase 1) | **+0.088** | **100%** |
| sigmoid only (64ch, Phase 1) | -0.002 | 33% |
| vpred + sigmoid (128ch, Phase 2 base) | -0.001 | 33% |

**v-prediction alone reliably produces positive kurtosis and passes fat tails in 100% of runs.** When combined with sigmoid schedule, kurtosis drops to ~0 and fat tails fails. The sigmoid schedule's redistribution of noise levels suppresses the tail-producing effect of v-prediction.

This means our "Phase 1 winner" (vpred+sigmoid) sacrificed a stylized fact (fat tails) for better discriminative score. Vpred-only may actually achieve 4/6 SF at full scale, closer to NormFlow's 5/6.

## Experiment Consistency Warning

Phase 1 and Phase 2 used different base_channels (64 vs 128), making direct metric comparison unreliable. The full-scale h2h numbers (MMD=0.020, Disc=0.66 for DDPM vs MMD=0.011, Disc=0.74 for NormFlow) used 400 epochs, while all ablations used 200 epochs. A fair comparison requires running all variants at the same settings.

## Recommendation (Revised)

1. **Run vpred-only (no sigmoid) at 128ch, 400 epochs** -- this may recover fat tails and reach 4/6 SF
2. **Run vpred + aux_sf_loss (no sigmoid) at 128ch, 400 epochs** -- combine the kurtosis-friendly base with the ACF-improving auxiliary loss
3. **Re-run NormFlow at 400 epochs with 3 seeds** for a statistically fair comparison
4. **Do not assume vpred+sigmoid is optimal** -- the interaction effect means we need to re-evaluate
5. See `docs/experiment_audit_and_next_steps.md` for the full audit and proposed experiment batch

# Phase 3: Fair Comparison Experiment Report

**Date**: April 16, 2026
**Setup**: All variants at 128 channels, 400 epochs (except 800ep extension), T=1000, batch=64, lr=2e-4, seeds 42/123/456, Apple MPS, 1059 windows (60-day x 16-asset), n_gen=1000.

---

## Main Results Table

| Model | SF (mean+-std) | MMD | Disc. Score | W1 | CorrDist |
|-------|:--------------:|:---:|:-----------:|:--:|:--------:|
| **p3_normflow** | **5.0+-0.0** | 0.0053 | 0.74 | 0.0852 | **1.18** |
| **p3_vpred** | **4.7+-0.5** | 0.0195 | 0.82 | 0.1606 | 3.46 |
| **p3_vpred_auxsf_800ep** | **4.3+-0.5** | **0.0031** | 0.75 | **0.0623** | 2.73 |
| p3_vpred_auxsf | 4.3+-0.5 | 0.0236 | 0.79 | 0.1910 | 4.54 |
| p3_vpred_auxsf_hi | 4.3+-0.5 | 0.0248 | 0.75 | 0.1986 | 4.49 |
| p3_vpred_sigmoid_auxsf | 3.0+-0.8 | 0.0212 | 0.67 | 0.1665 | 6.67 |
| p3_baseline | 3.0+-0.0 | 0.2694 | 0.97 | 1.1818 | 3.87 |
| p3_vpred_sigmoid | 2.7+-0.5 | 0.0207 | **0.67** | 0.1651 | 6.65 |
| p3_sigmoid | 2.7+-0.5 | 0.0266 | 0.71 | 0.1948 | 6.71 |

## Per-Fact Pass Rates

| Model | Fat Tails | Vol Cluster | Leverage | Slow ACF | Cross-Asset | No Autocorr |
|-------|:---------:|:-----------:|:--------:|:--------:|:-----------:|:-----------:|
| p3_normflow | 100% | 100% | 100% | 100% | 100% | 0% |
| p3_vpred | 100% | **100%** | 100% | **67%** | 100% | 0% |
| p3_vpred_auxsf_800ep | 100% | **100%** | 100% | 33% | 100% | 0% |
| p3_vpred_auxsf | 100% | **100%** | 100% | 33% | 100% | 0% |
| p3_vpred_auxsf_hi | 100% | **100%** | 100% | 33% | 100% | 0% |
| p3_vpred_sigmoid | 100% | 0% | 100% | 0% | 100% | 67% |
| p3_vpred_sigmoid_auxsf | 100% | 0% | 100% | 0% | 100% | 67% |
| p3_sigmoid | 67% | 0% | 100% | 0% | 100% | 67% |
| p3_baseline | 0% | 67% | 100% | 33% | 100% | 0% |

---

## Key Findings

### 1. Sigmoid Schedule is Harmful -- Not Helpful

This is the headline finding of Phase 3.

| Metric | vpred only | vpred + sigmoid | Effect |
|--------|:----------:|:---------------:|:------:|
| SF | **4.7** | 2.7 | **-2.0 SF** |
| Fat Tails kurtosis | 0.28 | 0.02 | Suppressed by 14x |
| Vol Clustering ARCH-p | **0.000** | 0.699 | Destroyed |
| Slow ACF n_positive | 14 | 0 | Destroyed |
| MMD | 0.0195 | 0.0207 | Similar |
| Disc Score | 0.82 | **0.67** | Sigmoid closer to 0.5 |

Sigmoid schedule was previously chosen as the "Phase 1 winner" because it improved discriminative score (closer to 0.5). But this came at the cost of **destroying three stylized facts**: volatility clustering, slow ACF decay, and (indirectly) fat tails. At full scale (128ch, 400ep), v-prediction alone passes 4-5/6 SF while vpred+sigmoid drops to 2-3/6.

The sigmoid schedule concentrates noise at mid-levels, which makes samples that are harder for a classifier to distinguish (better disc score) but eliminates the temporal and distributional structure that financial data requires.

### 2. v-Prediction is the Single Most Important Innovation

v-prediction alone (no other modifications) achieves:
- **4.7/6 SF** (vs baseline's 3.0/6) -- passes fat tails, volatility clustering, leverage, and often slow ACF
- **Kurtosis 0.23-0.30** (vs baseline's -0.05) -- genuine heavy tails
- **ARCH-LM p = 0.0** in all seeds -- genuine volatility clustering
- **MMD = 0.0195** (vs baseline's 0.2694) -- 14x improvement

This is a purely algorithmic improvement: same model size, same training time, different loss target (velocity instead of noise). The balanced gradients across noise levels allow the model to learn structure at all scales.

### 3. DDPM Now Matches NormFlow on 4 of 6 Stylized Facts

| Fact | NormFlow | DDPM (vpred) | Gap |
|------|:--------:|:------------:|:---:|
| Fat Tails | PASS | **PASS** | Closed |
| Vol Clustering | PASS | **PASS** | Closed |
| Leverage | PASS | **PASS** | Tied |
| Slow ACF | PASS | **67% pass** | Partially closed |
| Cross-Asset | PASS | **PASS** | Tied |
| No Autocorr | FAIL | FAIL | Both fail |

The remaining gap is Slow ACF Decay (long memory). vpred passes this 67% of the time (2/3 seeds) vs NormFlow's 100%. Neither model passes No Raw Autocorrelation (both real data and NormFlow fail this test too).

### 4. Auxiliary SF Loss + 800 Epochs Achieves MMD Parity with NormFlow

| Model | MMD | W1 | CorrDist |
|-------|:---:|:--:|:--------:|
| NormFlow (400ep) | 0.0053 | 0.0852 | **1.18** |
| vpred + aux_sf (800ep) | **0.0031** | **0.0623** | 2.73 |
| vpred only (400ep) | 0.0195 | 0.1606 | 3.46 |

At 800 epochs with auxiliary SF loss, the DDPM achieves **MMD = 0.0031** -- lower than NormFlow's 0.0053. It also beats NormFlow on Wasserstein-1 (0.0623 vs 0.0852). The remaining gap is correlation matrix distance (2.73 vs 1.18), where NormFlow's flattened-MLP architecture has an inherent advantage.

The kurtosis at 800ep reaches **1.95-2.52** (up from 0.23-0.30 at 400ep), showing the auxiliary loss has increasing effect with training time.

### 5. Higher aux_sf Weight (0.5 vs 0.1) Shows No Benefit

| Config | SF | MMD | W1 |
|--------|:--:|:---:|:--:|
| vpred + aux_sf (weight=0.1) | 4.3 | 0.0236 | 0.1910 |
| vpred + aux_sf (weight=0.5) | 4.3 | 0.0248 | 0.1986 |

Identical SF count. The higher weight slightly worsens MMD and W1. Weight 0.1 is sufficient.

### 6. aux_sf Cannot Rescue Sigmoid

| Config | SF | Fat Tails | Vol Cluster |
|--------|:--:|:---------:|:-----------:|
| vpred + sigmoid | 2.7 | PASS (barely) | FAIL |
| vpred + sigmoid + aux_sf | 3.0 | PASS | FAIL |
| vpred only | **4.7** | **PASS** | **PASS** |

Adding aux_sf to vpred+sigmoid improves SF from 2.7 to 3.0 but cannot recover volatility clustering. The sigmoid schedule's fundamental noise redistribution cannot be patched by an auxiliary loss. The correct fix is to remove sigmoid entirely.

---

## Parameter Count Fairness

| Model | Parameters | SF | MMD |
|-------|----------:|:--:|:---:|
| NormFlow | 6,711,936 | 5.0 | 0.0053 |
| DDPM vpred (128ch) | 8,970,896 | 4.7 | 0.0195 |

DDPM uses 1.34x more parameters than NormFlow. At 64ch (2.3M params, 3x fewer than NormFlow), Phase 1 vpred achieved 3.0/6 SF and MMD=0.051 at 200 epochs. This suggests v-prediction works regardless of model size, though the 64ch result used fewer epochs (200 vs 400) so is not a direct comparison. Phase 4 will test 64ch at 400 epochs for a true parameter-fair comparison.

---

## Recommended Final Configuration

For the project submission, use **vpred only (no sigmoid, no aux_sf)** as the primary DDPM configuration:

- It achieves the highest SF count (4.7/6 mean, matching NormFlow on 4 of 6 facts)
- It is the simplest configuration (single flag change from baseline)
- It presents the clearest scientific story (v-prediction fixes gradient balance)

For the distributional matching comparison, additionally show **vpred + aux_sf at 800 epochs**:
- MMD = 0.0031 (beats NormFlow's 0.0053)
- W1 = 0.0623 (beats NormFlow's 0.0852)
- Demonstrates that with sufficient training, DDPM can match NormFlow on distributional metrics

---

## Corrected Narrative for Final Report

**Old narrative** (incorrect): "vpred+sigmoid is our best DDPM, scoring 3/6 SF vs NormFlow's 5/6. The gap is structural."

**New narrative** (correct): "v-prediction alone is the key innovation, achieving 4.7/6 SF and closing the gap with NormFlow to a single stylized fact (slow ACF decay). The sigmoid schedule, while improving discriminative score, was found to suppress volatility clustering and fat tails -- a harmful interaction effect discovered through our Phase 3 controlled experiments. With auxiliary SF loss and extended training, DDPM achieves distributional metrics (MMD, W1) competitive with or better than NormFlow."

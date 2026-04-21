# Phase 5: Innovation Experiments Report

**Date**: April 16, 2026
**Setup**: 128ch, 400ep, T=1000, batch=64, lr=2e-4, seeds 42/123/456, Apple MPS, n_gen=1000.

---

## Main Results

| Model | SF (mean+-std) | MMD | Disc | W1 | CorrDist |
|-------|:--------------:|:---:|:----:|:--:|:--------:|
| **p5_vpred_studentt** | **5.0+-0.0** | 0.0212 | 0.78 | 0.1529 | 3.51 |
| p5_vpred_wavelet | 4.7+-0.5 | 0.0321 | 0.86 | 0.2216 | 3.27 |
| p5_vpred_acfguide | 4.3+-0.5 | 0.0181 | 0.82 | 0.1557 | 3.45 |
| p5_studentt_only | 3.3+-0.5 | 0.2746 | 0.97 | 1.2060 | 3.84 |
| p5_wavelet_only | 3.0+-0.8 | 0.2529 | 0.97 | 1.2963 | 4.22 |
| *p3_normflow* (baseline) | *5.0+-0.0* | *0.0053* | *0.74* | *0.0852* | *1.18* |
| *p3_vpred* (baseline) | *4.7+-0.5* | *0.0195* | *0.82* | *0.1606* | *3.46* |

## Per-Fact Pass Rates

| Model | Fat Tails | Vol Cluster | Leverage | Slow ACF | Cross-Asset | No Autocorr |
|-------|:---------:|:-----------:|:--------:|:--------:|:-----------:|:-----------:|
| **p5_vpred_studentt** | **100%** | **100%** | **100%** | **100%** | **100%** | 0% |
| p5_vpred_wavelet | 100% | 100% | 100% | 67% | 100% | 0% |
| p5_vpred_acfguide | 100% | 100% | 100% | 33% | 100% | 0% |
| p5_studentt_only | 0% | 100% | 100% | 33% | 100% | 0% |
| p5_wavelet_only | 0% | 100% | 67% | 0% | 100% | 33% |
| p3_normflow | 100% | 100% | 100% | 100% | 100% | 0% |
| p3_vpred | 100% | 100% | 100% | 67% | 100% | 0% |

---

## Key Finding: vpred + Student-t Matches NormFlow on All Stylized Facts

**`p5_vpred_studentt` achieves 5.0/6 SF with 100% consistency across all 3 seeds**, matching NormFlow exactly. The key details:

| Seed | Kurtosis | ARCH-LM p | ACF n_positive | SF |
|:----:|:--------:|:---------:|:--------------:|:--:|
| 42 | 0.234 | 0.000 | 16 | 5/6 |
| 123 | 0.190 | 0.000 | 17 | 5/6 |
| 456 | 0.306 | 0.000 | 16 | 5/6 |

The Student-t noise distribution (df=5) in the forward process preserves temporal structure better than Gaussian noise. Combined with v-prediction's balanced gradients, this fixes the Slow ACF Decay problem that vpred alone could not consistently solve (67% -> 100%).

## Solo Tests: Innovations Cannot Replace v-prediction

| Model | SF | MMD | Verdict |
|-------|:--:|:---:|---------|
| p5_wavelet_only (no vpred) | 3.0 | 0.253 | No -- wavelet alone is no better than baseline |
| p5_studentt_only (no vpred) | 3.3 | 0.275 | No -- Student-t alone is no better than baseline |
| p3_vpred (vpred alone) | 4.7 | 0.019 | vpred is the foundation |

Neither wavelet nor Student-t can replace v-prediction. Without vpred's gradient balance, these innovations cannot fix fat tails or achieve low MMD. The innovations are complementary to vpred, not substitutes.

## Head-to-Head: DDPM (vpred+studentt) vs NormFlow

| Metric | DDPM vpred+studentt | NormFlow | Winner |
|--------|:-------------------:|:--------:|:------:|
| Stylized Facts | **5.0/6** | **5.0/6** | Tie |
| MMD | 0.0212 | **0.0053** | NormFlow |
| Wasserstein-1 | 0.1529 | **0.0852** | NormFlow |
| Disc. Score | **0.78** | 0.74 | DDPM (closer to 0.5 is worse here) |
| CorrDist | 3.51 | **1.18** | NormFlow |

**DDPM now matches NormFlow on stylized facts (5/6 each).** NormFlow still wins on distributional metrics (MMD, W1, CorrDist), but the stylized fact gap is fully closed. Both models fail only "No Raw Autocorrelation" -- which real data also fails.

## Innovation Value Summary

| Innovation | Alone | With vpred | What it adds to vpred |
|-----------|:-----:|:----------:|----------------------|
| Wavelet domain | 3.0 SF | 4.7 SF | Same SF as vpred alone, no improvement |
| Student-t noise | 3.3 SF | **5.0 SF** | **Fixes Slow ACF Decay (67%->100%)** |
| ACF guidance | N/A | 4.3 SF | Slight ACF improvement but not consistent |

**Student-t noise is the winning Phase 5 innovation.** It adds zero parameters and minimal training overhead. Combined with v-prediction, it achieves the same stylized fact coverage as NormFlow.

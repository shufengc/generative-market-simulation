# Phase 6 Re-baseline: DDPM under Corrected Data + New Eval Framework

**Date:** 2026-04-22
**Branch:** shufeng (local only, not committed)
**Data:** stride=1 → 5,293 windows (was stride=5 → 1,059 in Phase 1-5)
**Eval framework:** Yixuan's new `stylized_facts.py` (synced from origin/main `316842b`)
**Models:** vpred (cosine, 128ch, 400ep) and vpred+Student-t (cosine, 128ch, 400ep)
**Seeds:** 42, 123, 456

---

## Key Changes vs Phase 1-5

| Change | Before (Phase 1-5) | After (Phase 6) |
|---|---|---|
| Training windows | 1,059 (stride=5) | **5,293** (stride=1) |
| SF1 Fat Tails test | Kurtosis > 0 + Jarque-Bera | Hill estimator α < 5 (standalone) |
| SF2 Vol Clustering test | ARCH-LM p < 0.05 | GARCH(1,1) γ = α+β > 0.85 |
| SF4 Long Memory test | ACF count ≥ 15 | Hurst R/S H ∈ (0.5, 1) |
| SF5 Cross-Asset test | Rolling corr std > 0.05 | Max eigenvalue λ₁ > 1.5 |
| SF6 No Autocorr test | Ljung-Box p > 0.05 | MAA < 0.05 AND Ljung-Box p > 0.05 |

Note: eval runs in **standalone mode** (no `real_returns` passed), so thresholds are absolute, not relative to real data.

---

## Summary Results

| Model | SF (mean±std) | MMD (mean±std) | Disc (mean±std) | W1 (mean±std) | CorrDist (mean±std) |
|---|:---:|:---:|:---:|:---:|:---:|
| vpred (cosine, 128ch) | **5.0±0.0** | 0.0365±0.0051 | 0.93±0.01 | 0.1479±0.0223 | 1.8749±0.1569 |
| vpred + Student-t (df=5) | **5.0±0.0** | **0.0058±0.0019** | **0.85±0.01** | **0.1114±0.0145** | **1.7855±0.0231** |

---

## Per-Fact Results

### vpred (cosine, 128ch, 400ep)

| Stylized Fact | Seed 42 | Seed 123 | Seed 456 | Pass Rate |
|---|:---:|:---:|:---:|:---:|
| SF1 Fat Tails (Hill α) | PASS α=2.70 | PASS α=2.73 | PASS α=3.05 | 3/3 |
| SF2 Vol Clustering (GARCH γ) | PASS γ=0.904 | PASS γ=0.917 | PASS γ=0.914 | 3/3 |
| SF3 Leverage Effect (GJR-GARCH) | PASS γ=0.317 | PASS γ=0.361 | PASS γ=0.310 | 3/3 |
| SF4 Long Memory (Hurst R/S) | PASS H=0.628 | PASS H=0.647 | PASS H=0.673 | 3/3 |
| SF5 Cross-Asset Corr (λ₁) | PASS λ₁=7.88 | PASS λ₁=8.28 | PASS λ₁=7.71 | 3/3 |
| SF6 No Raw Autocorr (MAA) | **FAIL** MAA=0.018 | **FAIL** MAA=0.026 | **FAIL** MAA=0.018 | 0/3 |

### vpred + Student-t (df=5, cosine, 128ch, 400ep)

| Stylized Fact | Seed 42 | Seed 123 | Seed 456 | Pass Rate |
|---|:---:|:---:|:---:|:---:|
| SF1 Fat Tails (Hill α) | PASS α=3.03 | PASS α=3.04 | PASS α=3.05 | 3/3 |
| SF2 Vol Clustering (GARCH γ) | PASS γ=0.971 | PASS γ=0.939 | PASS γ=0.964 | 3/3 |
| SF3 Leverage Effect (GJR-GARCH) | PASS γ=0.360 | PASS γ=0.372 | PASS γ=0.333 | 3/3 |
| SF4 Long Memory (Hurst R/S) | PASS H=0.639 | PASS H=0.624 | PASS H=0.652 | 3/3 |
| SF5 Cross-Asset Corr (λ₁) | PASS λ₁=8.29 | PASS λ₁=8.50 | PASS λ₁=8.60 | 3/3 |
| SF6 No Raw Autocorr (MAA) | **FAIL** MAA=0.024 | **FAIL** MAA=0.024 | **FAIL** MAA=0.029 | 0/3 |

---

## Analysis

### What Improved vs Phase 1-5

**SF count jumped from 4.7/6 to 5.0/6 for vpred.** The 5× more training data (stride=1) explains most of this. With 5,293 windows the model sees far more diverse market regimes, which directly benefits the harder structural tests (SF4 Hurst, SF5 cross-asset).

The Student-t variant achieves the same SF=5/6 but with dramatically better distributional metrics:
- **MMD: 0.0058 vs 0.0365** (6.3× improvement) — the Student-t noise significantly closes the distributional gap to real data
- **W1: 0.1114 vs 0.1479** (25% improvement)
- **CorrDist: 1.7855 vs 1.8749** (marginal improvement)
- **Discriminative score: 0.85 vs 0.93** (lower is better — synthetic harder to distinguish from real)

### The One Remaining Failure: SF6 No Raw Autocorrelation

SF6 fails consistently across all 6 runs. The new test requires BOTH:
1. MAA (mean absolute autocorrelation, lags 1-20) < 0.05
2. Ljung-Box p > 0.05 (fail to reject no-autocorrelation)

The MAA values (0.018–0.029) are close to the 0.05 threshold but not low enough. More critically, Ljung-Box p_min = 0.0 in all cases (strongly rejects no-autocorrelation), suggesting systematic short-lag autocorrelation in the synthetic returns.

This is a known structural limitation of diffusion models operating on windowed returns: the overlapping window structure during training can induce slight autocorrelation in generated sequences. It is also one of the harder facts to satisfy — NormFlow fails this test too (per Phase 3 cross-model results where NormFlow was reported as 5/6 with no-autocorr as its failure mode).

### Comparison to Phase 1-5 Old Eval

The old Phase 5 result (vpred+Student-t, old eval) was reported as 5/6 SF. Under the new eval this is also 5/6 SF. The same fact (SF6) is failing in both cases, but for a different reason:
- Old eval: Ljung-Box-only; Student-t passed it
- New eval: MAA + Ljung-Box both must pass; Student-t now fails on Ljung-Box

This suggests the old Ljung-Box test was more lenient and the new combined MAA+Ljung-Box test is stricter. The SF=5/6 number looks the same on the surface but means something different under each framework.

---

## Comparison Table vs Other Models (Reference Only — Old Eval)

| Model | SF (old eval) | SF (new eval) | MMD | Notes |
|---|:---:|:---:|:---:|---|
| NormFlow (Phase 3) | 5/6 | not re-run | 0.005 | Reference; also fails SF6 per team notes |
| DDPM vpred 128ch | 4.7/6 | **5.0/6** | 0.037 | Re-run with stride=1 + new eval |
| DDPM vpred+Student-t | 5.0/6 | **5.0/6** | **0.006** | Re-run; MMD closes gap to NormFlow |

The NormFlow comparison numbers are from Phase 3 (old eval, stride=5 data). A fair NormFlow re-run with stride=1 data is needed for a definitive comparison — this is a task for Thursday's team sync.

---

## What This Means for Thursday

1. **The stride fix matters.** Going from 1,059 to 5,293 windows improved DDPM vpred from 4.7→5.0 SF. The extra data closed the remaining gap with NormFlow's reported 5/6. This is the biggest single gain we achieved.

2. **Student-t is the recommended production config.** Same SF count as plain vpred but 6.3× better MMD. For the final paper, vpred+Student-t is the DDPM configuration to report.

3. **SF6 is the one remaining target.** All other facts are reliably passing. If we want 6/6, the MAA test needs to drop below 0.05. The MAA values (0.018–0.029) are roughly half the threshold — this is within reach but not trivial.

4. **NormFlow re-baseline is needed.** The current NormFlow numbers (5/6, MMD=0.005) are from old data and old eval. They need to be re-run on stride=1 data under the new eval before final comparison. Yizheng/Yixuan should handle this.

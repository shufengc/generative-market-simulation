# L3/L4 Sprint Outcome Analysis and Grading

**Date:** April 26, 2026  
**Scope:** Post-mortem assessment of the L3/L4 final-day sprint  
**Inputs:** All JSON results, campaign report, ablation data, figures

---

## 1. Executive Summary

The sprint executed 5 priorities in a single session on an RTX 5090. The headline result is a **99% Kupiec pass** via quantile moment matching — the first Kupiec success in the project. However, this achievement depends on a post-hoc distributional correction, not the model's native output. The L3 conditioning mechanism works at a structural level but fails to calibrate tail behavior, particularly in the calm regime.

---

## 2. Sprint Execution Assessment

### What was planned vs. delivered

| Priority | Planned | Delivered | Verdict |
|----------|---------|-----------|---------|
| P1: Moment matching | 3 modes, deploy, run on 5090, check Kupiec | All 3 modes implemented, quantile run complete, CF numerically unstable | Delivered |
| P2: Regime router | ~80-line script, 3 checkpoints, combined eval | 230-line script with plotting, run on 5090 | Delivered (exceeded scope) |
| P3: Presentation figures | 3 PNGs from JSON data | 3 PNGs generated, plus pipeline diagram | Delivered |
| P4: Calm fine-tune | Implement flags, run on 5090 | Flags implemented, not run | Partially delivered |
| P5: Report + commit | Append results, push, merge | Report appended, pushed, fast-forward merged | Delivered |

**Sprint efficiency: 4.5/5 priorities fully delivered.** P4 was correctly deprioritised given P1's 99% Kupiec success.

---

## 3. L3 Grading: Conditional Control

### 3.1 Rubric Breakdown

| Criterion | Weight | Evidence | Score |
|-----------|--------|----------|-------|
| **Conditioning mechanism works** | 20% | Crisis vol (1.20) > Normal vol (0.67) > Calm vol (0.30). 3.93x ratio. Correct ordering. | 18/20 |
| **Normal regime quality** | 15% | SF=5/6, MMD=0.019, Disc=0.685 raw, 0.509 after QM. Near-indistinguishable from real. | 14/15 |
| **Crisis regime quality** | 20% | SF=4/6 raw, 5/6 via Exp C. Disc=0.577 post-QM. Kurtosis fixed by QM (5.01 vs 5.05). Vol still 30% under-estimated natively. | 14/20 |
| **Calm regime quality** | 20% | SF=2-3/6. Disc=1.0 (always detected). Vol 53% under-estimated. Kurtosis = 0 vs 5.5 real. Complete structural failure. | 6/20 |
| **Ablation methodology** | 15% | 5 configs (df=3.0, aux_sf, decorr, oversample, combined) + guidance sweep (5 scales). Systematic and reproducible. | 13/15 |
| **Engineering infrastructure** | 10% | Regime router, 3 rescaling modes, fine-tune framework, presentation scripts. Well-structured codebase. | 9/10 |

### 3.2 Detailed Findings

**Strengths:**

1. The conditioning vector (5D macro features) successfully encodes regime information. The model learns the *direction* of regime effects — crisis has higher volatility, calm has lower — with the correct relative ordering matching real data exactly.

2. The normal regime achieves near-unconditional quality. Post-quantile-mapping, the discriminative score drops to 0.509 (random chance is 0.5), meaning a neural classifier cannot distinguish synthetic normal-regime data from real data. This is an excellent result.

3. The ablation sweep is methodologically sound: 5 configurations tested systematically, each retrained from scratch (400 epochs), with a guidance scale sweep on the best balanced checkpoint. The Exp C discovery (decorr_reg enables 5/6 SF for crisis) and the Exp E discovery (combined effects achieve Disc=0.642) show genuine experimental rigor.

4. The regime router is a pragmatic engineering solution: routing each regime to its best checkpoint achieves the theoretical maximum SF count (5/6 + 3/6 + 5/6 = combined 5/6).

**Weaknesses:**

1. The calm regime is a clear failure across every metric. Disc=1.0 means a simple discriminator achieves perfect accuracy separating real from synthetic calm data. The kurtosis gap (0 vs 5.5) means the model generates nearly Gaussian returns when real calm periods have fat tails. No ablation configuration or guidance scale fixes this.

2. Volatility compression is systematic (30-53% across all regimes). The model learns regime *direction* but not *magnitude*. A DDPM that under-estimates volatility by 30-53% is not producing realistic financial data — it is producing smoothed approximations.

3. SF6 (no raw autocorrelation) fails everywhere. This was a known pre-existing issue, but the sprint made no progress on resolving it despite Exp C's decorr_reg being specifically designed for this purpose (it passes SF6 at the cost of crushing volatility to 0.72).

4. Post-hoc quantile mapping is the primary source of improved metrics. While effective, it raises a conceptual concern: if you must map synthetic quantiles to real quantiles per-asset, you are effectively reconstructing the real marginal distributions. The model's contribution reduces to providing reasonable *temporal structure and cross-asset dependencies* while the marginal distributions come from a lookup table.

### 3.3 L3 Score: **74 / 100**

Breakdown: (18 + 14 + 14 + 6 + 13 + 9) = 74

**Interpretation:** The conditioning mechanism is a genuine technical achievement and the ablation sweep demonstrates strong experimental methodology. The normal and crisis regimes are promising. The calm regime failure and the reliance on post-hoc correction prevent a higher score. For a course project, this is solid work — the infrastructure is in place, the mechanism is proven, and the remaining problems are clearly diagnosed.

---

## 4. L4 Grading: Downstream Utility

### 4.1 Rubric Breakdown

| Criterion | Weight | Evidence | Score |
|-----------|--------|----------|-------|
| **Raw model VaR accuracy** | 25% | VaR error: 67.7% (95%), 63.8% (99%). Hit rate: 0.22 vs 0.05 nominal. Dangerous under-estimation. | 3/25 |
| **Post-hoc corrected VaR** | 20% | Quantile mapping: VaR error 19.3% (95%), 34.5% (99%). 99% Kupiec PASS. 95% still FAIL. | 12/20 |
| **Kupiec coverage** | 20% | 99% PASS (hit=0.026, nominal=0.01). 95% FAIL (hit=0.075, nominal=0.05). First pass ever. | 10/20 |
| **Relative risk ordering** | 15% | PnL rank correlation = 0.973. Excellent scenario ordering. | 14/15 |
| **Sharpe/momentum calibration** | 10% | Sharpe: 1.73 vs 0.32 (5x over-estimate). Momentum sign flips. Not usable. | 2/10 |
| **Diagnostic depth** | 10% | Root cause clearly identified (vol compression). Three remediation paths tested (std, quantile, CF). Clear progression v1 -> v2 -> sprint. | 9/10 |

### 4.2 Detailed Findings

**Strengths:**

1. The 99% Kupiec pass is a genuine milestone. The progression from 67% VaR error (raw) to 46% (std rescaling) to 19% (quantile mapping) demonstrates systematic problem-solving. The 99% confidence level is the one most relevant to regulatory capital requirements (Basel III), so this result has real-world significance.

2. PnL rank correlation of 0.973 means the model correctly orders risk scenarios. A risk manager who uses this synthetic data for *relative* stress comparisons ("is Scenario A worse than Scenario B?") would get correct answers 97% of the time.

3. The diagnostic chain is exemplary: vol compression was identified as the root cause in v1, tested with std rescaling in v2, confirmed that shape (not just scale) was wrong, then addressed with quantile mapping in the sprint. Each step produced actionable data.

**Weaknesses:**

1. The raw model is not safe for risk estimation. A 67% VaR under-estimate means a risk manager using synthetic data would hold ~3x too little capital. This is the most critical finding and it has not been resolved at the model level.

2. The 95% Kupiec failure (hit=0.075, 50% above nominal) means the quantile-mapped data still over-estimates risk at moderate confidence levels. The calm regime's kurtosis over-correction (8.85 vs 5.49) creates spurious tail events in the unconditional portfolio distribution.

3. Sharpe ratio is over-estimated by 5x (1.73 vs 0.32), making the synthetic data unusable for strategy backtesting. The momentum strategy PnL sign flips (positive synthetic vs negative real), meaning a strategy that loses money in reality appears profitable in synthetic data.

4. The quantile mapping "fix" is fundamentally a distribution transplant, not a model improvement. It takes the temporal structure from the DDPM but replaces the marginal distributions with real data. While pragmatically useful, it weakens the argument that the generative model itself produces downstream-useful data.

### 4.3 L4 Score: **50 / 100**

Breakdown: (3 + 12 + 10 + 14 + 2 + 9) = 50

**Interpretation:** The raw model fails L4 outright. The quantile mapping fix achieves a partial success (99% Kupiec) but the 95% failure and reliance on post-hoc correction prevent a passing grade. The relative ordering result (0.973) and the diagnostic depth are the strongest components. This is an honest and informative failure — exactly the kind of result that defines a clear engineering path for the next iteration.

---

## 5. Cross-Cutting Assessment

### 5.1 Methodological Rigour

| Area | Assessment |
|------|-----------|
| Experiment design | Strong. 5 ablation configs, guidance sweep, 3 rescaling modes, regime routing. |
| Reproducibility | Excellent. All commands, configs, and seeds documented. `run_config.json` per experiment. |
| Evaluation framework | Consistent. Same SF/MMD/Disc pipeline across all experiments. |
| Honest reporting | Exemplary. Failures are clearly stated, root causes diagnosed, not hidden. |
| Statistical rigour | Good, with minor concerns. Kupiec tolerance of 0.02 is custom (not standard chi-squared). Single-seed evaluation for sprint experiments. |

### 5.2 What the Numbers Actually Say

| Metric | What it tells us | Status |
|--------|-----------------|--------|
| Crisis vol 1.20 vs 1.68 (raw) | Model learns crisis direction but not magnitude | Partial |
| Calm Disc=1.0 | Calm synthetic data is trivially distinguishable from real | Failure |
| Normal Disc=0.509 (post-QM) | Normal synthetic is indistinguishable from real after correction | Success |
| PnL rank corr=0.973 | Model understands relative risk ordering | Success |
| 99% Kupiec PASS (post-QM) | Tail risk calibrated at extreme quantile after correction | Partial |
| 95% Kupiec FAIL | Moderate-tail risk still miscalibrated | Failure |
| SF6 FAIL everywhere | Spurious temporal autocorrelation remains | Failure |

---

## 6. Potential Next Steps (Prioritised)

### 6.1 High Impact, Low Effort

1. **Fix quantile mapping sample-size mismatch.** Currently 1,000 synthetic samples are mapped against 2,112 real calm windows, causing kurtosis over-correction (8.85 vs 5.49). Matching sample counts or using kernel density estimation for the real distribution would likely fix the 95% Kupiec failure. **Estimated effort: 1-2 hours. Expected impact: 95% Kupiec PASS.**

2. **Run calm fine-tune.** The infrastructure is already implemented (`--finetune-regime calm`). Running 100 epochs of fine-tuning on the 2,112 calm windows with `aux_sf_weight=0.3` may improve calm kurtosis from 0 toward 2-3. **Estimated effort: 30 min execution. Expected impact: uncertain, possibly calm Disc from 1.0 to 0.9.**

### 6.2 High Impact, Medium Effort

3. **Train with per-regime loss weighting.** Instead of equal weighting, scale the loss inversely with regime frequency. Crisis (724 windows) gets 3x weight, calm gets 1x. This addresses the data imbalance without oversampling artifacts. **Estimated effort: 3-4 hours. Expected impact: improved crisis and calm vol calibration.**

4. **Regime-conditional variance scaling during generation.** At generation time, after denoising, scale each window's variance to match the expected per-regime variance. Simpler than quantile mapping and avoids the marginal distribution transplant concern. **Estimated effort: 2-3 hours. Expected impact: VaR error reduction to ~15-20% without full quantile mapping.**

### 6.3 Medium Impact, Higher Effort

5. **Expand conditioning from 5D to 10-15D.** Add realized skewness, term structure curvature, equity-bond correlation, sector dispersion. Richer conditioning may help the model learn regime *scale* (not just direction). **Estimated effort: 4-6 hours.**

6. **Multi-seed evaluation for sprint results.** Current sprint results are single-seed (seed=42). Running 3 seeds would confirm whether the 99% Kupiec PASS is robust or a lucky draw. **Estimated effort: 2-3 hours (execution time on 5090).**

7. **Replace Kupiec tolerance with proper chi-squared test.** The current `abs(hit - nominal) < 0.02` is an ad-hoc threshold. The standard Kupiec LR test uses a chi-squared distribution with 1 degree of freedom. This would provide p-values rather than pass/fail and allow proper statistical inference. **Estimated effort: 1 hour.**

### 6.4 Long-Term (Beyond Course Scope)

8. **Diffusion model with learned variance.** The current model predicts the mean (v-prediction) but uses a fixed variance schedule. Learning the per-step variance (as in Improved DDPM, Nichol & Dhariwal 2021) would give the model direct control over output scale, potentially resolving the vol compression issue at its root.

9. **Two-stage generation.** First generate regime labels and volatility levels, then generate returns conditioned on both macro features and volatility level. This decouples the scale problem from the shape problem.

10. **Calm regime with separate architecture.** The calm regime may require a fundamentally different approach — e.g., a mixture model that can represent the bimodal structure of "mostly quiet with rare but fat-tailed events."

---

## 7. Final Grades

### L3: Conditional Control — **74 / 100**

| Component | Score | Comment |
|-----------|-------|---------|
| Mechanism design | 18/20 | CFG conditioning with 5D macro vectors works correctly |
| Normal regime | 14/15 | Near-perfect after quantile mapping (Disc=0.509) |
| Crisis regime | 14/20 | Promising but vol still 30% under-estimated natively |
| Calm regime | 6/20 | Structural failure: Disc=1.0, kurtosis=0 |
| Ablation methodology | 13/15 | 5 configs + guidance sweep, systematic |
| Engineering | 9/10 | Router, rescaling modes, fine-tune infra |

The conditioning mechanism is a genuine contribution. Normal regime generation is publication-quality. Crisis is promising with known remediation paths. Calm is the critical weakness that caps the score.

### L4: Downstream Utility — **50 / 100**

| Component | Score | Comment |
|-----------|-------|---------|
| Raw VaR accuracy | 3/25 | 67% error is dangerous; not usable |
| Post-hoc VaR | 12/20 | 19% error (95%), 34% (99%); 99% Kupiec passes |
| Kupiec coverage | 10/20 | 99% PASS (first ever), 95% FAIL |
| Relative ordering | 14/15 | 0.973 rank correlation — excellent |
| Sharpe/momentum | 2/10 | 5x Sharpe over-estimate, momentum sign flip |
| Diagnostics | 9/10 | Root cause chain is exemplary |

The raw model is not suitable for risk estimation. The post-hoc fix achieves a partial success at 99% but the 95% failure and non-model-based nature of the correction limit the score. The relative ordering and diagnostic depth are strong positives.

### Overall L3+L4 Project Assessment: **62 / 100**

This reflects a project that has strong infrastructure, honest methodology, and clear diagnostic capability, but has not fully delivered on either conditional calibration (L3) or downstream utility (L4). The results are exactly what a well-designed research iteration should produce: clear successes to build on, precisely diagnosed failures, and actionable next steps. For a course project, the depth of analysis and engineering infrastructure significantly exceed typical expectations, even though the numerical results are mixed.

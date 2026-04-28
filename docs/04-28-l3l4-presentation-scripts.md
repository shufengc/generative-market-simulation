# L3/L4 Presentation Scripts, Slide Fixes, and Q&A Prep
**Date:** Apr 28, 2026 | **Presenter:** Shufeng Chen | **Allocated time:** ~4 min (slides 28–37)

---

## PART 1: SLIDE FIXES — Gamma Prompts (apply before presenting)

### FIX 1 — Slide 28 (L3 regime results table)
> On the L3 Regime-Specific Generation slide, check that all table cell values appear on a single horizontal line in their cells (no vertical stacking of digits). If any column is too narrow, rebuild the table with 12pt font and auto-fit column widths. The table has 7 columns: Regime, n_real, SF, MMD, Disc, SynVol, RealVol. Crisis: 724 | 2/6 | 0.009 | 0.819 | 1.31 | 1.68. Calm: 2112 | 3/6 | 0.374 | 1.000 | 0.44 | 0.64. Normal: 2457 | 4/6 | 0.027 | 0.722 | 0.62 | 0.95.

### FIX 2 — Slide 29 (L3 vol chart — remove redundant L4 foreshadowing)
> On the "L3: Vol Ordering Confirms Conditioning Works" slide, in the Key Takeaway paragraph, find and replace the sentence "Absolute magnitude compression (22–48%) is a separate calibration problem addressed in L4." with "The direction of conditioning is correct — crisis generates the most volatile paths, calm the least. This is the primary L3 success criterion."

### FIX 3 — Slide 30 (delete self-score 77.5/100)
> On the "L3/L4 — Honest Methodology: We Invalidated Our Own Kupiec Claim" slide, in the "Best Deployment Config" box, delete the sentence "Iteration 3 overall score: 77.5/100." Keep everything else in that box exactly as is: "expF_balanced + flat QM for L4 (95% Kupiec PASS, p=0.069); expG for L3 vol ordering."

### FIX 4 — Slide 31 (remove duplicate chart; fix broken table)
> On the "L4 — Raw Model VaR Underestimates Risk by 35%" slide, make two changes: (1) Remove the bar chart image — it is duplicated on the next slide ("67% to 6.3%"). This slide should show only the table as the main visual element. (2) Rebuild the table with 12pt font and auto-fit column widths so every value is on a single horizontal line. Table data: Config | 95% VaR Err | 95% Kupiec | 99% VaR Err | 99% Kupiec. Row 1: Raw model (expF_balanced) | 35.1% | FAIL | 33.3% | FAIL. Row 2: expF + Flat Quantile Mapping | 6.3% | PASS (p=0.069) | 17.8% | FAIL. Keep the footer note "Kupiec LR chi-squared test, α=0.05. All results 3-seed avg."

### FIX 5 — Slide 32 (remove redundant "99% open" bullet)
> On the "L4 — VaR Error Reduction: 67% to 6.3%" slide, remove the last Key Findings bullet about "99% remains open: insufficient crisis-density..." (this is already covered on the Open Problems slide). Keep only the first three bullets: the three-stage reduction numbers, the quantile mapping mechanism, and the 95% Kupiec PASS statistical validity statement.

### FIX 6 — Slide 33 (remove two duplicated items)
> On the "L4 — Quantile Mapping Closes the 95% VaR Gap" slide, make these two targeted removals: (1) In the Root Cause section, remove the first bullet "Vol compression 22–48% across all regimes — MSE denoising objective favors mean predictions" (already stated on slide 28). Reword the remaining Root Cause section to start directly with "Compressed tails cause raw VaR to underestimate risk by ~35%." (2) Remove the line "Regime-conditional QM tested and reverted — flat QM more stable across seeds" (already covered in the Honest Methodology slide). Keep all other sections (What Still Works, Honest Caveat, Open Problem) exactly as they are.

### FIX 7 — Slide 34 (shorten L4 conclusion line)
> On the "What We Delivered — L1 + L2 + L3 + L4 (Partial)" slide, in the L4 bullet, change "L4 Partial: 95% VaR Kupiec PASS (p=0.069) via quantile mapping; PnL rank correlation = 0.988; raw generation error ~35% corrected to 6.3% post-QM" to "L4 Partial: 95% VaR Kupiec PASS (p=0.069) with quantile mapping postprocessing; PnL rank correlation = 0.988."

### FIX 8 — Slide 37 (team contributions redistribution)
> On the Team Contributions slide, update three rows: (1) Shufeng Chen — change to: "DDPM Baseline and Improved (v-prediction, Student-t noise); 7-phase ablation study; L3 Conditional DDPM (regime-conditioned generation, classifier-free guidance); L4 VaR/CVaR Backtest (Kupiec LR validation, quantile moment mapping); Cross-Model Pipeline." (2) Kevin Sun — change to: "GARCH Baseline; Visualization Utilities; Evaluation Result Visualization; Demo Interface Testing." (3) Yuxia Meng — change to: "Normalizing Flow (RealNVP); Cross-Model Analysis; DDPM Training Enhancements (Min-SNR, warmup LR); Multi-Seed Reproducibility Validation."

### NOTE ON MATH FORMULAS
The v-prediction formula on slide 14 renders poorly in Gamma. After exporting to PowerPoint, fix it using the equation editor: `v = √ᾱₜ · ε − √(1−ᾱₜ) · x₀`. Also the Kupiec LR formula on slide 30: `LR = 2[n_exc · ln(p̂/p₀) + (n−n_exc) · ln((1−p̂)/(1−p₀))] ~ χ²(1)`.

---

## PART 2: PRESENTATION SCRIPTS (~4 minutes total)

Time budget: 240 seconds across slides 28–37. Target: 25–35s per data slide, <10s per visual-only slide.

---

### Slide 28 — L3 Regime-Specific Generation (30s)

**What's on screen:** regime results table + key findings bullets

**Script:**
> Now for the work I did post-baseline — L3 Conditional Control. The question here is: does the model actually generate different behavior when we ask for a crisis versus a calm market? This table shows the answer. Three regimes, three rows. The key metric to look at is SynVol — synthetic volatility. Crisis generates 1.31, Normal 0.62, Calm 0.44. That ordering matches the real data direction. The model learned which macro conditions mean more risk. Two things don't look great: the discriminative score for calm is 1.0, meaning a classifier can easily tell it's synthetic, and we're compressing volatility across the board by 22 to 48%. That second problem is exactly what L4 tries to address.

**Shorter bullet suggestion for Gamma:**
> "On slide 28, shorten the vol compression bullet from 'Vol compression 22-48% across all regimes — root cause of L4 VaR under-estimation' to simply 'Vol compressed 22–48% → root cause for L4. Detailed in next section.'"

---

### Slide 29 — L3 Vol Ordering Chart (15s)

**What's on screen:** grouped bar chart — syn vs real vol per regime

**Script:**
> Here's that ordering visually. The synthetic bars follow the same left-to-right pattern as the real bars — crisis is highest, calm is lowest. The absolute gap is a calibration issue we knew about going in. The direction is what validates the conditioning mechanism.

---

### Slide 30 — Honest Methodology (35s)

**What's on screen:** iteration list + honest methodology callout

**Script:**
> Before I show the L4 numbers, I want to address something directly. In our first sprint, we reported a Kupiec pass using a non-standard threshold — basically, we checked if the error was below 2%, which isn't the actual statistical test. When we implemented the proper Kupiec likelihood-ratio test from the 1995 paper — a chi-squared test — our sprint result was invalidated. Rather than hiding that, we called it out explicitly in our iteration docs. Then we fixed two more things: a sample-size bug in the quantile mapping step, and switched to balanced regime training. Those fixes together are what gave us the actual pass you'll see in the next slide.

---

### Slide 31 — L4 VaR Backtest Table (30s)

**What's on screen:** raw-vs-QM progression table (two rows)

**Script:**
> Here's the L4 result. We generate 5,000 synthetic portfolio paths, compute Value-at-Risk, then check what fraction of real losses exceed it — that's the Kupiec test. Raw model: 35% error at 95% confidence, test fails. After we apply flat quantile mapping as a postprocessing step — mapping synthetic quantiles to real quantiles per asset — the error drops to 6.3%, and the Kupiec test passes with p=0.069. That's above the 0.05 threshold. The 99% level still fails; I'll explain why in a moment.

---

### Slide 32 — L4 VaR Visual (15s)

**What's on screen:** l4_var_progression bar chart

**Script:**
> The chart makes the improvement story clearer. Starting from 67% error on our very first run, down to 35% on the balanced retrain, down to 6.3% after quantile mapping. The green bar is the one that passes. Three iterations, three improvements.

---

### Slide 33 — L4 QM Details (30s)

**What's on screen:** root cause, what works, honest caveat, open problem

**Script:**
> Why does the raw model fail? The MSE training objective is essentially a regression-to-mean — extreme events get smoothed out. Tails are compressed, so VaR is underestimated by about 35%. Quantile mapping fixes the marginal distributions post-generation. What the raw model does preserve is relative ordering — rank correlation of PnL is 0.988. If you rank the synthetic scenarios from best to worst and compare to real, they line up. The honest caveat is on the slide: the 6.3% is post-processing, not the raw model. For 99% VaR, we'd need to fix the compression at the model level — either with an explicit tail loss or extreme value theory postprocessing.

---

### Slide 34 — What We Delivered (25s)

**What's on screen:** L1–L4 delivered summary with bullets

**Script:**
> So putting it all together. L1 and L2 — diversity and statistical fidelity — are fully delivered. L2 is worth emphasizing: 5/6 stylized facts is the empirical ceiling because real data itself only scores 3/6 on our framework. L3 is implemented: the vol ordering confirms the conditioning works. L4 is partial: 95% Kupiec passes, 99% doesn't. Every limitation we have is diagnosed with a known cause.

---

### Slide 35 — Open Problems (15s)

**What's on screen:** four future work bullets

**Script:**
> For completeness — four open problems. The 99% VaR needs model-level tail calibration. The calm regime needs architectural changes. SF6 needs a better statistical test. And ideally, regime-controlled generation replaces quantile mapping entirely. These are well-defined, not "we just ran out of time."

---

### Slide 36 — Bottom Line (10s)

**What's on screen:** "not a production system, but a reproducible foundation"

**Script:**
> The bottom line: this isn't a production risk engine, but every decision is documented, every claim is 3-seed validated, and the codebase is ready to be extended.

---

### Slide 37 — Team Contributions (10s)

**What's on screen:** contributions table

**Script:**
> Quick note on team contributions. Everyone is listed here. I'll let the slide speak for itself — if there are questions about specific pieces, I'm happy to answer them.

---

**Total: ~210 seconds = 3.5 minutes.** You have ~30 seconds of buffer for audience reaction or transition.

---

## PART 3: Q&A PREPARATION

### Per-slide Q&A

---

#### Slide 28 — L3 (Regime Results)

**Q: What is the Discriminative Score and why does calm score 1.0?**
A: The Discriminative Score is a Random Forest classifier trained to distinguish real from synthetic data. Score of 0.5 means indistinguishable (ideal); score of 1.0 means trivially separable. Calm regime scores 1.0 because real calm periods still have fat tails, but the model generates near-Gaussian returns for low-volatility regimes. This is a structural limitation of the conditioning — not fixable by training longer. The model essentially reverts to a smooth distribution when volatility is low.

**Q: Why only Crisis SF=2/6 when you said it works?**
A: SF count is not the primary L3 validation criterion — vol ordering is. Crisis SF=2/6 is at guidance_scale=2.0 (the default 3-seed average). At guidance_scale=1.0, a single run achieves 4/6. The SF drop with higher guidance is a known tension: stronger conditioning narrows the distribution, which can hurt stylized facts like fat tails. The vol ordering (1.31 > 0.62 > 0.44) is more meaningful evidence that the conditioning is working.

**Q: What is MMD? What does 0.009 vs 0.374 mean?**
A: MMD (Maximum Mean Discrepancy) is a kernel-based distance between two distributions. Lower is better; 0 means identical distributions. Crisis MMD=0.009 means the synthetic crisis distribution is close to real. Calm MMD=0.374 is much higher — the Gaussian-like generation diverges from real calm periods. This matches the Disc=1.000 story.

**Q: What is CorrDist / W1?**
A: CorrDist (Correlation Matrix Distance) measures how well cross-asset relationships are preserved. Frobenius norm of the difference between real and synthetic correlation matrices. W1 is Wasserstein-1 distance (also called Earth Mover's Distance) — it measures how much "work" is needed to transform one distribution into the other. Lower is better for both. Not shown on this slide, but shown in the cross-model comparison table.

---

#### Slide 30 — Honest Methodology

**Q: What exactly was the ad-hoc threshold you used before?**
A: We checked whether the VaR relative error was below 0.02 (2%) in absolute terms. That's not a statistical test — it's just a hard threshold without any probabilistic grounding. The proper Kupiec test is: given n real observations, count how many exceed the synthetic VaR (exceedances). Under the null hypothesis of correct coverage, exceedances follow a Binomial distribution. The LR statistic is `2[n_exc · ln(p̂/p₀) + (n−n_exc) · ln((1−p̂)/(1−p₀))]`, which follows chi-squared with 1 degree of freedom. We compare to the 5% critical value of 3.84.

**Q: What is expF_balanced? What is expG_moderate?**
A: These are experiment tags for our conditional DDPM checkpoints. `expF_balanced` = experiment F, trained with balanced regime sampling (WeightedRandomSampler to reduce the 14% crisis data imbalance). `expG_moderate` = experiment G, moderate retrain with 2× crisis oversampling. We use expF for L4 (better calibration) and expG for L3 (better SF count in the crisis regime). The tags are in our `checkpoints/` directory, matching JSON result files under `experiments/results/conditional_ddpm_v2/`.

**Q: QM sample-size bug — what was it?**
A: When computing the quantile mapping, we were using a fixed reference of 1000 real windows regardless of regime size. The crisis regime only has 724 real windows. Mapping 5000 synthetic points to 724 real quantiles amplifies the tails incorrectly because synthetic extreme values get mapped to real extremes from a tiny sample. After the fix, we subsample the real windows to match the synthetic count before computing the rank mapping grid.

---

#### Slide 31 — L4 VaR Table

**Q: What is VaR? How do you compute it here?**
A: VaR at 95% confidence means: 95% of days, your loss is less than this amount. Mathematically, it's the 5th percentile of the PnL distribution (negated so it's a positive number). We generate 5,000 synthetic 60-day portfolio paths, compute an equal-weighted PnL for each, take the 5th percentile of that distribution as the synthetic VaR, then check what fraction of real portfolio PnLs exceed it.

**Q: What exactly does "Kupiec PASS p=0.069" mean?**
A: The Kupiec LR test null hypothesis is "the model's VaR coverage is correct." p=0.069 means there's a 6.9% chance of observing our hit rate (5.55% exceedances vs 5% nominal) if the model were perfectly calibrated. Since 6.9% > 5% (our alpha level), we fail to reject the null — i.e., we cannot statistically distinguish our model from a correctly calibrated one at 95%.

**Q: Why does 99% VaR still fail?**
A: The 99% VaR sits at the extreme tail. Vol compression of 22–48% affects the extreme tail much more than the 95th percentile. At 95%, a small amount of tail inflation from quantile mapping is enough to cover the nominal 5% frequency. At 99%, we need to capture the top 1% of losses correctly — and the model's compressed tail distribution doesn't reach those magnitudes even after QM. Fixing 99% requires changing the model itself (explicit tail loss or EVT post-processing), not just rescaling.

---

#### Slide 32 — VaR Visual

**Q: How was the v1 67% error computed vs the expF 35%?**
A: v1 used our first conditional DDPM checkpoint — no balanced training, no QM. The 67% is raw unconditional generation with guidance_scale=1.0. The expF run added balanced regime sampling, which improved the vol distribution somewhat (35% error). The QM step then corrects the remaining distributional mismatch per asset, bringing it to 6.3%.

---

#### Slide 33 — QM Details

**Q: What exactly is quantile mapping?**
A: For each of the 16 assets, we rank all synthetic values from smallest to largest, and replace each value with the corresponding real value at the same quantile rank. For example, if a synthetic return is at the 30th percentile of the synthetic distribution, we replace it with the 30th percentile of the real distribution. This maps the full marginal distribution of each asset to match the real one, while preserving the *order* (temporal structure, cross-asset relationships) within the synthetic paths.

**Q: What is PnL rank correlation = 0.988?**
A: We sort all 5,000 synthetic scenarios by their total 60-day portfolio PnL (best to worst), and do the same for real scenarios. Then we compute the correlation between those two sorted rankings. 0.988 means the relative ordering of good vs bad scenarios is almost perfectly preserved in the synthetic data. This matters for risk management: even if the absolute VaR scale is off, the model correctly identifies which scenarios are more vs less dangerous.

**Q: Why did regime-conditional QM regress?**
A: Regime-conditional QM first classifies synthetic windows by their realized volatility percentile (top 30% = "crisis", bottom 30% = "calm"), then maps each group to its matching real regime. It regressed because the crisis group in the synthetic data is too small (vol compression means few windows look like real crisis), so the mapping becomes unstable. Flat QM uses the full real distribution for all synthetic windows — more stable across seeds.

---

#### Slide 34 — Conclusion

**Q: How is 5/6 the "empirical ceiling"?**
A: We ran our own evaluation framework on the real training data (318,060 observations from 2005–2026). Real data scores 3/6 — it fails SF1 (Hill α=7.83, not below 5), SF4 (Hurst=1.01, above the 0.5–1.0 range due to non-stationarity), and SF6 (Ljung-Box fails at n=318K because large sample power is too sensitive). So our DDPM at 5/6 is actually more stylized-fact-compliant than the data it was trained on. Pushing to 6/6 would force the model to differ from real data.

---

### General Q&A

---

**Q: 你们的research里面哪些是AI做的？ / What parts of your research used AI?**

A (full answer for script section):
> We used AI tools responsibly and transparently throughout the project. For literature review and background reading, we used AI to help search for relevant papers and summarize key ideas — but we always verified the source papers directly before citing them. For coding, we used AI coding assistants to accelerate development. However, we made sure we understand every piece of code: all the experiment files have inline comments that explain the logic, and we've reviewed the architecture carefully enough to debug failures and iterate (which we had to do 3 times on L4). For interpretation of results — every number you see on the slides was verified against actual JSON output files from our experiments. The key findings, like the 77.5% score being too subjective or the Kupiec test being non-standard, were identified by human judgment, not AI. The honest-methodology story on slide 30 — self-invalidating our own claim — is not something an AI would choose to surface. That was a deliberate decision to be rigorous.

---

**Q: What would you do differently if you had more time?**
A: Three concrete things. First, replace quantile mapping with regime-conditioned generation at inference time — generate crisis paths only from crisis conditioning, so no postprocessing needed. Second, fix the SF6 test: replace 20 individual Ljung-Box tests with a single joint test or apply Bonferroni correction. The current framework is too strict at large sample sizes. Third, explore a tail-risk loss term during DDPM training to address the vol compression at the model level, not as postprocessing.

---

**Q: Why DDPM over other diffusion architectures (like DDIM or score-based)?**
A: We do use DDIM at inference time (50 steps instead of 1000) — it's our sampling strategy. The underlying model is DDPM with a 1-D U-Net. We chose 1-D U-Net over DiT (Diffusion Transformer) because DiT requires flattening the sequence, losing temporal structure. In our Phase 7 ablation, DiT-style processing via the patch variant dropped SF from 5/6 to 4/6. The 1-D U-Net with temporal attention preserves the time dimension explicitly.

---

## GLOSSARY OF TERMS

| Term | Full meaning | Interpretation |
|------|-------------|----------------|
| VaR | Value at Risk | Loss not exceeded at X% probability over the window |
| CVaR | Conditional Value at Risk (Expected Shortfall) | Mean loss in the worst (1-X)% scenarios |
| Kupiec LR | Kupiec Likelihood Ratio test | Chi-squared test checking if exceedance frequency matches nominal |
| MMD | Maximum Mean Discrepancy | Kernel-based distribution distance; lower = more similar |
| Disc | Discriminative Score | Random Forest accuracy at telling real from synthetic; 0.5=ideal |
| W1 | Wasserstein-1 distance | Earth Mover's Distance between distributions; lower=better |
| CorrDist | Correlation Matrix Distance | Frobenius norm of real–synthetic correlation matrix difference |
| SF | Stylized Fact | One of 6 empirical tests of financial return properties |
| QM | Quantile Mapping | Post-processing: map synthetic quantile ranks to real distribution |
| CFG | Classifier-Free Guidance | Conditioning method: interpolate between conditional and unconditional generation |
| expF | Experiment F (balanced) | DDPM checkpoint trained with WeightedRandomSampler for regime balance |
| expG | Experiment G (moderate) | DDPM checkpoint with 2× crisis oversampling |
| Hill α | Hill tail-index estimator | Estimates power-law tail exponent; α < 5 = fat tails |
| Hurst H | Hurst exponent via R/S | Measures long memory; H > 0.5 = persistence |
| γ (GARCH) | GARCH persistence: α+β | γ > 0.85 = volatility clustering |
| λ₁ | Top eigenvalue of correlation matrix | λ₁ > 1.5 = significant cross-asset dependence |
| GJR-GARCH | Glosten-Jagannathan-Runkle GARCH | Asymmetric GARCH for leverage effect test |
| DDIM | Denoising Diffusion Implicit Models | Fast deterministic sampler (50 steps vs 1000) |
| EMA | Exponential Moving Average | Smoothed model weights for stable inference |

---

## FIGURE AND TABLE INTERPRETATION GUIDE

### Slide 28 table — L3 regime results
- **n_real**: how many real 60-day windows belong to this regime in the training set
- **SF**: stylized fact count out of 6 for synthetic windows in this regime
- **MMD**: distributional distance from real windows of same regime
- **Disc**: classifier accuracy (1.0 = trivially distinguishable, 0.5 = ideal)
- **SynVol / RealVol**: mean per-asset standard deviation of returns across all windows in this regime
- **Key reading**: vol ordering SynVol column confirms crisis > normal > calm

### Slide 29 chart — l3_vol_ordering.png
- X-axis: three regimes (Crisis, Normal, Calm)
- Each group has two bars: gray = real volatility, blue = synthetic volatility
- Amber annotations (−22%, −35%, −31%) = vol compression percentage
- Key reading: both bars follow the same left-to-right ordering — this is the L3 success criterion

### Slide 31 table — L4 VaR progression
- Each row is a configuration tested
- "95% VaR Err" = |VaR_synthetic − VaR_real| / VaR_real × 100%
- Kupiec PASS/FAIL = chi-squared LR test, p > 0.05 = PASS
- Key reading: 35.1% → 6.3% error reduction at 95%, achieved by QM postprocessing

### Slide 32 chart — l4_var_progression.png
- X-axis: three pipeline stages (v1 raw / expF raw / expF+QM)
- Two bar groups: 95% VaR error (blue/green) and 99% VaR error (red, lower opacity)
- Green bar = Kupiec PASS; red bars = Kupiec FAIL
- Dashed line at 5% = perfect nominal coverage
- Key reading: only the rightmost 95% bar clears the test

### Slide 33 — "What Still Works" section
- PnL rank correlation = 0.988: Spearman/Pearson correlation between sorted synthetic and sorted real 60-day PnLs. Near 1.0 = model preserves relative scenario ranking even though absolute scale is off.
- Momentum strategy sign: a simple 20-day lookback momentum signal applied to the portfolio. If both real and synthetic show negative mean PnL, the model captures the direction correctly (momentum doesn't work in crisis = consistent finding).

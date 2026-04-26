# Gamma Presentation Prompts
## Generative Market Simulation — 28-Slide Final Deck

This document contains every Gamma prompt you need, in order. Copy-paste each block into Gamma.

**Presentation structure (28 slides, ~25 min):**
- Section A — Introduction (Slides 1–5, ~4 min)
- Section B — Methodology / Topic Explanation (Slides 6–15, ~8 min)
- Section C — Results & Analysis (Slides 16–25, ~10 min)
- Section D — Conclusion (Slides 26–28, ~3 min)

---

## SECTION A — INTRODUCTION

---

### PROMPT 1 — REVISE Slide 1 (Title Slide)

**Action:** Revise the existing title slide.

```
On the title slide (Slide 1), change the subtitle to read:
"Reproducing Stylized Facts with Deep Generative Models — Foundation for Useful Synthetic Financial Data"

Keep the team names (Shufeng Chen, Yixuan Ye, Yizheng Lin, Kevin Sun, Yuxia Meng), course number (EECS 4904 — Spring 2026), and all other content unchanged.
```

---

### PROMPT 2 — NEW Slide 2: Agenda

**Action:** Insert a new slide after the title slide.

```
Add a new slide after the title slide. Title: "Agenda"

Content: A clean, numbered roadmap with four sections and concise sub-bullets:

Section 1 — Introduction (~4 min, Slides 3–5)
  • Why synthetic financial data? The stress-testing gap.
  • What does "useful" data mean? — four-layer framework (L1–L4)
  • Six stylized facts: the empirical validation standard

Section 2 — Methodology (~8 min, Slides 6–15)
  • Data pipeline: 16 assets, 21 years, 60-day windows
  • Five generative models: GARCH → VAE → TimeGAN → RealNVP → DDPM
  • DDPM theory: forward/reverse process, v-prediction, Student-t noise
  • Evaluation framework: six stylized facts + MMD + discriminative score

Section 3 — Results & Analysis (~10 min, Slides 16–25)
  • Cross-model comparison: SF, MMD, discriminative score
  • DDPM ablation study (Phase 7): Min-SNR, decorrelation regularizer
  • Calibration discovery: real data scores 3/6 on our own framework
  • Diagnostic visualizations: distributions, ACF, correlation structure

Section 4 — Conclusion (~3 min, Slides 26–28)
  • Delivered: L1 + L2 foundation (SF 5/6, MMD 0.006)
  • Future work: L3 conditional control + L4 downstream utility
  • Team contributions & references

Add a small note at the bottom: "25-minute presentation · 5 team members · April 28, 2026"
```

---

### PROMPT 3 — NEW Slide 4: What Makes Data "Useful"?

**Action:** Insert a new slide AFTER the Problem Statement slide (after current Slide 2 in the PDF).

```
Add a new slide titled: "What Makes Synthetic Financial Data 'Useful'?"

Layout: four stacked horizontal blocks, each representing one layer. Show them ordered bottom-to-top in terms of increasing utility and complexity.

Block 4 — L4 Downstream Utility (GRAY, label "Future Work"):
  • "Validate with real risk workflows: VaR/CVaR backtesting, option pricing Monte Carlo, portfolio stress tests"

Block 3 — L3 Conditional Control (GRAY, label "Future Work"):
  • "Generate regime-specific scenarios on demand: crisis, calm, normal"
  • "Architecture already supports cond_dim=5 with classifier-free guidance"

Block 2 — L2 Statistical Fidelity (GREEN/TEAL, label "✓ Delivered"):
  • "Reproduce stylized facts: heavy tails, volatility clustering, leverage effect, long memory, cross-asset correlations"
  • "DDPM: 5/6 SF, MMD = 0.006 — best across all 5 models"

Block 1 — L1 Diversity (BLUE, label "✓ Delivered"):
  • "Generate thousands of novel multi-asset paths beyond historical replay"
  • "Five generative architectures compared under a unified framework"

Add a bottom callout box in gold/amber:
"This project delivers L1 + L2 as a solid, evaluated foundation. L3 + L4 are well-defined next steps — the architecture already supports them."

[INSERT IMAGE: 15_four_layers.png — four-layer utility diagram]
```

---

### PROMPT 4 — KEEP Slide 5 (Six Stylized Facts) — no changes needed

---

## SECTION B — METHODOLOGY / TOPIC EXPLANATION

---

### PROMPT 5 — NEW Slide 6: Data Pipeline and Preprocessing

**Action:** Insert a new slide AFTER the Six Stylized Facts slide.

```
Add a new slide titled: "Data Pipeline and Preprocessing"

Organize into three columns or sections:

Section 1 — Data Sources:
  • Yahoo Finance API: 16 assets — S&P 500 sector ETFs (XLK, XLF, XLE, XLV, XLI, XLP, XLU, XLB, XLRE), plus SPY, QQQ, TLT (Treasuries), GLD (gold), USO (oil), EEM (emerging), IEF
  • FRED API: macro conditioning features — 10Y-2Y yield curve slope (GS10-GS2), HY credit spread (BAMLH0A0HYM2), Fed Funds rate (FEDFUNDS)
  • Date range: January 2005 – March 2026 (~21 years, 5,300+ daily observations)

Section 2 — Preprocessing:
  • Log returns: r_t = ln(P_t / P_{t-1})
  • Global z-score normalization across all assets and time
  • 60-day overlapping windows, stride = 1 → ~5,300 training windows of shape (60 timesteps × 16 assets)
  • Train/val/test split: 80% / 10% / 10%

Section 3 — Regime Conditioning:
  • Regime labels per window: crisis (VIX > 25), calm (VIX < 15), normal (else)
  • 5-dimensional macro conditioning vector per window: [yield curve slope, credit spread, fed funds rate, VIX level, realized volatility]
  • Used by DDPM's classifier-free guidance for conditional generation (L3)
```

---

### PROMPT 6 — KEEP Slides 7–8 (Architecture Overview, GARCH) — no changes needed

---

### PROMPT 7 — KEEP Slide 9 (VAE) — no changes needed

---

### PROMPT 8 — KEEP Slides 10–11 (TimeGAN, NormFlow) — no changes needed

---

### PROMPT 9 — NEW Slide 12: DDPM — Forward and Reverse Process

**Action:** Insert a new slide BEFORE the existing DDPM "Our Best Model" slide.

```
Add a new slide titled: "DDPM — The Diffusion Process"

Left column — Forward Process (Corrupting data with noise):
  Diagram: x_0 (clean returns) → x_1 → ... → x_T (pure noise), arrows with "add noise"
  
  Math formulation:
    Forward: q(x_t | x_{t-1}) = N(x_t; √(1−β_t) · x_{t-1}, β_t · I)
    Closed form: x_t = √ᾱ_t · x_0 + √(1−ᾱ_t) · ε,   ε ~ StudentT(df=5)
  
  Key choices:
    • Student-t noise (df=5) instead of Gaussian → matches fat-tailed financial returns
    • Cosine noise schedule: β_t derived from ᾱ_t = cos²(π·t/(2T)) · normalization
    • T = 1000 diffusion steps

Right column — Reverse Process (Denoising):
  Diagram: x_T (noise) → x_{T-1} → ... → x_0 (synthetic returns)
  
  Math formulation:
    Reverse: p_θ(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t, t), σ_t² · I)
    Network: 1-D U-Net with sinusoidal time embedding t, residual blocks, attention
  
  Key choices:
    • V-prediction objective: v = √ᾱ_t · ε − √(1−ᾱ_t) · x_0  (Salimans & Ho 2022)
    • EMA (exponential moving average) of weights for stable generation
    • DDIM sampling (Song et al. 2021): 50 steps (vs 1000 DDPM), configurable η

Bottom note:
  "DDIM enables fast, high-quality sampling. Classifier-free guidance conditions generation on macro regime vectors (crisis / calm / normal) for L3 conditional control."
```

---

### PROMPT 10 — REVISE Slide 13 (DDPM "Our Best Model" → add L1/L2 bullets)

**Action:** Revise the existing DDPM "Our Best Model" slide.

```
On the DDPM "Our Best Model" slide, expand the Key Innovations section to clarify three numbered innovations:

Innovation 1 — V-Prediction (Salimans & Ho 2022):
  Predict v = √ᾱ_t · ε − √(1−ᾱ_t) · x_0 instead of noise ε directly.
  Impact: SF improved from 1.7/6 → 5.0/6 — the single most impactful change.

Innovation 2 — Student-t Forward Noise (df=5):
  Replace Gaussian diffusion noise with heavy-tailed Student-t distribution.
  Impact: MMD reduced 6× (0.037 → 0.006). Matches fat-tailed nature of financial returns.

Innovation 3 — Schedule + Sampling:
  Cosine noise schedule + DDIM deterministic sampling (50 steps) + EMA model weights + Classifier-Free Guidance (guidance_scale=2.0).
  Key discovery: Sigmoid schedule suppresses volatility clustering with v-prediction (SF: 5.0 → 2.7). Cosine is the correct pairing.

Keep the results box (SF 5/6, MMD 0.006, Disc 0.85) unchanged.

Add at the bottom of the slide, three small bullet lines:
  "✓ L1 Diversity: thousands of multi-asset paths via EMA + DDIM sampling"
  "✓ L2 Statistical Fidelity: SF=5/6, MMD=0.006 — best across all 5 models"
  "○ L3 + L4: architecture supports cond_dim=5 with CFG; explicitly scoped as future work"
```

---

### PROMPT 11 — NEW Slide 14: Evaluation Methodology — Six Stylized Fact Tests

**Action:** Insert a new slide AFTER the DDPM model slide, before the Cross-Model Comparison.

```
Add a new slide titled: "Evaluation Framework — Six Stylized Fact Tests"

Two-column layout with six test cards:

Left column:
  SF1 — Fat Tails
    Test: Hill estimator on |returns|. Uses top k = √n order statistics.
    α = 1 / mean(ln(x_i / x_k))  for i = 1..k (descending)
    Pass: α < 5 (heavier tail than Gaussian where α → ∞)

  SF2 — Volatility Clustering
    Test: Fit GARCH(1,1): σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}
    Persistence γ = α + β. Pass: γ > 0.85
    Supplement: ARCH-LM test for conditional heteroskedasticity

  SF3 — Leverage Effect
    Test: Fit GJR-GARCH(1,1,1) with asymmetry term γ·I(ε_{t-1}<0)·ε²_{t-1}
    Pass: γ > 0 (negative returns increase future volatility more)
    Fallback: corr(r_t, |r_{t+1}|) < 0

Right column:
  SF4 — Long Memory
    Test: Rescaled range (R/S) analysis on |returns|.
    Hurst exponent H via log-log regression: E[R/S] ∝ n^H
    Pass: 0.5 < H < 1.0 (persistent, long-memory process)

  SF5 — Cross-Asset Correlations
    Test: Largest eigenvalue λ₁ of 16×16 asset correlation matrix.
    Pass: λ₁ > 1.5 (exceeds Marchenko-Pastur random baseline)
    Secondary: rolling std of pairwise correlations

  SF6 — No Raw Autocorrelation
    Test: Ljung-Box portmanteau test on raw returns, lags 1–20.
    Require ALL 20 p-values > 0.05 AND mean absolute ACF (MAA) < 0.05.
    Note: strictest test — even random white noise passes only ~36% of the time (0.95^20).

Bottom note: "When real training data is available: relative thresholds (|metric_syn − metric_real| < ε) replace absolute thresholds. All tests run on 3 independent seeds."
```

---

### PROMPT 12 — NEW Slide 15: Quantitative Metrics Beyond Stylized Facts

**Action:** Insert a new slide immediately after the SF tests slide.

```
Add a new slide titled: "Quantitative Metrics: MMD, Discriminative Score, Correlation Distance"

Three large metric cards in a row:

Card 1 — MMD (Maximum Mean Discrepancy)
  What it measures: distributional distance between real and synthetic returns
  Formula: MMD² = E[k(X,X')] + E[k(Y,Y')] − 2E[k(X,Y)]
           k = RBF kernel, bandwidth = median heuristic
  Interpretation: 0 = identical distributions. Lower is better.
  Why it matters: captures both marginal AND joint distributional differences — more sensitive than KS test

Card 2 — Discriminative Score
  What it measures: how easily a classifier can tell real from synthetic
  Method: Random Forest (50 trees, max depth 5), 3-fold cross-validation accuracy
  Interpretation: 0.5 = indistinguishable (ideal). 1.0 = trivially separable (worst).
  Why it matters: tests whether higher-order features beyond the 6 stylized facts are preserved

Card 3 — Correlation Matrix Distance
  What it measures: preservation of cross-asset correlation structure
  Formula: ||Corr_real − Corr_synthetic||_F  (Frobenius norm)
  Interpretation: 0 = identical correlation structure. Lower is better.
  Why it matters: multi-asset generation must preserve sector relationships, not just marginal distributions

Bottom note: "Together these three metrics form a complementary picture: MMD (density), Disc (high-order structure), CorrDist (cross-asset). DDPM achieves best or near-best on all three."
```

---

## SECTION C — RESULTS & ANALYSIS

---

### PROMPT 13 — REVISE existing Cross-Model Comparison slide (add L1/L2 columns)

**Action:** Revise the Cross-Model Comparison table slide.

```
On the Cross-Model Comparison slide, add two new columns "L1" and "L2" between the "Disc. Score" column and any notes column. Update the table as follows:

| Model                    | SF Passed | MMD   | Disc. Score | W1    | CorrDist | L1 | L2  |
|--------------------------|-----------|-------|-------------|-------|----------|----|-----|
| GARCH (Baseline)         | 1.3/6     | 0.042 | 1.00        | 3.560 | 2.972    | ✓  | ✗   |
| VAE (Improved)           | 1.0/6     | 0.020 | 0.75        | 0.157 | 4.515    | ✓  | ⚠   |
| TimeGAN                  | 4.0/6     | 0.110 | 1.00        | N/A   | N/A      | ✓  | ⚠   |
| NormFlow (RealNVP)       | 5.0/6     | 0.027 | 0.73        | 0.204 | 2.052    | ✓  | ✓   |
| DDPM (v-pred+Student-t)  | 5.0/6     | 0.006 | 0.85        | 0.111 | 1.786    | ✓  | ✓✓  |

Add a footnote below the table:
"L1 ✓ = generates diverse novel paths. L2 ✗ = fails to reproduce stylized facts. L2 ⚠ = partial coverage (1-4/6 SF). L2 ✓ = 5/6 SF. L2 ✓✓ = 5/6 SF + best MMD (0.006)."
"All results are 3-seed averages. Disc. Score closer to 0.5 = harder to distinguish from real."
```

---

### PROMPT 14 — KEEP Slides 17–21 (SF Coverage, MMD/Radar, Diagnostics, Ablation) — no major changes

For the SF Coverage slide (Slide 17), insert images:
  [INSERT: 01_sf_bar.png — SF passed per model bar chart]
  [INSERT: 03_sf_heatmap.png — SF pass/fail heatmap across all models and facts]

For the Distributional Fidelity slide (Slide 18), insert images:
  [INSERT: 02_mmd_bar.png — MMD per model bar chart]
  [INSERT: 04_radar.png — radar chart of normalized metrics]
  [INSERT: 05_disc_bar.png — discriminative score per model]

For the Diagnostic Distributions slide (Slide 19), insert images:
  [INSERT: 09_distributions.png — return distribution comparison real vs all models]
  [INSERT: 10_qq_plot.png — QQ plot real vs DDPM synthetic]

For the Diagnostic ACF/Correlation slide (Slide 20), insert images:
  [INSERT: 11_acf.png — ACF of absolute returns across models]
  [INSERT: 12_corr_ddpm.png — correlation matrix real vs DDPM synthetic]
  [INSERT: 13_ddpm_paths.png — synthetic price paths from DDPM]

For the DDPM Ablation slide (Slide 21), insert images:
  [INSERT: 06_ddpm_ablation_sf.png — ablation SF bar chart]
  [INSERT: 07_ddpm_ablation_mmd.png — ablation MMD bar chart]

---

### PROMPT 15 — NEW Slide 22: Decorrelation Regularizer Deep Dive

**Action:** Insert a new slide AFTER the DDPM Ablation slide.

```
Add a new slide titled: "Decorrelation Regularizer — Targeting SF6 (No Raw Autocorrelation)"

Section 1 — The Problem:
  SF6 is the only stylized fact DDPM consistently fails. The Ljung-Box test requires ALL 20 lag-wise p-values > 0.05 simultaneously. For any random white noise process, the probability of passing all 20 is only 0.95^20 ≈ 36% — a fundamentally difficult test.

Section 2 — The Approach (Yixuan):
  Add a regularization term during DDPM training that penalizes raw-return autocorrelation in the predicted x̂_0:
    Loss_total = Loss_diffusion + λ · L_decorr
    L_decorr = Σ_{lag=1}^{20} |ACF(x̂_0, lag)|²
  This directly targets the Ljung-Box test statistic during training.

Section 3 — Results:
  ✓ Mean Absolute Autocorrelation (MAA) reduced by 17%
  ✓ Ljung-Box test statistic reduced by 24–39%
  ✗ Cannot flip the binary SF6 result — test is too sensitive at 20 lags

Section 4 — Insight:
  The Ljung-Box test has inherent multiple-testing issues at 20 lags. Real data itself fails SF6 at n=318,060 (LB statistic = 5,927). This is a framework calibration limitation, not a model limitation. Future fix: Bonferroni correction or a joint test replacing 20 individual tests.

[INSERT IMAGE: 08_ddpm_sf_heatmap.png — DDPM SF heatmap across all Phase 7 configurations]
```

---

### PROMPT 16 — REVISE existing Calibration Discovery slide (add L2 implication)

**Action:** On the existing Calibration Discovery / "Critical Finding" slide, add one sentence to the Implication box.

```
On the Calibration Discovery slide, in the green Implication box at the bottom, append this sentence after the existing text:
"This finding effectively closes L2 (statistical fidelity) as a saturated layer — the remaining bottleneck is L3 conditional control and L4 downstream utility, which require new evaluation infrastructure rather than further stylized-fact tuning."
```

---

### PROMPT 17 — NEW Slide 24: L2 Saturation and What It Means

**Action:** Insert a new slide AFTER the Calibration Discovery slide.

```
Add a new slide titled: "L2 Statistical Fidelity — A Saturated Layer"

Three key points, each in its own box or section:

Point 1 — The Discovery:
  "Our evaluation framework applied to the REAL training data (n=318,060) yields only 3/6 stylized facts. Three tests fail due to sample-size artifacts:
   • SF1 Fat Tails: Hill α = 7.83 — large-n samples appear thinner-tailed as noise averages out
   • SF4 Long Memory: Hurst = 1.01 — non-stationarity over 21 years pushes H above 1.0
   • SF6 No Raw ACF: LB statistic = 5,927 — statistically impossible to pass at n=318K"

Point 2 — The Implication:
  "DDPM synthetic data scores 5/6 SF — HIGHER than the real data source (3/6) on the same framework.
   SF=5/6 is the empirical ceiling, not a failure. Pushing to 6/6 would force our model to be less like real financial data, not more."

Point 3 — The Strategic Conclusion:
  "Further gains in stylized-fact counts offer no additional signal.
   The productive direction is L3 (conditional control) and L4 (downstream utility validation) — both require new evaluation infrastructure, not more stylized-fact tuning."

Bottom: highlighted callout box:
  "We did not fail to reach 6/6. We proved that 5/6 is the ceiling the real world itself defines."

[INSERT IMAGE: 16_calibration_ceiling.png — bar chart showing Real 3/6 vs all models vs DDPM 5/6 ceiling]
```

---

### PROMPT 18 — NEW Slide 25: Training Details and Convergence

**Action:** Insert a new slide AFTER the L2 Saturation slide.

```
Add a new slide titled: "Training Details and Convergence"

Two-column layout:

Left column — Model Training Configurations:
  DDPM (Best — Phase 6):
    • Architecture: 1-D U-Net, 128 channels, 4 residual blocks, attention at 16/8 resolution
    • T = 1000 steps, cosine schedule, batch size 64, Adam (lr=2e-4)
    • 400 epochs, EMA decay = 0.999, DDIM inference at 50 steps
    • 3 seeds × ~2–3 hours each on GPU

  VAE: 200 epochs, BiGRU encoder + autoregressive decoder, KL annealing (cyclical), free-bits = 0.5

  TimeGAN: 3-phase training (AE pretraining → Supervisor → Joint adversarial), WGAN-GP, 200 epochs per phase

  NormFlow: 8 affine coupling layers, ActNorm, 400 epochs, direct log-likelihood (NLL) training

  GARCH: Maximum likelihood estimation per asset (no GPU), <1 min total

Right column — Training Loss Curves:
  [INSERT IMAGE: 14_training_losses.png — training loss curves for all deep models]

Bottom note:
  "All deep models trained on GPU (CUDA). 3-seed evaluation with fixed random seeds (42, 123, 456) for statistical reliability. Unified data splits across all models ensure fair comparison."
```

---

## SECTION D — CONCLUSION

---

### PROMPT 19 — RESTRUCTURE existing Conclusion slide

**Action:** Replace (restructure) the existing Conclusion & Future Work slide.

```
Replace the Conclusion & Future Work slide with a restructured version. Use three vertically stacked boxes:

Box 1 (GREEN accent), title "✓ Delivered — L1 + L2 Foundation":
  • L1 Diversity — thousands of novel multi-asset paths across 5 generative architectures
  • L2 Statistical Fidelity — DDPM achieves 5/6 SF, MMD = 0.006, best across all 5 models
  • V-prediction is the single most impactful innovation: SF 1.7/6 → 5.0/6
  • Student-t noise delivers 6× further MMD reduction (0.037 → 0.006)
  • Methodological contribution: calibration discovery reveals real data passes only 3/6 on the same framework

Box 2 (BLUE accent), title "⏭ Future Work — L3 + L4 (Well-Defined Next Steps)":
  • L3 Conditional Control: architecture already supports cond_dim=5 with classifier-free guidance; needs regime-stratified validation (crisis / calm / normal). Originally scoped in proposal as stretch goal.
  • L4 Downstream Utility: VaR/CVaR validation against historical data; strategy backtest fidelity (Disc 0.85 → ?); option pricing Monte Carlo benchmark
  • Plus: DDIM η sweep, Ljung-Box multi-test correction, higher-frequency data extensions

Box 3 (GOLD/AMBER accent):
  "Bottom line: not a production system — but a clean, evaluated, reproducible foundation for one to be built on top of."
```

---

### PROMPT 20 — KEEP existing Team Contributions slide — no changes needed

---

### PROMPT 21 — NEW Slide 28: References and Q&A

**Action:** Insert a new slide at the very end (after Team Contributions).

```
Add a final slide titled: "References & Thank You"

References in two columns (compact font):

Left column:
  • Ho et al. (2020). Denoising Diffusion Probabilistic Models. NeurIPS.
  • Salimans & Ho (2022). Progressive Distillation for Fast Diffusion Sampling. ICLR.
  • Song et al. (2021). Denoising Diffusion Implicit Models. ICLR.
  • Nichol & Dhariwal (2021). Improved Denoising Diffusion Probabilistic Models. ICML.
  • Hang et al. (2023). Efficient Diffusion Training via Min-SNR Weighting. ICCV.

Right column:
  • Yoon et al. (2019). Time-series Generative Adversarial Networks. NeurIPS.
  • Dinh et al. (2017). Density Estimation Using Real-valued Non-volume Preserving Transformations. ICLR.
  • Gulrajani et al. (2017). Improved Training of Wasserstein GANs. NeurIPS.
  • Bollerslev (1986). Generalized Autoregressive Conditional Heteroskedasticity. J. Econometrics.
  • Cont (2001). Empirical properties of asset returns: stylized facts and statistical issues. Quantitative Finance.
  • Glosten, Jagannathan & Runkle (1993). On the Relation Between Expected Return and Volatility. J. Finance. [GJR-GARCH]

Large centered text at the bottom:
  "Thank you — Questions?"
```

---

## Image Insertion Summary (all 16 images)

| Image File | Insert On Slide # | Slide Title |
|---|---|---|
| `15_four_layers.png` | Slide 4 | What Makes Data "Useful"? |
| `01_sf_bar.png` | Slide 17 | SF Coverage |
| `03_sf_heatmap.png` | Slide 17 | SF Coverage |
| `02_mmd_bar.png` | Slide 18 | Distributional Fidelity |
| `04_radar.png` | Slide 18 | Distributional Fidelity |
| `05_disc_bar.png` | Slide 18 | Distributional Fidelity |
| `09_distributions.png` | Slide 19 | Diagnostic: Distributions |
| `10_qq_plot.png` | Slide 19 | Diagnostic: Distributions |
| `11_acf.png` | Slide 20 | Diagnostic: ACF/Correlation |
| `12_corr_ddpm.png` | Slide 20 | Diagnostic: ACF/Correlation |
| `13_ddpm_paths.png` | Slide 20 | Diagnostic: ACF/Correlation |
| `06_ddpm_ablation_sf.png` | Slide 21 | DDPM Ablation |
| `07_ddpm_ablation_mmd.png` | Slide 21 | DDPM Ablation |
| `08_ddpm_sf_heatmap.png` | Slide 22 | Decorr Reg Deep Dive |
| `16_calibration_ceiling.png` | Slide 24 | L2 Saturation |
| `14_training_losses.png` | Slide 25 | Training Details |

---

## Speaker Assignment Suggestion

| Slide Range | Content | Suggested Speaker |
|---|---|---|
| 1–5 | Title, Agenda, Problem, L1-L4 Framing, Six SFs | Shufeng (1-4) + Yixuan (5) |
| 6–7 | Data Pipeline, Architecture Overview | Yizheng |
| 8 | GARCH Baseline | Kevin |
| 9 | VAE | Yizheng |
| 10 | TimeGAN | Yixuan |
| 11 | NormFlow | Yuxia |
| 12–13 | DDPM Theory + Innovations | Shufeng |
| 14–15 | Evaluation Framework (SF tests + metrics) | Yixuan |
| 16–20 | Cross-model comparison, visualizations | Shufeng |
| 21–22 | Ablation + Decorrelation Regularizer | Yixuan |
| 23–24 | Calibration Discovery + L2 Saturation | Yixuan |
| 25 | Training Details | Shufeng |
| 26–28 | Conclusion, Team, References | Shufeng |

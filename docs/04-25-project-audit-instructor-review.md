# Project Audit: Instructor-Perspective Critical Review

**Date:** 2026-04-25  
**Scope:** Full codebase on `main` branch (commit `890b226`)  
**Reviewer stance:** Grading instructor, first time seeing this project, honest assessment

---

## Executive Summary

**Overall grade: B+ / A- territory.** The project is *substantially* above average for a course project. It has a clear research question, 5 real model implementations, a unified evaluation framework, 3-seed reproducibility, and a compelling DDPM improvement story. However, there are several issues — ranging from cosmetic to methodologically significant — that an attentive instructor would flag.

**Strengths:**
- Clear problem statement and real financial data (16 S&P assets, 20 years)
- 5 distinct generative model implementations, all functional
- Unified 6-stylized-facts evaluation framework with quantitative metrics
- Multi-seed experiments (3 seeds, same hyperparameters) for all models
- Strong ablation story across 7 phases of DDPM improvement
- Impressive headline result: DDPM vpred+Student-t achieves SF=5/6, MMD=0.006
- Calibration discovery (real data only passes 3/6 SFs) shows genuine scientific insight

**Weaknesses (from most to least critical):**
1. README.md cross-model table uses STALE numbers from an old evaluation run
2. Two parallel evaluation codepaths produce different numbers for the same metrics
3. Ljung-Box SF6 test has a multiple-testing issue that makes it impossible to pass
4. No train/test temporal split — all evaluation is in-distribution
5. FRED API key hardcoded in source code
6. Several gitignored directories (`/references/`, `/proposal/`) are not tracked

---

## Section 1: Code Quality and Architecture

### 1.1 Data Pipeline (`src/data/`)

**download.py — Grade: B-**
- Downloads Yahoo Finance (18 tickers) and FRED macro data — correct and functional
- **CRITICAL: FRED API key hardcoded** as `DEFAULT_FRED_KEY` (literal hex string in source). This is a security issue and bad practice. Should use env var with graceful skip if unset
- Download end date is `datetime.today()` — not deterministic across runs. For scientific reproducibility, the date range should be fixed
- Forward-fill + backward-fill on prices can introduce lookahead bias (`bfill` uses future data to fill early NaN)

**preprocess.py — Grade: B+**
- Windowing math is correct and consistent (stride applied uniformly)
- **Global normalization issue**: z-scoring uses full-sample mean/std. For a generative model this is acceptable (not forecasting), but should be explicitly stated
- `prices.ffill().bfill()` — the `bfill` is problematic: early missing values are filled from future prices, which is a form of lookahead in the level series before computing returns

**regime_labels.py — Grade: B**
- VIX-based regime labeling is reasonable
- Full-sample z-scoring of conditioning features (lookahead)
- `mode()` tie-breaking for window regime labels is undefined — could produce different results across pandas versions

**config.py — Grade: B+**
- Clean constants file
- `MODEL_NAMES` list does not include `ddpm_improved` even though it's the primary model — inconsistency with `run_pipeline.py`

**run_pipeline.py — Grade: B+**
- Clear step-based orchestration with logging
- GARCH is trained on `returns_flat` (the full flat return series), same as all other models get windowed data — this is correct, GARCH fits on flat returns and generates windows
- `--quick` mode uses stride=10, reducing dataset size — fine for testing but not documented in output
- Passes `DEFAULT_FRED_KEY` as CLI default (perpetuates the hardcoded key issue)

### 1.2 Model Implementations (`src/models/`)

**ddpm_improved.py — Grade: A-**
- v-prediction math is correct: target = sqrt(α̅)ε − sqrt(1−α̅)x₀
- Student-t noise scaling by sqrt(ν/(ν−2)) correctly normalizes variance to 1
- Min-SNR weighting is correctly implemented
- Decorrelation regularizer is well-designed and properly gated behind flags
- **Minor issue**: Student-t forward process is a heuristic — the reverse process still uses the Gaussian DDPM/DDIM math. This is a methodological shortcut, not a bug, but should be acknowledged
- **Minor issue**: ACF guidance step (`_acf_guidance_step`) does not use `_net_with_self_cond` — would break if `use_self_cond=True` is combined with ACF guidance. **Not a problem for reported results** since self-cond is not used in any reported experiment
- Code is well-documented with clear phase labels

**ddpm.py — Grade: B+**
- Functional baseline DDPM with DDIM sampling and CFG
- Standard implementation, no issues found

**gan.py (TimeGAN) — Grade: B**
- Implements the TimeGAN *architecture* (embedder, recovery, generator, supervisor, discriminator) but uses WGAN-GP losses instead of the original TimeGAN losses
- More accurately described as "TimeGAN-style architecture with WGAN-GP training" than a faithful TimeGAN reimplementation
- CuDNN disabled for gradient penalty through RNN — good practice
- Generation path (supervisor → recovery) is correct

**vae.py — Grade: B+**
- Feature-rich: BiGRU encoder, autoregressive decoder, Student-t likelihood, factor loadings for cross-asset structure
- The "free bits" KL implementation uses `clamp(kl, min=free_bits)` which is a floor (minimum KL), not the standard "don't penalize until you've used c nats" ceiling — a nonstandard variant
- Teacher-forcing exposure bias is the standard issue with autoregressive decoders

**garch.py — Grade: B-**
- Fits AR(1) + GARCH(1,1) per asset via `arch` library — correct
- **Issue**: AR(1) mean dynamics are fitted but **ignored in generation** — `_simulate_garch` only uses constant `mu`, not the AR coefficient. Generated paths don't use the fitted mean model
- **Issue**: Student-t df=5 is hardcoded in generation regardless of fitted df — minor inconsistency
- `self.means` is computed but never used in `generate()`
- Cross-asset correlation via Cholesky decomposition of residual correlations — correct approach
- **For the purpose of being a baseline**: these issues are acceptable. GARCH is not the focus model

**normalizing_flow.py (RealNVP) — Grade: B+**
- Flat-vector RealNVP with ActNorm is correctly implemented and works for reported results
- **Multi-scale path is broken**: dimensions don't line up after splits, `forward()` never implements splitting. But `multi_scale=False` by default, and all reported results use the default — so this doesn't affect any claimed numbers
- ActNorm (default) is the safer choice over batch norm for flow models — good

### 1.3 Evaluation Framework (`src/evaluation/`)

**stylized_facts.py — Grade: B**
- The 6-test framework is the project's defining contribution
- **SF6 (Ljung-Box) has a multiple-testing problem**: requires min(p_value) > 0.05 across all 20 lags. Even for true white noise, the probability of all 20 individual p-values exceeding 0.05 is ≈0.36 (= 0.95^20). This makes SF6 nearly impossible to pass at any reasonable sample size — and explains why no model (including real data) passes it
- **SF5 (Cross-Asset Correlations)**: the "Marchenko-Pastur" label for the λ₁>1.5 threshold is misleading — MP bound depends on T/D ratio, 1.5 is just an ad-hoc lower bound
- **SF4 (Hurst)**: R/S estimator is a "vintage" approach with known finite-sample bias; DFA would be more robust but this is acceptable for a course project
- **SF1-SF3**: implementations are correct and reasonable
- Tests operate on mean-across-assets 1D series (except SF5 which needs 2D). This is a modeling choice that can hide per-asset failures

**metrics.py — Grade: B+**
- MMD with median heuristic bandwidth — standard and correct
- Discriminative score via RandomForest cross-val — a reasonable "can you tell real from fake?" proxy, not identical to TimeGAN's original discriminative score
- MMD subsampling (2000 points) does not seed the RNG locally — minor reproducibility issue
- All metrics flatten multi-asset windows to 1D — destroys temporal and cross-asset structure, but is fair across models if applied uniformly

**cross_model_analysis.py — CRITICAL ISSUE: Grade: C+**
- **Implements its OWN versions of Hill, Hurst, GARCH persistence, and GJR** that are DIFFERENT from stylized_facts.py:
  - Hill: uses top 5% fraction vs √n in stylized_facts
  - GARCH: uses moment heuristic (ACF of r²) vs arch MLE
  - Leverage: uses mean(r²|neg) vs mean(r²|pos) heuristic vs GJR fit
  - Hurst: log-spaced lags vs powers-of-2 blocks
- This means the "Table 1" style analysis in cross_model_analysis produces **different numbers** from what `stylized_facts.run_all_tests()` reports
- The `run_all_tests()` calls in the main loop do NOT pass `real_returns`, so they use absolute thresholds rather than the "compare to real" mode
- This dual-evaluation-system is the biggest methodological concern in the project

### 1.4 Experiment Scripts and Results

**Results integrity: Grade: A-**
- `comparison_table.csv` numbers match the source JSON files for all 5 models
- `ddpm_ablation_table.csv` numbers match source JSONs for all 4 DDPM configs
- All ANALYSIS.md files are consistent with their companion JSON data
- **One error found**: GARCH's ANALYSIS.md references "DDPM Disc=0.54" — this is from the OLD cross_model numbers, not the current rebaseline (should be 0.85)

---

## Section 2: README.md Inconsistencies (CRITICAL)

The README's main "Cross-Model Comparison Summary" table (lines 93-100) uses numbers from an **OLD** evaluation run (`results/cross_model/cross_model_metrics.json`), not the current 3-seed rebaseline results:

| Model | README says | 3-seed rebaseline says | Match? |
|-------|-----------|----------------------|--------|
| GARCH SF | 4/6 | 1.3/6 | **NO** |
| GARCH MMD | 0.281 | 0.042 | **NO** |
| VAE SF | 3/6 | 1.0/6 | **NO** |
| VAE MMD | 0.415 | 0.020 | **NO** |
| NormFlow Disc | 0.54 | 0.73 | **NO** |
| TimeGAN MMD | 0.065 | 0.110 | **NO** |
| DDPM | Correct | Correct | Yes |

The footnote says "Multi-seed re-runs for all models are in progress" — this is **stale**; the re-runs are complete. The README table MUST be updated before presentation.

### Impact
- An instructor reading only the README would see dramatically different numbers than what the actual experiment data shows
- GARCH appears to pass 4/6 SFs in the README but actually only passes 1.3/6 under the unified evaluation
- VAE appears to pass 3/6 but actually only passes 1.0/6

### What happened
The old `results/cross_model/` data was generated by `cross_model_analysis.py` which uses **different** statistical tests (see Section 1.3 above). The new `experiments/results/final_comparison/` data uses `stylized_facts.run_all_tests()` — a stricter and more principled framework. The README was never updated to reflect the new numbers.

---

## Section 3: Gitignore and Tracking Issues

| Item | Tracked? | Should be? | Issue |
|------|:--------:|:----------:|-------|
| `/references/` | **No** (gitignored) | Debatable | Literature reviews and meeting notes are lost for collaborators |
| `/proposal/` | **No** (gitignored) | Debatable | Original proposal not accessible in repo |
| `/data/` | **No** (gitignored) | Correct | Regenerable via pipeline |
| `/checkpoints/` | **No** (gitignored) | Correct | Large binary files |
| `PROJECT_STATUS.md` | **No** (untracked) | Your choice | Currently in root, untracked |
| `docs/` | **Yes** (tracked) | Yes | Good |
| `experiments/results/` | **Yes** (tracked) | Yes | Good |

---

## Section 4: Does the Project "Work" End-to-End?

**Yes.** A fresh user could:
1. `pip install -r requirements.txt`
2. `PYTHONPATH=. python3 src/run_pipeline.py` (full pipeline)
3. Get trained models, evaluation results, and a demo dashboard

**Caveats:**
- FRED download will fail silently if the hardcoded API key is revoked
- No `setup.py` or `pyproject.toml` — not installable as a package
- Demo (`FastAPI + Chart.js`) is a nice touch for presentation

---

## Section 5: Kevin's Concern — Is It Valid?

### What Kevin said (paraphrased from WeChat voice messages):
"I ran the full training for 10+ hours and didn't get results, then restarted from yesterday 9:48 PM and it's still running now. Every time I ask AI for progress, it says logs are buffered."

Kevin seems to be talking about running the full pipeline on his machine (possibly via Claude) and experiencing extremely long training times with no visible output (buffered stdout).

### Is the "code skeleton and data" concern valid?

**Short answer: Not really a problem for the final results, but Kevin's experience reveals a real UX issue.**

1. **The code skeleton is sound.** The data pipeline downloads real Yahoo Finance data, computes log returns, creates overlapping windows with stride=1, and feeds them to all models identically. There is no fundamental error in the data flow.

2. **Kevin's GARCH runs fine.** We already ran his GARCH model ourselves using `run_garch_rebaseline.py` with 3 seeds under unified settings. It completed in seconds (GARCH fitting is fast — it's statistical, not deep learning). The results are on main.

3. **The "10+ hours" issue** is likely because Kevin ran the **full pipeline** (`run_pipeline.py`) which trains ALL models (DDPM 400 epochs, TimeGAN, VAE, NormFlow, GARCH). On CPU, DDPM alone takes hours. If he's running on CPU without GPU, this is expected behavior. The buffered output issue is because Python's stdout is line-buffered by default in non-interactive mode — use `python -u` or `PYTHONUNBUFFERED=1`.

4. **Kevin said Claude's results were "better than ours"** when fed a different project. This is irrelevant — Claude was comparing a different codebase with different data/evaluation. Our pipeline, data, and evaluation framework are self-consistent.

5. **The "data problem" concern**: There is no evidence of corrupted or incorrect data. The data pipeline downloads from Yahoo Finance, which is a standard source. All models consume the same preprocessed windows. The only methodological "issue" is the global normalization (not a bug, a design choice) and the `bfill` on missing prices (very minor effect).

### Do we need to take Kevin's concern seriously?

**No, for the following reasons:**

1. **His GARCH experiment is already complete.** We ran it ourselves under unified settings. Results: SF=1.3/6, MMD=0.042. This is on `origin/main`.

2. **His code is merged.** Kevin's `garch.py` and `visualization.py` are on main. His visualization utilities are genuinely useful.

3. **The "better results from Claude" claim is not actionable** without knowing what Claude ran, on what data, with what evaluation framework. Our evaluation framework is self-consistent across all 5 models.

4. **His runtime issue** is a UX problem (buffered output, full pipeline on CPU), not a data/code correctness problem.

### Suggested reply to Kevin (Chinese):

> Kevin，GARCH 的实验我们已经帮你跑完了，3个seed，统一设置(stride=1, 400 windows, 新的评估框架)，结果在 main 上面了。SF=1.3/6，MMD=0.042，作为 baseline 完全够用。
>
> 关于你说的跑了10多个小时还在跑：你应该是跑了完整 pipeline（包含 DDPM 400 epoch 训练），在 CPU 上确实要很久。GARCH 本身只需要几秒就跑完。如果你只需要跑 GARCH，可以用 `--models garch` 参数。
>
> 关于 Claude 的结果比我们好：那个是不同的代码和数据，不能直接比较。我们现在5个模型用的是同一套数据管道和评估框架，结果是一致可比的。
>
> 你现在不用再跑实验了，可以帮忙准备 PPT 的 Introduction + Data pipeline 部分。

---

## Section 6: Items That MUST Be Fixed Before Presentation

### Priority 1 (Critical — affects credibility)

1. **Update README.md cross-model table** to use the 3-seed rebaseline numbers from `comparison_table.csv`. The current table shows GARCH at 4/6 SF when it's actually 1.3/6 — a >3x discrepancy that any reviewer would catch

2. **Remove or update the footnote** "Multi-seed re-runs for all models are in progress" — they're done

3. **Fix GARCH ANALYSIS.md** reference to "DDPM Disc=0.54" → should be 0.85

### Priority 2 (Should fix)

4. **Update old `results/cross_model/` figures** or clearly separate them from the `experiments/results/final_comparison/` outputs. README images currently point to the OLD cross_model figures which were generated from different evaluation code

5. **Remove hardcoded FRED API key** from `download.py` — change default to `None` with env var fallback

### Priority 3 (Nice to have)

6. Add a note in the README or presentation that global normalization is intentional (not a data leakage issue for generative modeling)

7. Add a note about the SF6 Ljung-Box impossibility (Yixuan's calibration discovery) somewhere prominent

8. Clean up `cross_model_analysis.py`'s duplicate statistical tests or add a clear disclaimer that it uses different estimators

---

## Section 7: Honest Assessment — Is This Project Convincing?

**Yes, with caveats.**

**What's genuinely impressive:**
- The DDPM improvement trajectory (SF 1.7→5/6 over 7 phases) tells a compelling research story
- MMD=0.006 for the best DDPM is an exceptionally good result
- The calibration discovery (real data only passes 3/6 SFs) is a genuine scientific contribution to the evaluation methodology
- 5 real model implementations with reproducible 3-seed experiments
- The interactive demo is a nice presentation touch

**What would concern a careful reviewer:**
- The README table doesn't match the actual experiment data (fixable today)
- Two different evaluation codepaths exist with different statistical tests
- No held-out temporal test set (all evaluation is in-distribution) — this is standard for generative modeling benchmarks but should be acknowledged
- Some of the stylized fact thresholds are arbitrary and not formally justified
- The "NormFlow Disc=0.54" in the old table (which made NormFlow look best on discriminative score) becomes 0.73 in the rebaseline — the old number was from different evaluation code

**Bottom line for the 20-minute presentation:**
If you fix the README numbers and present from the `final_comparison` pipeline outputs, the project tells a strong, data-backed story: GARCH < VAE < TimeGAN < NormFlow ≈ DDPM on SF count, with DDPM winning decisively on MMD (0.006 vs 0.027). The Phase 7 ablation adds depth. The SF calibration finding adds intellectual honesty. This is presentation-ready content.

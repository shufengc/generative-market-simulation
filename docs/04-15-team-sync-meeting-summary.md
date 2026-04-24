# Team Sync Status & Problems
*Branch: shufeng | Date: April 15, 2026*
*For: Shufeng Chen + Claude Code collaboration*

---

## Current Team Status (from WeChat, April 14-15 evening)

### Meeting Outcome
The team (Shufeng, Yixuan, Yizheng, Meng, Kevin) met after class and agreed:
1. **Confirm everyone is using the same dataset** (data + stride)
2. **Re-run baseline experiments** for each model with unified data + eval
3. Starting **Thursday**: Focus on DDPM refinement
4. **Do not merge shufeng branch to main yet** (Lin: "wait until data and eval are confirmed")

---

## Problem 1: Stride Inconsistency (CONFIRMED)

**Status:** Confirmed by branch diff. This is the main issue Ye raised.

| Branch | `run_pipeline.py` default stride | Impact |
|--------|----------------------------------|--------|
| `shufeng` | **stride=5** | ~1,059 training windows |
| `main`, `yixuan`, `yizheng`, `yuxia` | **stride=1** | ~5,200+ training windows |

**What this means for Shufeng:** All Phase 1-5 DDPM experiments were trained on ~5× fewer data windows than other models. The 4.7/6 and 5.0/6 SF scores previously reported need to be re-validated with stride=1.

**Fix:** Update `src/run_pipeline.py` to use `stride=1` (matching main).

---

## Problem 2: Evaluation Framework Changed (CRITICAL)

**Status:** Yixuan completely rewrote `src/evaluation/stylized_facts.py`, now merged into `main`.

Yixuan's message (translated from Chinese):
> "I updated data and evaluate into the yixuan branch. Mainly changed: `src/data/preprocess.py`, `src/data/regime_labels.py`, `src/evaluation/stylized_facts.py`, `src/run_pipeline.py`. You can check them — no issues, same goal as the proposal, and the evaluate metrics are universal for all models."

### Changes to Stylized Facts Tests:

| Test | Old (shufeng branch) | New (main branch) | Reason |
|------|---------------------|-------------------|--------|
| SF1 Fat Tails | Kurtosis + Jarque-Bera | **Hill estimator** (α comparison vs real) | More robust, proposal-aligned |
| SF2 Vol Clustering | ARCH-LM p < 0.05 | **GARCH(1,1)** γ=α+β comparison | Quantitative, like real data |
| SF3 Leverage Effect | Correlation coeff | Same + outputs real comparison value | Unchanged core test |
| SF4 Long Memory | ACF count ≥ 15 | **Hurst R/S** `\|H_syn−H_real\| < 0.05, H∈(0.5,1)` | ACF count was unreliable |
| SF5 Cross-Asset Corr | Rolling corr std | **Max eigenvalue** relative error < 5% | Captures full correlation structure |
| SF6 No Autocorr | Ljung-Box | **MAA** (mean abs autocorr < 0.05) + Ljung-Box | MAA is the primary criterion |

**New API:** `run_all_tests(returns, real_returns=None)` — when `real_returns` is passed, all comparison-based tests use quantitative gaps against real data.

**Impact on our Phase 1-5 results:** ALL previous SF counts (3.7/6, 4.3/6, 4.7/6, 5.0/6) were measured with the OLD tests. Under the new framework, these numbers are meaningless and must be re-run.

---

## Problem 3: FRED API Key Was Broken

**Status:** RESOLVED by Lin (林一正).

Lin's message (translated):
> "I went through the entire data pipeline. The dataset issue was caused by FRED API key problems. Already applied for a new one. Currently tested and working. Shall we still use FRED? Macro data would be more accurate."

**Current decision:** Use Yahoo Finance for stock prices (already working). FRED for macro conditioning is optional but accurate. Lin confirmed the pipeline passes end-to-end.

---

## Problem 4: `demo.ipynb` Version Mismatch

**Status:** Lin noted his local `demo.ipynb` is the old version. Using Shufeng's updated version.

Lin's message:
> "demo.ipynb is different — mine is still the original version. I'll use your updated one."

**Action:** No code change needed. The updated `demo.ipynb` in shufeng branch is the reference.

---

## Problem 5: Accidental Files in Main Root Directory

Ye said:
> "In main I accidentally uploaded those 4 files directly to the root directory. They can be deleted later."

The 4 files referred to are: `preprocess.py`, `regime_labels.py`, `stylized_facts.py`, `run_pipeline.py` (the ones from yixuan branch that were supposed to go into `src/` subdirectories).

**Status:** Lin said he can clean this up. Not Shufeng's responsibility.

---

## Problem 6: run_pipeline.py `fix(pipeline)` commit

In Yixuan's yixuan branch commit history, there's a commit:
> `fix(pipeline): correct stride in main() to match step_preprocess default`

This suggests there was a known stride bug in the pipeline that Yixuan fixed in yixuan branch (set `stride=1` to match `step_preprocess` default). The shufeng branch never received this fix.

---

## Summary of Actions for Shufeng + Claude Code

### Immediate (before Thursday sync):

- [ ] **Update `src/evaluation/stylized_facts.py`** → copy from `origin/main` (Yixuan's version)
- [ ] **Update `src/run_pipeline.py`** → change `stride: int = 5` → `stride: int = 1` in `step_preprocess()` default, and fix `main()` strides accordingly  
- [ ] **Verify data downloads and pipeline runs** end-to-end with new settings
- [ ] **Re-run DDPM vpred baseline** (best Phase 3-4 config: `vpred=True, cosine schedule, 128ch`) using new data (stride=1) and new eval framework

### Do NOT do yet:
- Do not commit or push to any branch
- Do not modify the DDPM model architecture (wait for team sync Thursday)
- Do not run Phase 5 innovations yet (stride baseline first)

---

## TA's Key Guidance

From TA Gaurav's feedback:
- DDPM is the primary model; GARCH is the baseline (not NormFlow)
- **Do not skip the leverage effect test** — it's the hardest one and most convincing
- Add TimeGAN/VAE/Flows as time allows, but DDPM core result is the priority

---

## Notes for Claude Code

1. **Do not assume the experiment results in `experiments/results/` are valid** — they were all run with stride=5 and old eval. They are reference only.
2. When running `run_pipeline.py`, always check that stride=1 is active.
3. The new `stylized_facts.py` requires `real_returns` to be passed for quantitative comparison. The pipeline already handles this in main branch — make sure shufeng branch matches.
4. If you need to check what main branch has: `git show origin/main:src/evaluation/stylized_facts.py`
5. The DDPM model (`src/models/ddpm_improved.py`) does NOT need changes — only the data pipeline and evaluation framework need to be synced.

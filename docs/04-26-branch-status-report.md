# Branch Status Report
**Date:** April 26, 2026 — 1:40 PM ET
**Author:** Shufeng (auto-generated)
**Current branch:** `shufeng`

---

## 1. Branch Map

```
* shufeng   (HEAD, 9 commits ahead of main, 0 behind)
  main      (9366954 — vae 3-seed rebaseline)
  remotes/origin/shufeng
  remotes/origin/main
  remotes/origin/phase7-decorr-reg      ← MERGED into shufeng (ancestor)
  remotes/origin/yizheng-update         ← MERGED into shufeng (ancestor)
  remotes/origin/kevin/garch-visualization  ← MERGED into shufeng (ancestor)
  remotes/origin/yizheng                ← NOT merged (2 extra commits — stride experiment results only)
  remotes/origin/experiment/timegan-3seed-colab  ← old, already in main
  remotes/origin/archive/timegan-experiments     ← archived
```

## 2. Shufeng Branch (Current) — 9 Commits Ahead of Main

```
1ef4c30 fix(docs+security): update README with rebaseline numbers
890b226 feat(comparison): add Phase 7 DDPM ablation section
ba81b4b merge(phase7): Yixuan's decorrelation regularizer + calibration discovery
f4eefd9 Merge pull request #13 from yizheng
52b4e6e feat(phase7): add full patch-based experiment pipeline and results
7ac788f feat(yuxia): DDPM enhanced rebaseline results (Min-SNR, warmup)
f1cf0b0 docs(phase7): add ANALYSIS.md and phase7_results.json (decorr_reg)
25f9381 feat(pipeline): add --seed and --skip-train CLI flags
dddf562 feat(ddpm): add decorrelation regularizer for SF6 (Phase 7)
```

This branch contains ALL team contributions merged: Phase 7 decorr_reg (Yixuan), patch training (Yizheng via yizheng-update), DDPM enhancements (Yuxia), GARCH (Kevin).

## 3. Working Tree Status (Uncommitted)

All new files from the PPT work session — none modify existing tracked code:
```
?? ELEN4904_ProjectGradingRubric.pdf
?? Generative-Market-Simulation.pdf
?? PROJECT_STATUS.md
?? docs/gamma-presentation-content.md
?? docs/gamma-prompts-final.md
?? docs/team-reply-draft.md
?? experiments/evaluate_regimes.py          ← NEW L3 script (has bugs)
?? experiments/run_conditional_ddpm.py      ← NEW L3 script (has bugs)
?? experiments/var_backtest.py              ← NEW L4 script (has bugs)
?? presentation_assets/
?? scripts/
```

No modified tracked files. Clean diff against HEAD for tracked content.

## 4. Unmerged Branch: origin/yizheng

Two commits not in shufeng:
```
625d814 experiments: improved DDPM stride comparison
588f56a add temporary improved ddpm variant with min-snr and cfg updates
```

These add `src/models/ddpm_improved_temp.py` and stride experiment results/charts. **They do NOT modify** any file in `src/models/ddpm_improved.py`, `src/data/`, or `src/evaluation/`. No conflict risk for L3/L4 work.

## 5. Existing Checkpoint Analysis — CRITICAL FINDING

The local `checkpoints/ddpm.pt` has this saved config:
```
n_features: 15     ← MISMATCH: current data has 16 assets
seq_len: 60
T: 1000
padded_len: 64
cond_dim: 5        ← Already trained with conditioning!
cfg_drop_prob: 0.1
```

**Issues:**
- `n_features=15` does not match `windows.npy` which is `(5293, 60, 16)`. This checkpoint was trained on an older 15-asset dataset.
- The config saved in the checkpoint is incomplete — it does NOT include the boolean flags (`use_vpred`, `use_student_t_noise`, etc.). This was saved by an older version of `save()`.
- We CANNOT load this checkpoint into a model constructed for 16 assets — the weight shapes will not match.

**Conclusion:** The existing `ddpm.pt` cannot be used for L3/L4. We MUST retrain from scratch on the current 16-asset data.

However, looking at `run_pipeline.py`, the `ddpm_improved` training path (lines 134-164) automatically:
- Reads `n_features` from `windows.npy` (16)
- Loads `window_cond.npy` → `cond_dim=5`
- Uses `use_vpred=True` and `use_student_t_noise=True`
- This is exactly the Phase 6 config that achieved SF 5/6, MMD 0.006

## 6. Data Status

```
data/windows.npy       — (5293, 60, 16) float32 — current, 16 assets
data/window_cond.npy   — (5293, 5) float32     — ready (macro conditioning)
data/window_regimes.npy — (5293,) int64         — ready
  Regime distribution: normal=2457, crisis=724, calm=2112
data/returns.csv       — daily returns for all assets
data/prices.csv        — raw prices
data/asset_names.json  — 16 asset names
data/scaler_mean.npy   — normalization params
data/scaler_std.npy    — normalization params
```

All data is ready. No re-download needed.

## 7. Interference Risks for L3/L4

| Risk | Status | Notes |
|------|--------|-------|
| Checkpoint mismatch (15 vs 16 features) | HIGH | Must retrain, cannot reuse ddpm.pt |
| `load_processed_data` doesn't exist | BUG | Scripts reference non-existent function |
| `StyleizedFactsEvaluator` doesn't exist | BUG | Class never existed; use `run_all_tests()` |
| Constructor param names wrong | BUG | Scripts use `use_v_prediction` not `use_vpred` |
| `origin/yizheng` not merged | LOW | Only adds stride experiment results, no code conflicts |
| Server `/root/dd_campaign/` | NONE | Completely separate directory, no shared files |
| Missing pip packages on server | MEDIUM | `arch`, `statsmodels`, `scikit-learn`, `tqdm` not installed |
| `base_channels` default is 64, pipeline uses 128 | MEDIUM | Must match pipeline: `base_channels=128, channel_mults=(1,2,4)` |

# Data Pipeline and Training Report

**For external verification (e.g., by classmate/Gemini)**
**Date**: April 16, 2026

---

## 1. Data Source and Authenticity

### Source
All market data is downloaded live from **Yahoo Finance** via the `yfinance` Python library. The download code is in `src/data/download.py`.

### Tickers Downloaded (19 total)
| Ticker | Asset | Category |
|--------|-------|----------|
| SPY | S&P 500 ETF | Broad Market |
| XLK, XLF, XLE, XLV, XLI, XLP, XLY, XLU, XLB, XLRE, XLC | S&P 500 Sector ETFs | Sectors |
| TLT, IEF, SHY | Treasury Bond ETFs (20Y, 7-10Y, 1-3Y) | Fixed Income |
| GLD | Gold ETF | Commodities |
| USO | Oil ETF | Commodities |
| UUP | Dollar Index ETF | Currency |
| ^VIX | CBOE Volatility Index | Volatility (conditioning only) |

### Date Range
- **Raw download**: January 3, 2005 to April 14, 2026
- **Trading days**: 5,353 rows in `data/prices.csv`
- All data is **real, publicly available** from Yahoo Finance. Nothing is synthetic or fabricated.

### Filtering Applied
- VIX is excluded from training data (used only for regime conditioning)
- XLC (launched 2018) and XLRE (launched 2015) have >50% missing data and are dropped by the 80% coverage threshold in `preprocess.py` line 112
- USO and UUP have some early missing data, filled via forward-fill and back-fill
- **Final training assets: 16** (after dropping VIX, XLC, XLRE)

### Conditioning Data (Optional)
- Downloaded from **FRED API** (Federal Reserve Economic Data): yield curve slope (GS10-GS2), credit spreads (BAMLH0A0HYM2, BAMLC0A4CBBB), fed funds rate
- Combined with VIX to form 5-dimensional conditioning vectors
- Used for classifier-free guidance in DDPM, not for NormFlow

---

## 2. Preprocessing Pipeline

Code: `src/data/preprocess.py`

1. **Log returns**: `r_t = log(P_t / P_{t-1})` for each asset (line 42)
2. **Z-score normalization**: Per-asset `(r_t - mean) / std` (lines 119-121)
3. **Windowing**: Slice into overlapping 60-day windows with stride=5 (lines 58-70)

### Resulting Training Data
| Dimension | Value |
|-----------|-------|
| Number of windows | 1,059 |
| Window length | 60 days |
| Assets per window | 16 |
| Values per sample | 60 x 16 = 960 |
| Total data points | 1,059 x 960 = 1,016,640 |
| Data type | float32 |

---

## 3. Why Training Takes Minutes, Not Hours

Gemini's estimate of 20+ hours is likely based on image diffusion models (e.g., 256x256x3 images). Our data is fundamentally smaller.

### Size Comparison
| Quantity | Our Data | Typical Image DDPM |
|----------|:--------:|:-----------------:|
| Values per sample | 960 | 196,608 (256x256x3) |
| Batch size | 64 | 64 |
| Values per batch | 61,440 | 12,582,912 |
| **Ratio** | **1x** | **205x larger** |

Our training data is **205x smaller per batch** than a typical image diffusion model. This directly translates to proportionally faster training.

### Actual Training Times (Apple M-series MPS GPU)

| Configuration | Params | Epochs | Time per epoch | Total |
|--------------|-------:|-------:|:--------------:|:-----:|
| DDPM 64ch | 2.3M | 200 | ~0.5s | ~94s |
| DDPM 128ch | 9.0M | 200 | ~0.7s | ~140s |
| DDPM 128ch | 9.0M | 400 | ~0.7s | ~275s (~4.6 min) |
| DDPM 128ch | 9.0M | 800 | ~0.7s | ~555s (~9.3 min) |
| NormFlow | 6.7M | 400 | ~0.35s | ~140s (~2.3 min) |

The entire Phase 3 experiment batch (27 runs) completed in ~2 hours.

### Why So Fast
1. **Tiny data**: 960 values per sample vs 196,608 for images = 205x smaller
2. **Small dataset**: 1,059 training windows = only 16 batches per epoch
3. **1D convolutions**: Conv1d is much cheaper than Conv2d at same channel count
4. **MPS acceleration**: Apple's Metal Performance Shaders provide GPU acceleration

---

## 4. Model Architectures

### DDPM (Our Primary Model)
- **Architecture**: 1D UNet with ResBlocks, GroupNorm, sinusoidal time embeddings
- **Input**: (batch, 16 assets, 60 timesteps) -- 960 values
- **Diffusion steps (T)**: 1000
- **Sampler**: DDIM with 50 steps
- **Key innovation**: v-prediction (Salimans & Ho, 2022) instead of epsilon-prediction
- **Conditioning**: 5-dimensional macro regime vector via classifier-free guidance

### NormFlow (Comparison Model)
- **Architecture**: RealNVP with 6 affine coupling layers, batch normalization
- **Input**: Flattened 960-dimensional vector
- **Hidden dim**: 256
- **Training**: Exact log-likelihood maximization
- **No conditioning**: operates unconditionally

### Parameter Counts
| Model | Parameters |
|-------|----------:|
| NormFlow (hidden=256, 6 layers) | 6,711,936 |
| DDPM 64ch (UNet, mults 1,2,4) | 2,335,120 |
| DDPM 128ch (UNet, mults 1,2,4) | 8,970,896 |

---

## 5. Evaluation Framework

All models are evaluated on the same generated samples using:

### 6 Stylized Facts (higher pass count = better)
1. Fat tails: Excess kurtosis > 0 and Jarque-Bera test rejects normality
2. Volatility clustering: ARCH-LM test p-value < 0.05
3. Leverage effect: Negative correlation between r_t and |r_{t+1}|
4. Slow ACF decay: At least 15 of first 20 lags of ACF(|returns|) are positive
5. Cross-asset correlations: Rolling correlation std > 0.05
6. No raw autocorrelation: Ljung-Box test p-value > 0.05

### Distributional Metrics (lower = better, except disc score -> 0.5)
- MMD (Maximum Mean Discrepancy): RBF kernel, median heuristic
- Wasserstein-1 distance
- Discriminative score: Random forest accuracy (0.5 = indistinguishable)
- Correlation matrix distance: Frobenius norm between real and synthetic correlation matrices

---

## 6. Reproducibility

To reproduce our results:
```bash
pip install -r requirements.txt
export PYTHONPATH=.

# Download data (requires internet)
python3 src/data/download.py

# Run Phase 3 experiment (requires ~2 hours on Mac with MPS)
python3 experiments/run_ddpm_ablation.py \
  --models p3_vpred p3_normflow \
  --epochs 400 --base-channels 128 --n-gen 1000 --seeds 42 123 456 \
  --out-dir experiments/results/phase3_fair_comparison
```

All random seeds are fixed (42, 123, 456). Results should be reproducible on the same hardware (Apple MPS).

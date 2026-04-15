# Generative Market Simulation

Synthetic financial time series generation via deep generative models, validated against the six stylized facts of financial returns.

> EECS 4904 &mdash; Spring 2026 Final Project

---

## Motivation

Risk management teams need thousands of realistic market scenarios to stress-test portfolios. Simply replaying history produces only a single realized path. Classical models like GARCH generate conditionally Gaussian returns that fail to capture the heavy tails, volatility clustering, and leverage effects universally observed in real markets.

This project trains multiple deep generative models to produce synthetic multi-asset return series that faithfully reproduce the statistical properties of real financial data, and compares them under a rigorous validation framework.

## Six Stylized Facts

Every generated dataset is validated against six well-documented empirical regularities:

| # | Property | Description |
|---|----------|-------------|
| 1 | Fat tails | Return distributions are heavier-tailed than Gaussian |
| 2 | Volatility clustering | Large moves tend to follow large moves |
| 3 | Leverage effect | Negative returns increase subsequent volatility more than positive returns |
| 4 | Slow autocorrelation decay | Absolute returns show long-memory autocorrelation |
| 5 | Time-varying cross-asset correlations | Correlations between assets change over time |
| 6 | No autocorrelation in raw returns | Raw returns are approximately uncorrelated |

## Models

| Model | Type | Key Idea |
|-------|------|----------|
| **DDPM** | Diffusion | 1-D U-Net denoiser with v-prediction, sigmoid schedule, DDIM sampling, EMA, and classifier-free guidance |
| **TimeGAN** | GAN | Embedding + supervisor + adversarial training for temporal latent dynamics |
| **VAE** | Variational | GRU encoder-decoder with KL annealing |
| **GARCH** | Statistical | Per-asset GARCH(1,1) with correlated Student-t innovations |
| **RealNVP** | Flow | Affine coupling layers with batch normalization |

## Architecture

```
Yahoo Finance + FRED API
        |
   download.py ──> preprocess.py ──> regime_labels.py
        |                |                  |
     prices.csv      windows.npy      window_cond.npy
                         |                  |
                   ┌─────┴──────────────────┘
                   v
        ┌──────────────────────┐
        │   5 Generative Models │
        │  DDPM | GAN | VAE    │
        │  GARCH | NormFlow    │
        └──────────┬───────────┘
                   v
        ┌──────────────────────┐
        │     Evaluation        │
        │  6 Stylized Facts     │
        │  MMD / Wasserstein    │
        │  Discriminative Score │
        └──────────┬───────────┘
                   v
        ┌──────────────────────┐
        │    Interactive Demo   │
        │  FastAPI + Chart.js   │
        └──────────────────────┘
```

## Results

Training on 16 assets (S&P 500 sector ETFs, Treasuries, gold, oil, dollar index), 2005-2026 daily returns, 60-day overlapping windows, 400 epochs.

| Model | Stylized Facts | MMD | Wasserstein-1 | Discriminative Score |
|-------|:--------------:|:---:|:-------------:|:--------------------:|
| **DDPM (Improved)** | 3 / 6 | 0.020 | 0.176 | **0.66** |
| **NormFlow** | **5 / 6** | **0.011** | **0.101** | 0.74 |

*Discriminative score: accuracy of a classifier distinguishing real from synthetic (0.5 = indistinguishable). DDPM achieves the best discriminative score.*

The improved DDPM uses v-prediction parameterization and a sigmoid noise schedule, yielding a 93% improvement in MMD (0.020 vs 0.276) and 32% improvement in discriminative score (0.66 vs 0.98) over the baseline DDPM. Full ablation study results are in `experiments/results/`.

### DDPM Ablation Study

Seven DDPM variants were tested across 3 random seeds. See `experiments/results/ANALYSIS.md` for the full report.

<p align="center">
  <img src="experiments/results/fig_radar_chart.png" width="700" alt="Radar chart of normalized metrics across DDPM variants">
</p>

<p align="center">
  <img src="experiments/results/fig_stylized_facts_heatmap.png" width="700" alt="Stylized facts pass/fail across DDPM variants">
</p>

### Return Distribution Comparison

<p align="center">
  <img src="experiments/results/fig_distributions_h2h.png" width="700" alt="Return distributions: real vs DDPM vs NormFlow">
</p>

### QQ-Plot Comparison

<p align="center">
  <img src="experiments/results/fig_qq_h2h.png" width="700" alt="QQ-plots: real vs synthetic">
</p>

### Autocorrelation of |Returns|

<p align="center">
  <img src="experiments/results/fig_acf_h2h.png" width="700" alt="ACF of absolute returns">
</p>

### Synthetic Price Paths

<p align="center">
  <img src="experiments/results/fig_paths_ddpm_final.png" width="700" alt="DDPM synthetic paths">
</p>

### Correlation Matrix: Real vs Synthetic

<p align="center">
  <img src="experiments/results/fig_corr_ddpm.png" width="700" alt="Correlation matrices: real vs DDPM">
</p>

### Training Loss Curves

<p align="center">
  <img src="experiments/results/fig_training_losses.png" width="600" alt="Training loss curves">
</p>

## Data Sources

- **Yahoo Finance** via `yfinance`: daily prices for 18 tickers (sector ETFs, Treasuries, commodities, VIX) spanning 2005--2026
- **FRED API** via `fredapi`: yield curve slope, credit spreads, fed funds rate for macro regime conditioning

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the full pipeline (download, preprocess, train, evaluate, dashboard)
PYTHONPATH=. python3 src/run_pipeline.py

# Quick test mode (20 epochs, ~5 min)
PYTHONPATH=. python3 src/run_pipeline.py --quick

# Launch the interactive demo
PYTHONPATH=. python3 -m src.demo.app
# Open http://localhost:8000
```

### Conditional Generation

The DDPM supports regime-conditioned generation (crisis, calm, normal) via classifier-free guidance:

```python
from src.models.ddpm_improved import ImprovedDDPM
from src.data.regime_labels import get_regime_conditioning_vectors

model = ImprovedDDPM(n_features=16, seq_len=60, cond_dim=5, device="mps",
                     use_vpred=True, use_sigmoid_schedule=True)
model.load("checkpoints/ddpm.pt")

crisis_paths = model.generate(1000, cond=get_regime_conditioning_vectors()["crisis"])
```

## Project Structure

```
├── src/
│   ├── data/
│   │   ├── download.py          # Yahoo Finance + FRED data acquisition
│   │   ├── preprocess.py        # Log returns, normalization, windowing
│   │   └── regime_labels.py     # Crisis/calm/normal regime classification
│   ├── models/
│   │   ├── base_model.py        # Abstract interface for all models
│   │   ├── ddpm.py              # DDPM baseline
│   │   ├── ddpm_improved.py     # DDPM with v-prediction, sigmoid schedule, self-conditioning
│   │   ├── garch.py             # GARCH(1,1) with correlated innovations
│   │   ├── vae.py               # GRU VAE with KL annealing
│   │   ├── gan.py               # TimeGAN with gradient penalty
│   │   └── normalizing_flow.py  # RealNVP with batch normalization
│   ├── evaluation/
│   │   ├── stylized_facts.py    # Six statistical tests
│   │   ├── metrics.py           # MMD, Wasserstein, discriminative score
│   │   └── visualization.py     # Comparison dashboards and plots
│   ├── demo/
│   │   ├── app.py               # FastAPI backend
│   │   └── index.html           # Interactive Chart.js frontend
│   ├── utils/
│   │   └── config.py            # Central configuration
│   └── run_pipeline.py          # End-to-end orchestration
├── experiments/
│   ├── run_ddpm_ablation.py     # Ablation study: 7 variants x 3 seeds
│   ├── report_ddpm.py           # 3-level evaluation report generator
│   └── results/                 # Figures, tables, raw JSON results
├── notebooks/
│   └── demo.ipynb               # Jupyter demo notebook
└── requirements.txt
```

## Team

| Member | Role |
|--------|------|
| Shufeng Chen | Lead, DDPM, Integration, Demo |
| Yixuan Ye | TimeGAN, Evaluation Framework |
| Yizheng Lin | VAE, Data Pipeline |
| Kevin Sun | GARCH Baseline, Visualization |
| Yuxia Meng | Normalizing Flow, Proposal |

## References

- Coletta et al. (2025). *TRADES: Generating Realistic Market Simulations with Diffusion Models.* arXiv:2502.07071
- Li et al. (2024). *Beyond Monte Carlo: Harnessing Diffusion Models to Simulate Financial Market Dynamics.* arXiv:2412.00036
- Zhang et al. (2024). *Generation of Synthetic Financial Time Series by Diffusion Models.* arXiv:2410.18897
- Du et al. (2024). *FTS-Diffusion: Generative Learning for Financial Time Series.* ICLR 2024
- Wiese et al. (2020). *Quant GANs: Deep Generation of Financial Time Series.* Quantitative Finance
- Yoon et al. (2019). *Time-series Generative Adversarial Networks.* NeurIPS 2019
- Cont (2001). *Empirical Properties of Asset Returns: Stylized Facts and Statistical Issues.* Quantitative Finance

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
| **DDPM** | Diffusion | 1-D U-Net denoiser with v-prediction, DDIM sampling, EMA, and classifier-free guidance |
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

Training on 16 assets (S&P 500 sector ETFs, Treasuries, gold, oil, dollar index), 2005-2026 daily returns, 60-day overlapping windows, 400 epochs, 3 seeds (42, 123, 456).

| Model | Params | Stylized Facts | MMD | Wasserstein-1 | Discriminative Score |
|-------|-------:|:--------------:|:---:|:-------------:|:--------------------:|
| **DDPM (v-prediction)** | 9.0M | **4.7 / 6** | 0.019 | 0.161 | 0.82 |
| **DDPM (v-prediction)** | 2.3M | **4.3 / 6** | 0.039 | 0.249 | 0.89 |
| **NormFlow** | 6.7M | **5.0 / 6** | **0.005** | **0.085** | 0.74 |

*Stylized facts: fat tails, volatility clustering, leverage effect, slow ACF decay, cross-asset correlations, no raw autocorrelation.*

The key algorithmic innovation is **v-prediction** (Salimans & Ho, 2022), which replaces the standard noise prediction target with a velocity target. This single change improves stylized facts from 1.7/6 to 4.3/6 and MMD by 9x -- with zero additional parameters or training time. Even at 2.3M params (3x fewer than NormFlow), v-prediction matches NormFlow on 4 of 6 stylized facts.

**Key discovery**: The sigmoid noise schedule, previously believed to be beneficial, was found to *suppress* volatility clustering and fat tails when combined with v-prediction (dropping SF from 4.7 to 2.7). This interaction effect was identified through controlled Phase 3 experiments.

Full experiment results across 5 phases of ablation are in `experiments/results/`.

### DDPM Ablation Study

Multiple DDPM variants were tested across 5 experiment phases. See `experiments/results/phase3_fair_comparison/ANALYSIS.md` for the controlled comparison and `experiments/results/phase4_low_compute/ANALYSIS.md` for the parameter-fair test.

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
                     use_vpred=True)
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
│   │   ├── ddpm_improved.py     # DDPM with v-prediction, ablation-ready improvements
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
│   ├── run_ddpm_ablation.py     # Ablation study: multi-phase, 20+ variants x 3 seeds
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

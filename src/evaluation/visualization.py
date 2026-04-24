"""
Visualization tools for comparing real vs synthetic financial data.

Produces:
  - Distribution comparison plots (histogram + QQ)
  - ACF comparison plots
  - Rolling correlation heatmaps
  - Stylized facts pass/fail heatmap
  - Synthetic path / terminal wealth plots
  - Correlation matrix comparison
  - Training loss curves
  - Model comparison dashboard
"""

from __future__ import annotations

import math
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats

ACCENT_COLOR = "#C87533"
COLORS = ["#2196F3", "#FF9800", "#4CAF50", "#E91E63", "#9C27B0", "#607D8B"]


def plot_return_distributions(real: np.ndarray, synthetic: dict[str, np.ndarray],
                              save_path: str | None = None):
    """Plot return distribution comparison: real vs each model."""
    n_models = len(synthetic)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4), sharey=True)
    if n_models == 1:
        axes = [axes]

    for ax, (name, syn) in zip(axes, synthetic.items()):
        ax.hist(real, bins=100, density=True, alpha=0.5, label="Real", color="#2196F3")
        ax.hist(syn, bins=100, density=True, alpha=0.5, label=name, color="#FF9800")
        ax.set_title(name, fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.set_xlabel("Return")

    axes[0].set_ylabel("Density")
    fig.suptitle("Return Distribution Comparison", fontsize=14, fontweight="bold")
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_qq_comparison(real: np.ndarray, synthetic: dict[str, np.ndarray],
                        save_path: str | None = None):
    """QQ-plots comparing real quantiles vs synthetic quantiles."""
    n_models = len(synthetic)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4))
    if n_models == 1:
        axes = [axes]

    quantiles = np.linspace(0.01, 0.99, 200)
    real_q = np.quantile(real, quantiles)

    for ax, (name, syn) in zip(axes, synthetic.items()):
        syn_q = np.quantile(syn, quantiles)
        ax.scatter(real_q, syn_q, s=8, alpha=0.6, color="#E91E63")
        lims = [min(real_q.min(), syn_q.min()), max(real_q.max(), syn_q.max())]
        ax.plot(lims, lims, "k--", lw=1, alpha=0.5)
        ax.set_xlabel("Real Quantiles")
        ax.set_ylabel("Synthetic Quantiles")
        ax.set_title(f"QQ: {name}", fontsize=12, fontweight="bold")
        ax.set_aspect("equal")

    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_acf_comparison(real: np.ndarray, synthetic: dict[str, np.ndarray],
                         max_lag: int = 50, mode: str = "absolute",
                         save_path: str | None = None):
    """Plot ACF of |returns| or squared returns for real vs synthetic."""
    def compute_acf(x, max_lag):
        x = x - x.mean()
        n = len(x)
        var = x.var()
        acf = [np.mean(x[lag:] * x[:-lag]) / var if var > 0 else 0 for lag in range(1, max_lag + 1)]
        return np.array(acf)

    transform = np.abs if mode == "absolute" else lambda x: x ** 2
    label = "|returns|" if mode == "absolute" else "returns^2"

    fig, ax = plt.subplots(figsize=(10, 5))
    lags = np.arange(1, max_lag + 1)

    real_acf = compute_acf(transform(real), max_lag)
    ax.plot(lags, real_acf, "k-", lw=2, label="Real", alpha=0.8)

    colors = plt.cm.Set2(np.linspace(0, 1, len(synthetic)))
    for (name, syn), color in zip(synthetic.items(), colors):
        syn_acf = compute_acf(transform(syn), max_lag)
        ax.plot(lags, syn_acf, "--", lw=1.5, label=name, color=color, alpha=0.8)

    ax.set_xlabel("Lag", fontsize=11)
    ax.set_ylabel(f"ACF of {label}", fontsize=11)
    ax.set_title(f"Autocorrelation of {label}", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.axhline(0, color="gray", lw=0.5)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_stylized_facts_heatmap(results: dict[str, list[dict]], save_path: str | None = None):
    """Heatmap of pass/fail across models and stylized facts."""
    models = list(results.keys())
    facts = [r["name"] for r in results[models[0]]]
    data = np.array([[1 if r.get("pass") else 0 for r in results[m]] for m in models])

    fig, ax = plt.subplots(figsize=(10, max(3, len(models) * 0.8 + 1)))
    sns.heatmap(
        data, annot=True, fmt="d", cmap=["#FFCDD2", "#C8E6C9"],
        xticklabels=facts, yticklabels=models,
        cbar=False, linewidths=1, linecolor="white", ax=ax,
    )
    ax.set_title("Stylized Facts: Pass (1) / Fail (0)", fontsize=13, fontweight="bold")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_synthetic_paths(synthetic: np.ndarray, n_paths: int = 50,
                          asset_idx: int = 0, title: str = "Synthetic Paths",
                          save_path: str | None = None):
    """Plot synthetic cumulative return paths for one asset."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    paths = synthetic[:n_paths, :, asset_idx] if synthetic.ndim == 3 else synthetic[:n_paths]
    cum = np.exp(np.cumsum(paths, axis=1)) * 100
    for i in range(min(n_paths, cum.shape[0])):
        ax.plot(cum[i], alpha=0.3, lw=0.8, color=ACCENT_COLOR)
    ax.set_xlabel("Trading Day")
    ax.set_ylabel("Price (start=100)")
    ax.set_title(f"{title} -- Cumulative Returns", fontweight="bold")

    ax = axes[1]
    all_paths = synthetic[:, :, asset_idx] if synthetic.ndim == 3 else synthetic
    all_cum = np.exp(np.cumsum(all_paths, axis=1)) * 100
    terminal = all_cum[:, -1]

    ax.hist(terminal, bins=40, color="#FF9800", edgecolor="white", alpha=0.8)
    ax.axvline(np.mean(terminal), color="red", lw=2, label=f"Mean: ${np.mean(terminal):.1f}")
    q5 = np.percentile(terminal, 5)
    ax.axvline(q5, color="darkred", lw=2, ls="--", label=f"5% VaR: ${q5:.1f}")
    ax.set_xlabel("Terminal Price")
    ax.set_ylabel("Count")
    ax.set_title("Terminal Price Distribution", fontweight="bold")
    ax.legend()

    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_correlation_matrices(real: np.ndarray, synthetic: np.ndarray,
                               asset_names: list[str] | None = None,
                               save_path: str | None = None):
    """Side-by-side correlation matrix heatmaps for real vs synthetic."""
    if real.ndim == 3:
        real = real.reshape(-1, real.shape[-1])
    if synthetic.ndim == 3:
        synthetic = synthetic.reshape(-1, synthetic.shape[-1])

    n = min(real.shape[1], synthetic.shape[1])
    corr_real = np.corrcoef(real[:, :n].T)
    corr_syn = np.corrcoef(synthetic[:, :n].T)

    labels = asset_names[:n] if asset_names and len(asset_names) >= n else None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    kwargs = dict(vmin=-1, vmax=1, cmap="RdBu_r", square=True,
                  linewidths=0.5, linecolor="white")

    sns.heatmap(corr_real, ax=ax1, xticklabels=labels, yticklabels=labels,
                annot=n <= 10, fmt=".2f", **kwargs)
    ax1.set_title("Real Correlations", fontweight="bold")

    sns.heatmap(corr_syn, ax=ax2, xticklabels=labels, yticklabels=labels,
                annot=n <= 10, fmt=".2f", **kwargs)
    ax2.set_title("Synthetic Correlations", fontweight="bold")

    fig.suptitle("Cross-Asset Correlation Comparison", fontsize=14, fontweight="bold")
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_training_losses(losses: dict[str, list[float]], save_path: str | None = None):
    """Plot training loss curves for multiple models."""
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (name, loss_values) in enumerate(losses.items()):
        ax.plot(loss_values, label=name, color=COLORS[i % len(COLORS)], lw=1.5)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss Curves", fontweight="bold")
    ax.legend()
    ax.set_yscale("log")
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_model_comparison_table(metrics: dict[str, dict], save_path: str | None = None):
    """Render a model comparison table as a figure."""
    models = list(metrics.keys())
    columns = ["MMD", "Wasserstein", "KS Stat", "Disc. Score", "Corr. Dist.", "SF Passed"]

    cell_data = []
    for m in models:
        info = metrics[m]
        row = [
            f"{info.get('mmd', '-'):.4f}" if isinstance(info.get('mmd'), (int, float)) else "-",
            f"{info.get('wasserstein_1d', '-'):.4f}" if isinstance(info.get('wasserstein_1d'), (int, float)) else "-",
            f"{info.get('ks_stat', '-'):.4f}" if isinstance(info.get('ks_stat'), (int, float)) else "-",
            f"{info.get('discriminative_score', '-'):.2f}" if isinstance(info.get('discriminative_score'), (int, float)) else "-",
            f"{info.get('correlation_matrix_distance', '-'):.2f}" if isinstance(info.get('correlation_matrix_distance'), (int, float)) else "-",
            str(info.get("sf_passed", "-")),
        ]
        cell_data.append(row)

    fig, ax = plt.subplots(figsize=(12, max(2, len(models) * 0.6 + 1.5)))
    ax.axis("off")
    table = ax.table(
        cellText=cell_data,
        rowLabels=models,
        colLabels=columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#E0E0E0")
            cell.set_text_props(fontweight="bold")

    ax.set_title("Model Comparison Summary", fontsize=14, fontweight="bold", pad=20)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def create_comparison_dashboard(
    real_returns: np.ndarray,
    model_results: dict[str, dict],
    asset_names: list[str] | None = None,
    save_dir: str = "results",
):
    """
    Generate a full set of comparison plots and save to save_dir.

    Args:
        real_returns: (T, D) real return matrix or 1D.
        model_results: dict mapping model_name -> {
            "synthetic": np.ndarray,
            "stylized_facts": list[dict],
            "metrics": dict (optional),
            "losses": list[float] (optional),
        }
    """
    os.makedirs(save_dir, exist_ok=True)

    real_flat = real_returns.flatten() if real_returns.ndim > 1 else real_returns
    syn_dict = {}
    for name, info in model_results.items():
        s = info["synthetic"]
        syn_dict[name] = s.flatten()

    plot_return_distributions(real_flat, syn_dict,
                               save_path=os.path.join(save_dir, "distributions.png"))
    plot_qq_comparison(real_flat, syn_dict,
                        save_path=os.path.join(save_dir, "qq_plots.png"))
    plot_acf_comparison(real_flat, syn_dict, mode="absolute",
                         save_path=os.path.join(save_dir, "acf_absolute.png"))
    plot_acf_comparison(real_flat, syn_dict, mode="squared",
                         save_path=os.path.join(save_dir, "acf_squared.png"))

    sf_results = {name: info["stylized_facts"] for name, info in model_results.items()
                  if "stylized_facts" in info}
    if sf_results:
        plot_stylized_facts_heatmap(sf_results,
                                     save_path=os.path.join(save_dir, "stylized_facts_heatmap.png"))

    # Correlation matrices for the primary model
    first_model = list(model_results.keys())[0]
    first_syn = model_results[first_model]["synthetic"]
    if real_returns.ndim >= 2 and first_syn.ndim >= 2:
        plot_correlation_matrices(real_returns, first_syn, asset_names,
                                   save_path=os.path.join(save_dir, "correlation_matrices.png"))

    # Training losses
    all_losses = {}
    for name, info in model_results.items():
        if "losses" in info and info["losses"]:
            all_losses[name] = info["losses"]
    if all_losses:
        plot_training_losses(all_losses, save_path=os.path.join(save_dir, "training_losses.png"))

    # Metrics comparison table
    metrics_dict = {}
    for name, info in model_results.items():
        if "metrics" in info:
            m = info["metrics"].copy()
            m["sf_passed"] = sum(1 for r in info.get("stylized_facts", []) if r.get("pass"))
            metrics_dict[name] = m
    if metrics_dict:
        plot_model_comparison_table(metrics_dict,
                                     save_path=os.path.join(save_dir, "comparison_table.png"))

    print(f"Dashboard saved to {save_dir}/")


def plot_garch_diagnostics(real: np.ndarray, synthetic: np.ndarray,
                            max_lag: int = 30, save_path: str | None = None):
    """Four-panel GARCH diagnostics: return distribution, ACF(r²), rolling vol, leverage profile.

    Args:
        real:      1-D array of real returns (flat).
        synthetic: 1-D array of synthetic returns (flat).
        max_lag:   Maximum ACF lag for the squared-return panel.
    """
    def _acf_sq(x, max_lag):
        x2 = x ** 2 - (x ** 2).mean()
        var = x2.var()
        return np.array([
            float(np.mean(x2[lag:] * x2[:-lag]) / var) if var > 0 else 0.0
            for lag in range(1, max_lag + 1)
        ])

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # --- Panel 1: Return distribution ---
    ax = axes[0, 0]
    ax.hist(real, bins=120, density=True, alpha=0.5, label="Real", color="#2196F3")
    ax.hist(synthetic, bins=120, density=True, alpha=0.5, label="GARCH", color="#FF9800")
    x_grid = np.linspace(min(real.min(), synthetic.min()),
                          max(real.max(), synthetic.max()), 300)
    from scipy.stats import norm
    ax.plot(x_grid, norm.pdf(x_grid, real.mean(), real.std()), "k--", lw=1, label="Normal fit")
    ax.set_title("Return Distribution", fontweight="bold")
    ax.set_xlabel("Return")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8)

    # --- Panel 2: ACF of squared returns ---
    ax = axes[0, 1]
    lags = np.arange(1, max_lag + 1)
    acf_real = _acf_sq(real, max_lag)
    acf_syn  = _acf_sq(synthetic, max_lag)
    ax.bar(lags - 0.2, acf_real, width=0.4, label="Real",  color="#2196F3", alpha=0.7)
    ax.bar(lags + 0.2, acf_syn,  width=0.4, label="GARCH", color="#FF9800", alpha=0.7)
    conf = 1.96 / math.sqrt(len(real))
    ax.axhline(conf,  color="gray", ls="--", lw=0.8)
    ax.axhline(-conf, color="gray", ls="--", lw=0.8)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_title("ACF of Squared Returns (Volatility Clustering)", fontweight="bold")
    ax.set_xlabel("Lag")
    ax.set_ylabel("ACF(r²)")
    ax.legend(fontsize=8)

    # --- Panel 3: Rolling 20-day volatility ---
    ax = axes[1, 0]
    window = 20
    real_ser = pd.Series(real)
    syn_ser  = pd.Series(synthetic)
    roll_real = real_ser.rolling(window).std() * math.sqrt(252)
    roll_syn  = syn_ser.rolling(window).std()  * math.sqrt(252)
    ax.plot(roll_real.values, color="#2196F3", lw=0.8, alpha=0.8, label="Real")
    ax.plot(roll_syn.values,  color="#FF9800", lw=0.8, alpha=0.8, label="GARCH")
    ax.set_title(f"Rolling {window}-Day Annualised Volatility", fontweight="bold")
    ax.set_xlabel("Observation")
    ax.set_ylabel("Ann. Vol.")
    ax.legend(fontsize=8)

    # --- Panel 4: Leverage-effect profile (signed-shock vs next-period vol) ---
    ax = axes[1, 1]
    for arr, label, color in [(real, "Real", "#2196F3"), (synthetic, "GARCH", "#FF9800")]:
        if len(arr) < 3:
            continue
        shocks   = arr[:-1]
        next_vol = np.abs(arr[1:])
        bins     = np.linspace(np.percentile(shocks, 2), np.percentile(shocks, 98), 25)
        idx      = np.digitize(shocks, bins)
        bin_centers = [(bins[b - 1] + bins[b]) / 2 if 1 <= b < len(bins) else np.nan
                       for b in idx]
        df_tmp = pd.DataFrame({"center": bin_centers, "nv": next_vol}).dropna()
        grp = df_tmp.groupby("center")["nv"].mean()
        ax.plot(grp.index, grp.values, "o-", ms=4, lw=1.2, label=label, color=color, alpha=0.85)
    ax.axvline(0, color="gray", lw=0.8, ls="--")
    ax.set_title("Leverage-Effect Profile", fontweight="bold")
    ax.set_xlabel("Return shock (t)")
    ax.set_ylabel("Mean |return| (t+1)")
    ax.legend(fontsize=8)

    fig.suptitle("GARCH Diagnostics", fontsize=14, fontweight="bold")
    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_conditional_volatility(returns: np.ndarray, cond_vol: np.ndarray,
                                 title: str = "Conditional Volatility",
                                 save_path: str | None = None):
    """Two-panel plot: return time-series (with ±2σ band) and conditional volatility.

    Args:
        returns:  1-D array of returns.
        cond_vol: 1-D array of conditional volatility estimates (same length).
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    t = np.arange(len(returns))

    ax1.plot(t, returns, lw=0.6, color="#2196F3", alpha=0.8, label="Returns")
    ax1.fill_between(t, -2 * cond_vol, 2 * cond_vol,
                     color="#FF9800", alpha=0.25, label="±2σ band")
    ax1.axhline(0, color="black", lw=0.4)
    ax1.set_ylabel("Return")
    ax1.set_title(title, fontweight="bold")
    ax1.legend(fontsize=9)

    ax2.plot(t, cond_vol, lw=0.8, color="#FF9800", label="Cond. Vol (σ_t)")
    ax2.set_ylabel("Conditional Volatility")
    ax2.set_xlabel("Observation")
    ax2.legend(fontsize=9)

    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_garch_param_summary(models_fitted: list[dict], save_path: str | None = None):
    """Bar-chart summary of fitted GARCH/EGARCH parameters (ω, α, β, γ) per asset.

    Args:
        models_fitted: List of dicts as stored in GARCHModel.models_fitted
                       [{"params": {...}, "success": bool, "vol_type": str}, ...].
    """
    n = len(models_fitted)
    asset_ids = list(range(n))

    omega  = [m["params"].get("omega",    np.nan) if m["success"] else np.nan for m in models_fitted]
    alpha  = [m["params"].get("alpha[1]", np.nan) if m["success"] else np.nan for m in models_fitted]
    beta   = [m["params"].get("beta[1]",  np.nan) if m["success"] else np.nan for m in models_fitted]
    gamma  = [m["params"].get("gamma[1]", np.nan) if m["success"] else np.nan for m in models_fitted]

    fig, axes = plt.subplots(4, 1, figsize=(max(8, n * 0.5 + 2), 10), sharex=True)
    bar_kw = dict(edgecolor="white", linewidth=0.5)

    axes[0].bar(asset_ids, omega, color="#2196F3", **bar_kw)
    axes[0].set_ylabel("ω (omega)")
    axes[0].set_title("GARCH Parameter Summary per Asset", fontweight="bold")

    axes[1].bar(asset_ids, alpha, color="#4CAF50", **bar_kw)
    axes[1].set_ylabel("α (alpha)")
    axes[1].axhline(0, color="black", lw=0.4)

    axes[2].bar(asset_ids, beta, color="#FF9800", **bar_kw)
    axes[2].set_ylabel("β (beta)")
    axes[2].axhline(0.9, color="gray", ls="--", lw=0.7, label="β=0.9")
    axes[2].legend(fontsize=7)

    axes[3].bar(asset_ids, gamma, color="#E91E63", **bar_kw)
    axes[3].set_ylabel("γ (gamma / leverage)")
    axes[3].axhline(0, color="black", lw=0.4)
    axes[3].set_xlabel("Asset index")

    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig

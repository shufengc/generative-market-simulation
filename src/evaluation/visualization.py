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

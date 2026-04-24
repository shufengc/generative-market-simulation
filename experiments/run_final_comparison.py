"""
Final Cross-Model Comparison Pipeline.

Loads results from all model rebaseline experiments and generates:
  1. Unified comparison table (CSV)
  2. Bar chart: SF passed per model
  3. Bar chart: MMD per model
  4. Radar chart: normalized metrics per model
  5. SF pass/fail heatmap per model

Usage:
    PYTHONPATH=. python experiments/run_final_comparison.py
    PYTHONPATH=. python experiments/run_final_comparison.py --output-dir results/final_comparison
"""

from __future__ import annotations

import os
import sys
import json
import csv
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

EXPERIMENTS_DIR = os.path.dirname(__file__)
DEFAULT_OUTPUT = os.path.join(EXPERIMENTS_DIR, "results", "final_comparison")

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_ddpm_phase6() -> dict:
    """Load DDPM Phase 6 vpred+Student-t results (best config: p5_vpred_studentt)."""
    path = os.path.join(EXPERIMENTS_DIR, "results", "phase6_rebaseline", "ablation_results.json")
    with open(path) as f:
        data = json.load(f)

    # Filter to the best config: p5_vpred_studentt
    entries = [e for e in data if e.get("model") == "p5_vpred_studentt"]
    if not entries:
        # Fallback: use vpred
        entries = [e for e in data if e.get("model") == "vpred"]

    sf_vals = [e.get("n_pass", 0) for e in entries]
    mmd_vals = [e.get("mmd", float("nan")) for e in entries]
    disc_vals = [e.get("discriminative_score", float("nan")) for e in entries]
    w1_vals = [e.get("wasserstein_1d", float("nan")) for e in entries]
    corr_vals = [e.get("correlation_matrix_distance", float("nan")) for e in entries]

    # SF breakdown from first entry's stylized_facts
    sf_breakdown = _extract_sf_breakdown(entries[0].get("stylized_facts", []))

    return {
        "name": "DDPM\nvpred+Student-t",
        "short_name": "DDPM",
        "sf_mean": float(np.mean(sf_vals)),
        "sf_std": float(np.std(sf_vals)),
        "mmd_mean": float(np.nanmean(mmd_vals)),
        "mmd_std": float(np.nanstd(mmd_vals)),
        "disc_mean": float(np.nanmean(disc_vals)),
        "disc_std": float(np.nanstd(disc_vals)),
        "w1_mean": float(np.nanmean(w1_vals)),
        "w1_std": float(np.nanstd(w1_vals)),
        "corr_mean": float(np.nanmean(corr_vals)),
        "corr_std": float(np.nanstd(corr_vals)),
        "n_seeds": len(entries),
        "sf_breakdown": sf_breakdown,
        "color": "#2196F3",  # blue
    }


def load_normflow() -> dict:
    """Load NormFlow rebaseline results (default config)."""
    path = os.path.join(EXPERIMENTS_DIR, "results", "normflow_rebaseline", "normflow_results.json")
    with open(path) as f:
        data = json.load(f)

    entries = [e for e in data if e.get("model") == "default"]
    if not entries:
        entries = data[:3]

    sf_vals = [e.get("n_pass", e.get("sf_passed", 0)) for e in entries]
    mmd_vals = [e.get("mmd", float("nan")) for e in entries]
    disc_vals = [e.get("discriminative_score", float("nan")) for e in entries]
    w1_vals = [e.get("wasserstein_1d", float("nan")) for e in entries]
    corr_vals = [e.get("correlation_matrix_distance", float("nan")) for e in entries]

    sf_breakdown = _extract_sf_breakdown(entries[0].get("stylized_facts", []))

    return {
        "name": "NormFlow\n(RealNVP)",
        "short_name": "NormFlow",
        "sf_mean": float(np.mean(sf_vals)),
        "sf_std": float(np.std(sf_vals)),
        "mmd_mean": float(np.nanmean(mmd_vals)),
        "mmd_std": float(np.nanstd(mmd_vals)),
        "disc_mean": float(np.nanmean(disc_vals)),
        "disc_std": float(np.nanstd(disc_vals)),
        "w1_mean": float(np.nanmean(w1_vals)),
        "w1_std": float(np.nanstd(w1_vals)),
        "corr_mean": float(np.nanmean(corr_vals)),
        "corr_std": float(np.nanstd(corr_vals)),
        "n_seeds": len(entries),
        "sf_breakdown": sf_breakdown,
        "color": "#4CAF50",  # green
    }


def load_timegan() -> dict:
    """Load TimeGAN 3-seed results from ANALYSIS.md numbers."""
    # TimeGAN results are in ANALYSIS.md (no JSON), hardcode from the file
    return {
        "name": "TimeGAN\n(WGAN-GP)",
        "short_name": "TimeGAN",
        "sf_mean": 4.0,
        "sf_std": 0.0,
        "mmd_mean": 0.1103,
        "mmd_std": 0.025,
        "disc_mean": 1.0,
        "disc_std": 0.0,
        "w1_mean": float("nan"),
        "w1_std": float("nan"),
        "corr_mean": float("nan"),
        "corr_std": float("nan"),
        "n_seeds": 3,
        "sf_breakdown": {
            "Fat Tails": True,
            "Vol Clustering": True,
            "Leverage Effect": True,
            "Long Memory": True,
            "Cross-Asset Corr": False,
            "No Raw Autocorr": False,
        },
        "color": "#FF9800",  # orange
    }


def load_garch() -> dict:
    """Load GARCH rebaseline results."""
    path = os.path.join(EXPERIMENTS_DIR, "results", "garch_rebaseline", "garch_results.json")
    with open(path) as f:
        data = json.load(f)

    sf_vals = [e.get("sf_passed", 0) for e in data]
    mmd_vals = [e.get("mmd", float("nan")) for e in data]
    disc_vals = [e.get("discriminative_score", float("nan")) for e in data]
    w1_vals = [e.get("wasserstein_1d", float("nan")) for e in data]
    corr_vals = [e.get("correlation_matrix_distance", float("nan")) for e in data]

    sf_breakdown = _extract_sf_breakdown(data[0].get("sf_results", []))

    return {
        "name": "GARCH\n(Baseline)",
        "short_name": "GARCH",
        "sf_mean": float(np.mean(sf_vals)),
        "sf_std": float(np.std(sf_vals)),
        "mmd_mean": float(np.nanmean(mmd_vals)),
        "mmd_std": float(np.nanstd(mmd_vals)),
        "disc_mean": float(np.nanmean(disc_vals)),
        "disc_std": float(np.nanstd(disc_vals)),
        "w1_mean": float(np.nanmean(w1_vals)),
        "w1_std": float(np.nanstd(w1_vals)),
        "corr_mean": float(np.nanmean(corr_vals)),
        "corr_std": float(np.nanstd(corr_vals)),
        "n_seeds": len(data),
        "sf_breakdown": sf_breakdown,
        "color": "#9E9E9E",  # grey
    }


def load_vae() -> dict | None:
    """Load VAE rebaseline results if available."""
    path = os.path.join(EXPERIMENTS_DIR, "results", "vae_rebaseline", "vae_results.json")
    if not os.path.exists(path):
        print("VAE results not yet available -- skipping from comparison.")
        return None

    with open(path) as f:
        data = json.load(f)

    sf_vals = [e.get("sf_passed", 0) for e in data]
    mmd_vals = [e.get("mmd", float("nan")) for e in data]
    disc_vals = [e.get("discriminative_score", float("nan")) for e in data]
    w1_vals = [e.get("wasserstein_1d", float("nan")) for e in data]
    corr_vals = [e.get("correlation_matrix_distance", float("nan")) for e in data]

    sf_breakdown = _extract_sf_breakdown(data[0].get("sf_results", []) if data else [])

    return {
        "name": "VAE\n(Improved)",
        "short_name": "VAE",
        "sf_mean": float(np.mean(sf_vals)),
        "sf_std": float(np.std(sf_vals)),
        "mmd_mean": float(np.nanmean(mmd_vals)),
        "mmd_std": float(np.nanstd(mmd_vals)),
        "disc_mean": float(np.nanmean(disc_vals)),
        "disc_std": float(np.nanstd(disc_vals)),
        "w1_mean": float(np.nanmean(w1_vals)),
        "w1_std": float(np.nanstd(w1_vals)),
        "corr_mean": float(np.nanmean(corr_vals)),
        "corr_std": float(np.nanstd(corr_vals)),
        "n_seeds": len(data),
        "sf_breakdown": sf_breakdown,
        "color": "#9C27B0",  # purple
    }


def _extract_sf_breakdown(sf_list: list) -> dict:
    """Extract per-SF pass/fail from stylized_facts list."""
    name_map = {
        "Fat Tails": "Fat Tails",
        "Volatility Clustering": "Vol Clustering",
        "Leverage Effect": "Leverage Effect",
        "Long Memory": "Long Memory",
        "Long Memory (Hurst)": "Long Memory",
        "Cross-Asset Correlations": "Cross-Asset Corr",
        "No Raw Autocorrelation": "No Raw Autocorr",
    }
    result = {}
    for sf in sf_list:
        if isinstance(sf, dict):
            name = sf.get("name", "")
            mapped = name_map.get(name, name)
            result[mapped] = bool(sf.get("pass", False))
    return result


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

SF_NAMES = [
    "Fat Tails",
    "Vol Clustering",
    "Leverage Effect",
    "Long Memory",
    "Cross-Asset Corr",
    "No Raw Autocorr",
]


def plot_sf_bar(models: list[dict], output_dir: str):
    """Bar chart: SF passed (mean ± std) per model."""
    fig, ax = plt.subplots(figsize=(9, 5))

    x = np.arange(len(models))
    width = 0.55
    bars = ax.bar(
        x,
        [m["sf_mean"] for m in models],
        width,
        yerr=[m["sf_std"] for m in models],
        color=[m["color"] for m in models],
        capsize=5, edgecolor="white", linewidth=0.8,
        error_kw={"elinewidth": 1.5, "ecolor": "#444"},
    )

    ax.set_xticks(x)
    ax.set_xticklabels([m["short_name"] for m in models], fontsize=11)
    ax.set_ylabel("Stylized Facts Passed (out of 6)", fontsize=11)
    ax.set_title("Stylized Facts Coverage — Cross-Model Comparison", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 7)
    ax.axhline(6, color="green", linestyle="--", linewidth=1.2, alpha=0.6, label="Perfect (6/6)")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    for bar, m in zip(bars, models):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f"{m['sf_mean']:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    fig.tight_layout()
    path = os.path.join(output_dir, "fig_sf_bar.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def plot_mmd_bar(models: list[dict], output_dir: str):
    """Bar chart: MMD per model (log scale)."""
    valid = [m for m in models if not np.isnan(m["mmd_mean"])]
    fig, ax = plt.subplots(figsize=(9, 5))

    x = np.arange(len(valid))
    width = 0.55
    bars = ax.bar(
        x,
        [m["mmd_mean"] for m in valid],
        width,
        yerr=[m["mmd_std"] for m in valid],
        color=[m["color"] for m in valid],
        capsize=5, edgecolor="white", linewidth=0.8,
        error_kw={"elinewidth": 1.5, "ecolor": "#444"},
    )

    ax.set_xticks(x)
    ax.set_xticklabels([m["short_name"] for m in valid], fontsize=11)
    ax.set_ylabel("Maximum Mean Discrepancy (MMD)", fontsize=11)
    ax.set_title("MMD — Lower is Better", fontsize=13, fontweight="bold")
    ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.3)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.3f"))

    for bar, m in zip(bars, valid):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.2,
                f"{m['mmd_mean']:.4f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    path = os.path.join(output_dir, "fig_mmd_bar.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def plot_disc_bar(models: list[dict], output_dir: str):
    """Bar chart: Discriminative score per model (closer to 0.5 = better)."""
    valid = [m for m in models if not np.isnan(m["disc_mean"])]
    fig, ax = plt.subplots(figsize=(9, 5))

    x = np.arange(len(valid))
    width = 0.55
    bars = ax.bar(
        x,
        [m["disc_mean"] for m in valid],
        width,
        yerr=[m["disc_std"] for m in valid],
        color=[m["color"] for m in valid],
        capsize=5, edgecolor="white", linewidth=0.8,
        error_kw={"elinewidth": 1.5, "ecolor": "#444"},
    )

    ax.set_xticks(x)
    ax.set_xticklabels([m["short_name"] for m in valid], fontsize=11)
    ax.set_ylabel("Discriminative Score (closer to 0.5 = better)", fontsize=11)
    ax.set_title("Discriminative Score — Cross-Model Comparison", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1.15)
    ax.axhline(0.5, color="green", linestyle="--", linewidth=1.2, alpha=0.6, label="Perfect (0.5)")
    ax.axhline(1.0, color="red", linestyle=":", linewidth=1.0, alpha=0.4, label="Trivial (1.0)")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    for bar, m in zip(bars, valid):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{m['disc_mean']:.3f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    path = os.path.join(output_dir, "fig_disc_bar.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def plot_sf_heatmap(models: list[dict], output_dir: str):
    """Heatmap: per-SF pass/fail for each model."""
    models_with_breakdown = [m for m in models if m.get("sf_breakdown")]
    if not models_with_breakdown:
        return

    n_models = len(models_with_breakdown)
    n_sf = len(SF_NAMES)

    matrix = np.zeros((n_models, n_sf))
    for i, m in enumerate(models_with_breakdown):
        bd = m.get("sf_breakdown", {})
        for j, sf_name in enumerate(SF_NAMES):
            matrix[i, j] = 1.0 if bd.get(sf_name, False) else 0.0

    fig, ax = plt.subplots(figsize=(10, max(3.5, n_models * 1.0 + 1.5)))
    cmap = matplotlib.colors.ListedColormap(["#FFCDD2", "#C8E6C9"])

    ax.imshow(matrix, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(n_sf))
    ax.set_xticklabels(SF_NAMES, rotation=35, ha="right", fontsize=10)
    ax.set_yticks(range(n_models))
    ax.set_yticklabels([m["short_name"] for m in models_with_breakdown], fontsize=11)
    ax.set_title("Stylized Facts Pass/Fail Heatmap\n(green=PASS, red=FAIL)", fontsize=12, fontweight="bold")

    for i in range(n_models):
        for j in range(n_sf):
            label = "✓" if matrix[i, j] == 1 else "✗"
            color = "#1B5E20" if matrix[i, j] == 1 else "#B71C1C"
            ax.text(j, i, label, ha="center", va="center", fontsize=14, color=color)

    fig.tight_layout()
    path = os.path.join(output_dir, "fig_sf_heatmap.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def plot_radar(models: list[dict], output_dir: str):
    """Radar chart: normalized multi-metric view."""
    metrics = ["SF / 6", "1 - MMD×10", "1 - |Disc-0.5|×2"]
    n_metrics = len(metrics)

    # Compute normalized scores (higher = better for all)
    def normalize(m: dict) -> list[float]:
        sf_score = m["sf_mean"] / 6.0
        mmd_score = max(0, 1.0 - m["mmd_mean"] * 10)
        disc_score = max(0, 1.0 - abs(m["disc_mean"] - 0.5) * 2)
        return [sf_score, mmd_score, disc_score]

    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

    for m in models:
        values = normalize(m)
        if any(np.isnan(values)):
            continue
        values += values[:1]
        ax.plot(angles, values, "o-", linewidth=2, label=m["short_name"], color=m["color"])
        ax.fill(angles, values, alpha=0.1, color=m["color"])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.5", "0.75", "1.0"], fontsize=8)
    ax.set_title("Normalized Multi-Metric Radar\n(larger = better)", fontsize=12,
                 fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=10)

    fig.tight_layout()
    path = os.path.join(output_dir, "fig_radar.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
# Table output
# ---------------------------------------------------------------------------

def save_comparison_table(models: list[dict], output_dir: str):
    """Save CSV and print markdown table."""
    fieldnames = [
        "Model", "N_Seeds", "SF_Mean", "SF_Std",
        "MMD_Mean", "MMD_Std", "Disc_Mean", "Disc_Std",
        "W1_Mean", "W1_Std", "CorrDist_Mean", "CorrDist_Std",
    ]
    rows = []
    for m in models:
        rows.append({
            "Model": m["short_name"],
            "N_Seeds": m["n_seeds"],
            "SF_Mean": round(m["sf_mean"], 2),
            "SF_Std": round(m["sf_std"], 2),
            "MMD_Mean": round(m["mmd_mean"], 4),
            "MMD_Std": round(m["mmd_std"], 4),
            "Disc_Mean": round(m["disc_mean"], 4),
            "Disc_Std": round(m["disc_std"], 4),
            "W1_Mean": round(m["w1_mean"], 4) if not np.isnan(m["w1_mean"]) else "N/A",
            "W1_Std": round(m["w1_std"], 4) if not np.isnan(m["w1_std"]) else "N/A",
            "CorrDist_Mean": round(m["corr_mean"], 4) if not np.isnan(m["corr_mean"]) else "N/A",
            "CorrDist_Std": round(m["corr_std"], 4) if not np.isnan(m["corr_std"]) else "N/A",
        })

    path_csv = os.path.join(output_dir, "comparison_table.csv")
    with open(path_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {path_csv}")

    # Print markdown table
    print("\n## Cross-Model Comparison Table\n")
    header = "| Model | SF | MMD | Disc | W1 | CorrDist |"
    sep    = "|-------|:--:|:---:|:----:|:--:|:--------:|"
    print(header)
    print(sep)
    for r in rows:
        sf = f"{r['SF_Mean']} ± {r['SF_Std']}"
        mmd = f"{r['MMD_Mean']} ± {r['MMD_Std']}"
        disc = f"{r['Disc_Mean']} ± {r['Disc_Std']}"
        w1 = f"{r['W1_Mean']} ± {r['W1_Std']}"
        corr = f"{r['CorrDist_Mean']} ± {r['CorrDist_Std']}"
        print(f"| {r['Model']} | {sf} | {mmd} | {disc} | {w1} | {corr} |")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading model results...")
    models_raw = [
        load_garch(),
        load_timegan(),
        load_normflow(),
        load_vae(),
        load_ddpm_phase6(),
    ]
    models = [m for m in models_raw if m is not None]
    print(f"Loaded {len(models)} models: {[m['short_name'] for m in models]}")

    print("\nGenerating comparison table...")
    save_comparison_table(models, args.output_dir)

    print("\nGenerating plots...")
    plot_sf_bar(models, args.output_dir)
    plot_mmd_bar(models, args.output_dir)
    plot_disc_bar(models, args.output_dir)
    plot_sf_heatmap(models, args.output_dir)
    plot_radar(models, args.output_dir)

    print(f"\nAll outputs saved to: {args.output_dir}/")
    print("Files:")
    for f in sorted(os.listdir(args.output_dir)):
        print(f"  {f}")


if __name__ == "__main__":
    main()

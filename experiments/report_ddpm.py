"""
Generate the full DDPM ablation report: tables, charts, and composite analysis.

Three levels of evaluation:
  Level 1: Individual metric rankings (per-dimension)
  Level 2: Composite scoring (aggregate)
  Level 3: Trade-off and Pareto analysis (multi-objective)

Usage:
    PYTHONPATH=. python experiments/report_ddpm.py
    PYTHONPATH=. python experiments/report_ddpm.py --results experiments/results/ablation_results.json
"""

from __future__ import annotations

import os
import sys
import json
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

SF_NAMES = [
    "Fat Tails", "Volatility Clustering", "Leverage Effect",
    "Slow ACF Decay", "Cross-Asset Correlations", "No Raw Autocorrelation",
]

METRIC_KEYS = ["n_pass", "mmd", "discriminative_score", "wasserstein_1d",
               "correlation_matrix_distance", "train_time_s"]

METRIC_LABELS = {
    "n_pass": "Stylized Facts (out of 6)",
    "mmd": "MMD",
    "discriminative_score": "Discriminative Score",
    "wasserstein_1d": "Wasserstein-1",
    "correlation_matrix_distance": "Corr. Matrix Distance",
    "train_time_s": "Training Time (s)",
}

LOWER_IS_BETTER = {"mmd", "wasserstein_1d", "correlation_matrix_distance",
                    "train_time_s"}


def load_results(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def aggregate_by_model(results: list[dict]) -> dict[str, dict]:
    by_model = defaultdict(list)
    for r in results:
        by_model[r["model"]].append(r)

    agg = {}
    for name, runs in by_model.items():
        entry = {}
        for key in METRIC_KEYS:
            vals = [r[key] for r in runs]
            entry[key] = {"mean": float(np.mean(vals)),
                          "std": float(np.std(vals)),
                          "values": vals}
        entry["disc_distance"] = {
            "mean": float(np.mean([abs(r["discriminative_score"] - 0.5)
                                   for r in runs])),
            "std": float(np.std([abs(r["discriminative_score"] - 0.5)
                                 for r in runs])),
        }
        sf_pass = {}
        for sf_name in SF_NAMES:
            passes = []
            for r in runs:
                sf_data = r.get("stylized_facts", {})
                if sf_name in sf_data:
                    passes.append(1 if sf_data[sf_name].get("pass", False) else 0)
            sf_pass[sf_name] = float(np.mean(passes)) if passes else 0.0
        entry["sf_pass_rates"] = sf_pass

        entry["losses_mean"] = np.mean(
            [r["losses"] for r in runs], axis=0).tolist()
        entry["n_runs"] = len(runs)
        agg[name] = entry
    return agg


# =========================================================================
# Level 1: Individual metric rankings
# =========================================================================

def table_metrics(agg: dict, out_dir: str):
    """Table 2: Distributional metrics comparison."""
    rows = []
    for name, data in agg.items():
        row = {"Model": name}
        for key in METRIC_KEYS:
            m = data[key]["mean"]
            s = data[key]["std"]
            row[METRIC_LABELS[key]] = f"{m:.4f} +/- {s:.4f}"
        rows.append(row)
    df = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, "table_metrics.csv")
    df.to_csv(csv_path, index=False)
    print(f"  Saved {csv_path}")
    return df


def table_stylized_facts(agg: dict, out_dir: str):
    """Table 1: Stylized facts pass/fail heatmap data."""
    rows = []
    for name, data in agg.items():
        row = {"Model": name}
        for sf in SF_NAMES:
            row[sf] = data["sf_pass_rates"].get(sf, 0.0)
        row["Total"] = data["n_pass"]["mean"]
        rows.append(row)
    df = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, "table_stylized_facts.csv")
    df.to_csv(csv_path, index=False)
    print(f"  Saved {csv_path}")
    return df


def fig_per_metric_bars(agg: dict, out_dir: str):
    """Figure 1: Per-metric bar charts."""
    models = list(agg.keys())
    keys_to_plot = ["n_pass", "mmd", "discriminative_score", "wasserstein_1d",
                    "correlation_matrix_distance"]

    fig, axes = plt.subplots(1, len(keys_to_plot), figsize=(4 * len(keys_to_plot), 5))
    colors = sns.color_palette("husl", len(models))

    for ax, key in zip(axes, keys_to_plot):
        means = [agg[m][key]["mean"] for m in models]
        stds = [agg[m][key]["std"] for m in models]
        bars = ax.bar(range(len(models)), means, yerr=stds, capsize=3,
                      color=colors, edgecolor="black", linewidth=0.5)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=45, ha="right", fontsize=8)
        ax.set_title(METRIC_LABELS[key], fontsize=10)
        ax.tick_params(axis="y", labelsize=8)

        best_idx = int(np.argmax(means)) if key == "n_pass" else int(np.argmin(means))
        if key == "discriminative_score":
            dists = [abs(m - 0.5) for m in means]
            best_idx = int(np.argmin(dists))
        bars[best_idx].set_edgecolor("red")
        bars[best_idx].set_linewidth(2.5)

    fig.suptitle("Level 1: Individual Metric Comparison", fontsize=13,
                 fontweight="bold")
    fig.tight_layout()
    path = os.path.join(out_dir, "fig_per_metric_bars.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def fig_stylized_facts_heatmap(agg: dict, out_dir: str):
    """Stylized facts pass rate heatmap."""
    models = list(agg.keys())
    data = np.array([[agg[m]["sf_pass_rates"].get(sf, 0) for sf in SF_NAMES]
                      for m in models])
    fig, ax = plt.subplots(figsize=(10, max(3, len(models) * 0.6)))
    sns.heatmap(data, annot=True, fmt=".1f", cmap="RdYlGn",
                xticklabels=[s.replace(" ", "\n") for s in SF_NAMES],
                yticklabels=models, vmin=0, vmax=1, ax=ax,
                linewidths=0.5, cbar_kws={"label": "Pass Rate"})
    ax.set_title("Stylized Facts Pass Rates Across Models", fontweight="bold")
    fig.tight_layout()
    path = os.path.join(out_dir, "fig_stylized_facts_heatmap.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def fig_training_losses(agg: dict, out_dir: str):
    """Figure 5: Training loss curves."""
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = sns.color_palette("husl", len(agg))
    for (name, data), color in zip(agg.items(), colors):
        losses = data["losses_mean"]
        ax.plot(losses, label=name, color=color, linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss Curves", fontweight="bold")
    ax.legend(fontsize=8)
    ax.set_yscale("log")
    fig.tight_layout()
    path = os.path.join(out_dir, "fig_training_losses.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# =========================================================================
# Level 2: Composite scoring
# =========================================================================

def compute_composite(agg: dict) -> pd.DataFrame:
    """Compute min-max normalized composite scores."""
    models = list(agg.keys())
    metrics_for_composite = {
        "n_pass": ("mean", False),        # higher is better
        "mmd": ("mean", True),            # lower is better
        "disc_distance": ("mean", True),  # lower is better
        "wasserstein_1d": ("mean", True),
        "correlation_matrix_distance": ("mean", True),
    }
    weights = {"n_pass": 0.30, "mmd": 0.25, "disc_distance": 0.25,
               "wasserstein_1d": 0.10, "correlation_matrix_distance": 0.10}

    raw = {}
    for key, (stat, lower_better) in metrics_for_composite.items():
        vals = [agg[m][key][stat] for m in models]
        raw[key] = vals

    normalized = {}
    for key, vals in raw.items():
        v_min, v_max = min(vals), max(vals)
        rng = v_max - v_min if v_max > v_min else 1e-8
        _, lower_better = metrics_for_composite[key]
        if lower_better:
            normalized[key] = [(v_max - v) / rng for v in vals]
        else:
            normalized[key] = [(v - v_min) / rng for v in vals]

    composites_weighted = []
    composites_equal = []
    for i in range(len(models)):
        w_score = sum(weights[k] * normalized[k][i] for k in weights)
        e_score = sum(normalized[k][i] for k in weights) / len(weights)
        composites_weighted.append(w_score)
        composites_equal.append(e_score)

    rows = []
    for i, m in enumerate(models):
        row = {"Model": m}
        for key in metrics_for_composite:
            row[f"{key}_raw"] = raw[key][i]
            row[f"{key}_norm"] = normalized[key][i]
        row["composite_weighted"] = composites_weighted[i]
        row["composite_equal"] = composites_equal[i]
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("composite_weighted", ascending=False)
    return df


def fig_radar_chart(agg: dict, out_dir: str):
    """Figure 2: Radar/spider chart."""
    models = list(agg.keys())
    comp_df = compute_composite(agg)
    metric_cols = [c for c in comp_df.columns if c.endswith("_norm")]
    metric_labels = [c.replace("_norm", "") for c in metric_cols]

    N = len(metric_cols)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    colors = sns.color_palette("husl", len(models))

    for idx, (_, row) in enumerate(comp_df.iterrows()):
        values = [row[c] for c in metric_cols]
        values += values[:1]
        ax.plot(angles, values, color=colors[idx], linewidth=2,
                label=row["Model"])
        ax.fill(angles, values, color=colors[idx], alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels, fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.set_title("Level 2: Radar Chart (Normalized Metrics)", fontweight="bold",
                 pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)
    fig.tight_layout()
    path = os.path.join(out_dir, "fig_radar_chart.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def fig_composite_bars(agg: dict, out_dir: str):
    """Figure 3: Composite score bar chart."""
    comp_df = compute_composite(agg).sort_values("composite_weighted",
                                                  ascending=True)
    fig, ax = plt.subplots(figsize=(8, max(3, len(comp_df) * 0.7)))
    colors = sns.color_palette("husl", len(comp_df))

    ax.barh(range(len(comp_df)), comp_df["composite_weighted"],
            color=colors, edgecolor="black", linewidth=0.5)
    ax.set_yticks(range(len(comp_df)))
    ax.set_yticklabels(comp_df["Model"])
    ax.set_xlabel("Weighted Composite Score")
    ax.set_title("Level 2: Composite Ranking", fontweight="bold")
    ax.set_xlim(0, 1.05)

    for i, v in enumerate(comp_df["composite_weighted"]):
        ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=9)

    fig.tight_layout()
    path = os.path.join(out_dir, "fig_composite_bars.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# =========================================================================
# Level 3: Pareto and trade-off analysis
# =========================================================================

def fig_pareto_plots(agg: dict, out_dir: str):
    """Figure 4: Pareto frontier scatter plots."""
    models = list(agg.keys())
    pairs = [
        ("mmd", "disc_distance", "MMD", "|Disc. Score - 0.5|"),
        ("n_pass", "mmd", "Stylized Facts Passed", "MMD"),
    ]

    fig, axes = plt.subplots(1, len(pairs), figsize=(6 * len(pairs), 5))
    colors = sns.color_palette("husl", len(models))

    for ax, (xkey, ykey, xlabel, ylabel) in zip(axes, pairs):
        xs = [agg[m][xkey]["mean"] for m in models]
        ys = [agg[m][ykey]["mean"] for m in models]

        for i, m in enumerate(models):
            ax.scatter(xs[i], ys[i], color=colors[i], s=100, zorder=5,
                       edgecolors="black", linewidth=0.5)
            ax.annotate(m, (xs[i], ys[i]), textcoords="offset points",
                        xytext=(5, 5), fontsize=8)

        pts = list(zip(xs, ys, models))
        if xkey == "n_pass":
            pts_sorted = sorted(pts, key=lambda p: (-p[0], p[1]))
        else:
            pts_sorted = sorted(pts, key=lambda p: (p[0], p[1]))

        pareto = [pts_sorted[0]]
        for p in pts_sorted[1:]:
            if p[1] <= pareto[-1][1]:
                pareto.append(p)

        if len(pareto) > 1:
            px = [p[0] for p in pareto]
            py = [p[1] for p in pareto]
            ax.plot(px, py, "r--", alpha=0.5, linewidth=1.5,
                    label="Pareto frontier")
            ax.legend(fontsize=8)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f"Pareto: {xlabel} vs {ylabel}", fontweight="bold")

    fig.suptitle("Level 3: Trade-off Analysis", fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(out_dir, "fig_pareto_plots.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def fig_win_count_heatmap(agg: dict, out_dir: str):
    """Figure 8: Win-count heatmap (pairwise head-to-head)."""
    models = list(agg.keys())
    compare_keys = ["n_pass", "mmd", "wasserstein_1d",
                    "correlation_matrix_distance"]
    n = len(models)
    win_matrix = np.zeros((n, n), dtype=int)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            wins = 0
            for key in compare_keys:
                vi = agg[models[i]][key]["mean"]
                vj = agg[models[j]][key]["mean"]
                if key == "n_pass":
                    if vi > vj:
                        wins += 1
                else:
                    if vi < vj:
                        wins += 1
            di = agg[models[i]]["disc_distance"]["mean"]
            dj = agg[models[j]]["disc_distance"]["mean"]
            if di < dj:
                wins += 1
            win_matrix[i, j] = wins

    fig, ax = plt.subplots(figsize=(max(6, n * 0.9), max(5, n * 0.8)))
    total_metrics = len(compare_keys) + 1
    sns.heatmap(win_matrix, annot=True, fmt="d",
                xticklabels=models, yticklabels=models,
                cmap="YlOrRd", vmin=0, vmax=total_metrics,
                ax=ax, linewidths=0.5,
                cbar_kws={"label": f"Wins (out of {total_metrics})"})
    ax.set_xlabel("Opponent")
    ax.set_ylabel("Model")
    ax.set_title("Level 3: Win-Count Matrix (row beats column on N metrics)",
                 fontweight="bold")
    fig.tight_layout()
    path = os.path.join(out_dir, "fig_win_count_heatmap.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# =========================================================================
# Summary
# =========================================================================

def print_summary(agg: dict):
    comp_df = compute_composite(agg)
    print("\n" + "=" * 70)
    print("ABLATION STUDY REPORT SUMMARY")
    print("=" * 70)

    print("\n--- Level 1: Individual Metric Rankings ---")
    models = list(agg.keys())
    for key in METRIC_KEYS:
        vals = [(m, agg[m][key]["mean"]) for m in models]
        if key == "n_pass":
            vals.sort(key=lambda x: -x[1])
        elif key == "discriminative_score":
            vals.sort(key=lambda x: abs(x[1] - 0.5))
        else:
            vals.sort(key=lambda x: x[1])
        winner = vals[0]
        print(f"  {METRIC_LABELS[key]:35s} -> BEST: {winner[0]} ({winner[1]:.4f})")

    print("\n--- Level 2: Composite Ranking ---")
    for _, row in comp_df.iterrows():
        print(f"  {row['Model']:12s}  weighted={row['composite_weighted']:.4f}  "
              f"equal={row['composite_equal']:.4f}")

    best = comp_df.iloc[0]
    print(f"\n  >>> OVERALL BEST: {best['Model']} "
          f"(composite={best['composite_weighted']:.4f})")

    print("\n--- Level 3: Trade-off Notes ---")
    for _, row in comp_df.iterrows():
        strengths = []
        weaknesses = []
        for key in ["n_pass", "mmd", "disc_distance", "wasserstein_1d",
                     "correlation_matrix_distance"]:
            norm_val = row[f"{key}_norm"]
            if norm_val >= 0.8:
                strengths.append(key)
            elif norm_val <= 0.2:
                weaknesses.append(key)
        s_str = ", ".join(strengths) if strengths else "none"
        w_str = ", ".join(weaknesses) if weaknesses else "none"
        print(f"  {row['Model']:12s}  strengths=[{s_str}]  weaknesses=[{w_str}]")

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Generate DDPM ablation report")
    parser.add_argument("--results",
                        default=os.path.join(RESULTS_DIR, "ablation_results.json"))
    parser.add_argument("--out-dir", default=RESULTS_DIR)
    args = parser.parse_args()

    print(f"Loading results from {args.results}")
    results = load_results(args.results)
    print(f"Loaded {len(results)} experiment runs")

    agg = aggregate_by_model(results)
    os.makedirs(args.out_dir, exist_ok=True)

    print("\nGenerating tables...")
    table_stylized_facts(agg, args.out_dir)
    table_metrics(agg, args.out_dir)

    comp_df = compute_composite(agg)
    comp_df.to_csv(os.path.join(args.out_dir, "table_composite.csv"), index=False)
    print(f"  Saved {os.path.join(args.out_dir, 'table_composite.csv')}")

    print("\nGenerating figures...")
    fig_per_metric_bars(agg, args.out_dir)
    fig_stylized_facts_heatmap(agg, args.out_dir)
    fig_training_losses(agg, args.out_dir)
    fig_radar_chart(agg, args.out_dir)
    fig_composite_bars(agg, args.out_dir)
    fig_pareto_plots(agg, args.out_dir)
    fig_win_count_heatmap(agg, args.out_dir)

    print_summary(agg)


# =========================================================================
# Domain-specific financial visualizations (require re-generating samples)
# =========================================================================

def generate_domain_visualizations(out_dir: str, data_dir: str, device: str):
    """
    Re-train top models briefly and generate domain-specific plots:
    return distributions, QQ-plots, ACF, synthetic paths, correlation matrices.
    """
    import sys
    sys.path.insert(0, PROJECT_ROOT)
    from src.models.ddpm_improved import ImprovedDDPM
    from src.evaluation.visualization import (
        plot_return_distributions, plot_qq_comparison,
        plot_acf_comparison, plot_synthetic_paths,
        plot_correlation_matrices,
    )

    windows = np.load(os.path.join(data_dir, "windows.npy"))
    cond_path = os.path.join(data_dir, "window_cond.npy")
    cond = np.load(cond_path) if os.path.exists(cond_path) else None
    asset_names_path = os.path.join(data_dir, "asset_names.json")
    asset_names = json.load(open(asset_names_path)) if os.path.exists(asset_names_path) else None

    n_features = windows.shape[2]
    seq_len = windows.shape[1]
    cond_dim = cond.shape[1] if cond is not None else 0

    models_to_viz = {
        "baseline": {},
        "vpred": {"use_vpred": True},
        "sigmoid": {"use_sigmoid_schedule": True},
        "best_combo": {"use_vpred": True, "use_self_cond": True,
                       "use_sigmoid_schedule": True},
    }

    np.random.seed(42)
    import torch
    torch.manual_seed(42)

    synthetic_windows = {}
    for name, flags in models_to_viz.items():
        print(f"  Training {name} for visualization (100 epochs)...")
        model = ImprovedDDPM(
            n_features=n_features, seq_len=seq_len, T=1000,
            base_channels=64, channel_mults=(1, 2, 4),
            cond_dim=cond_dim, device=device, **flags,
        )
        model.train(windows, cond=cond, epochs=100, batch_size=64, lr=2e-4)
        syn = model.generate(500, seq_len)
        synthetic_windows[name] = syn
        print(f"    {name}: generated {syn.shape}")

    real_flat = windows.reshape(-1, n_features).flatten()
    syn_flat_dict = {k: v.reshape(-1, v.shape[-1]).flatten()
                     for k, v in synthetic_windows.items()}

    print("  Plotting return distributions...")
    plot_return_distributions(
        real_flat[:50000], {k: v[:50000] for k, v in syn_flat_dict.items()},
        save_path=os.path.join(out_dir, "fig_return_distributions.png"))

    print("  Plotting QQ-plots...")
    plot_qq_comparison(
        real_flat[:50000], {k: v[:50000] for k, v in syn_flat_dict.items()},
        save_path=os.path.join(out_dir, "fig_qq_plots.png"))

    print("  Plotting ACF of |returns|...")
    plot_acf_comparison(
        real_flat[:50000], {k: v[:50000] for k, v in syn_flat_dict.items()},
        mode="absolute",
        save_path=os.path.join(out_dir, "fig_acf_absolute.png"))

    print("  Plotting ACF of returns^2...")
    plot_acf_comparison(
        real_flat[:50000], {k: v[:50000] for k, v in syn_flat_dict.items()},
        mode="squared",
        save_path=os.path.join(out_dir, "fig_acf_squared.png"))

    for name, syn in synthetic_windows.items():
        print(f"  Plotting synthetic paths for {name}...")
        plot_synthetic_paths(
            syn, n_paths=50, asset_idx=0, title=f"{name} (asset 0)",
            save_path=os.path.join(out_dir, f"fig_paths_{name}.png"))

    best_name = "sigmoid"
    best_syn = synthetic_windows[best_name]
    print(f"  Plotting correlation matrices (real vs {best_name})...")
    plot_correlation_matrices(
        windows, best_syn, asset_names=asset_names,
        save_path=os.path.join(out_dir, "fig_correlation_matrices.png"))

    print(f"  All domain visualizations saved to {out_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Generate DDPM ablation report")
    parser.add_argument("--results",
                        default=os.path.join(RESULTS_DIR, "ablation_results.json"))
    parser.add_argument("--out-dir", default=RESULTS_DIR)
    parser.add_argument("--skip-domain-viz", action="store_true",
                        help="Skip domain-specific visualizations (faster)")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    print(f"Loading results from {args.results}")
    results = load_results(args.results)
    print(f"Loaded {len(results)} experiment runs")

    agg = aggregate_by_model(results)
    os.makedirs(args.out_dir, exist_ok=True)

    print("\nGenerating tables...")
    table_stylized_facts(agg, args.out_dir)
    table_metrics(agg, args.out_dir)

    comp_df = compute_composite(agg)
    comp_df.to_csv(os.path.join(args.out_dir, "table_composite.csv"), index=False)
    print(f"  Saved {os.path.join(args.out_dir, 'table_composite.csv')}")

    print("\nGenerating aggregate figures...")
    fig_per_metric_bars(agg, args.out_dir)
    fig_stylized_facts_heatmap(agg, args.out_dir)
    fig_training_losses(agg, args.out_dir)
    fig_radar_chart(agg, args.out_dir)
    fig_composite_bars(agg, args.out_dir)
    fig_pareto_plots(agg, args.out_dir)
    fig_win_count_heatmap(agg, args.out_dir)

    if not args.skip_domain_viz:
        data_dir = args.data_dir
        if data_dir is None:
            from src.utils.config import DATA_DIR, DEFAULT_DEVICE
            data_dir = DATA_DIR
        device = args.device
        if device is None:
            from src.utils.config import DEFAULT_DEVICE
            device = DEFAULT_DEVICE

        print("\nGenerating domain-specific financial visualizations...")
        generate_domain_visualizations(args.out_dir, data_dir, device)

    print_summary(agg)


if __name__ == "__main__":
    main()

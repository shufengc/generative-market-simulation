"""
evaluate_regimes.py
===================
Regime-stratified evaluation of a trained conditional DDPM.

Loads .npy samples saved by run_conditional_ddpm.py and compares them
against real windows of the same regime.

Usage:
    python experiments/evaluate_regimes.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.evaluation.metrics import (                            # noqa: E402
    maximum_mean_discrepancy,
    discriminative_score,
    correlation_matrix_distance,
)
from src.evaluation.stylized_facts import run_all_tests         # noqa: E402

_DEFAULT_RESULTS_DIR = os.path.join(ROOT, "experiments", "results", "conditional_ddpm")
DATA_DIR             = os.path.join(ROOT, "data")

REGIMES    = ["crisis", "calm", "normal"]
REGIME_INT = {"crisis": 1, "calm": 2, "normal": 0}


def load_real_windows():
    windows        = np.load(os.path.join(DATA_DIR, "windows.npy"))
    window_regimes = np.load(os.path.join(DATA_DIR, "window_regimes.npy"))
    return windows, window_regimes


def evaluate_regime(real_windows: np.ndarray, synthetic_windows: np.ndarray,
                    regime_name: str) -> dict:
    from scipy.stats import kurtosis, skew  # noqa: PLC0415

    # cap at 1000 to keep evaluation runtime manageable
    n = min(len(real_windows), len(synthetic_windows), 1000)
    real_sub = real_windows[:n]
    syn_sub  = synthetic_windows[:n]

    # flatten (N, T, D) -> (N, T*D) for distribution-level metrics
    mmd      = maximum_mean_discrepancy(real_sub.reshape(n, -1), syn_sub.reshape(n, -1))
    disc     = discriminative_score(real_sub, syn_sub)
    corr_dist = correlation_matrix_distance(real_sub, syn_sub)

    # Stylized facts on synthetic alone (absolute thresholds)
    sf_list  = run_all_tests(syn_sub)
    sf_count = sum(1 for r in sf_list if r.get("pass", False))
    sf_details = {r.get("name", f"sf{i}"): r.get("pass", False)
                  for i, r in enumerate(sf_list)}

    # per-asset vol: mean of per-asset std across all windows and time steps
    syn_flat  = syn_sub.reshape(-1, syn_sub.shape[-1])
    real_flat = real_sub.reshape(-1, real_sub.shape[-1])
    syn_vol   = float(np.std(syn_flat, axis=0).mean())
    real_vol  = float(np.std(real_flat, axis=0).mean())
    syn_kurt  = float(kurtosis(syn_sub.flatten(), fisher=True))
    real_kurt = float(kurtosis(real_sub.flatten(), fisher=True))
    syn_skew  = float(skew(syn_sub.flatten()))
    real_skew = float(skew(real_sub.flatten()))

    return {
        "regime":              regime_name,
        "n_real":              len(real_windows),
        "n_synthetic":         len(synthetic_windows),
        "n_evaluated":         n,
        "sf_count":            sf_count,
        "sf_details":          sf_details,
        "mmd":                 round(float(mmd), 6),
        "discriminative_score": round(float(disc), 4),
        "correlation_distance": round(float(corr_dist), 4),
        "syn_vol_mean":        round(syn_vol, 4),
        "real_vol_mean":       round(real_vol, 4),
        "syn_excess_kurtosis": round(syn_kurt, 3),
        "real_excess_kurtosis": round(real_kurt, 3),
        "syn_skewness":        round(syn_skew, 4),
        "real_skewness":       round(real_skew, 4),
    }


def print_summary(all_results: list[dict]) -> None:
    print("\n" + "=" * 90)
    print("REGIME-STRATIFIED EVALUATION SUMMARY")
    print("=" * 90)
    hdr = (f"{'Regime':<10} {'n_real':>8} {'SF':>6} {'MMD':>8} "
           f"{'Disc':>8} {'CorrDist':>10} {'SynVol':>8} {'RealVol':>8} "
           f"{'SynKurt':>8} {'RealKurt':>9}")
    print(hdr)
    print("-" * 90)
    for r in all_results:
        print(
            f"{r['regime']:<10} {r['n_real']:>8} {r['sf_count']:>4}/6 "
            f"{r['mmd']:>8.4f} {r['discriminative_score']:>8.3f} "
            f"{r['correlation_distance']:>10.4f} {r['syn_vol_mean']:>8.4f} "
            f"{r['real_vol_mean']:>8.4f} {r['syn_excess_kurtosis']:>8.2f} "
            f"{r['real_excess_kurtosis']:>9.2f}"
        )
    print("=" * 90)

    by_name = {r["regime"]: r for r in all_results}
    if "crisis" in by_name and "calm" in by_name:
        c_vol  = by_name["crisis"]["syn_vol_mean"]
        k_vol  = by_name["calm"]["syn_vol_mean"]
        c_kurt = by_name["crisis"]["syn_excess_kurtosis"]
        k_kurt = by_name["calm"]["syn_excess_kurtosis"]
        print("\nConditioning sanity checks:")
        # Primary L3 success criterion: crisis should produce higher vol than calm
        print(f"  Crisis vol {c_vol:.4f} > Calm vol {k_vol:.4f}  "
              + ("PASS" if c_vol > k_vol else "FAIL (crisis should have higher vol)"))
        print(f"  Crisis kurtosis {c_kurt:.2f} > Calm kurtosis {k_kurt:.2f}  "
              + ("PASS" if c_kurt > k_kurt else "FAIL (crisis should have fatter tails)"))


def make_comparison_plot(all_results: list[dict], windows: np.ndarray,
                          window_regimes: np.ndarray) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    regimes = [r["regime"] for r in all_results]
    c_map   = {"crisis": "#e53935", "calm": "#43a047", "normal": "#1e88e5"}
    c_list  = [c_map[r] for r in regimes]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    # Volatility
    ax = axes[0]
    x = np.arange(len(regimes)); w = 0.35
    ax.bar(x - w/2, [r["real_vol_mean"] for r in all_results], w,
           label="Real", color="#90a4ae", edgecolor="#546e7a")
    ax.bar(x + w/2, [r["syn_vol_mean"]  for r in all_results], w,
           label="Synthetic", color=c_list, edgecolor="white", linewidth=0.8)
    ax.set_xticks(x); ax.set_xticklabels([r.capitalize() for r in regimes])
    ax.set_title("Volatility by Regime", fontweight="bold"); ax.legend()
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    # MMD
    ax = axes[1]
    bars = ax.bar(regimes, [r["mmd"] for r in all_results],
                  color=c_list, edgecolor="white")
    for bar, v in zip(bars, [r["mmd"] for r in all_results]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f"{v:.4f}", ha="center", va="bottom", fontsize=9)
    ax.set_title("MMD per Regime (lower = better)", fontweight="bold")
    ax.set_xticklabels([r.capitalize() for r in regimes])
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    # SF count
    ax = axes[2]
    bars = ax.bar(regimes, [r["sf_count"] for r in all_results],
                  color=c_list, edgecolor="white")
    ax.axhline(5, linestyle="--", color="#1b5e20", linewidth=1.5,
               label="Unconditional DDPM (5/6)")
    for bar, v in zip(bars, [r["sf_count"] for r in all_results]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f"{v}/6", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylim(0, 7); ax.set_yticks(range(7))
    ax.set_title("SF Count per Regime", fontweight="bold")
    ax.set_xticklabels([r.capitalize() for r in regimes])
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    fig.suptitle("Regime-Stratified Evaluation — Conditional DDPM", fontweight="bold")
    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, "regime_comparison.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--results-dir", type=str, default=None,
                        help="Directory containing synthetic_*.npy files "
                             "(default: experiments/results/conditional_ddpm)")
    args = parser.parse_args()

    RESULTS_DIR = args.results_dir or _DEFAULT_RESULTS_DIR
    os.makedirs(RESULTS_DIR, exist_ok=True)
    windows, window_regimes = load_real_windows()
    all_results = []

    for regime_name in REGIMES:
        npy_path = os.path.join(RESULTS_DIR, f"synthetic_{regime_name}.npy")
        if not os.path.exists(npy_path):
            print(f"  SKIP {regime_name}: {npy_path} not found "
                  "(run run_conditional_ddpm.py first)")
            continue
        synthetic = np.load(npy_path)
        mask = window_regimes == REGIME_INT[regime_name]
        print(f"Evaluating {regime_name}: {mask.sum()} real, {len(synthetic)} synthetic ...")
        result = evaluate_regime(windows[mask], synthetic, regime_name)
        all_results.append(result)

    if not all_results:
        print("No results found. Run run_conditional_ddpm.py first.")
        return

    print_summary(all_results)

    out_json = os.path.join(RESULTS_DIR, "regime_eval_summary.json")
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {out_json}")

    if not args.no_plot:
        make_comparison_plot(all_results, windows, window_regimes)


if __name__ == "__main__":
    main()

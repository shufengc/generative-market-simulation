"""
GARCH Rebaseline -- fit and evaluate GARCHModel under unified settings.

Matches Phase 6 DDPM and NormFlow rebaseline setup:
  - stride=1 data (5,293 windows, 16 assets, seq_len=60)
  - Current evaluation framework (Hill, GARCH gamma, Hurst R/S, max-eigenvalue, MAA+LB)
  - 3 seeds (42, 123, 456)
  - 1,000 generated samples for evaluation

Usage:
    PYTHONPATH=. python experiments/run_garch_rebaseline.py
    PYTHONPATH=. python experiments/run_garch_rebaseline.py --quick
"""

from __future__ import annotations

import os
import sys
import json
import time
import argparse

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.config import DATA_DIR
from src.models.garch import GARCHModel
from src.evaluation.stylized_facts import run_all_tests, count_passes
from src.evaluation.metrics import full_evaluation

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "garch_rebaseline")
SEEDS = [42, 123, 456]


def set_seed(seed: int):
    np.random.seed(seed)


def run_single(windows: np.ndarray, real_windows: np.ndarray,
               seed: int, n_gen: int) -> dict:
    """Fit one GARCH model with one seed and return results."""
    set_seed(seed)
    n_features = windows.shape[2]
    seq_len = windows.shape[1]

    print(f"\n{'='*60}")
    print(f"GARCH | Seed: {seed}")
    print(f"  Data: {windows.shape}, n_gen={n_gen}")
    print(f"{'='*60}")

    model = GARCHModel(n_features=n_features, seq_len=seq_len, device="cpu")

    t0 = time.time()
    model.train(windows)
    train_time = time.time() - t0
    print(f"  Training time: {train_time:.1f}s")

    t1 = time.time()
    syn = model.generate(n_samples=n_gen)
    gen_time = time.time() - t1
    print(f"  Generation time: {gen_time:.1f}s, shape: {syn.shape}")

    # Stylized facts evaluation
    real_returns = real_windows.reshape(-1, real_windows.shape[-1])
    syn_returns = syn.reshape(-1, syn.shape[-1])

    sf_results = run_all_tests(syn_returns, real_returns=real_returns)
    n_pass = count_passes(sf_results)
    print(f"  SF: {n_pass}/6  ->  {sf_results}")

    # Full metrics (MMD, Wasserstein, discriminative score, correlation distance)
    metrics = full_evaluation(syn, real_windows)
    print(f"  MMD: {metrics.get('mmd', float('nan')):.4f}")
    print(f"  Wasserstein-1D: {metrics.get('wasserstein_1d', float('nan')):.4f}")
    print(f"  Discriminative: {metrics.get('discriminative_score', float('nan')):.4f}")
    print(f"  CorrDist: {metrics.get('correlation_matrix_distance', float('nan')):.4f}")

    return {
        "seed": seed,
        "sf_results": sf_results,
        "sf_passed": n_pass,
        "train_time_s": round(train_time, 2),
        "gen_time_s": round(gen_time, 2),
        **{k: round(float(v), 6) for k, v in metrics.items()
           if isinstance(v, (int, float))},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="Fast run: 1 seed, 200 samples")
    parser.add_argument("--n-gen", type=int, default=1000,
                        help="Number of samples to generate per seed")
    args = parser.parse_args()

    seeds = [42] if args.quick else SEEDS
    n_gen = 200 if args.quick else args.n_gen

    # Load data
    windows = np.load(os.path.join(DATA_DIR, "windows.npy"))
    real_windows = windows.copy()
    print(f"Loaded windows: {windows.shape}")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    all_results = []
    for seed in seeds:
        result = run_single(windows, real_windows, seed=seed, n_gen=n_gen)
        all_results.append(result)

    # Save raw results
    out_json = os.path.join(RESULTS_DIR, "garch_results.json")
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved results to {out_json}")

    # Summary statistics
    sf_vals = [r["sf_passed"] for r in all_results]
    mmd_vals = [r.get("mmd", float("nan")) for r in all_results]
    disc_vals = [r.get("discriminative_score", float("nan")) for r in all_results]
    w1_vals = [r.get("wasserstein_1d", float("nan")) for r in all_results]
    corr_vals = [r.get("correlation_matrix_distance", float("nan")) for r in all_results]

    print(f"\n{'='*60}")
    print(f"GARCH REBASELINE SUMMARY ({len(seeds)} seeds)")
    print(f"{'='*60}")
    print(f"  SF passed:    {np.mean(sf_vals):.1f} ± {np.std(sf_vals):.2f}")
    print(f"  MMD:          {np.mean(mmd_vals):.4f} ± {np.std(mmd_vals):.4f}")
    print(f"  Discriminative: {np.mean(disc_vals):.4f} ± {np.std(disc_vals):.4f}")
    print(f"  Wasserstein:  {np.mean(w1_vals):.4f} ± {np.std(w1_vals):.4f}")
    print(f"  CorrDist:     {np.mean(corr_vals):.4f} ± {np.std(corr_vals):.4f}")

    # Save summary CSV
    import csv
    out_csv = os.path.join(RESULTS_DIR, "summary.csv")
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "seed", "sf_passed", "mmd", "wasserstein_1d",
            "discriminative_score", "correlation_matrix_distance",
            "train_time_s", "gen_time_s",
        ])
        writer.writeheader()
        for r in all_results:
            writer.writerow({k: r.get(k, "") for k in writer.fieldnames})
    print(f"Saved summary to {out_csv}")

    return all_results


if __name__ == "__main__":
    main()

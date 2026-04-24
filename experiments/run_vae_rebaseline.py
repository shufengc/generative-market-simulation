"""
VAE Improved Rebaseline -- train and evaluate FinancialVAE under unified settings.

Matches Phase 6 DDPM and NormFlow rebaseline setup:
  - stride=1 data (5,293 windows, 16 assets, seq_len=60)
  - Current evaluation framework (Hill, GARCH gamma, Hurst R/S, max-eigenvalue, MAA+LB)
  - 400 epochs, 3 seeds (42, 123, 456)
  - 1,000 generated samples for evaluation

Usage:
    PYTHONPATH=. python experiments/run_vae_rebaseline.py
    PYTHONPATH=. python experiments/run_vae_rebaseline.py --quick
"""

from __future__ import annotations

import os
import sys
import json
import time
import argparse
import csv

import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.config import DATA_DIR, DEFAULT_DEVICE
from src.models.vae import FinancialVAE
from src.evaluation.stylized_facts import run_all_tests, count_passes
from src.evaluation.metrics import full_evaluation

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "vae_rebaseline")
SEEDS = [42, 123, 456]


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        pass  # MPS doesn't have per-op seed control, np seed handles data shuffling


def run_single(windows: np.ndarray, cond: np.ndarray | None,
               real_windows: np.ndarray, seed: int,
               epochs: int, batch_size: int, n_gen: int,
               device: str) -> dict:
    """Train one VAE with one seed and return evaluation results."""
    set_seed(seed)
    n_features = windows.shape[2]
    seq_len = windows.shape[1]
    cond_dim = int(cond.shape[1]) if cond is not None else 0

    print(f"\n{'='*60}")
    print(f"VAE Improved | Seed: {seed} | Epochs: {epochs} | Device: {device}")
    print(f"  Data: {windows.shape}, cond_dim={cond_dim}, n_gen={n_gen}")
    print(f"{'='*60}")

    model = FinancialVAE(
        n_features=n_features,
        seq_len=seq_len,
        cond_dim=cond_dim,
        factor_dim=2,
        device=device,
    )

    t0 = time.time()
    model.train(
        windows,
        cond=cond,
        epochs=epochs,
        batch_size=batch_size,
        teacher_forcing_end=0.3,
    )
    train_time = time.time() - t0
    print(f"  Training time: {train_time:.1f}s ({train_time/60:.1f} min)")

    t1 = time.time()
    syn = model.generate(n_samples=n_gen, cond=cond)
    gen_time = time.time() - t1
    print(f"  Generation time: {gen_time:.1f}s, shape: {syn.shape}")

    real_returns = real_windows.reshape(-1, real_windows.shape[-1])
    syn_returns = syn.reshape(-1, syn.shape[-1])

    sf_results = run_all_tests(syn_returns, real_returns=real_returns)
    n_pass = count_passes(sf_results)
    print(f"  SF: {n_pass}/6")

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
                        help="Fast run: 1 seed, 20 epochs, 200 samples")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--n-gen", type=int, default=1000)
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    args = parser.parse_args()

    seeds = [42] if args.quick else SEEDS
    epochs = 20 if args.quick else (args.epochs or 400)
    n_gen = 200 if args.quick else args.n_gen

    # Load data
    windows = np.load(os.path.join(DATA_DIR, "windows.npy"))
    real_windows = windows.copy()
    print(f"Loaded windows: {windows.shape}")

    cond_path = os.path.join(DATA_DIR, "window_cond.npy")
    cond = np.load(cond_path).astype(np.float32) if os.path.exists(cond_path) else None
    if cond is not None:
        print(f"Loaded conditioning: {cond.shape}")
    else:
        print("No conditioning found, running unconditional.")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    all_results = []
    for seed in seeds:
        result = run_single(
            windows, cond, real_windows,
            seed=seed, epochs=epochs,
            batch_size=64, n_gen=n_gen,
            device=args.device,
        )
        all_results.append(result)

    # Save raw results
    out_json = os.path.join(RESULTS_DIR, "vae_results.json")
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
    print(f"VAE IMPROVED REBASELINE SUMMARY ({len(seeds)} seeds)")
    print(f"{'='*60}")
    print(f"  SF passed:    {np.mean(sf_vals):.1f} ± {np.std(sf_vals):.2f}")
    print(f"  MMD:          {np.mean(mmd_vals):.4f} ± {np.std(mmd_vals):.4f}")
    print(f"  Discriminative: {np.mean(disc_vals):.4f} ± {np.std(disc_vals):.4f}")
    print(f"  Wasserstein:  {np.mean(w1_vals):.4f} ± {np.std(w1_vals):.4f}")
    print(f"  CorrDist:     {np.mean(corr_vals):.4f} ± {np.std(corr_vals):.4f}")

    # Save summary CSV
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

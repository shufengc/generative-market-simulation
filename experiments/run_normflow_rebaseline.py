"""
NormFlow Rebaseline -- train and evaluate NormFlow under unified settings.

Matches the Phase 6 DDPM ablation setup:
  - stride=1 data (5,293 windows)
  - Current evaluation framework (Hill estimator, GARCH gamma, Hurst R/S, etc.)
  - 400 epochs, 3 seeds (42, 123, 456)
  - 1,000 generated samples for evaluation

Usage:
    PYTHONPATH=. python experiments/run_normflow_rebaseline.py
    PYTHONPATH=. python experiments/run_normflow_rebaseline.py --quick
    PYTHONPATH=. python experiments/run_normflow_rebaseline.py --models default actnorm_off deep
"""

from __future__ import annotations

import os
import sys
import json
import time
import argparse
from collections import defaultdict

import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.config import DATA_DIR, DEFAULT_DEVICE
from src.models.normalizing_flow import NormalizingFlowModel
from src.evaluation.stylized_facts import run_all_tests, count_passes
from src.evaluation.metrics import full_evaluation

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "normflow_rebaseline")

# ---------------------------------------------------------------------------
# NormFlow variant configs for ablation
# ---------------------------------------------------------------------------

MODEL_CONFIGS = {
    # Default config (matches existing cross-model runs)
    "default": dict(
        hidden_dim=256, n_flow_layers=8, n_blocks_per_layer=2,
        use_actnorm=True, use_batchnorm=True, multi_scale=False,
    ),
    # Deeper model
    "deep": dict(
        hidden_dim=256, n_flow_layers=12, n_blocks_per_layer=2,
        use_actnorm=True, use_batchnorm=True, multi_scale=False,
    ),
    # Wider hidden dim
    "wide": dict(
        hidden_dim=512, n_flow_layers=8, n_blocks_per_layer=2,
        use_actnorm=True, use_batchnorm=True, multi_scale=False,
    ),
    # Without ActNorm (BatchNorm only)
    "actnorm_off": dict(
        hidden_dim=256, n_flow_layers=8, n_blocks_per_layer=2,
        use_actnorm=False, use_batchnorm=True, multi_scale=False,
    ),
    # Multi-scale architecture
    "multi_scale": dict(
        hidden_dim=256, n_flow_layers=8, n_blocks_per_layer=2,
        use_actnorm=True, use_batchnorm=True, multi_scale=True,
    ),
    # More residual blocks per coupling layer
    "deep_coupling": dict(
        hidden_dim=256, n_flow_layers=8, n_blocks_per_layer=4,
        use_actnorm=True, use_batchnorm=True, multi_scale=False,
    ),
}


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_single(model_name: str, config: dict, windows: np.ndarray,
               real_windows: np.ndarray, seed: int, epochs: int,
               batch_size: int, lr: float, n_gen: int,
               device: str) -> dict:
    """Train one NormFlow variant with one seed and return results."""
    set_seed(seed)
    n_features = windows.shape[2]
    seq_len = windows.shape[1]

    print(f"\n{'='*60}")
    print(f"Model: {model_name} | Seed: {seed} | Epochs: {epochs}")
    print(f"  Config: {config}")
    print(f"{'='*60}")

    model = NormalizingFlowModel(
        n_features=n_features, seq_len=seq_len,
        device=device, **config,
    )

    n_params = sum(p.numel() for p in model.flow.parameters())
    print(f"  Parameters: {n_params:,}")

    t0 = time.time()
    history = model.train(windows, epochs=epochs, batch_size=batch_size, lr=lr)
    train_time = time.time() - t0

    print(f"  Training took {train_time:.1f}s")
    print(f"  Generating {n_gen} samples...")

    t1 = time.time()
    synthetic = model.generate(n_gen, seq_len)
    gen_time = time.time() - t1
    print(f"  Generation took {gen_time:.1f}s")

    # Check for NaN/Inf in generated data
    n_bad = np.isnan(synthetic).sum() + np.isinf(synthetic).sum()
    if n_bad > 0:
        print(f"  WARNING: {n_bad} NaN/Inf values in generated data")
        synthetic = np.nan_to_num(synthetic, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"  Running evaluation...")
    sf_results = run_all_tests(synthetic)
    n_pass = count_passes(sf_results)

    eval_real = real_windows[:n_gen] if len(real_windows) >= n_gen else real_windows
    metrics = full_evaluation(eval_real, synthetic[:len(eval_real)])

    sf_details = {}
    for r in sf_results:
        name = r.get("name", "unknown")
        sf_details[name] = {
            "pass": r.get("pass", False),
            **{k: v for k, v in r.items() if k not in ("name", "pass", "acf_values")},
        }

    result = {
        "model": model_name,
        "seed": seed,
        "epochs": epochs,
        "n_params": n_params,
        "config": config,
        "train_time_s": round(train_time, 2),
        "gen_time_s": round(gen_time, 2),
        "n_pass": n_pass,
        "stylized_facts": sf_details,
        "mmd": metrics["mmd"],
        "wasserstein_1d": metrics["wasserstein_1d"],
        "discriminative_score": metrics["discriminative_score"],
        "correlation_matrix_distance": metrics["correlation_matrix_distance"],
        "ks_stat": metrics["ks_stat"],
        "moments": metrics["moments"],
        "losses": history.get("losses", []),
    }

    print(f"  Result: SF={n_pass}/6 | MMD={metrics['mmd']:.4f} | "
          f"Disc={metrics['discriminative_score']:.2f} | "
          f"W1={metrics['wasserstein_1d']:.4f} | "
          f"CorrDist={metrics['correlation_matrix_distance']:.4f}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="NormFlow Rebaseline: unified eval (stride=1, 400 epochs, 3 seeds)")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Model variants to run (default: all)")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--n-gen", type=int, default=1000)
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: 30 epochs, 200 samples")
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--out-dir", default=RESULTS_DIR)
    args = parser.parse_args()

    if args.quick:
        args.epochs = args.epochs if args.epochs != 400 else 30
        args.n_gen = min(args.n_gen, 200)

    models_to_run = args.models or list(MODEL_CONFIGS.keys())
    print(f"Device: {args.device}")
    print(f"Models: {models_to_run}")
    print(f"Seeds: {args.seeds}")
    print(f"Epochs: {args.epochs}")

    # Load stride=1 data
    windows_path = os.path.join(args.data_dir, "windows.npy")
    if not os.path.exists(windows_path):
        print(f"ERROR: Data not found at {windows_path}")
        print("Run the pipeline first: PYTHONPATH=. python src/run_pipeline.py")
        sys.exit(1)

    windows = np.load(windows_path)
    print(f"Data: {windows.shape[0]} windows, {windows.shape[1]} steps, "
          f"{windows.shape[2]} features")

    # Resume support
    out_path = os.path.join(args.out_dir, "normflow_results.json")
    all_results = []
    if os.path.exists(out_path):
        with open(out_path) as f:
            all_results = json.load(f)
        existing = {(r["model"], r["seed"]) for r in all_results}
        print(f"Resuming: {len(all_results)} existing runs loaded")
    else:
        existing = set()

    for model_name in models_to_run:
        if model_name not in MODEL_CONFIGS:
            print(f"Unknown model: {model_name}, skipping")
            continue

        config = MODEL_CONFIGS[model_name]

        for seed in args.seeds:
            if (model_name, seed) in existing:
                print(f"  Skipping {model_name} seed={seed} (already done)")
                continue

            result = run_single(
                model_name=model_name,
                config=config,
                windows=windows,
                real_windows=windows,
                seed=seed,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                n_gen=args.n_gen,
                device=args.device,
            )
            all_results.append(result)

            os.makedirs(args.out_dir, exist_ok=True)
            with open(out_path, "w") as f:
                json.dump(all_results, f, indent=2, default=str)
            print(f"  Saved intermediate results -> {out_path}")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("ALL EXPERIMENTS COMPLETE")
    print(f"{'='*60}")
    print(f"Results: {out_path}")

    print(f"\n{'='*60}")
    print("SUMMARY (mean +/- std across seeds)")
    print(f"{'='*60}")
    by_model = defaultdict(list)
    for r in all_results:
        by_model[r["model"]].append(r)

    print(f"  {'Model':<20s}  {'SF':>10s}  {'MMD':>16s}  {'Disc':>12s}  "
          f"{'W1':>16s}  {'CorrDist':>16s}")
    print(f"  {'-'*20}  {'-'*10}  {'-'*16}  {'-'*12}  {'-'*16}  {'-'*16}")

    for name, runs in by_model.items():
        sfs = [r["n_pass"] for r in runs]
        mmds = [r["mmd"] for r in runs]
        discs = [r["discriminative_score"] for r in runs]
        w1s = [r["wasserstein_1d"] for r in runs]
        cds = [r["correlation_matrix_distance"] for r in runs]
        print(f"  {name:<20s}  "
              f"{np.mean(sfs):4.1f}+/-{np.std(sfs):3.1f}  "
              f"{np.mean(mmds):.4f}+/-{np.std(mmds):.4f}  "
              f"{np.mean(discs):.2f}+/-{np.std(discs):.2f}  "
              f"{np.mean(w1s):.4f}+/-{np.std(w1s):.4f}  "
              f"{np.mean(cds):.4f}+/-{np.std(cds):.4f}")

    # Save summary CSV
    summary_path = os.path.join(args.out_dir, "summary.csv")
    with open(summary_path, "w") as f:
        f.write("model,seed,n_params,epochs,sf_passed,mmd,wasserstein_1d,"
                "discriminative_score,correlation_matrix_distance,train_time_s\n")
        for r in all_results:
            f.write(f"{r['model']},{r['seed']},{r.get('n_params','')},{r['epochs']},"
                    f"{r['n_pass']},{r['mmd']},{r['wasserstein_1d']},"
                    f"{r['discriminative_score']},{r['correlation_matrix_distance']},"
                    f"{r['train_time_s']}\n")
    print(f"\nSummary CSV: {summary_path}")


if __name__ == "__main__":
    main()

"""
End-to-end pipeline for the Generative Market Simulation project.

Usage:
    python -m src.run_pipeline                     # full pipeline
    python -m src.run_pipeline --skip-download      # skip data download
    python -m src.run_pipeline --models ddpm ddpm_improved garch   # train only selected models
    python -m src.run_pipeline --quick              # fast run for testing
"""

from __future__ import annotations

import os
import sys
import json
import argparse
import time

import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.config import (
    DATA_DIR, CHECKPOINTS_DIR, RESULTS_DIR,
    DEFAULT_EPOCHS, DEFAULT_BATCH_SIZE, DEFAULT_LR, SEED, DEFAULT_DEVICE,
)


def set_seed(seed: int = SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def step_download(data_dir: str, fred_key: str | None = None):
    """Step 1: Download market data."""
    print("\n" + "=" * 60)
    print("STEP 1: Downloading market data")
    print("=" * 60)
    from src.data.download import download_market_data, download_fred_data

    prices = download_market_data(output_dir=data_dir)
    macro = download_fred_data(output_dir=data_dir, api_key=fred_key)
    return prices, macro


def step_preprocess(data_dir: str, window_size: int = 60, stride: int = 1):
    """Step 2: Preprocess data into returns and windows."""
    print("\n" + "=" * 60)
    print("STEP 2: Preprocessing data")
    print("=" * 60)
    from src.data.preprocess import prepare_dataset

    dataset = prepare_dataset(data_dir, window_size=window_size, stride=stride)

    os.makedirs(data_dir, exist_ok=True)
    np.save(os.path.join(data_dir, "windows.npy"), dataset["windows"])
    np.save(os.path.join(data_dir, "scaler_mean.npy"), dataset["scaler_params"]["mean"])
    np.save(os.path.join(data_dir, "scaler_std.npy"), dataset["scaler_params"]["std"])
    dataset["returns_df"].to_csv(os.path.join(data_dir, "returns.csv"))
    with open(os.path.join(data_dir, "asset_names.json"), "w") as f:
        json.dump(dataset["asset_names"], f)

    print(f"  Returns: {dataset['returns_df'].shape}")
    print(f"  Windows: {dataset['windows'].shape}")
    print(f"  Assets: {dataset['asset_names']}")
    return dataset


def step_regime_labels(dataset: dict, data_dir: str):
    """Step 3: Compute regime labels and conditioning vectors."""
    print("\n" + "=" * 60)
    print("STEP 3: Computing regime labels")
    print("=" * 60)
    from src.data.regime_labels import prepare_regime_data

    regime_data = prepare_regime_data(
        returns_df=dataset["returns_df"],
        vix_df=dataset.get("vix"),
        macro_df=dataset.get("macro_df"),
        window_dates=dataset["window_dates"],
        data_dir=data_dir,
    )
    return regime_data


def step_train(data_dir: str, models_to_train: list[str],
               epochs: int = DEFAULT_EPOCHS, batch_size: int = DEFAULT_BATCH_SIZE,
               lr: float = DEFAULT_LR, device: str = "cpu"):
    """Step 4: Train all selected models."""
    print("\n" + "=" * 60)
    print("STEP 4: Training models")
    print("=" * 60)

    windows = np.load(os.path.join(data_dir, "windows.npy"))
    n_features = windows.shape[2]
    seq_len = windows.shape[1]

    cond_path = os.path.join(data_dir, "window_cond.npy")
    cond = np.load(cond_path) if os.path.exists(cond_path) else None
    cond_dim = cond.shape[1] if cond is not None else 0

    returns_path = os.path.join(data_dir, "returns.csv")
    import pandas as pd
    returns_flat = pd.read_csv(returns_path, index_col=0).values.astype(np.float32)

    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    trained_models = {}
    training_losses = {}

    for model_name in models_to_train:
        print(f"\n--- Training {model_name.upper()} ---")
        t0 = time.time()

        if model_name == "ddpm":
            from src.models.ddpm import DDPMModel
            is_quick = epochs < 100
            model = DDPMModel(
                n_features=n_features, seq_len=seq_len,
                cond_dim=cond_dim, device=device,
                T=200 if is_quick else 1000,
                base_channels=32 if is_quick else 128,
                channel_mults=(1, 2) if is_quick else (1, 2, 4),
            )
            history = model.train(windows, cond=cond, epochs=epochs, batch_size=batch_size, lr=lr)
            model.save(os.path.join(CHECKPOINTS_DIR, "ddpm.pt"))
            training_losses[model_name] = history["losses"]

        elif model_name == "ddpm_improved":
            from src.models.ddpm_improved import ImprovedDDPM
            is_quick = epochs < 100
            model = ImprovedDDPM(
                n_features=n_features, seq_len=seq_len,
                cond_dim=cond_dim, device=device,
                T=200 if is_quick else 1000,
                base_channels=32 if is_quick else 128,
                channel_mults=(1, 2) if is_quick else (1, 2, 4),
                # Project's strongest configuration from ablation phases.
                use_vpred=True,
                use_student_t_noise=True,
                student_t_df=5.0,
            )
            history = model.train(windows, cond=cond, epochs=epochs, batch_size=batch_size, lr=lr)
            model.save(os.path.join(CHECKPOINTS_DIR, "ddpm_improved.pt"))
            training_losses[model_name] = history["losses"]

        elif model_name == "garch":
            from src.models.garch import GARCHModel
            model = GARCHModel(n_features=n_features, seq_len=seq_len, device=device)
            history = model.train(returns_flat)
            model.save(os.path.join(CHECKPOINTS_DIR, "garch.npz"))

        elif model_name == "vae":
            from src.models.vae import FinancialVAE
            model = FinancialVAE(n_features=n_features, seq_len=seq_len, device=device)
            history = model.train(windows, epochs=epochs, batch_size=batch_size)
            model.save(os.path.join(CHECKPOINTS_DIR, "vae.pt"))
            training_losses[model_name] = history.get("total", [])

        elif model_name == "timegan":
            from src.models.gan import TimeGANModel
            # TimeGAN requires double backward through GRU; run on CPU to avoid CuDNN limitation.
            tgan_device = "cpu"
            if device != "cpu":
                print(f"  TimeGAN forcing device to CPU (requested device was {device})")
            model = TimeGANModel(n_features=n_features, seq_len=seq_len, device=tgan_device)
            tgan_epochs = int(epochs * 1.5)
            history = model.train(windows, epochs=tgan_epochs, batch_size=batch_size)
            model.save(os.path.join(CHECKPOINTS_DIR, "timegan.pt"))
            training_losses[model_name] = history.get("g", [])

        elif model_name == "flow":
            from src.models.normalizing_flow import NormalizingFlowModel
            model = NormalizingFlowModel(n_features=n_features, seq_len=seq_len, device=device)
            history = model.train(windows, epochs=epochs, batch_size=batch_size)
            model.save(os.path.join(CHECKPOINTS_DIR, "flow.pt"))
            training_losses[model_name] = history.get("losses", [])

        else:
            print(f"  Unknown model: {model_name}, skipping")
            continue

        elapsed = time.time() - t0
        print(f"  {model_name.upper()} trained in {elapsed:.1f}s")
        trained_models[model_name] = model

    return trained_models, training_losses


def step_evaluate(trained_models: dict, data_dir: str, n_samples: int = 500):
    """Step 5: Generate synthetic data and run evaluations."""
    print("\n" + "=" * 60)
    print("STEP 5: Evaluation")
    print("=" * 60)

    windows = np.load(os.path.join(data_dir, "windows.npy"))
    from src.evaluation.stylized_facts import run_all_tests, count_passes
    from src.evaluation.metrics import full_evaluation

    results = {}
    for name, model in trained_models.items():
        print(f"\n--- Evaluating {name.upper()} ---")
        try:
            synthetic = model.generate(n_samples)
            sf = run_all_tests(synthetic)
            n_pass = count_passes(sf)
            metrics = full_evaluation(windows[:n_samples], synthetic[:n_samples])

            results[name] = {
                "synthetic": synthetic,
                "stylized_facts": sf,
                "metrics": metrics,
                "n_pass": n_pass,
            }
            print(f"  Stylized facts: {n_pass}/6 passed")
            print(f"  MMD: {metrics['mmd']:.4f}, Disc. score: {metrics['discriminative_score']:.2f}")
        except Exception as e:
            print(f"  ERROR: {e}")
            results[name] = {"error": str(e)}

    return results


def step_dashboard(results: dict, data_dir: str, training_losses: dict):
    """Step 6: Generate comparison dashboard."""
    print("\n" + "=" * 60)
    print("STEP 6: Generating dashboard")
    print("=" * 60)

    windows = np.load(os.path.join(data_dir, "windows.npy"))
    names_path = os.path.join(data_dir, "asset_names.json")
    asset_names = json.load(open(names_path)) if os.path.exists(names_path) else None

    from src.evaluation.visualization import create_comparison_dashboard

    dashboard_input = {}
    for name, info in results.items():
        if "synthetic" in info:
            entry = {
                "synthetic": info["synthetic"],
                "stylized_facts": info.get("stylized_facts", []),
                "metrics": info.get("metrics", {}),
            }
            if name in training_losses:
                entry["losses"] = training_losses[name]
            dashboard_input[name] = entry

    if dashboard_input:
        create_comparison_dashboard(
            windows, dashboard_input,
            asset_names=asset_names,
            save_dir=RESULTS_DIR,
        )


def main():
    parser = argparse.ArgumentParser(description="Run full pipeline")
    parser.add_argument("--skip-download", action="store_true", help="Skip data download")
    parser.add_argument("--models", nargs="+", default=["ddpm", "ddpm_improved", "garch", "vae", "timegan", "flow"],
                        help="Models to train")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--quick", action="store_true", help="Quick run with reduced epochs/samples")
    parser.add_argument("--n-samples", type=int, default=2000)
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--fred-key", default=None)
    args = parser.parse_args()

    if args.quick:
        epochs = args.epochs or 20
        n_samples = min(args.n_samples, 200)
        stride = 10
    else:
        epochs = args.epochs or DEFAULT_EPOCHS
        n_samples = args.n_samples
        stride = 1

    device = DEFAULT_DEVICE
    print(f"Device: {device}")
    set_seed()

    # Step 1: Download
    if not args.skip_download:
        step_download(args.data_dir, args.fred_key)

    # Step 2: Preprocess
    dataset = step_preprocess(args.data_dir, stride=stride)

    # Step 3: Regime labels
    step_regime_labels(dataset, args.data_dir)

    # Step 4: Train
    trained_models, training_losses = step_train(
        args.data_dir, args.models, epochs=epochs, device=device
    )

    # Step 5: Evaluate
    results = step_evaluate(trained_models, args.data_dir, n_samples=n_samples)

    # Step 6: Dashboard
    step_dashboard(results, args.data_dir, training_losses)

    # Summary
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    for name, info in results.items():
        if "n_pass" in info:
            print(f"  {name.upper():12s}  SF: {info['n_pass']}/6  "
                  f"MMD: {info['metrics']['mmd']:.4f}  "
                  f"Disc: {info['metrics']['discriminative_score']:.2f}")
    print(f"\nResults saved to: {RESULTS_DIR}/")
    print(f"Checkpoints saved to: {CHECKPOINTS_DIR}/")
    print(f"To start demo: python -m src.demo.app --data-dir {args.data_dir}")


if __name__ == "__main__":
    main()
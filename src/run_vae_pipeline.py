"""
VAE-only pipeline: same steps as `run_pipeline`, but trains and evaluates
only the improved VAE (`src.models.vae`) and the original VAE (`src.models.vae_origin`).

Usage:
    python -m src.run_vae_pipeline
    python -m src.run_vae_pipeline --skip-download
    python -m src.run_vae_pipeline --models vae              # improved only
    python -m src.run_vae_pipeline --models vae_origin       # original only
    python -m src.run_vae_pipeline --quick
"""

from __future__ import annotations

import os
import sys
import argparse
import time

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.config import (
    DATA_DIR, CHECKPOINTS_DIR, RESULTS_DIR,
    DEFAULT_EPOCHS, DEFAULT_BATCH_SIZE, DEFAULT_DEVICE,
)
from src.data.download import DEFAULT_FRED_KEY

# Shared steps: identical behavior to `run_pipeline`.
from src.run_pipeline import (
    set_seed,
    step_download,
    step_preprocess,
    step_regime_labels,
    step_evaluate,
    step_dashboard,
)

VALID_VAE_MODELS = ("vae", "vae_origin")
DEFAULT_VAE_MODELS = ["vae", "vae_origin"]


def step_train_vae(
    data_dir: str,
    models_to_train: list[str],
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    factor_dim: int = 2,
    teacher_forcing_end: float = 0.3,
    device: str = "cpu",
) -> tuple[dict, dict]:
    """Train only VAE (improved) and/or vae_origin; checkpoints mirror `run_pipeline` for `vae`."""
    print("\n" + "=" * 60)
    print("STEP 4: Training VAE models")
    print("=" * 60)

    windows = np.load(os.path.join(data_dir, "windows.npy"))
    n_features = windows.shape[2]
    seq_len = windows.shape[1]
    cond_path = os.path.join(data_dir, "window_cond.npy")
    cond = np.load(cond_path).astype(np.float32) if os.path.exists(cond_path) else None
    cond_dim = int(cond.shape[1]) if cond is not None else 0

    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    trained_models: dict = {}
    training_losses: dict = {}

    for model_name in models_to_train:
        if model_name not in VALID_VAE_MODELS:
            print(f"  Skipping unknown model (not a VAE pipeline model): {model_name}")
            continue

        print(f"\n--- Training {model_name.upper()} ---")
        t0 = time.time()

        if model_name == "vae":
            from src.models.vae import FinancialVAE

            model = FinancialVAE(
                n_features=n_features,
                seq_len=seq_len,
                cond_dim=cond_dim,
                factor_dim=factor_dim,
                device=device,
            )
            history = model.train(
                windows,
                cond=cond,
                epochs=epochs,
                batch_size=batch_size,
                teacher_forcing_end=teacher_forcing_end,
            )
            model.save(os.path.join(CHECKPOINTS_DIR, "vae.pt"))
            training_losses[model_name] = history.get("total", [])

        else:  # vae_origin
            from src.models.vae_origin import FinancialVAE as FinancialVAEOrigin

            model = FinancialVAEOrigin(
                n_features=n_features, seq_len=seq_len, device=device
            )
            history = model.train(windows, epochs=epochs, batch_size=batch_size)
            model.save(os.path.join(CHECKPOINTS_DIR, "vae_origin.pt"))
            training_losses[model_name] = history.get("total", [])

        elapsed = time.time() - t0
        print(f"  {model_name.upper()} trained in {elapsed:.1f}s")
        trained_models[model_name] = model

    return trained_models, training_losses


def main():
    parser = argparse.ArgumentParser(
        description="Run VAE-only pipeline (improved + original), same as run_pipeline otherwise."
    )
    parser.add_argument("--skip-download", action="store_true", help="Skip data download")
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(DEFAULT_VAE_MODELS),
        help=f"Subset of {list(VALID_VAE_MODELS)}; default: both",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--factor-dim", type=int, default=2, help="Low-rank factor dimension for improved VAE")
    parser.add_argument(
        "--teacher-forcing-end",
        type=float,
        default=0.3,
        help="Final teacher forcing ratio for improved VAE scheduled sampling",
    )
    parser.add_argument(
        "--quick", action="store_true", help="Quick run with reduced epochs/samples"
    )
    parser.add_argument("--n-samples", type=int, default=2000)
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument(
        "--fred-key",
        default=DEFAULT_FRED_KEY,
        help="FRED API key (default: src.data.download.DEFAULT_FRED_KEY)",
    )
    args = parser.parse_args()

    unknown = [m for m in args.models if m not in VALID_VAE_MODELS]
    if unknown:
        raise SystemExit(
            f"Invalid --models: {unknown}. Use only: {list(VALID_VAE_MODELS)}"
        )
    if not args.models:
        args.models = list(DEFAULT_VAE_MODELS)

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

    if not args.skip_download:
        step_download(args.data_dir, args.fred_key)

    dataset = step_preprocess(args.data_dir, stride=stride)
    step_regime_labels(dataset, args.data_dir)

    trained_models, training_losses = step_train_vae(
        args.data_dir,
        args.models,
        epochs=epochs,
        batch_size=DEFAULT_BATCH_SIZE,
        factor_dim=args.factor_dim,
        teacher_forcing_end=args.teacher_forcing_end,
        device=device,
    )
    results = step_evaluate(trained_models, args.data_dir, n_samples=n_samples)
    step_dashboard(results, args.data_dir, training_losses)

    print("\n" + "=" * 60)
    print("VAE PIPELINE COMPLETE")
    print("=" * 60)
    for name, info in results.items():
        if "n_pass" in info:
            print(
                f"  {name.upper():14s}  SF: {info['n_pass']}/6  "
                f"MMD: {info['metrics']['mmd']:.4f}  "
                f"Disc: {info['metrics']['discriminative_score']:.2f}"
            )
    print(f"\nResults saved to: {RESULTS_DIR}/")
    print(f"Checkpoints saved to: {CHECKPOINTS_DIR}/  (vae.pt, vae_origin.pt as trained)")
    print(f"To start demo: python -m src.demo.app --data-dir {args.data_dir}")


if __name__ == "__main__":
    main()

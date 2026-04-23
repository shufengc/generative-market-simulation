"""
VAE-only cross-model evaluation.

Reuses `run_cross_model_analysis` from `cross_model_analysis.py` (same metrics:
bootstrap CIs, pairwise significance, per-regime breakdown, Borda ranking,
radar chart, temporal coherence, proposal criteria, etc.) but only loads
synthetic data from:

  - ``vae.pt``          → improved ``FinancialVAE`` (``src.models.vae``)
  - ``vae_origin.pt`` → original ``FinancialVAE`` (``src.models.vae_origin``)

Run after ``python -m src.run_vae_pipeline`` (or equivalent training).

Usage:
    python -m src.evaluation.vae_cross_model_analysis
    python -m src.evaluation.vae_cross_model_analysis --n-samples 800 --n-bootstrap 40
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.config import CHECKPOINTS_DIR, DATA_DIR, DEFAULT_DEVICE, RESULTS_DIR


def _vae_addendum_text(model_metrics: dict[str, dict]) -> str:
    """Replace generic NormFlow narrative when only VAE checkpoints are compared."""
    lines = [
        "=" * 70,
        "VAE-ONLY COMPARISON NOTE",
        "=" * 70,
        "",
        "Models:",
        "  VAE_improved  — src.models.vae.FinancialVAE  (checkpoint: vae.pt)",
        "  VAE_original  — src.models.vae_origin.FinancialVAE  (checkpoint: vae_origin.pt)",
        "",
        "Outputs (radar, CIs, regime heatmaps, ranking, report) use the same",
        "layout as src.evaluation.cross_model_analysis.run_cross_model_analysis.",
        "The generic NormFlow narrative is omitted for this two-VAE run.",
        "",
        "Quick metric snapshot:",
        "-" * 50,
    ]
    for name in sorted(model_metrics.keys()):
        m = model_metrics[name]
        lines.append(f"  {name}")
        for k in (
            "sf_passed",
            "mmd",
            "discriminative_score",
            "proposal_criteria_passed",
            "sig_w1",
        ):
            v = m.get(k)
            if v is None:
                continue
            lines.append(f"    {k}: {v}")
        lines.append("")
    lines.append("Full detail: cross_model_report.txt and cross_model_metrics.json in save_dir.")
    lines.append("=" * 70)
    return "\n".join(lines)


def load_vae_pair_synthetics(
    data_dir: str,
    checkpoints_dir: str,
    n_features: int,
    seq_len: int,
    n_samples: int,
    device: str,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Load both VAE checkpoints, generate windows, record generation time per model."""
    model_synthetics: dict[str, np.ndarray] = {}
    timing: dict[str, float] = {}

    specs: list[tuple[str, str, str]] = [
        ("VAE_improved", "vae.pt", "vae"),
        ("VAE_original", "vae_origin.pt", "vae_origin"),
    ]

    for display_name, ckpt_file, kind in specs:
        ckpt_path = os.path.join(checkpoints_dir, ckpt_file)
        if not os.path.isfile(ckpt_path):
            print(f"  Skipping {display_name} (missing {ckpt_path})")
            continue
        try:
            t0 = time.time()
            if kind == "vae":
                from src.models.vae import FinancialVAE

                model = FinancialVAE(
                    n_features=n_features, seq_len=seq_len, device=device
                )
                model.load(ckpt_path)
                syn = model.generate(n_samples)
            else:
                from src.models.vae_origin import FinancialVAE as FinancialVAEOrigin

                model = FinancialVAEOrigin(
                    n_features=n_features, seq_len=seq_len, device=device
                )
                model.load(ckpt_path)
                syn = model.generate(n_samples)
            elapsed = time.time() - t0
            model_synthetics[display_name] = syn
            timing[display_name] = elapsed
            print(f"  {display_name}: generated {syn.shape} in {elapsed:.1f}s")
        except Exception as e:
            print(f"  ERROR loading {display_name}: {e}")

    return model_synthetics, timing


def run_vae_cross_model_analysis(
    data_dir: str | None = None,
    checkpoints_dir: str | None = None,
    save_dir: str | None = None,
    n_samples: int = 500,
    n_bootstrap: int = 50,
    device: str | None = None,
) -> dict[str, Any]:
    """Load real windows + both VAE checkpoints, then delegate to ``run_cross_model_analysis``."""
    from src.evaluation.cross_model_analysis import run_cross_model_analysis

    data_dir = data_dir or DATA_DIR
    checkpoints_dir = checkpoints_dir or CHECKPOINTS_DIR
    save_dir = save_dir or os.path.join(RESULTS_DIR, "vae_cross_model")
    device = device or DEFAULT_DEVICE

    windows = np.load(os.path.join(data_dir, "windows.npy"))
    n_features = int(windows.shape[2])
    seq_len = int(windows.shape[1])

    regime_path = os.path.join(data_dir, "window_regimes.npy")
    regimes = np.load(regime_path) if os.path.exists(regime_path) else None

    print("\n" + "=" * 60)
    print("VAE CROSS-MODEL ANALYSIS (engine: cross_model_analysis)")
    print("=" * 60)
    print("\nLoading checkpoints and generating synthetic windows ...")

    model_synthetics, timing = load_vae_pair_synthetics(
        data_dir, checkpoints_dir, n_features, seq_len, n_samples, device
    )

    if not model_synthetics:
        raise RuntimeError(
            "No VAE checkpoints produced synthetic data. Train both models first "
            f"(expected {os.path.join(checkpoints_dir, 'vae.pt')} and "
            f"{os.path.join(checkpoints_dir, 'vae_origin.pt')})."
        )

    out = run_cross_model_analysis(
        real_windows=windows,
        model_synthetics=model_synthetics,
        window_regimes=regimes,
        timing=timing,
        n_bootstrap=n_bootstrap,
        save_dir=save_dir,
    )

    if save_dir and "metrics" in out:
        note_path = os.path.join(save_dir, "normflow_analysis.txt")
        with open(note_path, "w", encoding="utf-8") as f:
            f.write(_vae_addendum_text(out["metrics"]))
        print(f"\n  Wrote VAE-specific note to {note_path}")

    return out


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Cross-model style evaluation for improved vs original VAE only."
    )
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--checkpoints-dir", default=None)
    parser.add_argument("--save-dir", default=None)
    parser.add_argument("--n-samples", type=int, default=500)
    parser.add_argument("--n-bootstrap", type=int, default=50)
    args = parser.parse_args()

    run_vae_cross_model_analysis(
        data_dir=args.data_dir,
        checkpoints_dir=args.checkpoints_dir,
        save_dir=args.save_dir,
        n_samples=args.n_samples,
        n_bootstrap=args.n_bootstrap,
    )


if __name__ == "__main__":
    main()

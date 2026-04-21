"""
DDPM Ablation Study -- train and evaluate all model variants.

Usage:
    PYTHONPATH=. python experiments/run_ddpm_ablation.py
    PYTHONPATH=. python experiments/run_ddpm_ablation.py --quick   # fast test
    PYTHONPATH=. python experiments/run_ddpm_ablation.py --models baseline dit vpred
"""

from __future__ import annotations

import os
import sys
import json
import time
import argparse

import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.config import DATA_DIR, DEFAULT_DEVICE, COND_DIM
from src.models.ddpm_improved import ImprovedDDPM
from src.evaluation.stylized_facts import run_all_tests, count_passes
from src.evaluation.metrics import full_evaluation

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


MODEL_CONFIGS = {
    # --- Phase 1 variants ---
    "baseline": dict(
        use_dit=False, use_vpred=False, use_self_cond=False,
        use_sigmoid_schedule=False, use_cross_attn=False,
    ),
    "dit": dict(
        use_dit=True, use_vpred=False, use_self_cond=False,
        use_sigmoid_schedule=False, use_cross_attn=False,
    ),
    "vpred": dict(
        use_dit=False, use_vpred=True, use_self_cond=False,
        use_sigmoid_schedule=False, use_cross_attn=False,
    ),
    "selfcond": dict(
        use_dit=False, use_vpred=False, use_self_cond=True,
        use_sigmoid_schedule=False, use_cross_attn=False,
    ),
    "sigmoid": dict(
        use_dit=False, use_vpred=False, use_self_cond=False,
        use_sigmoid_schedule=True, use_cross_attn=False,
    ),
    "crossattn": dict(
        use_dit=False, use_vpred=False, use_self_cond=False,
        use_sigmoid_schedule=False, use_cross_attn=True,
    ),
    # --- Phase 2 variants (all build on vpred+sigmoid as Phase 1 winner) ---
    "temporal_attn": dict(
        use_vpred=True, use_sigmoid_schedule=True,
        use_temporal_attn=True,
    ),
    "hetero_noise": dict(
        use_vpred=True, use_sigmoid_schedule=True,
        use_hetero_noise=True,
    ),
    "aux_sf_loss": dict(
        use_vpred=True, use_sigmoid_schedule=True,
        use_aux_sf_loss=True,
    ),
    "phase2_combo": dict(
        use_vpred=True, use_sigmoid_schedule=True,
        use_temporal_attn=True,
        use_hetero_noise=True,
        use_aux_sf_loss=True,
    ),
    # --- Phase 3: fair comparison at uniform 128ch/400ep ---
    "p3_baseline": dict(),
    "p3_vpred": dict(use_vpred=True),
    "p3_sigmoid": dict(use_sigmoid_schedule=True),
    "p3_vpred_sigmoid": dict(use_vpred=True, use_sigmoid_schedule=True),
    "p3_vpred_auxsf": dict(use_vpred=True, use_aux_sf_loss=True),
    "p3_vpred_auxsf_hi": dict(use_vpred=True, use_aux_sf_loss=True, aux_sf_weight=0.5),
    "p3_vpred_sigmoid_auxsf": dict(use_vpred=True, use_sigmoid_schedule=True, use_aux_sf_loss=True),
    "p3_vpred_auxsf_800ep": dict(use_vpred=True, use_aux_sf_loss=True),
    # --- Phase 4: parameter-fair test at 64ch/400ep ---
    "p4_baseline_64ch": dict(),
    "p4_vpred_64ch": dict(use_vpred=True),
    # --- Phase 5: innovations (solo + vpred combos) ---
    "p5_wavelet_only": dict(use_wavelet=True),
    "p5_studentt_only": dict(use_student_t_noise=True, student_t_df=5.0),
    "p5_vpred_acfguide": dict(use_vpred=True, use_acf_guidance=True),
    "p5_vpred_wavelet": dict(use_vpred=True, use_wavelet=True),
    "p5_vpred_studentt": dict(use_vpred=True, use_student_t_noise=True, student_t_df=5.0),
}

NORMFLOW_MODELS = {"p3_normflow"}


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_normflow(model_name: str, windows: np.ndarray,
                 real_windows: np.ndarray, seed: int, epochs: int,
                 batch_size: int, lr: float, n_gen: int,
                 device: str) -> dict:
    """Train and evaluate NormFlow for fair cross-model comparison."""
    from src.models.normalizing_flow import NormalizingFlowModel

    set_seed(seed)
    n_features = windows.shape[2]
    seq_len = windows.shape[1]

    print(f"\n{'='*60}")
    print(f"Model: {model_name} (NormFlow) | Seed: {seed} | Epochs: {epochs}")
    print(f"{'='*60}")

    model = NormalizingFlowModel(
        n_features=n_features, seq_len=seq_len,
        hidden_dim=256, n_flow_layers=6, device=device,
    )

    t0 = time.time()
    history = model.train(windows, epochs=epochs, batch_size=batch_size, lr=1e-4)
    train_time = time.time() - t0

    print(f"  Training took {train_time:.1f}s")
    print(f"  Generating {n_gen} samples...")

    t1 = time.time()
    synthetic = model.generate(n_gen, seq_len)
    gen_time = time.time() - t1
    print(f"  Generation took {gen_time:.1f}s")

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


def run_single(model_name: str, config: dict, windows: np.ndarray,
               cond: np.ndarray | None, real_windows: np.ndarray,
               seed: int, epochs: int, batch_size: int, lr: float,
               n_gen: int, device: str, quick: bool,
               base_channels: int = 64) -> dict:
    """Train one model variant with one seed and return results."""
    set_seed(seed)
    n_features = windows.shape[2]
    seq_len = windows.shape[1]
    cond_dim = cond.shape[1] if cond is not None else 0

    T_val = 200 if quick else 1000
    base_ch = 32 if quick else base_channels
    ch_mults = (1, 2) if quick else (1, 2, 4)

    print(f"\n{'='*60}")
    print(f"Model: {model_name} | Seed: {seed} | Epochs: {epochs}")
    print(f"{'='*60}")

    model = ImprovedDDPM(
        n_features=n_features,
        seq_len=seq_len,
        T=T_val,
        base_channels=base_ch,
        channel_mults=ch_mults,
        cond_dim=cond_dim,
        device=device,
        dit_d_model=128 if quick else 256,
        dit_n_heads=4 if quick else 8,
        dit_n_layers=3 if quick else 6,
        **config,
    )

    t0 = time.time()
    history = model.train(windows, cond=cond, epochs=epochs,
                          batch_size=batch_size, lr=lr)
    train_time = time.time() - t0

    print(f"  Training took {train_time:.1f}s")
    print(f"  Generating {n_gen} samples...")

    t1 = time.time()
    synthetic = model.generate(n_gen, seq_len)
    gen_time = time.time() - t1
    print(f"  Generation took {gen_time:.1f}s")

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
        "losses": history["losses"],
    }

    print(f"  Result: SF={n_pass}/6 | MMD={metrics['mmd']:.4f} | "
          f"Disc={metrics['discriminative_score']:.2f} | "
          f"W1={metrics['wasserstein_1d']:.4f} | "
          f"CorrDist={metrics['correlation_matrix_distance']:.4f}")

    return result


def main():
    parser = argparse.ArgumentParser(description="DDPM Ablation Study")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Model variants to run (default: all)")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--n-gen", type=int, default=1000)
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: fewer epochs, smaller model")
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--out-dir", default=RESULTS_DIR,
                        help="Output directory for results JSON")
    parser.add_argument("--base-channels", type=int, default=64,
                        help="Base channel width (64 for Phase 1, 128 for Phase 2)")
    args = parser.parse_args()

    if args.quick:
        args.epochs = args.epochs if args.epochs != 200 else 30
        args.n_gen = min(args.n_gen, 200)

    all_known = set(MODEL_CONFIGS.keys()) | NORMFLOW_MODELS
    models_to_run = args.models or list(MODEL_CONFIGS.keys())
    print(f"Device: {args.device}")
    print(f"Models: {models_to_run}")
    print(f"Seeds: {args.seeds}")
    print(f"Epochs: {args.epochs}")
    print(f"Base channels: {args.base_channels}")

    windows = np.load(os.path.join(args.data_dir, "windows.npy"))
    cond_path = os.path.join(args.data_dir, "window_cond.npy")
    cond = np.load(cond_path) if os.path.exists(cond_path) else None

    print(f"Data: {windows.shape[0]} windows, {windows.shape[1]} steps, "
          f"{windows.shape[2]} features")
    if cond is not None:
        print(f"Conditioning: {cond.shape}")

    # Load existing results if resuming into the same output directory
    out_path = os.path.join(args.out_dir, "ablation_results.json")
    all_results = []
    if os.path.exists(out_path):
        with open(out_path) as f:
            all_results = json.load(f)
        existing = {(r["model"], r["seed"]) for r in all_results}
        print(f"Resuming: {len(all_results)} existing runs loaded")
    else:
        existing = set()

    for model_name in models_to_run:
        if model_name not in all_known:
            print(f"Unknown model: {model_name}, skipping")
            continue

        for seed in args.seeds:
            if (model_name, seed) in existing:
                print(f"  Skipping {model_name} seed={seed} (already done)")
                continue

            if model_name in NORMFLOW_MODELS:
                result = run_normflow(
                    model_name=model_name,
                    windows=windows,
                    real_windows=windows,
                    seed=seed,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    n_gen=args.n_gen,
                    device=args.device,
                )
            else:
                config = MODEL_CONFIGS[model_name]
                result = run_single(
                    model_name=model_name,
                    config=config,
                    windows=windows,
                    cond=cond,
                    real_windows=windows,
                    seed=seed,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    n_gen=args.n_gen,
                    device=args.device,
                    quick=args.quick,
                    base_channels=args.base_channels,
                )
            all_results.append(result)

            os.makedirs(args.out_dir, exist_ok=True)
            with open(out_path, "w") as f:
                json.dump(all_results, f, indent=2, default=str)
            print(f"  Saved intermediate results -> {out_path}")

    print(f"\n{'='*60}")
    print("ALL EXPERIMENTS COMPLETE")
    print(f"{'='*60}")
    print(f"Results: {out_path}")

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    from collections import defaultdict
    by_model = defaultdict(list)
    for r in all_results:
        by_model[r["model"]].append(r)

    for name, runs in by_model.items():
        sfs = [r["n_pass"] for r in runs]
        mmds = [r["mmd"] for r in runs]
        discs = [r["discriminative_score"] for r in runs]
        print(f"  {name:25s}  SF={np.mean(sfs):.1f}+-{np.std(sfs):.1f}  "
              f"MMD={np.mean(mmds):.4f}+-{np.std(mmds):.4f}  "
              f"Disc={np.mean(discs):.2f}+-{np.std(discs):.2f}")


if __name__ == "__main__":
    main()

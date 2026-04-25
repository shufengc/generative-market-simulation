"""
Phase 7 — patch-based training + evaluation for the strongest ImprovedDDPM (vpred + Student-t).

Pipeline (default): download data → preprocess (stride=1) + regime labels → train each
(patch_stride × seed) → evaluate each checkpoint (stylized facts + distributional metrics).

CLI:
  Default: re-download, re-train, evaluate.
  ``--skip-download``: skip download only (still preprocesses + regime unless eval-only + skip-preprocess).
  ``--eval-only`` / ``--skip-train``: skip training; evaluate existing ``ddpm_improved.pt`` under ``--out-dir``.
  ``--skip-preprocess`` (requires eval-only): use existing ``windows.npy`` / ``window_cond.npy`` without refreshing.

Patch-based training (Rank 8 in docs/04-24-ddpm-phase7-improvement-ideas.md):
  patch_stride == 1: use all windows.
  patch_stride  > 1: random subset of size max(2*batch_size, n // patch_stride) without replacement.

Outputs:
  - checkpoints: checkpoints/phase7_patch_training/
  - metrics/json/plots: experiments/results/phase7_patch_training/

Usage:
  PYTHONPATH=. python experiments/run_phase7_patch_training.py
  PYTHONPATH=. python experiments/run_phase7_patch_training.py --quick
  PYTHONPATH=. python experiments/run_phase7_patch_training.py --skip-download
  PYTHONPATH=. python experiments/run_phase7_patch_training.py --eval-only --skip-download
  PYTHONPATH=. python experiments/run_phase7_patch_training.py --eval-only --skip-download \\
      --skip-preprocess
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.download import DEFAULT_FRED_KEY
from src.evaluation.metrics import full_evaluation
from src.evaluation.stylized_facts import count_passes, run_all_tests
from src.models.ddpm_improved import ImprovedDDPM
from src.run_pipeline import set_seed, step_download, step_preprocess, step_regime_labels
from src.utils.config import CHECKPOINTS_DIR, DATA_DIR, DEFAULT_DEVICE, DEFAULT_EPOCHS


PATCH_STRIDES = [1, 2, 3, 5, 10, 20]
DEFAULT_SEEDS = [42, 123, 456]

DEFAULT_OUT_DIR = os.path.join(
    PROJECT_ROOT, "experiments", "results", "phase7_patch_training"
)
DEFAULT_CKPT_DIR = os.path.join(CHECKPOINTS_DIR, "phase7_patch_training")


def _json_default(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    return str(v)


def _infer_channel_mults(net_state: dict, base_channels: int) -> tuple[int, ...]:
    mults: list[int] = []
    idx = 0
    while True:
        key = f"down_blocks.{idx}.block1.2.weight"
        if key not in net_state:
            break
        out_ch = int(net_state[key].shape[0])
        mults.append(max(out_ch // base_channels, 1))
        idx += 1
    return tuple(mults) if mults else (1, 2, 4)


def _infer_dit_layers(net_state: dict) -> int:
    layer_ids = set()
    for key in net_state:
        m = re.match(r"blocks\.(\d+)\.", key)
        if m:
            layer_ids.add(int(m.group(1)))
    return max(layer_ids) + 1 if layer_ids else 6


def load_improved_ddpm_from_ckpt(ckpt_path: str, device: str) -> ImprovedDDPM:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    net_state = ckpt["net_state"]
    use_dit = bool(cfg.get("use_dit", False))

    kwargs = dict(
        n_features=int(cfg.get("n_features", 16)),
        seq_len=int(cfg.get("seq_len", 60)),
        T=int(cfg.get("T", 1000)),
        cond_dim=int(cfg.get("cond_dim", 0)),
        cfg_drop_prob=float(cfg.get("cfg_drop_prob", 0.1)),
        device=device,
        use_dit=use_dit,
        use_vpred=bool(cfg.get("use_vpred", False)),
        use_self_cond=bool(cfg.get("use_self_cond", False)),
        use_sigmoid_schedule=bool(cfg.get("use_sigmoid_schedule", False)),
        use_cross_attn=bool(cfg.get("use_cross_attn", False)),
        use_temporal_attn=bool(cfg.get("use_temporal_attn", False)),
        use_hetero_noise=bool(cfg.get("use_hetero_noise", False)),
        use_aux_sf_loss=bool(cfg.get("use_aux_sf_loss", False)),
        use_acf_guidance=bool(cfg.get("use_acf_guidance", False)),
        use_wavelet=bool(cfg.get("use_wavelet", False)),
        use_student_t_noise=bool(cfg.get("use_student_t_noise", False)),
        student_t_df=float(cfg.get("student_t_df", 5.0)),
    )

    if use_dit:
        kwargs["dit_d_model"] = int(net_state["input_proj.weight"].shape[0])
        kwargs["dit_n_layers"] = _infer_dit_layers(net_state)
    else:
        base_channels = int(net_state["init_conv.weight"].shape[0])
        kwargs["base_channels"] = base_channels
        kwargs["channel_mults"] = _infer_channel_mults(net_state, base_channels)

    model = ImprovedDDPM(**kwargs)
    model.load(ckpt_path)
    return model


def build_patch_indices(
    n_windows: int,
    patch_stride: int,
    seed: int,
    batch_size: int,
) -> np.ndarray:
    if patch_stride <= 1:
        return np.arange(n_windows, dtype=np.int64)

    n_sub = max(2 * batch_size, n_windows // patch_stride)
    n_sub = min(n_sub, n_windows)
    rng = np.random.default_rng(int(seed))
    return rng.choice(n_windows, size=n_sub, replace=False)


def load_training_arrays(data_dir: str) -> tuple[np.ndarray, np.ndarray | None]:
    windows_path = os.path.join(data_dir, "windows.npy")
    if not os.path.exists(windows_path):
        raise FileNotFoundError(
            f"Missing {windows_path}. Run without --eval-only first, or preprocess stride=1."
        )
    windows = np.load(windows_path)
    cond_path = os.path.join(data_dir, "window_cond.npy")
    cond = np.load(cond_path) if os.path.exists(cond_path) else None
    if cond is not None and len(cond) != len(windows):
        raise ValueError(
            f"window_cond.npy length {len(cond)} != windows.npy length {len(windows)}"
        )
    return windows, cond


def sample_cond_if_available(data_dir: str, n_samples: int) -> np.ndarray | None:
    cond_path = os.path.join(data_dir, "window_cond.npy")
    if not os.path.exists(cond_path):
        return None
    cond = np.load(cond_path)
    if cond.ndim != 2 or len(cond) == 0:
        return None
    idx = np.random.choice(len(cond), size=n_samples, replace=True)
    return cond[idx]


def evaluate_checkpoint(
    *,
    ckpt_path: str,
    windows_full: np.ndarray,
    data_dir: str,
    device: str,
    n_gen: int,
    seed: int,
    run_sub: str,
) -> dict:
    set_seed(seed)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(ckpt_path)

    model = load_improved_ddpm_from_ckpt(ckpt_path, device)
    n_gen_eff = min(int(n_gen), len(windows_full))
    cond_eval = sample_cond_if_available(data_dir, n_gen_eff)

    t0 = time.time()
    synthetic = model.generate(n_gen_eff, cond=cond_eval)
    gen_time = time.time() - t0

    real_eval = windows_full[:n_gen_eff]
    synthetic = synthetic[: len(real_eval)]

    sf_results = run_all_tests(synthetic)
    n_pass = count_passes(sf_results)
    metrics = full_evaluation(real_eval, synthetic)

    sf_details = {}
    sf_pass_vector: dict[str, int] = {}
    for i, r in enumerate(sf_results, start=1):
        name = r.get("name", "unknown")
        passed = bool(r.get("pass", False))
        sf_details[name] = {
            "pass": passed,
            **{k: v for k, v in r.items() if k not in ("name", "pass", "acf_values")},
        }
        sf_pass_vector[f"sf{i}_pass"] = int(passed)

    eval_out = {
        "n_gen": n_gen_eff,
        "gen_time_s": round(gen_time, 3),
        "n_pass": int(n_pass),
        "sf_pass_vector": sf_pass_vector,
        "stylized_facts": sf_details,
        "mmd": float(metrics["mmd"]),
        "wasserstein_1d": float(metrics["wasserstein_1d"]),
        "discriminative_score": float(metrics["discriminative_score"]),
        "correlation_matrix_distance": float(metrics["correlation_matrix_distance"]),
        "ks_stat": float(metrics["ks_stat"]),
        "sig_w1": float(metrics.get("sig_w1", np.nan)),
        "moment_mean_diff": float(metrics.get("moments", {}).get("mean_diff", np.nan)),
        "moment_std_diff": float(metrics.get("moments", {}).get("std_diff", np.nan)),
        "moment_skew_diff": float(metrics.get("moments", {}).get("skew_diff", np.nan)),
        "moment_kurt_diff": float(metrics.get("moments", {}).get("kurt_diff", np.nan)),
    }

    os.makedirs(run_sub, exist_ok=True)
    with open(os.path.join(run_sub, "eval_results.json"), "w", encoding="utf-8") as f:
        json.dump(eval_out, f, indent=2, default=_json_default)

    print(
        f"    eval: SF={n_pass}/6 | MMD={metrics['mmd']:.4f} | "
        f"Disc={metrics['discriminative_score']:.2f} | {gen_time:.1f}s gen"
    )
    return eval_out


def prepare_data_pipeline(data_dir: str, fred_key: str | None, skip_download: bool) -> None:
    if not skip_download:
        step_download(data_dir, fred_key)
    dataset = step_preprocess(data_dir, stride=1)
    step_regime_labels(dataset, data_dir)


def run_result_subdir(out_dir: str, patch_stride: int, seed: int) -> str:
    return os.path.join(out_dir, f"patch_stride_{int(patch_stride)}", f"seed_{int(seed)}")


def run_checkpoint_path(ckpt_dir: str, patch_stride: int, seed: int) -> str:
    return os.path.join(
        ckpt_dir,
        f"patch_stride_{int(patch_stride)}",
        f"seed_{int(seed)}",
        "ddpm_improved.pt",
    )


def resolve_checkpoint_path(
    *,
    ckpt_dir: str,
    out_dir: str,
    patch_stride: int,
    seed: int,
) -> str | None:
    preferred = run_checkpoint_path(ckpt_dir, patch_stride, seed)
    if os.path.isfile(preferred):
        return preferred
    # Backward compatibility for older runs that saved checkpoints under out_dir.
    legacy = os.path.join(
        run_result_subdir(out_dir, patch_stride, seed),
        "ddpm_improved.pt",
    )
    if os.path.isfile(legacy):
        return legacy
    return None


def save_visualizations(rows: list[dict], out_dir: str) -> None:
    records: list[dict] = []
    for row in rows:
        eval_info = row.get("eval") or {}
        records.append(
            {
                "patch_stride": row["patch_stride"],
                "seed": row["seed"],
                "sf_passed": eval_info.get("n_pass"),
                "mmd": eval_info.get("mmd"),
                "discriminative_score": eval_info.get("discriminative_score"),
                "wasserstein_1d": eval_info.get("wasserstein_1d"),
                "correlation_matrix_distance": eval_info.get("correlation_matrix_distance"),
                "ks_stat": eval_info.get("ks_stat"),
                "sig_w1": eval_info.get("sig_w1"),
                "moment_mean_diff": eval_info.get("moment_mean_diff"),
                "moment_std_diff": eval_info.get("moment_std_diff"),
                "moment_skew_diff": eval_info.get("moment_skew_diff"),
                "moment_kurt_diff": eval_info.get("moment_kurt_diff"),
                **(eval_info.get("sf_pass_vector") or {}),
            }
        )
    df = pd.DataFrame(records)
    if df.empty:
        return
    df.to_csv(os.path.join(out_dir, "summary_metrics.csv"), index=False)
    agg = (
        df.groupby("patch_stride", as_index=False)
        .agg(
            sf_passed_mean=("sf_passed", "mean"),
            mmd_mean=("mmd", "mean"),
            disc_mean=("discriminative_score", "mean"),
            w1_mean=("wasserstein_1d", "mean"),
            corr_dist_mean=("correlation_matrix_distance", "mean"),
            ks_mean=("ks_stat", "mean"),
        )
        .sort_values("patch_stride")
    )
    agg.to_csv(os.path.join(out_dir, "summary_by_patch_stride.csv"), index=False)

    sf_cols = sorted([c for c in df.columns if c.startswith("sf") and c.endswith("_pass")])
    if sf_cols:
        sf_rate = df.groupby("patch_stride", as_index=False)[sf_cols].mean().sort_values("patch_stride")
        sf_rate.to_csv(os.path.join(out_dir, "sf_pass_rate_by_patch_stride.csv"), index=False)

    plots = [
        ("sf_passed", "SF Passed", "sf_passed_vs_patch_stride.png"),
        ("mmd", "MMD", "mmd_vs_patch_stride.png"),
        ("discriminative_score", "Discriminative Score", "disc_vs_patch_stride.png"),
    ]
    for col, title, fn in plots:
        if col not in df.columns or df[col].dropna().empty:
            continue
        plt.figure(figsize=(8, 5))
        sns.lineplot(
            data=df.sort_values("patch_stride"),
            x="patch_stride",
            y=col,
            marker="o",
            estimator="mean",
            errorbar=("sd", 1),
        )
        plt.title(f"{title} vs Patch Stride", fontweight="bold")
        plt.xlabel("Patch Stride")
        plt.ylabel(title)
        plt.grid(alpha=0.2)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, fn), dpi=180, bbox_inches="tight")
        plt.close()

    heat = df.pivot(index="seed", columns="patch_stride", values="mmd")
    if not heat.empty:
        plt.figure(figsize=(10, 4))
        sns.heatmap(heat, annot=True, fmt=".4f", cmap="YlOrRd", linewidths=0.4)
        plt.title("MMD Heatmap (seed x patch_stride)", fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "mmd_heatmap.png"), dpi=180, bbox_inches="tight")
        plt.close()

    if sf_cols:
        sf_heat = (
            df.groupby("patch_stride", as_index=False)[sf_cols]
            .mean()
            .set_index("patch_stride")
            .T
        )
        plt.figure(figsize=(10, 4))
        sns.heatmap(sf_heat, annot=True, fmt=".2f", cmap="YlGnBu", linewidths=0.4, vmin=0.0, vmax=1.0)
        plt.title("SF Pass Rate Heatmap (fact x patch_stride)", fontweight="bold")
        plt.xlabel("Patch Stride")
        plt.ylabel("Stylized Fact")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "sf_pass_rate_heatmap.png"), dpi=180, bbox_inches="tight")
        plt.close()


def train_one(
    *,
    windows_full: np.ndarray,
    cond_full: np.ndarray | None,
    patch_stride: int,
    seed: int,
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
    out_dir: str,
    ckpt_dir: str,
    quick: bool,
) -> dict:
    set_seed(seed)
    n = len(windows_full)
    idx = build_patch_indices(n, patch_stride, seed, batch_size)
    w_train = windows_full[idx]
    c_train = cond_full[idx] if cond_full is not None else None

    n_features = w_train.shape[2]
    seq_len = w_train.shape[1]
    cond_dim = c_train.shape[1] if c_train is not None else 0

    is_quick = quick or epochs < 100
    model = ImprovedDDPM(
        n_features=n_features,
        seq_len=seq_len,
        cond_dim=cond_dim,
        device=device,
        T=200 if is_quick else 1000,
        base_channels=32 if is_quick else 128,
        channel_mults=(1, 2) if is_quick else (1, 2, 4),
        use_vpred=True,
        use_student_t_noise=True,
        student_t_df=5.0,
    )

    t0 = time.time()
    history = model.train(
        w_train,
        cond=c_train,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
    )
    elapsed = time.time() - t0

    run_sub = run_result_subdir(out_dir, patch_stride, seed)
    os.makedirs(run_sub, exist_ok=True)
    ckpt_path = run_checkpoint_path(ckpt_dir, patch_stride, seed)
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    model.save(ckpt_path)

    train_cfg = {
        "phase": "phase7_patch_training",
        "patch_stride": patch_stride,
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "device": device,
        "quick": is_quick,
        "n_windows_full_pool": int(n),
        "n_windows_train": int(len(w_train)),
        "patch_index_min": int(idx.min()),
        "patch_index_max": int(idx.max()),
        "checkpoint": ckpt_path,
        "train_time_s": round(elapsed, 2),
    }
    with open(os.path.join(run_sub, "train_config.json"), "w", encoding="utf-8") as f:
        json.dump(train_cfg, f, indent=2)

    with open(os.path.join(run_sub, "losses.json"), "w", encoding="utf-8") as f:
        json.dump({"losses": history.get("losses", [])}, f, indent=2, default=_json_default)

    print(
        f"  train: patch_stride={patch_stride} seed={seed} | "
        f"n_train={len(w_train)} | {elapsed:.1f}s -> {ckpt_path}"
    )
    return train_cfg


def main():
    parser = argparse.ArgumentParser(
        description="Phase 7 patch-based DDPM: download, train, evaluate (or eval-only)"
    )
    parser.add_argument("--data-dir", default=DATA_DIR, help="Project data directory")
    parser.add_argument(
        "--out-dir",
        default=DEFAULT_OUT_DIR,
        help="Root folder for metrics JSON and plots",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=DEFAULT_CKPT_DIR,
        help="Root folder to save/load trained checkpoints",
    )
    parser.add_argument(
        "--patch-strides",
        type=int,
        nargs="+",
        default=PATCH_STRIDES,
        help="Patch strides for subsampled training",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=DEFAULT_SEEDS,
        help="Random seeds (patch indices + torch/np + generation cond sampling)",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--n-gen", type=int, default=1000, help="Samples for evaluation")
    parser.add_argument("--quick", action="store_true", help="Small model + T=200 + default 30 epochs")
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip market/macro download (still runs preprocess + regime by default)",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip training; load checkpoints under --out-dir and run evaluation only",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Alias for --eval-only (no training, evaluate existing checkpoints)",
    )
    parser.add_argument(
        "--skip-preprocess",
        action="store_true",
        help="Only with --eval-only: skip preprocess/regime; require windows.npy (+ window_cond) on disk",
    )
    parser.add_argument(
        "--fred-key",
        default=DEFAULT_FRED_KEY,
        help="FRED API key for download step",
    )
    args = parser.parse_args()

    eval_only = bool(args.eval_only or args.skip_train)

    if args.skip_preprocess and not eval_only:
        parser.error("--skip-preprocess requires --eval-only (or --skip-train)")

    if args.quick:
        epochs = args.epochs or 30
    else:
        epochs = args.epochs or DEFAULT_EPOCHS

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    if eval_only:
        if not args.skip_preprocess:
            if not args.skip_download:
                step_download(args.data_dir, args.fred_key)
            dataset = step_preprocess(args.data_dir, stride=1)
            step_regime_labels(dataset, args.data_dir)
        # else: require windows on disk
    else:
        prepare_data_pipeline(args.data_dir, args.fred_key, skip_download=args.skip_download)

    windows_full, cond_full = load_training_arrays(args.data_dir)

    run_meta = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "data_dir": os.path.abspath(args.data_dir),
        "out_dir": os.path.abspath(args.out_dir),
        "checkpoint_dir": os.path.abspath(args.checkpoint_dir),
        "patch_strides": list(args.patch_strides),
        "seeds": list(args.seeds),
        "epochs": epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "device": args.device,
        "n_gen": args.n_gen,
        "quick": bool(args.quick),
        "skip_download": bool(args.skip_download),
        "eval_only": eval_only,
        "skip_train_flag": bool(args.skip_train),
        "skip_preprocess": bool(args.skip_preprocess),
        "windows_shape": list(windows_full.shape),
    }
    with open(os.path.join(args.out_dir, "run_meta.json"), "w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2)

    print("Phase 7 patch-based pipeline")
    print(f"  data_dir={args.data_dir}")
    print(f"  out_dir={args.out_dir}")
    print(f"  checkpoint_dir={args.checkpoint_dir}")
    print(
        f"  download={'skip' if args.skip_download else 'on'} | "
        f"train={'skip (eval-only)' if eval_only else 'on'} | "
        f"eval=on | n_windows={len(windows_full)}"
    )

    all_rows: list[dict] = []
    for ps in args.patch_strides:
        for seed in args.seeds:
            run_sub = run_result_subdir(args.out_dir, int(ps), int(seed))
            ckpt_path = run_checkpoint_path(args.checkpoint_dir, int(ps), int(seed))

            row: dict = {"patch_stride": int(ps), "seed": int(seed), "checkpoint": ckpt_path}

            if not eval_only:
                row["train"] = train_one(
                    windows_full=windows_full,
                    cond_full=cond_full,
                    patch_stride=int(ps),
                    seed=int(seed),
                    epochs=epochs,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    device=args.device,
                    out_dir=args.out_dir,
                    ckpt_dir=args.checkpoint_dir,
                    quick=args.quick,
                )
            else:
                resolved = resolve_checkpoint_path(
                    ckpt_dir=args.checkpoint_dir,
                    out_dir=args.out_dir,
                    patch_stride=int(ps),
                    seed=int(seed),
                )
                if resolved is None:
                    print(f"  [skip] missing checkpoint for patch_stride={ps}, seed={seed}")
                    row["train"] = None
                    row["eval"] = None
                    row["error"] = "missing_checkpoint"
                    all_rows.append(row)
                    continue
                ckpt_path = resolved
                row["checkpoint"] = ckpt_path

            print(f"  eval: patch_stride={ps} seed={seed}")
            try:
                row["eval"] = evaluate_checkpoint(
                    ckpt_path=ckpt_path,
                    windows_full=windows_full,
                    data_dir=args.data_dir,
                    device=args.device,
                    n_gen=args.n_gen,
                    seed=int(seed),
                    run_sub=run_sub,
                )
            except Exception as e:
                print(f"    ERROR: {e}")
                row["eval"] = None
                row["error"] = str(e)

            if eval_only and os.path.isfile(os.path.join(run_sub, "train_config.json")):
                with open(os.path.join(run_sub, "train_config.json"), encoding="utf-8") as f:
                    row["train"] = json.load(f)
            all_rows.append(row)

    summary_path = os.path.join(args.out_dir, "summary_runs.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_rows, f, indent=2, default=_json_default)
    save_visualizations(all_rows, args.out_dir)

    print(f"\nDone. Summary: {summary_path}")


if __name__ == "__main__":
    main()

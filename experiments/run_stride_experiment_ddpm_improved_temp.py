"""
Stride experiment runner for DDPM Improved vs DDPM Improved Temp.

This script follows experiments/run_stride_experiment.py, and evaluates:
  - ddpm_improved
  - ddpm_improved_temp

Usage:
  python -m experiments.run_stride_experiment_ddpm_improved_temp
  python -m experiments.run_stride_experiment_ddpm_improved_temp --quick
  python -m experiments.run_stride_experiment_ddpm_improved_temp --retest-only
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import time
from dataclasses import dataclass
from typing import Any

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import torch

from experiments.run_stride_experiment import (
    STRIDES,
    _build_improved_ddpm_from_ckpt,
    _json_default,
    _plot_heatmap,
    _plot_lines,
    _sample_cond_if_available,
    _save_stride_analysis_outputs,
    _to_float_or_none,
)
from src.data.download import DEFAULT_FRED_KEY
from src.evaluation.cross_model_analysis import run_cross_model_analysis
from src.models.ddpm_improved_temp import ImprovedDDPM
from src.run_pipeline import (
    CHECKPOINTS_DIR,
    DEFAULT_BATCH_SIZE,
    DEFAULT_DEVICE,
    DEFAULT_EPOCHS,
    DEFAULT_LR,
    set_seed,
    step_download,
    step_preprocess,
    step_regime_labels,
    step_train,
)
from src.utils.config import DATA_DIR, PROJECT_ROOT


MODELS = ["ddpm_improved", "ddpm_improved_temp"]


@dataclass
class EvalRow:
    stride: int
    model: str
    checkpoint: str
    sf_passed: int
    proposal_criteria_passed: int | None
    mmd: float | None
    wasserstein_1d: float | None
    sig_w1: float | None
    ks_stat: float | None
    discriminative_score: float | None
    correlation_matrix_distance: float | None
    temporal_coherence: float | None


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


def _build_improved_ddpm_temp_from_ckpt(ckpt_path: str, device: str) -> ImprovedDDPM:
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
        use_min_snr_loss=bool(cfg.get("use_min_snr_loss", True)),
        min_snr_gamma=float(cfg.get("min_snr_gamma", 5.0)),
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


def _train_ddpm_improved_temp(
    data_dir: str,
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
) -> str:
    windows = np.load(os.path.join(data_dir, "windows.npy"))
    cond_path = os.path.join(data_dir, "window_cond.npy")
    cond = np.load(cond_path) if os.path.exists(cond_path) else None

    n_features = windows.shape[2]
    seq_len = windows.shape[1]
    cond_dim = cond.shape[1] if cond is not None else 0
    is_quick = epochs < 100

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
        use_min_snr_loss=True,
        min_snr_gamma=5.0,
    )
    model.train(windows, cond=cond, epochs=epochs, batch_size=batch_size, lr=lr)

    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    ckpt_path = os.path.join(CHECKPOINTS_DIR, "ddpm_improved_temp.pt")
    model.save(ckpt_path)
    return ckpt_path


def _save_stride_checkpoints(stride: int):
    for model_name in MODELS:
        src = os.path.join(CHECKPOINTS_DIR, f"{model_name}.pt")
        dst = os.path.join(CHECKPOINTS_DIR, f"{model_name}_{stride}.pt")
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"  Saved checkpoint: {dst}")


def _run_full_analysis_for_stride(
    stride: int,
    data_dir: str,
    n_samples: int,
    n_bootstrap: int,
    device: str,
    run_dir: str,
) -> list[EvalRow]:
    stride_dir = os.path.join(run_dir, f"stride_{stride}")
    os.makedirs(stride_dir, exist_ok=True)

    dataset = step_preprocess(data_dir, stride=stride)
    real_eval = dataset["windows"][:n_samples]
    cond_eval = _sample_cond_if_available(data_dir, n_samples)

    model_synthetics: dict[str, np.ndarray] = {}
    timing: dict[str, float] = {}
    ckpt_map: dict[str, str] = {}

    for model_name in MODELS:
        ckpt_path = os.path.join(CHECKPOINTS_DIR, f"{model_name}_{stride}.pt")
        if not os.path.exists(ckpt_path):
            print(f"[skip] Missing checkpoint: {ckpt_path}")
            continue

        t0 = time.time()
        if model_name == "ddpm_improved":
            model = _build_improved_ddpm_from_ckpt(ckpt_path, device)
        else:
            model = _build_improved_ddpm_temp_from_ckpt(ckpt_path, device)
        synthetic = model.generate(n_samples, cond=cond_eval)
        model_synthetics[model_name] = synthetic[:len(real_eval)]
        timing[model_name] = time.time() - t0
        ckpt_map[model_name] = ckpt_path
        print(f"  Loaded and generated: {model_name} ({timing[model_name]:.1f}s)")

    if not model_synthetics:
        return []

    analysis = run_cross_model_analysis(
        real_windows=real_eval,
        model_synthetics=model_synthetics,
        window_regimes=None,
        timing=timing,
        n_bootstrap=n_bootstrap,
        save_dir=None,
    )
    _save_stride_analysis_outputs(analysis, stride_dir)

    rows: list[EvalRow] = []
    metrics_by_model = analysis.get("metrics", {})
    tc_by_model = analysis.get("temporal_coherence", {})
    for model_name, metrics in metrics_by_model.items():
        tc = tc_by_model.get(model_name, {})
        rows.append(
            EvalRow(
                stride=stride,
                model=model_name,
                checkpoint=ckpt_map.get(model_name, ""),
                sf_passed=int(metrics.get("sf_passed", 0)),
                proposal_criteria_passed=metrics.get("proposal_criteria_passed"),
                mmd=_to_float_or_none(metrics.get("mmd")),
                wasserstein_1d=_to_float_or_none(metrics.get("wasserstein_1d")),
                sig_w1=_to_float_or_none(metrics.get("sig_w1")),
                ks_stat=_to_float_or_none(metrics.get("ks_stat")),
                discriminative_score=_to_float_or_none(metrics.get("discriminative_score")),
                correlation_matrix_distance=_to_float_or_none(metrics.get("correlation_matrix_distance")),
                temporal_coherence=_to_float_or_none(tc.get("overall_score")),
            )
        )
    return rows


def run_stride_experiment(
    data_dir: str,
    epochs: int,
    n_samples: int,
    n_bootstrap: int,
    batch_size: int,
    lr: float,
    device: str,
    run_dir: str,
    retrain: bool,
):
    all_rows: list[EvalRow] = []
    stride_dirs: list[str] = []

    for stride in STRIDES:
        print("\n" + "#" * 80)
        print(f"STRIDE EXPERIMENT (DDPM IMPROVED TEMP): stride={stride}")
        print("#" * 80)
        set_seed()

        if retrain:
            dataset = step_preprocess(data_dir, stride=stride)
            step_regime_labels(dataset, data_dir)
            step_train(
                data_dir=data_dir,
                models_to_train=["ddpm_improved"],
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                device=device,
            )
            _train_ddpm_improved_temp(
                data_dir=data_dir,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                device=device,
            )
            _save_stride_checkpoints(stride)

        rows = _run_full_analysis_for_stride(
            stride=stride,
            data_dir=data_dir,
            n_samples=n_samples,
            n_bootstrap=n_bootstrap,
            device=device,
            run_dir=run_dir,
        )
        all_rows.extend(rows)
        stride_dirs.append(os.path.join(run_dir, f"stride_{stride}"))

    if not all_rows:
        raise RuntimeError("No results collected. Check stride checkpoints and runtime logs.")

    summary_df = pd.DataFrame([r.__dict__ for r in all_rows]).sort_values(["stride", "model"]).reset_index(drop=True)
    summary_df.to_csv(os.path.join(run_dir, "summary_metrics.csv"), index=False)
    with open(os.path.join(run_dir, "summary_metrics.json"), "w") as f:
        json.dump(summary_df.to_dict(orient="records"), f, indent=2)
    with open(os.path.join(run_dir, "run_config.json"), "w") as f:
        json.dump(
            {
                "strides": STRIDES,
                "models": MODELS,
                "epochs": epochs,
                "n_samples": n_samples,
                "n_bootstrap": n_bootstrap,
                "batch_size": batch_size,
                "lr": lr,
                "device": device,
                "retrain": retrain,
                "stride_dirs": stride_dirs,
            },
            f,
            indent=2,
        )

    _plot_lines(summary_df, "mmd", "MMD", os.path.join(run_dir, "mmd_vs_stride.png"))
    _plot_lines(
        summary_df,
        "discriminative_score",
        "Discriminative Score",
        os.path.join(run_dir, "discriminative_vs_stride.png"),
    )
    _plot_lines(
        summary_df,
        "temporal_coherence",
        "Temporal Coherence",
        os.path.join(run_dir, "temporal_coherence_vs_stride.png"),
    )
    _plot_lines(
        summary_df,
        "sf_passed",
        "Stylized Facts Passed",
        os.path.join(run_dir, "sf_passed_vs_stride.png"),
    )
    _plot_heatmap(summary_df, os.path.join(run_dir, "metrics_heatmap.png"))

    print("\n" + "=" * 80)
    print("STRIDE EXPERIMENT (DDPM IMPROVED VS TEMP) COMPLETE")
    print("=" * 80)
    print(f"Output directory: {run_dir}")
    print(f"  {os.path.join(run_dir, 'summary_metrics.csv')}")
    print(f"  {os.path.join(run_dir, 'metrics_heatmap.png')}")


def main():
    parser = argparse.ArgumentParser(description="Stride experiment runner for ddpm_improved vs ddpm_improved_temp")
    parser.add_argument("--skip-download", action="store_true", help="Skip data download")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--quick", action="store_true", help="Quick run with reduced epochs/samples/bootstrap")
    parser.add_argument("--retest-only", action="store_true", help="Skip training and only retest saved checkpoints")
    parser.add_argument("--n-samples", type=int, default=500)
    parser.add_argument("--n-bootstrap", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--out-dir", default=None, help="Full output dir under experiments/results")
    parser.add_argument(
        "--fred-key",
        default=DEFAULT_FRED_KEY,
        help="FRED API key (default: src.data.download.DEFAULT_FRED_KEY)",
    )
    args = parser.parse_args()

    data_dir = args.data_dir or DATA_DIR
    if args.quick:
        epochs = args.epochs or 20
        n_samples = min(args.n_samples, 200)
        n_bootstrap = min(args.n_bootstrap, 20)
    else:
        epochs = args.epochs or DEFAULT_EPOCHS
        n_samples = args.n_samples
        n_bootstrap = args.n_bootstrap

    if args.out_dir:
        run_dir = args.out_dir
    else:
        run_dir = os.path.join(PROJECT_ROOT, "experiments", "results", "stride_experiment_ddpm_improved_vs_temp")
    os.makedirs(run_dir, exist_ok=True)

    print(f"Device: {DEFAULT_DEVICE}")
    print(f"Strides: {STRIDES}")
    print(f"Models: {MODELS}")
    print(f"Mode: {'retest-only' if args.retest_only else 'train+test'}")
    print(f"Output dir: {run_dir}")

    set_seed()
    if not args.skip_download:
        step_download(data_dir, args.fred_key)

    run_stride_experiment(
        data_dir=data_dir,
        epochs=epochs,
        n_samples=n_samples,
        n_bootstrap=n_bootstrap,
        batch_size=args.batch_size,
        lr=args.lr,
        device=DEFAULT_DEVICE,
        run_dir=run_dir,
        retrain=not args.retest_only,
    )


if __name__ == "__main__":
    main()

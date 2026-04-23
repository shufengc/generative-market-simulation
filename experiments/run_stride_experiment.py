"""
Unified stride experiment script.

This script merges:
1) training + evaluation across fixed strides
2) retest-only mode using existing stride checkpoints

Main models:
  - ddpm
  - ddpm_improved

Usage:
  python -m experiments.run_stride_experiment
  python -m experiments.run_stride_experiment --quick
  python -m experiments.run_stride_experiment --retest-only
  python -m experiments.run_stride_experiment --retest-only --n-samples 1000
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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from src.data.download import DEFAULT_FRED_KEY
from src.evaluation.cross_model_analysis import (
    plot_radar_chart,
    plot_regime_heatmap,
    plot_temporal_coherence,
    run_cross_model_analysis,
)
from src.models.ddpm import DDPMModel
from src.models.ddpm_improved import ImprovedDDPM
from src.run_pipeline import (
    CHECKPOINTS_DIR,
    DEFAULT_DEVICE,
    DEFAULT_EPOCHS,
    set_seed,
    step_download,
    step_preprocess,
    step_regime_labels,
    step_train,
)
from src.utils.config import DATA_DIR, PROJECT_ROOT


STRIDES = [1, 2, 3, 5, 10, 20]
MODELS = ["ddpm", "ddpm_improved"]


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


def _to_float_or_none(v: Any):
    if isinstance(v, (int, float)) and np.isfinite(v):
        return float(v)
    return None


def _json_default(v: Any):
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


def _build_ddpm_from_ckpt(ckpt_path: str, device: str) -> DDPMModel:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    net_state = ckpt["net_state"]

    base_channels = int(net_state["init_conv.weight"].shape[0])
    channel_mults = _infer_channel_mults(net_state, base_channels)

    model = DDPMModel(
        n_features=int(cfg.get("n_features", 16)),
        seq_len=int(cfg.get("seq_len", 60)),
        T=int(cfg.get("T", 1000)),
        base_channels=base_channels,
        channel_mults=channel_mults,
        cond_dim=int(cfg.get("cond_dim", 0)),
        cfg_drop_prob=float(cfg.get("cfg_drop_prob", 0.1)),
        device=device,
    )
    model.load(ckpt_path)
    return model


def _build_improved_ddpm_from_ckpt(ckpt_path: str, device: str) -> ImprovedDDPM:
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


def _load_model(model_name: str, ckpt_path: str, device: str):
    if model_name == "ddpm":
        return _build_ddpm_from_ckpt(ckpt_path, device)
    if model_name == "ddpm_improved":
        return _build_improved_ddpm_from_ckpt(ckpt_path, device)
    raise ValueError(f"Unsupported model: {model_name}")


def _sample_cond_if_available(data_dir: str, n_samples: int) -> np.ndarray | None:
    cond_path = os.path.join(data_dir, "window_cond.npy")
    if not os.path.exists(cond_path):
        return None
    cond = np.load(cond_path)
    if cond.ndim != 2 or len(cond) == 0:
        return None
    idx = np.random.choice(len(cond), size=n_samples, replace=True)
    return cond[idx]


def _save_stride_checkpoints(stride: int):
    for model_name in MODELS:
        src = os.path.join(CHECKPOINTS_DIR, f"{model_name}.pt")
        dst = os.path.join(CHECKPOINTS_DIR, f"{model_name}_{stride}.pt")
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"  Saved checkpoint: {dst}")


def _plot_lines(df: pd.DataFrame, metric: str, ylabel: str, out_path: str):
    if metric not in df.columns or df[metric].dropna().empty:
        return
    plt.figure(figsize=(8, 5))
    sns.lineplot(data=df, x="stride", y=metric, hue="model", marker="o", linewidth=2)
    plt.title(f"{ylabel} vs Stride", fontweight="bold")
    plt.xlabel("Stride")
    plt.ylabel(ylabel)
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()


def _plot_heatmap(df: pd.DataFrame, out_path: str):
    view = df.copy()
    view["name"] = view["model"] + "_s" + view["stride"].astype(str)
    cols = [
        "sf_passed",
        "proposal_criteria_passed",
        "mmd",
        "discriminative_score",
        "correlation_matrix_distance",
        "temporal_coherence",
    ]
    cols = [c for c in cols if c in view.columns]
    mat = view.set_index("name")[cols]
    plt.figure(figsize=(10, max(4, 0.45 * len(view) + 2)))
    sns.heatmap(mat, annot=True, fmt=".3g", cmap="YlOrRd", linewidths=0.4)
    plt.title("Stride Experiment Metrics Heatmap", fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()


def _save_ci_plot_safe(ci_results: dict[str, dict[str, dict]], metric_name: str, save_path: str):
    names = list(ci_results.keys())
    means, errors_low, errors_high = [], [], []

    for name in names:
        ci = ci_results.get(name, {}).get(metric_name, {})
        mean = ci.get("mean", np.nan)
        ci_low = ci.get("ci_low", np.nan)
        ci_high = ci.get("ci_high", np.nan)
        if not (np.isfinite(mean) and np.isfinite(ci_low) and np.isfinite(ci_high)):
            mean = 0.0
            low = 0.0
            high = 0.0
        else:
            low = max(float(mean - ci_low), 0.0)
            high = max(float(ci_high - mean), 0.0)

        means.append(float(mean))
        errors_low.append(low)
        errors_high.append(high)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(
        names,
        means,
        yerr=[errors_low, errors_high],
        capsize=6,
        color=["#4C78A8", "#F58518", "#54A24B", "#E45756"][: len(names)],
        edgecolor="white",
        alpha=0.85,
    )
    ax.set_ylabel(metric_name.replace("_", " ").title(), fontsize=11)
    ax.set_title(f"{metric_name.replace('_', ' ').title()} with 95% Bootstrap CI", fontsize=13, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_stride_analysis_outputs(analysis: dict[str, Any], stride_dir: str):
    os.makedirs(stride_dir, exist_ok=True)
    metrics = analysis["metrics"]
    ranking = analysis["ranking"]
    ci = analysis["ci"]
    regime = analysis["regime"]
    significance = analysis["significance"]
    tc = analysis["temporal_coherence"]
    report = analysis["report"]
    normflow_analysis = analysis["normflow_analysis"]

    plot_radar_chart(metrics, save_path=os.path.join(stride_dir, "radar_chart.png"))
    for metric in ("mmd", "wasserstein", "discriminative_score"):
        _save_ci_plot_safe(ci, metric_name=metric, save_path=os.path.join(stride_dir, f"ci_{metric}.png"))
    for metric in ("sf_passed", "mmd"):
        plot_regime_heatmap(regime, metric_name=metric, save_path=os.path.join(stride_dir, f"regime_{metric}.png"))
    plot_temporal_coherence(tc, save_path=os.path.join(stride_dir, "temporal_coherence.png"))

    ranking.to_csv(os.path.join(stride_dir, "model_ranking.csv"))
    significance.to_csv(os.path.join(stride_dir, "pairwise_significance.csv"))

    with open(os.path.join(stride_dir, "cross_model_report.txt"), "w", encoding="utf-8") as f:
        f.write(report)
    with open(os.path.join(stride_dir, "normflow_analysis.txt"), "w", encoding="utf-8") as f:
        f.write(normflow_analysis)

    with open(os.path.join(stride_dir, "cross_model_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, default=_json_default)

    tc_serializable = {}
    for name, tc_item in tc.items():
        tc_copy = {k: v for k, v in tc_item.items() if k != "tests"}
        tc_copy["tests"] = {}
        for tk, tv in tc_item["tests"].items():
            if isinstance(tv, dict):
                tc_copy["tests"][tk] = {str(kk): vv for kk, vv in tv.items()}
            else:
                tc_copy["tests"][tk] = tv
        tc_serializable[name] = tc_copy
    with open(os.path.join(stride_dir, "temporal_coherence.json"), "w") as f:
        json.dump(tc_serializable, f, indent=2, default=_json_default)


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
        model = _load_model(model_name, ckpt_path, device)
        syn = model.generate(n_samples, cond=cond_eval)
        model_synthetics[model_name] = syn[:len(real_eval)]
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
    for model_name, m in metrics_by_model.items():
        tc = tc_by_model.get(model_name, {})
        rows.append(
            EvalRow(
                stride=stride,
                model=model_name,
                checkpoint=ckpt_map.get(model_name, ""),
                sf_passed=int(m.get("sf_passed", 0)),
                proposal_criteria_passed=m.get("proposal_criteria_passed"),
                mmd=_to_float_or_none(m.get("mmd")),
                wasserstein_1d=_to_float_or_none(m.get("wasserstein_1d")),
                sig_w1=_to_float_or_none(m.get("sig_w1")),
                ks_stat=_to_float_or_none(m.get("ks_stat")),
                discriminative_score=_to_float_or_none(m.get("discriminative_score")),
                correlation_matrix_distance=_to_float_or_none(m.get("correlation_matrix_distance")),
                temporal_coherence=_to_float_or_none(tc.get("overall_score")),
            )
        )
    return rows


def run_stride_experiment(
    data_dir: str,
    epochs: int,
    n_samples: int,
    n_bootstrap: int,
    device: str,
    run_dir: str,
    retrain: bool,
):
    all_rows: list[EvalRow] = []
    stride_dirs: list[str] = []

    for stride in STRIDES:
        print("\n" + "#" * 80)
        print(f"STRIDE EXPERIMENT: stride={stride}")
        print("#" * 80)
        set_seed()

        if retrain:
            dataset = step_preprocess(data_dir, stride=stride)
            step_regime_labels(dataset, data_dir)
            step_train(
                data_dir=data_dir,
                models_to_train=MODELS,
                epochs=epochs,
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
    print("STRIDE EXPERIMENT COMPLETE")
    print("=" * 80)
    print(f"Output directory: {run_dir}")
    print("Top-level files:")
    print(f"  {os.path.join(run_dir, 'summary_metrics.csv')}")
    print(f"  {os.path.join(run_dir, 'metrics_heatmap.png')}")
    print("Per-stride full-test outputs:")
    for d in stride_dirs:
        print(f"  {d}")


def main():
    parser = argparse.ArgumentParser(description="Unified stride experiment runner")
    parser.add_argument("--skip-download", action="store_true", help="Skip data download")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--quick", action="store_true", help="Quick run with reduced epochs/samples/bootstrap")
    parser.add_argument("--retest-only", action="store_true", help="Skip training and only retest saved checkpoints")
    parser.add_argument("--n-samples", type=int, default=500)
    parser.add_argument("--n-bootstrap", type=int, default=50)
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
        run_dir = os.path.join(PROJECT_ROOT, "experiments", "results", "stride_experiment")
    os.makedirs(run_dir, exist_ok=True)

    print(f"Device: {DEFAULT_DEVICE}")
    print(f"Strides: {STRIDES}")
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
        device=DEFAULT_DEVICE,
        run_dir=run_dir,
        retrain=not args.retest_only,
    )


if __name__ == "__main__":
    main()

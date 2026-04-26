"""
run_conditional_ddpm.py
=======================
Train the ImprovedDDPM with cond_dim=5 (macro conditioning vectors)
using classifier-free guidance, then generate regime-specific samples
and evaluate them.

Usage:
    python experiments/run_conditional_ddpm.py
    python experiments/run_conditional_ddpm.py --skip-train   # eval only
    python experiments/run_conditional_ddpm.py --skip-eval    # train only
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.data.regime_labels import get_regime_conditioning_vectors  # noqa: E402
from src.models.ddpm_improved import ImprovedDDPM                   # noqa: E402
from src.evaluation.metrics import full_evaluation                   # noqa: E402
from src.evaluation.stylized_facts import run_all_tests             # noqa: E402


# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CFG = {
    "cond_dim": 5,
    "seq_len": 60,
    "T": 1000,
    "base_channels": 128,
    "channel_mults": (1, 2, 4),
    "use_vpred": True,
    "use_student_t_noise": True,
    "student_t_df": 5.0,
    "ema_decay": 0.9999,
    "use_ddim": True,
    "ddim_steps": 50,
    "ddim_eta": 0.3,
    "guidance_scale": 2.0,
    "cfg_drop_prob": 0.1,
    "epochs": 400,
    "batch_size": 64,
    "lr": 2e-4,
    "seed": 42,
    "data_dir":       os.path.join(ROOT, "data"),
    "checkpoint_dir": os.path.join(ROOT, "checkpoints"),
    "output_dir":     os.path.join(ROOT, "experiments", "results", "conditional_ddpm"),
}

REGIMES = ["crisis", "calm", "normal"]


# ─────────────────────────────────────────────────────────────
# Data loading  (all files already exist in data/)
# ─────────────────────────────────────────────────────────────
def load_data():
    data_dir = CFG["data_dir"]
    windows       = np.load(os.path.join(data_dir, "windows.npy"))        # (N, 60, D)
    window_cond   = np.load(os.path.join(data_dir, "window_cond.npy"))    # (N, 5)
    window_regimes = np.load(os.path.join(data_dir, "window_regimes.npy"))# (N,)

    print(f"  Windows:  {windows.shape}  dtype={windows.dtype}")
    print(f"  Cond:     {window_cond.shape}  dtype={window_cond.dtype}")
    print(f"  Regimes:  {window_regimes.shape}")
    counts = {name: int((window_regimes == i).sum())
              for i, name in enumerate(["normal", "crisis", "calm"])}
    print(f"  Regime distribution: {counts}")
    return windows, window_cond, window_regimes


# ─────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────
def build_model(n_features: int) -> ImprovedDDPM:
    return ImprovedDDPM(
        n_features=n_features,
        seq_len=CFG["seq_len"],
        cond_dim=CFG["cond_dim"],
        T=CFG["T"],
        base_channels=CFG["base_channels"],
        channel_mults=CFG["channel_mults"],
        use_vpred=CFG["use_vpred"],
        use_student_t_noise=CFG["use_student_t_noise"],
        student_t_df=CFG["student_t_df"],
        cfg_drop_prob=CFG["cfg_drop_prob"],
        device=DEVICE,
    )


def train(windows: np.ndarray, window_cond: np.ndarray) -> ImprovedDDPM:
    np.random.seed(CFG["seed"])
    n_features = windows.shape[2]

    print(f"\nBuilding ImprovedDDPM  n_features={n_features}  cond_dim={CFG['cond_dim']}  device={DEVICE}")
    model = build_model(n_features)
    n_params = sum(p.numel() for p in model.net.parameters())
    print(f"  Parameters: {n_params:,}")

    print(f"Training {CFG['epochs']} epochs  batch={CFG['batch_size']}  lr={CFG['lr']} ...")
    t0 = time.time()
    model.train(
        windows,
        cond=window_cond,
        epochs=CFG["epochs"],
        batch_size=CFG["batch_size"],
        lr=CFG["lr"],
        ema_decay=CFG["ema_decay"],
    )
    elapsed = time.time() - t0
    print(f"Training completed in {elapsed / 3600:.2f} hours  ({elapsed:.0f} s)")

    os.makedirs(CFG["checkpoint_dir"], exist_ok=True)
    ckpt_path = os.path.join(CFG["checkpoint_dir"], "ddpm_conditional.pt")
    model.save(ckpt_path)
    print(f"Checkpoint saved: {ckpt_path}")
    return model


# ─────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────
def evaluate_conditional(model: ImprovedDDPM, windows: np.ndarray,
                          window_regimes: np.ndarray) -> dict:
    os.makedirs(CFG["output_dir"], exist_ok=True)
    regime_vectors = get_regime_conditioning_vectors()
    results = {}
    n_gen = 1000

    for regime_name in REGIMES:
        print(f"\n--- Generating {n_gen} samples: regime={regime_name} ---")
        cond_vec = regime_vectors[regime_name]   # shape (5,)

        synthetic = model.generate(
            n_samples=n_gen,
            use_ddim=CFG["use_ddim"],
            ddim_steps=CFG["ddim_steps"],
            guidance_scale=CFG["guidance_scale"],
            ddim_eta=CFG["ddim_eta"],
            cond=cond_vec,
        )
        print(f"  Generated shape: {synthetic.shape}")

        # Matched real windows for this regime
        regime_int = {"crisis": 1, "calm": 2, "normal": 0}[regime_name]
        mask = window_regimes == regime_int
        real_regime = windows[mask]
        if len(real_regime) < 50:
            print(f"  WARNING: only {len(real_regime)} real windows for '{regime_name}'")
        n_real = min(len(real_regime), n_gen)
        real_subset = real_regime[:n_real]

        # Distributional metrics (real vs synthetic)
        metrics = full_evaluation(real_subset, synthetic[:n_real])

        # Stylized facts on synthetic (absolute thresholds, no reference)
        sf_list = run_all_tests(synthetic)
        sf_count = sum(1 for r in sf_list if r.get("pass", False))

        # Volatility sanity check
        syn_flat = synthetic.reshape(-1, synthetic.shape[-1])
        vol_mean = float(np.std(syn_flat, axis=0).mean())
        vol_max  = float(np.std(syn_flat, axis=0).max())

        results[regime_name] = {
            "n_generated":            n_gen,
            "n_real_windows":         int(mask.sum()),
            "sf_count":               sf_count,
            "vol_mean_across_assets": round(vol_mean, 4),
            "vol_max_asset":          round(vol_max, 4),
            **metrics,
        }
        print(f"  SF: {sf_count}/6  MMD: {metrics['mmd']:.4f}  "
              f"Disc: {metrics['discriminative_score']:.3f}  "
              f"Vol(mean): {vol_mean:.4f}")

        np.save(os.path.join(CFG["output_dir"], f"synthetic_{regime_name}.npy"), synthetic)

    # Save JSON
    out_json = os.path.join(CFG["output_dir"], "conditional_eval_results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {out_json}")

    print("\n" + "=" * 65)
    print(f"{'Regime':<10} {'SF':>4} {'MMD':>8} {'Disc':>8} {'Vol(mean)':>12}")
    print("-" * 65)
    for name in REGIMES:
        r = results[name]
        print(f"{name:<10} {r['sf_count']:>4} {r['mmd']:>8.4f} "
              f"{r['discriminative_score']:>8.3f} {r['vol_mean_across_assets']:>12.4f}")
    print("=" * 65)
    return results


def make_regime_plots(windows: np.ndarray, window_regimes: np.ndarray) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"crisis": "#e53935", "calm": "#43a047", "normal": "#1e88e5"}

    # Return distribution per regime
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=False)
    for ax, regime_name in zip(axes, REGIMES):
        syn_path = os.path.join(CFG["output_dir"], f"synthetic_{regime_name}.npy")
        if not os.path.exists(syn_path):
            continue
        synthetic = np.load(syn_path)
        regime_int = {"crisis": 1, "calm": 2, "normal": 0}[regime_name]
        mask = window_regimes == regime_int
        ax.hist(windows[mask].reshape(-1), bins=80, alpha=0.5, density=True,
                color="#546e7a", label="Real")
        ax.hist(synthetic.reshape(-1), bins=80, alpha=0.6, density=True,
                color=colors[regime_name], label=f"Synthetic ({regime_name})")
        ax.set_title(f"Regime: {regime_name.capitalize()}", fontsize=13, fontweight="bold")
        ax.set_xlabel("Return (z-scored)", fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.legend(fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Return Distribution by Regime — Real vs Conditional DDPM",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    out = os.path.join(CFG["output_dir"], "regime_distributions.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out}")

    # Volatility profile per regime
    fig, ax = plt.subplots(figsize=(10, 5))
    for regime_name in REGIMES:
        syn_path = os.path.join(CFG["output_dir"], f"synthetic_{regime_name}.npy")
        if not os.path.exists(syn_path):
            continue
        synthetic = np.load(syn_path)   # (N, 60, D)
        rolling_vol = np.array([
            np.std(synthetic[:, max(0, t - 5):t + 1, :])
            for t in range(60)
        ])
        ax.plot(rolling_vol, label=regime_name.capitalize(),
                color=colors[regime_name], linewidth=2)

    ax.set_xlabel("Timestep within 60-day window", fontsize=12)
    ax.set_ylabel("Rolling Volatility (std)", fontsize=12)
    ax.set_title("Volatility Profile by Regime — Conditional DDPM",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    out = os.path.join(CFG["output_dir"], "regime_volatility_profiles.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out}")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Conditional DDPM training and evaluation")
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip training, load existing checkpoint")
    parser.add_argument("--skip-eval",  action="store_true",
                        help="Train only, skip evaluation and plots")
    args = parser.parse_args()

    print(f"Device: {DEVICE}")
    if DEVICE == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print("\nLoading data ...")
    windows, window_cond, window_regimes = load_data()
    n_features = windows.shape[2]

    if args.skip_train:
        ckpt_path = os.path.join(CFG["checkpoint_dir"], "ddpm_conditional.pt")
        print(f"\nLoading checkpoint: {ckpt_path}")
        model = build_model(n_features)
        model.load(ckpt_path)
    else:
        model = train(windows, window_cond)

    if not args.skip_eval:
        evaluate_conditional(model, windows, window_regimes)
        make_regime_plots(windows, window_regimes)


if __name__ == "__main__":
    main()

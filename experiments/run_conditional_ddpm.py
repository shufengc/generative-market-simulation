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

v2 ablation flags:
    --tag TAG            output subdirectory tag (default: v1)
    --student-t-df DF   Student-t degrees of freedom (default: 5.0)
    --aux-sf-loss        enable auxiliary kurtosis+ACF loss
    --decorr-reg         enable decorrelation regularizer (SF6)
    --crisis-oversample N  oversample crisis windows N times (default: 1)
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
# Base config (overridden by CLI where applicable)
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
    "student_t_df": 5.0,          # overridden by --student-t-df
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
    # ablation flags (overridden by CLI)
    "use_aux_sf_loss":  False,
    "aux_sf_weight":    0.1,
    "use_decorr_reg":   False,
    "decorr_weight":    0.05,
    "crisis_oversample": 1,
    "use_regime_weight": False,   # inverse-frequency weighting for all regimes
    "tag": "v1",
    "data_dir":       os.path.join(ROOT, "data"),
    "checkpoint_dir": os.path.join(ROOT, "checkpoints"),
}

REGIMES = ["crisis", "calm", "normal"]


def _output_dir() -> str:
    tag = CFG["tag"]
    if tag == "v1":
        return os.path.join(ROOT, "experiments", "results", "conditional_ddpm")
    return os.path.join(ROOT, "experiments", "results", "conditional_ddpm_v2", tag)


def _ckpt_name() -> str:
    tag = CFG["tag"]
    if tag == "v1":
        return "ddpm_conditional.pt"
    return f"ddpm_conditional_{tag}.pt"


# ─────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────
def load_data():
    data_dir = CFG["data_dir"]
    windows        = np.load(os.path.join(data_dir, "windows.npy"))
    window_cond    = np.load(os.path.join(data_dir, "window_cond.npy"))
    window_regimes = np.load(os.path.join(data_dir, "window_regimes.npy"))

    print(f"  Windows:  {windows.shape}  dtype={windows.dtype}")
    print(f"  Cond:     {window_cond.shape}  dtype={window_cond.dtype}")
    print(f"  Regimes:  {window_regimes.shape}")
    counts = {name: int((window_regimes == i).sum())
              for i, name in enumerate(["normal", "crisis", "calm"])}
    print(f"  Regime distribution: {counts}")

    # Optional crisis oversampling
    n_oversample = CFG["crisis_oversample"]
    if n_oversample > 1:
        crisis_mask = window_regimes == 1   # crisis == 1
        n_before = windows.shape[0]
        extra_w = np.tile(windows[crisis_mask], (n_oversample - 1, 1, 1))
        extra_c = np.tile(window_cond[crisis_mask], (n_oversample - 1, 1))
        extra_r = np.tile(window_regimes[crisis_mask], n_oversample - 1)
        windows        = np.concatenate([windows, extra_w], axis=0)
        window_cond    = np.concatenate([window_cond, extra_c], axis=0)
        window_regimes = np.concatenate([window_regimes, extra_r], axis=0)
        print(f"  Crisis oversampled {n_oversample}x: {n_before} -> {windows.shape[0]} windows")

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
        use_aux_sf_loss=CFG["use_aux_sf_loss"],
        aux_sf_weight=CFG["aux_sf_weight"],
        use_decorr_reg=CFG["use_decorr_reg"],
        decorr_weight=CFG["decorr_weight"],
        device=DEVICE,
    )


def train(windows: np.ndarray, window_cond: np.ndarray,
          window_regimes: np.ndarray | None = None) -> ImprovedDDPM:
    np.random.seed(CFG["seed"])
    n_features = windows.shape[2]

    print(f"\nBuilding ImprovedDDPM  n_features={n_features}  cond_dim={CFG['cond_dim']}  device={DEVICE}")
    print(f"  student_t_df={CFG['student_t_df']}  aux_sf_loss={CFG['use_aux_sf_loss']}"
          f"  decorr_reg={CFG['use_decorr_reg']}  crisis_oversample={CFG['crisis_oversample']}x"
          f"  regime_weight={CFG['use_regime_weight']}")
    model = build_model(n_features)
    n_params = sum(p.numel() for p in model.net.parameters())
    print(f"  Parameters: {n_params:,}")

    # Compute inverse-frequency sample weights when --regime-weight is set
    sample_weights = None
    if CFG["use_regime_weight"] and window_regimes is not None:
        unique_r, counts_r = np.unique(window_regimes, return_counts=True)
        freq = dict(zip(unique_r, counts_r))
        n_total = len(window_regimes)
        # Weight each sample inversely proportional to its regime frequency
        weights = np.array([n_total / freq[r] for r in window_regimes], dtype=np.float32)
        weights /= weights.mean()   # normalise so mean weight = 1
        sample_weights = weights
        regime_names = {1: "crisis", 2: "calm", 0: "normal"}
        for r, cnt in zip(unique_r, counts_r):
            print(f"  regime {regime_names.get(r, r)}: {cnt} windows, weight={n_total/freq[r]:.2f}x")

    print(f"Training {CFG['epochs']} epochs  batch={CFG['batch_size']}  lr={CFG['lr']} ...")
    t0 = time.time()
    model.train(
        windows,
        cond=window_cond,
        epochs=CFG["epochs"],
        batch_size=CFG["batch_size"],
        lr=CFG["lr"],
        ema_decay=CFG["ema_decay"],
        sample_weights=sample_weights,
    )
    elapsed = time.time() - t0
    print(f"Training completed in {elapsed / 3600:.2f} hours  ({elapsed:.0f} s)")

    os.makedirs(CFG["checkpoint_dir"], exist_ok=True)
    ckpt_path = os.path.join(CFG["checkpoint_dir"], _ckpt_name())
    model.save(ckpt_path)
    print(f"Checkpoint saved: {ckpt_path}")
    return model


# ─────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────
def evaluate_conditional(model: ImprovedDDPM, windows: np.ndarray,
                          window_regimes: np.ndarray) -> dict:
    out_dir = _output_dir()
    os.makedirs(out_dir, exist_ok=True)
    regime_vectors = get_regime_conditioning_vectors()
    results = {}
    n_gen = 1000

    # Use unoversampled windows for evaluation (real distribution unchanged)
    windows_npy = np.load(os.path.join(CFG["data_dir"], "windows.npy"))
    reg_npy     = np.load(os.path.join(CFG["data_dir"], "window_regimes.npy"))

    for regime_name in REGIMES:
        print(f"\n--- Generating {n_gen} samples: regime={regime_name} ---")
        cond_vec = regime_vectors[regime_name]

        synthetic = model.generate(
            n_samples=n_gen,
            use_ddim=CFG["use_ddim"],
            ddim_steps=CFG["ddim_steps"],
            guidance_scale=CFG["guidance_scale"],
            ddim_eta=CFG["ddim_eta"],
            cond=cond_vec,
        )
        print(f"  Generated shape: {synthetic.shape}")

        regime_int  = {"crisis": 1, "calm": 2, "normal": 0}[regime_name]
        mask        = reg_npy == regime_int
        real_regime = windows_npy[mask]
        if len(real_regime) < 50:
            print(f"  WARNING: only {len(real_regime)} real windows for '{regime_name}'")
        n_real      = min(len(real_regime), n_gen)
        real_subset = real_regime[:n_real]

        metrics  = full_evaluation(real_subset, synthetic[:n_real])
        sf_list  = run_all_tests(synthetic)
        sf_count = sum(1 for r in sf_list if r.get("pass", False))

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

        np.save(os.path.join(out_dir, f"synthetic_{regime_name}.npy"), synthetic)

    out_json = os.path.join(out_dir, "conditional_eval_results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    # also save the config used
    cfg_json = os.path.join(out_dir, "run_config.json")
    with open(cfg_json, "w") as f:
        json.dump({k: v for k, v in CFG.items() if not k.endswith("_dir")
                   and not isinstance(v, tuple)}, f, indent=2)
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

    out_dir = _output_dir()
    # Use unoversampled windows for real distribution plots
    windows_npy = np.load(os.path.join(CFG["data_dir"], "windows.npy"))
    reg_npy     = np.load(os.path.join(CFG["data_dir"], "window_regimes.npy"))

    colors = {"crisis": "#e53935", "calm": "#43a047", "normal": "#1e88e5"}

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=False)
    for ax, regime_name in zip(axes, REGIMES):
        syn_path = os.path.join(out_dir, f"synthetic_{regime_name}.npy")
        if not os.path.exists(syn_path):
            continue
        synthetic  = np.load(syn_path)
        regime_int = {"crisis": 1, "calm": 2, "normal": 0}[regime_name]
        mask       = reg_npy == regime_int
        ax.hist(windows_npy[mask].reshape(-1), bins=80, alpha=0.5, density=True,
                color="#546e7a", label="Real")
        ax.hist(synthetic.reshape(-1), bins=80, alpha=0.6, density=True,
                color=colors[regime_name], label=f"Synthetic ({regime_name})")
        ax.set_title(f"Regime: {regime_name.capitalize()}", fontsize=13, fontweight="bold")
        ax.set_xlabel("Return (z-scored)", fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.legend(fontsize=9)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    fig.suptitle(f"Return Distribution by Regime — DDPM [{CFG['tag']}]",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    out = os.path.join(out_dir, "regime_distributions.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out}")

    fig, ax = plt.subplots(figsize=(10, 5))
    for regime_name in REGIMES:
        syn_path = os.path.join(out_dir, f"synthetic_{regime_name}.npy")
        if not os.path.exists(syn_path):
            continue
        synthetic   = np.load(syn_path)
        rolling_vol = np.array([
            np.std(synthetic[:, max(0, t - 5):t + 1, :])
            for t in range(60)
        ])
        ax.plot(rolling_vol, label=regime_name.capitalize(),
                color=colors[regime_name], linewidth=2)
    ax.set_xlabel("Timestep within 60-day window", fontsize=12)
    ax.set_ylabel("Rolling Volatility (std)", fontsize=12)
    ax.set_title(f"Volatility Profile by Regime — DDPM [{CFG['tag']}]",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=11); ax.grid(alpha=0.3)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout()
    out = os.path.join(out_dir, "regime_volatility_profiles.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out}")


# ─────────────────────────────────────────────────────────────
# Fine-tuning
# ─────────────────────────────────────────────────────────────

def _run_finetune(model: ImprovedDDPM, windows: np.ndarray, window_cond: np.ndarray,
                  window_regimes: np.ndarray, regime: str,
                  lr: float = 5e-5, epochs: int = 100) -> None:
    """
    Fine-tune the already-loaded model on windows from a single regime.

    Overwrites the checkpoint at _ckpt_name() after fine-tuning and updates the
    output tag to include '_ft{regime}' so results go to a separate directory.
    """
    regime_int = {"crisis": 1, "calm": 2, "normal": 0}[regime]
    # Use the original (unoversampled) real windows for fine-tune
    real_w = np.load(os.path.join(CFG["data_dir"], "windows.npy"))
    real_c = np.load(os.path.join(CFG["data_dir"], "window_cond.npy"))
    real_r = np.load(os.path.join(CFG["data_dir"], "window_regimes.npy"))

    mask = real_r == regime_int
    ft_windows = real_w[mask]
    ft_cond    = real_c[mask]
    print(f"\n[Fine-tune] Regime={regime}  windows={ft_windows.shape[0]}"
          f"  lr={lr}  epochs={epochs}")

    if ft_windows.shape[0] < 32:
        print(f"  [SKIP] Too few windows ({ft_windows.shape[0]}) for fine-tuning.")
        return

    # Update tag so the fine-tuned checkpoint and results go to a new directory
    original_tag = CFG["tag"]
    CFG["tag"] = f"{original_tag}_ft{regime}"

    model.train(
        ft_windows,
        cond=ft_cond,
        epochs=epochs,
        batch_size=min(CFG["batch_size"], ft_windows.shape[0]),
        lr=lr,
        ema_decay=CFG["ema_decay"],
    )

    os.makedirs(CFG["checkpoint_dir"], exist_ok=True)
    ckpt_path = os.path.join(CFG["checkpoint_dir"], _ckpt_name())
    model.save(ckpt_path)
    print(f"Fine-tuned checkpoint saved: {ckpt_path}")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Conditional DDPM training and evaluation")
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip training, load existing checkpoint")
    parser.add_argument("--skip-eval",  action="store_true",
                        help="Train only, skip evaluation and plots")
    # v2 ablation flags
    parser.add_argument("--tag", type=str, default="v1",
                        help="Output subdirectory tag, e.g. 'expA_df3' (default: v1)")
    parser.add_argument("--student-t-df", type=float, default=5.0,
                        help="Degrees of freedom for Student-t noise (default: 5.0)")
    parser.add_argument("--aux-sf-loss", action="store_true",
                        help="Enable auxiliary kurtosis+ACF loss")
    parser.add_argument("--decorr-reg", action="store_true",
                        help="Enable decorrelation regularizer (targets SF6)")
    parser.add_argument("--crisis-oversample", type=int, default=1,
                        help="Oversample crisis windows N times in training (default: 1)")
    parser.add_argument("--regime-weight", action="store_true",
                        help="Use inverse-frequency weighting across all regimes during training")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    # fine-tune flags
    parser.add_argument("--finetune-regime", type=str, default=None,
                        choices=["crisis", "calm", "normal"],
                        help="If set, fine-tune on only this regime's windows "
                             "(requires an existing checkpoint for --tag).")
    parser.add_argument("--finetune-lr", type=float, default=5e-5,
                        help="Learning rate for fine-tune phase (default: 5e-5)")
    parser.add_argument("--finetune-epochs", type=int, default=100,
                        help="Number of fine-tune epochs (default: 100)")
    args = parser.parse_args()

    # Apply CLI overrides to CFG
    CFG["tag"]              = args.tag
    CFG["student_t_df"]     = args.student_t_df
    CFG["use_aux_sf_loss"]  = args.aux_sf_loss
    CFG["use_decorr_reg"]   = args.decorr_reg
    CFG["crisis_oversample"] = args.crisis_oversample
    CFG["use_regime_weight"] = args.regime_weight
    CFG["seed"]              = args.seed

    print(f"Device: {DEVICE}")
    if DEVICE == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Tag: {CFG['tag']}  student_t_df={CFG['student_t_df']}"
          f"  aux_sf_loss={CFG['use_aux_sf_loss']}"
          f"  decorr_reg={CFG['use_decorr_reg']}"
          f"  crisis_oversample={CFG['crisis_oversample']}x")

    print("\nLoading data ...")
    windows, window_cond, window_regimes = load_data()
    n_features = windows.shape[2]

    if args.skip_train:
        ckpt_path = os.path.join(CFG["checkpoint_dir"], _ckpt_name())
        print(f"\nLoading checkpoint: {ckpt_path}")
        model = build_model(n_features)
        model.load(ckpt_path)
    else:
        model = train(windows, window_cond, window_regimes)

    # Optional fine-tune on a single regime
    if args.finetune_regime is not None:
        _run_finetune(model, windows, window_cond, window_regimes,
                      regime=args.finetune_regime,
                      lr=args.finetune_lr,
                      epochs=args.finetune_epochs)

    if not args.skip_eval:
        evaluate_conditional(model, windows, window_regimes)
        make_regime_plots(windows, window_regimes)


if __name__ == "__main__":
    main()

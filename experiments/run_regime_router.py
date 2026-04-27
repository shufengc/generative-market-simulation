"""
run_regime_router.py
====================
Best-of-breed checkpoint routing for regime-conditioned generation.

Loads the best checkpoint per regime and generates regime-conditioned samples:
  Crisis  -> expC_decorr    (5/6 SF, best crisis quality)
  Calm    -> expB_aux_sf    (3/6 SF, best calm quality)
  Normal  -> expD_oversample (5/6 SF, best crisis Disc 0.704)

Combines per-regime samples and evaluates jointly, reporting the "theoretical
maximum" SF count achievable without further retraining.

Outputs:
  experiments/results/conditional_ddpm_v2/regime_router/
    regime_router_results.json   -- per-regime and combined metrics
    regime_router_summary.png    -- bar chart comparison
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.models.ddpm_improved import ImprovedDDPM                   # noqa: E402
from src.data.regime_labels import get_regime_conditioning_vectors  # noqa: E402
from src.evaluation.metrics import (                                 # noqa: E402
    maximum_mean_discrepancy,
    discriminative_score,
    correlation_matrix_distance,
)
from src.evaluation.stylized_facts import run_all_tests             # noqa: E402

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR   = os.path.join(ROOT, "data")
CKPT_DIR   = os.path.join(ROOT, "checkpoints")
OUT_DIR    = os.path.join(ROOT, "experiments", "results", "conditional_ddpm_v2", "regime_router")
REGIMES    = ["crisis", "calm", "normal"]
REGIME_INT = {"crisis": 1, "calm": 2, "normal": 0}

# Best checkpoint per regime based on ablation sweep results
REGIME_CKPT = {
    "crisis": os.path.join(CKPT_DIR, "ddpm_conditional_expC_decorr.pt"),   # 5/6 SF
    "calm":   os.path.join(CKPT_DIR, "ddpm_conditional_expB_aux_sf.pt"),   # 3/6 SF
    "normal": os.path.join(CKPT_DIR, "ddpm_conditional_expD_oversample.pt"), # 5/6 SF, Disc 0.704
}

# Constructor flags matching each checkpoint (must mirror run_conditional_ddpm.py build_model)
REGIME_FLAGS = {
    "crisis": {"use_decorr_reg": True,   "use_aux_sf_loss": False, "student_t_df": 5.0},
    "calm":   {"use_decorr_reg": False,  "use_aux_sf_loss": True,  "student_t_df": 5.0},
    "normal": {"use_decorr_reg": False,  "use_aux_sf_loss": False, "student_t_df": 5.0},
}


def load_model(regime: str, n_features: int) -> ImprovedDDPM:
    flags  = REGIME_FLAGS[regime]
    ckpt   = REGIME_CKPT[regime]
    model  = ImprovedDDPM(
        n_features=n_features, seq_len=60, cond_dim=5, T=1000,
        base_channels=128, channel_mults=(1, 2, 4),
        use_vpred=True, use_student_t_noise=True,
        student_t_df=flags["student_t_df"],
        use_aux_sf_loss=flags["use_aux_sf_loss"],
        use_decorr_reg=flags["use_decorr_reg"],
        device=DEVICE,
    )
    model.load(ckpt)
    return model


def eval_one(real: np.ndarray, syn: np.ndarray, regime_name: str) -> dict:
    from scipy.stats import kurtosis, skew  # noqa: PLC0415
    n = min(len(real), len(syn), 1000)
    r, s = real[:n], syn[:n]

    mmd       = float(maximum_mean_discrepancy(r.reshape(n, -1), s.reshape(n, -1)))
    disc      = float(discriminative_score(r, s))
    corr_dist = float(correlation_matrix_distance(r, s))
    sf_list   = run_all_tests(s)
    sf_count  = sum(1 for x in sf_list if x.get("pass", False))
    sf_detail = {x.get("name", f"sf{i}"): x.get("pass", False)
                 for i, x in enumerate(sf_list)}
    syn_flat  = s.reshape(-1, s.shape[-1])
    real_flat = r.reshape(-1, r.shape[-1])
    return {
        "regime":              regime_name,
        "n":                   n,
        "sf_count":            sf_count,
        "sf_details":          sf_detail,
        "mmd":                 round(mmd, 6),
        "discriminative_score": round(disc, 4),
        "correlation_distance": round(float(corr_dist), 4),
        "syn_vol":             round(float(np.std(syn_flat, axis=0).mean()), 4),
        "real_vol":            round(float(np.std(real_flat, axis=0).mean()), 4),
        "syn_kurtosis":        round(float(kurtosis(s.flatten(), fisher=True)), 3),
        "real_kurtosis":       round(float(kurtosis(r.flatten(), fisher=True)), 3),
    }


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Device: {DEVICE}")
    if DEVICE == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    windows        = np.load(os.path.join(DATA_DIR, "windows.npy"))
    window_regimes = np.load(os.path.join(DATA_DIR, "window_regimes.npy"))
    n_features     = windows.shape[2]
    print(f"Real windows: {windows.shape}")

    regime_vecs = get_regime_conditioning_vectors()
    results = {}
    all_syn, all_real = [], []

    for regime_name in REGIMES:
        ckpt = REGIME_CKPT[regime_name]
        print(f"\n[{regime_name}] Loading {os.path.basename(ckpt)} ...")
        model = load_model(regime_name, n_features)

        print(f"[{regime_name}] Generating 1000 samples ...")
        syn = model.generate(
            n_samples=1000, use_ddim=True, ddim_steps=50,
            guidance_scale=1.0, ddim_eta=0.3,
            cond=regime_vecs[regime_name],
        )
        mask     = window_regimes == REGIME_INT[regime_name]
        real_reg = windows[mask]

        results[regime_name] = eval_one(real_reg, syn, regime_name)
        np.save(os.path.join(OUT_DIR, f"synthetic_{regime_name}.npy"), syn)

        all_syn.append(syn)
        all_real.append(real_reg[:1000])

        r = results[regime_name]
        print(f"  SF={r['sf_count']}/6  MMD={r['mmd']:.5f}  Disc={r['discriminative_score']:.3f}"
              f"  Vol={r['syn_vol']:.4f}/{r['real_vol']:.4f}  Kurt={r['syn_kurtosis']:.2f}/{r['real_kurtosis']:.2f}")

    # Combined evaluation across all regimes
    print("\n[combined] Evaluating all routed samples together ...")
    syn_combined  = np.concatenate(all_syn, axis=0)
    real_combined = np.concatenate(all_real, axis=0)
    results["combined"] = eval_one(real_combined, syn_combined, "combined")
    r = results["combined"]
    print(f"  SF={r['sf_count']}/6  MMD={r['mmd']:.5f}  Disc={r['discriminative_score']:.3f}")

    # Save JSON
    out_json = os.path.join(OUT_DIR, "regime_router_results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_json}")

    # Summary table
    print("\n" + "=" * 70)
    print("REGIME ROUTER SUMMARY")
    print("=" * 70)
    print(f"{'Regime':<12} {'Checkpoint':<35} {'SF':>5} {'Disc':>7} {'MMD':>9}")
    print("-" * 70)
    ckpt_labels = {
        "crisis": "expC_decorr",
        "calm":   "expB_aux_sf",
        "normal": "expD_oversample",
        "combined": "(all combined)",
    }
    for name in REGIMES + ["combined"]:
        r = results[name]
        print(f"{name:<12} {ckpt_labels[name]:<35} {r['sf_count']:>3}/6  {r['discriminative_score']:>7.3f}  {r['mmd']:>9.5f}")
    print("=" * 70)

    _make_plot(results)


def _make_plot(results: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    configs  = list(REGIMES) + ["combined"]
    sf_vals  = [results[c]["sf_count"] for c in configs]
    disc_vals = [results[c]["discriminative_score"] for c in configs]
    vol_syn  = [results[c]["syn_vol"] for c in configs if c in REGIMES]
    vol_real = [results[c]["real_vol"] for c in configs if c in REGIMES]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    x  = np.arange(len(configs)); w = 0.5
    colors = ["#e53935", "#43a047", "#1e88e5", "#7b1fa2"]

    ax = axes[0]
    bars = ax.bar(x, sf_vals, w, color=colors, edgecolor="white", alpha=0.85)
    for bar, val in zip(bars, sf_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f"{val}/6", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels([c.capitalize() for c in configs])
    ax.set_ylim(0, 7); ax.set_yticks(range(7))
    ax.set_title("Stylized Facts Passed — Regime Router", fontweight="bold")
    ax.set_ylabel("SF count (out of 6)")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    ax = axes[1]
    bars = ax.bar(x, disc_vals, w, color=colors, edgecolor="white", alpha=0.85)
    for bar, val in zip(bars, disc_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=10)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1.2, label="Random (0.5)")
    ax.set_xticks(x); ax.set_xticklabels([c.capitalize() for c in configs])
    ax.set_ylim(0, 1.15)
    ax.set_title("Discriminative Score — Regime Router", fontweight="bold")
    ax.set_ylabel("Disc score (lower = more realistic)")
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    fig.suptitle("Best-of-Breed Checkpoint Routing per Regime", fontweight="bold")
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "regime_router_summary.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()

"""
run_rescaling_ablation.py
=========================
Part 1 of the L3/L4 v2 iteration.

Post-hoc variance rescaling: loads the existing conditional DDPM checkpoint,
generates regime-conditioned samples, then rescales each asset's per-window
standard deviation to match the real data's distribution.

Key question: if the shape (tail ordering, correlations) is correct and only
the scale is wrong, does rescaling fix VaR coverage and stylized facts?

Outputs:
  experiments/results/conditional_ddpm_v2/rescaling/
    regime_eval_rescaled.json     -- per-regime SF/MMD/Disc before & after
    var_summary_rescaled.json     -- L4 VaR Kupiec before & after rescaling
    rescaling_comparison.png      -- bar chart: vol / kurtosis real vs raw vs rescaled
    var_comparison_rescaled.png   -- VaR/CVaR chart after rescaling

Usage:
    python3 experiments/run_rescaling_ablation.py
    python3 experiments/run_rescaling_ablation.py --n-paths 5000
"""

from __future__ import annotations

import argparse
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

DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR     = os.path.join(ROOT, "data")
CKPT_PATH    = os.path.join(ROOT, "checkpoints", "ddpm_conditional.pt")
OUT_DIR      = os.path.join(ROOT, "experiments", "results", "conditional_ddpm_v2", "rescaling")
REGIMES      = ["crisis", "calm", "normal"]
REGIME_INT   = {"crisis": 1, "calm": 2, "normal": 0}


# ─────────────────────────────────────────────────────────────
# Rescaling helpers
# ─────────────────────────────────────────────────────────────

def rescale_to_real(synthetic: np.ndarray, real: np.ndarray) -> np.ndarray:
    """
    Match the per-asset (per-feature) standard deviation of synthetic to real.

    Both arrays: (N, T, D). Rescaling is computed over the flattened (N*T) axis
    so it matches the marginal per-asset std of the real distribution.
    """
    syn = synthetic.copy()
    real_flat  = real.reshape(-1, real.shape[-1])   # (N*T, D)
    syn_flat   = syn.reshape(-1, syn.shape[-1])
    real_std   = real_flat.std(axis=0) + 1e-8       # (D,)
    syn_std    = syn_flat.std(axis=0)  + 1e-8       # (D,)
    scale      = real_std / syn_std                  # (D,)
    syn        = syn * scale[None, None, :]          # broadcast over (N, T, D)
    return syn


# ─────────────────────────────────────────────────────────────
# Per-regime evaluation (before and after rescaling)
# ─────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────
# VaR helpers (reused from var_backtest.py)
# ─────────────────────────────────────────────────────────────

def portfolio_returns(w: np.ndarray) -> np.ndarray:
    weights = np.ones(w.shape[-1]) / w.shape[-1]
    return (w * weights[None, None, :]).sum(-1)

def var_cvar(returns: np.ndarray, conf: float) -> tuple[float, float]:
    q    = np.quantile(returns, 1.0 - conf)
    tail = returns[returns <= q]
    return float(-q), float(-np.mean(tail)) if len(tail) else float(-q)

def kupiec(real_pnl, var_syn, conf):
    hit = float((-real_pnl > var_syn).sum() / len(real_pnl))
    return {"hit_rate": round(hit, 4), "nominal": 1.0 - conf,
            "kupiec_pass": abs(hit - (1.0 - conf)) < 0.02}


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-paths", type=int, default=5000)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Device: {DEVICE}")
    if DEVICE == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── Load data
    windows        = np.load(os.path.join(DATA_DIR, "windows.npy"))
    window_regimes = np.load(os.path.join(DATA_DIR, "window_regimes.npy"))
    n_features     = windows.shape[2]
    print(f"Real windows: {windows.shape}")

    # ── Load model
    print(f"\nLoading checkpoint: {CKPT_PATH}")
    model = ImprovedDDPM(
        n_features=n_features, seq_len=60, cond_dim=5, T=1000,
        base_channels=128, channel_mults=(1, 2, 4),
        use_vpred=True, use_student_t_noise=True, student_t_df=5.0,
        device=DEVICE,
    )
    model.load(CKPT_PATH)

    regime_vecs = get_regime_conditioning_vectors()

    # ── Per-regime: generate, evaluate before/after rescaling
    results = {}
    for regime_name in REGIMES:
        print(f"\n[{regime_name}] Generating 1000 samples ...")
        syn_raw = model.generate(
            n_samples=1000, use_ddim=True, ddim_steps=50,
            guidance_scale=2.0, ddim_eta=0.3,
            cond=regime_vecs[regime_name],
        )
        mask     = window_regimes == REGIME_INT[regime_name]
        real_reg = windows[mask]

        print(f"  Evaluating raw ...")
        before = eval_one(real_reg, syn_raw, regime_name)

        syn_resc = rescale_to_real(syn_raw, real_reg)
        print(f"  Evaluating rescaled ...")
        after  = eval_one(real_reg, syn_resc, regime_name)

        results[regime_name] = {"before": before, "after": after}
        np.save(os.path.join(OUT_DIR, f"synthetic_{regime_name}_rescaled.npy"), syn_resc)

        print(f"  Vol:  raw={before['syn_vol']:.4f}  rescaled={after['syn_vol']:.4f}  real={before['real_vol']:.4f}")
        print(f"  Kurt: raw={before['syn_kurtosis']:.2f}  rescaled={after['syn_kurtosis']:.2f}  real={before['real_kurtosis']:.2f}")
        print(f"  SF:   raw={before['sf_count']}/6  rescaled={after['sf_count']}/6")
        print(f"  Disc: raw={before['discriminative_score']:.3f}  rescaled={after['discriminative_score']:.3f}")
        print(f"  MMD:  raw={before['mmd']:.5f}  rescaled={after['mmd']:.5f}")

    # ── L4 VaR backtest: raw vs rescaled unconditional
    print(f"\n[L4] Generating {args.n_paths} unconditional paths ...")
    syn_uncond_raw = model.generate(
        n_samples=args.n_paths, use_ddim=True, ddim_steps=50,
        guidance_scale=1.0, ddim_eta=0.0, cond=None,
    )
    syn_uncond_resc = rescale_to_real(syn_uncond_raw, windows)
    real_pnl        = portfolio_returns(windows).sum(-1)
    pnl_raw         = portfolio_returns(syn_uncond_raw).sum(-1)
    pnl_resc        = portfolio_returns(syn_uncond_resc).sum(-1)

    var_summary = {}
    for conf in [0.95, 0.99]:
        key           = f"{conf:.0%}"
        var_r, cvar_r = var_cvar(real_pnl, conf)
        var_raw, cvar_raw   = var_cvar(pnl_raw,  conf)
        var_resc, cvar_resc = var_cvar(pnl_resc, conf)
        var_summary[key] = {
            "VaR_real":       round(var_r,    5),
            "VaR_raw":        round(var_raw,  5),
            "VaR_rescaled":   round(var_resc, 5),
            "CVaR_real":      round(cvar_r,   5),
            "CVaR_raw":       round(cvar_raw, 5),
            "CVaR_rescaled":  round(cvar_resc, 5),
            "kupiec_raw":     kupiec(real_pnl, var_raw,  conf),
            "kupiec_rescaled": kupiec(real_pnl, var_resc, conf),
            "VaR_err_raw_pct":   round(abs(var_raw  - var_r) / (abs(var_r) + 1e-8) * 100, 2),
            "VaR_err_resc_pct":  round(abs(var_resc - var_r) / (abs(var_r) + 1e-8) * 100, 2),
        }
        k_raw  = var_summary[key]["kupiec_raw"]
        k_resc = var_summary[key]["kupiec_rescaled"]
        print(f"\n{key}:")
        print(f"  VaR  real={var_r:.4f}  raw={var_raw:.4f} ({var_summary[key]['VaR_err_raw_pct']:.1f}% err)"
              f"  rescaled={var_resc:.4f} ({var_summary[key]['VaR_err_resc_pct']:.1f}% err)")
        print(f"  Kupiec raw: hit={k_raw['hit_rate']:.4f} {'PASS' if k_raw['kupiec_pass'] else 'FAIL'}")
        print(f"  Kupiec rescaled: hit={k_resc['hit_rate']:.4f} {'PASS' if k_resc['kupiec_pass'] else 'FAIL'}")

    # ── Save JSONs
    out_regime = os.path.join(OUT_DIR, "regime_eval_rescaled.json")
    out_var    = os.path.join(OUT_DIR, "var_summary_rescaled.json")
    with open(out_regime, "w") as f:
        json.dump(results, f, indent=2)
    with open(out_var, "w") as f:
        json.dump(var_summary, f, indent=2)
    print(f"\nSaved: {out_regime}")
    print(f"Saved: {out_var}")

    # ── Summary table
    print("\n" + "=" * 80)
    print("RESCALING SUMMARY")
    print("=" * 80)
    print(f"{'Regime':<10} {'SF_raw':>7} {'SF_resc':>8} {'Disc_raw':>9} {'Disc_resc':>10}"
          f" {'Vol_raw':>8} {'Vol_resc':>9} {'Vol_real':>9}")
    print("-" * 80)
    for rn in REGIMES:
        b = results[rn]["before"]
        a = results[rn]["after"]
        print(f"{rn:<10} {b['sf_count']:>5}/6  {a['sf_count']:>6}/6  "
              f"{b['discriminative_score']:>9.3f}  {a['discriminative_score']:>10.3f}"
              f"  {b['syn_vol']:>8.4f}  {a['syn_vol']:>9.4f}  {b['real_vol']:>9.4f}")
    print("=" * 80)

    # ── Plots
    if not args.no_plot:
        _make_plots(results, var_summary, real_pnl, pnl_raw, pnl_resc)


def _make_plots(results, var_summary, real_pnl, pnl_raw, pnl_resc):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    c_map = {"crisis": "#e53935", "calm": "#43a047", "normal": "#1e88e5"}

    # Figure 1: Vol and kurtosis before/after by regime
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(len(REGIMES)); w = 0.25

    ax = axes[0]
    real_vols = [results[r]["before"]["real_vol"]   for r in REGIMES]
    raw_vols  = [results[r]["before"]["syn_vol"]    for r in REGIMES]
    resc_vols = [results[r]["after"]["syn_vol"]     for r in REGIMES]
    ax.bar(x - w, real_vols, w, label="Real",     color="#546e7a", edgecolor="white")
    ax.bar(x,     raw_vols,  w, label="Raw DDPM", color="#90a4ae", edgecolor="white")
    ax.bar(x + w, resc_vols, w, label="Rescaled", color=[c_map[r] for r in REGIMES], edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels([r.capitalize() for r in REGIMES])
    ax.set_title("Volatility: Real vs Raw vs Rescaled", fontweight="bold")
    ax.set_ylabel("Mean std across assets"); ax.legend()
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    ax = axes[1]
    real_k = [results[r]["before"]["real_kurtosis"]  for r in REGIMES]
    raw_k  = [results[r]["before"]["syn_kurtosis"]   for r in REGIMES]
    resc_k = [results[r]["after"]["syn_kurtosis"]    for r in REGIMES]
    ax.bar(x - w, real_k, w, label="Real",     color="#546e7a", edgecolor="white")
    ax.bar(x,     raw_k,  w, label="Raw DDPM", color="#90a4ae", edgecolor="white")
    ax.bar(x + w, resc_k, w, label="Rescaled", color=[c_map[r] for r in REGIMES], edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels([r.capitalize() for r in REGIMES])
    ax.set_title("Excess Kurtosis: Real vs Raw vs Rescaled", fontweight="bold")
    ax.set_ylabel("Excess kurtosis"); ax.legend()
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    fig.suptitle("Post-Hoc Rescaling Effect on Distribution Shape", fontweight="bold")
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "rescaling_comparison.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out}")

    # Figure 2: VaR comparison (raw vs rescaled vs real)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, conf in zip(axes, [0.95, 0.99]):
        key = f"{conf:.0%}"
        v = var_summary[key]
        labels = ["VaR Real", "VaR Raw", "VaR Rescaled", "CVaR Real", "CVaR Raw", "CVaR Rescaled"]
        vals   = [v["VaR_real"], v["VaR_raw"], v["VaR_rescaled"],
                  v["CVaR_real"], v["CVaR_raw"], v["CVaR_rescaled"]]
        colors = ["#546e7a", "#90a4ae", "#1e88e5",
                  "#546e7a", "#90a4ae", "#1e88e5"]
        bars = ax.bar(labels, vals, color=colors, edgecolor="white", alpha=0.85)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + max(vals) * 0.015,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)
        k_r = v["kupiec_rescaled"]
        kupiec_col = "#2e7d32" if k_r["kupiec_pass"] else "#c62828"
        ax.text(0.5, 0.97,
                f"Rescaled Kupiec: hit={k_r['hit_rate']:.3f}  nominal={k_r['nominal']:.2f}  "
                + ("PASS" if k_r["kupiec_pass"] else "FAIL"),
                transform=ax.transAxes, ha="center", va="top",
                fontsize=9, color=kupiec_col,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=kupiec_col))
        ax.set_title(f"{key} — VaR/CVaR After Rescaling", fontweight="bold")
        ax.set_ylabel("Loss (positive = loss)")
        ax.tick_params(axis="x", rotation=15)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.suptitle("L4: VaR Backtest — Raw vs Post-Hoc Rescaled", fontweight="bold")
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "var_comparison_rescaled.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()

"""
run_rescaling_ablation.py
=========================
Part 1 of the L3/L4 v2 iteration.

Post-hoc rescaling: loads the existing conditional DDPM checkpoint,
generates regime-conditioned samples, then rescales to match the real
marginal distribution using one of four modes.

Modes:
  std             -- match per-asset standard deviation only (original)
  quantile        -- full quantile mapping per asset (matches kurtosis by construction)
  cornish-fisher  -- Cornish-Fisher moment matching (mean, std, skew, kurtosis)
  regime-quantile -- regime-stratified quantile mapping (novel; avoids cross-regime contamination)

Outputs:
  experiments/results/conditional_ddpm_v2/{rescaling|moment_matching}/
    regime_eval_rescaled.json     -- per-regime SF/MMD/Disc before & after
    var_summary_rescaled.json     -- L4 VaR Kupiec before & after rescaling
    rescaling_comparison.png      -- bar chart: vol / kurtosis real vs raw vs rescaled
    var_comparison_rescaled.png   -- VaR/CVaR chart after rescaling

Usage:
    python3 experiments/run_rescaling_ablation.py
    python3 experiments/run_rescaling_ablation.py --n-paths 5000
    python3 experiments/run_rescaling_ablation.py --mode quantile --n-paths 5000
    python3 experiments/run_rescaling_ablation.py --mode cornish-fisher --n-paths 5000
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
OUT_DIR_STD  = os.path.join(ROOT, "experiments", "results", "conditional_ddpm_v2", "rescaling")
OUT_DIR_QM   = os.path.join(ROOT, "experiments", "results", "conditional_ddpm_v2", "moment_matching")
OUT_DIR_CF   = os.path.join(ROOT, "experiments", "results", "conditional_ddpm_v2", "moment_matching_cf")
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


def quantile_map(synthetic: np.ndarray, real: np.ndarray,
                  rng: np.random.Generator | None = None) -> np.ndarray:
    """
    Per-asset quantile mapping: map each synthetic value to the corresponding
    real quantile via rank-based interpolation.

    Size-mismatch handling: when n_real > n_syn, subsample real values to n_syn
    before computing the mapping grid.  Without this, compressed synthetic
    extreme values (vol compression) get mapped to real extremes, artificially
    amplifying tails (observed as calm kurtosis 8.85 vs real 5.49).

    Both arrays: (N, T, D).  Mapping is computed over the flattened (N*T) axis.
    """
    if rng is None:
        rng = np.random.default_rng(seed=42)
    N, T, D = synthetic.shape
    syn = synthetic.copy()
    for i in range(D):
        syn_flat  = syn[:, :, i].flatten()
        real_flat = real[:, :, i].flatten()
        n_syn = len(syn_flat)
        n_real = len(real_flat)
        # Subsample real to syn size to avoid tail amplification from size mismatch
        if n_real > n_syn:
            idx = rng.choice(n_real, size=n_syn, replace=False)
            real_for_map = np.sort(real_flat[idx])
        else:
            real_for_map = np.sort(real_flat)
        n_map = len(real_for_map)
        syn_ranks = np.argsort(np.argsort(syn_flat))
        quantiles = syn_ranks / (n_syn - 1)
        mapped = np.interp(quantiles, np.linspace(0, 1, n_map), real_for_map)
        syn[:, :, i] = mapped.reshape(N, T)
    return syn


def cornish_fisher_match(synthetic: np.ndarray, real: np.ndarray) -> np.ndarray:
    """
    Cornish-Fisher moment matching: standardise synthetic, apply Cornish-Fisher
    expansion to match real skewness and excess kurtosis, then rescale to real
    mean and std. Operates per-asset over the flattened (N*T) axis.

    CF expansion: z_cf = z + (s/6)*(z²-1) + (k/24)*(z³-3z) - (s²/36)*(2z³-5z)
    where s = target skew, k = target excess kurtosis.
    """
    from scipy.stats import skew as sp_skew, kurtosis as sp_kurt  # noqa: PLC0415
    N, T, D = synthetic.shape
    syn = synthetic.copy()
    for i in range(D):
        syn_flat  = syn[:, :, i].flatten()
        real_flat = real[:, :, i].flatten()
        r_mean = float(real_flat.mean())
        r_std  = float(real_flat.std()) + 1e-8
        r_skew = float(sp_skew(real_flat))
        r_kurt = float(sp_kurt(real_flat, fisher=True))  # excess kurtosis
        s_mean = float(syn_flat.mean())
        s_std  = float(syn_flat.std()) + 1e-8
        z = (syn_flat - s_mean) / s_std
        s, k = r_skew, r_kurt
        z_cf = (z
                + (s / 6) * (z**2 - 1)
                + (k / 24) * (z**3 - 3 * z)
                - (s**2 / 36) * (2 * z**3 - 5 * z))
        # Rescale to real mean/std
        matched = z_cf * r_std + r_mean
        syn[:, :, i] = matched.reshape(N, T)
    return syn


def apply_rescaling(synthetic: np.ndarray, real: np.ndarray, mode: str) -> np.ndarray:
    """Dispatch to the selected rescaling mode."""
    if mode == "std":
        return rescale_to_real(synthetic, real)
    elif mode == "quantile":
        return quantile_map(synthetic, real)
    elif mode == "cornish-fisher":
        return cornish_fisher_match(synthetic, real)
    else:
        raise ValueError(f"Unknown rescaling mode: {mode}")


def regime_quantile_map(
    synthetic: np.ndarray,
    real_windows: np.ndarray,
    real_regimes: np.ndarray,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Regime-conditional quantile mapping for unconditional synthetic paths.

    Standard (flat) QM mixes regime distributions: calm-like synthetic windows
    get mapped against crisis-regime real quantiles, artificially amplifying tails
    (observed as calm kurtosis 8.72 vs real 5.49, driving 95% Kupiec failure).

    This function:
    1. Estimates each synthetic window's regime by its per-window realized vol.
    2. Derives vol-percentile thresholds from the *real* data distribution
       (no training label leakage — only the real data's own statistics).
    3. Applies quantile_map() per group using only the matching real-regime windows.

    Classification uses the 30th and 70th percentiles of real per-window vol,
    chosen to roughly match crisis/calm/normal proportions (14%/40%/46%).
    """
    if rng is None:
        rng = np.random.default_rng(seed=42)

    # ── Real data vol thresholds (derived from real distribution only)
    real_per_window_vol = real_windows.reshape(real_windows.shape[0], -1).std(axis=1)
    thresh_low  = float(np.percentile(real_per_window_vol, 30))   # calm  = below 30th pct
    thresh_high = float(np.percentile(real_per_window_vol, 70))   # crisis = above 70th pct

    # ── Classify synthetic windows by realized vol
    syn_vol = synthetic.reshape(synthetic.shape[0], -1).std(axis=1)
    mask_crisis = syn_vol >= thresh_high
    mask_calm   = syn_vol <  thresh_low
    mask_normal = ~mask_crisis & ~mask_calm

    print(f"  Regime-QM window counts: crisis={mask_crisis.sum()}  "
          f"calm={mask_calm.sum()}  normal={mask_normal.sum()}  "
          f"(vol thresholds: low={thresh_low:.4f} high={thresh_high:.4f})")

    # ── Real windows per regime
    real_by_regime = {
        "crisis": real_windows[real_regimes == 1],   # crisis=1
        "calm":   real_windows[real_regimes == 2],   # calm=2
        "normal": real_windows[real_regimes == 0],   # normal=0
    }

    result = synthetic.copy()

    for mask, regime_key in [
        (mask_crisis, "crisis"),
        (mask_calm,   "calm"),
        (mask_normal, "normal"),
    ]:
        if mask.sum() == 0:
            continue
        syn_group  = synthetic[mask]
        real_group = real_by_regime[regime_key]
        if len(real_group) == 0:
            print(f"  WARNING: no real windows for regime '{regime_key}', skipping QM")
            continue
        result[mask] = quantile_map(syn_group, real_group, rng=rng)

    return result


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

def kupiec(real_pnl: np.ndarray, var_syn: float, conf: float) -> dict:
    """Kupiec (1995) LR test: LR ~ chi2(1) under correct coverage null."""
    from scipy.stats import chi2 as _chi2
    n = len(real_pnl)
    n_exc = int((-real_pnl > var_syn).sum())
    p_hat = n_exc / n
    p_0   = 1.0 - conf
    if n_exc == 0 or n_exc == n:
        lr_stat, p_value = 0.0, 1.0
    else:
        lr_stat = 2.0 * (
            n_exc * np.log(p_hat / p_0)
            + (n - n_exc) * np.log((1.0 - p_hat) / (1.0 - p_0))
        )
        p_value = float(1.0 - _chi2.cdf(lr_stat, df=1))
    return {
        "hit_rate":    round(p_hat, 4),
        "nominal":     p_0,
        "kupiec_pass": p_value > 0.05,
        "p_value":     round(p_value, 4),
        "lr_stat":     round(lr_stat, 4),
    }


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-paths", type=int, default=5000)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument(
        "--mode",
        choices=["std", "quantile", "cornish-fisher", "regime-quantile"],
        default="std",
        help="Rescaling mode: std (std-match only), quantile (full quantile map), "
             "cornish-fisher (moment matching via CF expansion), "
             "regime-quantile (stratified QM per synthetic regime). Default: std",
    )
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Override checkpoint path (default: ddpm_conditional.pt)")
    parser.add_argument("--aux-sf-loss", action="store_true",
                        help="Load model with aux_sf_loss=True (needed for expF and later)")
    parser.add_argument("--out-tag", type=str, default=None,
                        help="Extra tag for output directory, e.g. expF_balanced")
    args = parser.parse_args()

    out_base = OUT_DIR_STD if args.mode == "std" else (OUT_DIR_CF if args.mode == "cornish-fisher" else OUT_DIR_QM)
    if args.mode == "regime-quantile":
        out_base = os.path.join(ROOT, "experiments", "results", "conditional_ddpm_v2", "moment_matching_rqm")
    if args.out_tag:
        out_base = out_base.rstrip("/") + f"_{args.out_tag}"
    OUT_DIR = out_base
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Device: {DEVICE}  Mode: {args.mode}  Output: {OUT_DIR}")
    if DEVICE == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── Load data
    windows        = np.load(os.path.join(DATA_DIR, "windows.npy"))
    window_regimes = np.load(os.path.join(DATA_DIR, "window_regimes.npy"))
    n_features     = windows.shape[2]
    print(f"Real windows: {windows.shape}")

    # ── Load model
    ckpt_path = args.ckpt or CKPT_PATH
    print(f"\nLoading checkpoint: {ckpt_path}")
    model = ImprovedDDPM(
        n_features=n_features, seq_len=60, cond_dim=5, T=1000,
        base_channels=128, channel_mults=(1, 2, 4),
        use_vpred=True, use_student_t_noise=True, student_t_df=5.0,
        use_aux_sf_loss=args.aux_sf_loss,
        device=DEVICE,
    )
    model.load(ckpt_path)

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

        syn_resc = apply_rescaling(syn_raw, real_reg,
                                   "quantile" if args.mode == "regime-quantile" else args.mode)
        print(f"  Evaluating {args.mode}-rescaled ...")
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
    if args.mode == "regime-quantile":
        print(f"[L4] Applying regime-conditional quantile mapping ...")
        rng = np.random.default_rng(seed=42)
        syn_uncond_resc = regime_quantile_map(syn_uncond_raw, windows, window_regimes, rng=rng)
    else:
        syn_uncond_resc = apply_rescaling(syn_uncond_raw, windows, args.mode)
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
    print(f"RESCALING SUMMARY  (mode={args.mode})")
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
        _make_plots(results, var_summary, real_pnl, pnl_raw, pnl_resc, OUT_DIR, args.mode)


def _make_plots(results, var_summary, real_pnl, pnl_raw, pnl_resc, out_dir, mode="std"):
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
    out = os.path.join(out_dir, "rescaling_comparison.png")
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
        ax.set_title(f"{key} — VaR/CVaR After {mode} Rescaling", fontweight="bold")
        ax.set_ylabel("Loss (positive = loss)")
        ax.tick_params(axis="x", rotation=15)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.suptitle(f"L4: VaR Backtest — Raw vs Post-Hoc {mode} Rescaled", fontweight="bold")
    fig.tight_layout()
    out = os.path.join(out_dir, "var_comparison_rescaled.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()

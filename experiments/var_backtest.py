"""
var_backtest.py
===============
L4 Downstream Utility — VaR/CVaR Backtesting

Uses the trained conditional DDPM to generate synthetic paths and validates
them against historical portfolio VaR/CVaR via Kupiec coverage test.

Usage:
    python experiments/var_backtest.py
    python experiments/var_backtest.py --n-paths 5000
    python experiments/var_backtest.py --no-plot
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Optional

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

_DEFAULT_RESULTS_DIR = os.path.join(ROOT, "experiments", "results", "var_backtest")
DATA_DIR       = os.path.join(ROOT, "data")
CHECKPOINT_DIR = os.path.join(ROOT, "checkpoints")
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
CONFIDENCE_LEVELS = [0.95, 0.99]


# ─────────────────────────────────────────────────────────────
# Portfolio / VaR utilities
# ─────────────────────────────────────────────────────────────

def portfolio_returns(windows: np.ndarray,
                      weights: Optional[np.ndarray] = None) -> np.ndarray:
    """Equal-weight portfolio: (N, T, D) -> (N, T)."""
    if weights is None:
        weights = np.ones(windows.shape[-1]) / windows.shape[-1]
    return (windows * weights[None, None, :]).sum(axis=-1)


def compute_window_pnl(windows: np.ndarray) -> np.ndarray:
    """Cumulative 60-day portfolio PnL per window: (N, T, D) -> (N,)."""
    return portfolio_returns(windows).sum(axis=-1)


def var_cvar(returns: np.ndarray, confidence: float = 0.95) -> tuple[float, float]:
    q    = np.quantile(returns, 1.0 - confidence)
    var  = float(-q)
    tail = returns[returns <= q]
    cvar = float(-np.mean(tail)) if len(tail) > 0 else var
    return var, cvar


def hit_rate(real_pnl: np.ndarray, var_synthetic: float) -> float:
    """Fraction of real losses exceeding synthetic VaR (should ~ 1-confidence)."""
    return float((-real_pnl > var_synthetic).sum() / len(real_pnl))


def kupiec_lr(real_pnl: np.ndarray, var_syn: float, conf: float) -> dict:
    """Kupiec (1995) LR unconditional coverage test.  LR ~ chi2(1) under H0."""
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


def sharpe_ratio(returns: np.ndarray, risk_free: float = 0.0) -> float:
    excess = returns - risk_free / 252
    return float(np.mean(excess) / np.std(excess) * np.sqrt(252)) if np.std(excess) > 1e-8 else 0.0


def sharpe_distribution(windows: np.ndarray) -> np.ndarray:
    port = portfolio_returns(windows)
    return np.array([sharpe_ratio(port[i]) for i in range(len(port))])


def momentum_strategy_pnl(windows: np.ndarray, lookback: int = 20) -> np.ndarray:
    port = portfolio_returns(windows)
    signal = np.sign(port[:, :lookback].mean(axis=1))
    return (signal[:, None] * port[:, lookback:]).sum(axis=-1)


# ─────────────────────────────────────────────────────────────
# Generate synthetic samples
# ─────────────────────────────────────────────────────────────

def load_or_generate(n_paths: int, ckpt_override: str = None) -> np.ndarray:
    """Load cached samples or generate from DDPM checkpoint."""
    from src.models.ddpm_improved import ImprovedDDPM  # noqa: PLC0415

    if ckpt_override:
        ckpt_path = ckpt_override
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    else:
        ckpt_candidates = [
            os.path.join(CHECKPOINT_DIR, "ddpm_conditional.pt"),
            os.path.join(CHECKPOINT_DIR, "ddpm_improved.pt"),
        ]
        ckpt_path = next((p for p in ckpt_candidates if os.path.exists(p)), None)
        if ckpt_path is None:
            raise FileNotFoundError(
                "No DDPM checkpoint found in checkpoints/. "
                "Run run_conditional_ddpm.py first."
            )
    print(f"  Loading checkpoint: {ckpt_path}")

    # Read n_features from data
    windows    = np.load(os.path.join(DATA_DIR, "windows.npy"))
    n_features = windows.shape[2]

    model = ImprovedDDPM(
        n_features=n_features,
        seq_len=60,
        cond_dim=5,
        T=1000,
        base_channels=128,
        channel_mults=(1, 2, 4),
        use_vpred=True,
        use_student_t_noise=True,
        student_t_df=5.0,
        device=DEVICE,
    )
    model.load(ckpt_path)

    print(f"  Generating {n_paths} paths (DDIM 50 steps, guidance=1.0, no conditioning) ...")
    synthetic = model.generate(
        n_samples=n_paths,
        use_ddim=True,
        ddim_steps=50,
        guidance_scale=1.0,
        ddim_eta=0.0,
    )
    return synthetic


# ─────────────────────────────────────────────────────────────
# Main backtest
# ─────────────────────────────────────────────────────────────

def run_backtest(real_windows: np.ndarray, synthetic_windows: np.ndarray,
                 n_paths: int) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    idx = np.random.permutation(len(synthetic_windows))
    syn = synthetic_windows[idx[:n_paths]]

    real_pnl    = compute_window_pnl(real_windows)
    syn_pnl     = compute_window_pnl(syn)
    real_sharpe = sharpe_distribution(real_windows)
    syn_sharpe  = sharpe_distribution(syn)

    results: dict = {
        "n_real": len(real_windows),
        "n_synthetic": n_paths,
        "checkpoint": "ddpm_conditional.pt",
        "device": DEVICE,
        "var_cvar": {},
        "hit_rates": {},
        "sharpe": {},
        "momentum_strategy": {},
    }

    for conf in CONFIDENCE_LEVELS:
        var_real, cvar_real = var_cvar(real_pnl, conf)
        var_syn,  cvar_syn  = var_cvar(syn_pnl,  conf)
        nominal = 1.0 - conf
        key = f"{conf:.0%}"
        results["var_cvar"][key] = {
            "VaR_real":              round(var_real, 5),
            "VaR_synthetic":         round(var_syn,  5),
            "CVaR_real":             round(cvar_real, 5),
            "CVaR_synthetic":        round(cvar_syn,  5),
        "VaR_relative_error_pct": round(abs(var_syn - var_real) / (abs(var_real) + 1e-8) * 100, 2),
        }
        results["hit_rates"][key] = kupiec_lr(real_pnl, var_syn, conf)

    results["sharpe"] = {
        "real_mean":    round(float(np.mean(real_sharpe)), 3),
        "real_std":     round(float(np.std(real_sharpe)),  3),
        "syn_mean":     round(float(np.mean(syn_sharpe)),  3),
        "syn_std":      round(float(np.std(syn_sharpe)),   3),
        "mean_abs_diff": round(abs(float(np.mean(real_sharpe)) - float(np.mean(syn_sharpe))), 3),
    }

    real_mom = momentum_strategy_pnl(real_windows)
    syn_mom  = momentum_strategy_pnl(syn)
    n_min = min(len(real_mom), len(syn_mom))
    results["momentum_strategy"] = {
        "real_mean_pnl": round(float(np.mean(real_mom)), 4),
        "syn_mean_pnl":  round(float(np.mean(syn_mom)),  4),
        "real_sharpe":   round(sharpe_ratio(real_mom),   3),
        "syn_sharpe":    round(sharpe_ratio(syn_mom),    3),
        "pnl_rank_corr": round(float(np.corrcoef(
            np.sort(real_mom[:n_min]),
            np.sort(syn_mom[:n_min]))[0, 1]), 3),
    }

    return results, real_pnl, syn_pnl, real_sharpe, syn_sharpe


def print_summary(results: dict) -> None:
    print("\n" + "=" * 70)
    print("VaR / CVaR BACKTEST SUMMARY")
    print("=" * 70)
    for key, v in results["var_cvar"].items():
        h = results["hit_rates"][key]
        verdict = "PASS" if h["kupiec_pass"] else "FAIL"
        print(f"\n{key} Confidence Level:")
        print(f"  VaR    Real={v['VaR_real']:>9.5f}  Synthetic={v['VaR_synthetic']:>9.5f}  "
              f"RelErr={v['VaR_relative_error_pct']:>5.1f}%")
        print(f"  CVaR   Real={v['CVaR_real']:>9.5f}  Synthetic={v['CVaR_synthetic']:>9.5f}")
        print(f"  Kupiec LR: hit={h['hit_rate']:.4f}  nominal={h['nominal']:.4f}  "
              f"p={h.get('p_value', 'n/a'):.4f}  {verdict}")

    s = results["sharpe"]
    print(f"\nSharpe Ratio:  Real mean={s['real_mean']:.3f} std={s['real_std']:.3f}  "
          f"Syn mean={s['syn_mean']:.3f} std={s['syn_std']:.3f}  "
          f"mean_abs_diff={s['mean_abs_diff']:.3f}")

    m = results["momentum_strategy"]
    print(f"Momentum:  Real PnL={m['real_mean_pnl']:.4f}  Syn PnL={m['syn_mean_pnl']:.4f}  "
          f"rank_corr={m['pnl_rank_corr']:.3f}")
    print("=" * 70)

    all_pass = all(v["kupiec_pass"] for v in results["hit_rates"].values())
    if all_pass:
        print("\nL4 VERDICT: Kupiec PASSES at all confidence levels.")
        print("  Synthetic data is calibrated for risk estimation workflows.")
    else:
        failed = [k for k, v in results["hit_rates"].items() if not v["kupiec_pass"]]
        print(f"\nL4 VERDICT: Kupiec FAILS at {failed}.")
        print("  VaR may be over- or under-estimated for those confidence levels.")


def make_plots(results: dict, real_pnl, syn_pnl, real_sharpe, syn_sharpe,
               results_dir: str = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _results_dir = results_dir or _DEFAULT_RESULTS_DIR
    os.makedirs(_results_dir, exist_ok=True)

    # Figure 1: VaR/CVaR bar chart
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, conf in zip(axes, [0.95, 0.99]):
        key = f"{conf:.0%}"
        v = results["var_cvar"][key]
        h = results["hit_rates"][key]
        labels = ["VaR (Real)", "VaR (Syn)", "CVaR (Real)", "CVaR (Syn)"]
        values = [v["VaR_real"], v["VaR_synthetic"], v["CVaR_real"], v["CVaR_synthetic"]]
        clrs   = ["#546e7a", "#1e88e5", "#546e7a", "#1e88e5"]
        bars = ax.bar(labels, values, color=clrs, edgecolor="white", alpha=0.85)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + max(values) * 0.015,
                    f"{val:.4f}", ha="center", va="bottom", fontsize=9)
        kupiec_col = "#2e7d32" if h["kupiec_pass"] else "#c62828"
        pv_str = f"p={h.get('p_value', 0.0):.3f}"
        ax.text(0.5, 0.97,
                f"Kupiec LR: hit={h['hit_rate']:.3f} nominal={h['nominal']:.2f} {pv_str}  "
                + ("PASS" if h["kupiec_pass"] else "FAIL"),
                transform=ax.transAxes, ha="center", va="top",
                fontsize=9, color=kupiec_col,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=kupiec_col))
        ax.set_title(f"{key} VaR/CVaR — Real vs Synthetic", fontweight="bold")
        ax.set_ylabel("Loss (positive = loss)")
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.tick_params(axis="x", rotation=10)

    fig.suptitle("L4 Downstream Validation: VaR/CVaR Backtest", fontweight="bold")
    fig.tight_layout()
    out = os.path.join(_results_dir, "var_comparison.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out}")

    # Figure 2: PnL and Sharpe distributions
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.hist(real_pnl, bins=60, alpha=0.55, density=True, color="#546e7a", label="Real")
    ax.hist(syn_pnl,  bins=60, alpha=0.55, density=True, color="#1e88e5", label="Synthetic")
    for conf, ls in zip([0.95, 0.99], ["--", ":"]):
        key = f"{conf:.0%}"
        ax.axvline(-results["var_cvar"][key]["VaR_real"],      color="#546e7a", linestyle=ls,
                   linewidth=1.8, label=f"VaR{key} Real")
        ax.axvline(-results["var_cvar"][key]["VaR_synthetic"], color="#1e88e5", linestyle=ls,
                   linewidth=1.8, label=f"VaR{key} Syn")
    ax.set_xlabel("60-day Portfolio Return"); ax.set_ylabel("Density")
    ax.set_title("Portfolio PnL Distribution", fontweight="bold")
    ax.legend(fontsize=7)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    ax = axes[1]
    ax.hist(real_sharpe, bins=50, alpha=0.55, density=True, color="#546e7a", label="Real")
    ax.hist(syn_sharpe,  bins=50, alpha=0.55, density=True, color="#1e88e5", label="Synthetic")
    s = results["sharpe"]
    ax.axvline(s["real_mean"], color="#546e7a", linestyle="--", linewidth=2,
               label=f"Real mean={s['real_mean']:.2f}")
    ax.axvline(s["syn_mean"],  color="#1e88e5", linestyle="--", linewidth=2,
               label=f"Syn  mean={s['syn_mean']:.2f}")
    ax.set_xlabel("Annualised Sharpe"); ax.set_ylabel("Density")
    ax.set_title("Sharpe Ratio Distribution", fontweight="bold")
    ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    fig.suptitle("L4 Downstream Validation: PnL and Sharpe", fontweight="bold")
    fig.tight_layout()
    out = os.path.join(_results_dir, "pnl_sharpe_distribution.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out}")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="L4 VaR/CVaR backtest")
    parser.add_argument("--n-paths", type=int, default=5000)
    parser.add_argument("--no-plot",   action="store_true")
    parser.add_argument("--results-dir", type=str, default=None,
                        help="Output directory for results (default: experiments/results/var_backtest)")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Path to DDPM checkpoint (overrides default search)")
    args = parser.parse_args()

    RESULTS_DIR = args.results_dir or _DEFAULT_RESULTS_DIR
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"Device: {DEVICE}")
    if DEVICE == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load real data
    print("\nLoading real windows ...")
    real_windows = np.load(os.path.join(DATA_DIR, "windows.npy"))
    print(f"  Real windows: {real_windows.shape}")

    # Generate synthetic
    print("\nGenerating synthetic samples ...")
    synthetic = load_or_generate(args.n_paths, ckpt_override=args.ckpt)
    print(f"  Synthetic: {synthetic.shape}")

    results, real_pnl, syn_pnl, real_sharpe, syn_sharpe = run_backtest(
        real_windows, synthetic, args.n_paths
    )

    print_summary(results)

    out_json = os.path.join(RESULTS_DIR, "var_summary.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_json}")

    if not args.no_plot:
        make_plots(results, real_pnl, syn_pnl, real_sharpe, syn_sharpe,
                   results_dir=RESULTS_DIR)


if __name__ == "__main__":
    main()

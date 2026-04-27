"""
generate_v2_figures.py
======================
Generate three v2 presentation figures from ablation sweep JSON results.

Figures produced:
  presentation_assets/v2_ablation_comparison.png
      Grouped bar chart: v1 baseline + 5 ablation configs (A-E) + moment-matched
      + regime-routed, showing per-regime SF count and Discriminative Score.

  presentation_assets/v2_moment_matching.png
      Before/after bar chart for vol and kurtosis comparing raw, std-rescaled,
      and quantile-mapped synthetic data against real.

  presentation_assets/v2_guidance_sweep.png
      Line plot of crisis Disc score and crisis vol vs guidance scale (1.0-7.0)
      for the Exp B checkpoint.

All data pulled from experiments/results/conditional_ddpm_v2/ JSON files.
Run after Priority 1 (moment matching) and Priority 2 (regime router) complete.
"""

from __future__ import annotations

import json
import os

import numpy as np

ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V2_DIR     = os.path.join(ROOT, "experiments", "results", "conditional_ddpm_v2")
ASSETS_DIR = os.path.join(ROOT, "presentation_assets")
REGIMES    = ["crisis", "calm", "normal"]

os.makedirs(ASSETS_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────
# Data loading helpers
# ─────────────────────────────────────────────────────────────

def _load_json(path: str) -> dict | None:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    print(f"  [WARN] Not found: {path}")
    return None


def load_ablation_data() -> dict[str, dict]:
    """
    Load per-regime SF count and Disc score for each config.
    Returns dict: config_label -> {regime -> {sf, disc, vol}}.
    """
    configs = {}

    # v1 baseline (from the main conditional_ddpm results)
    v1_path = os.path.join(ROOT, "experiments", "results", "conditional_ddpm", "conditional_eval_results.json")
    d = _load_json(v1_path)
    if d:
        configs["v1\n(baseline)"] = {
            r: {"sf": d[r]["sf_count"],
                "disc": d[r]["discriminative_score"],
                "vol": d[r]["vol_mean_across_assets"]}
            for r in REGIMES if r in d
        }

    # Ablation configs A-E
    ablation_map = {
        "Exp A\n(df=3.0)":          "expA_df3",
        "Exp B\n(aux_sf)":          "expB_aux_sf",
        "Exp C\n(decorr)":          "expC_decorr",
        "Exp D\n(oversample)":      "expD_oversample",
        "Exp E\n(combined)":        "expE_combined",
    }
    for label, tag in ablation_map.items():
        path = os.path.join(V2_DIR, tag, "conditional_eval_results.json")
        d = _load_json(path)
        if d:
            configs[label] = {
                r: {"sf": d[r]["sf_count"],
                    "disc": d[r]["discriminative_score"],
                    "vol": d[r].get("vol_mean_across_assets", 0.0)}
                for r in REGIMES if r in d
            }

    # Quantile moment matching (Priority 1)
    mm_path = os.path.join(V2_DIR, "moment_matching", "regime_eval_rescaled.json")
    d = _load_json(mm_path)
    if d:
        configs["Quantile\nMatched"] = {
            r: {"sf": d[r]["after"]["sf_count"],
                "disc": d[r]["after"]["discriminative_score"],
                "vol": d[r]["after"]["syn_vol"]}
            for r in REGIMES if r in d
        }

    # Regime router (Priority 2)
    rr_path = os.path.join(V2_DIR, "regime_router", "regime_router_results.json")
    d = _load_json(rr_path)
    if d:
        configs["Regime\nRouter"] = {
            r: {"sf": d[r]["sf_count"],
                "disc": d[r]["discriminative_score"],
                "vol": d[r]["syn_vol"]}
            for r in REGIMES if r in d
        }

    return configs


# ─────────────────────────────────────────────────────────────
# Figure 1: Ablation comparison bar chart
# ─────────────────────────────────────────────────────────────

def make_ablation_comparison() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print("Generating v2_ablation_comparison.png ...")
    configs = load_ablation_data()
    if not configs:
        print("  [SKIP] No ablation data found.")
        return

    config_labels = list(configs.keys())
    n_configs = len(config_labels)
    regime_colors = {"crisis": "#e53935", "calm": "#43a047", "normal": "#1e88e5"}
    bar_width = 0.25
    x = np.arange(n_configs)

    fig, axes = plt.subplots(1, 2, figsize=(max(14, n_configs * 1.8), 6))

    # SF count per regime
    ax = axes[0]
    for j, regime in enumerate(REGIMES):
        sf_vals = [configs[c].get(regime, {}).get("sf", 0) for c in config_labels]
        offset  = (j - 1) * bar_width
        bars = ax.bar(x + offset, sf_vals, bar_width,
                      label=regime.capitalize(), color=regime_colors[regime],
                      edgecolor="white", alpha=0.85)
        for bar, val in zip(bars, sf_vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                        str(val), ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(x); ax.set_xticklabels(config_labels, fontsize=8)
    ax.set_ylim(0, 7); ax.set_yticks(range(7))
    ax.set_title("Stylized Facts Passed per Regime", fontweight="bold")
    ax.set_ylabel("SF count (out of 6)")
    ax.legend(fontsize=9, loc="upper left")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    # Discriminative score per regime
    ax = axes[1]
    for j, regime in enumerate(REGIMES):
        disc_vals = [configs[c].get(regime, {}).get("disc", 1.0) for c in config_labels]
        offset    = (j - 1) * bar_width
        bars = ax.bar(x + offset, disc_vals, bar_width,
                      label=regime.capitalize(), color=regime_colors[regime],
                      edgecolor="white", alpha=0.85)
        for bar, val in zip(bars, disc_vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=7)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1.2, label="Random (0.5)", zorder=0)
    ax.set_xticks(x); ax.set_xticklabels(config_labels, fontsize=8)
    ax.set_ylim(0, 1.15)
    ax.set_title("Discriminative Score per Regime", fontweight="bold")
    ax.set_ylabel("Disc score (lower = more realistic)")
    ax.legend(fontsize=9, loc="upper right")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    fig.suptitle("v2 Ablation Sweep — SF Count and Discriminative Score",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    out = os.path.join(ASSETS_DIR, "v2_ablation_comparison.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {out}")


# ─────────────────────────────────────────────────────────────
# Figure 2: Moment matching before/after
# ─────────────────────────────────────────────────────────────

def make_moment_matching() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print("Generating v2_moment_matching.png ...")

    # Try quantile-mapped data first, fall back to std-rescaled
    mm_path  = os.path.join(V2_DIR, "moment_matching", "regime_eval_rescaled.json")
    std_path = os.path.join(V2_DIR, "rescaling", "regime_eval_rescaled.json")
    d = _load_json(mm_path) or _load_json(std_path)
    if not d:
        print("  [SKIP] No moment-matching or rescaling data found.")
        return

    has_mm  = os.path.exists(mm_path)
    mode_label = "Quantile Mapped" if has_mm else "Std Rescaled"
    c_map = {"crisis": "#e53935", "calm": "#43a047", "normal": "#1e88e5"}
    x = np.arange(len(REGIMES)); w = 0.22

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Volatility panel
    ax = axes[0]
    real_vols  = [d[r]["before"]["real_vol"]   for r in REGIMES]
    raw_vols   = [d[r]["before"]["syn_vol"]    for r in REGIMES]
    resc_vols  = [d[r]["after"]["syn_vol"]     for r in REGIMES]
    ax.bar(x - w, real_vols, w, label="Real",          color="#546e7a", edgecolor="white")
    ax.bar(x,     raw_vols,  w, label="Raw DDPM",      color="#90a4ae", edgecolor="white")
    ax.bar(x + w, resc_vols, w, label=mode_label,
           color=[c_map[r] for r in REGIMES], edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels([r.capitalize() for r in REGIMES])
    ax.set_title("Volatility: Real vs Raw vs Rescaled", fontweight="bold")
    ax.set_ylabel("Mean std across assets")
    ax.legend(); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    # Kurtosis panel
    ax = axes[1]
    real_k  = [d[r]["before"]["real_kurtosis"]  for r in REGIMES]
    raw_k   = [d[r]["before"]["syn_kurtosis"]   for r in REGIMES]
    resc_k  = [d[r]["after"]["syn_kurtosis"]    for r in REGIMES]
    ax.bar(x - w, real_k, w, label="Real",         color="#546e7a", edgecolor="white")
    ax.bar(x,     raw_k,  w, label="Raw DDPM",     color="#90a4ae", edgecolor="white")
    ax.bar(x + w, resc_k, w, label=mode_label,
           color=[c_map[r] for r in REGIMES], edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels([r.capitalize() for r in REGIMES])
    ax.set_title("Excess Kurtosis: Real vs Raw vs Rescaled", fontweight="bold")
    ax.set_ylabel("Excess kurtosis")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.legend(); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    fig.suptitle(f"Post-Hoc Moment Matching ({mode_label}) — Distribution Shape Recovery",
                 fontweight="bold")
    fig.tight_layout()
    out = os.path.join(ASSETS_DIR, "v2_moment_matching.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {out}")


# ─────────────────────────────────────────────────────────────
# Figure 3: Guidance sweep
# ─────────────────────────────────────────────────────────────

def make_guidance_sweep() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print("Generating v2_guidance_sweep.png ...")
    path = os.path.join(V2_DIR, "guidance_sweep", "guidance_sweep.json")
    d = _load_json(path)
    if not d:
        print("  [SKIP] No guidance sweep data found.")
        return

    scales = sorted(float(k) for k in d)
    crisis_disc  = [d[f"{s:.1f}"]["crisis"]["disc"]  for s in scales]
    crisis_vol   = [d[f"{s:.1f}"]["crisis"]["vol"]   for s in scales]
    calm_disc    = [d[f"{s:.1f}"]["calm"]["disc"]    for s in scales]
    normal_disc  = [d[f"{s:.1f}"]["normal"]["disc"]  for s in scales]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.plot(scales, crisis_disc,  "o-", color="#e53935", linewidth=2,  label="Crisis Disc")
    ax.plot(scales, calm_disc,    "s-", color="#43a047", linewidth=2,  label="Calm Disc")
    ax.plot(scales, normal_disc,  "^-", color="#1e88e5", linewidth=2,  label="Normal Disc")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1.2, label="Random (0.5)")
    for s, v in zip(scales, crisis_disc):
        ax.annotate(f"{v:.3f}", (s, v), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8, color="#e53935")
    ax.set_xlabel("Guidance Scale"); ax.set_ylabel("Discriminative Score")
    ax.set_title("Disc Score vs Guidance Scale (Exp B checkpoint)", fontweight="bold")
    ax.set_xticks(scales)
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    ax = axes[1]
    ax.plot(scales, crisis_vol, "o-", color="#e53935", linewidth=2, label="Crisis Vol (syn)")
    ax.axhline(1.6835, color="#e53935", linestyle="--", linewidth=1.5,
               label="Crisis Vol (real=1.684)", alpha=0.7)
    for s, v in zip(scales, crisis_vol):
        ax.annotate(f"{v:.3f}", (s, v), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8, color="#e53935")
    ax.set_xlabel("Guidance Scale"); ax.set_ylabel("Synthetic Volatility (mean std)")
    ax.set_title("Crisis Volatility vs Guidance Scale (Exp B checkpoint)", fontweight="bold")
    ax.set_xticks(scales)
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    fig.suptitle("Guidance Scale Sweep — Exp B (aux_sf_loss) Checkpoint",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    out = os.path.join(ASSETS_DIR, "v2_guidance_sweep.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {out}")


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    make_ablation_comparison()
    make_moment_matching()
    make_guidance_sweep()
    print("\nDone. Figures saved to presentation_assets/")

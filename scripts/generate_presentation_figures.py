"""
Generate new presentation figures for the Generative Market Simulation deck.

Produces:
  presentation_assets/15_four_layers.png  -- L1-L4 "What makes data useful" diagram
  presentation_assets/16_calibration_ceiling.png  -- Real 3/6 vs DDPM 5/6 ceiling chart
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "presentation_assets")
os.makedirs(OUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# Figure 1: L1-L4 Four Layers Diagram
# ─────────────────────────────────────────────
def make_four_layers():
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    layers = [
        # (y_bottom, height, color, tag, title, subtitle, status_label, status_color)
        (0.5,  1.9, "#2e7d32", "L4", "Downstream Utility",
         "VaR backtesting · option pricing · strategy backtest fidelity",
         "Future Work", "#9e9e9e"),
        (2.6,  1.9, "#1565c0", "L3", "Conditional Control",
         "Regime-specific generation (crisis / calm / normal) · CFG guidance",
         "Future Work", "#9e9e9e"),
        (4.7,  1.9, "#00897b", "L2", "Statistical Fidelity",
         "Heavy tails · vol clustering · leverage · cross-asset correlations",
         "✓ Delivered", "#43a047"),
        (6.8,  1.9, "#1976d2", "L1", "Diversity",
         "Thousands of novel multi-asset paths · 5 generative architectures",
         "✓ Delivered", "#43a047"),
    ]

    for (yb, h, color, tag, title, subtitle, status, st_color) in layers:
        delivered = "Delivered" in status


        box_color = color if delivered else "#bdbdbd"
        text_color = "white" if delivered else "#424242"
        alpha = 1.0 if delivered else 0.55

        rect = FancyBboxPatch((0.3, yb), 7.2, h - 0.1,
                              boxstyle="round,pad=0.07",
                              facecolor=box_color, edgecolor="white",
                              linewidth=2, alpha=alpha)
        ax.add_patch(rect)


        ax.text(0.95, yb + h / 2, tag,
                ha="center", va="center",
                fontsize=18, fontweight="bold",
                color="white" if delivered else "#616161",
                alpha=alpha)

        ax.text(2.0, yb + h * 0.62, title,
                ha="left", va="center",
                fontsize=14, fontweight="bold",
                color=text_color, alpha=alpha)

        ax.text(2.0, yb + h * 0.28, subtitle,
                ha="left", va="center",
                fontsize=10, color=text_color,
                alpha=alpha * 0.9)

        badge_color = st_color if delivered else "#9e9e9e"
        badge_rect = FancyBboxPatch((7.7, yb + h * 0.3), 2.0, h * 0.4,
                                    boxstyle="round,pad=0.05",
                                    facecolor=badge_color,
                                    edgecolor="none", alpha=alpha)
        ax.add_patch(badge_rect)
        ax.text(8.7, yb + h / 2, status,
                ha="center", va="center",
                fontsize=9.5, fontweight="bold",
                color="white", alpha=alpha)

    # Arrow on the right side indicating increasing utility
    ax.annotate("", xy=(9.6, 8.7), xytext=(9.6, 0.5),
                arrowprops=dict(arrowstyle="-|>", color="#555", lw=2))
    ax.text(9.75, 4.7, "Increasing\nutility", ha="center", va="center",
            fontsize=9, color="#555", rotation=90)

    ax.text(5.0, 9.3, 'What Makes Synthetic Financial Data "Useful"?',
            ha="center", va="center",
            fontsize=16, fontweight="bold", color="#212121")

    # Bottom note
    note = ("This project delivers L1 + L2 as a solid, evaluated foundation.  "
            "L3 + L4 are explicitly scoped as next steps.")
    ax.text(5.0, 0.15, note, ha="center", va="center",
            fontsize=10, color="#555", style="italic")

    fig.tight_layout()
    out = os.path.join(OUT_DIR, "15_four_layers.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out}")


# ─────────────────────────────────────────────
# Figure 2: Calibration Ceiling Bar Chart
# ─────────────────────────────────────────────
def make_calibration_ceiling():
    fig, ax = plt.subplots(figsize=(9, 6))

    models = ["Real Data\n(n=318,060)", "GARCH\nBaseline", "VAE\n(Improved)",
              "TimeGAN", "NormFlow\n(RealNVP)", "DDPM\n(v-pred+Student-t)"]
    sf_values = [3.0, 1.33, 1.0, 4.0, 5.0, 5.0]
    bar_colors = ["#90a4ae", "#ef9a9a", "#ef9a9a", "#ffe082", "#a5d6a7", "#43a047"]
    edge_colors = ["#546e7a", "#c62828", "#c62828", "#f9a825", "#2e7d32", "#1b5e20"]

    bars = ax.bar(models, sf_values, color=bar_colors, edgecolor=edge_colors,
                  linewidth=1.8, width=0.6, zorder=3)

    # Annotate values on bars
    for bar, val in zip(bars, sf_values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.08,
                f"{val:.1f}/6",
                ha="center", va="bottom",
                fontsize=12, fontweight="bold", color="#212121")

    # Ceiling line at 5.0
    ax.axhline(5.0, color="#1b5e20", linestyle="--", linewidth=2.0,
               label="Empirical ceiling (5/6)", zorder=2)

    # Real data reference line at 3.0
    ax.axhline(3.0, color="#546e7a", linestyle=":", linewidth=1.8,
               label="Real data baseline (3/6)", zorder=2)

    # Annotations
    ax.text(5.6, 5.1, "Empirical ceiling", fontsize=9.5,
            color="#1b5e20", fontweight="bold")
    ax.text(5.6, 3.1, "Real data scores 3/6\n(same framework)", fontsize=8.5,
            color="#546e7a")

    ax.set_ylim(0, 6.5)
    ax.set_yticks([0, 1, 2, 3, 4, 5, 6])
    ax.set_yticklabels([str(i) for i in range(7)], fontsize=11)
    ax.set_ylabel("Stylized Facts Passed (out of 6)", fontsize=13)
    ax.set_title("Stylized Fact Coverage: All Models vs Real Data Baseline",
                 fontsize=14, fontweight="bold", pad=14)
    ax.tick_params(axis="x", labelsize=10)
    ax.grid(axis="y", alpha=0.35, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Highlight the critical insight
    insight = ("DDPM synthetic data (5/6) scores HIGHER than real\n"
               "training data (3/6) on the same evaluation framework.")
    ax.text(0.5, 0.96, insight, transform=ax.transAxes,
            ha="center", va="top", fontsize=9.5,
            color="#1b5e20", style="italic",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#e8f5e9", edgecolor="#a5d6a7"))

    fig.tight_layout()
    out = os.path.join(OUT_DIR, "16_calibration_ceiling.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    make_four_layers()
    make_calibration_ceiling()
    print("Done. Both figures saved to presentation_assets/")

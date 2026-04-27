"""
generate_pipeline_diagram.py
============================
Generates a high-level project pipeline diagram for the README.

Shows the full project data flow:
  Yahoo Finance + FRED API
    -> download.py / preprocess.py / regime_labels.py
    -> 5 Generative Models (DDPM, GAN, VAE, GARCH, NormFlow)
    -> Evaluation (6 Stylized Facts, MMD/Wasserstein, Disc Score)
    -> Interactive Demo (FastAPI + Chart.js)

Output: presentation_assets/pipeline_overview.png
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT  = os.path.join(ROOT, "presentation_assets", "pipeline_overview.png")

C_DATA   = "#1565c0"   # blue    — data sources
C_PIPE   = "#5c6bc0"   # indigo  — data pipeline
C_MODEL  = "#e65100"   # orange  — models
C_EVAL   = "#37474f"   # blue-grey — evaluation
C_DEMO   = "#2e7d32"   # green   — demo
C_ARROW  = "#546e7a"
C_BG     = "#fafafa"


def _box(ax, x, y, w, h, label, sublabel=None, color=C_DATA, alpha=0.92,
         fontsize=10, subfontsize=8):
    rect = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                          boxstyle="round,pad=0.05",
                          linewidth=1.8, edgecolor=color,
                          facecolor=color, alpha=alpha, zorder=3)
    ax.add_patch(rect)
    ax.text(x, y + (0.06 if sublabel else 0), label,
            ha="center", va="center", fontsize=fontsize,
            fontweight="bold", color="white", zorder=4)
    if sublabel:
        ax.text(x, y - 0.18, sublabel,
                ha="center", va="center", fontsize=subfontsize,
                color="white", alpha=0.90, zorder=4)


def _arrow(ax, x0, y0, x1, y1, label=None, color=C_ARROW, lw=1.8):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                connectionstyle="arc3,rad=0.0"),
                zorder=5)
    if label:
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        ax.text(mx + 0.05, my, label, fontsize=7.5, color=color,
                va="center", zorder=6)


fig, ax = plt.subplots(figsize=(14, 8))
ax.set_xlim(0, 14); ax.set_ylim(0, 8)
ax.axis("off")
fig.patch.set_facecolor(C_BG)
ax.set_facecolor(C_BG)

fig.suptitle("Generative Market Simulation — Project Pipeline",
             fontsize=14, fontweight="bold", y=0.97, color="#1a237e")

# ── Row 1: Data Sources  (y = 7.0)
_box(ax, 3.5, 7.0, 5.5, 0.75, "Data Sources",
     "Yahoo Finance (18 tickers, 2005-2026)  ·  FRED API (macro features)",
     C_DATA, fontsize=11, subfontsize=8.5)

# ── Row 2: Pipeline scripts  (y = 5.6)
_box(ax, 2.0, 5.6, 2.4, 0.65, "download.py",
     "Daily OHLCV + macro data", C_PIPE, fontsize=9.5, subfontsize=7.5)
_box(ax, 5.0, 5.6, 2.4, 0.65, "preprocess.py",
     "Log returns · z-score · windows.npy", C_PIPE, fontsize=9.5, subfontsize=7.5)
_box(ax, 8.0, 5.6, 2.4, 0.65, "regime_labels.py",
     "VIX+yield → crisis/calm/normal", C_PIPE, fontsize=9.5, subfontsize=7.5)

# Arrows: sources -> scripts
_arrow(ax, 3.5, 6.62, 3.5, 5.93, color=C_DATA)
ax.annotate("", xy=(2.0, 5.93), xytext=(3.5, 6.62),
            arrowprops=dict(arrowstyle="-|>", color=C_DATA, lw=1.8,
                            connectionstyle="arc3,rad=0.15"), zorder=5)
ax.annotate("", xy=(5.0, 5.93), xytext=(3.5, 6.62),
            arrowprops=dict(arrowstyle="-|>", color=C_DATA, lw=1.8,
                            connectionstyle="arc3,rad=0.0"), zorder=5)
ax.annotate("", xy=(8.0, 5.93), xytext=(3.5, 6.62),
            arrowprops=dict(arrowstyle="-|>", color=C_DATA, lw=1.8,
                            connectionstyle="arc3,rad=-0.2"), zorder=5)

# Artefact labels
ax.text(2.0, 4.95, "prices.csv", ha="center", fontsize=8, color=C_PIPE, style="italic")
ax.text(5.0, 4.95, "windows.npy", ha="center", fontsize=8, color=C_PIPE, style="italic")
ax.text(8.0, 4.95, "window_cond.npy", ha="center", fontsize=8, color=C_PIPE, style="italic")

# ── Row 3: Generative Models  (y = 3.6)
_box(ax, 7.0, 3.6, 10.0, 0.9, "5 Generative Models",
     "DDPM (v-pred + Student-t)  ·  TimeGAN  ·  VAE  ·  GARCH  ·  NormFlow (RealNVP)",
     C_MODEL, fontsize=11, subfontsize=8.5)

# Arrows: pipeline -> models
for px in [2.0, 5.0, 8.0]:
    ax.annotate("", xy=(7.0, 4.05), xytext=(px, 5.27),
                arrowprops=dict(arrowstyle="-|>", color=C_PIPE, lw=1.5,
                                connectionstyle="arc3,rad=0.0"), zorder=5)

# ── Row 4: Evaluation  (y = 2.2)
_box(ax, 7.0, 2.2, 10.0, 0.9, "Evaluation Framework",
     "6 Stylized Facts  ·  MMD / Wasserstein  ·  Discriminative Score  ·  VaR/CVaR (Kupiec)",
     C_EVAL, fontsize=11, subfontsize=8.5)

_arrow(ax, 7.0, 3.15, 7.0, 2.65, color=C_ARROW)

# ── Row 5: Demo  (y = 0.85)
_box(ax, 7.0, 0.85, 6.0, 0.75, "Interactive Demo",
     "FastAPI backend  ·  Chart.js frontend  ·  http://localhost:8000",
     C_DEMO, fontsize=11, subfontsize=8.5)

_arrow(ax, 7.0, 1.75, 7.0, 1.23, color=C_ARROW)

# ── Legend
legend_items = [
    mpatches.Patch(color=C_DATA,  label="Data Sources"),
    mpatches.Patch(color=C_PIPE,  label="Data Pipeline"),
    mpatches.Patch(color=C_MODEL, label="Generative Models"),
    mpatches.Patch(color=C_EVAL,  label="Evaluation"),
    mpatches.Patch(color=C_DEMO,  label="Interactive Demo"),
]
ax.legend(handles=legend_items, loc="lower right", fontsize=8,
          framealpha=0.9, ncol=5, bbox_to_anchor=(1.0, 0.0))

fig.tight_layout(rect=[0, 0.02, 1, 0.96])
fig.savefig(OUT, dpi=160, bbox_inches="tight", facecolor=C_BG)
plt.close(fig)
print(f"Saved: {OUT}")

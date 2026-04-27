"""
generate_architecture_diagram.py
=================================
Generates a clean DDPM pipeline architecture diagram for presentation.
Shows the full flow: Data → Forward Process → U-Net 1D → Reverse Process → Evaluation.

Output: presentation_assets/architecture_ddpm_pipeline.png
"""

import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT  = os.path.join(ROOT, "presentation_assets", "architecture_ddpm_pipeline.png")

# ── Colour palette
C_DATA    = "#1565c0"   # dark blue  — data / input
C_FWD     = "#6a1b9a"   # purple     — forward process
C_UNET    = "#e65100"   # deep orange — U-Net denoiser
C_REV     = "#2e7d32"   # green      — reverse / generation
C_EVAL    = "#37474f"   # blue-grey  — evaluation
C_AUX     = "#ad1457"   # pink       — auxiliary losses
C_COND    = "#00838f"   # teal       — conditioning
C_ARROW   = "#455a64"
C_BG      = "#fafafa"

def box(ax, x, y, w, h, label, sublabel=None, color=C_DATA, alpha=0.92,
        fontsize=9, subfontsize=7.5):
    rect = FancyBboxPatch((x - w/2, y - h/2), w, h,
                          boxstyle="round,pad=0.04",
                          linewidth=1.5, edgecolor=color,
                          facecolor=color, alpha=alpha,
                          zorder=3)
    ax.add_patch(rect)
    ax.text(x, y + (0.05 if sublabel else 0), label,
            ha="center", va="center", fontsize=fontsize,
            fontweight="bold", color="white", zorder=4,
            wrap=True)
    if sublabel:
        ax.text(x, y - 0.15, sublabel,
                ha="center", va="center", fontsize=subfontsize,
                color="white", alpha=0.92, zorder=4)

def section_bg(ax, x, y, w, h, label, color, alpha=0.06):
    rect = FancyBboxPatch((x, y), w, h,
                          boxstyle="round,pad=0.06",
                          linewidth=1.2, edgecolor=color,
                          facecolor=color, alpha=alpha, zorder=1)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h + 0.05, label,
            ha="center", va="bottom", fontsize=8,
            color=color, fontweight="bold", zorder=2)

def arrow(ax, x0, y0, x1, y1, color=C_ARROW, lw=1.5, label=None):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>",
                                color=color, lw=lw,
                                connectionstyle="arc3,rad=0.0"),
                zorder=5)
    if label:
        mx, my = (x0 + x1)/2, (y0 + y1)/2
        ax.text(mx + 0.05, my, label, fontsize=6.5, color=color,
                va="center", zorder=6)

def curved_arrow(ax, x0, y0, x1, y1, rad=0.3, color=C_ARROW, lw=1.4, label=None):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                connectionstyle=f"arc3,rad={rad}"),
                zorder=5)
    if label:
        mx, my = (x0 + x1)/2, (y0 + y1)/2
        ax.text(mx, my + 0.12, label, fontsize=6.5, color=color,
                ha="center", zorder=6)

# ── Figure
fig, ax = plt.subplots(figsize=(18, 10))
ax.set_xlim(0, 18)
ax.set_ylim(0, 10)
ax.axis("off")
fig.patch.set_facecolor(C_BG)
ax.set_facecolor(C_BG)

fig.suptitle("DDPM Architecture for Conditional Synthetic Financial Data Generation",
             fontsize=14, fontweight="bold", y=0.98, color="#1a237e")

# ──────────────────────────────────────────────────────────────
# SECTION BACKGROUNDS
# ──────────────────────────────────────────────────────────────
section_bg(ax, 0.3,  6.0,  2.6, 3.4, "Data Preparation",  C_DATA)
section_bg(ax, 3.2,  6.0,  2.6, 3.4, "Forward Process",   C_FWD)
section_bg(ax, 6.2,  1.8,  5.6, 7.6, "U-Net 1D Denoiser", C_UNET)
section_bg(ax, 12.1, 6.0,  2.6, 3.4, "Reverse / Generate",C_REV)
section_bg(ax, 15.1, 6.0,  2.6, 3.4, "Evaluation",        C_EVAL)
section_bg(ax, 6.2,  0.3,  5.6, 1.3, "Auxiliary Losses",  C_AUX)
section_bg(ax, 0.3,  1.8,  2.6, 3.8, "Conditioning",      C_COND)

# ──────────────────────────────────────────────────────────────
# DATA PREPARATION  (col ≈ 1.6)
# ──────────────────────────────────────────────────────────────
box(ax, 1.6, 9.0, 2.0, 0.55, "16 Assets × Daily Returns",
    "S&P 500 / Bonds / Commodities / FX", C_DATA, fontsize=8, subfontsize=6.5)
box(ax, 1.6, 8.0, 2.0, 0.55, "Rolling Windows",
    "N=5293, T=60 days, z-scored", C_DATA, fontsize=8, subfontsize=6.5)
box(ax, 1.6, 7.0, 2.0, 0.55, "Regime Labels",
    "Crisis=724  Calm=2112  Normal=2457", C_DATA, fontsize=8, subfontsize=6.5)

arrow(ax, 1.6, 8.72, 1.6, 8.28, C_DATA)
arrow(ax, 1.6, 7.72, 1.6, 7.28, C_DATA)

# ──────────────────────────────────────────────────────────────
# CONDITIONING  (col ≈ 1.6, lower section)
# ──────────────────────────────────────────────────────────────
box(ax, 1.6, 5.2, 2.0, 0.55, "Macro Features (5-dim)",
    "Yield slope, credit spread, VIX, RV, Fed funds", C_COND, fontsize=8, subfontsize=6)
box(ax, 1.6, 4.2, 2.0, 0.55, "CFG Conditioning",
    "10% null-cond drop  ·  guidance_scale=2.0", C_COND, fontsize=8, subfontsize=6)
box(ax, 1.6, 3.2, 2.0, 0.55, "Cond MLP",
    "Linear(5→128)→SiLU→Linear(128→128)", C_COND, fontsize=8, subfontsize=6)

arrow(ax, 1.6, 4.92, 1.6, 4.48, C_COND)
arrow(ax, 1.6, 3.92, 1.6, 3.48, C_COND)

# ──────────────────────────────────────────────────────────────
# FORWARD PROCESS  (col ≈ 4.5)
# ──────────────────────────────────────────────────────────────
box(ax, 4.5, 9.0, 2.0, 0.55, "Student-t Noise",
    "df=5.0  ·  heavier tails than Gaussian", C_FWD, fontsize=8, subfontsize=6.5)
box(ax, 4.5, 8.0, 2.0, 0.55, "Cosine Noise Schedule",
    "T=1000 steps  ·  α̅_t = cos²(πt/2T·s)", C_FWD, fontsize=8, subfontsize=6.5)
box(ax, 4.5, 7.0, 2.0, 0.55, "V-Prediction Target",
    "v = √ᾱ ε − √(1−ᾱ) x₀  [Salimans 2022]", C_FWD, fontsize=8, subfontsize=6.5)

arrow(ax, 4.5, 8.72, 4.5, 8.28, C_FWD)
arrow(ax, 4.5, 7.72, 4.5, 7.28, C_FWD)

# data → forward
arrow(ax, 2.6, 7.0, 3.5, 7.0, C_ARROW)

# ──────────────────────────────────────────────────────────────
# U-NET 1D  (centre cols 7–11)
# ──────────────────────────────────────────────────────────────
# Encoder
box(ax, 7.6, 9.0, 2.2, 0.55, "Input Conv1d",
    "in_ch=16  →  base_ch=128", C_UNET, fontsize=8, subfontsize=6.5)
box(ax, 7.6, 8.0, 2.2, 0.55, "ResBlock (128ch) + Pool",
    "GroupNorm · SiLU · Conv1d × 2  ·  stride-2 pool", C_UNET, fontsize=8, subfontsize=6.5)
box(ax, 7.6, 7.0, 2.2, 0.55, "ResBlock (256ch) + Pool",
    "+ TemporalAttn(256)  ·  stride-2 pool", C_UNET, fontsize=8, subfontsize=6.5)
box(ax, 7.6, 6.0, 2.2, 0.55, "ResBlock (512ch) + Pool",
    "+ TemporalAttn(512)  ·  stride-2 pool", C_UNET, fontsize=8, subfontsize=6.5)

# Bottleneck
box(ax, 9.9, 7.5, 1.8, 0.55, "Bottleneck",
    "ResBlock(512ch) + TemporalAttn", C_UNET, fontsize=8, subfontsize=6.5)

# Decoder
box(ax, 11.4, 8.0, 2.2, 0.55, "Up + ResBlock (256ch)",
    "ConvTranspose + skip concat", C_UNET, fontsize=8, subfontsize=6.5)
box(ax, 11.4, 7.0, 2.2, 0.55, "Up + ResBlock (128ch)",
    "ConvTranspose + skip concat", C_UNET, fontsize=8, subfontsize=6.5)
box(ax, 11.4, 6.0, 2.2, 0.55, "Output Conv1d",
    "GroupNorm · SiLU · Conv1d → 16ch", C_UNET, fontsize=8, subfontsize=6.5)

# Time embedding  (shared)
box(ax, 9.5, 9.0, 2.0, 0.55, "Sinusoidal Time Emb",
    "SinPosEmb(128) → Linear → SiLU  [t]", C_UNET, fontsize=8, subfontsize=6.5)

# Encoder arrows
arrow(ax, 7.6, 8.72, 7.6, 8.28, C_UNET)
arrow(ax, 7.6, 7.72, 7.6, 7.28, C_UNET)
arrow(ax, 7.6, 6.72, 7.6, 6.28, C_UNET)
# time → resblocks
arrow(ax, 9.5, 8.72, 9.5, 8.3,  C_UNET)   # time emb down to reach unet body
arrow(ax, 8.7, 9.0,  8.7, 9.0, C_UNET)    # (no-op placeholder)
# bottleneck
arrow(ax, 7.6, 5.72, 8.8, 7.5,  C_UNET, label="bottleneck")
# decoder
arrow(ax, 8.8, 7.5,  10.3, 8.0, C_UNET)
arrow(ax, 11.3, 8.0, 11.3, 7.28, C_UNET)
arrow(ax, 11.3, 7.0, 11.3, 6.28, C_UNET)

# Skip connections (dashed)
for y_enc, y_dec in [(8.0, 8.0), (7.0, 7.0)]:
    ax.annotate("", xy=(10.3, y_dec), xytext=(8.7, y_enc),
                arrowprops=dict(arrowstyle="-|>", color=C_UNET, lw=1.2,
                                linestyle="dashed",
                                connectionstyle="arc3,rad=-0.25"),
                zorder=5)

# Cond MLP → U-Net add
arrow(ax, 2.6, 3.2, 6.5, 3.2, C_COND)
ax.annotate("", xy=(7.6, 7.5), xytext=(6.5, 3.2),
            arrowprops=dict(arrowstyle="-|>", color=C_COND, lw=1.2,
                            connectionstyle="arc3,rad=-0.3"),
            zorder=5)
ax.text(5.8, 5.5, "add to t_emb\n(all ResBlocks)", fontsize=6.5, color=C_COND,
        ha="center", zorder=6)

# forward → UNet input
arrow(ax, 5.5, 7.0, 6.5, 7.0, C_ARROW, label="noisy x_t")

# ──────────────────────────────────────────────────────────────
# REVERSE / GENERATION  (col ≈ 13.4)
# ──────────────────────────────────────────────────────────────
box(ax, 13.4, 9.0, 2.0, 0.55, "DDIM Sampling",
    "50 steps  ·  η=0.3  ·  deterministic", C_REV, fontsize=8, subfontsize=6.5)
box(ax, 13.4, 8.0, 2.0, 0.55, "EMA Model Weights",
    "decay=0.9999  ·  EMA for inference", C_REV, fontsize=8, subfontsize=6.5)
box(ax, 13.4, 7.0, 2.0, 0.55, "CFG Interpolation",
    "ε_guided = ε_uncond + s·(ε_cond − ε_uncond)", C_REV, fontsize=8, subfontsize=6.5)

arrow(ax, 13.4, 8.72, 13.4, 8.28, C_REV)
arrow(ax, 13.4, 7.72, 13.4, 7.28, C_REV)

# U-Net output → Reverse
arrow(ax, 12.5, 6.0, 12.5, 7.0, C_ARROW)
arrow(ax, 12.5, 7.0, 12.4, 7.0, C_ARROW)

# ──────────────────────────────────────────────────────────────
# EVALUATION  (col ≈ 16.4)
# ──────────────────────────────────────────────────────────────
box(ax, 16.4, 9.0, 2.0, 0.55, "6 Stylized Facts",
    "Fat tails · Vol clust. · Leverage · ACF…", C_EVAL, fontsize=8, subfontsize=6.5)
box(ax, 16.4, 8.0, 2.0, 0.55, "MMD + Disc Score",
    "RBF kernel MMD  ·  1D-CNN discriminator", C_EVAL, fontsize=8, subfontsize=6.5)
box(ax, 16.4, 7.0, 2.0, 0.55, "VaR / CVaR (Kupiec)",
    "95% / 99%  ·  rolling portfolio backtest", C_EVAL, fontsize=8, subfontsize=6.5)

arrow(ax, 13.4, 6.72, 15.4, 6.72, C_ARROW)
ax.annotate("", xy=(16.4, 7.0), xytext=(15.4, 6.72),
            arrowprops=dict(arrowstyle="-|>", color=C_ARROW, lw=1.5,
                            connectionstyle="arc3,rad=0.0"),
            zorder=5)
arrow(ax, 16.4, 8.72, 16.4, 8.28, C_EVAL)
arrow(ax, 16.4, 7.72, 16.4, 7.28, C_EVAL)

# ──────────────────────────────────────────────────────────────
# AUXILIARY LOSSES  (bottom strip)
# ──────────────────────────────────────────────────────────────
box(ax, 8.4, 0.9, 2.2, 0.48, "Aux SF Loss",
    "kurtosis match + ACF match  (w=0.1)", C_AUX, fontsize=8, subfontsize=6.5)
box(ax, 11.0, 0.9, 2.2, 0.48, "Decorr Reg",
    "penalise raw-return ACF on x̂₀  (w=0.05)", C_AUX, fontsize=8, subfontsize=6.5)

# arrows up to U-Net bottom
ax.annotate("", xy=(8.0, 6.0), xytext=(8.4, 1.14),
            arrowprops=dict(arrowstyle="-|>", color=C_AUX, lw=1.1,
                            linestyle="dotted",
                            connectionstyle="arc3,rad=0.2"),
            zorder=5)
ax.annotate("", xy=(11.0, 6.0), xytext=(11.0, 1.14),
            arrowprops=dict(arrowstyle="-|>", color=C_AUX, lw=1.1,
                            linestyle="dotted",
                            connectionstyle="arc3,rad=0.0"),
            zorder=5)

# ──────────────────────────────────────────────────────────────
# LEGEND
# ──────────────────────────────────────────────────────────────
legend_items = [
    mpatches.Patch(color=C_DATA,  label="Data Preparation"),
    mpatches.Patch(color=C_FWD,   label="Forward Process"),
    mpatches.Patch(color=C_UNET,  label="U-Net 1D Denoiser"),
    mpatches.Patch(color=C_REV,   label="Reverse / Generate"),
    mpatches.Patch(color=C_EVAL,  label="Evaluation"),
    mpatches.Patch(color=C_COND,  label="Conditioning"),
    mpatches.Patch(color=C_AUX,   label="Auxiliary Losses"),
]
ax.legend(handles=legend_items, loc="lower right", fontsize=7.5,
          framealpha=0.9, ncol=4, bbox_to_anchor=(1.0, 0.0))

fig.tight_layout(rect=[0, 0.02, 1, 0.97])
fig.savefig(OUT, dpi=160, bbox_inches="tight", facecolor=C_BG)
plt.close(fig)
print(f"Saved: {OUT}")

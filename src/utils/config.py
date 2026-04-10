"""
Central configuration for the project.
"""

import os
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CHECKPOINTS_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
DEMO_DIR = os.path.join(PROJECT_ROOT, "src", "demo")

DEFAULT_WINDOW_SIZE = 60
DEFAULT_N_ASSETS = 18
DEFAULT_EPOCHS = 400
DEFAULT_BATCH_SIZE = 64
DEFAULT_LR = 2e-4
def _best_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

DEFAULT_DEVICE = _best_device()

SEED = 42

# Regime definitions
REGIME_MAP = {0: "normal", 1: "crisis", 2: "calm"}
REGIME_INV = {"normal": 0, "crisis": 1, "calm": 2}
N_REGIMES = 3

# Macro conditioning feature names (from FRED)
MACRO_FEATURES = ["yield_curve_slope", "credit_spread", "fed_funds", "vix_level", "realized_vol"]
COND_DIM = len(MACRO_FEATURES)

# VIX thresholds for regime classification
VIX_CRISIS_THRESHOLD = 25.0
VIX_CALM_THRESHOLD = 15.0

# Model names for iteration
MODEL_NAMES = ["ddpm", "garch", "vae", "timegan", "flow"]

"""
FastAPI backend for the Generative Market Simulation demo.

Serves:
  - /generate          POST  Generate synthetic paths for a given regime/model
  - /models            GET   List available models
  - /stylized-facts    POST  Run stylized fact tests on generated data
  - /comparison        GET   Full model comparison results
  - /                  GET   Serve the frontend HTML
"""

from __future__ import annotations

import os
import sys
import json
import argparse
import re

import numpy as np

# Ensure project root is on path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
import uvicorn

from src.data.regime_labels import get_regime_conditioning_vectors
from src.evaluation.stylized_facts import run_all_tests, count_passes
from src.evaluation.metrics import full_evaluation

app = FastAPI(title="Generative Market Simulation", version="1.0")


def _to_native(obj):
    """Recursively convert numpy types to native Python for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_native(v) for v in obj]
    if isinstance(obj, (np.bool_, np.bool)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

LOADED_MODELS: dict = {}
REAL_DATA: dict = {}
ASSET_NAMES: list[str] = []
SCALER: dict = {}


def _get_device():
    from src.utils.config import DEFAULT_DEVICE
    return DEFAULT_DEVICE


def _build_ddpm_kwargs_from_checkpoint(ckpt_path: str, defaults: dict) -> dict:
    """
    Build DDPM init kwargs by combining runtime defaults and checkpoint config.
    This avoids architecture mismatches when checkpoint was trained with a
    different DDPM width/depth than current defaults.
    """
    kwargs = dict(defaults)
    import torch as _torch

    ckpt = _torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})
    net_state = ckpt.get("net_state", {})

    kwargs["n_features"] = cfg.get("n_features", kwargs["n_features"])
    kwargs["seq_len"] = cfg.get("seq_len", kwargs["seq_len"])
    kwargs["T"] = cfg.get("T", kwargs.get("T", 1000))
    kwargs["cond_dim"] = cfg.get("cond_dim", kwargs["cond_dim"])
    kwargs["cfg_drop_prob"] = cfg.get("cfg_drop_prob", kwargs.get("cfg_drop_prob", 0.1))

    init_conv_w = net_state.get("init_conv.weight")
    if init_conv_w is not None and init_conv_w.ndim == 3:
        base_channels = int(init_conv_w.shape[0])
        kwargs["base_channels"] = base_channels

        down_channels = {}
        pattern = re.compile(r"down_blocks\.(\d+)\.block1\.2\.weight")
        for key, value in net_state.items():
            match = pattern.match(key)
            if match and hasattr(value, "shape") and len(value.shape) == 3:
                idx = int(match.group(1))
                down_channels[idx] = int(value.shape[0])

        if down_channels:
            channel_mults = tuple(
                max(1, down_channels[idx] // max(base_channels, 1))
                for idx in sorted(down_channels.keys())
            )
            kwargs["channel_mults"] = channel_mults

    return kwargs


def load_all_models(checkpoints_dir: str, data_dir: str):
    """Load all available model checkpoints."""
    import torch

    device = _get_device()
    windows_path = os.path.join(data_dir, "windows.npy")
    if not os.path.exists(windows_path):
        print("WARNING: windows.npy not found -- models cannot be loaded without data shape info")
        return

    windows = np.load(windows_path)
    n_features = windows.shape[2]
    seq_len = windows.shape[1]

    cond_path = os.path.join(data_dir, "window_cond.npy")
    cond_dim = np.load(cond_path).shape[1] if os.path.exists(cond_path) else 0

    # Store real data for comparison
    returns_path = os.path.join(data_dir, "returns.csv")
    if os.path.exists(returns_path):
        import pandas as pd
        REAL_DATA["returns"] = pd.read_csv(returns_path, index_col=0).values.astype(np.float32)

    REAL_DATA["windows"] = windows

    names_path = os.path.join(data_dir, "asset_names.json")
    if os.path.exists(names_path):
        with open(names_path) as f:
            ASSET_NAMES.extend(json.load(f))

    scaler_mean = os.path.join(data_dir, "scaler_mean.npy")
    scaler_std = os.path.join(data_dir, "scaler_std.npy")
    if os.path.exists(scaler_mean):
        SCALER["mean"] = np.load(scaler_mean)
        SCALER["std"] = np.load(scaler_std)

    # Read DDPM checkpoint config so demo can construct matching architecture.
    ddpm_ckpt_path = os.path.join(checkpoints_dir, "ddpm.pt")
    ddpm_kwargs = {"n_features": n_features, "seq_len": seq_len, "cond_dim": cond_dim, "device": device}
    if os.path.exists(ddpm_ckpt_path):
        try:
            ddpm_kwargs = _build_ddpm_kwargs_from_checkpoint(ddpm_ckpt_path, ddpm_kwargs)
        except Exception as e:
            print(f"  WARNING: Could not infer DDPM config from checkpoint: {e}")

    model_configs = {
        "ddpm": {
            "file": "ddpm.pt",
            "class": "DDPMModel",
            "kwargs": ddpm_kwargs,
        },
        "garch": {
            "file": "garch.npz",
            "class": "GARCHModel",
            "kwargs": {"n_features": n_features, "seq_len": seq_len, "device": device},
        },
        "vae": {
            "file": "vae.pt",
            "class": "FinancialVAE",
            "kwargs": {"n_features": n_features, "seq_len": seq_len, "device": device},
        },
        "timegan": {
            "file": "timegan.pt",
            "class": "TimeGANModel",
            "kwargs": {"n_features": n_features, "seq_len": seq_len, "device": device},
        },
        "flow": {
            "file": "flow.pt",
            "class": "NormalizingFlowModel",
            "kwargs": {"n_features": n_features, "seq_len": seq_len, "device": device},
        },
    }

    from src.models import DDPMModel, GARCHModel, FinancialVAE, TimeGANModel, NormalizingFlowModel
    class_map = {
        "DDPMModel": DDPMModel,
        "GARCHModel": GARCHModel,
        "FinancialVAE": FinancialVAE,
        "TimeGANModel": TimeGANModel,
        "NormalizingFlowModel": NormalizingFlowModel,
    }

    for name, cfg in model_configs.items():
        ckpt_path = os.path.join(checkpoints_dir, cfg["file"])
        if os.path.exists(ckpt_path):
            try:
                model_cls = class_map[cfg["class"]]
                model = model_cls(**cfg["kwargs"])
                model.load(ckpt_path)
                LOADED_MODELS[name] = model
                print(f"  Loaded {name} from {ckpt_path}")
            except Exception as e:
                print(f"  WARNING: Failed to load {name}: {e}")
        else:
            print(f"  Skipped {name} (no checkpoint at {ckpt_path})")


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class GenerateRequest(BaseModel):
    regime: str = "normal"
    n_paths: int = 50
    model: str = "ddpm"
    asset_idx: int = 0


class StylizedFactsRequest(BaseModel):
    model: str = "ddpm"
    n_samples: int = 500


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    html_path = os.path.join(os.path.dirname(__file__), "index.html")
    if os.path.exists(html_path):
        with open(html_path) as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>Frontend not found</h1>")


@app.get("/api/models")
async def list_models():
    return {
        "models": list(LOADED_MODELS.keys()),
        "assets": ASSET_NAMES,
    }


@app.post("/api/generate")
async def generate(req: GenerateRequest):
    if req.model not in LOADED_MODELS:
        raise HTTPException(404, f"Model '{req.model}' not loaded. Available: {list(LOADED_MODELS.keys())}")

    model = LOADED_MODELS[req.model]
    n_paths = min(req.n_paths, 500)

    cond = None
    if req.model == "ddpm" and model.cond_dim > 0:
        regime_vectors = get_regime_conditioning_vectors()
        if req.regime in regime_vectors:
            cond = regime_vectors[req.regime]

    try:
        if req.model == "ddpm":
            synthetic = model.generate(n_paths, cond=cond)
        else:
            synthetic = model.generate(n_paths)
    except Exception as e:
        raise HTTPException(500, f"Generation failed: {e}")

    asset_idx = min(req.asset_idx, synthetic.shape[2] - 1) if synthetic.ndim == 3 else 0

    if synthetic.ndim == 3:
        paths_raw = synthetic[:, :, asset_idx]
    else:
        paths_raw = synthetic

    # De-normalize if scaler available
    if SCALER and "mean" in SCALER and asset_idx < len(SCALER["mean"]):
        paths_denorm = paths_raw * SCALER["std"][asset_idx] + SCALER["mean"][asset_idx]
    else:
        paths_denorm = paths_raw

    prices = np.exp(np.cumsum(paths_denorm, axis=1)) * 100
    terminal = prices[:, -1].tolist()

    return {
        "prices": prices.tolist(),
        "terminal": terminal,
        "mean_terminal": round(float(np.mean(terminal)), 1),
        "var_5": round(float(np.percentile(terminal, 5)), 1),
        "n_paths": n_paths,
        "regime": req.regime,
        "model": req.model,
        "seq_len": int(prices.shape[1]),
        "asset": ASSET_NAMES[asset_idx] if asset_idx < len(ASSET_NAMES) else f"Asset {asset_idx}",
    }


@app.post("/api/stylized-facts")
async def stylized_facts(req: StylizedFactsRequest):
    if req.model not in LOADED_MODELS:
        raise HTTPException(404, f"Model '{req.model}' not loaded")

    model = LOADED_MODELS[req.model]
    try:
        synthetic = model.generate(min(req.n_samples, 1000))
    except Exception as e:
        raise HTTPException(500, f"Generation failed: {e}")

    results = run_all_tests(synthetic)
    n_pass = count_passes(results)

    return _to_native({
        "model": req.model,
        "results": results,
        "n_pass": n_pass,
        "n_total": len(results),
    })


@app.get("/api/comparison")
async def comparison():
    if not LOADED_MODELS:
        raise HTTPException(400, "No models loaded")

    real = REAL_DATA.get("windows")
    if real is None:
        raise HTTPException(400, "No real data loaded")

    comparison_results = {}
    for name, model in LOADED_MODELS.items():
        try:
            synthetic = model.generate(min(500, real.shape[0]))
            sf = run_all_tests(synthetic)
            metrics = full_evaluation(real[:500], synthetic[:500])
            comparison_results[name] = {
                "stylized_facts": sf,
                "n_pass": count_passes(sf),
                **metrics,
            }
        except Exception as e:
            comparison_results[name] = {"error": str(e)}

    return _to_native(comparison_results)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Demo server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--data-dir", default=os.path.join(PROJECT_ROOT, "data"))
    parser.add_argument("--checkpoints-dir", default=os.path.join(PROJECT_ROOT, "checkpoints"))
    args = parser.parse_args()

    print("Loading models...")
    load_all_models(args.checkpoints_dir, args.data_dir)
    print(f"Loaded {len(LOADED_MODELS)} models: {list(LOADED_MODELS.keys())}")

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

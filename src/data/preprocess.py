"""
Preprocess raw price / macro data into model-ready tensors.

Produces:
  - Log returns  (n_days x n_assets)
  - Windowed sequences for temporal models
  - Macro conditioning vectors aligned by date
"""

from __future__ import annotations

import os
import json
import argparse

import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")


def load_prices(data_dir: str | None = None) -> pd.DataFrame:
    if data_dir is None:
        data_dir = DATA_DIR
    path = os.path.join(data_dir, "prices.csv")
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df


def load_macro(data_dir: str | None = None) -> pd.DataFrame | None:
    if data_dir is None:
        data_dir = DATA_DIR
    path = os.path.join(data_dir, "macro.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute log returns and drop the first NaN row."""
    returns = np.log(prices / prices.shift(1)).dropna()
    return returns


def compute_yield_curve_slope(macro: pd.DataFrame) -> pd.Series | None:
    if macro is None or "GS10" not in macro.columns or "GS2" not in macro.columns:
        return None
    return macro["GS10"] - macro["GS2"]


def compute_credit_spread(macro: pd.DataFrame) -> pd.Series | None:
    if macro is None or "BAMLH0A0HYM2" not in macro.columns:
        return None
    return macro["BAMLH0A0HYM2"]


def make_windows(
    returns: np.ndarray,
    window_size: int = 60,
    stride: int = 1,
) -> np.ndarray:
    """Slice a (T, D) array into overlapping windows of shape (N, W, D)."""
    T, D = returns.shape
    n_windows = (T - window_size) // stride + 1
    windows = np.empty((n_windows, window_size, D), dtype=returns.dtype)
    for i in range(n_windows):
        start = i * stride
        windows[i] = returns[start : start + window_size]
    return windows


def make_window_dates(
    dates: np.ndarray,
    window_size: int = 60,
    stride: int = 1,
) -> list[tuple]:
    """Return (start_date, end_date) for each window."""
    n_windows = (len(dates) - window_size) // stride + 1
    result = []
    for i in range(n_windows):
        start = i * stride
        result.append((dates[start], dates[start + window_size - 1]))
    return result


def prepare_dataset(
    data_dir: str | None = None,
    window_size: int = 60,
    stride: int = 1,
    exclude_vix: bool = True,
) -> dict:
    """
    Full preprocessing pipeline.

    Returns a dict with:
      - returns_df: pd.DataFrame of log returns
      - windows: np.ndarray (N, W, D)
      - scaler_params: dict with mean/std for de-normalization
      - asset_names: list[str]
      - macro_df: pd.DataFrame or None
      - vix: pd.DataFrame or None
      - window_dates: list of (start, end) date tuples
    """
    prices = load_prices(data_dir)

    vix = None
    if exclude_vix and "^VIX" in prices.columns:
        vix = prices[["^VIX"]].copy()
        prices = prices.drop(columns=["^VIX"])

    prices = prices.dropna(axis=1, thresh=int(len(prices) * 0.8))
    prices = prices.ffill().bfill()

    returns_df = compute_log_returns(prices)
    asset_names = list(returns_df.columns)

    returns_arr = returns_df.values.astype(np.float32)
    mu = returns_arr.mean(axis=0)
    sigma = returns_arr.std(axis=0) + 1e-8
    normalized = (returns_arr - mu) / sigma

    windows = make_windows(normalized, window_size=window_size, stride=stride)
    dates = returns_df.index.values
    window_dates = make_window_dates(dates, window_size=window_size, stride=stride)

    macro = load_macro(data_dir)

    return {
        "returns_df": returns_df,
        "prices_df": prices,
        "windows": windows,
        "scaler_params": {"mean": mu, "std": sigma},
        "asset_names": asset_names,
        "macro_df": macro,
        "vix": vix,
        "window_dates": window_dates,
    }


def main():
    parser = argparse.ArgumentParser(description="Preprocess market data")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--window-size", type=int, default=60)
    parser.add_argument("--stride", type=int, default=1)
    args = parser.parse_args()

    dataset = prepare_dataset(args.data_dir, args.window_size, args.stride)
    print(f"Returns shape: {dataset['returns_df'].shape}")
    print(f"Windows shape: {dataset['windows'].shape}")
    print(f"Assets: {dataset['asset_names']}")

    out_dir = args.data_dir or DATA_DIR
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "windows.npy"), dataset["windows"])
    np.save(os.path.join(out_dir, "scaler_mean.npy"), dataset["scaler_params"]["mean"])
    np.save(os.path.join(out_dir, "scaler_std.npy"), dataset["scaler_params"]["std"])
    dataset["returns_df"].to_csv(os.path.join(out_dir, "returns.csv"))

    with open(os.path.join(out_dir, "asset_names.json"), "w") as f:
        json.dump(dataset["asset_names"], f)

    print(f"Saved preprocessed data to {out_dir}")


if __name__ == "__main__":
    main()

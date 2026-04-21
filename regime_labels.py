"""
Regime labeling and macro conditioning vector construction.

Labels each trading day (and window) as one of:
  - crisis  (high VIX, high realized vol, inverted yield curve)
  - calm    (low VIX, low realized vol)
  - normal  (everything else)

Builds conditioning vectors from macro features for each training window.
"""

from __future__ import annotations

import os
import json

import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")


def label_daily_regimes(
    returns_df: pd.DataFrame,
    vix_df: pd.DataFrame | None = None,
    macro_df: pd.DataFrame | None = None,
    vix_crisis: float = 25.0,
    vix_calm: float = 15.0,
    rvol_window: int = 20,
) -> pd.Series:
    """
    Assign a regime label to each trading day.

    Uses a combination of:
      1. VIX level (primary signal if available)
      2. Realized volatility of the first asset (SPY proxy)
      3. Yield curve slope (inverted = crisis boost)

    Returns:
        pd.Series with values in {0=normal, 1=crisis, 2=calm}
    """
    idx = returns_df.index
    labels = pd.Series(0, index=idx, name="regime", dtype=int)

    # --- VIX-based labeling ---
    if vix_df is not None and len(vix_df) > 0:
        vix_col = vix_df.columns[0] if isinstance(vix_df, pd.DataFrame) else "^VIX"
        vix_vals = vix_df[vix_col] if isinstance(vix_df, pd.DataFrame) else vix_df
        vix_aligned = vix_vals.reindex(idx).ffill().bfill()

        labels[vix_aligned > vix_crisis] = 1   # crisis
        labels[vix_aligned < vix_calm] = 2     # calm
    else:
        # Fallback: use realized volatility of the first asset
        spy_ret = returns_df.iloc[:, 0]
        rvol = spy_ret.rolling(rvol_window).std() * np.sqrt(252)
        rvol = rvol.bfill()
        q75 = rvol.quantile(0.75)
        q25 = rvol.quantile(0.25)
        labels[rvol > q75] = 1   # crisis
        labels[rvol < q25] = 2   # calm

    # --- Yield curve inversion boost ---
    if macro_df is not None and "GS10" in macro_df.columns and "GS2" in macro_df.columns:
        slope = (macro_df["GS10"] - macro_df["GS2"]).reindex(idx).ffill().bfill()
        inverted = slope < 0
        labels.loc[inverted & (labels == 0)] = 1

    return labels


def build_macro_conditioning(
    returns_df: pd.DataFrame,
    vix_df: pd.DataFrame | None = None,
    macro_df: pd.DataFrame | None = None,
    prices_df: pd.DataFrame | None = None,
    rvol_window: int = 20,
) -> pd.DataFrame:
    """
    Build a per-day conditioning vector from available data.

    Columns (all z-scored):
      yield_curve_slope  -- TLT/SHY log-price spread (long minus short end proxy)
      credit_spread      -- TLT/SPY log-price ratio (flight-to-quality / risk-off proxy)
      rate_level         -- TLT rolling return mean (rate environment proxy)
      vix_level          -- VIX level (from Yahoo Finance ^VIX)
      realized_vol       -- 20-day annualised vol of SPY

    All five features are derived from Yahoo Finance data already downloaded,
    so no FRED API key is required. FRED macro_df is accepted for compatibility
    but is not used.
    """
    idx = returns_df.index
    cond = pd.DataFrame(index=idx)

    # --- Yield curve slope proxy: TLT log-price minus SHY log-price ---
    # Long-end debt rises when rates fall / curve steepens; short-end is sticky.
    if prices_df is not None and "TLT" in prices_df.columns and "SHY" in prices_df.columns:
        tlt = np.log(prices_df["TLT"].reindex(idx).ffill().bfill())
        shy = np.log(prices_df["SHY"].reindex(idx).ffill().bfill())
        cond["yield_curve_slope"] = (tlt - shy).values
    else:
        cond["yield_curve_slope"] = 0.0

    # --- Credit / risk-off spread proxy: TLT log-price minus SPY log-price ---
    # When credit stress rises, capital rotates from equities (SPY) to Treasuries (TLT).
    if prices_df is not None and "TLT" in prices_df.columns and "SPY" in prices_df.columns:
        tlt = np.log(prices_df["TLT"].reindex(idx).ffill().bfill())
        spy_p = np.log(prices_df["SPY"].reindex(idx).ffill().bfill())
        cond["credit_spread"] = (tlt - spy_p).values
    else:
        cond["credit_spread"] = 0.0

    # --- Rate level proxy: 60-day rolling mean of TLT log returns ---
    # Sustained TLT gains signal a falling-rate (loose) environment; losses signal tightening.
    if prices_df is not None and "TLT" in prices_df.columns:
        tlt_ret = np.log(prices_df["TLT"] / prices_df["TLT"].shift(1))
        tlt_ret = tlt_ret.reindex(idx).ffill().bfill()
        cond["rate_level"] = tlt_ret.rolling(60, min_periods=1).mean().values
    else:
        cond["rate_level"] = 0.0

    # --- VIX level (downloaded from Yahoo Finance as ^VIX) ---
    if vix_df is not None and len(vix_df) > 0:
        vix_col = vix_df.columns[0] if isinstance(vix_df, pd.DataFrame) else "^VIX"
        vix_vals = vix_df[vix_col] if isinstance(vix_df, pd.DataFrame) else vix_df
        cond["vix_level"] = vix_vals.reindex(idx).ffill().bfill().values
    else:
        cond["vix_level"] = 0.0

    # --- Realised volatility: 20-day annualised vol of SPY ---
    spy_ret = returns_df.iloc[:, 0]
    cond["realized_vol"] = spy_ret.rolling(rvol_window, min_periods=1).std().bfill() * np.sqrt(252)

    # Z-score each column
    for col in cond.columns:
        mu = cond[col].mean()
        sigma = cond[col].std() + 1e-8
        cond[col] = (cond[col] - mu) / sigma

    return cond.astype(np.float32)


def assign_window_regimes(
    daily_regimes: pd.Series,
    window_dates: list[tuple],
) -> np.ndarray:
    """
    Assign a single regime label to each window via majority vote.

    Returns:
        np.ndarray of shape (N,) with values in {0, 1, 2}
    """
    labels = np.zeros(len(window_dates), dtype=np.int64)
    for i, (start, end) in enumerate(window_dates):
        mask = (daily_regimes.index >= start) & (daily_regimes.index <= end)
        window_labels = daily_regimes[mask]
        if len(window_labels) > 0:
            labels[i] = int(window_labels.mode().iloc[0])
    return labels


def assign_window_conditioning(
    daily_cond: pd.DataFrame,
    window_dates: list[tuple],
) -> np.ndarray:
    """
    Build a conditioning vector for each window by averaging daily values.

    Returns:
        np.ndarray of shape (N, cond_dim)
    """
    cond_dim = daily_cond.shape[1]
    cond = np.zeros((len(window_dates), cond_dim), dtype=np.float32)
    for i, (start, end) in enumerate(window_dates):
        mask = (daily_cond.index >= start) & (daily_cond.index <= end)
        chunk = daily_cond[mask]
        if len(chunk) > 0:
            cond[i] = chunk.values.mean(axis=0)
    return cond


def get_regime_conditioning_vectors() -> dict[str, np.ndarray]:
    """
    Return canonical conditioning vectors for each regime.
    Used at generation time to request a specific regime.

    Feature order (all z-scored):
      [yield_curve_slope, credit_spread, rate_level, vix_level, realized_vol]

    Signs reflect the Yahoo-derived proxies:
      yield_curve_slope: TLT - SHY log-price  (high = steep/normal, low = flat/inverted)
      credit_spread:     TLT - SPY log-price  (high = risk-off/crisis, low = risk-on/calm)
      rate_level:        60d mean TLT return   (high = rates falling, low = rates rising)
      vix_level:         VIX                  (high = fearful, low = complacent)
      realized_vol:      SPY 20d annualised vol (high = turbulent, low = quiet)
    """
    return {
        # Crisis: inverted curve, strong flight-to-quality, high VIX & vol
        "crisis": np.array([-1.5,  2.0,  1.0,  2.0,  2.0], dtype=np.float32),
        # Calm:   steep curve, risk-on, rates stable, low VIX & vol
        "calm":   np.array([ 1.0, -1.5, -0.5, -1.5, -1.5], dtype=np.float32),
        # Normal: all features near their historical average
        "normal": np.array([ 0.0,  0.0,  0.0,  0.0,  0.0], dtype=np.float32),
    }


def prepare_regime_data(
    returns_df: pd.DataFrame,
    vix_df: pd.DataFrame | None,
    macro_df: pd.DataFrame | None,
    window_dates: list[tuple],
    data_dir: str | None = None,
    prices_df: pd.DataFrame | None = None,
) -> dict:
    """
    Full regime-labeling pipeline. Computes daily regimes, window labels,
    and conditioning vectors, then saves to disk.
    """
    if data_dir is None:
        data_dir = DATA_DIR

    daily_regimes = label_daily_regimes(returns_df, vix_df, macro_df)
    daily_cond = build_macro_conditioning(returns_df, vix_df, macro_df, prices_df=prices_df)

    window_regimes = assign_window_regimes(daily_regimes, window_dates)
    window_cond = assign_window_conditioning(daily_cond, window_dates)

    os.makedirs(data_dir, exist_ok=True)
    np.save(os.path.join(data_dir, "window_regimes.npy"), window_regimes)
    np.save(os.path.join(data_dir, "window_cond.npy"), window_cond)
    daily_regimes.to_csv(os.path.join(data_dir, "daily_regimes.csv"))

    regime_counts = {
        "normal": int((window_regimes == 0).sum()),
        "crisis": int((window_regimes == 1).sum()),
        "calm": int((window_regimes == 2).sum()),
    }
    print(f"  Regime distribution: {regime_counts}")

    return {
        "daily_regimes": daily_regimes,
        "daily_cond": daily_cond,
        "window_regimes": window_regimes,
        "window_cond": window_cond,
    }

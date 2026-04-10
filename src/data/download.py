"""
Download financial market data from Yahoo Finance and FRED.

Data covers:
  - S&P 500 sector ETFs, Treasuries, gold, oil, dollar index, VIX
  - FRED macro: yield curve slope, credit spreads, fed funds rate
"""

from __future__ import annotations

import os
import argparse
from datetime import datetime

import pandas as pd
import numpy as np
import yfinance as yf

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")

MARKET_TICKERS = {
    "SPY": "S&P 500",
    "XLK": "Technology",
    "XLF": "Financials",
    "XLE": "Energy",
    "XLV": "Healthcare",
    "XLI": "Industrials",
    "XLP": "Consumer Staples",
    "XLY": "Consumer Discretionary",
    "XLU": "Utilities",
    "XLB": "Materials",
    "XLRE": "Real Estate",
    "XLC": "Communication Services",
    "TLT": "20+ Year Treasury",
    "IEF": "7-10 Year Treasury",
    "SHY": "1-3 Year Treasury",
    "GLD": "Gold",
    "USO": "Oil",
    "UUP": "Dollar Index",
    "^VIX": "VIX",
}

FRED_SERIES = {
    "GS10": "10-Year Treasury Yield",
    "GS2": "2-Year Treasury Yield",
    "BAMLH0A0HYM2": "High Yield OAS",
    "BAMLC0A4CBBB": "BBB Corporate OAS",
    "FEDFUNDS": "Fed Funds Rate",
}


def download_market_data(
    start: str = "2005-01-01",
    end: str | None = None,
    output_dir: str | None = None,
) -> pd.DataFrame:
    """Download daily price data from Yahoo Finance."""
    if end is None:
        end = datetime.today().strftime("%Y-%m-%d")
    if output_dir is None:
        output_dir = DATA_DIR
    os.makedirs(output_dir, exist_ok=True)

    tickers = list(MARKET_TICKERS.keys())
    print(f"Downloading {len(tickers)} tickers from Yahoo Finance ...")
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, threads=True)

    if len(tickers) == 1:
        prices = raw[["Close"]].copy()
        prices.columns = tickers
    elif isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"] if "Close" in raw.columns.get_level_values(0) else raw
        if isinstance(prices.columns, pd.MultiIndex):
            prices = prices.droplevel(0, axis=1)
    else:
        prices = raw[["Close"]].copy()

    prices.columns.name = None
    prices = prices.dropna(how="all")

    out_path = os.path.join(output_dir, "prices.csv")
    prices.to_csv(out_path)
    print(f"Saved prices -> {out_path}  ({prices.shape})")
    return prices


def download_fred_data(
    start: str = "2005-01-01",
    end: str | None = None,
    output_dir: str | None = None,
    api_key: str | None = None,
) -> pd.DataFrame | None:
    """Download macro data from FRED (requires fredapi + API key)."""
    if output_dir is None:
        output_dir = DATA_DIR
    os.makedirs(output_dir, exist_ok=True)

    if api_key is None:
        api_key = os.environ.get("FRED_API_KEY")
    if api_key is None:
        print(
            "FRED_API_KEY not set. Skipping FRED download.\n"
            "Get a free key at https://fred.stlouisfed.org/docs/api/api_key.html\n"
            "Then: export FRED_API_KEY=your_key"
        )
        return None

    from fredapi import Fred

    fred = Fred(api_key=api_key)
    frames = {}
    for series_id, label in FRED_SERIES.items():
        print(f"  FRED: {label} ({series_id}) ...")
        try:
            s = fred.get_series(series_id, observation_start=start, observation_end=end)
            frames[series_id] = s
        except Exception as e:
            print(f"    WARNING: could not fetch {series_id}: {e}")

    if not frames:
        return None

    macro = pd.DataFrame(frames)
    macro.index.name = "Date"
    macro = macro.ffill()

    out_path = os.path.join(output_dir, "macro.csv")
    macro.to_csv(out_path)
    print(f"Saved macro -> {out_path}  ({macro.shape})")
    return macro


def main():
    parser = argparse.ArgumentParser(description="Download market data")
    parser.add_argument("--start", default="2005-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--fred-key", default=None)
    args = parser.parse_args()

    prices = download_market_data(args.start, args.end, args.output_dir)
    macro = download_fred_data(args.start, args.end, args.output_dir, args.fred_key)

    print("\n=== Download complete ===")
    print(f"Prices: {prices.shape[0]} days x {prices.shape[1]} assets")
    if macro is not None:
        print(f"Macro:  {macro.shape[0]} days x {macro.shape[1]} series")


if __name__ == "__main__":
    main()

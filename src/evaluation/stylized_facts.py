"""
Statistical tests for the six stylized facts of financial returns.

Each test takes a returns array and returns a dict with test statistics,
p-values, and a boolean pass/fail. Tests can run per-asset or on aggregated
series.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch


def _to_1d(returns: np.ndarray, method: str = "mean") -> np.ndarray:
    """Convert multi-asset returns to 1D for univariate tests."""
    if returns.ndim == 3:
        returns = returns.reshape(-1, returns.shape[-1])
    if returns.ndim == 2:
        if method == "mean":
            returns = returns.mean(axis=1)
        else:
            returns = returns[:, 0]
    return returns[np.isfinite(returns)]


def test_fat_tails(returns: np.ndarray, alpha: float = 0.05) -> dict:
    """
    Test 1: Fat tails -- returns should be heavier-tailed than Gaussian.

    Metrics: excess kurtosis (should be > 0), Jarque-Bera test (should reject normality).
    """
    flat = _to_1d(returns)
    kurt = float(stats.kurtosis(flat, fisher=True))
    jb_stat, jb_p = stats.jarque_bera(flat)

    return {
        "name": "Fat Tails",
        "excess_kurtosis": round(kurt, 4),
        "jarque_bera_stat": round(float(jb_stat), 4),
        "jarque_bera_p": round(float(jb_p), 6),
        "pass": kurt > 0 and jb_p < alpha,
    }


def test_fat_tails_per_asset(returns: np.ndarray, alpha: float = 0.05) -> dict:
    """Per-asset fat tails test. Returns aggregate pass rate."""
    if returns.ndim == 3:
        returns = returns.reshape(-1, returns.shape[-1])
    if returns.ndim == 1:
        return test_fat_tails(returns, alpha)

    n_assets = returns.shape[1]
    n_pass = 0
    kurts = []
    for i in range(n_assets):
        col = returns[:, i]
        col = col[np.isfinite(col)]
        if len(col) < 20:
            continue
        k = float(stats.kurtosis(col, fisher=True))
        _, p = stats.jarque_bera(col)
        kurts.append(k)
        if k > 0 and p < alpha:
            n_pass += 1

    pass_rate = n_pass / max(n_assets, 1)
    return {
        "name": "Fat Tails",
        "pass_rate": round(pass_rate, 4),
        "mean_kurtosis": round(float(np.mean(kurts)), 4) if kurts else None,
        "n_assets_tested": n_assets,
        "pass": pass_rate >= 0.6,
    }


def test_volatility_clustering(returns: np.ndarray, nlags: int = 10, alpha: float = 0.05) -> dict:
    """
    Test 2: Volatility clustering -- large moves follow large moves.
    ARCH-LM test on returns.
    """
    flat = _to_1d(returns)
    try:
        arch_result = het_arch(flat, nlags=nlags)
        arch_lm_stat = float(arch_result[0])
        arch_lm_p = float(arch_result[1])
    except Exception:
        arch_lm_stat, arch_lm_p = np.nan, np.nan

    sq = flat ** 2
    acf1_sq = float(np.corrcoef(sq[:-1], sq[1:])[0, 1]) if len(sq) > 1 else np.nan
    passed = arch_lm_p < alpha if np.isfinite(arch_lm_p) else False

    return {
        "name": "Volatility Clustering",
        "arch_lm_stat": round(arch_lm_stat, 4) if np.isfinite(arch_lm_stat) else None,
        "arch_lm_p": round(arch_lm_p, 6) if np.isfinite(arch_lm_p) else None,
        "acf1_squared_returns": round(acf1_sq, 4) if np.isfinite(acf1_sq) else None,
        "pass": passed,
    }


def test_leverage_effect(returns: np.ndarray) -> dict:
    """
    Test 3: Leverage effect -- negative returns increase future volatility.
    Metric: correlation between r_t and |r_{t+1}| should be negative.
    """
    flat = _to_1d(returns)
    r_t = flat[:-1]
    abs_r_next = np.abs(flat[1:])
    corr = float(np.corrcoef(r_t, abs_r_next)[0, 1])

    return {
        "name": "Leverage Effect",
        "corr_r_abs_r_next": round(corr, 4),
        "pass": corr < 0,
    }


def test_slow_acf_decay(returns: np.ndarray, max_lag: int = 100) -> dict:
    """
    Test 4: Slow autocorrelation decay in absolute returns.
    ACF of |returns| should remain positive and decay slowly.
    """
    flat = _to_1d(returns)
    abs_ret = np.abs(flat)

    n = len(abs_ret)
    max_lag = min(max_lag, n // 4)
    mean_abs = abs_ret.mean()
    var_abs = abs_ret.var()

    acf_values = []
    for lag in range(1, max_lag + 1):
        cov = np.mean((abs_ret[lag:] - mean_abs) * (abs_ret[:-lag] - mean_abs))
        acf_values.append(cov / var_abs if var_abs > 0 else 0.0)

    acf_arr = np.array(acf_values)
    acf_at_10 = float(acf_arr[9]) if len(acf_arr) > 9 else np.nan
    acf_at_50 = float(acf_arr[49]) if len(acf_arr) > 49 else np.nan
    n_positive = int(np.sum(acf_arr[:20] > 0))

    return {
        "name": "Slow ACF Decay",
        "acf_lag_10": round(acf_at_10, 4) if np.isfinite(acf_at_10) else None,
        "acf_lag_50": round(acf_at_50, 4) if np.isfinite(acf_at_50) else None,
        "n_positive_first_20": n_positive,
        "acf_values": [round(float(v), 4) for v in acf_arr],
        "pass": n_positive >= 15,
    }


def test_cross_asset_correlations(returns: np.ndarray, window: int = 60,
                                   n_pairs: int = 5) -> dict:
    """
    Test 5: Time-varying cross-asset correlations.
    Rolling correlations between multiple asset pairs should vary over time.
    """
    if returns.ndim == 3:
        returns = returns.reshape(-1, returns.shape[-1])
    if returns.ndim == 1:
        return {"name": "Cross-Asset Correlations", "note": "Requires multi-asset", "pass": False}

    n_t, n_assets = returns.shape
    if n_assets < 2 or n_t < window + 10:
        return {"name": "Cross-Asset Correlations", "note": "Insufficient data", "pass": False}

    pairs_to_test = min(n_pairs, n_assets * (n_assets - 1) // 2)
    all_stds = []

    pair_idx = 0
    for a in range(min(n_assets, 10)):
        for b in range(a + 1, min(n_assets, 10)):
            if pair_idx >= pairs_to_test:
                break
            rolling_corrs = []
            for t in range(window, n_t):
                chunk = returns[t - window : t, :]
                c = np.corrcoef(chunk[:, a], chunk[:, b])[0, 1]
                if np.isfinite(c):
                    rolling_corrs.append(c)
            if rolling_corrs:
                all_stds.append(float(np.std(rolling_corrs)))
            pair_idx += 1
        if pair_idx >= pairs_to_test:
            break

    mean_std = float(np.mean(all_stds)) if all_stds else 0.0

    return {
        "name": "Cross-Asset Correlations",
        "mean_rolling_corr_std": round(mean_std, 4),
        "n_pairs_tested": len(all_stds),
        "pass": mean_std > 0.05,
    }


def test_no_raw_autocorrelation(returns: np.ndarray, nlags: int = 10, alpha: float = 0.05) -> dict:
    """
    Test 6: No autocorrelation in raw returns.
    Ljung-Box test on raw returns should NOT reject.
    """
    flat = _to_1d(returns)
    try:
        lb_result = acorr_ljungbox(flat, lags=nlags, return_df=True)
        lb_p_min = float(lb_result["lb_pvalue"].min())
        lb_stat_max = float(lb_result["lb_stat"].max())
    except Exception:
        lb_p_min, lb_stat_max = np.nan, np.nan

    passed = lb_p_min > alpha if np.isfinite(lb_p_min) else False

    return {
        "name": "No Raw Autocorrelation",
        "ljung_box_stat_max": round(lb_stat_max, 4) if np.isfinite(lb_stat_max) else None,
        "ljung_box_p_min": round(lb_p_min, 6) if np.isfinite(lb_p_min) else None,
        "pass": passed,
    }


def run_all_tests(returns: np.ndarray, per_asset: bool = False) -> list[dict]:
    """Run all 6 stylized fact tests and return a list of result dicts."""
    if per_asset:
        fat_tail_fn = test_fat_tails_per_asset
    else:
        fat_tail_fn = test_fat_tails

    tests = [
        fat_tail_fn,
        test_volatility_clustering,
        test_leverage_effect,
        test_slow_acf_decay,
        test_cross_asset_correlations,
        test_no_raw_autocorrelation,
    ]
    results = []
    for test_fn in tests:
        try:
            result = test_fn(returns)
        except Exception as e:
            result = {"name": test_fn.__name__, "error": str(e), "pass": False}
        results.append(result)
    return results


def count_passes(results: list[dict]) -> int:
    return sum(1 for r in results if r.get("pass"))


def print_report(results: list[dict]) -> None:
    """Pretty-print the stylized facts test report."""
    print("\n" + "=" * 60)
    print("STYLIZED FACTS VALIDATION REPORT")
    print("=" * 60)
    n_pass = 0
    for r in results:
        status = "PASS" if r.get("pass") else "FAIL"
        if r.get("pass"):
            n_pass += 1
        print(f"\n[{status}] {r.get('name', 'Unknown')}")
        for k, v in r.items():
            if k in ("name", "pass", "acf_values"):
                continue
            print(f"  {k}: {v}")
    print(f"\n{'=' * 60}")
    print(f"Result: {n_pass}/{len(results)} tests passed")
    print("=" * 60)


if __name__ == "__main__":
    import argparse, os

    parser = argparse.ArgumentParser(description="Run stylized facts tests")
    parser.add_argument("--model", default="real", help="Model name or 'real' for actual data")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--per-asset", action="store_true")
    args = parser.parse_args()

    data_dir = args.data_dir or os.path.join(os.path.dirname(__file__), "..", "..", "data")

    if args.model == "real":
        returns_path = os.path.join(data_dir, "returns.csv")
        df = pd.read_csv(returns_path, index_col=0, parse_dates=True)
        returns = df.values.astype(np.float32)
        print(f"Testing real data: {returns.shape}")
    else:
        gen_path = os.path.join(data_dir, f"generated_{args.model}.npy")
        returns = np.load(gen_path)
        print(f"Testing {args.model} generated data: {returns.shape}")

    results = run_all_tests(returns, per_asset=args.per_asset)
    print_report(results)

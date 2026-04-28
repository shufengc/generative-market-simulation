"""
Statistical tests for the six stylized facts of financial returns.

Each test accepts a synthetic returns array and an optional real_returns array
for quantitative comparison (as per the proposal's Table 1 criteria).
When real_returns is provided, tests compute |metric_syn - metric_real| gaps;
otherwise they fall back to absolute thresholds.
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox


def _to_1d(returns: np.ndarray, method: str = "mean") -> np.ndarray:
    """Flatten multi-asset / windowed returns to a 1-D series."""
    if returns.ndim == 3:
        returns = returns.reshape(-1, returns.shape[-1])
    if returns.ndim == 2:
        returns = returns.mean(axis=1) if method == "mean" else returns[:, 0]
    return returns[np.isfinite(returns)]


def _to_2d(returns: np.ndarray) -> np.ndarray:
    """Return (T, D) array, collapsing window dim if present."""
    if returns.ndim == 3:
        return returns.reshape(-1, returns.shape[-1])
    if returns.ndim == 1:
        return returns.reshape(-1, 1)
    return returns


# ---------------------------------------------------------------------------
# Hill Estimator helper
# ---------------------------------------------------------------------------

def _hill_estimator(x: np.ndarray, k: int | None = None) -> float:
    """
    Hill tail-index estimator on the absolute values of x.
    Returns α (tail exponent); higher α = lighter tail.
    Uses k = floor(sqrt(n)) tail observations by default.
    """
    x = np.abs(x[np.isfinite(x)])
    n = len(x)
    if n < 20:
        return np.nan
    if k is None:
        k = max(10, int(np.sqrt(n)))   # sqrt(n) is a standard bias-variance tradeoff for k
    k = min(k, n - 1)
    x_sorted = np.sort(x)[::-1]          # descending
    log_ratios = np.log(x_sorted[:k] / x_sorted[k])
    mean_log = log_ratios.mean()
    return float(1.0 / mean_log) if mean_log > 0 else np.nan


# ---------------------------------------------------------------------------
# Hurst Exponent helper (R/S analysis)
# ---------------------------------------------------------------------------

def _hurst_rs(ts: np.ndarray, min_chunk: int = 20) -> float:
    """
    Hurst exponent via rescaled-range (R/S) analysis.
    H > 0.5 → long memory / persistence.
    """
    ts = ts[np.isfinite(ts)]
    n = len(ts)
    if n < 50:
        return np.nan

    max_power = int(np.log2(n))
    ns, rs_means = [], []
    for p in range(4, max_power + 1):
        chunk_size = int(2 ** p)
        if chunk_size < min_chunk or chunk_size > n // 2:
            continue
        n_chunks = n // chunk_size
        rs_vals = []
        for i in range(n_chunks):
            chunk = ts[i * chunk_size : (i + 1) * chunk_size]
            mean_c = chunk.mean()
            devs = np.cumsum(chunk - mean_c)   # cumulative deviation from mean
            R = devs.max() - devs.min()         # range of cumulative deviation
            S = chunk.std()
            if S > 0:
                rs_vals.append(R / S)           # rescaled range for this chunk
        if rs_vals:
            ns.append(np.log(chunk_size))
            rs_means.append(np.log(np.mean(rs_vals)))

    if len(ns) < 2:
        return np.nan
    # Hurst exponent is the slope of log(R/S) vs log(n) in OLS
    H, _ = np.polyfit(ns, rs_means, 1)
    return float(H)


# ---------------------------------------------------------------------------
# Test 1: Fat Tails — Hill estimator + Tail KS
# ---------------------------------------------------------------------------

def test_fat_tails(returns: np.ndarray,
                   real_returns: np.ndarray | None = None,
                   alpha_tol: float = 0.1,
                   alpha: float = 0.05) -> dict:
    """
    Test 1: Fat tails.

    Primary metric: Hill tail-index estimator (α).
    Pass criterion (with real data): |α_syn − α_real| < alpha_tol (default 0.1).
    Pass criterion (standalone): α_syn < 5 (heavier-tailed than thin Gaussian proxy).
    Secondary: two-sample KS test on the upper tail (top 10 %).
    """
    flat_syn = _to_1d(returns)
    alpha_syn = _hill_estimator(flat_syn)

    result: dict = {
        "name": "Fat Tails",
        "hill_alpha_syn": round(alpha_syn, 4) if np.isfinite(alpha_syn) else None,
    }

    if real_returns is not None:
        flat_real = _to_1d(real_returns)
        alpha_real = _hill_estimator(flat_real)
        result["hill_alpha_real"] = round(alpha_real, 4) if np.isfinite(alpha_real) else None

        if np.isfinite(alpha_syn) and np.isfinite(alpha_real):
            gap = abs(alpha_syn - alpha_real)
            result["hill_alpha_gap"] = round(gap, 4)
            passed = gap < alpha_tol
        else:
            passed = False

        # Tail KS on absolute returns (top 10 %)
        abs_syn = np.abs(flat_syn)
        abs_real = np.abs(flat_real)
        thresh = np.quantile(abs_real, 0.90)
        tail_syn = abs_syn[abs_syn > thresh]
        tail_real = abs_real[abs_real > thresh]
        if len(tail_syn) > 5 and len(tail_real) > 5:
            ks_stat, ks_p = stats.ks_2samp(tail_real, tail_syn)
            result["tail_ks_stat"] = round(float(ks_stat), 4)
            result["tail_ks_p"] = round(float(ks_p), 6)
            # non-rejection of KS (p > alpha) is a secondary pass condition
            passed = passed and (ks_p > alpha)
    else:
        # Standalone: just check α is in a "fat-tail" range
        passed = np.isfinite(alpha_syn) and alpha_syn < 5.0

    result["pass"] = passed
    return result


# ---------------------------------------------------------------------------
# Test 2: Volatility Clustering — GARCH(1,1) persistence γ = α + β
# ---------------------------------------------------------------------------

def test_volatility_clustering(returns: np.ndarray,
                                real_returns: np.ndarray | None = None,
                                gamma_tol: float = 0.02) -> dict:
    """
    Test 2: Volatility clustering.

    Fits GARCH(1,1) to the returns series and extracts γ = α + β (persistence).
    Pass criterion (with real data): |γ_syn − γ_real| < gamma_tol (default 0.02).
    Pass criterion (standalone): γ_syn > 0.85 (high persistence, as observed in equity returns).
    Falls back to ARCH-LM if GARCH fitting fails.
    """
    def _fit_garch_gamma(ts: np.ndarray) -> float:
        try:
            from arch import arch_model
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                am = arch_model(ts * 100, vol="Garch", p=1, q=1, dist="normal")
                res = am.fit(disp="off", show_warning=False)
            a = float(res.params.get("alpha[1]", np.nan))
            b = float(res.params.get("beta[1]", np.nan))
            return a + b if np.isfinite(a) and np.isfinite(b) else np.nan
        except Exception:
            return np.nan

    flat_syn = _to_1d(returns)
    gamma_syn = _fit_garch_gamma(flat_syn)

    result: dict = {
        "name": "Volatility Clustering",
        "garch_gamma_syn": round(gamma_syn, 4) if np.isfinite(gamma_syn) else None,
    }

    if real_returns is not None:
        flat_real = _to_1d(real_returns)
        gamma_real = _fit_garch_gamma(flat_real)
        result["garch_gamma_real"] = round(gamma_real, 4) if np.isfinite(gamma_real) else None

        if np.isfinite(gamma_syn) and np.isfinite(gamma_real):
            gap = abs(gamma_syn - gamma_real)
            result["garch_gamma_gap"] = round(gap, 4)
            passed = gap < gamma_tol
        else:
            passed = False
    else:
        passed = np.isfinite(gamma_syn) and gamma_syn > 0.85

    # Always report ARCH-LM as supplementary info
    try:
        from statsmodels.stats.diagnostic import het_arch
        arch_stat, arch_p, _, _ = het_arch(flat_syn, nlags=10)
        result["arch_lm_p"] = round(float(arch_p), 6)
    except Exception:
        pass

    result["pass"] = passed
    return result


# ---------------------------------------------------------------------------
# Test 3: Leverage Effect — correlation(r_t, |r_{t+1}|)
# ---------------------------------------------------------------------------

def _gjr_gamma(series: np.ndarray) -> float:
    """
    Fit GJR-GARCH(1,1,1) and return the asymmetry coefficient γ.
    Returns nan on failure (short series, convergence issues, etc.).
    """
    try:
        from arch import arch_model
        am = arch_model(series * 100, vol="Garch", p=1, o=1, q=1, dist="normal")
        res = am.fit(disp="off", show_warning=False)
        return float(res.params.get("gamma[1]", np.nan))
    except Exception:
        return np.nan


def test_leverage_effect(returns: np.ndarray,
                          real_returns: np.ndarray | None = None) -> dict:
    """
    Test 3: Leverage effect.

    Primary metric: GJR-GARCH(1,1,1) asymmetry coefficient γ.
    γ > 0 means negative shocks raise volatility more than positive (leverage effect).
    Fallback: Pearson corr(r_t, |r_{t+1}|) < 0 if GJR-GARCH fails.
    Pass: γ_syn > 0 (sign test); if real provided, also |γ_syn − γ_real| / max(|γ_real|, 0.01) < 1.
    """
    flat = _to_1d(returns)

    gamma_syn = _gjr_gamma(flat)

    # Fallback to correlation if GJR-GARCH failed
    if not np.isfinite(gamma_syn):
        r_t = flat[:-1]
        abs_r_next = np.abs(flat[1:])
        corr = float(np.corrcoef(r_t, abs_r_next)[0, 1])
        passed = corr < 0
        result: dict = {
            "name": "Leverage Effect",
            "method": "corr_fallback",
            "corr_r_abs_r_next_syn": round(corr, 4),
            "pass": passed,
        }
        if real_returns is not None:
            flat_real = _to_1d(real_returns)
            corr_real = float(np.corrcoef(flat_real[:-1], np.abs(flat_real[1:]))[0, 1])
            result["corr_r_abs_r_next_real"] = round(corr_real, 4)
        return result

    passed = gamma_syn > 0
    result = {
        "name": "Leverage Effect",
        "method": "gjr_garch",
        "gamma_syn": round(gamma_syn, 4),
        "pass": passed,
    }

    if real_returns is not None:
        flat_real = _to_1d(real_returns)
        gamma_real = _gjr_gamma(flat_real)
        if np.isfinite(gamma_real):
            result["gamma_real"] = round(gamma_real, 4)
            gap = abs(gamma_syn - gamma_real) / max(abs(gamma_real), 0.01)
            result["pass"] = bool(gamma_syn > 0 and gap < 1.0)

    return result


# ---------------------------------------------------------------------------
# Test 4: Long Memory — Hurst Exponent (R/S analysis)
# ---------------------------------------------------------------------------

def test_long_memory(returns: np.ndarray,
                     real_returns: np.ndarray | None = None,
                     hurst_tol: float = 0.05) -> dict:
    """
    Test 4: Long memory via Hurst exponent on absolute returns.

    H is estimated on |returns| using R/S analysis.
    Pass criterion (with real data): |H_syn − H_real| < hurst_tol AND H_syn ∈ (0.5, 1).
    Pass criterion (standalone): H_syn ∈ (0.5, 1.0).
    """
    flat_syn = np.abs(_to_1d(returns))
    H_syn = _hurst_rs(flat_syn)

    result: dict = {
        "name": "Long Memory (Hurst)",
        "hurst_syn": round(H_syn, 4) if np.isfinite(H_syn) else None,
    }

    if real_returns is not None:
        flat_real = np.abs(_to_1d(real_returns))
        H_real = _hurst_rs(flat_real)
        result["hurst_real"] = round(H_real, 4) if np.isfinite(H_real) else None

        if np.isfinite(H_syn) and np.isfinite(H_real):
            gap = abs(H_syn - H_real)
            result["hurst_gap"] = round(gap, 4)
            passed = gap < hurst_tol and 0.5 < H_syn < 1.0
        else:
            passed = False
    else:
        passed = np.isfinite(H_syn) and 0.5 < H_syn < 1.0

    result["pass"] = passed
    return result


# ---------------------------------------------------------------------------
# Test 5: Cross-Asset Correlations — top eigenvalue λ₁ + time-varying corr
# ---------------------------------------------------------------------------

def test_cross_asset_correlations(returns: np.ndarray,
                                   real_returns: np.ndarray | None = None,
                                   lambda1_tol: float = 0.05,
                                   rolling_window: int = 60,
                                   n_pairs: int = 5) -> dict:
    """
    Test 5: Cross-asset correlation structure.

    Primary metric: top eigenvalue λ₁ of the correlation matrix (captures market mode).
    Pass criterion (with real data): |λ₁_syn − λ₁_real| / λ₁_real < lambda1_tol (default 5 %).
    Pass criterion (standalone): λ₁_syn > 1.5 (above Marchenko-Pastur upper bound proxy).

    Secondary: mean std of rolling pairwise correlations (time-varying structure).
    """
    arr_syn = _to_2d(returns)
    if arr_syn.shape[1] < 2:
        return {"name": "Cross-Asset Correlations", "note": "Requires multi-asset", "pass": False}

    def _top_eigenvalue(arr: np.ndarray) -> float:
        clean = arr[np.isfinite(arr).all(axis=1)]
        if len(clean) < 10:
            return np.nan
        corr = np.corrcoef(clean.T)
        eigs = np.linalg.eigvalsh(corr)
        return float(eigs[-1])

    lambda1_syn = _top_eigenvalue(arr_syn)
    result: dict = {
        "name": "Cross-Asset Correlations",
        "lambda1_syn": round(lambda1_syn, 4) if np.isfinite(lambda1_syn) else None,
    }

    if real_returns is not None:
        arr_real = _to_2d(real_returns)
        lambda1_real = _top_eigenvalue(arr_real)
        result["lambda1_real"] = round(lambda1_real, 4) if np.isfinite(lambda1_real) else None

        if np.isfinite(lambda1_syn) and np.isfinite(lambda1_real) and lambda1_real > 0:
            rel_gap = abs(lambda1_syn - lambda1_real) / lambda1_real
            result["lambda1_rel_gap"] = round(rel_gap, 4)
            passed = rel_gap < lambda1_tol
        else:
            passed = False
    else:
        passed = np.isfinite(lambda1_syn) and lambda1_syn > 1.5

    # Supplementary: time-varying correlation (rolling std of pairwise corrs)
    n_t, n_assets = arr_syn.shape
    pairs = min(n_pairs, n_assets * (n_assets - 1) // 2)
    all_stds = []
    pair_idx = 0
    for a in range(min(n_assets, 10)):
        for b in range(a + 1, min(n_assets, 10)):
            if pair_idx >= pairs:
                break
            if n_t > rolling_window + 10:
                rc = [
                    np.corrcoef(arr_syn[t - rolling_window:t, a],
                                arr_syn[t - rolling_window:t, b])[0, 1]
                    for t in range(rolling_window, n_t)
                ]
                rc = [v for v in rc if np.isfinite(v)]
                if rc:
                    all_stds.append(float(np.std(rc)))
            pair_idx += 1
        if pair_idx >= pairs:
            break

    if all_stds:
        result["rolling_corr_std_mean"] = round(float(np.mean(all_stds)), 4)

    result["pass"] = passed
    return result


# ---------------------------------------------------------------------------
# Test 6: No Raw Autocorrelation — Ljung-Box + MAA
# ---------------------------------------------------------------------------

def test_no_raw_autocorrelation(returns: np.ndarray,
                                 real_returns: np.ndarray | None = None,
                                 nlags: int = 20,
                                 alpha: float = 0.05,
                                 maa_tol: float = 0.05) -> dict:
    """
    Test 6: No autocorrelation in raw returns.

    Ljung-Box test: p_min > 0.05 (fail to reject → no autocorrelation).
    MAA (Mean Absolute Autocorrelation, lags 1-20) < maa_tol (default 0.05).
    Both must pass.
    """
    flat = _to_1d(returns)

    # Ljung-Box
    try:
        lb = acorr_ljungbox(flat, lags=nlags, return_df=True)
        lb_p_min = float(lb["lb_pvalue"].min())
        lb_stat_max = float(lb["lb_stat"].max())
    except Exception:
        lb_p_min, lb_stat_max = np.nan, np.nan

    # MAA: mean |ACF| for lags 1..nlags
    n = len(flat)
    mean_f = flat.mean()
    var_f = flat.var()
    acf_abs = []
    for lag in range(1, nlags + 1):
        if lag >= n:
            break
        cov = np.mean((flat[lag:] - mean_f) * (flat[:-lag] - mean_f))
        acf_abs.append(abs(cov / var_f) if var_f > 0 else 0.0)
    maa = float(np.mean(acf_abs)) if acf_abs else np.nan

    lb_pass = lb_p_min > alpha if np.isfinite(lb_p_min) else False
    maa_pass = maa < maa_tol if np.isfinite(maa) else False

    result: dict = {
        "name": "No Raw Autocorrelation",
        "ljung_box_p_min": round(lb_p_min, 6) if np.isfinite(lb_p_min) else None,
        "ljung_box_stat_max": round(lb_stat_max, 4) if np.isfinite(lb_stat_max) else None,
        "maa_lags_1_20": round(maa, 6) if np.isfinite(maa) else None,
        "pass": lb_pass and maa_pass,
    }

    if real_returns is not None:
        flat_real = _to_1d(real_returns)
        mean_r = flat_real.mean()
        var_r = flat_real.var()
        acf_abs_r = []
        for lag in range(1, nlags + 1):
            if lag >= len(flat_real):
                break
            cov = np.mean((flat_real[lag:] - mean_r) * (flat_real[:-lag] - mean_r))
            acf_abs_r.append(abs(cov / var_r) if var_r > 0 else 0.0)
        result["maa_lags_1_20_real"] = round(float(np.mean(acf_abs_r)), 6) if acf_abs_r else None

    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_all_tests(returns: np.ndarray,
                  real_returns: np.ndarray | None = None,
                  per_asset: bool = False) -> list[dict]:
    """
    Run all 6 stylized fact tests.

    Args:
        returns:      Synthetic returns. Shape (N, W, D), (T, D), or (T,).
        real_returns: Real returns for quantitative comparison (optional but recommended).
        per_asset:    If True, run fat-tails test per asset (legacy option, ignored for other tests).
    """
    tests = [
        lambda r, rr: test_fat_tails(r, rr),
        lambda r, rr: test_volatility_clustering(r, rr),
        lambda r, rr: test_leverage_effect(r, rr),
        lambda r, rr: test_long_memory(r, rr),
        lambda r, rr: test_cross_asset_correlations(r, rr),
        lambda r, rr: test_no_raw_autocorrelation(r, rr),
    ]
    results = []
    for fn in tests:
        try:
            results.append(fn(returns, real_returns))
        except Exception as e:
            results.append({"name": fn.__name__, "error": str(e), "pass": False})
    return results


def count_passes(results: list[dict]) -> int:
    return sum(1 for r in results if r.get("pass"))


def print_report(results: list[dict]) -> None:
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
            if k in ("name", "pass"):
                continue
            print(f"  {k}: {v}")
    print(f"\n{'=' * 60}")
    print(f"Result: {n_pass}/{len(results)} tests passed")
    print("=" * 60)


if __name__ == "__main__":
    import argparse, os

    parser = argparse.ArgumentParser(description="Run stylized facts tests")
    parser.add_argument("--model", default="real")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--per-asset", action="store_true")
    args = parser.parse_args()

    data_dir = args.data_dir or os.path.join(os.path.dirname(__file__), "..", "..", "data")
    returns_path = os.path.join(data_dir, "returns.csv")
    real_df = pd.read_csv(returns_path, index_col=0, parse_dates=True)
    real_data = real_df.values.astype(np.float32)

    if args.model == "real":
        results = run_all_tests(real_data, per_asset=args.per_asset)
    else:
        gen_path = os.path.join(data_dir, f"generated_{args.model}.npy")
        syn_data = np.load(gen_path)
        results = run_all_tests(syn_data, real_returns=real_data, per_asset=args.per_asset)

    print_report(results)

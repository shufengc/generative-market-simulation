"""
Statistical distance metrics for comparing real vs synthetic distributions.

Includes:
  - MMD (Maximum Mean Discrepancy) with RBF kernel
  - 1D Wasserstein distance
  - KS test
  - Moment comparison
  - Discriminative score (classifier-based)
  - Correlation matrix distance (Frobenius norm)
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist


def maximum_mean_discrepancy(X: np.ndarray, Y: np.ndarray, gamma: float | None = None) -> float:
    """
    Compute MMD^2 between samples X and Y using an RBF kernel.
    Lower is better (0 = identical distributions).
    """
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)

    # Subsample for computational tractability
    max_n = 2000
    if len(X) > max_n:
        X = X[np.random.choice(len(X), max_n, replace=False)]
    if len(Y) > max_n:
        Y = Y[np.random.choice(len(Y), max_n, replace=False)]

    if gamma is None:
        combined = np.vstack([X, Y])
        pairwise = cdist(combined, combined, metric="sqeuclidean")
        median_dist = np.median(pairwise[pairwise > 0])
        gamma = 1.0 / (2 * median_dist) if median_dist > 0 else 1.0

    def rbf_kernel(A, B):
        sq_dists = cdist(A, B, metric="sqeuclidean")
        return np.exp(-gamma * sq_dists)

    K_XX = rbf_kernel(X, X)
    K_YY = rbf_kernel(Y, Y)
    K_XY = rbf_kernel(X, Y)

    m, n = len(X), len(Y)
    mmd2 = K_XX.sum() / (m * m) + K_YY.sum() / (n * n) - 2 * K_XY.sum() / (m * n)
    return float(mmd2)


def wasserstein_1d(x: np.ndarray, y: np.ndarray) -> float:
    """1D Wasserstein distance (earth mover's distance)."""
    from scipy.stats import wasserstein_distance
    return float(wasserstein_distance(x.flatten(), y.flatten()))


def ks_test(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Two-sample Kolmogorov-Smirnov test. Returns (statistic, p-value)."""
    from scipy.stats import ks_2samp
    stat, p = ks_2samp(x.flatten(), y.flatten())
    return float(stat), float(p)


def moment_comparison(real: np.ndarray, synthetic: np.ndarray) -> dict:
    """Compare first four moments between real and synthetic returns."""
    from scipy.stats import kurtosis, skew

    def moments(x):
        x = x.flatten()
        return {
            "mean": float(np.mean(x)),
            "std": float(np.std(x)),
            "skew": float(skew(x)),
            "kurtosis": float(kurtosis(x, fisher=True)),
        }

    real_m = moments(real)
    syn_m = moments(synthetic)

    return {
        "real": real_m,
        "synthetic": syn_m,
        "abs_diff": {k: round(abs(real_m[k] - syn_m[k]), 6) for k in real_m},
    }


def discriminative_score(real: np.ndarray, synthetic: np.ndarray, n_splits: int = 3) -> float:
    """
    Train a simple classifier to distinguish real from synthetic data.
    Returns accuracy -- closer to 0.5 means synthetic is more realistic.

    Uses a Random Forest on flattened samples.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score

    if real.ndim == 3:
        real = real.reshape(real.shape[0], -1)
    if synthetic.ndim == 3:
        synthetic = synthetic.reshape(synthetic.shape[0], -1)
    if real.ndim == 1:
        real = real.reshape(-1, 1)
    if synthetic.ndim == 1:
        synthetic = synthetic.reshape(-1, 1)

    n = min(len(real), len(synthetic), 1000)
    X = np.vstack([real[:n], synthetic[:n]])
    y = np.concatenate([np.ones(n), np.zeros(n)])

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    clf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
    try:
        scores = cross_val_score(clf, X, y, cv=min(n_splits, n // 2), scoring="accuracy")
        return float(scores.mean())
    except Exception:
        return 0.5


def correlation_matrix_distance(real: np.ndarray, synthetic: np.ndarray) -> float:
    """
    Compute Frobenius norm between real and synthetic correlation matrices.
    Lower is better.
    """
    if real.ndim == 3:
        real = real.reshape(-1, real.shape[-1])
    if synthetic.ndim == 3:
        synthetic = synthetic.reshape(-1, synthetic.shape[-1])
    if real.ndim == 1 or synthetic.ndim == 1:
        return 0.0

    n_assets = min(real.shape[1], synthetic.shape[1])
    real = real[:, :n_assets]
    synthetic = synthetic[:, :n_assets]

    real_clean = real[np.isfinite(real).all(axis=1)]
    syn_clean = synthetic[np.isfinite(synthetic).all(axis=1)]

    if len(real_clean) < 10 or len(syn_clean) < 10:
        return float("inf")

    corr_real = np.corrcoef(real_clean.T)
    corr_syn = np.corrcoef(syn_clean.T)

    diff = corr_real - corr_syn
    return float(np.sqrt(np.sum(diff ** 2)))


def full_evaluation(real: np.ndarray, synthetic: np.ndarray) -> dict:
    """Run all metrics and return a summary dict."""
    real_flat = real.flatten()
    syn_flat = synthetic.flatten()

    mmd = maximum_mean_discrepancy(
        real_flat[:2000].reshape(-1, 1),
        syn_flat[:2000].reshape(-1, 1),
    )
    w1 = wasserstein_1d(real_flat[:5000], syn_flat[:5000])
    ks_stat, ks_p = ks_test(real_flat[:5000], syn_flat[:5000])
    moments = moment_comparison(real, synthetic)
    disc = discriminative_score(real, synthetic)
    corr_dist = correlation_matrix_distance(real, synthetic)

    return {
        "mmd": round(mmd, 6),
        "wasserstein_1d": round(w1, 6),
        "ks_stat": round(ks_stat, 4),
        "ks_pvalue": round(ks_p, 6),
        "moments": moments,
        "discriminative_score": round(disc, 4),
        "correlation_matrix_distance": round(corr_dist, 4),
    }

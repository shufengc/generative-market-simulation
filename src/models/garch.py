"""
GARCH-based baseline for synthetic financial return generation.

Uses univariate GARCH(1,1) or EGARCH(1,1) fitted per asset, then generates
paths by sampling from the fitted conditional distribution. Cross-asset
dependence is captured via a static correlation matrix applied to innovations.
"""

from __future__ import annotations

import os
import argparse
import warnings

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from src.models.base_model import BaseGenerativeModel

warnings.filterwarnings("ignore")


class GARCHModel(BaseGenerativeModel):
    """DCC-GARCH-style baseline using the `arch` library."""

    def __init__(self, n_features: int = 18, seq_len: int = 60,
                 vol_model: str = "Garch", device: str = "cpu"):
        super().__init__(name="GARCH", device=device)
        self.n_features = n_features
        self.seq_len = seq_len
        self.vol_model = vol_model
        self.models_fitted = []
        self.corr_matrix = None
        self.means = None

    def train(self, data: np.ndarray, **kwargs) -> dict:
        """
        Fit GARCH(1,1) or EGARCH(1,1) to each asset's return series.

        Args:
            data: (N, seq_len, n_features) windowed returns OR (T, n_features) flat returns.
        """
        from arch import arch_model

        if data.ndim == 3:
            flat = data.reshape(-1, data.shape[-1])
        else:
            flat = data

        self.n_features = flat.shape[1]
        self.means = flat.mean(axis=0)

        residuals = np.zeros_like(flat)
        self.models_fitted = []

        for i in range(self.n_features):
            series = flat[:, i] * 100
            success = False
            params = {}

            for vol_type in [self.vol_model, "Garch"]:
                try:
                    am = arch_model(series, vol=vol_type, p=1, q=1,
                                    mean="AR", lags=1, dist="t")
                    res = am.fit(disp="off", show_warning=False)
                    params = dict(res.params)
                    std_resid = res.resid / res.conditional_volatility
                    std_resid = np.nan_to_num(std_resid, nan=0.0, posinf=0.0, neginf=0.0)
                    residuals[:, i] = std_resid
                    success = True
                    break
                except Exception:
                    continue

            self.models_fitted.append({
                "params": params,
                "success": success,
            })
            if not success:
                residuals[:, i] = flat[:, i]

            if (i + 1) % 5 == 0:
                print(f"  Fitted {i+1}/{self.n_features} assets")

        valid = np.isfinite(residuals).all(axis=1)
        if valid.sum() > 10:
            self.corr_matrix = np.corrcoef(residuals[valid].T)
        else:
            self.corr_matrix = np.eye(self.n_features)

        if not np.all(np.isfinite(self.corr_matrix)):
            self.corr_matrix = np.eye(self.n_features)

        self.is_trained = True
        n_success = sum(1 for m in self.models_fitted if m["success"])
        print(f"  GARCH fitted: {n_success}/{self.n_features} assets successful")
        return {"n_fitted": n_success}

    def generate(self, n_samples: int, seq_len: int | None = None, **kwargs) -> np.ndarray:
        """Generate synthetic returns using fitted GARCH parameters."""
        if seq_len is None:
            seq_len = self.seq_len

        results = np.zeros((n_samples, seq_len, self.n_features))

        reg_matrix = self.corr_matrix + 1e-6 * np.eye(self.n_features)
        eigvals = np.linalg.eigvalsh(reg_matrix)
        if eigvals.min() < 0:
            reg_matrix += (-eigvals.min() + 1e-6) * np.eye(self.n_features)
        L = np.linalg.cholesky(reg_matrix)

        for sample_idx in range(n_samples):
            z_indep = np.random.standard_t(df=5, size=(seq_len, self.n_features))
            z_corr = z_indep @ L.T

            for i, model_info in enumerate(self.models_fitted):
                if not model_info["success"]:
                    results[sample_idx, :, i] = z_corr[:, i] * 0.01
                    continue

                params = model_info["params"]
                omega = params.get("omega", 0.01)
                alpha = params.get("alpha[1]", 0.05)
                beta = params.get("beta[1]", 0.90)
                mu = params.get("Const", 0.0)

                alpha_beta_sum = alpha + beta
                if alpha_beta_sum >= 1.0:
                    alpha = 0.05
                    beta = 0.90

                sigma2 = omega / max(1 - alpha - beta, 0.01)
                returns = np.zeros(seq_len)

                for t in range(seq_len):
                    sigma = np.sqrt(max(sigma2, 1e-10))
                    returns[t] = mu + sigma * z_corr[t, i]
                    sigma2 = omega + alpha * (returns[t] - mu) ** 2 + beta * sigma2
                    sigma2 = max(sigma2, 1e-10)

                results[sample_idx, :, i] = returns / 100

        return results

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        np.savez(
            path,
            corr_matrix=self.corr_matrix,
            means=self.means,
            n_features=self.n_features,
            seq_len=self.seq_len,
            vol_model=self.vol_model,
            params=[m["params"] for m in self.models_fitted],
            success=[m["success"] for m in self.models_fitted],
        )

    def load(self, path: str) -> None:
        data = np.load(path, allow_pickle=True)
        self.corr_matrix = data["corr_matrix"]
        self.means = data["means"]
        self.n_features = int(data["n_features"])
        self.seq_len = int(data["seq_len"])
        params_list = data["params"]
        success_list = data["success"]
        self.models_fitted = [
            {"params": dict(p), "success": bool(s)} for p, s in zip(params_list, success_list)
        ]
        self.is_trained = True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GARCH baseline")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--vol-model", default="Garch", choices=["Garch", "EGARCH"])
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--checkpoint", default="checkpoints/garch.npz")
    args = parser.parse_args()

    data_dir = args.data_dir or os.path.join(os.path.dirname(__file__), "..", "..", "data")

    if args.train:
        returns = pd.read_csv(os.path.join(data_dir, "returns.csv"), index_col=0).values.astype(np.float32)
        print(f"Training GARCH on {returns.shape}")
        model = GARCHModel(n_features=returns.shape[1], vol_model=args.vol_model)
        model.train(returns)
        model.save(args.checkpoint)

    if args.generate:
        model = GARCHModel()
        model.load(args.checkpoint)
        synthetic = model.generate(args.n_samples)
        np.save(os.path.join(data_dir, "generated_garch.npy"), synthetic)
        print(f"Generated {synthetic.shape}")

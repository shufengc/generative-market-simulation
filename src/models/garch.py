"""
GARCH-based baseline for synthetic financial return generation.

Uses univariate GARCH(1,1) or EGARCH(1,1) fitted per asset, then generates
paths by sampling from the fitted conditional distribution. Cross-asset
dependence is captured via a static correlation matrix applied to innovations.
"""

from __future__ import annotations

import json
import math
import os
import argparse
import warnings

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from src.models.base_model import BaseGenerativeModel

warnings.filterwarnings("ignore")

# E[|z|] for standard normal = sqrt(2/pi), used in EGARCH update
_E_ABS_Z_NORMAL = math.sqrt(2.0 / math.pi)


class GARCHModel(BaseGenerativeModel):
    """DCC-GARCH-style baseline using the `arch` library.

    Fits a per-asset GARCH(1,1) or EGARCH(1,1) model with Student-t innovations
    and an AR(1) conditional mean. Cross-asset dependence is captured via a
    Cholesky decomposition of the empirical standardised-residual correlation matrix.
    """

    def __init__(self, n_features: int = 18, seq_len: int = 60,
                 vol_model: str = "Garch", device: str = "cpu"):
        super().__init__(name="GARCH", device=device)
        self.n_features = n_features
        self.seq_len = seq_len
        self.vol_model = vol_model
        self.models_fitted: list[dict] = []
        self.corr_matrix: np.ndarray | None = None
        self.means: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

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
            series = flat[:, i] * 100  # scale to percent returns for numerical stability
            success = False
            params: dict = {}
            fitted_vol_type = "Garch"

            for vol_type in [self.vol_model, "Garch"]:
                try:
                    am = arch_model(series, vol=vol_type, p=1, q=1,
                                    mean="AR", lags=1, dist="t")
                    res = am.fit(disp="off", show_warning=False)
                    params = {str(k): float(v) for k, v in res.params.items()}
                    std_resid = res.resid / res.conditional_volatility
                    std_resid = np.nan_to_num(std_resid, nan=0.0, posinf=0.0, neginf=0.0)
                    residuals[:, i] = std_resid
                    fitted_vol_type = vol_type
                    success = True
                    break
                except Exception:
                    continue

            self.models_fitted.append({
                "params": params,
                "success": success,
                "vol_type": fitted_vol_type,
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
        n_egarch = sum(1 for m in self.models_fitted if m.get("vol_type") == "EGARCH")
        print(f"  GARCH fitted: {n_success}/{self.n_features} assets successful"
              f"  (EGARCH: {n_egarch})")
        return {"n_fitted": n_success, "n_egarch": n_egarch}

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(self, n_samples: int, seq_len: int | None = None, **kwargs) -> np.ndarray:
        """Generate synthetic returns using fitted GARCH / EGARCH parameters.

        Correlated Student-t(5) innovations are drawn jointly via Cholesky
        decomposition of the residual correlation matrix, preserving cross-asset
        dependence structure.
        """
        if seq_len is None:
            seq_len = self.seq_len

        results = np.zeros((n_samples, seq_len, self.n_features))

        # Regularise and factorise the correlation matrix once
        reg_matrix = self.corr_matrix + 1e-6 * np.eye(self.n_features)
        eigvals = np.linalg.eigvalsh(reg_matrix)
        if eigvals.min() < 0:
            reg_matrix += (-eigvals.min() + 1e-6) * np.eye(self.n_features)
        L = np.linalg.cholesky(reg_matrix)

        for sample_idx in range(n_samples):
            # Draw correlated Student-t(5) innovations for all assets simultaneously
            z_indep = np.random.standard_t(df=5, size=(seq_len, self.n_features))
            z_corr = z_indep @ L.T  # (seq_len, n_features)

            for i, model_info in enumerate(self.models_fitted):
                if not model_info["success"]:
                    results[sample_idx, :, i] = z_corr[:, i] * 0.01
                    continue

                params = model_info["params"]
                vol_type = model_info.get("vol_type", "Garch")

                omega = params.get("omega", 0.01)
                alpha = params.get("alpha[1]", 0.05)
                beta  = params.get("beta[1]", 0.90)
                mu    = params.get("Const", 0.0)

                if vol_type == "EGARCH":
                    results[sample_idx, :, i] = self._simulate_egarch(
                        z_corr[:, i], omega, alpha, beta,
                        gamma=params.get("gamma[1]", 0.0), mu=mu,
                    )
                else:
                    results[sample_idx, :, i] = self._simulate_garch(
                        z_corr[:, i], omega, alpha, beta, mu,
                    )

        return results

    # ------------------------------------------------------------------
    # Internal simulation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _simulate_garch(z: np.ndarray, omega: float, alpha: float,
                         beta: float, mu: float) -> np.ndarray:
        """Simulate one return path using GARCH(1,1) equations.

        sigma2_t = omega + alpha * eps2_{t-1} + beta * sigma2_{t-1}
        """
        ab = alpha + beta
        if ab >= 1.0:
            scale = 0.95 / ab
            alpha *= scale
            beta  *= scale

        seq_len = len(z)
        returns = np.empty(seq_len)
        sigma2 = omega / max(1.0 - alpha - beta, 0.01)

        for t in range(seq_len):
            sigma = math.sqrt(max(sigma2, 1e-10))
            returns[t] = mu + sigma * z[t]
            sigma2 = omega + alpha * (returns[t] - mu) ** 2 + beta * sigma2
            sigma2 = max(sigma2, 1e-10)

        return returns / 100.0

    @staticmethod
    def _simulate_egarch(z: np.ndarray, omega: float, alpha: float,
                          beta: float, gamma: float, mu: float) -> np.ndarray:
        """Simulate one return path using EGARCH(1,1) equations.

        ln sigma2_t = omega + alpha*(|z_{t-1}| - E|z|) + gamma*z_{t-1} + beta*ln sigma2_{t-1}

        gamma < 0 captures the leverage effect: negative shocks raise volatility
        more than positive shocks of equal magnitude.
        """
        seq_len = len(z)
        returns = np.empty(seq_len)

        denom = max(1.0 - abs(beta), 0.01)
        log_sigma2 = omega / denom

        for t in range(seq_len):
            sigma2 = math.exp(min(log_sigma2, 20.0))
            sigma  = math.sqrt(max(sigma2, 1e-10))
            returns[t] = mu + sigma * z[t]

            z_t = z[t]
            log_sigma2 = (omega
                          + alpha * (abs(z_t) - _E_ABS_Z_NORMAL)
                          + gamma * z_t
                          + beta  * log_sigma2)

        return returns / 100.0

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_fitted_params_df(self) -> pd.DataFrame:
        """Return a DataFrame summarising the fitted GARCH/EGARCH parameters."""
        rows = []
        for i, m in enumerate(self.models_fitted):
            row = {"asset_idx": i, "vol_type": m.get("vol_type", "Garch"),
                   "success": m["success"]}
            row.update(m["params"])
            rows.append(row)
        return pd.DataFrame(rows)

    def get_unconditional_vol(self) -> np.ndarray:
        """Return per-asset unconditional annualised volatility (%)."""
        vols = []
        for m in self.models_fitted:
            if not m["success"]:
                vols.append(np.nan)
                continue
            p = m["params"]
            omega = p.get("omega", 0.01)
            alpha = p.get("alpha[1]", 0.05)
            beta  = p.get("beta[1]", 0.90)
            denom = max(1.0 - alpha - beta, 1e-4)
            uncond_var = omega / denom
            vols.append(math.sqrt(uncond_var * 252))
        return np.array(vols)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save model to .npz. Parameters are JSON-serialised for portability."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        params_json  = json.dumps([m["params"] for m in self.models_fitted])
        success_arr  = np.array([m["success"]  for m in self.models_fitted], dtype=bool)
        vol_type_arr = np.array([m.get("vol_type", "Garch") for m in self.models_fitted])

        np.savez(
            path,
            corr_matrix = self.corr_matrix,
            means       = self.means,
            n_features  = np.array(self.n_features),
            seq_len     = np.array(self.seq_len),
            vol_model   = np.array(self.vol_model),
            params_json = np.array(params_json),
            success     = success_arr,
            vol_types   = vol_type_arr,
        )

    def load(self, path: str) -> None:
        """Load model from .npz (supports both JSON and legacy pickle formats)."""
        data = np.load(path, allow_pickle=True)
        self.corr_matrix = data["corr_matrix"]
        self.means       = data["means"]
        self.n_features  = int(data["n_features"])
        self.seq_len     = int(data["seq_len"])
        self.vol_model   = str(data["vol_model"])

        success_list = data["success"].tolist()

        if "params_json" in data:
            params_list = json.loads(str(data["params_json"]))
        else:
            params_list = [dict(p) for p in data["params"]]

        if "vol_types" in data:
            vol_type_list = data["vol_types"].tolist()
        else:
            vol_type_list = ["Garch"] * len(params_list)

        self.models_fitted = [
            {"params": p, "success": bool(s), "vol_type": vt}
            for p, s, vt in zip(params_list, success_list, vol_type_list)
        ]
        self.is_trained = True


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GARCH baseline")
    parser.add_argument("--train",     action="store_true")
    parser.add_argument("--generate",  action="store_true")
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--vol-model", default="Garch", choices=["Garch", "EGARCH"])
    parser.add_argument("--data-dir",  default=None)
    parser.add_argument("--checkpoint", default="checkpoints/garch.npz")
    args = parser.parse_args()

    data_dir = args.data_dir or os.path.join(os.path.dirname(__file__), "..", "..", "data")

    if args.train:
        returns = pd.read_csv(
            os.path.join(data_dir, "returns.csv"), index_col=0
        ).values.astype(np.float32)
        print(f"Training GARCH on {returns.shape}")
        model = GARCHModel(n_features=returns.shape[1], vol_model=args.vol_model)
        model.train(returns)
        model.save(args.checkpoint)
        print("\nFitted parameter summary:")
        print(model.get_fitted_params_df().to_string(index=False))

    if args.generate:
        model = GARCHModel()
        model.load(args.checkpoint)
        synthetic = model.generate(args.n_samples)
        np.save(os.path.join(data_dir, "generated_garch.npy"), synthetic)
        print(f"Generated {synthetic.shape}")

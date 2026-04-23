"""
Variational Autoencoder for financial time series generation.

Key design choices (vs. a vanilla GRU-VAE):

1. Bidirectional GRU encoder -> richer posterior summary of the window.
2. Autoregressive GRU decoder with teacher forcing. The baseline approach
   of broadcasting a single latent across all time steps collapses to
   near-constant trajectories; autoregression is what lets the model
   produce heteroscedastic, clustered dynamics.
3. Heteroscedastic Student-t output distribution with a learnable degrees
   of freedom. Directly targets fat tails (Hill alpha) and asymmetric
   conditional variance, which Gaussian/MSE VAEs cannot reach.
4. Free-bits + cyclical KL annealing to avoid posterior collapse.
5. Auxiliary moment-matching regularisers:
     - ACF on raw returns      (targets SF Test 6)
     - ACF on |returns|        (targets SF Tests 2 and 4)
     - Cross-asset correlation (targets SF Test 5) -- required because the
       per-feature Student-t likelihood is conditionally diagonal and would
       otherwise leave the top eigenvalue of Corr(X) far below real data.
6. Per-step low-rank "market factor" emission (k factors).
   The decoder emits a loading matrix L_t (F, K) at every step; we sample
   f_t ~ N(0, I_K) and set the Student-t location to (mu_t + L_t f_t).
   This yields rank-K + diagonal conditional covariance and is what
   actually makes free-running generation preserve the top eigenvalue
   of Corr(X) -- a purely diagonal emission cannot, regardless of how
   well the correlation regulariser is minimised in teacher forcing.
   KL(q(f)||p(f)) = 0, so this stays a valid ELBO.
7. Aggregate-posterior sampling at inference time to mitigate the
   "prior hole" gap between q(z|x) and the N(0, I) generative prior.
"""

from __future__ import annotations

import os
import math
import argparse
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import StudentT
from torch.utils.data import DataLoader, TensorDataset

from src.models.base_model import BaseGenerativeModel


# ---------------------------------------------------------------------------
# Modules
# ---------------------------------------------------------------------------


class Encoder(nn.Module):
    """Bidirectional GRU encoder producing (mu, logvar) of a Gaussian posterior."""

    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int,
                 n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.gru = nn.GRU(
            input_dim, hidden_dim, n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.fc_mu = nn.Linear(hidden_dim * 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim * 2, latent_dim)

    def forward(self, x: torch.Tensor):
        out, _ = self.gru(x)          # (B, T, 2H)
        summary = out[:, -1]           # last step combines forward & backward
        mu = self.fc_mu(summary)
        logvar = self.fc_logvar(summary).clamp(min=-8.0, max=8.0)
        return mu, logvar


class AutoregressiveDecoder(nn.Module):
    """
    Autoregressive GRU decoder with Student-t heteroscedastic emissions.

    At each step t:
        input_t = [x_{t-1}, z]       (x_{-1} is a learnable start token)
        h_t     = GRU(input_t, h_{t-1})
        (mu_t, log_sigma_t) = heads(h_t)
        p(x_t | ...) = StudentT(df, mu_t, sigma_t)
    """

    def __init__(self, latent_dim: int, hidden_dim: int, output_dim: int,
                 seq_len: int, n_layers: int = 2, dropout: float = 0.1,
                 cond_dim: int = 0,
                 factor_dim: int = 2,
                 min_log_sigma: float = -5.0, max_log_sigma: float = 2.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.seq_len = seq_len
        self.n_layers = n_layers
        self.cond_dim = max(int(cond_dim), 0)
        self.factor_dim = factor_dim
        self.min_log_sigma = min_log_sigma
        self.max_log_sigma = max_log_sigma

        init_input_dim = latent_dim + self.cond_dim
        self.fc_init = nn.Linear(init_input_dim, hidden_dim * n_layers)
        self.start_token = nn.Parameter(torch.zeros(1, 1, output_dim))

        self.gru = nn.GRU(
            output_dim + latent_dim + self.cond_dim, hidden_dim, n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.head_mu = nn.Linear(hidden_dim, output_dim)
        self.head_logsigma = nn.Linear(hidden_dim, output_dim)
        if factor_dim > 0:
            self.head_loading = nn.Linear(hidden_dim, output_dim * factor_dim)
            # Start with small loadings so initial emission is near-diagonal
            # Student-t; let training grow them to explain cross-asset variance.
            nn.init.normal_(self.head_loading.weight, mean=0.0, std=0.05)
            nn.init.zeros_(self.head_loading.bias)
        else:
            self.head_loading = None

        # Global learnable Student-t degrees of freedom, shared across features.
        # Parameterised so that df = 2.1 + softplus(log_df) >= 2.1 (finite variance).
        # Initial df ~ 3, reasonable prior for daily returns.
        self.log_df = nn.Parameter(torch.tensor(math.log(math.expm1(1.0))))

    @property
    def df(self) -> torch.Tensor:
        return 2.1 + F.softplus(self.log_df)

    def _init_hidden(self, z: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        B = z.size(0)
        if self.cond_dim > 0:
            if cond is None:
                cond = torch.zeros(B, self.cond_dim, device=z.device, dtype=z.dtype)
            init_input = torch.cat([z, cond], dim=-1)
        else:
            init_input = z
        h0 = torch.tanh(self.fc_init(init_input)).view(B, self.n_layers, self.hidden_dim)
        return h0.transpose(0, 1).contiguous()          # (n_layers, B, H)

    def _heads(self, h_out: torch.Tensor):
        """
        From GRU output (..., H) compute the three heads.
        Returns mu (..., F), sigma (..., F), loading (..., F, K) or None.
        """
        mu = self.head_mu(h_out)
        log_sigma = self.head_logsigma(h_out).clamp(self.min_log_sigma, self.max_log_sigma)
        sigma = torch.exp(log_sigma)
        loading = None
        if self.head_loading is not None:
            lshape = h_out.shape[:-1] + (self.output_dim, self.factor_dim)
            loading = self.head_loading(h_out).view(*lshape)
        return mu, sigma, loading

    def forward_teacher(
        self,
        z: torch.Tensor,
        x: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        teacher_forcing_ratio: float = 1.0,
    ):
        """
        Teacher-forced pass.
        Returns cond_mean (B, T, F), sigma (B, T, F), loading (B, T, F, K) or None.
        cond_mean = mu + L @ f with f ~ N(0, I_K) drawn fresh each call.
        """
        B, T, _ = x.shape
        if self.cond_dim > 0 and cond is None:
            cond = torch.zeros(B, self.cond_dim, device=x.device, dtype=x.dtype)
        cond_step = cond.unsqueeze(1) if cond is not None else None
        z_step = z.unsqueeze(1)

        h = self._init_hidden(z, cond)
        x_prev = self.start_token.expand(B, 1, self.output_dim)

        mu_steps = []
        sigma_steps = []
        loading_steps = [] if self.head_loading is not None else None

        tf_ratio = float(max(0.0, min(1.0, teacher_forcing_ratio)))
        for t in range(T):
            parts = [x_prev, z_step]
            if cond_step is not None:
                parts.append(cond_step)
            gru_in = torch.cat(parts, dim=-1)
            out, h = self.gru(gru_in, h)
            mu_t, sigma_t, loading_t = self._heads(out)

            mu_steps.append(mu_t)
            sigma_steps.append(sigma_t)
            if loading_steps is not None:
                loading_steps.append(loading_t)

            if t + 1 < T:
                x_true = x[:, t:t + 1]
                if tf_ratio >= 1.0:
                    x_prev = x_true
                elif tf_ratio <= 0.0:
                    x_prev = mu_t.detach()
                else:
                    use_teacher = (torch.rand(B, 1, 1, device=x.device) < tf_ratio)
                    x_prev = torch.where(use_teacher, x_true, mu_t.detach())

        mu = torch.cat(mu_steps, dim=1)
        sigma = torch.cat(sigma_steps, dim=1)
        loading = torch.cat(loading_steps, dim=1) if loading_steps is not None else None

        if loading is not None:
            f = torch.randn(B, T, self.factor_dim, device=x.device, dtype=x.dtype)
            # (B, T, F, K) x (B, T, K, 1) -> (B, T, F, 1) -> (B, T, F)
            factor_shift = (loading * f.unsqueeze(-2)).sum(dim=-1)
            cond_mean = mu + factor_shift
        else:
            cond_mean = mu
        return cond_mean, sigma, loading

    @torch.no_grad()
    def sample(
        self,
        z: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> torch.Tensor:
        """Free-running generation from latent z. Returns (B, T, F)."""
        B = z.size(0)
        if self.cond_dim > 0 and cond is None:
            cond = torch.zeros(B, self.cond_dim, device=z.device, dtype=z.dtype)
        cond_step = cond.unsqueeze(1) if cond is not None else None
        h = self._init_hidden(z, cond)
        x_prev = self.start_token.expand(B, 1, self.output_dim).contiguous()
        df = self.df
        outputs = []
        for _ in range(self.seq_len):
            parts = [x_prev, z.unsqueeze(1)]
            if cond_step is not None:
                parts.append(cond_step)
            gru_in = torch.cat(parts, dim=-1)
            out, h = self.gru(gru_in, h)
            mu, sigma, loading = self._heads(out)

            if loading is not None:
                f = torch.randn(B, 1, self.factor_dim, device=z.device, dtype=z.dtype)
                factor_shift = (loading * f.unsqueeze(-2)).sum(dim=-1)
                cond_mean = mu + factor_shift
            else:
                cond_mean = mu

            if deterministic:
                x_t = cond_mean
            else:
                dist = StudentT(df=df, loc=cond_mean, scale=sigma)
                x_t = dist.rsample()
            outputs.append(x_t)
            x_prev = x_t
        return torch.cat(outputs, dim=1)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _batch_acf(x: torch.Tensor, max_lag: int) -> torch.Tensor:
    """
    Differentiable per-window ACF averaged over the batch and feature dims.

    Args:
        x: (B, T, F) tensor of returns.
        max_lag: number of lags to compute (k=1..max_lag).

    Returns:
        (max_lag,) tensor, mean lag-k autocorrelation across B and F.
    """
    centered = x - x.mean(dim=1, keepdim=True)
    var = centered.pow(2).mean(dim=1, keepdim=True) + 1e-8  # (B, 1, F)
    acfs = []
    for k in range(1, max_lag + 1):
        num = (centered[:, k:] * centered[:, :-k]).mean(dim=1)   # (B, F)
        acfs.append((num / var.squeeze(1)).mean())
    return torch.stack(acfs)


def _batch_corr(x: torch.Tensor) -> torch.Tensor:
    """
    Differentiable cross-feature correlation matrix of a batch of windows.
    x: (B, T, F)  ->  (F, F) correlation over all (B*T) rows.
    """
    flat = x.reshape(-1, x.shape[-1])
    centered = flat - flat.mean(dim=0, keepdim=True)
    std = centered.std(dim=0, unbiased=False) + 1e-6
    normed = centered / std
    n = normed.shape[0]
    return (normed.t() @ normed) / max(n, 1)


def _cyclical_beta(step: int, total_steps: int, beta_max: float,
                   n_cycles: int = 4, ramp_ratio: float = 0.5) -> float:
    """
    Cyclical KL annealing (Fu et al. 2019). Each cycle ramps beta from 0
    to beta_max over the first ramp_ratio of the cycle, then holds.
    """
    if total_steps <= 0:
        return beta_max
    period = max(total_steps / max(n_cycles, 1), 1.0)
    pos = (step % period) / period             # in [0, 1)
    if pos < ramp_ratio:
        return beta_max * (pos / ramp_ratio)
    return beta_max


def _linear_schedule(
    step: int,
    total_steps: int,
    start: float,
    end: float,
    decay_ratio: float = 1.0,
) -> float:
    """Linear schedule from start to end over (decay_ratio * total_steps)."""
    if total_steps <= 0:
        return end
    decay_steps = max(int(total_steps * max(min(decay_ratio, 1.0), 1e-6)), 1)
    alpha = min(max(step / decay_steps, 0.0), 1.0)
    return (1.0 - alpha) * start + alpha * end


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------


class FinancialVAE(BaseGenerativeModel):
    """VAE for generating financial return sequences."""

    def __init__(
        self,
        n_features: int = 18,
        seq_len: int = 60,
        hidden_dim: int = 128,
        latent_dim: int = 32,
        n_layers: int = 2,
        dropout: float = 0.1,
        cond_dim: int = 0,
        factor_dim: int = 2,
        device: str = "cpu",
    ):
        super().__init__(name="VAE", device=device)
        self.n_features = n_features
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.cond_dim = max(int(cond_dim), 0)
        self.factor_dim = factor_dim

        self.encoder = Encoder(n_features, hidden_dim, latent_dim, n_layers, dropout).to(self.device)
        self.decoder = AutoregressiveDecoder(
            latent_dim, hidden_dim, n_features, seq_len, n_layers, dropout,
            cond_dim=self.cond_dim,
            factor_dim=factor_dim,
        ).to(self.device)

        # Aggregate-posterior cache (populated at end of training).
        self._agg_mu: Optional[torch.Tensor] = None
        self._agg_logvar: Optional[torch.Tensor] = None
        self._agg_cond: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(
        self,
        x: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        teacher_forcing_ratio: float = 1.0,
    ):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        cond_mean, sigma_x, _loading = self.decoder.forward_teacher(
            z, x, cond=cond, teacher_forcing_ratio=teacher_forcing_ratio
        )
        return cond_mean, sigma_x, mu, logvar

    def _loss(
        self,
        x: torch.Tensor,
        mu_x: torch.Tensor,
        sigma_x: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        beta: float,
        free_bits: float,
        acf_ref: Optional[torch.Tensor],
        acf_abs_ref: Optional[torch.Tensor],
        corr_ref: Optional[torch.Tensor],
        acf_weight: float,
        acf_abs_weight: float,
        corr_weight: float,
        acf_max_lag: int,
    ):
        # --- Reconstruction: Student-t NLL ---
        df = self.decoder.df
        dist = StudentT(df=df, loc=mu_x, scale=sigma_x)
        recon_loss = -dist.log_prob(x).mean()

        # --- KL with free bits (per latent dim) ---
        kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())     # (B, D)
        kl_mean_per_dim = kl_per_dim.mean(dim=0)                         # (D,)
        kl_free = torch.clamp(kl_mean_per_dim, min=free_bits).sum()
        kl_raw = kl_mean_per_dim.sum()

        loss = recon_loss + beta * kl_free

        # --- Moment-matching regularisers on a differentiable sample ---
        # rsample() is reparameterised and carries gradients through mu_x / sigma_x.
        acf_r_loss = torch.zeros((), device=x.device)
        acf_abs_loss = torch.zeros((), device=x.device)
        corr_loss = torch.zeros((), device=x.device)

        need_sample = (
            (acf_weight > 0 and acf_ref is not None)
            or (acf_abs_weight > 0 and acf_abs_ref is not None)
            or (corr_weight > 0 and corr_ref is not None)
        )
        if need_sample:
            x_samp = dist.rsample()

            if acf_weight > 0 and acf_ref is not None:
                acf_s = _batch_acf(x_samp, acf_max_lag)
                acf_r_loss = F.mse_loss(acf_s, acf_ref.to(x.device))
                loss = loss + acf_weight * acf_r_loss

            if acf_abs_weight > 0 and acf_abs_ref is not None:
                acf_abs_s = _batch_acf(x_samp.abs(), acf_max_lag)
                acf_abs_loss = F.mse_loss(acf_abs_s, acf_abs_ref.to(x.device))
                loss = loss + acf_abs_weight * acf_abs_loss

            if corr_weight > 0 and corr_ref is not None:
                corr_s = _batch_corr(x_samp)
                corr_loss = F.mse_loss(corr_s, corr_ref.to(x.device))
                loss = loss + corr_weight * corr_loss

        stats = {
            "recon": recon_loss.detach().item(),
            "kl": kl_raw.detach().item(),
            "df": float(df.detach().item()),
            "acf_r": float(acf_r_loss.detach().item()),
            "acf_abs": float(acf_abs_loss.detach().item()),
            "corr": float(corr_loss.detach().item()),
        }
        return loss, stats

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        data: np.ndarray,
        epochs: int = 200,
        batch_size: int = 64,
        lr: float = 1e-3,
        beta_max: float = 0.5,
        free_bits: float = 0.05,
        n_kl_cycles: int = 4,
        teacher_forcing_start: float = 1.0,
        teacher_forcing_end: float = 0.3,
        teacher_forcing_decay_ratio: float = 0.7,
        acf_weight: float = 1.0,
        acf_abs_weight: float = 2.0,
        corr_weight: float = 3.0,
        acf_max_lag: int = 10,
        grad_clip: float = 1.0,
        weight_decay: float = 1e-5,
        verbose: bool = True,
        **kwargs,
    ) -> dict:
        x_tensor = torch.tensor(data, dtype=torch.float32)
        if x_tensor.ndim == 2:
            x_tensor = x_tensor.unsqueeze(-1)
        assert x_tensor.shape[1] == self.seq_len, (
            f"data seq_len {x_tensor.shape[1]} does not match model seq_len {self.seq_len}"
        )
        cond_np = kwargs.get("cond", None)
        cond_tensor = None
        if self.cond_dim > 0:
            if cond_np is None:
                cond_tensor = torch.zeros((x_tensor.shape[0], self.cond_dim), dtype=torch.float32)
            else:
                cond_tensor = torch.tensor(cond_np, dtype=torch.float32)
                if cond_tensor.ndim == 1:
                    cond_tensor = cond_tensor.unsqueeze(1)
                if cond_tensor.shape[0] != x_tensor.shape[0]:
                    raise ValueError(
                        f"cond rows ({cond_tensor.shape[0]}) must match data windows ({x_tensor.shape[0]})"
                    )
                if cond_tensor.shape[1] != self.cond_dim:
                    raise ValueError(
                        f"cond dim ({cond_tensor.shape[1]}) must match model cond_dim ({self.cond_dim})"
                    )

        if cond_tensor is not None:
            dataset = TensorDataset(x_tensor, cond_tensor)
        else:
            dataset = TensorDataset(x_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        # Reference ACF on real training data (compute once).
        with torch.no_grad():
            x_full = x_tensor.to(self.device)
            acf_ref = _batch_acf(x_full, acf_max_lag).detach()
            acf_abs_ref = _batch_acf(x_full.abs(), acf_max_lag).detach()
            corr_ref = _batch_corr(x_full).detach()

        all_params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        self.encoder.train(True)
        self.decoder.train(True)
        optimizer = torch.optim.Adam(all_params, lr=lr, weight_decay=weight_decay)

        total_steps = max(epochs * max(len(loader), 1), 1)
        losses = {"total": [], "recon": [], "kl": [], "acf_r": [], "acf_abs": [],
                  "corr": [], "df": [], "beta": [], "tf_ratio": []}

        global_step = 0
        for epoch in range(epochs):
            tot = rec = klv = ar = aa = co = dfv = 0.0
            beta = 0.0
            tf_ratio = 1.0
            for packed in loader:
                if cond_tensor is not None:
                    batch, cond_batch = packed
                    cond_batch = cond_batch.to(self.device)
                else:
                    (batch,) = packed
                    cond_batch = None
                batch = batch.to(self.device)
                tf_ratio = _linear_schedule(
                    global_step,
                    total_steps,
                    start=teacher_forcing_start,
                    end=teacher_forcing_end,
                    decay_ratio=teacher_forcing_decay_ratio,
                )
                mu_x, sigma_x, mu, logvar = self.forward(
                    batch, cond=cond_batch, teacher_forcing_ratio=tf_ratio
                )

                beta = _cyclical_beta(global_step, total_steps, beta_max, n_kl_cycles)
                loss, stats = self._loss(
                    batch, mu_x, sigma_x, mu, logvar,
                    beta=beta, free_bits=free_bits,
                    acf_ref=acf_ref, acf_abs_ref=acf_abs_ref, corr_ref=corr_ref,
                    acf_weight=acf_weight, acf_abs_weight=acf_abs_weight,
                    corr_weight=corr_weight, acf_max_lag=acf_max_lag,
                )

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(all_params, grad_clip)
                optimizer.step()

                tot += loss.item()
                rec += stats["recon"]
                klv += stats["kl"]
                ar += stats["acf_r"]
                aa += stats["acf_abs"]
                co += stats["corr"]
                dfv += stats["df"]
                global_step += 1

            n = max(len(loader), 1)
            losses["total"].append(tot / n)
            losses["recon"].append(rec / n)
            losses["kl"].append(klv / n)
            losses["acf_r"].append(ar / n)
            losses["acf_abs"].append(aa / n)
            losses["corr"].append(co / n)
            losses["df"].append(dfv / n)
            losses["beta"].append(beta)
            losses["tf_ratio"].append(tf_ratio)

            if verbose and ((epoch + 1) % 20 == 0 or epoch == 0):
                print(
                    f"  Epoch {epoch+1}/{epochs} | loss={losses['total'][-1]:.4f} "
                    f"recon={losses['recon'][-1]:.4f} kl={losses['kl'][-1]:.4f} "
                    f"acf_r={losses['acf_r'][-1]:.4f} acf_abs={losses['acf_abs'][-1]:.4f} "
                    f"corr={losses['corr'][-1]:.4f} "
                    f"df={losses['df'][-1]:.2f} beta={beta:.3f} tf={tf_ratio:.2f}"
                )

        # --- Cache aggregate posterior for generation ---
        self._cache_aggregate_posterior(
            x_tensor,
            cond_tensor=cond_tensor,
            batch_size=max(batch_size, 256),
        )
        self.is_trained = True
        return losses

    @torch.no_grad()
    def _cache_aggregate_posterior(
        self,
        x_tensor: torch.Tensor,
        cond_tensor: Optional[torch.Tensor] = None,
        batch_size: int = 256,
    ):
        self.encoder.eval()
        if cond_tensor is not None:
            loader = DataLoader(TensorDataset(x_tensor, cond_tensor), batch_size=batch_size, shuffle=False)
        else:
            loader = DataLoader(TensorDataset(x_tensor), batch_size=batch_size, shuffle=False)
        mus, logvars = [], []
        conds = [] if cond_tensor is not None else None
        for packed in loader:
            if cond_tensor is not None:
                b, c = packed
                conds.append(c.cpu())
            else:
                (b,) = packed
            mu, logvar = self.encoder(b.to(self.device))
            mus.append(mu.cpu())
            logvars.append(logvar.cpu())
        self._agg_mu = torch.cat(mus, dim=0)
        self._agg_logvar = torch.cat(logvars, dim=0)
        self._agg_cond = torch.cat(conds, dim=0) if conds is not None else None

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        n_samples: int,
        seq_len: int | None = None,
        use_aggregate_posterior: bool = True,
        cond: Optional[np.ndarray] = None,
        regime: Optional[str] = None,
        deterministic: bool = False,
        **kwargs,
    ) -> np.ndarray:
        self.decoder.eval()
        if seq_len is not None and seq_len != self.seq_len:
            # Decoder rolls for self.seq_len steps; temporarily override.
            original_len = self.decoder.seq_len
            self.decoder.seq_len = seq_len
        else:
            original_len = None

        idx = None
        if use_aggregate_posterior and self._agg_mu is not None:
            idx = torch.randint(0, self._agg_mu.size(0), (n_samples,))
            mu = self._agg_mu[idx].to(self.device)
            std = torch.exp(0.5 * self._agg_logvar[idx]).to(self.device)
            z = mu + std * torch.randn_like(std)
        else:
            z = torch.randn(n_samples, self.latent_dim, device=self.device)

        cond_t = None
        if self.cond_dim > 0:
            if regime is not None:
                from src.data.regime_labels import get_regime_conditioning_vectors

                vecs = get_regime_conditioning_vectors()
                if regime not in vecs:
                    raise ValueError(f"Unknown regime '{regime}'. Use one of {list(vecs.keys())}")
                cond_arr = np.repeat(vecs[regime][None, :], n_samples, axis=0)
                cond_t = torch.tensor(cond_arr, dtype=torch.float32, device=self.device)
            elif cond is not None:
                cond_arr = np.asarray(cond, dtype=np.float32)
                if cond_arr.ndim == 1:
                    cond_arr = np.repeat(cond_arr[None, :], n_samples, axis=0)
                if cond_arr.shape != (n_samples, self.cond_dim):
                    raise ValueError(
                        f"cond must have shape ({n_samples}, {self.cond_dim}), got {cond_arr.shape}"
                    )
                cond_t = torch.tensor(cond_arr, dtype=torch.float32, device=self.device)
            elif self._agg_cond is not None:
                if idx is None:
                    idx = torch.randint(0, self._agg_cond.size(0), (n_samples,))
                cond_t = self._agg_cond[idx].to(self.device)
            else:
                cond_t = torch.zeros((n_samples, self.cond_dim), dtype=torch.float32, device=self.device)

        x_gen = self.decoder.sample(z, cond=cond_t, deterministic=deterministic)

        if original_len is not None:
            self.decoder.seq_len = original_len
        return x_gen.cpu().numpy()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            "encoder": self.encoder.state_dict(),
            "decoder": self.decoder.state_dict(),
            "agg_mu": self._agg_mu,
            "agg_logvar": self._agg_logvar,
            "agg_cond": self._agg_cond,
            "config": {
                "n_features": self.n_features,
                "seq_len": self.seq_len,
                "hidden_dim": self.hidden_dim,
                "latent_dim": self.latent_dim,
                "n_layers": self.n_layers,
                "dropout": self.dropout,
                "cond_dim": self.cond_dim,
                "factor_dim": self.factor_dim,
            },
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        cfg = ckpt.get("config", {})

        ckpt_hidden = int(cfg.get("hidden_dim", self.hidden_dim))
        ckpt_latent = int(cfg.get("latent_dim", self.latent_dim))
        ckpt_layers = int(cfg.get("n_layers", self.n_layers))
        ckpt_dropout = float(cfg.get("dropout", self.dropout))
        ckpt_cond = int(cfg.get("cond_dim", self.cond_dim))
        ckpt_factor = int(cfg.get("factor_dim", self.factor_dim))

        need_rebuild = (
            ckpt_hidden != self.hidden_dim
            or ckpt_latent != self.latent_dim
            or ckpt_layers != self.n_layers
            or abs(ckpt_dropout - self.dropout) > 1e-12
            or ckpt_cond != self.cond_dim
            or ckpt_factor != self.factor_dim
        )
        if need_rebuild:
            self.hidden_dim = ckpt_hidden
            self.latent_dim = ckpt_latent
            self.n_layers = ckpt_layers
            self.dropout = ckpt_dropout
            self.cond_dim = ckpt_cond
            self.factor_dim = ckpt_factor
            self.encoder = Encoder(
                self.n_features,
                self.hidden_dim,
                self.latent_dim,
                self.n_layers,
                dropout=self.dropout,
            ).to(self.device)
            self.decoder = AutoregressiveDecoder(
                self.latent_dim,
                self.hidden_dim,
                self.n_features,
                self.seq_len,
                self.n_layers,
                dropout=self.dropout,
                cond_dim=self.cond_dim,
                factor_dim=self.factor_dim,
            ).to(self.device)

        self.encoder.load_state_dict(ckpt["encoder"])
        self.decoder.load_state_dict(ckpt["decoder"])
        self._agg_mu = ckpt.get("agg_mu", None)
        self._agg_logvar = ckpt.get("agg_logvar", None)
        self._agg_cond = ckpt.get("agg_cond", None)
        self.is_trained = True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    from src.utils.config import DEFAULT_DEVICE
    parser = argparse.ArgumentParser(description="VAE for financial time series")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--checkpoint", default="checkpoints/vae.pt")
    args = parser.parse_args()

    data_dir = args.data_dir or os.path.join(os.path.dirname(__file__), "..", "..", "data")

    if args.train:
        windows = np.load(os.path.join(data_dir, "windows.npy"))
        print(f"Training VAE on {windows.shape}")
        model = FinancialVAE(
            n_features=windows.shape[2], seq_len=windows.shape[1],
            device=DEFAULT_DEVICE,
        )
        model.train(windows, epochs=args.epochs)
        model.save(args.checkpoint)

    if args.generate:
        windows = np.load(os.path.join(data_dir, "windows.npy"))
        model = FinancialVAE(
            n_features=windows.shape[2], seq_len=windows.shape[1],
            device=DEFAULT_DEVICE,
        )
        model.load(args.checkpoint)
        synthetic = model.generate(args.n_samples)
        np.save(os.path.join(data_dir, "generated_vae.npy"), synthetic)
        print(f"Generated {synthetic.shape}")

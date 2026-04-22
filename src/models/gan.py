"""
TimeGAN for financial time series generation.

Architecture (Yoon et al., NeurIPS 2019):
  Embedder  : X -> H
  Recovery  : H -> X
  Generator : Z -> E_hat
  Supervisor: H -> H
  Discriminator: H -> logit

Loss: BCE-with-logits (original TimeGAN formulation) augmented with
stylized-fact-aware auxiliary losses that target the six evaluation
tests directly:
  L_acf   : ACF of |r| at lags 1..20 (volatility clustering, long memory)
  L_lev   : corr(r_t, |r_{t+1}|)      (leverage effect)
  L_tail  : kurtosis + |r| 90% quantile (fat tails)
  L_corr  : cross-asset correlation matrix (correlation structure)

No WGAN-GP -> no cuDNN-disable penalty -> significantly faster on CUDA.
"""

from __future__ import annotations

import os
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.models.base_model import BaseGenerativeModel


class RNNBlock(nn.Module):
    """GRU-based sequence model shared across all TimeGAN components."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 n_layers: int = 2):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.gru(x)
        return self.fc(h)


# ---------------------------------------------------------------------------
# Stylized-fact-aware auxiliary statistics (all differentiable, batch-level)
# ---------------------------------------------------------------------------

def _flat_time(x: torch.Tensor) -> torch.Tensor:
    """(B, T, D) -> (B*T, D)."""
    return x.reshape(-1, x.shape[-1])


def _pool(x: torch.Tensor) -> torch.Tensor:
    """(B, T, D) -> (B*T,) via mean across assets — matches evaluator's _to_1d."""
    return x.mean(dim=-1).reshape(-1)


def _acf_abs_pooled(x: torch.Tensor, max_lag: int = 40) -> torch.Tensor:
    """ACF of |pooled series| at lags 1..max_lag. Returns (max_lag,)."""
    s = _pool(x).abs()
    s = s - s.mean()
    var = (s ** 2).mean() + 1e-8
    return torch.stack([(s[lag:] * s[:-lag]).mean() / var
                        for lag in range(1, max_lag + 1)])


def _acf_raw_pooled(x: torch.Tensor, max_lag: int = 20) -> torch.Tensor:
    """ACF of raw pooled returns (we want this ≈ 0 — targets test 6)."""
    s = _pool(x)
    s = s - s.mean()
    var = (s ** 2).mean() + 1e-8
    return torch.stack([(s[lag:] * s[:-lag]).mean() / var
                        for lag in range(1, max_lag + 1)])


def _leverage_pooled(x: torch.Tensor) -> torch.Tensor:
    """corr(r_t, |r_{t+1}|) on the pooled series — matches test 3's input."""
    s = _pool(x)
    r_t = s[:-1]
    abs_r_next = s[1:].abs()
    r_t = r_t - r_t.mean()
    abs_r_next = abs_r_next - abs_r_next.mean()
    return (r_t * abs_r_next).mean() / (r_t.std() * abs_r_next.std() + 1e-8)


def _kurtosis_pooled(x: torch.Tensor) -> torch.Tensor:
    """Excess kurtosis on the pooled series."""
    s = _pool(x)
    m = s.mean()
    sd = s.std() + 1e-8
    z = (s - m) / sd
    return (z ** 4).mean() - 3.0


def _q90_abs_pooled(x: torch.Tensor) -> torch.Tensor:
    """90% quantile of |pooled|."""
    return torch.quantile(_pool(x).abs(), 0.9)


def _corr_matrix(x: torch.Tensor) -> torch.Tensor:
    """Cross-asset correlation matrix (D, D)."""
    flat = _flat_time(x)
    flat = flat - flat.mean(dim=0, keepdim=True)
    s = flat.std(dim=0) + 1e-8
    flat_n = flat / s
    return (flat_n.T @ flat_n) / flat_n.shape[0]


class TimeGANModel(BaseGenerativeModel):
    """TimeGAN for multi-asset financial time series (BCE + stylized-fact losses)."""

    def __init__(
        self,
        n_features: int = 18,
        seq_len: int = 60,
        hidden_dim: int = 64,
        latent_dim: int = 32,
        n_layers: int = 2,
        device: str = "cpu",
    ):
        super().__init__(name="TimeGAN", device=device)
        self.n_features = n_features
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.embedder      = RNNBlock(n_features, hidden_dim, latent_dim, n_layers).to(self.device)
        self.recovery      = RNNBlock(latent_dim, hidden_dim, n_features, n_layers).to(self.device)
        self.generator     = RNNBlock(latent_dim, hidden_dim, latent_dim, n_layers).to(self.device)
        self.supervisor    = RNNBlock(latent_dim, hidden_dim, latent_dim, n_layers).to(self.device)
        self.discriminator = RNNBlock(latent_dim, hidden_dim, 1,          n_layers).to(self.device)

    def _noise(self, batch_size: int, seq_len: int) -> torch.Tensor:
        return torch.randn(batch_size, seq_len, self.latent_dim, device=self.device)

    def train(
        self,
        data: np.ndarray,
        epochs: int = 200,
        batch_size: int = 64,
        lr: float = 1e-4,
        gamma: float = 1.0,
        ae_ratio: float = 0.3,
        sup_ratio: float = 0.2,
        n_gen_steps: int = 2,
        w_acf: float = 4.0,
        w_lev: float = 0.3,
        w_tail: float = 1.0,
        w_corr: float = 3.0,
        w_acf_raw: float = 2.0,
        d_skip_hi: float = 0.9,
        d_skip_lo: float = 0.1,
        d_warmup_epochs: int = 10,
        ckpt_path: str | None = None,
        ckpt_every: int = 40,
        **kwargs,
    ) -> dict:
        """
        Three-phase TimeGAN training with BCE adversarial loss and
        stylized-fact-aware auxiliary losses.

        Args:
            ae_ratio / sup_ratio: fraction of epochs for AE and supervisor pretraining.
            gamma:        weight on E_hat adversarial branch.
            n_gen_steps:  generator updates per outer iteration.
            w_acf/lev/tail/corr: weights for the four stylized-fact losses.
            d_skip_hi/lo: skip D update when sigmoid(D(real)).mean() > d_skip_hi
                          AND sigmoid(D(fake)).mean() < d_skip_lo (D already winning).
        """
        x_tensor = torch.tensor(data, dtype=torch.float32)
        if x_tensor.ndim == 2:
            x_tensor = x_tensor.unsqueeze(-1)
        loader = DataLoader(TensorDataset(x_tensor),
                            batch_size=batch_size, shuffle=True, drop_last=True)

        mse = nn.MSELoss()
        bce = nn.BCEWithLogitsLoss()

        opt_ae  = torch.optim.Adam(
            list(self.embedder.parameters()) + list(self.recovery.parameters()), lr=lr)
        opt_sup = torch.optim.Adam(self.supervisor.parameters(), lr=lr)
        # G gets higher LR than D (2:1)
        opt_g   = torch.optim.Adam(
            list(self.generator.parameters()) + list(self.supervisor.parameters()),
            lr=lr, betas=(0.5, 0.9))
        opt_d   = torch.optim.Adam(self.discriminator.parameters(),
                                   lr=lr * 0.5, betas=(0.5, 0.9))

        losses = {"ae": [], "sup": [], "g": [], "d": []}

        ae_epochs    = max(1, int(epochs * ae_ratio))
        sup_epochs   = max(1, int(epochs * sup_ratio))
        joint_epochs = max(1, epochs - ae_epochs - sup_epochs)

        # ------------------------------------------------------------------
        # Phase 1: Autoencoder pre-training
        # ------------------------------------------------------------------
        print(f"  Phase 1: Autoencoder pre-training ({ae_epochs} epochs)")
        for _ in range(ae_epochs):
            epoch_loss = 0.0
            for (batch,) in loader:
                batch = batch.to(self.device)
                h     = self.embedder(batch)
                x_hat = self.recovery(h)
                loss  = 10 * torch.sqrt(mse(x_hat, batch) + 1e-8)
                opt_ae.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.embedder.parameters()) + list(self.recovery.parameters()), 1.0)
                opt_ae.step()
                epoch_loss += loss.item()
            losses["ae"].append(epoch_loss / len(loader))

        # ------------------------------------------------------------------
        # Phase 2: Supervisor pre-training
        # ------------------------------------------------------------------
        print(f"  Phase 2: Supervisor pre-training ({sup_epochs} epochs)")
        for _ in range(sup_epochs):
            epoch_loss = 0.0
            for (batch,) in loader:
                batch = batch.to(self.device)
                with torch.no_grad():
                    h = self.embedder(batch)
                h_sup = self.supervisor(h[:, :-1, :])
                loss  = mse(h_sup, h[:, 1:, :])
                opt_sup.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.supervisor.parameters(), 1.0)
                opt_sup.step()
                epoch_loss += loss.item()
            losses["sup"].append(epoch_loss / len(loader))

        # ------------------------------------------------------------------
        # Phase 3: Joint adversarial training
        # ------------------------------------------------------------------
        print(f"  Phase 3: Joint adversarial training ({joint_epochs} epochs)")
        sched_g = torch.optim.lr_scheduler.CosineAnnealingLR(opt_g, joint_epochs, eta_min=lr * 0.1)
        sched_d = torch.optim.lr_scheduler.CosineAnnealingLR(opt_d, joint_epochs, eta_min=lr * 0.05)

        for epoch in range(joint_epochs):
            g_loss_total, d_loss_total = 0.0, 0.0
            d_skipped = 0

            for (batch,) in loader:
                batch = batch.to(self.device)
                B     = batch.shape[0]

                # ---- (a) Generator + Supervisor update -------------------
                g_loss_val = 0.0
                for _ in range(n_gen_steps):
                    noise  = self._noise(B, self.seq_len)
                    e_hat  = self.generator(noise)
                    h_hat  = self.supervisor(e_hat)
                    x_hat  = self.recovery(h_hat)

                    # BCE adversarial: want D(fake)=1
                    d_h = self.discriminator(h_hat)
                    d_e = self.discriminator(e_hat)
                    g_adv_h = bce(d_h, torch.ones_like(d_h))
                    g_adv_e = bce(d_e, torch.ones_like(d_e))

                    with torch.no_grad():
                        h_real = self.embedder(batch)
                    g_sup_loss = mse(
                        self.supervisor(h_real[:, :-1, :]),
                        h_real[:, 1:, :],
                    )

                    g_moment = (
                        torch.abs(x_hat.mean(dim=0) - batch.mean(dim=0)).mean() +
                        torch.abs(x_hat.std(dim=0)  - batch.std(dim=0)).mean()
                    )

                    # --- stylized-fact auxiliary losses ---
                    with torch.no_grad():
                        acf_real  = _acf_abs_pooled(batch)
                        lev_real  = _leverage_pooled(batch)
                        kurt_real = _kurtosis_pooled(batch)
                        q90_real  = _q90_abs_pooled(batch)
                        corr_real = _corr_matrix(batch)

                    L_acf     = mse(_acf_abs_pooled(x_hat),  acf_real)
                    L_lev     = (_leverage_pooled(x_hat) - lev_real) ** 2
                    L_tail    = ((_kurtosis_pooled(x_hat) - kurt_real) ** 2 +
                                 (_q90_abs_pooled(x_hat)  - q90_real)  ** 2)
                    L_corr    = mse(_corr_matrix(x_hat), corr_real)
                    # Push raw-return ACF toward zero (test 6)
                    L_acf_raw = (_acf_raw_pooled(x_hat) ** 2).mean()

                    g_loss = (
                        g_adv_h
                        + gamma * g_adv_e
                        + 10.0 * torch.sqrt(g_sup_loss + 1e-8)
                        + 10.0 * torch.sqrt(g_moment   + 1e-8)
                        + w_acf      * L_acf
                        + w_lev      * L_lev
                        + w_tail     * L_tail
                        + w_corr     * L_corr
                        + w_acf_raw  * L_acf_raw
                    )
                    opt_g.zero_grad()
                    g_loss.backward()
                    nn.utils.clip_grad_norm_(
                        list(self.generator.parameters()) + list(self.supervisor.parameters()), 1.0)
                    opt_g.step()
                    g_loss_val = g_loss.item()

                # ---- (b) Discriminator update ---------------------------
                with torch.no_grad():
                    h_real_d = self.embedder(batch)
                    e_hat_d  = self.generator(self._noise(B, self.seq_len))
                    h_hat_d  = self.supervisor(e_hat_d)

                # Probe: is D already winning?
                with torch.no_grad():
                    p_real = torch.sigmoid(self.discriminator(h_real_d)).mean().item()
                    p_fake = torch.sigmoid(self.discriminator(h_hat_d)).mean().item()

                in_warmup = epoch < d_warmup_epochs
                if (not in_warmup) and p_real > d_skip_hi and p_fake < d_skip_lo:
                    d_skipped += 1
                    d_loss_val = 0.0
                else:
                    logit_real = self.discriminator(h_real_d)
                    logit_h    = self.discriminator(h_hat_d)
                    logit_e    = self.discriminator(e_hat_d)
                    d_loss = (
                        bce(logit_real, torch.ones_like(logit_real))
                        + bce(logit_h,  torch.zeros_like(logit_h))
                        + gamma * bce(logit_e, torch.zeros_like(logit_e))
                    )
                    opt_d.zero_grad()
                    d_loss.backward()
                    nn.utils.clip_grad_norm_(self.discriminator.parameters(), 1.0)
                    opt_d.step()
                    d_loss_val = d_loss.item()

                # ---- (c) Embedder + Recovery fine-tune ------------------
                h_e     = self.embedder(batch)
                x_tilde = self.recovery(h_e)
                h_sup_e = self.supervisor(h_e[:, :-1, :])

                e_recon = mse(x_tilde, batch)
                e_sup   = mse(h_sup_e, h_e[:, 1:, :].detach())
                e_loss  = torch.sqrt(e_recon + 1e-8) + 0.1 * e_sup

                opt_ae.zero_grad()
                e_loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.embedder.parameters()) + list(self.recovery.parameters()), 1.0)
                opt_ae.step()

                g_loss_total += g_loss_val
                d_loss_total += d_loss_val

            sched_g.step()
            sched_d.step()
            losses["g"].append(g_loss_total / len(loader))
            losses["d"].append(d_loss_total / len(loader))
            if (epoch + 1) % 20 == 0:
                print(f"  Joint {epoch+1}/{joint_epochs} | "
                      f"G={losses['g'][-1]:.4f}  D={losses['d'][-1]:.4f}  "
                      f"D_skip={d_skipped}/{len(loader)}")
            if ckpt_path and (epoch + 1) % ckpt_every == 0:
                self.is_trained = True
                self.save(ckpt_path)

        self.is_trained = True
        return losses

    @torch.no_grad()
    def generate(self, n_samples: int, seq_len: int | None = None, **kwargs) -> np.ndarray:
        """Generate n_samples windows. To mirror the evaluator's real windows
        (which are built with stride=1 from one long series), we unroll the GRU
        over a single long sequence of length (n_samples + seq_len - 1) and
        slide stride-1 windows, so synthetic and real share the same overlap
        structure (critical for Hurst / GARCH persistence tests)."""
        if seq_len is None:
            seq_len = self.seq_len
        self.generator.eval()
        self.supervisor.eval()
        self.recovery.eval()

        total_len = n_samples + seq_len - 1
        # Chunk the long rollout to avoid blowing memory on very large n_samples
        chunk = max(total_len, 4096)
        noise = torch.randn(1, total_len, self.latent_dim, device=self.device)
        e_hat = self.generator(noise)
        h_hat = self.supervisor(e_hat)
        x_long = self.recovery(h_hat).squeeze(0).cpu().numpy()   # (total_len, D)

        # Stride-1 sliding windows
        windows = np.lib.stride_tricks.sliding_window_view(
            x_long, window_shape=seq_len, axis=0
        )  # (n_samples, D, seq_len)
        windows = windows.transpose(0, 2, 1).astype(np.float32)  # (n_samples, seq_len, D)
        return np.ascontiguousarray(windows[:n_samples])

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            "embedder":      self.embedder.state_dict(),
            "recovery":      self.recovery.state_dict(),
            "generator":     self.generator.state_dict(),
            "supervisor":    self.supervisor.state_dict(),
            "discriminator": self.discriminator.state_dict(),
            "config": {
                "n_features": self.n_features,
                "seq_len":    self.seq_len,
                "hidden_dim": self.hidden_dim,
                "latent_dim": self.latent_dim,
            },
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.embedder.load_state_dict(ckpt["embedder"])
        self.recovery.load_state_dict(ckpt["recovery"])
        self.generator.load_state_dict(ckpt["generator"])
        self.supervisor.load_state_dict(ckpt["supervisor"])
        self.discriminator.load_state_dict(ckpt["discriminator"])
        self.is_trained = True


if __name__ == "__main__":
    from src.utils.config import DEFAULT_DEVICE

    parser = argparse.ArgumentParser(description="TimeGAN for financial time series")
    parser.add_argument("--train",     action="store_true")
    parser.add_argument("--generate",  action="store_true")
    parser.add_argument("--epochs",    type=int, default=200)
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--data-dir",  default=None)
    parser.add_argument("--checkpoint", default="checkpoints/timegan.pt")
    args = parser.parse_args()

    data_dir = args.data_dir or os.path.join(os.path.dirname(__file__), "..", "..", "data")

    if args.train:
        windows = np.load(os.path.join(data_dir, "windows.npy"))
        print(f"Training TimeGAN on {windows.shape}")
        model = TimeGANModel(
            n_features=windows.shape[2],
            seq_len=windows.shape[1],
            device=DEFAULT_DEVICE,
        )
        model.train(windows, epochs=args.epochs)
        model.save(args.checkpoint)

    if args.generate:
        windows = np.load(os.path.join(data_dir, "windows.npy"))
        model = TimeGANModel(
            n_features=windows.shape[2],
            seq_len=windows.shape[1],
            device=DEFAULT_DEVICE,
        )
        model.load(args.checkpoint)
        synthetic = model.generate(args.n_samples)
        np.save(os.path.join(data_dir, "generated_timegan.npy"), synthetic)
        print(f"Generated {synthetic.shape}")

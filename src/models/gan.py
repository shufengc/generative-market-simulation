"""
TimeGAN for financial time series generation.

Architecture (Yoon et al., NeurIPS 2019):
  Embedder  : X -> H  (real space to latent space)
  Recovery  : H -> X  (latent back to real)
  Generator : Z -> E_hat  (noise to raw latent)
  Supervisor: H -> H  (captures temporal dynamics in latent space)
  Discriminator: H -> scalar  (real vs fake latent)

Training phases:
  Phase 1 — Autoencoder pre-training (Embedder + Recovery)
  Phase 2 — Supervisor pre-training  (Supervisor on real latents)
  Phase 3 — Joint adversarial training:
             (a) Generator + Supervisor update
             (b) Discriminator update  (on both H_hat and E_hat)
             (c) Embedder + Recovery update  (reconstruction + supervisor)

Loss: Wasserstein + Gradient Penalty (replaces unstable BCE+GP mix).
"""

from __future__ import annotations

import os
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

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


class TimeGANModel(BaseGenerativeModel):
    """TimeGAN for multi-asset financial time series."""

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

        self.embedder     = RNNBlock(n_features,  hidden_dim, latent_dim, n_layers).to(self.device)
        self.recovery     = RNNBlock(latent_dim,  hidden_dim, n_features, n_layers).to(self.device)
        self.generator    = RNNBlock(latent_dim,  hidden_dim, latent_dim, n_layers).to(self.device)
        self.supervisor   = RNNBlock(latent_dim,  hidden_dim, latent_dim, n_layers).to(self.device)
        self.discriminator = RNNBlock(latent_dim, hidden_dim, 1,          n_layers).to(self.device)

    def _noise(self, batch_size: int, seq_len: int) -> torch.Tensor:
        """Gaussian noise in latent space dimension."""
        return torch.randn(batch_size, seq_len, self.latent_dim, device=self.device)

    def _gradient_penalty(self, real_h: torch.Tensor, fake_h: torch.Tensor) -> torch.Tensor:
        """WGAN-GP gradient penalty between real and fake latent sequences.
        CuDNN is disabled during the forward pass because double-backward
        is not supported for CuDNN RNNs.
        """
        B = real_h.shape[0]
        alpha = torch.rand(B, 1, 1, device=self.device)
        interp = (alpha * real_h + (1 - alpha) * fake_h).requires_grad_(True)
        with torch.backends.cudnn.flags(enabled=False):
            d_interp = self.discriminator(interp)
        grad = torch.autograd.grad(
            outputs=d_interp, inputs=interp,
            grad_outputs=torch.ones_like(d_interp),
            create_graph=True, retain_graph=True,
        )[0]
        grad_norm = grad.reshape(B, -1).norm(2, dim=1)
        return ((grad_norm - 1) ** 2).mean()

    def train(
        self,
        data: np.ndarray,
        epochs: int = 200,
        batch_size: int = 64,
        lr: float = 1e-4,
        gp_weight: float = 10.0,
        gamma: float = 1.0,
        ae_ratio: float = 0.4,
        sup_ratio: float = 0.2,
        **kwargs,
    ) -> dict:
        """
        Three-phase TimeGAN training.

        Args:
            epochs:    Total training epochs.
            ae_ratio:  Fraction of epochs for autoencoder pre-training (default 0.4).
            sup_ratio: Fraction of epochs for supervisor pre-training  (default 0.2).
                       Joint phase gets the remaining 1 - ae_ratio - sup_ratio.
            gamma:     Weight for discriminating raw generator output E_hat.
            gp_weight: Gradient penalty coefficient.
        """
        x_tensor = torch.tensor(data, dtype=torch.float32)
        if x_tensor.ndim == 2:
            x_tensor = x_tensor.unsqueeze(-1)
        dataset = TensorDataset(x_tensor)
        loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        mse = nn.MSELoss()

        opt_ae  = torch.optim.Adam(
            list(self.embedder.parameters()) + list(self.recovery.parameters()), lr=lr)
        opt_sup = torch.optim.Adam(self.supervisor.parameters(), lr=lr)
        opt_g   = torch.optim.Adam(
            list(self.generator.parameters()) + list(self.supervisor.parameters()), lr=lr)
        opt_d   = torch.optim.Adam(self.discriminator.parameters(), lr=lr * 0.5)

        losses = {"ae": [], "sup": [], "g": [], "d": []}

        ae_epochs    = max(1, int(epochs * ae_ratio))
        sup_epochs   = max(1, int(epochs * sup_ratio))
        joint_epochs = max(1, epochs - ae_epochs - sup_epochs)

        # ------------------------------------------------------------------
        # Phase 1: Autoencoder pre-training (Embedder + Recovery)
        # ------------------------------------------------------------------
        print(f"  Phase 1: Autoencoder pre-training ({ae_epochs} epochs)")
        for epoch in range(ae_epochs):
            epoch_loss = 0.0
            for (batch,) in loader:
                batch = batch.to(self.device)
                h      = self.embedder(batch)
                x_hat  = self.recovery(h)
                loss   = 10 * torch.sqrt(mse(x_hat, batch) + 1e-8)
                opt_ae.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.embedder.parameters()) + list(self.recovery.parameters()), 1.0)
                opt_ae.step()
                epoch_loss += loss.item()
            losses["ae"].append(epoch_loss / len(loader))

        # ------------------------------------------------------------------
        # Phase 2: Supervisor pre-training (learns temporal dynamics)
        # ------------------------------------------------------------------
        print(f"  Phase 2: Supervisor pre-training ({sup_epochs} epochs)")
        for epoch in range(sup_epochs):
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

            for (batch,) in loader:
                batch = batch.to(self.device)
                B     = batch.shape[0]

                # ---- (a) Generator + Supervisor update ----------------
                noise  = self._noise(B, self.seq_len)
                e_hat  = self.generator(noise)          # raw latent from noise
                h_hat  = self.supervisor(e_hat)         # supervised latent
                x_hat  = self.recovery(h_hat)

                # Wasserstein adversarial losses (higher = more real)
                g_adv_h = -self.discriminator(h_hat).mean()     # on supervised latent
                g_adv_e = -self.discriminator(e_hat).mean()     # on raw latent

                # Supervisor loss on real latent sequences
                with torch.no_grad():
                    h_real = self.embedder(batch)
                g_sup_loss = mse(
                    self.supervisor(h_real[:, :-1, :]),
                    h_real[:, 1:, :],
                )

                # Moment matching (mean + std per feature per timestep)
                g_moment = (
                    torch.abs(x_hat.mean(dim=0) - batch.mean(dim=0)).mean() +
                    torch.abs(x_hat.std(dim=0)  - batch.std(dim=0)).mean()
                )

                g_loss = (
                    g_adv_h
                    + gamma * g_adv_e
                    + 10.0 * torch.sqrt(g_sup_loss  + 1e-8)
                    + 10.0 * torch.sqrt(g_moment    + 1e-8)
                )
                opt_g.zero_grad()
                g_loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.generator.parameters()) + list(self.supervisor.parameters()), 1.0)
                opt_g.step()

                # ---- (b) Discriminator update --------------------------
                with torch.no_grad():
                    h_real_d = self.embedder(batch)
                    e_hat_d  = self.generator(self._noise(B, self.seq_len))
                    h_hat_d  = self.supervisor(e_hat_d)

                # Wasserstein loss: real high, fake low
                d_real    = self.discriminator(h_real_d).mean()
                d_fake_h  = self.discriminator(h_hat_d).mean()
                d_fake_e  = self.discriminator(e_hat_d).mean()

                gp = self._gradient_penalty(h_real_d, h_hat_d)

                d_loss = (d_fake_h + gamma * d_fake_e) - d_real + gp_weight * gp
                opt_d.zero_grad()
                d_loss.backward()
                nn.utils.clip_grad_norm_(self.discriminator.parameters(), 1.0)
                opt_d.step()

                # ---- (c) Embedder + Recovery update -------------------
                # Jointly fine-tune with reconstruction + supervisor loss.
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

                g_loss_total += g_loss.item()
                d_loss_total += d_loss.item()

            sched_g.step()
            sched_d.step()
            losses["g"].append(g_loss_total / len(loader))
            losses["d"].append(d_loss_total / len(loader))
            if (epoch + 1) % 20 == 0:
                print(f"  Joint {epoch+1}/{joint_epochs} | "
                      f"G={losses['g'][-1]:.4f}  D={losses['d'][-1]:.4f}")

        self.is_trained = True
        return losses

    @torch.no_grad()
    def generate(self, n_samples: int, seq_len: int | None = None, **kwargs) -> np.ndarray:
        if seq_len is None:
            seq_len = self.seq_len
        self.generator.eval()
        self.supervisor.eval()
        self.recovery.eval()

        noise = self._noise(n_samples, seq_len)
        h_hat = self.supervisor(self.generator(noise))
        x_hat = self.recovery(h_hat)
        return x_hat.cpu().numpy()

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
"""
RealNVP-style Normalizing Flow for financial time series generation.

Uses affine coupling layers with batch normalization to learn an invertible
mapping between a simple base distribution (Gaussian) and the data distribution.
Operates on flattened (seq_len * n_features) vectors.
"""

from __future__ import annotations

import os
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.base_model import BaseGenerativeModel


class BatchNormFlow(nn.Module):
    """Batch normalization as a flow layer (invertible)."""

    def __init__(self, dim: int, momentum: float = 0.1, eps: float = 1e-5):
        super().__init__()
        self.dim = dim
        self.momentum = momentum
        self.eps = eps
        self.log_gamma = nn.Parameter(torch.zeros(dim))
        self.beta = nn.Parameter(torch.zeros(dim))
        self.register_buffer("running_mean", torch.zeros(dim))
        self.register_buffer("running_var", torch.ones(dim))

    def forward(self, x: torch.Tensor):
        if self.training:
            mean = x.mean(0)
            var = x.var(0) + self.eps
            self.running_mean.mul_(1 - self.momentum).add_(mean.detach() * self.momentum)
            self.running_var.mul_(1 - self.momentum).add_(var.detach() * self.momentum)
        else:
            mean = self.running_mean
            var = self.running_var + self.eps

        x_hat = (x - mean) / var.sqrt()
        y = x_hat * self.log_gamma.exp() + self.beta
        log_det = (self.log_gamma - 0.5 * var.log()).sum()
        return y, log_det

    def inverse(self, y: torch.Tensor):
        mean = self.running_mean
        var = self.running_var + self.eps
        x_hat = (y - self.beta) / self.log_gamma.exp()
        x = x_hat * var.sqrt() + mean
        return x


class CouplingLayer(nn.Module):
    """Affine coupling layer: splits input, transforms one half conditioned on the other."""

    def __init__(self, dim: int, hidden_dim: int = 256, mask_type: str = "even"):
        super().__init__()
        self.dim = dim
        if mask_type == "even":
            self.mask = torch.arange(dim) % 2 == 0
        else:
            self.mask = torch.arange(dim) % 2 == 1

        self.scale_net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, dim),
            nn.Tanh(),
        )
        self.translate_net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor):
        mask = self.mask.to(x.device).float()
        x_masked = x * mask
        s = self.scale_net(x_masked) * (1 - mask)
        t = self.translate_net(x_masked) * (1 - mask)
        y = x_masked + (1 - mask) * (x * torch.exp(s) + t)
        log_det = s.sum(dim=-1)
        return y, log_det

    def inverse(self, y: torch.Tensor):
        mask = self.mask.to(y.device).float()
        y_masked = y * mask
        s = self.scale_net(y_masked) * (1 - mask)
        t = self.translate_net(y_masked) * (1 - mask)
        x = y_masked + (1 - mask) * (y - t) * torch.exp(-s)
        return x


class RealNVPFlow(nn.Module):
    """Stack of affine coupling layers with batch normalization."""

    def __init__(self, dim: int, hidden_dim: int = 256, n_layers: int = 6,
                 use_batchnorm: bool = True):
        super().__init__()
        self.layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList() if use_batchnorm else None

        for i in range(n_layers):
            mask_type = "even" if i % 2 == 0 else "odd"
            self.layers.append(CouplingLayer(dim, hidden_dim, mask_type))
            if use_batchnorm and i < n_layers - 1:
                self.bn_layers.append(BatchNormFlow(dim))

    def forward(self, x: torch.Tensor):
        log_det_total = torch.zeros(x.shape[0], device=x.device)
        z = x
        for i, layer in enumerate(self.layers):
            z, log_det = layer(z)
            log_det_total += log_det
            if self.bn_layers is not None and i < len(self.bn_layers):
                z, bn_log_det = self.bn_layers[i](z)
                log_det_total += bn_log_det
        return z, log_det_total

    def inverse(self, z: torch.Tensor):
        x = z
        for i in range(len(self.layers) - 1, -1, -1):
            if self.bn_layers is not None and i < len(self.bn_layers):
                x = self.bn_layers[i].inverse(x)
            x = self.layers[i].inverse(x)
        return x


class NormalizingFlowModel(BaseGenerativeModel):
    """RealNVP normalizing flow for financial return windows."""

    def __init__(
        self,
        n_features: int = 18,
        seq_len: int = 60,
        hidden_dim: int = 256,
        n_flow_layers: int = 6,
        device: str = "cpu",
    ):
        super().__init__(name="NormalizingFlow", device=device)
        self.n_features = n_features
        self.seq_len = seq_len
        self.flat_dim = n_features * seq_len

        self.flow = RealNVPFlow(
            self.flat_dim, hidden_dim, n_flow_layers, use_batchnorm=True
        ).to(self.device)

    def train(self, data: np.ndarray, epochs: int = 200, batch_size: int = 64,
              lr: float = 1e-4, **kwargs) -> dict:
        x = torch.tensor(data, dtype=torch.float32)
        if x.ndim == 3:
            x = x.reshape(x.shape[0], -1)

        actual_dim = x.shape[1]
        if actual_dim != self.flat_dim:
            self.flat_dim = actual_dim
            self.n_features = data.shape[2] if data.ndim == 3 else self.n_features
            self.seq_len = actual_dim // self.n_features
            self.flow = RealNVPFlow(actual_dim, 256, 6, use_batchnorm=True).to(self.device)

        dataset = TensorDataset(x)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        optimizer = torch.optim.Adam(self.flow.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)

        losses = []
        self.flow.train()
        for epoch in range(epochs):
            epoch_loss = 0
            for (batch,) in loader:
                batch = batch.to(self.device)
                z, log_det = self.flow(batch)
                log_prob_z = -0.5 * (z ** 2 + np.log(2 * np.pi)).sum(dim=-1)
                loss = -(log_prob_z + log_det).mean()

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.flow.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()

            scheduler.step()
            avg = epoch_loss / max(len(loader), 1)
            losses.append(avg)
            if (epoch + 1) % 20 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1}/{epochs} | NLL={avg:.4f}")

        self.is_trained = True
        return {"losses": losses}

    @torch.no_grad()
    def generate(self, n_samples: int, seq_len: int | None = None, **kwargs) -> np.ndarray:
        self.flow.eval()
        if seq_len is None:
            seq_len = self.seq_len

        z = torch.randn(n_samples, self.flat_dim, device=self.device)
        x = self.flow.inverse(z)
        result = x.cpu().numpy()

        n_feat = self.n_features
        actual_seq = self.flat_dim // n_feat
        return result.reshape(n_samples, actual_seq, n_feat)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            "flow": self.flow.state_dict(),
            "config": {
                "n_features": self.n_features,
                "seq_len": self.seq_len,
                "flat_dim": self.flat_dim,
            },
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        cfg = ckpt["config"]
        self.flat_dim = cfg["flat_dim"]
        self.n_features = cfg.get("n_features", self.n_features)
        self.seq_len = cfg.get("seq_len", self.seq_len)
        self.flow = RealNVPFlow(self.flat_dim, 256, 6, use_batchnorm=True).to(self.device)
        self.flow.load_state_dict(ckpt["flow"])
        self.is_trained = True


if __name__ == "__main__":
    from src.utils.config import DEFAULT_DEVICE
    parser = argparse.ArgumentParser(description="Normalizing Flow for financial time series")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--checkpoint", default="checkpoints/flow.pt")
    args = parser.parse_args()

    data_dir = args.data_dir or os.path.join(os.path.dirname(__file__), "..", "..", "data")

    if args.train:
        windows = np.load(os.path.join(data_dir, "windows.npy"))
        print(f"Training NormalizingFlow on {windows.shape}")
        model = NormalizingFlowModel(
            n_features=windows.shape[2], seq_len=windows.shape[1],
            device=DEFAULT_DEVICE,
        )
        model.train(windows, epochs=args.epochs)
        model.save(args.checkpoint)

    if args.generate:
        windows = np.load(os.path.join(data_dir, "windows.npy"))
        model = NormalizingFlowModel(
            n_features=windows.shape[2], seq_len=windows.shape[1],
            device=DEFAULT_DEVICE,
        )
        model.load(args.checkpoint)
        synthetic = model.generate(args.n_samples)
        np.save(os.path.join(data_dir, "generated_flow.npy"), synthetic)
        print(f"Generated {synthetic.shape}")

"""
Abstract base class for all generative models.

Every model must implement train(), generate(), and save()/load().
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import torch


class BaseGenerativeModel(ABC):
    """Base class that all generative models inherit from."""

    def __init__(self, name: str, device: str = "cpu"):
        self.name = name
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = torch.device(device)
        self.is_trained = False

    @abstractmethod
    def train(self, data: np.ndarray, **kwargs) -> dict:
        """
        Train the model on real data.

        Args:
            data: np.ndarray of shape (N, seq_len, n_features) -- windowed returns.
            **kwargs: Model-specific training arguments.

        Returns:
            dict with training metrics (loss history, etc.)
        """
        ...

    @abstractmethod
    def generate(self, n_samples: int, seq_len: int, **kwargs) -> np.ndarray:
        """
        Generate synthetic return sequences.

        Args:
            n_samples: Number of sequences to generate.
            seq_len: Length of each sequence.
            **kwargs: Conditioning variables, etc.

        Returns:
            np.ndarray of shape (n_samples, seq_len, n_features)
        """
        ...

    @abstractmethod
    def save(self, path: str) -> None:
        """Save model checkpoint."""
        ...

    @abstractmethod
    def load(self, path: str) -> None:
        """Load model checkpoint."""
        ...

    def evaluate(self, real_data: np.ndarray, n_samples: int = 1000) -> dict:
        """Generate synthetic data and run stylized facts tests."""
        from src.evaluation.stylized_facts import run_all_tests
        from src.evaluation.metrics import maximum_mean_discrepancy, moment_comparison

        seq_len = real_data.shape[1] if real_data.ndim == 3 else 252
        n_features = real_data.shape[2] if real_data.ndim == 3 else 1

        synthetic = self.generate(n_samples, seq_len)

        if synthetic.ndim == 3:
            syn_flat = synthetic.reshape(-1, synthetic.shape[-1])
        else:
            syn_flat = synthetic.flatten()

        if real_data.ndim == 3:
            real_flat = real_data.reshape(-1, real_data.shape[-1])
        else:
            real_flat = real_data.flatten()

        stylized = run_all_tests(syn_flat)

        mmd = maximum_mean_discrepancy(
            real_flat[:2000].flatten().reshape(-1, 1),
            syn_flat[:2000].flatten().reshape(-1, 1),
        )
        moments = moment_comparison(real_flat.flatten(), syn_flat.flatten())

        return {
            "model": self.name,
            "stylized_facts": stylized,
            "mmd": mmd,
            "moments": moments,
        }

    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}', device='{self.device}')"

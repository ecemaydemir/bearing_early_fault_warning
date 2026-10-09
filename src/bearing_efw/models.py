"""Anomaly detectors. All are fitted on healthy data only (no failure labels).

Every detector maps a bearing's standardized feature matrix ``X`` (n_snapshots x
n_features, time-ordered) to one anomaly score per snapshot; higher = less healthy.
Detectors are fitted once per test on the pooled healthy period of all bearings,
because each bearing's features are already standardized against its own baseline.
"""
from __future__ import annotations

import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import IsolationForest


class RMSThreshold:
    """Classic condition-monitoring baseline: overall vibration level (z-score of log RMS)."""

    name = "RMS threshold"

    def __init__(self, columns: list[str]):
        self.idx = [i for i, c in enumerate(columns) if c.startswith("rms_")]

    def fit(self, Xs: list[np.ndarray]) -> "RMSThreshold":
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        return X[:, self.idx].max(axis=1)


class Mahalanobis:
    """Distance from the healthy feature distribution (shrinkage covariance)."""

    name = "Mahalanobis"

    def fit(self, Xs):
        X = np.vstack(Xs)
        self.cov = LedoitWolf().fit(X)
        return self

    def score(self, X):
        return np.sqrt(self.cov.mahalanobis(X))


class IForest:
    name = "Isolation Forest"

    def __init__(self, n_estimators: int = 300, seed: int = 0):
        self.model = IsolationForest(n_estimators=n_estimators, random_state=seed)

    def fit(self, Xs):
        self.model.fit(np.vstack(Xs))
        return self

    def score(self, X):
        return -self.model.score_samples(X)


class LSTMAutoencoder:
    """Sequence autoencoder over windows of consecutive snapshots.

    The score of snapshot ``t`` is the reconstruction error of the window ending at ``t``,
    so it only uses past and present data (causal, deployable online).
    """

    name = "LSTM autoencoder"

    def __init__(self, n_features: int, window=12, hidden=32, latent=8, epochs=60, lr=2e-3, batch_size=64, noise=0.1, seed=0):
        import torch
        from torch import nn

        torch.manual_seed(seed)
        self.torch, self.window, self.epochs, self.lr, self.batch_size, self.noise = torch, window, epochs, lr, batch_size, noise
        self.rng = np.random.default_rng(seed)

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.enc = nn.LSTM(n_features, hidden, batch_first=True)
                self.to_latent = nn.Linear(hidden, latent)
                self.from_latent = nn.Linear(latent, hidden)
                self.dec = nn.LSTM(hidden, hidden, batch_first=True)
                self.out = nn.Linear(hidden, n_features)

            def forward(self, x):
                _, (h, _) = self.enc(x)
                z = self.to_latent(h[-1])
                rep = self.from_latent(z).unsqueeze(1).repeat(1, x.shape[1], 1)
                y, _ = self.dec(rep)
                return self.out(y)

        self.net = Net()

    def _windows(self, X: np.ndarray) -> np.ndarray:
        pad = np.repeat(X[:1], self.window - 1, axis=0)
        Xp = np.vstack([pad, X])
        return np.lib.stride_tricks.sliding_window_view(Xp, self.window, axis=0).transpose(0, 2, 1).copy()

    def fit(self, Xs):
        torch = self.torch
        # Windows are built per bearing so a sequence never spans two bearings.
        W = torch.tensor(np.concatenate([self._windows(X)[self.window - 1 :] for X in Xs]))
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        self.net.train()
        for _ in range(self.epochs):
            for idx in np.array_split(self.rng.permutation(len(W)), max(1, len(W) // self.batch_size)):
                xb = W[idx]
                # denoising objective: reconstruct the clean window from a noisy copy,
                # which keeps the small model from memorizing the training windows
                loss = ((self.net(xb + self.noise * torch.randn_like(xb)) - xb) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
        return self

    def score(self, X):
        torch = self.torch
        self.net.eval()
        with torch.no_grad():
            W = torch.tensor(self._windows(X))
            err = ((self.net(W) - W) ** 2).mean(dim=(1, 2))
        return err.numpy()


def build_models(cfg: dict, columns: list[str]) -> list:
    m = cfg["models"]
    models = []
    if m["rms"]["enabled"]:
        models.append(RMSThreshold(columns))
    if m["mahalanobis"]["enabled"]:
        models.append(Mahalanobis())
    if m["isolation_forest"]["enabled"]:
        models.append(IForest(m["isolation_forest"]["n_estimators"], m["isolation_forest"]["seed"]))
    if m["lstm_ae"]["enabled"]:
        p = {k: v for k, v in m["lstm_ae"].items() if k != "enabled"}
        models.append(LSTMAutoencoder(len(columns), **p))
    return models

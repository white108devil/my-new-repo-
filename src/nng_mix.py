from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors


class NNGMixGenerator:
    """Nearest-Neighbor Gaussian Mixup generator for pseudo anomalies."""

    def __init__(
        self,
        k_neighbors: int,
        pseudo_per_anomaly: int,
        mix_mu: float,
        mix_sigma: float,
        noise_std: float,
        random_state: int | None = None,
    ) -> None:
        self.k_neighbors = max(1, int(k_neighbors))
        self.pseudo_per_anomaly = max(1, int(pseudo_per_anomaly))
        self.mix_mu = float(mix_mu)
        self.mix_sigma = float(mix_sigma)
        self.noise_std = float(noise_std)
        self.rng = np.random.default_rng(random_state)

    def generate(self, labeled_anomalies: np.ndarray, unlabeled: np.ndarray) -> np.ndarray:
        if labeled_anomalies.size == 0 or unlabeled.size == 0:
            return np.empty((0, unlabeled.shape[1] if unlabeled.ndim == 2 else 0))

        nn = NearestNeighbors(n_neighbors=self.k_neighbors, metric="euclidean")
        nn.fit(unlabeled)
        _, indices = nn.kneighbors(labeled_anomalies)

        pseudo = []
        for idx, x_anom in enumerate(labeled_anomalies):
            neighbors = unlabeled[indices[idx]]
            for _ in range(self.pseudo_per_anomaly):
                neighbor = neighbors[self.rng.integers(0, len(neighbors))]
                alpha = self.rng.normal(self.mix_mu, self.mix_sigma)
                alpha = float(np.clip(alpha, 0.0, 1.5))
                mixed = x_anom + alpha * (x_anom - neighbor)
                if self.noise_std > 0:
                    mixed = mixed + self.rng.normal(0.0, self.noise_std, size=mixed.shape)
                pseudo.append(mixed)

        return np.asarray(pseudo)

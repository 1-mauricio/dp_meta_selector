"""Aplicação de mecanismos DP em dados tabulares."""

import logging

import numpy as np

_log = logging.getLogger(__name__)

from .calibration import DELTA_DEFAULT, FAMILY_EPSILON, MECHANISM_EPSILON
from .mechanisms import (
    FAMILY_OF,
    Staircase,
)


class DPApplicator:
    def __init__(self, delta: float = DELTA_DEFAULT):
        self.delta = delta

    def apply(self, name: str, X: np.ndarray) -> np.ndarray:
        # Per-mechanism calibration takes precedence over per-family.
        eps = MECHANISM_EPSILON.get(name) or FAMILY_EPSILON[FAMILY_OF[name]]
        X_orig = X.astype(float).copy()
        X_out = X_orig.copy()
        for j in range(X_orig.shape[1]):
            col = X_orig[:, j]
            c_min = col.min()
            c_max = col.max()
            c_range = c_max - c_min + 1e-9
            col_n = (col - c_min) / c_range
            # B3: exceções propagam para o caller (_score_mechanism loga e penaliza com 0.0)
            noisy_n = self._col(name, col_n, eps)
            X_out[:, j] = noisy_n * c_range + c_min
        return X_out

    def _col(self, name: str, col_n: np.ndarray, eps: float) -> np.ndarray:
        n = len(col_n)

        # ── Contínuos simples: NumPy nativo ──────────────────────────────────
        if name == "Laplace":
            return col_n + np.random.laplace(0.0, 1.0 / eps, size=n)

        if name == "Gaussian":
            d = max(self.delta, 1e-4)
            sigma = np.sqrt(2.0 * np.log(1.25 / d)) / min(eps, 50.0)
            return col_n + np.random.normal(0.0, sigma, size=n)

        # ── PF4: mecanismos com distribuição analítica derivável ─────────────
        if name == "GaussianAnalytic":
            # Gaussiana analítica: mesma fórmula de sigma, sem objeto diffprivlib
            sigma = np.sqrt(2.0 * np.log(1.25 / max(self.delta, 1e-9))) / eps
            return col_n + np.random.normal(0.0, sigma, size=n)

        if name == "Staircase":
            # Staircase ≈ Laplace + discret. discreta; fallback para objeto (baixo n)
            m = Staircase(epsilon=eps, sensitivity=1.0)
            return np.array([float(m.randomise(float(v))) for v in col_n])

        if name == "LaplaceTruncated":
            # Truncated Laplace: ruído Laplace clipado em [0,1]
            scale = 1.0 / eps
            raw = col_n + np.random.laplace(0.0, scale, size=n)
            return np.clip(raw, 0.0, 1.0)

        if name == "LaplaceFolded":
            # Folded Laplace: |Laplace(x, b)| refletido em [0,1]
            scale = 1.0 / eps
            raw = col_n + np.random.laplace(0.0, scale, size=n)
            return np.abs(np.mod(raw, 2.0) - 1.0)  # dobramento em [0,1]

        if name == "Snapping":
            # Snapping: Laplace + arredondamento para potência de 2 mais próxima
            scale = 1.0 / eps
            raw = col_n + np.random.laplace(0.0, scale, size=n)
            # snap para resolução lambda = 2^ceil(log2(sensitivity/eps))
            lam = 2.0 ** np.ceil(np.log2(1.0 / eps + 1e-30))
            snapped = np.round(raw / lam) * lam
            return np.clip(snapped, 0.0, 1.0)

        if name == "Uniform":
            # Uniform: adiciona ruído uniforme em [-1/(2ε), 1/(2ε)] por linha
            half = 1.0 / (2.0 * eps)
            return col_n + np.random.uniform(-half, half, size=n)

        # ── PF4: discretos — usa vetorização nativa de NumPy sem loop Python ─
        if name in ("Geometric", "GeometricTruncated", "GeometricFolded"):
            col_int = np.round(col_n * 100).astype(int)
            p = 1.0 - np.exp(-eps)  # p da geométrica bi-lateral
            # Geométrica bilateral: diferença de duas geométricas unilaterais
            g1 = np.random.geometric(p, size=n) - 1
            g2 = np.random.geometric(p, size=n) - 1
            noise = g1 - g2
            noisy = (col_int + noise).astype(float)
            if name == "GeometricTruncated":
                noisy = np.clip(noisy, 0, 100)
            elif name == "GeometricFolded":
                noisy = np.abs(np.mod(noisy, 200) - 100).astype(float)
            return noisy / 100.0

        # ── PF6: Exponential — sem loop de objetos por linha ─────────────────
        if name == "Exponential":
            n_bins = min(20, max(2, len(np.unique(col_n))))
            centers = np.linspace(0.0, 1.0, n_bins)
            # utilidades: -|v - c| para cada (amostra × centro) — shape (n, n_bins)
            util = -np.abs(col_n[:, None] - centers[None, :])  # broadcasting
            # probabilidades via softmax estável: exp(eps/2 * u) / sum
            log_w = (eps / 2.0) * util
            log_w -= log_w.max(axis=1, keepdims=True)  # estabilidade numérica
            w = np.exp(log_w)
            w /= w.sum(axis=1, keepdims=True)
            # amostrar um índice por linha com base nas probabilidades
            cdf = np.cumsum(w, axis=1)
            u = np.random.uniform(size=(n, 1))
            idx = (cdf < u).sum(axis=1).clip(0, n_bins - 1)
            return centers[idx]

        raise ValueError(f"Mecanismo DP desconhecido: '{name}'")

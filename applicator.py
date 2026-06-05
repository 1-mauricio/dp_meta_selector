"""Aplicação de mecanismos DP em dados tabulares."""

import logging

import numpy as np

_log = logging.getLogger(__name__)

from .calibration import DELTA_DEFAULT, FAMILY_EPSILON
from .mechanisms import (
    FAMILY_OF,
    Exponential,
    Gaussian,
    GaussianAnalytic,
    Geometric,
    GeometricFolded,
    GeometricTruncated,
    Laplace,
    LaplaceFolded,
    LaplaceTruncated,
    Snapping,
    Staircase,
    Uniform,
)


class DPApplicator:
    def __init__(self, delta: float = DELTA_DEFAULT):
        self.delta = delta

    def apply(self, name: str, X: np.ndarray) -> np.ndarray:
        eps = FAMILY_EPSILON[FAMILY_OF[name]]
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
        # P1: mecanismos contínuos simples — vetorização total com NumPy
        if name == "Laplace":
            return col_n + np.random.laplace(0.0, 1.0 / eps, size=len(col_n))

        if name == "Gaussian":
            d = max(self.delta, 1e-4)
            sigma = np.sqrt(2.0 * np.log(1.25 / d)) / min(eps, 50.0)
            return col_n + np.random.normal(0.0, sigma, size=len(col_n))

        # P1: demais mecanismos contínuos — cria objeto uma vez por coluna
        if name == "GaussianAnalytic":
            m = GaussianAnalytic(epsilon=eps, delta=self.delta, sensitivity=1.0)
            return np.vectorize(m.randomise)(col_n)

        if name == "Staircase":
            m = Staircase(epsilon=eps, sensitivity=1.0)
            return np.vectorize(m.randomise)(col_n)

        if name == "LaplaceTruncated":
            m = LaplaceTruncated(epsilon=eps, sensitivity=1.0, lower=0.0, upper=1.0)
            return np.vectorize(m.randomise)(col_n)

        if name == "LaplaceFolded":
            m = LaplaceFolded(epsilon=eps, sensitivity=1.0, lower=0.0, upper=1.0)
            return np.vectorize(m.randomise)(col_n)

        if name == "Snapping":
            m = Snapping(epsilon=eps, sensitivity=1.0, lower=0.0, upper=1.0)
            return np.vectorize(m.randomise)(col_n)

        if name == "Uniform":
            m = Uniform(delta=float(np.clip(self.delta, 1e-9, 0.5)), sensitivity=1.0)
            return np.vectorize(m.randomise)(col_n)

        # P1: mecanismos discretos — inteiros, cria objeto uma vez por coluna
        if name in ("Geometric", "GeometricTruncated", "GeometricFolded"):
            col_int = np.round(col_n * 100).astype(int)
            if name == "Geometric":
                m = Geometric(epsilon=eps, sensitivity=1)
            elif name == "GeometricTruncated":
                m = GeometricTruncated(epsilon=eps, sensitivity=1, lower=0, upper=100)
            else:
                m = GeometricFolded(epsilon=eps, sensitivity=1, lower=0, upper=100)
            noisy = np.vectorize(lambda v: float(m.randomise(int(v))))(col_int)
            return noisy / 100.0

        # Exponential: candidatos dependem do valor — loop inevitável
        if name == "Exponential":
            n_bins = min(20, max(2, len(np.unique(col_n))))
            centers = np.linspace(0.0, 1.0, n_bins)
            out = []
            for v in col_n:
                mech = Exponential(
                    epsilon=eps,
                    sensitivity=1.0,
                    utility=(-np.abs(v - centers)).tolist(),
                    candidates=centers.tolist(),
                )
                out.append(float(mech.randomise()))
            return np.array(out)

        raise ValueError(f"Mecanismo DP desconhecido: '{name}'")

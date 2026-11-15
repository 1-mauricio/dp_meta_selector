"""Scale-aware L2 distortion and relative regret for released NB statistics."""

from __future__ import annotations

import numpy as np

from .privacy import sufficient_statistic_sensitivities


def sufficient_statistic_l2_distortion(clean, released) -> float:
    """RMS L2 error across statistic blocks, each divided by public L2 sensitivity.

    This dimensionless research metric gives equal weight to counts, sums, and
    second moments. It uses only their replace-one public sensitivity bounds for
    scaling; the clean values are needed to score an experiment, not as selector
    features or deployment-time inputs.
    """
    clean_blocks = {
        "counts": np.asarray(clean.counts, dtype=float),
        "sums": np.asarray(clean.sums, dtype=float),
        "seconds": np.asarray(clean.seconds, dtype=float),
    }
    released_blocks = {
        "counts": np.asarray(released.counts, dtype=float),
        "sums": np.asarray(released.sums, dtype=float),
        "seconds": np.asarray(released.seconds, dtype=float),
    }
    p = clean_blocks["sums"].shape[1] if clean_blocks["sums"].ndim == 2 else 0
    if p < 1:
        raise ValueError("somas devem ter shape (classes, atributos) não vazio")
    sensitivities = sufficient_statistic_sensitivities(p)
    normalized_errors = []
    for name in ("counts", "sums", "seconds"):
        clean_value, released_value = clean_blocks[name], released_blocks[name]
        if clean_value.shape != released_value.shape:
            raise ValueError(f"shapes incompatíveis para {name}")
        if not np.isfinite(clean_value).all() or not np.isfinite(released_value).all():
            raise ValueError("estatísticas contêm valores não finitos")
        normalized_errors.append(
            np.linalg.norm(released_value - clean_value) / sensitivities[name].l2
        )
    return float(np.sqrt(np.mean(np.square(normalized_errors))))


def relative_l2_regrets(distortions: np.ndarray) -> np.ndarray:
    """Normalize candidate distortions by the best candidate in one condition."""
    distortions = np.asarray(distortions, dtype=float)
    if distortions.ndim != 1 or not len(distortions):
        raise ValueError("distortions precisa ser um vetor não vazio")
    if not np.isfinite(distortions).all() or np.any(distortions < 0):
        raise ValueError("distortions precisam ser finitas e não negativas")
    best = float(distortions.min())
    if best == 0:
        if np.all(distortions == 0):
            return np.ones_like(distortions)
        raise ValueError("regret relativo indefinido: melhor distorção é zero")
    return distortions / best

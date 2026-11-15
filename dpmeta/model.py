"""Exact and private Gaussian Naive Bayes fits from sufficient statistics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .mechanisms import make_release, perturb_vector, privacy_rng, zcdp_rho_for_epsilon
from .privacy import PrivacyLedger, sufficient_statistic_sensitivities

VARIANCE_FLOOR = 1e-6
COUNT_FLOOR = 1e-6
DELTA_TOTAL = 1e-6
ALLOCATION_EQUAL = (1 / 3, 1 / 3, 1 / 3)
ALLOCATION_204040 = (0.2, 0.4, 0.4)
ALLOCATIONS = {
    "equal": ALLOCATION_EQUAL,
    "count_20_sum_40_second_40": ALLOCATION_204040,
}


@dataclass(frozen=True)
class SufficientStatistics:
    counts: np.ndarray
    sums: np.ndarray
    seconds: np.ndarray


@dataclass(frozen=True)
class GaussianNBModel:
    means: np.ndarray
    variances: np.ndarray
    priors: np.ndarray


def sufficient_statistics(x: np.ndarray, y: np.ndarray, n_classes: int) -> SufficientStatistics:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=int)
    if x.ndim != 2 or y.shape != (x.shape[0],) or n_classes < 2:
        raise ValueError("shapes inválidos para as estatísticas")
    if not np.isfinite(x).all() or np.any(np.abs(x) > 1) or np.any(y < 0) or np.any(y >= n_classes):
        raise ValueError("os dados devem estar em [-1,1] e nos índices públicos de classe")
    p = x.shape[1]
    counts = np.bincount(y, minlength=n_classes).astype(float)
    sums = np.zeros((n_classes, p), dtype=float)
    seconds = np.zeros((n_classes, p), dtype=float)
    for klass in range(n_classes):
        block = x[y == klass]
        if len(block):
            sums[klass] = block.sum(axis=0)
            seconds[klass] = np.square(block).sum(axis=0)
    return SufficientStatistics(counts, sums, seconds)


def model_from_statistics(stats: SufficientStatistics) -> GaussianNBModel:
    counts = np.maximum(np.asarray(stats.counts, dtype=float), 0.0)
    denom = np.maximum(counts, COUNT_FLOOR)
    means = np.clip(np.asarray(stats.sums, dtype=float) / denom[:, None], -1.0, 1.0)
    raw_variance = np.asarray(stats.seconds, dtype=float) / denom[:, None] - np.square(means)
    variances = np.maximum(raw_variance, VARIANCE_FLOOR)
    priors = denom / denom.sum()
    return GaussianNBModel(means, variances, priors)


def fit_clean(x: np.ndarray, y: np.ndarray, n_classes: int) -> GaussianNBModel:
    return model_from_statistics(sufficient_statistics(x, y, n_classes))


def fit_private(
    x: np.ndarray,
    y: np.ndarray,
    n_classes: int,
    *,
    epsilon: float,
    mechanism: str,
    allocation: str,
    seed: int | None = None,
    delta: float | None = None,
    stats: SufficientStatistics | None = None,
) -> tuple[GaussianNBModel, PrivacyLedger]:
    if stats is None:
        stats = sufficient_statistics(x, y, n_classes)
    noisy, ledger = privatize_sufficient_statistics(
        stats, epsilon=epsilon, mechanism=mechanism, allocation=allocation,
        seed=seed, delta=delta,
    )
    return model_from_statistics(noisy), ledger


def privatize_sufficient_statistics(
    stats: SufficientStatistics,
    *,
    epsilon: float,
    mechanism: str,
    allocation: str,
    seed: int | None = None,
    delta: float | None = None,
) -> tuple[SufficientStatistics, PrivacyLedger]:
    """Release the three sufficient-statistic blocks with one composed budget."""
    if allocation not in ALLOCATIONS:
        raise ValueError(f"alocação desconhecida: {allocation}")
    if mechanism not in {"laplace", "staircase", "analytic_gaussian", "gaussian_zcdp"}:
        raise ValueError(f"mecanismo desconhecido: {mechanism}")
    p = np.asarray(stats.sums).shape[1]
    if (np.asarray(stats.counts).ndim != 1 or np.asarray(stats.sums).ndim != 2
            or np.asarray(stats.seconds).shape != np.asarray(stats.sums).shape
            or np.asarray(stats.counts).shape[0] != np.asarray(stats.sums).shape[0]
            or p < 1):
        raise ValueError("shapes inválidos para as estatísticas suficientes")
    sensitivities = sufficient_statistic_sensitivities(p)
    shares = ALLOCATIONS[allocation]
    if mechanism in {"analytic_gaussian", "gaussian_zcdp"}:
        delta_total = DELTA_TOTAL if delta is None else float(delta)
        if not 0 < delta_total < 1:
            raise ValueError("delta precisa estar em (0,1) para mecanismos Gaussianos")
    else:
        if delta not in (None, 0, 0.0):
            raise ValueError("mecanismos puros requerem delta=0")
        delta_total = 0.0
    ledger = PrivacyLedger(
        epsilon, delta_total,
        accounting="zcdp" if mechanism == "gaussian_zcdp" else "basic",
    )
    rho_total = (zcdp_rho_for_epsilon(epsilon, delta_total)
                 if mechanism == "gaussian_zcdp" else 0.0)
    rng = privacy_rng(seed)
    originals = {"counts": stats.counts, "sums": stats.sums, "seconds": stats.seconds}
    noisy = {}
    for index, name in enumerate(("counts", "sums", "seconds")):
        eps_i = epsilon * shares[index]
        delta_i = (ledger.delta_limit * shares[index]
                   if mechanism == "analytic_gaussian" else 0.0)
        rho_i = rho_total * shares[index]
        perturbed, scale = perturb_vector(
            originals[name], mechanism=mechanism, epsilon=eps_i, delta=delta_i,
            sensitivity=sensitivities[name], rng=rng,
            rho=rho_i if mechanism == "gaussian_zcdp" else None,
        )
        ledger.record(make_release(
            name, mechanism, 0.0 if mechanism == "gaussian_zcdp" else eps_i,
            delta_i, sensitivities[name], scale, rho=rho_i if mechanism == "gaussian_zcdp" else 0.0,
        ))
        noisy[name] = perturbed
    ledger.verify()
    return SufficientStatistics(**noisy), ledger


def predict(x: np.ndarray, model: GaussianNBModel) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    scores = np.empty((len(x), len(model.priors)), dtype=float)
    for klass, prior in enumerate(model.priors):
        log_likelihood = -0.5 * np.sum(
            np.log(2 * np.pi * model.variances[klass])
            + np.square(x - model.means[klass]) / model.variances[klass],
            axis=1,
        )
        scores[:, klass] = log_likelihood + np.log(max(prior, COUNT_FLOOR))
    return scores.argmax(axis=1)


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    if not len(y_true) or y_true.shape != y_pred.shape:
        raise ValueError("rótulos vazios ou com shapes diferentes")
    recalls = [np.mean(y_pred[y_true == klass] == klass)
               for klass in np.unique(y_true)]
    return float(np.mean(recalls))

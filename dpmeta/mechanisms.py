"""Calibrated noise distributions used by the benchmark."""

from __future__ import annotations

from functools import lru_cache
from math import exp

import numpy as np
from scipy.special import ndtr

from .privacy import Release, Sensitivity
from .randomness import make_rng


@lru_cache(maxsize=4096)
def analytic_gaussian_sigma(epsilon: float, delta: float, sensitivity_l2: float) -> float:
    """Return sigma for the analytic Gaussian mechanism (Balle & Wang 2018)."""
    if epsilon <= 0 or not 0 < delta < 1 or sensitivity_l2 <= 0:
        raise ValueError("epsilon, delta ou sensibilidade inválidos")

    def achieved_delta(mu: float) -> float:
        # mu = sensitivity / sigma. The expression is monotone increasing in mu.
        first = ndtr(mu / 2.0 - epsilon / mu)
        second = exp(epsilon) * ndtr(-mu / 2.0 - epsilon / mu)
        return float(max(0.0, first - second))

    low, high = 0.0, 1.0
    while achieved_delta(high) < delta:
        high *= 2.0
        if high > 1e8:
            raise ArithmeticError("falha ao calibrar sigma Gaussiano")
    for _ in range(120):
        middle = (low + high) / 2.0
        if achieved_delta(middle) < delta:
            low = middle
        else:
            high = middle
    mu = (low + high) / 2.0
    return sensitivity_l2 / mu


def zcdp_rho_for_epsilon(epsilon: float, delta: float) -> float:
    """Invert the standard rho-zCDP to (epsilon, delta)-DP conversion."""
    if epsilon <= 0 or not 0 < delta < 1:
        raise ValueError("epsilon e delta inválidos para zCDP")
    root_log_delta = np.sqrt(np.log(1.0 / delta))
    return float((np.sqrt(root_log_delta**2 + epsilon) - root_log_delta) ** 2)


def calibration_scale(
    *, mechanism: str, epsilon: float, delta: float, sensitivity: Sensitivity,
    output_size: int, rho: float | None = None,
) -> float:
    """Return the calibrated noise scale or a documented staircase proxy."""
    if output_size < 1:
        raise ValueError("output_size precisa ser positivo")
    if mechanism == "laplace":
        return sensitivity.l1 / epsilon
    if mechanism == "analytic_gaussian":
        return analytic_gaussian_sigma(epsilon, delta, sensitivity.l2)
    if mechanism == "gaussian_zcdp":
        if rho is None or rho <= 0:
            raise ValueError("rho positivo é necessário para Gaussian zCDP")
        return sensitivity.l2 / np.sqrt(2.0 * rho)
    if mechanism == "staircase":
        # Product composition is over coordinates changed by one record, not
        # every coordinate in the released vector. If no sparsity bound is
        # supplied, output_size remains a conservative fallback.
        affected = sensitivity.max_changed_coordinates or output_size
        return sensitivity.linf * affected / epsilon
    raise ValueError(f"mecanismo desconhecido: {mechanism}")


def _staircase_noise(
    shape: tuple[int, ...], *, epsilon: float, sensitivity: float,
    rng: np.random.Generator, gamma: float = 0.5,
) -> np.ndarray:
    """Sample the symmetric staircase distribution by a geometric mixture."""
    if sensitivity <= 0 or epsilon <= 0 or not 0 < gamma < 1:
        raise ValueError("parâmetros staircase inválidos")
    size = int(np.prod(shape))
    q = exp(-epsilon)
    normalizer = gamma + q * (1.0 - gamma)
    shells = rng.geometric(1.0 - q, size=size) - 1
    inner = rng.random(size) < gamma / normalizer
    offsets = np.empty(size, dtype=float)
    first = int(np.count_nonzero(inner))
    offsets[inner] = rng.random(first) * gamma * sensitivity
    outer_count = size - first
    offsets[~inner] = (gamma + rng.random(outer_count) * (1.0 - gamma)) * sensitivity
    signs = np.where(rng.random(size) < 0.5, -1.0, 1.0)
    return (signs * (shells * sensitivity + offsets)).reshape(shape)


def perturb_vector(
    value: np.ndarray,
    *,
    mechanism: str,
    epsilon: float,
    delta: float,
    sensitivity: Sensitivity,
    rng,
    rho: float | None = None,
) -> tuple[np.ndarray, float]:
    """Perturb a joint vector and return (release, scale) for the ledger."""
    if epsilon <= 0:
        raise ValueError("epsilon por consulta deve ser positivo")
    value = np.asarray(value, dtype=float)
    if mechanism == "laplace":
        if delta != 0:
            raise ValueError("Laplace requer delta=0")
        scale = sensitivity.l1 / epsilon
        noise = rng.laplace(0.0, scale, size=value.shape)
    elif mechanism == "analytic_gaussian":
        scale = analytic_gaussian_sigma(epsilon, delta, sensitivity.l2)
        noise = rng.normal(0.0, scale, size=value.shape)
    elif mechanism == "gaussian_zcdp":
        if delta != 0 or rho is None or rho <= 0:
            raise ValueError("Gaussian zCDP requer rho positivo e delta por consulta zero")
        scale = calibration_scale(
            mechanism=mechanism, epsilon=epsilon, delta=0,
            sensitivity=sensitivity, output_size=value.size, rho=rho,
        )
        noise = rng.normal(0.0, scale, size=value.shape)
    elif mechanism == "staircase":
        if delta != 0:
            raise ValueError("staircase requer delta=0")
        affected = sensitivity.max_changed_coordinates or value.size
        coordinate_epsilon = epsilon / affected
        scale = calibration_scale(
            mechanism=mechanism, epsilon=epsilon, delta=0,
            sensitivity=sensitivity, output_size=value.size,
        )
        noise = _staircase_noise(
            value.shape, epsilon=coordinate_epsilon,
            sensitivity=sensitivity.linf, rng=rng,
        )
    else:
        raise ValueError(f"mecanismo desconhecido: {mechanism}")
    return value + noise, scale


def privacy_rng(seed: int | None):
    """Benchmark seeds are deterministic; None uses OS cryptographic entropy."""
    return make_rng(seed)


def make_release(name: str, mechanism: str, epsilon: float, delta: float,
                 sensitivity: Sensitivity, scale: float, *, rho: float = 0.0) -> Release:
    return Release(name, mechanism, float(epsilon), float(delta), sensitivity,
                   float(scale), float(rho))

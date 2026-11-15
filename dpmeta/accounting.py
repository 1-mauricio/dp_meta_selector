"""Small, auditable RDP helpers for Gaussian release composition."""

from __future__ import annotations

from math import exp, log, sqrt

import numpy as np
from scipy.optimize import brentq
from scipy.special import ndtr


DEFAULT_RDP_ORDERS = tuple(
    sorted({
        1.01, 1.02, 1.05, 1.1, 1.2, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0,
        10.0, 12.0, 16.0, 20.0, 24.0, 32.0, 48.0, 64.0, 96.0, 128.0,
        192.0, 256.0, 384.0, 512.0, 768.0, 1024.0,
    })
)


def gaussian_rdp(order: float, sensitivity_l2: float, noise_scale: float) -> float:
    """RDP order ``order`` of a Gaussian mechanism."""
    if order <= 1 or sensitivity_l2 < 0 or noise_scale <= 0:
        raise ValueError("ordem, sensibilidade ou escala inválida")
    return float(order * sensitivity_l2**2 / (2.0 * noise_scale**2))


def gaussian_privacy_profile_delta(epsilon: float, mu: float) -> float:
    """Exact hockey-stick profile of a Gaussian mechanism.

    ``mu`` is the L2 mean shift divided by the Gaussian standard deviation.
    Independent Gaussian releases compose by adding the squared ``mu`` values.
    """
    if epsilon < 0 or mu < 0:
        raise ValueError("epsilon/mu inválidos")
    if mu == 0:
        return 0.0
    value = ndtr(mu / 2.0 - epsilon / mu) - exp(epsilon) * ndtr(
        -mu / 2.0 - epsilon / mu,
    )
    return float(max(0.0, value))


def gaussian_mu_for_delta(epsilon: float, delta: float) -> float:
    """Largest Gaussian privacy-loss shift compatible with ``(epsilon,delta)``."""
    if epsilon <= 0 or not 0 < delta < 1:
        raise ValueError("epsilon/delta inválidos")
    low, high = 0.0, 1.0
    while gaussian_privacy_profile_delta(epsilon, high) < delta:
        high *= 2.0
        if high > 1e8:
            raise ArithmeticError("falha ao inverter o perfil Gaussiano")
    return float(brentq(
        lambda mu: gaussian_privacy_profile_delta(epsilon, mu) - delta,
        low + 1e-15, high,
        xtol=1e-13,
    ))


def compose_gaussian_privacy_profile(
    releases: list[tuple[float, float]],
    *,
    epsilon_gaussian: float,
    delta: float,
    pure_dp_epsilon: float = 0.0,
) -> dict:
    """Compose Gaussian releases with their exact Gaussian privacy profile."""
    if not releases or epsilon_gaussian <= 0 or not 0 < delta < 1:
        raise ValueError("releases, epsilon_gaussian ou delta inválidos")
    if pure_dp_epsilon < 0:
        raise ValueError("epsilon puro inválido")
    mu_squared = sum((sensitivity / scale) ** 2 for sensitivity, scale in releases)
    mu = sqrt(mu_squared)
    achieved_delta = gaussian_privacy_profile_delta(epsilon_gaussian, mu)
    return {
        "accounting": "exact_gaussian_privacy_profile_plus_pure_dp",
        "epsilon_gaussian": float(epsilon_gaussian),
        "epsilon_used": float(epsilon_gaussian + pure_dp_epsilon),
        "delta_used": float(achieved_delta),
        "delta_limit": float(delta),
        "mu_total": float(mu),
        "gaussian_releases": len(releases),
        "pure_dp_epsilon": float(pure_dp_epsilon),
        "within_delta": bool(achieved_delta <= delta + 1e-12),
    }


def compose_gaussian_rdp(
    releases: list[tuple[float, float]],
    *,
    delta: float,
    pure_dp_epsilon: float = 0.0,
    orders: tuple[float, ...] = DEFAULT_RDP_ORDERS,
) -> dict:
    """Compose Gaussian releases and convert the result to ``(ε,δ)`` DP.

    ``releases`` contains ``(sensitivity_l2, noise_scale)`` pairs. The pure-DP
    term is added after conversion because sequential composition between a
    pure-DP mechanism and the RDP composition is valid.
    """
    if not releases or not 0 < delta < 1 or pure_dp_epsilon < 0:
        raise ValueError("releases, delta ou epsilon puro inválidos")
    if not orders or any(order <= 1 for order in orders):
        raise ValueError("ordens RDP devem ser maiores que 1")
    values = []
    for order in orders:
        total_rdp = sum(
            gaussian_rdp(order, sensitivity, scale)
            for sensitivity, scale in releases
        )
        converted = total_rdp + log(1.0 / delta) / (order - 1.0)
        values.append((converted + pure_dp_epsilon, float(order), total_rdp))
    best = min(values, key=lambda item: item[0])
    return {
        "epsilon": float(best[0]),
        "delta": float(delta),
        "optimal_order": best[1],
        "gaussian_rdp_at_optimal_order": best[2],
        "pure_dp_epsilon": float(pure_dp_epsilon),
        "orders_checked": len(orders),
    }


def analytic_gaussian_hpo_rdp_bound(
    epsilon_total: float,
    *,
    delta_total: float,
    n_candidates: int = 3,
    tuning_epsilon_fraction: float = 0.10,
    selection_epsilon_fraction: float = 0.10,
    tuning_delta_fraction: float = 0.10,
    orders: tuple[float, ...] = DEFAULT_RDP_ORDERS,
) -> dict:
    """Audit RDP for the current analytic-Gaussian HPO budget split.

    Sensitivity cancels because each output perturbation scale is calibrated
    from the same sensitivity. This function audits the accounting only; it
    does not assume privacy amplification from the Poisson sample.
    """
    if epsilon_total <= 0 or n_candidates < 1:
        raise ValueError("epsilon_total/n_candidates inválidos")
    epsilon_tuning_total = epsilon_total * tuning_epsilon_fraction
    epsilon_selection = epsilon_total * selection_epsilon_fraction
    epsilon_final = epsilon_total - epsilon_tuning_total - epsilon_selection
    delta_tuning_total = delta_total * tuning_delta_fraction
    delta_final = delta_total - delta_tuning_total
    epsilon_candidate = epsilon_tuning_total / n_candidates
    delta_candidate = delta_tuning_total / n_candidates
    # Use unit sensitivity: sigma/Delta is invariant to the dataset dimension.
    from .mechanisms import analytic_gaussian_sigma

    candidate_scale = analytic_gaussian_sigma(
        epsilon_candidate, delta_candidate, 1.0,
    )
    final_scale = analytic_gaussian_sigma(epsilon_final, delta_final, 1.0)
    bound = compose_gaussian_rdp(
        [(1.0, candidate_scale)] * n_candidates + [(1.0, final_scale)],
        delta=delta_total,
        pure_dp_epsilon=epsilon_selection,
        orders=orders,
    )
    bound.update({
        "epsilon_total_limit": float(epsilon_total),
        "epsilon_basic_composition": float(epsilon_total),
        "epsilon_candidate": float(epsilon_candidate),
        "epsilon_final": float(epsilon_final),
        "epsilon_selection": float(epsilon_selection),
        "delta_candidate": float(delta_candidate),
        "delta_final": float(delta_final),
        "n_candidates": int(n_candidates),
    })
    return bound


__all__ = [
    "DEFAULT_RDP_ORDERS",
    "analytic_gaussian_hpo_rdp_bound",
    "compose_gaussian_privacy_profile",
    "compose_gaussian_rdp",
    "gaussian_mu_for_delta",
    "gaussian_privacy_profile_delta",
    "gaussian_rdp",
]

"""Budgeted DP diagnostics for data-dependent classifier selection."""

from __future__ import annotations

from math import log, sqrt

import numpy as np

from .mechanisms import analytic_gaussian_sigma, privacy_rng
from .privacy import PrivacyLedger, Release, Sensitivity

DP_DIAGNOSTIC_NAMES = (
    "class_entropy_normalized",
    "class_max_fraction",
    "class_min_fraction",
    "class_support_cv",
    "absent_class_fraction",
    "feature_abs_mean_mean",
    "feature_std_q10",
    "feature_std_median",
    "feature_std_q90",
    "feature_variance_q10",
    "feature_variance_q90",
    "class_separation_mean",
    "class_separation_q90",
)


def sufficient_statistic_l2_sensitivity(n_features: int) -> float:
    """Replace-one sensitivity for joint class counts, sums, and second moments.

    Each bounded record contributes a vector with squared norm at most
    ``1 + 2*p``: one class-count coordinate, at most ``p`` in squared feature
    values, and at most ``p`` in their squares. Replacing one record changes
    the vector by at most twice this norm.
    """
    if n_features < 1:
        raise ValueError("n_features precisa ser positivo")
    return 2.0 * sqrt(1.0 + 2.0 * n_features)


def dp_training_diagnostics(
    x: np.ndarray,
    y: np.ndarray,
    n_classes: int,
    *,
    epsilon: float,
    delta: float,
    seed: int | None = None,
) -> tuple[np.ndarray, dict]:
    """Release DP descriptors derived from noisy train-only sufficient stats.

    The single analytic-Gaussian vector release covers class counts, sums and
    second moments jointly under fixed-size replace-one adjacency. All returned
    descriptors are post-processing of that one release.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=int)
    if x.ndim != 2 or y.shape != (len(x),) or not len(y):
        raise ValueError("shapes inválidos para diagnósticos DP")
    if n_classes < 2 or np.any(y < 0) or np.any(y >= n_classes):
        raise ValueError("rótulos/classes inválidos para diagnósticos DP")
    if not np.isfinite(x).all() or np.any(np.abs(x) > 1.0 + 1e-12):
        raise ValueError("features dos diagnósticos precisam estar no domínio público [-1,1]")
    if epsilon <= 0 or not 0 < delta < 1:
        raise ValueError("epsilon/delta inválidos para diagnósticos DP")

    n, p = x.shape
    counts = np.bincount(y, minlength=n_classes).astype(float)
    sums = np.zeros((n_classes, p), dtype=float)
    seconds = np.zeros((n_classes, p), dtype=float)
    for klass in range(n_classes):
        rows = x[y == klass]
        if len(rows):
            sums[klass] = rows.sum(axis=0)
            seconds[klass] = np.square(rows).sum(axis=0)
    query = np.r_[counts, sums.ravel(), seconds.ravel()]

    sensitivity_l2 = sufficient_statistic_l2_sensitivity(p)
    sigma = analytic_gaussian_sigma(epsilon, delta, sensitivity_l2)
    noisy = query + privacy_rng(seed).normal(0.0, sigma, size=query.shape)

    count_end = n_classes
    sum_end = count_end + n_classes * p
    noisy_counts = noisy[:count_end].reshape(n_classes)
    noisy_sums = noisy[count_end:sum_end].reshape(n_classes, p)
    noisy_seconds = noisy[sum_end:].reshape(n_classes, p)

    # Projection and all feature construction below are post-processing.
    projected_counts = np.maximum(noisy_counts, 0.0)
    if projected_counts.sum() <= 1e-12:
        fractions = np.full(n_classes, 1.0 / n_classes)
        projected_counts = fractions * n
    else:
        fractions = projected_counts / projected_counts.sum()
        projected_counts = fractions * n
    denominators = np.maximum(projected_counts, 1.0)
    class_means = np.clip(noisy_sums / denominators[:, None], -1.0, 1.0)

    overall_mean = np.clip(noisy_sums.sum(axis=0) / n, -1.0, 1.0)
    overall_second = noisy_seconds.sum(axis=0) / n
    total_variance = np.maximum(overall_second - np.square(overall_mean), 0.0)
    standard_deviations = np.sqrt(total_variance)
    between = np.sum(
        fractions[:, None] * np.square(class_means - overall_mean[None, :]), axis=0,
    )
    separation = np.divide(
        between, total_variance, out=np.zeros_like(between), where=total_variance > 1e-12,
    )
    separation = np.clip(separation, 0.0, 1.0)

    positive = fractions[fractions > 0]
    entropy = -float(np.sum(positive * np.log(positive)))
    normalized_entropy = entropy / log(n_classes)
    std_q10, std_median, std_q90 = np.quantile(standard_deviations, [0.1, 0.5, 0.9])
    var_q10, var_q90 = np.quantile(total_variance, [0.1, 0.9])
    features = np.asarray([
        normalized_entropy,
        float(fractions.max()),
        float(fractions.min()),
        float(np.std(projected_counts) / max(np.mean(projected_counts), 1e-12)),
        float(np.mean(projected_counts <= 1e-12)),
        float(np.mean(np.abs(overall_mean))),
        float(std_q10),
        float(std_median),
        float(std_q90),
        float(var_q10),
        float(var_q90),
        float(np.mean(separation)),
        float(np.quantile(separation, 0.9)),
    ], dtype=float)
    if not np.isfinite(features).all():
        raise FloatingPointError("diagnósticos DP resultaram em features não finitas")

    sensitivity = Sensitivity(
        l1=sqrt(len(query)) * sensitivity_l2,
        l2=sensitivity_l2,
        linf=sensitivity_l2,
    )
    ledger = PrivacyLedger(float(epsilon), float(delta))
    ledger.record(Release(
        name="train_sufficient_statistics_for_meta_features",
        mechanism="analytic_gaussian_joint_vector",
        epsilon=float(epsilon),
        delta=float(delta),
        sensitivity=sensitivity,
        noise_scale=float(sigma),
    ))
    return features, ledger.verify().as_dict()

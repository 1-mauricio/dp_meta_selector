"""DP-calibration and research-only data-derived meta-features."""

from __future__ import annotations

from math import log

import numpy as np

from .mechanisms import calibration_scale, zcdp_rho_for_epsilon
from .model import ALLOCATIONS, DELTA_TOTAL
from .privacy import PrivacyLedger, Release, Sensitivity, sufficient_statistic_sensitivities
from .randomness import make_rng
from .schema import PublicSchema

PRIVATE_META_NAMES = (
    "class_entropy_normalized", "class_max_fraction", "class_min_fraction",
    "class_support_cv", "absent_class_fraction", "feature_abs_mean_mean",
    "feature_std_median", "feature_std_q10", "feature_std_q90",
    "feature_variance_q10", "feature_variance_q90", "near_zero_fraction",
    "boundary_fraction", "feature_abs_skew_median", "class_separation_mean",
    "class_separation_q90", "abs_correlation_mean", "abs_correlation_q90",
)
PRIVATE_META_COLUMNS = tuple(f"private_meta__{name}" for name in PRIVATE_META_NAMES)

# This is a deliberately small release of train-only descriptors.  It uses
# bounded scalar/vector summaries instead of trying to privatize arbitrary
# quantiles and correlations.  The release is useful for a first deployment
# experiment; it is separate from the model epsilon and must be composed with
# it in a real application.
DP_META_NAMES = (
    "class_entropy_normalized", "class_max_fraction", "class_min_fraction",
    "class_support_cv", "absent_class_fraction", "feature_abs_mean_mean",
    "feature_second_moment_mean", "feature_abs_class_gap",
    "feature_second_class_gap", "near_zero_fraction", "boundary_fraction",
)
DP_META_COLUMNS = tuple(f"dp_meta__{name}" for name in DP_META_NAMES)
DP_META_EPSILON_DEFAULT = 1.0

_STAT_NAMES = ("counts", "sums", "seconds")
_DP_PER_STAT_NAMES = (
    "budget_share", "delta_share", "log_sens_l1", "log_sens_l2",
    "log_sens_linf", "log_calibration_scale", "log_scale_over_n",
    "log_scale_over_n_per_feature",
)
DP_CALIBRATION_FEATURE_NAMES = (
    "log_zcdp_rho_total",
    *(f"{name}__{stat}" for stat in _STAT_NAMES for name in _DP_PER_STAT_NAMES),
)


def _log1p_nonnegative(value: float) -> float:
    return float(np.log1p(max(0.0, float(value))))


def dp_calibration_features(
    n: int, schema: PublicSchema, epsilon: float, mechanism: str, allocation: str,
    *, delta_total: float = DELTA_TOTAL,
) -> np.ndarray:
    """Public, action-specific features from the DP calibration, not row values."""
    if n < 1 or epsilon <= 0:
        raise ValueError("n e epsilon precisam ser positivos")
    if mechanism not in {"laplace", "staircase", "analytic_gaussian", "gaussian_zcdp"}:
        raise ValueError(f"mecanismo desconhecido: {mechanism}")
    if allocation not in ALLOCATIONS:
        raise ValueError(f"alocação desconhecida: {allocation}")
    if mechanism in {"analytic_gaussian", "gaussian_zcdp"} and not 0 < delta_total < 1:
        raise ValueError("delta_total precisa estar em (0,1) para mecanismos Gaussianos")

    p, n_classes = schema.encoded_dim, len(schema.classes)
    sizes = (n_classes, n_classes * p, n_classes * p)
    shares = ALLOCATIONS[allocation]
    sensitivities = sufficient_statistic_sensitivities(p)
    rho_total = (zcdp_rho_for_epsilon(epsilon, delta_total)
                 if mechanism == "gaussian_zcdp" else 0.0)
    values = [_log1p_nonnegative(rho_total)]
    for index, stat_name in enumerate(_STAT_NAMES):
        sensitivity = sensitivities[stat_name]
        eps_i = epsilon * shares[index]
        delta_i = (delta_total * shares[index]
                   if mechanism == "analytic_gaussian" else 0.0)
        rho_i = rho_total * shares[index]
        scale = calibration_scale(
            mechanism=mechanism, epsilon=eps_i, delta=delta_i,
            sensitivity=sensitivity, output_size=sizes[index],
            rho=rho_i if mechanism == "gaussian_zcdp" else None,
        )
        values.extend([
            float(shares[index]),
            float(delta_i / delta_total) if delta_i else 0.0,
            _log1p_nonnegative(sensitivity.l1),
            _log1p_nonnegative(sensitivity.l2),
            _log1p_nonnegative(sensitivity.linf),
            _log1p_nonnegative(scale),
            _log1p_nonnegative(scale / n),
            _log1p_nonnegative(scale / (n / p)),
        ])
    return np.asarray(values, dtype=float)


def private_training_meta_features(x: np.ndarray, y: np.ndarray,
                                   n_classes: int) -> np.ndarray:
    """Compute train-only data descriptors; these are NOT differentially private.

    They are included solely as a research diagnostic/upper-bound branch. A
    selector that consumes them on sensitive data is not covered by the final
    model's DP guarantee unless the feature release and adaptive selection are
    separately privatized and composed.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=int)
    if x.ndim != 2 or y.shape != (x.shape[0],) or not len(y):
        raise ValueError("shapes inválidos para meta-features privadas")
    if not np.isfinite(x).all() or np.any(y < 0) or np.any(y >= n_classes):
        raise ValueError("dados inválidos para meta-features privadas")

    n, p = x.shape
    counts = np.bincount(y, minlength=n_classes).astype(float)
    fractions = counts / n
    nonzero_fractions = fractions[fractions > 0]
    entropy = -float(np.sum(nonzero_fractions * np.log(nonzero_fractions)))
    entropy_normalized = entropy / log(n_classes) if n_classes > 1 else 0.0
    support_cv = float(np.std(counts) / np.mean(counts)) if np.mean(counts) else 0.0

    means = x.mean(axis=0)
    centered = x - means
    variances = np.mean(np.square(centered), axis=0)
    standard_deviations = np.sqrt(variances)
    active = standard_deviations > 1e-12
    if np.any(active):
        skew = np.abs(np.mean(np.power(centered[:, active], 3), axis=0)
                       / np.power(standard_deviations[active], 3))
        skew_median = float(np.median(skew))
    else:
        skew_median = 0.0

    between = np.zeros(p, dtype=float)
    for klass in range(n_classes):
        if counts[klass] > 0:
            class_mean = x[y == klass].mean(axis=0)
            between += fractions[klass] * np.square(class_mean - means)
    separation = np.divide(between, variances, out=np.zeros_like(between),
                           where=variances > 1e-12)
    separation = np.clip(separation, 0.0, 1.0)

    active_indices = np.flatnonzero(active)
    if len(active_indices) > 64:
        # Deterministic schema-order subsample keeps this diagnostic bounded.
        active_indices = active_indices[np.linspace(0, len(active_indices) - 1, 64).astype(int)]
    if len(active_indices) >= 2:
        corr = np.corrcoef(x[:, active_indices], rowvar=False)
        upper = np.abs(corr[np.triu_indices_from(corr, k=1)])
        upper = upper[np.isfinite(upper)]
    else:
        upper = np.asarray([], dtype=float)
    corr_mean = float(np.mean(upper)) if upper.size else 0.0
    corr_q90 = float(np.quantile(upper, 0.9)) if upper.size else 0.0
    std_quantiles = np.quantile(standard_deviations, [0.1, 0.5, 0.9])
    variance_quantiles = np.quantile(variances, [0.1, 0.9])

    values = np.asarray([
        entropy_normalized,
        float(np.max(fractions)),
        float(np.min(fractions)),
        support_cv,
        float(np.mean(counts == 0)),
        float(np.mean(np.abs(means))),
        float(std_quantiles[1]),
        float(std_quantiles[0]),
        float(std_quantiles[2]),
        float(variance_quantiles[0]),
        float(variance_quantiles[1]),
        float(np.mean(np.abs(x) <= 1e-12)),
        float(np.mean(np.abs(x) >= 1.0 - 1e-12)),
        skew_median,
        float(np.mean(separation)),
        float(np.quantile(separation, 0.9)),
        corr_mean,
        corr_q90,
    ], dtype=float)
    return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)


def dp_training_meta_features(
    x: np.ndarray,
    y: np.ndarray,
    n_classes: int,
    *,
    epsilon: float = DP_META_EPSILON_DEFAULT,
    seed: int | None = None,
) -> tuple[np.ndarray, PrivacyLedger]:
    """Release a compact DP descriptor vector from bounded training rows.

    ``x`` must already be encoded in ``[-1, 1]``.  The mechanism releases four
    bounded queries with sequential Laplace composition: class counts, class
    absolute/second moments, and two indicator totals.  All downstream ratios,
    clipping and normalization are post-processing.  The returned ledger is
    intentionally explicit so callers can add this release to the model
    budget instead of silently treating it as free metadata.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=int)
    if x.ndim != 2 or y.shape != (x.shape[0],) or not len(y):
        raise ValueError("shapes inválidos para meta-features DP")
    if n_classes < 2 or np.any(y < 0) or np.any(y >= n_classes):
        raise ValueError("classes inválidas para meta-features DP")
    if not np.isfinite(x).all() or np.min(x) < -1.0 or np.max(x) > 1.0:
        raise ValueError("meta-features DP requerem x em [-1,1]")
    if epsilon <= 0.0 or not np.isfinite(epsilon):
        raise ValueError("epsilon das meta-features precisa ser positivo e finito")

    # The four vector queries have replace-one L1 sensitivity at most two.
    shares = (0.30, 0.30, 0.25, 0.15)
    generator = make_rng(seed)
    ledger = PrivacyLedger(float(epsilon), 0.0)

    row_abs = np.mean(np.abs(x), axis=1)
    row_second = np.mean(np.square(x), axis=1)
    indicators = np.column_stack([
        np.mean(np.abs(x) <= 1e-12, axis=1),
        np.mean(np.abs(x) >= 1.0 - 1e-12, axis=1),
    ])
    raw_counts = np.bincount(y, minlength=n_classes).astype(float)
    raw_abs = np.bincount(y, weights=row_abs, minlength=n_classes).astype(float)
    raw_second = np.bincount(y, weights=row_second, minlength=n_classes).astype(float)
    raw_indicators = indicators.sum(axis=0).astype(float)

    def noisy(values: np.ndarray, share: float, sensitivity: float = 2.0) -> np.ndarray:
        eps_i = float(epsilon * share)
        scale = sensitivity / eps_i
        return np.asarray(values, dtype=float) + generator.laplace(
            0.0, scale, size=np.asarray(values).shape,
        )

    noisy_counts = noisy(raw_counts, shares[0])
    noisy_abs = noisy(raw_abs, shares[1])
    noisy_second = noisy(raw_second, shares[2])
    noisy_indicators = noisy(raw_indicators, shares[3])
    for index, (name, share) in enumerate(zip(
            ("class_counts", "class_moments", "feature_indicators"),
            (shares[0], shares[1], shares[3]))):
        if index == 0:
            sensitivity = Sensitivity(2.0, np.sqrt(2.0), 1.0)
            scale = 2.0 / (epsilon * share)
        elif index == 1:
            sensitivity = Sensitivity(2.0, np.sqrt(2.0), 1.0)
            scale = 2.0 / (epsilon * share)
        else:
            sensitivity = Sensitivity(2.0, np.sqrt(2.0), 1.0)
            scale = 2.0 / (epsilon * share)
        ledger.record(Release(
            name=f"dp_meta:{name}", mechanism="laplace_vector", epsilon=epsilon * share,
            delta=0.0, sensitivity=sensitivity, noise_scale=scale,
        ))
    # Counts and class moments are released together as one vector query each;
    # keeping a separate ledger entry for the second moment makes composition
    # auditable rather than hiding it in a larger sensitivity constant.
    ledger.record(Release(
        name="dp_meta:class_second_moments", mechanism="laplace_vector",
        epsilon=epsilon * shares[2], delta=0.0,
        sensitivity=Sensitivity(2.0, np.sqrt(2.0), 1.0),
        noise_scale=2.0 / (epsilon * shares[2]),
    ))

    positive_counts = np.maximum(noisy_counts, 0.0)
    total = float(positive_counts.sum())
    if total <= 0.0 or not np.isfinite(total):
        fractions = np.full(n_classes, 1.0 / n_classes)
        total = 1.0
    else:
        fractions = positive_counts / total
    abs_means = np.clip(
        noisy_abs / np.maximum(positive_counts, 1.0), 0.0, 1.0,
    )
    second_means = np.clip(
        noisy_second / np.maximum(positive_counts, 1.0), 0.0, 1.0,
    )
    global_abs = float(np.clip(np.sum(positive_counts * abs_means) / total, 0.0, 1.0))
    global_second = float(np.clip(np.sum(positive_counts * second_means) / total, 0.0, 1.0))
    noisy_indicators = np.clip(noisy_indicators / total, 0.0, 1.0)
    entropy_terms = fractions[fractions > 0.0]
    entropy = -float(np.sum(entropy_terms * np.log(entropy_terms)))
    values = np.asarray([
        entropy / log(n_classes) if n_classes > 1 else 0.0,
        float(np.max(fractions)), float(np.min(fractions)),
        float(np.std(positive_counts) / max(np.mean(positive_counts), 1e-12)),
        float(np.mean(positive_counts <= 0.0)), global_abs, global_second,
        float(np.mean(np.abs(abs_means - global_abs))),
        float(np.mean(np.abs(second_means - global_second))),
        float(noisy_indicators[0]), float(noisy_indicators[1]),
    ], dtype=float)
    return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0), ledger.verify()

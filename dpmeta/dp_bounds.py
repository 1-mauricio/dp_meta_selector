"""Differentially private numeric bounds from public coarse intervals.

The mechanism releases a fixed-bin histogram for each numeric attribute. A
record replacement changes the histogram vector by L1 at most two, so a
coordinate-wise Laplace vector release with scale ``2 / epsilon`` is pure
epsilon-DP. The per-attribute releases compose sequentially.

This module intentionally requires finite coarse bounds supplied by a public
contract. It never derives those coarse bounds from the input values.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .privacy import PrivacyLedger, Release, Sensitivity
from .randomness import make_rng


HISTOGRAM_SENSITIVITY = Sensitivity(
    l1=2.0,
    l2=float(np.sqrt(2.0)),
    linf=1.0,
)


@dataclass(frozen=True)
class DPBoundsResult:
    """Estimated quantile bounds and the ledger for all releases."""

    bounds: dict[str, tuple[float, float]]
    ledger: PrivacyLedger


def _validate_interval(lower: float, upper: float) -> tuple[float, float]:
    lower, upper = float(lower), float(upper)
    if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
        raise ValueError("bounds coarse públicos precisam ser finitos e crescentes")
    return lower, upper


def _quantile_from_noisy_histogram(noisy_counts: np.ndarray, edges: np.ndarray,
                                   quantile: float) -> float:
    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile precisa estar estritamente entre zero e um")
    counts = np.maximum(np.asarray(noisy_counts, dtype=float), 0.0)
    total = float(counts.sum())
    if not np.isfinite(total) or total <= 0.0:
        return float((edges[0] + edges[-1]) / 2.0)
    target = quantile * total
    cumulative = np.cumsum(counts)
    index = int(np.searchsorted(cumulative, target, side="left"))
    index = min(index, len(counts) - 1)
    previous = float(cumulative[index - 1]) if index else 0.0
    mass = float(counts[index])
    fraction = 0.5 if mass <= 0.0 else (target - previous) / mass
    fraction = float(np.clip(fraction, 0.0, 1.0))
    return float(edges[index] + fraction * (edges[index + 1] - edges[index]))


def estimate_dp_numeric_bounds(
    values,
    *,
    coarse_lower: float,
    coarse_upper: float,
    epsilon: float,
    quantiles: tuple[float, float] = (0.01, 0.99),
    bins: int = 32,
    rng: np.random.Generator | None = None,
) -> tuple[tuple[float, float], float]:
    """Estimate two quantile bounds from one numeric column.

    Returns ``((lower_quantile, upper_quantile), laplace_scale)``. Missing and
    non-finite values are omitted from the histogram. The interval used for
    clipping is public and supplied by the caller.
    """
    coarse_lower, coarse_upper = _validate_interval(coarse_lower, coarse_upper)
    if epsilon <= 0.0 or not np.isfinite(epsilon):
        raise ValueError("epsilon precisa ser positivo e finito")
    if len(quantiles) != 2 or not 0.0 < quantiles[0] < quantiles[1] < 1.0:
        raise ValueError("quantiles precisam satisfazer 0 < inferior < superior < 1")
    if bins < 2 or bins != int(bins):
        raise ValueError("bins precisa ser um inteiro >= 2")
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float, na_value=np.nan)
    finite = numeric[np.isfinite(numeric)]
    clipped = np.clip(finite, coarse_lower, coarse_upper)
    edges = np.linspace(coarse_lower, coarse_upper, int(bins) + 1)
    counts, _ = np.histogram(clipped, bins=edges)
    scale = 2.0 / float(epsilon)
    generator = make_rng(None) if rng is None else rng
    noisy = counts.astype(float) + generator.laplace(0.0, scale, size=int(bins))
    estimated = tuple(
        float(np.clip(_quantile_from_noisy_histogram(noisy, edges, quantile),
                      coarse_lower, coarse_upper))
        for quantile in quantiles
    )
    if estimated[0] >= estimated[1]:
        # The public interval remains a valid fallback when noise reverses the
        # two estimates. This is post-processing and spends no extra budget.
        estimated = (coarse_lower, coarse_upper)
    return estimated, scale


def estimate_dp_bounds_frame(
    frame: pd.DataFrame,
    coarse_bounds: dict[str, tuple[float, float]],
    *,
    epsilon: float,
    quantiles: tuple[float, float] = (0.01, 0.99),
    bins: int = 32,
    seed: int | None = None,
) -> DPBoundsResult:
    """Estimate bounds for all named columns with sequential composition."""
    if not coarse_bounds:
        raise ValueError("ao menos um atributo numérico é necessário")
    missing = set(coarse_bounds) - set(frame.columns)
    if missing:
        raise ValueError(f"atributos ausentes no frame: {sorted(missing)}")
    if epsilon <= 0.0 or not np.isfinite(epsilon):
        raise ValueError("epsilon precisa ser positivo e finito")
    per_feature_epsilon = float(epsilon) / len(coarse_bounds)
    generator = make_rng(seed)
    ledger = PrivacyLedger(float(epsilon), 0.0)
    bounds: dict[str, tuple[float, float]] = {}
    for name, (lower, upper) in coarse_bounds.items():
        estimated, scale = estimate_dp_numeric_bounds(
            frame[name], coarse_lower=lower, coarse_upper=upper,
            epsilon=per_feature_epsilon, quantiles=quantiles, bins=bins,
            rng=generator,
        )
        bounds[str(name)] = estimated
        ledger.record(Release(
            name=f"numeric_bounds:{name}",
            mechanism="laplace_fixed_bin_histogram",
            epsilon=per_feature_epsilon,
            delta=0.0,
            sensitivity=HISTOGRAM_SENSITIVITY,
            noise_scale=scale,
        ))
    return DPBoundsResult(bounds=bounds, ledger=ledger.verify())

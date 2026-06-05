"""Calibração adaptativa de epsilon por família de mecanismo."""

import logging
from typing import Dict

import numpy as np

from .config import TARGET_NOISE_RATIO

_log = logging.getLogger(__name__)

DELTA_DEFAULT: float = 1e-5  # mantido por retrocompatibilidade de import


def calibrate_epsilon(family: str) -> float:
    """
    Retorna epsilon tal que o ruído introduzido seja ~30% do sinal,
    comparável entre famílias.
    """
    if family == "continuous":
        return float(np.sqrt(2) / (TARGET_NOISE_RATIO * 0.5))

    if family == "discrete":
        return float(1.0 / (TARGET_NOISE_RATIO * 0.5 * 100))

    return 3.0


FAMILY_EPSILON: Dict[str, float] = {
    fam: calibrate_epsilon(fam) for fam in ("continuous", "discrete", "categorical")
}

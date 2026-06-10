"""Calibração adaptativa de epsilon por família de mecanismo."""

import logging
from typing import Dict

import numpy as np

from .config import TARGET_NOISE_RATIO

_log = logging.getLogger(__name__)

DELTA_DEFAULT: float = 1e-5  # mantido por retrocompatibilidade de import

# ---------------------------------------------------------------------------
# Epsilons de TREINAMENTO — calibrados para que cada mecanismo produza o MESMO
# nível de ruído efetivo no espaço normalizado [0, 1].
#
# Alvo: ~20% de ruído relativo ao range da coluna para todos os mecanismos.
#
# Laplace em [0,1]:     scale = 1/ε → ε = 1/0.20 = 5.0
# Geometric em [0,100]: E[|g1-g2|]/100 = 0.20 → empiricamente ε ≈ 0.04
#   (fórmula 2*(1-p)/p subestima o ruído real; medição empírica usada)
# Exponential:          ε = 2.0 (razoável para categorical)
#
# Isso cria um sinal discriminativo: ao mesmo nível de ruído absoluto, Geometric
# preserva estrutura inteira (caudas mais finas), enquanto Laplace espalha
# continuamente. A diferença é detectável via histogram_score e f1_macro com RF.
# ---------------------------------------------------------------------------
COMPARISON_EPSILON: float = 3.0  # referência (usado na chave de cache — bump invalida cache antigo)

FAMILY_EPSILON: Dict[str, float] = {
    "continuous": 5.0,    # Laplace: noise ≈ 19.5% do range [0,1]
    "discrete":   0.04,   # Geometric: noise ≈ 20% (equivalente ao Laplace)
    "categorical": 2.0,   # Exponential: orçamento razoável para categórico
}

# ---------------------------------------------------------------------------
# Calibração POR MECANISMO — sobrescreve FAMILY_EPSILON quando especificado.
#
# GaussianAnalytic/Gaussian usam sigma = sqrt(2*ln(1.25/δ))/ε.
# Para δ=1e-5 e E[|noise|]=0.20:
#   sigma_target = 0.20 * sqrt(π/2) ≈ 0.2506  (E[|N(0,σ)|] = σ*sqrt(2/π))
#   ε = sqrt(2*ln(1.25/δ)) / sigma_target ≈ 19.34
#
# Uniform: half-width = 1/(2ε); E[|noise|] = half/2 = 1/(4ε) = 0.20 → ε = 1.25
# ---------------------------------------------------------------------------
_sigma_target: float = 0.20 * np.sqrt(np.pi / 2.0)      # ≈ 0.2506
_GAUSSIAN_EPS: float = round(
    np.sqrt(2.0 * np.log(1.25 / DELTA_DEFAULT)) / _sigma_target, 2
)  # ≈ 19.34 → E[|noise|] ≈ 20%

MECHANISM_EPSILON: Dict[str, float] = {
    "GaussianAnalytic": _GAUSSIAN_EPS,  # ≈ 19.34 → E[|noise|] ≈ 20%
    "Gaussian":         _GAUSSIAN_EPS,  # mesma fórmula
    "Uniform":          1.25,           # half = 1/(2*1.25) = 0.4; E[|noise|] = 0.4/2 = 0.2
}


def calibrate_epsilon(mechanism_or_family: str) -> float:
    """Retorna epsilon calibrado para ~20% de ruído efetivo.

    Verifica MECHANISM_EPSILON primeiro (por mecanismo), depois FAMILY_EPSILON (por família).
    """
    if mechanism_or_family in MECHANISM_EPSILON:
        return MECHANISM_EPSILON[mechanism_or_family]
    return FAMILY_EPSILON.get(mechanism_or_family, 1.0)

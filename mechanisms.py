"""Registro de mecanismos DP suportados."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from .config import DEFAULT_CACHE_DIR

_log = logging.getLogger(__name__)

from diffprivlib.mechanisms import (
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


@dataclass
class DPMechanism:
    name: str
    description: str
    family: str  # 'continuous' | 'discrete' | 'categorical'


DP_MECHANISMS = [
    DPMechanism("Laplace", "Laplace clássico", "continuous"),
    DPMechanism("Gaussian", "Gaussiano (ε,δ)", "continuous"),
    DPMechanism("GaussianAnalytic", "Gaussiano analítico Balle–Wang", "continuous"),
    DPMechanism("Staircase", "Staircase (mistura geométrica)", "continuous"),
    DPMechanism("LaplaceTruncated", "Laplace truncado [0,1]", "continuous"),
    DPMechanism("LaplaceFolded", "Laplace folded [0,1]", "continuous"),
    DPMechanism("Snapping", "Snapping Mironov", "continuous"),
    DPMechanism("Exponential", "Exponencial (candidatos)", "categorical"),
    DPMechanism("Uniform", "Uniform δ-DP", "continuous"),
]

MECHANISM_NAMES = [m.name for m in DP_MECHANISMS]
FAMILY_OF: Dict[str, str] = {m.name: m.family for m in DP_MECHANISMS}

# Um representante por família para screening rápido (fase 1).
SCREENING_MECHANISMS = ["Laplace", "GaussianAnalytic", "Exponential"]

# Re-exporta classes diffprivlib usadas pelo aplicador.
__all__ = [
    "DPMechanism",
    "DP_MECHANISMS",
    "MECHANISM_NAMES",
    "FAMILY_OF",
    "SCREENING_MECHANISMS",
    "DEFAULT_CACHE_DIR",
    "Laplace",
    "LaplaceTruncated",
    "LaplaceFolded",
    "Gaussian",
    "GaussianAnalytic",
    "Geometric",
    "GeometricTruncated",
    "GeometricFolded",
    "Exponential",
    "Staircase",
    "Uniform",
    "Snapping",
]

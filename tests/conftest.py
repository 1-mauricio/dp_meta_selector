"""Fixtures compartilhados para os testes do dp_meta_selector."""

import sys
from pathlib import Path

import numpy as np
import pytest

# garante que o pacote raiz é importável
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def rng():
    return np.random.RandomState(0)


@pytest.fixture
def small_continuous(rng):
    """Dataset numérico pequeno (100×4), 2 classes."""
    X = rng.randn(100, 4)
    y = (X[:, 0] > 0).astype(int)
    return X, y


@pytest.fixture
def small_categorical(rng):
    """Dataset com colunas inteiras (0-4), 3 classes."""
    X = rng.randint(0, 5, size=(80, 3)).astype(float)
    y = rng.randint(0, 3, size=80)
    return X, y


@pytest.fixture
def tiny(rng):
    """Dataset mínimo para smoke tests (20×2)."""
    X = rng.randn(20, 2)
    y = (X[:, 0] > 0).astype(int)
    return X, y

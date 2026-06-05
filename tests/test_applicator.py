"""Testes para DPApplicator — aplicação de mecanismos DP."""

import numpy as np
import pytest

from dp_meta_selector.applicator import DPApplicator
from dp_meta_selector.mechanisms import MECHANISM_NAMES


@pytest.fixture
def applicator():
    return DPApplicator(delta=1e-5)


@pytest.fixture
def X_2d():
    """Dataset 2D (50×1) para testar apply por coluna."""
    rng = np.random.RandomState(99)
    return rng.randn(50, 4).astype(float)


def test_all_mechanisms_produce_same_shape(applicator, X_2d):
    """Cada mecanismo deve processar X sem erro e preservar o shape."""
    for name in MECHANISM_NAMES:
        out = applicator.apply(name, X_2d)
        assert out.shape == X_2d.shape, f"{name}: shape mismatch"


def test_laplace_output_shape(applicator, X_2d):
    out = applicator.apply("Laplace", X_2d)
    assert out.shape == X_2d.shape


def test_gaussian_output_shape(applicator, X_2d):
    out = applicator.apply("Gaussian", X_2d)
    assert out.shape == X_2d.shape


def test_geometric_integer_output(applicator):
    X = np.arange(1, 26, dtype=float).reshape(5, 5)
    out = applicator.apply("Geometric", X)
    assert np.issubdtype(out.dtype, np.number)
    assert out.shape == X.shape


def test_unknown_mechanism_raises(applicator, X_2d):
    with pytest.raises((ValueError, KeyError)):
        applicator.apply("NonExistentMechanism", X_2d)


def test_no_silent_return_original(applicator, X_2d):
    """Com ruído contínuo, Laplace deve produzir valores distintos dos originais."""
    out = applicator.apply("Laplace", X_2d)
    assert not np.allclose(out, X_2d, atol=1e-10), (
        "Saída idêntica à entrada: suspeito de retorno silencioso"
    )


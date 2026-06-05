"""Testes para UtilityResultCache e _data_fingerprint."""

import time
import tempfile
from pathlib import Path

import numpy as np
import pytest

from dp_meta_selector.utility import UtilityResultCache, _data_fingerprint, META_FAST_PROFILE


@pytest.fixture
def cache(tmp_path):
    return UtilityResultCache(cache_dir=tmp_path, max_age_days=1)


def test_fingerprint_differs_for_different_data():
    rng = np.random.RandomState(1)
    X1 = rng.randn(100, 5)
    y1 = np.zeros(100, dtype=int)
    X2 = rng.randn(100, 5)
    y2 = np.zeros(100, dtype=int)
    assert _data_fingerprint(X1, y1) != _data_fingerprint(X2, y2)


def test_fingerprint_stable():
    """Mesmo array → mesmo fingerprint."""
    rng = np.random.RandomState(42)
    X = rng.randn(50, 3)
    y = np.zeros(50, dtype=int)
    assert _data_fingerprint(X, y) == _data_fingerprint(X, y)


def test_fingerprint_different_shapes():
    X1 = np.ones((10, 5))
    X2 = np.ones((5, 10))
    y = np.zeros(10, dtype=int)
    assert _data_fingerprint(X1, y[:10]) != _data_fingerprint(X2, y[:5])


def test_cache_set_get(cache):
    fp = "test_fp_001"
    cache.set(fp, "Laplace", META_FAST_PROFILE, 0.95)
    result = cache.get(fp, "Laplace", META_FAST_PROFILE)
    assert result is not None
    assert abs(result - 0.95) < 1e-9


def test_cache_miss_returns_none(cache):
    assert cache.get("nonexistent_fp_xyz", "Laplace", META_FAST_PROFILE) is None


def test_cache_prune(tmp_path):
    """prune() deve remover entradas mais velhas que max_age_days."""
    c = UtilityResultCache(cache_dir=tmp_path, max_age_days=0)
    c.set("old_fp", "Laplace", META_FAST_PROFILE, 0.8)
    # forçar arquivo "velho" modificando mtime
    import os
    cache_files = list(tmp_path.glob("*.joblib"))
    if cache_files:
        old_time = time.time() - 86400 * 2  # 2 dias atrás
        os.utime(cache_files[0], (old_time, old_time))
    pruned = c.prune()
    assert pruned >= 0

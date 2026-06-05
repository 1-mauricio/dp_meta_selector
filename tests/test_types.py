"""Testes para dp_meta_selector.types — Dataset dataclass."""

import numpy as np
import pytest

from dp_meta_selector.types import Dataset


def test_dataset_basic_attrs():
    X = np.zeros((30, 5))
    y = np.zeros(30, dtype=int)
    ds = Dataset(X=X, y=y, name="test")
    assert ds.n_samples == 30
    assert ds.n_features == 5
    assert ds.n_classes == 1  # só 0


def test_dataset_n_classes():
    X = np.zeros((10, 2))
    y = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
    ds = Dataset(X=X, y=y, name="multiclass")
    assert ds.n_classes == 3


def test_dataset_backward_compat_unpack():
    """Dataset deve ser desempacotável como (X, y, name) — retrocompat."""
    X = np.ones((5, 2))
    y = np.zeros(5, dtype=int)
    ds = Dataset(X=X, y=y, name="compat")
    X2, y2, name = ds
    np.testing.assert_array_equal(X, X2)
    np.testing.assert_array_equal(y, y2)
    assert name == "compat"


def test_dataset_getitem():
    X = np.ones((5, 2))
    y = np.zeros(5, dtype=int)
    ds = Dataset(X=X, y=y, name="idx")
    assert ds[2] == "idx"


def test_dataset_len():
    X = np.zeros((12, 3))
    y = np.zeros(12, dtype=int)
    ds = Dataset(X=X, y=y, name="lentest")
    assert len(ds) == 3  # len == n_features (tuple-compat = 3 elements)


def test_dataset_shape_mismatch_raises():
    X = np.zeros((10, 3))
    y = np.zeros(5, dtype=int)
    with pytest.raises((ValueError, AssertionError)):
        Dataset(X=X, y=y, name="bad")


def test_dataset_empty_name_raises():
    X = np.zeros((5, 2))
    y = np.zeros(5, dtype=int)
    with pytest.raises((ValueError, AssertionError)):
        Dataset(X=X, y=y, name="")

"""Testes para MetaFeatureExtractor."""

import numpy as np
import pytest

from dp_meta_selector.meta_features import MetaFeatureExtractor

EXPECTED_MIN_FEATURES = 30  # aumentou de 27 → 32 com ML4


@pytest.fixture
def extractor():
    return MetaFeatureExtractor()


def test_feature_count(extractor, small_continuous):
    X, y = small_continuous
    feats = extractor.extract(X, y)
    assert len(feats) >= EXPECTED_MIN_FEATURES, (
        f"Esperado >= {EXPECTED_MIN_FEATURES} features, obtido {len(feats)}"
    )


def test_feature_names_consistent(extractor, small_continuous, small_categorical):
    X1, y1 = small_continuous
    X2, y2 = small_categorical
    names1 = list(extractor.extract(X1, y1).keys())
    names2 = list(extractor.extract(X2, y2).keys())
    assert names1 == names2, "Nomes das features devem ser consistentes entre datasets"


def test_dp_relevance_features_present(extractor, small_continuous):
    """Features de DP-relevância (ML4) devem estar presentes."""
    X, y = small_continuous
    feats = extractor.extract(X, y)
    dp_keys = {"mean_sensitivity", "max_sensitivity", "outlier_ratio"}
    for key in dp_keys:
        assert key in feats, f"Feature DP '{key}' ausente"


def test_no_nan_or_inf(extractor, small_continuous):
    X, y = small_continuous
    feats = extractor.extract(X, y)
    for k, v in feats.items():
        assert np.isfinite(v), f"Feature '{k}' = {v} não é finita"


def test_fast_landmarks_feature_names_consistent(extractor, small_continuous):
    """Dois calls a extract() devem produzir as mesmas keys."""
    X, y = small_continuous
    feats1 = extractor.extract(X, y)
    feats2 = extractor.extract(X, y)
    assert set(feats1.keys()) == set(feats2.keys())

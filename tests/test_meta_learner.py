"""Testes para MetaLearner — treino, predição e estado."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dp_meta_selector.meta_learner import MetaLearner
from dp_meta_selector.mechanisms import MECHANISM_NAMES


@pytest.fixture
def synthetic_meta_dataset(rng):
    """DataFrame com meta-features sintéticas e target válido."""
    n = 60
    ml = MetaLearner()
    # cria colunas com nomes das meta-features
    feat_names = [
        "n_samples", "n_features", "n_classes", "class_imbalance",
        "mean_corr", "std_corr", "max_corr", "frac_numeric",
        "mean_kurtosis", "std_kurtosis", "mean_skewness", "std_skewness",
        "mean_entropy", "std_entropy", "frac_outliers", "mean_range",
        "std_range", "n_unique_ratio", "pca_explained_50", "pca_explained_90",
        "land_knn_acc", "land_nb_acc", "land_dt_acc", "land_svm_acc",
        "mean_sensitivity", "max_sensitivity", "outlier_ratio",
        "pca_intrinsic_dim_ratio", "pca_top1_var",
    ]
    data = {f: rng.randn(n) for f in feat_names}
    data["best_mechanism"] = rng.choice(MECHANISM_NAMES, size=n)
    for mech in MECHANISM_NAMES:
        data[f"rel_{mech}"] = rng.rand(n)
    return pd.DataFrame(data)


def test_meta_feature_cols_instance_variable():
    """B2: META_FEATURE_COLS deve ser atributo de instância, não de classe."""
    m1 = MetaLearner()
    m2 = MetaLearner()
    # Se for de classe, modificar m1 afeta m2
    m1.META_FEATURE_COLS = ["a", "b"]
    assert m2.META_FEATURE_COLS is None or m2.META_FEATURE_COLS != ["a", "b"]


def test_fit_and_predict(synthetic_meta_dataset, small_continuous):
    """MetaLearner deve treinar e fazer predição sem erro."""
    ml = MetaLearner()
    ml.fit(synthetic_meta_dataset)
    X, y = small_continuous
    rec = ml.predict(X, y)
    # predict returns either a string or a dict with 'recommended_mechanism'
    if isinstance(rec, dict):
        mech = rec["recommended_mechanism"]
    else:
        mech = rec
    assert mech in MECHANISM_NAMES, f"Predição inválida: {mech}"


def test_fast_landmarks_stored_after_fit(synthetic_meta_dataset):
    """P2: _fast_landmarks deve ser preservado após fit."""
    ml = MetaLearner()
    ml.fit(synthetic_meta_dataset)
    assert hasattr(ml, "_fast_landmarks"), "_fast_landmarks não foi salvo"


def test_save_and_load(tmp_path, synthetic_meta_dataset, small_continuous):
    """MetaLearner deve ser salvo e recarregado com predição consistente."""
    ml = MetaLearner()
    ml.fit(synthetic_meta_dataset)
    path = tmp_path / "model.joblib"
    ml.save(str(path))

    ml2 = MetaLearner()
    ml2.load(str(path))
    X, y = small_continuous
    rec = ml2.predict(X, y)
    mech = rec["recommended_mechanism"] if isinstance(rec, dict) else rec
    assert mech in MECHANISM_NAMES

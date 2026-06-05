"""Carregamento de datasets OpenML e utilitários de split."""

import logging
from functools import lru_cache
from typing import Dict, List, Optional, Union

import numpy as np
import openml
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

from .config import MAX_ROWS_PER_DATASET, OPENML_CC18_SUITE_ID, OPENML_TRAINING_TARGET
from .types import Dataset, DatasetTuple

_log = logging.getLogger(__name__)

OPENML_TRAINING_SPECS_CORE = [
    {"data_id": 61, "label": "iris"},
    {"data_id": 31, "label": "credit-g"},
    {"data_id": 37, "label": "diabetes"},
    {"data_id": 44, "label": "spambase"},
    {"data_id": 54, "label": "vehicle"},
    {"data_id": 1464, "label": "blood-transfusion"},
    {"data_id": 1489, "label": "phoneme"},
    {"data_id": 1494, "label": "qsar-biodeg"},
    {"data_id": 1067, "label": "hill-valley"},
    {"data_id": 4534, "label": "australian"},
    {"data_id": 4538, "label": "heart-statlog"},
    {"data_id": 846, "label": "cardiotocography"},
    {"data_id": 12, "label": "mfeat-factors"},
    {"data_id": 14, "label": "mfeat-fourier"},
    {"data_id": 16, "label": "mfeat-karhunen"},
    {"data_id": 18, "label": "mfeat-zernike"},
    {"data_id": 59, "label": "chip"},
    {"data_id": 40701, "label": "churn"},
    {"data_id": 3, "label": "kr-vs-kp"},
    {"data_id": 40975, "label": "run-or-walk"},
]

OPENML_TRAINING_SPECS: List[Dict] = []  # mantido para retrocompatibilidade de API


@lru_cache(maxsize=1)
def _get_default_training_specs() -> tuple:
    """Lazy-build e cache das specs padrão. Retorna tuple para hashability."""
    return tuple(build_openml_training_specs())


def build_openml_training_specs(
    target_n: int = OPENML_TRAINING_TARGET,
    core: Optional[List[Dict]] = None,
) -> List[Dict]:
    """
    Monta até ``target_n`` specs: núcleo curado, suite OpenML-CC18 e classificação tabular extra.
    """
    specs: List[Dict] = list(core or OPENML_TRAINING_SPECS_CORE)
    seen_ids = {s["data_id"] for s in specs}
    seen_names = {s["label"].lower() for s in specs}

    def _append(did: int, label: str) -> None:
        nonlocal specs
        if len(specs) >= target_n:
            return
        nm = label.lower()
        if did in seen_ids or nm in seen_names:
            return
        specs.append({"data_id": did, "label": label})
        seen_ids.add(did)
        seen_names.add(nm)

    dfs = None
    try:
        cc18_ids = openml.study.get_suite(OPENML_CC18_SUITE_ID).data
        dfs = openml.datasets.list_datasets(output_format="dataframe", status="active")
        for did in cc18_ids:
            if len(specs) >= target_n:
                break
            did = int(did)
            row = dfs[dfs["did"] == did]
            label = str(row.iloc[0]["name"]) if len(row) else str(did)
            _append(did, label)
    except Exception as exc:
        _log.warning("OpenML-CC18 indisponível (%s); usando apenas núcleo + busca extra.", exc)

    if len(specs) < target_n:
        if dfs is None:
            dfs = openml.datasets.list_datasets(output_format="dataframe", status="active")
        pool = dfs[
            (dfs["NumberOfClasses"].fillna(0) >= 2)
            & (dfs["NumberOfClasses"].fillna(0) <= 30)
            & (dfs["NumberOfInstances"].fillna(0) >= 200)
            & (dfs["NumberOfInstances"].fillna(0) <= 50000)
            & (dfs["NumberOfFeatures"].fillna(0) >= 3)
            & (dfs["NumberOfFeatures"].fillna(0) <= 150)
            & (~dfs["did"].isin(seen_ids))
        ].sort_values("NumberOfInstances")
        skip_substr = ("fri_c", "arcene_seed", "analcatdata", "dataset_analcat")
        for _, row in pool.iterrows():
            if len(specs) >= target_n:
                break
            label = str(row["name"])
            if any(s in label.lower() for s in skip_substr):
                continue
            _append(int(row["did"]), label)

    return specs[:target_n]


def _openml_X_to_float(X) -> np.ndarray:
    if isinstance(X, pd.DataFrame):
        parts = []
        for c in X.columns:
            s = X[c]
            if pd.api.types.is_numeric_dtype(s):
                col = pd.to_numeric(s, errors="coerce")
            else:
                col = pd.factorize(s.astype(str))[0].astype(float)
            parts.append(np.asarray(col, dtype=float).reshape(-1, 1))
        return np.hstack(parts)
    X = np.asarray(X)
    if X.dtype.kind in "UO" or X.dtype == object:
        rows = []
        for j in range(X.shape[1]):
            col = X[:, j]
            if np.issubdtype(col.dtype, np.number):
                rows.append(np.asarray(col, dtype=float).reshape(-1, 1))
            else:
                _, codes = np.unique(col.astype(str), return_inverse=True)
                rows.append(codes.astype(float).reshape(-1, 1))
        return np.hstack(rows)
    return np.asarray(X, dtype=float)


def load_openml_dataset(
    data_id: int,
    label: str = "",
    max_rows: int = MAX_ROWS_PER_DATASET,
    seed: int = 42,
) -> Optional[Dataset]:
    try:
        ds = openml.datasets.get_dataset(data_id, download_data=True)
        X, y, _, _ = ds.get_data(
            target=ds.default_target_attribute,
            dataset_format="dataframe",
        )
        X = _openml_X_to_float(X)
        y = np.asarray(y)
        nm = label or ds.name

        if y.dtype.kind in "UO" or y.dtype == object:
            y = LabelEncoder().fit_transform(y.astype(str))
        else:
            y = LabelEncoder().fit_transform(y)

        if np.isnan(X).any() or np.isinf(X).any():
            X = SimpleImputer(strategy="median").fit_transform(X)

        name = f"openml:{nm}"
        if len(y) > max_rows:
            rng = np.random.RandomState(seed)
            idx = rng.choice(len(y), max_rows, replace=False)
            X, y = X[idx], y[idx]
            name = f"{name}[sub={max_rows}]"

        return Dataset(X=X.astype(float), y=y.astype(int), name=name)
    except Exception as exc:
        _log.warning("Pulando OpenML %d (%s): %s", data_id, label or data_id, exc)
        return None


def load_openml_training_datasets(
    specs: Optional[List[Dict]] = None,
    max_rows: int = MAX_ROWS_PER_DATASET,
    seed: int = 42,
) -> List[Dataset]:
    # Q1: sem mutação de global; usa lru_cache para default specs
    if specs is None:
        specs = list(_get_default_training_specs())
    datasets: List[Dataset] = []
    for spec in specs:
        got = load_openml_dataset(
            data_id=spec["data_id"],
            label=spec.get("label", ""),
            max_rows=max_rows,
            seed=seed,
        )
        if got is not None:
            datasets.append(got)

    if not datasets:
        raise RuntimeError(
            "Nenhum dataset OpenML carregado. Verifique rede, cache OpenML "
            "e os IDs em OPENML_TRAINING_SPECS."
        )
    _log.info("OpenML: %d/%d datasets carregados.", len(datasets), len(specs))
    return datasets


def split_meta_datasets(
    datasets: List[Union[Dataset, DatasetTuple]],
    test_ratio: float = 0.3,
    seed: int = 42,
):
    rng = np.random.RandomState(seed)
    idx = np.arange(len(datasets))
    rng.shuffle(idx)

    split = int(len(datasets) * (1 - test_ratio))
    train_idx, test_idx = idx[:split], idx[split:]

    train = [datasets[i] for i in train_idx]
    test = [datasets[i] for i in test_idx]

    return train, test


OPENML_TRAINING_SPECS: List[Dict] = []  # mantido para retrocompatibilidade de API


@lru_cache(maxsize=1)
def _get_default_training_specs() -> tuple:
    """Lazy-build e cache das specs padrão. Retorna tuple para hashability."""
    return tuple(build_openml_training_specs())


def build_openml_training_specs(
    target_n: int = OPENML_TRAINING_TARGET,
    core: Optional[List[Dict]] = None,
) -> List[Dict]:
    """
    Monta até ``target_n`` specs: núcleo curado, suite OpenML-CC18 e classificação tabular extra.
    """
    specs: List[Dict] = list(core or OPENML_TRAINING_SPECS_CORE)
    seen_ids = {s["data_id"] for s in specs}
    seen_names = {s["label"].lower() for s in specs}

    def _append(did: int, label: str) -> None:
        nonlocal specs
        if len(specs) >= target_n:
            return
        nm = label.lower()
        if did in seen_ids or nm in seen_names:
            return
        specs.append({"data_id": did, "label": label})
        seen_ids.add(did)
        seen_names.add(nm)

    dfs = None
    try:
        cc18_ids = openml.study.get_suite(OPENML_CC18_SUITE_ID).data
        dfs = openml.datasets.list_datasets(output_format="dataframe", status="active")
        for did in cc18_ids:
            if len(specs) >= target_n:
                break
            did = int(did)
            row = dfs[dfs["did"] == did]
            label = str(row.iloc[0]["name"]) if len(row) else str(did)
            _append(did, label)
    except Exception as exc:
        print(f"  [aviso] OpenML-CC18 indisponível ({exc}); usando apenas núcleo + busca extra.")

    if len(specs) < target_n:
        if dfs is None:
            dfs = openml.datasets.list_datasets(output_format="dataframe", status="active")
        pool = dfs[
            (dfs["NumberOfClasses"].fillna(0) >= 2)
            & (dfs["NumberOfClasses"].fillna(0) <= 30)
            & (dfs["NumberOfInstances"].fillna(0) >= 200)
            & (dfs["NumberOfInstances"].fillna(0) <= 50000)
            & (dfs["NumberOfFeatures"].fillna(0) >= 3)
            & (dfs["NumberOfFeatures"].fillna(0) <= 150)
            & (~dfs["did"].isin(seen_ids))
        ].sort_values("NumberOfInstances")
        skip_substr = ("fri_c", "arcene_seed", "analcatdata", "dataset_analcat")
        for _, row in pool.iterrows():
            if len(specs) >= target_n:
                break
            label = str(row["name"])
            if any(s in label.lower() for s in skip_substr):
                continue
            _append(int(row["did"]), label)

    return specs[:target_n]


def _openml_X_to_float(X) -> np.ndarray:
    if isinstance(X, pd.DataFrame):
        parts = []
        for c in X.columns:
            s = X[c]
            if pd.api.types.is_numeric_dtype(s):
                col = pd.to_numeric(s, errors="coerce")
            else:
                col = pd.factorize(s.astype(str))[0].astype(float)
            parts.append(np.asarray(col, dtype=float).reshape(-1, 1))
        return np.hstack(parts)
    X = np.asarray(X)
    if X.dtype.kind in "UO" or X.dtype == object:
        rows = []
        for j in range(X.shape[1]):
            col = X[:, j]
            if np.issubdtype(col.dtype, np.number):
                rows.append(np.asarray(col, dtype=float).reshape(-1, 1))
            else:
                _, codes = np.unique(col.astype(str), return_inverse=True)
                rows.append(codes.astype(float).reshape(-1, 1))
        return np.hstack(rows)
    return np.asarray(X, dtype=float)


def load_openml_dataset(
    data_id: int,
    label: str = "",
    max_rows: int = 3000,
    seed: int = 42,
) -> Optional[DatasetTuple]:
    try:
        ds = openml.datasets.get_dataset(data_id, download_data=True)
        X, y, _, _ = ds.get_data(
            target=ds.default_target_attribute,
            dataset_format="dataframe",
        )
        X = _openml_X_to_float(X)
        y = np.asarray(y)
        nm = label or ds.name

        if y.dtype.kind in "UO" or y.dtype == object:
            y = LabelEncoder().fit_transform(y.astype(str))
        else:
            y = LabelEncoder().fit_transform(y)

        if np.isnan(X).any() or np.isinf(X).any():
            X = SimpleImputer(strategy="median").fit_transform(X)

        name = f"openml:{nm}"
        if len(y) > max_rows:
            rng = np.random.RandomState(seed)
            idx = rng.choice(len(y), max_rows, replace=False)
            X, y = X[idx], y[idx]
            name = f"{name}[sub={max_rows}]"

        return X.astype(float), y.astype(int), name
    except Exception as exc:
        print(f"  [skip] OpenML {data_id} ({label or data_id}): {exc}")
        return None


def load_openml_training_datasets(
    specs: Optional[List[Dict]] = None,
    max_rows: int = 3000,
    seed: int = 42,
) -> List[DatasetTuple]:
    # Q1: sem mutação de global; usa lru_cache para default specs
    if specs is None:
        specs = list(_get_default_training_specs())
    datasets: List[DatasetTuple] = []
    for spec in specs:
        got = load_openml_dataset(
            data_id=spec["data_id"],
            label=spec.get("label", ""),
            max_rows=max_rows,
            seed=seed,
        )
        if got is not None:
            datasets.append(got)

    if not datasets:
        raise RuntimeError(
            "Nenhum dataset OpenML carregado. Verifique rede, cache OpenML "
            "e os IDs em OPENML_TRAINING_SPECS."
        )
    print(f"  OpenML: {len(datasets)}/{len(specs)} datasets carregados.")
    return datasets


def split_meta_datasets(datasets, test_ratio=0.3, seed=42):
    rng = np.random.RandomState(seed)
    idx = np.arange(len(datasets))
    rng.shuffle(idx)

    split = int(len(datasets) * (1 - test_ratio))
    train_idx, test_idx = idx[:split], idx[split:]

    train = [datasets[i] for i in train_idx]
    test = [datasets[i] for i in test_idx]

    return train, test

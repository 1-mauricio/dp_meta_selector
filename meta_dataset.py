"""Construção do meta-dataset para meta-aprendizagem."""

import logging
from typing import List, Optional

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from .calibration import DELTA_DEFAULT, FAMILY_EPSILON
from .mechanisms import DP_MECHANISMS, FAMILY_OF, MECHANISM_NAMES
from .meta_features import MetaFeatureExtractor
from .baseline_store import DEFAULT_BASELINE_REGISTRY, BaselineRegistry, BaselineStore
from .utility import (
    META_FAST_PROFILE,
    DPUtilityEvaluator,
    UtilityProfile,
    UtilityResultCache,
    _data_fingerprint,
)

_log = logging.getLogger(__name__)


class MetaDatasetBuilder:
    def __init__(
        self,
        delta: float = DELTA_DEFAULT,
        profile: UtilityProfile = META_FAST_PROFILE,
        cache: Optional[UtilityResultCache] = None,
        fast_landmarks: bool = True,
        n_runs: Optional[int] = None,
        baseline_store: Optional[BaselineStore] = None,
        baseline_registry: Optional[BaselineRegistry] = None,
        baseline_id: str = "meta_logreg",
        n_jobs: int = -1,  # PF1: paralelismo de datasets
    ):
        registry = baseline_registry or DEFAULT_BASELINE_REGISTRY
        self.extractor = MetaFeatureExtractor(fast_landmarks=fast_landmarks)
        self.evaluator = DPUtilityEvaluator(
            delta=delta,
            profile=profile,
            cache=cache,
            n_runs=n_runs,
            baseline_store=baseline_store,
            baseline_registry=registry,
            baseline_id=baseline_id,
        )
        self.n_jobs = n_jobs

    def _process_one(self, item) -> Optional[dict]:
        """PF1: processa um dataset — chamado em paralelo por build()."""
        X, y, name = item
        y = LabelEncoder().fit_transform(y)
        meta = self.extractor.extract(X, y)
        meta["dataset_name"] = name
        # PF5: calcula fingerprint uma vez e repassa para baseline + evaluate_all
        fp = _data_fingerprint(X, y)
        meta["baseline_acc"] = self.evaluator.baseline(X, y, dataset_id=name, fp=fp)
        dp = self.evaluator.evaluate_all(X, y)
        for k, v in dp.items():
            meta[f"acc_{k}"] = v
        base = meta["baseline_acc"] + 1e-9
        rel = {m: dp[m] / base for m in dp}
        best_rel = max(rel.values())
        # ML1: tie-breaking determinístico pela ordem em MECHANISM_NAMES
        candidates = [m for m in MECHANISM_NAMES if rel.get(m, 0.0) >= best_rel - 1e-9]
        meta["best_mechanism"] = candidates[0] if candidates else max(rel, key=rel.get)
        meta["best_relative_acc"] = max(rel.values())
        return meta

    def build(self, datasets) -> pd.DataFrame:
        _log.info(
            "[meta-build] perfil=%s screening=%s clf=%s cv=%d runs=%d",
            self.evaluator.profile.name,
            self.evaluator.profile.use_screening,
            self.evaluator.profile.clf,
            self.evaluator.profile.cv_splits,
            self.evaluator.profile.n_runs,
        )

        # PF1: converte para lista de tuplas (X, y, name) para serialização joblib
        items = [(ds.X, ds.y, ds.name) if hasattr(ds, "X") else ds for ds in datasets]
        n = len(items)

        effective_jobs = self.n_jobs
        # Com poucos datasets, o overhead de processos não compensa
        if n <= 4:
            effective_jobs = 1

        if effective_jobs == 1:
            rows = []
            for item in tqdm(items, desc="Construindo meta-dataset"):
                r = self._process_one(item)
                if r is not None:
                    rows.append(r)
        else:
            _log.info("[meta-build] paralelo: n_jobs=%s datasets=%d", effective_jobs, n)
            # prefer="threads": numpy/sklearn liberam o GIL → paralelismo real sem fork
            # (evita o problema de subprocessos sem acesso ao venv)
            results = Parallel(n_jobs=effective_jobs, prefer="threads", verbose=0)(
                delayed(self._process_one)(item)
                for item in tqdm(items, desc="Construindo meta-dataset")
            )
            rows = [r for r in results if r is not None]

        df = pd.DataFrame(rows)
        self._log_diagnostics(df)
        _log.info("[meta-build] %s", self.evaluator.cache.summary())
        if self.evaluator.baseline_store is not None:
            _log.info("[meta-build] %s", self.evaluator.baseline_store.summary())
        return df
        return df

    def _log_diagnostics(self, df):
        _log.info("[Meta-Dataset] Distribuição de melhores mecanismos:")
        vc = df["best_mechanism"].value_counts()
        for mech, cnt in vc.items():
            fam = FAMILY_OF.get(mech, "?")
            _log.info("   %-22s %2d  %s", mech, cnt, fam)

        _log.info("[Meta-Dataset] Acurácia média pós-DP por família:")
        for fam, eps in FAMILY_EPSILON.items():
            cols = [
                f"acc_{m.name}"
                for m in DP_MECHANISMS
                if m.family == fam and f"acc_{m.name}" in df.columns
            ]
            mean = df[cols].values.mean() if cols else float("nan")
            _log.info("   %-12s ε=%.3f  acurácia_média=%.4f", fam, eps, mean)

"""Construção do meta-dataset para meta-aprendizagem."""

import logging
from typing import List, Optional

import numpy as np
import pandas as pd
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

    def build(self, datasets) -> pd.DataFrame:
        rows = []
        _log.info(
            "[meta-build] perfil=%s screening=%s clf=%s cv=%d runs=%d",
            self.evaluator.profile.name,
            self.evaluator.profile.use_screening,
            self.evaluator.profile.clf,
            self.evaluator.profile.cv_splits,
            self.evaluator.profile.n_runs,
        )
        for X, y, name in tqdm(datasets, desc="Construindo meta-dataset"):
            y = LabelEncoder().fit_transform(y)
            meta = self.extractor.extract(X, y)
            meta["dataset_name"] = name
            meta["baseline_acc"] = self.evaluator.baseline(X, y, dataset_id=name)
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
            rows.append(meta)
        df = pd.DataFrame(rows)
        self._log_diagnostics(df)
        _log.info("[meta-build] %s", self.evaluator.cache.summary())
        if self.evaluator.baseline_store is not None:
            _log.info("[meta-build] %s", self.evaluator.baseline_store.summary())
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

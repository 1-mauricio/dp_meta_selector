"""Meta-modelos para prever o melhor mecanismo DP."""

import logging
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import (
    LeaveOneOut,
    StratifiedKFold,
    cross_val_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

from .mechanisms import MECHANISM_NAMES
from .meta_features import MetaFeatureExtractor

_log = logging.getLogger(__name__)


class MetaLearner:
    def __init__(self, fast_mode: bool = True, fast_landmarks: bool = True):
        self.META_FEATURE_COLS: Optional[List[str]] = None  # B2: instância, não classe
        self._fast_landmarks = fast_landmarks  # P2: armazena para uso em predict()
        if fast_mode:
            self.models = {
                "RandomForest": Pipeline([
                    ("s", StandardScaler()),
                    ("clf", RandomForestClassifier(n_estimators=150, random_state=42)),
                ]),
                "GradientBoosting": Pipeline([
                    ("s", StandardScaler()),
                    ("clf", GradientBoostingClassifier(n_estimators=60, random_state=42)),
                ]),
            }
        else:
            self.models = {
                "RandomForest": Pipeline([
                    ("s", StandardScaler()),
                    ("clf", RandomForestClassifier(n_estimators=300, random_state=42)),
                ]),
                "GradientBoosting": Pipeline([
                    ("s", StandardScaler()),
                    ("clf", GradientBoostingClassifier(n_estimators=100, random_state=42)),
                ]),
                "KNN": Pipeline([
                    ("s", StandardScaler()),
                    ("clf", KNeighborsClassifier(n_neighbors=3)),
                ]),
                "SVM": Pipeline([
                    ("s", StandardScaler()),
                    ("clf", SVC(kernel="rbf", probability=True, random_state=42)),
                ]),
            }
        self.best_model_name: Optional[str] = None
        self.label_encoder = LabelEncoder()

    def fit(self, meta_df: pd.DataFrame) -> Dict[str, float]:
        excl = (
            {"dataset_name", "best_mechanism", "best_relative_acc", "baseline_acc"}
            | {f"acc_{m}" for m in MECHANISM_NAMES}
        )
        self.META_FEATURE_COLS = [c for c in meta_df.columns if c not in excl]

        X_meta = np.nan_to_num(
            meta_df[self.META_FEATURE_COLS].values.astype(float),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        y_meta = self.label_encoder.fit_transform(meta_df["best_mechanism"])

        min_class = int(np.min(np.bincount(y_meta)))
        n_cls = len(np.unique(y_meta))

        if min_class < 2:
            cv, cv_name = LeaveOneOut(), "LeaveOneOut"
        else:
            k = min(5, min_class)
            cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
            cv_name = f"StratifiedKFold(k={k})"

        _log.info("[Meta-Modelos] CV=%s  n=%d  classes=%d", cv_name, len(meta_df), n_cls)
        _log.info(
            "  Distribuição das classes: %s",
            {k: int(v) for k, v in zip(self.label_encoder.classes_, np.bincount(y_meta))},
        )

        scores = {}
        for name, model in self.models.items():
            try:
                s = cross_val_score(model, X_meta, y_meta, cv=cv, scoring="f1_macro")
                scores[name] = float(s.mean())
            except Exception:
                scores[name] = float("nan")
            model.fit(X_meta, y_meta)

        # ML2: calibração de probabilidade após treino (Platt scaling)
        if len(X_meta) >= 10:
            for name, model in list(self.models.items()):
                try:
                    cal_method = "isotonic" if len(X_meta) >= 30 else "sigmoid"
                    cal = CalibratedClassifierCV(model, cv="prefit", method=cal_method)
                    cal.fit(X_meta, y_meta)
                    self.models[name] = cal
                except Exception:
                    pass  # mantém não calibrado se falhar

        valid = {k: v for k, v in scores.items() if not np.isnan(v)}
        self.best_model_name = (
            max(valid, key=valid.get) if valid else list(scores)[0]
        )
        return scores

    def predict(self, X, y, model_name=None) -> Dict:
        if self.META_FEATURE_COLS is None:
            raise RuntimeError("Chame fit() antes de predict().")
        y_enc = LabelEncoder().fit_transform(y)
        feats = MetaFeatureExtractor(fast_landmarks=self._fast_landmarks).extract(X, y_enc)  # P2
        row = np.array([[feats.get(c, 0.0) for c in self.META_FEATURE_COLS]])

        if model_name is not None:
            # Modelo específico solicitado
            proba = self.models[model_name].predict_proba(row)[0]
            used_name = model_name
        else:
            # ML3: soft-voting ensemble — média das probabilidades de todos os modelos
            probas = []
            for m in self.models.values():
                try:
                    probas.append(m.predict_proba(row)[0])
                except Exception:
                    pass
            if probas:
                proba = np.mean(probas, axis=0)
                used_name = "ensemble"
            else:
                proba = self.models[self.best_model_name].predict_proba(row)[0]
                used_name = self.best_model_name

        classes = self.label_encoder.inverse_transform(np.arange(len(proba)))
        best = int(np.argmax(proba))
        return {
            "recommended_mechanism": classes[best],
            "confidence": float(proba[best]),
            "all_proba": dict(zip(classes, proba.tolist())),
            "meta_model_used": used_name,
        }

    def save(self, path):
        joblib.dump(
            {
                "models": self.models,
                "best_model_name": self.best_model_name,
                "label_encoder": self.label_encoder,
                "meta_feature_cols": self.META_FEATURE_COLS,
                "fast_landmarks": self._fast_landmarks,  # P2
            },
            path,
        )
        _log.info("[MetaLearner] Salvo em '%s'", path)

    def load(self, path):
        d = joblib.load(path)
        self.models = d["models"]
        self.best_model_name = d["best_model_name"]
        self.label_encoder = d["label_encoder"]
        self.META_FEATURE_COLS = d["meta_feature_cols"]
        self._fast_landmarks = d.get("fast_landmarks", True)  # P2: retrocompat
        _log.info("[MetaLearner] Carregado de '%s'", path)

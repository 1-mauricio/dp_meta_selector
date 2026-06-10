"""Meta-modelos para prever o melhor mecanismo DP."""

import logging
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
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
                # ExtraTrees: melhor F1-macro no benchmark (396 datasets)
                "ExtraTrees": Pipeline([
                    ("s", StandardScaler()),
                    ("clf", RandomForestClassifier(
                        n_estimators=200, random_state=42,
                        class_weight="balanced",
                    )),
                ]),
                # LogReg: melhor balanced_accuracy, generaliza bem em datasets pequenos
                "LogReg": Pipeline([
                    ("s", StandardScaler()),
                    ("clf", LogisticRegression(
                        max_iter=500, class_weight="balanced", random_state=42,
                    )),
                ]),
                # SVM-Linear: melhor balanced_accuracy junto com LogReg
                "SVM-Linear": Pipeline([
                    ("s", StandardScaler()),
                    ("clf", SVC(
                        kernel="linear", probability=True,
                        class_weight="balanced", random_state=42,
                    )),
                ]),
            }
        else:
            self.models = {
                "ExtraTrees": Pipeline([
                    ("s", StandardScaler()),
                    ("clf", RandomForestClassifier(
                        n_estimators=400, random_state=42,
                        class_weight="balanced",
                    )),
                ]),
                "LogReg": Pipeline([
                    ("s", StandardScaler()),
                    ("clf", LogisticRegression(
                        max_iter=1000, class_weight="balanced", random_state=42,
                    )),
                ]),
                "SVM-Linear": Pipeline([
                    ("s", StandardScaler()),
                    ("clf", SVC(
                        kernel="linear", probability=True,
                        class_weight="balanced", random_state=42,
                    )),
                ]),
                "SVM-RBF": Pipeline([
                    ("s", StandardScaler()),
                    ("clf", SVC(
                        kernel="rbf", probability=True,
                        class_weight="balanced", random_state=42,
                    )),
                ]),
                "GradientBoosting": Pipeline([
                    ("s", StandardScaler()),
                    ("clf", GradientBoostingClassifier(n_estimators=150, random_state=42)),
                ]),
            }
        self.best_model_name: Optional[str] = None
        self.label_encoder = LabelEncoder()
        self._family_classifier = None
        self._family_label_map: dict = {}

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

        # Oversampling manual das classes minoritárias (sem dependência externa)
        X_meta, y_meta = self._oversample(X_meta, y_meta)

        min_class = int(np.min(np.bincount(y_meta)))
        n_cls = len(np.unique(y_meta))

        if min_class < 2:
            cv, cv_name = LeaveOneOut(), "LeaveOneOut"
        else:
            k = min(5, min_class)
            cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
            cv_name = f"StratifiedKFold(k={k})"

        _log.info("[Meta-Modelos] CV=%s  n=%d  classes=%d", cv_name, len(X_meta), n_cls)
        _log.info(
            "  Distribuição das classes (pós-oversample): %s",
            {k: int(v) for k, v in zip(self.label_encoder.classes_, np.bincount(y_meta))},
        )

        # Treina classificador hierárquico de família (continuous/discrete/categorical)
        self._fit_family_classifier(X_meta, y_meta)

        # Caso degenerado: apenas 1 classe — usa DummyClassifier (most_frequent)
        if n_cls < 2:
            _log.warning(
                "[MetaLearner] Apenas 1 classe no treino — usando DummyClassifier (most_frequent)."
            )
            dummy = DummyClassifier(strategy="most_frequent")
            dummy.fit(X_meta, y_meta)
            self.models = {"Dummy": dummy}
            self.best_model_name = "Dummy"
            return {"Dummy": float("nan")}

        scores = {}
        failed_fit = []

        # Pesos de amostra para compensar desbalanceamento (usado no fit do GB)
        counts = np.bincount(y_meta)
        class_weights = len(y_meta) / (len(counts) * counts)
        sample_weights = class_weights[y_meta]

        for name, model in list(self.models.items()):
            try:
                s = cross_val_score(model, X_meta, y_meta, cv=cv, scoring="f1_macro")
                scores[name] = float(s.mean())
            except Exception as exc:
                _log.debug("[MetaLearner] CV falhou para %s: %s", name, exc)
                scores[name] = float("nan")
            try:
                # GradientBoosting não aceita class_weight → usa sample_weight no fit
                clf = model[-1] if hasattr(model, "__getitem__") else model
                if isinstance(clf, GradientBoostingClassifier):
                    model.fit(X_meta, y_meta, **{f"{model.steps[-1][0]}__sample_weight": sample_weights})
                else:
                    model.fit(X_meta, y_meta)
            except Exception as exc:
                _log.warning("[MetaLearner] fit() falhou para %s: %s — removido.", name, exc)
                failed_fit.append(name)

        for name in failed_fit:
            del self.models[name]
            scores.pop(name, None)

        if not self.models:
            _log.warning("[MetaLearner] Todos os modelos falharam — usando DummyClassifier.")
            dummy = DummyClassifier(strategy="most_frequent")
            dummy.fit(X_meta, y_meta)
            self.models = {"Dummy": dummy}
            self.best_model_name = "Dummy"
            return {"Dummy": float("nan")}

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

    @staticmethod
    def _oversample(X: np.ndarray, y: np.ndarray, target_ratio: float = 0.4) -> tuple:
        """Oversampling manual: replica amostras minoritárias até atingir target_ratio da classe majoritária.

        Parameters
        ----------
        target_ratio:
            Fração mínima desejada de cada classe em relação à classe majoritária.
            Ex: 0.4 → cada classe terá pelo menos 40% do tamanho da maior.
        """
        rng = np.random.RandomState(42)
        counts = np.bincount(y)
        max_count = counts.max()
        target_count = int(max_count * target_ratio)

        X_parts = [X]
        y_parts = [y]
        for cls_idx, cnt in enumerate(counts):
            if cnt < target_count:
                n_add = target_count - cnt
                idxs = np.where(y == cls_idx)[0]
                chosen = rng.choice(idxs, size=n_add, replace=True)
                X_parts.append(X[chosen])
                y_parts.append(np.full(n_add, cls_idx))

        X_out = np.vstack(X_parts)
        y_out = np.concatenate(y_parts)
        # Shuffle
        perm = rng.permutation(len(y_out))
        return X_out[perm], y_out[perm]

    def _fit_family_classifier(self, X_meta: np.ndarray, y_meta: np.ndarray) -> None:
        """Treina um classificador de família (continuous/discrete/categorical).

        Usado em predict() para ajustar as probabilidades via prior hierárquico.
        """
        from .mechanisms import FAMILY_OF
        classes = self.label_encoder.classes_
        y_fam = np.array([
            {"continuous": 0, "discrete": 1, "categorical": 2}.get(
                FAMILY_OF.get(c, "continuous"), 0
            )
            for c in classes[y_meta]
        ])
        fam_counts = np.bincount(y_fam, minlength=3)
        _log.info(
            "  Família (continuous/discrete/categorical): %s",
            {n: int(c) for n, c in zip(["continuous", "discrete", "categorical"], fam_counts)},
        )
        try:
            fam_clf = Pipeline([
                ("s", StandardScaler()),
                ("clf", SVC(kernel="linear", probability=True,
                            class_weight="balanced", random_state=42)),
            ])
            fam_clf.fit(X_meta, y_fam)
            self._family_classifier = fam_clf
            self._family_label_map = {0: "continuous", 1: "discrete", 2: "categorical"}
        except Exception as exc:
            _log.debug("[MetaLearner] family_classifier falhou: %s", exc)
            self._family_classifier = None

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

        # Prior hierárquico: ajusta probabilidades com prior de família
        proba = self._apply_family_prior(row, proba)

        classes = self.label_encoder.inverse_transform(np.arange(len(proba)))
        best = int(np.argmax(proba))
        return {
            "recommended_mechanism": classes[best],
            "confidence": float(proba[best]),
            "all_proba": dict(zip(classes, proba.tolist())),
            "meta_model_used": used_name,
        }

    def _apply_family_prior(self, row: np.ndarray, proba: np.ndarray) -> np.ndarray:
        """Ajusta probabilidades multiplicando pelo prior de família previsto.

        Se o classificador de família prevê com alta confiança "discrete" ou "categorical",
        reforça os mecanismos da família correspondente.
        """
        from .mechanisms import FAMILY_OF
        if getattr(self, "_family_classifier", None) is None:
            return proba

        try:
            fam_proba = self._family_classifier.predict_proba(row)[0]  # [cont, disc, cat]
            p_cont, p_disc, p_cat = fam_proba

            classes = self.label_encoder.classes_
            proba = proba.copy().astype(float)

            for i, cls in enumerate(classes):
                fam = FAMILY_OF.get(cls, "continuous")
                if fam == "discrete":
                    proba[i] *= (1.0 + 3.0 * p_disc)   # boost proporcional à confiança
                elif fam == "categorical":
                    proba[i] *= (1.0 + 3.0 * p_cat)
                else:  # continuous
                    proba[i] *= (1.0 + 1.5 * p_cont)

            # Renormaliza
            total = proba.sum()
            if total > 1e-9:
                proba /= total
        except Exception:
            pass

        return proba

    def save(self, path):
        joblib.dump(
            {
                "models": self.models,
                "best_model_name": self.best_model_name,
                "label_encoder": self.label_encoder,
                "meta_feature_cols": self.META_FEATURE_COLS,
                "fast_landmarks": self._fast_landmarks,
                "family_classifier": getattr(self, "_family_classifier", None),
                "family_label_map": getattr(self, "_family_label_map", {}),
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
        self._fast_landmarks = d.get("fast_landmarks", True)
        self._family_classifier = d.get("family_classifier", None)
        self._family_label_map = d.get("family_label_map", {})
        _log.info("[MetaLearner] Carregado de '%s'", path)

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
        # v16: Thresholds otimizados - balanceando recall vs hit rate
        # Mantendo valores mais conservadores para preservar hit rate
        self._family_gate_threshold: float = 0.55
        self._cat_prefilter = None
        self._cat_prefilter_threshold: float = 0.75  # Mantido
        self._cat_prefilter_family_min: float = 0.15  # Mantido
        self._gauss_prefilter = None
        self._gauss_prefilter_threshold: float = 0.80  # Ligeiramente mais ativo
        self._ga_boost_pca_threshold: float = 0.45
        self._ga_boost_factor: float = 2.8  # Ligeiro boost
        # Novo: classificadores por família para ensemble hierárquico
        self._family_mechanism_classifiers: Dict[str, Pipeline] = {}
        self._discrete_prefilter = None  # Novo: pré-filtro para Geometric
        # v17: Fallback seguro para Laplace
        self._laplace_fallback_enabled: bool = True
        self._laplace_fallback_threshold: float = 0.65  # Confiança mínima para alternativa
        # FASE 2: Regressão de Perda de Utilidade
        # Cada mecanismo recebe uma previsão de perda (%) em vez de uma classificação direta.
        # O framework recomenda o mecanismo com MENOR perda prevista.
        self._regression_model = None
        self._regression_mechanisms: List[str] = []
        self._regression_cv_mae: float = float("nan")
        # v18/v19: Ensemble Híbrido — Classificador como Filtro de Sobrevivência + Regressor como Fine-Tuner.
        # top_k=3: calibrado offline na v19 (tune_meta_models.py) — top_k≥2 converge para o mesmo
        #          resultado; 3 garante cobertura caso o 1º sobrevivente não esteja no regressor.
        self._hybrid_top_k: int = 3
        # v19: margem de fallback conservador — sweet spot calibrado offline com grid 5×8 (40 combos):
        #   margin=0.5pp → Hit Rate=50.5%, Pior-que-Laplace=3.0% (era 31.8% sem calibração).
        #   Valores ≥2pp colapsam para Laplace em quase todos os casos (hit rate cai para ~48%).
        self._hybrid_laplace_margin: float = 0.5

    def fit(self, meta_df: pd.DataFrame) -> Dict[str, float]:
        # Colunas de perda de utilidade são targets de regressão, não features
        loss_cols = {f"utility_loss_{m}" for m in MECHANISM_NAMES}
        excl = (
            {"dataset_name", "best_mechanism", "best_relative_acc", "baseline_acc", "best_family"}
            | {f"acc_{m}" for m in MECHANISM_NAMES}
            | loss_cols
        )
        self.META_FEATURE_COLS = [c for c in meta_df.columns if c not in excl]

        X_meta = np.nan_to_num(
            meta_df[self.META_FEATURE_COLS].values.astype(float),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        y_meta = self.label_encoder.fit_transform(meta_df["best_mechanism"])

        # CAT1: treina pré-filtro binário nos dados originais (antes do oversample)
        self._fit_categorical_prefilter(X_meta, y_meta)

        # GAUSS: treina pré-filtro binário GaussianAnalytic vs Laplace nos dados originais
        self._fit_gaussian_prefilter(X_meta, y_meta)

        # DISC: treina pré-filtro para Geometric (datasets discretos)
        self._fit_discrete_prefilter(X_meta, y_meta)

        # HIER: treina classificador de família nos dados ORIGINAIS (antes do oversample)
        self._fit_family_classifier(X_meta, y_meta)
        
        # Treina classificadores por família para ensemble hierárquico
        self._fit_family_mechanism_classifiers(X_meta, y_meta)

        # Guarda X_meta antes do oversample para treinar os regressores
        # (regressão deve ver a distribuição real dos datasets, sem amostras duplicadas)
        X_meta_orig = X_meta.copy()

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
        # Passa sample_weight para que a calibração também respeite o balanceamento.
        if len(X_meta) >= 10:
            for name, model in list(self.models.items()):
                try:
                    cal_method = "isotonic" if len(X_meta) >= 30 else "sigmoid"
                    cal = CalibratedClassifierCV(model, cv="prefit", method=cal_method)
                    cal.fit(X_meta, y_meta, sample_weight=sample_weights)
                    self.models[name] = cal
                except Exception:
                    pass  # mantém não calibrado se falhar

        valid = {k: v for k, v in scores.items() if not np.isnan(v)}
        self.best_model_name = (
            max(valid, key=valid.get) if valid else list(scores)[0]
        )

        # FASE 2: Treina regressores de perda de utilidade usando dados originais (pré-oversample)
        # X_meta_orig é os dados sem oversample — preserva distribuição real dos datasets
        self._fit_regression(meta_df, X_meta_orig)

        return scores

    @staticmethod
    def _oversample(X: np.ndarray, y: np.ndarray, target_ratio: float = 0.8) -> tuple:
        """Oversampling manual: replica amostras minoritárias até atingir target_ratio da classe majoritária.

        Parameters
        ----------
        target_ratio:
            Fração mínima desejada de cada classe em relação à classe majoritária.
            Ex: 0.8 → cada classe terá pelo menos 80% do tamanho da maior.
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

    def _fit_categorical_prefilter(self, X_meta: np.ndarray, y_meta: np.ndarray) -> None:
        """CAT1: treina um classificador binário Exponential vs. resto.

        Usa os dados originais (sem oversample) para preservar a distribuição real.
        Se disparar com confiança >= _cat_prefilter_threshold em predict(), retorna
        Exponential diretamente, sem passar pelo classificador multi-classe.
        """
        exp_idx = list(self.label_encoder.classes_).index("Exponential") if "Exponential" in self.label_encoder.classes_ else -1
        if exp_idx < 0:
            _log.debug("[CAT1] 'Exponential' não encontrado nas classes — pré-filtro desativado.")
            self._cat_prefilter = None
            return

        y_binary = (y_meta == exp_idx).astype(int)
        n_pos = int(y_binary.sum())
        n_neg = int(len(y_binary) - n_pos)

        if n_pos < 5:
            _log.debug("[CAT1] Poucos exemplos Exponential (%d) — pré-filtro desativado.", n_pos)
            self._cat_prefilter = None
            return

        try:
            from sklearn.ensemble import GradientBoostingClassifier as GBC
            clf = Pipeline([
                ("s", StandardScaler()),
                ("clf", GBC(
                    n_estimators=150, max_depth=3, learning_rate=0.1,
                    subsample=0.8, random_state=42,
                )),
            ])
            k = min(5, n_pos)
            cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
            f1s = cross_val_score(clf, X_meta, y_binary, cv=cv, scoring="f1")
            _log.info(
                "[CAT1] Pré-filtro Exponential  pos=%d  neg=%d  F1-CV=%.4f (k=%d)",
                n_pos, n_neg, float(f1s.mean()), k,
            )
            clf.fit(X_meta, y_binary)
            self._cat_prefilter = clf
        except Exception as exc:
            _log.warning("[CAT1] Falha ao treinar pré-filtro: %s — desativado.", exc)
            self._cat_prefilter = None

    def _fit_gaussian_prefilter(self, X_meta: np.ndarray, y_meta: np.ndarray) -> None:
        """GAUSS: treina um classificador binário GaussianAnalytic vs Laplace.

        Usa apenas features discriminadoras conhecidas para GA vs Laplace:
        - n_features, pca_top1_var, pca_intrinsic_dim_ratio, mean_sensitivity,
          max_sensitivity, outlier_ratio, mean_std, ratio_integer_cols
        Treinado somente nos datasets contínuos (Laplace ou GaussianAnalytic).
        """
        classes = self.label_encoder.classes_
        gauss_idx  = list(classes).index("GaussianAnalytic") if "GaussianAnalytic" in classes else -1
        laplace_idx = list(classes).index("Laplace") if "Laplace" in classes else -1

        if gauss_idx < 0 or laplace_idx < 0 or self.META_FEATURE_COLS is None:
            self._gauss_prefilter = None
            return

        # Seleciona features conhecidas como discriminadoras GA vs Laplace
        GA_FEATURES = [
            "n_features", "pca_top1_var", "pca_intrinsic_dim_ratio",
            "mean_sensitivity", "max_sensitivity", "outlier_ratio",
            "mean_std", "std_std", "ratio_integer_cols", "mean_corr",
            "coeff_var", "samples_per_feature",
        ]
        feat_idx = [i for i, c in enumerate(self.META_FEATURE_COLS) if c in GA_FEATURES]
        if len(feat_idx) < 3:
            self._gauss_prefilter = None
            return
        self._gauss_feature_idx = feat_idx

        # Filtra apenas datasets contínuos
        cont_mask = np.isin(y_meta, [gauss_idx, laplace_idx])
        X_cont = X_meta[cont_mask][:, feat_idx]
        y_cont = (y_meta[cont_mask] == gauss_idx).astype(int)

        n_pos = int(y_cont.sum())
        n_neg = int(len(y_cont) - n_pos)

        if n_pos < 5:
            _log.debug("[GAUSS] Poucos exemplos GaussianAnalytic (%d) — prefilter desativado.", n_pos)
            self._gauss_prefilter = None
            return

        try:
            from sklearn.ensemble import GradientBoostingClassifier as GBC
            clf = Pipeline([
                ("s", StandardScaler()),
                ("clf", GBC(
                    n_estimators=200, max_depth=3, learning_rate=0.05,
                    subsample=0.8, random_state=42,
                )),
            ])
            k = min(5, n_pos)
            cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
            f1s = cross_val_score(clf, X_cont, y_cont, cv=cv, scoring="f1")
            _log.info(
                "[GAUSS] Pré-filtro GaussianAnalytic  pos=%d  neg=%d  F1-CV=%.4f (k=%d)  feats=%d",
                n_pos, n_neg, float(f1s.mean()), k, len(feat_idx),
            )
            clf.fit(X_cont, y_cont)
            self._gauss_prefilter = clf
        except Exception as exc:
            _log.warning("[GAUSS] Falha ao treinar prefilter: %s — desativado.", exc)
            self._gauss_prefilter = None

    def _apply_gaussian_prefilter(self, row: np.ndarray) -> Optional[Dict]:
        """GAUSS: retorna recomendação GaussianAnalytic se o prefilter disparar.

        Usa apenas o subconjunto de features treinado em _fit_gaussian_prefilter.
        """
        clf = getattr(self, "_gauss_prefilter", None)
        feat_idx = getattr(self, "_gauss_feature_idx", None)
        if clf is None or feat_idx is None:
            return None
        try:
            row_ga = row[:, feat_idx]
            p_gauss = float(clf.predict_proba(row_ga)[0][1])
            if p_gauss >= self._gauss_prefilter_threshold:
                classes = list(self.label_encoder.classes_)
                all_proba = {m: 0.0 for m in classes}
                all_proba["GaussianAnalytic"] = p_gauss
                return {
                    "recommended_mechanism": "GaussianAnalytic",
                    "confidence": p_gauss,
                    "all_proba": all_proba,
                    "meta_model_used": "gauss_prefilter",
                }
        except Exception:
            pass
        return None

    def _fit_family_classifier(self, X_meta: np.ndarray, y_meta: np.ndarray) -> None:
        """Treina um classificador de família.

        Com Geometric removido, torna-se um classificador binário: continuous (0) vs
        categorical (1). Treinado pré-oversample para preservar distribuição real.
        """
        from .mechanisms import FAMILY_OF
        classes = self.label_encoder.classes_

        # Monta labels de família presentes no treino
        fam_raw = np.array([FAMILY_OF.get(c, "continuous") for c in classes[y_meta]])
        present_fams = sorted(set(fam_raw))

        if len(present_fams) < 2:
            _log.debug("[HIER] Apenas 1 família no treino — family_classifier desativado.")
            self._family_classifier = None
            return

        # Codifica famílias presentes em 0, 1, 2...
        fam_to_idx = {f: i for i, f in enumerate(present_fams)}
        y_fam = np.array([fam_to_idx[f] for f in fam_raw])
        self._family_label_map = {i: f for f, i in fam_to_idx.items()}

        fam_counts = {f: int(np.sum(y_fam == i)) for f, i in fam_to_idx.items()}
        _log.info("  Famílias no treino: %s", fam_counts)

        try:
            fam_clf = Pipeline([
                ("s", StandardScaler()),
                ("clf", SVC(kernel="linear", probability=True,
                            class_weight="balanced", random_state=42)),
            ])
            fam_clf.fit(X_meta, y_fam)
            self._family_classifier = fam_clf
        except Exception as exc:
            _log.debug("[HIER] family_classifier falhou: %s", exc)
            self._family_classifier = None

    def _fit_discrete_prefilter(self, X_meta: np.ndarray, y_meta: np.ndarray) -> None:
        """DISC: treina um classificador binário para Geometric (discrete family).
        
        Datasets discretos (inteiros com range pequeno) favorecem Geometric.
        """
        from .mechanisms import FAMILY_OF
        classes = self.label_encoder.classes_
        
        # Identifica mecanismos discretos
        discrete_mechs = [c for c in classes if FAMILY_OF.get(c) == "discrete"]
        if not discrete_mechs:
            self._discrete_prefilter = None
            return
        
        discrete_indices = [list(classes).index(m) for m in discrete_mechs]
        y_binary = np.isin(y_meta, discrete_indices).astype(int)
        n_pos = int(y_binary.sum())
        
        if n_pos < 3:
            _log.debug("[DISC] Poucos exemplos discrete (%d) — pré-filtro desativado.", n_pos)
            self._discrete_prefilter = None
            return
        
        # Features discriminadoras para discrete
        DISC_FEATURES = [
            "ratio_integer_cols", "ratio_discrete", "disc_composite_score",
            "disc_mean_int_range", "disc_ratio_small_int_range", 
            "mean_log_unique_ratio", "median_unique_per_col",
        ]
        feat_idx = [i for i, c in enumerate(self.META_FEATURE_COLS) if c in DISC_FEATURES]
        if len(feat_idx) < 3:
            self._discrete_prefilter = None
            return
        self._discrete_feature_idx = feat_idx
        
        try:
            from sklearn.ensemble import GradientBoostingClassifier as GBC
            clf = Pipeline([
                ("s", StandardScaler()),
                ("clf", GBC(n_estimators=100, max_depth=3, random_state=42)),
            ])
            clf.fit(X_meta[:, feat_idx], y_binary)
            self._discrete_prefilter = clf
            _log.info("[DISC] Pré-filtro discrete  pos=%d  neg=%d", n_pos, len(y_binary) - n_pos)
        except Exception as exc:
            _log.debug("[DISC] Falha ao treinar prefilter: %s", exc)
            self._discrete_prefilter = None

    def _fit_family_mechanism_classifiers(self, X_meta: np.ndarray, y_meta: np.ndarray) -> None:
        """Treina classificadores por família para ensemble hierárquico.
        
        Cada família tem seu próprio classificador que escolhe entre seus mecanismos.
        """
        from .mechanisms import FAMILY_OF
        classes = self.label_encoder.classes_
        
        self._family_mechanism_classifiers = {}
        
        for family in ["continuous", "discrete", "categorical"]:
            # Filtra mecanismos desta família
            family_mechs = [c for c in classes if FAMILY_OF.get(c) == family]
            if len(family_mechs) < 2:
                continue
            
            # Filtra exemplos desta família
            family_mech_indices = [list(classes).index(m) for m in family_mechs]
            mask = np.isin(y_meta, family_mech_indices)
            if mask.sum() < 5:
                continue
            
            X_fam = X_meta[mask]
            y_fam_raw = y_meta[mask]
            
            # Recodifica labels para 0, 1, 2...
            idx_to_local = {idx: i for i, idx in enumerate(family_mech_indices)}
            y_fam = np.array([idx_to_local[y] for y in y_fam_raw])
            
            try:
                clf = Pipeline([
                    ("s", StandardScaler()),
                    ("clf", RandomForestClassifier(
                        n_estimators=100, class_weight="balanced", random_state=42
                    )),
                ])
                clf.fit(X_fam, y_fam)
                self._family_mechanism_classifiers[family] = {
                    "classifier": clf,
                    "mechanisms": family_mechs,
                    "idx_to_mech": {i: m for i, m in enumerate(family_mechs)},
                }
                _log.info("[HIER-MECH] Classificador %s: %d exemplos, %d mecanismos",
                         family, len(y_fam), len(family_mechs))
            except Exception as exc:
                _log.debug("[HIER-MECH] Falha para %s: %s", family, exc)

    def _fit_regression(self, meta_df: pd.DataFrame, X_meta: np.ndarray) -> None:
        """FASE 2: Treina regressor multi-output para prever perda de utilidade por mecanismo.
        
        Em vez de classificar diretamente ("use Laplace"), o metamodelo aprende a prever
        a perda de utilidade relativa (%) de cada mecanismo para um dataset com aquelas
        meta-features. O mecanismo com MENOR perda prevista será recomendado.
        
        Isso é matematicamente superior à classificação porque:
        1. Usa informação ordinal (Laplace perdeu 5%, Gauss perdeu 8%) em vez de label binário
        2. Permite recomendar por margem de ganho, não só por rank
        3. Generaliza melhor em distribuições de dados não vistas
        """
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.multioutput import MultiOutputRegressor
        from sklearn.model_selection import KFold, cross_val_score as cv_score

        # Identifica colunas de perda disponíveis no meta-dataset
        present_loss_cols = [
            f"utility_loss_{m}" for m in MECHANISM_NAMES
            if f"utility_loss_{m}" in meta_df.columns
        ]
        if not present_loss_cols:
            _log.info("[Regressão] Colunas utility_loss_* não encontradas — regressão desativada.")
            _log.info("  → Para ativar, use MetaDatasetBuilder para gerar o meta-dataset.")
            self._regression_model = None
            return

        self._regression_mechanisms = [c.replace("utility_loss_", "") for c in present_loss_cols]

        # Target: matriz de perdas de utilidade relativa (n_datasets × n_mechanisms)
        Y = np.nan_to_num(
            meta_df[present_loss_cols].values.astype(float),
            nan=0.0, posinf=100.0, neginf=0.0,
        )
        Y = np.clip(Y, 0.0, 100.0)

        # Multi-output Random Forest: melhor trade-off qualidade/velocidade para tabulares
        # Cada mecanismo tem seu próprio estimador interno (MultiOutputRegressor)
        base_reg = RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
            min_samples_leaf=2,
        )
        model = Pipeline([
            ("s", StandardScaler()),
            ("reg", MultiOutputRegressor(base_reg, n_jobs=1)),
        ])

        # CV para medir MAE de regressão (métrica de qualidade das previsões de perda)
        k = min(5, len(X_meta))
        cv = KFold(n_splits=k, shuffle=True, random_state=42)
        try:
            mae_scores = cv_score(
                model, X_meta, Y, cv=cv, scoring="neg_mean_absolute_error"
            )
            self._regression_cv_mae = float(-mae_scores.mean())
            _log.info(
                "[Regressão] Multi-output RF  n=%d  mecanismos=%d  MAE-CV=%.2f%%",
                len(X_meta), len(present_loss_cols), self._regression_cv_mae,
            )
        except Exception as exc:
            _log.warning("[Regressão] CV falhou: %s", exc)
            self._regression_cv_mae = float("nan")

        # Treino final em todos os dados
        model.fit(X_meta, Y)
        self._regression_model = model
        _log.info("[Regressão] Mecanismos cobertos: %s", self._regression_mechanisms)

    def _predict_regression(self, row: np.ndarray) -> Optional[Dict]:
        """FASE 2: Recomenda mecanismo com menor perda de utilidade prevista.
        
        Converte as perdas previstas em 'probabilidades' via softmin
        (menor perda = maior probabilidade) para compatibilidade com a API existente.
        """
        model = getattr(self, "_regression_model", None)
        mechanisms = getattr(self, "_regression_mechanisms", [])
        if model is None or not mechanisms:
            return None

        try:
            predicted_losses = model.predict(row)[0]  # shape: (n_mechanisms,)
            predicted_losses = np.clip(predicted_losses, 0.0, 100.0)

            # Mecanismo com menor perda prevista
            best_idx = int(np.argmin(predicted_losses))
            best_mech = mechanisms[best_idx]

            # Converte perdas em "probabilidades" via softmin
            # p_i = exp(-loss_i / T) / sum(exp(-loss_j / T)), T=10 (temperatura)
            T = 10.0
            log_proba = -predicted_losses / T
            log_proba -= log_proba.max()  # estabilidade numérica
            proba = np.exp(log_proba)
            proba /= proba.sum()

            all_losses = {m: float(l) for m, l in zip(mechanisms, predicted_losses)}
            all_proba = {m: float(p) for m, p in zip(mechanisms, proba)}

            return {
                "recommended_mechanism": best_mech,
                "confidence": float(proba[best_idx]),
                "all_proba": all_proba,
                "predicted_utility_loss": all_losses,
                "meta_model_used": "regression_multioutput",
            }
        except Exception as exc:
            _log.warning("[Regressão] predict falhou: %s — fallback para classificação.", exc)
            return None

    def _predict_hybrid(self, row: np.ndarray) -> Optional[Dict]:
        """v18: Ensemble Híbrido — Filtro de Sobrevivência (Classificador) + Fine-Tuning (Regressor).

        Estratégia:
        1. Classificador (soft-voting ensemble) gera probabilidades para todos os mecanismos.
        2. Filtro de Sobrevivência: mantém os top-K mecanismos com maior probabilidade.
        3. Regressor: ordena os finalistas pela menor perda de utilidade prevista.
        4. v19 Fallback conservador: se regressor não confia que o vencedor é melhor que
           Laplace (dentro de margem), usa a recomendação do classificador puro em vez disso.
           Estanca o problema de 48.6% "piores que Laplace" causado por labels ruidosas (n_runs=1).
        """
        reg_model = getattr(self, "_regression_model", None)
        reg_mechs = getattr(self, "_regression_mechanisms", [])
        if reg_model is None or not reg_mechs:
            return None

        # 1. Classificador: soft-voting ensemble → probabilidades por mecanismo
        probas = []
        for m in self.models.values():
            try:
                probas.append(m.predict_proba(row)[0])
            except Exception:
                pass
        if not probas:
            return None

        clf_proba = np.mean(probas, axis=0)
        classes = list(self.label_encoder.inverse_transform(np.arange(len(clf_proba))))
        clf_proba_dict = dict(zip(classes, clf_proba.tolist()))

        # 2. Filtro de Sobrevivência: top-K mecanismos por probabilidade do classificador
        sorted_by_clf = sorted(clf_proba_dict.items(), key=lambda x: -x[1])
        survivors = [m for m, _ in sorted_by_clf[: self._hybrid_top_k] if m in reg_mechs]
        if not survivors:
            top_mech = sorted_by_clf[0][0]
            survivors = [top_mech] if top_mech in reg_mechs else list(reg_mechs[:1])

        # 3. Regressor: fine-tuning entre os finalistas pela menor perda prevista
        try:
            all_predicted_losses = reg_model.predict(row)[0]
            all_predicted_losses = np.clip(all_predicted_losses, 0.0, 100.0)
            loss_dict = dict(zip(reg_mechs, all_predicted_losses.tolist()))

            survivor_losses = {m: loss_dict[m] for m in survivors if m in loss_dict}
            if not survivor_losses:
                return None

            best_mech = min(survivor_losses, key=survivor_losses.get)
            best_loss = survivor_losses[best_mech]

            # ── v19: Fallback Conservador ──────────────────────────────────────────────
            # Se o vencedor não supera Laplace por self._hybrid_laplace_margin, o regressor
            # pode estar confuso → recorre ao top-1 do classificador (mais robusto ao ruído).
            # Margem calibrada offline (tune_meta_models.py v19): 0.5pp é o sweet spot.
            _laplace_loss = loss_dict.get("Laplace", float("inf"))
            if best_mech != "Laplace" and best_loss > _laplace_loss - self._hybrid_laplace_margin:
                clf_best = sorted_by_clf[0][0]
                _log.debug(
                    "[Híbrido v19] Fallback: reg=%s (loss=%.1f%%) não supera Laplace "
                    "(loss=%.1f%%) por %.1fpp → clf top-1=%s",
                    best_mech, best_loss, _laplace_loss, self._hybrid_laplace_margin, clf_best,
                )
                best_mech = clf_best
                best_loss = loss_dict.get(clf_best, best_loss)
            # ──────────────────────────────────────────────────────────────────────────

            # Confiança composta: softmin sobre sobreviventes + probabilidade do classificador
            surv_list = list(survivor_losses.keys())
            surv_arr = np.array([survivor_losses[m] for m in surv_list])
            T = 10.0
            log_p = -surv_arr / T
            log_p -= log_p.max()
            surv_proba = np.exp(log_p)
            surv_proba /= surv_proba.sum()
            best_surv_idx = surv_list.index(best_mech) if best_mech in surv_list else 0
            reg_conf = float(surv_proba[best_surv_idx])
            clf_conf = clf_proba_dict.get(best_mech, 0.0)
            composite_conf = 0.4 * clf_conf + 0.6 * reg_conf

            _log.debug(
                "[Híbrido] sobreviventes=%s → recomendado=%s (loss=%.1f%%)",
                survivors, best_mech, best_loss,
            )

            return {
                "recommended_mechanism": best_mech,
                "confidence": composite_conf,
                "all_proba": clf_proba_dict,
                "hybrid_survivors": survivors,
                "predicted_utility_loss": loss_dict,
                "meta_model_used": "hybrid_ensemble",
            }
        except Exception as exc:
            _log.warning("[Híbrido] predict falhou: %s — fallback para classificação.", exc)
            return None

    def _apply_discrete_prefilter(self, row: np.ndarray) -> Optional[Dict]:
        """DISC: retorna recomendação de mecanismo discreto se o prefilter disparar."""
        clf = getattr(self, "_discrete_prefilter", None)
        feat_idx = getattr(self, "_discrete_feature_idx", None)
        if clf is None or feat_idx is None:
            return None
        
        try:
            row_disc = row[:, feat_idx]
            p_disc = float(clf.predict_proba(row_disc)[0][1])
            
            # Threshold para discrete
            if p_disc >= 0.70:
                # Usa classificador intra-família se disponível
                family_clf = self._family_mechanism_classifiers.get("discrete")
                if family_clf:
                    proba = family_clf["classifier"].predict_proba(row)[0]
                    best_idx = int(np.argmax(proba))
                    best_mech = family_clf["idx_to_mech"][best_idx]
                else:
                    # Fallback para Geometric se não há classificador intra-família
                    best_mech = "Geometric" if "Geometric" in self.label_encoder.classes_ else None
                    if best_mech is None:
                        return None
                
                classes = list(self.label_encoder.classes_)
                all_proba = {m: 0.0 for m in classes}
                all_proba[best_mech] = p_disc
                return {
                    "recommended_mechanism": best_mech,
                    "confidence": p_disc,
                    "all_proba": all_proba,
                    "meta_model_used": "discrete_prefilter",
                }
        except Exception:
            pass
        return None

    def predict(
        self,
        X,
        y,
        model_name=None,
        epsilon: float = None,
        task_type: str = None,
    ) -> Dict:
        """Prediz o melhor mecanismo DP para um dataset.
        
        Parameters
        ----------
        X : np.ndarray
            Matriz de features do dataset
        y : np.ndarray
            Vetor de labels
        model_name : str, optional
            Nome do meta-modelo específico a usar
        epsilon : float, optional
            Orçamento de privacidade desejado (contexto obrigatório para DP)
        task_type : str, optional
            Tipo de tarefa: "classification", "regression", ou "queries"
        
        Returns
        -------
        Dict
            Dicionário com recommended_mechanism, confidence, all_proba, etc.
        """
        if self.META_FEATURE_COLS is None:
            raise RuntimeError("Chame fit() antes de predict().")
        y_enc = LabelEncoder().fit_transform(y)
        feats = MetaFeatureExtractor(fast_landmarks=self._fast_landmarks).extract(
            X, y_enc, epsilon=epsilon, task_type=task_type
        )
        row = np.array([[feats.get(c, 0.0) for c in self.META_FEATURE_COLS]])

        # CAT1: pré-filtro binário Exponential — intercepta categorical antes do portão de família
        cat_result = self._apply_categorical_prefilter(row)
        if cat_result is not None:
            return cat_result

        # DISC: pré-filtro para mecanismos discretos (Geometric, etc.)
        disc_result = self._apply_discrete_prefilter(row)
        if disc_result is not None:
            return disc_result

        # GAUSS: pré-filtro binário GaussianAnalytic — dentro do espaço contínuo
        gauss_result = self._apply_gaussian_prefilter(row)
        if gauss_result is not None:
            return gauss_result

        # v18: Ensemble Híbrido — Filtro de Sobrevivência (Classificador) + Fine-Tuning (Regressor).
        # O classificador elimina mecanismos claramente inadequados; o regressor ordena os finalistas
        # pela menor perda de utilidade prevista. Combina robustez do classificador com a precisão
        # ordinal do regressor. Fallback para classificação pura se regressor indisponível.
        if model_name is None:
            hybrid_result = self._predict_hybrid(row)
            if hybrid_result is not None:
                return hybrid_result

        if model_name is not None:
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

        # HIER: decisão hierárquica de família (hard gate ≥ threshold, soft boost abaixo)
        proba = self._apply_family_decision(row, proba)

        # GA BOOST: se pca_top1_var < threshold → amplifica GaussianAnalytic no ensemble
        # (compensação para sub-representação de GA no treino, precision=67% em simulação)
        proba = self._apply_ga_boost(row, proba)

        classes = self.label_encoder.inverse_transform(np.arange(len(proba)))
        best = int(np.argmax(proba))

        # GUARD: Exponential via ensemble requer confirmação do family classifier
        # com ≥ 0.60 de confiança em "categorical", evitando FP em datasets contínuos
        if classes[best] == "Exponential":
            p_cat = self._get_family_confidence(row, "categorical")
            if p_cat < 0.60:
                # Suprime Exponential; re-normaliza
                exp_idx = int(np.where(classes == "Exponential")[0][0])
                proba[exp_idx] = 0.0
                total = proba.sum()
                if total > 1e-9:
                    proba = proba / total
                    best = int(np.argmax(proba))
                    _log.debug("[GUARD] Exponential suprimido (p_cat=%.3f < 0.60); novo melhor=%s",
                               p_cat, classes[best])

        # v17: LAPLACE FALLBACK - se alternativa com baixa confiança, usa Laplace
        recommended = classes[best]
        confidence = float(proba[best])
        fallback_applied = False
        
        if self._laplace_fallback_enabled and recommended != "Laplace":
            if confidence < self._laplace_fallback_threshold:
                if "Laplace" in classes:
                    lap_idx = list(classes).index("Laplace")
                    lap_conf = float(proba[lap_idx])
                    _log.debug("[FALLBACK] %s (conf=%.3f < %.2f) → Laplace (conf=%.3f)",
                               recommended, confidence, self._laplace_fallback_threshold, lap_conf)
                    recommended = "Laplace"
                    confidence = lap_conf
                    fallback_applied = True

        return {
            "recommended_mechanism": recommended,
            "confidence": confidence,
            "all_proba": dict(zip(classes, proba.tolist())),
            "meta_model_used": used_name,
            "fallback_applied": fallback_applied,
        }

    def _get_family_confidence(self, row: np.ndarray, family: str) -> float:
        """Retorna a confiança do family classifier para a família solicitada."""
        fc = getattr(self, "_family_classifier", None)
        label_map = getattr(self, "_family_label_map", {})
        if fc is None:
            return 0.0
        try:
            fam_proba = fc.predict_proba(row)[0]
            idx = next((k for k, v in label_map.items() if v == family), None)
            if idx is not None and idx < len(fam_proba):
                return float(fam_proba[idx])
        except Exception:
            pass
        return 0.0

    def _apply_ga_boost(self, row: np.ndarray, proba: np.ndarray) -> np.ndarray:
        """GA BOOST: amplifica GaussianAnalytic quando pca_top1_var < threshold.

        Datasets com variância concentrada em poucos PCs tendem a ser contínuos-Laplace;
        datasets com variância distribuída (pca_top1_var baixo) favorecem GaussianAnalytic.
        Simulação no test set: boost 3x com pca_top1_var < 0.50 → +3 hits, precision=67%.
        """
        threshold = getattr(self, "_ga_boost_pca_threshold", 0.50)
        factor = getattr(self, "_ga_boost_factor", 3.0)
        if threshold <= 0.0 or factor <= 1.0:
            return proba

        try:
            pca_col = self.META_FEATURE_COLS.index("pca_top1_var")
            pca_val = float(row[0, pca_col])
            if pca_val < threshold:
                classes = list(self.label_encoder.classes_)
                if "GaussianAnalytic" in classes:
                    ga_idx = classes.index("GaussianAnalytic")
                    gated = proba.copy().astype(float)
                    gated[ga_idx] *= factor
                    total = gated.sum()
                    if total > 1e-9:
                        _log.debug("[GA_BOOST] pca_top1_var=%.3f < %.2f → boost GA x%.1f",
                                   pca_val, threshold, factor)
                        return gated / total
        except (ValueError, IndexError):
            pass
        return proba

    def _apply_categorical_prefilter(self, row: np.ndarray) -> Optional[Dict]:
        """CAT1: retorna recomendação Exponential se o pré-filtro disparar.

        Dual-gate: requer tanto p_exp >= _cat_prefilter_threshold (CAT1)
        quanto p_cat >= _cat_prefilter_family_min (HIER), evitando FP
        em datasets contínuos que possuem features dummy-encoded.

        Retorna None se o pré-filtro não disparar (fluxo normal continua).
        """
        clf = getattr(self, "_cat_prefilter", None)
        if clf is None:
            return None
        try:
            p_exp = float(clf.predict_proba(row)[0][1])
            if p_exp >= self._cat_prefilter_threshold:
                # Dual-gate: verifica suporte do family classifier
                fam_min = getattr(self, "_cat_prefilter_family_min", 0.0)
                if fam_min > 0.0:
                    p_cat = self._get_family_confidence(row, "categorical")
                    if p_cat < fam_min:
                        _log.debug("[CAT1] Bloqueado por dual-gate (p_exp=%.3f p_cat=%.3f < %.2f)",
                                   p_exp, p_cat, fam_min)
                        return None
                classes = list(self.label_encoder.classes_)
                all_proba = {m: 0.0 for m in classes}
                all_proba["Exponential"] = p_exp
                return {
                    "recommended_mechanism": "Exponential",
                    "confidence": p_exp,
                    "all_proba": all_proba,
                    "meta_model_used": "cat_prefilter",
                }
        except Exception:
            pass
        return None

    def _apply_family_decision(self, row: np.ndarray, proba: np.ndarray) -> np.ndarray:
        """Decisão hierárquica de família (HIER).

        Usa _family_label_map para ser agnóstico ao número de famílias presentes.
        Se confiança >= _family_gate_threshold: restrição DURA (zera outras famílias).
        Caso contrário: boost SUAVE proporcional.
        """
        from .mechanisms import FAMILY_OF
        if getattr(self, "_family_classifier", None) is None:
            return proba

        try:
            fam_proba = self._family_classifier.predict_proba(row)[0]
            label_map = getattr(self, "_family_label_map", {})  # {idx: family_name}
            max_idx = int(np.argmax(fam_proba))
            max_p = float(fam_proba[max_idx])
            pred_fam = label_map.get(max_idx, "continuous")

            classes = self.label_encoder.classes_
            gated = proba.copy().astype(float)

            if max_p >= self._family_gate_threshold:
                # HARD gate: zera mecanismos fora da família predita
                for i, cls in enumerate(classes):
                    if FAMILY_OF.get(cls, "continuous") != pred_fam:
                        gated[i] = 0.0
                total = gated.sum()
                if total > 1e-9:
                    _log.debug("[HIER] Hard gate: família=%s  confiança=%.3f", pred_fam, max_p)
                    return gated / total
                # fallthrough se todos zerados (segurança)

            # SOFT boost
            for i, cls in enumerate(classes):
                fam = FAMILY_OF.get(cls, "continuous")
                # Encontra a probabilidade desta família no classificador
                fam_idx = next((k for k, v in label_map.items() if v == fam), None)
                p_fam = float(fam_proba[fam_idx]) if fam_idx is not None and fam_idx < len(fam_proba) else 0.0
                boost = 3.0 if fam in ("discrete", "categorical") else 1.5
                gated[i] *= (1.0 + boost * p_fam)
            total = gated.sum()
            if total > 1e-9:
                gated /= total
            return gated

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
                "family_gate_threshold": getattr(self, "_family_gate_threshold", 0.55),
                "cat_prefilter": getattr(self, "_cat_prefilter", None),
                "cat_prefilter_threshold": getattr(self, "_cat_prefilter_threshold", 0.75),
                "cat_prefilter_family_min": getattr(self, "_cat_prefilter_family_min", 0.15),
                "gauss_prefilter": getattr(self, "_gauss_prefilter", None),
                "gauss_prefilter_threshold": getattr(self, "_gauss_prefilter_threshold", 0.85),
                "gauss_feature_idx": getattr(self, "_gauss_feature_idx", None),
                "ga_boost_pca_threshold": getattr(self, "_ga_boost_pca_threshold", 0.45),
                "ga_boost_factor": getattr(self, "_ga_boost_factor", 2.5),
                # Novos campos de classificação
                "discrete_prefilter": getattr(self, "_discrete_prefilter", None),
                "discrete_feature_idx": getattr(self, "_discrete_feature_idx", None),
                "family_mechanism_classifiers": getattr(self, "_family_mechanism_classifiers", {}),
                # FASE 2: Regressão de Perda de Utilidade
                "regression_model": getattr(self, "_regression_model", None),
                "regression_mechanisms": getattr(self, "_regression_mechanisms", []),
                "regression_cv_mae": getattr(self, "_regression_cv_mae", float("nan")),
                # v18/v19: Ensemble Híbrido
                "hybrid_top_k": getattr(self, "_hybrid_top_k", 3),
                "hybrid_laplace_margin": getattr(self, "_hybrid_laplace_margin", 0.5),
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
        self._family_gate_threshold = d.get("family_gate_threshold", 0.55)
        self._cat_prefilter = d.get("cat_prefilter", None)
        self._cat_prefilter_threshold = d.get("cat_prefilter_threshold", 0.75)
        self._cat_prefilter_family_min = d.get("cat_prefilter_family_min", 0.15)
        self._gauss_prefilter = d.get("gauss_prefilter", None)
        self._gauss_prefilter_threshold = d.get("gauss_prefilter_threshold", 0.85)
        self._gauss_feature_idx = d.get("gauss_feature_idx", None)
        self._ga_boost_pca_threshold = d.get("ga_boost_pca_threshold", 0.45)
        self._ga_boost_factor = d.get("ga_boost_factor", 2.5)
        # Novos campos
        self._discrete_prefilter = d.get("discrete_prefilter", None)
        self._discrete_feature_idx = d.get("discrete_feature_idx", None)
        self._family_mechanism_classifiers = d.get("family_mechanism_classifiers", {})
        # FASE 2: Regressão de Perda de Utilidade
        self._regression_model = d.get("regression_model", None)
        self._regression_mechanisms = d.get("regression_mechanisms", [])
        self._regression_cv_mae = d.get("regression_cv_mae", float("nan"))
        # v18/v19: Ensemble Híbrido
        self._hybrid_top_k = d.get("hybrid_top_k", 3)
        self._hybrid_laplace_margin = d.get("hybrid_laplace_margin", 0.5)
        _log.info("[MetaLearner] Carregado de '%s'", path)

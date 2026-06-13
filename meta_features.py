"""Extração de meta-features de datasets tabulares."""

from typing import Dict, Optional

import numpy as np
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import GaussianNB  # Q2: import no topo
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


# Constantes para tipos de tarefa (contexto obrigatório)
TASK_CLASSIFICATION = "classification"
TASK_REGRESSION = "regression"
TASK_QUERIES = "queries"
TASK_TYPES = [TASK_CLASSIFICATION, TASK_REGRESSION, TASK_QUERIES]


class MetaFeatureExtractor:
    def __init__(self, fast_landmarks: bool = False):
        self.fast_landmarks = fast_landmarks

    def extract(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epsilon: Optional[float] = None,
        task_type: Optional[str] = None,
    ) -> Dict[str, float]:
        """Extrai meta-features do dataset.
        
        Parameters
        ----------
        X : np.ndarray
            Matriz de features (n_samples, n_features)
        y : np.ndarray
            Vetor de labels
        epsilon : float, optional
            Orçamento de privacidade desejado pelo usuário (contexto obrigatório para DP)
        task_type : str, optional
            Tipo de tarefa: "classification", "regression", ou "queries"
        
        Returns
        -------
        Dict[str, float]
            Dicionário com meta-features estáticas + contexto
        """
        f = {}
        f.update(self._stat(X, y))
        f.update(self._info(X, y))
        f.update(self._land(X, y))
        f.update(self._dp_relevance(X, y))  # ML4
        f.update(self._categorical_signal(X, y))  # CAT1
        f.update(self._discrete_signal(X, y))     # DISC
        f.update(self._family_discriminators(X, y))  # MELHORIA: novos discriminadores
        
        # NOVO: Meta-features específicas para DP
        f.update(self._dp_clipping_signal(X, y))  # Razão max/mediana, curtose
        f.update(self._dp_sparsity_dimensionality(X, y))  # Esparsidade, rank efetivo
        f.update(self._dp_subgroup_entropy(X, y))  # Entropia de subgrupos
        
        # NOVO: Variáveis de contexto obrigatórias (concatenadas ao vetor X)
        f.update(self._context_features(epsilon, task_type))
        
        return f

    def _stat(self, X, y):
        n, d = X.shape
        nc = len(np.unique(y))
        ms = X.mean(0)
        ss = X.std(0) + 1e-9
        import warnings as _w
        with _w.catch_warnings():
            _w.simplefilter("ignore", RuntimeWarning)
            sk = stats.skew(X, axis=0, nan_policy="omit")
            ku = stats.kurtosis(X, axis=0, nan_policy="omit")
        sk = np.nan_to_num(sk, nan=0.0, posinf=0.0, neginf=0.0)
        ku = np.nan_to_num(ku, nan=0.0, posinf=0.0, neginf=0.0)
        rng = X.max(0) - X.min(0) + 1e-9

        # Contagem de colunas com poucos valores únicos (ratio_discrete original)
        unique_counts = np.array([len(np.unique(X[:, j])) for j in range(d)])
        nd = int(np.sum(unique_counts <= max(10, 0.05 * n)))

        # Novas features discriminadoras para família discrete vs continuous
        # ratio_integer_cols: % de colunas onde todos os valores são inteiros
        int_mask = np.array([
            np.all(np.isfinite(X[:, j])) and np.allclose(X[:, j], np.floor(X[:, j]))
            for j in range(d)
        ])
        ratio_integer_cols = float(int_mask.mean())

        # ratio_binary_cols: % de colunas com apenas 2 valores únicos
        ratio_binary_cols = float(np.mean(unique_counts == 2))

        # mean_log_unique_ratio: média de log(unique/n) — discrimina muito bem contínuo vs discreto
        # valores perto de 0 = discreto, perto de -inf (i.e., muito negativo) = binário, perto de log(1)=0 = contínuo
        log_unique = np.log(unique_counts.astype(float) / n + 1e-9)
        mean_log_unique_ratio = float(log_unique.mean())
        std_log_unique_ratio  = float(log_unique.std())

        # median_unique_per_col: mediana de valores únicos por coluna (valor absoluto)
        median_unique_per_col = float(np.median(unique_counts))
        max_unique_per_col    = float(np.max(unique_counts))

        return {
            "n_samples": n,
            "n_features": d,
            "n_classes": nc,
            "samples_per_feature": n / d,
            "samples_per_class": n / nc,
            "class_imbalance": np.std([np.sum(y == c) for c in np.unique(y)]) / n,
            "mean_mean": ms.mean(),
            "std_mean": ms.std(),
            "mean_std": ss.mean(),
            "std_std": ss.std(),
            "mean_skew": np.abs(sk).mean(),
            "max_skew": np.abs(sk).max(),
            "mean_kurt": ku.mean(),
            "max_kurt": ku.max(),
            "mean_range": rng.mean(),
            "max_range": rng.max(),
            "ratio_discrete": nd / d,
            "ratio_integer_cols": ratio_integer_cols,
            "ratio_binary_cols": ratio_binary_cols,
            "mean_log_unique_ratio": mean_log_unique_ratio,
            "std_log_unique_ratio": std_log_unique_ratio,
            "median_unique_per_col": median_unique_per_col,
            "max_unique_per_col": max_unique_per_col,
            "mean_corr": self._corr(X),
            "sparsity": np.sum(X == 0) / X.size,
            "coeff_var": (ss / (np.abs(ms) + 1e-9)).mean(),
        }

    def _corr(self, X):
        if X.shape[1] < 2:
            return 0.0
        try:
            # Remove colunas constantes (std=0) para evitar NaN no corrcoef
            mask = X.std(0) > 1e-9
            Xv = X[:, mask]
            if Xv.shape[1] < 2:
                return 0.0
            c = np.corrcoef(Xv.T)
            tri = np.triu(np.ones(c.shape, dtype=bool), k=1)
            vals = c[tri]
            vals = vals[np.isfinite(vals)]
            return float(np.abs(vals).mean()) if len(vals) else 0.0
        except Exception:
            return 0.0

    def _entropy(self, y):
        _, c = np.unique(y, return_counts=True)
        p = c / c.sum()
        return float(-np.sum(p * np.log2(p + 1e-9)))

    def _mi(self, x, y):
        bins = min(20, int(np.sqrt(len(x))))
        xb = np.digitize(x, np.histogram_bin_edges(x, bins=bins))
        hx = self._entropy(xb)
        hy = self._entropy(y)
        # PF8: joint entropy via ravel_multi_index — sem alocação de strings
        n_xb = int(xb.max()) + 1
        n_y  = int(y.max())  + 1
        joint_idx = np.ravel_multi_index([xb, y.astype(int)], dims=[n_xb, n_y])
        hxy = self._entropy(joint_idx)
        return max(0.0, hx + hy - hxy)

    def _info(self, X, y):
        mi = np.array([self._mi(X[:, j], y) for j in range(X.shape[1])])
        return {
            "mean_mi": mi.mean(),
            "max_mi": mi.max(),
            "min_mi": mi.min(),
            "std_mi": mi.std(),
            "class_entropy": self._entropy(y),
        }

    def _land(self, X, y):
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        Xs = StandardScaler().fit_transform(X)
        # PF2: n_jobs=1 aqui — evita nested parallelism quando chamado de Parallel externo
        n_jobs_cv = 1

        if self.fast_landmarks:
            clfs = {
                "lm_stump": DecisionTreeClassifier(max_depth=1, random_state=42),
                "lm_lin": LogisticRegression(max_iter=200, random_state=42),
            }
        else:
            clfs = {
                "lm_stump": DecisionTreeClassifier(max_depth=1, random_state=42),
                "lm_nb": GaussianNB(),  # Q2: usa import do topo
                "lm_1nn": KNeighborsClassifier(n_neighbors=1),
                "lm_lin": LogisticRegression(max_iter=200, random_state=42),
            }
        out = {}
        for k, clf in clfs.items():
            try:
                s = cross_val_score(clf, Xs, y, cv=cv, scoring="accuracy", n_jobs=n_jobs_cv)
                out[k] = float(s.mean())
            except Exception:
                out[k] = 0.5
        return out

    def _dp_relevance(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """ML4: meta-features relevantes para seleção de mecanismo DP."""
        n, d = X.shape
        sensitivity = X.max(0) - X.min(0)
        mu = X.mean(0)
        sigma = X.std(0) + 1e-9
        outlier_mask = np.abs(X - mu) > 3 * sigma

        result: Dict[str, float] = {
            "mean_sensitivity": float(sensitivity.mean()),
            "max_sensitivity": float(sensitivity.max()),
            "outlier_ratio": float(outlier_mask.mean()),
        }

        k = min(d, n, 10)
        if k >= 2:
            try:
                pca = PCA(n_components=k).fit(X)
                cum_var = np.cumsum(pca.explained_variance_ratio_)
                n95 = int(np.searchsorted(cum_var, 0.95)) + 1
                result["pca_intrinsic_dim_ratio"] = float(n95 / d)
                result["pca_top1_var"] = float(pca.explained_variance_ratio_[0])
            except Exception:
                result["pca_intrinsic_dim_ratio"] = 1.0
                result["pca_top1_var"] = 1.0 / max(d, 1)
        else:
            result["pca_intrinsic_dim_ratio"] = 1.0
            result["pca_top1_var"] = 1.0

        return result

    def _categorical_signal(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """CAT1: meta-features discriminadoras para datasets categóricos (Exponential).

        Datasets onde o Exponential vence tendem a ter muitas colunas de baixa
        cardinalidade, distribuição de classes uniforme e entropia alta por coluna.
        """
        n, d = X.shape
        unique_counts = np.array([len(np.unique(X[:, j])) for j in range(d)])

        # Proporção de colunas com baixa cardinalidade (≤ 10 valores distintos)
        ratio_low_cardinality = float(np.mean(unique_counts <= 10))

        # Proporção de colunas com cardinalidade muito baixa (≤ 5)
        ratio_very_low_cardinality = float(np.mean(unique_counts <= 5))

        # Entropia média por coluna de feature (alta → mais categórico)
        col_entropies = []
        for j in range(d):
            _, counts = np.unique(X[:, j], return_counts=True)
            p = counts / counts.sum()
            col_entropies.append(float(-np.sum(p * np.log2(p + 1e-9))))
        mean_col_entropy = float(np.mean(col_entropies)) if col_entropies else 0.0
        max_col_entropy = float(np.max(col_entropies)) if col_entropies else 0.0

        # Uniformidade da distribuição de classes alvo (classes balanceadas → Exponential)
        _, y_counts = np.unique(y, return_counts=True)
        y_probs = y_counts / y_counts.sum()
        target_entropy = float(-np.sum(y_probs * np.log2(y_probs + 1e-9)))
        max_target_entropy = float(np.log2(len(y_counts) + 1e-9))
        target_entropy_ratio = target_entropy / (max_target_entropy + 1e-9)

        # Dominância nominal: % de colunas onde um único valor ocupa ≥ 50% das linhas
        dominant = np.array([np.max(np.bincount(np.searchsorted(np.unique(X[:, j]), X[:, j]))) / n
                             for j in range(d)])
        ratio_dominant_cols = float(np.mean(dominant >= 0.5))

        return {
            "cat_ratio_low_cardinality": ratio_low_cardinality,
            "cat_ratio_very_low_cardinality": ratio_very_low_cardinality,
            "cat_mean_col_entropy": mean_col_entropy,
            "cat_max_col_entropy": max_col_entropy,
            "cat_target_entropy": target_entropy,
            "cat_target_entropy_ratio": target_entropy_ratio,
            "cat_ratio_dominant_cols": ratio_dominant_cols,
        }

    def _discrete_signal(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """DISC: meta-features discriminadoras para datasets discretos (Geometric).

        Datasets onde o Geometric vence têm colunas inteiras com range pequeno.
        Características: ratio_integer_cols alto, range por coluna pequeno, 
        poucos valores únicos relativos ao range possível (alta densidade discreta).
        """
        n, d = X.shape
        unique_counts = np.array([len(np.unique(X[:, j])) for j in range(d)])
        col_ranges = X.max(0) - X.min(0) + 1e-9

        # Máscara de colunas inteiras
        int_mask = np.array([
            np.all(np.isfinite(X[:, j])) and np.allclose(X[:, j], np.floor(X[:, j]))
            for j in range(d)
        ])
        ratio_integer_cols = float(int_mask.mean())

        # Range médio e máximo das colunas inteiras
        if int_mask.any():
            int_ranges = col_ranges[int_mask]
            disc_mean_int_range = float(int_ranges.mean())
            disc_max_int_range = float(int_ranges.max())
            # Proporção de colunas inteiras com range pequeno (≤ 50)
            disc_ratio_small_int_range = float(np.mean(int_ranges <= 50))
            # Densidade: unique/range (1.0 = todos os inteiros do range estão presentes)
            density = unique_counts[int_mask] / (int_ranges + 1)
            disc_mean_int_density = float(density.mean())
        else:
            disc_mean_int_range = float(col_ranges.mean())
            disc_max_int_range = float(col_ranges.max())
            disc_ratio_small_int_range = 0.0
            disc_mean_int_density = 0.0

        # Score composto: sinal forte de dataset discreto
        # Alto quando: muitas colunas inteiras + range pequeno + alta densidade
        disc_composite_score = float(
            ratio_integer_cols * disc_ratio_small_int_range * disc_mean_int_density
        )

        # Proporção de colunas com range ≤ 100 (discrimina discreto do contínuo)
        disc_ratio_small_range = float(np.mean(col_ranges <= 100))

        # Proporção de colunas onde todos os valores são não-negativos (típico de contagens)
        disc_ratio_nonneg_cols = float(np.mean(X.min(0) >= 0))

        return {
            "disc_ratio_integer_cols": ratio_integer_cols,
            "disc_mean_int_range": disc_mean_int_range,
            "disc_max_int_range": disc_max_int_range,
            "disc_ratio_small_int_range": disc_ratio_small_int_range,
            "disc_mean_int_density": disc_mean_int_density,
            "disc_composite_score": disc_composite_score,
            "disc_ratio_small_range": disc_ratio_small_range,
            "disc_ratio_nonneg_cols": disc_ratio_nonneg_cols,
        }

    def _gaussian_signal(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """GAUSS: meta-features discriminadoras para GaussianAnalytic vs Laplace.

        GaussianAnalytic (Balle-Wang) vence Laplace principalmente em datasets de
        alta dimensionalidade, com variância distribuída entre muitas features e
        baixa sensibilidade normalizada por feature.
        """
        n, d = X.shape

        # Sinal principal: log(n_features) normalizado — GA domina em alta dimensão
        ga_log_features = float(np.log1p(d))

        # Proporção de features verdadeiramente contínuas (muitos valores únicos)
        unique_counts = np.array([len(np.unique(X[:, j])) for j in range(d)])
        ga_ratio_continuous_cols = float(np.mean(unique_counts > min(20, 0.1 * n)))

        # Sensibilidade normalizada: baixa → GA melhor (ruído gaussiano menor que laplaciano)
        sensitivity = X.max(0) - X.min(0) + 1e-9
        ga_mean_norm_sensitivity = float((sensitivity / (X.std(0) + 1e-9)).mean())
        ga_max_norm_sensitivity  = float((sensitivity / (X.std(0) + 1e-9)).max())

        # Spread do PCA: GA vence quando variância está distribuída entre muitas componentes
        # (pca_top1_var baixo = spread alto = GA favorecido)
        # Já temos pca_top1_var e pca_intrinsic_dim_ratio em _dp_relevance.
        # Adicionamos o complemento como feature explícita para o GAUSS prefilter.
        k = min(d, n, 10)
        if k >= 2:
            try:
                from sklearn.decomposition import PCA as _PCA
                pca = _PCA(n_components=k).fit(X)
                ga_pca_spread = float(1.0 - pca.explained_variance_ratio_[0])
                ga_pca_n50pct = float(
                    int(np.searchsorted(np.cumsum(pca.explained_variance_ratio_), 0.50)) + 1
                ) / d
            except Exception:
                ga_pca_spread = 0.5
                ga_pca_n50pct = 0.5
        else:
            ga_pca_spread = 0.0
            ga_pca_n50pct = 1.0 / max(d, 1)

        # Score composto: sinal direto de "alta dimensionalidade + variância distribuída"
        ga_composite_score = float(ga_log_features / 10.0 * ga_ratio_continuous_cols * ga_pca_spread)

        return {
            "ga_log_features": ga_log_features,
            "ga_ratio_continuous_cols": ga_ratio_continuous_cols,
            "ga_mean_norm_sensitivity": ga_mean_norm_sensitivity,
            "ga_max_norm_sensitivity": ga_max_norm_sensitivity,
            "ga_pca_spread": ga_pca_spread,
            "ga_pca_n50pct": ga_pca_n50pct,
            "ga_composite_score": ga_composite_score,
        }

    def _family_discriminators(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """MELHORIA: meta-features adicionais para discriminação de família.
        
        Estas features ajudam a distinguir quando usar:
        - Continuous (Laplace, Gaussian): dados contínuos com alta cardinalidade
        - Discrete (Geometric): dados inteiros com range pequeno
        - Categorical (Exponential): dados com poucos valores únicos
        """
        n, d = X.shape
        
        # Análise de tipo de dados por coluna
        unique_counts = np.array([len(np.unique(X[:, j])) for j in range(d)])
        col_ranges = X.max(0) - X.min(0) + 1e-9
        
        # Máscara de colunas inteiras
        int_mask = np.array([
            np.all(np.isfinite(X[:, j])) and np.allclose(X[:, j], np.floor(X[:, j]))
            for j in range(d)
        ])
        
        # Score de "continuidade": 1.0 = totalmente contínuo, 0.0 = totalmente discreto
        # Baseado em: alta cardinalidade + valores não-inteiros
        continuity_per_col = unique_counts / n  # 0 a 1
        continuity_score = float(np.mean(continuity_per_col) * (1 - int_mask.mean()))
        
        # Score de "discretude": 1.0 = totalmente discreto, 0.0 = contínuo
        # Baseado em: colunas inteiras + range pequeno
        small_range_mask = col_ranges <= 100
        discreteness_score = float(int_mask.mean() * small_range_mask.mean())
        
        # Score de "categoricidade": 1.0 = categórico, 0.0 = não-categórico
        # Baseado em: baixa cardinalidade (<= 10 valores únicos)
        categorical_mask = unique_counts <= 10
        categoricity_score = float(categorical_mask.mean())
        
        # Gini impurity médio por coluna de feature
        # Alta impureza = mais uniforme = melhor para Exponential
        gini_per_col = []
        for j in range(d):
            _, counts = np.unique(X[:, j], return_counts=True)
            p = counts / counts.sum()
            gini = 1.0 - np.sum(p ** 2)
            gini_per_col.append(gini)
        mean_gini = float(np.mean(gini_per_col)) if gini_per_col else 0.0
        
        # Proporção de colunas com distribuição uniforme (alta entropia)
        uniform_threshold = 0.9 * np.log2(10)  # ~90% da entropia máxima para 10 valores
        uniform_mask = []
        for j in range(d):
            _, counts = np.unique(X[:, j], return_counts=True)
            p = counts / counts.sum()
            entropy = -np.sum(p * np.log2(p + 1e-9))
            max_entropy = np.log2(len(counts) + 1e-9)
            uniform_mask.append(entropy >= 0.9 * max_entropy if max_entropy > 0 else False)
        ratio_uniform_cols = float(np.mean(uniform_mask))
        
        # One-hot encoding detection: muitas colunas binárias esparsas
        binary_mask = unique_counts == 2
        if binary_mask.any():
            binary_cols = X[:, binary_mask]
            sparsity = np.mean(binary_cols == 0, axis=0)
            is_onehot = float(np.mean(sparsity >= 0.7))
        else:
            is_onehot = 0.0
        
        # Score composto de família
        # Usa soft-max para normalizar os três scores
        scores = np.array([continuity_score, discreteness_score, categoricity_score])
        exp_scores = np.exp(scores * 3)  # Temperatura
        family_proba = exp_scores / exp_scores.sum()
        
        # ========== NOVAS FEATURES v16 ==========
        
        # Features para melhorar recall de Exponential (categorical)
        max_cardinality = int(unique_counts.max()) if d > 0 else 0
        pct_cols_under_10 = float((unique_counts < 10).mean())
        pct_cols_under_5 = float((unique_counts < 5).mean())
        mean_cardinality = float(unique_counts.mean())
        
        # Razão valor/linha por coluna (baixa = mais categórico)
        value_count_ratios = unique_counts / n
        mean_value_ratio = float(value_count_ratios.mean())
        min_value_ratio = float(value_count_ratios.min()) if d > 0 else 0.0
        
        # Features para melhorar recall de GaussianAnalytic
        # GA performa melhor em alta dimensionalidade e features correlacionadas
        feature_to_sample_ratio = d / n
        is_high_dim = float(feature_to_sample_ratio > 0.1)  # mais features que 10% das amostras
        
        # Correlação média entre features (alta = bom para GA)
        try:
            if d >= 2 and n >= 10:
                corr_matrix = np.corrcoef(X.T)
                # Pega triângulo superior (sem diagonal)
                upper_tri = np.triu_indices(d, k=1)
                correlations = np.abs(corr_matrix[upper_tri])
                mean_feature_corr = float(np.nanmean(correlations))
                max_feature_corr = float(np.nanmax(correlations))
            else:
                mean_feature_corr = 0.0
                max_feature_corr = 0.0
        except Exception:
            mean_feature_corr = 0.0
            max_feature_corr = 0.0
        
        # PCA: variância explicada pelos primeiros componentes
        try:
            if d >= 3 and n >= 10:
                X_scaled = (X - X.mean(0)) / (X.std(0) + 1e-9)
                pca = PCA(n_components=min(3, d, n))
                pca.fit(X_scaled)
                pca_var_top3 = float(sum(pca.explained_variance_ratio_[:3]))
            else:
                pca_var_top3 = 1.0
        except Exception:
            pca_var_top3 = 1.0
        
        # Score composto para GA (alta dim + alta correlação + variância concentrada)
        ga_score = (
            0.3 * is_high_dim + 
            0.3 * mean_feature_corr + 
            0.4 * (1 - pca_var_top3)  # Se variância NÃO está concentrada = muitas dimensões úteis
        )
        
        return {
            "fam_continuity_score": continuity_score,
            "fam_discreteness_score": discreteness_score,
            "fam_categoricity_score": categoricity_score,
            "fam_mean_gini": mean_gini,
            "fam_ratio_uniform_cols": ratio_uniform_cols,
            "fam_is_onehot": is_onehot,
            "fam_p_continuous": float(family_proba[0]),
            "fam_p_discrete": float(family_proba[1]),
            "fam_p_categorical": float(family_proba[2]),
            # Novas features v16
            "fam_max_cardinality": max_cardinality,
            "fam_pct_cols_under_10": pct_cols_under_10,
            "fam_pct_cols_under_5": pct_cols_under_5,
            "fam_mean_cardinality": mean_cardinality,
            "fam_mean_value_ratio": mean_value_ratio,
            "fam_min_value_ratio": min_value_ratio,
            "fam_feature_to_sample_ratio": feature_to_sample_ratio,
            "fam_is_high_dim": is_high_dim,
            "fam_mean_feature_corr": mean_feature_corr,
            "fam_max_feature_corr": max_feature_corr,
            "fam_pca_var_top3": pca_var_top3,
            "fam_ga_score": ga_score,
        }

    # =========================================================================
    # NOVAS META-FEATURES ESPECÍFICAS PARA DIFFERENTIAL PRIVACY (DP)
    # =========================================================================

    def _dp_clipping_signal(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Meta-features para prever impacto do clipping em DP.
        
        O clipping (limitação de valores extremos) é crítico em DP para controlar
        a sensibilidade. Datasets com outliers severos sofrem mais com clipping.
        
        Features:
        - max_median_ratio: razão máximo/mediana por coluna (outlier severity)
        - mean/max_kurtosis: curtose indica caudas pesadas (mais outliers)
        - iqr_ratio: razão IQR/range (baixo = outliers dominam o range)
        - clipping_loss_estimate: perda estimada com clipping 3σ
        """
        n, d = X.shape
        
        # Razão máximo/mediana por coluna (outlier severity indicator)
        medians = np.median(X, axis=0)
        maxs = np.abs(X).max(axis=0)
        # Evita divisão por zero para colunas constantes
        safe_medians = np.where(np.abs(medians) > 1e-9, np.abs(medians), 1.0)
        max_median_ratios = maxs / safe_medians
        
        dp_mean_max_median_ratio = float(np.mean(max_median_ratios))
        dp_max_max_median_ratio = float(np.max(max_median_ratios))
        
        # Curtose (kurtosis) - caudas pesadas indicam mais outliers
        import warnings as _w
        with _w.catch_warnings():
            _w.simplefilter("ignore", RuntimeWarning)
            kurtosis = stats.kurtosis(X, axis=0, nan_policy="omit")
        kurtosis = np.nan_to_num(kurtosis, nan=0.0, posinf=10.0, neginf=-2.0)
        
        dp_mean_kurtosis = float(np.mean(kurtosis))
        dp_max_kurtosis = float(np.max(kurtosis))
        dp_std_kurtosis = float(np.std(kurtosis))
        # Proporção de colunas com curtose alta (> 3, indicando caudas pesadas)
        dp_ratio_heavy_tails = float(np.mean(kurtosis > 3))
        
        # Razão IQR/Range - baixo valor indica que outliers dominam o range
        q1 = np.percentile(X, 25, axis=0)
        q3 = np.percentile(X, 75, axis=0)
        iqr = q3 - q1
        col_range = X.max(axis=0) - X.min(axis=0) + 1e-9
        iqr_ratio = iqr / col_range
        
        dp_mean_iqr_ratio = float(np.mean(iqr_ratio))
        dp_min_iqr_ratio = float(np.min(iqr_ratio))
        
        # Estimativa de perda por clipping 3σ
        # Conta proporção de valores que seriam clipados
        mu = X.mean(axis=0)
        sigma = X.std(axis=0) + 1e-9
        clipping_mask = np.abs(X - mu) > 3 * sigma
        dp_clipping_loss_estimate = float(clipping_mask.mean())
        
        # Sensibilidade global estimada (range normalizado)
        # Importante: em DP, sensibilidade = max diferença que um único registro causa
        dp_global_sensitivity_norm = float(col_range.mean() / (sigma.mean() + 1e-9))
        
        return {
            "dp_mean_max_median_ratio": dp_mean_max_median_ratio,
            "dp_max_max_median_ratio": dp_max_max_median_ratio,
            "dp_mean_kurtosis": dp_mean_kurtosis,
            "dp_max_kurtosis": dp_max_kurtosis,
            "dp_std_kurtosis": dp_std_kurtosis,
            "dp_ratio_heavy_tails": dp_ratio_heavy_tails,
            "dp_mean_iqr_ratio": dp_mean_iqr_ratio,
            "dp_min_iqr_ratio": dp_min_iqr_ratio,
            "dp_clipping_loss_estimate": dp_clipping_loss_estimate,
            "dp_global_sensitivity_norm": dp_global_sensitivity_norm,
        }

    def _dp_sparsity_dimensionality(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Meta-features de esparsidade e dimensionalidade efetiva.
        
        Em DP, alta dimensionalidade e esparsidade podem causar colapso de utilidade:
        - Ruído acumula com mais features
        - Dados esparsos têm menos sinal para preservar
        
        Features:
        - zero_ratio: proporção de zeros (esparsidade)
        - effective_rank: rank numérico via SVD (dimensionalidade intrínseca)
        - condition_number: número de condição da matriz (estabilidade numérica)
        """
        n, d = X.shape
        
        # Esparsidade: proporção de zeros
        dp_zero_ratio = float(np.sum(X == 0) / X.size)
        
        # Esparsidade por coluna (para detectar colunas muito esparsas)
        col_sparsity = np.mean(X == 0, axis=0)
        dp_max_col_sparsity = float(np.max(col_sparsity))
        dp_mean_col_sparsity = float(np.mean(col_sparsity))
        # Proporção de colunas com >50% zeros
        dp_ratio_sparse_cols = float(np.mean(col_sparsity > 0.5))
        
        # Rank numérico e dimensionalidade efetiva via SVD
        try:
            # Centraliza e normaliza para estabilidade numérica
            X_centered = X - X.mean(axis=0)
            X_scaled = X_centered / (X.std(axis=0) + 1e-9)
            
            # SVD truncada para eficiência
            k = min(d, n, 50)
            from scipy.linalg import svd
            _, s, _ = svd(X_scaled, full_matrices=False)
            s = s[:k]
            
            # Rank numérico: conta valores singulares > threshold
            threshold = max(n, d) * np.finfo(float).eps * s[0]
            dp_numerical_rank = int(np.sum(s > threshold))
            dp_numerical_rank_ratio = float(dp_numerical_rank / d)
            
            # Dimensionalidade efetiva (entropia dos valores singulares normalizados)
            s_norm = s / (s.sum() + 1e-9)
            effective_dim = float(np.exp(-np.sum(s_norm * np.log(s_norm + 1e-9))))
            dp_effective_dim_ratio = float(effective_dim / d)
            
            # Número de condição (razão maior/menor valor singular)
            # Alto número de condição = matriz mal condicionada = DP mais difícil
            s_nonzero = s[s > 1e-9]
            if len(s_nonzero) >= 2:
                dp_condition_number = float(s_nonzero[0] / s_nonzero[-1])
                dp_log_condition_number = float(np.log10(dp_condition_number + 1))
            else:
                dp_condition_number = 1.0
                dp_log_condition_number = 0.0
                
            # Proporção de variância explicada pelos top-k componentes
            var_explained = s**2 / (np.sum(s**2) + 1e-9)
            dp_var_top1 = float(var_explained[0])
            dp_var_top5 = float(np.sum(var_explained[:min(5, len(var_explained))]))
            
        except Exception:
            dp_numerical_rank = d
            dp_numerical_rank_ratio = 1.0
            dp_effective_dim_ratio = 1.0
            dp_condition_number = 1.0
            dp_log_condition_number = 0.0
            dp_var_top1 = 1.0 / max(d, 1)
            dp_var_top5 = min(5, d) / max(d, 1)
        
        return {
            "dp_zero_ratio": dp_zero_ratio,
            "dp_max_col_sparsity": dp_max_col_sparsity,
            "dp_mean_col_sparsity": dp_mean_col_sparsity,
            "dp_ratio_sparse_cols": dp_ratio_sparse_cols,
            "dp_numerical_rank": dp_numerical_rank,
            "dp_numerical_rank_ratio": dp_numerical_rank_ratio,
            "dp_effective_dim_ratio": dp_effective_dim_ratio,
            "dp_condition_number": dp_condition_number,
            "dp_log_condition_number": dp_log_condition_number,
            "dp_var_top1": dp_var_top1,
            "dp_var_top5": dp_var_top5,
        }

    def _dp_subgroup_entropy(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Meta-features de entropia de subgrupos para mitigar Disparate Impact.
        
        O ruído DP afeta desproporcionalmente grupos minoritários. Estas features
        ajudam a prever quando isso será um problema:
        - Desbalanceamento de classes
        - Tamanho do menor grupo
        - Entropia da distribuição de grupos
        
        Features:
        - minority_class_ratio: proporção da menor classe
        - class_imbalance_ratio: razão maior/menor classe
        - class_entropy: entropia da distribuição de classes
        - gini_impurity: impureza de Gini das classes
        """
        n = len(y)
        
        # Contagem de classes
        classes, counts = np.unique(y, return_counts=True)
        n_classes = len(classes)
        
        if n_classes < 2:
            return {
                "dp_minority_class_ratio": 1.0,
                "dp_class_imbalance_ratio": 1.0,
                "dp_class_entropy": 0.0,
                "dp_class_entropy_normalized": 0.0,
                "dp_gini_impurity": 0.0,
                "dp_minority_class_size": n,
                "dp_majority_class_ratio": 1.0,
                "dp_effective_n_classes": 1.0,
            }
        
        # Proporções de cada classe
        probs = counts / n
        
        # Proporção da menor classe (grupo minoritário)
        dp_minority_class_ratio = float(np.min(probs))
        dp_minority_class_size = int(np.min(counts))
        
        # Proporção da maior classe
        dp_majority_class_ratio = float(np.max(probs))
        
        # Razão de desbalanceamento (maior/menor)
        dp_class_imbalance_ratio = float(np.max(counts) / (np.min(counts) + 1))
        
        # Entropia de Shannon das classes
        dp_class_entropy = float(-np.sum(probs * np.log2(probs + 1e-9)))
        # Entropia normalizada (0-1, onde 1 = distribuição uniforme)
        max_entropy = np.log2(n_classes)
        dp_class_entropy_normalized = float(dp_class_entropy / (max_entropy + 1e-9))
        
        # Impureza de Gini
        dp_gini_impurity = float(1.0 - np.sum(probs ** 2))
        
        # Número efetivo de classes (entropia exponenciada)
        dp_effective_n_classes = float(np.exp(dp_class_entropy * np.log(2)))
        
        # Score composto de risco de Disparate Impact
        # Alto quando: classes muito desbalanceadas + grupo minoritário pequeno
        dp_disparate_impact_risk = float(
            (1 - dp_minority_class_ratio) * (1 - dp_class_entropy_normalized)
        )
        
        return {
            "dp_minority_class_ratio": dp_minority_class_ratio,
            "dp_minority_class_size": dp_minority_class_size,
            "dp_majority_class_ratio": dp_majority_class_ratio,
            "dp_class_imbalance_ratio": dp_class_imbalance_ratio,
            "dp_class_entropy": dp_class_entropy,
            "dp_class_entropy_normalized": dp_class_entropy_normalized,
            "dp_gini_impurity": dp_gini_impurity,
            "dp_effective_n_classes": dp_effective_n_classes,
            "dp_disparate_impact_risk": dp_disparate_impact_risk,
        }

    def _context_features(
        self,
        epsilon: Optional[float] = None,
        task_type: Optional[str] = None,
    ) -> Dict[str, float]:
        """Variáveis de contexto obrigatórias concatenadas ao vetor de features.
        
        O meta-modelo NÃO deve tentar adivinhar o melhor algoritmo olhando apenas
        para o dataset. O orçamento de privacidade (epsilon) e o tipo de tarefa
        são informações críticas que o usuário deve fornecer.
        
        Parameters
        ----------
        epsilon : float
            Orçamento de privacidade desejado. Valores típicos:
            - 0.1-0.5: privacidade forte (muito ruído)
            - 1.0: padrão (ruído moderado)
            - 5.0-10.0: privacidade fraca (pouco ruído)
            
        task_type : str
            Tipo de tarefa: "classification", "regression", ou "queries"
        """
        features = {}
        
        # Epsilon (orçamento de privacidade)
        if epsilon is not None:
            features["ctx_epsilon"] = float(epsilon)
            features["ctx_log_epsilon"] = float(np.log(epsilon + 1e-9))
            # Buckets de epsilon para facilitar aprendizado
            features["ctx_epsilon_low"] = float(epsilon < 1.0)
            features["ctx_epsilon_medium"] = float(1.0 <= epsilon < 5.0)
            features["ctx_epsilon_high"] = float(epsilon >= 5.0)
        else:
            # Valores padrão quando não especificado (epsilon = 1.0)
            features["ctx_epsilon"] = 1.0
            features["ctx_log_epsilon"] = 0.0
            features["ctx_epsilon_low"] = 0.0
            features["ctx_epsilon_medium"] = 1.0
            features["ctx_epsilon_high"] = 0.0
        
        # Tipo de tarefa (one-hot encoding)
        if task_type is not None and task_type in TASK_TYPES:
            features["ctx_task_classification"] = float(task_type == TASK_CLASSIFICATION)
            features["ctx_task_regression"] = float(task_type == TASK_REGRESSION)
            features["ctx_task_queries"] = float(task_type == TASK_QUERIES)
        else:
            # Padrão: classificação
            features["ctx_task_classification"] = 1.0
            features["ctx_task_regression"] = 0.0
            features["ctx_task_queries"] = 0.0
        
        return features

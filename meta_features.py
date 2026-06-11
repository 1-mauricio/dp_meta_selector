"""Extração de meta-features de datasets tabulares."""

from typing import Dict

import numpy as np
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import GaussianNB  # Q2: import no topo
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


class MetaFeatureExtractor:
    def __init__(self, fast_landmarks: bool = False):
        self.fast_landmarks = fast_landmarks

    def extract(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        f = {}
        f.update(self._stat(X, y))
        f.update(self._info(X, y))
        f.update(self._land(X, y))
        f.update(self._dp_relevance(X, y))  # ML4
        f.update(self._categorical_signal(X, y))  # CAT1
        f.update(self._discrete_signal(X, y))     # DISC
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

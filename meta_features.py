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

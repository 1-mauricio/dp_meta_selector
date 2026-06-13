"""Avaliação de utilidade pós-DP (perfis de custo, cache, screening)."""

import hashlib
import logging
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional

_log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .baseline_store import BaselineRegistry, BaselineStore

import joblib
import numpy as np
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .applicator import DPApplicator
from .calibration import COMPARISON_EPSILON, DELTA_DEFAULT
from .config import DEFAULT_CACHE_DIR, FINGERPRINT_SAMPLE_SIZE
from .mechanisms import (
    FAMILY_OF,
    MECHANISM_NAMES,
    SCREENING_MECHANISMS,
)


@dataclass(frozen=True)
class UtilityProfile:
    """Configuração do protocolo de utilidade (proxy vs avaliação forte)."""

    name: str
    clf: str = "logreg"
    n_estimators: int = 30
    cv_splits: int = 3
    n_runs: int = 1
    use_screening: bool = True
    refine_top_k: int = 3
    parallel: bool = True

    def cache_key(self) -> str:
        return (
            f"{self.name}|{self.clf}|{self.n_estimators}|{self.cv_splits}|"
            f"{self.n_runs}|{self.use_screening}|{self.refine_top_k}"
        )


META_FAST_PROFILE = UtilityProfile(
    name="meta_fast",
    clf="logreg",
    n_estimators=0,
    cv_splits=3,
    n_runs=1,
    use_screening=True,
    refine_top_k=3,
    parallel=True,
)

# Perfil de rotulagem alinhado com EVAL_FAST: mesmos clf/cv/runs → labels menos ruidosas.
META_ALIGNED_PROFILE = UtilityProfile(
    name="meta_aligned",
    clf="rf",
    n_estimators=30,
    cv_splits=3,
    n_runs=2,
    use_screening=True,
    refine_top_k=5,
    parallel=True,
)

EVAL_FAST_PROFILE = UtilityProfile(
    name="eval_fast",
    clf="rf",
    n_estimators=30,
    cv_splits=3,
    n_runs=2,
    use_screening=True,
    refine_top_k=5,
    parallel=True,
)

EVAL_FULL_PROFILE = UtilityProfile(
    name="eval_full",
    clf="rf",
    n_estimators=50,
    cv_splits=5,
    n_runs=3,
    use_screening=False,
    refine_top_k=0,
    parallel=True,
)

# Perfil estável para construção de meta-dataset de alta qualidade.
# Usa n_runs=5 para calcular a média sobre múltiplas sementes aleatórias,
# eliminando o ruído estocástico da DP dos labels de treino do metamodelo.
# Recomendado quando se quer labels de treino confiáveis (mais lento que META_FAST).
META_STABLE_PROFILE = UtilityProfile(
    name="meta_stable",
    clf="rf",
    n_estimators=30,
    cv_splits=3,
    n_runs=5,
    use_screening=True,
    refine_top_k=3,
    parallel=True,
)


def _make_utility_pipeline(profile: UtilityProfile) -> Pipeline:
    if profile.clf == "logreg":
        clf = LogisticRegression(max_iter=300, random_state=42)
    else:
        n = max(10, profile.n_estimators)
        clf = RandomForestClassifier(n_estimators=n, random_state=42, n_jobs=1)
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])


def _data_fingerprint(X: np.ndarray, y: np.ndarray) -> str:
    """Q5: fingerprint robusto com sample + estatísticas por coluna + shape completo."""
    h = hashlib.sha256()
    n = min(FINGERPRINT_SAMPLE_SIZE, len(y))
    if n > 0:
        # sample igualmente espaçado (determinístico)
        idx = np.linspace(0, len(y) - 1, n, dtype=int)
        h.update(np.ascontiguousarray(X[idx]).tobytes())
        h.update(y[idx].tobytes())
    # estatísticas globais por coluna — diferencia datasets com mesmo sample
    h.update(X.min(axis=0).tobytes())
    h.update(X.max(axis=0).tobytes())
    h.update(X.mean(axis=0).tobytes())
    h.update(f"{X.shape}|{len(np.unique(y))}|{int(y.sum())}".encode())
    return h.hexdigest()[:24]


class UtilityResultCache:
    """Cache em disco de acurácias pós-DP por (dados, mecanismo, perfil).
    
    PF7: L1 in-memory dict como camada rápida antes do disco.
    """

    def __init__(
        self,
        cache_dir: Path = DEFAULT_CACHE_DIR,
        enabled: bool = True,
        max_age_days: Optional[float] = None,  # Q3: TTL
    ):
        self.enabled = enabled
        self.cache_dir = Path(cache_dir)
        self.max_age_days = max_age_days
        if enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0
        self._lock = threading.Lock()
        self._mem: Dict[str, float] = {}  # PF7: L1 cache in-memory

    def _inc_hits(self) -> None:
        with self._lock:
            self.hits += 1

    def _inc_misses(self) -> None:
        with self._lock:
            self.misses += 1

    def _cache_key(self, fp: str, mechanism: str, profile: UtilityProfile) -> str:
        # Inclui epsilon de comparação na chave para auto-invalidar quando mudar
        return hashlib.md5(
            f"{fp}|{mechanism}|{profile.cache_key()}|eps={COMPARISON_EPSILON:.4f}".encode()
        ).hexdigest()

    def _path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.joblib"

    def _is_expired(self, p: Path) -> bool:
        """Q3: retorna True se o arquivo for mais antigo que max_age_days."""
        if self.max_age_days is None:
            return False
        mtime = datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc)
        return datetime.now(tz=timezone.utc) - mtime > timedelta(days=self.max_age_days)

    def get(self, fp: str, mechanism: str, profile: UtilityProfile) -> Optional[float]:
        if not self.enabled:
            return None
        key = self._cache_key(fp, mechanism, profile)
        # PF7: verifica L1 primeiro — zero I/O
        with self._lock:
            if key in self._mem:
                self.hits += 1
                return self._mem[key]
        p = self._path(key)
        if p.is_file():
            if self._is_expired(p):
                p.unlink(missing_ok=True)
                return None
            val = float(joblib.load(p))
            with self._lock:
                self._mem[key] = val  # popula L1
                self.hits += 1
            return val
        return None

    def set(self, fp: str, mechanism: str, profile: UtilityProfile, score: float) -> None:
        if not self.enabled:
            return
        key = self._cache_key(fp, mechanism, profile)
        with self._lock:
            self._mem[key] = float(score)  # PF7: grava em L1 imediatamente
        joblib.dump(float(score), self._path(key))  # persiste em disco

    def prune(self) -> int:
        """Q3: remove entradas expiradas; retorna número de arquivos removidos."""
        if not self.enabled or self.max_age_days is None:
            return 0
        removed = 0
        for p in self.cache_dir.glob("*.joblib"):
            if self._is_expired(p):
                p.unlink(missing_ok=True)
                removed += 1
        if removed:
            _log.info("Cache: %d entradas expiradas removidas.", removed)
        return removed

    def summary(self) -> str:
        if not self.enabled:
            return "cache=off"
        total = self.hits + self.misses
        if total == 0:
            return "cache=empty"
        ttl_str = f" ttl={self.max_age_days}d" if self.max_age_days else ""
        mem_str = f" mem={len(self._mem)}"
        return f"cache hits={self.hits} misses={self.misses} ({100*self.hits/total:.0f}%){ttl_str}{mem_str}"


class DPUtilityEvaluator:
    def __init__(
        self,
        delta: float = DELTA_DEFAULT,
        profile: UtilityProfile = META_FAST_PROFILE,
        cache: Optional[UtilityResultCache] = None,
        n_runs: Optional[int] = None,
        baseline_store: Optional["BaselineStore"] = None,
        baseline_registry: Optional["BaselineRegistry"] = None,
        baseline_id: Optional[str] = None,
        screening_mechanisms: Optional[List[str]] = None,  # P3
    ):
        self.applicator = DPApplicator(delta=delta)
        if n_runs is not None:
            self.profile = UtilityProfile(
                name=profile.name,
                clf=profile.clf,
                n_estimators=profile.n_estimators,
                cv_splits=profile.cv_splits,
                n_runs=n_runs,
                use_screening=profile.use_screening,
                refine_top_k=profile.refine_top_k,
                parallel=profile.parallel,
            )
        else:
            self.profile = profile
        self.cache = cache or UtilityResultCache(enabled=True)
        self.baseline_store = baseline_store
        self.baseline_registry = baseline_registry
        self._default_baseline_id = baseline_id
        self._screening_mechanisms: List[str] = list(
            screening_mechanisms or SCREENING_MECHANISMS
        )

    def _cv_score(
        self, X: np.ndarray, y: np.ndarray, profile: Optional[UtilityProfile] = None,
        n_jobs: int = 1,
    ) -> float:
        """Métrica composta: f1_macro + balanced_accuracy.

        f1_macro é mais discriminativa que balanced_accuracy quando mecanismos
        introduzem ruído assimétrico por classe.
        """
        profile = profile or self.profile
        n_splits = min(profile.cv_splits, int(np.min(np.bincount(y))))
        if n_splits < 2:
            return 0.0
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        pipe = _make_utility_pipeline(profile)
        try:
            f1  = float(cross_val_score(pipe, X, y, cv=cv, scoring="f1_macro",          n_jobs=n_jobs).mean())
            bal = float(cross_val_score(pipe, X, y, cv=cv, scoring="balanced_accuracy", n_jobs=n_jobs).mean())
            return 0.6 * f1 + 0.4 * bal
        except Exception:
            return 0.0

    @staticmethod
    def _histogram_score(X_orig: np.ndarray, X_dp: np.ndarray, n_bins: int = 20) -> float:
        """Mede preservação de histograma coluna a coluna (Opção 3).

        Discrimina Geometric (caudas finas = menos outliers) de Laplace
        (caudas pesadas = mais valores extremos) ao mesmo nível médio de ruído.
        Retorna valor em [0, 1] onde 1 = histogramas idênticos.
        """
        scores = []
        for j in range(X_orig.shape[1]):
            col_orig = X_orig[:, j]
            col_dp   = X_dp[:, j]
            c_min, c_max = col_orig.min(), col_orig.max()
            if c_max - c_min < 1e-9:
                scores.append(1.0)
                continue
            edges = np.linspace(c_min, c_max + 1e-9, n_bins + 1)
            hist_orig, _ = np.histogram(col_orig, bins=edges)
            hist_dp,   _ = np.histogram(np.clip(col_dp, c_min, c_max), bins=edges)
            total = hist_orig.sum() + 1e-9
            # Usa intersecção de histogramas (métrica mais robusta que MAE)
            intersection = np.minimum(hist_orig, hist_dp).sum() / total
            scores.append(float(intersection))
        return float(np.mean(scores)) if scores else 0.0

    def baseline(
        self,
        X,
        y,
        profile: Optional[UtilityProfile] = None,
        dataset_id: Optional[str] = None,
        baseline_id: Optional[str] = None,
        fp: Optional[str] = None,  # PF5: aceita fingerprint pré-calculado
    ) -> float:
        profile = profile or self.profile
        bid = baseline_id or self._default_baseline_id
        if bid is None and self.baseline_registry is not None:
            bid = self.baseline_registry.resolve_id(profile)

        if self.baseline_store is not None and self.baseline_store.enabled and bid is not None:
            from sklearn.preprocessing import LabelEncoder

            y_enc = LabelEncoder().fit_transform(y)
            fp = fp or _data_fingerprint(X, y_enc)  # PF5: reutiliza se já calculado
            ds_id = dataset_id or fp
            cached = self.baseline_store.get(ds_id, bid, fp)
            if cached is not None:
                return cached
            self.baseline_store._inc_misses()  # B1: thread-safe
            score = self._cv_score(X, y_enc, profile)
            self.baseline_store.set(ds_id, bid, fp, profile, score)
            return score

        return self._cv_score(X, y, profile)

    def _score_mechanism(
        self,
        mechanism: str,
        X: np.ndarray,
        y: np.ndarray,
        profile: UtilityProfile,
        fp: str,
    ) -> float:
        cached = self.cache.get(fp, mechanism, profile)
        if cached is not None:
            return cached
        self.cache._inc_misses()
        accs = []
        for _run in range(profile.n_runs):
            try:
                X_dp = self.applicator.apply(mechanism, X)
                clf_score  = self._cv_score(X_dp, y, profile)
                hist_score = self._histogram_score(X, X_dp)
                # Blend: classificador + preservação de histograma
                # O histograma discrimina Geometric (quase sem perda) de Laplace
                # (ruído massivo com ε=1.0) em dados discretos/inteiros
                accs.append(0.6 * clf_score + 0.4 * hist_score)
            except Exception as exc:
                _log.warning(
                    "Mecanismo '%s' falhou (run %d): %s", mechanism, _run + 1, exc
                )
                accs.append(0.0)
        score = float(np.mean(accs)) if accs else 0.0
        self.cache.set(fp, mechanism, profile, score)
        return score

    def _evaluate_mechanisms(
        self,
        mechanisms: List[str],
        X: np.ndarray,
        y: np.ndarray,
        profile: UtilityProfile,
        fp: str,
    ) -> Dict[str, float]:
        if profile.parallel and len(mechanisms) > 1:
            # PF2: prefer="threads" — mecanismos são I/O+compute bound, threads evitam GIL issues
            # n_jobs limitado a min(len, cores) para não oversaturar
            pairs = Parallel(n_jobs=-1, prefer="threads")(
                delayed(self._score_mechanism)(m, X, y, profile, fp) for m in mechanisms
            )
            return dict(zip(mechanisms, pairs))
        return {m: self._score_mechanism(m, X, y, profile, fp) for m in mechanisms}

    def _refine_profile(self) -> UtilityProfile:
        p = self.profile
        clf = "rf" if p.clf == "logreg" else p.clf
        return UtilityProfile(
            name=f"{p.name}_refine",
            clf=clf,
            n_estimators=max(20, p.n_estimators),
            cv_splits=p.cv_splits,
            n_runs=max(2, p.n_runs),
            use_screening=False,
            refine_top_k=0,
            parallel=p.parallel,
        )

    def evaluate_all(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        # PF5: fingerprint calculado uma única vez e repassado para todos os sub-métodos
        fp = _data_fingerprint(X, y)
        profile = self.profile

        if not profile.use_screening:
            return self._evaluate_mechanisms(MECHANISM_NAMES, X, y, profile, fp)

        screen_scores = self._evaluate_mechanisms(
            self._screening_mechanisms, X, y, profile, fp  # P3: configurable
        )
        ranked = sorted(screen_scores.items(), key=lambda x: -x[1])
        top_k = [m for m, _ in ranked[: max(1, profile.refine_top_k)]]

        refine_profile = self._refine_profile()
        refined = self._evaluate_mechanisms(top_k, X, y, refine_profile, fp)

        results: Dict[str, float] = dict(screen_scores)
        results.update(refined)
        for mech in MECHANISM_NAMES:
            if mech not in results:
                fam = FAMILY_OF[mech]
                fam_scores = [
                    s for n, s in results.items() if FAMILY_OF.get(n) == fam
                ]
                results[mech] = float(np.mean(fam_scores)) if fam_scores else 0.0
        return {m: results[m] for m in MECHANISM_NAMES}

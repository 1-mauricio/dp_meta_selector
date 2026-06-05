"""Interface principal de seleção de mecanismos DP."""

import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from sklearn.preprocessing import LabelEncoder

from .applicator import DPApplicator
from .calibration import DELTA_DEFAULT, FAMILY_EPSILON
from .config import DEFAULT_CACHE_DIR, FRAMEWORK_VERSION, TARGET_NOISE_RATIO
from .mechanisms import FAMILY_OF, MECHANISM_NAMES
from .meta_dataset import MetaDatasetBuilder
from .meta_learner import MetaLearner
from .baseline_store import DEFAULT_BASELINE_REGISTRY, BaselineStore
from .utility import (
    EVAL_FAST_PROFILE,
    META_FAST_PROFILE,
    DPUtilityEvaluator,
    UtilityProfile,
    UtilityResultCache,
)

_log = logging.getLogger(__name__)


class DPMechanismSelector:
    """
    Interface de alto nível para seleção automática de mecanismo DP.

    >>> selector = DPMechanismSelector()
    >>> selector.fit(datasets)
    >>> rec = selector.recommend(X_new, y_new)
    >>> X_private = selector.apply(X_new, rec["recommended_mechanism"])
    """

    def __init__(
        self,
        delta: float = DELTA_DEFAULT,
        n_runs: Optional[int] = None,
        meta_profile: UtilityProfile = META_FAST_PROFILE,
        eval_profile: UtilityProfile = EVAL_FAST_PROFILE,
        cache_dir: Path = DEFAULT_CACHE_DIR,
        use_cache: bool = True,
        fast_meta_models: bool = True,
        baseline_store: Optional[BaselineStore] = None,
    ):
        self.delta = delta
        self.n_runs = n_runs
        self.meta_profile = meta_profile
        self.eval_profile = eval_profile
        self._cache = UtilityResultCache(cache_dir=cache_dir, enabled=use_cache)
        self._baseline_store = baseline_store or BaselineStore(
            db_path=Path(cache_dir) / "baselines.sqlite",
            enabled=use_cache,
        )
        eval_baseline_id = DEFAULT_BASELINE_REGISTRY.resolve_id(eval_profile)
        self._builder = MetaDatasetBuilder(
            delta=delta,
            profile=meta_profile,
            cache=self._cache,
            fast_landmarks=True,
            n_runs=n_runs,
            baseline_store=self._baseline_store,
            baseline_id="meta_logreg",
        )
        self._learner = MetaLearner(fast_mode=fast_meta_models, fast_landmarks=True)
        self._applicator = DPApplicator(delta=delta)
        self._evaluator = DPUtilityEvaluator(
            delta=delta,
            profile=eval_profile,
            cache=self._cache,
            n_runs=n_runs,
            baseline_store=self._baseline_store,
            baseline_registry=DEFAULT_BASELINE_REGISTRY,
            baseline_id=eval_baseline_id,
        )
        self.meta_df = None
        self.cv_scores = None

    def fit(self, datasets) -> "DPMechanismSelector":
        _log.info("=" * 65)
        _log.info("  DP MECHANISM SELECTOR v%s", FRAMEWORK_VERSION)
        _log.info("=" * 65)
        _log.info("  δ = %s  |  SNR-alvo = %.0f%%", self.delta, TARGET_NOISE_RATIO * 100)
        _log.info(
            "  Meta-build : %s (%s, cv=%d, runs=%d, screening=%s)",
            self.meta_profile.name, self.meta_profile.clf,
            self.meta_profile.cv_splits, self.meta_profile.n_runs,
            self.meta_profile.use_screening,
        )
        _log.info(
            "  Avaliação  : %s (%s, cv=%d, runs=%d)",
            self.eval_profile.name, self.eval_profile.clf,
            self.eval_profile.cv_splits, self.eval_profile.n_runs,
        )
        _log.info(
            "  Cache      : %s (%s)",
            self._cache.cache_dir, "on" if self._cache.enabled else "off",
        )
        _log.info(
            "  Baselines  : %s (%s)",
            self._baseline_store.db_path,
            "on" if self._baseline_store.enabled else "off",
        )
        _log.info(
            "  Epsilons calibrados: %s",
            {k: f"{v:.3f}" for k, v in FAMILY_EPSILON.items()},
        )
        _log.info("  Datasets: %d", len(datasets))

        self.meta_df = self._builder.build(datasets)
        self.cv_scores = self._learner.fit(self.meta_df)

        _log.info("[Meta-Modelos] F1-macro (CV):")
        for name, score in self.cv_scores.items():
            marker = " ◄ melhor" if name == self._learner.best_model_name else ""
            s = f"{score:.4f}" if not np.isnan(score) else "  n/a "
            _log.info("   %-22s %s%s", name, s, marker)
        _log.info("  Melhor meta-modelo: %s", self._learner.best_model_name)
        return self

    def recommend(self, X, y, meta_model=None, verbose=True) -> Dict:
        result = self._learner.predict(X, y, model_name=meta_model)
        if verbose:
            mech = result["recommended_mechanism"]
            fam = FAMILY_OF.get(mech, "?")
            eps = FAMILY_EPSILON.get(fam, "?")
            _log.info("=" * 65)
            _log.info("  RECOMENDAÇÃO DE MECANISMO DP")
            _log.info("=" * 65)
            _log.info("  Mecanismo  : %s", mech)
            _log.info("  Família    : %s  (ε = %.3f)", fam, eps)
            _log.info("  Confiança  : %.2f%%", result["confidence"] * 100)
            _log.info("  Meta-modelo: %s", result["meta_model_used"])
            _log.info("  Todas as probabilidades:")
            for m, p in sorted(result["all_proba"].items(), key=lambda x: -x[1]):
                fam_m = FAMILY_OF.get(m, "?")
                bar = "█" * int(p * 28)
                _log.info("   %-22s %.3f  [%-12s]  %s", m, p, fam_m, bar)
        return result

    def apply(self, X, mechanism: str, verbose=True) -> np.ndarray:
        if mechanism not in MECHANISM_NAMES:
            raise ValueError(f"Desconhecido: {mechanism}")
        X_dp = self._applicator.apply(mechanism, X)
        if verbose:
            fam = FAMILY_OF[mechanism]
            eps = FAMILY_EPSILON[fam]
            noise = float(np.mean(np.abs(X_dp - X.astype(float))))
            _log.info("[Aplicação] '%s'  família=%s  ε=%.3f", mechanism, fam, eps)
            _log.info("  Ruído médio absoluto: %.6f", noise)
        return X_dp

    def evaluate(
        self, X, y, dataset_id: Optional[str] = None
    ) -> Dict[str, float]:
        y_enc = LabelEncoder().fit_transform(y)
        base = self._evaluator.baseline(X, y_enc, dataset_id=dataset_id)
        dp = self._evaluator.evaluate_all(X, y_enc)

        _log.info("=" * 72)
        _log.info("  AVALIAÇÃO DE UTILIDADE")
        _log.info("=" * 72)
        _log.info("  Baseline (sem DP): %.4f", base)
        _log.info(
            "  %-22s  %7s  %-12s  %6s  %6s  %s",
            "Mecanismo", "ε", "Família", "Acc", "Rel%", "Barra",
        )
        _log.info("  %s", "-" * 70)
        for mech in MECHANISM_NAMES:
            acc = dp[mech]
            rel = acc / (base + 1e-9)
            fam = FAMILY_OF[mech]
            eps = FAMILY_EPSILON[fam]
            bar = "█" * int(rel * 20)
            _log.info(
                "  %-22s  %7.3f  %-12s  %.4f  %5.1f%%  %s",
                mech, eps, fam, acc, rel * 100, bar,
            )
        return {"baseline": base, **dp}

    def save(self, path="dp_meta_selector.joblib"):
        self._learner.save(path)

    def load(self, path="dp_meta_selector.joblib"):
        self._learner.load(path)

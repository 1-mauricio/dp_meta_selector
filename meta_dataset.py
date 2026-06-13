"""Construção do meta-dataset para meta-aprendizagem."""

import logging
import time
from pathlib import Path
from typing import List, Optional, Set

import joblib
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
        checkpoint_path: Optional[Path] = None,  # v18: checkpoint para retomada após interrupção
        checkpoint_every: int = 10,  # v18: salva checkpoint a cada N datasets
        save_path: Optional[Path] = None,  # v19: diretório para persistir meta-dataset em CSV
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
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.checkpoint_every = checkpoint_every
        self.save_path = Path(save_path) if save_path else None

    def _save_checkpoint(self, rows: List[dict], processed: Set[str]) -> None:
        """Salva progresso em disco para permitir retomada após interrupção."""
        if self.checkpoint_path is None:
            return
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"rows": rows, "processed": list(processed)}, self.checkpoint_path)
        _log.info(
            "[checkpoint] %d datasets salvos → %s",
            len(rows), self.checkpoint_path,
        )

    def _load_checkpoint(self):
        """Carrega progresso anterior se o arquivo de checkpoint existir."""
        if self.checkpoint_path is None or not self.checkpoint_path.exists():
            return [], set()
        data = joblib.load(self.checkpoint_path)
        rows = data.get("rows", [])
        processed = set(data.get("processed", []))
        _log.info(
            "[checkpoint] Retomando de '%s': %d datasets já processados.",
            self.checkpoint_path, len(rows),
        )
        return rows, processed

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

        # utility_gap: diferença entre o 1º e 2º melhor mecanismo
        sorted_rel = sorted(rel.values(), reverse=True)
        meta["utility_gap"] = float(sorted_rel[0] - sorted_rel[1]) if len(sorted_rel) > 1 else 0.0
        meta["utility_best_abs"] = float(max(dp.values()))
        meta["utility_worst_abs"] = float(min(dp.values()))
        meta["utility_range"] = meta["utility_best_abs"] - meta["utility_worst_abs"]

        # FASE 3: Perda de Utilidade Relativa por mecanismo (target para regressão).
        # Definição: quanto o mecanismo M perde em relação ao baseline sem DP.
        # utility_loss_M = max(0, (baseline - dp_acc) / baseline) * 100  [percentual]
        # Um mecanismo que preserva toda a utilidade tem loss=0.
        # O framework escolherá o mecanismo com MENOR perda prevista.
        for m in dp:
            loss = max(0.0, (base - dp[m]) / base) * 100.0
            meta[f"utility_loss_{m}"] = float(loss)

        # MELHORIA: Seleção de best_mechanism com desempate por família
        # Prioriza mecanismo mais específico quando há empate
        meta["best_mechanism"] = self._select_best_mechanism(dp, rel, meta)
        meta["best_relative_acc"] = max(rel.values())
        
        # Registra família do melhor mecanismo para diagnóstico
        meta["best_family"] = FAMILY_OF.get(meta["best_mechanism"], "continuous")
        return meta

    def _select_best_mechanism(
        self, dp: dict, rel: dict, meta: dict, margin: float = 0.005
    ) -> str:
        """Seleciona o melhor mecanismo com desempate inteligente por família.
        
        Mudanças vs. versão anterior:
        1. Desempate por família: prefere mecanismo da família mais adequada ao dataset
        2. Considera sinais do dataset (ratio_integer_cols, cat_ratio_low_cardinality)
        3. Quando empate dentro da família, escolhe o mais simples/eficiente
        """
        best_rel = max(rel.values())
        candidates = [m for m in MECHANISM_NAMES if rel.get(m, 0.0) >= best_rel - margin]
        
        if len(candidates) == 1:
            return candidates[0]
        
        # Extrai sinais do dataset para decisão de família
        ratio_int = meta.get("ratio_integer_cols", 0.0)
        ratio_discrete = meta.get("ratio_discrete", 0.0)
        cat_low_card = meta.get("cat_ratio_low_cardinality", 0.0)
        disc_score = meta.get("disc_composite_score", 0.0)
        
        # Heurística de família baseada em meta-features
        if cat_low_card >= 0.7 and ratio_int >= 0.8:
            preferred_family = "categorical"
        elif ratio_int >= 0.8 and disc_score >= 0.3:
            preferred_family = "discrete"
        elif ratio_discrete >= 0.5 and ratio_int >= 0.5:
            preferred_family = "discrete"
        else:
            preferred_family = "continuous"
        
        # Filtra candidatos pela família preferida se houver algum
        family_candidates = [m for m in candidates if FAMILY_OF.get(m) == preferred_family]
        if family_candidates:
            candidates = family_candidates
        
        # Desempate final: maior acurácia absoluta, depois ordem canônica
        candidates.sort(key=lambda m: (-dp[m], MECHANISM_NAMES.index(m)))
        return candidates[0]

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
        n_total = len(items)

        # v18: retoma de checkpoint se disponível
        rows, processed_names = self._load_checkpoint()
        items_todo = [
            item for item in items
            if (item[2] if isinstance(item, tuple) else getattr(item, "name", "")) not in processed_names
        ]
        n_done_start = len(rows)
        if n_done_start > 0:
            _log.info(
                "[meta-build] Retomando: %d/%d datasets já processados, %d restantes.",
                n_done_start, n_total, len(items_todo),
            )

        effective_jobs = self.n_jobs
        if n_total <= 4:
            effective_jobs = 1

        t0 = time.time()

        if effective_jobs == 1:
            for i, item in enumerate(tqdm(items_todo, desc="Construindo meta-dataset",
                                          initial=n_done_start, total=n_total)):
                r = self._process_one(item)
                if r is not None:
                    rows.append(r)
                    processed_names.add(r.get("dataset_name", f"idx_{n_done_start + i}"))

                # v18: checkpoint periódico + log de progresso com ETA
                n_done = len(rows)
                if (i + 1) % self.checkpoint_every == 0:
                    self._save_checkpoint(rows, processed_names)
                    elapsed = time.time() - t0
                    remaining = n_total - n_done
                    eta = (elapsed / max(n_done - n_done_start, 1)) * remaining
                    _log.info(
                        "[progresso] %d/%d  elapsed=%ds  ETA=%ds  (%.0f%% concluído)",
                        n_done, n_total, int(elapsed), int(eta),
                        100.0 * n_done / n_total,
                    )
        else:
            _log.info("[meta-build] paralelo: n_jobs=%s datasets=%d", effective_jobs, len(items_todo))
            # prefer="threads": numpy/sklearn liberam o GIL → paralelismo real sem fork
            results = Parallel(n_jobs=effective_jobs, prefer="threads", verbose=0)(
                delayed(self._process_one)(item)
                for item in tqdm(items_todo, desc="Construindo meta-dataset",
                                 initial=n_done_start, total=n_total)
            )
            new_rows = [r for r in results if r is not None]
            rows.extend(new_rows)
            for r in new_rows:
                processed_names.add(r.get("dataset_name", ""))

        # Salva checkpoint final
        self._save_checkpoint(rows, processed_names)
        elapsed_total = time.time() - t0
        _log.info(
            "[meta-build] Concluído: %d datasets em %.0fs (%.1f s/dataset)",
            len(rows), elapsed_total, elapsed_total / max(len(rows), 1),
        )

        df = pd.DataFrame(rows)
        self._log_diagnostics(df)
        self._save_meta_dataset(df)  # v19: persiste features + targets em CSV
        _log.info("[meta-build] %s", self.evaluator.cache.summary())
        if self.evaluator.baseline_store is not None:
            _log.info("[meta-build] %s", self.evaluator.baseline_store.summary())
        return df

    def _save_meta_dataset(self, df: pd.DataFrame) -> None:
        """v19: persiste meta-features (X) e targets de regressão (Y) em CSV.

        Salva dois arquivos no diretório self.save_path:
          - meta_features_{profile}.csv   — todas as 116 features + dataset_name
          - meta_targets_{profile}.csv    — utility_loss_* por mecanismo + dataset_name

        Isso permite retunar os modelos ML sem recalcular o loop caro de n_runs=5.
        """
        if self.save_path is None:
            return
        self.save_path.mkdir(parents=True, exist_ok=True)
        profile_tag = self.evaluator.profile.name

        from .mechanisms import MECHANISM_NAMES
        loss_cols = [f"utility_loss_{m}" for m in MECHANISM_NAMES if f"utility_loss_{m}" in df.columns]
        meta_cols  = [c for c in df.columns if c not in loss_cols]

        features_path = self.save_path / f"meta_features_{profile_tag}.csv"
        targets_path  = self.save_path / f"meta_targets_{profile_tag}.csv"

        df[meta_cols].to_csv(features_path, index=False)
        df[["dataset_name"] + loss_cols].to_csv(targets_path, index=False)

        _log.info(
            "[meta-build] Meta-dataset salvo em '%s'  (features=%d  targets=%d  rows=%d)",
            self.save_path, len(meta_cols), len(loss_cols), len(df),
        )

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

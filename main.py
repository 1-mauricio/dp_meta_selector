"""Pipeline CLI do meta-seletor de mecanismos DP."""

import argparse
import logging
from pathlib import Path

import numpy as np

from .baseline_store import (
    DEFAULT_BASELINE_REGISTRY,
    BaselineStore,
    precompute_baselines,
)
from .config import DEFAULT_CACHE_DIR, DEFAULT_MODEL_PATH, FRAMEWORK_VERSION, LOG_FORMAT
from .datasets import load_openml_training_datasets, split_meta_datasets
from .evaluator import FrameworkEvaluator
from .selector import DPMechanismSelector
from .utility import EVAL_FAST_PROFILE, EVAL_FULL_PROFILE, META_FAST_PROFILE, UtilityProfile

_log = logging.getLogger(__name__)


def _setup_logging(verbose: bool = False) -> None:
    """Configura o sistema de logging para uso via CLI."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format=LOG_FORMAT)
    # reduz verbosidade de bibliotecas externas ruidosas
    for noisy in ("openml", "urllib3", "requests", "sklearn"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _baseline_ids_for_run(
    meta_profile: UtilityProfile,
    eval_profile: UtilityProfile,
    extra: list[str] | None,
) -> list[str]:
    ids = {
        "meta_logreg",
        DEFAULT_BASELINE_REGISTRY.resolve_id(eval_profile),
    }
    if extra:
        ids.update(extra)
    return sorted(ids)


def run_precompute_baselines(
    baseline_ids: list[str] | None = None,
    use_cache: bool = True,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    export_path: Path | None = None,
) -> BaselineStore:
    _log.info("=" * 65)
    _log.info("  PRÉ-COMPUTAÇÃO DE BASELINES")
    _log.info("=" * 65)

    store = BaselineStore(
        db_path=Path(cache_dir) / "baselines.sqlite",
        enabled=use_cache,
    )
    datasets = load_openml_training_datasets()
    ids = baseline_ids or DEFAULT_BASELINE_REGISTRY.ids()

    _log.info("  Datasets : %d", len(datasets))
    _log.info("  Algoritmos: %s", ids)
    _log.info("  Store    : %s", store.db_path)

    precompute_baselines(
        datasets,
        baseline_ids=ids,
        registry=DEFAULT_BASELINE_REGISTRY,
        store=store,
    )

    if export_path is not None:
        out = store.export_table(export_path)
        _log.info("  Exportado: %s", out)

    return store


def main(
    meta_profile: UtilityProfile = META_FAST_PROFILE,
    eval_profile: UtilityProfile = EVAL_FAST_PROFILE,
    use_cache: bool = True,
    full_oracle_test: bool = False,
    precompute_baselines_first: bool = True,
    extra_baseline_ids: list[str] | None = None,
):
    _log.info("=" * 65)
    _log.info("  DP META-SELECTOR v%s — perfis de custo", FRAMEWORK_VERSION)
    _log.info("=" * 65)

    np.random.seed(42)

    _log.info("[1/5] Carregando datasets de treino (OpenML)...")
    datasets = load_openml_training_datasets()

    if precompute_baselines_first and use_cache:
        bid = _baseline_ids_for_run(meta_profile, eval_profile, extra_baseline_ids)
        store = BaselineStore(db_path=DEFAULT_CACHE_DIR / "baselines.sqlite")
        precompute_baselines(
            datasets,
            baseline_ids=bid,
            registry=DEFAULT_BASELINE_REGISTRY,
            store=store,
        )

    train_ds, test_ds = split_meta_datasets(datasets)
    _log.info("      Train: %d | Test: %d", len(train_ds), len(test_ds))

    _log.info("[2/5] Treinando seletor (meta-build rápido)...")
    selector = DPMechanismSelector(
        meta_profile=meta_profile,
        eval_profile=eval_profile,
        use_cache=use_cache,
        fast_meta_models=True,
    )
    selector.fit(train_ds)

    _log.info("[3/5] Avaliando framework...")
    evaluator = FrameworkEvaluator(selector, use_full_oracle=full_oracle_test)
    results_df = evaluator.evaluate(test_ds)

    _log.info("[4/5] Salvando modelo...")
    selector.save(str(DEFAULT_MODEL_PATH))

    _log.info("[5/5] Teste rápido...")
    X_test, y_test = test_ds[0][:2]
    rec = selector.recommend(X_test, y_test)
    selector.apply(X_test, rec["recommended_mechanism"])

    _log.info("=" * 65)
    _log.info("FIM")
    _log.info("=" * 65)

    return selector, results_df


def cli():
    parser = argparse.ArgumentParser(
        description="Meta-seletor de mecanismos DP (treino OpenML, perfis de custo)."
    )
    parser.add_argument(
        "--eval-full",
        action="store_true",
        help="Avaliação hold-out com EVAL_FULL (todos os mecanismos, mais lento).",
    )
    parser.add_argument(
        "--full-oracle-test",
        action="store_true",
        help="Oracle no teste com EVAL_FULL (ablation; muito lento).",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Desativa cache em .dp_meta_cache/ (mecanismos DP e baselines).",
    )
    parser.add_argument(
        "--precompute-baselines",
        action="store_true",
        help="Só pré-computa baselines e encerra (sem treinar o seletor).",
    )
    parser.add_argument(
        "--baseline-id",
        action="append",
        dest="baseline_ids",
        metavar="ID",
        help=(
            "Algoritmo de baseline a calcular (repita para vários). "
            f"Padrão no treino: meta_logreg + perfil de avaliação. "
            f"Disponíveis: {DEFAULT_BASELINE_REGISTRY.ids()}"
        ),
    )
    parser.add_argument(
        "--export-baselines",
        metavar="PATH",
        default=None,
        help="Exporta tabela de baselines (Parquet ou CSV) após pré-computo.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Ativa logging detalhado (DEBUG).",
    )
    parser.add_argument(
        "--skip-baseline-precompute",
        action="store_true",
        help="Não pré-computa baselines antes do treino (usa store sob demanda).",
    )
    args = parser.parse_args()
    _setup_logging(verbose=args.verbose)  # Q7: configura logging antes de tudo

    eval_profile = EVAL_FULL_PROFILE if args.eval_full else EVAL_FAST_PROFILE
    use_cache = not args.no_cache

    if args.precompute_baselines:
        run_precompute_baselines(
            baseline_ids=args.baseline_ids or DEFAULT_BASELINE_REGISTRY.ids(),
            use_cache=use_cache,
            export_path=Path(args.export_baselines) if args.export_baselines else None,
        )
        return None, None

    return main(
        meta_profile=META_FAST_PROFILE,
        eval_profile=eval_profile,
        use_cache=use_cache,
        full_oracle_test=args.full_oracle_test,
        precompute_baselines_first=not args.skip_baseline_precompute,
        extra_baseline_ids=args.baseline_ids,
    )


if __name__ == "__main__":
    cli()

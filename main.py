"""Pipeline CLI do meta-seletor de mecanismos DP."""

import argparse
import logging
import time
from pathlib import Path

import numpy as np

from .baseline_store import (
    DEFAULT_BASELINE_REGISTRY,
    BaselineStore,
    precompute_baselines,
)
from .config import DEFAULT_CACHE_DIR, DEFAULT_MODEL_PATH, FRAMEWORK_VERSION, LOG_FORMAT
from .datasets import load_openml_training_datasets, split_meta_datasets
from .diagnostics import run_full_diagnostics
from .evaluator import FrameworkEvaluator
from .reporter import generate_report
from .selector import DPMechanismSelector
from .utility import EVAL_FAST_PROFILE, EVAL_FULL_PROFILE, META_FAST_PROFILE, META_STABLE_PROFILE, UtilityProfile

_log = logging.getLogger(__name__)


def _setup_logging(verbose: bool = False, log_file: Path | None = None) -> None:
    """Configura o sistema de logging para uso via CLI.

    Parameters
    ----------
    verbose:
        Ativa nível DEBUG (padrão: INFO).
    log_file:
        Caminho do arquivo onde os logs serão gravados em paralelo ao console.
        O diretório pai é criado automaticamente se não existir.
        Se ``None``, grava somente no console.
    """
    level = logging.DEBUG if verbose else logging.INFO
    handlers: list[logging.Handler] = [logging.StreamHandler()]

    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
        handlers.append(file_handler)
        print(f"[INFO] Logs salvos em: {log_file.resolve()}")

    logging.basicConfig(level=level, format=LOG_FORMAT, handlers=handlers)
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
    log_file: Path | None = None,
    report_dir: Path | None = None,
    run_diagnostics: bool = False,
    checkpoint_path: Path | None = None,  # v18
    checkpoint_every: int = 25,           # v18: padrão mais granular para runs longas
    save_meta_dataset: Path | None = None, # v19
):
    start_time = time.time()
    _log.info("=" * 65)
    _log.info("  DP META-SELECTOR v%s — perfis de custo", FRAMEWORK_VERSION)
    if log_file is not None:
        _log.info("  Log file: %s", Path(log_file).resolve())
    _log.info("=" * 65)

    np.random.seed(42)

    _log.info("[1/5] Carregando datasets de treino (OpenML)...")
    datasets = load_openml_training_datasets()
    
    # MELHORIA: Adiciona datasets sintéticos para melhor cobertura de famílias
    from .synthetic_datasets import augment_training_datasets
    datasets = augment_training_datasets(datasets, synthetic_ratio=0.2, min_synthetic=30)
    _log.info("      Total (com sintéticos): %d datasets", len(datasets))
    
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
        checkpoint_path=checkpoint_path,
        checkpoint_every=checkpoint_every,
        save_path=save_meta_dataset,
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

    # Relatório completo
    _log.info("[6/6] Gerando relatório estruturado...")
    _report_dir = Path(report_dir) if report_dir else Path("reports")
    generate_report(
        selector=selector,
        results_df=results_df,
        train_ds=train_ds,
        test_ds=test_ds,
        all_ds=datasets,
        start_time=start_time,
        output_dir=_report_dir,
        log_file=Path(log_file) if log_file else None,
    )

    # Diagnósticos avançados (opcional)
    if run_diagnostics:
        _log.info("[7/7] Executando diagnósticos avançados...")
        run_full_diagnostics(
            selector=selector,
            results_df=results_df,
            train_datasets=train_ds,
            test_datasets=test_ds,
            output_dir=_report_dir,
        )

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
        "--report-dir",
        metavar="DIR",
        default="reports",
        help="Diretório onde o relatório JSON da run será salvo (padrão: reports/).",
    )
    parser.add_argument(
        "--log-file",
        metavar="PATH",
        default=None,
        help=(
            "Salva todos os logs da run em arquivo (além do console). "
            "Ex.: --log-file logs/run_2024-01.log"
        ),
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Ativa logging detalhado (DEBUG).",
    )
    parser.add_argument(
        "--diagnostics",
        action="store_true",
        help="Executa diagnósticos avançados (F1 por família, confusion matrix, calibração).",
    )
    parser.add_argument(
        "--stable",
        action="store_true",
        help=(
            "v19: usa META_STABLE_PROFILE (n_runs=5) para gerar meta-dataset de alta "
            "qualidade. Labels são médias de 5 runs → regressor mais preciso. "
            "~5× mais lento que o padrão."
        ),
    )
    parser.add_argument(
        "--save-meta-dataset",
        metavar="DIR",
        default=None,
        help=(
            "v19: salva meta_features_{profile}.csv e meta_targets_{profile}.csv no "
            "diretório especificado assim que o loop de datasets terminar. "
            "Permite retunar modelos ML sem recalcular n_runs=5."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        metavar="FILE",
        default=None,
        help=(
            "v18: arquivo de checkpoint (.joblib) para retomada após interrupção. "
            "Ex.: --checkpoint .dp_meta_cache/ckpt_stable.joblib"
        ),
    )
    parser.add_argument(
        "--checkpoint-every",
        metavar="N",
        type=int,
        default=25,
        help="v18: salva checkpoint a cada N datasets (padrão: 25).",
    )
    parser.add_argument(
        "--skip-baseline-precompute",
        action="store_true",
        help="Não pré-computa baselines antes do treino (usa store sob demanda).",
    )
    args = parser.parse_args()
    log_file = Path(args.log_file) if args.log_file else None
    _setup_logging(verbose=args.verbose, log_file=log_file)

    meta_profile = META_STABLE_PROFILE if args.stable else META_FAST_PROFILE
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
        meta_profile=meta_profile,
        eval_profile=eval_profile,
        use_cache=use_cache,
        full_oracle_test=args.full_oracle_test,
        precompute_baselines_first=not args.skip_baseline_precompute,
        extra_baseline_ids=args.baseline_ids,
        log_file=log_file,
        report_dir=Path(args.report_dir),
        run_diagnostics=args.diagnostics,
        checkpoint_path=Path(args.checkpoint) if args.checkpoint else None,
        checkpoint_every=args.checkpoint_every,
        save_meta_dataset=Path(args.save_meta_dataset) if args.save_meta_dataset else None,
    )


if __name__ == "__main__":
    cli()

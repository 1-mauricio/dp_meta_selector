"""Private validation selection using disjoint shards and parallel composition.

Each candidate model is trained on a disjoint, value-independent shard. A
separate disjoint validation shard is used by the exponential mechanism. The
full epsilon is available in each branch; basic parallel composition bounds
the joint release by the maximum branch privacy cost.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from math import sqrt
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dpmeta.classifiers import (
    encoded_numeric_mask,
    fit_clean_softmax,
    fit_private_random_forest,
    fit_private_sgd_softmax,
    make_random_forest_plan,
    privatize_softmax,
)
from dpmeta.model import DELTA_TOTAL, balanced_accuracy, fit_private
from dpmeta.openml_benchmark import (
    DEFAULT_CACHE,
    DEFAULT_SEED,
    _get_openml_data,
    _load_manifest,
    _openml,
    _seed_for,
    stratified_holdout_indices,
)
from dpmeta.privacy import ParallelPrivacyLedger, Release, Sensitivity
from scripts.private_validation_benchmark import (
    DELTA,
    DP_SGD_STEPS,
    PIPELINES,
    PIPELINE_DISPLAY,
    VALIDATION_FRACTION,
    _model_ledger_dict,
    _paired_summary,
    _predict_candidate,
    exponential_select,
    summarize,
)


def _compose_parallel_ledger(epsilon_total: float, delta_total: float,
                             candidate_ledgers: dict[str, dict],
                             n_validation: int) -> dict:
    ledger = ParallelPrivacyLedger(epsilon_total, delta_total)
    for pipeline, item in candidate_ledgers.items():
        ledger.record(Release(
            name=f"candidate_shard:{pipeline}",
            mechanism="private_model_on_disjoint_shard",
            epsilon=item["epsilon_used"],
            delta=item["delta_used"],
            sensitivity=Sensitivity(0.0, 0.0, 0.0),
            noise_scale=0.0,
        ))
    sensitivity = 1.0 / n_validation
    ledger.record(Release(
        name="private_validation_shard:pipeline_choice",
        mechanism="exponential_mechanism_mean_accuracy",
        epsilon=epsilon_total,
        delta=0.0,
        sensitivity=Sensitivity(sensitivity, sensitivity, sensitivity),
        noise_scale=0.0,
    ))
    return ledger.verify().as_dict()


def _implementation_sha256() -> str:
    sources = (
        Path(__file__),
        Path("scripts/private_validation_benchmark.py"),
        Path("dpmeta/classifiers.py"),
        Path("dpmeta/mechanisms.py"),
        Path("dpmeta/model.py"),
        Path("dpmeta/openml_benchmark.py"),
        Path("dpmeta/privacy.py"),
        Path("dpmeta/schema.py"),
    )
    digest = hashlib.sha256()
    for source in sources:
        digest.update(str(source).encode("utf-8") + b"\0")
        digest.update(source.read_bytes())
    return digest.hexdigest()


def _fit_parallel_candidates(x_shards: list[np.ndarray], y_shards: list[np.ndarray],
                             x_validation: np.ndarray, y_validation: np.ndarray,
                             schema, epsilon: float, seed: int
                             ) -> tuple[dict[str, object], dict[str, dict], int]:
    if len(x_shards) != len(PIPELINES) or len(y_shards) != len(PIPELINES):
        raise ValueError("é necessário um shard disjunto por pipeline")
    if any(len(y) == 0 for y in y_shards):
        raise ValueError("cada pipeline precisa de ao menos uma linha de treino")
    n_classes = len(schema.classes)
    models: dict[str, object] = {}
    ledgers: dict[str, dict] = {}

    x, y = x_shards[0], y_shards[0]
    gnb, gnb_ledger = fit_private(
        x, y, n_classes, epsilon=epsilon, mechanism="gaussian_zcdp",
        allocation="equal", seed=_seed_for(seed, "gnb"), delta=DELTA,
    )
    models[PIPELINES[0]] = gnb
    ledgers[PIPELINES[0]] = _model_ledger_dict(gnb_ledger)

    x, y = x_shards[1], y_shards[1]
    clean_softmax = fit_clean_softmax(
        x, y, n_classes, regularization=0.1, optimizer_tolerance=1e-5,
        record_norm_bound=sqrt(len(schema.feature_names) + 1),
    )
    softmax, softmax_ledger = privatize_softmax(
        clean_softmax, epsilon=epsilon, seed=_seed_for(seed, "softmax-output"),
        delta=DELTA,
    )
    models[PIPELINES[1]] = softmax
    ledgers[PIPELINES[1]] = _model_ledger_dict(softmax_ledger)

    x, y = x_shards[2], y_shards[2]
    forest_plan = make_random_forest_plan(
        len(y), encoded_numeric_mask(schema), seed=_seed_for(seed, "forest-plan"),
        n_trees=5, depth=3,
    )
    forest, forest_ledger = fit_private_random_forest(
        x, y, n_classes, epsilon=epsilon, seed=_seed_for(seed, "forest"),
        plan=forest_plan,
    )
    models[PIPELINES[2]] = forest
    ledgers[PIPELINES[2]] = _model_ledger_dict(forest_ledger)

    x, y = x_shards[3], y_shards[3]
    sgd_softmax, sgd_ledger = fit_private_sgd_softmax(
        x, y, n_classes, epsilon=epsilon, delta=DELTA,
        seed=_seed_for(seed, "dp-sgd"), n_steps=DP_SGD_STEPS,
        gradient_clip=1.0, regularization=0.1,
    )
    models[PIPELINES[3]] = sgd_softmax
    ledgers[PIPELINES[3]] = _model_ledger_dict(sgd_ledger)

    accuracies = np.asarray([
        np.mean(_predict_candidate(name, models[name], x_validation) == y_validation)
        for name in PIPELINES
    ], dtype=float)
    selected = exponential_select(
        accuracies, epsilon, len(y_validation), seed=_seed_for(seed, "choice"),
    )
    composed = _compose_parallel_ledger(epsilon, DELTA, ledgers, len(y_validation))
    return models, {
        "candidate_ledgers": ledgers,
        "composed_ledger": composed,
        "epsilon_selection": epsilon,
        "epsilon_per_candidate": epsilon,
        "delta_per_approximate_candidate": DELTA,
    }, selected


def run_dataset(openml, row: pd.Series, epsilons: tuple[float, ...], repeats: int,
                seed: int, numeric_transform: str) -> list[dict]:
    x, y, schema = _get_openml_data(openml, row, numeric_transform=numeric_transform)
    rows = []
    for repetition in range(repeats):
        train_indices, test_indices = stratified_holdout_indices(
            y, _seed_for(seed, int(row.dataset_id), repetition, "outer-split"),
        )
        x_outer, y_outer = x[train_indices], y[train_indices]
        permutation = np.random.default_rng(_seed_for(
            seed, int(row.dataset_id), repetition, "parallel-private-partition",
        )).permutation(len(y_outer))
        n_validation = max(1, int(round(VALIDATION_FRACTION * len(y_outer))))
        if n_validation + len(PIPELINES) > len(y_outer):
            raise ValueError("split interno insuficiente para validar e treinar os candidatos")
        validation_positions = permutation[:n_validation]
        training_positions = permutation[n_validation:]
        shard_positions = np.array_split(training_positions, len(PIPELINES))
        x_shards = [x_outer[index] for index in shard_positions]
        y_shards = [y_outer[index] for index in shard_positions]
        x_validation = x_outer[validation_positions]
        y_validation = y_outer[validation_positions]
        x_test, y_test = x[test_indices], y[test_indices]

        for epsilon in epsilons:
            models, accounting, selected_index = _fit_parallel_candidates(
                x_shards, y_shards, x_validation, y_validation, schema, epsilon,
                _seed_for(seed, int(row.dataset_id), repetition, epsilon),
            )
            selected_pipeline = PIPELINES[selected_index]
            selected_prediction = _predict_candidate(
                selected_pipeline, models[selected_pipeline], x_test,
            )

            # Reference only: this candidate uses all outer-training records,
            # unlike the disjoint-shard selector being evaluated.
            baseline_clean = fit_clean_softmax(
                x_outer, y_outer, len(schema.classes), regularization=0.1,
                optimizer_tolerance=1e-5,
                record_norm_bound=sqrt(len(schema.feature_names) + 1),
            )
            baseline_model, baseline_ledger = privatize_softmax(
                baseline_clean, epsilon=epsilon,
                seed=_seed_for(seed, int(row.dataset_id), repetition, epsilon,
                               "full-softmax"),
                delta=DELTA,
            )
            rows.append({
                "dataset_id": int(row.dataset_id),
                "family_id": str(row.family_id),
                "epsilon_total": float(epsilon),
                "repetition": repetition,
                "numeric_transform": numeric_transform,
                "selected_pipeline": selected_pipeline,
                "selected_test_balanced_accuracy": balanced_accuracy(
                    y_test, selected_prediction,
                ),
                "full_epsilon_softmax_test_balanced_accuracy": balanced_accuracy(
                    y_test, baseline_model.predict(x_test),
                ),
                "epsilon_used": float(accounting["composed_ledger"]["epsilon_used"]),
                "epsilon_limit": float(accounting["composed_ledger"]["epsilon_limit"]),
                "delta_used": float(accounting["composed_ledger"]["delta_used"]),
                "delta_limit": float(accounting["composed_ledger"]["delta_limit"]),
                "epsilon_selection": float(accounting["epsilon_selection"]),
                "epsilon_per_candidate": float(accounting["epsilon_per_candidate"]),
                "validation_fraction": VALIDATION_FRACTION,
                "n_validation": len(y_validation),
                "training_n": int(sum(map(len, y_shards))),
                "candidate_training_sizes_json": json.dumps(
                    {pipeline: len(shard) for pipeline, shard in zip(PIPELINES, y_shards)},
                    sort_keys=True,
                ),
                "candidate_ledger_json": json.dumps(
                    accounting["candidate_ledgers"], sort_keys=True,
                ),
                "composed_ledger_json": json.dumps(
                    accounting["composed_ledger"], sort_keys=True,
                ),
                "baseline_epsilon_used": float(baseline_ledger.epsilon_used),
                "baseline_delta_used": float(baseline_ledger.delta_used),
            })
    return rows


def write_report(summary: dict) -> str:
    comparison = summary["private_validation_selector_vs_full_epsilon_softmax"]
    low, high = comparison["ci95_family_bootstrap"]
    lines = [
        "# Seleção privada por validação com composição paralela",
        "",
        f"{summary['n_datasets']} datasets, {summary['n_families']} famílias, "
        f"{summary['n_conditions']} condições sem falhas.",
        "",
        "Uma permutação independente dos valores divide o treino externo em quatro "
        "shards de candidato e um shard de validação separado. Cada modelo e o "
        "mecanismo exponencial recebem o ε integral. A utilidade de validação é "
        "acurácia média em [0,1], com sensibilidade replace-one `1/n_val`.",
        "",
        "## Resultado pareado por família",
        "",
        f"- BA média do seletor: {summary['selector_mean_test_ba']:.4f}",
        f"- BA média do softmax de referência: {summary['full_epsilon_softmax_mean_test_ba']:.4f}",
        f"- diferença seletor − softmax: {comparison['mean']:.4f}; IC95% "
        f"[{low:.4f}, {high:.4f}], p={comparison['paired_sign_flip_p_two_sided']:.4f}",
        f"- seleção por pipeline: `{json.dumps(summary['selected_pipeline_frequency'], sort_keys=True)}`",
        "",
        "## Contabilidade e limites",
        "",
        "A contabilidade usa composição paralela básica: cada registro pertence a "
        "exatamente um ramo, e a partição é independente dos valores. O custo conjunto "
        "é o máximo entre os ramos de treino e o ramo de seleção. O runner verifica "
        "`epsilon_used <= epsilon_limit` e `delta_used <= delta_limit` em cada condição. "
        "O mecanismo exponencial recebe os modelos já privatizados; para os registros "
        "dos shards de treino, essa etapa é pós-processamento dos modelos, e para os "
        "registros da validação os modelos são independentes.",
        "",
        "Este desenho reduz cada candidato a cerca de 18,75% das linhas do treino "
        "externo (75% dividido por quatro). O softmax de referência usa todas as linhas "
        "e ε integral, portanto é um baseline de desempenho deliberadamente forte. "
        "Balanced accuracy de teste pode divergir da acurácia privada de validação.",
        "",
        "O benchmark fixa seeds para reproduzir os mecanismos de ruído e seleção. "
        "Os números com seeds fixos medem utilidade em dados OpenML públicos; não "
        "são uma liberação DP real. Em deployment, a aleatoriedade tem de vir de uma "
        "fonte criptográfica não revelada.",
        "",
        "A transformação `public_bounds`, quando usada, deriva limites dos dados OpenML "
        "públicos completos, incluindo atributos do holdout externo. Ela serve como "
        "diagnóstico transdutivo no benchmark público; não valida preprocessing DP de "
        "dados privados. Para usar em produção, bounds devem vir de uma fonte "
        "genuinamente pública ou ser estimados com DP e compostos no orçamento.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/openml_manifest.csv"))
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--epsilons", type=lambda value: tuple(float(x) for x in value.split(",")),
                        default=(0.5, 2.0, 8.0))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--numeric-transform",
                        choices=("tanh", "asinh10", "public_bounds"), default="tanh")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = _load_manifest(args.manifest).head(args.limit).copy()
    if len(manifest) != args.limit or args.repeats < 1 or any(e <= 0 for e in args.epsilons):
        parser.error("manifest/limit, repeats ou epsilons inválidos")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    conditions_path = args.output_dir / "conditions.csv"
    if conditions_path.exists():
        raise FileExistsError(f"{conditions_path} já existe")
    config = {
        "manifest": str(args.manifest),
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "implementation_sha256": _implementation_sha256(),
        "limit": args.limit,
        "repeats": args.repeats,
        "epsilons": list(args.epsilons),
        "seed": args.seed,
        "numeric_transform": args.numeric_transform,
        "validation_fraction": VALIDATION_FRACTION,
        "dp_sgd_steps": DP_SGD_STEPS,
        "pipelines": list(PIPELINES),
        "composition": "parallel_disjoint_record_partitions",
    }
    (args.output_dir / "run_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
    )
    openml = _openml(args.cache_dir)
    failures = []
    for _, row in manifest.iterrows():
        try:
            rows = run_dataset(
                openml, row, args.epsilons, args.repeats, args.seed,
                args.numeric_transform,
            )
            pd.DataFrame(rows).to_csv(
                conditions_path, mode="a", header=not conditions_path.exists(), index=False,
            )
            print(f"finished {int(row.dataset_id)} {row.dataset_name}: {len(rows)} rows", flush=True)
        except Exception as error:
            failures.append({"dataset_id": int(row.dataset_id),
                             "dataset_name": str(row.dataset_name),
                             "error": f"{type(error).__name__}: {error}"})
            print(f"failed {row.dataset_name}: {failures[-1]['error']}", file=sys.stderr,
                  flush=True)
    if not conditions_path.exists():
        raise RuntimeError(f"nenhum dataset completou; falhas={failures}")
    conditions = pd.read_csv(conditions_path)
    if failures or conditions.dataset_id.nunique() != args.limit:
        raise RuntimeError(f"run incompleto; falhas={failures}")
    if (conditions.epsilon_used > conditions.epsilon_limit + 1e-9).any():
        raise AssertionError("composição paralela excedeu epsilon")
    if (conditions.delta_used > conditions.delta_limit + 1e-12).any():
        raise AssertionError("composição paralela excedeu delta")
    summary = summarize(conditions)
    summary.update({"failed_datasets": failures, "seed": args.seed,
                    "numeric_transform": args.numeric_transform,
                    "manifest_sha256": config["manifest_sha256"],
                    "implementation_sha256": config["implementation_sha256"]})
    summary["privacy_config"] = {
        "candidate_count": len(PIPELINES),
        "candidate_epsilon_each": "epsilon_total",
        "selection_epsilon": "epsilon_total",
        "delta_per_approximate_candidate": DELTA,
        "delta_total": DELTA,
        "composition": "basic parallel composition on value-independent disjoint record partitions",
        "validation_score": "mean 0/1 accuracy, range [0,1], replace-one sensitivity 1/n_validation",
        "dp_sgd": {"steps": DP_SGD_STEPS, "gradient_clip": 1.0,
                   "accounting": "full-batch Gaussian zCDP, no subsampling amplification"},
    }
    summary["primary_success"] = bool(
        summary["private_validation_selector_vs_full_epsilon_softmax"][
            "ci95_family_bootstrap"][0] > 0
        and summary["private_validation_selector_vs_full_epsilon_softmax"][
            "paired_sign_flip_p_two_sided"] < 0.05
    )
    summary["exploratory_only"] = True
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
    )
    report = write_report(summary)
    (args.output_dir / "report.md").write_text(report)
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

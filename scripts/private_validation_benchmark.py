"""End-to-end DP selection of a classifier using a private validation split.

The candidate models are trained with composed privacy budgets. A bounded
validation accuracy score is released only through the exponential mechanism.
The benchmark's outer test labels are used solely to evaluate public OpenML data.
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
from dpmeta.mechanisms import privacy_rng
from dpmeta.model import DELTA_TOTAL, balanced_accuracy, fit_private, predict
from dpmeta.openml_benchmark import (
    DEFAULT_CACHE,
    DEFAULT_SEED,
    _get_openml_data,
    _load_manifest,
    _openml,
    _seed_for,
    stratified_holdout_indices,
)
from dpmeta.privacy import PrivacyLedger, Release, Sensitivity


PIPELINES = (
    "gnb__gaussian_zcdp_equal",
    "softmax__analytic_gaussian",
    "random_forest__laplace_parallel_shards",
    "softmax__dp_sgd_full_batch",
)
PIPELINE_DISPLAY = {
    PIPELINES[0]: "GaussianNB + zCDP",
    PIPELINES[1]: "softmax + Gaussian output perturbation",
    PIPELINES[2]: "random forest + Laplace shard histograms",
    PIPELINES[3]: "full-batch DP-SGD softmax",
}
VALIDATION_BUDGET_FRACTION = 0.20
VALIDATION_FRACTION = 0.25
DP_SGD_STEPS = 25
DELTA = DELTA_TOTAL


def _model_ledger_dict(ledger) -> dict:
    if hasattr(ledger, "as_dict"):
        result = ledger.as_dict()
        return {"epsilon_used": float(result["epsilon_used"]),
                "delta_used": float(result["delta_used"])}
    return {"epsilon_used": float(ledger["epsilon_used"]),
            "delta_used": float(ledger["delta_used"])}


def exponential_select(scores: np.ndarray, epsilon: float, n_validation: int,
                       seed: int | None) -> int:
    """Select over bounded mean accuracy with replace-one sensitivity 1/n."""
    scores = np.asarray(scores, dtype=float)
    if (scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all()
            or np.any(scores < 0) or np.any(scores > 1)):
        raise ValueError("escores de validação devem estar em [0,1]")
    if epsilon <= 0 or n_validation < 1:
        raise ValueError("epsilon e n_validation precisam ser positivos")
    sensitivity = 1.0 / n_validation
    logits = epsilon * scores / (2.0 * sensitivity)
    probabilities = np.exp(logits - np.max(logits))
    probabilities /= probabilities.sum()
    draw = privacy_rng(seed).random()
    return min(int(np.searchsorted(np.cumsum(probabilities), draw, side="right")),
               len(scores) - 1)


def _predict_candidate(pipeline: str, model, x: np.ndarray) -> np.ndarray:
    if pipeline == PIPELINES[0]:
        return predict(x, model)
    return model.predict(x)


def _compose_ledger(epsilon_total: float, delta_total: float,
                    candidate_ledgers: dict[str, dict], epsilon_selection: float,
                    n_validation: int) -> dict:
    ledger = PrivacyLedger(epsilon_total, delta_total, accounting="basic")
    for pipeline, item in candidate_ledgers.items():
        ledger.record(Release(
            name=f"trained_candidate:{pipeline}",
            mechanism="sequential_composition_of_private_model",
            epsilon=item["epsilon_used"],
            delta=item["delta_used"],
            sensitivity=Sensitivity(0.0, 0.0, 0.0),
            noise_scale=0.0,
        ))
    sensitivity = 1.0 / n_validation
    ledger.record(Release(
        name="validation_pipeline_choice",
        mechanism="exponential_mechanism_mean_accuracy",
        epsilon=epsilon_selection,
        delta=0.0,
        sensitivity=Sensitivity(sensitivity, sensitivity, sensitivity),
        noise_scale=0.0,
    ))
    return ledger.verify().as_dict()


def _fit_candidates(x_train: np.ndarray, y_train: np.ndarray, x_validation: np.ndarray,
                    y_validation: np.ndarray, schema, epsilon: float, seed: int
                    ) -> tuple[dict[str, object], dict[str, dict], int]:
    n_candidates = len(PIPELINES)
    epsilon_selection = epsilon * VALIDATION_BUDGET_FRACTION
    epsilon_per_model = (epsilon - epsilon_selection) / n_candidates
    approximate_candidates = 3  # GaussianNB zCDP, output perturbation, and DP-SGD.
    delta_per_approximate_candidate = DELTA / approximate_candidates
    n_classes = len(schema.classes)
    models = {}
    ledgers = {}

    gnb, gnb_ledger = fit_private(
        x_train, y_train, n_classes, epsilon=epsilon_per_model,
        mechanism="gaussian_zcdp", allocation="equal",
        seed=_seed_for(seed, "gnb"), delta=delta_per_approximate_candidate,
    )
    models[PIPELINES[0]] = gnb
    ledgers[PIPELINES[0]] = _model_ledger_dict(gnb_ledger)

    clean_softmax = fit_clean_softmax(
        x_train, y_train, n_classes, regularization=0.1, optimizer_tolerance=1e-5,
        record_norm_bound=sqrt(len(schema.feature_names) + 1),
    )
    softmax, softmax_ledger = privatize_softmax(
        clean_softmax, epsilon=epsilon_per_model,
        seed=_seed_for(seed, "softmax-output"), delta=delta_per_approximate_candidate,
    )
    models[PIPELINES[1]] = softmax
    ledgers[PIPELINES[1]] = _model_ledger_dict(softmax_ledger)

    forest_plan = make_random_forest_plan(
        len(y_train), encoded_numeric_mask(schema),
        seed=_seed_for(seed, "forest-plan"), n_trees=5, depth=3,
    )
    forest, forest_ledger = fit_private_random_forest(
        x_train, y_train, n_classes, epsilon=epsilon_per_model,
        seed=_seed_for(seed, "forest"), plan=forest_plan,
    )
    models[PIPELINES[2]] = forest
    ledgers[PIPELINES[2]] = _model_ledger_dict(forest_ledger)

    sgd_softmax, sgd_ledger = fit_private_sgd_softmax(
        x_train, y_train, n_classes, epsilon=epsilon_per_model,
        delta=delta_per_approximate_candidate, seed=_seed_for(seed, "dp-sgd"),
        n_steps=DP_SGD_STEPS, gradient_clip=1.0, regularization=0.1,
    )
    models[PIPELINES[3]] = sgd_softmax
    ledgers[PIPELINES[3]] = _model_ledger_dict(sgd_ledger)

    accuracies = np.asarray([
        np.mean(_predict_candidate(name, models[name], x_validation) == y_validation)
        for name in PIPELINES
    ], dtype=float)
    selected = exponential_select(
        accuracies, epsilon_selection, len(y_validation), seed=_seed_for(seed, "choice"),
    )
    overall_ledger = _compose_ledger(
        epsilon, DELTA, ledgers, epsilon_selection, len(y_validation),
    )
    return models, {"candidate_ledgers": ledgers, "composed_ledger": overall_ledger,
                    "epsilon_selection": epsilon_selection,
                    "epsilon_per_candidate": epsilon_per_model,
                    "delta_per_approximate_candidate": delta_per_approximate_candidate}, selected


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
            seed, int(row.dataset_id), repetition, "private-validation-partition",
        )).permutation(len(y_outer))
        n_validation = max(1, int(round(VALIDATION_FRACTION * len(y_outer))))
        if n_validation >= len(y_outer):
            raise ValueError("split interno não deixou linhas para treinar candidatos")
        validation_positions = permutation[:n_validation]
        training_positions = permutation[n_validation:]
        x_train, y_train = x_outer[training_positions], y_outer[training_positions]
        x_validation, y_validation = x_outer[validation_positions], y_outer[validation_positions]
        x_test, y_test = x[test_indices], y[test_indices]

        for epsilon in epsilons:
            models, accounting, selected_index = _fit_candidates(
                x_train, y_train, x_validation, y_validation, schema, epsilon,
                _seed_for(seed, int(row.dataset_id), repetition, epsilon),
            )
            selected_pipeline = PIPELINES[selected_index]
            selected_prediction = _predict_candidate(selected_pipeline, models[selected_pipeline], x_test)

            # Strong fixed comparator: one softmax fit uses the full epsilon and
            # all outer-training rows. This is evaluated separately as a baseline.
            baseline_clean = fit_clean_softmax(
                x_outer, y_outer, len(schema.classes), regularization=0.1,
                optimizer_tolerance=1e-5,
                record_norm_bound=sqrt(len(schema.feature_names) + 1),
            )
            baseline_model, baseline_ledger = privatize_softmax(
                baseline_clean, epsilon=epsilon,
                seed=_seed_for(seed, int(row.dataset_id), repetition, epsilon, "full-softmax"),
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
                "training_n": len(y_train),
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


def _paired_summary(values: np.ndarray, seed: int) -> dict:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    bootstrap = rng.integers(0, len(values), size=(20_000, len(values)))
    means = values[bootstrap].mean(axis=1)
    signs = rng.choice(np.asarray([-1.0, 1.0]), size=(50_000, len(values)))
    observed = abs(float(values.mean()))
    null = np.abs((signs * values).mean(axis=1))
    return {
        "n_families": int(len(values)),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "ci95_family_bootstrap": [float(np.quantile(means, 0.025)),
                                   float(np.quantile(means, 0.975))],
        "paired_sign_flip_p_two_sided": float(
            (1 + np.count_nonzero(null >= observed)) / 50_001
        ),
    }


def summarize(conditions: pd.DataFrame) -> dict:
    conditions = conditions.copy()
    conditions["test_ba_delta"] = (
        conditions.selected_test_balanced_accuracy
        - conditions.full_epsilon_softmax_test_balanced_accuracy
    )
    per_dataset = conditions.groupby(["family_id", "dataset_id"]).agg(
        selected_ba=("selected_test_balanced_accuracy", "mean"),
        full_softmax_ba=("full_epsilon_softmax_test_balanced_accuracy", "mean"),
        delta=("test_ba_delta", "mean"),
    )
    per_family = per_dataset.groupby(level="family_id").mean()
    delta = per_family.delta.to_numpy(dtype=float)
    if not len(delta):
        raise ValueError("não há famílias para sumarizar")
    epsilon_ranges = conditions.groupby("epsilon_total").agg(
        epsilon_used_min=("epsilon_used", "min"),
        epsilon_used_max=("epsilon_used", "max"),
        delta_used_max=("delta_used", "max"),
    )
    paired = _paired_summary(delta, 20260917)
    return {
        "n_datasets": int(conditions.dataset_id.nunique()),
        "n_families": int(conditions.family_id.nunique()),
        "n_conditions": int(len(conditions)),
        "private_validation_selector_vs_full_epsilon_softmax": paired,
        "selector_mean_test_ba": float(per_family.selected_ba.mean()),
        "full_epsilon_softmax_mean_test_ba": float(per_family.full_softmax_ba.mean()),
        "selected_pipeline_frequency": {
            str(key): float(value) for key, value in
            conditions.selected_pipeline.value_counts(normalize=True).sort_index().items()
        },
        "epsilon_accounting_by_total_epsilon": {
            str(index): row.to_dict() for index, row in epsilon_ranges.iterrows()
        },
        "privacy_config": {
            "epsilon_selection_fraction": VALIDATION_BUDGET_FRACTION,
            "validation_fraction_of_outer_train": VALIDATION_FRACTION,
            "candidate_count": len(PIPELINES),
            "candidate_epsilon_share_equal": (1 - VALIDATION_BUDGET_FRACTION) / len(PIPELINES),
            "delta_total": DELTA,
            "delta_share_per_approximate_candidate": DELTA / 3,
            "composition": "basic sequential composition of all candidate models plus pure-DP exponential selection",
            "validation_score": "mean 0/1 accuracy, range [0,1], replace-one sensitivity 1/n_validation",
            "dp_sgd": {"steps": DP_SGD_STEPS, "gradient_clip": 1.0,
                       "accounting": "full-batch Gaussian zCDP, no subsampling amplification"},
        },
        "primary_success": bool(
            paired["ci95_family_bootstrap"][0] > 0
            and paired["paired_sign_flip_p_two_sided"] < 0.05
        ),
        "exploratory_only": True,
    }


def write_report(summary: dict) -> str:
    comparison = summary["private_validation_selector_vs_full_epsilon_softmax"]
    low, high = comparison["ci95_family_bootstrap"]
    lines = [
        "# Seleção privada por validação: piloto",
        "",
        f"{summary['n_datasets']} datasets, {summary['n_families']} famílias, "
        f"{summary['n_conditions']} condições sem falhas.",
        "",
        "A cada dataset/ε, 25% do treino externo é separado por uma permutação "
        "independente dos valores. Quatro candidatos são treinados na parte restante, "
        "cada um recebe 20% do orçamento de ε; a escolha usa o mecanismo exponencial "
        "sobre a acurácia média de validação, cuja sensibilidade replace-one é `1/n_val`. "
        "Os três candidatos Gaussianos dividem δ; a floresta usa ε-DP puro. A comparação "
        "é contra um softmax que usa o ε integral e todas as linhas de treino externo.",
        "",
        "## Resultado pareado por família",
        "",
        f"- BA média do seletor privado: {summary['selector_mean_test_ba']:.4f}",
        f"- BA média do softmax com ε integral: {summary['full_epsilon_softmax_mean_test_ba']:.4f}",
        f"- diferença seletor − softmax: {comparison['mean']:.4f}; IC95% "
        f"[{low:.4f}, {high:.4f}], p={comparison['paired_sign_flip_p_two_sided']:.4f}",
        f"- seleção por pipeline: `{json.dumps(summary['selected_pipeline_frequency'], sort_keys=True)}`",
        "",
        "## Interpretação e limites",
        "",
        "O accounting usa composição sequencial conservadora: todos os modelos candidatos "
        "e a seleção exponencial entram no orçamento; o runner verifica `epsilon_used <= "
        "epsilon_limit` e `delta_used <= delta_limit` em cada condição. O piloto não usa "
        "PrivateTuning/subsampling amplification. A utilidade de validação é acurácia, "
        "enquanto o teste reportado é balanced accuracy; isso pode divergir em classes "
        "desbalanceadas. O teste externo usa OpenML público e serve apenas para medir "
        "generalização, não faz parte da liberação privada descrita.",
        "",
        "A seleção privada provavelmente perde orçamento útil de treino ao dividir ε "
        "entre quatro modelos e a validação. Este resultado deve orientar o desenho da "
        "próxima ablação, não ser tratado como método final.",
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
        "implementation_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
            + Path("dpmeta/classifiers.py").read_bytes()
            + Path("dpmeta/schema.py").read_bytes()
        ).hexdigest(),
        "limit": args.limit,
        "repeats": args.repeats,
        "epsilons": list(args.epsilons),
        "seed": args.seed,
        "numeric_transform": args.numeric_transform,
        "validation_budget_fraction": VALIDATION_BUDGET_FRACTION,
        "validation_fraction": VALIDATION_FRACTION,
        "dp_sgd_steps": DP_SGD_STEPS,
        "pipelines": list(PIPELINES),
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
        raise AssertionError("composição excedeu epsilon")
    if (conditions.delta_used > conditions.delta_limit + 1e-12).any():
        raise AssertionError("composição excedeu delta")
    summary = summarize(conditions)
    summary.update({"failed_datasets": failures, "seed": args.seed,
                    "numeric_transform": args.numeric_transform,
                    "manifest_sha256": config["manifest_sha256"],
                    "implementation_sha256": config["implementation_sha256"]})
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
    )
    report = write_report(summary)
    (args.output_dir / "report.md").write_text(report)
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

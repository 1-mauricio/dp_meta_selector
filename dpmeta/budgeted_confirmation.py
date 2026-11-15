"""Fresh-holdout evaluation of a selector with explicitly budgeted DP diagnostics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from .classifiers import (
    encoded_numeric_mask,
    fit_clean_random_forest,
    fit_clean_softmax,
    fit_private_random_forest,
    make_random_forest_plan,
    privatize_softmax,
)
from .dp_diagnostics import DP_DIAGNOSTIC_NAMES, dp_training_diagnostics
from .external_confirmation import (
    DEFAULT_CACHE,
    _hash,
    _implementation_hash,
    _load_manifest,
    _manifest_identity,
    select_disjoint_manifest,
)
from .model import DELTA_TOTAL, balanced_accuracy, fit_clean, fit_private, predict
from .openml_benchmark import (
    _get_openml_data,
    _openml,
    _parse_epsilon,
    _seed_for,
    stratified_holdout_indices,
)
from .portfolio_benchmark import (
    PIPELINE_DISPLAY,
    PIPELINES,
    FOREST_CANDIDATE_TREE_COUNT,
    SOFTMAX_OPTIMIZER_TOLERANCE,
    SOFTMAX_REGULARIZATION,
    TREE_DEPTH,
    _aggregate,
    _best_fixed,
    _bootstrap_ci,
    _candidate_features,
    _condition_score,
    _fit_selector,
    _metric_summary,
    _public_dp_features,
    _sign_flip_p,
    run_dataset,
)
from .schema import PublicSchema, public_features

DEFAULT_TRAIN_MANIFEST = Path("data/openml_manifest.csv")
DEFAULT_TRAIN_RESULTS = Path("results/classifier-portfolio-feasibility/conditions.csv")
DEFAULT_TEST_MANIFEST = Path("data/openml_budgeted_dp_manifest.csv")
DEFAULT_OUTPUT = Path("results/budgeted-dp-selector-2026-09-18")
DEFAULT_PROTOCOL = Path("BUDGETED_SELECTOR_PROTOCOL.md")
DEFAULT_SEED = 20260920
DEFAULT_EPSILONS = (0.5, 2.0, 8.0)
MIN_TRAIN_FAMILIES = 40
MIN_TEST_FAMILIES = 120
DIAGNOSTIC_EPSILON_SHARE = 0.2
DIAGNOSTIC_DELTA_SHARE = 0.5


def _diagnostic_columns() -> list[str]:
    return [f"dpdiag__{name}" for name in DP_DIAGNOSTIC_NAMES]


def _budget_parameters(epsilon_total: float) -> dict[str, float]:
    epsilon_diag = DIAGNOSTIC_EPSILON_SHARE * float(epsilon_total)
    epsilon_model = float(epsilon_total) - epsilon_diag
    delta_diag = DIAGNOSTIC_DELTA_SHARE * DELTA_TOTAL
    delta_model = DELTA_TOTAL - delta_diag
    if epsilon_model <= 0 or delta_diag <= 0 or delta_model <= 0:
        raise ValueError("divisão do orçamento DP inválida")
    return {
        "epsilon_diag": epsilon_diag,
        "epsilon_model": epsilon_model,
        "delta_diag": delta_diag,
        "delta_model": delta_model,
    }


def run_dataset_budgeted(openml, row: pd.Series, epsilons: tuple[float, ...],
                         repeats: int, seed: int) -> list[dict]:
    """Run the data-dependent selector candidate with end-to-end budget split."""
    x, y, schema = _get_openml_data(openml, row)
    output: list[dict] = []
    for repetition in range(repeats):
        train_indices, test_indices = stratified_holdout_indices(
            y, _seed_for(seed, int(row.dataset_id), repetition, "portfolio_split"),
        )
        x_train, x_test = x[train_indices], x[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]

        clean_gnb = fit_clean(x_train, y_train, len(schema.classes))
        clean_softmax = fit_clean_softmax(
            x_train, y_train, len(schema.classes),
            regularization=SOFTMAX_REGULARIZATION,
            optimizer_tolerance=SOFTMAX_OPTIMIZER_TOLERANCE,
            record_norm_bound=np.sqrt(len(schema.feature_names) + 1),
        )
        forest_plan = make_random_forest_plan(
            len(y_train), encoded_numeric_mask(schema),
            seed=_seed_for(seed, int(row.dataset_id), repetition, "tree_plan"),
            n_trees=FOREST_CANDIDATE_TREE_COUNT, depth=TREE_DEPTH,
        )
        clean_forest = fit_clean_random_forest(
            x_train, y_train, len(schema.classes), forest_plan,
        )
        clean_scores = {
            PIPELINES[0]: balanced_accuracy(y_test, predict(x_test, clean_gnb)),
            PIPELINES[4]: balanced_accuracy(y_test, clean_softmax.model.predict(x_test)),
            PIPELINES[6]: balanced_accuracy(y_test, clean_forest.predict(x_test)),
        }
        best_clean_score = max(clean_scores.values())

        for epsilon_total in epsilons:
            budget = _budget_parameters(epsilon_total)
            diagnostics, diagnostic_ledger = dp_training_diagnostics(
                x_train, y_train, len(schema.classes),
                epsilon=budget["epsilon_diag"], delta=budget["delta_diag"],
                seed=_seed_for(seed, int(row.dataset_id), repetition,
                               epsilon_total, "dp_diagnostic"),
            )
            features_by_name = dict(zip(DP_DIAGNOSTIC_NAMES, diagnostics))

            gnb, gnb_ledger = fit_private(
                x_train, y_train, len(schema.classes),
                epsilon=budget["epsilon_model"], delta=budget["delta_model"],
                mechanism="gaussian_zcdp", allocation="equal",
                seed=_seed_for(seed, int(row.dataset_id), repetition,
                               epsilon_total, "budgeted_gnb"),
            )
            softmax, softmax_ledger = privatize_softmax(
                clean_softmax, epsilon=budget["epsilon_model"],
                delta=budget["delta_model"],
                seed=_seed_for(seed, int(row.dataset_id), repetition,
                               epsilon_total, "budgeted_softmax"),
            )
            forest, forest_ledger = fit_private_random_forest(
                x_train, y_train, len(schema.classes),
                epsilon=budget["epsilon_model"],
                seed=_seed_for(seed, int(row.dataset_id), repetition,
                               epsilon_total, "budgeted_forest"),
                plan=forest_plan,
            )
            models = {
                PIPELINES[0]: (predict(x_test, gnb), gnb_ledger.as_dict()),
                PIPELINES[4]: (softmax.predict(x_test), softmax_ledger.as_dict()),
                PIPELINES[6]: (forest.predict(x_test), forest_ledger),
            }
            for pipeline, (predicted, model_ledger) in models.items():
                private_score = balanced_accuracy(y_test, predicted)
                epsilon_used = budget["epsilon_diag"] + float(model_ledger["epsilon_used"])
                delta_used = budget["delta_diag"] + float(model_ledger["delta_used"])
                if (not np.isclose(epsilon_used, epsilon_total, rtol=0.0, atol=1e-8)
                        or delta_used > DELTA_TOTAL + 1e-12):
                    raise AssertionError("composição do diagnóstico e do modelo excede o contrato")
                composition = {
                    "contract": "replace_one_fixed_n_record_level",
                    "epsilon_total_limit": float(epsilon_total),
                    "delta_total_limit": DELTA_TOTAL,
                    "epsilon_used_basic_composition": epsilon_used,
                    "delta_used_basic_composition": delta_used,
                    "diagnostic": diagnostic_ledger,
                    "model": model_ledger,
                }
                output.append({
                    "dataset_id": int(row.dataset_id),
                    "family_id": str(row.family_id),
                    "epsilon": float(epsilon_total),
                    "epsilon_diag": budget["epsilon_diag"],
                    "epsilon_model": budget["epsilon_model"],
                    "delta_diag": budget["delta_diag"],
                    "delta_model": budget["delta_model"],
                    "pipeline": pipeline,
                    "repetition": repetition,
                    "private_ba": private_score,
                    "clean_ba": clean_scores[pipeline],
                    "best_clean_portfolio_ba": best_clean_score,
                    "pipeline_privacy_penalty": clean_scores[pipeline] - private_score,
                    "portfolio_regret": best_clean_score - private_score,
                    "train_n": len(train_indices),
                    "test_n": len(test_indices),
                    "encoded_dim": schema.encoded_dim,
                    "original_feature_count": len(schema.feature_names),
                    "class_count": len(schema.classes),
                    "epsilon_used": epsilon_used,
                    "delta_used": delta_used,
                    "ledger_json": json.dumps(composition, sort_keys=True),
                    **{f"dpdiag__{name}": float(value)
                       for name, value in features_by_name.items()},
                })
    return output


def _budgeted_candidate_features(meta_row: pd.Series, epsilon_total: float,
                                 pipeline: str, row: pd.Series,
                                 include_diagnostic: bool) -> np.ndarray:
    schema = PublicSchema.from_manifest_json(meta_row.public_schema_json)
    n = int(meta_row.n_instances)
    budget = _budget_parameters(epsilon_total)
    parts = [
        public_features(n, schema, epsilon_total),
        np.asarray([float(pipeline == item) for item in PIPELINES]),
        np.asarray([
            budget["epsilon_diag"] / epsilon_total,
            budget["epsilon_model"] / epsilon_total,
            budget["delta_diag"] / DELTA_TOTAL,
            budget["delta_model"] / DELTA_TOTAL,
        ]),
        _public_dp_features(
            n, schema, budget["epsilon_model"], pipeline,
            delta_total=budget["delta_model"],
        ),
    ]
    if include_diagnostic:
        parts.append(row[_diagnostic_columns()].to_numpy(dtype=float))
    return np.concatenate(parts)


def _row_weights(frame: pd.DataFrame) -> np.ndarray:
    family_dataset_counts = (frame[["family_id", "dataset_id"]].drop_duplicates()
                             .groupby("family_id").dataset_id.nunique().to_dict())
    n_families = frame.family_id.nunique()
    epsilon_counts = frame[["dataset_id", "epsilon"]].drop_duplicates().groupby(
        "dataset_id").epsilon.nunique().to_dict()
    repetition_counts = frame[["dataset_id", "epsilon", "repetition"]].drop_duplicates().groupby(
        ["dataset_id", "epsilon"]).repetition.nunique().to_dict()
    pipeline_counts = frame[["dataset_id", "epsilon", "repetition", "pipeline"]].drop_duplicates().groupby(
        ["dataset_id", "epsilon", "repetition"]).pipeline.nunique().to_dict()
    weights = []
    for row in frame.itertuples(index=False):
        weights.append(
            1.0 / n_families / family_dataset_counts[row.family_id]
            / epsilon_counts[row.dataset_id]
            / repetition_counts[(row.dataset_id, row.epsilon)]
            / pipeline_counts[(row.dataset_id, row.epsilon, row.repetition)]
        )
    values = np.asarray(weights, dtype=float)
    return values / values.sum()


def _checkpoint_is_complete(path: Path, dataset_ids: set[int],
                            expected_rows_per_dataset: int) -> bool:
    """Whether a cached conditions file is safe to reuse for analysis-only resume."""
    if not path.exists():
        return False
    try:
        frame = pd.read_csv(path, usecols=["dataset_id"])
    except (OSError, ValueError, pd.errors.ParserError):
        return False
    counts = frame.groupby("dataset_id").size()
    return (set(counts.index.astype(int)) == dataset_ids
            and bool((counts == expected_rows_per_dataset).all()))


def _fit_budgeted_selector(train: pd.DataFrame, manifest: pd.DataFrame, *,
                           include_diagnostic: bool, seed: int):
    meta = manifest.drop_duplicates("dataset_id").set_index("dataset_id")
    x = np.vstack([
        _budgeted_candidate_features(
            meta.loc[int(row.dataset_id)], float(row.epsilon), row.pipeline,
            pd.Series(row._asdict()), include_diagnostic,
        ) for row in train.itertuples(index=False)
    ])
    model = RandomForestRegressor(
        n_estimators=1000, min_samples_leaf=3, max_features=0.9,
        random_state=seed, n_jobs=1,
    )
    model.fit(x, train.private_ba.to_numpy(dtype=float), sample_weight=_row_weights(train))
    return model, meta


def _metric_comparison(predictions: pd.DataFrame, baseline_column: str,
                       seed: int) -> dict:
    per_dataset = (predictions.groupby(["family_id", "dataset_id"], as_index=False)
                   .agg({"selector_regret": "mean", baseline_column: "mean"}))
    per_family = (per_dataset.groupby("family_id", as_index=False)
                  .agg({"selector_regret": "mean", baseline_column: "mean"}))
    gain = (per_family[baseline_column] - per_family.selector_regret).to_numpy()
    mean_gain, lower, upper = _bootstrap_ci(gain, seed)
    selector_metrics = _metric_summary(per_family.selector_regret.to_numpy())
    baseline_metrics = _metric_summary(per_family[baseline_column].to_numpy())
    tail_noninferiority = _tail_noninferiority_lcb(
        per_family.selector_regret.to_numpy(),
        per_family[baseline_column].to_numpy(),
        seed + 2,
    )
    no_worse_tail = bool(all(value >= 0.0 for value in
                             tail_noninferiority["lower_95_one_sided"].values()))
    return {
        "mean_regret_gain": mean_gain,
        "gain_ci95_family_bootstrap": [lower, upper],
        "paired_sign_flip_p_two_sided": _sign_flip_p(gain, seed + 1),
        "selector_regret": selector_metrics,
        "baseline_regret": baseline_metrics,
        "tail_noninferiority_margin": 0.0,
        "tail_noninferiority_ci95_one_sided": tail_noninferiority,
        "no_worse_median_q90_cvar90": no_worse_tail,
    }


def _tail_noninferiority_lcb(selector: np.ndarray, baseline: np.ndarray,
                             seed: int, draws: int = 20000) -> dict:
    """Paired family bootstrap lower bounds for baseline-minus-selector tails."""
    selector = np.asarray(selector, dtype=float)
    baseline = np.asarray(baseline, dtype=float)
    if selector.shape != baseline.shape or selector.ndim != 1 or not len(selector):
        raise ValueError("vetores inválidos para bootstrap de cauda")
    rng = np.random.default_rng(seed)
    n = len(selector)
    tail_count = max(1, int(np.ceil(0.10 * n)))
    differences = {"median": np.empty(draws), "q90": np.empty(draws), "cvar90": np.empty(draws)}
    batch_size = 1000
    for start in range(0, draws, batch_size):
        stop = min(draws, start + batch_size)
        indices = rng.integers(0, n, size=(stop - start, n))
        selector_sample = selector[indices]
        baseline_sample = baseline[indices]
        differences["median"][start:stop] = (
            np.median(baseline_sample, axis=1) - np.median(selector_sample, axis=1)
        )
        differences["q90"][start:stop] = (
            np.quantile(baseline_sample, 0.90, axis=1)
            - np.quantile(selector_sample, 0.90, axis=1)
        )
        selector_tail = np.partition(selector_sample, n - tail_count, axis=1)[:, -tail_count:]
        baseline_tail = np.partition(baseline_sample, n - tail_count, axis=1)[:, -tail_count:]
        differences["cvar90"][start:stop] = (
            baseline_tail.mean(axis=1) - selector_tail.mean(axis=1)
        )
    return {
        "estimand": "baseline_regret_minus_selector_regret",
        "lower_95_one_sided": {
            name: float(np.quantile(values, 0.05))
            for name, values in differences.items()
        },
    }


def _summarize_branch(predictions: pd.DataFrame, *, seed: int) -> dict:
    metrics = {
        "vs_fixed_global_full_budget": _metric_comparison(
            predictions, "fixed_global_regret", seed,
        ),
        "vs_fixed_epsilon_full_budget": _metric_comparison(
            predictions, "fixed_epsilon_regret", seed + 11,
        ),
    }
    primary = metrics["vs_fixed_epsilon_full_budget"]
    return {
        "n_test_datasets": int(predictions.dataset_id.nunique()),
        "n_test_families": int(predictions.family_id.nunique()),
        "selector_top1_auxiliary": float(predictions.selector_top1.mean()),
        "selected_pipeline_frequency": {
            str(k): float(v) for k, v in
            predictions.selector_pipeline.value_counts(normalize=True).sort_index().items()
        },
        **metrics,
        "primary_success": bool(
            primary["gain_ci95_family_bootstrap"][0] > 0
            and primary["paired_sign_flip_p_two_sided"] < 0.05
            and primary["no_worse_median_q90_cvar90"]
            and predictions.family_id.nunique() >= MIN_TEST_FAMILIES
        ),
    }


def _write_report(summary: dict) -> str:
    branches = summary["branches"]
    lines = [
        "# Validação externa exploratória — seletor com diagnóstico DP",
        "",
        "> O holdout é novo em relação às coortes anteriores, mas a hipótese foi proposta após a inspeção delas; não é confirmação preregistrada.",
        "> Cada diagnóstico é um único vetor Gaussiano DP; features são seu pós-processamento. Em deployment, uma só execução consome o orçamento descrito.",
        "",
        f"- Datasets tentados/completos: {summary['attempted_datasets']}/{summary['completed_datasets']}",
        f"- Famílias completas: {summary['primary_n_test_families']}",
        f"- Divisão fixa: ε_diag={DIAGNOSTIC_EPSILON_SHARE:.0%} e ε_model={1-DIAGNOSTIC_EPSILON_SHARE:.0%}; δ dividido ao meio.",
        "",
        "## Seletor com diagnóstico DP vs. melhor fixo por ε (orçamento integral)",
    ]
    main = branches["budgeted_dp_diagnostic"]["vs_fixed_epsilon_full_budget"]
    lines.extend([
        f"- Ganho médio de regret: {main['mean_regret_gain']:.6f}",
        f"- IC95% bootstrap por família: [{main['gain_ci95_family_bootstrap'][0]:.6f}, {main['gain_ci95_family_bootstrap'][1]:.6f}]",
        f"- Sign-flip bilateral: p={main['paired_sign_flip_p_two_sided']:.4f}",
        f"- Mediana/Q90/CVaR90 não piores: {main['no_worse_median_q90_cvar90']}",
        f"- Critério primário de viabilidade: {branches['budgeted_dp_diagnostic']['primary_success']}",
        "",
        "## Comparação de features com orçamento de modelo igual",
    ])
    ablation = summary["dp_diagnostic_vs_public_split"]
    lines.extend([
        f"- Ganho médio DP-diagnostic vs. seletor público: {ablation['mean_regret_gain']:.6f}",
        f"- IC95%: [{ablation['gain_ci95_family_bootstrap'][0]:.6f}, {ablation['gain_ci95_family_bootstrap'][1]:.6f}], p={ablation['paired_sign_flip_p_two_sided']:.4f}",
        "",
        "## Outros ramos",
    ])
    for name, label in (
        ("public_full_budget", "Público + calibração DP, ε integral"),
        ("public_split_budget", "Público + calibração DP, ε_model"),
        ("budgeted_dp_diagnostic", "Diagnóstico DP + ε_model"),
    ):
        item = branches[name]["vs_fixed_epsilon_full_budget"]
        lines.append(
            f"- {label}: ganho={item['mean_regret_gain']:.6f}; IC95% "
            f"[{item['gain_ci95_family_bootstrap'][0]:.6f}, "
            f"{item['gain_ci95_family_bootstrap'][1]:.6f}]; "
            f"p={item['paired_sign_flip_p_two_sided']:.4f}."
        )
    lines.extend([
        "",
        "Registre como evidência externa exploratória, não como garantia geral. O benchmark OpenML mede utilidade entre famílias públicas; a DP formal refere-se ao procedimento de deployment com uma única consulta diagnóstica e um único pipeline escolhido.",
    ])
    return "\n".join(lines) + "\n"


def _build_predictions(test_full: pd.DataFrame, test_budgeted: pd.DataFrame,
                       test_manifest: pd.DataFrame, train_full: pd.DataFrame,
                       train_manifest: pd.DataFrame,
                       train_budgeted: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, dict]:
    train_meta = train_manifest.drop_duplicates("dataset_id")
    train_full_aggregated = _aggregate(train_full, train_manifest)
    fixed_global = _best_fixed(train_full_aggregated)
    fixed_epsilon = {
        float(epsilon): _best_fixed(train_full_aggregated, float(epsilon))
        for epsilon in sorted(train_full_aggregated.epsilon.unique())
    }
    public_full_selector, _ = _fit_selector(
        train_full_aggregated, train_meta, "public_plus_dp", {}, seed=seed,
    )
    public_split_selector, _ = _fit_budgeted_selector(
        train_budgeted, train_manifest, include_diagnostic=False, seed=seed + 1,
    )
    diagnostic_selector, _ = _fit_budgeted_selector(
        train_budgeted, train_manifest, include_diagnostic=True, seed=seed + 2,
    )
    selectors = {
        "public_full_budget": public_full_selector,
        "public_split_budget": public_split_selector,
        "budgeted_dp_diagnostic": diagnostic_selector,
    }

    test_meta = test_manifest.drop_duplicates("dataset_id").set_index("dataset_id")
    rows: list[dict] = []
    key = ["dataset_id", "epsilon", "repetition"]
    full_groups = {tuple((int(r.dataset_id), float(r.epsilon), int(r.repetition))): g
                   for _, g in test_full.groupby(key, sort=False)
                   for r in [g.iloc[0]]}
    for combo, budget_condition in test_budgeted.groupby(key, sort=True):
        dataset_id, epsilon, repetition = int(combo[0]), float(combo[1]), int(combo[2])
        full_condition = full_groups[(dataset_id, epsilon, repetition)]
        meta_row = test_meta.loc[dataset_id]
        row_features = budget_condition.iloc[0]

        full_x = np.vstack([
            _candidate_features(meta_row, epsilon, pipeline, "public_plus_dp", {})
            for pipeline in PIPELINES
        ])
        public_split_x = np.vstack([
            _budgeted_candidate_features(meta_row, epsilon, pipeline, row_features, False)
            for pipeline in PIPELINES
        ])
        diagnostic_x = np.vstack([
            _budgeted_candidate_features(meta_row, epsilon, pipeline, row_features, True)
            for pipeline in PIPELINES
        ])
        chosen_by_branch = {
            "public_full_budget": PIPELINES[int(np.argmax(
                selectors["public_full_budget"].predict(full_x)))],
            "public_split_budget": PIPELINES[int(np.argmax(
                selectors["public_split_budget"].predict(public_split_x)))],
            "budgeted_dp_diagnostic": PIPELINES[int(np.argmax(
                selectors["budgeted_dp_diagnostic"].predict(diagnostic_x)))],
        }
        full_oracle = max(PIPELINES, key=lambda pipeline: _condition_score(
            full_condition, pipeline, "private_ba"))
        split_oracle = max(PIPELINES, key=lambda pipeline: _condition_score(
            budget_condition, pipeline, "private_ba"))
        global_pipeline = fixed_global
        epsilon_pipeline = fixed_epsilon[epsilon]
        fixed_global_regret = _condition_score(full_condition, global_pipeline, "portfolio_regret")
        fixed_epsilon_regret = _condition_score(full_condition, epsilon_pipeline, "portfolio_regret")
        for branch, chosen in chosen_by_branch.items():
            condition = (full_condition if branch == "public_full_budget"
                         else budget_condition)
            oracle = full_oracle if branch == "public_full_budget" else split_oracle
            rows.append({
                "branch": branch,
                "dataset_id": dataset_id,
                "family_id": str(row_features.family_id),
                "epsilon": epsilon,
                "repetition": repetition,
                "selector_pipeline": chosen,
                "selector_regret": _condition_score(condition, chosen, "portfolio_regret"),
                "selector_ba": _condition_score(condition, chosen, "private_ba"),
                "selector_top1": float(chosen == oracle),
                "fixed_global_pipeline": global_pipeline,
                "fixed_epsilon_pipeline": epsilon_pipeline,
                "fixed_global_regret": fixed_global_regret,
                "fixed_epsilon_regret": fixed_epsilon_regret,
                "epsilon_fixed_regret": fixed_epsilon_regret,
                "gnb_full_budget_regret": _condition_score(
                    full_condition, PIPELINES[0], "portfolio_regret"),
            })
    predictions = pd.DataFrame(rows)

    # Direct paired ablation: same budgeted model outcomes, with/without the
    # noisy data-dependent diagnostic.
    paired = predictions[predictions.branch.isin(
        ["public_split_budget", "budgeted_dp_diagnostic"]
    )].pivot_table(
        index=["family_id", "dataset_id", "epsilon", "repetition"],
        columns="branch", values="selector_regret", aggfunc="first",
    ).reset_index()
    family_gain = (paired.assign(gain=paired.public_split_budget
                                 - paired.budgeted_dp_diagnostic)
                   .groupby("family_id", as_index=False).gain.mean().gain.to_numpy())
    mean_gain, lower, upper = _bootstrap_ci(family_gain, seed + 30)
    ablation = {
        "mean_regret_gain": mean_gain,
        "gain_ci95_family_bootstrap": [lower, upper],
        "paired_sign_flip_p_two_sided": _sign_flip_p(family_gain, seed + 31),
        "positive_favors_dp_diagnostic": True,
    }
    return predictions, {
        "fixed_global_pipeline": fixed_global,
        "fixed_epsilon_pipeline": {str(k): v for k, v in fixed_epsilon.items()},
        "selector_training_families": int(train_budgeted.family_id.nunique()),
        "dp_diagnostic_vs_public_split": ablation,
    }


def run_experiment(*, train_manifest_path: Path = DEFAULT_TRAIN_MANIFEST,
                   train_results_path: Path = DEFAULT_TRAIN_RESULTS,
                   test_manifest_path: Path = DEFAULT_TEST_MANIFEST,
                   exclusion_manifest_paths: tuple[Path, ...] = (),
                   protocol_path: Path = DEFAULT_PROTOCOL,
                   output_dir: Path = DEFAULT_OUTPUT,
                   epsilons: tuple[float, ...] = DEFAULT_EPSILONS,
                   repeats: int = 3, seed: int = DEFAULT_SEED,
                   cache_dir: Path = DEFAULT_CACHE, resume: bool = False) -> dict:
    if repeats < 1 or not epsilons or min(epsilons) <= 0:
        raise ValueError("repetições/epsilons inválidos")
    if not test_manifest_path.exists():
        raise FileNotFoundError("selecione e congele o manifesto antes da avaliação")
    if output_dir.exists() and any(output_dir.iterdir()) and not resume:
        raise FileExistsError(f"{output_dir} não está vazio; use --resume ou outro diretório")
    output_dir.mkdir(parents=True, exist_ok=True)
    train_manifest = _load_manifest(train_manifest_path)
    test_manifest = _load_manifest(test_manifest_path)
    old_train_full = pd.read_csv(train_results_path)
    excluded_ids, excluded_families = _manifest_identity(
        (train_manifest_path, *exclusion_manifest_paths)
    )
    test_families = set(test_manifest.family_id.astype(str))
    if excluded_ids & set(test_manifest.dataset_id.astype(int)):
        raise ValueError("sobreposição de IDs entre treino/exclusões e novo holdout")
    if excluded_families & test_families:
        raise ValueError("sobreposição de famílias entre treino/exclusões e novo holdout")
    if len(test_manifest) != 150 or test_manifest.family_id.nunique() != len(test_manifest):
        raise ValueError("o manifesto precisa conter 150 famílias únicas")
    if set(old_train_full.dataset_id.astype(int)) != set(train_manifest.dataset_id.astype(int)):
        raise ValueError("outcomes completos de meta-treino não correspondem ao manifesto")

    config = {
        "protocol": str(protocol_path),
        "protocol_sha256": _hash(protocol_path),
        "implementation_sha256": _implementation_hash(),
        "train_manifest_sha256": _hash(train_manifest_path),
        "train_full_results_sha256": _hash(train_results_path),
        "test_manifest_sha256": _hash(test_manifest_path),
        "excluded_manifests_sha256": {
            str(path): _hash(path) for path in exclusion_manifest_paths
        },
        "seed": seed,
        "epsilons": list(epsilons),
        "repeats": repeats,
        "delta_total": DELTA_TOTAL,
        "diagnostic_epsilon_share": DIAGNOSTIC_EPSILON_SHARE,
        "diagnostic_delta_share": DIAGNOSTIC_DELTA_SHARE,
        "portfolio": list(PIPELINES),
        "new_holdout": True,
        "confirmatory": False,
    }
    config_path = output_dir / "run_config.json"
    analysis_only_resume = False
    if resume and config_path.exists():
        saved = json.loads(config_path.read_text())
        if saved != config:
            saved_without_code = {key: value for key, value in saved.items()
                                  if key != "implementation_sha256"}
            current_without_code = {key: value for key, value in config.items()
                                    if key != "implementation_sha256"}
            expected_rows = len(epsilons) * repeats * len(PIPELINES)
            test_ids = set(test_manifest.dataset_id.astype(int))
            train_ids = set(train_manifest.dataset_id.astype(int))
            complete_artifacts = all((
                _checkpoint_is_complete(
                    output_dir / "meta_train_budgeted_conditions.csv", train_ids,
                    expected_rows,
                ),
                _checkpoint_is_complete(
                    output_dir / "test_full_conditions.csv", test_ids, expected_rows,
                ),
                _checkpoint_is_complete(
                    output_dir / "test_budgeted_conditions.csv", test_ids, expected_rows,
                ),
            ))
            analysis_only_resume = (
                saved_without_code == current_without_code
                and saved.get("implementation_sha256") != config["implementation_sha256"]
                and complete_artifacts
            )
            if not analysis_only_resume:
                raise ValueError(
                    "--resume recusado: config/hash diverge; mudança de implementação só "
                    "é permitida para reanálise quando todos os checkpoints de condições "
                    "estão completos e os demais parâmetros/hash coincidem"
                )
    else:
        config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")

    openml = _openml(cache_dir)
    train_budget_path = output_dir / "meta_train_budgeted_conditions.csv"
    train_fail_path = output_dir / "meta_train_failures.csv"
    if train_budget_path.exists():
        train_budgeted = pd.read_csv(train_budget_path)
        expected_train_rows = len(epsilons) * repeats * len(PIPELINES)
        train_counts = train_budgeted.groupby("dataset_id").size()
        completed_train = set(train_counts[train_counts == expected_train_rows].index.astype(int))
        if resume and len(completed_train) != train_budgeted.dataset_id.nunique():
            train_budgeted = train_budgeted[train_budgeted.dataset_id.isin(completed_train)]
            train_budgeted.to_csv(train_budget_path, index=False)
    else:
        train_budgeted = pd.DataFrame()
        completed_train = set()
    train_failures: list[dict] = []
    for _, row in train_manifest.iterrows():
        dataset_id = int(row.dataset_id)
        if dataset_id in completed_train:
            continue
        try:
            rows = run_dataset_budgeted(openml, row, epsilons, repeats, seed - 3)
            frame = pd.DataFrame(rows)
            frame.to_csv(train_budget_path, mode="a", header=not train_budget_path.exists(), index=False)
            completed_train.add(dataset_id)
        except Exception as error:
            failure = {"dataset_id": dataset_id, "dataset_name": row.dataset_name,
                       "error": f"{type(error).__name__}: {error}"}
            train_failures.append(failure)
            print(f"meta-train failed {row.dataset_name}: {failure['error']}",
                  file=sys.stderr, flush=True)
        pd.DataFrame(train_failures).to_csv(train_fail_path, index=False)
        if len(completed_train) % 10 == 0 and dataset_id in completed_train:
            print(f"budgeted meta-train: {len(completed_train)}/{len(train_manifest)}", flush=True)
    if len(completed_train) < MIN_TRAIN_FAMILIES:
        raise RuntimeError("menos de 40 famílias de meta-treino completaram o novo orçamento")
    train_budgeted = pd.read_csv(train_budget_path)

    full_test_path = output_dir / "test_full_conditions.csv"
    budgeted_test_path = output_dir / "test_budgeted_conditions.csv"
    failure_path = output_dir / "test_failures.csv"
    full_test = pd.read_csv(full_test_path) if full_test_path.exists() else pd.DataFrame()
    budgeted_test = pd.read_csv(budgeted_test_path) if budgeted_test_path.exists() else pd.DataFrame()
    expected_test_rows = len(epsilons) * repeats * len(PIPELINES)
    full_counts = full_test.groupby("dataset_id").size() if len(full_test) else pd.Series(dtype=int)
    budget_counts = budgeted_test.groupby("dataset_id").size() if len(budgeted_test) else pd.Series(dtype=int)
    completed_full = set(full_counts[full_counts == expected_test_rows].index.astype(int))
    completed_budget = set(budget_counts[budget_counts == expected_test_rows].index.astype(int))
    completed_test = completed_full & completed_budget
    if resume:
        if len(full_test) and full_test.dataset_id.nunique() != len(completed_test):
            full_test = full_test[full_test.dataset_id.isin(completed_test)]
            full_test.to_csv(full_test_path, index=False)
        if len(budgeted_test) and budgeted_test.dataset_id.nunique() != len(completed_test):
            budgeted_test = budgeted_test[budgeted_test.dataset_id.isin(completed_test)]
            budgeted_test.to_csv(budgeted_test_path, index=False)
    failures: list[dict] = []
    if failure_path.exists():
        failures = pd.read_csv(failure_path).to_dict(orient="records")
    for _, row in test_manifest.iterrows():
        dataset_id = int(row.dataset_id)
        if dataset_id in completed_test:
            continue
        try:
            full_rows = run_dataset(
                openml, row, epsilons, repeats, seed,
                include_research_features=False,
            )
            budget_rows = run_dataset_budgeted(openml, row, epsilons, repeats, seed)
            full_frame = pd.DataFrame(full_rows)
            budget_frame = pd.DataFrame(budget_rows)
            # A shared split must imply an identical clean-reference endpoint.
            clean_full = full_frame.groupby("repetition").best_clean_portfolio_ba.first()
            clean_budget = budget_frame.groupby("repetition").best_clean_portfolio_ba.first()
            if not np.allclose(clean_full.sort_index(), clean_budget.sort_index()):
                raise AssertionError("full/split-budget evaluations used different clean splits")
            full_frame.to_csv(full_test_path, mode="a", header=not full_test_path.exists(), index=False)
            budget_frame.to_csv(
                budgeted_test_path, mode="a", header=not budgeted_test_path.exists(), index=False,
            )
            completed_test.add(dataset_id)
            failures = [item for item in failures if int(item["dataset_id"]) != dataset_id]
        except Exception as error:
            failure = {"dataset_id": dataset_id, "dataset_name": row.dataset_name,
                       "error": f"{type(error).__name__}: {error}"}
            failures.append(failure)
            print(f"holdout failed {row.dataset_name}: {failure['error']}",
                  file=sys.stderr, flush=True)
        pd.DataFrame(failures, columns=["dataset_id", "dataset_name", "error"]).to_csv(
            failure_path, index=False,
        )
        if len(completed_test) % 10 == 0 and dataset_id in completed_test:
            print(f"fresh holdout: {len(completed_test)}/{len(test_manifest)}", flush=True)

    full_test = pd.read_csv(full_test_path)
    budgeted_test = pd.read_csv(budgeted_test_path)
    predictions, training_summary = _build_predictions(
        full_test, budgeted_test, test_manifest, old_train_full,
        train_manifest, train_budgeted, seed + 400,
    )
    branches = {}
    for index, branch in enumerate(sorted(predictions.branch.unique())):
        branches[branch] = _summarize_branch(
            predictions[predictions.branch == branch], seed=seed + 500 + index * 13,
        )
    primary_n_families = branches["budgeted_dp_diagnostic"]["n_test_families"]
    summary = {
        "analysis_type": "fresh_holdout_budgeted_dp_diagnostic_selector",
        "confirmatory": False,
        "exploratory": True,
        "interpretation": (
            "evidência externa exploratória favorável ao caminho DP-diagnostic"
            if branches["budgeted_dp_diagnostic"]["primary_success"] else
            "o caminho DP-diagnostic não demonstrou viabilidade pelo critério primário"
            if primary_n_families >= MIN_TEST_FAMILIES else
            "inconclusivo: menos de 120 famílias novas completaram"
        ),
        "attempted_datasets": int(len(test_manifest)),
        "completed_datasets": int(len(completed_test)),
        "failed_datasets": failures,
        "primary_n_test_families": primary_n_families,
        "minimum_test_families_required": MIN_TEST_FAMILIES,
        "diagnostic_release": {
            "statistics": ["class_counts", "class_feature_sums", "class_feature_second_moments"],
            "features": list(DP_DIAGNOSTIC_NAMES),
            "mechanism": "analytic_gaussian_joint_vector",
            "sensitivity_l2": "2*sqrt(1+2*p)",
            "epsilon_share": DIAGNOSTIC_EPSILON_SHARE,
            "delta_share": DIAGNOSTIC_DELTA_SHARE,
            "postprocessing_only": True,
        },
        "privacy_contract": {
            "adjacency": "replace_one_full_record_fixed_n",
            "total_epsilon_grid": list(epsilons),
            "delta_total": DELTA_TOTAL,
            "model_epsilon_fraction": 1.0 - DIAGNOSTIC_EPSILON_SHARE,
            "basic_composition_checked_per_candidate": True,
            "deployment_transcript": "one diagnostic release and one selected pipeline",
            "benchmark_warning": "all candidate alternatives and repeated public OpenML runs are not one sensitive deployment transcript",
        },
        "training": training_summary,
        "primary_comparison": "budgeted DP-diagnostic selector vs epsilon-specific fixed pipeline trained with full epsilon",
        "branches": branches,
        "dp_diagnostic_vs_public_split": training_summary["dp_diagnostic_vs_public_split"],
        "protocol": str(protocol_path),
        "protocol_sha256": _hash(protocol_path),
        "run_config_sha256": _hash(config_path),
        "execution_implementation_sha256": json.loads(config_path.read_text())[
            "implementation_sha256"],
        "analysis_implementation_sha256": _implementation_hash(),
        "analysis_only_resume_after_code_change": analysis_only_resume,
        "test_manifest_sha256": _hash(test_manifest_path),
        "fresh_holdout_disjoint_from_all_previous_manifests": True,
        "note": "This new path was designed after inspection of prior cohorts; fresh family holdout prevents retuning on this cohort but is not a preregistered study of the thesis.",
    }
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (output_dir / "report.md").write_text(_write_report(summary))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, default=DEFAULT_TRAIN_MANIFEST)
    parser.add_argument("--train-results", type=Path, default=DEFAULT_TRAIN_RESULTS)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_TEST_MANIFEST)
    parser.add_argument("--exclude-manifest", type=Path, action="append", default=[])
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--limit", type=int, default=150)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--epsilons", type=_parse_epsilon, default=DEFAULT_EPSILONS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--select-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite-manifest", action="store_true")
    args = parser.parse_args(argv)
    if args.limit < MIN_TEST_FAMILIES or args.repeats < 1:
        parser.error("--limit precisa ser >=120 e --repeats >=1")
    if args.select_only:
        select_disjoint_manifest(
            old_manifest_path=args.train_manifest,
            exclusion_manifest_paths=tuple(args.exclude_manifest),
            output_path=args.manifest,
            cache_dir=args.cache_dir,
            target_count=args.limit,
            seed=args.seed,
            catalog_status="all",
            max_rows=10000,
            overwrite=args.overwrite_manifest,
        )
        return 0
    run_experiment(
        train_manifest_path=args.train_manifest,
        train_results_path=args.train_results,
        test_manifest_path=args.manifest,
        exclusion_manifest_paths=tuple(args.exclude_manifest),
        protocol_path=args.protocol,
        output_dir=args.output,
        epsilons=args.epsilons,
        repeats=args.repeats,
        seed=args.seed,
        cache_dir=args.cache_dir,
        resume=args.resume,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

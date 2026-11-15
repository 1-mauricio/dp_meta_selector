"""Feasibility benchmark for selecting among different private classifiers."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from functools import lru_cache
from math import ceil, log1p, sqrt
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from scipy.stats import binomtest, friedmanchisquare, ttest_1samp, wilcoxon

from .classifiers import (
    CATEGORICAL_NB_NUMERIC_BINS,
    categorical_nb_statistics,
    categorical_nb_features,
    encoded_numeric_mask,
    fit_clean_categorical_nb,
    fit_clean_random_forest,
    fit_clean_softmax,
    fit_private_categorical_nb,
    fit_private_random_forest,
    fit_private_random_forest_exponential,
    fit_private_random_forest_gaussian,
    fit_private_sgd_softmax,
    make_random_forest_plan,
    privatize_softmax,
    random_forest_leaf_counts,
)
from .mechanisms import analytic_gaussian_sigma, zcdp_rho_for_epsilon
from .meta_features import (
    DP_META_COLUMNS,
    DP_META_EPSILON_DEFAULT,
    DP_CALIBRATION_FEATURE_NAMES,
    DP_META_NAMES,
    PRIVATE_META_COLUMNS,
    PRIVATE_META_NAMES,
    dp_calibration_features,
    dp_training_meta_features,
    private_training_meta_features,
)
from .model import (
    DELTA_TOTAL,
    balanced_accuracy,
    fit_private,
    model_from_statistics,
    predict,
    sufficient_statistics,
)
from .openml_benchmark import (
    DEFAULT_CACHE,
    DEFAULT_SEED,
    _get_openml_data,
    _load_manifest,
    _openml,
    _parse_epsilon,
    _seed_for,
    stratified_holdout_indices,
)
from .provenance import (numerical_implementation_sha256, runtime_fingerprint,
                         save_source_snapshot)
from .schema import PUBLIC_FEATURE_NAMES, PublicSchema, public_features

PIPELINES = (
    "gnb__gaussian_zcdp_equal",
    "gnb__laplace_equal",
    "gnb__staircase_vector_equal",
    "gnb__analytic_gaussian_equal",
    "softmax__analytic_gaussian",
    "softmax__dp_sgd",
    "random_forest__laplace_parallel_shards_25",
    "softmax__dp_sgd_steps100",
    "softmax__dp_sgd_clip2_reg003",
    "random_forest__analytic_gaussian_parallel_shards",
    "categorical_naive_bayes__laplace_histograms",
    "random_forest__exponential_majority_parallel_shards",
)
# Preserve the historical tuple/default and selector feature layout.
NB8U_PIPELINE = "categorical_naive_bayes__laplace_histograms_b8u"
CATEGORICAL_NB_OPTIONS = {
    PIPELINES[10]: {"numeric_bins": 16, "uniform_prior": False},
    NB8U_PIPELINE: {"numeric_bins": 8, "uniform_prior": True},
}
LAMRULE_PIPELINE = "softmax__dp_sgd_clip2_lamrule"
LAM0003_PIPELINE = "softmax__dp_sgd_clip2_lam0003"
SGD_NEW_PIPELINES = (LAMRULE_PIPELINE, LAM0003_PIPELINE)
PIPELINE_REGISTRY = (*PIPELINES, NB8U_PIPELINE, *SGD_NEW_PIPELINES)
LEGACY_PIPELINE_IDS = {
    "gnb__staircase_equal",
    "random_forest__laplace_parallel_shards",
    "random_forest__laplace_parallel_shards_depth2",
}
H27_PIPELINE = PIPELINES[8]
PUBLIC_LAMBDA_RULE = Path(__file__).resolve().parents[1] / "results/experiments/e6_e7/E6_final_public_rule.json"


def sgd_regularization(pipeline: str, n_instances: int, epsilon: float) -> float:
    """Frozen public policy: n is the manifest total, never the training split."""
    if pipeline == H27_PIPELINE:
        return DP_SGD_CLIP2_REGULARIZATION
    if pipeline == LAM0003_PIPELINE:
        return 0.0003
    if pipeline != LAMRULE_PIPELINE:
        raise ValueError(f"not a clip2 SGD option: {pipeline}")
    if n_instances <= 0 or epsilon <= 0:
        raise ValueError("public n_instances and epsilon must be positive")
    rule = json.loads(PUBLIC_LAMBDA_RULE.read_text())
    edges = np.asarray([float(v) for v in rule["bin_edges_log10_n_epsilon"]])
    index = int(np.digitize(np.log10(n_instances * epsilon), edges[1:-1], right=False))
    return float(rule["lambdas"][index])
PIPELINE_DISPLAY = {
    PIPELINES[0]: "Gaussiano (zCDP) · GaussianNB",
    PIPELINES[1]: "Laplace · GaussianNB",
    PIPELINES[2]: "Staircase vetorial · GaussianNB",
    PIPELINES[3]: "Gaussiano analítico · GaussianNB",
    PIPELINES[4]: "Gaussiano analítico · Softmax",
    PIPELINES[5]: "DP-SGD · Softmax (25 passos)",
    PIPELINES[6]: "Laplace · Random Forest (25 árvores, profundidade 3)",
    PIPELINES[7]: "DP-SGD · Softmax (100 passos)",
    PIPELINES[8]: "DP-SGD · Softmax (100 passos, clip 2, λ=0,03)",
    PIPELINES[9]: "Gaussiano analítico · Random Forest (25 árvores, profundidade 3)",
    PIPELINES[10]: "Laplace · Naive Bayes categórico",
    PIPELINES[11]: "Exponencial · Random Forest (25 árvores, profundidade 3)",
}
TREE_COUNT = 1000
SELECTOR_TREE_COUNT = TREE_COUNT
SELECTOR_N_JOBS = min(8, os.cpu_count() or 1)
FOREST_CANDIDATE_TREE_COUNT = 25
FOREST_SMALL_TREE_COUNT = 25
TREE_DEPTH = 3
DP_SGD_STEPS = 25
DP_SGD_STEPS_ALTERNATE = 100
DP_SGD_GRADIENT_CLIP = 1.0
DP_SGD_REGULARIZATION = 0.1
DP_SGD_CLIP2_GRADIENT_CLIP = 2.0
DP_SGD_CLIP2_REGULARIZATION = 0.03
SOFTMAX_REGULARIZATION = 0.1
SOFTMAX_OPTIMIZER_TOLERANCE = 1e-5
FEATURE_MODES = (
    "public_schema_only", "public_plus_dp", "research_privileged", "research_rich",
    "dp_research_rich",
)
PUBLIC_ONLY_FEATURE_MODES = ("public_schema_only",)
DEPLOYABLE_FEATURE_MODES = ("public_schema_only", "public_plus_dp")
PRIMARY_FEATURE_MODES = ("public_plus_dp",)
GNB_FEATURE_WIDTH = len(DP_CALIBRATION_FEATURE_NAMES)
SOFTMAX_FEATURE_WIDTH = 6
DP_SGD_FEATURE_WIDTH = 8
FOREST_FEATURE_WIDTH = 8
CATEGORICAL_NB_FEATURE_WIDTH = 8
DP_FEATURE_WIDTH = (
    GNB_FEATURE_WIDTH + SOFTMAX_FEATURE_WIDTH + DP_SGD_FEATURE_WIDTH
    + FOREST_FEATURE_WIDTH + CATEGORICAL_NB_FEATURE_WIDTH
)
DEFAULT_EPSILONS = (0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
CONDITION_KEYS = ("dataset_id", "epsilon", "pipeline", "repetition")


def complete_dataset_ids(results: pd.DataFrame, *, epsilons: tuple[float, ...],
                         repeats: int, pipelines: tuple[str, ...] = PIPELINES) -> set[int]:
    """Identify datasets with exactly the expected condition grid."""
    required = set(CONDITION_KEYS)
    if required - set(results.columns):
        raise ValueError(f"colunas de condição ausentes: {sorted(required - set(results.columns))}")
    expected = len(epsilons) * len(pipelines) * repeats
    allowed_epsilon = set(map(float, epsilons))
    allowed_pipeline = set(pipelines)
    allowed_repetition = set(range(repeats))
    completed = set()
    for dataset_id, block in results.groupby("dataset_id", sort=False):
        if (len(block) == expected
                and not block.duplicated(list(CONDITION_KEYS)).any()
                and set(block.epsilon.astype(float)) <= allowed_epsilon
                and set(block.pipeline) <= allowed_pipeline
                and set(block.repetition.astype(int)) <= allowed_repetition):
            completed.add(int(dataset_id))
    return completed


def validate_condition_grid(results: pd.DataFrame, manifest: pd.DataFrame, *,
                            epsilons: tuple[float, ...], repeats: int,
                            pipelines: tuple[str, ...] = PIPELINES) -> None:
    """Reject partial or foreign conditions before analysis."""
    observed = set(results.dataset_id.astype(int))
    expected_ids = set(manifest.dataset_id.astype(int))
    if not observed or not observed <= expected_ids:
        raise ValueError("conditions.csv está vazio ou contém IDs fora do manifesto")
    completed = complete_dataset_ids(results, epsilons=epsilons, repeats=repeats,
                                     pipelines=pipelines)
    if completed != observed:
        raise ValueError(f"grade incompleta ou inválida em {len(observed - completed)} datasets")
    metrics = ("private_ba", "clean_ba", "best_clean_portfolio_ba",
               "portfolio_regret", "pipeline_privacy_penalty")
    if any(column not in results for column in metrics):
        raise ValueError("colunas de score obrigatórias ausentes")
    if not np.isfinite(results[list(metrics)].to_numpy(dtype=float)).all():
        raise ValueError("pontuações não finitas em conditions.csv")


def _ledger_fields(ledger) -> dict:
    if hasattr(ledger, "as_dict"):
        value = ledger.as_dict()
        return {
            "epsilon_used": float(value["epsilon_used"]),
            "delta_used": float(value["delta_used"]),
            "ledger_json": json.dumps(value, sort_keys=True),
        }
    return {
        "epsilon_used": float(ledger["epsilon_used"]),
        "delta_used": float(ledger["delta_used"]),
        "ledger_json": json.dumps(ledger, sort_keys=True),
    }


def _run_clip2_options(openml, row, epsilons, repeats, seed, *, pipelines,
                       include_research_features, numeric_transform, dp_meta_epsilon,
                       best_clean_by_condition):
    """Parameterized H27 variants; retain H27 seed and exact numerical kernel."""
    x, y, schema = _get_openml_data(openml, row, numeric_transform=numeric_transform)
    rows, metadata = [], {}
    for repetition in range(repeats):
        tr, te = stratified_holdout_indices(
            y, _seed_for(seed, int(row.dataset_id), repetition, "portfolio_split"))
        xx, yy, xt, yt = x[tr], y[tr], x[te], y[te]
        if repetition == 0 and include_research_features:
            research = private_training_meta_features(xx, yy, len(schema.classes))
            dp_research, meta_ledger = dp_training_meta_features(
                xx, yy, len(schema.classes), epsilon=dp_meta_epsilon,
                seed=_seed_for(seed, int(row.dataset_id), repetition, "dp_meta"))
            metadata = {
                **dict(zip(PRIVATE_META_COLUMNS, map(float, research))),
                **dict(zip(DP_META_COLUMNS, map(float, dp_research))),
                "metadata_feature_mode": "dp_released_train",
                "dp_meta_epsilon": float(dp_meta_epsilon),
                "dp_meta_epsilon_used": float(meta_ledger.epsilon_used),
                "dp_meta_delta_used": float(meta_ledger.delta_used),
                "dp_meta_ledger_json": json.dumps(meta_ledger.as_dict(), sort_keys=True),
            }
        elif not include_research_features:
            metadata = dict(metadata_feature_mode="public_schema_only", dp_meta_epsilon=0.,
                            dp_meta_epsilon_used=0., dp_meta_delta_used=0.)
        # Cache only exact duplicate fits (same split, lambda, epsilon and seed).
        clean_cache = {}
        for epsilon in epsilons:
            private_cache = {}
            fitted = []
            for pipeline in pipelines:
                lam = sgd_regularization(pipeline, int(row.n_instances), epsilon)
                if lam not in clean_cache:
                    clean = fit_clean_softmax(
                        xx, yy, len(schema.classes), regularization=lam,
                        optimizer_tolerance=SOFTMAX_OPTIMIZER_TOLERANCE,
                        record_norm_bound=sqrt(len(schema.feature_names) + 1))
                    clean_cache[lam] = balanced_accuracy(yt, clean.model.predict(xt))
                if lam not in private_cache:
                    model, ledger = fit_private_sgd_softmax(
                        xx, yy, len(schema.classes), epsilon=epsilon, delta=DELTA_TOTAL,
                        seed=_seed_for(seed, int(row.dataset_id), repetition, epsilon,
                                       "dp_sgd_clip2_reg003"),
                        n_steps=DP_SGD_STEPS_ALTERNATE,
                        gradient_clip=DP_SGD_CLIP2_GRADIENT_CLIP, regularization=lam)
                    private_cache[lam] = (balanced_accuracy(yt, model.predict(xt)), ledger)
                score, ledger = private_cache[lam]
                fitted.append((pipeline, lam, score, ledger, clean_cache[lam]))
            best = (best_clean_by_condition[(float(epsilon), repetition)]
                    if best_clean_by_condition is not None else max(v[4] for v in fitted))
            for pipeline, lam, score, ledger, clean_ba in fitted:
                fields = _ledger_fields(ledger)
                rows.append({
                    "dataset_id": int(row.dataset_id), "family_id": str(row.family_id),
                    "epsilon": float(epsilon), "pipeline": pipeline, "repetition": repetition,
                    "private_ba": score, "clean_ba": clean_ba,
                    "best_clean_portfolio_ba": best,
                    "pipeline_privacy_penalty": clean_ba - score,
                    "portfolio_regret": best - score,
                    "train_n": len(tr), "test_n": len(te), "encoded_dim": schema.encoded_dim,
                    "original_feature_count": len(schema.feature_names),
                    "class_count": len(schema.classes), "numeric_transform": numeric_transform,
                    "regularization": lam, "public_n_instances": int(row.n_instances),
                    **metadata, **fields,
                    "total_epsilon_basic_bound": fields["epsilon_used"] + metadata["dp_meta_epsilon_used"],
                    "total_delta_bound": fields["delta_used"] + metadata["dp_meta_delta_used"],
                })
    return rows


def _run_dataset_legacy(openml, row: pd.Series, epsilons: tuple[float, ...], repeats: int,
                seed: int, *, include_research_features: bool = True,
                numeric_transform: str = "tanh",
                dp_meta_epsilon: float = DP_META_EPSILON_DEFAULT) -> list[dict]:
    """Evaluate one dataset and emit private-model outcomes.

    When ``include_research_features`` is false, only public schema metadata is
    available to the selector.  In particular, no train-row metadata release
    is created and the metadata privacy ledger contributes exactly zero.
    """
    x, y, schema = _get_openml_data(openml, row, numeric_transform=numeric_transform)
    rows: list[dict] = []
    research_features = None
    dp_research_features = None
    for repetition in range(repeats):
        train_indices, test_indices = stratified_holdout_indices(
            y, _seed_for(seed, int(row.dataset_id), repetition, "portfolio_split"),
        )
        x_train, x_test = x[train_indices], x[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]

        # Optional train-row descriptors are computed only on the first split.
        # The privileged diagnostic is not DP; the separate DP release is
        # explicit and is never part of the public-only selector.
        if repetition == 0 and include_research_features:
            research_features = private_training_meta_features(
                x_train, y_train, len(schema.classes),
            )
            dp_research_features, dp_meta_ledger = dp_training_meta_features(
                x_train, y_train, len(schema.classes), epsilon=dp_meta_epsilon,
                seed=_seed_for(seed, int(row.dataset_id), repetition, "dp_meta"),
            )

        gnb_stats = sufficient_statistics(x_train, y_train, len(schema.classes))
        clean_gnb = model_from_statistics(gnb_stats)
        clean_softmax = fit_clean_softmax(
            x_train, y_train, len(schema.classes),
            regularization=SOFTMAX_REGULARIZATION,
            optimizer_tolerance=SOFTMAX_OPTIMIZER_TOLERANCE,
            record_norm_bound=sqrt(len(schema.feature_names) + 1),
        )
        clean_softmax_clip2 = fit_clean_softmax(
            x_train, y_train, len(schema.classes),
            regularization=DP_SGD_CLIP2_REGULARIZATION,
            optimizer_tolerance=SOFTMAX_OPTIMIZER_TOLERANCE,
            record_norm_bound=sqrt(len(schema.feature_names) + 1),
        )
        categorical_stats = categorical_nb_statistics(
            x_train, y_train, len(schema.classes), schema,
        )
        clean_categorical_nb, _ = fit_clean_categorical_nb(
            x_train, y_train, len(schema.classes), schema,
            statistics=categorical_stats,
        )
        categorical_test, _ = categorical_nb_features(x_test, schema)
        forest_plan_small = make_random_forest_plan(
            len(y_train), encoded_numeric_mask(schema),
            seed=_seed_for(seed, int(row.dataset_id), repetition, "tree_plan_small"),
            n_trees=FOREST_SMALL_TREE_COUNT, depth=TREE_DEPTH,
        )
        forest_counts_small = random_forest_leaf_counts(
            x_train, y_train, len(schema.classes), forest_plan_small,
        )
        clean_forest_small = fit_clean_random_forest(
            x_train, y_train, len(schema.classes), forest_plan_small,
            leaf_counts=forest_counts_small,
        )
        clean_models = {
            PIPELINES[0]: predict(x_test, clean_gnb),
            PIPELINES[1]: predict(x_test, clean_gnb),
            PIPELINES[2]: predict(x_test, clean_gnb),
            PIPELINES[3]: predict(x_test, clean_gnb),
            PIPELINES[4]: clean_softmax.model.predict(x_test),
            PIPELINES[5]: clean_softmax.model.predict(x_test),
            PIPELINES[6]: clean_forest_small.predict(x_test),
            PIPELINES[7]: clean_softmax.model.predict(x_test),
            PIPELINES[8]: clean_softmax_clip2.model.predict(x_test),
            PIPELINES[9]: clean_forest_small.predict(x_test),
            PIPELINES[10]: clean_categorical_nb.predict(categorical_test),
            PIPELINES[11]: clean_forest_small.predict(x_test),
        }
        clean_scores = {
            key: balanced_accuracy(y_test, prediction)
            for key, prediction in clean_models.items()
        }
        best_clean_score = max(clean_scores.values())

        for epsilon in epsilons:
            gnb_models = {}
            for pipeline, mechanism in (
                (PIPELINES[0], "gaussian_zcdp"),
                (PIPELINES[1], "laplace"),
                (PIPELINES[2], "staircase"),
                (PIPELINES[3], "analytic_gaussian"),
            ):
                gnb_models[pipeline] = fit_private(
                    x_train, y_train, len(schema.classes), epsilon=epsilon,
                    mechanism=mechanism, allocation="equal",
                    seed=_seed_for(seed, int(row.dataset_id), repetition, epsilon, pipeline),
                    stats=gnb_stats,
                )
            softmax_model, softmax_ledger = privatize_softmax(
                clean_softmax, epsilon=epsilon,
                seed=_seed_for(seed, int(row.dataset_id), repetition, epsilon, "softmax"),
            )
            dp_sgd_model, dp_sgd_ledger = fit_private_sgd_softmax(
                x_train, y_train, len(schema.classes), epsilon=epsilon,
                delta=DELTA_TOTAL,
                seed=_seed_for(seed, int(row.dataset_id), repetition, epsilon, "dp_sgd"),
                n_steps=DP_SGD_STEPS,
                gradient_clip=DP_SGD_GRADIENT_CLIP,
                regularization=DP_SGD_REGULARIZATION,
            )
            dp_sgd_model_steps100, dp_sgd_ledger_steps100 = fit_private_sgd_softmax(
                x_train, y_train, len(schema.classes), epsilon=epsilon,
                delta=DELTA_TOTAL,
                seed=_seed_for(seed, int(row.dataset_id), repetition, epsilon, "dp_sgd_steps100"),
                n_steps=DP_SGD_STEPS_ALTERNATE,
                gradient_clip=DP_SGD_GRADIENT_CLIP,
                regularization=DP_SGD_REGULARIZATION,
            )
            dp_sgd_model_clip2, dp_sgd_ledger_clip2 = fit_private_sgd_softmax(
                x_train, y_train, len(schema.classes), epsilon=epsilon,
                delta=DELTA_TOTAL,
                seed=_seed_for(seed, int(row.dataset_id), repetition, epsilon,
                               "dp_sgd_clip2_reg003"),
                n_steps=DP_SGD_STEPS_ALTERNATE,
                gradient_clip=DP_SGD_CLIP2_GRADIENT_CLIP,
                regularization=DP_SGD_CLIP2_REGULARIZATION,
            )
            forest_model, forest_ledger = fit_private_random_forest(
                x_train, y_train, len(schema.classes), epsilon=epsilon,
                seed=_seed_for(seed, int(row.dataset_id), repetition, epsilon,
                               "forest_laplace_small"),
                plan=forest_plan_small, leaf_counts=forest_counts_small,
            )
            forest_gaussian_model, forest_gaussian_ledger = fit_private_random_forest_gaussian(
                x_train, y_train, len(schema.classes), epsilon=epsilon,
                delta=DELTA_TOTAL,
                seed=_seed_for(seed, int(row.dataset_id), repetition, epsilon,
                               "forest_analytic_gaussian"),
                plan=forest_plan_small, leaf_counts=forest_counts_small,
            )
            categorical_nb_model, _, categorical_nb_ledger = fit_private_categorical_nb(
                x_train, y_train, len(schema.classes), schema, epsilon=epsilon,
                seed=_seed_for(seed, int(row.dataset_id), repetition, epsilon,
                               "categorical_nb_laplace"), statistics=categorical_stats,
            )
            forest_exponential_model, forest_exponential_ledger = (
                fit_private_random_forest_exponential(
                    x_train, y_train, len(schema.classes), epsilon=epsilon,
                    seed=_seed_for(seed, int(row.dataset_id), repetition, epsilon,
                                   "forest_exponential"),
                    plan=forest_plan_small, leaf_counts=forest_counts_small,
                )
            )
            private_models = {
                **{
                    pipeline: (predict(x_test, model), ledger)
                    for pipeline, (model, ledger) in gnb_models.items()
                },
                PIPELINES[4]: (softmax_model.predict(x_test), softmax_ledger),
                PIPELINES[5]: (dp_sgd_model.predict(x_test), dp_sgd_ledger),
                PIPELINES[6]: (forest_model.predict(x_test), forest_ledger),
                PIPELINES[7]: (dp_sgd_model_steps100.predict(x_test), dp_sgd_ledger_steps100),
                PIPELINES[8]: (dp_sgd_model_clip2.predict(x_test), dp_sgd_ledger_clip2),
                PIPELINES[9]: (forest_gaussian_model.predict(x_test), forest_gaussian_ledger),
                PIPELINES[10]: (
                    categorical_nb_model.predict(categorical_test), categorical_nb_ledger,
                ),
                PIPELINES[11]: (
                    forest_exponential_model.predict(x_test), forest_exponential_ledger,
                ),
            }
            for pipeline in PIPELINES:
                prediction, ledger = private_models[pipeline]
                private_score = balanced_accuracy(y_test, prediction)
                ledger_data = _ledger_fields(ledger)
                if dp_research_features is not None:
                    metadata_fields = {
                        "metadata_feature_mode": "dp_released_train",
                        "dp_meta_epsilon": float(dp_meta_epsilon),
                        "dp_meta_epsilon_used": float(dp_meta_ledger.epsilon_used),
                        "dp_meta_delta_used": float(dp_meta_ledger.delta_used),
                        "dp_meta_ledger_json": json.dumps(
                            dp_meta_ledger.as_dict(), sort_keys=True,
                        ),
                        # Conservative basic-composition view for deployment:
                        # the DP metadata release is emitted once per split
                        # and must be added to the model release.
                        "total_epsilon_basic_bound": float(
                            ledger_data["epsilon_used"] + dp_meta_ledger.epsilon_used
                        ),
                        "total_delta_bound": float(
                            ledger_data["delta_used"] + dp_meta_ledger.delta_used
                        ),
                    }
                else:
                    metadata_fields = {
                        "metadata_feature_mode": "public_schema_only",
                        "dp_meta_epsilon": 0.0,
                        "dp_meta_epsilon_used": 0.0,
                        "dp_meta_delta_used": 0.0,
                        # Public metadata is post-processing of the public
                        # manifest/schema; no metadata release is composed.
                        "total_epsilon_basic_bound": float(ledger_data["epsilon_used"]),
                        "total_delta_bound": float(ledger_data["delta_used"]),
                    }
                rows.append({
                    "dataset_id": int(row.dataset_id),
                    "family_id": str(row.family_id),
                    "epsilon": float(epsilon),
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
                    "numeric_transform": numeric_transform,
                    **({column: float(value) for column, value in
                        zip(PRIVATE_META_COLUMNS, research_features)}
                       if research_features is not None else {}),
                    **({column: float(value) for column, value in
                        zip(DP_META_COLUMNS, dp_research_features)}
                       if dp_research_features is not None else {}),
                    **metadata_fields,
                    **ledger_data,
                })
    return rows


def run_dataset(openml, row: pd.Series, epsilons: tuple[float, ...], repeats: int,
                seed: int, *, include_research_features: bool = True,
                numeric_transform: str = "tanh",
                dp_meta_epsilon: float = DP_META_EPSILON_DEFAULT,
                pipelines: tuple[str, ...] | None = None,
                best_clean_by_condition: dict[tuple[float, int], float] | None = None,
                ) -> list[dict]:
    """Evaluate selected options; the default preserves the original 12-option run.

    NB variants share the historical NB seed namespace, including its prior RNG
    draw. A subset run can supply the original clean portfolio reference. Without
    that reference, its regret is relative to the selected clean NB options.
    """
    selected = PIPELINES if pipelines is None else tuple(pipelines)
    if not selected or len(set(selected)) != len(selected) or set(selected) - set(PIPELINE_REGISTRY):
        raise ValueError("pipelines vazias, duplicadas ou desconhecidas")
    kwargs = dict(include_research_features=include_research_features,
                  numeric_transform=numeric_transform, dp_meta_epsilon=dp_meta_epsilon)
    if selected == PIPELINES and best_clean_by_condition is None:
        return _run_dataset_legacy(openml, row, epsilons, repeats, seed, **kwargs)
    nb_selected = tuple(p for p in selected if p in CATEGORICAL_NB_OPTIONS)
    sgd_selected = tuple(p for p in selected if p in SGD_NEW_PIPELINES)
    other_selected = tuple(p for p in selected
                           if p not in CATEGORICAL_NB_OPTIONS and p not in SGD_NEW_PIPELINES)
    rows = []
    if other_selected == (H27_PIPELINE,) and best_clean_by_condition is not None:
        sgd_selected = (H27_PIPELINE, *sgd_selected)
        other_selected = ()
    if sgd_selected:
        rows.extend(_run_clip2_options(
            openml, row, epsilons, repeats, seed, pipelines=sgd_selected,
            best_clean_by_condition=best_clean_by_condition, **kwargs))
    if other_selected:
        rows.extend(r for r in _run_dataset_legacy(openml, row, epsilons, repeats, seed, **kwargs)
                    if r["pipeline"] in selected)
        # The legacy run already emitted the current NB, if selected.
        nb_selected = tuple(p for p in nb_selected if p != PIPELINES[10])
    if nb_selected:
        x, y, schema = _get_openml_data(openml, row, numeric_transform=numeric_transform)
        metadata = {}
        for repetition in range(repeats):
            split_seed = _seed_for(seed, int(row.dataset_id), repetition, "portfolio_split")
            tr, te = stratified_holdout_indices(y, split_seed)
            xx, yy, xt, yt = x[tr], y[tr], x[te], y[te]
            if repetition == 0 and include_research_features:
                research = private_training_meta_features(xx, yy, len(schema.classes))
                dp_research, meta_ledger = dp_training_meta_features(
                    xx, yy, len(schema.classes), epsilon=dp_meta_epsilon,
                    seed=_seed_for(seed, int(row.dataset_id), repetition, "dp_meta"))
                metadata = {
                    **dict(zip(PRIVATE_META_COLUMNS, map(float, research))),
                    **dict(zip(DP_META_COLUMNS, map(float, dp_research))),
                    "metadata_feature_mode": "dp_released_train",
                    "dp_meta_epsilon": float(dp_meta_epsilon),
                    "dp_meta_epsilon_used": float(meta_ledger.epsilon_used),
                    "dp_meta_delta_used": float(meta_ledger.delta_used),
                    "dp_meta_ledger_json": json.dumps(meta_ledger.as_dict(), sort_keys=True),
                }
            elif not include_research_features:
                metadata = dict(metadata_feature_mode="public_schema_only", dp_meta_epsilon=0.,
                                dp_meta_epsilon_used=0., dp_meta_delta_used=0.)
            prepared = {}
            for pipeline in nb_selected:
                params = CATEGORICAL_NB_OPTIONS[pipeline]
                stats = categorical_nb_statistics(xx, yy, len(schema.classes), schema,
                                                 numeric_bins=params["numeric_bins"])
                clean, _ = fit_clean_categorical_nb(xx, yy, len(schema.classes), schema,
                                                   statistics=stats, **params)
                test, _ = categorical_nb_features(xt, schema, numeric_bins=params["numeric_bins"])
                prepared[pipeline] = (stats, test, balanced_accuracy(yt, clean.predict(test)))
            for epsilon in epsilons:
                best = (best_clean_by_condition[(float(epsilon), repetition)]
                        if best_clean_by_condition is not None else
                        max([v[2] for v in prepared.values()] + [
                            r["best_clean_portfolio_ba"] for r in rows
                            if r["epsilon"] == epsilon and r["repetition"] == repetition]))
                for pipeline, (stats, test, clean_ba) in prepared.items():
                    dp_seed = _seed_for(seed, int(row.dataset_id), repetition, epsilon,
                                        "categorical_nb_laplace")
                    model, _, ledger = fit_private_categorical_nb(
                        xx, yy, len(schema.classes), schema, epsilon=epsilon, seed=dp_seed,
                        statistics=stats, **CATEGORICAL_NB_OPTIONS[pipeline])
                    score = balanced_accuracy(yt, model.predict(test))
                    fields = _ledger_fields(ledger)
                    rows.append({
                        "dataset_id": int(row.dataset_id), "family_id": str(row.family_id),
                        "epsilon": float(epsilon), "pipeline": pipeline, "repetition": repetition,
                        "private_ba": score, "clean_ba": clean_ba,
                        "best_clean_portfolio_ba": best,
                        "pipeline_privacy_penalty": clean_ba - score,
                        "portfolio_regret": best - score,
                        "train_n": len(tr), "test_n": len(te), "encoded_dim": schema.encoded_dim,
                        "original_feature_count": len(schema.feature_names),
                        "class_count": len(schema.classes), "numeric_transform": numeric_transform,
                        **metadata, **fields,
                        "total_epsilon_basic_bound": fields["epsilon_used"] + metadata["dp_meta_epsilon_used"],
                        "total_delta_bound": fields["delta_used"] + metadata["dp_meta_delta_used"],
                    })
    if best_clean_by_condition is not None:
        for result in rows:
            best = best_clean_by_condition[(result["epsilon"], result["repetition"])]
            result["best_clean_portfolio_ba"] = best
            result["portfolio_regret"] = best - result["private_ba"]
    return sorted(rows, key=lambda r: (r["repetition"], epsilons.index(r["epsilon"]),
                                     selected.index(r["pipeline"])))


def _public_dp_features(n: int, schema: PublicSchema, epsilon: float,
                        pipeline: str, *, delta_total: float = DELTA_TOTAL) -> np.ndarray:
    """Public, action-aware calibration descriptors, not private-data statistics."""
    if pipeline in SGD_NEW_PIPELINES:
        values = _public_dp_features(n, schema, epsilon, H27_PIPELINE, delta_total=delta_total)
        values[GNB_FEATURE_WIDTH + SOFTMAX_FEATURE_WIDTH + 5] = log1p(
            sgd_regularization(pipeline, n, epsilon))
        return values
    values = np.zeros(DP_FEATURE_WIDTH, dtype=float)
    if pipeline == PIPELINES[0]:
        calibration = dp_calibration_features(
            n, schema, epsilon, "gaussian_zcdp", "equal", delta_total=delta_total,
        )
        values[:GNB_FEATURE_WIDTH] = calibration
        return values
    if pipeline == PIPELINES[1]:
        calibration = dp_calibration_features(
            n, schema, epsilon, "laplace", "equal", delta_total=0.0,
        )
        values[:GNB_FEATURE_WIDTH] = calibration
        return values
    if pipeline == PIPELINES[2]:
        calibration = dp_calibration_features(
            n, schema, epsilon, "staircase", "equal", delta_total=0.0,
        )
        values[:GNB_FEATURE_WIDTH] = calibration
        return values
    if pipeline == PIPELINES[3]:
        calibration = dp_calibration_features(
            n, schema, epsilon, "analytic_gaussian", "equal", delta_total=delta_total,
        )
        values[:GNB_FEATURE_WIDTH] = calibration
        return values
    if pipeline == PIPELINES[4]:
        dimension = (schema.encoded_dim + 1) * len(schema.classes)
        record_norm = sqrt(len(schema.feature_names) + 1)
        sensitivity = (
            2 * sqrt(2) * record_norm / (SOFTMAX_REGULARIZATION * n)
            + 2 * SOFTMAX_OPTIMIZER_TOLERANCE
        )
        sigma = analytic_gaussian_sigma(epsilon, delta_total, sensitivity)
        block = np.asarray([
            log1p(sensitivity), log1p(sigma), log1p(sigma * sqrt(dimension)),
            log1p(SOFTMAX_REGULARIZATION), log1p(dimension),
            log1p(n * SOFTMAX_REGULARIZATION),
        ], dtype=float)
        start = GNB_FEATURE_WIDTH
        values[start:start + SOFTMAX_FEATURE_WIDTH] = block
        return values
    if pipeline in {PIPELINES[5], PIPELINES[7], PIPELINES[8]}:
        dimension = (schema.encoded_dim + 1) * len(schema.classes)
        rho_total = zcdp_rho_for_epsilon(epsilon, delta_total)
        if pipeline == PIPELINES[8]:
            gradient_clip = DP_SGD_CLIP2_GRADIENT_CLIP
            regularization = DP_SGD_CLIP2_REGULARIZATION
            n_steps = DP_SGD_STEPS_ALTERNATE
        else:
            gradient_clip = DP_SGD_GRADIENT_CLIP
            regularization = DP_SGD_REGULARIZATION
            n_steps = DP_SGD_STEPS if pipeline == PIPELINES[5] else DP_SGD_STEPS_ALTERNATE
        sensitivity = 2.0 * gradient_clip / n
        rho_step = rho_total / n_steps
        sigma = sensitivity / sqrt(2.0 * rho_step)
        block = np.asarray([
            log1p(sensitivity), log1p(sigma), log1p(sigma * sqrt(dimension)),
            log1p(n_steps), log1p(gradient_clip),
            log1p(regularization), log1p(dimension), log1p(rho_total),
        ], dtype=float)
        start = GNB_FEATURE_WIDTH + SOFTMAX_FEATURE_WIDTH
        values[start:start + DP_SGD_FEATURE_WIDTH] = block
        return values
    if pipeline in {PIPELINES[6], PIPELINES[9], PIPELINES[11]}:
        depth = TREE_DEPTH
        tree_count = FOREST_CANDIDATE_TREE_COUNT
        if pipeline == PIPELINES[9]:
            scale = analytic_gaussian_sigma(epsilon, delta_total, sqrt(2.0))
        elif pipeline == PIPELINES[11]:
            # Exponential mechanism uses epsilon/2 per leaf to account for
            # replacement adjacency affecting at most two leaf outputs.
            scale = 4.0 / epsilon
        else:
            scale = 2.0 / epsilon
        leaf_count = 2**depth
        shard_size = n / min(tree_count, n)
        block = np.asarray([
            log1p(2.0), log1p(scale), log1p(scale * sqrt(leaf_count * len(schema.classes))),
            log1p(tree_count), log1p(depth), log1p(leaf_count),
            log1p(shard_size), log1p(leaf_count * len(schema.classes)),
        ], dtype=float)
        start = GNB_FEATURE_WIDTH + SOFTMAX_FEATURE_WIDTH + DP_SGD_FEATURE_WIDTH
        values[start:start + FOREST_FEATURE_WIDTH] = block
        return values
    if pipeline in CATEGORICAL_NB_OPTIONS:
        numeric_bins = CATEGORICAL_NB_OPTIONS[pipeline]["numeric_bins"]
        n_features = len(schema.feature_names)
        n_queries = n_features + 1
        epsilon_per_query = epsilon / n_queries
        noise_scale = 2.0 / epsilon_per_query
        n_nominal = sum(kind == "nominal" for kind in schema.feature_kinds)
        n_categories = sum(
            len(domain) + 1 if kind == "nominal" else numeric_bins
            for kind, domain in zip(schema.feature_kinds, schema.nominal_domains)
        )
        block = np.asarray([
            log1p(n_features), log1p(n_queries), log1p(n_categories),
            log1p(epsilon_per_query), log1p(noise_scale), log1p(n * epsilon_per_query),
            log1p(n_nominal), log1p(numeric_bins),
        ], dtype=float)
        start = GNB_FEATURE_WIDTH + SOFTMAX_FEATURE_WIDTH + DP_SGD_FEATURE_WIDTH
        start += FOREST_FEATURE_WIDTH
        values[start:start + CATEGORICAL_NB_FEATURE_WIDTH] = block
        return values
    raise ValueError(f"pipeline desconhecido: {pipeline}")


def _candidate_features(meta_row: pd.Series, epsilon: float, pipeline: str,
                        mode: str, research_by_dataset: dict[int, np.ndarray],
                        dp_research_by_dataset: dict[int, np.ndarray] | None = None
                        ) -> np.ndarray:
    # Share the H27 action identity; lambda is already a public SGD calibration
    # coordinate. All historical feature vectors keep their exact shape/values.
    feature_pipeline = H27_PIPELINE if pipeline in SGD_NEW_PIPELINES else pipeline
    schema = _cached_public_schema(meta_row.public_schema_json)
    n = int(meta_row.n_instances)
    base = public_features(n, schema, epsilon)
    action = np.asarray([float(feature_pipeline == item) for item in PIPELINES])
    values = [base, action]
    if mode in {"public_plus_dp", "research_privileged", "research_rich", "dp_research_rich"}:
        values.append(_public_dp_features(n, schema, epsilon, pipeline))
    if mode in {"research_privileged", "research_rich"}:
        dataset_id = int(meta_row.get("dataset_id", meta_row.name))
        private = research_by_dataset.get(dataset_id)
        if private is None:
            raise ValueError("meta-features privilegiadas ausentes")
        values.append(private)
        if mode == "research_rich":
            action = np.asarray([float(feature_pipeline == item) for item in PIPELINES])
            values.append(np.concatenate([private * indicator for indicator in action]))
            values.append(private * np.log1p(float(epsilon)))
    if mode == "dp_research_rich":
        dataset_id = int(meta_row.get("dataset_id", meta_row.name))
        private = (dp_research_by_dataset or {}).get(dataset_id)
        if private is None:
            raise ValueError("meta-features DP ausentes")
        values.append(private)
        action = np.asarray([float(feature_pipeline == item) for item in PIPELINES])
        values.append(np.concatenate([private * indicator for indicator in action]))
        values.append(private * np.log1p(float(epsilon)))
    if mode not in FEATURE_MODES:
        raise ValueError(f"modo de features desconhecido: {mode}")
    return np.concatenate(values)


_CALIBRATION_BLOCKS = (
    ("calib_gnb", DP_CALIBRATION_FEATURE_NAMES),
    ("calib_softmax_output", ("sensitivity", "sigma", "sigma_sqrt_dim", "regularization",
                              "dimension", "n_regularization")),
    ("calib_dp_sgd", ("sensitivity", "sigma", "sigma_sqrt_dim", "steps", "clip",
                      "regularization", "dimension", "rho_total")),
    ("calib_forest", ("sensitivity_l1", "scale", "scale_sqrt_cells", "trees", "depth",
                      "leaves", "shard_size", "cells")),
    ("calib_categorical_nb", ("features", "queries", "categories", "epsilon_per_query",
                              "noise_scale", "n_epsilon_per_query", "nominal",
                              "numeric_bins")),
)


def candidate_feature_names(mode: str) -> tuple[tuple[str, str], ...]:
    """(name, group) for each column of ``_candidate_features`` in ``mode``."""
    if mode not in FEATURE_MODES:
        raise ValueError(f"modo de features desconhecido: {mode}")
    names = [(name, "schema") for name in PUBLIC_FEATURE_NAMES]
    names += [(f"option={pipeline}", "option") for pipeline in PIPELINES]
    if mode != "public_schema_only":
        names += [(f"{group}__{name}", group)
                  for group, block in _CALIBRATION_BLOCKS for name in block]

    def train_block(prefix: str, meta_names: tuple[str, ...], group: str,
                    interactions: bool) -> list[tuple[str, str]]:
        block = [(f"{prefix}{name}", group) for name in meta_names]
        if interactions:
            block += [(f"{prefix}{name}*option={pipeline}", group)
                      for pipeline in PIPELINES for name in meta_names]
            block += [(f"{prefix}{name}*log1p_epsilon", group) for name in meta_names]
        return block

    if mode in {"research_privileged", "research_rich"}:
        names += train_block("private_meta__", PRIVATE_META_NAMES, "train_nondp",
                             mode == "research_rich")
    if mode == "dp_research_rich":
        names += train_block("dp_meta__", DP_META_NAMES, "train_dp", True)
    return tuple(names)


@lru_cache(maxsize=2048)
def _cached_public_schema(raw: str) -> PublicSchema:
    return PublicSchema.from_manifest_json(raw)


def _cached_candidate_features(
    meta_row: pd.Series, epsilon: float, pipeline: str, mode: str,
    research_by_dataset: dict[int, np.ndarray],
    dp_research_by_dataset: dict[int, np.ndarray] | None,
    feature_cache: dict[tuple, np.ndarray] | None,
) -> np.ndarray:
    if feature_cache is None:
        return _candidate_features(meta_row, epsilon, pipeline, mode,
                                   research_by_dataset, dp_research_by_dataset)
    key = (int(meta_row.get("dataset_id", meta_row.name)), float(epsilon), pipeline, mode)
    if key not in feature_cache:
        feature_cache[key] = _candidate_features(
            meta_row, epsilon, pipeline, mode, research_by_dataset, dp_research_by_dataset,
        )
    return feature_cache[key]


def _available_feature_modes(aggregated: pd.DataFrame) -> tuple[str, ...]:
    """Return modes supported by a conditions table without breaking old runs."""
    modes = ["public_schema_only", "public_plus_dp"]
    has_private_features = all(column in aggregated.columns for column in PRIVATE_META_COLUMNS)
    if has_private_features:
        modes.extend(("research_privileged", "research_rich"))
    if all(column in aggregated.columns for column in DP_META_COLUMNS):
        modes.append("dp_research_rich")
    return tuple(modes)


def _aggregate(results: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    # The old coordinate-composed staircase run has a different calibration
    # than the current vector implementation and is not comparable by ID.
    results = results[~results.pipeline.isin(LEGACY_PIPELINE_IDS)].copy()
    feature_columns = [column for column in (*PRIVATE_META_COLUMNS, *DP_META_COLUMNS)
                       if column in results.columns]
    averaged = (results.groupby(["dataset_id", "epsilon", "pipeline"], as_index=False,
                               sort=True)
                .agg(private_ba=("private_ba", "mean"), clean_ba=("clean_ba", "mean"),
                     best_clean_portfolio_ba=("best_clean_portfolio_ba", "mean"),
                     portfolio_regret=("portfolio_regret", "mean"),
                     pipeline_privacy_penalty=("pipeline_privacy_penalty", "mean"),
                     **{name: (name, "first") for name in feature_columns}))
    family_map = manifest.drop_duplicates("dataset_id").set_index("dataset_id").family_id
    averaged["family_id"] = averaged.dataset_id.map(family_map).astype(str)
    return averaged


def _row_weights(train: pd.DataFrame) -> np.ndarray:
    dataset_counts = train[["family_id", "dataset_id"]].drop_duplicates().groupby(
        "family_id").dataset_id.nunique()
    eps_counts = train[["dataset_id", "epsilon"]].drop_duplicates().groupby(
        "dataset_id").epsilon.nunique()
    pipeline_counts = train[["dataset_id", "epsilon", "pipeline"]].drop_duplicates().groupby(
        ["dataset_id", "epsilon"]).pipeline.nunique()
    n_families = train.family_id.nunique()
    weights = []
    for row in train.itertuples(index=False):
        weights.append(1.0 / n_families / dataset_counts[row.family_id]
                       / eps_counts[row.dataset_id]
                       / pipeline_counts.loc[(row.dataset_id, row.epsilon)])
    weights = np.asarray(weights, dtype=float)
    return weights / weights.sum()


def _fit_selector(train: pd.DataFrame, manifest_one: pd.DataFrame, mode: str,
                  research_by_dataset: dict[int, np.ndarray], seed: int,
                  dp_research_by_dataset: dict[int, np.ndarray] | None = None,
                  target_column: str = "private_ba",
                  feature_cache: dict[tuple, np.ndarray] | None = None):
    meta = manifest_one.set_index("dataset_id")
    features = [
        _cached_candidate_features(meta.loc[int(row.dataset_id)], float(row.epsilon),
                                   row.pipeline, mode, research_by_dataset,
                                   dp_research_by_dataset, feature_cache)
        for row in train.itertuples(index=False)
    ]
    model = RandomForestRegressor(
        n_estimators=SELECTOR_TREE_COUNT, min_samples_leaf=3, max_features=0.9, max_depth=6,
        max_samples=0.5,
        random_state=seed, n_jobs=SELECTOR_N_JOBS,
    )
    if target_column not in train:
        raise ValueError(f"alvo do seletor ausente: {target_column}")
    model.fit(np.vstack(features), train[target_column].to_numpy(dtype=float),
              sample_weight=_row_weights(train))
    # Prediction is repeatedly called on tiny portfolio blocks below;
    # avoid joblib process overhead there while retaining parallel fitting.
    model.set_params(n_jobs=1)
    return model, meta


SELECTOR_STRATEGIES = (
    "absolute_shared",
    "absolute_per_epsilon",
    "centered_per_epsilon",
    "regret_per_epsilon",
)


def _selector_training_block(train: pd.DataFrame, strategy: str) -> tuple[pd.DataFrame, str]:
    """Create the out-of-fold target used by one selector model.

    Centering removes the dataset/epsilon difficulty level, so the model learns
    which candidate is relatively strong for that condition.  The per-epsilon
    model is important because the ranking and the noise regime change sharply
    across the privacy budget.
    """
    block = train.copy()
    if strategy.startswith("centered"):
        block["_selector_target"] = (
            block.private_ba
            - block.groupby(["dataset_id", "epsilon"], sort=False)
            .private_ba.transform("mean")
        )
        return block, "_selector_target"
    if strategy.startswith("regret"):
        block["_selector_target"] = -block.portfolio_regret
        return block, "_selector_target"
    return block, "private_ba"


def _best_fixed(train: pd.DataFrame, epsilon: float | None = None) -> str:
    subset = train if epsilon is None else train[train.epsilon == epsilon]
    family_scores = (subset.groupby(["family_id", "pipeline"], as_index=False)
                     .private_ba.mean())
    scores = family_scores.groupby("pipeline").private_ba.mean()
    available = tuple(pipeline for pipeline in PIPELINES if pipeline in scores.index)
    if not available:
        raise ValueError("nenhum pipeline disponível para o baseline fixo")
    return max(available, key=lambda pipeline: float(scores.loc[pipeline]))


def _condition_score(condition: pd.DataFrame, pipeline: str, column: str) -> float:
    match = condition[condition.pipeline == pipeline]
    if len(match) != 1:
        raise ValueError("condição deve conter cada pipeline exatamente uma vez")
    return float(match.iloc[0][column])


def _bootstrap_ci(values: np.ndarray, seed: int, draws: int = 20000) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    choices = rng.integers(0, len(values), size=(draws, len(values)))
    means = values[choices].mean(axis=1)
    return float(values.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _bootstrap_tail_gains(selector: np.ndarray, fixed: np.ndarray, seed: int,
                          draws: int = 20000) -> dict:
    """Paired family bootstrap for regret median, q90 and CVaR90 gains."""
    selector = np.asarray(selector, dtype=float)
    fixed = np.asarray(fixed, dtype=float)
    rng = np.random.default_rng(seed)
    choices = rng.integers(0, len(selector), size=(draws, len(selector)))
    selected = selector[choices]
    baseline = fixed[choices]
    tail_count = max(1, ceil(0.10 * len(selector)))
    selected_sorted = np.sort(selected, axis=1)
    baseline_sorted = np.sort(baseline, axis=1)
    values = {
        "median": np.quantile(baseline, 0.5, axis=1)
                 - np.quantile(selected, 0.5, axis=1),
        "q90": np.quantile(baseline, 0.9, axis=1)
               - np.quantile(selected, 0.9, axis=1),
        "cvar90": baseline_sorted[:, -tail_count:].mean(axis=1)
                  - selected_sorted[:, -tail_count:].mean(axis=1),
    }
    observed = _metric_summary(fixed)
    selected_observed = _metric_summary(selector)
    point = {
        "median": observed["median"] - selected_observed["median"],
        "q90": observed["q90"] - selected_observed["q90"],
        "cvar90": observed["cvar90"] - selected_observed["cvar90"],
    }
    return {
        name: {
            "gain": float(point[name]),
            "ci95": [float(np.quantile(values[name], 0.025)),
                     float(np.quantile(values[name], 0.975))],
        }
        for name in values
    }


def _sign_flip_p(values: np.ndarray, seed: int, draws: int = 50000) -> float:
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.asarray([-1.0, 1.0]), size=(draws, len(values)))
    observed = abs(float(values.mean()))
    simulated = np.abs(np.mean(signs * values, axis=1))
    return float((1 + np.count_nonzero(simulated >= observed)) / (draws + 1))


def _metric_summary(values: np.ndarray) -> dict:
    tail_count = max(1, ceil(0.10 * len(values)))
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "q90": float(np.quantile(values, 0.90)),
        "cvar90": float(np.mean(np.sort(values)[-tail_count:])),
    }


def _safe_wilcoxon(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values) or np.allclose(values, 0.0):
        return 1.0
    return float(wilcoxon(values, zero_method="wilcox", alternative="two-sided").pvalue)


def _rank_biserial(values: np.ndarray) -> float:
    """Paired rank-biserial effect size, positive when selector gains."""
    values = np.asarray(values, dtype=float)
    nonzero = values[np.abs(values) > 1e-15]
    if not len(nonzero):
        return 0.0
    ranks = pd.Series(np.abs(nonzero)).rank(method="average").to_numpy()
    total = len(nonzero) * (len(nonzero) + 1) / 2.0
    return float((ranks[nonzero > 0].sum() - ranks[nonzero < 0].sum()) / total)


def _mc_nemar(selector_correct: np.ndarray, fixed_correct: np.ndarray) -> dict:
    selector_correct = np.asarray(selector_correct, dtype=bool)
    fixed_correct = np.asarray(fixed_correct, dtype=bool)
    selector_wins = int(np.count_nonzero(selector_correct & ~fixed_correct))
    fixed_wins = int(np.count_nonzero(~selector_correct & fixed_correct))
    discordant = selector_wins + fixed_wins
    p_value = (float(binomtest(selector_wins, discordant, 0.5,
                               alternative="two-sided").pvalue)
               if discordant else 1.0)
    return {
        "unit_of_analysis": "dataset_id",
        "selector_correct_fixed_wrong": selector_wins,
        "selector_wrong_fixed_correct": fixed_wins,
        "discordant_pairs": discordant,
        "exact_binomial_p_two_sided": p_value,
    }


def _holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    adjusted = {}
    running = 0.0
    total = len(ordered)
    for index, (name, p_value) in enumerate(ordered):
        corrected = min(1.0, (total - index) * float(p_value))
        running = max(running, corrected)
        adjusted[name] = running
    return adjusted


def _summarize_mode(predictions: pd.DataFrame, mode: str, seed: int) -> dict:
    mode_rows = predictions[predictions.feature_mode == mode]
    dataset_aggregates = {
        "selector_regret": "mean", "best_fixed_regret": "mean",
        "gnb_fixed_regret": "mean", "epsilon_fixed_regret": "mean",
        "oracle_regret": "mean", "selector_ba": "mean",
        "best_fixed_ba": "mean", "gnb_fixed_ba": "mean",
        "selector_top1": "mean", "fixed_top1": "mean",
    }
    has_h27 = ("h27_fixed_regret" in mode_rows.columns
               and mode_rows.h27_fixed_regret.notna().any())
    if has_h27:
        dataset_aggregates.update({"h27_fixed_regret": "mean", "h27_fixed_ba": "mean"})
    per_dataset = (mode_rows.groupby(["family_id", "dataset_id"], as_index=False)
                   .agg(dataset_aggregates))
    per_family = (per_dataset.groupby("family_id", as_index=False)
                  .agg({column: "mean" for column in per_dataset.columns
                        if column not in {"family_id", "dataset_id"}}))
    gain = (per_family.best_fixed_regret - per_family.selector_regret).to_numpy()
    mean_gain, lower, upper = _bootstrap_ci(gain, seed)
    p_value = _sign_flip_p(gain, seed + 1)
    gain_vs_gnb = (per_family.gnb_fixed_regret - per_family.best_fixed_regret).to_numpy()
    gnb_mean_gain, gnb_lower, gnb_upper = _bootstrap_ci(gain_vs_gnb, seed + 2)
    gnb_p_value = _sign_flip_p(gain_vs_gnb, seed + 3)
    regret_selector = _metric_summary(per_family.selector_regret.to_numpy())
    regret_fixed = _metric_summary(per_family.best_fixed_regret.to_numpy())
    gain_wilcoxon = _safe_wilcoxon(gain)
    gain_ttest = float(ttest_1samp(gain, 0.0).pvalue) if len(gain) > 1 else 1.0
    tail_gains = _bootstrap_tail_gains(
        per_family.selector_regret.to_numpy(),
        per_family.best_fixed_regret.to_numpy(), seed + 4,
    )
    # Each dataset contributes one paired outcome: majority Top-1 accuracy
    # over its epsilon grid. This avoids treating repeated epsilon conditions
    # from the same dataset as independent McNemar observations.
    top1_test = _mc_nemar(
        per_dataset.selector_top1.to_numpy() > 0.5,
        per_dataset.fixed_top1.to_numpy() > 0.5,
    )
    summary = {
        "mean_regret_gain_vs_best_fixed": mean_gain,
        "gain_ci95_family_bootstrap": [lower, upper],
        "paired_sign_flip_p_two_sided": p_value,
        "wilcoxon_signed_rank_p_two_sided": gain_wilcoxon,
        "paired_t_test_p_two_sided": gain_ttest,
        "rank_biserial_effect": _rank_biserial(gain),
        "regret_tail_gains": tail_gains,
        "selector_regret": regret_selector,
        "best_fixed_regret": regret_fixed,
        "gnb_zcdp_equal_regret": _metric_summary(per_family.gnb_fixed_regret.to_numpy()),
        "best_fixed_by_epsilon_regret": _metric_summary(
            per_family.epsilon_fixed_regret.to_numpy()),
        "oracle_regret": _metric_summary(per_family.oracle_regret.to_numpy()),
        "selector_private_balanced_accuracy": _metric_summary(
            per_family.selector_ba.to_numpy()),
        "best_fixed_private_balanced_accuracy": _metric_summary(
            per_family.best_fixed_ba.to_numpy()),
        "top1_auxiliary": float(per_family.selector_top1.mean()),
        "fixed_top1_auxiliary": float(per_family.fixed_top1.mean()),
        "mcnemar_top1": top1_test,
        "no_worse_median_and_tail": bool(
            regret_selector["median"] <= regret_fixed["median"]
            and regret_selector["q90"] <= regret_fixed["q90"]
            and regret_selector["cvar90"] <= regret_fixed["cvar90"]
        ),
        "best_fixed_gain_vs_gnb_mean": gnb_mean_gain,
        "best_fixed_gain_vs_gnb_ci95_family_bootstrap": [gnb_lower, gnb_upper],
        "best_fixed_vs_gnb_sign_flip_p_two_sided": gnb_p_value,
        "selected_pipeline_frequency": {
            PIPELINE_DISPLAY.get(str(key), str(key)): float(value) for key, value in
            mode_rows.selector_pipeline.value_counts(normalize=True).sort_index().items()
        },
    }
    if has_h27:
        h27_gain = (per_family.h27_fixed_regret - per_family.selector_regret).to_numpy()
        h27_mean, h27_lower, h27_upper = _bootstrap_ci(h27_gain, seed + 7)
        summary.update({
            "h27_fixed_regret": _metric_summary(per_family.h27_fixed_regret.to_numpy()),
            "h27_fixed_private_balanced_accuracy": _metric_summary(
                per_family.h27_fixed_ba.to_numpy()),
            "mean_regret_gain_vs_h27_fixed": h27_mean,
            "gain_vs_h27_ci95_family_bootstrap": [h27_lower, h27_upper],
            "paired_sign_flip_vs_h27_p_two_sided": _sign_flip_p(h27_gain, seed + 8),
            "h27_no_worse_median_and_tail": bool(
                regret_selector["median"] <= _metric_summary(
                    per_family.h27_fixed_regret.to_numpy())["median"]
                and regret_selector["q90"] <= _metric_summary(
                    per_family.h27_fixed_regret.to_numpy())["q90"]
                and regret_selector["cvar90"] <= _metric_summary(
                    per_family.h27_fixed_regret.to_numpy())["cvar90"]
            ),
        })
    return summary


def evaluate_grouped(results: pd.DataFrame, manifest: pd.DataFrame, *,
                     n_splits: int | None = 5, seed: int = DEFAULT_SEED,
                     selector_strategy: str = "absolute_shared",
                     feature_modes_override: tuple[str, ...] | None = None,
                     progress: bool = False,
                     selector_sink: dict | None = None,
                     fold_subset: set[int] | None = None,
                     ) -> tuple[dict | None, pd.DataFrame]:
    """Grouped out-of-fold evaluation. ``selector_sink``, when given, receives
    ``{fold: {"selectors": ..., "test_ids": ...}}`` for post-hoc analyses.

    ``n_splits=None`` runs leave-one-family-out (LODO: each held-out dataset's
    whole family is excluded from training). ``fold_subset`` runs only those
    folds and returns ``(None, predictions)``; merge the shards with
    ``summarize_grouped``."""
    if selector_strategy not in SELECTOR_STRATEGIES:
        raise ValueError(
            f"estratégia de seletor desconhecida: {selector_strategy}; "
            f"use uma de {SELECTOR_STRATEGIES}"
        )
    aggregated = _aggregate(results, manifest)
    manifest_one = manifest.drop_duplicates("dataset_id").copy()
    dataset_families = aggregated[["dataset_id", "family_id"]].drop_duplicates()
    family_count = dataset_families.family_id.nunique()
    lodo = n_splits is None
    if lodo:
        n_splits = family_count
    if family_count < n_splits or family_count < 5:
        raise ValueError(f"avaliação agrupada requer >= {n_splits} famílias; há {family_count}")
    has_private_features = all(column in aggregated.columns for column in PRIVATE_META_COLUMNS)
    private_by_dataset = {
        int(dataset_id): group.iloc[0][list(PRIVATE_META_COLUMNS)].to_numpy(dtype=float)
        for dataset_id, group in aggregated.groupby("dataset_id", sort=False)
    } if has_private_features else {}
    dp_by_dataset = {
        int(dataset_id): group.iloc[0][list(DP_META_COLUMNS)].to_numpy(dtype=float)
        for dataset_id, group in aggregated.groupby("dataset_id", sort=False)
    } if all(column in aggregated.columns for column in DP_META_COLUMNS) else {}
    feature_modes = tuple(
        feature_modes_override if feature_modes_override is not None
        else _available_feature_modes(aggregated)
    )
    unknown_modes = set(feature_modes) - set(FEATURE_MODES)
    if unknown_modes:
        raise ValueError(f"modos de features desconhecidos: {sorted(unknown_modes)}")
    missing_modes = set(feature_modes) - set(_available_feature_modes(aggregated))
    if missing_modes:
        raise ValueError(f"modos de features ausentes: {sorted(missing_modes)}")
    if not feature_modes:
        raise ValueError("é necessário selecionar ao menos um modo de features")
    splitter = LeaveOneGroupOut() if lodo else GroupKFold(n_splits=n_splits)
    dataset_ids = dataset_families.dataset_id.to_numpy()
    groups = dataset_families.family_id.to_numpy()
    prediction_rows = []
    feature_cache: dict[tuple, np.ndarray] = {}
    for fold, (train_index, test_index) in enumerate(splitter.split(dataset_ids, groups=groups)):
        if fold_subset is not None and fold not in fold_subset:
            continue
        if progress:
            print(f"selector fold {fold + 1}/{n_splits}: fitting", flush=True)
        train_ids = set(dataset_ids[train_index])
        test_ids = set(dataset_ids[test_index])
        train = aggregated[aggregated.dataset_id.isin(train_ids)]
        test = aggregated[aggregated.dataset_id.isin(test_ids)]
        per_epsilon = selector_strategy.endswith("_per_epsilon")
        selectors = {}
        for index, mode in enumerate(feature_modes):
            stable_mode_index = FEATURE_MODES.index(mode)
            if per_epsilon:
                selectors[mode] = {}
                for epsilon in sorted(train.epsilon.unique()):
                    block, target_column = _selector_training_block(
                        train[train.epsilon == epsilon], selector_strategy,
                    )
                    fit_seed = (seed + fold * 31 + stable_mode_index * 1009
                                + int(float(epsilon) * 1000))
                    selectors[mode][float(epsilon)] = _fit_selector(
                        block, manifest_one, mode, private_by_dataset,
                        fit_seed, dp_by_dataset, target_column, feature_cache,
                    )[0]
            else:
                block, target_column = _selector_training_block(train, selector_strategy)
                selectors[mode] = _fit_selector(
                    block, manifest_one, mode, private_by_dataset,
                    seed + fold * 31 + stable_mode_index, dp_by_dataset, target_column,
                    feature_cache,
                )[0]
        if selector_sink is not None:
            selector_sink[fold] = {"selectors": selectors, "test_ids": test_ids}
        fixed = _best_fixed(train)
        fixed_by_epsilon = {
            float(epsilon): _best_fixed(train, float(epsilon))
            for epsilon in sorted(test.epsilon.unique())
        }
        test_conditions = list(test.groupby(
            ["family_id", "dataset_id", "epsilon"], sort=True,
        ))
        manifest_indexed = manifest_one.set_index("dataset_id")
        available_by_condition = []
        for _, condition in test_conditions:
            present = set(condition.pipeline)
            available = tuple(pipeline for pipeline in PIPELINES if pipeline in present)
            if not available:
                raise ValueError("condição sem pipelines conhecidos")
            available_by_condition.append(available)
        predicted_by_mode: dict[str, list[np.ndarray]] = {}
        for mode in feature_modes:
            predicted_blocks: list[np.ndarray | None] = [None] * len(test_conditions)
            for epsilon in sorted(test.epsilon.unique()):
                indices = [index for index, (key, _) in enumerate(test_conditions)
                           if float(key[2]) == float(epsilon)]
                matrices = []
                for index in indices:
                    (_, dataset_id, _), _ = test_conditions[index]
                    meta_row = manifest_indexed.loc[int(dataset_id)]
                    matrices.append(np.vstack([
                        _cached_candidate_features(
                            meta_row, float(epsilon), pipeline, mode,
                            private_by_dataset, dp_by_dataset, feature_cache,
                        )
                        for pipeline in available_by_condition[index]
                    ]))
                model = (selectors[mode][float(epsilon)] if per_epsilon
                         else selectors[mode])
                flat_predictions = model.predict(np.vstack(matrices))
                offset = 0
                for index, matrix in zip(indices, matrices):
                    width = len(matrix)
                    predicted_blocks[index] = flat_predictions[offset:offset + width]
                    offset += width
            predicted_by_mode[mode] = predicted_blocks
        for index, ((family_id, dataset_id, epsilon), condition) in enumerate(test_conditions):
            available_pipelines = available_by_condition[index]
            fixed_epsilon = fixed_by_epsilon[float(epsilon)]
            actual_best = max(available_pipelines, key=lambda pipe: _condition_score(
                condition, pipe, "private_ba"))
            has_h27 = H27_PIPELINE in available_pipelines
            for mode in feature_modes:
                predicted = predicted_by_mode[mode][index]
                chosen = available_pipelines[int(np.argmax(predicted))]
                prediction_rows.append({
                    "feature_mode": mode,
                    "fold": fold,
                    "family_id": family_id,
                    "dataset_id": int(dataset_id),
                    "epsilon": float(epsilon),
                    "selector_pipeline": chosen,
                    "selector_display": PIPELINE_DISPLAY[chosen],
                    "best_fixed_pipeline": fixed,
                    "best_fixed_display": PIPELINE_DISPLAY[fixed],
                    "epsilon_fixed_pipeline": fixed_epsilon,
                    "h27_fixed_pipeline": H27_PIPELINE if has_h27 else None,
                    "h27_fixed_regret": (
                        _condition_score(condition, H27_PIPELINE, "portfolio_regret")
                        if has_h27 else np.nan
                    ),
                    "h27_fixed_ba": (
                        _condition_score(condition, H27_PIPELINE, "private_ba")
                        if has_h27 else np.nan
                    ),
                    "h27_top1": (float(H27_PIPELINE == actual_best) if has_h27 else np.nan),
                    "selector_regret": _condition_score(condition, chosen, "portfolio_regret"),
                    "best_fixed_regret": _condition_score(condition, fixed, "portfolio_regret"),
                    "gnb_fixed_regret": _condition_score(condition, PIPELINES[0], "portfolio_regret"),
                    "epsilon_fixed_regret": _condition_score(
                        condition, fixed_epsilon, "portfolio_regret"),
                    "oracle_regret": _condition_score(condition, actual_best, "portfolio_regret"),
                    "selector_ba": _condition_score(condition, chosen, "private_ba"),
                    "best_fixed_ba": _condition_score(condition, fixed, "private_ba"),
                    "gnb_fixed_ba": _condition_score(condition, PIPELINES[0], "private_ba"),
                    "selector_top1": float(chosen == actual_best),
                    "fixed_top1": float(fixed == actual_best),
                })
        if progress:
            print(f"selector fold {fold + 1}/{n_splits}: complete", flush=True)
    predictions = pd.DataFrame(prediction_rows)
    if fold_subset is not None:
        return None, predictions
    return summarize_grouped(
        predictions, aggregated, n_splits=n_splits, seed=seed,
        selector_strategy=selector_strategy, feature_modes=feature_modes,
        validation_scheme="LODO" if lodo else "GroupKFold",
    ), predictions


def summarize_grouped(predictions: pd.DataFrame, aggregated: pd.DataFrame, *,
                      n_splits: int, seed: int, selector_strategy: str,
                      feature_modes: tuple[str, ...],
                      validation_scheme: str = "GroupKFold") -> dict:
    """Summary statistics over out-of-fold predictions from ``evaluate_grouped``."""
    family_count = aggregated.family_id.nunique()
    summaries = {
        mode: _summarize_mode(predictions, mode, seed + FEATURE_MODES.index(mode) * 101)
        for mode in feature_modes
    }
    mode_family_regrets = {
        mode: (predictions[predictions.feature_mode == mode]
               .groupby("family_id").selector_regret.mean())
        for mode in feature_modes
    }
    common_families = sorted(set.intersection(*(
        set(values.index) for values in mode_family_regrets.values()
    ))) if mode_family_regrets else []
    if len(common_families) >= 3 and len(mode_family_regrets) >= 3:
        friedman = friedmanchisquare(*[
            mode_family_regrets[mode].reindex(common_families).to_numpy()
            for mode in feature_modes
        ])
        friedman_result = {
            "statistic": float(friedman.statistic),
            "p_two_sided": float(friedman.pvalue),
            "n_families": len(common_families),
            "modes": list(feature_modes),
        }
    else:
        friedman_result = {
            "statistic": None, "p_two_sided": None,
            "n_families": len(common_families), "modes": list(feature_modes),
        }
    wilcoxon_raw = {
        mode: result["wilcoxon_signed_rank_p_two_sided"]
        for mode, result in summaries.items()
    }
    wilcoxon_holm = _holm_adjust(wilcoxon_raw)
    mean_superior_modes = [
        mode for mode, result in summaries.items()
        if result["gain_ci95_family_bootstrap"][0] > 0
        and (result["paired_sign_flip_p_two_sided"] < 0.05
             or result["wilcoxon_signed_rank_p_two_sided"] < 0.05)
    ]
    strict_superior_modes = [
        mode for mode, result in summaries.items()
        if mode in mean_superior_modes and result["no_worse_median_and_tail"]
    ]
    best_mode = max(
        summaries, key=lambda mode: summaries[mode]["mean_regret_gain_vs_best_fixed"]
    )
    deployable_modes = tuple(mode for mode in feature_modes
                             if mode in DEPLOYABLE_FEATURE_MODES)
    best_deployable_mode = max(
        deployable_modes,
        key=lambda mode: summaries[mode]["mean_regret_gain_vs_best_fixed"],
    ) if deployable_modes else None
    family_pipeline = (aggregated.groupby(["family_id", "pipeline"], as_index=False)
                       .agg(private_ba=("private_ba", "mean"),
                            clean_ba=("clean_ba", "mean"),
                            portfolio_regret=("portfolio_regret", "mean"),
                            pipeline_privacy_penalty=("pipeline_privacy_penalty", "mean")))
    pipeline_summaries = (family_pipeline.groupby("pipeline").mean(numeric_only=True)
                          .to_dict(orient="index"))
    reference_mode = next(
        (mode for mode in ("public_plus_dp", "public_schema_only") if mode in summaries),
        next(iter(summaries)),
    )
    best_fixed_pipeline_by_fold = (predictions[predictions.feature_mode == reference_mode]
                                   .groupby("fold").best_fixed_pipeline.first().to_dict())
    primary_mode = next(
        (mode for mode in (*PRIMARY_FEATURE_MODES, "public_plus_dp", "public_schema_only")
         if mode in summaries),
        next(iter(summaries)),
    )
    primary = summaries[primary_mode]
    primary_success = bool(
        primary["gain_ci95_family_bootstrap"][0] > 0
        and primary["paired_sign_flip_p_two_sided"] < 0.05
        and primary["no_worse_median_and_tail"]
    )
    summary = {
        "n_datasets": int(aggregated.dataset_id.nunique()),
        "n_families": int(family_count),
        "n_outer_folds": int(n_splits),
        "validation_scheme": validation_scheme,
        "pipelines": {key: value for key, value in PIPELINE_DISPLAY.items()},
        "selector_strategy": selector_strategy,
        "primary_feature_mode": primary_mode,
        "primary_metadata_epsilon_used": 0.0,
        "public_only": bool(feature_modes == PUBLIC_ONLY_FEATURE_MODES),
        "metadata_epsilon_used": 0.0 if feature_modes == PUBLIC_ONLY_FEATURE_MODES else None,
        "primary_comparison": (
            f"{primary_mode} selector vs fold-specific best fixed option and fixed H27 baseline"
        ),
        "primary_success": primary_success,
        "primary_result": primary,
        "feature_modes": summaries,
        "statistical_tests": {
            "friedman_across_feature_modes": friedman_result,
            "wilcoxon_holm_adjusted_by_feature_mode": wilcoxon_holm,
            "wilcoxon_raw_by_feature_mode": wilcoxon_raw,
        },
        "expanded_validation": {
            "mean_superior_modes": mean_superior_modes,
            "strict_superior_modes": strict_superior_modes,
            "best_mode_by_mean_regret_gain": best_mode,
            "best_deployable_mode_by_mean_regret_gain": best_deployable_mode,
            "strict_criterion": (
                "IC95% do ganho > 0, sign-flip ou Wilcoxon p < 0,05 e "
                "mediana/q90/CVaR90 sem piora"
            ),
        },
        "equal_family_pipeline_summary": pipeline_summaries,
        "evaluated_option_ids": [pipeline for pipeline in PIPELINES
                                 if pipeline in set(aggregated.pipeline)],
        "n_evaluated_options": int(aggregated.pipeline.nunique()),
        "best_fixed_pipeline_by_outer_fold": {
            str(key): str(value) for key, value in best_fixed_pipeline_by_fold.items()
        },
        "interpretation": (
            "evidência exploratória de ganho fora da amostra" if primary_success else
            "não demonstrou ganho do seletor implantável sobre a melhor opção fixa"
        ),
        "research_privileged_caveat": (
            "research_privileged e research_rich usam descritores calculados nos dados de treino, "
            "sem DP; são somente tetos diagnósticos e não oferecem garantia formal de privacidade. "
            "dp_research_rich usa um release Laplace separado, com epsilon registrado em "
            "dp_meta_epsilon, e precisa ser composto com o orçamento do modelo."
        ),
    }
    return summary


def _implementation_hash() -> str:
    return numerical_implementation_sha256()


def _write_report(summary: dict, conditions: pd.DataFrame, manifest: pd.DataFrame) -> str:
    primary = summary["primary_result"]
    lines = [
        "# Portfolio de mecanismos DP e modelos: validação de viabilidade",
        "",
        f"**Conclusão:** {summary['interpretation']}",
        "",
        f"Transformação numérica: `{summary['numeric_transform']}`.",
        "",
        f"Estratégia do seletor: `{summary.get('selector_strategy', 'absolute_shared')}`.",
        "",
        (
            "Metafeatures: somente schema público; orçamento DP adicional de metafeatures: ε=0."
            if summary.get("public_only") else
            "Metafeatures: modos públicos/DP disponíveis conforme o manifesto; qualquer release "
            "DP de metafeatures é registrado separadamente."
        ),
        "",
        f"{summary['n_datasets']} datasets OpenML, {summary['n_families']} famílias, "
        f"validação cruzada {summary.get('validation_scheme', 'GroupKFold')} "
        f"com {summary['n_outer_folds']} folds.",
        "",
        f"Regret = melhor balanced accuracy entre as "
        f"{summary.get('n_evaluated_options', len(PIPELINES))} opções avaliadas de mecanismo e "
        "modelo ajustadas limpas no split, menos a "
        "balanced accuracy privada. Valores negativos são possíveis. Cada comparação de "
        "opção fixa é selecionada somente nos grupos de treino do fold.",
        "",
        "## Comparação primária",
        "",
        f"Seletor `{summary.get('primary_feature_mode', 'public_plus_dp')}` contra melhor "
        "opção fixa e baseline H27. `public_plus_dp` não usa ε adicional de metadados; "
        "as estimativas de regret e IC são agregadas por família:",
        "",
        f"- ganho médio de regret: {primary['mean_regret_gain_vs_best_fixed']:.4f}; "
        f"IC bootstrap 95% [{primary['gain_ci95_family_bootstrap'][0]:.4f}, "
        f"{primary['gain_ci95_family_bootstrap'][1]:.4f}]",
        f"- teste pareado sign-flip: p = {primary['paired_sign_flip_p_two_sided']:.4f}",
        f"- Wilcoxon pareado: p = {primary['wilcoxon_signed_rank_p_two_sided']:.4f}; "
        f"tamanho de efeito rank-biserial = {primary['rank_biserial_effect']:.3f}",
        f"- teste t pareado: p = {primary['paired_t_test_p_two_sided']:.4f}",
        f"- McNemar exato no Top-1: p = "
        f"{primary['mcnemar_top1']['exact_binomial_p_two_sided']:.4f}",
        f"- regret mediano: seletor {primary['selector_regret']['median']:.4f}, "
        f"fixo {primary['best_fixed_regret']['median']:.4f}",
        f"- regret q90 / CVaR90: seletor {primary['selector_regret']['q90']:.4f} / "
        f"{primary['selector_regret']['cvar90']:.4f}; fixo "
        f"{primary['best_fixed_regret']['q90']:.4f} / "
        f"{primary['best_fixed_regret']['cvar90']:.4f}",
        f"- balanced accuracy mediana: seletor "
        f"{primary['selector_private_balanced_accuracy']['median']:.4f}, melhor fixo "
        f"{primary['best_fixed_private_balanced_accuracy']['median']:.4f}",
        f"- melhor opção fixa vs Gaussiano (zCDP) · GaussianNB: "
        f"ganho médio {primary['best_fixed_gain_vs_gnb_mean']:.4f}; "
        f"IC95% [{primary['best_fixed_gain_vs_gnb_ci95_family_bootstrap'][0]:.4f}, "
        f"{primary['best_fixed_gain_vs_gnb_ci95_family_bootstrap'][1]:.4f}], "
        f"p = {primary['best_fixed_vs_gnb_sign_flip_p_two_sided']:.4f}",
        f"- top-1 auxiliar: {primary['top1_auxiliary']:.3f}",
        f"- ganhos de cauda (fixo - seletor), mediana/q90/CVaR90: "
        f"{primary['regret_tail_gains']['median']['gain']:.4f} / "
        f"{primary['regret_tail_gains']['q90']['gain']:.4f} / "
        f"{primary['regret_tail_gains']['cvar90']['gain']:.4f}",
        "",
        "## Validação expandida",
        "",
        f"Modos superiores na média com IC e teste: "
        f"`{', '.join(summary['expanded_validation']['mean_superior_modes']) or 'nenhum'}`.",
        f"Modos que também passam a restrição de cauda: "
        f"`{', '.join(summary['expanded_validation']['strict_superior_modes']) or 'nenhum'}`.",
        f"Melhor modo pela média de ganho: "
        f"`{summary['expanded_validation']['best_mode_by_mean_regret_gain']}`.",
        f"Melhor modo implantável pela média de ganho: "
        f"`{summary['expanded_validation']['best_deployable_mode_by_mean_regret_gain']}`.",
        "",
        "## Outros seletores e baselines",
        "",
        "`dp_research_rich` é uma ablação que libera metafeatures com ε próprio. "
        "Os números nominais desse modo não são uma comparação justa por orçamento total; "
        "a análise justa usa `scripts/scoped_total_epsilon.py`.",
        "",
        "| Seletor | ganho médio vs fixo | IC95% por família | p sign-flip | regret mediano | "
        "q90 | CVaR90 | top-1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mode, result in summary["feature_modes"].items():
        low, high = result["gain_ci95_family_bootstrap"]
        lines.append(
            f"| {mode} | {result['mean_regret_gain_vs_best_fixed']:.4f} | "
            f"[{low:.4f}, {high:.4f}] | {result['paired_sign_flip_p_two_sided']:.4f} | "
            f"{result['selector_regret']['median']:.4f} | {result['selector_regret']['q90']:.4f} | "
            f"{result['selector_regret']['cvar90']:.4f} | {result['top1_auxiliary']:.3f} |"
        )
    lines.extend([
        "",
        "## Desempenho agregado das opções",
        "",
        "Médias com peso igual por família, dataset e epsilon; `privacy_penalty` compara cada "
        "opção ao próprio ajuste limpo, enquanto `portfolio_regret` usa a referência limpa "
        "comum do melhor candidato.",
        "",
        "| Mecanismo · modelo | BA privada | BA limpa | regret do portfólio | penalidade própria |",
        "|---|---:|---:|---:|---:|",
    ])
    if "mean_regret_gain_vs_h27_fixed" in primary:
        h27 = primary["h27_fixed_regret"]
        low, high = primary["gain_vs_h27_ci95_family_bootstrap"]
        lines.extend([
            "",
            "## Baseline fixo H27",
            "",
            f"H27 (`{PIPELINE_DISPLAY[H27_PIPELINE]}`) é comparado como opção fixa, "
            "sem selecionar o baseline pelo conjunto de teste.",
            f"- regret médio/mediano/q90/CVaR90: {h27['mean']:.4f} / "
            f"{h27['median']:.4f} / {h27['q90']:.4f} / {h27['cvar90']:.4f}",
            f"- ganho médio do seletor sobre H27 (H27 − seletor): "
            f"{primary['mean_regret_gain_vs_h27_fixed']:.4f}; IC95% por família "
            f"[{low:.4f}, {high:.4f}], sign-flip p = "
            f"{primary['paired_sign_flip_vs_h27_p_two_sided']:.4f}",
            f"- gate de cauda contra H27: "
            f"{'passa' if primary['h27_no_worse_median_and_tail'] else 'não passa'}",
        ])
    for pipeline, values in summary["equal_family_pipeline_summary"].items():
        lines.append(
            f"| {PIPELINE_DISPLAY[pipeline]} | {values['private_ba']:.4f} | "
            f"{values['clean_ba']:.4f} | {values['portfolio_regret']:.4f} | "
            f"{values['pipeline_privacy_penalty']:.4f} |"
        )
    lines.extend([
        "",
        "Na coluna `gnb_zcdp_equal_regret` do summary está o baseline fixo "
        "Gaussiano (zCDP) · GaussianNB, "
        "para comparação direta com o mecanismo fixo anteriormente selecionado. `oracle_regret` "
        "é apenas teto retrospectivo, nunca disponível ao seletor.",
        "",
        "## Escopo e limites",
        "",
        f"Isto é um piloto exploratório, sem ajuste baseado no teste; {len(summary['epsilons'])} "
        "epsilons e "
        f"{summary['repeats']} "
        "repetições por dataset. A validação usa benchmarks públicos OpenML e não prova "
        "privacidade em dados reais. O seletor implantável usa schema público, epsilon e "
        "identidade da opção de mecanismo e modelo; quando habilitada, a calibração é derivada de sensibilidades "
        "públicas. O modo "
        "`research_privileged` e `research_rich` usam dados de treino não protegidos e não são "
        "implantáveis sob a garantia DP reportada.",
        "",
        "As sementes fixas tornam este benchmark reproduzível em OpenML público; elas não "
        "devem ser usadas para liberar modelos de dados sensíveis. As APIs de ruído aceitam "
        "`seed=None` para obter entropia do sistema operacional, mas a amostragem finita ainda "
        "precisa de auditoria antes de produção.",
        "",
        "Os histogramas de folhas usam árvores e partições aleatórias independentes dos valores; "
        "cada registro pertence a um único shard, com histograma conjunto de sensibilidade L1 2 "
        "e composição paralela. A logística usa perturbação gaussiana dos coeficientes, com "
        "sensibilidade analítica baseada em forte convexidade e em uma cota conservadora de erro "
        "para número fixo de iterações do otimizador, sem critério de parada dependente dos dados. "
        "A garantia matemática pressupõe aritmética exata; este protótipo não é uma auditoria de "
        "precisão finita, canais laterais ou implementação de produção.",
        "",
        "O Top-1 compara a proporção de epsilons em que seletor e baseline acertam "
        "a melhor opção, agregada primeiro por dataset. O McNemar exato usa uma observação "
        "por dataset (maioria de epsilons corretos), nunca uma linha por condição.",
        "",
        "A mediana e as caudas são comparadas entre famílias. Evidência positiva requer IC95% "
        "inteiramente acima de zero, teste pareado bilateral p < 0,05 e nenhum aumento em mediana, "
        "q90 ou CVaR dos 10% piores regrets. Resultados dos outros modos são diagnósticos "
        "exploratórios, sem afirmação confirmatória.",
        "",
    ])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path,
        default=Path("data/openml_manifest_984_completed.csv"),
    )
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--limit", type=int, default=984)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--pipelines", nargs="+", choices=PIPELINE_REGISTRY,
                        help="opções a executar; padrão preserva as 12 opções originais")
    parser.add_argument("--epsilons", type=_parse_epsilon, default=DEFAULT_EPSILONS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--numeric-transform", choices=("tanh", "asinh10", "public_bounds"),
                        default="tanh", help=(
                            "transformação numérica; public_bounds requer limites explícitos "
                            "no schema do manifesto"
                        ))
    parser.add_argument("--dp-meta-epsilon", type=float, default=DP_META_EPSILON_DEFAULT,
                        help="epsilon separado para o release DP de metafeatures train-only")
    parser.add_argument(
        "--selector-strategy", choices=SELECTOR_STRATEGIES,
        default=None,
        help=(
            "alvo e especialização do seletor; absolute_shared treina um único "
            "modelo para toda a grade de epsilon"
        ),
    )
    parser.add_argument("--output", type=Path,
                        default=Path("results/classifier-portfolio-984"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--summary-only", action="store_true",
                        help="calcula o resumo a partir de conditions.csv existente")
    parser.add_argument("--deployable-only", action="store_true",
                        help="resume apenas public_schema_only e public_plus_dp")
    parser.add_argument("--primary-only", action="store_true",
                        help="resume apenas public_plus_dp, o modo principal sem epsilon extra")
    parser.add_argument(
        "--public-only", action="store_true",
        help=(
            "executa somente o seletor com metafeatures do schema público; "
            "não gera nem contabiliza release DP de metafeatures"
        ),
    )
    args = parser.parse_args(argv)
    effective_selector_strategy = args.selector_strategy or "absolute_shared"
    if args.limit < 5 or args.repeats < 1:
        parser.error("--limit precisa ser >= 5 e --repeats >= 1")
    selected_mode_flags = sum(
        bool(value) for value in (args.deployable_only, args.primary_only, args.public_only)
    )
    if selected_mode_flags > 1:
        parser.error("--public-only, --deployable-only e --primary-only são mutuamente exclusivos")
    manifest = _load_manifest(args.manifest).head(args.limit).copy()
    if len(manifest) < args.limit:
        raise ValueError(f"manifest contém {len(manifest)} datasets, limite={args.limit}")
    args.output.mkdir(parents=True, exist_ok=True)
    paths = {
        "conditions": args.output / "conditions.csv",
        "failures": args.output / "failures.csv",
        "summary": args.output / "summary.json",
        "predictions": args.output / "outer_predictions.csv",
        "report": args.output / "report.md",
        "config": args.output / "run_config.json",
    }
    saved_config = {}
    if args.summary_only:
        if not paths["conditions"].exists():
            raise FileNotFoundError(f"conditions.csv não existe em {args.output}")
        # `failures.csv` belongs to the benchmark run, not to summary output.
        # Preserve it when regenerating derived artifacts from saved conditions.
        derived_paths = [paths[key] for key in ("summary", "predictions", "report")]
        if not args.overwrite and any(path.exists() for path in derived_paths):
            raise FileExistsError(
                f"resumo já existe em {args.output}; use --overwrite explicitamente"
            )
        if args.overwrite:
            for path in derived_paths:
                if path.exists():
                    path.unlink()
        if paths["config"].exists():
            saved_config = json.loads(paths["config"].read_text())
    else:
        if not args.overwrite and any(path.exists() for path in paths.values()):
            raise FileExistsError(f"saída já existe em {args.output}; use --overwrite explicitamente")
        if args.overwrite:
            for path in paths.values():
                if path.exists():
                    path.unlink()
    manifest_hash = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    config = {
        "manifest_sha256": manifest_hash,
        "implementation_sha256": _implementation_hash(),
        "runtime": runtime_fingerprint(),
        "limit": args.limit,
        "epsilons": list(args.epsilons),
        "repeats": args.repeats,
        "seed": args.seed,
        "numeric_transform": args.numeric_transform,
        "pipelines": list(args.pipelines or PIPELINES),
        "tree_count": SELECTOR_TREE_COUNT,
        "selector_tree_count": SELECTOR_TREE_COUNT,
        "forest_candidate_tree_count": FOREST_CANDIDATE_TREE_COUNT,
        "forest_small_tree_count": FOREST_SMALL_TREE_COUNT,
        "tree_depth": TREE_DEPTH,
        "dp_sgd_steps": DP_SGD_STEPS,
        "dp_sgd_steps_alternate": DP_SGD_STEPS_ALTERNATE,
        "dp_sgd_gradient_clip": DP_SGD_GRADIENT_CLIP,
        "dp_sgd_regularization": DP_SGD_REGULARIZATION,
        "dp_sgd_clip2_gradient_clip": DP_SGD_CLIP2_GRADIENT_CLIP,
        "dp_sgd_clip2_regularization": DP_SGD_CLIP2_REGULARIZATION,
        "softmax_regularization": SOFTMAX_REGULARIZATION,
        "categorical_nb_numeric_bins": CATEGORICAL_NB_NUMERIC_BINS,
        "softmax_optimizer_tolerance": SOFTMAX_OPTIMIZER_TOLERANCE,
        "dp_meta_epsilon": (args.dp_meta_epsilon if not
                            (args.public_only or args.primary_only or args.deployable_only)
                            else 0.0),
        "dp_meta_epsilon_requested": args.dp_meta_epsilon,
        "metadata_feature_mode": (
            "public_schema_only" if args.public_only else
            "public_plus_dp" if args.primary_only else
            "deployable_only" if args.deployable_only else "all_available"
        ),
        "metadata_epsilon_used": (args.dp_meta_epsilon if not
                                  (args.public_only or args.primary_only or args.deployable_only)
                                  else 0.0),
        "selector_strategy": effective_selector_strategy,
    }
    if args.summary_only and saved_config:
        config = saved_config
        if config.get("manifest_sha256") != manifest_hash:
            raise ValueError("manifesto atual difere do manifesto registrado no run")
    else:
        paths["config"].write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
        save_source_snapshot(args.output)
    failures = []
    if args.summary_only:
        if paths["failures"].exists():
            failures = pd.read_csv(paths["failures"]).to_dict("records")
    else:
        openml = _openml(args.cache_dir)
        completed_ids = set()
        for _, row in manifest.iterrows():
            try:
                rows = run_dataset(openml, row, args.epsilons, args.repeats, args.seed,
                                   include_research_features=not (
                                       args.public_only or args.primary_only or args.deployable_only),
                                   numeric_transform=args.numeric_transform,
                                   dp_meta_epsilon=args.dp_meta_epsilon,
                                   pipelines=tuple(args.pipelines) if args.pipelines else None)
                pd.DataFrame(rows).to_csv(
                    paths["conditions"], mode="a", header=not paths["conditions"].exists(),
                    index=False,
                )
                completed_ids.add(int(row.dataset_id))
                print(f"finished {int(row.dataset_id)} {row.dataset_name}: {len(rows)} rows", flush=True)
            except Exception as error:
                failure = {"dataset_id": int(row.dataset_id), "dataset_name": row.dataset_name,
                           "error": f"{type(error).__name__}: {error}"}
                failures.append(failure)
                print(f"failed {row.dataset_name}: {failure['error']}", file=sys.stderr, flush=True)
            pd.DataFrame(failures, columns=["dataset_id", "dataset_name", "error"]).to_csv(
                paths["failures"], index=False,
            )
    if not paths["conditions"].exists():
        raise RuntimeError("nenhum dataset foi avaliado; veja failures.csv")
    conditions = pd.read_csv(paths["conditions"])
    run_epsilons = tuple(config.get("epsilons", args.epsilons))
    run_repeats = int(config.get("repeats", args.repeats))
    validate_condition_grid(
        conditions, manifest, epsilons=run_epsilons, repeats=run_repeats,
        pipelines=tuple(config.get("pipelines", PIPELINES)),
    )
    if args.pipelines:
        print(f"selected pipelines completed: {len(conditions)} conditions", flush=True)
        return 0
    used_manifest = manifest[manifest.dataset_id.astype(int).isin(
        conditions.dataset_id.astype(int).unique())].copy()
    if used_manifest.family_id.nunique() < 5:
        raise RuntimeError("menos de cinco famílias completaram")
    run_seed = int(config.get("seed", args.seed))
    run_numeric_transform = config.get("numeric_transform", args.numeric_transform)
    summary, predictions = evaluate_grouped(
        conditions, used_manifest, seed=run_seed,
        selector_strategy=(args.selector_strategy or
                           config.get("selector_strategy", effective_selector_strategy)),
        feature_modes_override=(PUBLIC_ONLY_FEATURE_MODES if args.public_only else
                                PRIMARY_FEATURE_MODES if args.primary_only else
                                DEPLOYABLE_FEATURE_MODES if args.deployable_only else None),
    )
    summary.update({
        "seed": run_seed,
        "numeric_transform": run_numeric_transform,
        "epsilons": run_epsilons,
        "repeats": run_repeats,
        "manifest": str(args.manifest),
        "manifest_sha256": manifest_hash,
        "implementation_sha256": config["implementation_sha256"],
        "analysis_implementation_sha256": numerical_implementation_sha256(),
        "run_source_matches_analysis": (
            config.get("implementation_sha256") == numerical_implementation_sha256()
        ),
        "attempted_datasets": int(len(manifest)),
        "completed_datasets": int(conditions.dataset_id.nunique()),
        "failed_datasets": failures,
        "delta_gaussian": DELTA_TOTAL,
        "exploratory_pilot": True,
        "dp_meta_epsilon": float(config.get("dp_meta_epsilon", args.dp_meta_epsilon)),
        "dp_meta_epsilon_requested": float(config.get(
            "dp_meta_epsilon_requested", args.dp_meta_epsilon)),
        "metadata_feature_mode": config.get(
            "metadata_feature_mode",
            "public_schema_only" if args.public_only else "all_available",
        ),
        "conditions_metadata_epsilon_used": float(config.get(
            "metadata_epsilon_used", args.dp_meta_epsilon)),
        "metadata_epsilon_used": (
            0.0 if (args.public_only or args.primary_only or args.deployable_only)
            else float(config.get("metadata_epsilon_used", args.dp_meta_epsilon))
        ),
    })
    paths["summary"].write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    predictions.to_csv(paths["predictions"], index=False)
    paths["report"].write_text(_write_report(summary, conditions, used_manifest))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

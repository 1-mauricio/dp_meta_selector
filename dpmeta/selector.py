"""Grouped evaluation of public, DP-aware, and privileged selector features."""

from __future__ import annotations

from math import ceil

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold

from .meta_features import (
    DP_CALIBRATION_FEATURE_NAMES,
    PRIVATE_META_COLUMNS,
    dp_calibration_features,
)
from .l2_metrics import relative_l2_regrets
from .model import ALLOCATIONS
from .schema import PublicSchema, public_features

MECHANISMS = ("laplace", "staircase", "analytic_gaussian", "gaussian_zcdp")
ACTIONS = tuple((mechanism, allocation) for mechanism in MECHANISMS for allocation in ALLOCATIONS)
ACTION_NAMES = tuple(f"{mechanism}__{allocation}" for mechanism, allocation in ACTIONS)
FEATURE_MODES = ("public", "dp_aware", "private_research_only")
ROPE = 0.01


def _action_features(mechanism: str, allocation: str) -> list[float]:
    return [float(mechanism == item) for item in MECHANISMS] + [
        float(allocation == item) for item in ALLOCATIONS
    ]


def _dataset_public_features(manifest_row: pd.Series, epsilon: float) -> np.ndarray:
    schema = PublicSchema.from_manifest_json(manifest_row["public_schema_json"])
    return public_features(int(manifest_row["n_instances"]), schema, float(epsilon))


def aggregate_repetitions(results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate repeat outcomes and derive condition-relative statistic regret."""
    required = {"family_id", "dataset_id", "epsilon", "mechanism", "allocation", "regret"}
    missing = required - set(results.columns)
    if missing:
        raise ValueError(f"colunas ausentes: {sorted(missing)}")
    private_columns = [column for column in results if column.startswith("private_meta__")]
    aggregation = {
        "regret": ("regret", "mean"),
        "clean_ba": ("clean_ba", "mean"),
        "private_ba": ("private_ba", "mean"),
    }
    if "l2_distortion" in results:
        aggregation["l2_distortion"] = ("l2_distortion", "mean")
    # Inputs are emitted in repetition order; first is the deployment-like
    # single-split descriptor, not an average over multiple hypothetical splits.
    aggregation.update({column: (column, "first") for column in private_columns})
    aggregated = (results.groupby(["family_id", "dataset_id", "epsilon", "mechanism", "allocation"],
                                  as_index=False, sort=True)
                  .agg(**aggregation))
    if "l2_distortion" in aggregated:
        aggregated["l2_regret"] = np.nan
        for _, indices in aggregated.groupby(["dataset_id", "epsilon"], sort=False).groups.items():
            indices = np.asarray(list(indices), dtype=int)
            aggregated.loc[indices, "l2_regret"] = relative_l2_regrets(
                aggregated.loc[indices, "l2_distortion"].to_numpy(dtype=float)
            )
    return aggregated


def _candidate_features(
    manifest_row: pd.Series, epsilon: float, action: tuple[str, str], mode: str,
    private_values: np.ndarray | None = None,
) -> np.ndarray:
    mechanism, allocation = action
    schema = PublicSchema.from_manifest_json(manifest_row["public_schema_json"])
    n = int(manifest_row["n_instances"])
    base = public_features(n, schema, float(epsilon))
    action_features = np.asarray(_action_features(mechanism, allocation), dtype=float)
    pieces = [base, action_features]
    if mode in {"dp_aware", "private_research_only"}:
        pieces.append(dp_calibration_features(n, schema, epsilon, mechanism, allocation))
    if mode == "private_research_only":
        if private_values is None:
            raise ValueError("features privadas de treino ausentes no modo de pesquisa")
        pieces.append(np.asarray(private_values, dtype=float))
    if mode not in FEATURE_MODES:
        raise ValueError(f"modo de features desconhecido: {mode}")
    return np.concatenate(pieces)


def _fit_predictor(train: pd.DataFrame, manifest: pd.DataFrame, mode: str,
                   private_by_dataset: dict[int, np.ndarray], seed: int,
                   target_column: str = "regret"):
    meta = manifest.drop_duplicates("dataset_id").set_index("dataset_id")
    x_values, y_values = [], []
    for row in train.itertuples(index=False):
        x_values.append(_candidate_features(
            meta.loc[row.dataset_id], row.epsilon, (row.mechanism, row.allocation), mode,
            private_by_dataset.get(int(row.dataset_id)),
        ))
        y_values.append(float(getattr(row, target_column)))
    model = RandomForestRegressor(
        n_estimators=1000, min_samples_leaf=3, max_features=0.9,
        random_state=seed, n_jobs=1,
    )
    model.fit(np.vstack(x_values), np.asarray(y_values))
    return model, meta


def _predict_action(model, manifest_row: pd.Series, epsilon: float, mode: str,
                    private_values: np.ndarray | None = None) -> tuple[str, str]:
    x = np.vstack([_candidate_features(manifest_row, epsilon, action, mode, private_values)
                   for action in ACTIONS])
    predicted = model.predict(x)
    return ACTIONS[int(np.argmin(predicted))]


def _best_fixed(train: pd.DataFrame, target_column: str = "regret") -> tuple[str, str]:
    means = train.groupby(["mechanism", "allocation"], sort=True)[target_column].mean()
    return min(ACTIONS, key=lambda action: float(means.loc[action]))


def _best_by_epsilon(train: pd.DataFrame, epsilon: float,
                     target_column: str = "regret") -> tuple[str, str]:
    at_epsilon = train[train.epsilon == epsilon]
    means = at_epsilon.groupby(["mechanism", "allocation"], sort=True)[target_column].mean()
    return min(ACTIONS, key=lambda action: float(means.loc[action]))


def _regret_for(condition: pd.DataFrame, action: tuple[str, str],
                target_column: str = "regret") -> float:
    row = condition[(condition.mechanism == action[0]) & (condition.allocation == action[1])]
    if len(row) != 1:
        raise ValueError("cada condição precisa de exatamente um regret agregado por ação")
    return float(row.iloc[0][target_column])


def _bootstrap_mean_ci(values: np.ndarray, seed: int, draws: int = 10_000) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    means = np.empty(draws)
    for index in range(draws):
        means[index] = rng.choice(values, size=len(values), replace=True).mean()
    return float(values.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _sign_flip_pvalue(values: np.ndarray, seed: int, draws: int = 20_000) -> float:
    """Two-sided paired randomization test over independent family-level gains."""
    rng = np.random.default_rng(seed)
    observed = abs(float(values.mean()))
    signs = rng.choice(np.array([-1.0, 1.0]), size=(draws, len(values)))
    randomized = np.abs((signs * values).mean(axis=1))
    return float((1 + np.count_nonzero(randomized >= observed)) / (draws + 1))


def _distribution_summary(values: np.ndarray) -> dict:
    tail_n = max(1, ceil(0.10 * len(values)))
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "q10": float(np.quantile(values, 0.10)),
        "q90": float(np.quantile(values, 0.90)),
        "cvar10": float(np.mean(np.sort(values)[-tail_n:])),
    }


def _summarize_predictions(predictions: pd.DataFrame, *, seed: int) -> dict:
    per_dataset = (predictions.groupby(["family_id", "dataset_id"], as_index=False)
                   .agg({"selector_regret": "mean", "fixed_regret": "mean",
                         "epsilon_fixed_regret": "mean", "default_regret": "mean",
                         "oracle_regret": "mean", "top1": "mean"}))
    per_family = (per_dataset.groupby("family_id", as_index=False)
                  .agg({"selector_regret": "mean", "fixed_regret": "mean",
                        "epsilon_fixed_regret": "mean", "default_regret": "mean",
                        "oracle_regret": "mean", "top1": "mean"}))
    gain = (per_family.fixed_regret - per_family.selector_regret).to_numpy()
    mean_gain, ci_low, ci_high = _bootstrap_mean_ci(gain, seed)
    wins = int(np.count_nonzero(gain > ROPE))
    losses = int(np.count_nonzero(gain < -ROPE))
    selector_values = per_family.selector_regret.to_numpy()
    fixed_values = per_family.fixed_regret.to_numpy()
    return {
        "primary_comparison": "selector vs fold-specific best fixed action",
        "mean_gain": mean_gain,
        "gain_ci95_family_bootstrap": [ci_low, ci_high],
        "sign_flip_p_two_sided": _sign_flip_pvalue(gain, seed + 1),
        "wins_ties_losses_rope_0_01": {
            "wins": wins, "ties": int(len(gain) - wins - losses), "losses": losses,
        },
        "selector": _distribution_summary(selector_values),
        "best_fixed": _distribution_summary(fixed_values),
        "best_fixed_by_epsilon": _distribution_summary(per_family.epsilon_fixed_regret.to_numpy()),
        "preselected_laplace_equal": _distribution_summary(per_family.default_regret.to_numpy()),
        "oracle_reference": _distribution_summary(per_family.oracle_regret.to_numpy()),
        "top1_auxiliary": float(per_family.top1.mean()),
        "success": bool(ci_low > 0 and np.median(selector_values) <= np.median(fixed_values)
                         and np.quantile(selector_values, 0.90)
                         <= np.quantile(fixed_values, 0.90) + ROPE),
        "interpretation": (
            "selector meets preregistered utility criteria" if ci_low > 0 else
            "no out-of-sample mean improvement demonstrated against best fixed action"
        ),
    }


def _summarize_l2_classifier_utility(predictions: pd.DataFrame, *, seed: int) -> dict:
    """Score classifier impact after selecting actions against the L2 target."""
    per_dataset = (predictions.groupby(["family_id", "dataset_id"], as_index=False)
                   .agg({"classifier_regret_selected_l2": "mean",
                         "classifier_regret_best_utility_fixed": "mean"}))
    per_family = (per_dataset.groupby("family_id", as_index=False)
                  .agg({"classifier_regret_selected_l2": "mean",
                        "classifier_regret_best_utility_fixed": "mean"}))
    selected = per_family.classifier_regret_selected_l2.to_numpy(dtype=float)
    fixed = per_family.classifier_regret_best_utility_fixed.to_numpy(dtype=float)
    gain = fixed - selected
    mean_gain, ci_low, ci_high = _bootstrap_mean_ci(gain, seed)
    return {
        "comparison": "classifier regret of L2-selected action vs training-selected fixed utility action",
        "mean_gain": mean_gain,
        "gain_ci95_family_bootstrap": [ci_low, ci_high],
        "sign_flip_p_two_sided": _sign_flip_pvalue(gain, seed + 1),
        "l2_selected_classifier_regret": _distribution_summary(selected),
        "best_utility_fixed_classifier_regret": _distribution_summary(fixed),
        "success": bool(ci_low > 0 and np.median(selected) <= np.median(fixed)
                         and np.quantile(selected, 0.90) <= np.quantile(fixed, 0.90)),
    }


def evaluate_grouped(results: pd.DataFrame, manifest: pd.DataFrame, *,
                     n_splits: int = 5, seed: int = 20260917,
                     target_column: str = "regret") -> tuple[dict, pd.DataFrame]:
    """Outer grouped CV; target is classifier regret or relative sufficient-stat L2 regret."""
    aggregated = aggregate_repetitions(results)
    if target_column not in aggregated:
        raise ValueError(f"alvo meta-learning ausente: {target_column}")
    private_columns = [column for column in PRIVATE_META_COLUMNS if column in aggregated]
    if private_columns and len(private_columns) != len(PRIVATE_META_COLUMNS):
        raise ValueError("vetor de meta-features privadas está incompleto")
    private_available = len(private_columns) == len(PRIVATE_META_COLUMNS)
    modes = ["public", "dp_aware"] + (["private_research_only"] if private_available else [])

    manifest_one = manifest.drop_duplicates("dataset_id").copy()
    family_map = manifest_one.set_index("dataset_id").family_id.to_dict()
    aggregated["family_id"] = aggregated.dataset_id.map(family_map)
    private_by_dataset: dict[int, np.ndarray] = {}
    if private_available:
        descriptor_rows = aggregated.drop_duplicates("dataset_id").set_index("dataset_id")
        private_by_dataset = {
            int(dataset_id): descriptor_rows.loc[dataset_id, private_columns].to_numpy(dtype=float)
            for dataset_id in descriptor_rows.index
        }

    dataset_families = (aggregated[["dataset_id", "family_id"]].drop_duplicates()
                        .sort_values("dataset_id", kind="stable"))
    dataset_ids = dataset_families.dataset_id.to_numpy()
    groups = dataset_families.family_id.to_numpy()
    unique_groups = np.unique(groups)
    if len(unique_groups) < n_splits:
        raise ValueError(f"avaliação requer {n_splits} famílias, encontradas {len(unique_groups)}")
    splitter = GroupKFold(n_splits=n_splits)
    yield_rows: list[dict] = []
    for fold, (train_ds_idx, test_ds_idx) in enumerate(splitter.split(dataset_ids, groups=groups)):
        train_ids, test_ids = set(dataset_ids[train_ds_idx]), set(dataset_ids[test_ds_idx])
        train = aggregated[aggregated.dataset_id.isin(train_ids)]
        test = aggregated[aggregated.dataset_id.isin(test_ids)]
        models = {
            mode: _fit_predictor(
                train, manifest_one, mode, private_by_dataset, seed + fold * 11 + idx,
                target_column=target_column,
            )[0]
            for idx, mode in enumerate(modes)
        }
        fixed = _best_fixed(train, target_column)
        utility_fixed = _best_fixed(train, "regret") if target_column == "l2_regret" else None
        for (family_id, dataset_id, epsilon), condition in test.groupby(
                ["family_id", "dataset_id", "epsilon"], sort=True):
            row = manifest_one[manifest_one.dataset_id == dataset_id].iloc[0]
            private_values = private_by_dataset.get(int(dataset_id))
            fixed_epsilon = _best_by_epsilon(train, float(epsilon), target_column)
            fixed_regret = _regret_for(condition, fixed, target_column)
            epsilon_fixed_regret = _regret_for(condition, fixed_epsilon, target_column)
            default_regret = _regret_for(condition, ("laplace", "equal"), target_column)
            actual_best = min(ACTIONS, key=lambda action: _regret_for(condition, action, target_column))
            for mode in modes:
                chosen = _predict_action(models[mode], row, float(epsilon), mode,
                                         private_values if mode == "private_research_only" else None)
                yield_rows.append({
                    "feature_mode": mode,
                    "fold": fold,
                    "family_id": family_id,
                    "dataset_id": int(dataset_id),
                    "epsilon": float(epsilon),
                    "selector_action": f"{chosen[0]}__{chosen[1]}",
                    "fixed_action": f"{fixed[0]}__{fixed[1]}",
                    "epsilon_fixed_action": f"{fixed_epsilon[0]}__{fixed_epsilon[1]}",
                    "selector_regret": _regret_for(condition, chosen, target_column),
                    "fixed_regret": fixed_regret,
                    "epsilon_fixed_regret": epsilon_fixed_regret,
                    "default_regret": default_regret,
                    "oracle_regret": _regret_for(condition, actual_best, target_column),
                    "top1": float(chosen == actual_best),
                    **({
                        "classifier_regret_selected_l2": _regret_for(condition, chosen, "regret"),
                        "classifier_regret_best_utility_fixed": _regret_for(
                            condition, utility_fixed, "regret",
                        ),
                    } if target_column == "l2_regret" else {}),
                })

    predictions = pd.DataFrame(yield_rows)
    mode_summaries = {}
    for index, mode in enumerate(modes):
        mode_predictions = predictions[predictions.feature_mode == mode]
        mode_summaries[mode] = _summarize_predictions(mode_predictions, seed=seed + 90 + index * 10)
    summary = {
        "target_column": target_column,
        "n_datasets": int(predictions.dataset_id.nunique()),
        "n_families": int(predictions.family_id.nunique()),
        "n_outer_folds": int(n_splits),
        "actions": list(ACTION_NAMES),
        "feature_modes": mode_summaries,
        "private_research_only_note": (
            "Uses unprotected descriptors of training rows/labels. This mode is not a DP mechanism "
            "and its selection is outside the final model's formal privacy guarantee."
        ),
    }
    if target_column == "l2_regret":
        summary["classifier_utility_of_l2_selected_action"] = {
            mode: _summarize_l2_classifier_utility(
                predictions[predictions.feature_mode == mode], seed=seed + 180 + index * 10,
            )
            for index, mode in enumerate(modes)
        }
    # Keep the original public-only keys as the primary result for compatibility.
    summary.update(mode_summaries["public"])
    return summary, predictions

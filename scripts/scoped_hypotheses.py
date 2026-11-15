"""Small, reproducible selector ablations on an existing conditions table.

This script deliberately does not refit private classifiers.  It is used to
screen hypotheses whose only change is the public description supplied to the
meta-selector (for example using the actual train size rather than the full
manifest size).  A candidate is promoted only after a fresh portfolio run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold

from dpmeta.meta_features import DP_META_COLUMNS, PRIVATE_META_COLUMNS
from dpmeta.portfolio_benchmark import (
    PIPELINES,
    TREE_COUNT,
    _aggregate,
    _candidate_features,
    _row_weights,
    _condition_score,
)


def _cvar90(values: np.ndarray) -> float:
    values = np.sort(np.asarray(values, dtype=float))
    return float(values[int(np.floor(0.9 * len(values))):].mean())


def _feature_variant(meta_row: pd.Series, epsilon: float, pipeline: str,
                     private: dict[int, np.ndarray], dp: dict[int, np.ndarray],
                     *, train_n: int | None, effective_tree_count: bool) -> np.ndarray:
    row = meta_row.copy()
    if train_n is not None:
        row["n_instances"] = int(train_n)
    values = _candidate_features(
        row, epsilon, pipeline, "dp_research_rich", private, dp,
    )
    if effective_tree_count and train_n is not None:
        # public_features (17), portfolio action indicators, then the public DP block;
        # the forest block is [log 2, log scale, log noise, log trees, ...].
        forest_tree_count_index = 17 + len(PIPELINES) + 25 + 6 + 8 + 3
        values[forest_tree_count_index] = np.log1p(min(TREE_COUNT, int(train_n)))
    return values


def run(path: Path, manifest_path: Path, *, n_estimators: int = 300,
        n_splits: int = 5) -> dict:
    raw = pd.read_csv(path)
    manifest = pd.read_csv(manifest_path, keep_default_na=False)
    aggregated = _aggregate(raw, manifest)
    train_n = (raw.groupby(["dataset_id", "epsilon"], as_index=False)
                  .train_n.mean().rename(columns={"train_n": "mean_train_n"}))
    aggregated = aggregated.merge(train_n, on=["dataset_id", "epsilon"], how="left")
    manifest_one = manifest.drop_duplicates("dataset_id").set_index("dataset_id")
    private = {
        int(dataset_id): group.iloc[0][list(PRIVATE_META_COLUMNS)].to_numpy(float)
        for dataset_id, group in aggregated.groupby("dataset_id", sort=False)
    }
    dp = {
        int(dataset_id): group.iloc[0][list(DP_META_COLUMNS)].to_numpy(float)
        for dataset_id, group in aggregated.groupby("dataset_id", sort=False)
    }
    rows = []
    for (dataset_id, family_id, epsilon), group in aggregated.groupby(
            ["dataset_id", "family_id", "epsilon"], sort=True):
        rows.append({
            "dataset_id": int(dataset_id), "family_id": str(family_id),
            "epsilon": float(epsilon),
            "group": group.set_index("pipeline"),
            "train_n": int(round(group.mean_train_n.iloc[0])),
        })

    results = {}
    for name, train_size, effective_trees in (
            ("baseline_manifest_n", None, False),
            ("train_n", "train", False),
        ("train_n_effective_tree_count", "train", True)):
        predictions = []
        frame = pd.DataFrame(rows)
        feature_cache = {}
        for item in rows:
            meta = manifest_one.loc[item["dataset_id"]]
            n = item["train_n"] if train_size == "train" else None
            for pipeline in PIPELINES:
                feature_cache[(item["dataset_id"], item["epsilon"], pipeline)] = _feature_variant(
                    meta, item["epsilon"], pipeline, private, dp,
                    train_n=n, effective_tree_count=effective_trees,
                )
        groups = frame.family_id.to_numpy()
        splitter = GroupKFold(n_splits=n_splits)
        for fold, (train_index, test_index) in enumerate(
                splitter.split(frame, groups=groups)):
            train_rows = frame.iloc[train_index]
            test_rows = frame.iloc[test_index]
            train_conditions = aggregated[aggregated.dataset_id.isin(
                train_rows.dataset_id)]
            for epsilon in sorted(train_conditions.epsilon.unique()):
                block = train_conditions[train_conditions.epsilon == epsilon].copy()
                block["target"] = block.private_ba - block.groupby(
                    ["dataset_id", "epsilon"], sort=False).private_ba.transform("mean")
                features = []
                for row in block.itertuples(index=False):
                    features.append(feature_cache[(int(row.dataset_id), float(row.epsilon), row.pipeline)])
                model = RandomForestRegressor(
                    n_estimators=n_estimators, min_samples_leaf=3,
                    max_features=0.9, max_depth=6, max_samples=0.5,
                    random_state=1729 + fold + int(epsilon * 1000), n_jobs=4,
                )
                model.fit(np.vstack(features), block.target.to_numpy(float),
                          sample_weight=_row_weights(block))
                model.set_params(n_jobs=1)
                for row in test_rows[test_rows.epsilon == epsilon].itertuples(index=False):
                    condition = row.group
                    available = [p for p in PIPELINES if p in condition.index]
                    x = np.vstack([feature_cache[(int(row.dataset_id), float(epsilon), p)]
                                   for p in available])
                    chosen = available[int(np.argmax(model.predict(x)))]
                    train_fixed = train_conditions[train_conditions.epsilon == epsilon]
                    fixed_scores = train_fixed.groupby(["family_id", "pipeline"]).private_ba.mean()
                    fixed = fixed_scores.groupby("pipeline").mean().idxmax()
                    predictions.append({
                        "variant": name, "fold": fold,
                        "dataset_id": row.dataset_id, "epsilon": epsilon,
                        "family_id": row.family_id,
                        "epsilon": epsilon,
                        "selector_regret": float(condition.loc[chosen, "portfolio_regret"]),
                        "fixed_regret": float(condition.loc[fixed, "portfolio_regret"]),
                        "oracle_regret": float(condition.portfolio_regret.min()),
                    })
        pred = pd.DataFrame(predictions)
        family = pred.groupby("family_id").mean(numeric_only=True)
        gain = family.fixed_regret.mean() - family.selector_regret.mean()
        values = family.selector_regret.to_numpy(float)
        fixed_values = family.fixed_regret.to_numpy(float)
        results[name] = {
            "n_estimators": n_estimators,
            "mean_selector_regret": float(values.mean()),
            "median_selector_regret": float(np.median(values)),
            "q90_selector_regret": float(np.quantile(values, 0.9)),
            "cvar90_selector_regret": _cvar90(values),
            "mean_fixed_regret": float(fixed_values.mean()),
            "mean_gain_vs_fixed": float(gain),
            "mean_oracle_regret": float(family.oracle_regret.mean()),
        }
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--conditions", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--estimators", type=int, default=300)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.conditions, args.manifest, n_estimators=args.estimators,
                 n_splits=args.folds)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

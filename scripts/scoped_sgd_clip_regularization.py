"""Pilot an additional DP-SGD clip/regularization policy as a portfolio action."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold

from dpmeta.classifiers import fit_clean_softmax, fit_private_sgd_softmax
from dpmeta.meta_features import DP_META_COLUMNS, PRIVATE_META_COLUMNS
from dpmeta.model import balanced_accuracy
from dpmeta.openml_benchmark import (
    _get_openml_data, _openml, _seed_for, stratified_holdout_indices,
)
from dpmeta.portfolio_benchmark import PIPELINES, _candidate_features, _row_weights


CANDIDATE = "softmax__dp_sgd_clip2_reg003"
BASELINE_PIPELINES = tuple(pipeline for pipeline in PIPELINES if pipeline != CANDIDATE)


def _make_candidate(manifest, old, cache_dir: Path, *, clip: float,
                    regularization: float, limit: int) -> pd.DataFrame:
    openml = _openml(cache_dir)
    rows = []
    for _, row in manifest.head(limit).iterrows():
        x, y, schema = _get_openml_data(openml, row)
        for repetition in sorted(old[old.dataset_id == row.dataset_id].repetition.unique()):
            train, test = stratified_holdout_indices(
                y, _seed_for(20260917, int(row.dataset_id), int(repetition),
                              "portfolio_split"),
            )
            x_train, y_train = x[train], y[train]
            clean = fit_clean_softmax(
                x_train, y_train, len(schema.classes), regularization=0.1,
                optimizer_tolerance=1e-5,
                record_norm_bound=np.sqrt(len(schema.feature_names) + 1),
            )
            clean_ba = balanced_accuracy(y[test], clean.model.predict(x[test]))
            for epsilon in sorted(old.epsilon.unique()):
                model, _ = fit_private_sgd_softmax(
                    x_train, y_train, len(schema.classes), epsilon=epsilon,
                    delta=1e-6,
                    seed=_seed_for(20260917, int(row.dataset_id), int(repetition),
                                   epsilon, "clip_regularization", clip,
                                   regularization),
                    n_steps=100, gradient_clip=clip,
                    regularization=regularization,
                )
                rows.append({
                    "dataset_id": int(row.dataset_id),
                    "family_id": str(row.family_id),
                    "epsilon": float(epsilon),
                    "pipeline": CANDIDATE,
                    "repetition": int(repetition),
                    "private_ba": balanced_accuracy(y[test], model.predict(x[test])),
                    "clean_ba": clean_ba,
                })
    return pd.DataFrame(rows)


def _features(aggregated, manifest, clip, regularization):
    meta = manifest.drop_duplicates("dataset_id").set_index("dataset_id")
    private = {int(i): g.iloc[0][list(PRIVATE_META_COLUMNS)].to_numpy(float)
               for i, g in aggregated.groupby("dataset_id")}
    dp = {int(i): g.iloc[0][list(DP_META_COLUMNS)].to_numpy(float)
          for i, g in aggregated.groupby("dataset_id")}
    values = {}
    for row in aggregated.itertuples(index=False):
        value = _candidate_features(
            meta.loc[int(row.dataset_id)], float(row.epsilon), row.pipeline,
            "dp_research_rich", private, dp,
        ).copy()
        values[(int(row.dataset_id), float(row.epsilon), row.pipeline)] = value
    return values


def _evaluate(aggregated, features, *, include_candidate: bool, folds: int) -> dict:
    pool = tuple(PIPELINES) if include_candidate else BASELINE_PIPELINES
    families = aggregated[["dataset_id", "family_id"]].drop_duplicates()
    selected = []
    for fold, (train_index, test_index) in enumerate(GroupKFold(folds).split(
            families.dataset_id, groups=families.family_id)):
        train_ids = set(families.iloc[train_index].dataset_id)
        test_ids = set(families.iloc[test_index].dataset_id)
        train = aggregated[aggregated.dataset_id.isin(train_ids)]
        test = aggregated[aggregated.dataset_id.isin(test_ids)]
        for epsilon in sorted(train.epsilon.unique()):
            block = train[(train.epsilon == epsilon) & train.pipeline.isin(pool)].copy()
            block["target"] = block.private_ba - block.groupby(
                ["dataset_id", "epsilon"], sort=False).private_ba.transform("mean")
            model = RandomForestRegressor(
                n_estimators=200, min_samples_leaf=3, max_features=0.9,
                max_depth=6, max_samples=0.5,
                random_state=20260917 + fold * 31 + int(epsilon * 1000), n_jobs=4,
            )
            model.fit(np.vstack([
                features[(int(row.dataset_id), float(row.epsilon), row.pipeline)]
                for row in block.itertuples(index=False)
            ]), block.target.to_numpy(float), sample_weight=_row_weights(block))
            model.set_params(n_jobs=1)
            for (_, dataset_id, eps), group in test[test.epsilon == epsilon].groupby(
                    ["family_id", "dataset_id", "epsilon"], sort=True):
                group = group[group.pipeline.isin(pool)]
                prediction = model.predict(np.vstack([
                    features[(int(dataset_id), float(eps), pipeline)]
                    for pipeline in group.pipeline
                ]))
                chosen = group.pipeline.iloc[int(np.argmax(prediction))]
                selected.append({
                    "family_id": group.family_id.iloc[0],
                    "selector_regret": float(
                        group.loc[group.pipeline == chosen, "portfolio_regret"].iloc[0]
                    ),
                    "oracle_regret": float(group.portfolio_regret.min()),
                    "chosen": chosen,
                })
    result = pd.DataFrame(selected)
    family = result.groupby("family_id").mean(numeric_only=True)
    regret = family.selector_regret.to_numpy(float)
    return {
        "mean": float(regret.mean()), "median": float(np.median(regret)),
        "q90": float(np.quantile(regret, 0.9)),
        "cvar90": float(np.sort(regret)[int(np.floor(0.9 * len(regret))):].mean()),
        "oracle_mean": float(family.oracle_regret.mean()),
        "candidate_share": float(np.mean(result.chosen == CANDIDATE)),
    }


def run(conditions: Path, manifest_path: Path, cache_dir: Path, *, limit: int,
        clip: float = 2.0, regularization: float = 0.03, folds: int = 3) -> dict:
    old = pd.read_csv(conditions)
    ids = old.dataset_id.drop_duplicates().head(limit)
    old = old[old.dataset_id.isin(ids)].copy()
    manifest = pd.read_csv(manifest_path, keep_default_na=False)
    manifest = manifest[manifest.dataset_id.isin(ids)].copy()
    candidate = _make_candidate(
        manifest, old, cache_dir, clip=clip, regularization=regularization,
        limit=limit,
    )
    for column in old.columns:
        if column not in candidate:
            candidate[column] = np.nan
    combined = pd.concat([old, candidate[old.columns]], ignore_index=True, sort=False)
    combined["best_clean_portfolio_ba"] = combined.groupby(
        ["dataset_id", "repetition"]
    ).clean_ba.transform("max")
    combined["portfolio_regret"] = (
        combined.best_clean_portfolio_ba - combined.private_ba
    )
    combined["pipeline_privacy_penalty"] = combined.clean_ba - combined.private_ba
    aggregated = (combined.groupby(["dataset_id", "epsilon", "pipeline"],
                                   as_index=False, sort=True)
                  .agg(private_ba=("private_ba", "mean"),
                       clean_ba=("clean_ba", "mean"),
                       best_clean_portfolio_ba=("best_clean_portfolio_ba", "mean"),
                       portfolio_regret=("portfolio_regret", "mean"),
                       pipeline_privacy_penalty=("pipeline_privacy_penalty", "mean"),
                       **{c: (c, "first") for c in (*DP_META_COLUMNS, *PRIVATE_META_COLUMNS)
                          if c in combined.columns}))
    family_map = manifest.set_index("dataset_id").family_id
    aggregated["family_id"] = aggregated.dataset_id.map(family_map).astype(str)
    features = _features(aggregated, manifest, clip, regularization)
    return {
        "baseline": _evaluate(aggregated[aggregated.pipeline.isin(BASELINE_PIPELINES)],
                               features, include_candidate=False, folds=folds),
        "expanded": _evaluate(aggregated, features, include_candidate=True, folds=folds),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--conditions", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.conditions, args.manifest, args.cache_dir,
                 limit=args.limit, folds=args.folds)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

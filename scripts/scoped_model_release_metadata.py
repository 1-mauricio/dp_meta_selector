"""Screen selector metadata derived from already-released private models.

The release summaries below are post-processing of the Gaussian-zCDP GNB
model (no extra privacy budget).  They are deliberately kept separate from
the production feature vector until the selector gate is passed.
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
from dpmeta.model import fit_private
from dpmeta.openml_benchmark import (
    _get_openml_data,
    _openml,
    _seed_for,
    stratified_holdout_indices,
)
from dpmeta.portfolio_benchmark import (
    PIPELINES,
    _aggregate,
    _candidate_features,
    _row_weights,
    run_dataset,
)


def _release_summary(model) -> np.ndarray:
    means = np.asarray(model.means, dtype=float)
    variances = np.asarray(model.variances, dtype=float)
    priors = np.asarray(model.priors, dtype=float)
    entropy = -float(np.sum(priors * np.log(np.maximum(priors, 1e-12))))
    return np.asarray([
        float(np.mean(np.abs(means))),
        float(np.mean(np.log(np.maximum(variances, 1e-6)))),
        float(np.mean(np.square(means))),
        entropy,
        float(np.max(priors)),
    ], dtype=float)


def _release_features(openml, manifest: pd.DataFrame, epsilons: tuple[float, ...],
                      repeats: int, seed: int) -> dict[tuple[int, float, int], np.ndarray]:
    output = {}
    for _, row in manifest.iterrows():
        x, y, schema = _get_openml_data(openml, row)
        for repetition in range(repeats):
            train, _ = stratified_holdout_indices(
                y, _seed_for(seed, int(row.dataset_id), repetition, "portfolio_split"),
            )
            x_train, y_train = x[train], y[train]
            for epsilon in epsilons:
                model, _ = fit_private(
                    x_train, y_train, len(schema.classes), epsilon=epsilon,
                    mechanism="gaussian_zcdp", allocation="equal",
                    seed=_seed_for(seed, int(row.dataset_id), repetition, epsilon,
                                   PIPELINES[0]),
                )
                output[(int(row.dataset_id), float(epsilon), repetition)] = _release_summary(model)
    return output


def _make_features(aggregated: pd.DataFrame, manifest: pd.DataFrame,
                   releases: dict[tuple[int, float, int], np.ndarray],
                   *, use_release: bool) -> dict:
    meta = manifest.drop_duplicates("dataset_id").set_index("dataset_id")
    private = {int(i): g.iloc[0][list(PRIVATE_META_COLUMNS)].to_numpy(float)
               for i, g in aggregated.groupby("dataset_id")}
    dp = {int(i): g.iloc[0][list(DP_META_COLUMNS)].to_numpy(float)
          for i, g in aggregated.groupby("dataset_id")}
    values = {}
    for row in aggregated.itertuples(index=False):
        base = _candidate_features(
            meta.loc[int(row.dataset_id)], float(row.epsilon), row.pipeline,
            "dp_research_rich", private, dp,
        )
        if use_release:
            # Match the current DP metadata policy: release only repetition 0.
            release = releases[(int(row.dataset_id), float(row.epsilon), 0)]
            values[(int(row.dataset_id), float(row.epsilon), row.pipeline)] = np.r_[
                base, release,
            ]
        else:
            values[(int(row.dataset_id), float(row.epsilon), row.pipeline)] = base
    return values


def _evaluate(aggregated: pd.DataFrame, features: dict, folds: int) -> dict:
    families = aggregated[["dataset_id", "family_id"]].drop_duplicates()
    selected = []
    for fold, (train_index, test_index) in enumerate(GroupKFold(folds).split(
            families.dataset_id, groups=families.family_id)):
        train_ids = set(families.iloc[train_index].dataset_id)
        test_ids = set(families.iloc[test_index].dataset_id)
        train = aggregated[aggregated.dataset_id.isin(train_ids)]
        test = aggregated[aggregated.dataset_id.isin(test_ids)]
        for epsilon in sorted(train.epsilon.unique()):
            block = train[train.epsilon == epsilon].copy()
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
                prediction = model.predict(np.vstack([
                    features[(int(dataset_id), float(eps), pipeline)]
                    for pipeline in group.pipeline
                ]))
                chosen = group.pipeline.iloc[int(np.argmax(prediction))]
                selected.append({
                    "family_id": group.family_id.iloc[0],
                    "regret": float(group.loc[group.pipeline == chosen,
                                               "portfolio_regret"].iloc[0]),
                })
    family = pd.DataFrame(selected).groupby("family_id").regret.mean().to_numpy(float)
    return {
        "mean": float(family.mean()),
        "median": float(np.median(family)),
        "q90": float(np.quantile(family, 0.9)),
        "cvar90": float(np.sort(family)[int(np.floor(0.9 * len(family))):].mean()),
    }


def run(manifest_path: Path, cache_dir: Path, *, limit: int, repeats: int,
        folds: int) -> dict:
    epsilons = (0.1, 1.0, 4.0)
    manifest = pd.read_csv(manifest_path, keep_default_na=False).head(limit)
    openml = _openml(cache_dir)
    rows = []
    for _, row in manifest.iterrows():
        rows.extend(run_dataset(openml, row, epsilons, repeats, 20260917,
                                include_research_features=True))
    raw = pd.DataFrame(rows)
    aggregated = _aggregate(raw, manifest)
    releases = _release_features(openml, manifest, epsilons, repeats, 20260917)
    # The summary is based on repetition 0, matching the current DP metadata
    # release policy.  It is already a post-processing of a private model.
    baseline_features = _make_features(aggregated, manifest, releases, use_release=False)
    release_features = _make_features(aggregated, manifest, releases, use_release=True)
    return {
        "baseline": _evaluate(aggregated, baseline_features, folds),
        "model_release_metadata": _evaluate(aggregated, release_features, folds),
        "limit": limit,
        "repeats": repeats,
        "folds": folds,
        "release_feature_names": [
            "mean_abs_gnb_means", "mean_log_gnb_variances",
            "mean_sq_gnb_means", "gnb_prior_entropy", "max_gnb_prior",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.manifest, args.cache_dir, limit=args.limit,
                 repeats=args.repeats, folds=args.folds)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

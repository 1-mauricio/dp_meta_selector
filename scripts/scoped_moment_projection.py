"""Scoped private screen for feasible-moment projection in Gaussian NB.

The candidate clips noisy sufficient statistics to the public feasible set
before constructing the model.  Clipping and the variance constraint are
post-processing, so the candidate uses exactly the baseline Gaussian-zCDP
release and cannot spend additional privacy.
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
from dpmeta.model import (
    DELTA_TOTAL,
    SufficientStatistics,
    balanced_accuracy,
    fit_clean,
    model_from_statistics,
    privatize_sufficient_statistics,
    sufficient_statistics,
    predict,
)
from dpmeta.openml_benchmark import (
    _get_openml_data,
    _openml,
    _seed_for,
    stratified_holdout_indices,
)
from dpmeta.portfolio_benchmark import (
    PIPELINES,
    _candidate_features,
    _row_weights,
    run_dataset,
)


CANDIDATE = "gnb__gaussian_zcdp_equal_moment_projection"


def _project_statistics(stats: SufficientStatistics) -> SufficientStatistics:
    counts = np.maximum(np.asarray(stats.counts, dtype=float), 1e-6)
    sums = np.clip(np.asarray(stats.sums, dtype=float), -counts[:, None], counts[:, None])
    seconds = np.asarray(stats.seconds, dtype=float)
    seconds = np.clip(seconds, 0.0, counts[:, None])
    # For every class/feature, E[X^2] >= E[X]^2 on [-1, 1].
    seconds = np.maximum(seconds, np.square(sums) / counts[:, None])
    return SufficientStatistics(counts, sums, seconds)


def _candidate_rows(openml, row: pd.Series, epsilons: tuple[float, ...],
                    repeats: int, seed: int, baseline: pd.DataFrame) -> pd.DataFrame:
    x, y, schema = _get_openml_data(openml, row)
    rows = []
    for repetition in range(repeats):
        train, test = stratified_holdout_indices(
            y, _seed_for(seed, int(row.dataset_id), repetition, "portfolio_split"),
        )
        x_train, x_test = x[train], x[test]
        y_train, y_test = y[train], y[test]
        clean_score = float(baseline.loc[
            (baseline.dataset_id == int(row.dataset_id))
            & (baseline.repetition == repetition), "best_clean_portfolio_ba"
        ].iloc[0])
        clean_gnb = fit_clean(x_train, y_train, len(schema.classes))
        clean_ba = balanced_accuracy(y_test, predict(x_test, clean_gnb))
        for epsilon in epsilons:
            stats = sufficient_statistics(x_train, y_train, len(schema.classes))
            noisy, ledger = privatize_sufficient_statistics(
                stats, epsilon=epsilon, mechanism="gaussian_zcdp", allocation="equal",
                seed=_seed_for(seed, int(row.dataset_id), repetition, epsilon,
                               "gnb_moment_projection"), delta=DELTA_TOTAL,
            )
            projected = model_from_statistics(_project_statistics(noisy))
            private_ba = balanced_accuracy(y_test, predict(x_test, projected))
            source = baseline.loc[
                (baseline.dataset_id == int(row.dataset_id))
                & (baseline.repetition == repetition)
                & np.isclose(baseline.epsilon, epsilon)
                & (baseline.pipeline == PIPELINES[0])
            ].iloc[0]
            output = source.to_dict()
            output.update({
                "epsilon": float(epsilon),
                "pipeline": CANDIDATE,
                "private_ba": float(private_ba),
                "clean_ba": float(clean_ba),
                "best_clean_portfolio_ba": clean_score,
                "portfolio_regret": clean_score - float(private_ba),
                "pipeline_privacy_penalty": clean_ba - float(private_ba),
                "epsilon_used": float(ledger.epsilon_used),
                "delta_used": float(ledger.delta_used),
                "ledger_json": json.dumps(ledger.as_dict(), sort_keys=True),
            })
            rows.append(output)
    return pd.DataFrame(rows)


def _features(aggregated: pd.DataFrame, manifest: pd.DataFrame) -> dict:
    meta = manifest.drop_duplicates("dataset_id").set_index("dataset_id")
    private = {int(i): g.iloc[0][list(PRIVATE_META_COLUMNS)].to_numpy(float)
               for i, g in aggregated.groupby("dataset_id")}
    dp = {int(i): g.iloc[0][list(DP_META_COLUMNS)].to_numpy(float)
          for i, g in aggregated.groupby("dataset_id")}
    values = {}
    for row in aggregated.itertuples(index=False):
        source = PIPELINES[0] if row.pipeline == CANDIDATE else row.pipeline
        value = _candidate_features(
            meta.loc[int(row.dataset_id)], float(row.epsilon), source,
            "dp_research_rich", private, dp,
        )
        # The extra action bit is the only new selector feature.
        values[(int(row.dataset_id), float(row.epsilon), row.pipeline)] = np.r_[
            value, float(row.pipeline == CANDIDATE),
        ]
    return values


def _evaluate(aggregated: pd.DataFrame, features: dict, *, include_candidate: bool,
              folds: int) -> dict:
    pool = tuple(PIPELINES) + ((CANDIDATE,) if include_candidate else ())
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
        "mean": float(regret.mean()),
        "median": float(np.median(regret)),
        "q90": float(np.quantile(regret, 0.9)),
        "cvar90": float(np.sort(regret)[int(np.floor(0.9 * len(regret))):].mean()),
        "oracle_mean": float(family.oracle_regret.mean()),
        "candidate_share": float(np.mean(result.chosen == CANDIDATE)),
    }


def run(manifest_path: Path, cache_dir: Path, *, limit: int,
        repeats: int = 2, folds: int = 3) -> dict:
    manifest = pd.read_csv(manifest_path, keep_default_na=False)
    selected_ids = manifest.dataset_id.head(limit).astype(int).tolist()
    manifest = manifest[manifest.dataset_id.isin(selected_ids)].copy()
    openml = _openml(cache_dir)
    baseline_rows = []
    for _, row in manifest.iterrows():
        baseline_rows.extend(run_dataset(
            openml, row, (0.1, 1.0, 4.0), repeats, 20260917,
            include_research_features=True,
        ))
    baseline = pd.DataFrame(baseline_rows)
    candidate = pd.concat([
        _candidate_rows(openml, row, (0.1, 1.0, 4.0), repeats, 20260917, baseline)
        for _, row in manifest.iterrows()
    ], ignore_index=True)
    combined = pd.concat([baseline, candidate], ignore_index=True, sort=False)
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
    features = _features(aggregated, manifest)
    return {
        "baseline": _evaluate(aggregated[aggregated.pipeline.isin(PIPELINES)],
                               features, include_candidate=False, folds=folds),
        "expanded": _evaluate(aggregated, features, include_candidate=True, folds=folds),
        "candidate": CANDIDATE,
        "limit": limit,
        "repeats": repeats,
        "folds": folds,
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

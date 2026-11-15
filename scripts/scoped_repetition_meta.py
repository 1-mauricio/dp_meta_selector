"""Pilot for repetition-aligned DP meta-features.

The current benchmark releases DP metadata on repetition zero and then
averages outcomes over repetitions.  This script compares that protocol with
one DP metadata release per hypothetical deployment split, keeping the outer
family grouping unchanged.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold

from dpmeta.meta_features import DP_META_COLUMNS, PRIVATE_META_COLUMNS, dp_training_meta_features
from dpmeta.openml_benchmark import _get_openml_data, _openml, _seed_for, stratified_holdout_indices
from dpmeta.portfolio_benchmark import PIPELINES, _aggregate, _candidate_features, _row_weights


def _cvar90(v):
    v = np.sort(np.asarray(v, dtype=float))
    return float(v[int(np.floor(0.9 * len(v))):].mean())


def _fit_predict(train, test, features, *, seed):
    block = train.copy()
    block["target"] = block.private_ba - block.groupby(
        ["dataset_id", "epsilon"], sort=False).private_ba.transform("mean")
    model = RandomForestRegressor(
        n_estimators=200, min_samples_leaf=3, max_features=0.9,
        max_depth=6, max_samples=0.5, random_state=seed, n_jobs=4,
    )
    x = np.vstack([features[k] for k in block["key"]])
    model.fit(x, block.target.to_numpy(float), sample_weight=_row_weights(block))
    model.set_params(n_jobs=1)
    selected = []
    group_cols = ["family_id", "dataset_id", "epsilon"]
    if "repetition" in test.columns:
        group_cols.append("repetition")
    for key, group in test.groupby(group_cols, sort=True):
        candidates = [p for p in PIPELINES if p in set(group.pipeline)]
        pred = model.predict(np.vstack([
            features[group[group.pipeline == p].iloc[0].key] for p in candidates
        ]))
        chosen = candidates[int(np.argmax(pred))]
        selected.append({
            "family_id": group.family_id.iloc[0],
            "dataset_id": group.dataset_id.iloc[0],
            "epsilon": group.epsilon.iloc[0],
            "selector_regret": float(group[group.pipeline == chosen].portfolio_regret.mean()),
            "oracle_regret": float(group.portfolio_regret.mean()),
        })
    return pd.DataFrame(selected)


def _summarize(pred):
    family = pred.groupby("family_id").selector_regret.mean().to_numpy(float)
    return {
        "mean": float(family.mean()), "median": float(np.median(family)),
        "q90": float(np.quantile(family, 0.9)), "cvar90": _cvar90(family),
    }


def run(conditions_path: Path, manifest_path: Path, cache_dir: Path,
        *, folds: int = 3, seed: int = 20260917) -> dict:
    raw = pd.read_csv(conditions_path)
    manifest = pd.read_csv(manifest_path, keep_default_na=False).head(
        raw.dataset_id.nunique(),
    )
    openml = _openml(cache_dir)
    meta = manifest.drop_duplicates("dataset_id").set_index("dataset_id")
    private_by_dataset = {
        int(i): g.iloc[0][list(PRIVATE_META_COLUMNS)].to_numpy(float)
        for i, g in raw.groupby("dataset_id", sort=False)
    }
    # Build an exact DP metadata release for each split.  This is a pilot, so
    # each release uses the same public epsilon and seed convention as the
    # portfolio benchmark.
    dp_by_key = {}
    for _, row in manifest.iterrows():
        x, y, schema = _get_openml_data(openml, row)
        for repetition in sorted(raw[raw.dataset_id == row.dataset_id].repetition.unique()):
            train, _ = stratified_holdout_indices(
                y, _seed_for(seed, int(row.dataset_id), int(repetition), "portfolio_split"),
            )
            values, _ = dp_training_meta_features(
                x[train], y[train], len(schema.classes), epsilon=1.0,
                seed=_seed_for(seed, int(row.dataset_id), int(repetition), "dp_meta"),
            )
            dp_by_key[(int(row.dataset_id), int(repetition))] = values

    # Current protocol: aggregate outcomes across repetitions and use release 0.
    agg = _aggregate(raw, manifest)
    current = []
    current_features = {}
    for row in agg.itertuples(index=False):
        key = (int(row.dataset_id), float(row.epsilon), "aggregate", row.pipeline)
        current.append({**row._asdict(), "key": key})
        current_features[key] = _candidate_features(
            meta.loc[int(row.dataset_id)], float(row.epsilon), row.pipeline,
            "dp_research_rich",
            private_by_dataset, {int(row.dataset_id): dp_by_key[(int(row.dataset_id), 0)]},
        )
    current = pd.DataFrame(current)

    # Repetition-aligned protocol: retain each repetition as a meta-example.
    aligned = []
    aligned_features = {}
    for row in raw.itertuples(index=False):
        did, rep, eps, pipeline = int(row.dataset_id), int(row.repetition), float(row.epsilon), row.pipeline
        synthetic = meta.loc[did].copy()
        synthetic["dataset_id"] = did * 10000 + rep
        aligned_key = (did, eps, rep, pipeline)
        aligned.append({**row._asdict(), "key": aligned_key})
        aligned_features[aligned_key] = _candidate_features(
            synthetic, eps, pipeline, "dp_research_rich", private_by_dataset,
            {did * 10000 + rep: dp_by_key[(did, rep)]},
        )
    aligned = pd.DataFrame(aligned)

    result = {}
    for name, table, features in (("current_aggregate", current, current_features),
                                  ("aligned_repetition", aligned, aligned_features)):
        outputs = []
        dataset_families = table[["dataset_id", "family_id"]].drop_duplicates()
        split = GroupKFold(n_splits=folds)
        for fold, (tr, te) in enumerate(split.split(
                dataset_families.dataset_id, groups=dataset_families.family_id)):
            train_ids = set(dataset_families.iloc[tr].dataset_id)
            test_ids = set(dataset_families.iloc[te].dataset_id)
            train = table[table.dataset_id.isin(train_ids)]
            test = table[table.dataset_id.isin(test_ids)]
            outputs.append(_fit_predict(
                train, test, features, seed=seed + fold * 31,
            ))
        pred = pd.concat(outputs, ignore_index=True)
        result[name] = _summarize(pred)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--conditions", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.conditions, args.manifest, args.cache_dir, folds=args.folds)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Scoped screen for a privacy-safe adaptive forest threshold policy.

Each threshold is a Laplace-noised bounded median.  The threshold releases
consume a fixed share of epsilon; the remaining share is used by the private
leaf histograms.  This is intentionally conservative and is not part of the
production portfolio unless it improves the full selector gate.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from dpmeta.classifiers import (
    RandomForestPlan,
    RandomTreeStructure,
    encoded_numeric_mask,
    fit_clean_random_forest,
    fit_private_random_forest,
    make_random_forest_plan,
)
from dpmeta.model import balanced_accuracy
from dpmeta.mechanisms import privacy_rng
from dpmeta.openml_benchmark import (
    _get_openml_data,
    _openml,
    _seed_for,
    stratified_holdout_indices,
)


def adaptive_plan(x_train, numeric_mask, *, seed: int, epsilon_threshold: float,
                  n_trees: int = 1000, depth: int = 3) -> RandomForestPlan:
    base = make_random_forest_plan(
        len(x_train), numeric_mask, seed=seed, n_trees=n_trees, depth=depth,
    )
    n_internal = 2**depth - 1
    total_releases = n_trees * n_internal
    per_release = epsilon_threshold / total_releases
    rng = privacy_rng(seed + 17)
    structures = []
    for structure in base.structures:
        thresholds = structure.thresholds.copy()
        for node, feature in enumerate(structure.features[:n_internal]):
            values = np.asarray(x_train[:, feature], dtype=float)
            center = float(np.median(values))
            scale = 2.0 / max(len(values) * per_release, 1e-12)
            thresholds[node] = np.clip(center + rng.laplace(0.0, scale), -1.0, 1.0)
        structures.append(RandomTreeStructure(structure.features, thresholds, depth))
    return RandomForestPlan(tuple(structures), base.shards)


def run(manifest_path: Path, cache_dir: Path, *, limit: int, repeats: int) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_path, keep_default_na=False).head(limit)
    openml = _openml(cache_dir)
    rows = []
    for _, row in manifest.iterrows():
        x, y, schema = _get_openml_data(openml, row)
        for repetition in range(repeats):
            train, test = stratified_holdout_indices(
                y, _seed_for(20260917, int(row.dataset_id), repetition,
                              "portfolio_split"),
            )
            x_train, y_train = x[train], y[train]
            mask = encoded_numeric_mask(schema)
            for epsilon in (0.1, 1.0, 4.0):
                threshold_share = 0.2
                plan = adaptive_plan(
                    x_train, mask,
                    seed=_seed_for(20260917, int(row.dataset_id), repetition,
                                   epsilon, "adaptive_threshold_plan"),
                    epsilon_threshold=epsilon * threshold_share,
                    n_trees=1000, depth=3,
                )
                clean = fit_clean_random_forest(
                    x_train, y_train, len(schema.classes), plan,
                )
                private, _ = fit_private_random_forest(
                    x_train, y_train, len(schema.classes),
                    epsilon=epsilon * (1.0 - threshold_share),
                    seed=_seed_for(20260917, int(row.dataset_id), repetition,
                                   epsilon, "adaptive_threshold_forest"),
                    plan=plan,
                )
                baseline_plan = make_random_forest_plan(
                    len(y_train), mask,
                    seed=_seed_for(20260917, int(row.dataset_id), repetition,
                                   "tree_plan"), n_trees=1000, depth=3,
                )
                baseline, _ = fit_private_random_forest(
                    x_train, y_train, len(schema.classes), epsilon=epsilon,
                    seed=_seed_for(20260917, int(row.dataset_id), repetition,
                                   epsilon, "forest"), plan=baseline_plan,
                )
                rows.append({
                    "dataset_id": int(row.dataset_id),
                    "repetition": repetition,
                    "epsilon": epsilon,
                    "baseline_private_ba": balanced_accuracy(y[test], baseline.predict(x[test])),
                    "adaptive_private_ba": balanced_accuracy(y[test], private.predict(x[test])),
                    "adaptive_clean_ba": balanced_accuracy(y[test], clean.predict(x[test])),
                })
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.manifest, args.cache_dir, limit=args.limit, repeats=args.repeats)
    result.to_csv(args.output, index=False)
    print(result.groupby("epsilon")[["baseline_private_ba", "adaptive_private_ba"]].mean())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

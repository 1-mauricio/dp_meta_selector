"""Private utility screen for class-balanced random-forest leaves."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from dpmeta.classifiers import (
    RandomForestModel,
    _counts_to_probabilities,
    _leaf_counts,
    encoded_numeric_mask,
    fit_clean_random_forest,
    fit_private_random_forest,
    make_random_forest_plan,
)
from dpmeta.mechanisms import privacy_rng
from dpmeta.model import balanced_accuracy
from dpmeta.openml_benchmark import (
    _get_openml_data,
    _openml,
    _seed_for,
    stratified_holdout_indices,
)


def _rebalance(model: RandomForestModel, prior: np.ndarray) -> RandomForestModel:
    prior = np.maximum(np.asarray(prior, dtype=float), 1e-6)
    prior /= prior.sum()
    probabilities = []
    for values in model.leaf_probabilities:
        adjusted = np.asarray(values, dtype=float) / prior[None, :]
        adjusted /= np.maximum(adjusted.sum(axis=1, keepdims=True), 1e-12)
        probabilities.append(adjusted)
    return RandomForestModel(model.structures, tuple(probabilities))


def fit_private_class_balanced(x, y, n_classes, *, epsilon, seed, plan):
    rng = privacy_rng(seed)
    scale = 2.0 / epsilon
    probabilities = []
    global_noisy = np.zeros(n_classes, dtype=float)
    for structure, shard in zip(plan.structures, plan.shards):
        counts = _leaf_counts(x[shard], y[shard], n_classes, structure)
        noisy = counts + rng.laplace(0.0, scale, size=counts.shape)
        global_noisy += noisy.sum(axis=0)
        probabilities.append(_counts_to_probabilities(noisy))
    model = RandomForestModel(plan.structures, tuple(probabilities))
    return _rebalance(model, global_noisy)


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
            plan = make_random_forest_plan(
                len(y_train), encoded_numeric_mask(schema),
                seed=_seed_for(20260917, int(row.dataset_id), repetition, "tree_plan"),
                n_trees=1000, depth=3,
            )
            clean = fit_clean_random_forest(x_train, y_train, len(schema.classes), plan)
            clean_prior = np.bincount(y_train, minlength=len(schema.classes)).astype(float)
            clean_balanced = _rebalance(clean, clean_prior)
            for epsilon in (0.1, 1.0, 4.0):
                private_base, _ = fit_private_random_forest(
                    x_train, y_train, len(schema.classes), epsilon=epsilon,
                    seed=_seed_for(20260917, int(row.dataset_id), repetition,
                                   epsilon, "forest"), plan=plan,
                )
                private = fit_private_class_balanced(
                    x_train, y_train, len(schema.classes), epsilon=epsilon,
                    seed=_seed_for(20260917, int(row.dataset_id), repetition,
                                   epsilon, "forest_class_balanced"), plan=plan,
                )
                rows.append({
                    "dataset_id": int(row.dataset_id),
                    "repetition": repetition,
                    "epsilon": epsilon,
                    "forest_clean_ba": balanced_accuracy(y[test], clean.predict(x[test])),
                    "balanced_clean_ba": balanced_accuracy(y[test], clean_balanced.predict(x[test])),
                    "forest_private_ba": balanced_accuracy(y[test], private_base.predict(x[test])),
                    "balanced_private_ba": balanced_accuracy(y[test], private.predict(x[test])),
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
    print(result.groupby("epsilon")[["forest_private_ba", "balanced_private_ba"]].mean())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

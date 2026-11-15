"""Private pilot for random-forest shard-count choices.

This measures classifier utility only; it does not silently add a candidate to
the portfolio. Promotion requires rerunning the complete selector benchmark.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from dpmeta.classifiers import (
    encoded_numeric_mask,
    fit_clean_random_forest,
    fit_private_random_forest,
    make_random_forest_plan,
)
from dpmeta.model import balanced_accuracy
from dpmeta.openml_benchmark import (
    _get_openml_data,
    _openml,
    _seed_for,
    stratified_holdout_indices,
)


def run(manifest_path: Path, cache_dir: Path, *, limit: int = 5,
        repeats: int = 2, epsilons: tuple[float, ...] = (0.1, 1.0, 4.0)) -> pd.DataFrame:
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
            for n_trees in (25, 50, 100, 250, 1000):
                plan = make_random_forest_plan(
                    len(y_train), mask,
                    seed=_seed_for(20260917, int(row.dataset_id), repetition,
                                   "tree_plan", n_trees),
                    n_trees=n_trees, depth=3,
                )
                clean = fit_clean_random_forest(
                    x_train, y_train, len(schema.classes), plan,
                )
                for epsilon in epsilons:
                    private, _ = fit_private_random_forest(
                        x_train, y_train, len(schema.classes), epsilon=epsilon,
                        seed=_seed_for(20260917, int(row.dataset_id), repetition,
                                       epsilon, "forest", n_trees),
                        plan=plan,
                    )
                    clean_ba = balanced_accuracy(y[test], clean.predict(x[test]))
                    private_ba = balanced_accuracy(y[test], private.predict(x[test]))
                    rows.append({
                        "dataset_id": int(row.dataset_id),
                        "repetition": repetition,
                        "epsilon": epsilon,
                        "n_trees": n_trees,
                        "clean_ba": clean_ba,
                        "private_ba": private_ba,
                    })
    result = pd.DataFrame(rows)
    result["regret_within_tree_count"] = (
        result.groupby(["dataset_id", "repetition", "epsilon"]).clean_ba.transform("max")
        - result.private_ba
    )
    return result


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
    print(result.groupby("n_trees")[["clean_ba", "private_ba",
                                     "regret_within_tree_count"]].mean())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

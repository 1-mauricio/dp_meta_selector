"""Pilot comparing empirical and uniform GNB class priors."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from dpmeta.model import balanced_accuracy, fit_clean, fit_private, predict
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
            clean = fit_clean(x_train, y_train, len(schema.classes))
            clean_uniform = clean.__class__(
                clean.means, clean.variances,
                np.full(len(schema.classes), 1.0 / len(schema.classes)),
            )
            for epsilon in epsilons:
                private, _ = fit_private(
                    x_train, y_train, len(schema.classes), epsilon=epsilon,
                    mechanism="analytic_gaussian", allocation="equal",
                    seed=_seed_for(20260917, int(row.dataset_id), repetition,
                                   epsilon, "uniform_gnb"),
                )
                private_uniform = private.__class__(
                    private.means, private.variances,
                    np.full(len(schema.classes), 1.0 / len(schema.classes)),
                )
                rows.extend([
                    {"dataset_id": int(row.dataset_id), "repetition": repetition,
                     "epsilon": epsilon, "prior": "empirical",
                     "clean_ba": balanced_accuracy(y[test], predict(x[test], clean)),
                     "private_ba": balanced_accuracy(y[test], predict(x[test], private))},
                    {"dataset_id": int(row.dataset_id), "repetition": repetition,
                     "epsilon": epsilon, "prior": "uniform",
                     "clean_ba": balanced_accuracy(y[test], predict(x[test], clean_uniform)),
                     "private_ba": balanced_accuracy(y[test], predict(x[test], private_uniform))},
                ])
    result = pd.DataFrame(rows)
    result["regret"] = result.clean_ba - result.private_ba
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
    print(result.groupby("prior")[["clean_ba", "private_ba", "regret"]].mean())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

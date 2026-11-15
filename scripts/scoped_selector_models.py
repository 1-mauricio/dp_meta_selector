"""Screen selector regressors/losses using an existing conditions table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.model_selection import GroupKFold

from dpmeta.meta_features import DP_META_COLUMNS, PRIVATE_META_COLUMNS
from dpmeta.portfolio_benchmark import (
    PIPELINES, _aggregate, _candidate_features, _row_weights,
)


def _cvar90(values: np.ndarray) -> float:
    values = np.sort(np.asarray(values, dtype=float))
    return float(values[int(np.floor(0.9 * len(values))):].mean())


def _make_model(kind: str, seed: int, n_estimators: int):
    common = dict(n_estimators=n_estimators, max_features=0.9,
                  max_samples=0.5, n_jobs=4, random_state=seed)
    if kind == "current":
        return RandomForestRegressor(min_samples_leaf=3, max_depth=6, **common)
    if kind == "leaf1":
        return RandomForestRegressor(min_samples_leaf=1, max_depth=None, **common)
    if kind == "leaf8":
        return RandomForestRegressor(min_samples_leaf=8, max_depth=8, **common)
    if kind == "extra":
        return ExtraTreesRegressor(min_samples_leaf=3, max_depth=None,
                                   bootstrap=True, **common)
    raise ValueError(kind)


def run(path: Path, manifest_path: Path, *, folds: int = 5,
        n_estimators: int = 100, requested: tuple[str, ...] | None = None) -> dict:
    raw = pd.read_csv(path)
    manifest = pd.read_csv(manifest_path, keep_default_na=False)
    agg = _aggregate(raw, manifest)
    meta = manifest.drop_duplicates("dataset_id").set_index("dataset_id")
    private = {int(i): g.iloc[0][list(PRIVATE_META_COLUMNS)].to_numpy(float)
               for i, g in agg.groupby("dataset_id", sort=False)}
    dp = {int(i): g.iloc[0][list(DP_META_COLUMNS)].to_numpy(float)
          for i, g in agg.groupby("dataset_id", sort=False)}
    grouped = []
    cache = {}
    available_pipelines = tuple(p for p in PIPELINES if p in set(agg.pipeline))
    for (did, family, eps), group in agg.groupby(
            ["dataset_id", "family_id", "epsilon"], sort=True):
        grouped.append((int(did), str(family), float(eps), group.set_index("pipeline")))
        for p in available_pipelines:
            cache[(int(did), float(eps), p)] = _candidate_features(
                meta.loc[int(did)], float(eps), p, "dp_research_rich", private, dp,
            )
    frame = pd.DataFrame(grouped, columns=["dataset_id", "family_id", "epsilon", "condition"])
    result = {}
    variants = (
        ("current_centered", "current", "centered", False),
        ("leaf1_centered", "leaf1", "centered", False),
        ("leaf8_centered", "leaf8", "centered", False),
        ("extra_centered", "extra", "centered", False),
        ("current_regret", "current", "regret", False),
        ("current_heads", "current", "centered", True),
    )
    if requested is not None:
        variants = tuple(item for item in variants if item[0] in requested)
    for name, model_kind, target_kind, heads in variants:
        out = []
        split = GroupKFold(n_splits=folds)
        for fold, (tr_idx, te_idx) in enumerate(
                split.split(frame, groups=frame.family_id.to_numpy())):
            train_ids = set(frame.iloc[tr_idx].dataset_id)
            train = agg[agg.dataset_id.isin(train_ids)]
            for eps in sorted(train.epsilon.unique()):
                block = train[train.epsilon == eps].copy()
                if target_kind == "centered":
                    block["target"] = block.private_ba - block.groupby(
                        ["dataset_id", "epsilon"], sort=False).private_ba.transform("mean")
                elif target_kind == "regret":
                    block["target"] = -block.portfolio_regret
                else:
                    block["target"] = block.private_ba
                models = {}
                model_pipes = available_pipelines if heads else (None,)
                for pipeline in model_pipes:
                    fit_block = block if pipeline is None else block[block.pipeline == pipeline]
                    features = np.vstack([cache[(int(r.dataset_id), float(r.epsilon), r.pipeline)]
                                          for r in fit_block.itertuples(index=False)])
                    # Match portfolio_benchmark's dp_research_rich seed path.
                    model = _make_model(
                        model_kind,
                        20260917 + fold * 31 + 4 * 1009 + int(eps * 1000),
                        n_estimators,
                    )
                    model.fit(features, fit_block.target.to_numpy(float),
                              sample_weight=_row_weights(fit_block))
                    model.set_params(n_jobs=1)
                    models[pipeline] = model
                fixed_scores = train[train.epsilon == eps].groupby(
                    ["family_id", "pipeline"]).private_ba.mean()
                fixed = fixed_scores.groupby("pipeline").mean().idxmax()
                tests = frame.iloc[te_idx]
                for row in tests[tests.epsilon == eps].itertuples(index=False):
                    condition = row.condition
                    available = [p for p in available_pipelines if p in condition.index]
                    x = np.vstack([cache[(int(row.dataset_id), float(eps), p)] for p in available])
                    if heads:
                        pred = np.asarray([models[p].predict(x[j:j + 1])[0]
                                           for j, p in enumerate(available)])
                    else:
                        pred = models[None].predict(x)
                    chosen = available[int(np.argmax(pred))]
                    out.append({
                        "dataset_id": row.dataset_id,
                        "epsilon": eps,
                        "family_id": row.family_id,
                        "selector_regret": float(condition.loc[chosen, "portfolio_regret"]),
                        "fixed_regret": float(condition.loc[fixed, "portfolio_regret"]),
                        "oracle_regret": float(condition.portfolio_regret.min()),
                    })
        pred = pd.DataFrame(out)
        fam = pred.groupby("family_id").mean(numeric_only=True)
        # The production report uses family as the independent unit.  Keep
        # all distributional metrics at that same level.
        sel = fam.selector_regret.to_numpy(float)
        fixed = fam.fixed_regret.to_numpy(float)
        result[name] = {
            "n_estimators": n_estimators,
            "mean_selector_regret": float(sel.mean()),
            "median_selector_regret": float(np.median(sel)),
            "q90_selector_regret": float(np.quantile(sel, 0.9)),
            "cvar90_selector_regret": _cvar90(sel),
            "mean_fixed_regret": float(fixed.mean()),
            "mean_gain_vs_fixed": float(fam.fixed_regret.mean() - fam.selector_regret.mean()),
            "mean_oracle_regret": float(fam.oracle_regret.mean()),
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--conditions", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--estimators", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variants", default="",
                        help="nomes separados por vírgula; vazio executa todos")
    args = parser.parse_args()
    requested = tuple(part for part in args.variants.split(",") if part) or None
    value = run(args.conditions, args.manifest, folds=args.folds,
                 n_estimators=args.estimators, requested=requested)
    args.output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

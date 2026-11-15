"""Reanalyse saved portfolio conditions with one selector shared across epsilon."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from threadpoolctl import threadpool_limits

from dpmeta.openml_benchmark import _load_manifest
from dpmeta.portfolio_benchmark import (
    PIPELINES, _aggregate, _write_report, evaluate_grouped, summarize_grouped,
    validate_condition_grid,
)
from dpmeta.provenance import (
    file_sha256, numerical_implementation_sha256, runtime_fingerprint,
    save_source_snapshot,
)


SCORE_COLUMNS = (
    "dataset_id", "epsilon", "pipeline", "repetition", "private_ba",
    "clean_ba", "best_clean_portfolio_ba", "portfolio_regret",
    "pipeline_privacy_penalty",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path,
                        default=Path("data/openml_manifest_984_completed.csv"))
    parser.add_argument("--conditions", type=Path,
                        default=Path("results/portfolio-984-current/conditions.csv"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/portfolio-984-shared-selector"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--lodo", action="store_true",
                        help="leave-one-family-out; ignora --folds")
    parser.add_argument("--num-shards", type=int, default=1,
                        help="divide os folds em N jobs independentes")
    parser.add_argument("--shard", type=int,
                        help="roda só este shard e grava shards/predictions_<i>.csv")
    parser.add_argument("--merge-shards", action="store_true",
                        help="junta shards/predictions_*.csv e gera o resumo")
    from dpmeta import selector_portfolio13 as sel
    parser.add_argument('--selector', choices=sel.SELECTORS, default='rf12')
    args = parser.parse_args(argv)
    if args.selector == 'gbm_residual_fallback':
        if args.output == Path('results/portfolio-984-shared-selector'):args.output=Path('results/dissertation-portfolio13/shared')
        args.output.mkdir(parents=True, exist_ok=True)
        c = pd.read_csv(args.conditions); m = _load_manifest(args.manifest)
        if args.lodo:
            from scripts.dissertation_analysis_portfolio13 import base
            base(args.conditions, args.manifest, args.output, Path('results/experiments/lodo_13/lodo_picks.csv'))
        else:
            with threadpool_limits(limits=1):
                summary, predictions = sel.grouped(c, m, n_splits=args.folds)
            predictions.to_csv(args.output / 'outer_predictions.csv', index=False)
            (args.output / 'summary.json').write_text(json.dumps(summary, indent=2))
        return 0

    if args.folds < 2:
        parser.error("--folds precisa ser >= 2")
    if args.shard is not None and not 0 <= args.shard < args.num_shards:
        parser.error("--shard precisa estar em [0, --num-shards)")
    if args.shard is not None and args.merge_shards:
        parser.error("--shard e --merge-shards são exclusivos")
    source_config_path = args.conditions.parent / "run_config.json"
    source_config = json.loads(source_config_path.read_text())
    manifest_hash = file_sha256(args.manifest)
    if manifest_hash != source_config["manifest_sha256"]:
        raise ValueError("manifesto difere do registrado nas condições originais")
    epsilons = tuple(float(value) for value in source_config["epsilons"])
    repeats = int(source_config["repeats"])
    pipelines = tuple(source_config["pipelines"])
    if pipelines != PIPELINES:
        raise ValueError("portfólio atual difere do portfólio das condições")

    n_splits = None if args.lodo else args.folds
    shard_dir = args.output / "shards"
    if args.shard is not None:
        manifest = _load_manifest(args.manifest).head(int(source_config["limit"])).copy()
        conditions = pd.read_csv(
            args.conditions, usecols=lambda column: column in SCORE_COLUMNS,
        )
        shard_dir.mkdir(parents=True, exist_ok=True)
        with threadpool_limits(limits=1):
            _, predictions = evaluate_grouped(
                conditions, manifest, n_splits=n_splits, seed=args.seed,
                selector_strategy="absolute_shared",
                feature_modes_override=("public_plus_dp",), progress=True,
                fold_subset=set(range(args.shard, len(manifest), args.num_shards)),
            )
        target = shard_dir / f"predictions_{args.shard}.csv"
        predictions.to_csv(target.with_suffix(".tmp"), index=False)
        target.with_suffix(".tmp").rename(target)
        print(json.dumps({"shard": args.shard, "folds": int(predictions.fold.nunique())}))
        return 0

    output_files = ("run_config.json", "summary.json", "outer_predictions.csv", "report.md")
    args.output.mkdir(parents=True, exist_ok=True)
    if not args.overwrite and any((args.output / name).exists() for name in output_files):
        raise FileExistsError(f"análise já existe em {args.output}; use --overwrite")

    manifest = _load_manifest(args.manifest).head(int(source_config["limit"])).copy()
    conditions = pd.read_csv(
        args.conditions, usecols=lambda column: column in SCORE_COLUMNS,
    )
    validate_condition_grid(conditions, manifest, epsilons=epsilons,
                            repeats=repeats, pipelines=pipelines)
    analysis_hash = numerical_implementation_sha256()
    config = {
        "source_conditions": str(args.conditions.resolve()),
        "source_conditions_sha256": file_sha256(args.conditions),
        "source_run_config_sha256": file_sha256(source_config_path),
        "source_run_implementation_sha256": source_config["implementation_sha256"],
        "source_manifest": str(args.manifest.resolve()),
        "source_manifest_sha256": manifest_hash,
        "analysis_implementation_sha256": analysis_hash,
        "analysis_script_sha256": file_sha256(Path(__file__)),
        "source_matches_current_implementation": (
            source_config["implementation_sha256"] == analysis_hash
        ),
        "selector_strategy": "absolute_shared",
        "selector_feature_mode": "public_plus_dp",
        "selector_target": "private_ba",
        "folds": "LODO" if args.lodo else args.folds,
        "num_shards": args.num_shards if args.merge_shards else 1,
        "seed": args.seed,
        "source_split_seed": int(source_config["seed"]),
        "epsilons": list(epsilons),
        "repeats": repeats,
        "pipelines": list(pipelines),
        "runtime": runtime_fingerprint(),
    }
    save_source_snapshot(args.output, extra_files=("scripts/analyze_shared_selector.py",))
    (args.output / "run_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n"
    )
    if args.merge_shards:
        paths = [shard_dir / f"predictions_{i}.csv" for i in range(args.num_shards)]
        missing = [str(path) for path in paths if not path.exists()]
        if missing:
            raise FileNotFoundError(f"shards ausentes: {missing}")
        predictions = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
        aggregated = _aggregate(conditions, manifest)
        folds = int(aggregated.family_id.nunique()) if args.lodo else args.folds
        if sorted(predictions.fold.unique()) != list(range(folds)):
            raise ValueError("shards não cobrem todos os folds exatamente")
        if predictions.duplicated(["dataset_id", "epsilon"]).any():
            raise ValueError("dataset/ε previsto em mais de um fold")
        summary = summarize_grouped(
            predictions, aggregated, n_splits=folds, seed=args.seed,
            selector_strategy="absolute_shared", feature_modes=("public_plus_dp",),
            validation_scheme="LODO" if args.lodo else "GroupKFold",
        )
    else:
        with threadpool_limits(limits=1):
            summary, predictions = evaluate_grouped(
                conditions, manifest, n_splits=n_splits, seed=args.seed,
                selector_strategy="absolute_shared",
                feature_modes_override=("public_plus_dp",),
                progress=True,
            )
    summary.update({
        "seed": args.seed,
        "numeric_transform": source_config["numeric_transform"],
        "epsilons": list(epsilons),
        "repeats": repeats,
        "manifest": str(args.manifest),
        "manifest_sha256": manifest_hash,
        "implementation_sha256": source_config["implementation_sha256"],
        "analysis_implementation_sha256": analysis_hash,
        "run_source_matches_analysis": config["source_matches_current_implementation"],
        "source_conditions_sha256": config["source_conditions_sha256"],
        "attempted_datasets": len(manifest),
        "completed_datasets": int(conditions.dataset_id.nunique()),
        "failed_datasets": [],
        "exploratory_pilot": True,
        "primary_metadata_epsilon_used": 0.0,
        "metadata_epsilon_used": 0.0,
        "conditions_metadata_epsilon_used": float(
            source_config.get("metadata_epsilon_used", 0.0)
        ),
    })
    predictions.to_csv(args.output / "outer_predictions.csv", index=False)
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    (args.output / "report.md").write_text(_write_report(summary, conditions, manifest))
    print(json.dumps({
        "output": str(args.output),
        "gain": summary["primary_result"]["mean_regret_gain_vs_best_fixed"],
        "ci95": summary["primary_result"]["gain_ci95_family_bootstrap"],
        "primary_success_exploratory": summary["primary_success"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

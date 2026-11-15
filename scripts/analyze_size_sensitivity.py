"""Size-stratified selector results and held-out class coverage for small datasets."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from dpmeta.openml_benchmark import (
    _get_openml_data, _load_manifest, _openml, _seed_for, stratified_holdout_indices,
)
from dpmeta.portfolio_benchmark import _bootstrap_ci
from dpmeta.provenance import file_sha256, runtime_fingerprint


SIZE_BINS = (0, 30, 100, 200, 1000, 10000, np.inf)
SIZE_LABELS = ("<30", "30–99", "100–199", "200–999", "1.000–9.999", "≥10.000")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path,
                        default=Path("data/openml_manifest_984_completed.csv"))
    parser.add_argument("--predictions", type=Path,
                        default=Path("results/portfolio-984-shared-selector/outer_predictions.csv"))
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/openml"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/portfolio-984-shared-selector"))
    parser.add_argument("--coverage-max-rows", type=int, default=200)
    from dpmeta import selector_portfolio13 as sel
    parser.add_argument('--selector', choices=sel.SELECTORS, default='rf12')
    parser.add_argument('--conditions', type=Path, default=Path('results/portfolio-984-current-slim/conditions.csv'))
    args = parser.parse_args(argv)
    if args.selector == 'gbm_residual_fallback':
        if args.output == Path('results/portfolio-984-shared-selector'):args.output=Path('results/dissertation-portfolio13/size')
        if args.predictions == Path('results/portfolio-984-shared-selector/outer_predictions.csv'):
            args.predictions=Path('results/dissertation-portfolio13/outer_predictions.csv')
        from scripts.dissertation_analysis_portfolio13 import size_analysis
        size_analysis(args.predictions, args.manifest, args.output, args.conditions)
        return 0

    config = json.loads((args.predictions.parent / "run_config.json").read_text())
    if file_sha256(args.manifest) != config["source_manifest_sha256"]:
        raise ValueError("manifesto difere do da análise do seletor")
    manifest = _load_manifest(args.manifest)
    predictions = pd.read_csv(args.predictions)
    predictions = predictions[predictions.feature_mode == "public_plus_dp"]
    per_dataset = (predictions.groupby("dataset_id", as_index=False)
                   .agg(selector_regret=("selector_regret", "mean"),
                        fixed_regret=("best_fixed_regret", "mean"),
                        selector_ba=("selector_ba", "mean")))
    per_dataset["gain"] = per_dataset.fixed_regret - per_dataset.selector_regret
    per_dataset = per_dataset.merge(
        manifest[["dataset_id", "family_id", "n_instances", "n_classes"]],
        on="dataset_id", validate="one_to_one",
    )
    per_dataset["size_band"] = pd.cut(
        per_dataset.n_instances, bins=SIZE_BINS, labels=SIZE_LABELS,
        right=False,
    )
    rows = []
    for index, (band, block) in enumerate(per_dataset.groupby("size_band", observed=True)):
        gain, lower, upper = _bootstrap_ci(block.gain.to_numpy(), 20260923 + index)
        rows.append({
            "size_band": str(band), "n_datasets": len(block),
            "mean_gain": gain, "gain_ci95_low": lower, "gain_ci95_high": upper,
            "median_gain": float(block.gain.median()),
            "mean_selector_regret": float(block.selector_regret.mean()),
            "mean_fixed_regret": float(block.fixed_regret.mean()),
            "median_n_instances": float(block.n_instances.median()),
        })
    size_summary = pd.DataFrame(rows)
    args.output.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(Path(__file__), args.output / "sensitivity_source.py")
    (args.output / "sensitivity_config.json").write_text(json.dumps({
        "manifest_sha256": file_sha256(args.manifest),
        "predictions_sha256": file_sha256(args.predictions),
        "script_sha256": file_sha256(Path(__file__)),
        "coverage_max_rows": args.coverage_max_rows,
        "runtime": runtime_fingerprint(),
    }, indent=2, sort_keys=True) + "\n")
    size_summary.to_csv(args.output / "sensitivity_by_size.csv", index=False)
    per_dataset.to_csv(args.output / "sensitivity_per_dataset.csv", index=False)

    small = manifest[manifest.n_instances < args.coverage_max_rows]
    openml = _openml(args.cache_dir)
    coverage_rows = []
    failures = []
    for _, row in small.iterrows():
        try:
            _, y, schema = _get_openml_data(openml, row)
            all_classes = len(np.unique(y))
            for repetition in range(int(config["repeats"])):
                train, test = stratified_holdout_indices(
                    y, _seed_for(int(config["source_split_seed"]), int(row.dataset_id),
                                 repetition, "portfolio_split"),
                )
                test_classes = len(np.unique(y[test]))
                coverage_rows.append({
                    "dataset_id": int(row.dataset_id),
                    "n_instances": int(row.n_instances),
                    "repetition": repetition,
                    "declared_classes": len(schema.classes),
                    "observed_classes": all_classes,
                    "train_classes": len(np.unique(y[train])),
                    "test_classes": test_classes,
                    "test_class_coverage": test_classes / all_classes,
                    "test_n": len(test),
                })
        except Exception as error:
            failures.append({"dataset_id": int(row.dataset_id),
                             "error": f"{type(error).__name__}: {error}"})
    coverage = pd.DataFrame(coverage_rows)
    coverage.to_csv(args.output / "small_dataset_coverage.csv", index=False)
    pd.DataFrame(failures, columns=("dataset_id", "error")).to_csv(
        args.output / "small_dataset_coverage_failures.csv", index=False,
    )
    incomplete = (int((coverage.test_class_coverage < 1).sum())
                  if not coverage.empty else 0)
    table = [
        "| Instâncias | Datasets | Ganho médio | IC95% | Regret seletor | Regret fixo |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in size_summary.itertuples(index=False):
        table.append(
            f"| {row.size_band} | {row.n_datasets} | {row.mean_gain:.4f} | "
            f"[{row.gain_ci95_low:.4f}, {row.gain_ci95_high:.4f}] | "
            f"{row.mean_selector_regret:.4f} | {row.mean_fixed_regret:.4f} |"
        )
    lines = [
        "# Sensibilidade ao tamanho do dataset",
        "",
        "Análise exploratória do seletor compartilhado, agregada primeiro por dataset.",
        "Intervalos bootstrap de 95% reamostram datasets dentro de cada faixa.",
        "",
        *table,
        "",
        f"Cobertura de classes no teste: {len(coverage)} splits de "
        f"{coverage.dataset_id.nunique() if not coverage.empty else 0} datasets com "
        f"menos de {args.coverage_max_rows} linhas; {incomplete} splits não contêm "
        "todas as classes observadas no dataset.",
        f"Falhas na leitura/reconstrução: {len(failures)} datasets.",
        "",
        "A análise usa as predições já calculadas; não refaz treinamento privado.",
    ]
    (args.output / "sensitivity_report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

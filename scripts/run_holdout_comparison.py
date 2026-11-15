"""Executa o holdout de 30 famílias com a grade congelada e os baselines da emenda
secundária (`CONFIRMATORY_PROTOCOL_AMENDMENT_SECONDARY.md`).

Por dataset grava ``shards/<id>.csv`` (condições, igual a ``run_confirmation``)
e ``shards/<id>_baselines.csv``; datasets já gravados são pulados na retomada.
Nenhuma métrica é agregada aqui; a análise fica em ``analyze_holdout.py``.

    OMP_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python scripts/run_holdout_comparison.py
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

import dpmeta.portfolio_benchmark as pb
from dpmeta.external_confirmation import CONFIRMATORY_SEED, _hash, confirmation_family
from dpmeta.holdout_baselines import run_baselines
from dpmeta.meta_features import DP_META_COLUMNS
from dpmeta.openml_benchmark import (
    DEFAULT_CACHE, _get_openml_data, _load_manifest, _openml, _seed_for,
    stratified_holdout_indices,
)
from dpmeta.provenance import numerical_implementation_sha256, runtime_fingerprint

_SELECTOR = None


def _init(selector_path: str) -> None:
    global _SELECTOR
    with open(selector_path, "rb") as handle:
        _SELECTOR = pickle.load(handle)


def evaluate(row: pd.Series, out: Path, cache: Path, seed: int, repeats: int) -> str:
    dataset_id = int(row.dataset_id)
    start = time.time()
    openml = _openml(cache)
    rows = pb.run_dataset(openml, row, pb.DEFAULT_EPSILONS, repeats, seed,
                          include_research_features=False)
    conditions = pd.DataFrame(rows)
    x, y, schema = _get_openml_data(openml, row)
    best_clean = conditions.groupby("repetition").best_clean_portfolio_ba.first()
    baseline_rows = []
    for repetition in range(repeats):
        train, test = stratified_holdout_indices(
            y, _seed_for(seed, dataset_id, repetition, "portfolio_split"))
        for epsilon in pb.DEFAULT_EPSILONS:
            results = run_baselines(x[train], y[train], x[test], y[test], schema,
                                    epsilon=epsilon, seed=seed, dataset_id=dataset_id,
                                    repetition=repetition, pythia_selector=_SELECTOR,
                                    meta_row=row)
            for method, (pipeline, score) in results.items():
                baseline_rows.append({
                    "dataset_id": dataset_id, "epsilon": epsilon, "repetition": repetition,
                    "method": method, "pipeline": pipeline, "private_ba": score,
                    "best_clean_portfolio_ba": float(best_clean.loc[repetition]),
                    "portfolio_regret": float(best_clean.loc[repetition]) - score})
    # grava baselines antes das condições: a presença de <id>.csv marca conclusão
    pd.DataFrame(baseline_rows).to_csv(out / f"{dataset_id}_baselines.csv", index=False)
    conditions.to_csv(out / f"{dataset_id}.csv", index=False)
    return f"{dataset_id} {row.dataset_name}: {len(conditions)} condições, {time.time() - start:.0f}s"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path,
                        default=Path("data/openml_confirmatory_manifest_30_holdout.csv"))
    parser.add_argument("--dev-manifest", type=Path,
                        default=Path("data/openml_manifest_984_completed.csv"))
    parser.add_argument("--dev-conditions", type=Path,
                        default=Path("results/portfolio-984-current/conditions.csv"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/openml-holdout-30-comparison"))
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=CONFIRMATORY_SEED)
    parser.add_argument("--smoke", action="store_true",
                        help="teste em datasets do desenvolvimento; ignora a checagem de sobreposição")
    args = parser.parse_args(argv)
    shards = args.output / "shards"
    shards.mkdir(parents=True, exist_ok=True)
    manifest = _load_manifest(args.manifest)
    dev_manifest = _load_manifest(args.dev_manifest)
    if len({confirmation_family(n) for n in manifest.dataset_name.astype(str)}) != len(manifest):
        raise ValueError("famílias repetidas no holdout")
    if not args.smoke and set(manifest.dataset_id) & set(dev_manifest.dataset_id):
        raise ValueError("holdout sobrepõe o desenvolvimento")

    selector_path = args.output / "pythia_like_selector.pkl"
    if not selector_path.exists():
        print("ajustando o seletor dp_research_rich do desenvolvimento", flush=True)
        columns = {"dataset_id", "epsilon", "pipeline", "repetition", "private_ba", "clean_ba",
                   "best_clean_portfolio_ba", "portfolio_regret", "pipeline_privacy_penalty",
                   *DP_META_COLUMNS}
        dev = pd.read_csv(args.dev_conditions, usecols=lambda c: c in columns)
        aggregated = pb._aggregate(dev, dev_manifest)
        dp = {int(d): g.iloc[0][list(DP_META_COLUMNS)].to_numpy(dtype=float)
              for d, g in aggregated.groupby("dataset_id")}
        selector, _ = pb._fit_selector(aggregated, dev_manifest.drop_duplicates("dataset_id"),
                                       "dp_research_rich", {}, seed=20260917,
                                       dp_research_by_dataset=dp)
        selector.set_params(n_jobs=1)
        selector_path.write_bytes(pickle.dumps(selector))
    config = {
        "protocols": {p: _hash(Path(p)) for p in (
            "CONFIRMATORY_PROTOCOL.md", "CONFIRMATORY_PROTOCOL_AMENDMENT_30_CANONICAL.md",
            "CONFIRMATORY_PROTOCOL_AMENDMENT_SECONDARY.md")},
        "manifest_sha256": _hash(args.manifest), "dev_manifest_sha256": _hash(args.dev_manifest),
        "implementation_sha256": numerical_implementation_sha256(),
        "baselines_sha256": _hash(Path("dpmeta/holdout_baselines.py")),
        "seed": args.seed, "repeats": args.repeats, "epsilons": list(pb.DEFAULT_EPSILONS),
        "runtime": runtime_fingerprint(),
    }
    (args.output / "run_config.json").write_text(json.dumps(config, indent=2) + "\n")

    done = {int(p.stem) for p in shards.glob("*.csv") if p.stem.isdigit()}
    todo = manifest[~manifest.dataset_id.isin(done)].copy()
    todo["_cells"] = todo.n_instances * todo.n_encoded_features
    todo = todo.sort_values("_cells", ascending=False)  # maiores primeiro
    print(f"{len(done)} prontos, {len(todo)} a executar", flush=True)
    failures = []
    with ProcessPoolExecutor(args.workers, initializer=_init,
                             initargs=(str(selector_path),)) as pool:
        futures = {pool.submit(evaluate, row, shards, args.cache_dir, args.seed, args.repeats):
                   row for _, row in todo.iterrows()}
        for future in as_completed(futures):
            row = futures[future]
            try:
                print(future.result(), flush=True)
            except Exception as error:  # noqa: BLE001
                failures.append({"dataset_id": int(row.dataset_id),
                                 "dataset_name": row.dataset_name,
                                 "error": f"{type(error).__name__}: {error}"})
                print(f"falhou {row.dataset_name}: {failures[-1]['error']}",
                      file=sys.stderr, flush=True)
    pd.DataFrame(failures, columns=["dataset_id", "dataset_name", "error"]).to_csv(
        args.output / "failures.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

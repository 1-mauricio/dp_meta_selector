"""Parallel executor for a full 984-dataset portfolio benchmark.

Dataset work is partitioned across processes. Each worker streams rows to a
small shard so interrupted runs can merge completed dataset results and resume
without repeating work. The canonical portfolio module handles analysis.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
from pathlib import Path

import pandas as pd
from threadpoolctl import threadpool_limits

from dpmeta.meta_features import DP_META_EPSILON_DEFAULT
from dpmeta.openml_benchmark import _load_manifest, _openml
from dpmeta.portfolio_benchmark import (
    CATEGORICAL_NB_NUMERIC_BINS,
    DEFAULT_EPSILONS,
    DP_SGD_CLIP2_GRADIENT_CLIP,
    DP_SGD_CLIP2_REGULARIZATION,
    DP_SGD_GRADIENT_CLIP,
    DP_SGD_REGULARIZATION,
    DP_SGD_STEPS,
    DP_SGD_STEPS_ALTERNATE,
    FOREST_CANDIDATE_TREE_COUNT,
    FOREST_SMALL_TREE_COUNT,
    PIPELINES,
    SELECTOR_TREE_COUNT,
    SOFTMAX_OPTIMIZER_TOLERANCE,
    SOFTMAX_REGULARIZATION,
    TREE_DEPTH,
    complete_dataset_ids,
    run_dataset,
    validate_condition_grid,
)
from dpmeta.provenance import (file_sha256, numerical_implementation_sha256,
                               runtime_fingerprint, save_source_snapshot)


FAILURE_COLUMNS = ("dataset_id", "dataset_name", "error")
ROW_KEYS = ("dataset_id", "epsilon", "pipeline", "repetition")
DEFAULT_SEED = 20260917
SHARD_DIR_NAME = "_portfolio_shards"


def _config(manifest_path: Path, limit: int, repeats: int, seed: int,
            numeric_transform: str, epsilons: tuple[float, ...],
            include_research_features: bool) -> dict:
    return {
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "implementation_sha256": numerical_implementation_sha256(),
        "executor_sha256": file_sha256(Path(__file__)),
        "runtime": runtime_fingerprint(),
        "limit": int(limit),
        "epsilons": list(epsilons),
        "repeats": int(repeats),
        "seed": int(seed),
        "numeric_transform": numeric_transform,
        "worker_blas_threads": 1,
        "dataset_scheduler": "largest_estimated_work_first_dynamic_queue",
        "pipelines": list(PIPELINES),
        "tree_count": SELECTOR_TREE_COUNT,
        "selector_tree_count": SELECTOR_TREE_COUNT,
        "forest_candidate_tree_count": FOREST_CANDIDATE_TREE_COUNT,
        "forest_small_tree_count": FOREST_SMALL_TREE_COUNT,
        "tree_depth": TREE_DEPTH,
        "dp_sgd_steps": DP_SGD_STEPS,
        "dp_sgd_steps_alternate": DP_SGD_STEPS_ALTERNATE,
        "dp_sgd_gradient_clip": DP_SGD_GRADIENT_CLIP,
        "dp_sgd_regularization": DP_SGD_REGULARIZATION,
        "dp_sgd_clip2_gradient_clip": DP_SGD_CLIP2_GRADIENT_CLIP,
        "dp_sgd_clip2_regularization": DP_SGD_CLIP2_REGULARIZATION,
        "softmax_regularization": SOFTMAX_REGULARIZATION,
        "categorical_nb_numeric_bins": CATEGORICAL_NB_NUMERIC_BINS,
        "softmax_optimizer_tolerance": SOFTMAX_OPTIMIZER_TOLERANCE,
        "dp_meta_epsilon": (DP_META_EPSILON_DEFAULT if include_research_features else 0.0),
        "dp_meta_epsilon_requested": DP_META_EPSILON_DEFAULT,
        "metadata_feature_mode": ("all_available" if include_research_features
                                  else "public_plus_dp"),
        "metadata_epsilon_used": (DP_META_EPSILON_DEFAULT if include_research_features
                                  else 0.0),
        "selector_strategy": "absolute_shared",
    }


def _read_csvs(paths: list[Path]) -> list[pd.DataFrame]:
    frames = []
    for path in paths:
        if not path.exists() or path.stat().st_size == 0:
            continue
        try:
            frames.append(pd.read_csv(path))
        except (pd.errors.EmptyDataError, pd.errors.ParserError) as error:
            print(f"ignoring incomplete shard {path.name}: {error}", flush=True)
    return frames


def _worker(index: int, work_queue, result_queue, manifest_path: str,
            cache_dir: str, output_dir: str, epsilons: tuple[float, ...],
            repeats: int, seed: int, numeric_transform: str,
            dp_meta_epsilon: float, include_research_features: bool) -> None:
    manifest = _load_manifest(Path(manifest_path))
    rows_by_id = {int(row.dataset_id): row for _, row in manifest.iterrows()}
    openml = _openml(Path(cache_dir))
    shard_dir = Path(output_dir) / SHARD_DIR_NAME
    shard_dir.mkdir(parents=True, exist_ok=True)
    conditions_path = shard_dir / f"conditions_{index:02d}.csv"
    failures_path = shard_dir / f"failures_{index:02d}.csv"
    for path in (conditions_path, failures_path):
        if path.exists():
            path.unlink()

    row_count = 0
    dataset_count = 0
    failures = []
    with threadpool_limits(limits=1):
        while True:
            dataset_id = work_queue.get()
            if dataset_id is None:
                break
            row = rows_by_id[int(dataset_id)]
            dataset_count += 1
            try:
                rows = run_dataset(
                    openml, row, epsilons, repeats, seed,
                    include_research_features=include_research_features,
                    numeric_transform=numeric_transform,
                    dp_meta_epsilon=dp_meta_epsilon,
                )
                pd.DataFrame(rows).to_csv(
                    conditions_path, mode="a", header=not conditions_path.exists(), index=False,
                )
                row_count += len(rows)
                print(f"worker {index}: finished {int(row.dataset_id)} {row.dataset_name} "
                      f"({row_count} linhas neste shard)", flush=True)
            except Exception as error:
                failures.append({
                    "dataset_id": int(row.dataset_id),
                    "dataset_name": row.dataset_name,
                    "error": f"{type(error).__name__}: {error}",
                })
                pd.DataFrame(failures, columns=FAILURE_COLUMNS).to_csv(failures_path, index=False)
                print(f"worker {index}: failed {row.dataset_name}: {failures[-1]['error']}",
                      flush=True)
    if not failures_path.exists():
        pd.DataFrame([], columns=FAILURE_COLUMNS).to_csv(failures_path, index=False)
    result_queue.put({"worker": index, "datasets": dataset_count, "rows": row_count,
                      "failures": len(failures)})


def _merge_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True, sort=False)
    if all(column in combined.columns for column in ROW_KEYS):
        combined = combined.drop_duplicates(subset=list(ROW_KEYS), keep="last")
        combined = combined.sort_values(list(ROW_KEYS))
    return combined


def run(manifest_path: Path, cache_dir: Path, output_dir: Path, *, workers: int,
        repeats: int, seed: int, limit: int, numeric_transform: str = "tanh",
        epsilons: tuple[float, ...] = DEFAULT_EPSILONS,
        dp_meta_epsilon: float = DP_META_EPSILON_DEFAULT,
        include_research_features: bool = False) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = _load_manifest(manifest_path).head(limit).copy()
    if len(manifest) < limit:
        raise ValueError(f"manifest contém {len(manifest)} datasets, limite={limit}")

    config_path = output_dir / "run_config.json"
    expected_config = _config(manifest_path, limit, repeats, seed,
                              numeric_transform, epsilons, include_research_features)
    expected_config["dp_meta_epsilon"] = (float(dp_meta_epsilon) if
                                          include_research_features else 0.0)
    expected_config["dp_meta_epsilon_requested"] = float(dp_meta_epsilon)
    expected_config["metadata_epsilon_used"] = (float(dp_meta_epsilon) if
                                                include_research_features else 0.0)
    if config_path.exists():
        previous_config = json.loads(config_path.read_text())
        if previous_config != expected_config:
            raise ValueError("a configuração de resume difere; use outra pasta")
    else:
        config_path.write_text(json.dumps(expected_config, indent=2, sort_keys=True) + "\n")
        save_source_snapshot(output_dir, extra_files=("scripts/parallel_h27_round.py",))

    shard_dir = output_dir / SHARD_DIR_NAME
    shard_dir.mkdir(parents=True, exist_ok=True)
    existing_path = output_dir / "conditions.csv"
    conditions_frames = _read_csvs(
        ([existing_path] if existing_path.exists() else [])
        + sorted(shard_dir.glob("conditions_*.csv"))
    )
    existing = _merge_frames(conditions_frames)
    if not existing.empty:
        existing.to_csv(existing_path, index=False)

    failure_paths = ([output_dir / "failures.csv"]
                     if (output_dir / "failures.csv").exists() else [])
    failure_paths += sorted(shard_dir.glob("failures_*.csv"))
    failure_frames = _read_csvs(failure_paths)
    old_failures = _merge_frames(failure_frames)
    if not old_failures.empty:
        old_failures = old_failures.drop_duplicates(subset=["dataset_id"], keep="last")
        old_failures.to_csv(output_dir / "failures.csv", index=False)
    # The shard data is safely consolidated into the root files before workers
    # start, so these old fragments can be removed and reused on this attempt.
    for path in list(shard_dir.glob("conditions_*.csv")) + list(shard_dir.glob("failures_*.csv")):
        path.unlink()

    completed_ids: set[int] = set()
    if not existing.empty:
        completed_ids = complete_dataset_ids(
            existing, epsilons=epsilons, repeats=repeats,
        )
    pending_ids = [int(value) for value in manifest.dataset_id
                   if int(value) not in completed_ids]
    if not pending_ids:
        return [{"worker": -1, "datasets": 0, "rows": 0, "failures": 0}]

    manifest_by_id = {int(row.dataset_id): row for _, row in manifest.iterrows()}
    pending_ids.sort(
        key=lambda dataset_id: (
            int(manifest_by_id[dataset_id].n_instances)
            * int(manifest_by_id[dataset_id].n_encoded_features)
            * int(manifest_by_id[dataset_id].n_classes)
        ),
        reverse=True,
    )
    context = mp.get_context("spawn")
    worker_count = min(workers, len(pending_ids))
    work_queue = context.Queue()
    result_queue = context.Queue()
    for dataset_id in pending_ids:
        work_queue.put(dataset_id)
    for _ in range(worker_count):
        work_queue.put(None)
    processes = [
        context.Process(
            target=_worker,
            args=(index, work_queue, result_queue, str(manifest_path), str(cache_dir),
                  str(output_dir), tuple(epsilons), repeats, seed, numeric_transform,
                  dp_meta_epsilon, include_research_features),
        )
        for index in range(worker_count)
    ]
    queues_drained = False
    try:
        for process in processes:
            process.start()
        for process in processes:
            process.join()
        failed_workers = [process for process in processes if process.exitcode != 0]
        if failed_workers:
            codes = [process.exitcode for process in failed_workers]
            raise RuntimeError(f"workers encerraram com códigos: {codes}")
        results = [result_queue.get() for _ in processes]
        queues_drained = True
    except KeyboardInterrupt:
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            if process.pid is not None:
                process.join()
        raise
    finally:
        for queue in (work_queue, result_queue):
            if queues_drained:
                queue.close()
                queue.join_thread()
            else:
                queue.cancel_join_thread()
                queue.close()

    shard_frames = _read_csvs(sorted(shard_dir.glob("conditions_*.csv")))
    final_frames = ([existing] if not existing.empty else []) + shard_frames
    combined = _merge_frames(final_frames)
    if combined.empty:
        raise RuntimeError("nenhum dataset completou; veja failures.csv")
    validate_condition_grid(combined, manifest, epsilons=epsilons, repeats=repeats)
    combined.to_csv(existing_path, index=False)

    failure_frames = _read_csvs(sorted(shard_dir.glob("failures_*.csv")))
    if failure_frames:
        current_failures = _read_csvs(
            [output_dir / "failures.csv"] if (output_dir / "failures.csv").exists() else []
        )
        failures = _merge_frames(current_failures + failure_frames)
        if not failures.empty:
            failures = failures.drop_duplicates(subset=["dataset_id"], keep="last")
        failures.to_csv(output_dir / "failures.csv", index=False)
    elif not (output_dir / "failures.csv").exists():
        pd.DataFrame([], columns=FAILURE_COLUMNS).to_csv(output_dir / "failures.csv", index=False)
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path,
                        default=Path("data/openml_manifest_984_completed.csv"))
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/openml"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=984)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--numeric-transform", choices=("tanh", "asinh10", "public_bounds"),
                        default="tanh")
    parser.add_argument("--dp-meta-epsilon", type=float, default=DP_META_EPSILON_DEFAULT)
    parser.add_argument("--include-research-features", action="store_true",
                        help="inclui descritores não DP e release DP extra nas condições")
    parser.add_argument("--epsilons", type=lambda value: tuple(
        float(item) for item in value.split(",") if item.strip()),
        default=DEFAULT_EPSILONS)
    args = parser.parse_args()
    if args.workers < 1 or args.repeats < 1 or args.limit < 5:
        parser.error("workers/repeats positivos e limit >= 5 são necessários")
    if not args.epsilons or any(value <= 0 for value in args.epsilons):
        parser.error("epsilons devem ser positivos")
    result = run(
        args.manifest, args.cache_dir, args.output,
        workers=args.workers, repeats=args.repeats, seed=args.seed,
        limit=args.limit, numeric_transform=args.numeric_transform,
        epsilons=args.epsilons, dp_meta_epsilon=args.dp_meta_epsilon,
        include_research_features=args.include_research_features,
    )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

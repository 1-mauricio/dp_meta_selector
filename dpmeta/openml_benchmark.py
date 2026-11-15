"""Select, download, and evaluate a fixed OpenML classification benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .l2_metrics import sufficient_statistic_l2_distortion
from .model import (
    ALLOCATIONS, balanced_accuracy, model_from_statistics, predict,
    privatize_sufficient_statistics, sufficient_statistics,
)
from .meta_features import PRIVATE_META_COLUMNS, private_training_meta_features
from .report import render_validation_report
from .schema import PublicSchema
from .provenance import numerical_implementation_sha256
from .selector import ACTIONS, evaluate_grouped

DEFAULT_EPSILONS = (0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
DEFAULT_SEED = 20260917
DEFAULT_CACHE = Path(".cache/openml")


@dataclass(frozen=True)
class DatasetCandidate:
    dataset_id: int
    task_id: int | None
    target_name: str
    name: str
    version: int
    n_instances: int
    feature_names: tuple[str, ...]
    feature_kinds: tuple[str, ...]
    nominal_domains: tuple[tuple[str, ...], ...]
    classes: tuple[str, ...]

    @property
    def schema(self) -> PublicSchema:
        return PublicSchema(self.feature_names, self.feature_kinds,
                            self.nominal_domains, self.classes)

    @property
    def family_id(self) -> str:
        return canonical_family(self.name)

    def manifest_row(self, source: str) -> dict:
        public_schema = {
            "feature_names": self.feature_names,
            "feature_kinds": self.feature_kinds,
            "nominal_domains": self.nominal_domains,
            "classes": self.classes,
        }
        return {
            "dataset_id": self.dataset_id,
            "dataset_version": self.version,
            "dataset_name": self.name,
            "task_id": self.task_id if self.task_id is not None else "",
            "target_name": self.target_name,
            "family_id": self.family_id,
            "n_instances": self.n_instances,
            "n_features": len(self.feature_names),
            "n_encoded_features": self.schema.encoded_dim,
            "n_classes": len(self.classes),
            "n_nominal_features": self.schema.nominal_count,
            "selection_source": source,
            "public_schema_json": json.dumps(public_schema, ensure_ascii=False, sort_keys=True),
        }


def canonical_family(name: str) -> str:
    """Conservative grouping for known variants in the selected catalog."""
    normalized = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    if normalized.startswith("mfeat_"):
        return "mfeat"
    if normalized in {"pc1", "pc3", "pc4", "kc1", "kc2"}:
        return "nasa_software_defects"
    if normalized.startswith("adult"):
        return "adult"
    normalized = re.sub(r"_seed_[0-9]+.*$", "", normalized)
    normalized = re.sub(r"_nrows_[0-9]+.*$", "", normalized)
    normalized = re.sub(r"_variant_[a-z0-9]+.*$", "", normalized)
    normalized = re.sub(r"_v[0-9]+$", "", normalized)
    return normalized


def _openml(cache_dir: Path):
    try:
        import openml
        import requests
    except ImportError as error:
        raise RuntimeError("instale as dependências com `pip install -r requirements.txt`") from error
    openml.config.set_root_cache_directory(str(cache_dir.resolve()))
    # The canonical www endpoint currently serves metadata more reliably than
    # api.openml.org, which can return transient 429s for the same requests.
    openml.config.server = "https://www.openml.org/api/v1/xml"
    openml.config.connection_n_retries = 1
    if not getattr(requests.Session, "_dpmeta_timeout_patch", False):
        original_get = requests.Session.get

        def get_with_timeout(session, url, **kwargs):
            kwargs.setdefault("timeout", 15)
            return original_get(session, url, **kwargs)

        requests.Session.get = get_with_timeout
        requests.Session._dpmeta_timeout_patch = True
    return openml


def _split_names(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return [str(item) for item in value]


def candidate_from_dataset(openml, dataset_id: int, task_id: int | None,
                           target_name: str | None, source: str,
                           *, min_rows: int, max_rows: int, max_classes: int,
                           max_expanded_features: int) -> DatasetCandidate | None:
    dataset = openml.datasets.get_dataset(
        int(dataset_id), download_data=False, download_qualities=True,
        download_features_meta_data=True,
    )
    target_name = target_name or dataset.default_target_attribute
    if not target_name:
        return None
    features = list(dataset.features.values())
    targets = [feature for feature in features if feature.name == target_name]
    if len(targets) != 1 or targets[0].data_type != "nominal":
        return None
    target = targets[0]
    classes = tuple(str(value) for value in (target.nominal_values or []))
    if len(classes) < 2 or len(classes) > max_classes or len(set(classes)) != len(classes):
        return None
    if target.number_missing_values:
        return None

    ignored = set(_split_names(dataset.ignore_attribute))
    if dataset.row_id_attribute:
        ignored.add(str(dataset.row_id_attribute))
    feature_names, feature_kinds, domains = [], [], []
    for feature in features:
        if feature.name == target_name or feature.name in ignored:
            continue
        if feature.data_type == "numeric":
            feature_names.append(str(feature.name))
            feature_kinds.append("numeric")
            domains.append(())
        elif feature.data_type == "nominal":
            domain = tuple(str(value) for value in (feature.nominal_values or []))
            if not domain or len(domain) != len(set(domain)):
                return None
            feature_names.append(str(feature.name))
            feature_kinds.append("nominal")
            domains.append(domain)
        else:
            return None
    if not feature_names:
        return None

    qualities = dataset.qualities or {}
    try:
        n_instances = int(float(qualities["NumberOfInstances"]))
    except (KeyError, TypeError, ValueError):
        return None
    schema = PublicSchema(tuple(feature_names), tuple(feature_kinds), tuple(domains), classes)
    if not min_rows <= n_instances <= max_rows or schema.encoded_dim > max_expanded_features:
        return None
    return DatasetCandidate(
        int(dataset_id), task_id, str(target_name), str(dataset.name), int(dataset.version),
        n_instances, tuple(feature_names), tuple(feature_kinds), tuple(domains), classes,
    )


def select_candidates(*, target_count: int = 50, cache_dir: Path = DEFAULT_CACHE,
                      seed: int = DEFAULT_SEED, min_rows: int = 200,
                      max_rows: int = 5000, max_classes: int = 30,
                      max_expanded_features: int = 512, workers: int = 4,
                      initial_rows: pd.DataFrame | None = None,
                      cache_only: bool = False) -> list[dict]:
    """Take a deterministic CC18-first sample based only on OpenML metadata.

    Metadata is fetched by `workers` threads in ordered chunks; acceptance runs
    sequentially in candidate order, so the result equals the sequential one.
    """
    openml = _openml(cache_dir)
    suite_candidates: list[tuple[int, int | None, str | None, str]] = []
    accepted: list[dict] = []
    accepted_ids: set[int] = set()
    accepted_families: set[str] = set()
    seen: set[int] = set()
    if initial_rows is not None:
        for row in initial_rows.to_dict("records"):
            dataset_id = int(row["dataset_id"])
            accepted.append(row)
            accepted_ids.add(dataset_id)
            accepted_families.add(str(row["family_id"]))
            seen.add(dataset_id)
    if not cache_only:
        suite = openml.study.get_suite("OpenML-CC18")
        if not suite.data or len(suite.data) != len(suite.tasks):
            raise RuntimeError("CC18 não informou pares alinhados de dataset e task IDs")
        for task_id, data_id in zip(suite.tasks, suite.data):
            dataset_id = int(data_id)
            if dataset_id in seen:
                continue
            seen.add(dataset_id)
            suite_candidates.append((dataset_id, int(task_id), None, "OpenML-CC18"))

    def fetch(item):
        dataset_id, task_id, target, source = item
        for attempt in range(5):
            try:
                return candidate_from_dataset(
                    openml, dataset_id, task_id, target, source, min_rows=min_rows,
                    max_rows=max_rows, max_classes=max_classes,
                    max_expanded_features=max_expanded_features,
                ), None
            except Exception as error:
                # Server overload looks like a permanent failure; retry before skipping.
                error_text = str(error).lower()
                if ("429" in error_text or "no features found" in error_text
                        or "timed out" in error_text or "timeout" in error_text):
                    return None, error
                if attempt == 4:
                    return None, error
                time.sleep(2 ** attempt)

    def consider(candidates: list[tuple[int, int | None, str | None, str]], offset: int) -> int:
        candidates = [item for item in candidates if item[0] not in accepted_ids]
        chunk = max(workers * 4, 1)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for start in range(0, len(candidates), chunk):
                if len(accepted) >= target_count:
                    break
                batch = candidates[start:start + chunk]
                for index, (item, (candidate, error)) in enumerate(
                        zip(batch, pool.map(fetch, batch)), start=start + 1):
                    if len(accepted) >= target_count:
                        break
                    if error is not None:
                        print(f"skip dataset {item[0]}: {error}", file=sys.stderr)
                        continue
                    source = item[3]
                    if candidate is None or candidate.dataset_id in accepted_ids:
                        continue
                    if candidate.family_id in accepted_families:
                        continue
                    accepted.append(candidate.manifest_row(source))
                    accepted_ids.add(candidate.dataset_id)
                    accepted_families.add(candidate.family_id)
                    if len(accepted) % 10 == 0:
                        print(f"selected {len(accepted)}/{target_count} "
                              f"at candidate {offset + index}", flush=True)
        return offset + len(candidates)

    cached_ids: set[int] | None = None
    if cache_only:
        cached_root = cache_dir / "org" / "openml" / "www" / "datasets"
        cached_ids = {
            int(path.name)
            for path in cached_root.iterdir()
            if path.is_dir() and path.name.isdigit()
            and all((path / filename).exists()
                    for filename in ("description.xml", "features.xml", "qualities.xml"))
        } if cached_root.exists() else set()
        suite_candidates = [item for item in suite_candidates if item[0] in cached_ids]
    print(f"checking {len(suite_candidates)} OpenML-CC18 dataset records", flush=True)
    next_offset = consider(suite_candidates, 0)
    if cache_only and len(accepted) < target_count:
        cached = []
        for dataset_id in sorted(cached_ids or ()):
            if dataset_id not in seen:
                seen.add(dataset_id)
                cached.append((dataset_id, None, None, "OpenML-cache"))
        print(f"checking {len(cached)} cached OpenML metadata records", flush=True)
        cached_candidates_by_family = {}
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for item, (candidate, error) in zip(cached, pool.map(fetch, cached)):
                if error is not None:
                    continue
                if candidate is not None and candidate.dataset_id not in accepted_ids:
                    current = cached_candidates_by_family.get(candidate.family_id)
                    if current is None or (
                        candidate.n_instances, candidate.schema.encoded_dim, candidate.dataset_id
                    ) < (
                        current.n_instances, current.schema.encoded_dim, current.dataset_id
                    ):
                        cached_candidates_by_family[candidate.family_id] = candidate
        # Prefer one compact representative per family so the expanded
        # benchmark remains computationally tractable and deterministic.
        cached_candidates = list(cached_candidates_by_family.values())
        cached_candidates.sort(key=lambda candidate: (
            candidate.n_instances, candidate.schema.encoded_dim, candidate.dataset_id,
        ))
        for candidate in cached_candidates:
            if len(accepted) >= target_count:
                break
            if candidate.family_id in accepted_families:
                continue
            accepted.append(candidate.manifest_row("OpenML-cache"))
            accepted_ids.add(candidate.dataset_id)
            accepted_families.add(candidate.family_id)
            if len(accepted) % 10 == 0:
                print(f"selected {len(accepted)}/{target_count} from cached metadata", flush=True)
    elif len(accepted) < target_count:
        catalog = openml.datasets.list_datasets(status="active", output_format="dataframe")
        id_column = next((col for col in catalog.columns if col.lower() in {"did", "dataid", "dataset_id"}), None)
        if id_column is None:
            raise RuntimeError(f"coluna de ID não encontrada no catálogo OpenML: {list(catalog.columns)}")
        # The active catalog is much larger than the eligible pool.  Apply
        # cheap metadata filters before fetching each dataset's schema; the
        # exact encoded-feature check still happens in candidate_from_dataset.
        instances_column = next(
            (col for col in catalog.columns if col.lower() == "numberofinstances"),
            None,
        )
        classes_column = next(
            (col for col in catalog.columns if col.lower() == "numberofclasses"),
            None,
        )
        if instances_column is not None:
            catalog[instances_column] = pd.to_numeric(catalog[instances_column], errors="coerce")
        if classes_column is not None:
            catalog[classes_column] = pd.to_numeric(catalog[classes_column], errors="coerce")
        catalog = catalog.sort_values(id_column, kind="stable")
        if instances_column is not None and classes_column is not None:
            likely = (
                catalog[instances_column].between(min_rows, max_rows)
                & catalog[classes_column].between(2, max_classes)
            )
            # Try the likely pool first, but retain all remaining records as
            # a fallback: catalog summaries can be missing or stale while
            # the full dataset metadata still passes the exact checks.
            catalog = pd.concat(
                [catalog.loc[likely], catalog.loc[~likely]],
                ignore_index=True,
            )
        extra = []
        for _, catalog_row in catalog.iterrows():
            dataset_id = int(catalog_row[id_column])
            if dataset_id not in seen:
                seen.add(dataset_id)
                extra.append((dataset_id, None, None, "OpenML-active-catalog"))
            if len(extra) >= target_count * 30:
                break
        print(f"checking {len(extra)} additional OpenML metadata records", flush=True)
        consider(extra, next_offset)

    if len(accepted) < target_count:
        raise RuntimeError(
            f"encontradas apenas {len(accepted)} famílias elegíveis para o alvo {target_count}; "
            "reduza filtros ou aumente o catálogo explicitamente"
        )
    return accepted


def _load_manifest(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, keep_default_na=False)
    required = {"dataset_id", "dataset_name", "family_id", "n_instances", "public_schema_json"}
    if required - set(frame.columns):
        raise ValueError(f"manifest incompleto: {sorted(required - set(frame.columns))}")
    for row in frame.itertuples(index=False):
        schema = PublicSchema.from_manifest_json(row.public_schema_json)
        if schema.encoded_dim != int(row.n_encoded_features):
            raise ValueError(f"schema e dimensão divergem para dataset {row.dataset_id}")
    return frame


def sparse_nominal_default(values: pd.Series, declared: list[str]) -> pd.Series:
    """No ARFF esparso, o valor omitido de um nominal é o primeiro valor declarado.

    O leitor devolve 0 nesses casos; sem esta correção, bases libSVM com alvo
    {1, -1} falhavam com o rótulo '0' fora do schema público.
    """
    if not declared or "0" in {str(v) for v in declared}:
        return values
    zero = {"0", "0.0"}
    return values.astype(object).map(lambda v: declared[0] if str(v) in zero else v)


def _get_openml_data(openml, row: pd.Series, *, numeric_transform: str = "tanh"):
    schema = PublicSchema.from_manifest_json(row.public_schema_json)
    local_data_path = getattr(row, "local_data_path", "")
    if pd.isna(local_data_path):
        local_data_path = ""
    if str(local_data_path).strip():
        path = Path(str(local_data_path))
        if not path.is_absolute():
            path = Path(__file__).resolve().parent.parent / path
        frame = pd.read_parquet(path)
        target_name = str(row.target_name)
        if target_name not in frame.columns:
            raise ValueError("alvo local não encontrado no parquet fixado no manifesto")
        if any(name not in frame.columns for name in schema.feature_names):
            raise ValueError("atributos locais diferem do schema fixado no manifesto")
        x_frame = frame.loc[:, list(schema.feature_names)]
        y = frame[target_name]
    else:
        dataset = openml.datasets.get_dataset(
            int(row.dataset_id), download_data=True, download_qualities=False,
            download_features_meta_data=True,
        )
        x_frame, y, _, names = dataset.get_data(target=str(row.target_name))
        if str(getattr(dataset, "format", "")).lower().startswith("sparse") and y is not None:
            declared = next(feature.nominal_values for feature in dataset.features.values()
                            if feature.name == str(row.target_name))
            y = sparse_nominal_default(y, declared)
        if tuple(str(name) for name in names) != schema.feature_names:
            raise ValueError("atributos baixados diferem do schema fixado no manifesto")
    if len(x_frame) != int(row.n_instances):
        raise ValueError("n de linhas baixadas difere do tamanho público fixado no manifesto")
    if y is None or y.isna().any():
        raise ValueError("alvo nulo ou ausente; o protocolo não remove linhas por valor privado")
    x = schema.encode(x_frame, numeric_transform=numeric_transform)
    labels = schema.class_indices(y.to_numpy(dtype=object))
    return x, labels, schema


def _seed_for(*parts) -> int:
    digest = hashlib.blake2b(repr(tuple(parts)).encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little") % (2**32 - 1)


def stratified_holdout_indices(y: np.ndarray, seed: int, test_fraction: float = 0.25):
    """Stable 75/25 split; singleton classes stay in train, never in test."""
    y = np.asarray(y, dtype=int)
    if np.min(np.unique(y, return_counts=True)[1]) >= 2:
        return train_test_split(
            np.arange(len(y)), test_size=test_fraction, random_state=seed, stratify=y,
        )
    rng = np.random.default_rng(seed)
    train_parts, test_parts = [], []
    for klass in np.unique(y):
        members = np.flatnonzero(y == klass).copy()
        rng.shuffle(members)
        if len(members) == 1:
            n_test = 0
        else:
            n_test = max(1, int(round(len(members) * test_fraction)))
            n_test = min(n_test, len(members) - 1)
        test_parts.append(members[:n_test])
        train_parts.append(members[n_test:])
    train = np.concatenate(train_parts)
    test = np.concatenate(test_parts)
    if len(test) == 0:
        raise ValueError("split sem exemplos de teste")
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def run_dataset(openml, row: pd.Series, epsilons: tuple[float, ...], repeats: int,
                seed: int) -> list[dict]:
    x, y, schema = _get_openml_data(openml, row)
    class_counts = np.bincount(y, minlength=len(schema.classes))
    if np.count_nonzero(class_counts) < 2:
        raise ValueError("menos de duas classes observadas")
    rows: list[dict] = []
    for repetition in range(repeats):
        train_indices, test_indices = stratified_holdout_indices(
            y, _seed_for(seed, int(row.dataset_id), repetition, "split"),
        )
        x_train, x_test = x[train_indices], x[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]
        # One deployment-like split per dataset supplies privileged diagnostics;
        # they are constant across candidate mechanisms and are not averaged
        # across repeated hypothetical deployments.
        private_meta = (dict(zip(
            PRIVATE_META_COLUMNS,
            private_training_meta_features(x_train, y_train, len(schema.classes)),
        )) if repetition == 0 else {})
        clean_stats = sufficient_statistics(x_train, y_train, len(schema.classes))
        clean_model = model_from_statistics(clean_stats)
        clean_ba = balanced_accuracy(y_test, predict(x_test, clean_model))
        for epsilon in epsilons:
            for mechanism, allocation in ACTIONS:
                released_stats, ledger = privatize_sufficient_statistics(
                    clean_stats, epsilon=epsilon,
                    mechanism=mechanism, allocation=allocation,
                    seed=_seed_for(seed, int(row.dataset_id), repetition, epsilon, mechanism, allocation),
                )
                model = model_from_statistics(released_stats)
                private_ba = balanced_accuracy(y_test, predict(x_test, model))
                l2_distortion = sufficient_statistic_l2_distortion(clean_stats, released_stats)
                rows.append({
                    "dataset_id": int(row.dataset_id), "family_id": str(row.family_id),
                    "epsilon": float(epsilon), "mechanism": mechanism,
                    "allocation": allocation, "repetition": repetition,
                    "clean_ba": clean_ba, "private_ba": private_ba,
                    "regret": clean_ba - private_ba,
                    "l2_distortion": l2_distortion,
                    **private_meta,
                    "ledger_epsilon": ledger.epsilon_used,
                    "ledger_delta": ledger.delta_used,
                    "ledger_json": json.dumps(ledger.as_dict(), sort_keys=True),
                    "dataset_n": len(y), "encoded_dim": schema.encoded_dim,
                })
    return rows


def _write_csv(rows: list[dict], path: Path) -> None:
    frame = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _implementation_sha256() -> str:
    return numerical_implementation_sha256()


def _parse_epsilon(raw: str) -> tuple[float, ...]:
    values = tuple(float(part) for part in raw.split(",") if part.strip())
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("epsilons devem ser uma lista positiva separada por vírgulas")
    return values


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/openml_manifest.csv"))
    parser.add_argument("--base-manifest", type=Path,
                        help="manifesto existente a preservar ao ampliar a seleção")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--select-only", action="store_true")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--epsilons", type=_parse_epsilon,
                        default=DEFAULT_EPSILONS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", type=Path, default=Path("results/openml-validation-expanded"))
    parser.add_argument("--protocol", type=Path,
                        help="protocolo existente; ausente indica análise exploratória")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--min-rows", type=int, default=200)
    parser.add_argument("--max-rows", type=int, default=5000)
    parser.add_argument("--max-classes", type=int, default=30)
    parser.add_argument("--max-expanded-features", type=int, default=512)
    parser.add_argument("--workers", type=int, default=4,
                        help="threads para buscar metadados do OpenML na seleção")
    parser.add_argument("--cache-only", action="store_true",
                        help="seleciona apenas datasets com metadados já presentes no cache")
    args = parser.parse_args(argv)
    if args.limit < 5 or args.repeats < 1:
        parser.error("--limit precisa ser >= 5 e --repeats >= 1")
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    if not args.manifest.exists() or args.overwrite:
        initial_rows = None
        if args.base_manifest is not None:
            if not args.base_manifest.exists():
                parser.error(f"manifesto base não encontrado: {args.base_manifest}")
            initial_rows = _load_manifest(args.base_manifest)
            if len(initial_rows) > args.limit:
                parser.error("--base-manifest não pode ter mais linhas que --limit")
        rows = select_candidates(
            target_count=args.limit, cache_dir=args.cache_dir, seed=args.seed,
            min_rows=args.min_rows, max_rows=args.max_rows,
            max_classes=args.max_classes,
            max_expanded_features=args.max_expanded_features, workers=args.workers,
            initial_rows=initial_rows, cache_only=args.cache_only,
        )
        _write_csv(rows, args.manifest)
    full_manifest = _load_manifest(args.manifest)
    if len(full_manifest) < args.limit:
        raise ValueError(f"manifest contém {len(full_manifest)} tarefas, mas --limit={args.limit}")
    canonical_families = full_manifest.dataset_name.map(canonical_family)
    if not np.array_equal(full_manifest.family_id.to_numpy(dtype=str), canonical_families.to_numpy(dtype=str)):
        full_manifest["family_id"] = canonical_families
        # Family aliases are part of split safety, so persist this deterministic correction.
        full_manifest.to_csv(args.manifest, index=False)
    manifest = full_manifest.head(args.limit).copy()
    print(f"manifest: {args.manifest} ({len(manifest)} datasets, "
          f"{manifest.family_id.nunique()} famílias)")
    if args.select_only:
        return 0

    args.output.mkdir(parents=True, exist_ok=True)
    results_path = args.output / "conditions.csv"
    failures_path = args.output / "failures.csv"
    summary_path = args.output / "summary.json"
    predictions_path = args.output / "outer_predictions.csv"
    l2_predictions_path = args.output / "outer_predictions_l2.csv"
    report_path = args.output / "report.md"
    config_path = args.output / "run_config.json"
    manifest_hash = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    if args.protocol is not None and not args.protocol.is_file():
        parser.error(f"protocolo não encontrado: {args.protocol}")
    protocol_hash = _sha256(args.protocol) if args.protocol else None
    implementation_hash = _implementation_sha256()
    config = {
        "manifest_sha256": manifest_hash,
        "protocol_sha256": protocol_hash,
        "implementation_sha256": implementation_hash,
        "limit": args.limit,
        "epsilons": list(args.epsilons),
        "repeats": args.repeats,
        "seed": args.seed,
        "secondary_target": "relative_l2_regret_of_sufficient_statistics",
        "l2_metric": "RMS of block L2 errors normalized by replace-one L2 sensitivity",
    }
    if results_path.exists() and not args.resume and not args.overwrite:
        raise FileExistsError(f"{results_path} existe; escolha --resume ou --overwrite")
    if args.overwrite:
        for path in (results_path, failures_path, summary_path, predictions_path,
                     l2_predictions_path, report_path, config_path):
            if path.exists():
                path.unlink()
    if args.resume and config_path.exists():
        previous = json.loads(config_path.read_text())
        if previous != config:
            raise ValueError("a configuração de resume difere; use outra pasta ou --overwrite")
    elif results_path.exists() and not args.resume:
        raise FileExistsError(f"{results_path} existe; use --resume ou --overwrite")
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")

    existing = pd.read_csv(results_path) if args.resume and results_path.exists() else pd.DataFrame()
    completed = set(existing.dataset_id.unique().astype(int)) if len(existing) else set()
    openml = _openml(args.cache_dir)
    failures: list[dict] = []
    if args.resume and failures_path.exists():
        try:
            failures = pd.read_csv(failures_path).to_dict("records")
        except pd.errors.EmptyDataError:
            failures = []
    for _, row in manifest.iterrows():
        dataset_id = int(row.dataset_id)
        if dataset_id in completed:
            continue
        try:
            rows = run_dataset(openml, row, args.epsilons, args.repeats, args.seed)
            pd.DataFrame(rows).to_csv(results_path, mode="a", header=not results_path.exists(), index=False)
            failures = [failure for failure in failures if int(failure["dataset_id"]) != dataset_id]
            print(f"finished {dataset_id} {row.dataset_name}: {len(rows)} observations")
        except Exception as error:
            failure = {"dataset_id": dataset_id, "dataset_name": row.dataset_name,
                       "error": f"{type(error).__name__}: {error}"}
            failures.append(failure)
            print(f"failed {dataset_id} {row.dataset_name}: {failure['error']}", file=sys.stderr)
        pd.DataFrame(failures, columns=["dataset_id", "dataset_name", "error"]).to_csv(
            failures_path, index=False,
        )

    if not results_path.exists():
        raise RuntimeError("nenhum dataset foi avaliado; veja failures.csv")
    results = pd.read_csv(results_path)
    used_ids = set(results.dataset_id.astype(int).unique())
    used_manifest = manifest[manifest.dataset_id.astype(int).isin(used_ids)].copy()
    if used_manifest.family_id.nunique() < 5:
        raise RuntimeError("menos de cinco famílias completaram a avaliação")
    summary, predictions = evaluate_grouped(results, used_manifest, seed=args.seed)
    l2_summary, l2_predictions = evaluate_grouped(
        results, used_manifest, seed=args.seed, target_column="l2_regret",
    )
    summary["sufficient_statistic_l2_selector"] = l2_summary
    summary.update({
        "protocol": str(args.protocol) if args.protocol else None,
        "seed": args.seed,
        "epsilons": list(args.epsilons),
        "repeats": args.repeats,
        "manifest": str(args.manifest),
        "manifest_sha256": manifest_hash,
        "protocol_sha256": protocol_hash,
        "implementation_sha256": implementation_hash,
        "attempted_datasets": int(len(manifest)),
        "completed_datasets": int(len(used_ids)),
        "failed_datasets": failures,
        "delta_gaussian": 1e-6,
    })
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    predictions.to_csv(predictions_path, index=False)
    l2_predictions.to_csv(l2_predictions_path, index=False)
    report_path.write_text(render_validation_report(summary, results, used_manifest, predictions))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

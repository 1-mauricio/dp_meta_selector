"""Frozen transfer test of the DP-classifier selector on unseen OpenML families."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .openml_benchmark import (
    DEFAULT_CACHE,
    _load_manifest,
    _openml,
    _parse_epsilon,
    _seed_for,
)
from .portfolio_benchmark import (
    DEFAULT_EPSILONS,
    H27_PIPELINE,
    PIPELINE_DISPLAY,
    PIPELINES,
    SELECTOR_TREE_COUNT,
    _aggregate,
    _best_fixed,
    _bootstrap_ci,
    _candidate_features,
    _condition_score,
    _fit_selector,
    _metric_summary,
    _sign_flip_p,
    run_dataset,
)
from .openml_benchmark import candidate_from_dataset, canonical_family
from .provenance import numerical_implementation_sha256

DEFAULT_OLD_MANIFEST = Path("data/openml_manifest_984_completed.csv")
DEFAULT_OLD_RESULTS = Path("results/portfolio-984-current/conditions.csv")
DEFAULT_NEW_MANIFEST = Path("data/openml_confirmatory_manifest_12_options.csv")
DEFAULT_OUTPUT = Path("results/openml-confirmatory-12-options-2026-09-23-family-deduped")
DEFAULT_PROTOCOL = Path("CONFIRMATORY_PROTOCOL.md")
CONFIRMATORY_SEED = 20260918
MIN_CONFIRMATORY_FAMILIES = 40
CONFIRMATORY_FAMILY_ALIASES = {
    "australian": "australian_credit_approval",
    "credit_approval_classification": "australian_credit_approval",
    "crx": "australian_credit_approval",
    "credit_approval": "australian_credit_approval",
    "credit": "credit_data_variants",
    "credit_g": "credit_data_variants",
    "credit_g_copy": "credit_data_variants",
    "dataset_credit_g": "credit_data_variants",
    "cars1": "car",
    "vehicle_reproduced": "vehicle",
    "glass_clean": "glass",
    "fetal_cardiotocography_dataset": "cardiotocography",
    "agaricus_lepiota": "mushroom",
    "mushroom": "mushroom",
    "diabetes": "pima_diabetes",
    "pima": "pima_diabetes",
    "eegeye_state": "eeg_eye_state",
    "eeg_eye_state": "eeg_eye_state",
}


def confirmation_family(name: str) -> str:
    """Conservative aliases for known parent datasets and generated variants."""
    # OpenML's BNG tasks are synthetic samples generated from a named source
    # dataset. Treat every BNG size/seed variant as that source family so a
    # generated copy cannot leak across development and holdout.
    bng_source = re.match(r"^\s*bng\s*\(\s*([^,)]+)", str(name), flags=re.IGNORECASE)
    if bng_source:
        return confirmation_family(bng_source.group(1).strip())
    normalized = canonical_family(name)
    if normalized.startswith("fri_c"):
        return "friedman_synthetic"
    if normalized.startswith("agrawal"):
        return "agrawal_synthetic"
    if normalized.startswith("stagger"):
        return "stagger_synthetic"
    if normalized.startswith("randomrbf"):
        return "random_rbf_synthetic"
    if normalized.startswith("hyperplane"):
        return "hyperplane_synthetic"
    if normalized.startswith("sea_") or normalized == "sea":
        return "sea_synthetic"
    if normalized.startswith("anneal"):
        return "anneal"
    if normalized.startswith("led"):
        return "led_synthetic"
    if normalized in {"credit_a", "credit_approval_australia"}:
        return "australian_credit_approval"
    if normalized == "higgs" or normalized.startswith("atlas_higgs"):
        return "atlas_higgs"
    if normalized.startswith("afro_mnist"):
        return "afro_mnist"
    if normalized.startswith("20_newsgroups"):
        return "20_newsgroups"
    if normalized.startswith("eda_home_mortgage_ny"):
        return "eda_home_mortgage_ny"
    if normalized.startswith("codrna"):
        return "codrna"
    if normalized.startswith("kuzushiji"):
        return "kuzushiji"
    if normalized.startswith("bridges_version"):
        return "bridges"
    if normalized == "iris_reproduced":
        return "iris"
    if normalized in {"forest_cover", "covertype", "covtype"}:
        return "covertype"
    if normalized in {"vehiclenorm", "vehicle_norm", "vehicle_reproduced"}:
        return "vehicle"
    if normalized in {"spect_test", "spectf"} or normalized.startswith("spectf_"):
        return "spectf"
    if re.fullmatch(r"(?:pc[1-5]|kc[1-4])", normalized):
        return "nasa_software_defects"
    if normalized in {"crx", "australian", "credit_approval",
                      "credit_approval_classification"}:
        return "australian_credit_approval"
    if (normalized == "glass2" or normalized == "prnn_fglass"
            or normalized == "glass_clean" or normalized.startswith("glass_")):
        return "glass"
    if normalized in {"satimage", "satellite", "satellite_image"}:
        return "satellite_image"
    if normalized == "gtsrb" or normalized.startswith("gtsrb_"):
        return "gtsrb"
    if (normalized in {"cifar_10", "cifar10", "cifar_100", "cifar100"}
            or normalized.startswith("cifar_10_small")
            or normalized.startswith("cifar10_small")):
        return "cifar"
    if normalized == "svhn" or normalized.startswith("svhn_"):
        return "svhn"
    if normalized == "shuttle" or normalized.startswith("shuttle_"):
        return "shuttle"
    if normalized.startswith("ml2017_challenge_"):
        return "ml2017_challenge"
    if normalized in {"cal_housing", "california_housing"}:
        return "california_housing"
    if normalized.startswith("thyroid_") or normalized in {
        "thyroid", "hypothyroid", "annthyroid", "sick", "sick_numeric",
    }:
        return "thyroid_disease"
    if normalized in {"ada", "ada_prior", "ada_agnostic"}:
        return "ada"
    if normalized in {"sylva", "sylva_prior", "sylva_agnostic"}:
        return "sylva"
    if normalized in {"sami_dataset", "sami_datasetagain"}:
        return "sami_dataset"
    if re.fullmatch(r"pie_?chart_?[0-9]+", normalized):
        return "piechart"
    if normalized in {"pen_local", "pen_global", "pendigits"}:
        return "pen_digits"
    if normalized.startswith("data_v3_"):
        return "data_v3"
    if normalized.startswith("ex") and "_features_holiday_" in normalized:
        return "holiday_person_event"
    if normalized == "xerces_init":
        return "promise_software_xerces"
    if normalized == "log4j" or normalized.startswith("log4j_"):
        return "promise_software_log4j"
    promise_projects = {
        "ant", "camel", "ivy", "jedit", "lucene", "poi", "prop", "synapse",
        "velocity", "xalan", "xerces", "tomcat",
    }
    match = re.fullmatch(r"([a-z]+)(?:_([0-9]+)(?:_([0-9]+))?)?", normalized)
    if match and match.group(1) in promise_projects:
        return f"promise_software_{match.group(1)}"
    return CONFIRMATORY_FAMILY_ALIASES.get(normalized, normalized)


def _id_column(frame: pd.DataFrame) -> str:
    column = next((name for name in frame.columns
                   if str(name).lower() in {"did", "dataid", "dataset_id"}), None)
    if column is None:
        raise RuntimeError(f"coluna de ID ausente no catálogo OpenML: {list(frame.columns)}")
    return column


def _metadata_prefilter(frame: pd.DataFrame, *, min_rows: int = 200,
                        max_rows: int = 5000, max_classes: int = 30,
                        max_catalog_features: int = 512) -> pd.DataFrame:
    """Apply public catalog filters before fetching detailed dataset metadata."""
    lower_map = {str(name).lower(): name for name in frame.columns}
    keep = pd.Series(True, index=frame.index)
    filters = (
        ("numberofinstances", min_rows, max_rows),
        ("numberofclasses", 2, max_classes),
        ("numberoffeatures", 1, max_catalog_features),
    )
    for key, lower, upper in filters:
        column = lower_map.get(key)
        if column is not None:
            values = pd.to_numeric(frame[column], errors="coerce")
            keep &= values.between(lower, upper, inclusive="both")
    return frame.loc[keep].copy()


def _manifest_identity(manifest_paths: tuple[Path, ...]) -> tuple[set[int], set[str]]:
    """Return IDs and both stored/canonical family labels for exclusion manifests."""
    excluded_ids: set[int] = set()
    excluded_families: set[str] = set()
    for path in manifest_paths:
        frame = _load_manifest(path)
        excluded_ids.update(frame.dataset_id.astype(int))
        for value in frame.family_id.astype(str):
            excluded_families.add(value)
            excluded_families.add(confirmation_family(value))
        excluded_families.update(
            confirmation_family(name) for name in frame.dataset_name.astype(str)
        )
        if "semantic_family_id" in frame:
            semantic_values = frame.semantic_family_id.astype(str)
            excluded_families.update(
                confirmation_family(value) for value in semantic_values
                if value and value.lower() != "nan"
            )
    return excluded_ids, excluded_families


def select_disjoint_manifest(*, old_manifest_path: Path = DEFAULT_OLD_MANIFEST,
                              exclusion_manifest_paths: tuple[Path, ...] = (),
                              output_path: Path = DEFAULT_NEW_MANIFEST,
                              cache_dir: Path = DEFAULT_CACHE,
                              target_count: int = 50,
                              seed: int = CONFIRMATORY_SEED,
                              catalog_status: str = "active",
                              min_rows: int = 200,
                              max_rows: int = 5000,
                              max_classes: int = 30,
                              max_catalog_features: int = 512,
                              max_expanded_features: int = 512,
                              max_public_cells: int | None = None,
                              minimum_families: int = MIN_CONFIRMATORY_FAMILIES,
                              protocol_path: Path = DEFAULT_PROTOCOL,
                              overwrite: bool = False) -> pd.DataFrame:
    """Sample eligible, family-disjoint datasets from the active OpenML catalog."""
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"{output_path} já existe; use --overwrite-manifest explicitamente")
    excluded_paths = (old_manifest_path, *exclusion_manifest_paths)
    old_ids, old_families = _manifest_identity(excluded_paths)

    openml = _openml(cache_dir)
    if catalog_status not in {"active", "in_preparation", "deactivated", "all"}:
        raise ValueError("catalog_status deve ser active, in_preparation, deactivated ou all")
    if min_rows < 1 or max_rows < min_rows:
        raise ValueError("limites de instâncias inválidos")
    if max_classes < 2 or max_catalog_features < 1 or max_expanded_features < 1:
        raise ValueError("limites públicos de classes/atributos inválidos")
    if max_public_cells is not None and max_public_cells < 1:
        raise ValueError("max_public_cells precisa ser positivo")
    if minimum_families < 1 or target_count < minimum_families:
        raise ValueError("target_count precisa ser >= minimum_families >= 1")
    catalog = openml.datasets.list_datasets(status=catalog_status, output_format="dataframe")
    id_column = _id_column(catalog)
    candidates = _metadata_prefilter(
        catalog, min_rows=min_rows, max_rows=max_rows, max_classes=max_classes,
        max_catalog_features=max_catalog_features,
    )
    candidates = candidates[~pd.to_numeric(candidates[id_column], errors="coerce").isin(old_ids)]
    candidate_ids = pd.to_numeric(candidates[id_column], errors="coerce").dropna().astype(int).to_numpy()
    name_column = next((name for name in candidates.columns if str(name).lower() == "name"), None)
    name_by_id = ({int(row[id_column]): str(row[name_column])
                   for _, row in candidates.iterrows()} if name_column is not None else {})
    order = np.random.default_rng(seed).permutation(candidate_ids)
    print(f"catalog entries after public size/class/feature filters: {len(order)}", flush=True)

    accepted: list[dict] = []
    seen_ids = set(old_ids)
    seen_families = set(old_families)
    for position, dataset_id in enumerate(order, start=1):
        dataset_id = int(dataset_id)
        if dataset_id in seen_ids:
            continue
        seen_ids.add(dataset_id)
        family_hint = confirmation_family(name_by_id.get(dataset_id, ""))
        if name_column is not None and family_hint in seen_families:
            continue
        try:
            candidate = candidate_from_dataset(
                openml, dataset_id, None, None,
                f"OpenML-{catalog_status}-catalog-disjoint",
                min_rows=min_rows, max_rows=max_rows, max_classes=max_classes,
                max_expanded_features=max_expanded_features,
            )
        except Exception as error:
            print(f"skip metadata {dataset_id}: {type(error).__name__}: {error}",
                  file=sys.stderr)
            continue
        if candidate is None:
            continue
        if (max_public_cells is not None
                and candidate.n_instances * candidate.schema.encoded_dim > max_public_cells):
            continue
        family_id = confirmation_family(candidate.name)
        if family_id in seen_families:
            continue
        accepted_row = candidate.manifest_row(f"OpenML-{catalog_status}-catalog-disjoint")
        accepted_row["family_id"] = family_id
        accepted.append(accepted_row)
        seen_families.add(family_id)
        if len(accepted) % 10 == 0:
            print(f"selected {len(accepted)}/{target_count} after {position} candidates",
                  flush=True)
        if len(accepted) == target_count:
            break
    if len(accepted) < target_count:
        raise RuntimeError(
            f"encontradas {len(accepted)}/{target_count} famílias novas elegíveis; "
            "amplie o catálogo/filtros explicitamente antes de medir utilidade"
        )
    selected = pd.DataFrame(accepted)
    overlap_ids = old_ids.intersection(selected.dataset_id.astype(int))
    overlap_families = old_families.intersection(selected.family_id.astype(str))
    if overlap_ids or overlap_families or selected.family_id.nunique() != len(selected):
        raise AssertionError("manifesto confirmatório não é disjunto por ID/família")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(output_path, index=False)
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    selection_record = {
        "manifest": str(output_path),
        "manifest_sha256": digest(output_path),
        "catalog_status": catalog_status,
        "target_families": target_count,
        "minimum_completed_families": minimum_families,
        "seed": seed,
        "eligibility": {
            "min_rows": min_rows,
            "max_rows": max_rows,
            "min_classes": 2,
            "max_classes": max_classes,
            "max_catalog_features": max_catalog_features,
            "max_expanded_features": max_expanded_features,
            "max_public_cells": max_public_cells,
        },
        "protocol": str(protocol_path),
        "protocol_sha256": digest(protocol_path) if protocol_path.exists() else None,
        "exclusion_manifest_sha256": {
            str(path): digest(path) for path in excluded_paths if path.exists()
        },
        "selector_implementation_sha256": digest(Path(__file__)),
        "n_unique_ids": int(selected.dataset_id.nunique()),
        "n_unique_canonical_families": int(selected.family_id.nunique()),
    }
    output_path.with_suffix(".selection.json").write_text(
        json.dumps(selection_record, indent=2, sort_keys=True) + "\n"
    )
    print(f"wrote {output_path}: {len(selected)} unseen datasets/families", flush=True)
    return selected


def _selector_frequency(predictions: pd.DataFrame) -> dict[str, float]:
    values = predictions.selector_pipeline.value_counts(normalize=True).sort_index()
    return {str(key): float(value) for key, value in values.items()}


def summarize_confirmation(predictions: pd.DataFrame, *, seed: int,
                           minimum_families: int = MIN_CONFIRMATORY_FAMILIES) -> dict:
    per_dataset = (predictions.groupby(["family_id", "dataset_id"], as_index=False)
                   .agg({"selector_regret": "mean", "fixed_regret": "mean",
                         "epsilon_fixed_regret": "mean", "gnb_regret": "mean",
                         "oracle_regret": "mean", "selector_ba": "mean",
                         "fixed_ba": "mean", "selector_top1": "mean"}))
    per_family = (per_dataset.groupby("family_id", as_index=False)
                  .agg({column: "mean" for column in per_dataset.columns
                        if column not in {"family_id", "dataset_id"}}))
    paired_gain = (per_family.fixed_regret - per_family.selector_regret).to_numpy()
    mean_gain, lower, upper = _bootstrap_ci(paired_gain, seed)
    p_value = _sign_flip_p(paired_gain, seed + 1)
    selector_regret = _metric_summary(per_family.selector_regret.to_numpy())
    fixed_regret = _metric_summary(per_family.fixed_regret.to_numpy())
    no_worse_tail = bool(
        selector_regret["median"] <= fixed_regret["median"]
        and selector_regret["q90"] <= fixed_regret["q90"]
        and selector_regret["cvar90"] <= fixed_regret["cvar90"]
    )
    n_families = int(per_family.family_id.nunique())
    enough_power = n_families >= minimum_families
    success = bool(enough_power and lower > 0 and p_value < 0.05 and no_worse_tail)
    return {
        "n_test_datasets": int(per_dataset.dataset_id.nunique()),
        "n_test_families": n_families,
        "primary_comparison": "frozen public+DP selector vs pre-registered fixed H27",
        "mean_regret_gain": mean_gain,
        "gain_ci95_family_bootstrap": [lower, upper],
        "paired_sign_flip_p_two_sided": p_value,
        "selector_regret": selector_regret,
        "fixed_regret": fixed_regret,
        "epsilon_fixed_regret": _metric_summary(per_family.epsilon_fixed_regret.to_numpy()),
        "gnb_fixed_regret": _metric_summary(per_family.gnb_regret.to_numpy()),
        "oracle_regret": _metric_summary(per_family.oracle_regret.to_numpy()),
        "selector_private_balanced_accuracy": _metric_summary(per_family.selector_ba.to_numpy()),
        "fixed_private_balanced_accuracy": _metric_summary(per_family.fixed_ba.to_numpy()),
        "selector_top1_auxiliary": float(per_family.selector_top1.mean()),
        "selector_pipeline_frequency": _selector_frequency(predictions),
        "no_worse_median_q90_cvar90": no_worse_tail,
        "minimum_40_families_met": enough_power,
        "minimum_family_threshold_met": enough_power,
        "minimum_families_required": minimum_families,
        "confirmatory_success": success,
        "status": ("conclusive" if enough_power else
                   f"inconclusive: fewer than {minimum_families} new families"),
    }


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _implementation_hash() -> str:
    return numerical_implementation_sha256()


def _write_report(summary: dict, fixed_pipeline: str) -> str:
    primary = summary["primary"]
    protocol_name = summary["protocol"]
    title = summary.get("report_title", "Confirmação externa do seletor de classificadores DP")
    lines = [
        f"# {title}",
        "",
        f"**Status:** {primary['status']}",
        "",
        f"**Baseline fixo H27, congelado antes do holdout:** "
        f"{PIPELINE_DISPLAY[fixed_pipeline]}.",
        f"**Conclusão:** {summary['interpretation']}.",
        summary.get("replication_context", ""),
        "",
        f"Teste em {primary['n_test_datasets']} datasets e "
        f"{primary['n_test_families']} famílias novas; nenhum resultado deste manifesto foi "
        "usado para treinar ou ajustar o seletor.",
        "",
        "## Comparação primária",
        "",
        f"- ganho de regret (fixo − seletor): {primary['mean_regret_gain']:.4f}",
        f"- IC95% bootstrap por família: [{primary['gain_ci95_family_bootstrap'][0]:.4f}, "
        f"{primary['gain_ci95_family_bootstrap'][1]:.4f}]",
        f"- teste pareado sign-flip bilateral: p = {primary['paired_sign_flip_p_two_sided']:.4f}",
        f"- mediana / q90 / CVaR90 do regret, seletor: "
        f"{primary['selector_regret']['median']:.4f} / {primary['selector_regret']['q90']:.4f} / "
        f"{primary['selector_regret']['cvar90']:.4f}",
        f"- mediana / q90 / CVaR90 do regret, fixo: "
        f"{primary['fixed_regret']['median']:.4f} / {primary['fixed_regret']['q90']:.4f} / "
        f"{primary['fixed_regret']['cvar90']:.4f}",
        f"- BA privada média por família, seletor/fixo: "
        f"{primary['selector_private_balanced_accuracy']['mean']:.4f} / "
        f"{primary['fixed_private_balanced_accuracy']['mean']:.4f}",
        f"- top-1 auxiliar: {primary['selector_top1_auxiliary']:.3f}",
        "",
        "## Baselines secundários",
        "",
        f"- GNB-zCDP fixo: regret médio {primary['gnb_fixed_regret']['mean']:.4f}",
        f"- fixo específico por epsilon: regret médio {primary['epsilon_fixed_regret']['mean']:.4f}",
        f"- oracle retrospectivo: regret médio {primary['oracle_regret']['mean']:.4f}",
        "",
        "## Limites",
        "",
        "OpenML é público; este resultado mede transferência para novas famílias, não protege dados "
        "reais. As 12 opções mecanismo–modelo, seus hiperparâmetros e features são os do protocolo congelado. "
        "Sementes são determinísticas para reprodutibilidade e não tornam os artifacts releases DP. "
        f"Consulte {protocol_name} para hipóteses, amostragem, métricas e semente. "
        "Sensibilidades e accounting estão registrados nos ledgers dos mecanismos; "
        "a implementação ainda não é uma auditoria de precisão finita.",
        "",
    ]
    return "\n".join(lines)


def run_confirmation(*, old_manifest_path: Path, old_results_path: Path,
                     new_manifest_path: Path, output_dir: Path,
                     exclusion_manifest_paths: tuple[Path, ...] = (),
                     protocol_path: Path = DEFAULT_PROTOCOL,
                     minimum_families: int = MIN_CONFIRMATORY_FAMILIES,
                     epsilons: tuple[float, ...] = DEFAULT_EPSILONS,
                     repeats: int = 3, seed: int = CONFIRMATORY_SEED,
                     cache_dir: Path = DEFAULT_CACHE,
                     overwrite: bool = False) -> dict:
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"{output_dir} não está vazio; use --overwrite-results explicitamente")
    output_dir.mkdir(parents=True, exist_ok=True)
    old_manifest = _load_manifest(old_manifest_path)
    new_manifest = _load_manifest(new_manifest_path)
    excluded_ids, excluded_families = _manifest_identity(
        (old_manifest_path, *exclusion_manifest_paths)
    )
    new_family = {confirmation_family(name) for name in new_manifest.dataset_name.astype(str)}
    overlap_ids = excluded_ids & set(new_manifest.dataset_id.astype(int))
    overlap_families = excluded_families & new_family
    if overlap_ids or overlap_families or len(new_family) != len(new_manifest):
        raise ValueError(f"holdout não é independente: ids={overlap_ids}, famílias={overlap_families}")
    old_results = pd.read_csv(old_results_path)
    old_aggregated = _aggregate(old_results, old_manifest)
    missing_options = set(PIPELINES) - set(old_aggregated.pipeline.unique())
    if missing_options:
        raise ValueError(
            "a confirmação externa exige resultados de desenvolvimento para todas as opções; "
            f"faltam {len(missing_options)}: {sorted(missing_options)}"
        )
    if H27_PIPELINE not in set(old_aggregated.pipeline):
        raise ValueError("a confirmação externa exige o baseline H27 no desenvolvimento")
    fixed_pipeline = H27_PIPELINE
    # This model is fitted only on the original meta-training set. No new test
    # labels, outcomes, or private descriptors are consulted before prediction.
    selector, _ = _fit_selector(
        old_aggregated, old_manifest.drop_duplicates("dataset_id"),
        "public_plus_dp", {}, seed=20260917,
    )
    config = {
        "protocol": str(protocol_path),
        "protocol_sha256": _hash(protocol_path),
        "old_manifest_sha256": _hash(old_manifest_path),
        "old_results_sha256": _hash(old_results_path),
        "new_manifest_sha256": _hash(new_manifest_path),
        "exclusion_manifest_sha256": {
            str(path): _hash(path) for path in exclusion_manifest_paths
        },
        "implementation_sha256": _implementation_hash(),
        "seed": seed,
        "epsilons": list(epsilons),
        "repeats": repeats,
        "minimum_families_required": minimum_families,
        "fixed_pipeline_frozen_before_test": fixed_pipeline,
        "fixed_pipeline_selection_rule": "pre-registered H27; no test data used",
        "selector_features": "public_plus_dp",
        "selector_strategy": "absolute_shared",
        "selector_target": "private_ba",
        "selector_fit": (
            f"RandomForestRegressor(n_estimators={SELECTOR_TREE_COUNT}, "
            "min_samples_leaf=3, max_features=0.9, max_depth=6, max_samples=0.5)"
        ),
    }
    (output_dir / "run_config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")

    openml = _openml(cache_dir)
    results_path = output_dir / "conditions.csv"
    failures = []
    for _, row in new_manifest.iterrows():
        try:
            rows = run_dataset(
                openml, row, epsilons, repeats, seed,
                include_research_features=False,
            )
            pd.DataFrame(rows).to_csv(
                results_path, mode="a", header=not results_path.exists(), index=False,
            )
            print(f"evaluated new {int(row.dataset_id)} {row.dataset_name}: {len(rows)} rows",
                  flush=True)
        except Exception as error:
            failures.append({"dataset_id": int(row.dataset_id),
                             "dataset_name": row.dataset_name,
                             "error": f"{type(error).__name__}: {error}"})
            print(f"failed {row.dataset_name}: {failures[-1]['error']}", file=sys.stderr,
                  flush=True)
        pd.DataFrame(failures, columns=["dataset_id", "dataset_name", "error"]).to_csv(
            output_dir / "failures.csv", index=False,
        )
    if not results_path.exists():
        raise RuntimeError("nenhuma família confirmatória completou")
    results = pd.read_csv(results_path)
    test_aggregated = _aggregate(results, new_manifest)
    meta_new = new_manifest.drop_duplicates("dataset_id")
    prediction_rows = []
    for (family_id, dataset_id, epsilon), condition in test_aggregated.groupby(
            ["family_id", "dataset_id", "epsilon"], sort=True):
        meta_row = meta_new[meta_new.dataset_id == dataset_id].iloc[0]
        candidate_x = np.vstack([
            _candidate_features(meta_row, float(epsilon), pipeline, "public_plus_dp", {})
            for pipeline in PIPELINES
        ])
        chosen = PIPELINES[int(np.argmax(selector.predict(candidate_x)))]
        fixed_for_epsilon = _best_fixed(old_aggregated, float(epsilon))
        oracle = max(PIPELINES, key=lambda pipeline: _condition_score(
            condition, pipeline, "private_ba"))
        prediction_rows.append({
            "family_id": family_id,
            "dataset_id": int(dataset_id),
            "epsilon": float(epsilon),
            "selector_pipeline": chosen,
            "fixed_pipeline": fixed_pipeline,
            "epsilon_fixed_pipeline": fixed_for_epsilon,
            "selector_regret": _condition_score(condition, chosen, "portfolio_regret"),
            "fixed_regret": _condition_score(condition, fixed_pipeline, "portfolio_regret"),
            "epsilon_fixed_regret": _condition_score(
                condition, fixed_for_epsilon, "portfolio_regret"),
            "gnb_regret": _condition_score(condition, PIPELINES[0], "portfolio_regret"),
            "oracle_regret": _condition_score(condition, oracle, "portfolio_regret"),
            "selector_ba": _condition_score(condition, chosen, "private_ba"),
            "fixed_ba": _condition_score(condition, fixed_pipeline, "private_ba"),
            "selector_top1": float(chosen == oracle),
        })
    predictions = pd.DataFrame(prediction_rows)
    primary = summarize_confirmation(
        predictions, seed=seed + 100, minimum_families=minimum_families,
    )
    completed = set(results.dataset_id.astype(int).unique())
    summary = {
        "protocol": str(protocol_path),
        "exploratory": False,
        "report_title": f"Confirmação externa do seletor DP — {len(new_manifest)} famílias novas",
        "replication_context": (
            "Amostra confirmatória independente, selecionada por filtros públicos e semente "
            "fixados no protocolo; nenhum resultado deste holdout será usado para retuning."
        ),
        "seed": seed,
        "epsilons": list(epsilons),
        "repeats": repeats,
        "new_manifest": str(new_manifest_path),
        "old_manifest": str(old_manifest_path),
        "excluded_manifests": [str(path) for path in exclusion_manifest_paths],
        "old_meta_results": str(old_results_path),
        "attempted_datasets": int(len(new_manifest)),
        "completed_datasets": int(len(completed)),
        "failed_datasets": failures,
        "fixed_pipeline": fixed_pipeline,
        "fixed_pipeline_description": PIPELINE_DISPLAY[fixed_pipeline],
        "selector_training_families": int(old_aggregated.family_id.nunique()),
        "no_overlap_families": True,
        "primary": primary,
        "interpretation": (
            "confirma ganho transferível do seletor" if primary["confirmatory_success"] else
            "não confirma ganho transferível do seletor; manter pipeline fixo como padrão"
            if primary["minimum_40_families_met"] else
            "inconclusivo: menos de 40 novas famílias completaram"
        ),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    (output_dir / "report.md").write_text(_write_report(summary, fixed_pipeline))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-manifest", type=Path, default=DEFAULT_OLD_MANIFEST)
    parser.add_argument("--old-results", type=Path, default=DEFAULT_OLD_RESULTS)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_NEW_MANIFEST)
    parser.add_argument("--exclude-manifest", type=Path, action="append", default=[])
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--minimum-families", type=int, default=MIN_CONFIRMATORY_FAMILIES)
    parser.add_argument("--epsilons", type=_parse_epsilon, default=DEFAULT_EPSILONS)
    parser.add_argument("--seed", type=int, default=CONFIRMATORY_SEED)
    parser.add_argument("--select-only", action="store_true")
    parser.add_argument("--overwrite-manifest", action="store_true")
    parser.add_argument("--overwrite-results", action="store_true")
    parser.add_argument("--catalog-status", choices=("active", "in_preparation", "deactivated", "all"),
                        default="active")
    parser.add_argument("--min-rows", type=int, default=200)
    parser.add_argument("--max-rows", type=int, default=5000)
    parser.add_argument("--max-classes", type=int, default=30)
    parser.add_argument("--max-catalog-features", type=int, default=512)
    parser.add_argument("--max-expanded-features", type=int, default=512)
    parser.add_argument("--max-public-cells", type=int)
    args = parser.parse_args(argv)
    if (args.minimum_families < 1 or args.limit < args.minimum_families
            or args.repeats < 1):
        parser.error("--limit precisa ser >= --minimum-families >= 1; --repeats >= 1")
    if args.select_only:
        select_disjoint_manifest(
            old_manifest_path=args.old_manifest, output_path=args.manifest,
            exclusion_manifest_paths=tuple(args.exclude_manifest),
            cache_dir=args.cache_dir, target_count=args.limit, seed=args.seed,
            catalog_status=args.catalog_status, min_rows=args.min_rows,
            max_rows=args.max_rows, max_classes=args.max_classes,
            max_catalog_features=args.max_catalog_features,
            max_expanded_features=args.max_expanded_features,
            max_public_cells=args.max_public_cells,
            minimum_families=args.minimum_families,
            protocol_path=args.protocol,
            overwrite=args.overwrite_manifest,
        )
        return 0
    if not args.manifest.exists():
        raise FileNotFoundError(
            f"manifesto confirmatório ausente: {args.manifest}; execute com --select-only, "
            "revise-o e então rode a confirmação"
        )
    run_confirmation(
        old_manifest_path=args.old_manifest, old_results_path=args.old_results,
        new_manifest_path=args.manifest, output_dir=args.output,
        exclusion_manifest_paths=tuple(args.exclude_manifest), protocol_path=args.protocol,
        minimum_families=args.minimum_families,
        epsilons=args.epsilons, repeats=args.repeats, seed=args.seed,
        cache_dir=args.cache_dir, overwrite=args.overwrite_results,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Train on the development OpenML cohort and confirm on fresh families."""

from __future__ import annotations

import argparse
from math import ceil
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from .model import (
    balanced_accuracy,
    model_from_statistics,
    predict,
    privatize_sufficient_statistics,
    sufficient_statistics,
)
from .families import conservative_family
from .openml_benchmark import (
    DEFAULT_CACHE,
    _get_openml_data,
    _load_manifest,
    _openml,
    _seed_for,
    stratified_holdout_indices,
)
from .selector import (
    ACTIONS,
    ACTION_NAMES,
    _fit_predictor,
    _predict_action,
    aggregate_repetitions,
)
from .provenance import numerical_implementation_sha256


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_MANIFEST = Path("data/openml_manifest.csv")
DEFAULT_TRAIN_RESULTS = Path("results/openml-l2-meta-50-2026-09-18/conditions.csv")
DEFAULT_TEST_MANIFEST = Path("data/openml_mechanism_confirmatory_150_manifest.csv")
DEFAULT_OUTPUT = Path("results/openml-mechanism-confirmatory-150-2026-09-18")
DEFAULT_EPSILONS = (0.5, 2.0, 8.0)
DEFAULT_SEED = 20260924
SELECTOR_MODES = ("dp_aware", "public")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _implementation_sha256() -> str:
    return numerical_implementation_sha256()


def family_equal_best_fixed(
    aggregated: pd.DataFrame, epsilon: float | None = None,
) -> tuple[str, str]:
    """Choose a fixed action after giving each training family equal weight."""
    data = aggregated if epsilon is None else aggregated[
        np.isclose(aggregated.epsilon.astype(float), float(epsilon))
    ]
    family_action = (data.groupby(["family_id", "mechanism", "allocation"], as_index=False)
                     .regret.mean())
    action_means = family_action.groupby(["mechanism", "allocation"]).regret.mean()
    if not len(action_means):
        raise ValueError("sem resultados de treino para escolher a ação fixa")
    available = [tuple(action) for action in action_means.index.to_list()]
    return min(available, key=lambda action: float(action_means.loc[action]))


def _distribution(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    tail_n = max(1, ceil(0.10 * len(values)))
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "q90": float(np.quantile(values, 0.90)),
        "cvar10": float(np.sort(values)[-tail_n:].mean()),
    }


def _sign_flip_pvalue(values: np.ndarray, seed: int, draws: int = 20_000) -> float:
    rng = np.random.default_rng(seed)
    observed = abs(float(np.mean(values)))
    signs = rng.choice(np.array([-1.0, 1.0]), size=(draws, len(values)))
    randomized = np.abs((signs * values).mean(axis=1))
    return float((1 + np.count_nonzero(randomized >= observed)) / (draws + 1))


def _bootstrap_ci(values: np.ndarray, seed: int, draws: int = 10_000) -> list[float]:
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    means = rng.choice(values, size=(draws, len(values)), replace=True).mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def _load_training(train_manifest_path: Path, train_results_path: Path):
    manifest = _load_manifest(train_manifest_path)
    manifest["family_id"] = manifest.dataset_name.map(conservative_family)
    results = pd.read_csv(train_results_path)
    expected_ids = set(manifest.dataset_id.astype(int))
    observed_ids = set(results.dataset_id.astype(int))
    if expected_ids != observed_ids:
        raise ValueError("resultados de desenvolvimento não cobrem exatamente o manifesto")
    if not set(results.mechanism.unique()) == {action[0] for action in ACTIONS}:
        raise ValueError("resultados de treino não contêm os quatro mecanismos congelados")
    aggregated = aggregate_repetitions(results)
    family_map = manifest.set_index("dataset_id").family_id.to_dict()
    aggregated["family_id"] = aggregated.dataset_id.map(family_map)
    if aggregated.family_id.isna().any():
        raise ValueError("ID de treino sem família no manifesto")
    expected_actions = set(ACTIONS)
    actual_actions = set(zip(aggregated.mechanism, aggregated.allocation))
    if expected_actions != actual_actions:
        raise ValueError("grade de ações dos resultados de treino difere do protocolo congelado")
    return manifest, aggregated


def _assert_fresh_manifest(test_manifest_path: Path, test_manifest: pd.DataFrame,
                           train_manifest: pd.DataFrame, output_dir: Path) -> None:
    if test_manifest.dataset_id.astype(int).duplicated().any():
        raise ValueError("manifesto de confirmação contém IDs duplicados")
    if test_manifest.family_id.astype(str).duplicated().any():
        raise ValueError("manifesto de confirmação contém famílias duplicadas")
    train_ids = set(train_manifest.dataset_id.astype(int))
    train_families = set(train_manifest.family_id.astype(str))
    if train_ids.intersection(test_manifest.dataset_id.astype(int)):
        raise ValueError("manifesto confirmatório reutiliza um dataset do desenvolvimento")
    if train_families.intersection(test_manifest.family_id.astype(str)):
        raise ValueError("manifesto confirmatório reutiliza uma família do desenvolvimento")

    known_ids, known_families = set(), set()
    for path in sorted((PROJECT_ROOT / "data").glob("*manifest.csv")):
        if path.resolve() == test_manifest_path.resolve():
            continue
        frame = pd.read_csv(path, keep_default_na=False)
        if "dataset_id" in frame:
            known_ids.update(pd.to_numeric(frame.dataset_id, errors="coerce")
                             .dropna().astype(int).tolist())
        for column in ("dataset_name", "family_id"):
            if column in frame:
                known_families.update(conservative_family(str(value)) for value in frame[column]
                                      if str(value).strip())
    for path in sorted((PROJECT_ROOT / "results").rglob("conditions*.csv")):
        if output_dir.resolve() in path.resolve().parents:
            continue
        try:
            frame = pd.read_csv(path, usecols=lambda column: column in
                                {"dataset_id", "dataset_name", "family_id"})
        except (ValueError, pd.errors.EmptyDataError):
            continue
        if "dataset_id" in frame:
            known_ids.update(pd.to_numeric(frame.dataset_id, errors="coerce")
                             .dropna().astype(int).tolist())
        for column in ("dataset_name", "family_id"):
            if column in frame:
                known_families.update(conservative_family(str(value)) for value in frame[column]
                                      if str(value).strip())
    overlap_ids = known_ids.intersection(test_manifest.dataset_id.astype(int))
    overlap_families = known_families.intersection(test_manifest.family_id.astype(str))
    if overlap_ids or overlap_families:
        raise ValueError(
            f"confirmação não é nova: {len(overlap_ids)} IDs e "
            f"{len(overlap_families)} famílias já aparecem no histórico"
        )


def _train_models(training: pd.DataFrame, manifest: pd.DataFrame, seed: int):
    private_by_dataset: dict[int, np.ndarray] = {}
    models = {}
    for index, mode in enumerate(SELECTOR_MODES):
        model, _ = _fit_predictor(
            training, manifest, mode, private_by_dataset,
            seed + index * 101, target_column="regret",
        )
        models[mode] = model
    return models


def _selected_action_table(test_manifest: pd.DataFrame, models: dict,
                           fixed: tuple[str, str], epsilon_fixed: dict[float, tuple[str, str]],
                           epsilons: tuple[float, ...]):
    rows = []
    for _, row in test_manifest.iterrows():
        for epsilon in epsilons:
            out = {
                "dataset_id": int(row.dataset_id),
                "dataset_name": str(row.dataset_name),
                "family_id": str(row.family_id),
                "epsilon": epsilon,
                "fixed_action": f"{fixed[0]}__{fixed[1]}",
                "epsilon_fixed_action": f"{epsilon_fixed[epsilon][0]}__{epsilon_fixed[epsilon][1]}",
                "preselected_laplace_equal": "laplace__equal",
                "preselected_zcdp_equal": "gaussian_zcdp__equal",
            }
            for mode, model in models.items():
                action = _predict_action(model, row, epsilon, mode)
                out[f"selector_action_{mode}"] = f"{action[0]}__{action[1]}"
            rows.append(out)
    return pd.DataFrame(rows)


def _run_dataset(openml, row: pd.Series, epsilons: tuple[float, ...],
                 action_row_by_epsilon: dict[float, dict], *, split_repeats: int,
                 noise_repeats: int, seed: int) -> list[dict]:
    x, labels, schema = _get_openml_data(openml, row)
    if np.count_nonzero(np.bincount(labels, minlength=len(schema.classes))) < 2:
        raise ValueError("menos de duas classes observadas")
    rows: list[dict] = []
    for split_repetition in range(split_repeats):
        train_indices, test_indices = stratified_holdout_indices(
            labels, _seed_for(seed, int(row.dataset_id), split_repetition, "split"),
        )
        x_train, x_test = x[train_indices], x[test_indices]
        y_train, y_test = labels[train_indices], labels[test_indices]
        clean_stats = sufficient_statistics(x_train, y_train, len(schema.classes))
        clean_ba = balanced_accuracy(y_test, predict(x_test, model_from_statistics(clean_stats)))
        for epsilon in epsilons:
            action_row = action_row_by_epsilon[float(epsilon)]
            for noise_repetition in range(noise_repeats):
                for mechanism, allocation in ACTIONS:
                    noisy_stats, ledger = privatize_sufficient_statistics(
                        clean_stats, epsilon=epsilon, mechanism=mechanism,
                        allocation=allocation,
                        seed=_seed_for(seed, int(row.dataset_id), split_repetition,
                                       epsilon, mechanism, allocation,
                                       "fresh-mechanism-confirmation", noise_repetition),
                    )
                    private_ba = balanced_accuracy(
                        y_test, predict(x_test, model_from_statistics(noisy_stats)),
                    )
                    rows.append({
                        "dataset_id": int(row.dataset_id),
                        "dataset_name": str(row.dataset_name),
                        "family_id": str(row.family_id),
                        "epsilon": float(epsilon),
                        "split_repetition": split_repetition,
                        "noise_repetition": noise_repetition,
                        "mechanism": mechanism,
                        "allocation": allocation,
                        "action": f"{mechanism}__{allocation}",
                        "clean_ba": clean_ba,
                        "private_ba": private_ba,
                        "regret": clean_ba - private_ba,
                        "ledger_epsilon": ledger.epsilon_used,
                        "ledger_delta": ledger.delta_used,
                        "encoded_dim": schema.encoded_dim,
                        **{
                            column: value for column, value in action_row.items()
                            if column.startswith("selector_action_")
                            or column in {"fixed_action", "epsilon_fixed_action",
                                          "preselected_laplace_equal", "preselected_zcdp_equal"}
                        },
                    })
    return rows


def _summarize_predictions(predictions: pd.DataFrame, *, seed: int) -> dict:
    per_dataset = (predictions.groupby(["family_id", "dataset_id"], as_index=False)
                   .agg({"selector_regret": "mean", "fixed_regret": "mean",
                         "epsilon_fixed_regret": "mean", "laplace_equal_regret": "mean",
                         "zcdp_equal_regret": "mean", "oracle_regret": "mean",
                         "top1": "mean"}))
    per_family = (per_dataset.groupby("family_id", as_index=False)
                  .agg({"selector_regret": "mean", "fixed_regret": "mean",
                        "epsilon_fixed_regret": "mean", "laplace_equal_regret": "mean",
                        "zcdp_equal_regret": "mean", "oracle_regret": "mean",
                        "top1": "mean"}))
    gain = (per_family.fixed_regret - per_family.selector_regret).to_numpy(dtype=float)
    fixed = per_family.fixed_regret.to_numpy(dtype=float)
    selected = per_family.selector_regret.to_numpy(dtype=float)
    ci = _bootstrap_ci(gain, seed)
    selected_distribution, fixed_distribution = _distribution(selected), _distribution(fixed)
    return {
        "primary_comparison": "selector vs best fixed action chosen on development families",
        "mean_gain_fixed_minus_selector": float(gain.mean()),
        "gain_ci95_family_bootstrap": ci,
        "sign_flip_p_two_sided": _sign_flip_pvalue(gain, seed + 1),
        "selector": selected_distribution,
        "best_fixed": fixed_distribution,
        "best_fixed_by_epsilon": _distribution(
            per_family.epsilon_fixed_regret.to_numpy(dtype=float)
        ),
        "preselected_laplace_equal": _distribution(
            per_family.laplace_equal_regret.to_numpy(dtype=float)
        ),
        "preselected_zcdp_equal": _distribution(
            per_family.zcdp_equal_regret.to_numpy(dtype=float)
        ),
        "oracle_reference": _distribution(per_family.oracle_regret.to_numpy(dtype=float)),
        "top1_auxiliary": float(per_family.top1.mean()),
        "success": bool(
            ci[0] > 0
            and _sign_flip_pvalue(gain, seed + 1) < 0.05
            and selected_distribution["median"] <= fixed_distribution["median"]
            and selected_distribution["q90"] <= fixed_distribution["q90"]
            and selected_distribution["cvar10"] <= fixed_distribution["cvar10"]
        ),
        "no_mean_improvement_demonstrated": bool(ci[0] <= 0),
    }


def _evaluate(raw: pd.DataFrame, actions: pd.DataFrame, seed: int) -> tuple[dict, pd.DataFrame]:
    means = (raw.groupby(["family_id", "dataset_id", "epsilon", "action"], as_index=False)
             .regret.mean())
    winners = means.loc[
        means.groupby(["dataset_id", "epsilon"]).regret.idxmin(),
        ["dataset_id", "epsilon", "action", "regret"],
    ].rename(columns={"action": "oracle_action", "regret": "oracle_regret"})
    indexed = means.set_index(["dataset_id", "epsilon", "action"]).regret
    predictions = actions.merge(winners, on=["dataset_id", "epsilon"], validate="one_to_one")
    action_to_metric = {
        "fixed_action": "fixed_regret",
        "epsilon_fixed_action": "epsilon_fixed_regret",
        "preselected_laplace_equal": "laplace_equal_regret",
        "preselected_zcdp_equal": "zcdp_equal_regret",
        "selector_action_dp_aware": "selector_regret_dp_aware",
        "selector_action_public": "selector_regret_public",
    }
    for column, metric in action_to_metric.items():
        predictions[metric] = [
            float(indexed.loc[(int(row.dataset_id), float(row.epsilon), str(getattr(row, column)))])
            for row in predictions.itertuples(index=False)
        ]
    predictions["top1_dp_aware"] = (
        predictions.selector_action_dp_aware == predictions.oracle_action
    ).astype(float)
    predictions["top1_public"] = (
        predictions.selector_action_public == predictions.oracle_action
    ).astype(float)
    dp = predictions.copy()
    dp["selector_regret"] = dp.selector_regret_dp_aware
    dp["top1"] = dp.top1_dp_aware
    public = predictions.copy()
    public["selector_regret"] = public.selector_regret_public
    public["top1"] = public.top1_public
    summary = {
        "dp_aware": _summarize_predictions(dp, seed=seed + 10),
        "public": _summarize_predictions(public, seed=seed + 20),
    }
    summary["action_frequencies"] = {
        mode: predictions[f"selector_action_{mode}"].value_counts().to_dict()
        for mode in SELECTOR_MODES
    }
    summary["oracle_action_frequencies"] = winners.oracle_action.value_counts().to_dict()
    return summary, predictions


def _render_report(summary: dict, config: dict, failures: list[dict]) -> str:
    primary = summary["dp_aware"]
    ci = primary["gain_ci95_family_bootstrap"]
    lines = [
        "# Confirmação externa do seletor de mecanismos DP",
        "",
        f"- Datasets completos: {summary['completed_datasets']}/{summary['attempted_datasets']}",
        f"- Famílias completas: {summary['n_families']}",
        f"- Liberações avaliadas: {summary['n_evaluation_releases']}",
        f"- Erro máximo do ledger em ε: {summary['max_ledger_epsilon_error']:.3g}",
        f"- Ação fixa escolhida apenas no desenvolvimento: `{config['best_fixed_action']}`",
        "",
        "## Seletor primário: DP-aware público vs. melhor ação fixa",
        "",
        f"- Ganho médio (fixo − seletor): {primary['mean_gain_fixed_minus_selector']:.5f}",
        f"- IC95% bootstrap por família: [{ci[0]:.5f}, {ci[1]:.5f}]",
        f"- Teste sign-flip bilateral: p = {primary['sign_flip_p_two_sided']:.5f}",
        f"- Mediana / q90 / CVaR10 seletor: {primary['selector']['median']:.5f} / "
        f"{primary['selector']['q90']:.5f} / {primary['selector']['cvar10']:.5f}",
        f"- Mediana / q90 / CVaR10 fixo: {primary['best_fixed']['median']:.5f} / "
        f"{primary['best_fixed']['q90']:.5f} / {primary['best_fixed']['cvar10']:.5f}",
        f"- Top-1 auxiliar: {primary['top1_auxiliary']:.3f}",
        "",
        "## Interpretação",
        "",
        ("Critérios confirmatórios atendidos." if primary["success"] else
         "INCONCLUSIVO: menos de 120 famílias completaram; não é possível declarar "
         "sucesso ou fracasso confirmatório." if not summary["confirmatory_threshold_met"] else
         "O seletor não demonstrou satisfazer todos os critérios confirmatórios; "
         "não foi ajustado após esta avaliação."),
        "A variante `public` schema-only é secundária; resultados completos estão em `summary.json`.",
        "OpenML é público: isto mede transferência de utilidade, não confidencialidade em dados reais.",
        "",
        f"Falhas de download/avaliação: {len(failures)}.",
    ]
    return "\n".join(lines) + "\n"


def run_confirmation(*, train_manifest_path: Path, train_results_path: Path,
                     test_manifest_path: Path, output_dir: Path, cache_dir: Path,
                     epsilons: tuple[float, ...] = DEFAULT_EPSILONS,
                     split_repeats: int = 3, noise_repeats: int = 3,
                     seed: int = DEFAULT_SEED, resume: bool = False) -> dict:
    if not epsilons or any(epsilon <= 0 for epsilon in epsilons):
        raise ValueError("epsilons precisam ser positivos")
    if split_repeats < 1 or noise_repeats < 1:
        raise ValueError("repetições precisam ser positivas")
    train_manifest, training = _load_training(train_manifest_path, train_results_path)
    test_manifest = _load_manifest(test_manifest_path)
    canonical = test_manifest.dataset_name.map(conservative_family)
    if not np.array_equal(test_manifest.family_id.astype(str).to_numpy(), canonical.astype(str).to_numpy()):
        test_manifest["family_id"] = canonical
    _assert_fresh_manifest(test_manifest_path, test_manifest, train_manifest, output_dir)
    models = _train_models(training, train_manifest, seed)
    fixed = family_equal_best_fixed(training)
    epsilon_fixed = {epsilon: family_equal_best_fixed(training, epsilon)
                     for epsilon in epsilons}
    actions = _selected_action_table(test_manifest, models, fixed, epsilon_fixed, epsilons)

    protocol_path = PROJECT_ROOT / "MECHANISM_CONFIRMATORY_PROTOCOL.md"
    selection_path = test_manifest_path.with_suffix(".selection.json")
    config = {
        "protocol": str(protocol_path),
        "protocol_sha256": _sha256(protocol_path),
        "implementation_sha256": _implementation_sha256(),
        "train_manifest_sha256": _sha256(train_manifest_path),
        "train_results_sha256": _sha256(train_results_path),
        "test_manifest_sha256": _sha256(test_manifest_path),
        "selection_record_sha256": _sha256(selection_path) if selection_path.is_file() else None,
        "epsilons": list(epsilons),
        "split_repeats": split_repeats,
        "noise_repeats_per_split": noise_repeats,
        "seed": seed,
        "actions": list(ACTION_NAMES),
        "selector_modes": list(SELECTOR_MODES),
        "primary_selector": "dp_aware",
        "best_fixed_action": f"{fixed[0]}__{fixed[1]}",
        "epsilon_fixed_actions": {
            str(epsilon): f"{action[0]}__{action[1]}"
            for epsilon, action in epsilon_fixed.items()
        },
        "expected_releases": int(len(test_manifest) * split_repeats * noise_repeats
                                  * len(epsilons) * len(ACTIONS)),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "run_config.json"
    results_path = output_dir / "conditions.csv"
    actions_path = output_dir / "actions.csv"
    failures_path = output_dir / "failures.csv"
    summary_path = output_dir / "summary.json"
    predictions_path = output_dir / "predictions.csv"
    report_path = output_dir / "report.md"
    if config_path.exists():
        previous = json.loads(config_path.read_text())
        if previous != config:
            raise ValueError("configuração difere da rodada existente; não retomar")
        if not resume:
            raise FileExistsError("saída existente; use --resume para continuar")
    elif any(path.exists() for path in (results_path, actions_path, summary_path)):
        raise FileExistsError("saída parcial sem configuração; preserve e use outro diretório")
    else:
        config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    if not actions_path.exists():
        actions.to_csv(actions_path, index=False)
    else:
        stored_actions = pd.read_csv(actions_path)
        if not stored_actions.equals(actions):
            raise ValueError("ações preditas mudaram em relação ao registro da rodada")

    existing = pd.read_csv(results_path) if results_path.exists() else pd.DataFrame()
    completed = set(existing.dataset_id.astype(int).unique()) if len(existing) else set()
    failures: list[dict] = []
    if failures_path.exists():
        try:
            failures = pd.read_csv(failures_path).to_dict("records")
        except pd.errors.EmptyDataError:
            failures = []
    openml = _openml(cache_dir)
    action_by_id = {
        int(dataset_id): {
            float(row.epsilon): row._asdict()
            for row in group.itertuples(index=False)
        }
        for dataset_id, group in actions.groupby("dataset_id", sort=False)
    }
    for _, row in test_manifest.iterrows():
        dataset_id = int(row.dataset_id)
        if dataset_id in completed:
            continue
        try:
            rows = _run_dataset(
                openml, row, epsilons, action_by_id[dataset_id],
                split_repeats=split_repeats, noise_repeats=noise_repeats, seed=seed,
            )
            pd.DataFrame(rows).to_csv(results_path, mode="a", header=not results_path.exists(),
                                      index=False)
            failures = [item for item in failures if int(item["dataset_id"]) != dataset_id]
            completed.add(dataset_id)
            print(f"finished {dataset_id} {row.dataset_name}: {len(rows)} releases", flush=True)
        except Exception as error:
            failures.append({"dataset_id": dataset_id, "dataset_name": row.dataset_name,
                             "error": f"{type(error).__name__}: {error}"})
            print(f"failed {dataset_id} {row.dataset_name}: {type(error).__name__}: {error}",
                  file=sys.stderr, flush=True)
        pd.DataFrame(failures, columns=["dataset_id", "dataset_name", "error"]).to_csv(
            failures_path, index=False,
        )

    if not results_path.is_file():
        raise RuntimeError("nenhum dataset completou; consulte failures.csv")
    results = pd.read_csv(results_path)
    completed_ids = set(results.dataset_id.astype(int).unique())
    completed_actions = actions[actions.dataset_id.astype(int).isin(completed_ids)].copy()
    summary, predictions = _evaluate(results, completed_actions, seed)
    family_count = int(results.family_id.nunique())
    summary.update({
        "attempted_datasets": int(len(test_manifest)),
        "completed_datasets": int(len(completed_ids)),
        "n_families": family_count,
        "n_splits": split_repeats,
        "n_noise_repeats_per_split": noise_repeats,
        "n_evaluation_releases": int(len(results)),
        "max_ledger_epsilon_error": float(np.max(np.abs(
            results.ledger_epsilon.to_numpy(dtype=float)
            - results.epsilon.to_numpy(dtype=float)
        ))),
        "failures": failures,
        "confirmatory_threshold_met": bool(family_count >= 120),
        "disclaimer": "OpenML é público; inferência sobre transferência entre famílias.",
    })
    if family_count < 120:
        for mode in SELECTOR_MODES:
            summary[mode]["success"] = False
            summary[mode]["interpretation"] = "inconclusivo: menos de 120 famílias completas"
    else:
        summary["dp_aware"]["interpretation"] = (
            "critérios confirmatórios atendidos" if summary["dp_aware"]["success"] else
            "não demonstrou ganho sem piora de mediana/cauda"
        )
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    predictions.to_csv(predictions_path, index=False)
    report_path.write_text(_render_report(summary, config, failures))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, default=DEFAULT_TRAIN_MANIFEST)
    parser.add_argument("--train-results", type=Path, default=DEFAULT_TRAIN_RESULTS)
    parser.add_argument("--test-manifest", type=Path, default=DEFAULT_TEST_MANIFEST)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epsilons", type=str, default="0.5,2,8")
    parser.add_argument("--split-repeats", type=int, default=3)
    parser.add_argument("--noise-repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    epsilons = tuple(float(item) for item in args.epsilons.split(",") if item.strip())
    run_confirmation(
        train_manifest_path=args.train_manifest,
        train_results_path=args.train_results,
        test_manifest_path=args.test_manifest,
        output_dir=args.output,
        cache_dir=args.cache_dir,
        epsilons=epsilons,
        split_repeats=args.split_repeats,
        noise_repeats=args.noise_repeats,
        seed=args.seed,
        resume=args.resume,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Post-hoc feature ablation on the already-used 150-family OpenML holdout.

This is explicitly exploratory. It reuses the existing utility outcomes and
fits a research-only selector with raw-data-derived meta-features; it is not a
new independent confirmation and carries no DP guarantee for the selector.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .external_confirmation import (
    DEFAULT_CACHE,
    _hash,
    _load_manifest,
)
from .openml_benchmark import (
    _get_openml_data,
    _openml,
    _seed_for,
    stratified_holdout_indices,
)
from .portfolio_benchmark import (
    PRIVATE_META_COLUMNS,
    PIPELINES,
    _aggregate,
    _best_fixed,
    _bootstrap_ci,
    _candidate_features,
    _condition_score,
    _fit_selector,
    _metric_summary,
    _sign_flip_p,
)
from .meta_features import private_training_meta_features


ROOT = Path(".")
OLD_MANIFEST = ROOT / "data/openml_manifest.csv"
OLD_RESULTS = ROOT / "results/classifier-portfolio-feasibility/conditions.csv"
TEST_MANIFEST = ROOT / "data/openml_confirmatory_150_manifest.csv"
TEST_RESULTS_DIR = ROOT / "results/openml-confirmatory-150-2026-09-18"
OUTPUT_DIR = ROOT / "results/exploratory-private-meta-150-2026-09-18"
SEED = 20260919


def _family_summaries(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    by_dataset = (predictions.groupby(["family_id", "dataset_id"], as_index=False)
                  .agg({column: "mean" for column in (
                      "selector_regret", "fixed_regret", "epsilon_fixed_regret",
                      "selector_ba", "fixed_ba", "selector_top1",
                  )}))
    by_family = (by_dataset.groupby("family_id", as_index=False)
                 .agg({column: "mean" for column in by_dataset.columns
                       if column not in {"family_id", "dataset_id"}}))
    return by_dataset, by_family


def _mode_metrics(predictions: pd.DataFrame, seed: int) -> dict:
    by_dataset, by_family = _family_summaries(predictions)
    gain = (by_family.fixed_regret - by_family.selector_regret).to_numpy()
    mean_gain, lower, upper = _bootstrap_ci(gain, seed)
    selector_regret = _metric_summary(by_family.selector_regret.to_numpy())
    fixed_regret = _metric_summary(by_family.fixed_regret.to_numpy())
    return {
        "n_test_datasets": int(by_dataset.dataset_id.nunique()),
        "n_test_families": int(by_family.family_id.nunique()),
        "mean_regret_gain_vs_fixed": mean_gain,
        "gain_ci95_family_bootstrap": [lower, upper],
        "paired_sign_flip_p_two_sided": _sign_flip_p(gain, seed + 1),
        "selector_regret": selector_regret,
        "fixed_regret": fixed_regret,
        "epsilon_fixed_regret": _metric_summary(by_family.epsilon_fixed_regret.to_numpy()),
        "selector_balanced_accuracy": _metric_summary(by_family.selector_ba.to_numpy()),
        "fixed_balanced_accuracy": _metric_summary(by_family.fixed_ba.to_numpy()),
        "selector_top1_auxiliary": float(by_family.selector_top1.mean()),
        "no_worse_median_q90_cvar90": bool(
            selector_regret["median"] <= fixed_regret["median"]
            and selector_regret["q90"] <= fixed_regret["q90"]
            and selector_regret["cvar90"] <= fixed_regret["cvar90"]
        ),
        "selector_pipeline_frequency": {
            str(k): float(v) for k, v in
            predictions.selector_pipeline.value_counts(normalize=True).sort_index().items()
        },
    }


def run(output_dir: Path = OUTPUT_DIR, cache_dir: Path = DEFAULT_CACHE,
        seed: int = SEED) -> dict:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"{output_dir} não está vazio; escolha outro diretório")
    old_manifest = _load_manifest(OLD_MANIFEST)
    test_manifest = _load_manifest(TEST_MANIFEST)
    old_conditions = pd.read_csv(OLD_RESULTS)
    test_conditions = pd.read_csv(TEST_RESULTS_DIR / "conditions.csv")
    prior_predictions = pd.read_csv(TEST_RESULTS_DIR / "predictions.csv")
    test_run_config = json.loads((TEST_RESULTS_DIR / "run_config.json").read_text())
    if int(test_run_config["seed"]) != seed:
        raise ValueError("seed diverge da rodada de 150 já salva")
    if test_run_config.get("selector_features") != "public_plus_dp":
        raise ValueError("a origem não parece ser a confirmação pública+DP esperada")

    old_ids = set(old_manifest.dataset_id.astype(int))
    test_ids = set(test_manifest.dataset_id.astype(int))
    if old_ids & test_ids:
        raise ValueError("sobreposição de dataset IDs entre meta-treino e teste")
    if len(test_manifest) != 150 or test_conditions.dataset_id.nunique() != 150:
        raise ValueError("a entrada não corresponde à coorte completa de 150 datasets")

    old_aggregated = _aggregate(old_conditions, old_manifest)
    missing_private = sorted(set(PRIVATE_META_COLUMNS) - set(old_aggregated.columns))
    if missing_private:
        raise ValueError(f"meta-treino não tem features privilegiadas: {missing_private}")
    private_train = {
        int(dataset_id): group.iloc[0][list(PRIVATE_META_COLUMNS)].to_numpy(dtype=float)
        for dataset_id, group in old_aggregated.groupby("dataset_id", sort=False)
    }
    selector, _ = _fit_selector(
        old_aggregated, old_manifest.drop_duplicates("dataset_id"),
        "research_privileged", private_train, seed=20260917,
    )

    openml = _openml(cache_dir)
    private_test: dict[int, np.ndarray] = {}
    for row in test_manifest.itertuples(index=False):
        row_series = pd.Series(row._asdict())
        x, y, schema = _get_openml_data(openml, row_series)
        train_indices, _ = stratified_holdout_indices(
            y, _seed_for(seed, int(row.dataset_id), 0, "portfolio_split"),
        )
        private_test[int(row.dataset_id)] = private_training_meta_features(
            x[train_indices], y[train_indices], len(schema.classes),
        )
        if len(private_test) % 25 == 0:
            print(f"features privadas de pesquisa: {len(private_test)}/150", flush=True)

    test_aggregated = _aggregate(test_conditions, test_manifest)
    test_meta = test_manifest.drop_duplicates("dataset_id").set_index("dataset_id")
    fixed_pipeline = _best_fixed(old_aggregated)
    predictions: list[dict] = []
    for (family_id, dataset_id, epsilon), condition in test_aggregated.groupby(
            ["family_id", "dataset_id", "epsilon"], sort=True):
        meta_row = test_meta.loc[int(dataset_id)]
        candidate_x = np.vstack([
            _candidate_features(meta_row, float(epsilon), pipeline,
                                "research_privileged", private_test)
            for pipeline in PIPELINES
        ])
        chosen = PIPELINES[int(np.argmax(selector.predict(candidate_x)))]
        fixed_epsilon = _best_fixed(old_aggregated, float(epsilon))
        oracle = max(PIPELINES, key=lambda pipeline: _condition_score(
            condition, pipeline, "private_ba"))
        predictions.append({
            "family_id": str(family_id), "dataset_id": int(dataset_id),
            "epsilon": float(epsilon), "feature_mode": "research_privileged",
            "selector_pipeline": chosen, "fixed_pipeline": fixed_pipeline,
            "epsilon_fixed_pipeline": fixed_epsilon,
            "selector_regret": _condition_score(condition, chosen, "portfolio_regret"),
            "fixed_regret": _condition_score(condition, fixed_pipeline, "portfolio_regret"),
            "epsilon_fixed_regret": _condition_score(
                condition, fixed_epsilon, "portfolio_regret"),
            "gnb_regret": _condition_score(
                condition, PIPELINES[0], "portfolio_regret"),
            "oracle_regret": _condition_score(condition, oracle, "portfolio_regret"),
            "selector_ba": _condition_score(condition, chosen, "private_ba"),
            "fixed_ba": _condition_score(condition, fixed_pipeline, "private_ba"),
            "selector_top1": float(chosen == oracle),
        })
    prediction_frame = pd.DataFrame(predictions)
    private_metrics = _mode_metrics(prediction_frame, seed + 100)

    # Direct, paired comparison to the already-run public+DP selector on these same rows.
    public = prior_predictions[["family_id", "dataset_id", "epsilon", "selector_regret"]].rename(
        columns={"selector_regret": "public_selector_regret"})
    comparison = prediction_frame.merge(
        public, on=["family_id", "dataset_id", "epsilon"], validate="one_to_one")
    per_family_delta = (comparison.assign(
        gain= comparison.public_selector_regret - comparison.selector_regret)
        .groupby("family_id", as_index=False).gain.mean().gain.to_numpy())
    private_vs_public = _bootstrap_ci(per_family_delta, seed + 200)
    private_vs_public_summary = {
        "mean_regret_gain_private_features_vs_public_plus_dp": private_vs_public[0],
        "gain_ci95_family_bootstrap": [private_vs_public[1], private_vs_public[2]],
        "paired_sign_flip_p_two_sided": _sign_flip_p(per_family_delta, seed + 201),
        "positive_favors_private_features": True,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "analysis_type": "post_hoc_exploratory_feature_ablation",
        "confirmatory": False,
        "exploratory": True,
        "warning": (
            "Reutiliza a coorte e os outcomes dos 150 datasets já inspecionados. "
            "Não é teste independente, não autoriza alegação confirmatória e "
            "features de pesquisa não são DP."
        ),
        "feature_mode": "research_privileged",
        "private_meta_feature_names": list(PRIVATE_META_COLUMNS),
        "private_features_use_only_train_rows": True,
        "utility_outcomes_reused_from": str(TEST_RESULTS_DIR / "conditions.csv"),
        "feature_split": "mesmo primeiro split 75/25 (seed 20260919) usado no benchmark",
        "fixed_pipeline_from_original_meta_train": fixed_pipeline,
        "selector_training_families": int(old_aggregated.family_id.nunique()),
        "test_run_seed": seed,
        "epsilons": sorted(float(x) for x in test_conditions.epsilon.unique()),
        "repeats_in_reused_utility_outcomes": int(test_conditions.repetition.nunique()),
        "research_privileged_vs_fixed": private_metrics,
        "research_privileged_vs_public_plus_dp_on_same_test": private_vs_public_summary,
        "input_hashes": {
            "old_manifest": _hash(OLD_MANIFEST),
            "old_conditions": _hash(OLD_RESULTS),
            "test_manifest": _hash(TEST_MANIFEST),
            "test_conditions": _hash(TEST_RESULTS_DIR / "conditions.csv"),
            "public_selector_predictions": _hash(TEST_RESULTS_DIR / "predictions.csv"),
        },
        "interpretation": (
            "diagnóstico pós-hoc do valor preditivo de meta-features derivadas das linhas; "
            "não mede uma recomendação com garantia de privacidade"
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n")
    prediction_frame.to_csv(output_dir / "predictions.csv", index=False)
    lines = [
        "# Ablation pós-hoc — meta-features de pesquisa não públicas",
        "",
        "> **Exploratória, não confirmatória.** Reutiliza os mesmos 150 datasets e outcomes já inspecionados.",
        "> O braço `research_privileged` lê estatísticas das linhas de treino e não é DP.",
        "",
        f"- Famílias retidas: {private_metrics['n_test_families']}",
        f"- Seletor privilegiado vs. melhor fixo: ganho médio de regret = {private_metrics['mean_regret_gain_vs_fixed']:.6f}",
        f"- IC95% bootstrap por família: [{private_metrics['gain_ci95_family_bootstrap'][0]:.6f}, {private_metrics['gain_ci95_family_bootstrap'][1]:.6f}]",
        f"- Sign-flip bilateral: p = {private_metrics['paired_sign_flip_p_two_sided']:.4f}",
        f"- Mediana/Q90/CVaR90 não piores: {private_metrics['no_worse_median_q90_cvar90']}",
        f"- Top-1 (auxiliar): {private_metrics['selector_top1_auxiliary']:.3f}",
        f"- Privilegiado vs. seletor público+DP, ganho médio: {private_vs_public[0]:.6f} ",
        f"(IC95% [{private_vs_public[1]:.6f}, {private_vs_public[2]:.6f}], p={private_vs_public_summary['paired_sign_flip_p_two_sided']:.4f})",
        "",
        "Não interpretar estes números como nova validação externa: o teste já foi aberto e a feature ablation é pós-hoc.",
    ]
    (output_dir / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)
    run(output_dir=args.output, cache_dir=args.cache_dir, seed=args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

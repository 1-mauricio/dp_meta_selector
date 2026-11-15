"""Compare selector feature modes and measure which meta-features drive the choice.

All modes are fitted on the same grouped folds. The analysis reports:

* each mode against the best fixed option and H27, with its metadata ε cost;
* paired mode-vs-mode comparisons at the same model ε (Holm-adjusted);
* modes that spend metadata ε against references given an equal or smaller
  total budget (``ε_model + ε_meta``) on the evaluated grid;
* grouped permutation importance: the increase in out-of-fold selector regret
  when one feature group is replaced by another dataset's values (same ε and
  option), plus impurity importance of individual features.

Research modes without DP are diagnostic ceilings with no privacy guarantee.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import dpmeta.portfolio_benchmark as pb
from dpmeta.meta_features import DP_META_COLUMNS, PRIVATE_META_COLUMNS
from dpmeta.openml_benchmark import _load_manifest
from dpmeta.provenance import (
    file_sha256, numerical_implementation_sha256, runtime_fingerprint,
    save_source_snapshot,
)

SCORE_COLUMNS = (
    "dataset_id", "epsilon", "pipeline", "repetition", "private_ba", "clean_ba",
    "best_clean_portfolio_ba", "portfolio_regret", "pipeline_privacy_penalty",
)
NON_DP_MODES = {"research_privileged", "research_rich"}


def metadata_cost(mode: str, dp_meta_epsilon: float) -> float | None:
    """Extra ε spent on meta-features; None means no DP guarantee at all."""
    if mode in NON_DP_MODES:
        return None
    return dp_meta_epsilon if mode == "dp_research_rich" else 0.0


def reference_epsilon(model_epsilon: float, meta_epsilon: float,
                      grid: tuple[float, ...]) -> float:
    """Largest grid ε not exceeding the selector's total budget."""
    return max(e for e in grid if e <= model_epsilon + meta_epsilon + 1e-12)


def _family_mean(frame: pd.DataFrame, column: str) -> pd.Series:
    frame = frame.assign(family_id=frame.family_id.astype(str))
    per_dataset = frame.groupby(["family_id", "dataset_id"])[column].mean()
    return per_dataset.groupby("family_id").mean()


def _paired(values: np.ndarray, seed: int) -> dict:
    mean, low, high = pb._bootstrap_ci(values, seed)
    return {"gain": mean, "ci95": [low, high], "sign_flip_p": pb._sign_flip_p(values, seed + 1)}


def mode_table(summary: dict, modes: tuple[str, ...], dp_meta_epsilon: float) -> pd.DataFrame:
    rows = []
    for mode in modes:
        result = summary["feature_modes"][mode]
        rows.append({
            "mode": mode,
            "metadata_epsilon": metadata_cost(mode, dp_meta_epsilon),
            "dp_guarantee": mode not in NON_DP_MODES,
            "regret_mean": result["selector_regret"]["mean"],
            "regret_median": result["selector_regret"]["median"],
            "regret_q90": result["selector_regret"]["q90"],
            "regret_cvar90": result["selector_regret"]["cvar90"],
            "gain_vs_best_fixed": result["mean_regret_gain_vs_best_fixed"],
            "gain_ci95_low": result["gain_ci95_family_bootstrap"][0],
            "gain_ci95_high": result["gain_ci95_family_bootstrap"][1],
            "gain_vs_h27": result.get("mean_regret_gain_vs_h27_fixed"),
            "no_worse_tails_vs_best_fixed": result["no_worse_median_and_tail"],
            "top1": result["top1_auxiliary"],
        })
    return pd.DataFrame(rows)


def pairwise_modes(predictions: pd.DataFrame, modes: tuple[str, ...], seed: int) -> pd.DataFrame:
    """Gain of ``mode_a`` over ``mode_b`` at the same model ε (positive: a better)."""
    regrets = {mode: _family_mean(predictions[predictions.feature_mode == mode],
                                  "selector_regret") for mode in modes}
    rows = []
    for index, (a, b) in enumerate(itertools.combinations(modes, 2)):
        diff = (regrets[b] - regrets[a].reindex(regrets[b].index)).to_numpy()
        rows.append({"mode_a": a, "mode_b": b, **_paired(diff, seed + 10 * index)})
    table = pd.DataFrame(rows)
    adjusted = pb._holm_adjust({f"{r.mode_a}|{r.mode_b}": r.sign_flip_p
                                for r in table.itertuples()})
    table["sign_flip_p_holm"] = [adjusted[f"{r.mode_a}|{r.mode_b}"] for r in table.itertuples()]
    return table


def equal_total_budget(predictions: pd.DataFrame, modes: tuple[str, ...],
                       dp_meta_epsilon: float, seed: int) -> pd.DataFrame:
    """Modes that spend ε_meta against references with total budget ≤ theirs.

    The reference runs at the largest grid ε not above ``ε_model + ε_meta``.
    When that is below the total, the comparison favours the spending mode, so
    only an exact match or a loss of the spending mode is conclusive.
    """
    grid = tuple(sorted(predictions.epsilon.unique()))
    references = [("best_fixed", None)] + [
        (mode, mode) for mode in modes if metadata_cost(mode, dp_meta_epsilon) == 0.0
    ]
    rows = []
    for mode in modes:
        cost = metadata_cost(mode, dp_meta_epsilon)
        if not cost:
            continue
        spending = predictions[predictions.feature_mode == mode]
        for model_epsilon in grid:
            ref_epsilon = reference_epsilon(model_epsilon, cost, grid)
            left = _family_mean(spending[spending.epsilon == model_epsilon], "selector_regret")
            for name, ref_mode in references:
                if ref_mode is None:
                    block = spending[spending.epsilon == ref_epsilon]
                    right = _family_mean(block, "best_fixed_regret")
                else:
                    block = predictions[(predictions.feature_mode == ref_mode)
                                        & (predictions.epsilon == ref_epsilon)]
                    right = _family_mean(block, "selector_regret")
                diff = (right - left.reindex(right.index)).to_numpy()
                result = _paired(diff, seed + len(rows))
                exact = abs(ref_epsilon - (model_epsilon + cost)) < 1e-12
                rows.append({
                    "mode": mode, "model_epsilon": model_epsilon,
                    "total_epsilon": model_epsilon + cost, "reference": name,
                    "reference_epsilon": ref_epsilon, "exact_budget_match": exact,
                    "mode_regret": float(left.mean()), "reference_regret": float(right.mean()),
                    **result,
                    # Exact budgets decide either way; a smaller reference budget
                    # only decides when the spending mode still loses.
                    "conclusive": bool(
                        (exact and (result["ci95"][0] > 0 or result["ci95"][1] < 0))
                        or (not exact and result["ci95"][1] < 0)
                    ),
                })
    return pd.DataFrame(rows)


def _condition_regret(test: pd.DataFrame, predicted: np.ndarray) -> pd.Series:
    """Per-family mean regret of the argmax option in each (dataset, ε)."""
    frame = test.assign(_score=predicted)
    chosen = frame.loc[frame.groupby(["dataset_id", "epsilon"])._score.idxmax()]
    return _family_mean(chosen, "portfolio_regret")


def _predict(selectors, test: pd.DataFrame, matrix: np.ndarray) -> np.ndarray:
    if not isinstance(selectors, dict):
        return selectors.predict(matrix)
    predicted = np.empty(len(test))
    for epsilon, model in selectors.items():
        mask = (test.epsilon == epsilon).to_numpy()
        predicted[mask] = model.predict(matrix[mask])
    return predicted


def permutation_importance(aggregated: pd.DataFrame, manifest: pd.DataFrame,
                           sink: dict, modes: tuple[str, ...], repeats: int,
                           seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Grouped permutation importance on out-of-fold selector regret."""
    meta = manifest.drop_duplicates("dataset_id").set_index("dataset_id")
    private = {int(d): g.iloc[0][list(PRIVATE_META_COLUMNS)].to_numpy(dtype=float)
               for d, g in aggregated.groupby("dataset_id")} \
        if set(PRIVATE_META_COLUMNS) <= set(aggregated.columns) else {}
    dp = {int(d): g.iloc[0][list(DP_META_COLUMNS)].to_numpy(dtype=float)
          for d, g in aggregated.groupby("dataset_id")} \
        if set(DP_META_COLUMNS) <= set(aggregated.columns) else {}
    order = {pipeline: index for index, pipeline in enumerate(pb.PIPELINES)}
    rng = np.random.default_rng(seed)
    group_rows, feature_rows = [], []
    for mode in modes:
        names = pb.candidate_feature_names(mode)
        groups = {}
        for column, (_, group) in enumerate(names):
            if group != "option":
                groups.setdefault(group, []).append(column)
        # Groups share information (calibration also depends on n and the
        # schema), so single groups measure non-redundant information only.
        calibration = [c for g, cols in groups.items() if g.startswith("calib_") for c in cols]
        if calibration:
            groups["all_calibration"] = calibration
        groups["all_meta_features"] = [c for c, (_, g) in enumerate(names) if g != "option"]
        deltas = {group: [] for group in groups}
        baselines = []
        impurity = np.zeros(len(names))
        n_models = 0
        for fold, entry in sink.items():
            selectors = entry["selectors"][mode]
            models = selectors.values() if isinstance(selectors, dict) else (selectors,)
            for model in models:
                impurity += model.feature_importances_
                n_models += 1
                model.set_params(n_jobs=pb.SELECTOR_N_JOBS)
            test = aggregated[aggregated.dataset_id.isin(entry["test_ids"])].copy()
            test["_order"] = test.pipeline.map(order)
            test = test.sort_values(["epsilon", "dataset_id", "_order"]).reset_index(drop=True)
            matrix = np.vstack([
                pb._candidate_features(meta.loc[int(row.dataset_id)], float(row.epsilon),
                                       row.pipeline, mode, private, dp)
                for row in test.itertuples(index=False)
            ])
            baseline = _condition_regret(test, _predict(selectors, test, matrix))
            baselines.append(baseline)
            n_options = len(pb.PIPELINES)
            for group, columns in groups.items():
                total = pd.Series(0.0, index=baseline.index)
                for _ in range(repeats):
                    permuted = matrix.copy()
                    for epsilon in test.epsilon.unique():
                        rows = np.flatnonzero((test.epsilon == epsilon).to_numpy())
                        block = permuted[rows].reshape(-1, n_options, matrix.shape[1])
                        donors = rng.permutation(len(block))
                        block[:, :, columns] = block[donors][:, :, columns]
                        permuted[rows] = block.reshape(len(rows), -1)
                    total += _condition_regret(test, _predict(selectors, test, permuted)) - baseline
                deltas[group].append(total / repeats)
            for model in models:
                model.set_params(n_jobs=1)
        for index, (group, parts) in enumerate(deltas.items()):
            values = pd.concat(parts).to_numpy()
            mean, low, high = pb._bootstrap_ci(values, seed + index)
            group_rows.append({
                "mode": mode, "group": group, "n_features": len(groups[group]),
                "regret_increase": mean, "ci95_low": low, "ci95_high": high,
                "impurity_share": float(impurity[groups[group]].sum() / n_models),
                "composite": group.startswith("all_"),
            })
        impurity /= n_models
        feature_rows += [{"mode": mode, "feature": name, "group": group,
                          "impurity_importance": float(value)}
                         for (name, group), value in zip(names, impurity)]
    groups_table = pd.DataFrame(group_rows).sort_values(
        ["mode", "regret_increase"], ascending=[True, False])
    features_table = pd.DataFrame(feature_rows).sort_values(
        ["mode", "impurity_importance"], ascending=[True, False])
    return groups_table, features_table


def _report(modes: pd.DataFrame, pairwise: pd.DataFrame, equal_total: pd.DataFrame,
            groups: pd.DataFrame, features: pd.DataFrame) -> str:
    def fmt(value):
        return "—" if value is None or (isinstance(value, float) and np.isnan(value)) \
            else f"{value:.4f}"
    lines = ["# Comparação de seletores e importância das meta-features", "",
             "Modos sem garantia DP (`research_*`) são tetos diagnósticos.", "",
             "## Modos frente à melhor opção fixa", "",
             "| Modo | ε meta | Regret médio | Ganho [IC95%] | Ganho vs H27 | Mediana | q90 | CVaR90 |",
             "|---|---:|---:|---|---:|---:|---:|---:|"]
    for r in modes.itertuples():
        cost = "sem DP" if not r.dp_guarantee else f"{r.metadata_epsilon:g}"
        lines.append(
            f"| {r.mode} | {cost} | {r.regret_mean:.4f} | {r.gain_vs_best_fixed:.4f} "
            f"[{r.gain_ci95_low:.4f}; {r.gain_ci95_high:.4f}] | {fmt(r.gain_vs_h27)} | "
            f"{r.regret_median:.4f} | {r.regret_q90:.4f} | {r.regret_cvar90:.4f} |")
    lines += ["", "## Comparações pareadas no mesmo ε do modelo", "",
              "Ganho positivo: o modo A tem regret menor que o B.", "",
              "| A | B | Ganho [IC95%] | p (Holm) |", "|---|---|---|---:|"]
    for r in pairwise.itertuples():
        lines.append(f"| {r.mode_a} | {r.mode_b} | {r.gain:.4f} [{r.ci95[0]:.4f}; "
                     f"{r.ci95[1]:.4f}] | {r.sign_flip_p_holm:.2g} |")
    if len(equal_total):
        lines += ["", "## Orçamento total igual ou menor para a referência", "",
                  "Ganho negativo: o modo que gasta ε_meta perde.", "",
                  "Conclusivo: ε igual com IC que exclui zero, ou referência com menos "
                  "orçamento e ainda assim melhor.", "",
                  "| Modo | ε modelo | ε total | Referência | ε ref. | ε igual | Ganho [IC95%] | Conclusivo |",
                  "|---|---:|---:|---|---:|---|---|---|"]
        for r in equal_total.itertuples():
            lines.append(f"| {r.mode} | {r.model_epsilon:g} | {r.total_epsilon:g} | "
                         f"{r.reference} | {r.reference_epsilon:g} | "
                         f"{'sim' if r.exact_budget_match else 'não'} | {r.gain:.4f} "
                         f"[{r.ci95[0]:.4f}; {r.ci95[1]:.4f}] | "
                         f"{'sim' if r.conclusive else 'não'} |")
    lines += ["", "## Importância por grupo (permutação)", "",
              "Aumento do regret fora da amostra ao trocar o grupo pelo de outra base "
              "(mesmo ε e opção). Esta é a medida de influência na recomendação. "
              "Grupos `all_*` permutam vários grupos juntos e capturam a informação "
              "que eles compartilham; grupos isolados medem só a parte não redundante.", "",
              "| Modo | Grupo | Features | Aumento do regret [IC95%] | Fração de impureza |",
              "|---|---|---:|---|---:|"]
    for r in groups.itertuples():
        lines.append(f"| {r.mode} | {r.group} | {r.n_features} | {r.regret_increase:.4f} "
                     f"[{r.ci95_low:.4f}; {r.ci95_high:.4f}] | {r.impurity_share:.3f} |")
    lines += ["", "## Features individuais mais usadas (impureza, top 10 por modo)", "",
              "A impureza mede o uso da feature para prever a utilidade, inclusive o seu "
              "nível; não mede o efeito sobre a escolha entre opções.", ""]
    for mode, block in features.groupby("mode", sort=False):
        lines.append(f"**{mode}:** " + "; ".join(
            f"{r.feature} ({r.impurity_importance:.3f})" for r in block.head(10).itertuples()))
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path,
                        default=Path("data/openml_manifest_984_completed.csv"))
    parser.add_argument("--conditions", type=Path,
                        default=Path("results/portfolio-984-current/conditions.csv"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/portfolio-984-selector-modes"))
    parser.add_argument("--modes", default=",".join(pb.FEATURE_MODES))
    parser.add_argument("--selector-strategy", choices=pb.SELECTOR_STRATEGIES,
                        default="absolute_shared")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--permutation-repeats", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None,
                        help="usa só os primeiros datasets do manifesto (teste rápido)")
    parser.add_argument("--overwrite", action="store_true")
    from dpmeta import selector_portfolio13 as sel
    parser.add_argument('--selector', choices=sel.SELECTORS, default='rf12')
    args = parser.parse_args(argv)
    if args.selector == 'gbm_residual_fallback':
        if args.output == Path('results/portfolio-984-selector-modes'):args.output=Path('results/dissertation-portfolio13/modes')
        from scripts.dissertation_analysis_portfolio13 import modes_analysis
        modes_analysis(args.conditions, args.manifest, args.output, args.folds,
                       args.permutation_repeats)
        return 0

    modes = tuple(mode.strip() for mode in args.modes.split(",") if mode.strip())
    if unknown := set(modes) - set(pb.FEATURE_MODES):
        parser.error(f"modos desconhecidos: {sorted(unknown)}")
    source_config_path = args.conditions.parent / "run_config.json"
    source_config = json.loads(source_config_path.read_text())
    manifest_hash = file_sha256(args.manifest)
    if manifest_hash != source_config["manifest_sha256"]:
        raise ValueError("manifesto difere do registrado nas condições originais")
    if tuple(source_config["pipelines"]) != pb.PIPELINES:
        raise ValueError("portfólio atual difere do portfólio das condições")
    outputs = ("summary.json", "report.md", "modes.csv", "pairwise.csv",
               "equal_total.csv", "importance_groups.csv", "importance_features.csv")
    args.output.mkdir(parents=True, exist_ok=True)
    if not args.overwrite and any((args.output / name).exists() for name in outputs):
        raise FileExistsError(f"análise já existe em {args.output}; use --overwrite")

    manifest = _load_manifest(args.manifest).head(
        args.limit or int(source_config["limit"])).copy()
    columns = {*SCORE_COLUMNS, *PRIVATE_META_COLUMNS, *DP_META_COLUMNS}
    conditions = pd.read_csv(args.conditions, usecols=lambda column: column in columns)
    conditions = conditions[conditions.dataset_id.isin(manifest.dataset_id)]
    epsilons = tuple(float(value) for value in source_config["epsilons"])
    pb.validate_condition_grid(conditions, manifest, epsilons=epsilons,
                               repeats=int(source_config["repeats"]), pipelines=pb.PIPELINES)
    dp_meta_epsilon = float(source_config.get("dp_meta_epsilon", 0.0))
    config = {
        "source_conditions_sha256": file_sha256(args.conditions),
        "source_run_implementation_sha256": source_config["implementation_sha256"],
        "source_manifest_sha256": manifest_hash,
        "analysis_implementation_sha256": numerical_implementation_sha256(),
        "analysis_script_sha256": file_sha256(Path(__file__)),
        "modes": list(modes), "selector_strategy": args.selector_strategy,
        "folds": args.folds, "seed": args.seed, "limit": args.limit,
        "permutation_repeats": args.permutation_repeats,
        "dp_meta_epsilon": dp_meta_epsilon, "runtime": runtime_fingerprint(),
    }
    save_source_snapshot(args.output, extra_files=("scripts/analyze_selector_modes.py",))
    (args.output / "run_config.json").write_text(json.dumps(config, indent=2) + "\n")

    sink: dict = {}
    with threadpool_limits(limits=1):
        summary, predictions = pb.evaluate_grouped(
            conditions, manifest, n_splits=args.folds, seed=args.seed,
            selector_strategy=args.selector_strategy, feature_modes_override=modes,
            progress=True, selector_sink=sink,
        )
        print("permutation importance", flush=True)
        groups, features = permutation_importance(
            pb._aggregate(conditions, manifest), manifest, sink, modes,
            args.permutation_repeats, args.seed,
        )
    modes_table = mode_table(summary, modes, dp_meta_epsilon)
    pairwise = pairwise_modes(predictions, modes, args.seed)
    equal_total = equal_total_budget(predictions, modes, dp_meta_epsilon, args.seed)

    predictions.to_csv(args.output / "outer_predictions.csv", index=False)
    for name, table in (("modes", modes_table), ("pairwise", pairwise),
                        ("equal_total", equal_total), ("importance_groups", groups),
                        ("importance_features", features)):
        table.to_csv(args.output / f"{name}.csv", index=False)
    (args.output / "summary.json").write_text(json.dumps({
        "config": config, "selector_summary": summary,
        "modes": modes_table.to_dict("records"),
        "pairwise": pairwise.to_dict("records"),
        "equal_total": equal_total.to_dict("records"),
        "importance_groups": groups.to_dict("records"),
    }, indent=2, default=float) + "\n")
    (args.output / "report.md").write_text(
        _report(modes_table, pairwise, equal_total, groups, features))
    print(modes_table.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

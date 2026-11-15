"""Análise do holdout de 30 famílias: primária congelada e secundárias da emenda.

Primária: mesma rotina de ``external_confirmation.run_confirmation`` (seletor
``public_plus_dp`` de 1.000 árvores contra H27, limiar de 40 famílias).
Secundárias (``CONFIRMATORY_PROTOCOL_AMENDMENT_SECONDARY.md``): seletores
alternativos treinados só no desenvolvimento, referências fixas e baselines que
pagam privacidade, contra H27 e contra a melhor fixa, com Holm, caudas,
Friedman/Nemenyi e estrato dentro/fora da faixa de n do desenvolvimento.

    PYTHONPATH=. .venv/bin/python scripts/analyze_holdout.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import friedmanchisquare, studentized_range  # noqa: E402
from sklearn.ensemble import (  # noqa: E402
    ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestClassifier,
)
from sklearn.linear_model import Ridge  # noqa: E402
from sklearn.neighbors import KNeighborsRegressor  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

import dpmeta.external_confirmation as ec  # noqa: E402
import dpmeta.portfolio_benchmark as pb  # noqa: E402
from dpmeta.figures import _cd_diagram, _save  # noqa: E402
from dpmeta.openml_benchmark import _load_manifest  # noqa: E402
from scripts.analyze_dissertation_extras_portfolio13 import (  # noqa: E402
    CAT, FeatureBuilder, classifier_matrix,
)

H27 = pb.H27_PIPELINE
BEST_FIXED = "categorical_naive_bayes__laplace_histograms"
DEV_MAX_N = 51336
LABELS = {
    "selector": "Seletor único (congelado)", "per_epsilon": "Seletor por ε",
    "schema_only": "Seletor só schema", "gb": "Gradient boosting", "et": "Extra-Trees",
    "knn": "k-NN", "ridge": "Ridge", "classifier": "Classificador da vencedora",
    "fixed_h27": "H27 (DP-SGD clip 2)", "fixed_nb": "Laplace · NB categórico",
    "fixed_gnb": "Gauss. zCDP · GNB", "search_shards": "Busca privada (partições)",
    "search_retrain": "Busca privada + retreino", "pythia_like": "Estilo Pythia (0,1ε)",
    "random": "Escolha aleatória", "oracle": "Oráculo",
}
DESCRIPTIVE = {"random", "oracle"}


def merge_shards(output: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    shards = output / "shards"
    conditions = pd.concat([pd.read_csv(p) for p in sorted(shards.glob("*.csv"))
                            if p.stem.isdigit()], ignore_index=True)
    baselines = pd.concat([pd.read_csv(p) for p in sorted(shards.glob("*_baselines.csv"))],
                          ignore_index=True)
    done = set(conditions.dataset_id)
    baselines = baselines[baselines.dataset_id.isin(done)]
    conditions.to_csv(output / "conditions.csv", index=False)
    baselines.to_csv(output / "baselines.csv", index=False)
    return conditions, baselines


def primary(dev_agg, dev_manifest, test_agg, holdout_manifest, seed, output) -> dict:
    """Idêntica à etapa final de ``run_confirmation``."""
    selector, _ = pb._fit_selector(dev_agg, dev_manifest.drop_duplicates("dataset_id"),
                                   "public_plus_dp", {}, seed=20260917)
    meta_new = holdout_manifest.drop_duplicates("dataset_id")
    rows = []
    for (family_id, dataset_id, epsilon), condition in test_agg.groupby(
            ["family_id", "dataset_id", "epsilon"], sort=True):
        meta_row = meta_new[meta_new.dataset_id == dataset_id].iloc[0]
        candidate_x = np.vstack([pb._candidate_features(meta_row, float(epsilon), p,
                                                        "public_plus_dp", {})
                                 for p in pb.PIPELINES])
        chosen = pb.PIPELINES[int(np.argmax(selector.predict(candidate_x)))]
        fixed_eps = pb._best_fixed(dev_agg, float(epsilon))
        oracle = max(pb.PIPELINES, key=lambda p: pb._condition_score(condition, p, "private_ba"))
        score = lambda p, c="portfolio_regret": pb._condition_score(condition, p, c)  # noqa: E731
        rows.append({
            "family_id": family_id, "dataset_id": int(dataset_id), "epsilon": float(epsilon),
            "selector_pipeline": chosen, "fixed_pipeline": H27,
            "epsilon_fixed_pipeline": fixed_eps, "selector_regret": score(chosen),
            "fixed_regret": score(H27), "epsilon_fixed_regret": score(fixed_eps),
            "gnb_regret": score(pb.PIPELINES[0]), "oracle_regret": score(oracle),
            "selector_ba": score(chosen, "private_ba"), "fixed_ba": score(H27, "private_ba"),
            "selector_top1": float(chosen == oracle)})
    predictions = pd.DataFrame(rows)
    summary = ec.summarize_confirmation(predictions, seed=seed + 100, minimum_families=40)
    predictions.to_csv(output / "predictions.csv", index=False)
    return summary, predictions


def dev_selectors(dev_agg, dev_manifest, builder: FeatureBuilder, seed: int) -> dict:
    """Seletores secundários, todos treinados só no desenvolvimento."""
    one = dev_manifest.drop_duplicates("dataset_id")
    fitted = {"schema_only": pb._fit_selector(dev_agg, one, "public_schema_only", {},
                                              seed=20260917)[0]}
    per_eps = {}
    for epsilon in sorted(dev_agg.epsilon.unique()):
        block, target = pb._selector_training_block(dev_agg[dev_agg.epsilon == epsilon],
                                                    "centered_per_epsilon")
        per_eps[float(epsilon)] = pb._fit_selector(
            block, one, "public_plus_dp", {}, seed=20260917 + int(float(epsilon) * 1000),
            target_column=target)[0]
    fitted["per_epsilon"] = per_eps
    x = builder.matrix(dev_agg)
    w = pb._row_weights(dev_agg)
    y = dev_agg.private_ba.to_numpy()
    fitted["gb"] = HistGradientBoostingRegressor(random_state=20260917).fit(x, y, sample_weight=w)
    fitted["et"] = ExtraTreesRegressor(500, min_samples_leaf=3, max_features=0.9,
                                       random_state=20260917, n_jobs=8).fit(x, y, sample_weight=w)
    fitted["knn"] = make_pipeline(StandardScaler(), KNeighborsRegressor(25)).fit(x, y)
    fitted["ridge"] = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(
        x, y, ridge__sample_weight=w)
    xc, cond = classifier_matrix(dev_agg, builder)
    winners = dev_agg.loc[dev_agg.groupby(["dataset_id", "epsilon"]).private_ba.idxmax()]
    labels = cond.merge(winners, on=["family_id", "dataset_id", "epsilon"]).pipeline
    fitted["classifier"] = RandomForestClassifier(
        500, min_samples_leaf=3, max_depth=12, random_state=20260917, n_jobs=8).fit(
        xc, labels.to_numpy())
    return fitted


def secondary_choices(test_agg, holdout_manifest, fitted, builder) -> pd.DataFrame:
    order = {p: i for i, p in enumerate(pb.PIPELINES)}
    test = test_agg.assign(_o=test_agg.pipeline.map(order)).sort_values(
        ["epsilon", "dataset_id", "_o"]).reset_index(drop=True)
    matrix = builder.matrix(test)
    frames = []

    def pick(name, scores):
        chosen = test.assign(_s=scores).loc[lambda f: f.groupby(
            ["dataset_id", "epsilon"])._s.idxmax()]
        frames.append(chosen.assign(method=name))

    for name in ("gb", "et", "knn", "ridge"):
        pick(name, fitted[name].predict(matrix))
    one = holdout_manifest.drop_duplicates("dataset_id").set_index("dataset_id")
    schema_x = np.vstack([pb._candidate_features(one.loc[int(r.dataset_id)], float(r.epsilon),
                                                 r.pipeline, "public_schema_only", {})
                          for r in test.itertuples()])
    pick("schema_only", fitted["schema_only"].predict(schema_x))
    scores = np.empty(len(test))
    for epsilon, model in fitted["per_epsilon"].items():
        mask = np.isclose(test.epsilon, epsilon)
        scores[mask] = model.predict(matrix[mask])
    pick("per_epsilon", scores)
    xc, cond = classifier_matrix(test, builder)
    chosen = cond.assign(pipeline=fitted["classifier"].predict(xc)).merge(
        test, on=["family_id", "dataset_id", "epsilon", "pipeline"])
    frames.append(chosen.assign(method="classifier"))
    return pd.concat(frames)[["method", "family_id", "dataset_id", "epsilon", "pipeline",
                              "portfolio_regret"]]


def method_table(test_agg, predictions, secondary, baselines) -> pd.DataFrame:
    """Arrependimento por (método, família, dataset, ε)."""
    keys = ["family_id", "dataset_id", "epsilon"]
    parts = [predictions[keys + ["selector_regret"]].rename(
        columns={"selector_regret": "regret"}).assign(method="selector"),
        secondary.rename(columns={"portfolio_regret": "regret"})[keys + ["regret", "method"]]]
    for name, pipeline in (("fixed_h27", H27), ("fixed_nb", BEST_FIXED),
                           ("fixed_gnb", pb.PIPELINES[0])):
        block = test_agg[test_agg.pipeline == pipeline]
        parts.append(block[keys + ["portfolio_regret"]].rename(
            columns={"portfolio_regret": "regret"}).assign(method=name))
    grouped = test_agg.groupby(keys).portfolio_regret
    parts.append(grouped.mean().rename("regret").reset_index().assign(method="random"))
    parts.append(grouped.min().rename("regret").reset_index().assign(method="oracle"))
    family = test_agg[["dataset_id", "family_id"]].drop_duplicates()
    base = (baselines.groupby(["method", "dataset_id", "epsilon"]).portfolio_regret.mean()
            .rename("regret").reset_index().merge(family, on="dataset_id"))
    parts.append(base)
    return pd.concat(parts, ignore_index=True)[["method", *keys, "regret"]]


def per_family(long: pd.DataFrame, datasets=None) -> pd.DataFrame:
    if datasets is not None:
        long = long[long.dataset_id.isin(datasets)]
    return (long.groupby(["method", "family_id", "dataset_id"]).regret.mean()
            .groupby(["method", "family_id"]).mean().unstack("method"))


def comparisons(wide: pd.DataFrame, seed: int) -> pd.DataFrame:
    rows = []
    for method in wide.columns:
        if method in DESCRIPTIVE:
            continue
        values = wide[method]
        summary = ec._metric_summary(values.to_numpy())
        for reference in ("fixed_h27", "fixed_nb"):
            if method == reference:
                continue
            diff = (wide[reference] - values).to_numpy()
            mean, low, high = pb._bootstrap_ci(diff, seed + len(rows))
            rows.append({"method": method, "label": LABELS[method], "reference": reference,
                         "regret": summary["mean"], "median": summary["median"],
                         "q90": summary["q90"], "cvar90": summary["cvar90"],
                         "gain": mean, "ci_low": low, "ci_high": high,
                         "sign_flip_p": pb._sign_flip_p(diff, seed + 1000 + len(rows))})
    table = pd.DataFrame(rows)
    secondary = {f"{r.method}|{r.reference}": r.sign_flip_p for r in table.itertuples()
                 if not (r.method == "selector" and r.reference == "fixed_h27")}
    holm = pb._holm_adjust(secondary)
    table["p_holm"] = [holm.get(f"{r.method}|{r.reference}", np.nan) for r in table.itertuples()]
    return table


def fig_cd(wide: pd.DataFrame, out: Path) -> dict:
    data = wide.drop(columns=[c for c in DESCRIPTIVE if c in wide]).dropna()
    data = data.rename(columns=LABELS)
    n, k = data.shape
    ranks = data.rank(axis=1).mean()
    cd = studentized_range.ppf(0.95, k, np.inf) / np.sqrt(2) * np.sqrt(k * (k + 1) / (6 * n))
    p = friedmanchisquare(*[data[c].to_numpy() for c in data]).pvalue
    fig, ax = plt.subplots(figsize=(12, 5.5))
    _cd_diagram(ax, ranks, cd)
    ax.set_title(f"Holdout: diferença crítica (Nemenyi, α=0,05, N={n} famílias, "
                 f"Friedman p={p:.2g})", fontsize=11)
    _save(fig, out, "fig13_holdout_cd")
    return {"ranks": ranks.to_dict(), "cd": float(cd), "friedman_p": float(p), "n": int(n)}


def fig_gains(table: pd.DataFrame, out: Path) -> None:
    t = table[table.reference == "fixed_h27"].sort_values("gain")
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    y = np.arange(len(t))
    colors = [CAT[1] if m in {"search_shards", "search_retrain", "pythia_like"} else
              CAT[0] if m not in {"fixed_nb", "fixed_gnb"} else "#8a8a8a" for m in t.method]
    ax.hlines(y, t.ci_low * 100, t.ci_high * 100, color="#555", lw=1.2)
    ax.scatter(t.gain * 100, y, c=colors, s=45, zorder=3)
    ax.axvline(0, color="#444", lw=1)
    ax.set_yticks(y, t.label)
    ax.set_xlabel("Redução do arrependimento frente a H27 (p.p., IC95%)")
    ax.set_title("Holdout de famílias novas: métodos frente a H27", fontsize=11)
    ax.scatter([], [], c=CAT[0], label="seletores gratuitos")
    ax.scatter([], [], c=CAT[1], label="pagam ε pela escolha")
    ax.scatter([], [], c="#8a8a8a", label="opções fixas")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(axis="y", visible=False)
    _save(fig, out, "fig14_holdout_gains_vs_h27")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/openml-holdout-30-comparison"))
    parser.add_argument("--manifest", type=Path,
                        default=Path("data/openml_confirmatory_manifest_30_holdout.csv"))
    parser.add_argument("--dev-manifest", type=Path,
                        default=Path("data/openml_manifest_984_completed.csv"))
    parser.add_argument("--dev-conditions", type=Path,
                        default=Path("results/portfolio-984-current-slim/conditions.csv"))
    parser.add_argument("--figures", type=Path, default=Path("figures/openml-holdout-30"))
    parser.add_argument("--seed", type=int, default=ec.CONFIRMATORY_SEED)
    from dpmeta import selector_portfolio13 as sel
    parser.add_argument('--selector', choices=sel.SELECTORS, default='rf12')
    parser.add_argument('--conditions', type=Path, help='holdout conditions; skips shard aggregation')
    args = parser.parse_args(argv)
    if args.selector == 'gbm_residual_fallback':
        if args.output == Path('results/openml-holdout-30-comparison'):args.output=Path('results/dissertation-portfolio13/holdout')
        if args.figures == Path('figures/openml-holdout-30'):args.figures=Path('figures/GBM-13')
        if args.dev_conditions == Path('results/portfolio-984-current-slim/conditions.csv'):
            args.dev_conditions=Path('results/portfolio-984-nb8u/conditions_13.csv')
        from scripts.dissertation_analysis_portfolio13 import holdout
        holdout(args.dev_conditions, args.conditions or args.output / 'conditions_13.csv',
                args.dev_manifest, args.manifest, args.output, args.figures)
        return 0

    args.figures.mkdir(parents=True, exist_ok=True)
    conditions, baselines = merge_shards(args.output)
    holdout = _load_manifest(args.manifest)
    dev_manifest = _load_manifest(args.dev_manifest)
    dev_agg = pb._aggregate(pd.read_csv(args.dev_conditions), dev_manifest)
    test_agg = pb._aggregate(conditions, holdout)
    pb.validate_condition_grid(conditions, holdout[holdout.dataset_id.isin(conditions.dataset_id)],
                               epsilons=pb.DEFAULT_EPSILONS, repeats=3, pipelines=pb.PIPELINES)

    summary, predictions = primary(dev_agg, dev_manifest, test_agg, holdout, args.seed,
                                   args.output)
    builder = FeatureBuilder(pd.concat([dev_manifest, holdout], ignore_index=True))
    fitted = dev_selectors(dev_agg, dev_manifest, builder, args.seed)
    secondary = secondary_choices(test_agg, holdout, fitted, builder)
    long = method_table(test_agg, predictions, secondary, baselines)
    long.to_csv(args.output / "method_regret.csv", index=False)
    wide = per_family(long)
    table = comparisons(wide, args.seed)
    table.to_csv(args.output / "comparisons.csv", index=False)
    meta = holdout.drop_duplicates("dataset_id").set_index("dataset_id")
    inside = [d for d in conditions.dataset_id.unique() if meta.loc[d, "n_instances"] <= DEV_MAX_N]
    outside = [d for d in conditions.dataset_id.unique() if d not in inside]
    strata = {}
    for name, ids in (("dentro_da_faixa", inside), ("fora_da_faixa", outside)):
        if len(ids) >= 3:
            strata[name] = comparisons(per_family(long, ids), args.seed).to_dict("records")
    cd = fig_cd(wide, args.figures)
    fig_gains(table, args.figures)
    choice_share = (pd.concat([
        predictions.assign(method="selector", pipeline=predictions.selector_pipeline),
        secondary, baselines], ignore_index=True)[["method", "pipeline"]]
        .groupby("method").pipeline.value_counts(normalize=True).rename("share").reset_index())
    choice_share.to_csv(args.output / "choice_share.csv", index=False)
    result = {
        "primary": summary, "completed_datasets": int(conditions.dataset_id.nunique()),
        "completed_families": int(test_agg.family_id.nunique()),
        "inside_dev_n_range": len(inside), "outside_dev_n_range": len(outside),
        "secondary": table.to_dict("records"), "strata": strata, "nemenyi": cd,
    }
    if result["completed_families"] < 40:
        result["note"] = "menos de 40 famílias: resultado primário exploratório."
    (args.output / "summary.json").write_text(json.dumps(result, indent=2, default=float) + "\n")
    print(json.dumps({k: summary[k] for k in ("n_test_families", "mean_regret_gain",
                                              "gain_ci95_family_bootstrap",
                                              "paired_sign_flip_p_two_sided",
                                              "no_worse_median_q90_cvar90", "status")},
                     indent=1, default=float))
    print(table.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Regera as figuras e números que dependem da opção vencedora, com empates divididos.

Em ~13% das condições várias opções empatam no topo (em 443 delas, as 12, em geral
no acaso). As versões anteriores usavam ``idxmax``, que dava a vitória a quem vinha
primeiro na tabela: a Figura 2 favorecia o NB categórico e a Figura 7 o GNB zCDP.
Este script refaz só o que depende disso, sem reajustar metamodelos:

* Figura 2 (fig1_winner_dispersion_12opcoes) e a tabela de vitórias por ε;
* Figura 7 (contribuição ao portfólio), Figura 8 (mapa por n·ε/p) e Figura 12 (t-SNE);
* interdependência mecanismo-modelo, gravada em ``summary.json``.

    PYTHONPATH=. .venv/bin/python scripts/regen_tied_winner_figures_portfolio13.py
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

_spec = importlib.util.spec_from_file_location(
    "extras", Path(__file__).with_name("analyze_dissertation_extras_portfolio13.py"))
x = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(x)
pb = x.pb

PT = {  # rótulos da Figura 2 da dissertação
    "gnb__gaussian_zcdp_equal": "zCDP · GNB",
    "gnb__laplace_equal": "Laplace · GNB",
    "gnb__staircase_vector_equal": "Staircase · GNB",
    "gnb__analytic_gaussian_equal": "Gauss. analítico · GNB",
    "softmax__analytic_gaussian": "Gauss. analítico · Softmax",
    "softmax__dp_sgd": "DP-SGD · Softmax (25 passos)",
    "random_forest__laplace_parallel_shards_25": "Laplace · Árvores",
    "softmax__dp_sgd_steps100": "DP-SGD · Softmax (100 passos)",
    "softmax__dp_sgd_clip2_reg003": "DP-SGD · Softmax (clip 2, λ=0,03)",
    "random_forest__analytic_gaussian_parallel_shards": "Gauss. analítico · Árvores",
    "categorical_naive_bayes__laplace_histograms": "Laplace · NB categórico",
    "random_forest__exponential_majority_parallel_shards": "Exponencial · Árvores",
}


def br(v: float, d: int = 1) -> str:
    return f"{v:.{d}f}".replace(".", ",")


def fig_winners_pt(agg: pd.DataFrame, out: Path) -> pd.DataFrame:
    w = x.tied_winners(agg)
    n = agg.groupby(["dataset_id", "epsilon"]).ngroups
    order = list(x.PIPES)
    share = (w.groupby("pipeline").weight.sum() / n).reindex(order) * 100
    by_eps = (pd.crosstab(w.epsilon, w.pipeline, values=w.weight, aggfunc="sum",
                          normalize="index").reindex(columns=order).sort_index() * 100)
    regret = agg.groupby("pipeline").portfolio_regret.mean()
    best, worst = regret.idxmin(), regret.idxmax()
    colors = {p: plt.get_cmap("tab20")(i) for i, p in enumerate(order)}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1, 1.3]})
    bars = ax1.barh([PT[p] for p in order], share.values, color=[colors[p] for p in order])
    ax1.invert_yaxis()
    ax1.set_xlim(0, share.max() + 14)
    ax1.set(xlabel=f"Vitórias (%) em {n:,} condições conjunto–ε".replace(",", "."),
            title="(a) Nenhum par mecanismo-modelo domina")
    for bar, p, v in zip(bars, order, share.values):
        tag = ("  ← menor arrependimento médio" if p == best else
               "  ← maior arrependimento médio" if p == worst else "")
        ax1.text(v + 0.5, bar.get_y() + bar.get_height() / 2, f"{br(v)}%{tag}",
                 va="center", fontsize=9)
        if tag:
            bar.set_edgecolor("black")
            bar.set_linewidth(1.4)
    bottom = np.zeros(len(by_eps))
    labels = [br(e, 2).rstrip("0").rstrip(",") for e in by_eps.index]
    for p in order:
        ax2.bar(labels, by_eps[p].values, bottom=bottom, label=PT[p], color=colors[p], width=0.72)
        bottom += by_eps[p].values
    ax2.set(xlabel="Orçamento de privacidade ε", ylabel="Vitórias (%)", ylim=(0, 100),
            title="(b) A vencedora muda com ε")
    ax2.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8)
    x._save(fig, out, "fig1_winner_dispersion_12opcoes")
    table = by_eps.T.assign(Total=share)
    return table.loc[table.Total.sort_values(ascending=False).index]


def main() -> int:
    import argparse
    from dpmeta import selector_portfolio13 as sel
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selector', choices=sel.SELECTORS, default='rf12')
    parser.add_argument('--conditions', type=Path, default=Path('results/portfolio-984-current-slim/conditions.csv'))
    parser.add_argument('--manifest', type=Path, default=Path('data/openml_manifest_984_completed.csv'))
    parser.add_argument('--lodo', type=Path, default=Path('results/portfolio-984-shared-selector-lodo/outer_predictions.csv'))
    parser.add_argument('--output', type=Path, default=Path('results/portfolio-984-dissertation-extras'))
    parser.add_argument('--figures', type=Path, default=Path('figures/portfolio-984-dissertation-extras'))
    args = parser.parse_args()
    if args.selector == 'gbm_residual_fallback':
        if args.output == Path('results/portfolio-984-dissertation-extras'):args.output=Path('results/dissertation-portfolio13/extras')
        if args.figures == Path('figures/portfolio-984-dissertation-extras'):args.figures=Path('figures/GBM-13')
        if args.lodo == Path('results/portfolio-984-shared-selector-lodo/outer_predictions.csv'):
            args.lodo=Path('results/experiments/lodo_13/lodo_picks.csv')
    extras, figures = args.output, args.figures
    extras.mkdir(parents=True, exist_ok=True); figures.mkdir(parents=True, exist_ok=True)
    manifest = x._load_manifest(args.manifest)
    if args.selector == 'gbm_residual_fallback':
        agg, manifest, pipes = sel.prepare(pd.read_csv(args.conditions), manifest)
        x.PIPES, x.EXPANDED = pipes, True
        PT[pb.NB8U_PIPELINE] = 'Laplace · NB b8u'
        lodo = sel.from_lodo(args.lodo, agg)
    else:
        agg = pb._aggregate(pd.read_csv(args.conditions), manifest)
        lodo = pd.read_csv(args.lodo)
        lodo = lodo[lodo.feature_mode == x.MODE].copy()

    table = fig_winners_pt(agg, figures)
    table.round(1).to_csv(extras / "win_share_by_epsilon.csv")
    print(table.round(1).to_string())

    contrib, corr = x.portfolio_contribution(agg)
    contrib.to_csv(extras / "portfolio_contribution.csv", index=False)
    x.fig_redundancy(corr, contrib, figures)
    print(contrib[["option", "win_share", "oracle_increase"]].round(4).to_string(index=False))

    regime = x.regime_frame(agg, lodo, manifest)
    bins = x.winner_map_bins(regime)
    bins.to_csv(extras / "winner_map_bins.csv")
    x.fig_regime(regime, figures)
    print(bins.iloc[[0, -1]].T.to_string())

    summary_path = extras / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["dp_changes_best_model"] = x.dp_changes_best_model(agg)
    summary["tie_handling"] = "empates no topo dividem a vitória (regen_tied_winner_figures_portfolio13.py)"
    summary_path.write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps(summary["dp_changes_best_model"], indent=1, default=float))

    # t-SNE: mesmas features e semente de analyze_dissertation_extras_portfolio13.py; muda só o rótulo
    builder = x.FeatureBuilder(manifest)
    per_dataset = x.most_frequent_winner(agg)
    feats = np.vstack([builder.row(d, d, d, 1.0, pb.PIPELINES[0])[:len(x.PUBLIC_FEATURE_NAMES)]
                       for d in per_dataset.index])
    keep = [i for i, name in enumerate(x.PUBLIC_FEATURE_NAMES) if "epsilon" not in name]
    emb = TSNE(perplexity=30, random_state=20260917, init="pca").fit_transform(
        StandardScaler().fit_transform(feats[:, keep]))
    x.fig_tsne(emb, per_dataset, figures)
    print(per_dataset.value_counts().to_string())

    # vencedora única e constante em todos os orçamentos
    w = x.tied_winners(agg)
    unique = w[w.weight == 1.0]
    constant = unique.groupby("dataset_id").filter(lambda f: len(f) == 8 and f.pipeline.nunique() == 1)
    print("conjuntos com a mesma vencedora única nos 8 orçamentos:", constant.dataset_id.nunique())
    print("condições com empate no topo:", float((w.groupby(["dataset_id", "epsilon"]).size() > 1).mean()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

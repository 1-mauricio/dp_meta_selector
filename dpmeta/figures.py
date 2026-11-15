"""Publication figures from a portfolio-benchmark run directory.

Ported from dp_meta_selector_v2 (generate_publication_figures / generate_h1_artifacts).
Needs ``conditions.csv``; the selector figures (regret/top-1 curves, CD diagram,
paired gain/ECDF) also need ``outer_predictions.csv``.

    python -m dpmeta.figures --run results/portfolio-250 --out figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import friedmanchisquare, studentized_range  # noqa: E402

from .portfolio_benchmark import (  # noqa: E402
    LEGACY_PIPELINE_IDS, PIPELINE_DISPLAY, PIPELINES,
)

plt.rcParams.update({"font.size": 11, "axes.labelsize": 12, "axes.titlesize": 12,
                     "legend.fontsize": 9, "axes.grid": True, "grid.alpha": 0.3,
                     "pdf.fonttype": 42, "ps.fonttype": 42})

PIPELINE_LABELS = dict(PIPELINE_DISPLAY)
PIPELINE_LABELS["gnb__staircase_equal"] = (
    "Staircase por coordenada · GaussianNB (calibração histórica)"
)
_TAB = plt.get_cmap("tab20")
PIPELINE_COLORS = {p: _TAB(i) for i, p in enumerate(PIPELINES)}


def _pipeline_order(cond: pd.DataFrame) -> list[str]:
    """Order the pipelines present in the data, not the ones in the current code.

    A conditions file may predate a change to ``PIPELINES``; filtering by the
    current tuple would silently drop candidates from the figure.
    """
    present = set(cond.pipeline)
    order = [p for p in PIPELINES if p in present]
    order += sorted(present - set(order))
    taken = {PIPELINE_COLORS[p] for p in order if p in PIPELINE_COLORS}
    spare = (c for c in (_TAB(i) for i in range(20)) if c not in taken)
    for pipeline in order:
        if pipeline not in PIPELINE_COLORS:
            PIPELINE_COLORS[pipeline] = next(spare, (0.4, 0.4, 0.4, 1.0))
        PIPELINE_LABELS.setdefault(pipeline, pipeline)
    return order

DEPLOYABLE_MODES = ("public_schema_only", "public_plus_dp")
MODE_LABELS = {"public_schema_only": "Seletor (schema)",
               "public_plus_dp": "Seletor (schema + calibração)",
               "dp_research_rich": "Seletor (meta-features DP)"}
BASELINES = {"oracle_regret": ("Oráculo (teto)", "#2ca02c", "--", "s"),
             "best_fixed_regret": ("Melhor opção fixa", "#ff7f0e", "-", "D"),
             "h27_fixed_regret": ("DP-SGD · Softmax (clip 2, λ=0,03)", "#8c564b", "--", "X"),
             "random_pipeline_regret": ("Opção aleatória", "#17becf", "--", "P"),
             "worst_fixed_regret": ("Pior opção fixa", "#d62728", ":", "D"),
             "worst_pipeline_regret": ("Pior opção", "#000000", ":", "v"),
             "gnb_fixed_regret": ("Gaussiano (zCDP) · GaussianNB", "#7f7f7f", ":", "x")}
MODE_STYLE = {"public_schema_only": ("#9467bd", "-.", "^"),
              "public_plus_dp": ("#1f77b4", "-", "o"),
              "dp_research_rich": ("#d62728", "-", "o")}


def _save(fig, out: Path, name: str) -> None:
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"wrote {out / name}.(png|pdf)")


def _condition_table(conditions: pd.DataFrame) -> pd.DataFrame:
    """Mean over repetitions per (dataset, epsilon, pipeline)."""
    conditions = conditions[~conditions.pipeline.isin(LEGACY_PIPELINE_IDS)]
    return (conditions.groupby(["dataset_id", "epsilon", "pipeline"], as_index=False)
            [["private_ba", "portfolio_regret"]].mean())


def _eps_labels(eps) -> list[str]:
    return [f"{e:g}" for e in eps]


def fig_winners(cond: pd.DataFrame, out: Path) -> None:
    """Which pipeline attains the highest private accuracy per (dataset, ε)."""
    keys = ["dataset_id", "epsilon"]
    top = cond.groupby(keys).private_ba.transform("max")
    winners = cond[(top - cond.private_ba).abs() < 1e-12]
    weight = 1.0 / winners.groupby(keys).private_ba.transform("size")  # empate divide a vitória
    n_conditions = cond.groupby(keys).ngroups
    order = _pipeline_order(cond)
    share = (weight.groupby(winners.pipeline).sum() / n_conditions).reindex(order, fill_value=0) * 100
    by_eps = (pd.crosstab(winners.epsilon, winners.pipeline, values=weight, aggfunc="sum",
                          normalize="index").reindex(columns=order, fill_value=0).sort_index() * 100)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1, 1.3]})
    bars = ax1.barh([PIPELINE_LABELS[p] for p in order], share.values,
                    color=[PIPELINE_COLORS[p] for p in order])
    ax1.invert_yaxis()
    ax1.set_xlim(0, max(35, share.max() + 6))
    ax1.set(xlabel=f"Win share (%) over {n_conditions:,} dataset–ε conditions",
            title="(a) No universal mechanism–model winner")
    mean_regret = cond.groupby("pipeline").portfolio_regret.mean().reindex(order)
    best, worst = mean_regret.idxmin(), mean_regret.idxmax()
    for bar, (pipeline, v) in zip(bars, zip(order, share.values)):
        tag = ""
        if pipeline == best:
            tag = "  ← best mean regret"
        elif pipeline == worst:
            tag = "  ← worst mean regret"
        ax1.text(v + 0.6, bar.get_y() + bar.get_height() / 2, f"{v:.1f}%{tag}",
                 va="center", fontsize=9)
        if tag:
            bar.set_edgecolor("black")
            bar.set_linewidth(1.4)
    bottom = np.zeros(len(by_eps))
    for p in order:
        ax2.bar(_eps_labels(by_eps.index), by_eps[p].values, bottom=bottom,
                label=PIPELINE_LABELS[p], color=PIPELINE_COLORS[p], width=0.72)
        bottom += by_eps[p].values
    ax2.set(xlabel=r"Privacy budget $\varepsilon$", ylabel="Win share (%)", ylim=(0, 100),
            title=r"(b) Winner shifts across $\varepsilon$")
    ax2.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    _save(fig, out, "fig1_winner_dispersion")


def fig_accuracy_range(cond: pd.DataFrame, out: Path) -> None:
    """H1: spread of private accuracy between best and worst pipeline per condition."""
    g = cond.groupby(["dataset_id", "epsilon"]).private_ba
    rng = ((g.max() - g.min()) * 100).rename("range").reset_index()
    eps = sorted(rng.epsilon.unique())
    x = np.arange(len(eps))
    by_eps = rng.groupby("epsilon").range.agg(["mean", "median"]).reindex(eps)
    high = rng.groupby("epsilon").range.apply(lambda s: (s >= 10).mean() * 100).reindex(eps)
    median, p90 = rng.range.median(), rng.range.quantile(0.9)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4.7))
    ax1.hist(rng.range, bins=np.arange(0, rng.range.max() + 2, 2), color="#4C78A8",
             edgecolor="white")
    ax1.axvline(median, color="k", label=f"Median: {median:.2f} p.p.")
    ax1.axvline(p90, color="#E45756", ls="--", label=f"P90: {p90:.2f} p.p.")
    ax1.set(xlabel="Best − worst option balanced accuracy (p.p.)", ylabel="Dataset–ε conditions",
            title="(a) Utility spread across mechanism–model options")
    ax1.legend()
    ax2.plot(x, by_eps["mean"], marker="o", lw=2.2, label="Mean")
    ax2.plot(x, by_eps["median"], marker="s", lw=2.2, label="Median")
    ax2.set(xticks=x, xticklabels=_eps_labels(eps), xlabel=r"Privacy budget $\varepsilon$",
            ylabel="Accuracy spread (p.p.)", title="(b) Typical spread by ε")
    ax2.legend()
    ax3.bar(x, high, color="#E45756", width=0.65)
    for xi, v in zip(x, high):
        ax3.text(xi, v + 0.5, f"{v:.1f}", ha="center", fontsize=9)
    ax3.set(xticks=x, xticklabels=_eps_labels(eps), xlabel=r"Privacy budget $\varepsilon$",
            ylabel="Conditions with spread ≥ 10 p.p. (%)", ylim=(0, max(20, high.max() + 5)),
            title="(c) Large option effects")
    _save(fig, out, "fig_h1_accuracy_range")


def _regret_methods(preds: pd.DataFrame, conditions: pd.DataFrame) -> pd.DataFrame:
    """Per (dataset, ε, method) regret and mode-agnostic baselines.

    ``Random option`` is the expected regret of a uniform draw from the
    candidate portfolio. ``Worst option`` is the largest regret observed in
    that dataset–epsilon condition. ``Worst fixed option`` always plays the
    option with the largest mean regret over the whole run; like the oracle,
    it is a hindsight reference, never a method.
    """
    modes = [m for m in DEPLOYABLE_MODES if m in set(preds.feature_mode)]
    ref = preds[preds.feature_mode == modes[0]]
    rows = []
    for col in BASELINES:
        if col in ref:
            rows.append(ref.assign(method=col)[["dataset_id", "epsilon", "method"]].assign(
                regret=ref[col].to_numpy()))

    portfolio_baselines = (conditions.groupby(["dataset_id", "epsilon"], as_index=False)
                           .agg(
                               random_pipeline_regret=("portfolio_regret", "mean"),
                               worst_pipeline_regret=("portfolio_regret", "max"),
                           ))
    baseline_ref = ref[["dataset_id", "epsilon"]].merge(
        portfolio_baselines, on=["dataset_id", "epsilon"], how="left", validate="one_to_one",
    )
    worst = worst_fixed_pipeline(conditions)
    worst_ref = ref[["dataset_id", "epsilon"]].merge(
        conditions[conditions.pipeline == worst][["dataset_id", "epsilon", "portfolio_regret"]],
        on=["dataset_id", "epsilon"], how="left", validate="one_to_one",
    ).rename(columns={"portfolio_regret": "regret"})
    rows.append(worst_ref.assign(method="worst_fixed_regret"))
    for col in ("random_pipeline_regret", "worst_pipeline_regret"):
        rows.append(baseline_ref.assign(method=col)[
            ["dataset_id", "epsilon", "method", col]
        ].rename(columns={col: "regret"}))
    for m in modes:
        s = preds[preds.feature_mode == m]
        rows.append(s.assign(method=m, regret=s.selector_regret)[
            ["dataset_id", "epsilon", "method", "regret"]])
    return pd.concat(rows, ignore_index=True), modes, ref


def worst_fixed_pipeline(conditions: pd.DataFrame) -> str:
    """Option with the largest mean regret, equal weight per dataset and ε."""
    return (conditions.groupby(["dataset_id", "pipeline"]).portfolio_regret.mean()
            .groupby("pipeline").mean().idxmax())


def fig_regret_top1(preds: pd.DataFrame, conditions: pd.DataFrame, out: Path) -> None:
    long, modes, ref = _regret_methods(preds, conditions)
    worst = worst_fixed_pipeline(conditions)
    worst_label = f"Pior opção fixa ({PIPELINE_LABELS.get(worst, worst)})"
    best_label = BASELINES["best_fixed_regret"][0]
    if ref.best_fixed_pipeline.nunique() == 1:
        best = ref.best_fixed_pipeline.iloc[0]
        best_label += f" ({PIPELINE_LABELS.get(best, best)})"
    eps = sorted(long.epsilon.unique())
    x = np.arange(len(eps))
    reg = long.groupby(["method", "epsilon"]).regret.mean().unstack().reindex(columns=eps)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    for col, (label, color, ls, mk) in BASELINES.items():
        if col not in reg.index:
            continue
        label = {"worst_fixed_regret": worst_label, "best_fixed_regret": best_label}.get(col, label)
        ax1.plot(x, reg.loc[col] * 100, color=color, ls=ls, marker=mk, lw=1.6, label=label)
    for m in modes:
        color, ls, mk = MODE_STYLE[m]
        ax1.plot(x, reg.loc[m] * 100, color=color, ls=ls, marker=mk, lw=2.4, label=MODE_LABELS[m])
        s = preds[preds.feature_mode == m].groupby("epsilon").selector_top1.mean().reindex(eps)
        ax2.plot(x, s * 100, color=color, ls=ls, marker=mk, lw=2.4, label=MODE_LABELS[m])
    fixed = ref.groupby("epsilon").fixed_top1.mean().reindex(eps)
    ax2.plot(x, fixed * 100, color=BASELINES["best_fixed_regret"][1], marker="D", lw=1.6,
             label=best_label)
    if "h27_top1" in ref and ref.h27_top1.notna().any():
        h27 = ref.groupby("epsilon").h27_top1.mean().reindex(eps)
        ax2.plot(x, h27 * 100, color=BASELINES["h27_fixed_regret"][1],
                 ls="--", marker="X", lw=1.6, label=BASELINES["h27_fixed_regret"][0])
    # Top-1 of the worst fixed option: share of conditions in which it is the winner.
    winners = conditions.loc[conditions.groupby(["dataset_id", "epsilon"]).private_ba.idxmax()]
    worst_top1 = ref[["dataset_id", "epsilon"]].merge(
        winners[["dataset_id", "epsilon", "pipeline"]], on=["dataset_id", "epsilon"],
        how="left", validate="one_to_one",
    ).assign(hit=lambda d: d.pipeline.eq(worst)).groupby("epsilon").hit.mean().reindex(eps)
    _, color, ls, mk = BASELINES["worst_fixed_regret"]
    ax2.plot(x, worst_top1 * 100, color=color, ls=ls, marker=mk, lw=1.6, label=worst_label)
    for ax, ylabel, title in ((ax1, "Arrependimento médio (p.p. de acurácia balanceada)",
                               r"(a) Arrependimento por $\varepsilon$"),
                              (ax2, "Acerto Top-1 (%)", r"(b) Acerto Top-1 por $\varepsilon$")):
        ax.set(xticks=x, xticklabels=_eps_labels(eps),
               xlabel=r"$\varepsilon$ do modelo (orçamento total igual para os seletores públicos)",
               ylabel=ylabel, title=title)
    # Every curve of panel (b) also appears in (a): one shared legend below both.
    fig.tight_layout()
    fig.legend(*ax1.get_legend_handles_labels(), loc="upper center",
               bbox_to_anchor=(0.5, 0.0), ncol=3, fontsize=9.5)
    _save(fig, out, "fig2_regret_and_top1_curves")


def _cd_diagram(ax, ranks: pd.Series, cd: float) -> None:
    ranks = ranks.sort_values()
    k, lo, hi = len(ranks), ranks.min() - 0.3, ranks.max() + 0.3
    ax.set_xlim(lo, hi)
    ax.axhline(0, color="k", lw=1)
    for r in np.arange(np.ceil(lo), np.floor(hi) + 1):
        ax.plot([r, r], [0, 0.08], color="k", lw=1)
        ax.text(r, 0.14, f"{r:g}", ha="center", fontsize=9)
    half = (k + 1) // 2
    for i, (name, r) in enumerate(ranks.items()):
        left = i < half
        h = 0.55 + 0.4 * (i if left else k - 1 - i)
        edge = lo if left else hi
        ax.plot([r, r, edge], [0, h, h], color="k", lw=0.8)
        ax.text(edge + (-0.05 if left else 0.05), h, f"{name} ({r:.2f})", va="center",
                ha="right" if left else "left", fontsize=9)
    # maximal groups of methods whose ranks differ by less than the CD
    vals, level, last_end = ranks.to_numpy(), 0, -1
    for i in range(k):
        j = max(j for j in range(i, k) if vals[j] - vals[i] < cd)
        if j > i and j > last_end:
            y = -0.15 - 0.13 * level
            ax.plot([vals[i] - 0.03, vals[j] + 0.03], [y, y], color="#d62728", lw=3)
            level, last_end = level + 1, j
    ax.plot([lo + 0.02, lo + 0.02 + cd], [-0.15 - 0.13 * (level + 1)] * 2, color="k", lw=2)
    ax.text(lo + 0.02 + cd / 2, -0.15 - 0.13 * (level + 1) - 0.1, f"CD = {cd:.2f}",
            ha="center", va="top", fontsize=9)
    ax.set_ylim(-0.15 - 0.13 * (level + 3), 0.55 + 0.4 * half)
    ax.axis("off")


def _inferential_methods(long: pd.DataFrame, modes: list[str]) -> pd.DataFrame:
    """Exclude hindsight/expected-reference curves from method inference."""
    eligible = set(modes) | {
        "best_fixed_regret", "h27_fixed_regret", "gnb_fixed_regret",
    }
    return long[long.method.isin(eligible)].copy()


def fig_cd(preds: pd.DataFrame, conditions: pd.DataFrame, out: Path) -> None:
    """Critical-difference diagram for executable, preselected methods only."""
    long, modes, _ = _regret_methods(preds, conditions)
    long = _inferential_methods(long, modes)
    names = {**{c: v[0] for c, v in BASELINES.items()}, **MODE_LABELS}
    wide = (long.groupby(["dataset_id", "method"]).regret.mean().unstack()
            .dropna(axis=1, how="all").dropna())
    wide = wide.rename(columns=names)
    n, k = wide.shape
    ranks = wide.rank(axis=1).mean()
    cd = studentized_range.ppf(0.95, k, np.inf) / np.sqrt(2) * np.sqrt(k * (k + 1) / (6 * n))
    p = friedmanchisquare(*[wide[c].to_numpy() for c in wide]).pvalue if k >= 3 else float("nan")
    fig, ax = plt.subplots(figsize=(11, 4.2))
    _cd_diagram(ax, ranks, cd)
    # p can underflow to exactly 0.0; printing "p=0" would be wrong.
    p_text = f"p={p:.2g}" if p > 0 else "p < 1e-300"
    ax.set_title(f"Critical difference on mean regret (Nemenyi, α=0.05, N={n:,} datasets, "
                 f"Friedman {p_text})", fontsize=11)
    _save(fig, out, "fig3_critical_difference_nemenyi")


def fig_paired_and_ecdf(preds: pd.DataFrame, conditions: pd.DataFrame, out: Path) -> None:
    long, modes, _ = _regret_methods(preds, conditions)
    wide = (long.groupby(["dataset_id", "method"]).regret.mean().unstack()
            .dropna(axis=1, how="all"))
    primary = "public_plus_dp" if "public_plus_dp" in modes else modes[-1]
    wide = wide.dropna(subset=[primary])
    fixed_reference = ("h27_fixed_regret" if "h27_fixed_regret" in wide
                       and wide.h27_fixed_regret.notna().any() else "best_fixed_regret")
    fixed_label = "H27 fixed" if fixed_reference == "h27_fixed_regret" else "Best fixed"
    diff = (wide[fixed_reference] - wide[primary]) * 100
    lim = max(10, np.ceil(diff.abs().quantile(0.99)))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.hist(diff.clip(-lim, lim), bins=np.linspace(-lim, lim, 61), color="#1f77b4", alpha=0.8,
             edgecolor="black", linewidth=0.5)
    ax1.axvline(0, color="k", ls="--")
    ax1.axvline(diff.median(), color="#d62728", lw=2, label=f"Median: {diff.median():+.2f} p.p.")
    w, t, l = (diff > 0).sum(), (diff == 0).sum(), (diff < 0).sum()
    ax1.text(0.03, 0.95, f"Wins {w} ({w / len(diff):.0%})\nTies {t} ({t / len(diff):.0%})\n"
             f"Losses {l} ({l / len(diff):.0%})", transform=ax1.transAxes, va="top",
             bbox=dict(boxstyle="round", fc="white", ec="#aaa"))
    ax1.set(xlabel=f"Regret reduction: {fixed_label} − {MODE_LABELS[primary]} (p.p.)",
            ylabel="Datasets", title=f"(a) Paired regret advantage (N={len(diff):,})")
    ax1.legend(loc="upper right")
    for col in wide.columns:
        label, color, ls, _ = (BASELINES[col] if col in BASELINES else
                               (MODE_LABELS[col], *MODE_STYLE[col][:2], None))
        v = np.sort(wide[col].to_numpy()) * 100
        ax2.plot(v, np.arange(1, len(v) + 1) / len(v), color=color, ls=ls, label=label,
                 lw=2.2 if col in MODE_LABELS else 1.5)
    ax2.set(xlabel="Mean regret per dataset (p.p.)", ylabel="ECDF", ylim=(0, 1.02),
            title=f"(b) Regret ECDF across {len(wide):,} datasets")
    ax2.legend(loc="lower right")
    _save(fig, out, "fig4_regret_distribution_and_headroom")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="portfolio_benchmark output dir")
    parser.add_argument("--conditions", type=Path,
                        help="conditions.csv original quando --run contém só nova análise")
    parser.add_argument("--out", type=Path, default=Path("figures"))
    from dpmeta import selector_portfolio13 as sel
    parser.add_argument('--selector', choices=sel.SELECTORS, default='rf12')
    parser.add_argument('--manifest', type=Path, default=Path('data/openml_manifest_984_completed.csv'))
    parser.add_argument('--lodo', type=Path, default=Path('results/experiments/lodo_13/lodo_picks.csv'))
    args = parser.parse_args(argv)
    if args.selector == 'gbm_residual_fallback':
        if args.out == Path('figures'):args.out=Path('figures/GBM-13')
        if args.conditions is None:args.conditions=Path('results/portfolio-984-nb8u/conditions_13.csv')
        from scripts.dissertation_analysis_portfolio13 import publication
        publication(args.conditions, args.manifest, args.run, args.out, args.lodo)
        return 0

    args.out.mkdir(parents=True, exist_ok=True)
    cond = _condition_table(pd.read_csv(
        args.conditions or args.run / "conditions.csv",
        usecols=("dataset_id", "epsilon", "pipeline", "private_ba", "portfolio_regret"),
    ))
    fig_winners(cond, args.out)
    fig_accuracy_range(cond, args.out)
    pred_path = args.run / "outer_predictions.csv"
    if not pred_path.exists():
        print(f"skipping selector figures: {pred_path} not found (run finished?)")
        return 0
    preds = pd.read_csv(pred_path)
    for fig in (fig_regret_top1, fig_cd, fig_paired_and_ecdf):
        fig(preds, cond, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

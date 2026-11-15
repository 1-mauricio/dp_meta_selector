"""Análises complementares do seletor único ``public_plus_dp`` para a dissertação.

Parte A reaproveita o LODO e as condições (sem reajuste):

* seletor contra cada uma das 12 opções fixas (V/E/D, Wilcoxon com Holm);
* ganho e posto médio por estrato (classes, tipo de atributo, dimensão, regime de ε);
* contribuição de cada opção ao oráculo e redundância entre opções;
* frequência com que a DP muda o modelo vencedor (interdependência mecanismo-modelo);
* seletor gratuito contra uma busca privada idealizada que gasta metade do ε;
* mapa das vencedoras em função de n·ε/p.

Parte B reajusta o seletor nas mesmas 5 partições agrupadas e guarda as utilidades
previstas de todas as opções:

* regret@k e acerto top-k (recomendação em lista);
* importância por permutação de cada meta-feature;
* decomposição de Shapley por grandeza pública primitiva (n, ε, classes,
  estrutura dos atributos), recalculando todas as features derivadas;
* metamodelos alternativos (inclusive a formulação por classificação);
* projeção t-SNE do espaço de meta-features e latência da recomendação.

    PYTHONPATH=. .venv/bin/python scripts/analyze_dissertation_extras_portfolio13.py
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import math
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.cluster.hierarchy import leaves_list, linkage  # noqa: E402
from scipy.stats import friedmanchisquare, studentized_range, wilcoxon  # noqa: E402
from sklearn.ensemble import (  # noqa: E402
    ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor,
)
from sklearn.linear_model import Ridge  # noqa: E402
from sklearn.manifold import TSNE  # noqa: E402
from sklearn.neighbors import KNeighborsRegressor  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

import dpmeta.portfolio_benchmark as pb  # noqa: E402
from dpmeta.figures import _cd_diagram, _save  # noqa: E402
from dpmeta.openml_benchmark import _load_manifest  # noqa: E402
from dpmeta.provenance import file_sha256  # noqa: E402
from dpmeta.schema import PUBLIC_FEATURE_NAMES, public_features  # noqa: E402

from dpmeta import dissertation_selector as sel

PIPES = pb.PIPELINES
EXPANDED = False
MODE = "public_plus_dp"
FAMILY = {  # modelo (e via de privatização da softmax) de cada opção
    "gnb": "GaussianNB", "softmax__analytic_gaussian": "Softmax · saída",
    "softmax__dp_sgd": "Softmax · DP-SGD", "random_forest": "Árvores",
    "categorical_naive_bayes": "NB categórico",
}
FAMILY_ORDER = ("NB categórico", "Softmax · DP-SGD", "Softmax · saída", "GaussianNB", "Árvores")
# Paleta categórica de referência (ordem fixa; validada para séries adjacentes).
CAT = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4")
MUTED = "#8a8a8a"
SHORT = {
    pb.NB8U_PIPELINE: "Laplace · NB b8u",
    "gnb__gaussian_zcdp_equal": "Gauss. zCDP · GNB",
    "gnb__laplace_equal": "Laplace · GNB",
    "gnb__staircase_vector_equal": "Staircase · GNB",
    "gnb__analytic_gaussian_equal": "Gauss. analítico · GNB",
    "softmax__analytic_gaussian": "Gauss. analítico · Softmax",
    "softmax__dp_sgd": "DP-SGD 25 passos",
    "random_forest__laplace_parallel_shards_25": "Laplace · árvores",
    "softmax__dp_sgd_steps100": "DP-SGD 100 passos",
    "softmax__dp_sgd_clip2_reg003": "DP-SGD clip 2",
    "random_forest__analytic_gaussian_parallel_shards": "Gauss. analítico · árvores",
    "categorical_naive_bayes__laplace_histograms": "Laplace · NB categórico",
    "random_forest__exponential_majority_parallel_shards": "Exponencial · árvores",
}
SHORT.update({'categorical_naive_bayes__laplace_histograms_b8u': 'Laplace · NB b8u', 'softmax__dp_sgd_clip2_lamrule': 'DP-SGD · Softmax (regra λ)', 'softmax__dp_sgd_clip2_lam0003': 'DP-SGD · Softmax (λ=0,0003)'})
PRIMITIVES = ("n", "epsilon", "classes", "attributes")
PRIMITIVE_LABELS = {"n": "Número de instâncias (n)", "epsilon": "Orçamento (ε)",
                    "classes": "Número de classes", "attributes": "Estrutura dos atributos"}


def family(pipeline: str) -> str:
    for prefix, name in FAMILY.items():
        if pipeline.startswith(prefix):
            return name
    raise ValueError(pipeline)


def tied_winners(frame: pd.DataFrame, value: str = "private_ba") -> pd.DataFrame:
    """Todas as opções empatadas no topo de cada condição, com peso 1/k.

    Em ~13% das condições várias opções empatam (em geral todas no acaso);
    ``idxmax`` daria a vitória a quem vem primeiro na tabela.
    """
    keys = ["dataset_id", "epsilon"]
    top = frame.groupby(keys)[value].transform("max")
    winners = frame[(top - frame[value]).abs() < 1e-12].copy()
    winners["weight"] = 1.0 / winners.groupby(keys)[value].transform("size")
    return winners


def ci(values, seed: int) -> dict:
    mean, low, high = pb._bootstrap_ci(np.asarray(values, dtype=float), seed)
    return {"mean": float(mean), "ci_low": float(low), "ci_high": float(high)}


def fam_mean(frame: pd.DataFrame, column: str) -> pd.Series:
    return frame.groupby(["family_id", "dataset_id"])[column].mean().groupby("family_id").mean()


# ---------------------------------------------------------------- parte A


def vs_each_option(agg: pd.DataFrame, lodo: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Seletor (LODO) contra cada opção fixa; positivo: seletor melhor."""
    selector = fam_mean(lodo, "selector_regret")
    rows, pvals = [], {}
    for index, pipeline in enumerate(PIPES):
        option = fam_mean(agg[agg.pipeline == pipeline], "portfolio_regret")
        diff = (option - selector.reindex(option.index)).to_numpy()
        tie = np.abs(diff) < 1e-9
        p = wilcoxon(diff[~tie]).pvalue
        pvals[pipeline] = p
        rows.append({"pipeline": pipeline, "option": SHORT[pipeline],
                     "option_regret": float(option.mean()),
                     **ci(diff, seed + index), "wins": int((diff > 0)[~tie].sum()),
                     "ties": int(tie.sum()), "losses": int((diff < 0)[~tie].sum()),
                     "wilcoxon_p": float(p)})
    table = pd.DataFrame(rows)
    holm = pb._holm_adjust(pvals)
    table["wilcoxon_p_holm"] = table.pipeline.map(holm)
    return table.sort_values("option_regret")


def dataset_strata(manifest: pd.DataFrame) -> pd.DataFrame:
    m = manifest.drop_duplicates("dataset_id").set_index("dataset_id")
    kind = np.where(m.n_nominal_features == 0, "só numéricos",
                    np.where(m.n_nominal_features == m.n_features, "só nominais", "mistos"))
    tercile = pd.qcut(m.n_encoded_features, 3, labels=["p baixo", "p médio", "p alto"])
    return pd.DataFrame({
        "Classes": np.where(m.n_classes == 2, "binário", "multiclasse"),
        "Atributos": kind, "Dimensão": tercile.astype(str),
    }, index=m.index)


def eps_regime(eps: pd.Series) -> pd.Series:
    return pd.cut(eps, [0, 0.5, 2, 100], labels=["ε ≤ 0,5", "1 ≤ ε ≤ 2", "ε ≥ 4"]).astype(str)


def strata_tables(agg: pd.DataFrame, lodo: pd.DataFrame, strata: pd.DataFrame,
                  seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    lodo = lodo.join(strata, on="dataset_id").assign(**{"Orçamento": eps_regime(lodo.epsilon)})
    gains = []
    for column in (*strata.columns, "Orçamento"):
        for value, block in lodo.groupby(column):
            diff = fam_mean(block, "best_fixed_regret") - fam_mean(block, "selector_regret")
            gains.append({"dimension": column, "stratum": value, "datasets": int(block.dataset_id.nunique()), "families": len(diff),
                          **ci(diff.to_numpy(), seed + len(gains))})
    # posto médio de cada opção e do seletor por condição, depois por estrato
    wide = agg.pivot_table(index=["dataset_id", "epsilon"], columns="pipeline",
                           values="portfolio_regret")[list(PIPES)]
    sel = lodo.set_index(["dataset_id", "epsilon"]).selector_regret
    wide["selector"] = sel.reindex(wide.index)
    ranks = wide.rank(axis=1).reset_index()
    ranks = ranks.join(strata, on="dataset_id").assign(
        **{"Orçamento": eps_regime(ranks.epsilon)})
    heat = {}
    for column in (*strata.columns, "Orçamento"):
        for value, block in ranks.groupby(column):
            per_dataset = block.groupby("dataset_id")[[*PIPES, "selector"]].mean()
            if EXPANDED:
                fmap = agg[['dataset_id','family_id']].drop_duplicates().set_index('dataset_id').family_id
                heat[(column, value)] = per_dataset.groupby(fmap.reindex(per_dataset.index)).mean().mean()
            else:
                heat[(column, value)] = per_dataset.mean()
    return pd.DataFrame(gains), pd.DataFrame(heat)


def portfolio_contribution(agg: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    wide = agg.pivot_table(index=["dataset_id", "epsilon"], columns="pipeline",
                           values="portfolio_regret")[list(PIPES)]
    oracle = wide.min(axis=1)
    tied = (wide.sub(oracle, axis=0).abs() < 1e-12)
    share = tied.div(tied.sum(axis=1), axis=0).mean()  # empate divide a vitória
    rows = []
    for pipeline in PIPES:
        reduced = wide.drop(columns=pipeline).min(axis=1)
        increase = (reduced - oracle).groupby(level=0).mean()
        rows.append({"pipeline": pipeline, "option": SHORT[pipeline],
                     "win_share": float(share[pipeline]),
                     "oracle_increase": float(increase.mean()),
                     "datasets_hurt_over_1pp": int((increase > 0.01).sum())})
    corr = wide.rank(axis=1).corr(method="spearman")
    return pd.DataFrame(rows).sort_values("oracle_increase", ascending=False), corr


def dp_changes_best_model(agg: pd.DataFrame) -> dict:
    agg = agg.assign(model=agg.pipeline.map(
        lambda p: "Softmax" if p.startswith("softmax") else family(p)))
    keys = ["dataset_id", "epsilon"]
    clean, private = tied_winners(agg, "clean_ba"), tied_winners(agg, "private_ba")
    sets = pd.DataFrame({"clean": clean.groupby(keys).model.agg(set),
                         "private": private.groupby(keys).model.agg(set)})
    # mudou quando nenhum modelo vencedor sem ruído está entre os vencedores sob DP
    changed = pd.Series([not (c & p) for c, p in zip(sets.clean, sets.private)], sets.index)

    def share(w: pd.DataFrame) -> dict:
        return (w.groupby("model").weight.sum() / w.groupby(keys).ngroups).to_dict()

    return {"overall": float(changed.mean()),
            "by_epsilon": changed.groupby(level="epsilon").mean().to_dict(),
            "clean_winner_share": share(clean), "private_winner_share": share(private)}


def idealized_private_search(agg: pd.DataFrame, lodo: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Seletor em ε contra o oráculo e a melhor fixa treinados com ε/2 e ε/4.

    O oráculo em ε/f representa uma busca privada que treina cada opção com
    ε/f e acerta sempre a melhor. A seleção privada de Liu e Talwar custa cerca
    de 3ε; na grade, ε/2 favorece a busca e ε/4 a desfavorece.
    """
    oracle = agg.groupby(["family_id", "dataset_id", "epsilon"]).portfolio_regret.min()
    fixed = agg[agg.pipeline == lodo.best_fixed_pipeline.iloc[0]].set_index(
        ["family_id", "dataset_id", "epsilon"]).portfolio_regret
    rows = []
    for factor, epsilon in itertools.product((2, 4), sorted(lodo.epsilon.unique())):
        share = epsilon / factor
        if not np.isclose(oracle.index.get_level_values("epsilon"), share).any():
            continue
        sel = fam_mean(lodo[lodo.epsilon == epsilon], "selector_regret")
        for name, ref in (("oracle", oracle), ("best_fixed", fixed)):
            block = ref[np.isclose(ref.index.get_level_values("epsilon"), share)]
            right = block.groupby(level=["family_id", "dataset_id"]).mean().groupby(level=0).mean()
            diff = (right - sel.reindex(right.index)).to_numpy()
            rows.append({"epsilon": epsilon, "factor": factor, "reference": name,
                         "selector": float(sel.mean()), "reference_regret": float(right.mean()),
                         **ci(diff, seed + len(rows))})
    return pd.DataFrame(rows)


def near_chance_gain(agg: pd.DataFrame, lodo: pd.DataFrame, manifest: pd.DataFrame,
                     seed: int, margin: float = 0.02) -> pd.DataFrame:
    """Ganho (LODO) separando conjuntos cuja melhor BA limpa fica perto do acaso."""
    classes = manifest.drop_duplicates("dataset_id").set_index("dataset_id").n_classes
    clean_col = "best_clean_portfolio_ba_gbm13" if EXPANDED else "best_clean_portfolio_ba"
    clean = agg.groupby("dataset_id")[clean_col].mean()
    near = (clean - 1 / classes.reindex(clean.index)) <= margin
    rows = []
    for name, ids in (("todos", clean.index), ("informativos", clean.index[~near]),
                      ("próximos do acaso", clean.index[near])):
        block = lodo[lodo.dataset_id.isin(ids)]
        diff = fam_mean(block, "best_fixed_regret") - fam_mean(block, "selector_regret")
        rows.append({"subset": name, "datasets": int(block.dataset_id.nunique()), "families": len(diff), **ci(diff.to_numpy(), seed + len(rows))})
    return pd.DataFrame(rows)


def regime_frame(agg: pd.DataFrame, lodo: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    m = manifest.drop_duplicates("dataset_id").set_index("dataset_id")
    winners = tied_winners(agg)[["dataset_id", "epsilon", "pipeline", "weight"]]
    conditions = lodo[["dataset_id", "epsilon", "selector_pipeline"]].copy()
    conditions["log_ratio"] = np.log10(
        m.n_instances.reindex(conditions.dataset_id).to_numpy() * conditions.epsilon
        / m.n_encoded_features.reindex(conditions.dataset_id).to_numpy())
    conditions["bin"] = pd.qcut(conditions.log_ratio, 10)  # decis sobre condições, não sobre empates
    frame = winners.merge(conditions, on=["dataset_id", "epsilon"])
    frame["oracle_family"] = frame.pipeline.map(family)
    frame["selector_family"] = frame.selector_pipeline.map(family)
    return frame


# ---------------------------------------------------------------- parte B


class FeatureBuilder:
    """Recalcula as features de ``public_plus_dp`` a partir de grandezas primitivas."""

    def __init__(self, manifest: pd.DataFrame):
        m = manifest.drop_duplicates("dataset_id").set_index("dataset_id")
        self.n = m.n_instances.astype(int).to_dict()
        self.schema = {int(d): pb._cached_public_schema(raw)
                       for d, raw in m.public_schema_json.items()}
        self.cache: dict[tuple, np.ndarray] = {}

    def row(self, n_from: int, attrs_from: int, classes_from: int, epsilon: float,
            pipeline: str) -> np.ndarray:
        key = (n_from, attrs_from, classes_from, epsilon, pipeline)
        if key not in self.cache:
            schema = self.schema[attrs_from]
            if classes_from != attrs_from:
                schema = dataclasses.replace(schema, classes=self.schema[classes_from].classes)
            n = self.n[n_from]
            original = pipeline
            feature_pipes = pb.PIPELINES if EXPANDED else PIPES
            identity = pb.H27_PIPELINE if original.startswith('softmax__dp_sgd_clip2_lam') else original
            action = np.asarray([float(identity == p) for p in feature_pipes])
            parts = [public_features(n, schema, epsilon), action,
                     pb._public_dp_features(n, schema, epsilon, original)]
            if EXPANDED:
                parts.append(np.asarray([float(pipeline == p) for p in PIPES]))
            self.cache[key] = np.concatenate(parts)
        return self.cache[key]

    def matrix(self, test: pd.DataFrame, donors: dict | None = None) -> np.ndarray:
        donors = donors or {}
        ids = test.dataset_id.to_numpy(int)
        eps = test.epsilon.to_numpy(float)
        pick = {k: donors.get(k) for k in ("n", "attributes", "classes")}
        eps_donor = donors.get("epsilon")
        out = []
        for i, (d, e, p) in enumerate(zip(ids, eps, test.pipeline)):
            out.append(self.row(
                pick["n"].get(d, d) if pick["n"] else d,
                pick["attributes"].get(d, d) if pick["attributes"] else d,
                pick["classes"].get(d, d) if pick["classes"] else d,
                float(eps_donor[i]) if eps_donor is not None else float(e), p))
        return np.vstack(out)


def choose(test: pd.DataFrame, scores: np.ndarray) -> pd.DataFrame:
    """Opção de maior utilidade prevista por condição."""
    frame = test.assign(_score=scores)
    return frame.loc[frame.groupby(["dataset_id", "epsilon"])._score.idxmax()]


def refit(conditions: pd.DataFrame, manifest: pd.DataFrame, seed: int, sink: dict) -> dict:
    if EXPANDED:
        summary, predictions = sel.grouped(conditions, manifest, sink=sink)
        sink['_predictions'] = predictions
        return summary
    with threadpool_limits(limits=1):
        summary, _ = pb.evaluate_grouped(
            conditions, manifest, n_splits=5, seed=seed,
            selector_strategy="absolute_shared", feature_modes_override=(MODE,),
            progress=True, selector_sink=sink)
    return summary


def fold_blocks(agg: pd.DataFrame, sink: dict):
    order = {p: i for i, p in enumerate(PIPES)}
    for fold, entry in sorted((k,v) for k,v in sink.items() if isinstance(k,int)):
        test = agg[agg.dataset_id.isin(entry["test_ids"])].copy()
        test["_order"] = test.pipeline.map(order)
        test = test.sort_values(["epsilon", "dataset_id", "_order"]).reset_index(drop=True)
        train = agg[~agg.dataset_id.isin(entry["test_ids"])]
        model = entry["selectors"][MODE]
        model.set_params(n_jobs=pb.SELECTOR_N_JOBS)
        yield fold, train, test, model


def topk(test: pd.DataFrame, scores: np.ndarray, fixed_rank: list[str]) -> pd.DataFrame:
    frame = test.assign(_score=scores, _fixed=test.pipeline.map(
        {p: -i for i, p in enumerate(fixed_rank)}))
    rows = []
    for (family_id, dataset_id, epsilon), c in frame.groupby(
            ["family_id", "dataset_id", "epsilon"]):
        best = c.portfolio_regret.min()
        by_sel = c.sort_values("_score", ascending=False).portfolio_regret.to_numpy()
        by_fix = c.sort_values("_fixed", ascending=False).portfolio_regret.to_numpy()
        for k in range(1, len(c) + 1):
            rows.append((family_id, dataset_id, epsilon, k, by_sel[:k].min(),
                         by_fix[:k].min(), float(by_sel[:k].min() <= best + 1e-12),
                         float(by_fix[:k].min() <= best + 1e-12)))
    return pd.DataFrame(rows, columns=["family_id", "dataset_id", "epsilon", "k",
                                       "selector", "fixed", "selector_hit", "fixed_hit"])


def column_importance(test, model, matrix, base_choice, rng, repeats, names):
    """Permutação coluna a coluna, entre bases, no mesmo ε e na mesma opção."""
    n_opt = len(PIPES)
    base_regret = fam_mean(base_choice, "portfolio_regret")
    out = {}
    for column, (name, group) in enumerate(names):
        if group == "option":
            continue
        slots = matrix.reshape(-1, n_opt, matrix.shape[1])[:, :, column]
        if np.ptp(slots, axis=0).max() == 0:  # permutar não altera nada
            out[name] = (pd.Series(0.0, index=base_regret.index), 0.0)
            continue
        deltas, changed = [], []
        for _ in range(repeats):
            permuted = matrix.copy()
            for epsilon in test.epsilon.unique():
                rows = np.flatnonzero((test.epsilon == epsilon).to_numpy())
                block = permuted[rows].reshape(-1, n_opt, matrix.shape[1])
                block[:, :, column] = block[rng.permutation(len(block))][:, :, column]
                permuted[rows] = block.reshape(len(rows), -1)
            chosen = choose(test, model.predict(permuted))
            deltas.append(fam_mean(chosen, "portfolio_regret") - base_regret)
            changed.append(float((chosen.pipeline.to_numpy()
                                  != base_choice.pipeline.to_numpy()).mean()))
        out[name] = (sum(deltas) / repeats, float(np.mean(changed)))
    return out


def shapley_primitives(test, model, builder, base_choice, rng, repeats):
    """Valores de Shapley do aumento de regret ao trocar grandezas primitivas.

    Cada grandeza é trocada pela de outra base de teste (ε: pelo de outra
    condição), e todas as features derivadas são recalculadas.
    """
    ids = np.array(sorted(test.dataset_id.unique()))
    base_regret = fam_mean(base_choice, "portfolio_regret")
    conditions = test[["dataset_id", "epsilon"]].drop_duplicates()
    k = len(PRIMITIVES)
    phi = {p: pd.Series(0.0, index=base_regret.index) for p in PRIMITIVES}
    phi_changed = dict.fromkeys(PRIMITIVES, 0.0)
    value_all, changed_all = pd.Series(0.0, index=base_regret.index), 0.0
    for _ in range(repeats):
        donors = {p: dict(zip(ids, rng.permutation(ids))) for p in ("n", "attributes", "classes")}
        cond_eps = rng.permutation(conditions.epsilon.to_numpy())
        eps_map = dict(zip(zip(conditions.dataset_id, conditions.epsilon), cond_eps))
        eps_rows = np.asarray([eps_map[(d, e)] for d, e in zip(test.dataset_id, test.epsilon)])
        values, changed = {}, {}
        for size in range(k + 1):
            for subset in itertools.combinations(PRIMITIVES, size):
                if not subset:
                    values[subset], changed[subset] = 0.0 * base_regret, 0.0
                    continue
                chosen_donors = {p: donors[p] for p in subset if p != "epsilon"}
                if "epsilon" in subset:
                    chosen_donors["epsilon"] = eps_rows
                chosen = choose(test, model.predict(builder.matrix(test, chosen_donors)))
                values[subset] = fam_mean(chosen, "portfolio_regret") - base_regret
                changed[subset] = float((chosen.pipeline.to_numpy()
                                         != base_choice.pipeline.to_numpy()).mean())
        for p in PRIMITIVES:
            others = [q for q in PRIMITIVES if q != p]
            for size in range(k):
                weight = math.factorial(size) * math.factorial(k - size - 1) / math.factorial(k)
                for subset in itertools.combinations(others, size):
                    with_p = tuple(q for q in PRIMITIVES if q in subset or q == p)
                    phi[p] += weight * (values[with_p] - values[subset]) / repeats
                    phi_changed[p] += weight * (changed[with_p] - changed[subset]) / repeats
        value_all += values[PRIMITIVES] / repeats
        changed_all += changed[PRIMITIVES] / repeats
        builder.cache.clear()  # linhas permutadas quase não se repetem
    return phi, phi_changed, value_all, changed_all


def candidate_models(seed: int) -> dict:
    if EXPANDED:
        return {
            "GBM residual + fallback (GBM-13)": None,
            "RF (RF-12)": RandomForestRegressor(n_estimators=pb.SELECTOR_TREE_COUNT,
                min_samples_leaf=3, max_features=.9, max_depth=6, max_samples=.5,
                random_state=seed, n_jobs=8),
            "GBM abs": sel.gbm(),
            "Extra-Trees": ExtraTreesRegressor(500,min_samples_leaf=3,max_features=.9,
                                               random_state=seed,n_jobs=8),
            "k-NN (k=25)": make_pipeline(StandardScaler(),KNeighborsRegressor(25)),
            "Ridge (linear)": make_pipeline(StandardScaler(),Ridge(alpha=1.)),
        }
    return {
        "Floresta aleatória (atual)": None,
        "Extra-Trees": ExtraTreesRegressor(n_estimators=500, min_samples_leaf=3,
                                           max_features=0.9, random_state=seed, n_jobs=8),
        "Gradient boosting": HistGradientBoostingRegressor(random_state=seed),
        "k-NN (k=25)": make_pipeline(StandardScaler(), KNeighborsRegressor(25)),
        "Ridge (linear)": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "Classificador da vencedora": RandomForestClassifier(
            n_estimators=500, min_samples_leaf=3, max_depth=12, random_state=seed, n_jobs=8),
    }


def classifier_matrix(frame: pd.DataFrame, builder: FeatureBuilder) -> tuple[np.ndarray, pd.DataFrame]:
    """Uma linha por condição: schema + calibração de todas as opções lado a lado."""
    conditions = frame[["family_id", "dataset_id", "epsilon"]].drop_duplicates().reset_index(drop=True)
    schema_width = len(PUBLIC_FEATURE_NAMES)
    rows = []
    for d, e in zip(conditions.dataset_id, conditions.epsilon):
        parts = [builder.row(d, d, d, e, PIPES[0])[:schema_width]]
        parts += [builder.row(d, d, d, e, p)[schema_width + len(PIPES):]
                  for p in PIPES]
        rows.append(np.concatenate(parts))
    return np.vstack(rows), conditions


# ---------------------------------------------------------------- figuras


def fig_vs_options(table: pd.DataFrame, out: Path) -> None:
    t = table.sort_values("mean")
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    y = np.arange(len(t))
    ax.hlines(y, t.ci_low * 100, t.ci_high * 100, color=CAT[0], lw=2)
    ax.plot(t["mean"] * 100, y, "o", color=CAT[0], ms=6)
    ax.axvline(0, color="#444", lw=1)
    ax.set_yticks(y, [f"{o}  ({w}/{e}/{d})" for o, w, e, d in
                      zip(t.option, t.wins, t.ties, t.losses)])
    ax.set_xlabel("Redução do arrependimento pelo seletor (p.p. de BA, IC95%)")
    ax.set_title('Seletor único (LODO) contra cada opção fixa — V/E/D por '+('família' if EXPANDED else 'conjunto'), fontsize=11)
    ax.grid(axis="y", visible=False)
    _save(fig, out, "fig5_selector_vs_each_option")


def fig_strata_heat(heat: pd.DataFrame, out: Path) -> None:
    rows = [*PIPES, "selector"]
    data = heat.loc[rows]
    labels = [SHORT.get(r, "Seletor único (LODO)") for r in rows]
    fig, ax = plt.subplots(figsize=(11, 5.8))
    low, high = float(data.min().min()), float(data.max().max())
    im = ax.imshow(data.to_numpy(), cmap="Blues_r", aspect="auto", vmin=low, vmax=high)
    ax.set_yticks(range(len(rows)), labels)
    ax.set_xticks(range(data.shape[1]), [v for _, v in data.columns], rotation=35, ha="right")
    for j, (dim, _) in enumerate(data.columns):
        if j and data.columns[j - 1][0] != dim:
            ax.axvline(j - 0.5, color="white", lw=3)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data.iat[i, j]
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=7.5,
                    color="white" if v < (low + high) / 2 else "#222")
    ax.axhline(len(rows) - 1.5, color="black", lw=1.2)
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.025, label="Posto médio (1 = menor arrependimento)")
    ax.set_title("Posto médio por estrato (classes · atributos · dimensão · orçamento)",
                 fontsize=11)
    _save(fig, out, "fig6_rank_heatmap_by_stratum")


def fig_redundancy(corr: pd.DataFrame, contrib: pd.DataFrame, out: Path) -> None:
    order = leaves_list(linkage(1 - corr.to_numpy()[np.triu_indices(len(corr), 1)], "average"))
    names = [corr.index[i] for i in order]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.6),
                                   gridspec_kw={"width_ratios": [1.25, 1]})
    im = ax1.imshow(corr.loc[names, names].to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    ax1.set_xticks(range(len(names)), [SHORT[n] for n in names], rotation=60, ha="right",
                   fontsize=8)
    ax1.set_yticks(range(len(names)), [SHORT[n] for n in names], fontsize=8)
    ax1.grid(False)
    fig.colorbar(im, ax=ax1, fraction=0.04, label="Spearman entre postos por condição")
    ax1.set_title("(a) Redundância entre opções", fontsize=11)
    c = contrib.sort_values("oracle_increase")
    ax2.barh([SHORT[p] for p in c.pipeline], c.oracle_increase * 100, color=CAT[0], height=0.6)
    for yi, (v, s) in enumerate(zip(c.oracle_increase * 100, c.win_share * 100)):
        ax2.text(v + 0.01, yi, f"{v:.2f} p.p. · vence {s:.1f}%".replace(".", ",").replace(",p,", ".p."),
                 va="center", fontsize=8)
    ax2.set_xlabel("Piora do oráculo sem a opção (p.p.)")
    ax2.set_xlim(0, c.oracle_increase.max() * 100 * 1.6)
    ax2.set_title("(b) Contribuição marginal ao portfólio", fontsize=11)
    ax2.grid(axis="y", visible=False)
    _save(fig, out, "fig7_portfolio_redundancy")


def winner_map_bins(frame: pd.DataFrame) -> pd.DataFrame:
    """Participação de cada família por decil: vencedoras (peso do empate) e escolhas do seletor."""
    choice = 1.0 / frame.groupby(["dataset_id", "epsilon"]).weight.transform("size")
    return pd.concat({
        "oracle_family": pd.crosstab(frame.bin, frame.oracle_family, values=frame.weight,
                                     aggfunc="sum", normalize="index"),
        "selector_family": pd.crosstab(frame.bin, frame.selector_family, values=choice,
                                       aggfunc="sum", normalize="index"),
    }, axis=1).round(3)


def fig_regime(frame: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    for ax, column, title in ((axes[0], "oracle_family", "(a) Vencedora observada"),
                              (axes[1], "selector_family", "(b) Escolha do seletor (LODO)")):
        weight = frame.weight if column == "oracle_family" else 1.0 / frame.groupby(
            ["dataset_id", "epsilon"]).weight.transform("size")  # uma escolha por condição
        share = pd.crosstab(frame.bin, frame[column], values=weight, aggfunc="sum",
                            normalize="index").reindex(
            columns=FAMILY_ORDER, fill_value=0) * 100
        centers = (frame.drop_duplicates(["dataset_id", "epsilon"])
                   .groupby("bin", observed=True).log_ratio.median().to_numpy())
        ax.stackplot(centers, share.T.to_numpy(), colors=CAT, edgecolor="white", lw=1.5,
                     labels=FAMILY_ORDER)
        ax.set_xlabel("log₁₀(n·ε/p) (mediana de cada decil)")
        ax.set_title(title, fontsize=11)
        ax.set_xlim(centers[0], centers[-1])
        ax.set_ylim(0, 100)
    axes[0].set_ylabel("Participação (%)")
    axes[1].legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=9, frameon=False)
    _save(fig, out, "fig8_winner_map_n_eps_p")


def fig_topk(curve: pd.DataFrame, oracle: float, out: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))
    ax1.plot(curve.k, curve.selector * 100, "-o", color=CAT[0], ms=5, lw=2,
             label="Lista do seletor")
    ax1.plot(curve.k, curve.fixed * 100, "-D", color=CAT[1], ms=5, lw=2,
             label="Lista fixa (ordem do treino)")
    ax1.axhline(oracle * 100, color=MUTED, ls="--", lw=1.5, label="Oráculo")
    ax1.set(xlabel="k opções da lista", ylabel="Arrependimento da melhor entre as k (p.p.)",
            title="(a) regret@k")
    ax1.legend(frameon=False)
    ax2.plot(curve.k, curve.selector_hit * 100, "-o", color=CAT[0], ms=5, lw=2)
    ax2.plot(curve.k, curve.fixed_hit * 100, "-D", color=CAT[1], ms=5, lw=2)
    ax2.set(xlabel="k opções da lista", ylabel="Condições com a vencedora na lista (%)",
            title="(b) Acerto top-k")
    for ax in (ax1, ax2):
        ax.set_xticks(range(1, len(PIPES) + 1))
    _save(fig, out, "fig9_topk_recommendation")


def fig_importance(shap: pd.DataFrame, columns: pd.DataFrame, out: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.2),
                                   gridspec_kw={"width_ratios": [1, 1.35]})
    s = shap.sort_values("share")
    y = np.arange(len(s))
    ax1.barh(y, s.share * 100, color=CAT[0], height=0.6)
    for yi, (v, d, ch) in enumerate(zip(s.share * 100, s["mean"] * 100, s.changed * 100)):
        ax1.text(v + 1, yi, f"{v:.0f}%  (Δ {d:.2f} p.p.; muda {ch:.0f}% das escolhas)"
                 .replace(".", ",").replace(",p,", ".p."),
                 va="center", fontsize=8.5)
    ax1.set_yticks(y, [PRIMITIVE_LABELS[p] for p in s.primitive])
    ax1.set_xlim(min(0, float(s.share.min()*100)-5) if EXPANDED else 0, max(100, float(s.share.max()*100)+10) if EXPANDED else 100)
    ax1.set_xlabel("Parcela do efeito total (Shapley, %)")
    ax1.set_title("(a) Grandezas públicas primitivas", fontsize=11)
    ax1.grid(axis="y", visible=False)
    top = columns.sort_values("mean", ascending=False).head(15).iloc[::-1]
    colors = [CAT[0] if g == "schema" else CAT[1] for g in top.group]
    y = np.arange(len(top))
    ax2.hlines(y, top.ci_low * 100, top.ci_high * 100, color="#555", lw=1)
    ax2.scatter(top["mean"] * 100, y, c=colors, s=40, zorder=3)
    ax2.set_yticks(y, top.label, fontsize=8.5)
    ax2.axvline(0, color="#444", lw=1)
    ax2.set_xlabel("Aumento do regret (p.p.; IC95%)")
    ax2.set_title("(b) 15 meta-features individuais mais influentes", fontsize=11)
    ax2.scatter([], [], c=CAT[0], label="schema")
    ax2.scatter([], [], c=CAT[1], label="calibração")
    ax2.legend(frameon=False, loc="lower right")
    ax2.grid(axis="y", visible=False)
    _save(fig, out, "fig10_metafeature_importance")


def fig_metamodels(wide: pd.DataFrame, out: Path) -> dict:
    n, k = wide.shape
    ranks = wide.rank(axis=1).mean()
    cd = studentized_range.ppf(0.95, k, np.inf) / np.sqrt(2) * np.sqrt(k * (k + 1) / (6 * n))
    p = friedmanchisquare(*[wide[c].to_numpy() for c in wide]).pvalue
    fig, ax = plt.subplots(figsize=(11, 4.2))
    _cd_diagram(ax, ranks, cd)
    ax.set_title(f"Metamodelos: diferença crítica (Nemenyi, α=0,05, N={n} {'famílias semânticas' if EXPANDED else 'conjuntos'}, "
                 f"Friedman p={p:.2g})", fontsize=11)
    _save(fig, out, "fig11_metamodel_cd")
    return {"ranks": ranks.to_dict(), "cd": float(cd), "friedman_p": float(p)}


def most_frequent_winner(agg: pd.DataFrame) -> pd.Series:
    """Família que mais vence (com peso do empate) nos orçamentos de cada conjunto."""
    w = tied_winners(agg).assign(fam=lambda f: f.pipeline.map(family))
    return w.groupby(["dataset_id", "fam"]).weight.sum().unstack(fill_value=0).idxmax(axis=1)


def fig_tsne(emb: np.ndarray, labels: pd.Series, out: Path) -> None:
    keep = ("NB categórico", "Softmax · DP-SGD", "GaussianNB")
    fig, ax = plt.subplots(figsize=(7.5, 6))
    other = ~labels.isin(keep).to_numpy()
    ax.scatter(emb[other, 0], emb[other, 1], s=14, c=MUTED, alpha=0.6,
               label="Outras (árvores, softmax · saída)", edgecolors="white", linewidths=0.4)
    for color, name in zip(CAT, keep):
        mask = (labels == name).to_numpy()
        ax.scatter(emb[mask, 0], emb[mask, 1], s=16, c=color, label=name,
                   edgecolors="white", linewidths=0.4)
    ax.set(xticks=[], yticks=[], xlabel="t-SNE 1", ylabel="t-SNE 2")
    ax.legend(frameon=False, fontsize=9, loc="best", title="Vencedora mais frequente")
    ax.set_title(f"Espaço das meta-features públicas ({len(labels)} conjuntos)", fontsize=11)
    _save(fig, out, "fig12_metafeature_space_tsne")


# ---------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selector", choices=sel.SELECTORS, default="rf12")
    parser.add_argument("--manifest", type=Path, default=Path("data/openml_manifest_984_completed.csv"))
    parser.add_argument("--conditions", type=Path,
                        default=Path("results/portfolio-984-current-slim/conditions.csv"))
    parser.add_argument("--lodo", type=Path,
                        default=Path("results/portfolio-984-shared-selector-lodo/outer_predictions.csv"))
    parser.add_argument("--output", type=Path, default=Path("results/portfolio-984-dissertation-extras"))
    parser.add_argument("--figures", type=Path, default=Path("figures/portfolio-984-dissertation-extras"))
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--skip-refit", action="store_true", help="só a parte A")
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    args.figures.mkdir(parents=True, exist_ok=True)
    global EXPANDED, PIPES
    EXPANDED = args.selector == 'gbm_residual_fallback'
    PIPES = pb.PIPELINES
    if EXPANDED:
        if args.output == Path('results/portfolio-984-dissertation-extras'):args.output=Path('results/dissertation/extras')
        if args.figures == Path('figures/portfolio-984-dissertation-extras'):args.figures=Path('figures/dissertation')
        if args.lodo == Path('results/portfolio-984-shared-selector-lodo/outer_predictions.csv'):
            args.lodo=Path('results/experiments/lodo_gbm15/lodo_picks.csv')
        args.output.mkdir(parents=True,exist_ok=True);args.figures.mkdir(parents=True,exist_ok=True)
    manifest = _load_manifest(args.manifest)
    conditions = pd.read_csv(args.conditions)
    if EXPANDED:
        agg, manifest, PIPES = sel.prepare(conditions, manifest)
        lodo = sel.from_lodo(args.lodo, agg)
    else:
        agg = pb._aggregate(conditions, manifest)
        lodo = pd.read_csv(args.lodo)
    lodo = lodo[lodo.feature_mode == MODE].copy()
    lodo["family_id"] = lodo.family_id.astype(str)
    summary: dict = {"config": {
        "conditions_sha256": file_sha256(args.conditions), "lodo_sha256": file_sha256(args.lodo),
        "manifest_sha256": file_sha256(args.manifest), "seed": args.seed,
        "repeats": args.repeats, "script_sha256": file_sha256(Path(__file__))}}

    # ---- parte A
    table = vs_each_option(agg, lodo, args.seed)
    table.to_csv(args.output / "selector_vs_each_option.csv", index=False)
    fig_vs_options(table, args.figures)
    strata = dataset_strata(manifest)
    gains, heat = strata_tables(agg, lodo, strata, args.seed)
    gains.to_csv(args.output / "gain_by_stratum.csv", index=False)
    heat.to_csv(args.output / "rank_by_stratum.csv")
    fig_strata_heat(heat, args.figures)
    contrib, corr = portfolio_contribution(agg)
    contrib.to_csv(args.output / "portfolio_contribution.csv", index=False)
    corr.to_csv(args.output / "option_rank_correlation.csv")
    fig_redundancy(corr, contrib, args.figures)
    summary["dp_changes_best_model"] = dp_changes_best_model(agg)
    search = idealized_private_search(agg, lodo, args.seed)
    search.to_csv(args.output / "idealized_private_search.csv", index=False)
    chance = near_chance_gain(agg, lodo, manifest, args.seed)
    chance.to_csv(args.output / "near_chance_gain.csv", index=False)
    print(chance.to_string(index=False))
    regime = regime_frame(agg, lodo, manifest)
    winner_map_bins(regime).to_csv(args.output / "winner_map_bins.csv")
    fig_regime(regime, args.figures)
    previous = args.output / "summary.json"
    if args.skip_refit and previous.exists():  # preserva os resultados da parte B
        summary = {**json.loads(previous.read_text()), **summary}
    previous.write_text(json.dumps(summary, indent=2, default=float))
    print(table.to_string(index=False))
    print(gains.to_string(index=False))
    print(contrib.to_string(index=False))
    print(json.dumps(summary["dp_changes_best_model"], indent=1, default=float))
    print(search.to_string(index=False))
    if args.skip_refit:
        return 0

    # ---- parte B
    sink: dict = {}
    t0 = time.time()
    grouped_summary = refit(conditions, manifest, args.seed, sink)
    summary["refit_5fold"] = grouped_summary["feature_modes"][MODE]["mean_regret_gain_vs_best_fixed"]
    print("refit", time.time() - t0, summary["refit_5fold"], flush=True)
    if EXPANDED:
        sink.pop('_predictions').to_csv(args.output / 'fivefold_predictions.csv', index=False)
    builder = FeatureBuilder(manifest)
    names = sel.feature_names(PIPES) if EXPANDED else pb.candidate_feature_names(MODE)
    rng = np.random.default_rng(args.seed)
    topk_parts, col_parts, shap_parts, meta_rows, latencies = [], {}, [], [], []
    shap_changed, all_values, all_changed = [], [], []
    for fold, train, test, model in fold_blocks(agg, sink):
        print(f"fold {fold}: análises", flush=True)
        matrix = builder.matrix(test)
        check = (sel.feature_matrix(test.head(2*len(PIPES)), manifest, PIPES) if EXPANDED else np.vstack([pb._candidate_features(
            builder_row_meta(manifest, int(r.dataset_id)), float(r.epsilon), r.pipeline, MODE, {}, {})
            for r in test.head(2*len(PIPES)).itertuples()]))
        assert np.allclose(check, matrix[:2*len(PIPES)]), "FeatureBuilder diverge de _candidate_features"
        scores = model.predict(matrix)
        base = choose(test, scores)
        fixed_scores = (train.groupby(["family_id", "pipeline"]).private_ba.mean()
                        .groupby("pipeline").mean().sort_values(ascending=False))
        topk_parts.append(topk(test, scores, list(fixed_scores.index)))
        for name, (delta, changed) in column_importance(
                test, model, matrix, base, rng, args.repeats, names).items():
            col_parts.setdefault(name, []).append((delta, changed))
        phi, phi_changed, value_all, changed_all = shapley_primitives(
            test, model, builder, base, rng, args.repeats)
        shap_parts.append(pd.DataFrame(phi))
        shap_changed.append(phi_changed)
        all_values.append(value_all)
        all_changed.append(changed_all)
        # metamodelos alternativos nas mesmas partições
        train_matrix = builder.matrix(train)
        weights = sel.weights(train) if EXPANDED else pb._row_weights(train)
        for name, estimator in candidate_models(args.seed + fold).items():
            if estimator is None:
                chosen = base
            elif name == "Classificador da vencedora":
                x_train, cond_train = classifier_matrix(train, builder)
                winners = train.loc[train.groupby(["dataset_id", "epsilon"]).private_ba.idxmax()]
                labels = cond_train.merge(winners, on=["family_id", "dataset_id", "epsilon"]).pipeline
                estimator.fit(x_train, labels.to_numpy())
                x_test, cond_test = classifier_matrix(test, builder)
                picked = cond_test.assign(pipeline=estimator.predict(x_test))
                chosen = picked.merge(test, on=["family_id", "dataset_id", "epsilon", "pipeline"])
            else:
                fit_kwargs = ({} if name.startswith("k-NN")  # k-NN não aceita pesos
                              else {"ridge__sample_weight": weights} if name.startswith("Ridge")
                              else {"sample_weight": weights})
                if EXPANDED and name == 'RF (RF-12)':
                    fit_kwargs['sample_weight'] = weights / weights.sum()
                estimator.fit(train_matrix, train.private_ba.to_numpy(), **fit_kwargs)
                chosen = choose(test, estimator.predict(matrix))
            best_rows = test.groupby(["dataset_id", "epsilon"], as_index=False) \
                .portfolio_regret.min().rename(columns={"portfolio_regret": "_best"})
            chosen = chosen.merge(best_rows, on=["dataset_id", "epsilon"])
            chosen["top1"] = chosen.portfolio_regret <= chosen._best + 1e-12
            meta_rows.append(chosen.assign(metamodel=name)[
                ["metamodel", "family_id", "dataset_id", "epsilon", "pipeline",
                 "portfolio_regret", "top1"]])
        # latência: uma recomendação = 12 linhas de features + uma predição
        model.set_params(n_jobs=1)
        for d in list(test.dataset_id.unique())[:20]:
            one = test[(test.dataset_id == d) & (test.epsilon == 1.0)]
            builder.cache.clear()
            start = time.perf_counter()
            model.predict(builder.matrix(one))
            latencies.append(time.perf_counter() - start)

    # ---- consolidação da parte B
    topk_frame = pd.concat(topk_parts)
    curve = (topk_frame.groupby(["k", "family_id", "dataset_id"])
             [["selector", "fixed", "selector_hit", "fixed_hit"]].mean()
             .groupby(["k", "family_id"]).mean().groupby("k").mean().reset_index())
    curve.to_csv(args.output / "topk.csv", index=False)
    oracle = float(topk_frame[topk_frame.k == len(PIPES)].groupby("dataset_id").selector.mean().mean())
    fig_topk(curve, oracle, args.figures)

    col_rows = []
    for index, (name, parts) in enumerate(col_parts.items()):
        values = pd.concat([p[0] for p in parts])
        group = dict(names)[name]
        col_rows.append({"feature": name, "group": group,
                         "label": name.replace("calib_", "").replace("__", " · "),
                         "changed": float(np.mean([p[1] for p in parts])),
                         **ci(values.to_numpy(), args.seed + index)})
    columns = pd.DataFrame(col_rows).sort_values("mean", ascending=False)
    total_all = pd.concat(all_values)
    columns["share_of_all"] = columns["mean"] / total_all.mean()
    columns.to_csv(args.output / "importance_columns.csv", index=False)

    phi_frame = pd.concat(shap_parts)
    shap_rows = []
    for index, p in enumerate(PRIMITIVES):
        shap_rows.append({"primitive": p, **ci(phi_frame[p].to_numpy(), args.seed + 100 + index),
                          "share": float(phi_frame[p].mean() / total_all.mean()),
                          "changed": float(np.mean([c[p] for c in shap_changed]))})
    shap = pd.DataFrame(shap_rows)
    shap.to_csv(args.output / "shapley_primitives.csv", index=False)
    summary["shapley_total"] = {**ci(total_all.to_numpy(), args.seed + 200),
                                "changed": float(np.mean(all_changed))}
    fig_importance(shap, columns, args.figures)

    metas = pd.concat(meta_rows)
    best_fixed_choice = agg[agg.pipeline == (sel.best_fixed(agg) if EXPANDED else pb._best_fixed(agg))]
    best_fixed = fam_mean(best_fixed_choice, "portfolio_regret")
    meta_table = []
    for index, (name, block) in enumerate(metas.groupby("metamodel", sort=False)):
        regret = fam_mean(block, "portfolio_regret")
        meta_table.append({"metamodel": name, "regret": float(regret.mean()),
                           "top1": float(block.top1.mean()),
                           **ci((best_fixed.reindex(regret.index) - regret).to_numpy(),
                                args.seed + 300 + index)})
    meta_table = pd.DataFrame(meta_table)
    meta_table.to_csv(args.output / "metamodels.csv", index=False)
    metas.to_csv(args.output / 'metamodel_predictions.csv', index=False)
    unit = 'family_id' if EXPANDED else 'dataset_id'
    wide = metas.groupby([unit, "metamodel"]).portfolio_regret.mean().unstack()
    wide["Melhor opção fixa"] = best_fixed_choice.groupby(unit).portfolio_regret.mean()
    summary["metamodel_cd"] = fig_metamodels(wide.dropna(), args.figures)

    per_dataset = most_frequent_winner(agg)
    dataset_features = np.vstack([builder.row(d, d, d, 1.0, PIPES[0])[:len(PUBLIC_FEATURE_NAMES)]
                                  for d in per_dataset.index])
    keep = [i for i, n in enumerate(PUBLIC_FEATURE_NAMES) if "epsilon" not in n]
    emb = TSNE(perplexity=30, random_state=args.seed, init="pca").fit_transform(
        StandardScaler().fit_transform(dataset_features[:, keep]))
    fig_tsne(emb, per_dataset, args.figures)
    summary["recommendation_latency_ms"] = {
        "median": float(np.median(latencies) * 1000), "p90": float(np.quantile(latencies, 0.9) * 1000)}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(curve.to_string(index=False))
    print(columns.head(20).to_string(index=False))
    print(shap.to_string(index=False))
    print(meta_table.to_string(index=False))
    print(json.dumps({k: summary[k] for k in ("shapley_total", "metamodel_cd",
                                              "recommendation_latency_ms", "refit_5fold")},
                     indent=1, default=float))
    return 0


def builder_row_meta(manifest: pd.DataFrame, dataset_id: int) -> pd.Series:
    return manifest.drop_duplicates("dataset_id").set_index("dataset_id").loc[dataset_id]


if __name__ == "__main__":
    raise SystemExit(main())

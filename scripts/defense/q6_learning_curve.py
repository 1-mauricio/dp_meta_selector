"""Q6: curva de aprendizado do seletor (floresta adotada e gradient boosting)
e, de brinde, 5 partições agrupadas por família SEMÂNTICA (questão 7 do parecer)."""
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import GroupKFold

sys.path.insert(0, "scripts")
import dpmeta.portfolio_benchmark as pb  # noqa: E402
from dpmeta.openml_benchmark import _load_manifest  # noqa: E402
from analyze_dissertation_extras_portfolio13 import FeatureBuilder  # noqa: E402

S = "results/defense-analyses/"
NBC = "categorical_naive_bayes__laplace_histograms"
man = _load_manifest("data/openml_manifest_984_completed.csv")
agg = pb._aggregate(pd.read_csv("results/portfolio-984-current-slim/conditions.csv"), man)
agg = agg.sort_values(["dataset_id", "epsilon", "pipeline"]).reset_index(drop=True)
X = FeatureBuilder(man).matrix(agg)
y = agg.private_ba.to_numpy()
sem = man.drop_duplicates("dataset_id").set_index("dataset_id").semantic_family_id


def rf():
    return RandomForestRegressor(n_estimators=1000, min_samples_leaf=3, max_features=0.9,
                                 max_depth=6, max_samples=0.5, random_state=0, n_jobs=10)


def gbm():
    return HistGradientBoostingRegressor(random_state=0)


def per_dataset_gain(test, scores):
    t = test.assign(s=scores)
    chosen = t.loc[t.groupby(["dataset_id", "epsilon"]).s.idxmax()]
    sel = chosen.groupby("dataset_id").portfolio_regret.mean()
    bf = test[test.pipeline == NBC].groupby("dataset_id").portfolio_regret.mean()
    return (bf - sel)


def run(groups_by_dataset, fractions, models, draws, tag):
    datasets = np.array(sorted(agg.dataset_id.unique()))
    groups = groups_by_dataset.reindex(datasets).astype(str).to_numpy()
    rows = []
    for fold, (tr, te) in enumerate(GroupKFold(5).split(datasets, groups=groups)):
        test = agg[agg.dataset_id.isin(datasets[te])]
        fam_tr = np.unique(groups[tr])
        for frac in fractions:
            for draw in range(draws if frac < 1 else 1):
                rng = np.random.default_rng(1000 * fold + draw)
                keep_f = rng.choice(fam_tr, max(5, int(round(frac * len(fam_tr)))), replace=False)
                keep = datasets[tr][np.isin(groups[tr], keep_f)]
                mask = agg.dataset_id.isin(keep).to_numpy()
                w = pb._row_weights(agg[mask])
                for name, make in models.items():
                    m = make()
                    m.fit(X[mask], y[mask], sample_weight=w)
                    g = per_dataset_gain(test, m.predict(X[test.index]))
                    rows.append(pd.DataFrame({"tag": tag, "fold": fold, "frac": frac, "draw": draw,
                                              "model": name, "train_datasets": len(keep),
                                              "dataset_id": g.index, "gain": g.to_numpy()}))
                    print(tag, fold, frac, draw, name, len(keep), round(g.mean(), 4), flush=True)
    return pd.concat(rows)


canon = man.drop_duplicates("dataset_id").set_index("dataset_id").family_id
lc = run(canon, (0.1, 0.25, 0.5, 0.75, 1.0), {"rf": rf, "gbm": gbm}, 2, "canonical")
semr = run(sem, (1.0,), {"rf": rf, "gbm": gbm}, 1, "semantic")
res = pd.concat([lc, semr])
res.to_csv(S + "q6_result.csv", index=False)
summ = (res.groupby(["tag", "model", "frac", "draw", "fold"]).gain.mean()
        .groupby(["tag", "model", "frac"]).agg(["mean", "std", "count"]))
print(summ)

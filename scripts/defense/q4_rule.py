"""Q4: uma regra de limiares sobre n·ε/(d+1) empata com o seletor?

Regras escolhem a opção por condição (dataset, ε) só com grandezas públicas.
Os limiares e as opções de cada faixa são escolhidos por LODO: para cada
conjunto, a configuração de menor arrependimento médio nos outros 983.
"""
import itertools
import json
import numpy as np
import pandas as pd

S = "results/defense-analyses/"
C = pd.read_csv("results/portfolio-984-current-slim/conditions.csv")
agg = C.groupby(["dataset_id", "epsilon", "pipeline"]).portfolio_regret.mean().unstack()
man = pd.read_csv("data/openml_manifest_984_completed.csv").drop_duplicates("dataset_id").set_index("dataset_id")
lodo = pd.read_csv("results/portfolio-984-shared-selector-lodo/outer_predictions.csv")
lodo = lodo[lodo.feature_mode == "public_plus_dp"].set_index(["dataset_id", "epsilon"])

ids = agg.index.get_level_values(0).to_numpy()
eps = agg.index.get_level_values(1).to_numpy()
n = man.n_instances.reindex(ids).to_numpy(float)
d = man.n_features.reindex(ids).to_numpy(float)
p = man.n_encoded_features.reindex(ids).to_numpy(float)
nominal_only = (man.n_nominal_features == man.n_features).reindex(ids).to_numpy()
x_nb = np.log10(n * eps / (d + 1))
x_p = np.log10(n * eps / p)
NBC = "categorical_naive_bayes__laplace_histograms"
OPTS = list(agg.columns)
R = agg.to_numpy()
col = {o: i for i, o in enumerate(OPTS)}
datasets = np.unique(ids)
pos = np.searchsorted(datasets, ids)


def per_dataset(choice_idx):
    r = R[np.arange(len(R)), choice_idx]
    return np.bincount(pos, r) / np.bincount(pos)


configs, names = [], []
# R1: um limiar; abaixo dele a opção A, acima o NB categórico
low_opts = [o for o in OPTS if o != NBC]
for x, xname in ((x_nb, "nε/(d+1)"), (x_p, "nε/p")):
    for t in np.round(np.arange(-1.0, 4.01, 0.1), 2):
        for a in low_opts:
            ch = np.where(x < t, col[a], col[NBC])
            configs.append(per_dataset(ch)); names.append(("R1", xname, t, None, a, None, False))
            ch2 = np.where(nominal_only, col[NBC], ch)
            configs.append(per_dataset(ch2)); names.append(("R1+nom", xname, t, None, a, None, True))
# R2: dois limiares, três faixas (A | B | NB cat), com a exceção dos só nominais
mids = ["softmax__dp_sgd_clip2_reg003", "softmax__dp_sgd", "softmax__dp_sgd_steps100",
        "random_forest__laplace_parallel_shards_25", "gnb__gaussian_zcdp_equal"]
for x, xname in ((x_nb, "nε/(d+1)"), (x_p, "nε/p")):
    grid = np.round(np.arange(-1.0, 4.01, 0.25), 2)
    for t1, t2 in itertools.combinations(grid, 2):
        for a, b in itertools.permutations(mids, 2):
            ch = np.where(x < t1, col[a], np.where(x < t2, col[b], col[NBC]))
            ch = np.where(nominal_only, col[NBC], ch)
            configs.append(per_dataset(ch)); names.append(("R2+nom", xname, t1, t2, a, b, True))
M = np.vstack(configs)  # configs × datasets
print("configs:", M.shape)

best_fixed = agg.groupby(level=0).mean()[NBC].reindex(datasets).to_numpy()
sel = lodo.selector_regret.groupby(level=0).mean().reindex(datasets).to_numpy()
lodo_bf = lodo.best_fixed_regret.groupby(level=0).mean().reindex(datasets).to_numpy()
print("sanity best-fixed diff:", np.abs(best_fixed - lodo_bf).max())


def boot(v, seed=1, B=10000):
    r = np.random.default_rng(seed)
    m = v[r.integers(0, len(v), (B, len(v)))].mean(1)
    return v.mean(), np.quantile(m, 0.025), np.quantile(m, 0.975)


out = {}
for fam in ("R1", "R1+nom", "R2+nom"):
    mask = np.array([nm[0] == fam for nm in names])
    Mf = M[mask]
    tot = Mf.sum(1, keepdims=True)
    loo = (tot - Mf) / (len(datasets) - 1)  # média nos outros conjuntos
    pick = loo.argmin(0)
    rule = Mf[pick, np.arange(len(datasets))]
    in_sample = np.array(names, dtype=object)[mask][Mf.mean(1).argmin()]
    chosen = pd.Series([str(np.array(names, dtype=object)[mask][i]) for i in pick]).value_counts()
    g_bf = boot(best_fixed - rule)
    g_sel = boot(sel - rule)
    q = lambda v: (np.median(v), np.quantile(v, 0.9), np.sort(v)[-len(v) // 10:].mean())
    out[fam] = {"regret": rule.mean(), "gain_vs_bestfixed": g_bf, "rule_minus_selector(neg=rule worse)": g_sel,
                "median_p90_cvar": q(rule), "insample_best": str(in_sample),
                "most_chosen_config": chosen.index[0], "share_folds": chosen.iloc[0] / len(datasets)}
out["selector"] = {"regret": sel.mean(), "gain_vs_bestfixed": boot(best_fixed - sel),
                   "median_p90_cvar": (np.median(sel), np.quantile(sel, 0.9), np.sort(sel)[-len(sel) // 10:].mean())}
out["best_fixed"] = {"regret": best_fixed.mean(),
                     "median_p90_cvar": (np.median(best_fixed), np.quantile(best_fixed, 0.9), np.sort(best_fixed)[-len(best_fixed) // 10:].mean())}
for k, v in out.items():
    print(k, json.dumps(v, default=lambda o: [round(float(a), 5) for a in o] if isinstance(o, tuple) else float(o), ensure_ascii=False, indent=1))
json.dump(out, open(S + "q4_result.json", "w"), default=str, indent=1)

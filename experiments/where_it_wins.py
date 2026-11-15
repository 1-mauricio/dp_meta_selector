"""Onde a final ganha: ganho vs b8u (LODO GBM-15) por estratos públicos. Exploratório/descritivo."""
import json, numpy as np, pandas as pd
B = "categorical_naive_bayes__laplace_histograms_b8u"
m = pd.read_csv("data/openml_manifest_984_completed.csv").drop_duplicates("dataset_id").set_index("dataset_id")
c = pd.read_csv("results/portfolio-984-sgd/conditions_15.csv", usecols=["dataset_id", "epsilon", "pipeline", "private_ba", "best_clean_portfolio_ba"])
a = c.groupby(["dataset_id", "epsilon", "pipeline"]).private_ba.mean()
r = pd.read_csv("results/experiments/lodo_gbm15/lodo_picks.csv")
r["g"] = [a.loc[(d, e, p)] - a.loc[(d, e, B)] for d, e, p in zip(r.dataset_id, r.epsilon, r.pipeline)]
r["switch"] = r.pipeline != B
mm = m.loc[r.dataset_id]
n, d, k = mm.n_instances.values, mm.n_encoded_features.values, mm.n_classes.values
r["n"] = pd.cut(n, [0, 500, 2000, 10000, 1e9], labels=["≤500", "501–2k", "2k–10k", ">10k"])
r["eps"] = r.epsilon
r["ratio"] = pd.cut(n * r.epsilon.values / (d + 1), [0, 10, 100, 1000, 1e12], labels=["<10", "10–100", "100–1k", ">1k"])
nom = mm.n_nominal_features.values; feat = mm.n_features.values
r["tipo"] = np.where(nom == feat, "só nominais", np.where(nom == 0, "só numéricos", "mistos"))
r["classes"] = pd.cut(k, [1, 2, 5, 1000], labels=["2", "3–5", ">5"])
cl = c.groupby(["dataset_id", "epsilon"]).best_clean_portfolio_ba.mean()
chance = 1 / pd.Series(k, index=r.index)
r["acaso"] = np.where([cl.loc[(x, e)] for x, e in zip(r.dataset_id, r.epsilon)] - chance.values <= 0.02, "perto do acaso", "informativo")
rng = np.random.default_rng(20261112); rows = []
for s in ["eps", "n", "ratio", "tipo", "classes", "acaso"]:
    for lv, x in r.groupby(s, observed=True):
        f = x.groupby("fam").g.mean().values
        bs = [f[rng.integers(0, len(f), len(f))].mean() for _ in range(5000)]
        cc = x.groupby(["dataset_id", "epsilon"]).g.mean()
        rows.append(dict(estrato=s, nivel=str(lv), condicoes=len(cc), familias=len(f), ganho_pp=100 * f.mean(),
                         ic_lo=100 * np.percentile(bs, 2.5), ic_hi=100 * np.percentile(bs, 97.5),
                         troca_opcao_pct=100 * x.switch.mean(), perde_1pp_pct=100 * (cc < -0.01).mean(),
                         ganha_1pp_pct=100 * (cc > 0.01).mean()))
s = pd.DataFrame(rows).round(2); s.to_csv("results/development-strata/gain_by_stratum.csv", index=False)
pick = r[r.switch].pipeline.value_counts(normalize=True).round(3).to_dict()
json.dump({"escolhas_quando_troca": pick}, open("results/development-strata/picks_when_switch.json", "w"), indent=1, ensure_ascii=False)
print(s.to_string(index=False)); print(pick)

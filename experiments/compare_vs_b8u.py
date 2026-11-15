"""Ganho vs NB b8u (base comum) e vs NB antigo para picks de LODO. Uso: compare_vs_b8u.py dir1 dir2 ..."""
import sys, numpy as np, pandas as pd
c = pd.read_csv("results/portfolio-984-sgd/conditions_15.csv", usecols=["dataset_id", "epsilon", "pipeline", "private_ba"])
a = c.groupby(["dataset_id", "epsilon", "pipeline"]).private_ba.mean()
B, OLD = "categorical_naive_bayes__laplace_histograms_b8u", "categorical_naive_bayes__laplace_histograms"
rows = []
for d in sys.argv[1:]:
    r = pd.read_csv(f"results/experiments/{d}/lodo_picks.csv")
    r["g"] = [a.loc[(x, e, p)] - a.loc[(x, e, B)] for x, e, p in zip(r.dataset_id, r.epsilon, r.pipeline)]
    r["go"] = [a.loc[(x, e, p)] - a.loc[(x, e, OLD)] for x, e, p in zip(r.dataset_id, r.epsilon, r.pipeline)]
    rng = np.random.default_rng(1)
    for (v, fb), x in r.groupby(["variant", "fallback"]):
        f = x.groupby("fam")[["g", "go"]].mean()
        bs = [f.g.values[rng.integers(0, len(f), len(f))].mean() for _ in range(3000)]
        cc = x.groupby(["dataset_id", "epsilon"]).g.mean()
        rows.append(dict(lodo=d, variante=v, fallback=fb, vs_b8u=f.g.mean(), ic_lo=np.percentile(bs, 2.5),
                         ic_hi=np.percentile(bs, 97.5), vs_NB_antigo=f.go.mean(),
                         pior_1pp_vs_b8u=(cc < -0.01).mean(), cvar90_perda=-np.sort(cc.values)[:len(cc)//10].mean()))
print(pd.DataFrame(rows).round(4).to_string(index=False))

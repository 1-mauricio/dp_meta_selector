"""E2/E3 (GBM-13): GBM absoluto/residual, portfólio reduzido e fallback.

Reaproveita results/portfolio-984-current-slim/conditions.csv. 5 partições
agrupadas por semantic_family_id. Ganho = BA privada média (3 rep.) da opção
escolhida menos a do NB categórico, peso igual por família semântica.
"""
import json, sys, time
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from dpmeta import portfolio_benchmark as pb

NB = "categorical_naive_bayes__laplace_histograms"
H27 = "softmax__dp_sgd_clip2_reg003"
OUT = "results/experiments"
SEED = 20261110

man = pd.read_csv("data/openml_manifest_984_completed.csv")
cond = pd.read_csv("results/portfolio-984-current-slim/conditions.csv",
                   usecols=["dataset_id", "epsilon", "pipeline", "repetition", "private_ba"])
agg = cond.groupby(["dataset_id", "epsilon", "pipeline"], as_index=False).private_ba.mean()
fam = man.drop_duplicates("dataset_id").set_index("dataset_id").semantic_family_id.astype(str)
agg["fam"] = agg.dataset_id.map(fam)
meta = man.drop_duplicates("dataset_id").set_index("dataset_id")
cache = {}
X = np.vstack([pb._cached_candidate_features(meta.loc[int(r.dataset_id)], float(r.epsilon),
                                              r.pipeline, "public_plus_dp", {}, None, cache)
               for r in agg.itertuples(index=False)])
nb_ba = agg.set_index(["dataset_id", "epsilon", "pipeline"]).private_ba.xs(NB, level=2)
agg["nb_ba"] = [nb_ba.loc[(d, e)] for d, e in zip(agg.dataset_id, agg.epsilon)]
agg["resid"] = agg.private_ba - agg.nb_ba

# contribuição ao oráculo, para o portfólio intermediário (top-5 de portfolio_contribution)
INTER = [NB, H27, "softmax__dp_sgd", "softmax__analytic_gaussian", "softmax__dp_sgd_steps100"]
PORTS = {"completo": None, "intermediario5": INTER, "NB+H27": [NB, H27]}


def model(kind):
    if kind == "rf":
        return RandomForestRegressor(n_estimators=300, min_samples_leaf=3, max_features=0.9,
                                     max_depth=6, max_samples=0.5, random_state=SEED, n_jobs=-1)
    return HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
                                         min_samples_leaf=40, l2_regularization=1.0,
                                         random_state=SEED)


def weights(df):
    w = 1.0 / df.groupby("fam").dataset_id.transform("nunique")
    return (w / df.groupby("dataset_id").pipeline.transform("size")).to_numpy()


def choose(df, pred, port, tau=None):
    d = df.assign(pred=pred)
    if port is not None:
        d = d[d.pipeline.isin(port)]
    idx = d.groupby(["dataset_id", "epsilon"]).pred.idxmax()
    pick = d.loc[idx, ["dataset_id", "epsilon", "pipeline", "pred"]]
    if tau is not None:  # fallback: fica no NB salvo vantagem prevista > tau
        nbp = d[d.pipeline == NB].set_index(["dataset_id", "epsilon"]).pred
        adv = pick.pred.to_numpy() - nbp.loc[list(zip(pick.dataset_id, pick.epsilon))].to_numpy()
        pick.loc[adv <= tau, "pipeline"] = NB
    return pick[["dataset_id", "epsilon", "pipeline"]]


def gain(df, pick):
    m = pick.merge(df[["dataset_id", "epsilon", "pipeline", "resid", "fam"]],
                   on=["dataset_id", "epsilon", "pipeline"])
    return m


variants = [("RF abs (300 árv.)", "rf", "private_ba"), ("GBM abs", "gbm", "private_ba"),
            ("GBM residual vs NB", "gbm", "resid")]
TAUS = [0.0, 0.0025, 0.005, 0.01, 0.02]
rows = []
t0 = time.time()
gkf = GroupKFold(n_splits=5)
for fold, (tr, te) in enumerate(gkf.split(X, groups=agg.fam)):
    dtr, dte = agg.iloc[tr], agg.iloc[te]
    for name, kind, target in variants:
        m = model(kind).fit(X[tr], dtr[target], sample_weight=weights(dtr))
        p = m.predict(X[te])
        for pname, port in PORTS.items():
            rows.append(gain(dte, choose(dte, p, port)).assign(variant=name, portfolio=pname,
                                                               fallback="não", fold=fold))
        # fallback: tau escolhido por validação interna (3 partições do treino)
        inner = GroupKFold(n_splits=3)
        score = {t: [] for t in TAUS}
        for itr, iva in inner.split(X[tr], groups=dtr.fam):
            a, b = dtr.iloc[itr], dtr.iloc[iva]
            mi = model(kind).fit(X[tr][itr], a[target], sample_weight=weights(a))
            pi = mi.predict(X[tr][iva])
            for t in TAUS:
                g = gain(b, choose(b, pi, None, t))
                score[t].append(g.groupby("fam").resid.mean().mean())
        tbest = max(TAUS, key=lambda t: np.mean(score[t]))
        rows.append(gain(dte, choose(dte, p, None, tbest)).assign(
            variant=name, portfolio="completo", fallback=f"tau={tbest}", fold=fold))
    print(f"fold {fold} {time.time()-t0:.0f}s", flush=True)

res = pd.concat(rows)
res.to_csv(f"{OUT}/e2_e3_picks.csv", index=False)
rng = np.random.default_rng(SEED)
out = []
for (v, p, fb), g in res.groupby(["variant", "portfolio", "fallback"]):
    g = g.assign(fallback_tau=fb)
    per_fam = g.groupby("fam").resid.mean()
    vals = per_fam.to_numpy()
    boots = [rng.choice(vals, len(vals)).mean() for _ in range(5000)]
    pd_ds = g.groupby(["dataset_id", "epsilon"]).resid.mean()
    out.append(dict(variante=v, portfolio=p, fallback=("sim" if fb != "não" else "não"),
                    ganho_vs_NB=vals.mean(), ic_baixo=np.percentile(boots, 2.5),
                    ic_alto=np.percentile(boots, 97.5),
                    pct_cond_pior_1pp=float((pd_ds < -0.01).mean()),
                    pct_escolhe_NB=float((g.pipeline == NB).mean())))
summ = pd.DataFrame(out).sort_values("ganho_vs_NB", ascending=False)
summ.to_csv(f"{OUT}/e2_e3_summary.csv", index=False)
taus = res[res.fallback != "não"].groupby(["variant", "fold"]).fallback.first()
print(summ.to_string(index=False))
print(taus.to_string())
print(f"tempo {time.time()-t0:.0f}s")

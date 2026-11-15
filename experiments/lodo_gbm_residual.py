"""LODO por família semântica: GBM residual vs NB (+ fallback τ fixo) e GBM absoluto.

τ = 0,01 fixado ANTES da execução (valor mais frequente na validação interna do E3).
Uso: lodo_gbm_residual.py <conditions.csv> <saida_dir> <n_jobs>
Ganho = BA privada média (3 rep.) da escolha − BA do NB categórico atual,
peso igual por família semântica; bootstrap por família.
"""
import sys, os, time, json
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import HistGradientBoostingRegressor
from dpmeta import portfolio_benchmark as pb

NB = "categorical_naive_bayes__laplace_histograms"
TAU = 0.01
SEED = 20261110
cond_path, out, n_jobs = sys.argv[1], sys.argv[2], int(sys.argv[3])
FALLBACK = sys.argv[4] if len(sys.argv) > 4 else "best_mean"
SAVE_PREDS = os.environ.get("SAVE_PREDS") == "1"  # "safe" = menor CVaR90 no treino
os.makedirs(out, exist_ok=True)

man = pd.read_csv("data/openml_manifest_984_completed.csv").drop_duplicates("dataset_id")
cond = pd.read_csv(cond_path, usecols=["dataset_id", "epsilon", "pipeline", "repetition", "private_ba"])
agg = cond.groupby(["dataset_id", "epsilon", "pipeline"], as_index=False).private_ba.mean()
pipes = sorted(agg.pipeline.unique())
full = agg.groupby(["dataset_id", "epsilon"]).pipeline.transform("nunique") == len(pipes)
agg = agg[full].reset_index(drop=True)
meta = man.set_index("dataset_id")
agg["fam"] = agg.dataset_id.map(meta.semantic_family_id.astype(str))
nb = agg[agg.pipeline == NB].set_index(["dataset_id", "epsilon"]).private_ba
agg["nb_ba"] = nb.loc[list(zip(agg.dataset_id, agg.epsilon))].to_numpy()
agg["resid"] = agg.private_ba - agg.nb_ba

# features públicas: as do benchmark + one-hot da opção (cobre opções novas)
base_pipes = [p for p in pipes if p in pb.PIPELINES]
cache = {}
def feat(r):
    # final: cada opção usa a própria calibração pública (b8u, lamrule, lam0003 suportadas em dpmeta)
    p = r.pipeline
    f = pb._cached_candidate_features(meta.loc[int(r.dataset_id)], float(r.epsilon), p,
                                      "public_plus_dp", {}, None, cache)
    return np.concatenate([f, [float(r.pipeline == q) for q in pipes]])
X = np.vstack([feat(r) for r in agg.itertuples(index=False)])

def weights(df):
    w = 1.0 / df.groupby("fam").dataset_id.transform("nunique")
    return (w / df.groupby("dataset_id").pipeline.transform("size")).to_numpy()

def gbm():
    return HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
                                         min_samples_leaf=40, l2_regularization=1.0,
                                         random_state=SEED)

def run_fold(f):
    te = (agg.fam == f).to_numpy(); tr = ~te
    dtr, dte = agg[tr], agg[te]
    res, preds = [], []
    for name, target in (("GBM abs", "private_ba"), ("GBM residual", "resid")):
        p = gbm().fit(X[tr], dtr[target], sample_weight=weights(dtr)).predict(X[te])
        d = dte.assign(pred=p)
        if SAVE_PREDS:
            preds.append(d[["dataset_id", "epsilon", "pipeline", "fam", "private_ba", "pred"]]
                             .assign(variant=name))
        pick = d.loc[d.groupby(["dataset_id", "epsilon"]).pred.idxmax()]
        # fallback para a melhor opção fixa do TREINO (média por família semântica)
        if FALLBACK == "safe":
            # opção mais segura do TREINO: menor CVaR90 da perda frente à melhor opção
            # de cada condição (peso igual por família semântica). Pré-fixado em 2026-11-11.
            best = dtr.groupby(["dataset_id", "epsilon"]).private_ba.transform("max")
            loss = (best - dtr.private_ba).rename("loss")
            t = dtr[["fam", "pipeline"]].assign(loss=loss.to_numpy(),
                                                w=weights(dtr) * dtr.groupby("dataset_id").pipeline.transform("size").to_numpy())
            def cvar(g):
                g = g.sort_values("loss", ascending=False)
                cw = g.w.cumsum(); k = cw <= 0.1 * g.w.sum()
                k.iloc[0] = True
                return float(np.average(g.loss[k], weights=g.w[k]))
            bf = t.groupby("pipeline").apply(cvar).idxmin()
        else:
            bf = dtr.groupby(["fam", "pipeline"]).private_ba.mean().groupby("pipeline").mean().idxmax()
        bfp = d[d.pipeline == bf].set_index(["dataset_id", "epsilon"]).pred
        adv = pick.pred.to_numpy() - bfp.loc[list(zip(pick.dataset_id, pick.epsilon))].to_numpy()
        for fb in ("não", "sim"):
            pp = pick.copy()
            if fb == "sim":
                pp.loc[adv <= TAU, "pipeline"] = bf
            pp = pp.assign(best_fixed=bf)
            m = pp[["dataset_id", "epsilon", "pipeline", "best_fixed"]].merge(
                dte[["dataset_id", "epsilon", "pipeline", "resid", "fam"]])
            res.append(m.assign(variant=name, fallback=fb))
    return pd.concat(res), (pd.concat(preds) if preds else None)

t0 = time.time()
fams = sorted(agg.fam.unique())
parts = Parallel(n_jobs=n_jobs, verbose=5)(delayed(run_fold)(f) for f in fams)
if SAVE_PREDS:
    pd.concat([q for _, q in parts]).to_csv(f"{out}/lodo_predictions.csv", index=False)
parts = [r for r, _ in parts]
res = pd.concat(parts); res.to_csv(f"{out}/lodo_picks.csv", index=False)

# oráculo e melhor fixa (no próprio corpus) como referência
ba = agg.set_index(["dataset_id", "epsilon", "pipeline"]).private_ba
orc = agg.loc[agg.groupby(["dataset_id", "epsilon"]).private_ba.idxmax()]
bf_map = res.drop_duplicates(["dataset_id", "epsilon"]).set_index(["dataset_id", "epsilon"]).best_fixed
orc = orc.assign(best_fixed=bf_map.loc[list(zip(orc.dataset_id, orc.epsilon))].to_numpy())
rng = np.random.default_rng(SEED); rows = []
def summ(name, g):
    g = g.assign(g_bf=[ba.loc[(d, e, p)] - ba.loc[(d, e, b)] for d, e, p, b in
                       zip(g.dataset_id, g.epsilon, g.pipeline, g.best_fixed)])
    row = dict(variante=name)
    for col, lab in (("g_bf", "vs_melhor_fixa"), ("resid", "vs_NB_atual")):
        v = g.groupby("fam")[col].mean().to_numpy()
        b = [rng.choice(v, len(v)).mean() for _ in range(5000)]
        row.update({f"ganho_{lab}": v.mean(), f"ic_lo_{lab}": np.percentile(b, 2.5),
                    f"ic_hi_{lab}": np.percentile(b, 97.5)})
    c = g.groupby(["dataset_id", "epsilon"]).g_bf.mean()
    row.update(cond_pior_1pp=float((c < -0.01).mean()),
               cvar90_perda=float(-np.sort(c.to_numpy())[: max(1, len(c) // 10)].mean()),
               pct_melhor_fixa=float((g.pipeline == g.best_fixed).mean()))
    rows.append(row)
for (v, fb), g in res.groupby(["variant", "fallback"]):
    summ(f"{v} | fallback={fb}", g)
summ("oráculo (média 3 rep.)", orc)
summ("melhor fixa do treino", orc.assign(pipeline=orc.best_fixed))
s = pd.DataFrame(rows).sort_values("ganho_vs_melhor_fixa", ascending=False)
s.to_csv(f"{out}/lodo_summary.csv", index=False)
json.dump(dict(conditions=cond_path, pipelines=pipes, families=len(fams),
               conditions_n=int(len(agg) / len(pipes)), tau=TAU, seconds=time.time() - t0),
          open(f"{out}/lodo_meta.json", "w"), indent=1)
print(s.to_string(index=False)); print(f"tempo {time.time()-t0:.0f}s")

"""Exploratório: regras de fallback a partir das previsões salvas do LODO (15 opções)."""
import numpy as np, pandas as pd
B = "categorical_naive_bayes__laplace_histograms_b8u"; L = "softmax__dp_sgd_clip2_lam0003"
OLD = "categorical_naive_bayes__laplace_histograms"
p = pd.read_csv("results/experiments/lodo_15_preds/lodo_predictions.csv")
p = p[p.variant == "GBM residual"]
ba = p.set_index(["dataset_id", "epsilon", "pipeline"]).private_ba
piv = p.pivot_table(index=["dataset_id", "epsilon"], columns="pipeline", values="pred")
fam = p.drop_duplicates(["dataset_id", "epsilon"]).set_index(["dataset_id", "epsilon"]).fam
top = piv.idxmax(1); topv = piv.max(1)
rows = []; rng = np.random.default_rng(1)
for tgt in (B, L):
    for tau in (0.0, 0.005, 0.01, 0.02, 0.03, 0.05):
        ch = top.where(topv - piv[tgt] > tau, tgt)
        g = pd.Series([ba[(d, e, c)] - ba[(d, e, B)] for (d, e), c in ch.items()], index=ch.index)
        go = pd.Series([ba[(d, e, c)] - ba[(d, e, OLD)] for (d, e), c in ch.items()], index=ch.index)
        f = g.groupby(fam).mean().values
        bs = [f[rng.integers(0, len(f), len(f))].mean() for _ in range(3000)]
        rows.append(dict(fallback=("NB b8u" if tgt == B else "DP-SGD lam0003"), tau=tau, vs_b8u=f.mean(),
                         ic_lo=np.percentile(bs, 2.5), ic_hi=np.percentile(bs, 97.5),
                         vs_NB_antigo=go.groupby(fam).mean().mean(), pior_1pp=(g < -0.01).mean(),
                         cvar90=-np.sort(g.values)[:len(g)//10].mean(), pct_fallback=(ch == tgt).mean()))
s = pd.DataFrame(rows).round(4); s.to_csv("results/experiments/lodo_15_preds/fallback_grid.csv", index=False)
print(s.to_string(index=False))

"""Q1: a conclusão depende da tanh? Mesmas 200 bases, partições e sementes."""
import sys
import numpy as np
import pandas as pd

S = "results/defense-analyses/"
man = pd.read_csv(S + "q1_manifest_sample.csv").set_index("dataset_id")
K = ["dataset_id", "epsilon"]
SHORT = {"categorical_naive_bayes__laplace_histograms": "NB cat", "softmax__dp_sgd_clip2_reg003": "SGD clip2",
         "softmax__dp_sgd": "SGD 25", "softmax__dp_sgd_steps100": "SGD 100",
         "softmax__analytic_gaussian": "Softmax saída", "gnb__gaussian_zcdp_equal": "GNB zCDP",
         "gnb__analytic_gaussian_equal": "GNB analít.", "gnb__laplace_equal": "GNB Laplace",
         "gnb__staircase_vector_equal": "GNB Staircase", "random_forest__laplace_parallel_shards_25": "RF Laplace",
         "random_forest__analytic_gaussian_parallel_shards": "RF Gauss", "random_forest__exponential_majority_parallel_shards": "RF Exp"}


def load(t):
    c = pd.read_csv(S + f"q1_{t}/conditions.csv", usecols=["dataset_id", "epsilon", "pipeline", "repetition",
                                                             "private_ba", "clean_ba"])
    return c.groupby(K + ["pipeline"], as_index=False)[["private_ba", "clean_ba"]].mean()


def summary(a):
    a = a.copy()
    a["best_clean"] = a.groupby(K).clean_ba.transform("max")
    a["regret"] = a.best_clean - a.private_ba  # referência dentro da própria transformação
    top = a.groupby(K).private_ba.transform("max")
    w = a[(top - a.private_ba).abs() < 1e-12].copy()
    w["wt"] = 1 / w.groupby(K).private_ba.transform("size")
    wins = w.groupby("pipeline").wt.sum() / w.groupby(K).ngroups
    reg = a.groupby(["dataset_id", "pipeline"]).regret.mean().groupby("pipeline").mean()
    pba = a.groupby(["dataset_id", "pipeline"]).private_ba.mean().groupby("pipeline").mean()
    cba = a.groupby(["dataset_id", "pipeline"]).clean_ba.mean().groupby("pipeline").mean()
    chance = 1 / man.n_classes
    bc = a.groupby("dataset_id").best_clean.mean()
    near = float(((bc - chance.reindex(bc.index)) <= 0.02).mean())
    t = pd.DataFrame({"win%": 100 * wins, "BA priv": pba, "BA limpa": cba, "regret": reg}).fillna(0)
    t.index = t.index.map(SHORT)
    return t.sort_values("regret"), near, float(bc.mean())


frames = {t: load(t) for t in sys.argv[1:]}
for t, a in frames.items():
    tab, near, bc = summary(a)
    print(f"\n=== {t}: melhor BA limpa média {bc:.3f}; bases ≤ acaso+2pp: {100*near:.1f}%")
    print(tab.round(4).to_string())
# mudança de ordem entre transformações: correlação de Spearman dos postos por condição
if len(frames) >= 2:
    names = list(frames)
    base = frames[names[0]].set_index(K + ["pipeline"]).private_ba
    for other in names[1:]:
        o = frames[other].set_index(K + ["pipeline"]).private_ba.reindex(base.index)
        d = pd.DataFrame({"a": base, "b": o}).reset_index()
        rho = d.groupby(K).apply(lambda g: g.a.rank().corr(g.b.rank(), method="pearson"), include_groups=False)
        wa = d.loc[d.groupby(K).a.idxmax()].set_index(K).pipeline
        wb = d.loc[d.groupby(K).b.idxmax()].set_index(K).pipeline
        print(f"\n{names[0]} vs {other}: Spearman mediano dos postos {rho.median():.2f}; "
              f"mesma vencedora em {100*(wa == wb).mean():.1f}% das condições; "
              f"ΔBA limpa (melhor opção) média {100*(frames[other].groupby(K).clean_ba.max().mean() - frames[names[0]].groupby(K).clean_ba.max().mean()):.2f} p.p.")

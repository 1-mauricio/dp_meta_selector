"""Q2: piso de ruído para a dispersão de vencedoras e para os 50,7%."""
import itertools
import json
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare

import argparse
from pathlib import Path
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--conditions', type=Path, required=True)
parser.add_argument('--manifest', type=Path, default=Path('data/openml_manifest_984_completed.csv'))
parser.add_argument('--output', type=Path, default=Path('results/dissertation/noise'))
args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
C = pd.read_csv(args.conditions)
C["model"] = C.pipeline.map(lambda p: "softmax" if p.startswith("softmax") else p.split("__")[0])
K = ["dataset_id", "epsilon"]
man = pd.read_csv(args.manifest).drop_duplicates("dataset_id").set_index("dataset_id")


def winsets(frame, value, col):
    top = frame.groupby(K)[value].transform("max")
    w = frame[(top - frame[value]).abs() < 1e-12]
    return w.groupby(K)[col].agg(frozenset)


def changed(a, b):
    a, b = a.align(b, join="inner")
    return pd.Series([not (x & y) for x, y in zip(a, b)], a.index)


def top_share(frame, value="private_ba"):
    top = frame.groupby(K)[value].transform("max")
    w = frame[(top - frame[value]).abs() < 1e-12].copy()
    w["wt"] = 1 / w.groupby(K)[value].transform("size")
    s = w.groupby("pipeline").wt.sum() / w.groupby(K).ngroups
    return s.sort_values(ascending=False)


def analyse(C, label):
    out = {"subset": label, "datasets": int(C.dataset_id.nunique())}
    reps = sorted(C.repetition.unique())
    per = {r: C[C.repetition == r] for r in reps}
    mean = C.groupby(K + ["pipeline", "model"], as_index=False)[["private_ba", "clean_ba"]].mean()

    # (a) métrica da dissertação: modelo limpo (média) vs modelo privado (média)
    out["dp_changes_model_mean3"] = changed(winsets(mean, "clean_ba", "model"),
                                            winsets(mean, "private_ba", "model")).mean()
    # (b) mesma métrica numa única repetição (nível de ruído de 1 rep)
    pairs = list(itertools.combinations(reps, 2))
    out["dp_changes_model_single_rep"] = np.mean([
        changed(winsets(per[r], "clean_ba", "model"), winsets(per[r], "private_ba", "model")).mean()
        for r in reps])
    # (c) piso: o modelo vencedor SEM ruído muda entre duas repetições (só partição muda)
    out["floor_clean_model_rep_vs_rep"] = np.mean([
        changed(winsets(per[a], "clean_ba", "model"), winsets(per[b], "clean_ba", "model")).mean()
        for a, b in pairs])
    # (d) piso: o modelo vencedor COM DP muda entre repetições
    out["floor_private_model_rep_vs_rep"] = np.mean([
        changed(winsets(per[a], "private_ba", "model"), winsets(per[b], "private_ba", "model")).mean()
        for a, b in pairs])
    # (e) opção (par) vencedora privada muda entre repetições
    out["floor_private_option_rep_vs_rep"] = np.mean([
        changed(winsets(per[a], "private_ba", "pipeline"), winsets(per[b], "private_ba", "pipeline")).mean()
        for a, b in pairs])
    # (f) split-half: vencedora da média de 2 reps prevê a da 3ª?
    hits = []
    for held in reps:
        rest = C[C.repetition != held].groupby(K + ["pipeline"], as_index=False).private_ba.mean()
        hits.append(1 - changed(winsets(rest, "private_ba", "pipeline"),
                                winsets(per[held], "private_ba", "pipeline")).mean())
    out["winner_2reps_predicts_3rd"] = float(np.mean(hits))

    # (g) decomposição de variância da BA privada centrada por condição
    x = C.copy()
    x["c"] = x.private_ba - x.groupby(K + ["repetition"]).private_ba.transform("mean")
    cell = x.groupby(K + ["pipeline"]).c
    resid_var = cell.var(ddof=1).mean()  # variância entre repetições (ruído DP + partição)
    inter_var_obs = cell.mean().var(ddof=1)  # variância das médias de 3 reps
    signal = inter_var_obs - resid_var / 3
    out["resid_sd_pp"] = 100 * np.sqrt(resid_var)
    out["signal_sd_pp"] = 100 * np.sqrt(max(signal, 0))
    out["reliability_mean3"] = signal / inter_var_obs  # fração da variância que é sinal

    share = top_share(mean)
    out["top_winner"] = share.index[0]
    out["top_winner_share"] = float(share.iloc[0])
    # Friedman/Kendall W sobre regret médio por conjunto
    reg = C.groupby(["dataset_id", "pipeline"]).portfolio_regret.mean().unstack()
    chi2, p = friedmanchisquare(*[reg[c] for c in reg.columns])
    n, kk = reg.shape
    out["friedman_chi2"] = chi2
    out["kendall_W"] = chi2 / (n * (kk - 1))
    return out


best_clean = C.groupby(["dataset_id", "epsilon", "repetition"]).clean_ba.max().groupby("dataset_id").mean()
near = (best_clean - 1 / man.n_classes.reindex(best_clean.index)) <= 0.02
rows = [analyse(C, "todos"),
        analyse(C[C.dataset_id.isin(best_clean.index[~near])], "informativos"),
        analyse(C[C.dataset_id.isin(best_clean.index[near])], "próximos do acaso")]
df = pd.DataFrame(rows).set_index("subset").T
pd.set_option("display.width", 200)
print(df.to_string())
df.to_csv(args.output / "q2_result.csv")
by_epsilon = []

# também por ε, conjunto informativo
inf = C[C.dataset_id.isin(best_clean.index[~near])]
for e, g in inf.groupby("epsilon"):
    r = analyse(g, str(e))
    print(e, round(r["dp_changes_model_mean3"], 3), round(r["floor_clean_model_rep_vs_rep"], 3),
          round(r["floor_private_option_rep_vs_rep"], 3), round(r["reliability_mean3"], 3))

    by_epsilon.append(r)
pd.DataFrame(by_epsilon).to_csv(args.output / "informative_by_epsilon.csv", index=False)

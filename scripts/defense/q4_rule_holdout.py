"""Q4 (holdout): a regra de um limiar, congelada como saiu do desenvolvimento
(limiar 2,2 em log10(nε/(d+1)), DP-SGD clip 2 abaixo, NB categórico acima e
nos conjuntos só nominais), aplicada às 41 famílias. Descritivo."""
import json
import numpy as np
import pandas as pd
import dpmeta.portfolio_benchmark as pb

H = "results/openml-holdout-41/"
NBC, SGD = "categorical_naive_bayes__laplace_histograms", "softmax__dp_sgd_clip2_reg003"
THRESHOLD, SEED, DEV_MAX_N = 2.2, 20260918, 51336

man = pd.read_csv("data/openml_confirmatory_manifest_41_combined.csv").set_index("dataset_id")
c = pd.read_csv(H + "conditions.csv", usecols=["dataset_id", "epsilon", "pipeline", "portfolio_regret"])
agg = c.groupby(["dataset_id", "epsilon", "pipeline"]).portfolio_regret.mean().unstack()
ids = agg.index.get_level_values(0)
m = man.reindex(ids)
x = np.log10(m.n_instances.to_numpy(float) * agg.index.get_level_values(1).to_numpy()
             / (m.n_features.to_numpy(float) + 1))
nominal_only = (m.n_nominal_features == m.n_features).to_numpy()
use_sgd = (x < THRESHOLD) & ~nominal_only
r = pd.DataFrame({"rule": np.where(use_sgd, agg[SGD], agg[NBC]), "nbc": agg[NBC].to_numpy(),
                  "h27": agg[SGD].to_numpy()}, index=agg.index).groupby(level=0).mean()
methods = pd.read_csv(H + "method_regret.csv")
r["selector"] = methods[methods.method == "selector"].groupby("dataset_id").regret.mean()
inside = (man.n_instances.reindex(r.index) <= DEV_MAX_N).to_numpy()


def gain(reference, mask=slice(None)):
    return pb._bootstrap_ci((r[reference] - r.rule).to_numpy()[mask], SEED)


out = {"regret": r.mean().to_dict(),
       "rule_vs_h27": gain("h27"), "rule_vs_selector": gain("selector"), "rule_vs_nbc": gain("nbc"),
       "inside_dev_range_families": int(inside.sum()),
       "rule_vs_nbc_inside_dev_range": gain("nbc", inside),
       "share_nbc_outside_dev_range": float(1 - use_sgd[~(m.n_instances <= DEV_MAX_N).to_numpy()].mean())}
print(json.dumps(out, indent=1))
json.dump(out, open("results/defense-analyses/q4_rule_holdout.json", "w"), indent=1)

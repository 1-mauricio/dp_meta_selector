"""Q7: ganho do LODO com peso igual por família semântica e bootstrap sobre
as famílias semânticas (frente à melhor opção fixa e frente ao H27)."""
import json
import pandas as pd
import dpmeta.portfolio_benchmark as pb

SEED = 20260917
man = pd.read_csv("data/openml_manifest_984_completed.csv").drop_duplicates("dataset_id").set_index("dataset_id")
lodo = pd.read_csv("results/portfolio-984-shared-selector-lodo/outer_predictions.csv")
d = lodo[lodo.feature_mode == "public_plus_dp"].groupby("dataset_id")[
    ["selector_regret", "best_fixed_regret", "h27_fixed_regret"]].mean()
sem = man.semantic_family_id.reindex(d.index)
out = {"semantic_families": int(sem.nunique())}
for name, column in (("vs_best_fixed", "best_fixed_regret"), ("vs_h27", "h27_fixed_regret")):
    per_family = (d[column] - d.selector_regret).groupby(sem.to_numpy()).mean().to_numpy()
    out[name] = pb._bootstrap_ci(per_family, SEED)
print(json.dumps(out, indent=1))
json.dump(out, open("results/defense-analyses/q7_semantic_bootstrap.json", "w"), indent=1)

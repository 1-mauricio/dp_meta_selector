"""Q1: amostra de conjuntos e manifesto com limites numéricos 'declarados'.

Os limites são o mínimo e o máximo observados em cada coluna: simulam um
curador que conhece o domínio de cada atributo. É um diagnóstico do custo da
tanh, não um procedimento implantável tal como está (ler min/max custa ε).
"""
import json
import sys
import numpy as np
import pandas as pd

S = sys.argv[1]
N = int(sys.argv[2])
m = pd.read_csv("data/openml_manifest_984_completed.csv")
has_numeric = m.public_schema_json.map(lambda s: "numeric" in json.loads(s)["feature_kinds"])
pool = m[has_numeric & (m.n_instances <= 20000)]
sample = pool.sample(n=N, random_state=20260926).sort_values("dataset_id")
sample.to_csv(f"{S}/q1_manifest_sample.csv", index=False)


def with_bounds(row):
    schema = json.loads(row.public_schema_json)
    frame = pd.read_parquet(row.local_data_path)
    bounds = []
    for name, kind in zip(schema["feature_names"], schema["feature_kinds"]):
        if kind != "numeric":
            bounds.append(None)
            continue
        v = pd.to_numeric(frame[name], errors="coerce").to_numpy(float)
        v = v[np.isfinite(v)]
        lo, hi = (float(v.min()), float(v.max())) if v.size else (0.0, 1.0)
        if hi <= lo:
            lo, hi = lo - 0.5, hi + 0.5
        bounds.append([lo, hi])
    schema["numeric_bounds"] = bounds
    return json.dumps(schema)


b = sample.copy()
b["public_schema_json"] = b.apply(with_bounds, axis=1)
b.to_csv(f"{S}/q1_manifest_sample_bounds.csv", index=False)
print(len(sample), sample.n_instances.describe().to_dict())

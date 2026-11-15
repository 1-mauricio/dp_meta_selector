"""Q3: por que a softmax (limpa e DP-SGD) cai ao acaso no holdout."""
import json
import numpy as np
import pandas as pd
from dpmeta.schema import PublicSchema
from dpmeta.classifiers import fit_clean_softmax
from dpmeta.model import balanced_accuracy
from dpmeta.openml_benchmark import _get_openml_data, _openml, stratified_holdout_indices

S = "results/defense-analyses/"
t = pd.read_csv(S + "q3_holdout.csv", index_col=0)
man = pd.read_csv("data/openml_confirmatory_manifest_41_combined.csv").set_index("dataset_id")
openml = _openml(__import__("pathlib").Path(".cache/openml"))
rows = []
rng = np.random.default_rng(0)
EXCLUDED = {382, 1039, 42140, 41216}  # alta dimensão: custo computacional
for did in [i for i in t.index if i not in EXCLUDED]:
    row = man.loc[did].copy()
    row["dataset_id"] = did
    try:
        x_t, y, schema = _get_openml_data(openml, row, numeric_transform="tanh")
        x_a, _, _ = _get_openml_data(openml, row, numeric_transform="asinh10")
    except Exception as exc:  # noqa: BLE001
        rows.append({"dataset_id": did, "error": str(exc)[:80]})
        continue
    num = np.array([k == "numeric" for k in schema.feature_kinds])
    enc_num = []
    for kind, dom in zip(schema.feature_kinds, schema.nominal_domains):
        enc_num += [True] if kind == "numeric" else [False] * (len(dom) + 1)
    enc_num = np.array(enc_num)
    sat = float((np.abs(x_t[:, enc_num]) > 0.995).mean()) if enc_num.any() else np.nan
    counts = np.bincount(y, minlength=len(schema.classes))
    maj = counts.max() / counts.sum()
    # diagnóstico limpo numa subamostra estratificada de até 20k linhas
    idx = np.arange(len(y))
    if len(idx) > 20000:
        idx = rng.choice(idx, 20000, replace=False)
    tr, te = stratified_holdout_indices(y[idx], 1)
    k = len(schema.classes)
    out = {"dataset_id": did, "n": len(y), "k": k, "majority": maj, "saturated": sat}
    for name, x in (("tanh", x_t), ("asinh10", x_a)):
        xs = x[idx]
        for lam in ((0.03, 1e-3) if x.shape[1] <= 300 else (0.03,)):
            fit = fit_clean_softmax(xs[tr], y[idx][tr], k, regularization=lam)
            out[f"{name}_lam{lam}"] = balanced_accuracy(y[idx][te], fit.model.predict(xs[te]))

    rows.append(out)
    print(out, flush=True)
d = pd.DataFrame(rows).set_index("dataset_id").join(t[["gap", "sgd_clean", "nbc_clean"]])
d.to_csv(S + "q3_diag_final.csv")
print(d.round(3).sort_values("gap").to_string())

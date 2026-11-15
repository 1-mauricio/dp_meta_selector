from pathlib import Path
import numpy as np,pandas as pd
from dpmeta import dissertation_selector as sel
from dpmeta.openml_benchmark import _load_manifest
OUT=Path('results/dissertation')
man=_load_manifest('data/openml_manifest_984_completed.csv')
c=pd.read_csv('results/portfolio-984-sgd/conditions_15.csv');a,m,pipes=sel.prepare(c,man);fmap=m.set_index('dataset_id').family_id
w=a.pivot(index=['dataset_id','epsilon'],columns='pipeline',values='private_ba')
tied=w.sub(w.max(axis=1),axis=0).abs().lt(1e-12);shares=tied.div(tied.sum(axis=1),axis=0)
families=sorted(m.family_id.unique());counts=m.family_id.value_counts().reindex(families).to_numpy()
rng=np.random.default_rng(sel.SEED)
# Counts of 20k family bootstrap samples; reuse the same paired resamples at all ε.
boots=rng.multinomial(len(families),np.repeat(1/len(families),len(families)),size=20000)
den=boots@counts;rows=[]
for epsilon in [*sorted(a.epsilon.unique()),'Total']:
    values=shares.groupby('dataset_id').mean() if epsilon=='Total' else shares.xs(epsilon,level='epsilon')
    sums=values.groupby(fmap.reindex(values.index)).sum().reindex(families)
    means=(boots@sums.to_numpy())/den[:,None]
    for j,pipeline in enumerate(w.columns):
        rows.append(dict(epsilon=epsilon,pipeline=pipeline,share=values[pipeline].mean(),
            ci_low=np.quantile(means[:,j],.025),ci_high=np.quantile(means[:,j],.975),
            estimand='equal dataset; percentile family cluster bootstrap',draws=20000,seed=sel.SEED))
pd.DataFrame(rows).to_csv(OUT/'winner_share_ci.csv',index=False)

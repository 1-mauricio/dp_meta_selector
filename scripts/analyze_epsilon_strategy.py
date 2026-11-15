"""Five semantic folds, GBM residual+fallback shared vs one model per ε."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import pandas as pd,numpy as np
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits
from dpmeta import dissertation_selector as sel
from dpmeta.openml_benchmark import _load_manifest
from scripts.dissertation_analysis import dump,paired_distribution,fam,mean_ci
out=Path('results/dissertation/strategies');out.mkdir(parents=True,exist_ok=True)
a,m,pipes=sel.prepare(pd.read_csv('results/portfolio-984-sgd/conditions_15.csv'),_load_manifest('data/openml_manifest_984_completed.csv'))
ids=m.sort_values('dataset_id');frames=[]
for fold,(tr,te) in enumerate(GroupKFold(5).split(ids,groups=ids.family_id)):
    train=a[a.dataset_id.isin(ids.iloc[tr].dataset_id)];test=a[a.dataset_id.isin(ids.iloc[te].dataset_id)]
    for e in sorted(a.epsilon.unique()):
        path=out/f'{fold}_{e}.csv'
        if path.exists():frames.append(pd.read_csv(path));continue
        tr_e=train[train.epsilon==e];te_e=sel.ordered(test[test.epsilon==e],pipes)
        with threadpool_limits(limits=1):
            model=sel.fit(tr_e,m,pipes);choice=sel.pick(te_e,model.predict(sel.feature_matrix(te_e,m,pipes)))
        pred=sel.predictions(te_e,choice,tr_e,fold);pred.to_csv(path,index=False);frames.append(pred)
    print('fold',fold,flush=True)
pred=pd.concat(frames,ignore_index=True)
# All methods are scored against the same training-global fixed comparator;
# per-epsilon models still use their own training-epsilon fallback internally.
score=a.set_index(['dataset_id','epsilon','pipeline']).portfolio_regret
for fold,g in pred.groupby('fold'):
    train=a[~a.dataset_id.isin(g.dataset_id)];fixed=sel.best_fixed(train)
    pred.loc[g.index,'fallback_pipeline_per_epsilon']=g.best_fixed_pipeline.to_numpy()
    pred.loc[g.index,'best_fixed_pipeline']=fixed
    pred.loc[g.index,'best_fixed_regret']=[score.loc[(d,e,fixed)] for d,e in zip(g.dataset_id,g.epsilon)]
pred.to_csv(out/'per_epsilon_predictions.csv',index=False)
shared=pd.read_csv('results/dissertation/extras/fivefold_predictions.csv')
summary={'per_epsilon_vs_fixed':paired_distribution(fam(pred,'selector_regret'),fam(pred,'best_fixed_regret')),
         'per_epsilon_vs_shared':paired_distribution(fam(pred,'selector_regret'),fam(shared,'selector_regret'))}
dump(out/'summary.json',summary)
rows=[]
for e,g in pred.groupby('epsilon'):
    rows.append(dict(epsilon=e,regret=fam(g,'selector_regret').mean(),**mean_ci(fam(g,'best_fixed_regret')-fam(g,'selector_regret'))))
pd.DataFrame(rows).to_csv(out/'by_epsilon.csv',index=False)

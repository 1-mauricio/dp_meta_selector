"""Resumable 5-fold semantic-family learning curves: RF-12 and GBM-13."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[name]='1'
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits
from dpmeta import dissertation_selector as sel
from dpmeta import portfolio_benchmark as pb
from dpmeta.openml_benchmark import _load_manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--conditions',type=Path,default=Path('results/portfolio-984-sgd/conditions_15.csv'))
    p.add_argument('--manifest',type=Path,default=Path('data/openml_manifest_984_completed.csv'))
    p.add_argument('--output',type=Path,default=Path('results/dissertation/learning_curve'))
    p.add_argument('--fold',type=int,choices=range(5),help='run one independent outer fold')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    a,m,pipes=sel.prepare(pd.read_csv(args.conditions),_load_manifest(args.manifest));a=sel.ordered(a,pipes)
    x=sel.feature_matrix(a,m,pipes);ids=m.sort_values('dataset_id')
    nb=a[a.pipeline==sel.NB].set_index(['dataset_id','epsilon']).private_ba
    residual=a.private_ba.to_numpy()-nb.loc[list(zip(a.dataset_id,a.epsilon))].to_numpy()
    for fold,(tr,te) in enumerate(GroupKFold(5).split(ids,groups=ids.family_id)):
        if args.fold is not None and fold != args.fold:continue
        families=np.unique(ids.iloc[tr].family_id);te_mask=a.dataset_id.isin(ids.iloc[te].dataset_id).to_numpy();test=a[te_mask]
        for fraction in (.1,.25,.5,.75,1.):
            for draw in range(2 if fraction<1 else 1):
                keep=np.random.default_rng(1000*fold+draw).choice(families,max(5,round(fraction*len(families))),replace=False)
                tr_mask=a.family_id.isin(keep).to_numpy();train=a[tr_mask];fixed=sel.best_fixed(train)
                for modelname in ('RF-12','GBM residual + fallback GBM-13'):
                    checkpoint=args.output/f'{fold}_{fraction}_{draw}_{modelname[:2]}.csv'
                    if checkpoint.exists():continue
                    with threadpool_limits(limits=1):
                        if modelname.startswith('RF'):
                            model=RandomForestRegressor(n_estimators=1000,min_samples_leaf=3,max_features=.9,
                                  max_depth=6,max_samples=.5,random_state=0,n_jobs=1)
                            weights=sel.weights(train);weights/=weights.sum()
                            model.fit(x[tr_mask],train.private_ba.to_numpy(),sample_weight=weights)
                        else:
                            raw=sel.gbm().fit(x[tr_mask],residual[tr_mask],sample_weight=sel.weights(train))
                            model=sel.FallbackPredictor(raw,pipes,fixed)
                        chosen=sel.pick(test,model.predict(x[te_mask]))
                    pred=sel.predictions(test,chosen,train,fold)
                    pred['gain']=pred.best_fixed_regret-pred.selector_regret
                    pred=pred.assign(model=modelname,fraction=fraction,draw=draw,train_families=len(keep),train_datasets=train.dataset_id.nunique())
                    pred.to_csv(checkpoint,index=False)
                    print(fold,fraction,draw,modelname,len(keep),pred.groupby('family_id').gain.mean().mean(),flush=True)
    if args.fold is not None:
        print('completed independent fold',args.fold,flush=True)
        return
    data=pd.concat([pd.read_csv(p) for p in args.output.glob('*.csv') if p.name not in ['predictions.csv','summary.csv']])
    data['family_id']=data.family_id.astype(str)
    data.to_csv(args.output/'predictions.csv',index=False)
    rows=[]
    for (model,frac),g in data.groupby(['model','fraction']):
        # Average repeated training draws within held-out family before bootstrap.
        means=g.groupby(['draw','family_id','dataset_id']).gain.mean().groupby(['family_id','dataset_id']).mean().groupby('family_id').mean()
        mean,lo,hi=pb._bootstrap_ci(means.to_numpy(),sel.SEED)
        rows.append(dict(model=model,fraction=frac,train_families=g.groupby(['fold','draw']).train_families.first().mean(),
            train_datasets=g.groupby(['fold','draw']).train_datasets.first().mean(),gain=mean,ci_low=lo,ci_high=hi))
    pd.DataFrame(rows).to_csv(args.output/'summary.csv',index=False)

if __name__=='__main__':main()

"""Separate portfolio expansion from attrition, using common datasets and semantic inference."""
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import friedmanchisquare,studentized_range
from dpmeta import dissertation_selector as sel
from dpmeta.openml_benchmark import _load_manifest
from scripts.dissertation_analysis import mean_ci,dump
out=Path('results/dissertation');m=_load_manifest('data/openml_manifest_984_completed.csv')
c=pd.read_csv('results/portfolio-984-sgd/conditions_15.csv');ids=set(c.dataset_id)
old=pd.read_csv('results/portfolio-984-current-slim/conditions.csv')
rows=[]
for name,source in [('v4_984_12',old),('common_973_12',c[c.pipeline.isin(sel.pb.PIPELINES)]),('gbm13_973_13',c[~c.pipeline.isin(sel.pb.SGD_NEW_PIPELINES)]),('gbm15_973_15',c)]:
    a,meta,pipes=sel.prepare(source,m);w=a.pivot(index=['dataset_id','epsilon'],columns='pipeline',values='private_ba')
    top=w.max(axis=1);wins=w.sub(top,axis=0).abs().lt(1e-12);share=wins.div(wins.sum(axis=1),axis=0).mean()
    fixed=sel.empirical_best(a);mean=a.groupby(['family_id','pipeline']).private_ba.mean().groupby('pipeline').mean()
    r=a.pivot_table(index='dataset_id',columns='pipeline',values='portfolio_regret');chi,p=friedmanchisquare(*[r[k] for k in r])
    row=dict(portfolio=name,datasets=len(meta),families=meta.family_id.nunique(),options=len(pipes),conditions=len(w),
        best_fixed=fixed,best_fixed_ba_equal_family=mean[fixed],oracle_ba_equal_dataset=top.mean(),
        oracle_gain_equal_dataset=(top-w[fixed]).mean(),friedman_chi2=chi,friedman_p=p,kendall_W=chi/(len(r)*(len(pipes)-1)),
        ties=int(wins.sum(axis=1).gt(1).sum()),all_tied=int(wins.all(axis=1).sum()),spread_median=(top-w.min(axis=1)).median(),
        top_two_gap_median=np.median(np.sort(w,axis=1)[:,-1]-np.sort(w,axis=1)[:,-2]))
    rows.append(row)
    share.rename('share').to_csv(out/f'{name}_winner_shares.csv')
pd.DataFrame(rows).to_csv(out/'portfolio_expansion_vs_attrition.csv',index=False)
a,meta,pipes=sel.prepare(c,m);fmap=meta.set_index('dataset_id').family_id
w=a.pivot(index=['dataset_id','epsilon'],columns='pipeline',values='private_ba');oracle=w.max(axis=1)
contributions=[]
for pipe in pipes:
    ds=(oracle-w.drop(columns=pipe).max(axis=1)).groupby('dataset_id').mean()
    f=pd.DataFrame({'value':ds,'family':fmap.reindex(ds.index)})
    byfam=f.groupby('family').value.agg(['sum','count','mean']);rng=np.random.default_rng(sel.SEED)
    ix=rng.integers(len(byfam),size=(20000,len(byfam)))
    boots=byfam['sum'].to_numpy()[ix].sum(axis=1)/byfam['count'].to_numpy()[ix].sum(axis=1)
    contributions.append(dict(pipeline=pipe,mean_equal_dataset=ds.mean(),ci_low_cluster_dataset=np.quantile(boots,.025),
         ci_high_cluster_dataset=np.quantile(boots,.975),**{f'family_{k}':v for k,v in mean_ci(byfam['mean']).items()}))
pd.DataFrame(contributions).to_csv(out/'portfolio_contribution_ci.csv',index=False)
# Contributions of jointly removing model families (single-option contributions do not add).
groups={'three_GNB_except_zcdp':[p for p in pipes if p.startswith('gnb') and 'zcdp' not in p],
        'all_GNB':[p for p in pipes if p.startswith('gnb')],
        'all_forests':[p for p in pipes if p.startswith('random_forest')],
        'all_categorical_NB':[p for p in pipes if p.startswith('categorical_naive_bayes')]}
rows=[]
for name,removed in groups.items():
    ds=(oracle-w.drop(columns=removed).max(axis=1)).groupby('dataset_id').mean()
    family=ds.groupby(fmap.reindex(ds.index)).mean()
    rows.append(dict(group=name,removed=';'.join(removed),mean_equal_dataset=ds.mean(),**mean_ci(family)))
pd.DataFrame(rows).to_csv(out/'portfolio_group_contribution.csv',index=False)

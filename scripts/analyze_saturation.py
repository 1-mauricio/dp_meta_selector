"""Explicit reconstruction of the saturation correlation on the retained numeric-only corpus.

The historical correlation script is absent. This reconstruction declares the
0.995 threshold used by the archived holdout diagnostic rather than guessing it.
"""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='1'
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np,pandas as pd
from scipy.stats import spearmanr
from dpmeta import selector_portfolio13 as sel,portfolio_benchmark as pb
from dpmeta.openml_benchmark import _load_manifest,_openml,_get_openml_data
from scripts.parallel_nb8u import disk_guard,source_row
from scripts.dissertation_analysis_portfolio13 import dump,mean_ci
OUT=Path('results/dissertation-portfolio13/saturation')


def work(raw):
    disk_guard();x,_,_=_get_openml_data(_openml(Path('.cache/openml')),source_row(pd.Series(raw)),numeric_transform='tanh')
    result=dict(dataset_id=int(raw['dataset_id']),saturated=float((abs(x)>.995).mean()),cells=int(x.size))
    disk_guard();return result


def main():
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'datasets').mkdir(exist_ok=True)
    c=pd.read_csv('results/portfolio-984-nb8u/conditions_13.csv')
    a,m,pipes=sel.prepare(c,_load_manifest('data/openml_manifest_984_completed.csv'))
    m=m[m.n_nominal_features.eq(0)]
    pending=[r.to_dict() for _,r in m.iterrows() if not (OUT/'datasets'/f'{int(r.dataset_id)}.json').exists()]
    failures=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        futures={pool.submit(work,r):int(r['dataset_id']) for r in pending}
        for future in as_completed(futures):
            d=futures[future]
            try:
                result=future.result();dump(OUT/'datasets'/f'{d}.json',result)
            except Exception as ex:failures.append(dict(dataset_id=d,error=str(ex)))
    dump(OUT/'failures.json',failures)
    sat=pd.DataFrame([json.loads(p.read_text()) for p in (OUT/'datasets').glob('*.json')]).set_index('dataset_id').sort_index()
    if len(sat)!=len(m):raise ValueError('Incomplete saturation diagnostic; see failures.json')
    w=a[a.dataset_id.isin(m.dataset_id)].pivot(index=['dataset_id','epsilon'],columns='pipeline',values='private_ba')
    rows=[]
    for name,nb,others in [('gbm13_b8u_vs_other12',sel.NB8U,[p for p in pipes if p!=sel.NB8U]),
                           ('v4_NB_vs_other11_common',sel.NB,[p for p in pb.PIPELINES if p!=sel.NB])]:
        top=w[others+[nb]].max(axis=1)
        tie=w[others+[nb]].sub(top,axis=0).abs().lt(1e-12)
        win=tie[nb]/tie.sum(axis=1)
        for definition in ['mean_condition_gap','gap_dataset_means']:
            gap=((w[nb]-w[others].max(axis=1)).groupby('dataset_id').mean() if definition=='mean_condition_gap'
                 else w[nb].groupby('dataset_id').mean()-w[others].groupby('dataset_id').mean().max(axis=1))
            frame=sat.join(gap.rename('gap')).join(win.groupby('dataset_id').mean().rename('win_share'))
            frame['tercile']=pd.qcut(frame.saturated.rank(method='first'),3,labels=['low','mid','high'])
            r,p=spearmanr(frame.saturated,frame.gap)
            for band,g in frame.groupby('tercile',observed=True):
                rows.append(dict(comparison=name,gap_definition=definition,tercile=str(band),datasets=len(g),
                     saturation_mean=g.saturated.mean(),gap_mean=g.gap.mean(),win_share=g.win_share.mean(),
                     spearman_r=r,spearman_p=p,**mean_ci(g.gap)))
            frame.to_csv(OUT/f'{name}_{definition}.csv')
    pd.DataFrame(rows).to_csv(OUT/'summary.csv',index=False)
    dump(OUT/'config.json',dict(threshold_abs_tanh=.995,datasets=len(m),workers=2,
         status='reconstrução explícita; script/limiar original da correlação em 588 datasets não foi preservado',
         tercile_ties='ordem de dataset_id para dividir empates de saturação; qcut(rank(method=first))'))

if __name__=='__main__':main()

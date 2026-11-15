"""Add NB b8u to the existing numerical-transform diagnostic; no replacements."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
import json,time,argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import pandas as pd,numpy as np
from threadpoolctl import threadpool_limits
from dpmeta import portfolio_benchmark as pb,selector_portfolio13 as sel
from dpmeta.openml_benchmark import _load_manifest,_openml
from scripts.parallel_nb8u import source_row,disk_guard
from scripts.dissertation_analysis_portfolio13 import mean_ci,dump
OUT=Path('results/dissertation-portfolio13/transformations')
SOURCE=Path('results/defense-analyses')


def run(raw,transform,refs):
    disk_guard()
    with threadpool_limits(limits=1):
        rows=pb.run_dataset(_openml(Path('.cache/openml')),source_row(pd.Series(raw)),
            pb.DEFAULT_EPSILONS,3,20260917,pipelines=(pb.NB8U_PIPELINE,),include_research_features=False,
            numeric_transform=transform,best_clean_by_condition=refs)
    disk_guard()
    cols=['dataset_id','epsilon','pipeline','repetition','private_ba','clean_ba',
          'best_clean_portfolio_ba','portfolio_regret','pipeline_privacy_penalty']
    return [{k:r[k] for k in cols} for r in rows]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--workers',type=int,default=4);args=p.parse_args()
    if not 1<=args.workers<=8:p.error('1–8 workers')
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'datasets').mkdir(exist_ok=True)
    dev=pd.read_csv('results/portfolio-984-nb8u/conditions_13.csv')
    valid=set(dev.dataset_id);frames={};tasks=[];manifests={}
    for transform in ['tanh','public_bounds','asinh10']:
        source=SOURCE/f'q1_{transform}';config=json.loads((source/'run_config.json').read_text())
        manifestfile=SOURCE/('q1_manifest_sample_bounds.csv' if transform=='public_bounds' else 'q1_manifest_sample.csv')
        m=_load_manifest(manifestfile);m=m[m.dataset_id.isin(valid)];manifests[transform]=m
        cols=['dataset_id','epsilon','pipeline','repetition','private_ba','clean_ba','best_clean_portfolio_ba','portfolio_regret','pipeline_privacy_penalty']
        old=pd.read_csv(source/'conditions.csv',usecols=cols);old=old[old.dataset_id.isin(valid)]
        frames[transform]=old
        for _,row in m.iterrows():
            d=int(row.dataset_id);path=OUT/'datasets'/f'{transform}_{d}.csv'
            if transform=='tanh':
                dev[(dev.dataset_id==d)&(dev.pipeline==sel.NB8U)][cols].to_csv(path,index=False)
            if path.exists():continue
            refs=old[old.dataset_id==d].groupby(['epsilon','repetition']).best_clean_portfolio_ba.first().to_dict()
            tasks.append((row.to_dict(),transform,refs,path))
    failures=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(run,raw,t,ref):(raw['dataset_id'],t,path) for raw,t,ref,path in tasks}
        for f in as_completed(futures):
            d,t,path=futures[f];disk_guard()
            try:pd.DataFrame(f.result()).to_csv(path,index=False);print('DONE',d,t,flush=True)
            except Exception as ex:failures.append(dict(dataset_id=d,transform=t,error=str(ex)))
    dump(OUT/'failures.json',failures)
    tables=[];wideframes={}
    for transform,old in frames.items():
        new=pd.concat([pd.read_csv(p) for p in (OUT/'datasets').glob(f'{transform}_*.csv')])
        if set(new.dataset_id)!=set(old.dataset_id):raise ValueError('Incomplete transformation; see failures')
        combined=pd.concat([old,new],ignore_index=True);combined.to_csv(OUT/f'conditions_{transform}.csv',index=False)
        a=combined.groupby(['dataset_id','epsilon','pipeline'],as_index=False)[['private_ba','clean_ba']].mean()
        m=manifests[transform].set_index('dataset_id');w=a.pivot(index=['dataset_id','epsilon'],columns='pipeline',values='private_ba')
        clean=a.groupby(['dataset_id','epsilon']).clean_ba.max().groupby('dataset_id').mean()
        wt=w.sub(w.max(axis=1),axis=0).abs().lt(1e-12);wt=wt.div(wt.sum(axis=1),axis=0)
        per=(w[sel.NB8U]-w[pb.H27_PIPELINE]).groupby('dataset_id').mean()
        summary=dict(transform=transform,datasets=len(m),best_clean=clean.mean(),near_chance=((clean-1/m.n_classes)<=.02).mean(),
                     nb8u_win_share=wt[sel.NB8U].mean(),**mean_ci(per))
        tables.append(summary);wideframes[transform]=w
    pd.DataFrame(tables).to_csv(OUT/'summary.csv',index=False)
    dump(OUT/'provenance.json',dict(status='diagnóstico exploratório; limites min/max observados',
         original_sample=200,retained_sample=len(manifests['tanh']),excluded=sorted(set(_load_manifest(SOURCE/'q1_manifest_sample.csv').dataset_id)-valid),
         seed=20260917,repeats=3,workers=args.workers,regret_reference='referência original preservada no CSV; tabela usa max clean das 13 opções'))

if __name__=='__main__':main()

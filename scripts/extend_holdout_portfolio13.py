"""Disk-guarded NB b8u extension of the opened 41-family holdout (EXPLORATORY)."""
from __future__ import annotations
import os
for v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[v]='1'
import argparse, json, time, shutil, traceback
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
from threadpoolctl import threadpool_limits
from dpmeta import portfolio_benchmark as pb
from dpmeta.openml_benchmark import _load_manifest, _openml
from dpmeta.external_confirmation import CONFIRMATORY_SEED
from dpmeta.provenance import file_sha256
from scripts.parallel_nb8u import source_row

OUT=Path('results/dissertation-portfolio13/holdout')
CACHE=Path('.cache/openml')
SOURCE=Path('results/openml-holdout-41/conditions.csv')
MANIFEST=Path('data/openml_confirmatory_manifest_41_combined.csv')
KEYS=['dataset_id','epsilon','repetition']


def guard():
    free=shutil.disk_usage(CACHE).free
    if free < 5*1024**3:
        raise RuntimeError('STOP: disk below 5 GiB')
    return free


def generate(raw, refs, pipeline):
    guard(); t=time.monotonic()
    with threadpool_limits(limits=1):
        rows=pb.run_dataset(_openml(CACHE), source_row(pd.Series(raw)), pb.DEFAULT_EPSILONS,
                            3, CONFIRMATORY_SEED, pipelines=(pipeline,),
                            include_research_features=False, numeric_transform='tanh',
                            best_clean_by_condition=refs)
    guard()
    return rows,time.monotonic()-t


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers',type=int,default=8)
    args=parser.parse_args()
    if not 1 <= args.workers <= 8: parser.error('1–8 workers')
    OUT.mkdir(parents=True,exist_ok=True); (OUT/'datasets').mkdir(exist_ok=True)
    guard()
    old=pd.read_csv(SOURCE,float_precision='round_trip'); man=_load_manifest(MANIFEST)
    man=man[man.dataset_id.isin(old.dataset_id)]
    if len(man)!=41: raise ValueError('Expected exactly 41 holdout datasets')
    refs=old.groupby(KEYS).best_clean_portfolio_ba.first()
    tasks=[]
    for _,row in man.iterrows():
        ref={(float(e),int(r)):float(b) for (_,e,r),b in refs.loc[[int(row.dataset_id)]].items()}
        tasks.append((row.to_dict(),ref))
    alignment=OUT/'alignment.json'
    if not alignment.exists():
        checks=[]; frames=[]
        # Three smallest datasets plus a numeric representative if necessary.
        selected=man.sort_values('n_instances').head(3).dataset_id
        for raw,ref in tasks:
            if raw['dataset_id'] not in set(selected): continue
            generated,seconds=generate(raw,ref,pb.PIPELINES[10])
            new=pd.DataFrame(generated)
            pair=new.merge(old[old.pipeline==pb.PIPELINES[10]],on=KEYS+['pipeline'],
                           suffixes=('_rerun','_original'),validate='one_to_one')
            for c in ['private_ba','clean_ba']:
                pair[c+'_error']=(pair[c+'_rerun']-pair[c+'_original']).abs()
            frames.append(pair[KEYS+['private_ba_rerun','private_ba_original','private_ba_error','clean_ba_error']])
            err=float(pair.private_ba_error.max()); clean=float(pair.clean_ba_error.max())
            checks.append(dict(dataset_id=int(raw['dataset_id']),conditions=len(pair),max_abs_error=err,
                               clean_max_abs_error=clean,seconds=seconds))
            print('ALIGN',checks[-1],flush=True)
        pd.concat(frames).to_csv(OUT/'alignment_conditions.csv',index=False)
        passed=len(checks)==3 and all(c['conditions']==24 and c['max_abs_error']<=1e-12 and
                                      c['clean_max_abs_error']<=1e-12 for c in checks)
        alignment.write_text(json.dumps(dict(passed=passed,checks=checks,seed=CONFIRMATORY_SEED,
                                              source_sha256=file_sha256(SOURCE)),indent=2))
    if not json.loads(alignment.read_text())['passed']:
        raise RuntimeError('Alignment failed; extension prohibited')
    pending=[t for t in tasks if not (OUT/'datasets'/f"{t[0]['dataset_id']}.json").exists()]
    failures=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(generate,raw,ref,pb.NB8U_PIPELINE):int(raw['dataset_id']) for raw,ref in pending}
        for future in as_completed(futures):
            guard(); d=futures[future]
            try:
                rows,seconds=future.result()
                path=OUT/'datasets'/f'{d}.json'; tmp=path.with_suffix('.tmp')
                tmp.write_text(json.dumps(dict(rows=rows,seconds=seconds),allow_nan=False)); tmp.replace(path)
                print('DONE',d,len(rows),round(seconds,2),flush=True)
            except Exception as ex:
                failures.append(dict(dataset_id=d,error=str(ex),traceback=traceback.format_exc()))
                print('FAIL',d,str(ex),flush=True)
    (OUT/'failures.json').write_text(json.dumps(failures,indent=2))
    new=pd.DataFrame([r for p in sorted((OUT/'datasets').glob('*.json'))
                      for r in json.loads(p.read_text())['rows']])
    new.to_csv(OUT/'conditions_new_option.csv',index=False)
    completed=set(new.dataset_id)
    combined=pd.concat([old[old.dataset_id.isin(completed)],new[old.columns]],ignore_index=True)
    pb.validate_condition_grid(combined,man[man.dataset_id.isin(completed)],epsilons=pb.DEFAULT_EPSILONS,
                                repeats=3,pipelines=(*pb.PIPELINES,pb.NB8U_PIPELINE))
    # Prior and histogram queries must have the same complete ledger, not just total ε.
    from dpmeta.privacy import PrivacyLedger,Release,Sensitivity
    audits=[]
    import numpy as np
    for r in new.itertuples():
        ledger=PrivacyLedger(r.epsilon,0.,accounting='basic'); q=r.original_feature_count+1
        for i in range(q):
            ledger.record(Release(name='class_prior_histogram' if i==0 else f'feature_{i-1}_histogram',
                mechanism='laplace',epsilon=r.epsilon/q,delta=0.,sensitivity=Sensitivity(2.,np.sqrt(2.),1.),
                noise_scale=2./(r.epsilon/q)))
        ledger.verify()
        if json.loads(r.ledger_json)!=ledger.as_dict(): raise ValueError('Ledger mismatch')
        audits.append(dict(dataset_id=r.dataset_id,epsilon=r.epsilon,repetition=r.repetition,identical=True))
    pd.DataFrame(audits).to_csv(OUT/'privacy_audit.csv',index=False)
    combined.to_csv(OUT/'conditions_13.csv',index=False)
    (OUT/'run_config.json').write_text(json.dumps(dict(status='EXPLORATORY: GBM-13 chosen after holdout opened',
        seed=CONFIRMATORY_SEED,repeats=3,numeric_transform='tanh',workers=args.workers,
        cache=str(CACHE),datasets=len(completed),source_sha256=file_sha256(SOURCE),
        manifest_sha256=file_sha256(MANIFEST),free_disk=guard()),indent=2))
    if len(completed)!=41: raise RuntimeError('Incomplete holdout; see failures.json')

if __name__=='__main__': main()

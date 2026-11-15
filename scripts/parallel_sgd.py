"""Resumable two-option H27 extension; alignment is mandatory before training."""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import time
import traceback

for variable in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
                 'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[variable]='1'
os.environ.setdefault('OPENML_CACHE_DIR',str(Path('.cache/openml').resolve()))
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from dpmeta.openml_benchmark import _load_manifest,_openml,_seed_for
from dpmeta.portfolio_benchmark import (PIPELINES,NB8U_PIPELINE,H27_PIPELINE,SGD_NEW_PIPELINES,
    PUBLIC_LAMBDA_RULE,LAMRULE_PIPELINE,run_dataset,validate_condition_grid)
from dpmeta.provenance import (file_sha256,numerical_implementation_sha256,
    runtime_fingerprint,save_source_snapshot)
from dpmeta.privacy import PrivacyLedger,Release,Sensitivity
from dpmeta.mechanisms import zcdp_rho_for_epsilon
from dpmeta.model import DELTA_TOTAL
from scripts.parallel_nb8u import atomic_json,source_row

OUT=Path('results/portfolio-984-sgd')
BASE=Path('results/portfolio-984-nb8u')
MANIFEST=Path('data/openml_manifest_984_completed.csv')
CACHE=Path('.cache/openml')
KEYS=['dataset_id','epsilon','repetition']
OLD_PIPELINES=(*PIPELINES,NB8U_PIPELINE)
MIN_FREE=5*1024**3

class DiskLimit(RuntimeError): pass

def disk_guard():
    free=shutil.disk_usage(CACHE).free
    if free<MIN_FREE: raise DiskLimit(f'free disk {free} bytes < 5 GiB; stopping')
    return free

def signature():
    return dict(implementation=numerical_implementation_sha256(),executor=file_sha256(Path(__file__)),
        rule=file_sha256(PUBLIC_LAMBDA_RULE),baseline=file_sha256(BASE/'conditions_13.csv'),
        manifest=file_sha256(MANIFEST),runtime=runtime_fingerprint())

def inputs():
    config=json.loads(Path('results/portfolio-984-current-slim/run_config.json').read_text())
    if file_sha256(MANIFEST)!=config['manifest_sha256']: raise ValueError('manifest changed')
    old=pd.read_csv(BASE/'conditions_13.csv',float_precision='round_trip')
    manifest=_load_manifest(MANIFEST)
    manifest=manifest[manifest.dataset_id.isin(old.dataset_id)].copy()
    validate_condition_grid(old,manifest,epsilons=tuple(config['epsilons']),
        repeats=config['repeats'],pipelines=OLD_PIPELINES)
    if len(manifest)!=973: raise ValueError('expected the same 973 datasets')
    return config,manifest,old

def audit_ledger(r):
    """Reconstruct every H27 release independently, without extra model fits."""
    sensitivity=4./r['train_n']
    rho=zcdp_rho_for_epsilon(r['epsilon'],DELTA_TOTAL)/100
    sigma=sensitivity/np.sqrt(2*rho)
    expected=PrivacyLedger(r['epsilon'],DELTA_TOTAL,accounting='zcdp')
    record=Sensitivity(float(sensitivity*np.sqrt((r['encoded_dim']+1)*r['class_count'])),
        float(sensitivity),float(sensitivity))
    for step in range(100):
        expected.record(Release(name=f'dp_sgd_gradient_{step+1}',
            mechanism='gaussian_zcdp_full_batch_gradient',epsilon=0.,delta=0.,
            sensitivity=record,noise_scale=float(sigma),rho=float(rho)))
    expected.verify()
    if json.loads(r['ledger_json'])!=expected.as_dict(): raise AssertionError('H27 ledger mismatch')
    if r['epsilon_used']!=expected.epsilon_used or r['delta_used']!=expected.delta_used:
        raise AssertionError('ledger totals mismatch')
    return True

def check_data(dataset_id):
    audit=pd.read_csv(BASE/'data_content_audit.csv',keep_default_na=False).set_index('dataset_id')
    row=audit.loc[dataset_id]
    path=Path(row.actual_path)
    if not path.is_file() or file_sha256(path)!=row.original_sha256:
        raise ValueError(f'dataset {dataset_id}: parquet differs from certified baseline')
    return dict(dataset_id=dataset_id,path=str(path),sha256=row.original_sha256)

def references(old,dataset_id):
    block=old[old.dataset_id.eq(dataset_id)].groupby(['epsilon','repetition']).best_clean_portfolio_ba.first()
    return {(float(e),int(r)):float(b) for (e,r),b in block.items()}

def alignment():
    config,manifest,old=inputs()
    selected_ids=pd.read_csv(BASE/'alignment_manifest.csv').dataset_id.tolist()
    selected=manifest.set_index('dataset_id').loc[selected_ids].reset_index()
    selected.to_csv(OUT/'alignment_manifest.csv',index=False)
    # The original consolidation reads raw shard CSVs with pandas' default
    # float parser. Preserve that serialization step, separately from numerical
    # equality to the raw fits. Never copy reference scores into generated fits.
    raw_parts=[]
    for path in sorted(Path('results/portfolio-984-current/_portfolio_shards').glob('conditions_*.csv')):
        for chunk in pd.read_csv(path,usecols=list(old.columns),float_precision='round_trip',chunksize=25000):
            raw_parts.append(chunk[chunk.dataset_id.isin(selected_ids)&chunk.pipeline.eq(H27_PIPELINE)])
    raw=pd.concat(raw_parts,ignore_index=True)
    raw.to_csv(OUT/'alignment_raw_shards.csv',index=False)
    om=_openml(CACHE); details=[]; data=[]; started=time.monotonic()
    raw_exact=[]
    with threadpool_limits(limits=1):
        for _,row in selected.iterrows():
            disk_guard(); data.append(check_data(int(row.dataset_id)))
            generated=run_dataset(om,source_row(row),tuple(config['epsilons']),config['repeats'],
                config['seed'],pipelines=(H27_PIPELINE,),include_research_features=False,
                numeric_transform=config['numeric_transform'],
                best_clean_by_condition=references(raw,int(row.dataset_id)))
            for r in generated: audit_ledger(r)
            generated=pd.DataFrame(generated)
            raw_ref=raw[raw.dataset_id.eq(row.dataset_id)]
            raw_pair=generated.merge(raw_ref,on=KEYS+['pipeline'],suffixes=('_rerun','_raw'),validate='one_to_one')
            raw_exact.append(all(raw_pair[c+'_rerun'].eq(raw_pair[c+'_raw']).all()
                for c in old.columns.difference(KEYS+['pipeline'])))
            # Exact reconstruction of the historical shard-to-consolidated step.
            generated=pd.read_csv(io.StringIO(generated[list(old.columns)].to_csv(index=False)))
            ref=old[old.dataset_id.eq(row.dataset_id)&old.pipeline.eq(H27_PIPELINE)]
            pair=generated.merge(ref,on=KEYS+['pipeline'],suffixes=('_rerun','_original'),validate='one_to_one')
            for metric in old.columns.difference(KEYS+['pipeline']):
                pair[metric+'_exact']=pair[metric+'_rerun'].eq(pair[metric+'_original'])
            details.append(pair)
            print('ALIGN',int(row.dataset_id),len(pair),flush=True)
    frame=pd.concat(details,ignore_index=True)
    exact=frame.filter(regex='_exact$').all().all() and all(raw_exact)
    frame.to_csv(OUT/'alignment_conditions.csv',index=False)
    pd.DataFrame(data).to_csv(OUT/'alignment_data_audit.csv',index=False)
    atomic_json(OUT/'alignment.json',dict(passed=bool(exact),datasets=len(selected),conditions=len(frame),
        exact_by_metric=frame.filter(regex='_exact$').all().to_dict(),
        raw_shard_all_metrics_exact=all(raw_exact),
        serialization='raw CSV -> pandas default read_csv -> consolidated CSV',
        max_private_error=float(abs(frame.private_ba_rerun-frame.private_ba_original).max()),
        seconds=time.monotonic()-started,signature=signature(),ledger_audits=len(frame)))
    if not exact: raise RuntimeError('exact H27 alignment failed; full run prohibited')

def worker(queue,stop,config,run_signature):
    om=_openml(CACHE)
    with threadpool_limits(limits=1):
        while not stop.is_set():
            task=queue.get()
            if task is None: return
            raw,reference=task; row=pd.Series(raw); dataset_id=int(row.dataset_id)
            started=time.monotonic()
            try:
                free=disk_guard(); data=check_data(dataset_id)
                generated=run_dataset(om,source_row(row),tuple(config['epsilons']),config['repeats'],config['seed'],
                    pipelines=SGD_NEW_PIPELINES,include_research_features=False,
                    numeric_transform=config['numeric_transform'],best_clean_by_condition=reference)
                # Deduplicate ledger payloads within a commit, retaining one audit per fit.
                ledgers={}
                for r in generated:
                    audit_ledger(r)
                    r['ledger_identical_to_h27']=True
                    r['split_seed']=_seed_for(config['seed'],dataset_id,r['repetition'],'portfolio_split')
                    r['dp_seed']=_seed_for(config['seed'],dataset_id,r['repetition'],r['epsilon'],'dp_sgd_clip2_reg003')
                    identifier=hashlib.sha256(r['ledger_json'].encode()).hexdigest()
                    ledgers[identifier]=r.pop('ledger_json'); r['ledger_id']=identifier
                validate_condition_grid(pd.DataFrame(generated),pd.DataFrame([raw]),
                    epsilons=tuple(config['epsilons']),repeats=config['repeats'],pipelines=SGD_NEW_PIPELINES)
                after=disk_guard()
                atomic_json(OUT/'datasets'/f'{dataset_id}.json',dict(dataset_id=dataset_id,
                    signature=run_signature,seconds=time.monotonic()-started,free_before=free,free_after=after,
                    data=data,rows=generated,ledgers=ledgers))
                print(f'DONE {dataset_id} {time.monotonic()-started:.2f}s free={after/1024**3:.2f}GiB',flush=True)
            except Exception as exc:
                atomic_json(OUT/'failure_details'/f'{dataset_id}.json',dict(dataset_id=dataset_id,
                    dataset_name=str(row.dataset_name),error_type=type(exc).__name__,error=str(exc),
                    traceback=traceback.format_exc(),seconds=time.monotonic()-started))
                print(f'FAIL {dataset_id}: {exc!r}',flush=True)
                if isinstance(exc,DiskLimit): stop.set(); return

def load_commits(config,manifest,run_signature):
    payloads=[]
    for path in sorted((OUT/'datasets').glob('*.json')):
        p=json.loads(path.read_text())
        if p['signature']!=run_signature: raise ValueError(f'resume signature changed: {path}')
        if p['dataset_id'] not in set(manifest.dataset_id): raise ValueError('foreign dataset')
        validate_condition_grid(pd.DataFrame(p['rows']),manifest[manifest.dataset_id.eq(p['dataset_id'])],
            epsilons=tuple(config['epsilons']),repeats=config['repeats'],pipelines=SGD_NEW_PIPELINES)
        for r in p['rows']:
            ledger=p['ledgers'][r['ledger_id']]
            if hashlib.sha256(ledger.encode()).hexdigest()!=r['ledger_id']: raise ValueError('ledger hash changed')
            audit_ledger(dict(r,ledger_json=ledger))
        payloads.append(p)
    return payloads

def run(workers,retry_failures=False):
    config,manifest,old=inputs(); sig=signature()
    alignment=json.loads((OUT/'alignment.json').read_text())
    if not alignment['passed'] or alignment['signature']!=sig: raise RuntimeError('passing current alignment required')
    disk_guard()
    for name in ('datasets','failure_details'): (OUT/name).mkdir(exist_ok=True)
    expected=dict(signature=sig,original_run_config=config,pipelines=SGD_NEW_PIPELINES,workers=workers,
        public_n='manifest n_instances',seed_namespace='dp_sgd_clip2_reg003',min_free_bytes=MIN_FREE,
        clean_reference='original best_clean_portfolio_ba; retained without expanding oracle clean',
        metadata_release=False)
    # JSON normalizes tuples for stable comparison.
    expected=json.loads(json.dumps(expected)); path=OUT/'run_config.json'
    if path.exists() and json.loads(path.read_text())!=expected: raise ValueError('resume configuration changed')
    atomic_json(path,expected)
    save_source_snapshot(OUT,extra_files=('scripts/parallel_sgd.py','tests/test_sgd_options.py',
        str(PUBLIC_LAMBDA_RULE.relative_to(Path.cwd()))))
    payloads=load_commits(config,manifest,sig); completed={p['dataset_id'] for p in payloads}
    failed={int(p.stem) for p in (OUT/'failure_details').glob('*.json')}
    pending=manifest[~manifest.dataset_id.isin(completed|(set() if retry_failures else failed))].copy()
    pending=pending.assign(work=pending.n_instances*pending.n_encoded_features*pending.n_classes).sort_values('work',ascending=False)
    print(f'RUN completed={len(completed)} failed={len(failed-completed)} pending={len(pending)} workers={workers}',flush=True)
    ctx=mp.get_context('spawn'); queue=ctx.Queue(); stop=ctx.Event()
    for _,row in pending.iterrows(): queue.put((row.drop('work').to_dict(),references(old,int(row.dataset_id))))
    for _ in range(workers): queue.put(None)
    processes=[ctx.Process(target=worker,args=(queue,stop,config,sig)) for _ in range(workers)]
    attempt=dict(started_unix=time.time(),pending=len(pending),completed_before=len(completed),workers=workers,free_before=disk_guard())
    attempts_path=OUT/'attempts.json'; attempts=json.loads(attempts_path.read_text()) if attempts_path.exists() else []
    attempts.append(attempt); atomic_json(attempts_path,attempts)
    try:
        for proc in processes: proc.start()
        while any(proc.is_alive() for proc in processes):
            try: disk_guard()
            except DiskLimit:
                stop.set()
                for proc in processes:
                    if proc.is_alive(): proc.terminate()
                break
            for proc in processes: proc.join(timeout=.1)
    except KeyboardInterrupt:
        for proc in processes:
            if proc.is_alive(): proc.terminate()
        raise
    finally:
        for proc in processes:
            if proc.pid is not None: proc.join()
        queue.cancel_join_thread(); queue.close()
        attempt.update(ended_unix=time.time(),seconds=time.time()-attempt['started_unix'],
            disk_stopped=stop.is_set(),free_after=shutil.disk_usage(CACHE).free,
            worker_exit_codes=[p.exitcode for p in processes])
        atomic_json(attempts_path,attempts)
    aggregate()
    if stop.is_set(): raise DiskLimit('disk threshold reached; committed datasets retained')
    if any(p.exitcode!=0 for p in processes): raise RuntimeError('worker exited unexpectedly; resume supported')

def aggregate():
    config,manifest,old=inputs()
    payloads=load_commits(config,manifest,signature())
    if not payloads: raise RuntimeError('no completed datasets')
    full=pd.DataFrame([r for p in payloads for r in p['rows']])
    new=full[list(old.columns)].sort_values(KEYS+['pipeline'])
    new.to_csv(OUT/'conditions_new_options.csv',index=False)
    for count,extra in ((14,(LAMRULE_PIPELINE,)),(15,SGD_NEW_PIPELINES)):
        combined=pd.concat([old,new[new.pipeline.isin(extra)]],ignore_index=True)
        counts=combined.groupby(KEYS).pipeline.agg(['size','nunique'])
        good=counts[(counts['size']==count)&(counts['nunique']==count)].index
        combined=combined.set_index(KEYS).loc[good].reset_index()[list(old.columns)].sort_values(KEYS+['pipeline'])
        validate_condition_grid(combined,manifest,epsilons=tuple(config['epsilons']),repeats=config['repeats'],pipelines=(*OLD_PIPELINES,*extra))
        # Original rows must round-trip exactly in every output population.
        retained=old[old.dataset_id.isin(combined.dataset_id)].sort_values(KEYS+['pipeline']).reset_index(drop=True)
        actual=combined[combined.pipeline.isin(OLD_PIPELINES)].sort_values(KEYS+['pipeline']).reset_index(drop=True)
        pd.testing.assert_frame_equal(actual,retained,check_exact=True)
        combined.to_csv(OUT/f'conditions_{count}.csv',index=False)
    ledgers={k:v for p in payloads for k,v in p['ledgers'].items()}
    pd.DataFrame([dict(ledger_id=k,ledger_json=v) for k,v in sorted(ledgers.items())]).to_csv(OUT/'ledgers.csv',index=False)
    audit_columns=KEYS+['pipeline','ledger_id','split_seed','dp_seed','train_n','test_n','encoded_dim',
        'class_count','epsilon_used','delta_used','regularization','public_n_instances','ledger_identical_to_h27']
    full[audit_columns].to_csv(OUT/'privacy_audit.csv',index=False)
    pd.DataFrame([p['data'] for p in payloads]).to_csv(OUT/'data_content_audit.csv',index=False)
    pd.DataFrame([{k:v for k,v in p.items() if k not in ('rows','ledgers','signature','data')} for p in payloads]).to_csv(OUT/'timing.csv',index=False)
    failures=[json.loads(p.read_text()) for p in (OUT/'failure_details').glob('*.json') if int(p.stem) not in set(full.dataset_id)]
    pd.DataFrame(failures,columns=['dataset_id','dataset_name','error_type','error','traceback','seconds']).to_csv(OUT/'failures.csv',index=False)
    atomic_json(OUT/'coverage.json',dict(planned=len(manifest),completed=full.dataset_id.nunique(),failed=len(failures),
        unattempted_ids=sorted(set(manifest.dataset_id)-set(full.dataset_id)-{f['dataset_id'] for f in failures}),
        new_rows=len(new),ledger_audits=len(full)))
    print(f'AGGREGATED {len(payloads)} datasets {len(new)} new rows',flush=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('alignment','run','aggregate'))
    parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--retry-failures',action='store_true')
    args=parser.parse_args(); OUT.mkdir(parents=True,exist_ok=True)
    if args.stage=='alignment': alignment()
    elif args.stage=='aggregate': aggregate()
    else: run(args.workers,args.retry_failures)
if __name__=='__main__': main()

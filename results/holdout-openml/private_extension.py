"""Exact private-outcome continuation of the exploratory 41-family extension.

Uses the existing DP-SGD kernel unchanged. Omits auxiliary non-private clean
softmax fits (not an endpoint in the final protocol); clean metrics remain missing, never copied
or fabricated. Requires bit-exact alignment against completed canonical runs.
"""
import os
for variable in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
                 'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[variable]='1'
from pathlib import Path
import json
import hashlib
import multiprocessing as mp
import time
import traceback
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from scripts.confirmatory import (OUT,EPS,KEYS,P15,CACHE,now,log,verify,freeze,
    source_row,_load_manifest,_openml,_get_openml_data,_seed_for,
    stratified_holdout_indices,pb,ec,conservative_family,audit_ledger,
    atomic_json,disk_guard,DiskLimit,recommendations,analyze_population)
from dpmeta.provenance import file_sha256,numerical_implementation_sha256,runtime_fingerprint

FOLDER=OUT/'exploratory_41'
FULL=FOLDER/'extension'
PRIVATE=FOLDER/'private_extension'
SOURCE=Path('results/dissertation-portfolio13/holdout/conditions_13.csv')
SOURCE_SHA='571631972451d0117372dd02d20ff4e1a247a78c9c636be865f9b1a6f24ff037'
PIPES=pb.SGD_NEW_PIPELINES

def generate(raw,epsilons=EPS):
    """Same arrays, seeds, parameters, cache and score calls as _run_clip2_options."""
    disk_guard();row=source_row(pd.Series(raw))
    x,y,schema=_get_openml_data(_openml(CACHE),row,numeric_transform='tanh')
    rows=[];did=int(row.dataset_id)
    with threadpool_limits(limits=1):
        for repetition in range(3):
            split_seed=_seed_for(ec.CONFIRMATORY_SEED,did,repetition,'portfolio_split')
            tr,te=stratified_holdout_indices(y,split_seed)
            xx,yy,xt,yt=x[tr],y[tr],x[te],y[te]
            for epsilon in epsilons:
                cache={}
                for pipeline in PIPES:
                    lam=pb.sgd_regularization(pipeline,int(row.n_instances),epsilon)
                    dp_seed=_seed_for(ec.CONFIRMATORY_SEED,did,repetition,epsilon,'dp_sgd_clip2_reg003')
                    if lam not in cache:
                        model,ledger=pb.fit_private_sgd_softmax(xx,yy,len(schema.classes),
                            epsilon=epsilon,delta=pb.DELTA_TOTAL,seed=dp_seed,
                            n_steps=pb.DP_SGD_STEPS_ALTERNATE,
                            gradient_clip=pb.DP_SGD_CLIP2_GRADIENT_CLIP,regularization=lam)
                        cache[lam]=(pb.balanced_accuracy(yt,model.predict(xt)),ledger)
                    score,ledger=cache[lam]
                    fields=pb._ledger_fields(ledger)
                    r=dict(dataset_id=did,family_id=str(row.family_id),epsilon=float(epsilon),
                        pipeline=pipeline,repetition=repetition,private_ba=score,
                        train_n=len(tr),test_n=len(te),encoded_dim=schema.encoded_dim,
                        original_feature_count=len(schema.feature_names),class_count=len(schema.classes),
                        numeric_transform='tanh',regularization=lam,public_n_instances=int(row.n_instances),
                        metadata_feature_mode='public_schema_only',dp_meta_epsilon=0.,
                        dp_meta_epsilon_used=0.,dp_meta_delta_used=0.,**fields,
                        total_epsilon_basic_bound=fields['epsilon_used'],
                        total_delta_bound=fields['delta_used'],split_seed=split_seed,dp_seed=dp_seed,
                        clean_metrics_evaluated=False,ledger_audit_passed=True)
                    audit_ledger(r);rows.append(r)
    disk_guard();return rows

def commits(path):
    return [json.loads(p.read_text()) for p in sorted((path/'datasets').glob('*.json'))]

def validate(frame,manifest,pipes):
    assert set(frame.dataset_id)==set(manifest.dataset_id)
    assert not frame.duplicated(KEYS).any()
    assert set(frame.pipeline)==set(pipes)
    assert set(frame.epsilon)==set(EPS) and set(frame.repetition)=={0,1,2}
    assert frame.groupby('dataset_id').size().eq(len(pipes)*8*3).all()
    assert frame.groupby(['dataset_id','epsilon','repetition']).pipeline.nunique().eq(len(pipes)).all()
    assert np.isfinite(frame.private_ba).all() and frame.private_ba.between(0,1).all()

def alignment(manifest,full):
    path=PRIVATE/'alignment.json'
    if path.exists():
        record=json.loads(path.read_text());assert record['passed']
        assert record['executor_sha256']==file_sha256(Path(__file__))
        log('Etapa 5: alinhamento privado exato anterior verificado.')
        return
    by_id={c['dataset_id']:c for c in full}
    selected=manifest[manifest.dataset_id.isin(by_id)].sort_values('n_instances').head(3)
    aligned=[];ledger_rows=[]
    for _,row in selected.iterrows():
        reference=by_id[int(row.dataset_id)]
        old={(r['epsilon'],r['repetition'],r['pipeline']):r for r in reference['rows']}
        new=generate(row.to_dict(),epsilons=(.1,1.,16.))
        for r in new:
            ref=old[(r['epsilon'],r['repetition'],r['pipeline'])]
            for col in ('private_ba','train_n','test_n','encoded_dim','class_count','epsilon_used',
                        'delta_used','regularization','public_n_instances','split_seed'):
                assert r[col]==ref[col],(int(row.dataset_id),col,r[col],ref[col])
            assert r['ledger_json']==reference['ledgers'][ref['ledger_id']]
            ledger_rows.append(dict(dataset_id=r['dataset_id'],epsilon=r['epsilon'],
                repetition=r['repetition'],pipeline=r['pipeline'],ledger_json=r['ledger_json'],passed=True))
            aligned.append({k:r[k] for k in KEYS+['private_ba']})
        log(f'Etapa 5: alinhamento privado exato ID {int(row.dataset_id)}, {len(new)} condições.')
    pd.DataFrame(aligned).to_csv(PRIVATE/'alignment_conditions.csv',index=False)
    pd.DataFrame(ledger_rows).to_csv(PRIVATE/'alignment_privacy_audit.csv',index=False)
    atomic_json(path,dict(passed=True,completed_at=now(),datasets=len(selected),rows=len(aligned),
        exact_fields=['private_ba','splits','accounting','regularization','complete ledger'],
        executor_sha256=file_sha256(Path(__file__))))
    assert len(selected)==3 and len(aligned)==54

def worker(queue,stop,signature):
    while not stop.is_set():
        raw=queue.get()
        if raw is None:return
        did=int(raw['dataset_id']);started=time.monotonic()
        try:
            rows=generate(raw);ledgers={}
            for r in rows:
                payload=r.pop('ledger_json');key=hashlib.sha256(payload.encode()).hexdigest()
                ledgers[key]=payload;r['ledger_id']=key
            validate(pd.DataFrame(rows),pd.DataFrame([raw]),PIPES)
            atomic_json(PRIVATE/'datasets'/f'{did}.json',dict(dataset_id=did,signature=signature,
                completed_at=now(),seconds=time.monotonic()-started,rows=rows,ledgers=ledgers,
                clean_metrics_evaluated=False))
            print(now(),'DONE_PRIVATE',did,len(rows),round(time.monotonic()-started,2),flush=True)
        except DiskLimit:
            stop.set();return
        except Exception as error:
            atomic_json(PRIVATE/'failure_details'/f'{did}.json',dict(dataset_id=did,at=now(),
                error=f'{type(error).__name__}: {error}',traceback=traceback.format_exc()))
            print(now(),'FAIL',did,str(error),flush=True)

def main():
    verify();freeze();assert file_sha256(SOURCE)==SOURCE_SHA
    assert (OUT/'confirmatory/analysis.json').exists()
    PRIVATE.mkdir(exist_ok=True)
    for name in ('datasets','failure_details'):(PRIVATE/name).mkdir(exist_ok=True)
    man=_load_manifest(Path('data/openml_confirmatory_manifest_41_combined.csv'))
    man['semantic_family_id']=man.dataset_name.map(conservative_family)
    old=pd.read_csv(SOURCE,float_precision='round_trip')
    full=commits(FULL)
    original_signature=json.loads((FULL/'run_config.json').read_text())
    for commit in full:
        assert commit['signature']==original_signature
        for r in commit['rows']:
            payload=commit['ledgers'][r['ledger_id']]
            assert hashlib.sha256(payload.encode()).hexdigest()==r['ledger_id']
            audit_ledger(dict(r,ledger_json=payload))
    signature=dict(executor_sha256=file_sha256(Path(__file__)),
        numerical=numerical_implementation_sha256(),runtime=runtime_fingerprint(),
        source_sha256=SOURCE_SHA,manifest_sha256=file_sha256(Path('data/openml_confirmatory_manifest_41_combined.csv')),
        frozen_sha256=file_sha256(OUT/'frozen.json'),public_rule_sha256=file_sha256(pb.PUBLIC_LAMBDA_RULE),
        pipes=list(PIPES),seed=ec.CONFIRMATORY_SEED,repeats=3,epsilons=list(EPS),workers=8,
        omp_num_threads=1,numeric_transform='tanh',clean_metrics_evaluated=False)
    config=PRIVATE/'run_config.json'
    if config.exists():assert json.loads(config.read_text())==signature
    else:atomic_json(config,signature)
    log('Etapa 5: continuação aditiva privada; kernel DP-SGD, dados, splits, seeds e ledgers originais. Ajustes limpos auxiliares omitidos e explicitamente ausentes.')
    alignment(man,full)
    saved=commits(PRIVATE)
    for commit in saved:
        assert commit['signature']==signature
        for r in commit['rows']:audit_ledger(dict(r,ledger_json=commit['ledgers'][r['ledger_id']]))
    done={c['dataset_id'] for c in full+saved}
    pending=man[~man.dataset_id.isin(done)].copy()
    pending['work']=pending.n_instances*pending.n_encoded_features*pending.n_classes
    ctx=mp.get_context('spawn');queue=ctx.Queue();stop=ctx.Event()
    for _,row in pending.sort_values('work',ascending=False).iterrows():queue.put(row.drop('work').to_dict())
    for _ in range(8):queue.put(None)
    procs=[ctx.Process(target=worker,args=(queue,stop,signature)) for _ in range(8)]
    log(f'Etapa 5: {len(done)} commits preservados, {len(pending)} pendentes; 8 workers privados após alinhamento exato.')
    try:
        for p in procs:p.start()
        while any(p.is_alive() for p in procs):
            disk_guard()
            if stop.is_set():raise DiskLimit('worker disk stop')
            for p in procs:p.join(timeout=.1)
    finally:
        for p in procs:
            if p.is_alive():p.terminate()
            p.join()
        queue.cancel_join_thread();queue.close()
    assert all(p.exitcode==0 for p in procs)
    all_commits=full+commits(PRIVATE)
    data=pd.DataFrame([r for c in all_commits for r in c['rows']])
    validate(data,man,PIPES)
    data.to_csv(FULL/'conditions.csv',index=False)
    ledgers={k:v for c in all_commits for k,v in c['ledgers'].items()}
    pd.DataFrame([dict(ledger_id=k,ledger_json=v) for k,v in ledgers.items()]).to_csv(FULL/'ledgers.csv',index=False)
    data[KEYS+['ledger_id','ledger_audit_passed','epsilon_used','delta_used','split_seed',
        'train_n','test_n','encoded_dim','class_count']].to_csv(FULL/'privacy_audit.csv',index=False)
    atomic_json(FULL/'coverage.json',dict(complete=True,completed_at=now(),datasets=41,
        rows=len(data),ledger_audits=len(data),canonical_commits=len(full),
        private_only_commits=len(all_commits)-len(full),clean_metrics_missing_are_intentional=True,
        auxiliary_alignment_audits=54))
    combined=pd.concat([old,data],ignore_index=True)
    validate(combined,man,P15)
    combined.to_csv(FOLDER/'conditions_15.csv',index=False)
    rec=pd.read_csv(FOLDER/'recommendations.csv')
    analyze_population(man,combined,rec,FOLDER,True)
    atomic_json(FOLDER/'sources.json',dict(source=str(SOURCE),sha256=SOURCE_SHA,
        manifest_sha256=signature['manifest_sha256'],seed=ec.CONFIRMATORY_SEED,
        exploratory=True,private_only_continuation=True,alignment_audits=54,
        auxiliary_clean_metrics_omitted_for_pending_datasets=True))
    log('Etapa 5 CONCLUÍDA — EXPLORATÓRIA: 41 famílias, grade privada completa; 1.968 saídas auditadas + 54 verificações de alinhamento.')

if __name__=='__main__':main()

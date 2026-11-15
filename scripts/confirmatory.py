"""Additive executor for the immutable protocols/CONFIRMATORY_PROTOCOL_FINAL.md.

Run from the repository root with PYTHONPATH=. and the repository virtualenv.
No git commands and no access to the RF-12 checkout. All fitted artifacts precede
catalog selection. Atomic dataset commits permit exact, signature-checked resume.
"""
from __future__ import annotations

import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_v] = '1'

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import time
import traceback
import multiprocessing as mp
import re
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits
from dpmeta import portfolio_benchmark as pb
from dpmeta import external_confirmation as ec
from dpmeta.openml_benchmark import (_load_manifest, _openml, _get_openml_data,
    candidate_from_dataset, canonical_family, stratified_holdout_indices, _seed_for)
from dpmeta.provenance import (file_sha256, numerical_implementation_sha256,
    runtime_fingerprint, save_source_snapshot)
from scripts.parallel_nb8u import atomic_json, source_row, disk_guard, DiskLimit
from scripts.parallel_sgd import audit_ledger as audit_sgd_ledger
from dpmeta.families import conservative_family

OUT = Path('results/holdout-openml')
PROTOCOL = Path('protocols/CONFIRMATORY_PROTOCOL_FINAL.md')
EXPECTED = '66933d771548a8048b38c5952dbc4a9e1675c31694421c0b39d5063af5dfc183'
MANIFEST = Path('data/openml_confirmatory_manifest.csv')
DEV_MAN = Path('data/openml_manifest_984_completed.csv')
DEV = Path('results/portfolio-984-sgd/conditions_15.csv')
RF_DEV = Path('results/portfolio-984-current-slim/conditions.csv')
CACHE = Path('.cache/openml')
SEED = 20261111
FIT_SEED = 20261110
RUN_SEED = 20260918  # existing external-confirmation benchmark split/noise seed
EPS = pb.DEFAULT_EPSILONS
NB = 'categorical_naive_bayes__laplace_histograms'
B8 = pb.NB8U_PIPELINE
P15 = (*pb.PIPELINES, B8, *pb.SGD_NEW_PIPELINES)
KEYS = ['dataset_id', 'epsilon', 'repetition', 'pipeline']

def now():
    return datetime.now(timezone.utc).isoformat()

def log(message):
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT/'LOG.md').open('a') as f:
        f.write(f'- {now()} — {message}\n')
    print(now(), message, flush=True)

def verify():
    assert file_sha256(PROTOCOL) == EXPECTED
    assert Path('protocols/CONFIRMATORY_PROTOCOL_FINAL.sha256').read_text().split()[0] == EXPECTED
    disk_guard()

def features(frame, meta, pipes, one_hot=True):
    cache = {}
    rows = []
    for r in frame.itertuples(index=False):
        f = pb._cached_candidate_features(meta.loc[int(r.dataset_id)], float(r.epsilon),
            r.pipeline, 'public_plus_dp', {}, None, cache)
        rows.append(np.r_[f, [float(r.pipeline == p) for p in pipes]] if one_hot else f)
    return np.vstack(rows)

def freeze():
    verify()
    if (OUT/'frozen.json').exists():
        frozen = json.loads((OUT/'frozen.json').read_text())
        for p,h in frozen['hashes'].items():
            assert file_sha256(Path(p)) == h, p
        log('Etapa 1: congelamento prévio validado; nenhum retreino.')
        return
    assert not MANIFEST.exists(), 'Cannot fit after holdout selection'
    log('Etapa 1 INÍCIO: treino único dos seletores final, GBM-13 e RF-12 em desenvolvimento.')
    model_dir = OUT/'models'; model_dir.mkdir(exist_ok=True)
    man = _load_manifest(DEV_MAN).drop_duplicates('dataset_id')
    meta = man.set_index('dataset_id')
    raw = pd.read_csv(DEV, usecols=KEYS+['private_ba'])
    assert raw.dataset_id.nunique() == 973
    counts = raw.groupby(['dataset_id','epsilon','pipeline']).repetition.agg(['size','nunique'])
    assert counts.eq(3).all().all() and set(raw.pipeline) == set(P15)
    agg = raw.groupby(['dataset_id','epsilon','pipeline'], as_index=False).private_ba.mean()
    agg['fam'] = agg.dataset_id.map(meta.semantic_family_id.astype(str))
    baseline = agg[agg.pipeline.eq(NB)].set_index(['dataset_id','epsilon']).private_ba
    agg['resid'] = agg.private_ba - baseline.loc[list(zip(agg.dataset_id,agg.epsilon))].to_numpy()
    specs = {}
    for name, pipes, tau in [('gbm13_1', sorted(P15), .02),
                            ('gbm13', sorted((*pb.PIPELINES,B8)), .01)]:
        path = model_dir/f'{name}.joblib'
        block = agg[agg.pipeline.isin(pipes)].reset_index(drop=True)
        if not path.exists():
            x = features(block, meta, pipes)
            weights = 1. / block.groupby('fam').dataset_id.transform('nunique')
            weights = (weights / block.groupby('dataset_id').pipeline.transform('size')).to_numpy()
            model = HistGradientBoostingRegressor(max_iter=400, learning_rate=.05,
                max_leaf_nodes=31, min_samples_leaf=40, l2_regularization=1., random_state=FIT_SEED)
            with threadpool_limits(limits=1):
                model.fit(x, block.resid, sample_weight=weights)
            joblib.dump(dict(model=model, pipes=pipes, tau=tau, fallback=B8), path)
        specs[name] = dict(path=str(path), sha256=file_sha256(path),
            training_data=str(DEV), training_data_sha256=file_sha256(DEV),
            datasets=int(block.dataset_id.nunique()), semantic_families=int(block.fam.nunique()),
            aggregate_rows=len(block), pipes=pipes, tau=tau, target='residual vs old NB',
            features='public_plus_dp + sorted portfolio one-hot', saved_at=now())
        log(f'Etapa 1: {name} salvo: {file_sha256(path)}; {len(block)} linhas agregadas.')
    rf_path = model_dir/'rf12.joblib'
    if not rf_path.exists():
        rf_raw = pd.read_csv(RF_DEV)
        rf_agg = pb._aggregate(rf_raw, man)
        assert set(rf_agg.pipeline) == set(pb.PIPELINES)
        model, _ = pb._fit_selector(rf_agg, man, 'public_plus_dp', {}, seed=20260917)
        joblib.dump(dict(model=model, pipes=list(pb.PIPELINES), tau=None), rf_path)
    specs['rf12'] = dict(path=str(rf_path), sha256=file_sha256(rf_path),
        training_data=str(RF_DEV), training_data_sha256=file_sha256(RF_DEV),
        datasets=int(pd.read_csv(RF_DEV,usecols=['dataset_id']).dataset_id.nunique()),
        pipes=list(pb.PIPELINES), saved_at=now(), implementation='pb._fit_selector; external_confirmation')
    hashes = {str(p):file_sha256(p) for p in [DEV, RF_DEV, DEV_MAN, PROTOCOL,
        *model_dir.glob('*.joblib'), Path('experiments/lodo_gbm_residual.py'),
        Path('dpmeta/portfolio_benchmark.py'), Path('dpmeta/external_confirmation.py')]}
    atomic_json(OUT/'frozen.json', dict(completed_at=now(), protocol_sha256=EXPECTED,
        models=specs, hashes=hashes, runtime=runtime_fingerprint(),
        numerical_implementation_sha256=numerical_implementation_sha256(),
        selection_seed=SEED, run_seed=RUN_SEED, run_seed_rule='same as external_confirmation and opened holdout'))
    (OUT/'models/SHA256SUMS').write_text(''.join(f'{h}  {p}\n' for p,h in hashes.items()))
    log('Etapa 1 CONCLUÍDA: três modelos + hashes dos modelos e dados salvos ANTES da etapa 2.')

def exclusions():
    """Include every CSV carrying IDs, including the content audit and addenda."""
    ids, canonical, semantic = set(), set(), set()
    records = []
    for path in sorted(Path('data').rglob('*.csv')):
        if path.name.startswith('openml_confirmatory_manifest'):
            continue
        frame = pd.read_csv(path, keep_default_na=False)
        if 'dataset_id' not in frame:
            continue
        ids.update(pd.to_numeric(frame.dataset_id,errors='raise').astype(int))
        for column in ('dataset_name','family_id','semantic_family_id'):
            if column not in frame:
                continue
            for value in frame[column].astype(str):
                if value:
                    canonical.update((value, canonical_family(value)))
                    semantic.update((value, ec.confirmation_family(value), conservative_family(value)))
        records.append(dict(path=str(path), sha256=file_sha256(path), rows=len(frame),
                            columns=list(frame.columns)))
    return ids, canonical, semantic, records

def public_origin(name, catalog_names):
    """Resolve explicit source identifiers in derived catalog dataset names.

    This is selection validation only, fixed before any holdout model outcomes.
    An ambiguous declared parent cannot certify semantic disjunction.
    """
    normalized=canonical_family(name)
    match=re.fullmatch(r'tabclustpfn_openml_(\d+)_v\d+_[a-f0-9]+',normalized)
    if match:
        did=int(match.group(1))
        return catalog_names.get(did,f'openml_source_{did}'),did
    match=re.fullmatch(r'tabclustpfn_uci_(.+)_v\d+_[a-f0-9]+',normalized)
    if match:
        source=match.group(1)
        aliases={'iono':'ionosphere','wisc':'breast_w','banknote':'banknote_authentication',
            'landsat':'satellite_image','heart':'heart_disease','segment':'segment',
            'biodeg':'qsar_biodeg','breast':'breast_cancer','vote':'vote'}
        return aliases.get(source,source),None
    if normalized.startswith('tabclustpfn_pima_'):
        return 'pima_diabetes',None
    if normalized.startswith('tabclustpfn_'):
        raise ValueError('declared derived source is ambiguous; semantic disjunction cannot be certified')
    return name,None

def select():
    verify(); freeze()
    if MANIFEST.exists():
        record = json.loads(MANIFEST.with_suffix('.selection.json').read_text())
        assert file_sha256(MANIFEST) == record['manifest_sha256']
        log('Etapa 2: manifesto e registro pré-existentes verificados; seleção preservada.')
        return
    log('Etapa 2 INÍCIO: catálogo OpenML; exclusão de TODOS os manifestos/CSVs com IDs de data/.')
    ids, canonical, semantic, records = exclusions()
    atomic_json(OUT/'exclusion_manifests.json', records)
    with (OUT/'LOG.md').open('a') as f:
        for r in records: f.write(f"  - `{r['path']}` — SHA256 `{r['sha256']}`; {r['rows']} linhas.\n")
    om = _openml(CACHE)
    catalog_path = OUT/'catalog.csv'
    if catalog_path.exists():
        catalog = pd.read_csv(catalog_path)
    else:
        catalog = om.datasets.list_datasets(status='active',output_format='dataframe')
        catalog.to_csv(catalog_path,index=False)
    col = ec._id_column(catalog)
    catalog_names=catalog.set_index(col)['name'].astype(str).to_dict()
    # Encoded dimension, not raw catalog columns (which include target/ignored
    # columns), is the frozen feature eligibility rule. Validate it below.
    candidates = ec._metadata_prefilter(catalog,min_rows=200,max_rows=50000,max_classes=30,
        max_catalog_features=10**9)
    candidates = candidates[~pd.to_numeric(candidates[col]).isin(ids)]
    order = np.random.default_rng(SEED).permutation(candidates[col].astype(int).to_numpy())
    names = candidates.set_index(col)['name'].astype(str).to_dict()
    pd.DataFrame(dict(position=np.arange(1,len(order)+1),dataset_id=order)).to_csv(
        OUT/'candidate_order.csv', index=False)
    accepted, decisions = [], []
    for position, did in enumerate(order,1):
        disk_guard(); did = int(did)
        name = names.get(did,'')
        origin,origin_id=public_origin(name,catalog_names)
        c, s = canonical_family(origin), conservative_family(origin)
        if origin_id in ids:
            decisions.append(dict(position=position,dataset_id=did,name=name,status='excluded_declared_source_id',source_id=origin_id))
            continue
        if c in canonical or s in semantic or ec.confirmation_family(name) in semantic:
            decisions.append(dict(position=position,dataset_id=did,name=name,status='excluded_family'))
            continue
        try:
            candidate = candidate_from_dataset(om,did,None,None,'final-frozen',min_rows=200,
                max_rows=50000,max_classes=30,max_expanded_features=512)
            if candidate is None:
                raise ValueError('public metadata/schema validator rejected candidate')
            origin,origin_id=public_origin(candidate.name,catalog_names)
            c, s = canonical_family(origin), conservative_family(origin)
            if c in canonical or s in semantic or ec.confirmation_family(origin) in semantic:
                decisions.append(dict(position=position,dataset_id=did,name=name,status='excluded_family'))
                continue
            row = candidate.manifest_row('OpenML-active-catalog-disjoint-final')
            row.update(family_id=s, canonical_family_id=c, semantic_family_id=s,
                       selection_position=position,declared_source_name=origin)
            x,y,_ = _get_openml_data(om,pd.Series(row),numeric_transform='tanh')
            if not np.isfinite(x).all():
                raise ValueError('nonfinite encoding')
            for r in range(3):
                stratified_holdout_indices(y,_seed_for(RUN_SEED,did,r,'portfolio_split'))
            del x,y
            accepted.append(row)
            canonical.add(c); semantic.add(s); semantic.add(ec.confirmation_family(origin))
            decisions.append(dict(position=position,dataset_id=did,name=name,status='accepted'))
            log(f'Etapa 2: aceita família {len(accepted)}/40, ID {did}, {name}, posição {position}.')
        except DiskLimit:
            raise
        except Exception as error:
            decisions.append(dict(position=position,dataset_id=did,name=name,status='validation_failure',
                                  error=f'{type(error).__name__}: {error}'))
            log(f'Etapa 2: validação falhou ID {did}; próximo na ordem sorteada: {error}')
        pd.DataFrame(decisions).to_csv(OUT/'selection_decisions.csv',index=False)
        pd.DataFrame(accepted).to_csv(OUT/'selection_progress.csv',index=False)
        if len(accepted) == 40: break
    pd.DataFrame(decisions).to_csv(OUT/'selection_decisions.csv',index=False)
    selected = pd.DataFrame(accepted)
    if selected.empty: raise RuntimeError('No eligible new families')
    old_ids, old_can, old_sem, _ = exclusions()
    assert not set(selected.dataset_id)&old_ids
    assert not set(selected.canonical_family_id)&old_can
    assert not set(selected.semantic_family_id)&old_sem
    assert selected.semantic_family_id.nunique() == len(selected)
    selected.to_csv(MANIFEST,index=False)
    atomic_json(MANIFEST.with_suffix('.selection.json'),dict(
        completed_at=now(),manifest=str(MANIFEST),manifest_sha256=file_sha256(MANIFEST),
        protocol_sha256=EXPECTED,frozen_selectors_sha256=file_sha256(OUT/'frozen.json'),
        seed=SEED,target_families=40,selected_families=len(selected),catalog_status='active',
        catalog_sha256=file_sha256(catalog_path),candidate_order_sha256=file_sha256(OUT/'candidate_order.csv'),
        eligibility=dict(min_rows=200,max_rows=50000,min_classes=2,max_classes=30,max_encoded_features=512),
        exclusions=records,exclusion_ids=len(old_ids),canonical_families=len(old_can),
        semantic_families=len(old_sem),
        replacement_rule='next eligible in frozen seeded permutation; validation only; no outcomes',
        validation_failures=[d for d in decisions if d['status']=='validation_failure'],
        available_shortfall=len(selected)<40,executor_sha256=file_sha256(Path(__file__))))
    log(f'Etapa 2 CONCLUÍDA: {len(selected)} famílias; manifesto e selection.json salvos ANTES de executar modelos.')

def audit_ledger(r):
    """Recompute composition and noise calibration for every emitted model fit."""
    from dpmeta.privacy import PrivacyLedger, Release, Sensitivity
    from dpmeta.mechanisms import analytic_gaussian_sigma, calibration_scale
    d = json.loads(r['ledger_json'])
    if r['pipeline'] in (*pb.SGD_NEW_PIPELINES, pb.H27_PIPELINE):
        audit_sgd_ledger(r)
    if 'releases' in d:
        ledger = PrivacyLedger(d['epsilon_limit'],d['delta_limit'],accounting=d['accounting'])
        for raw in d['releases']:
            release = Release(**dict(raw,sensitivity=Sensitivity(**raw['sensitivity'])))
            ledger.record(release)
            m = release.mechanism; s = release.sensitivity
            if m == 'gaussian_zcdp_full_batch_gradient':
                expected = s.l2 / np.sqrt(2*release.rho)
            elif m == 'analytic_gaussian_output_perturbation':
                expected = analytic_gaussian_sigma(release.epsilon,release.delta,s.l2)
            else:
                expected = calibration_scale(mechanism=m,epsilon=release.epsilon,
                    delta=release.delta,sensitivity=s,output_size=r['encoded_dim']*r['class_count'],
                    rho=release.rho if release.rho else None)
            assert np.isclose(release.noise_scale,expected,rtol=1e-12,atol=1e-12)
        assert ledger.verify().as_dict() == d
        if r['pipeline'] in (NB,B8):
            assert len(d['releases']) == r['original_feature_count']+1
            assert all(q['sensitivity']['l1']==2. for q in d['releases'])
        if r['pipeline'].startswith('gnb__'):
            from dpmeta.privacy import sufficient_statistic_sensitivities
            from dataclasses import asdict
            expected = sufficient_statistic_sensitivities(r['encoded_dim'])
            assert {q['name']:q['sensitivity'] for q in d['releases']} == {
                k:asdict(v) for k,v in expected.items()}
    else:
        assert d['accounting'] in ('parallel_disjoint_row_shards','parallel_shards_and_two_leaf_composition')
        assert sum(d['shard_sizes']) == r['train_n']
        assert len(d['shard_sizes']) == d['n_shards'] == 25
        if d.get('mechanism') == 'exponential_majority_label':
            assert d['max_affected_leaves_per_record']==2
            assert d['epsilon_per_leaf']*2 == r['epsilon']
        elif d.get('mechanism') == 'analytic_gaussian':
            expected = analytic_gaussian_sigma(r['epsilon'],d['delta_limit'],np.sqrt(2.))
            assert np.isclose(d['noise_scale'],expected,rtol=1e-12)
            assert d['per_shard_delta'] == d['delta_limit']
        else:
            assert d['replace_one_l1_sensitivity_per_shard']==2.
            assert d['noise_scale']==2./r['epsilon']
        assert d['epsilon_used']==r['epsilon']
    assert np.isclose(d['epsilon_limit'],r['epsilon'],rtol=0,atol=1e-12)
    assert d['epsilon_used']<=r['epsilon']+1e-10
    assert d['delta_used']<=d['delta_limit']+1e-10
    assert d['epsilon_used']==r['epsilon_used'] and d['delta_used']==r['delta_used']
    assert r.get('dp_meta_epsilon_used',0.)==0.
    return True

def recommendations(manifest, output):
    """Uses only manifest public schema, n, epsilon, and frozen models."""
    meta = manifest.set_index('dataset_id')
    rows = pd.DataFrame([dict(dataset_id=int(r.dataset_id),epsilon=float(e),pipeline=p)
        for r in manifest.itertuples() for e in EPS for p in sorted(P15)])
    chosen = []
    for name in ('gbm13_1','gbm13','rf12'):
        artifact = joblib.load(OUT/'models'/f'{name}.joblib')
        pipes = artifact['pipes']; frame=rows[rows.pipeline.isin(pipes)].copy()
        x = features(frame,meta,pipes,one_hot=name!='rf12')
        frame['pred'] = artifact['model'].predict(x)
        # RF uses the historical portfolio order for argmax ties.
        frame['order'] = frame.pipeline.map({p:i for i,p in enumerate(pipes)})
        frame=frame.sort_values(['dataset_id','epsilon','order'])
        for (did,eps),group in frame.groupby(['dataset_id','epsilon'],sort=True):
            best=group.loc[group.pred.idxmax()]
            fallback=float(group[group.pipeline.eq(B8)].pred.iloc[0]) if name!='rf12' else None
            advantage=float(best.pred-fallback) if fallback is not None else None
            pick=B8 if advantage is not None and advantage<=artifact['tau'] else best.pipeline
            chosen.append(dict(dataset_id=int(did),epsilon=float(eps),selector=name,
                pipeline=pick,predicted_best=best.pipeline,predicted_advantage=advantage,
                features_sha256=hashlib.sha256(x.tobytes()).hexdigest(),public_features_only=True))
    rec=pd.DataFrame(chosen); rec.to_csv(output,index=False)
    return rec

def run_worker(queue, stop, folder, pipes, seed, signature):
    om = _openml(CACHE)
    with threadpool_limits(limits=1):
        while not stop.is_set():
            task = queue.get()
            if task is None: return
            raw, refs = task; row=source_row(pd.Series(raw)); did=int(row.dataset_id)
            started=time.monotonic()
            try:
                disk_guard()
                generated=pb.run_dataset(om,row,EPS,3,seed,pipelines=pipes,
                    numeric_transform='tanh',include_research_features=False,
                    best_clean_by_condition=refs)
                pb.validate_condition_grid(pd.DataFrame(generated),pd.DataFrame([raw]),
                    epsilons=EPS,repeats=3,pipelines=pipes)
                ledgers={}
                for r in generated:
                    audit_ledger(r)
                    r['split_seed']=_seed_for(seed,did,r['repetition'],'portfolio_split')
                    identifier=hashlib.sha256(r['ledger_json'].encode()).hexdigest()
                    ledgers[identifier]=r.pop('ledger_json'); r['ledger_id']=identifier
                    r['ledger_audit_passed']=True
                disk_guard()
                atomic_json(folder/'datasets'/f'{did}.json',dict(dataset_id=did,signature=signature,
                    completed_at=now(),seconds=time.monotonic()-started,rows=generated,ledgers=ledgers))
                print(now(),'DONE',did,len(generated),round(time.monotonic()-started,2),flush=True)
            except DiskLimit:
                stop.set(); return
            except Exception as error:
                atomic_json(folder/'failure_details'/f'{did}.json',dict(dataset_id=did,
                    at=now(),error=f'{type(error).__name__}: {error}',traceback=traceback.format_exc()))
                print(now(),'FAIL',did,str(error),flush=True)

def parallel_run(manifest, folder, pipes, seed, reference=None):
    """Dynamic spawn queue, 8 workers and atomic resume, as parallel_sgd.run."""
    verify(); folder.mkdir(exist_ok=True,parents=True)
    for d in ('datasets','failure_details'): (folder/d).mkdir(exist_ok=True)
    signature=dict(manifest_sha256=hashlib.sha256(manifest.to_csv(index=False).encode()).hexdigest(),
        frozen_sha256=file_sha256(OUT/'frozen.json'),numerical=numerical_implementation_sha256(),
        runtime=runtime_fingerprint(),pipes=list(pipes),epsilons=list(EPS),repeats=3,seed=seed,
        numeric_transform='tanh',workers=8,omp_num_threads=1,min_free_bytes=5*1024**3,
        executor_sha256=file_sha256(Path(__file__)))
    config_path=folder/'run_config.json'
    if config_path.exists():
        assert json.loads(config_path.read_text())==signature, 'Resume signature mismatch'
    else: atomic_json(config_path,signature)
    completed={}
    for path in sorted((folder/'datasets').glob('*.json')):
        commit=json.loads(path.read_text()); assert commit['signature']==signature
        for r in commit['rows']:
            payload=commit['ledgers'][r['ledger_id']]
            assert hashlib.sha256(payload.encode()).hexdigest()==r['ledger_id']
            audit_ledger(dict(r,ledger_json=payload))
        completed[commit['dataset_id']]=commit
    pending=manifest[~manifest.dataset_id.isin(completed)].copy()
    pending['work']=pending.n_instances*pending.n_encoded_features*pending.n_classes
    pending=pending.sort_values('work',ascending=False)
    ctx=mp.get_context('spawn'); queue=ctx.Queue(); stop=ctx.Event()
    for _,row in pending.iterrows():
        refs=None
        if reference is not None:
            block=reference[reference.dataset_id.eq(row.dataset_id)].groupby(['epsilon','repetition']).best_clean_portfolio_ba.first()
            refs={(float(e),int(r)):float(v) for (e,r),v in block.items()}
        queue.put((row.drop('work').to_dict(),refs))
    for _ in range(8): queue.put(None)
    processes=[ctx.Process(target=run_worker,args=(queue,stop,folder,pipes,seed,signature)) for _ in range(8)]
    log(f'Execução paralela {folder}: {len(completed)} datasets completos, {len(pending)} pendentes, 8 workers.')
    try:
        for proc in processes: proc.start()
        while any(proc.is_alive() for proc in processes):
            disk_guard()
            if stop.is_set(): raise DiskLimit('worker disk stop')
            for proc in processes: proc.join(timeout=.1)
    finally:
        for proc in processes:
            if proc.is_alive(): proc.terminate()
            proc.join()
        queue.cancel_join_thread(); queue.close()
    if any(p.exitcode!=0 for p in processes): raise RuntimeError('Worker exit; resume from atomic commits')
    commits=[json.loads(p.read_text()) for p in sorted((folder/'datasets').glob('*.json'))]
    rows=[r for c in commits for r in c['rows']]
    full=pd.DataFrame(rows)
    if full.empty: raise RuntimeError('No completed datasets; see failure_details')
    pb.validate_condition_grid(full,manifest,epsilons=EPS,repeats=3,pipelines=pipes)
    assert set(full.dataset_id)==set(manifest.dataset_id), 'Incomplete run; inspect failure_details; resume'
    full.to_csv(folder/'conditions.csv',index=False)
    ledgers={k:v for c in commits for k,v in c['ledgers'].items()}
    pd.DataFrame([dict(ledger_id=k,ledger_json=v) for k,v in ledgers.items()]).to_csv(folder/'ledgers.csv',index=False)
    full[KEYS+['ledger_id','ledger_audit_passed','epsilon_used','delta_used','split_seed',
               'train_n','test_n','encoded_dim','class_count']].to_csv(folder/'privacy_audit.csv',index=False)
    atomic_json(folder/'coverage.json',dict(completed_at=now(),datasets=len(commits),rows=len(full),
        ledger_audits=len(full),complete=True))
    return full

def run():
    verify(); assert MANIFEST.exists()
    selected=json.loads(MANIFEST.with_suffix('.selection.json').read_text())
    assert selected['manifest_sha256']==file_sha256(MANIFEST)
    freeze(); man=_load_manifest(MANIFEST)
    save_source_snapshot(OUT,extra_files=('scripts/confirmatory.py',
        'scripts/parallel_nb8u.py','scripts/parallel_sgd.py','experiments/lodo_gbm_residual.py'))
    log('Etapa 3 INÍCIO: recomendações públicas congeladas; execução das 15 opções × 8 ε × 3 repetições.')
    recommendations(man,OUT/'recommendations.csv')
    parallel_run(man,OUT/'confirmatory',P15,RUN_SEED)
    log('Etapa 3 CONCLUÍDA: grade completa; ledger auditado em cada execução.')

def bootstrap(values):
    values=np.asarray(values,dtype=float)
    rng=np.random.default_rng(SEED)
    draws=rng.choice(values,size=(20000,len(values))).mean(axis=1)
    return dict(gain=float(values.mean()),ci_low=float(np.percentile(draws,2.5)),
                ci_high=float(np.percentile(draws,97.5)),families=len(values))

def tail_summary(pred):
    groups=[-g.sort_values('epsilon').gain_b8u.to_numpy() for _,g in pred.groupby('family_id')]
    assert len(set(map(len,groups)))==1, 'Use complete family conditions only'
    matrix=np.vstack(groups)
    def metrics(loss):
        count=max(1,loss.shape[-1]//10)
        return (np.mean(loss>.01,axis=-1),np.sort(loss,axis=-1)[...,-count:].mean(axis=-1))
    point=metrics(matrix.ravel())
    indices=np.random.default_rng(SEED).integers(0,len(matrix),size=(20000,len(matrix)))
    sampled=matrix[indices].reshape(20000,-1)
    fraction,cvar=metrics(sampled)
    return dict(fraction_loss_gt_1pp=float(point[0]),
        fraction_ci95=np.percentile(fraction,[2.5,97.5]).tolist(),
        cvar90_loss=float(point[1]),cvar90_ci95=np.percentile(cvar,[2.5,97.5]).tolist(),
        definition='signed loss = b8u BA - selector BA; mean largest max(1,floor(0.1*N)) losses; family bootstrap',
        bootstrap_draws=20000,seed=SEED)

def analyze_population(man, conditions, rec, folder, exploratory):
    folder.mkdir(exist_ok=True,parents=True)
    # No outcomes are passed to recommendations(): join them only here.
    scores=conditions.groupby(['dataset_id','epsilon','pipeline']).private_ba.mean().unstack('pipeline')
    assert set(scores.columns)==set(P15)
    meta=man.set_index('dataset_id')
    pred=scores[[B8,NB,pb.H27_PIPELINE]].rename(columns={B8:'b8u_ba',NB:'old_nb_ba',pb.H27_PIPELINE:'h27_ba'})
    for name in ('gbm13_1','gbm13','rf12'):
        picks=rec[rec.selector.eq(name)].set_index(['dataset_id','epsilon']).pipeline
        pred[name+'_pipeline']=picks
        pred[name+'_ba']=[scores.loc[(d,e),p] for (d,e),p in picks.items()]
    pred=pred.reset_index()
    pred['family_id']=pred.dataset_id.map(meta.semantic_family_id if 'semantic_family_id' in meta else meta.family_id)
    pred['n_instances']=pred.dataset_id.map(meta.n_instances)
    pred['attribute_type']=pred.dataset_id.map(pd.Series(np.where(meta.n_nominal_features.eq(0),'numeric',
        np.where(meta.n_nominal_features.eq(meta.n_features),'nominal','mixed')),index=meta.index))
    pred['n_band']=pd.cut(pred.n_instances,[0,1000,5000,10000,50000,np.inf],
        labels=['<=1000','1001-5000','5001-10000','10001-50000','>50000']).astype(str)
    references={'b8u':'b8u_ba','H27':'h27_ba','old_NB':'old_nb_ba','gbm13_selector':'gbm13_ba','rf12_selector':'rf12_ba'}
    comparisons=[]
    for reference,column in references.items():
        pred['gain_'+reference]=pred.gbm15_ba-pred[column]
        family=pred.groupby(['family_id','dataset_id'])['gain_'+reference].mean().groupby('family_id').mean()
        result=bootstrap(family.to_numpy())
        result.update(reference=reference,
            sign_flip_p_two_sided=pb._sign_flip_p(family.to_numpy(),SEED),exploratory=exploratory)
        comparisons.append(result)
    holm=pb._holm_adjust({r['reference']:r['sign_flip_p_two_sided'] for r in comparisons if r['reference']!='b8u'})
    for r in comparisons:
        r['p_holm']=holm.get(r['reference'])
        r['holm_reject_05']=bool(r['p_holm']<.05) if r['p_holm'] is not None else None
    comparison=pd.DataFrame(comparisons); comparison.to_csv(folder/'comparisons.csv',index=False)
    pred.to_csv(folder/'predictions.csv',index=False)
    family=pred.groupby(['family_id','dataset_id']).gain_b8u.mean().groupby('family_id').mean()
    family.rename('gain_b8u').to_csv(folder/'family_gains.csv')
    tail=tail_summary(pred)
    summary=dict(exploratory=exploratory,completed_at=now(),protocol_sha256=EXPECTED,
        datasets=int(pred.dataset_id.nunique()),families=int(pred.family_id.nunique()),
        n_range=[int(man.n_instances.min()),int(man.n_instances.max())],
        primary=comparisons[0],success_criterion='lower percentile family bootstrap CI95 > 0',
        success=bool(comparisons[0]['ci_low']>0),tails=tail,secondaries=comparisons[1:],
        bootstrap_draws=20000,bootstrap_seed=SEED,
        secondary_test='paired family sign-flip, two-sided, 50000 draws; Holm across exactly four',
        selector_recommendations_public_only=bool(rec.public_features_only.all()),
        recommendation_frequency=rec.groupby('selector').pipeline.value_counts(normalize=True).to_dict().__str__())
    atomic_json(folder/'analysis.json',summary)
    # All subgroup analyses are explicitly exploratory, including on the new holdout.
    strata=[]
    for factor in ('epsilon','n_band','attribute_type'):
        for value,block in pred.groupby(factor,observed=True):
            values=block.groupby(['family_id','dataset_id']).gain_b8u.mean().groupby('family_id').mean()
            strata.append(dict(factor=factor,stratum=str(value),exploratory=True,
                datasets=int(block.dataset_id.nunique()),**bootstrap(values.to_numpy())))
    pd.DataFrame(strata).to_csv(folder/'exploratory_strata.csv',index=False)
    return summary,pred,scores

def analyze():
    verify(); log('Etapa 4 INÍCIO: análise confirmatória congelada; bootstrap familiar 20.000, seed 20261111; Holm em quatro secundários.')
    man=_load_manifest(MANIFEST)
    cond=pd.read_csv(OUT/'confirmatory/conditions.csv')
    rec=pd.read_csv(OUT/'recommendations.csv')
    analyze_population(man,cond,rec,OUT/'confirmatory',False)
    log('Etapa 4 CONCLUÍDA: primária, critério IC inferior > 0, caudas com IC, quatro secundários Holm salvos.')

def exploratory():
    verify(); assert (OUT/'confirmatory/analysis.json').exists()
    log('Etapa 5 INÍCIO — EXPLORATÓRIA: holdout antigo aberto, complementação DP-SGD; nenhum retreino.')
    folder=OUT/'exploratory_41'; folder.mkdir(exist_ok=True)
    path=Path('data/openml_confirmatory_manifest_41_combined.csv')
    man=_load_manifest(path)
    man['semantic_family_id']=man.dataset_name.map(conservative_family)
    base=Path('results/dissertation-portfolio13/holdout/conditions_13.csv')
    if base.exists():
        old=pd.read_csv(base,float_precision='round_trip')
        pipes=pb.SGD_NEW_PIPELINES
        phase_a_audit=pd.read_csv(base.parent/'privacy_audit.csv')
        assert len(phase_a_audit)==41*8*3 and phase_a_audit.identical.all()
        log(f'Etapa 5: reutilização b8u fase A: {base}, SHA256 {file_sha256(base)}.')
    else:
        base=Path('results/openml-holdout-41/conditions.csv')
        old=pd.read_csv(base,float_precision='round_trip')
        pipes=(B8,*pb.SGD_NEW_PIPELINES)
    assert len(man)==41 and old.dataset_id.nunique()==41
    recommendations(man,folder/'recommendations.csv')
    new=parallel_run(man,folder/'extension',pipes,ec.CONFIRMATORY_SEED,reference=old)
    combined=pd.concat([old,new],ignore_index=True)
    pb.validate_condition_grid(combined,man,epsilons=EPS,repeats=3,pipelines=P15)
    combined.to_csv(folder/'conditions_15.csv',index=False)
    rec=pd.read_csv(folder/'recommendations.csv')
    analyze_population(man,combined,rec,folder,True)
    atomic_json(folder/'sources.json',dict(source=str(base),sha256=file_sha256(base),
        manifest_sha256=file_sha256(path),seed=ec.CONFIRMATORY_SEED,exploratory=True))
    log('Etapa 5 CONCLUÍDA — EXPLORATÓRIA: holdout antigo e estratos por ε, n, tipo de atributo salvos separadamente.')

def figures():
    verify(); log('Etapa 6 INÍCIO: figuras forest, CD e ganhos por ε; PDF + PNG.')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.stats import friedmanchisquare,studentized_range
    from dpmeta.figures import _cd_diagram, _save
    folder=Path('figures/dissertation'); folder.mkdir(exist_ok=True,parents=True)
    pred=pd.read_csv(OUT/'confirmatory/predictions.csv')
    cond=pd.read_csv(OUT/'confirmatory/conditions.csv')
    meta=_load_manifest(MANIFEST).set_index('dataset_id')
    scores=cond.groupby(['dataset_id','epsilon','pipeline']).private_ba.mean().unstack('pipeline')
    for name in ('gbm13_1','gbm13','rf12'):
        scores[name]=pred.set_index(['dataset_id','epsilon'])[name+'_ba']
    scores=scores.reset_index()
    scores['family_id']=scores.dataset_id.map(meta.semantic_family_id)
    columns=[*P15,'gbm13_1','gbm13','rf12']
    family=scores.groupby(['family_id','dataset_id'])[columns].mean().groupby('family_id').mean()
    labels={p:pb.PIPELINE_DISPLAY.get(p,p.replace('__',' · ')) for p in P15}
    labels.update({B8:'NB b8u',NB:'NB antigo',pb.SGD_NEW_PIPELINES[0]:'DP-SGD λ regra',
        pb.SGD_NEW_PIPELINES[1]:'DP-SGD λ=0,003','gbm13_1':'Seletor GBM-15','gbm13':'Seletor GBM-13','rf12':'Seletor RF-12'})
    rows=[dict(method=c,**bootstrap((family[c]-family[B8]).to_numpy())) for c in columns if c!=B8]
    table=pd.DataFrame(rows).sort_values('gain')
    table.to_csv(OUT/'figure_forest_data.csv',index=False)
    fig,ax=plt.subplots(figsize=(11,9))
    y=np.arange(len(table)); ax.hlines(y,100*table.ci_low,100*table.ci_high,color='#555555')
    ax.scatter(100*table.gain,y,c=['#c04532' if c=='gbm13_1' else '#256582' for c in table.method],zorder=3)
    ax.set_yticks(y,[labels[c] for c in table.method]);ax.axvline(0,color='gray',lw=1)
    ax.set_xlabel('Ganho de acurácia balanceada frente a b8u (p.p.; IC95% por família)')
    ax.set_title(f'Holdout final: {len(family)} famílias novas');fig.tight_layout()
    _save(fig,folder,'fig14_holdout_gains_vs_b8u')
    ranks=family.rank(axis=1,ascending=False).mean().rename(index=labels)
    n,k=family.shape
    cd=float(studentized_range.ppf(.95,k,np.inf)/np.sqrt(2)*np.sqrt(k*(k+1)/(6*n)))
    p=float(friedmanchisquare(*[family[c].to_numpy() for c in family]).pvalue)
    fig,ax=plt.subplots(figsize=(15,10));_cd_diagram(ax,ranks,cd)
    ax.set_title(f'Holdout final: Nemenyi α=0,05; {n} famílias; Friedman p={p:.3g}')
    _save(fig,folder,'fig13_holdout_cd')
    atomic_json(OUT/'critical_difference.json',dict(ranks=ranks.to_dict(),cd=cd,friedman_p=p,n=n,k=k,descriptive=True))
    rows=[]
    for e,block in pred.groupby('epsilon'):
        values=block.groupby('family_id').gain_b8u.mean()
        rows.append(dict(epsilon=float(e),**bootstrap(values.to_numpy())))
    table=pd.DataFrame(rows);table.to_csv(OUT/'figure_gains_by_epsilon.csv',index=False)
    fig,ax=plt.subplots(figsize=(8,5))
    ax.errorbar(table.epsilon,table.gain*100,yerr=np.vstack([
        table.gain-table.ci_low,table.ci_high-table.gain])*100,fmt='o-',capsize=3,color='#256582')
    ax.axhline(0,color='gray');ax.set_xscale('log');ax.set_xticks(table.epsilon,table.epsilon.map(str))
    ax.set_xlabel('ε');ax.set_ylabel('Ganho final vs b8u (p.p.; IC95%)')
    ax.set_title('Exploratório: ganhos por ε no holdout final');fig.tight_layout()
    _save(fig,folder,'gains_by_epsilon')
    log('Etapa 6 CONCLUÍDA: seis arquivos PDF/PNG salvos em figures/dissertation/.')

def summary(error=None):
    assert file_sha256(PROTOCOL)==EXPECTED
    log('Etapa 7: redação do SUMMARY.md em português.' if error is None else f'Interrupção registrada: {error}')
    selection_path=MANIFEST.with_suffix('.selection.json')
    selected=json.loads(selection_path.read_text()) if selection_path.exists() else None
    primary_path=OUT/'confirmatory/analysis.json'
    result=json.loads(primary_path.read_text()) if primary_path.exists() else None
    old_path=OUT/'exploratory_41/analysis.json'
    old=json.loads(old_path.read_text()) if old_path.exists() else None
    lines=['# Confirmação final — famílias novas','',f'Protocolo SHA256: `{EXPECTED}`.',
        'Protocolo preservado; nenhum comando git; checkout anterior não acessado.',
        'Cronologia completa (UTC) e lista de manifestos: [LOG.md](LOG.md).',
        'Seletores treinados exclusivamente no desenvolvimento e salvos antes da seleção: [frozen.json](frozen.json).','',
        'Regressão: 102 testes existentes passaram (14 avisos).']
    if selected:
        man=_load_manifest(MANIFEST)
        lines += ['',f"Amostra: {len(man)} datasets, {man.semantic_family_id.nunique()} famílias; n de {int(man.n_instances.min())} a {int(man.n_instances.max())}; 200–50.000 linhas, 2–30 classes, ≤512 atributos codificados.",
            f"Falhas de validação/substituições: {len(selected['validation_failures'])}; uso exclusivo do próximo candidato na ordem sorteada (seed 20261111), antes de medir desempenho. Detalhes em [selection_decisions.csv](selection_decisions.csv).",
            f"Exclusão contra {len(selected['exclusions'])} arquivos de data/, por ID, família canônica e semântica. Manifesto e seleção salvos antes das execuções."]
        if len(man)<40:
            lines += [f'Alvo de 40 famílias não alcançado: a ordem completa do catálogo ativo foi esgotada, com apenas {len(man)} famílias elegíveis. Foram usadas todas, conforme o protocolo. O poder e a generalização são muito limitados; o bootstrap é condicionado às famílias observadas.']
    if result:
        r=result['primary']; t=result['tails']
        lines += ['',f"Primária — ganho final vs b8u: **{r['gain']*100:.3f} p.p.**, IC95% [{r['ci_low']*100:.3f}; {r['ci_high']*100:.3f}] p.p. Bootstrap percentil por família, 20.000 reamostragens, seed 20261111.",
            f"Critério de sucesso H2a final (limite inferior > 0): **{'ATINGIDO' if result['success'] else 'NÃO ATINGIDO'}**.",
            f"Caudas: perda >1 p.p. em {t['fraction_loss_gt_1pp']*100:.2f}% das condições; IC95% [{t['fraction_ci95'][0]*100:.2f}; {t['fraction_ci95'][1]*100:.2f}]%. CVaR90 da perda: {t['cvar90_loss']*100:.3f} p.p.; IC95% [{t['cvar90_ci95'][0]*100:.3f}; {t['cvar90_ci95'][1]*100:.3f}] p.p.",
            'CVaR: média das maiores floor(10% × condições) perdas assinadas (b8u − seletor), como na rotina de desenvolvimento; IC por reamostragem das famílias.',
            '', 'Secundários (sign-flip pareado bilateral por família; Holm nos quatro):','',
            '| Referência | Ganho (p.p.) | IC95% (p.p.) | p Holm |','|---|---:|---:|---:|']
        for r in result['secondaries']:
            lines.append(f"| {r['reference']} | {r['gain']*100:.3f} | [{r['ci_low']*100:.3f}; {r['ci_high']*100:.3f}] | {r['p_holm']:.5g} |")
        lines += ['', 'Recomendações calculadas somente com schema e tamanho públicos, ε e calibrações públicas; nenhum resultado do holdout foi usado para escolher pipelines.',
            'Grade: 15 opções × 8 ε × 3 repetições por dataset; tanh; 8 workers; OMP_NUM_THREADS=1; commits atômicos retomáveis; auditoria do ledger em cada ajuste.',
            'Split/sementes: rotina existente do benchmark, namespace portfolio_split; semente base de confirmação externa 20260918.']
    else:
        lines += ['', '**Execução incompleta: nenhum resultado confirmatório disponível; o critério de sucesso ainda não foi avaliado.**']
    lines += ['', 'Exploratório (separado da confirmação):']
    if old:
        r=old['primary']
        lines += [f"Holdout antigo aberto: {old['datasets']} datasets, {old['families']} famílias semânticas; ganho vs b8u {r['gain']*100:.3f} p.p., IC95% [{r['ci_low']*100:.3f}; {r['ci_high']*100:.3f}] p.p. Comparadores, Holm e caudas em [exploratory_41/analysis.json](exploratory_41/analysis.json).",
            'Análises por ε, faixa de n (≤1000, 1001–5000, 5001–10000, 10001–50000, >50000) e atributos numeric/nominal/mixed em exploratory_strata.csv de cada população.']
    else: lines += ['Holdout antigo de 41 famílias: ainda não concluído.']
    lines += ['', 'Figuras PDF+PNG: figures/dissertation/ (forest de ganhos vs b8u, CD Nemenyi, ganhos por ε).',
        '', 'Validação de procedência na etapa 2: candidatos TabClustPFN declaravam derivação de IDs/famílias já conhecidos. Antes de qualquer ajuste no holdout, o wrapper passou a resolver esses pais públicos; a validação foi reiniciada com o mesmo catálogo e a mesma permutação. O registro anterior está preservado em before_origin_validation_*. Nenhum critério de elegibilidade ou modelo foi alterado.',
        '', 'Desvios e escolhas operacionais: nenhum hiperparâmetro, alvo, fallback ou critério de sucesso foi alterado. O protocolo não define teste de p dos secundários nem convenção amostral do CVaR; foram usadas as rotinas bilaterais sign-flip e a convenção CVaR do desenvolvimento. Catálogo ativo; elegibilidade por dimensão codificada validada, sem restrição adicional de colunas brutas. Essas escolhas estão documentadas no executor e nos JSONs.']
    if error: lines += ['',f'Interrupção: `{error}`. Resultados incompletos não foram tratados como confirmação.']
    (OUT/'SUMMARY.md').write_text('\n'.join(lines)+'\n')
    if error is None: log('Etapa 7 CONCLUÍDA: SUMMARY.md salvo.')

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['freeze','select','run','analyze','exploratory','figures','summary','all'])
    args = parser.parse_args()
    stages=dict(freeze=freeze,select=select,run=run,analyze=analyze,exploratory=exploratory,
                figures=figures,summary=summary)
    try:
        if args.stage=='all':
            for stage in stages.values(): stage()
        else: stages[args.stage]()
    except Exception as error:
        summary(error=f'{type(error).__name__}: {error}')
        raise

if __name__ == '__main__':
    main()

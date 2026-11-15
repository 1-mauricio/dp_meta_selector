"""STAGE 1 only. Never trains a selector and contains no primary analysis."""
from __future__ import annotations
import os
for variable in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[variable]='1'
os.environ.setdefault('OPENML_CACHE_DIR',str(__import__('pathlib').Path('.cache/openml').resolve()))
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import io
import json
import multiprocessing as mp
from pathlib import Path
import re
import shutil
import time
import traceback
import zipfile
import requests
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from dpmeta.uci_external import prepare_uci
from dpmeta.openml_benchmark import canonical_family, _get_openml_data, _seed_for, stratified_holdout_indices, _load_manifest
from dpmeta.families import conservative_family
from dpmeta.external_confirmation import confirmation_family
from dpmeta.provenance import file_sha256, runtime_fingerprint, numerical_implementation_sha256
from scripts import confirmatory as frozen
from scripts.parallel_nb8u import atomic_json, DiskLimit
from scripts.external_uci_catalog import log

OUT=Path('results/holdout-external')
MAN=Path('data/external_manifest_uci.csv')
AMENDMENT=Path('protocols/CONFIRMATORY_PROTOCOL_EXTERNAL_AMENDMENT.md')
SEED=20261112
# Source-name aliases, based on UCI descriptions and standard OpenML names.
# A match excludes conservatively, including uncertainty about a shared base.
ALIASES={8:['audiology'],15:['breast-w','breast_cancer_wisconsin'],17:['wdbc','breast_cancer_wisconsig_diag'],
22:['kr-vs-kp'],23:['kr-vs-k','king_rook_vs_king'],43:['haberman'],47:['colic'],50:['segment'],
59:['letter'],69:['splice'],74:['musk'],75:['musk'],78:['page-blocks'],80:['optdigits'],81:['pendigits'],
95:['SPECT'],96:['SPECTF'],101:['tic-tac-toe'],105:['vote'],107:['waveform-5000'],
143:['australian_credit_approval'],144:['credit-g','german_credit_data'],145:['heart-statlog'],
146:['satellite_image','satimage'],147:['segment'],149:['vehicle'],151:['sonar'],
159:['MagicTelescope','MAGIC-Gamma-Telescope-Dataset'],161:['mammography'],172:['ozone_level'],
212:['vertebra-column'],225:['Indian-Liver-Patient-Dataset'],257:['user-knowledge'],
270:['gas-drift-different-concentrations','gas-drift'],329:['diabetes_debrecen','diabetic_retinopathy'],
372:['Pulsar-Dataset-HTRU2'],380:['youtube-spam-psy'],419:['Autism-Screening'],426:['autism-adult-data'],
468:['online-shoppers-intention'],529:['early-stage-diabetes'],544:['Estimation_of_Obesity_Levels'],
563:['UCI_churn'],597:['garments-worker-productivity'],602:['beans'],
697:['students_dropout_and_academic_success'],722:['NATICUSdroid'],
857:['Predicting_Risk_Factors_of_Chronic_Kidney_Disease'],887:['NHANES_age'],890:['aids','AIDS_Virus_Infection_Prediction']}


def disk_guard():
    free=shutil.disk_usage(OUT).free
    if free<5*1024**3:raise DiskLimit(f'free disk {free} bytes < 5 GiB')
    return free


def verify():
    assert file_sha256(AMENDMENT)==AMENDMENT.with_suffix('.sha256').read_text().split()[0]
    assert file_sha256(frozen.PROTOCOL)==frozen.EXPECTED
    state=json.loads((frozen.OUT/'frozen.json').read_text())
    for path,digest in state['hashes'].items():assert file_sha256(Path(path))==digest,path
    assert state['run_seed']==frozen.RUN_SEED==20260918
    assert numerical_implementation_sha256()==state['numerical_implementation_sha256']
    for path,digest in json.loads((OUT/'protected_sha256.json').read_text()).items():
        assert file_sha256(Path(path))==digest,path
    disk_guard()
    return state


def compact(value):return re.sub('[^a-z0-9]','',str(value).casefold())

def forms(value):return {canonical_family(value),conservative_family(value),confirmation_family(value)}


def exclusion_index():
    records=[];index={};origins=[]
    for path in sorted(Path('data').rglob('*')):
        if not path.is_file() or path.name.startswith('external_manifest_uci'):continue
        if path.suffix not in {'.csv','.json'}:continue
        raw=path.read_text();origins.append(raw.casefold())
        records.append(dict(path=str(path),sha256=file_sha256(path)))
        if path.suffix=='.csv':
            frame=pd.read_csv(path,keep_default_na=False)
            for col in frame:
                if col in {'dataset_name','name','family_id','semantic_family_id','canonical_family_id','declared_source_name'}:
                    for value in frame[col].astype(str):
                        if value:
                            for key in forms(value):index.setdefault(compact(key),(str(path),col,value))
    cat=OUT/'openml_catalog_all.csv';frame=pd.read_csv(cat,keep_default_na=False)
    records.append(dict(path=str(cat),sha256=file_sha256(cat),rows=len(frame)))
    for r in frame.itertuples():
        for key in forms(str(r.name)):index.setdefault(compact(key),('OpenML-full-catalog',str(r.did),str(r.name)))
    # Public legacy URLs/slugs and DOI strings are compared against all manifest content.
    return index,'\n'.join(origins),records


def overlap(meta,index,origin_text):
    name=meta['name'];did=int(meta['uci_id'])
    for value in [name,*ALIASES.get(did,[])]:
        for key in sorted(forms(value)):
            if compact(key) in index:return 'excluded_name_or_family',dict(alias=value,match=index[compact(key)])
    for key in ('repository_url','dataset_doi'):
        value=str(meta.get(key) or '').strip()
        if value and value.casefold() in origin_text:return 'excluded_declared_origin',dict(origin=value)
    # Explicit derivative/near-name matches: conservative same-base uncertainty.
    needle=compact(canonical_family(name))
    tokens=set(canonical_family(name).split('_'))-{'data','dataset','set','uci','classification','the','of','and','for'}
    for key,evidence in index.items():
        if key not in {'dataset','classification','prediction','regression','collection','benchmark','synthetic','training','validation','testing','features','records'} and min(len(key),len(needle))>=7 and (needle in key or key in needle):
            return 'excluded_uncertain_same_base',dict(match=evidence,rule='compact-name containment >=7')
        other=set(canonical_family(evidence[2]).split('_'))-{'data','dataset','set','uci','classification','the','of','and','for'}
        if min(len(tokens),len(other))>=2 and (tokens<=other or other<=tokens):
            return 'excluded_uncertain_same_base',dict(match=evidence,rule='multi-token source-name containment')
    return None,None


def download(url,path):
    if path.exists():return path.read_bytes()
    disk_guard()
    for attempt in range(3):
        try:
            r=requests.get(url,timeout=60);r.raise_for_status()
            disk_guard();tmp=path.with_suffix('.tmp');tmp.write_bytes(r.content);tmp.replace(path)
            log(f'Download público {url}; {len(r.content)} bytes; SHA256 {file_sha256(path)}.')
            return r.content
        except DiskLimit:raise
        except Exception:
            if attempt==2:raise
            time.sleep(attempt+1)


def source_frame(meta):
    did=int(meta['uci_id']);raw=OUT/'raw';raw.mkdir(exist_ok=True)
    if meta.get('data_url'):
        content=download(meta['data_url'],raw/f'{did}.csv')
        return pd.read_csv(io.BytesIO(content),na_values=meta.get('missing_values_symbol') or None)
    # Never select an arbitrary file/variant from a source archive. A unique
    # table with exactly the source-declared names can be loaded losslessly.
    url=f'https://archive.ics.uci.edu/static/public/{did}/{meta["repository_url"].rsplit("/",1)[-1]}.zip'
    content=download(url,raw/f'{did}.zip')
    tables=[]
    expected={str(v['name']) for v in meta.get('variables',[])}
    archive=zipfile.ZipFile(io.BytesIO(content))
    def entries(book, depth=0):
        if depth>3: raise ValueError('archive_nesting_above_3')
        for entry in book.infolist():
            if entry.filename.startswith('__MACOSX/'): continue
            if entry.filename.casefold().endswith('.zip'):
                yield from entries(zipfile.ZipFile(io.BytesIO(book.read(entry))),depth+1)
            else: yield book,entry
    for archive,entry in entries(archive):
        if entry.file_size>1024**3:raise ValueError('archive_table_exceeds_1GiB')
        if entry.filename.startswith('__MACOSX/'):continue
        suffix=Path(entry.filename).suffix.casefold()
        if suffix not in {'.csv','.xlsx','.xls','.json','.data','.arff'}:continue
        try:
            data=archive.read(entry)
            if suffix in {'.xlsx','.xls'}:
                book=pd.ExcelFile(io.BytesIO(data))
                candidates=[pd.read_excel(book,sheet_name=s) for s in book.sheet_names]
            elif suffix=='.json':candidates=[pd.read_json(io.BytesIO(data))]
            elif suffix=='.arff':
                from scipy.io import arff
                arr,_=arff.loadarff(io.StringIO(data.decode()));df=pd.DataFrame(arr)
                for col in df:
                    if df[col].dtype==object:df[col]=df[col].map(lambda v:v.decode() if isinstance(v,bytes) else v)
                candidates=[df]
            else:candidates=[pd.read_csv(io.BytesIO(data),sep=None,engine='python',na_values=meta.get('missing_values_symbol') or None)]
            for df in candidates:
                if expected<=set(df.columns):tables.append((entry.filename,df))
        except ImportError:raise
        except Exception:continue
    if len(tables)!=1:raise ValueError(f'no_unique_archive_table_matching_declared_schema:{[n for n,_ in tables]}')
    log(f'Arquivo UCI {did}: única tabela declarada {tables[0][0]}.')
    return tables[0][1]


def select():
    verify()
    if MAN.exists():
        record=json.loads(MAN.with_suffix('.selection.json').read_text())
        assert record['manifest_sha256']==file_sha256(MAN)
        log('Seleção anterior e SHA256 verificados; nenhuma substituição após modelos.');return
    index,origins,records=exclusion_index();atomic_json(OUT/'exclusion_sources.json',records)
    catalog=pd.read_csv(OUT/'uci_catalog.csv');allitems=[];nonclass=[]
    for r in catalog.sort_values('id').itertuples():
        meta=json.loads((OUT/'api'/f'uci_{r.id}.json').read_text())['data']
        if 'Classification' in (meta.get('tasks') or []):allitems.append(meta)
        else:nonclass.append(dict(uci_id=r.id,name=r.name,reason='not_declared_classification'))
    pd.DataFrame(nonclass).to_csv(OUT/'uci_nonclassification.csv',index=False)
    order=np.random.default_rng(SEED).permutation(len(allitems))
    ordered=[allitems[int(i)] for i in order]
    pd.DataFrame([dict(position=i,uci_id=m['uci_id'],name=m['name']) for i,m in enumerate(ordered,1)]).to_csv(OUT/'candidate_order.csv',index=False)
    log(f'Ordem congelada ANTES de baixar dados: {len(ordered)} candidatos de classificação; seed {SEED}; {len(nonclass)} não classificação; {len(records)} fontes de exclusão.')
    accepted=[];decisions=[];selected_index={}
    for pos,meta in enumerate(ordered,1):
        disk_guard();did=int(meta['uci_id']);decision=dict(position=pos,uci_id=did,name=meta['name'],at=datetime.now(timezone.utc).isoformat())
        try:
            reason,evidence=overlap(meta,index,origins)
            if reason:
                decision.update(status=reason,reason=json.dumps(evidence,ensure_ascii=False))
            elif not 200<=int(meta.get('num_instances') or 0)<=50000:
                decision.update(status='excluded_rows',reason=f"declared rows={meta.get('num_instances')}")
            elif len(meta.get('target_col') or [])!=1:
                decision.update(status='excluded_declared_target',reason='source must declare exactly one target; no target choice/relabeling')
            elif int(meta.get('num_features') or 0)>512:
                decision.update(status='excluded_feature_lower_bound',reason='declared raw feature count >512')
            else:
                reason,evidence=overlap(meta,selected_index,'') if selected_index else (None,None)
                if reason:
                    decision.update(status='excluded_selected_family',reason=json.dumps(evidence))
                else:
                    log(f'Validação/carregamento UCI {did}, posição {pos}: {meta["name"]}.')
                    row=prepare_uci(meta,source_frame(meta),OUT/'datasets_public'/f'{did}.parquet')
                    for repetition in range(3):
                        x,y,_=_get_openml_data(None,pd.Series(row))
                        stratified_holdout_indices(y,_seed_for(frozen.RUN_SEED,row['dataset_id'],repetition,'portfolio_split'))
                    family=conservative_family(meta['name'])
                    row.update(family_id=family,semantic_family_id=family,canonical_family_id=canonical_family(meta['name']),selection_position=pos,
                        source_metadata_sha256=file_sha256(OUT/'api'/f'uci_{did}.json'))
                    accepted.append(row)
                    for key in forms(meta['name']):selected_index[compact(key)]=('selected-UCI',str(did),meta['name'])
                    decision.update(status='accepted',reason='public schema, encoding, 3 splits and conservative disjunction validated')
                    log(f'Aceita família UCI {len(accepted)}/40: {did} {meta["name"]}.')
        except DiskLimit:raise
        except Exception as error:
            decision.update(status='validation_loading_failure',reason=f'{type(error).__name__}: {error}')
        decisions.append(decision)
        pd.DataFrame(decisions).to_csv(OUT/'selection_decisions.csv',index=False)
        log(f'Decisão {pos}/{len(ordered)} UCI {did}: {decision["status"]}; {decision["reason"]}')
        if len(accepted)==40:break
    columns=['dataset_id','dataset_name','family_id','semantic_family_id','n_instances','n_encoded_features','public_schema_json']
    selected=pd.DataFrame(accepted) if accepted else pd.DataFrame(columns=columns)
    selected.to_csv(MAN,index=False)
    atomic_json(MAN.with_suffix('.selection.json'),dict(completed_at=datetime.now(timezone.utc).isoformat(),manifest_sha256=file_sha256(MAN),
        amendment_sha256=file_sha256(AMENDMENT),frozen_selectors_sha256=file_sha256(frozen.OUT/'frozen.json'),
        candidate_order_sha256=file_sha256(OUT/'candidate_order.csv'),decisions_sha256=file_sha256(OUT/'selection_decisions.csv'),
        selection_seed=SEED,run_seed=frozen.RUN_SEED,id_rule='1000000 + uci_id',selected_families=len(accepted),
        candidates=len(ordered),target=40,stage2_required=len(accepted)<30,sources=records,
        replacement_rule='seeded order, validation/loading only, no model outcomes',executor_sha256=file_sha256(Path(__file__))))
    log(f'Manifesto FINAL e decisões salvos ANTES dos modelos: {len(accepted)} famílias UCI.')


def worker(queue,stop,signature):
    with threadpool_limits(limits=1):
        while not stop.is_set():
            task=queue.get()
            if task is None:return
            row=pd.Series(task);did=int(row.dataset_id);started=time.monotonic()
            try:
                disk_guard()
                assert file_sha256(Path(row.local_data_path))==row.local_data_sha256
                log(f'Ajustes INÍCIO dataset {did}: 15 opções × 8 eps × 3 repetições.')
                generated=frozen.pb.run_dataset(None,row,frozen.EPS,3,frozen.RUN_SEED,pipelines=frozen.P15,
                    numeric_transform='tanh',include_research_features=False)
                frozen.pb.validate_condition_grid(pd.DataFrame(generated),pd.DataFrame([task]),epsilons=frozen.EPS,repeats=3,pipelines=frozen.P15)
                ledgers={}
                for r in generated:
                    frozen.audit_ledger(r)
                    r['split_seed']=_seed_for(frozen.RUN_SEED,did,r['repetition'],'portfolio_split')
                    payload=r.pop('ledger_json');key=hashlib.sha256(payload.encode()).hexdigest();ledgers[key]=payload
                    r.update(ledger_id=key,ledger_audit_passed=True)
                disk_guard()
                atomic_json(OUT/'confirmatory/datasets'/f'{did}.json',dict(dataset_id=did,signature=signature,rows=generated,ledgers=ledgers,
                    completed_at=datetime.now(timezone.utc).isoformat(),seconds=time.monotonic()-started))
                log(f'Ajustes CONCLUÍDOS {did}: {len(generated)} ledgers auditados; {time.monotonic()-started:.2f}s.')
            except DiskLimit:
                stop.set();log(f'PARADA DISCO dataset {did}.');return
            except Exception as error:
                atomic_json(OUT/'confirmatory/failure_details'/f'{did}.json',dict(dataset_id=did,error=f'{type(error).__name__}: {error}',traceback=traceback.format_exc()))
                log(f'FALHA execução {did}: {type(error).__name__}: {error}; sem substituição por resultados.')


def run():
    verify();record=json.loads(MAN.with_suffix('.selection.json').read_text());assert file_sha256(MAN)==record['manifest_sha256']
    man=_load_manifest(MAN)
    for d in ('datasets','failure_details'):(OUT/'confirmatory'/d).mkdir(exist_ok=True,parents=True)
    log('Recomendações dos três seletores congelados: SOMENTE features públicas; hashes verificados; nenhum retreino.')
    if len(man):frozen.recommendations(man,OUT/'recommendations.csv')
    signature=dict(manifest_sha256=file_sha256(MAN),frozen_sha256=file_sha256(frozen.OUT/'frozen.json'),
        numerical=numerical_implementation_sha256(),runtime=runtime_fingerprint(),pipelines=list(frozen.P15),epsilons=list(frozen.EPS),
        repeats=3,seed=frozen.RUN_SEED,numeric_transform='tanh',workers=8,OMP_NUM_THREADS=1,min_free_bytes=5*1024**3,
        executor_sha256=file_sha256(Path(__file__)),loader_sha256=file_sha256(Path('dpmeta/uci_external.py')))
    config=OUT/'confirmatory/run_config.json'
    if config.exists():assert json.loads(config.read_text())==signature,'resume signature mismatch'
    else:atomic_json(config,signature)
    completed=set()
    for path in (OUT/'confirmatory/datasets').glob('*.json'):
        commit=json.loads(path.read_text());assert commit['signature']==signature
        assert commit['dataset_id'] in set(man.dataset_id)
        frozen.pb.validate_condition_grid(pd.DataFrame(commit['rows']),man[man.dataset_id.eq(commit['dataset_id'])],epsilons=frozen.EPS,repeats=3,pipelines=frozen.P15)
        for r in commit['rows']:
            payload=commit['ledgers'][r['ledger_id']];assert hashlib.sha256(payload.encode()).hexdigest()==r['ledger_id']
            frozen.audit_ledger(dict(r,ledger_json=payload))
        completed.add(commit['dataset_id'])
    pending=man[~man.dataset_id.isin(completed)]
    ctx=mp.get_context('spawn');queue=ctx.Queue();stop=ctx.Event()
    for raw in pending.to_dict('records'):queue.put(raw)
    for _ in range(8):queue.put(None)
    processes=[ctx.Process(target=worker,args=(queue,stop,signature)) for _ in range(8)]
    log(f'Execução retomável: {len(completed)} completos; {len(pending)} pendentes; 8 workers; OMP_NUM_THREADS=1; limite disco 5 GiB.')
    try:
        for proc in processes:proc.start()
        while any(proc.is_alive() for proc in processes):
            disk_guard()
            if stop.is_set():raise DiskLimit('worker disk stop')
            for proc in processes:proc.join(timeout=.1)
    finally:
        for proc in processes:
            if proc.is_alive():proc.terminate()
            proc.join()
        queue.cancel_join_thread();queue.close()
    assert all(p.exitcode==0 for p in processes),'worker exited; resume required'
    commits=[json.loads(p.read_text()) for p in (OUT/'confirmatory/datasets').glob('*.json')]
    rows=[r for c in commits for r in c['rows']];full=pd.DataFrame(rows)
    if len(man):
        frozen.pb.validate_condition_grid(full,man,epsilons=frozen.EPS,repeats=3,pipelines=frozen.P15)
        assert set(full.dataset_id)==set(man.dataset_id),'incomplete execution; inspect failure_details and resume'
        full.to_csv(OUT/'confirmatory/conditions.csv',index=False)
        ledgers={k:v for c in commits for k,v in c['ledgers'].items()}
        pd.DataFrame([dict(ledger_id=k,ledger_json=v) for k,v in ledgers.items()]).to_csv(OUT/'confirmatory/ledgers.csv',index=False)
        full[frozen.KEYS+['ledger_id','ledger_audit_passed','epsilon_used','delta_used','split_seed','train_n','test_n','encoded_dim','class_count']].to_csv(OUT/'confirmatory/privacy_audit.csv',index=False)
    atomic_json(OUT/'confirmatory/coverage.json',dict(datasets=len(commits),rows=len(rows),ledger_audits=len(rows),complete=True))
    verify();log(f'Execução STAGE 1 concluída; {len(rows)} ajustes auditados. Nenhuma análise primária calculada.')


def status():
    verify();record=json.loads(MAN.with_suffix('.selection.json').read_text());dec=pd.read_csv(OUT/'selection_decisions.csv')
    coverage_path=OUT/'confirmatory/coverage.json';coverage=json.loads(coverage_path.read_text()) if coverage_path.exists() else {}
    failures=[json.loads(p.read_text()) for p in (OUT/'confirmatory/failure_details').glob('*.json')]
    first=re.search(r'(2026-\d\d-\d\dT[^ ]+)',(OUT/'LOG.md').read_text()).group(1)
    elapsed=(datetime.now(timezone.utc)-datetime.fromisoformat(first)).total_seconds()
    n=record['selected_families'];counts=dec.status.value_counts().to_dict()
    lines=['# STAGE 1 — UCI',f'Atualizado em {datetime.now(timezone.utc).isoformat()} (UTC).',
        f'Famílias UCI elegíveis obtidas: **{n}** (alvo: 40).',
        f'Classificação: {record["candidates"]} candidatos na permutação com semente 20261112. Catálogo total UCI: 690 entradas. Catálogo OpenML: 25.245 IDs, incluindo ativos, desativados e em preparação.',
        'Exclusões e decisões por motivo:']
    lines += [f'- `{k}`: {v}' for k,v in sorted(counts.items())]
    lines += [f'Falhas de validação/carregamento: {counts.get("validation_loading_failure",0)}; detalhes por candidato em selection_decisions.csv.',
        f'Execução: {coverage.get("datasets",0)}/{n} datasets; {coverage.get("rows",0)} ajustes de {n*360} previstos; {coverage.get("ledger_audits",0)} auditorias de ledger aprovadas.',
        f'Falhas de execução registradas: {len(failures)}.',f'Tempo decorrido: {elapsed:.1f} s ({elapsed/60:.1f} minutos), incluindo aquisição, validação, testes e execução.',
        f'Etapa 2 (Kaggle) necessária: **{"SIM" if n<30 else "NÃO"}**, pois foram obtidas {n} famílias UCI {"(<30)" if n<30 else "(≥30)"}. Kaggle não foi iniciado.',
        'As recomendações usam somente metadados públicos e os três seletores congelados, com hashes verificados. Protocolos e results/holdout-openml/ preservados por SHA256; sem comandos git e sem acesso ao checkout anterior.',
        'A análise primária, bootstrap, comparação de ganhos e testes de hipótese NÃO foram calculados.']
    (OUT/'STAGE1_STATUS.md').write_text('\n\n'.join(lines)+'\n')
    log('STAGE1_STATUS.md escrito em português; parada antes de análise primária e antes de Kaggle.')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['select','run','status']);args=parser.parse_args()
    globals()[args.stage]()

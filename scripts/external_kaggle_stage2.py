"""Additive Kaggle stage of the frozen external confirmation; no selector fit.

Credentials live only in requests' in-memory Basic auth. No Kaggle config,
request headers, response headers, cookies or exception dumps are persisted.
"""
from __future__ import annotations
import os
for _v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_v]='1'
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import re
import time
import threading
import zipfile
import numpy as np
import pandas as pd
import requests
from scripts import external_uci_stage1 as uci
from scripts import confirmatory as frozen
from scripts.parallel_nb8u import atomic_json, DiskLimit
from dpmeta.provenance import file_sha256
from dpmeta.uci_external import prepare_uci
from dpmeta.openml_benchmark import _get_openml_data, _seed_for, stratified_holdout_indices

ROOT=Path('results/holdout-external')
OUT=ROOT/'kaggle'
MAN=Path('data/external_manifest_kaggle.csv')
QUERY=dict(group='public',sortBy='votes',filetype='csv',tagids='tabular,classification')
BASE='https://www.kaggle.com/api/v1'
_UCI_WORKER=uci.worker
_RATE_LOCK=threading.Lock()
_NEXT_REQUEST=0.

def throttle(cooldown=0.):
    global _NEXT_REQUEST
    with _RATE_LOCK:
        if cooldown:_NEXT_REQUEST=max(_NEXT_REQUEST,time.monotonic()+cooldown)
        wait=max(0.,_NEXT_REQUEST-time.monotonic())
        _NEXT_REQUEST=max(_NEXT_REQUEST,time.monotonic())+1.1
    while wait>0:
        interval=min(wait,30.);time.sleep(interval);wait-=interval

def worker(queue,stop,signature):
    # Spawn imports modules afresh: redirect inside EACH child, not just parent.
    uci.OUT=OUT;uci.log=log
    return _UCI_WORKER(queue,stop,signature)

def now():return datetime.now(timezone.utc).isoformat()

def log(message):
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'LOG.md').open('a') as f:f.write(f'- {now()} — {message}\n')
    print(now(),message,flush=True)

def safe_bytes(content):
    for name in ('KAGGLE_USERNAME','KAGGLE_KEY'):
        value=os.environ.get(name)
        if value and value.encode() in content:
            raise ValueError('credential_value_detected_in_response; response not persisted')
    return content

def fetch(endpoint,params=None,binary=False):
    auth=(os.environ['KAGGLE_USERNAME'],os.environ['KAGGLE_KEY'])
    last_status=None
    for attempt in range(12):
        uci.disk_guard()
        try:
            throttle()
            with requests.get(BASE+endpoint,params=params,auth=auth,
                headers={'User-Agent':'kaggle-api/v1.7.0','Accept':'application/json'},
                timeout=60,stream=True) as response:
                if response.status_code==429:
                    log('HTTP 429: pausa coordenada de aquisição; candidato preservado para nova tentativa.')
                    retry=response.headers.get('Retry-After','70')
                    delay=max(75.,float(retry)) if retry.isdigit() else 75.
                    throttle(delay)
                    if attempt==11:raise RuntimeError('Kaggle_rate_limit_persistent; resume acquisition, do not exclude candidate')
                    continue
                if response.status_code!=200:
                    last_status=response.status_code
                    raise ValueError(f'Kaggle_HTTP_{response.status_code}')
                chunks=[];size=0
                for chunk in response.iter_content(1024*1024):
                    uci.disk_guard();size+=len(chunk)
                    if size>1024**3:raise ValueError('response_above_1GiB')
                    chunks.append(chunk)
                content=safe_bytes(b''.join(chunks))
                return content if binary else json.loads(content)
        except DiskLimit:raise
        except (requests.RequestException,ValueError):
            if attempt>=2:raise ValueError(f'Kaggle_request_failed_after_retries_HTTP_{last_status}') from None
            time.sleep(attempt+1)

def verify():
    assert file_sha256(uci.AMENDMENT)==uci.AMENDMENT.with_suffix('.sha256').read_text().split()[0]
    assert file_sha256(frozen.PROTOCOL)==frozen.EXPECTED
    state=json.loads((frozen.OUT/'frozen.json').read_text())
    for path,digest in state['hashes'].items():assert file_sha256(Path(path))==digest,path
    assert uci.numerical_implementation_sha256()==state['numerical_implementation_sha256']
    for path,digest in json.loads((ROOT/'protected_sha256.json').read_text()).items():
        assert file_sha256(Path(path))==digest,path
    uci.disk_guard()
    if (OUT/'protected_sha256.json').exists():
        for path,digest in json.loads((OUT/'protected_sha256.json').read_text()).items():
            assert file_sha256(Path(path))==digest,path

def catalog():
    OUT.mkdir(parents=True,exist_ok=True);verify()
    if not (OUT/'protected_sha256.json').exists():
        paths=[p for p in ROOT.rglob('*') if p.is_file() and OUT not in p.parents]
        paths += [p for p in Path('.').glob('CONFIRMATORY_PROTOCOL*') if p.is_file()]
        paths += [p for p in Path('data').rglob('*') if p.is_file()]
        atomic_json(OUT/'protected_sha256.json',{str(p):file_sha256(p) for p in paths})
    api=OUT/'api';api.mkdir(exist_ok=True)
    def page(number):
        path=api/f'list_{number:02d}.json'
        if path.exists():return json.loads(path.read_text())
        result=fetch('/datasets/list',dict(QUERY,page=number))
        assert isinstance(result,list) and len(result)==20,'expected 20 candidates per API page'
        atomic_json(path,result);return result
    with ThreadPoolExecutor(max_workers=8) as pool:pages=list(pool.map(page,range(1,31)))
    items=[x for page_ in pages for x in page_][:600]
    assert len(items)==600 and len({x['ref'] for x in items})==600
    assert all(not x.get('isPrivate',True) for x in items)
    assert all({'tabular','classification'}<={t['name'] for t in x['tags']} for x in items)
    assert all(items[i]['voteCount']>=items[i+1]['voteCount'] for i in range(599))
    path=OUT/'catalog.csv'
    if not path.exists():
        pd.DataFrame([dict(api_rank=i+1,kaggle_id=x['id'],ref=x['ref'],name=x['title'],votes=x['voteCount']) for i,x in enumerate(items)]).to_csv(path,index=False)
        order=np.random.default_rng(20261112).permutation(600)
        pd.read_csv(path).iloc[order].assign(position=np.arange(1,601)).to_csv(OUT/'candidate_order.csv',index=False)
        atomic_json(OUT/'query.json',dict(at=now(),endpoint=BASE+'/datasets/list',query=QUERY,pages=30,page_size=20,
            limit=600,seed=20261112,catalog_sha256=file_sha256(path),order_sha256=file_sha256(OUT/'candidate_order.csv')))
    log('600 candidatos públicos: ordem API por votos e ambas tags verificados; permutação 20261112 salva antes de downloads/modelos.')
    index,origins,_=exclusion_index()
    def metadata(item):
        # A name/family exclusion is final without another API request.
        if overlap(item['title'],index)[0]:return
        ref=item['ref'];path=api/f"view_{item['id']}.json"
        if not path.exists() or json.loads(path.read_text()).get('acquisition_failure'):
            try:atomic_json(path,fetch('/datasets/view/'+ref))
            except ValueError as error:atomic_json(path,dict(acquisition_failure=True,final_non_rate_limited_failure=True,reason=str(error)))
    ordered=[items[int(i)] for i in np.random.default_rng(20261112).permutation(600)]
    with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(metadata,ordered))
    verify();log('Metadados adquiridos para candidatos sem sobreposição nominal; inventários CSV sob demanda na validação; nenhuma execução de modelos.')

def exclusion_index():
    index,origins,records=uci.exclusion_index()
    def names(value,path):
        if isinstance(value,dict):
            for key,item in value.items():
                if key in {'dataset_name','name','family_id','semantic_family_id','canonical_family_id','declared_source_name'} and isinstance(item,str):
                    for form in uci.forms(item):index.setdefault(uci.compact(form),(str(path),key,item))
                names(item,path)
        elif isinstance(value,list):
            for item in value:names(item,path)
    for path in Path('data').rglob('*.json'):
        if path.name.startswith('external_manifest_kaggle'):continue
        names(json.loads(path.read_text()),path)
    for path in [ROOT/'uci_catalog.csv',uci.MAN]:
        frame=pd.read_csv(path,keep_default_na=False);records.append(dict(path=str(path),sha256=file_sha256(path)))
        col='name' if 'name' in frame else 'dataset_name'
        for name in frame[col]:
            for key in uci.forms(name):index.setdefault(uci.compact(key),(str(path),'source',name))
    # The UCI stage's exclusion_index skips its own manifest; include every
    # source URL/DOI and source name explicitly for stage 2.
    for path in (ROOT/'api').glob('uci_*.json'):
        m=json.loads(path.read_text()).get('data') or {}
        for field in ('repository_url','dataset_doi','data_url'):
            if m.get(field):origins+='\n'+str(m[field]).casefold()
    return index,origins,records

def overlap(name,index,origins='',description=''):
    result,evidence=uci.overlap(dict(name=name,uci_id=-1),index,origins)
    if result:return result,evidence
    lower=description.casefold()
    # Explicit repository provenance prevents differently named mirrors.
    if re.search(r'archive\.ics\.uci\.edu|\buci\b|openml\.org|\bopenml\b',lower):
        return 'excluded_declared_repository_origin',{'rule':'description declares UCI/OpenML provenance'}
    for doi in re.findall(r'10\.\d{4,9}/[^\s<>"{}\[\]]+',description):
        doi=doi.rstrip('.,;)').casefold()
        if doi in origins:return 'excluded_declared_origin',{'doi':doi}
    for url in re.findall(r'https?://[^\s<>)\]"\']+',description):
        if url.rstrip('/').casefold() in origins:
            return 'excluded_declared_origin',{'origin':url}
        slug=url.rstrip('/').rsplit('/',1)[-1]
        reason,evidence=uci.overlap(dict(name=slug,uci_id=-1),index,'')
        if reason:return 'excluded_origin_family',dict(url=url,evidence=evidence)
    return None,None

def select():
    verify();assert (OUT/'review.json').exists(),'explicit target/origin review required before selection'
    if MAN.exists():
        record=json.loads(MAN.with_suffix('.selection.json').read_text());assert file_sha256(MAN)==record['manifest_sha256'];return
    index,origins,records=exclusion_index();review=json.loads((OUT/'review.json').read_text())
    order=pd.read_csv(OUT/'candidate_order.csv');accepted=[];decisions=[];selected={}
    for r in order.itertuples(index=False):
        uci.disk_guard();mpath=OUT/'api'/f'view_{r.kaggle_id}.json'
        m=json.loads(mpath.read_text()) if mpath.exists() else {}
        decision=dict(position=int(r.position),kaggle_id=int(r.kaggle_id),ref=r.ref,name=r.name,at=now())
        reason,evidence=overlap(r.name,index,origins,m.get('description',''))
        entry=review.get(r.ref,{})
        if reason:decision.update(status=reason,reason=json.dumps(evidence,ensure_ascii=False))
        elif m.get('acquisition_failure'):decision.update(status='validation_loading_failure',reason=m['reason'])
        elif not entry or entry.get('status')!='eligible_for_validation':
            decision.update(status=entry.get('status','excluded_ambiguous_target_or_origin'),reason=entry.get('reason','No certified unique source-declared target and disjoint origin'))
        else:
            try:
                assert entry['target'] and entry['target_evidence'] and entry['origin_review']
                assert entry['target_evidence'] in m.get('description',''),'target evidence must be verbatim source metadata'
                reason,evidence=overlap(entry.get('family',r.name),index,origins)
                if not reason:reason,evidence=overlap(entry.get('family',r.name),selected)
                if reason:decision.update(status='excluded_reviewed_family',reason=json.dumps(evidence,ensure_ascii=False))
                else:
                    fpath=OUT/'api'/f'files_{r.kaggle_id}.json'
                    if not fpath.exists() or json.loads(fpath.read_text()).get('acquisition_failure'):
                        atomic_json(fpath,fetch('/datasets/list/'+r.ref))
                    files=json.loads(fpath.read_text())
                    # Large source-code repositories need not enumerate thousands
                    # of non-tabular .sol files: ask the API for a full-sized page.
                    if files.get('nextPageToken'):
                        files=fetch('/datasets/list/'+r.ref,params={'pageSize':10000})
                    all_files=list(files.get('datasetFiles',[]));token=files.get('nextPageToken');seen=set()
                    while token:
                        assert token not in seen,'repeated file pagination token';seen.add(token)
                        page=fetch('/datasets/list/'+r.ref,params={'pageToken':token,'pageSize':10000})
                        all_files.extend(page.get('datasetFiles',[]));token=page.get('nextPageToken')
                    files=dict(datasetFiles=all_files)
                    inventory=OUT/'api'/f'files_all_{r.kaggle_id}.json';atomic_json(inventory,files)
                    csvs=[f['name'] for f in files['datasetFiles'] if f['name'].casefold().endswith('.csv')]
                    assert csvs,'no CSV files'
                    if entry.get('source_file'):
                        assert entry['source_file'] in csvs and entry['source_file'] in m['description']
                        assert entry['table_evidence'] in m['description']
                        csvs=[entry['source_file']]
                    declared_target=entry['target'];target=declared_target;tables=[]
                    for filename in csvs:
                        raw=OUT/'raw'/f'{r.kaggle_id}_{hashlib.sha256(filename.encode()).hexdigest()[:16]}.download';raw.parent.mkdir(exist_ok=True)
                        if not raw.exists():raw.write_bytes(fetch('/datasets/download/'+r.ref+'/'+filename,binary=True))
                        content=raw.read_bytes()
                        if zipfile.is_zipfile(io.BytesIO(content)):
                            z=zipfile.ZipFile(io.BytesIO(content));entries=[x for x in z.infolist() if not x.is_dir()]
                            assert len(entries)==1,'nonunique archive'
                            assert entries[0].file_size<=1024**3,'archive_table_exceeds_1GiB (same UCI loader resource limit)'
                            content=z.read(entries[0])
                        frame=pd.read_csv(io.BytesIO(content))
                        if entry.get('target_position') is not None:
                            assert frame.shape[1]==entry['source_column_count'],'source-declared positional schema mismatch'
                            name=frame.columns[int(entry['target_position'])]
                            assert str(name) not in entry.get('declared_class_values',[]),'headerless CSV unsupported by existing named-schema loader'
                            matches=[name]
                        else:matches=[c for c in frame if uci.compact(c)==uci.compact(declared_target)]
                        if len(matches)==1:
                            tables.append((filename,frame,matches[0]))
                            assert len(tables)==1,'multiple CSV tables match source-declared target; no variant choice'
                            assert 200<=int(frame[matches[0]].notna().sum())<=50000,'row_count_outside_200_50000 after missing-target removal'
                    assert len(tables)==1,'no unique CSV matching source-declared target; no arbitrary variant or concatenation'
                    filename,frame,target=tables[0]
                    source_file=next(x for x in files['datasetFiles'] if x['name']==filename)
                    hints={x['name']:str(x.get('type') or '').casefold() for x in source_file.get('columns',[])}
                    type_map={'number':'numeric','float':'real','double':'real','string':'nominal','text':'nominal','bool':'binary','boolean':'binary'}
                    # Infer feature dtypes as explicitly allowed by the amendment;
                    # retain every non-target column, including source IDs.
                    meta=dict(uci_id=1000000+int(r.kaggle_id),name=r.name,tasks=['Classification'],num_instances=len(frame),target_col=[target],
                        variables=[dict(name=c,role='Target' if c==target else 'Feature',type=type_map.get(hints.get(c,''),hints.get(c,''))) for c in frame],
                        repository_url='https://www.kaggle.com/datasets/'+r.ref,data_url=BASE+'/datasets/download/'+r.ref+'/'+filename)
                    row=prepare_uci(meta,frame,OUT/'datasets_public'/f'{r.kaggle_id}.parquet')
                    family=entry.get('family',r.name)
                    row.update(kaggle_id=int(r.kaggle_id),kaggle_ref=r.ref,uci_id=None,dataset_id_rule='2000000 + kaggle_id',
                        source='Kaggle-public-API-external-final',family_id=uci.conservative_family(family),
                        semantic_family_id=uci.conservative_family(family),canonical_family_id=uci.canonical_family(family),
                        declared_source_name=family,selection_position=int(r.position),source_metadata_sha256=file_sha256(OUT/'api'/f'view_{r.kaggle_id}.json'),
                        declared_origin_url=entry.get('origin_url',meta['repository_url']),source_csv=filename,
                        declared_target_text=declared_target,target_matching='unique exact normalized name; no semantic guessing',
                        source_file_inventory_sha256=file_sha256(inventory),
                        feature_type_rule='source machine-readable type if present, otherwise pandas dtype')
                    for rep in range(3):
                        _,y,_=_get_openml_data(None,pd.Series(row));stratified_holdout_indices(y,_seed_for(frozen.RUN_SEED,row['dataset_id'],rep,'portfolio_split'))
                    accepted.append(row)
                    for key in uci.forms(family):selected[uci.compact(key)]=('selected-Kaggle',r.ref,family)
                    decision.update(status='accepted',reason='source target, conservative provenance, schema/tanh encoding and 3 splits validated')
            except DiskLimit:raise
            except Exception as e:decision.update(status='validation_loading_failure',reason=f'{type(e).__name__}: {e}')
        decisions.append(decision);pd.DataFrame(decisions).to_csv(OUT/'selection_decisions.csv',index=False)
        log(f'{r.position}/600 {r.ref}: {decision["status"]}')
        if len(accepted)+len(pd.read_csv(uci.MAN))==40:break
    assert accepted,'No eligible Kaggle families; cannot run or analyze'
    pd.DataFrame(accepted).to_csv(MAN,index=False)
    atomic_json(MAN.with_suffix('.selection.json'),dict(at=now(),manifest_sha256=file_sha256(MAN),
        review_sha256=file_sha256(OUT/'review.json'),query_sha256=file_sha256(OUT/'query.json'),
        candidate_order_sha256=file_sha256(OUT/'candidate_order.csv'),decisions_sha256=file_sha256(OUT/'selection_decisions.csv'),
        amendment_sha256=file_sha256(uci.AMENDMENT),frozen_sha256=file_sha256(frozen.OUT/'frozen.json'),
        selected_families=len(accepted),uci_families=len(pd.read_csv(uci.MAN)),target_total_external=40,seed=20261112,
        id_rule='2000000 + kaggle_id',sources=records,replacements='validation/loading only, seeded order, no outcomes',executor_sha256=file_sha256(Path(__file__))))
    verify();log(f'Manifesto FINAL pré-modelos salvo: {len(accepted)} famílias Kaggle.')

def run():
    verify();record=json.loads(MAN.with_suffix('.selection.json').read_text());assert file_sha256(MAN)==record['manifest_sha256']
    assert file_sha256(OUT/'review.json')==record['review_sha256']
    # Reuse the exact UCI executor without invoking its selection/report stages.
    # All mutations are redirected into the NEW Kaggle directory.
    execution=dict(stage2_executor_sha256=file_sha256(Path(__file__)),manifest_sha256=file_sha256(MAN),
        selection_sha256=file_sha256(MAN.with_suffix('.selection.json')))
    config=OUT/'stage2_execution.json'
    if config.exists():assert json.loads(config.read_text())==execution,'stage2 resume source mismatch'
    else:atomic_json(config,execution)
    uci.OUT=OUT;uci.MAN=MAN;uci.log=log;uci.verify=verify;uci.worker=worker
    uci.run()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['catalog','select','run']);args=p.parse_args();globals()[args.stage]()

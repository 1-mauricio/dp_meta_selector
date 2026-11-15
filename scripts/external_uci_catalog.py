"""Public metadata acquisition only; never fits models or changes internal results."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import time
import requests
import pandas as pd

OUT = Path('results/holdout-external')

def log(message):
    line = f'- {datetime.now(timezone.utc).isoformat()} — {message}\n'
    with (OUT/'LOG.md').open('a') as f: f.write(line)
    print(line, end='', flush=True)

def fetch(url, path):
    if path.exists(): return json.loads(path.read_text())
    for attempt in range(3):
        try:
            r=requests.get(url,timeout=45); r.raise_for_status(); payload=r.json()
            temp=path.with_suffix('.tmp'); temp.write_text(json.dumps(payload)); temp.replace(path)
            return payload
        except Exception:
            if attempt==2: raise
            time.sleep(1+attempt)

def main():
    OUT.mkdir(exist_ok=True); api=OUT/'api'; api.mkdir(exist_ok=True)
    protected=[Path('protocols/CONFIRMATORY_PROTOCOL_FINAL.md'),Path('protocols/CONFIRMATORY_PROTOCOL_EXTERNAL_AMENDMENT.md'),*Path('results/holdout-openml').rglob('*')]
    snapshot=OUT/'protected_sha256.json'
    if not snapshot.exists():
        snapshot.write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in protected if p.is_file()},indent=2))
    items=fetch('https://archive.ics.uci.edu/api/datasets/list',api/'list.json')['data']
    log(f'Catálogo UCI público: {len(items)} IDs; adquirindo metadados de TODOS, inclusive sem importação ucimlrepo.')
    def metadata(item):
        path=api/f"uci_{item['id']}.json"
        try:
            value=fetch(f"https://archive.ics.uci.edu/api/dataset?id={item['id']}",path)
            data=value.get('data')
            if not isinstance(data,dict): raise ValueError(str(value)[:250])
            return dict(id=item['id'],name=item['name'],tasks=data.get('tasks'),n=data.get('num_instances'),features=data.get('num_features'),metadata_ok=True)
        except Exception as e:return dict(id=item['id'],name=item['name'],metadata_ok=False,error=f'{type(e).__name__}: {e}')
    rows=[]
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures=[pool.submit(metadata,i) for i in items]
        for future in as_completed(futures):
            rows.append(future.result())
            if len(rows)%50==0:log(f'Metadados UCI: {len(rows)}/{len(items)} respostas processadas.')
    pd.DataFrame(rows).sort_values('id').to_csv(OUT/'uci_catalog.csv',index=False)
    log(f'Metadados UCI concluídos; falhas {sum(not r["metadata_ok"] for r in rows)}.')
    # Explicitly fetch every OpenML status; old active-only catalog is insufficient.
    frames=[pd.read_csv('results/holdout-openml/catalog.csv')]
    for status in ('active','deactivated','in_preparation'):
        payload=fetch(f'https://www.openml.org/api/v1/json/data/list/limit/100000/status/{status}',api/f'openml_{status}.json')
        datasets=payload.get('data',{}).get('dataset',[])
        if not datasets:raise RuntimeError(f'OpenML catalog incomplete: {status}: {str(payload)[:200]}')
        frames.append(pd.DataFrame(datasets))
        log(f'Catálogo OpenML {status}: {len(datasets)} nomes obtidos.')
    pd.concat(frames,ignore_index=True).drop_duplicates('did',keep='last').to_csv(OUT/'openml_catalog_all.csv',index=False)
    log('Catálogo OpenML integral de nomes salvo externamente; catálogo interno preservado.')

if __name__=='__main__':main()

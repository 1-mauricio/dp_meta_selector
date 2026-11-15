"""Public metadata prefetch only; never changes the ordered selection decision.

Install a complete metadata directory by atomic rename only when the main
OpenML cache directory does not exist. Never overwrite files of the selector.
"""
import os
os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import pandas as pd
from scripts.confirmatory import (OUT,CACHE,_openml,exclusions,public_origin,
    canonical_family,conservative_family,log,disk_guard)

def main():
    ids,canonical,semantic,_=exclusions()
    catalog=pd.read_csv(OUT/'catalog.csv').set_index('did')
    names=catalog.name.to_dict()
    order=pd.read_csv(OUT/'candidate_order.csv')
    position=int(pd.read_csv(OUT/'selection_decisions.csv').position.max())
    staging=OUT/'metadata_prefetch_cache'
    om=_openml(staging)
    relative=Path('org/openml/www/datasets')
    pending=[]
    for did in order.loc[order.position>position+10,'dataset_id'].astype(int):
        origin,source=public_origin(names[did],names)
        if source in ids or canonical_family(origin) in canonical or conservative_family(origin) in semantic:
            continue
        if not (CACHE/relative/str(did)).exists():pending.append(int(did))
    log(f'Etapa 2: antecipação paralela de metadados públicos de {len(pending)} candidatos em cache separado; ordem de decisão inalterada.')
    def fetch(did):
        disk_guard()
        om.datasets.get_dataset(did,download_data=False,download_qualities=True,download_features_meta_data=True)
        source=staging/relative/str(did);target=CACHE/relative/str(did)
        assert all((source/name).is_file() for name in ('description.xml','features.xml','qualities.xml'))
        if not target.exists():
            try: source.rename(target)
            except OSError:
                if not target.exists():raise
        return did
    successes=[]; failures=[]
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures={pool.submit(fetch,did):did for did in pending}
        for future in as_completed(futures):
            try:successes.append(future.result())
            except Exception as error:failures.append(dict(dataset_id=futures[future],error=str(error)))
    (OUT/'metadata_prefetch.json').write_text(json.dumps(dict(
        metadata_only=True,decision_order_unchanged=True,completed=sorted(successes),failures=failures),indent=2))
    log(f'Etapa 2: antecipação concluída: {len(successes)} metadados; {len(failures)} falhas (seleção principal decide na ordem original).')

if __name__=='__main__':main()

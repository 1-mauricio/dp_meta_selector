"""Fetch the exact official HAR parquet by validated HTTP ranges into the cache."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import hashlib,json,shutil,time
import requests

URL='https://data.openml.org/datasets/0000/1478/dataset_1478.pq'
SIZE=47271336
ETAG='01732d8950c5136ab69e19a1c8c3a3e3'
CACHE=Path('.cache/openml/org/openml/www/datasets/1478')
OUT=Path('results/portfolio-984-nb8u')
CHUNK=1024*1024

def main():
    t=time.monotonic()
    blocks=CACHE/'https_blocks';blocks.mkdir(exist_ok=True)
    # MinIO writes complete one-MiB chunks. Keep its prefix and verify the MD5
    # of the entire assembled file against the official object's ETag.
    prefix=CACHE/f'dataset_1478.pq.{ETAG}.part.minio'
    if prefix.exists():
        with prefix.open('rb') as f:
            for i in range(prefix.stat().st_size//CHUNK):
                p=blocks/f'{i:03d}.part'
                if not p.exists():p.write_bytes(f.read(CHUNK))
                else:f.seek(CHUNK,1)
    def fetch(i):
        start=i*CHUNK;end=min(SIZE,(i+1)*CHUNK)-1
        p=blocks/f'{i:03d}.part'
        if p.exists() and p.stat().st_size==end-start+1:return i,0.
        began=time.monotonic()
        error=None
        for attempt in range(3):
            if shutil.disk_usage(CACHE).free<5*1024**3:raise RuntimeError('disk below 5 GiB')
            try:
                # Distinct range requests also avoid an intermediary returning a
                # cached full response. Never accept a response with wrong range.
                r=requests.get(URL,headers={'Range':f'bytes={start}-{end}'},timeout=(15,30))
                r.raise_for_status()
                if r.status_code!=206 or r.headers.get('Content-Range')!=f'bytes {start}-{end}/{SIZE}':
                    raise ValueError(f'invalid range response: {r.status_code} {r.headers.get("Content-Range")}')
                if len(r.content)!=end-start+1:raise ValueError('truncated range')
                tmp=p.with_suffix('.tmp');tmp.write_bytes(r.content);tmp.replace(p)
                return i,time.monotonic()-began
            except Exception as exc:error=exc
        raise error
    results=[]
    with ThreadPoolExecutor(max_workers=8) as pool:
        for future in as_completed([pool.submit(fetch,i) for i in range((SIZE+CHUNK-1)//CHUNK)]):
            result=future.result();results.append(result);print('BLOCK',result,flush=True)
    tmp=CACHE/'dataset_1478.assembled.pq'
    with tmp.open('wb') as f:
        for i in range((SIZE+CHUNK-1)//CHUNK):f.write((blocks/f'{i:03d}.part').read_bytes())
    md5=hashlib.md5(tmp.read_bytes()).hexdigest()
    if tmp.stat().st_size!=SIZE or md5!=ETAG:raise ValueError(f'assembled file MD5 mismatch: {md5}')
    import pandas as pd
    frame=pd.read_parquet(tmp)
    if len(frame)!=10299 or len(frame.columns)!=562:raise ValueError('HAR shape differs')
    tmp.replace(CACHE/'dataset_1478.pq')
    metadata=dict(url=URL,bytes=SIZE,etag_md5=ETAG,md5=md5,
                  sha256=hashlib.sha256((CACHE/'dataset_1478.pq').read_bytes()).hexdigest(),
                  seconds=time.monotonic()-t,range_workers=8,shape=list(frame.shape))
    (OUT/'har_download.json').write_text(json.dumps(metadata,indent=2))
    print(json.dumps(metadata),flush=True)

if __name__=='__main__':main()

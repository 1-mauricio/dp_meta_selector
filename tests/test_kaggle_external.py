"""Guard against protected-output mutations, target guessing and mirror leakage."""
import json
import numpy as np
import pandas as pd
import pytest
from scripts import external_kaggle_stage2 as k
from scripts.external_analysis import where_it_wins


def test_secret_guard_does_not_persist_credentials(monkeypatch):
    monkeypatch.setenv('KAGGLE_USERNAME','fixture-secret-user')
    monkeypatch.setenv('KAGGLE_KEY','fixture-secret-key')
    assert k.safe_bytes(b'public response')==b'public response'
    for content in [b'fixture-secret-user',b'fixture-secret-key']:
        with pytest.raises(ValueError,match='not persisted'):k.safe_bytes(content)


def test_rate_limit_is_retried_using_server_delay(monkeypatch):
    monkeypatch.setenv('KAGGLE_USERNAME','fixture-user')
    monkeypatch.setenv('KAGGLE_KEY','fixture-key')
    monkeypatch.setattr(k.uci,'disk_guard',lambda:10**10)
    monkeypatch.setattr(k,'log',lambda _:None)
    pauses=[];monkeypatch.setattr(k,'throttle',lambda cooldown=0:pauses.append(cooldown))
    class Response:
        def __init__(self,status):self.status_code=status;self.headers={'Retry-After':'90'}
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def iter_content(self,*args):return [b'{"public":true}']
    responses=iter([Response(429),Response(200)])
    monkeypatch.setattr(k.requests,'get',lambda *args,**kwargs:next(responses))
    assert k.fetch('/datasets/view/fixture/public')=={'public':True}
    assert 90. in pauses


def test_declared_repository_mirrors_and_origin_aliases_excluded():
    index={k.uci.compact(x):('manifest','name','breast cancer') for x in k.uci.forms('breast cancer')}
    for desc in ['Source: https://archive.ics.uci.edu/dataset/1/renamed',
                 'Derived from OpenML', 'Original: https://example.org/breast-cancer']:
        reason,_=k.overlap('unrelated-new-name',index,'',desc)
        assert reason.startswith('excluded_')
    assert k.overlap('unrelated-new-name',index,'','Original measurements by named author')==(None,None)


def test_spawn_worker_redirects_before_any_fit(monkeypatch,tmp_path):
    monkeypatch.setattr(k,'OUT',tmp_path/'new-kaggle')
    observed=[]
    monkeypatch.setattr(k,'_UCI_WORKER',lambda q,s,signature:observed.append((k.uci.OUT,k.uci.log)))
    # Restore shared module globals after the wrapper's deliberate redirection.
    monkeypatch.setattr(k.uci,'OUT',k.ROOT)
    monkeypatch.setattr(k.uci,'log',k.uci.log)
    k.worker(None,None,{})
    assert observed==[(tmp_path/'new-kaggle',k.log)]


def test_dtype_inference_reuses_uci_encoding_and_retains_id(tmp_path):
    from dpmeta.uci_external import prepare_uci
    from dpmeta.openml_benchmark import _get_openml_data
    frame=pd.DataFrame({'id':np.arange(204),'x':[0.,1.,np.nan,2.]*51,'category':['a','b',None,'a']*51,'outcome':[0,1,0,1]*51})
    meta=dict(uci_id=1001234,name='Kaggle fixture',tasks=['Classification'],num_instances=204,target_col=['outcome'],
        variables=[dict(name=c,role='Target' if c=='outcome' else 'Feature',type='') for c in frame])
    row=prepare_uci(meta,frame,tmp_path/'fixture.parquet');x,y,schema=_get_openml_data(None,pd.Series(row))
    assert row['dataset_id']==2001234 and schema.feature_names==('id','x','category')
    assert schema.feature_kinds==('numeric','numeric','nominal')
    assert len(x)==204 and np.isfinite(x).all() and x[1,0]==np.tanh(1)

"""Disk guard, independent accounting audit, and resume provenance checks."""
import json
from types import SimpleNamespace
import numpy as np
import pytest
from dpmeta.classifiers import fit_private_sgd_softmax
from scripts import parallel_sgd as r


def test_disk_threshold(monkeypatch):
    monkeypatch.setattr(r.shutil,'disk_usage',lambda _:SimpleNamespace(free=r.MIN_FREE-1))
    with pytest.raises(r.DiskLimit): r.disk_guard()
    monkeypatch.setattr(r.shutil,'disk_usage',lambda _:SimpleNamespace(free=r.MIN_FREE))
    assert r.disk_guard()==r.MIN_FREE


def test_independent_ledger_audit_rejects_changed_release():
    _,ledger=fit_private_sgd_softmax(np.zeros((8,2)),np.array([0,1]*4),2,
        epsilon=1.,delta=r.DELTA_TOTAL,n_steps=100,gradient_clip=2.,regularization=.0003,seed=10)
    row=dict(train_n=8,encoded_dim=2,class_count=2,epsilon=1.,epsilon_used=ledger.epsilon_used,
        delta_used=ledger.delta_used,ledger_json=json.dumps(ledger.as_dict()))
    assert r.audit_ledger(row)
    broken=ledger.as_dict(); broken['releases'][0]['noise_scale']*=2
    with pytest.raises(AssertionError): r.audit_ledger(dict(row,ledger_json=json.dumps(broken)))


def test_resume_rejects_changed_signature(tmp_path,monkeypatch):
    (tmp_path/'datasets').mkdir()
    (tmp_path/'datasets/1.json').write_text(json.dumps(dict(signature={'changed':True})))
    monkeypatch.setattr(r,'OUT',tmp_path)
    with pytest.raises(ValueError,match='resume signature changed'):
        r.load_commits({},None,{'changed':False})

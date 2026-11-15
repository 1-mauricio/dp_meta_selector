"""Frozen-rule boundaries, exact historical outputs/features, and H27 accounting."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dpmeta import classifiers as c
from dpmeta import portfolio_benchmark as p
from dpmeta.schema import PublicSchema

FIXTURES = Path(__file__).parent / 'fixtures'


def sample(monkeypatch):
    schema = PublicSchema(('x', 'z'), ('numeric', 'numeric'), ((), ()), ('no', 'yes'))
    x = np.random.default_rng(739).uniform(-1, 1, (48, 2))
    y = np.array([0]*32 + [1]*16)
    monkeypatch.setattr(p, '_get_openml_data', lambda *a, **kw: (x, y, schema))
    return x, y, pd.Series(dict(dataset_id=123, family_id='synthetic', n_instances=48))


@pytest.mark.parametrize('n,eps,expected', [(999,.1,.03), (1000,.1,.01),
    (9999,.1,.01), (10000,.1,.0003), (100000,1,.0003), (1000000,1,.0003)])
def test_public_rule_boundaries(n, eps, expected):
    assert p.sgd_regularization(p.LAMRULE_PIPELINE, n, eps) == expected
    assert p.sgd_regularization(p.LAM0003_PIPELINE, n, eps) == .0003


def test_rule_reads_json_instead_of_duplicating_policy(tmp_path, monkeypatch):
    rule = json.loads(p.PUBLIC_LAMBDA_RULE.read_text())
    rule['lambdas'] = [.7]*5
    path = tmp_path/'rule.json'; path.write_text(json.dumps(rule))
    monkeypatch.setattr(p, 'PUBLIC_LAMBDA_RULE', path)
    assert p.sgd_regularization(p.LAMRULE_PIPELINE, 48, .1) == .7


@pytest.mark.parametrize('epsilon', p.DEFAULT_EPSILONS)
def test_ledger_identical_h27(monkeypatch, epsilon):
    x,y,_ = sample(monkeypatch)
    ledgers = [c.fit_private_sgd_softmax(x,y,2,epsilon=epsilon,delta=p.DELTA_TOTAL,
        seed=19,n_steps=100,gradient_clip=2.,regularization=lam)[1].as_dict()
        for lam in (.03,.01,.0003)]
    assert ledgers[0] == ledgers[1] == ledgers[2]
    assert len(ledgers[0]['releases']) == 100


def test_thirteen_existing_outputs_exact(monkeypatch):
    _,_,row = sample(monkeypatch)
    actual = p.run_dataset(None,row,(.1,1.),2,20260917,
        include_research_features=False,pipelines=(*p.PIPELINES,p.NB8U_PIPELINE))
    assert actual == json.loads((FIXTURES/'portfolio_13_golden.json').read_text())


def test_existing_feature_vectors_exact_and_new_calibration():
    meta=pd.read_csv('data/openml_manifest_984_completed.csv',keep_default_na=False).iloc[0]
    private={int(meta.dataset_id):np.zeros(len(p.PRIVATE_META_NAMES))}
    dp={int(meta.dataset_id):np.zeros(len(p.DP_META_NAMES))}
    expected=json.loads((FIXTURES/'portfolio_features_golden.json').read_text())
    for mode in p.FEATURE_MODES:
        for pipeline in p.PIPELINES:
            actual=p._candidate_features(meta,1.,pipeline,mode,private,dp)
            np.testing.assert_array_equal(actual,expected[mode+'|'+pipeline])
        ref=p._candidate_features(meta,1.,p.H27_PIPELINE,mode,private,dp)
        for pipeline in p.SGD_NEW_PIPELINES:
            value=p._candidate_features(meta,1.,pipeline,mode,private,dp)
            assert value.shape == ref.shape
            changed=np.flatnonzero(value != ref)
            assert len(changed) <= (0 if mode=='public_schema_only' else 1)


def test_new_subset_no_legacy_and_public_total(monkeypatch):
    _,_,row = sample(monkeypatch)
    row.n_instances=1000  # At epsilon .1 total selects .01, train_n would select .03.
    monkeypatch.setattr(p,'_run_dataset_legacy',lambda *a,**kw: pytest.fail('legacy executed'))
    rows=p.run_dataset(None,row,(.1,),1,20260917,pipelines=p.SGD_NEW_PIPELINES,
        include_research_features=False,best_clean_by_condition={(.1,0):.8})
    assert [r['regularization'] for r in rows] == [.01,.0003]
    assert rows[0]['ledger_json'] == rows[1]['ledger_json']
    assert all(r['best_clean_portfolio_ba']==.8 for r in rows)


def test_h27_parameterized_path_exact(monkeypatch):
    _,_,row=sample(monkeypatch)
    expected=json.loads((FIXTURES/'portfolio_13_golden.json').read_text())
    reference={(r['epsilon'],r['repetition']):r['best_clean_portfolio_ba'] for r in expected}
    actual=p.run_dataset(None,row,(.1,1.),2,20260917,pipelines=(p.H27_PIPELINE,),
        include_research_features=False,best_clean_by_condition=reference)
    for a,b in zip(actual,[r for r in expected if r['pipeline']==p.H27_PIPELINE]):
        assert {k:a[k] for k in b} == b


def test_nb8u_previously_missing_calibration_has_same_width():
    meta=pd.read_csv('data/openml_manifest_984_completed.csv',keep_default_na=False).iloc[0]
    features=p._candidate_features(meta,1.,p.NB8U_PIPELINE,'public_plus_dp',{})
    assert len(features)==len(p.candidate_feature_names('public_plus_dp'))
    assert features[-1]==np.log1p(8)

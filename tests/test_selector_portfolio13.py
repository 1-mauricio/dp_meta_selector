"""Protocol regressions for the opt-in GBM-13 selector (historical defaults stay intact)."""
import numpy as np
import pandas as pd
import pytest
from dpmeta import selector_portfolio13 as sel, portfolio_benchmark as pb


class Scores:
    def __init__(self,scores):self.scores=np.asarray(scores,float)
    def predict(self,x):return self.scores.copy()
    def get_params(self):return {}


def test_fallback_uses_training_best_option_and_inclusive_tau():
    pipes=('old_nb','b8u','other')
    # Two blocks have different option orders. A 0.01 advantage must abstain;
    # a clear advantage must be retained. The residual zero reference is not fallback.
    x=np.vstack([np.eye(3),np.eye(3)[[2,0,1]]])
    model=sel.FallbackPredictor(Scores([-.2,0,.01,.02,-.2,0]),pipes,'b8u')
    out=model.predict(x).reshape(2,3)
    assert out.argmax(axis=1).tolist()==[1,0]


def test_fallback_rejects_incomplete_or_duplicated_portfolio():
    model=sel.FallbackPredictor(Scores([0,1]),('a','b'),'a')
    with pytest.raises(ValueError,match='exactly once'):model.predict(np.array([[1,0],[1,0]]))
    with pytest.raises(ValueError,match='complete'):model.predict(np.array([[1,0]]))


def test_family_weights_keep_regularization_scale():
    rows=[]
    for family,datasets in [('A',[1,2]),('B',[3])]:
        for d in datasets:
            for e in [.1,1.]:
                for p in ['a','b','c']:rows.append(dict(family_id=family,dataset_id=d,epsilon=e,pipeline=p))
    data=pd.DataFrame(rows);w=sel.weights(data)
    assert w.sum()==pytest.approx(2.)
    assert pd.Series(w).groupby(data.family_id).sum().tolist()==pytest.approx([1.,1.])
    assert pd.Series(w).groupby(data.dataset_id).sum().tolist()==pytest.approx([.5,.5,1.])


def test_feature_contract_matches_original_lodo_for_new_nb():
    m=pd.read_csv('data/openml_manifest_984_completed.csv').head(1)
    data=pd.DataFrame({'dataset_id':[int(m.dataset_id.iloc[0])]*2,'epsilon':[1.,1.],
                       'pipeline':[sel.NB,sel.NB8U]})
    pipes=tuple(sorted(data.pipeline));x=sel.feature_matrix(data,m,pipes)
    old=pb._candidate_features(m.iloc[0],1.,sel.NB,'public_plus_dp',{})
    np.testing.assert_array_equal(x[0,:len(old)],old)
    np.testing.assert_array_equal(x[1,:len(old)],old)
    assert x.shape[1]==len(sel.feature_names(pipes))
    np.testing.assert_array_equal(x[:,-2:],np.eye(2))
    assert len(pb.PIPELINES)==12 and sel.NB8U not in pb.PIPELINES


def test_semantic_holdout_manifest_and_development_missing_policy():
    m=pd.DataFrame({'dataset_id':[1,2,3],'family_id':['x','y','z'],
                    'semantic_family_id':['parent',np.nan,np.nan]})
    assert sel.semantic_manifest(m).family_id.tolist()==['parent','nan','nan']
    assert sel.semantic_manifest(m.drop(columns='semantic_family_id')).family_id.tolist()==['x','y','z']
    assert m.family_id.tolist()==['x','y','z']


def test_serialized_missing_family_is_not_dropped_from_inference():
    from io import StringIO
    from scripts.dissertation_analysis_portfolio13 import fam
    frame=pd.read_csv(StringIO('family_id,dataset_id,value\nnan,1,1.0\nnan,2,3.0\nparent,3,4.0\n'))
    result=fam(frame,'value')
    assert result.index.tolist()==['nan','parent']
    assert result.tolist()==[2.,4.]

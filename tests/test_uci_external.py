import copy
import numpy as np
import pandas as pd
import pytest
from dpmeta.uci_external import prepare_uci
from dpmeta.openml_benchmark import _get_openml_data
from dpmeta.schema import PublicSchema


def source():
    meta=dict(uci_id=999,name='Fixture',tasks=['Classification'],num_instances=204,
        target_col=['outcome'],variables=[dict(name='id',role='ID',type='Integer'),
        dict(name='x',role='Feature',type='Real'),dict(name='category',role='Feature',type='Categorical'),
        dict(name='outcome',role='Target',type='Integer')])
    frame=pd.DataFrame(dict(id=range(204),x=[0.,1.,np.inf,np.nan]*51,
        category=['a','b',None,'a']*51,outcome=[0,1,0,1]*51))
    return meta,frame


def test_same_openml_pipeline_and_missing_targets(tmp_path):
    meta,frame=source();frame.loc[0,'outcome']=np.nan
    row=prepare_uci(meta,frame,tmp_path/'fixture.parquet')
    x,y,schema=_get_openml_data(None,pd.Series(row))
    assert isinstance(schema,PublicSchema)
    assert schema.feature_names==('x','category') and schema.feature_kinds==('numeric','nominal')
    assert schema.nominal_domains==((),('a','b')) and schema.classes==('0.0','1.0')
    assert row['n_instances']==203 and row['removed_missing_targets']==1
    assert x.shape==(203,4) and np.isfinite(x).all() and len(y)==203
    assert x[0,0]==np.tanh(1.) and x[1,0]==0 and x[1,-1]==1


@pytest.mark.parametrize('change,reason',[
    (lambda m:m.update(target_col=['outcome','other']),'ambiguous'),
    (lambda m:m['variables'][1].update(type='Date'),'unsupported'),
    (lambda m:m.update(num_instances=205),'row_count'),
    (lambda m:m.update(tasks=['Regression']),'classification')])
def test_invalid_metadata_rejected(tmp_path,change,reason):
    meta,frame=source();change(meta)
    with pytest.raises(ValueError,match=reason):prepare_uci(meta,frame,tmp_path/'invalid.parquet')
    assert not (tmp_path/'invalid.parquet').exists()


def test_encoded_dimension_and_class_eligibility(tmp_path):
    meta,frame=source();meta['variables'][1]['type']='Categorical'
    frame['x']=[str(i) for i in range(204)];frame['category']=[str(i) for i in range(204)]
    # Three high-cardinality nominal features exceed encoded dimension 512.
    meta['variables'].append(dict(name='extra',role='Feature',type='Categorical'))
    frame['extra']=frame['x']
    with pytest.raises(ValueError,match='encoded_features'):prepare_uci(meta,frame,tmp_path/'a.parquet')
    meta,frame=source();frame['outcome']=range(204)
    with pytest.raises(ValueError,match='classes'):prepare_uci(meta,frame,tmp_path/'b.parquet')

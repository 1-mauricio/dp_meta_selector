"""Exercise source packaging, without network or benchmark outcomes."""
import io
import zipfile
import pandas as pd
import pytest
from scripts import external_uci_stage1 as stage


def zipped(files):
    buffer=io.BytesIO()
    with zipfile.ZipFile(buffer,'w') as archive:
        for name,content in files.items():archive.writestr(name,content)
    return buffer.getvalue()


def setup(monkeypatch,tmp_path,payload):
    monkeypatch.setattr(stage,'OUT',tmp_path)
    monkeypatch.setattr(stage,'download',lambda url,path:payload)
    monkeypatch.setattr(stage,'log',lambda message:None)
    return dict(uci_id=1,repository_url='https://archive.ics.uci.edu/dataset/1/fixture',
        variables=[dict(name='x'),dict(name='target')])


def test_nested_archive_and_delimiter_preserve_source_values(monkeypatch,tmp_path):
    payload=zipped({'outer.zip':zipped({'table.csv':'x;target\n1;a\n2;b\n'})})
    meta=setup(monkeypatch,tmp_path,payload)
    pd.testing.assert_frame_equal(stage.source_frame(meta),pd.DataFrame({'x':[1,2],'target':['a','b']}))


def test_no_arbitrary_file_or_variant_choice(monkeypatch,tmp_path):
    payload=zipped({'train.csv':'x,target\n1,a\n','test.csv':'x,target\n2,b\n'})
    meta=setup(monkeypatch,tmp_path,payload)
    with pytest.raises(ValueError,match='no_unique_archive_table'):stage.source_frame(meta)


def test_generic_catalog_name_does_not_exclude_unrelated_source():
    assert stage.overlap(dict(uci_id=99999,name='A new source dataset'),{'dataset':('OpenML','1','Dataset')},'')==(None,None)
    reason,evidence=stage.overlap(dict(uci_id=59,name='Letter Recognition'),{'letter':('OpenML','6','letter')},'')
    assert reason=='excluded_name_or_family' and evidence['alias']=='letter'

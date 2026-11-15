"""Independent Phase-C invariants; read-only outside isolated validation output."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import hashlib,json,re
import numpy as np,pandas as pd,joblib
from dpmeta import dissertation_selector as v
from scripts import analyze_dissertation_extras as x
from dpmeta.openml_benchmark import _load_manifest
OUT=Path('results/dissertation');ROOT=Path.cwd();PAPERS=ROOT.parent/'mestrado-papers'
checks={}
def check(key,value):
    checks[key]=bool(value)
    if not value:raise AssertionError(key)
def main():
    c=pd.read_csv('results/portfolio-984-sgd/conditions_15.csv');a,m,pipes=v.prepare(c,_load_manifest('data/openml_manifest_984_completed.csv'))
    check('350280_evaluations',len(c)==973*15*8*3);check('667_semantic_groups',m.family_id.nunique()==667)
    p=pd.read_csv(OUT/'outer_predictions.csv');picks=pd.read_csv('results/experiments/lodo_gbm15/lodo_picks.csv');picks=picks[(picks.variant=='GBM residual')&(picks.fallback=='sim')]
    check('exact_source_LODO_picks',p.set_index(['dataset_id','epsilon']).selector_pipeline.sort_index().equals(picks.set_index(['dataset_id','epsilon']).pipeline.sort_index().rename('selector_pipeline')))
    check('LODO_fallback_b8u',p.best_fixed_pipeline.eq(v.NB8U).all());check('tau_002',v.TAU==.02)
    test=v.ordered(a[a.dataset_id.isin(m.dataset_id.iloc[:3])],pipes);x.EXPANDED=True;x.PIPES=pipes
    matrix=v.feature_matrix(test,m,pipes);check('99_features',matrix.shape[1]==99);check('FeatureBuilder_exact',np.array_equal(matrix,x.FeatureBuilder(m).matrix(test)))
    # Reproduce the independent frozen holdout recommendations with this adapter's features.
    artifact=joblib.load('results/holdout-openml/models/gbm13_1.joblib')
    hc=pd.read_csv('results/holdout-openml/confirmatory/conditions.csv')
    pred=pd.read_csv('results/holdout-openml/confirmatory/predictions.csv')
    meta=pd.read_csv('results/holdout-openml/confirmatory/manifest.csv') if Path('results/holdout-openml/confirmatory/manifest.csv').exists() else pd.read_csv('data/openml_confirmatory_manifest.csv')
    ha,hm,hp=v.prepare(hc,meta);ha=v.ordered(ha,hp)
    model=v.FallbackPredictor(artifact['model'],artifact['pipes'],v.NB8U,.02)
    chosen=v.pick(ha,model.predict(v.feature_matrix(ha,hm,hp)))
    check('frozen_holdout_recommendations_exact',chosen.set_index(['dataset_id','epsilon']).pipeline.sort_index().equals(pred.set_index(['dataset_id','epsilon']).gbm15_pipeline.sort_index().rename('pipeline')))
    # Feature names refer to 17 public schema + 55 calibration + 12 legacy + 15 own actions.
    check('feature_names_match',len(v.feature_names(pipes))==99)
    expected={p.name for p in Path('figures/GBM-13').glob('fig*.pdf')}
    for folder in ['latex-dissertacao','artigo','latex-problema-pesquisa']:
        check(folder+'_all_14_figures',all((PAPERS/folder/'figures'/name).exists() for name in expected))
        for name in expected:check(folder+'_'+name+'_hash',hashlib.sha256((PAPERS/folder/'figures'/name).read_bytes()).hexdigest()==hashlib.sha256((Path('figures/dissertation')/name).read_bytes()).hexdigest())
    before=json.loads((OUT/'protected_fingerprints.json').read_text());changed=[]
    for path,digest in before.items():
        p=Path(path)
        if not p.exists() or hashlib.sha256(p.read_bytes()).hexdigest()!=digest:changed.append(path)
    check('protected_files_unchanged',not changed)
    check('protocol_hash_unchanged',hashlib.sha256(Path('protocols/CONFIRMATORY_PROTOCOL_FINAL.md').read_bytes()).hexdigest()=='66933d771548a8048b38c5952dbc4a9e1675c31694421c0b39d5063af5dfc183')
    checks['protected_files_count']=len(before)
    (OUT/'validation.json').write_text(json.dumps(checks,indent=2)+'\n');print(json.dumps(checks,indent=2))
if __name__=='__main__':main()

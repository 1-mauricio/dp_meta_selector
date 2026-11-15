"""Finish the public-family replacement, retaining exact unchanged E7 fits."""
import json
import shutil
import time
import pandas as pd
from experiments.e6_e7 import OUT, EPS7, run_dataset, save_json


def main():
    # The provisional runner has already received its original payloads.
    # Wait for its final output before consolidating the corrected sample.
    while True:
        timing_path=OUT/'E7_timing.json'
        if timing_path.exists() and (OUT/'E7_runs.csv').exists():
            timing=json.loads(timing_path.read_text())
            if len(timing)==55:
                break
        time.sleep(5)
    assert all(r['complete'] for r in timing),timing
    archive=OUT/'provisional_before_alias_correction'
    archive.mkdir(exist_ok=True)
    for name in ['E7_runs.csv','E7_timing.json']:
        shutil.copyfile(OUT/name,archive/name)
    old=pd.read_csv(OUT/'provisional_predictions_before_alias_correction.csv')
    new=pd.read_csv(OUT/'E6_lodo_predictions.csv')
    retained=old.merge(new,on=['dataset_id','epsilon'],suffixes=('_old','_new'))
    changed=retained[retained.epsilon.isin(EPS7)&retained.selected_lambda_old.ne(retained.selected_lambda_new)]
    assert not len(changed),changed
    predictions={(int(r.dataset_id),float(r.epsilon)):float(r.selected_lambda) for r in new.itertuples()}
    sample=pd.read_csv(OUT/'sample_E6_E7.csv',keep_default_na=False)
    row=sample[sample.dataset_id.eq(46468)].iloc[0]
    print('START replacement E7 dataset 46468; retained LODO choices all identical',flush=True)
    result=run_dataset((row.to_dict(),'E7',predictions,[(t,c) for t in [100,200,400] for c in [.5,1.,2.,4.]]))
    print(json.dumps(result),flush=True)
    assert result['complete'],result
    frames=[pd.read_csv(OUT/'parts'/f'E7_{int(i)}.csv') for i in sample.dataset_id]
    pd.concat(frames,ignore_index=True).to_csv(OUT/'E7_runs.csv',index=False)
    timing=[r for r in timing if r['dataset_id']!=46173]+[result]
    save_json('E7_timing.json',timing)
    save_json('alias_correction_verification.json',dict(passed=True,retained_datasets=54,
        retained_E7_lodo_lambda_cells_checked=len(retained[retained.epsilon.isin(EPS7)]),
        changed_retained_cells=0,replacement_dataset_id=46468,
        inference_excludes_dataset_id=46173,selection_uses_public_information_only=True))


if __name__=='__main__': main()

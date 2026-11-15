"""Input validation must survive aggregation of otherwise complete run commits."""
import json

import pandas as pd

from dpmeta.portfolio_benchmark import NB8U_PIPELINE, PIPELINES
from scripts import parallel_nb8u as runner


def test_alignment_rejection_cannot_reenter_through_a_complete_commit(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, 'OUT', tmp_path)
    (tmp_path / 'datasets').mkdir()
    (tmp_path / 'failure_details').mkdir()
    base = dict(epsilon=1., repetition=0, private_ba=.4, clean_ba=.5,
                best_clean_portfolio_ba=.5, pipeline_privacy_penalty=.1, portfolio_regret=.1)
    old = pd.DataFrame([dict(base, dataset_id=d, pipeline=p)
                        for d in (1, 2) for p in PIPELINES])
    manifest = pd.DataFrame({'dataset_id': [1, 2]})
    monkeypatch.setattr(runner, 'inputs', lambda: ({'epsilons': [1.], 'repeats': 1}, manifest, old))
    for d in (1, 2):
        row = dict(base, dataset_id=d, pipeline=NB8U_PIPELINE, ledger_json='{}',
                   split_seed=3, dp_seed=4, train_n=8, test_n=4, encoded_dim=1,
                   epsilon_used=1., delta_used=0., ledger_identical_to_current_nb=True)
        (tmp_path / 'datasets' / f'{d}.json').write_text(json.dumps(dict(
            dataset_id=d, seconds=1., rows=[row])))
    (tmp_path / 'validation_rejections.json').write_text(json.dumps({'2': {'error': 'input mismatch'}}))
    (tmp_path / 'failure_details' / '2.json').write_text(json.dumps(dict(
        dataset_id=2, dataset_name='rejected', error_type='AlignmentError',
        error='input mismatch', traceback='validation', seconds=1.)))
    runner.aggregate()
    new = pd.read_csv(tmp_path / 'conditions_new_option.csv')
    combined = pd.read_csv(tmp_path / 'conditions_13.csv')
    failures = pd.read_csv(tmp_path / 'failures.csv')
    assert set(new.dataset_id) == set(combined.dataset_id) == {1}
    assert len(combined) == 13
    assert set(failures.dataset_id) == {2}
    assert runner.rejected_dataset_ids() == {2}
    assert (tmp_path / 'datasets/2.json').exists()  # Complete raw grid is insufficient.
    coverage = json.loads((tmp_path / 'coverage.json').read_text())
    assert coverage['completed'] == 1 and coverage['failed'] == 1
    assert coverage['unattempted_ids'] == []

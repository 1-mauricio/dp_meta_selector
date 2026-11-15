"""Regression and privacy tests for the parameterized E4 NB option."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dpmeta import classifiers as c
from dpmeta import portfolio_benchmark as p
from dpmeta.schema import PublicSchema


@pytest.fixture
def sample():
    schema = PublicSchema(('x', 'z'), ('numeric', 'numeric'), ((), ()), ('no', 'yes'))
    x = np.random.default_rng(739).uniform(-1, 1, (48, 2))
    return x, np.array([0] * 32 + [1] * 16), schema


@pytest.mark.parametrize('epsilon', (0.1, 0.25, 0.5, 1., 2., 4., 8., 16.))
def test_ledger_identical_and_prior_query_retained(sample, epsilon):
    x, y, schema = sample
    current, _, ledger = c.fit_private_categorical_nb(x, y, 2, schema, epsilon=epsilon, seed=17)
    new, _, new_ledger = c.fit_private_categorical_nb(
        x, y, 2, schema, epsilon=epsilon, seed=17, numeric_bins=8, uniform_prior=True)
    assert ledger.as_dict() == new_ledger.as_dict()
    assert new_ledger.as_dict()['releases'][0]['name'] == 'class_prior_histogram'
    np.testing.assert_array_equal(new.class_counts, [1., 1.])
    assert current.feature_cardinalities == (16, 16)
    assert new.feature_cardinalities == (8, 8)


def test_uniform_prior_changes_only_decision_prior(sample):
    x, y, schema = sample
    current, categories, ledger = c.fit_private_categorical_nb(
        x, y, 2, schema, epsilon=1, seed=77, numeric_bins=8)
    uniform, _, uniform_ledger = c.fit_private_categorical_nb(
        x, y, 2, schema, epsilon=1, seed=77, numeric_bins=8, uniform_prior=True)
    for a, b in zip(current.feature_counts, uniform.feature_counts):
        np.testing.assert_array_equal(a, b)
    wrapper = c.CategoricalNaiveBayesModel(np.ones(2), current.feature_counts, (8, 8))
    np.testing.assert_array_equal(uniform.predict(categories), wrapper.predict(categories))
    assert ledger.as_dict() == uniform_ledger.as_dict()
    # Unequal priors override weak likelihood evidence; a uniform prior does not.
    empirical = c.CategoricalNaiveBayesModel(np.array([99., 1.]), (np.array([[6., 4.], [1., 9.]]),), (2,))
    equal = c.CategoricalNaiveBayesModel(np.ones(2), empirical.feature_counts, (2,))
    np.testing.assert_array_equal(empirical.predict([[1]]), [0])
    np.testing.assert_array_equal(equal.predict([[1]]), [1])


def test_bins_boundaries_and_clean_prior():
    schema = PublicSchema(('x',), ('numeric',), ((),), ('a', 'b'))
    x = np.linspace(-1, 1, 9).reshape(-1, 1)
    categories, cards = c.categorical_nb_features(x, schema, numeric_bins=8)
    np.testing.assert_array_equal(categories.ravel(), [0, 1, 2, 3, 4, 5, 6, 7, 7])
    assert cards == (8,)
    stats = c.categorical_nb_statistics(x, np.array([0]*7 + [1]*2), 2, schema, numeric_bins=8)
    assert stats.feature_counts[0].shape == (2, 8)
    model, _ = c.fit_clean_categorical_nb(x, np.array([0]*7 + [1]*2), 2, schema,
                                        numeric_bins=8, uniform_prior=True)
    np.testing.assert_array_equal(model.class_counts, np.ones(2))


def test_original_twelve_outputs_exactly_unchanged(sample, monkeypatch):
    x, y, schema = sample
    monkeypatch.setattr(p, '_get_openml_data', lambda *args, **kwargs: (x, y, schema))
    row = pd.Series(dict(dataset_id=123, family_id='synthetic'))
    actual = p.run_dataset(None, row, (.1, 1.), 2, 20260917, include_research_features=False)
    expected = json.loads((Path(__file__).parent / 'fixtures/portfolio12_golden.json').read_text())
    assert actual == expected
    nb = p.run_dataset(None, row, (.1, 1.), 2, 20260917,
                       include_research_features=False, pipelines=(p.PIPELINES[10],))
    expected_nb = [r for r in expected if r['pipeline'] == p.PIPELINES[10]]
    for a, b in zip(nb, expected_nb):
        for key in ('private_ba', 'clean_ba', 'ledger_json'):
            assert a[key] == b[key]


def test_subset_runs_only_new_and_uses_original_reference(sample, monkeypatch):
    x, y, schema = sample
    monkeypatch.setattr(p, '_get_openml_data', lambda *a, **kw: (x, y, schema))
    def forbidden(*a, **kw):
        raise AssertionError('unselected portfolio executed')
    monkeypatch.setattr(p, '_run_dataset_legacy', forbidden)
    row = pd.Series(dict(dataset_id=123, family_id='synthetic'))
    rows = p.run_dataset(None, row, (1.,), 1, 20260917, include_research_features=False,
                         pipelines=(p.NB8U_PIPELINE,), best_clean_by_condition={(1., 0): 83.})
    assert len(rows) == 1 and rows[0]['pipeline'] == p.NB8U_PIPELINE
    assert rows[0]['best_clean_portfolio_ba'] == 83.
    assert rows[0]['portfolio_regret'] == 83. - rows[0]['private_ba']
    assert len(p.PIPELINES) == 12 and len(p.PIPELINE_REGISTRY) == 15

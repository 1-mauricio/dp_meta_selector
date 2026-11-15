import numpy as np
import pytest

from dpmeta.private_hpo import (
    fit_private_softmax_hpo,
    fit_private_softmax_hpo_profile,
    poisson_split,
)


def _toy_data():
    x = np.asarray([
        [-1.0, -1.0], [-0.8, -0.4], [-0.4, -0.7], [0.2, 0.3],
        [0.6, 0.7], [1.0, 1.0], [-0.2, 0.4], [0.8, -0.1],
    ])
    y = np.asarray([0, 0, 0, 1, 1, 1, 0, 1])
    return x, y


def test_poisson_split_is_disjoint_and_covers_rows():
    tuning, validation = poisson_split(100, 0.2, seed=12)
    assert len(tuning) > 0
    assert len(validation) > 0
    assert set(tuning).isdisjoint(validation)
    assert sorted(np.r_[tuning, validation].tolist()) == list(range(100))


def test_private_hpo_refit_is_reproducible_and_composed():
    x, y = _toy_data()
    first = fit_private_softmax_hpo(x, y, 2, epsilon=1.2, seed=19)
    second = fit_private_softmax_hpo(x, y, 2, epsilon=1.2, seed=19)
    assert first.selected_regularization == second.selected_regularization
    np.testing.assert_allclose(first.candidate_scores, second.candidate_scores)
    np.testing.assert_array_equal(first.model.predict(x), second.model.predict(x))
    assert first.ledger["epsilon_used"] == pytest.approx(1.2)
    assert first.ledger["delta_used"] == pytest.approx(1e-6)
    assert len(first.ledger["releases"]) == 5


def test_private_hpo_uses_full_refit_and_valid_budget():
    x, y = _toy_data()
    result = fit_private_softmax_hpo(x, y, 2, epsilon=0.5, seed=5)
    assert len(result.tuning_indices) + len(result.validation_indices) == len(x)
    assert result.epsilon_final_model == pytest.approx(0.4)
    assert result.epsilon_selection == pytest.approx(0.05)
    assert result.epsilon_tuning_models == pytest.approx(0.05)
    assert result.delta_final_model == pytest.approx(0.9e-6)
    assert np.isfinite(result.model.weights).all()


def test_private_hpo_profile_composes_exact_gaussian_releases():
    x, y = _toy_data()
    result = fit_private_softmax_hpo_profile(x, y, 2, epsilon=1.2, seed=19)
    assert result.ledger["accounting"] == "exact_gaussian_privacy_profile_plus_pure_dp"
    assert result.ledger["epsilon_used"] == pytest.approx(1.2)
    assert result.ledger["delta_used"] <= 1e-6 + 1e-12
    assert result.ledger["within_delta"] is True
    assert result.ledger["final_noise_scale"] > 0

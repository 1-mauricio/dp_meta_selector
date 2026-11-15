import numpy as np
import pytest

from dpmeta.privacy import ParallelPrivacyLedger, Release, Sensitivity
from scripts.private_validation_benchmark import exponential_select
from scripts.private_validation_parallel_benchmark import _compose_parallel_ledger


def _release(name, epsilon, delta):
    return Release(
        name=name, mechanism="test", epsilon=epsilon, delta=delta,
        sensitivity=Sensitivity(1.0, 1.0, 1.0), noise_scale=1.0,
    )


def test_parallel_ledger_uses_maximum_branch_cost_and_enforces_both_limits():
    ledger = ParallelPrivacyLedger(epsilon_limit=2.0, delta_limit=1e-6)
    ledger.record(_release("model", 2.0, 1e-6))
    ledger.record(_release("validation", 2.0, 0.0))
    assert ledger.epsilon_used == pytest.approx(2.0)
    assert ledger.delta_used == pytest.approx(1e-6)

    too_much_epsilon = ParallelPrivacyLedger(1.5, 1e-6)
    with pytest.raises(ValueError, match="epsilon"):
        too_much_epsilon.record(_release("model", 2.0, 0.0))

    too_much_delta = ParallelPrivacyLedger(2.0, 1e-7)
    with pytest.raises(ValueError, match="delta"):
        too_much_delta.record(_release("model", 1.0, 1e-6))


def test_private_validation_parallel_composition_covers_all_candidate_and_choice_branches():
    candidates = {
        "approximate": {"epsilon_used": 2.0, "delta_used": 1e-6},
        "pure": {"epsilon_used": 2.0, "delta_used": 0.0},
    }
    ledger = _compose_parallel_ledger(2.0, 1e-6, candidates, n_validation=25)
    assert ledger["accounting"] == "parallel_disjoint_record_partitions"
    assert ledger["epsilon_used"] == pytest.approx(2.0)
    assert ledger["delta_used"] == pytest.approx(1e-6)
    assert len(ledger["branches"]) == 3
    assert ledger["branches"][-1]["sensitivity"]["l1"] == pytest.approx(1 / 25)


def test_exponential_validation_choice_is_reproducible_and_favors_higher_accuracy():
    scores = np.array([0.1, 0.9])
    first = exponential_select(scores, epsilon=100.0, n_validation=20, seed=7)
    second = exponential_select(scores, epsilon=100.0, n_validation=20, seed=7)
    assert first == second == 1

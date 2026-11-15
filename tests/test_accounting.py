import pytest

from dpmeta.accounting import (
    analytic_gaussian_hpo_rdp_bound,
    gaussian_mu_for_delta,
    gaussian_privacy_profile_delta,
    compose_gaussian_rdp,
    gaussian_rdp,
)
from dpmeta.mechanisms import analytic_gaussian_sigma


def test_gaussian_rdp_formula():
    assert gaussian_rdp(2.0, 1.0, 1.0) == pytest.approx(1.0)


def test_rdp_composition_includes_pure_selection_term():
    bound = compose_gaussian_rdp(
        [(1.0, 2.0)], delta=1e-6, pure_dp_epsilon=0.3,
    )
    assert bound["epsilon"] > 0.3
    assert bound["pure_dp_epsilon"] == pytest.approx(0.3)
    assert bound["delta"] == pytest.approx(1e-6)


def test_hpo_rdp_audit_is_conservative_and_auditable():
    result = analytic_gaussian_hpo_rdp_bound(2.0, delta_total=1e-6)
    assert result["epsilon_total_limit"] == pytest.approx(2.0)
    assert result["epsilon_basic_composition"] == pytest.approx(2.0)
    assert result["optimal_order"] > 1.0
    assert result["orders_checked"] > 10


def test_gaussian_profile_inverse_matches_analytic_calibration():
    epsilon, delta = 1.2, 1e-6
    mu = gaussian_mu_for_delta(epsilon, delta)
    sigma = analytic_gaussian_sigma(epsilon, delta, 1.0)
    assert mu == pytest.approx(1.0 / sigma, rel=1e-9)
    assert gaussian_privacy_profile_delta(epsilon, mu) == pytest.approx(delta, rel=1e-7)

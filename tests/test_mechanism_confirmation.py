import pandas as pd

from dpmeta.families import conservative_family
from dpmeta.mechanism_confirmation import _evaluate, family_equal_best_fixed
from dpmeta.selector import ACTIONS


def test_family_equal_best_fixed_does_not_overweight_a_family():
    rows = []
    # Action A is better on the one-dataset family B, while action B wins on
    # the two highly correlated rows from family A. Equal-family aggregation
    # should still give each family exactly one vote.
    for family, dataset, a_score, b_score in (
        ("family_a", 1, 0.2, 0.0),
        ("family_a", 2, 0.2, 0.0),
        ("family_b", 3, 0.0, 0.3),
    ):
        rows.extend([
            {"family_id": family, "dataset_id": dataset, "epsilon": 0.5,
             "mechanism": "laplace", "allocation": "equal", "regret": a_score},
            {"family_id": family, "dataset_id": dataset, "epsilon": 0.5,
             "mechanism": "staircase", "allocation": "equal", "regret": b_score},
        ])
    result = family_equal_best_fixed(pd.DataFrame(rows))
    assert result == ("laplace", "equal")


def test_confirmatory_evaluation_scores_both_public_selectors():
    actions = pd.DataFrame([{
        "dataset_id": 1,
        "dataset_name": "toy",
        "family_id": "toy",
        "epsilon": 0.5,
        "fixed_action": "laplace__equal",
        "epsilon_fixed_action": "laplace__equal",
        "preselected_laplace_equal": "laplace__equal",
        "preselected_zcdp_equal": "gaussian_zcdp__equal",
        "selector_action_dp_aware": "gaussian_zcdp__equal",
        "selector_action_public": "staircase__equal",
    }])
    raw = pd.DataFrame([
        {"dataset_id": 1, "family_id": "toy", "epsilon": 0.5,
         "action": f"{mechanism}__{allocation}", "regret": float(index) / 10}
        for index, (mechanism, allocation) in enumerate(ACTIONS)
    ])
    summary, predictions = _evaluate(raw, actions, seed=7)
    assert predictions.loc[0, "selector_regret_dp_aware"] == 0.6
    assert predictions.loc[0, "selector_regret_public"] == 0.2
    assert "dp_aware" in summary and "public" in summary
    assert summary["dp_aware"]["selector"]["mean"] == 0.6
    assert summary["public"]["selector"]["mean"] == 0.2


def test_conservative_family_groups_aliases_from_prior_protocols():
    assert conservative_family("credit_approval_reproduced") == conservative_family("Australian")
    assert conservative_family("monks-problems-1") == conservative_family("monks-problems-2")
    assert conservative_family("fri_c2_250_25") == conservative_family("fri_c4_500_25")

import itertools

import numpy as np
import pandas as pd
import pytest
from scipy.special import ndtr
import dpmeta.portfolio_benchmark as portfolio

from dpmeta.mechanisms import analytic_gaussian_sigma, zcdp_rho_for_epsilon
from dpmeta.classifiers import (
    categorical_nb_statistics,
    encoded_numeric_mask,
    fit_clean_random_forest,
    fit_clean_softmax,
    fit_private_categorical_nb,
    fit_private_random_forest,
    fit_private_random_forest_exponential,
    fit_private_random_forest_gaussian,
    fit_private_sgd_softmax,
    make_random_forest_plan,
    privatize_softmax,
    random_forest_leaf_counts,
)
from dpmeta.external_confirmation import confirmation_family
from dpmeta.l2_metrics import relative_l2_regrets, sufficient_statistic_l2_distortion
from dpmeta.dp_diagnostics import (
    DP_DIAGNOSTIC_NAMES,
    dp_training_diagnostics,
    sufficient_statistic_l2_sensitivity as diagnostic_l2_sensitivity,
)
from dpmeta.dp_bounds import estimate_dp_bounds_frame, estimate_dp_numeric_bounds
from dpmeta.budgeted_confirmation import _checkpoint_is_complete, _tail_noninferiority_lcb
from dpmeta.figures import BASELINES, _inferential_methods, _regret_methods
from dpmeta.meta_features import (
    DP_META_COLUMNS,
    DP_CALIBRATION_FEATURE_NAMES,
    PRIVATE_META_COLUMNS,
    dp_calibration_features,
    dp_training_meta_features,
    private_training_meta_features,
)
from dpmeta.model import (
    SufficientStatistics, balanced_accuracy, fit_clean, fit_private,
    model_from_statistics, predict, privatize_sufficient_statistics,
    sufficient_statistics,
)
from dpmeta.openml_benchmark import stratified_holdout_indices
from dpmeta.portfolio_benchmark import (
    DEFAULT_EPSILONS,
    DP_FEATURE_WIDTH,
    DP_SGD_GRADIENT_CLIP,
    DP_SGD_REGULARIZATION,
    DP_SGD_STEPS,
    DP_SGD_STEPS_ALTERNATE,
    GNB_FEATURE_WIDTH,
    SOFTMAX_FEATURE_WIDTH,
    PIPELINES,
    PUBLIC_ONLY_FEATURE_MODES,
    complete_dataset_ids,
    validate_condition_grid,
    _candidate_features,
    _available_feature_modes,
    _mc_nemar,
    _public_dp_features,
    _rank_biserial,
    _safe_wilcoxon,
)
from dpmeta.privacy import sufficient_statistic_sensitivities
from dpmeta.randomness import SystemRandomGenerator
from dpmeta.schema import PublicSchema, public_features
from dpmeta.selector import ACTIONS, evaluate_grouped


def _stat_vector(rows, labels, n_classes):
    p = len(rows[0])
    counts = np.zeros(n_classes)
    sums = np.zeros((n_classes, p))
    seconds = np.zeros((n_classes, p))
    for row, label in zip(rows, labels):
        counts[label] += 1
        sums[label] += row
        seconds[label] += np.square(row)
    return np.r_[counts, sums.ravel(), seconds.ravel()]


def test_declared_sensitivities_cover_all_small_replace_one_neighbors():
    p, n_classes = 2, 2
    sensitivities = sufficient_statistic_sensitivities(p)
    records = list(itertools.product((-1.0, 1.0), repeat=p))
    queries = []
    for record, label in itertools.product(records, range(n_classes)):
        queries.append(_stat_vector([record], [label], n_classes))
    for left, right in itertools.combinations(queries, 2):
        difference = np.abs(left - right)
        # Neighbors here are exactly one-row replace-one datasets.
        assert difference.sum() <= sum(s.l1 for s in sensitivities.values())
        assert np.linalg.norm(difference) <= sum(s.l2 for s in sensitivities.values())
        assert difference.max() <= max(s.linf for s in sensitivities.values())

    # Also test each individual query with neighbors from exhaustive row values.
    one_row_stats = {
        "counts": [], "sums": [], "seconds": [],
    }
    for record, label in itertools.product(records, range(n_classes)):
        stats = sufficient_statistics(np.asarray([record]), np.asarray([label]), n_classes)
        one_row_stats["counts"].append(stats.counts.ravel())
        one_row_stats["sums"].append(stats.sums.ravel())
        one_row_stats["seconds"].append(stats.seconds.ravel())
    for name, vectors in one_row_stats.items():
        sensitivity = sensitivities[name]
        for left, right in itertools.combinations(vectors, 2):
            diff = np.abs(left - right)
            assert diff.sum() <= sensitivity.l1 + 1e-12
            assert np.linalg.norm(diff) <= sensitivity.l2 + 1e-12
            assert diff.max() <= sensitivity.linf + 1e-12


def test_analytic_gaussian_calibration_meets_requested_delta():
    epsilon, delta, sensitivity = 1.2, 1e-6, 3.5
    sigma = analytic_gaussian_sigma(epsilon, delta, sensitivity)
    mu = sensitivity / sigma
    achieved = ndtr(mu / 2 - epsilon / mu) - np.exp(epsilon) * ndtr(-mu / 2 - epsilon / mu)
    assert achieved <= delta * (1 + 1e-7)
    assert achieved >= delta * (1 - 1e-6)


def test_budgeted_dp_diagnostics_are_finite_reproducible_and_accounted():
    x = np.array([[-1.0, 0.0], [-0.4, 0.3], [0.7, 0.8], [1.0, -0.5]])
    y = np.array([0, 0, 1, 1])
    features, ledger = dp_training_diagnostics(
        x, y, 2, epsilon=0.4, delta=5e-7, seed=43,
    )
    again, second_ledger = dp_training_diagnostics(
        x, y, 2, epsilon=0.4, delta=5e-7, seed=43,
    )
    assert len(features) == len(DP_DIAGNOSTIC_NAMES) == 13
    assert np.isfinite(features).all()
    np.testing.assert_array_equal(features, again)
    assert ledger["epsilon_used"] == pytest.approx(0.4)
    assert ledger["delta_used"] == pytest.approx(5e-7)
    assert second_ledger["releases"] == ledger["releases"]
    assert ledger["releases"][0]["sensitivity"]["l2"] == pytest.approx(
        diagnostic_l2_sensitivity(x.shape[1])
    )


def test_budgeted_gaussian_nb_delta_override_respects_remaining_budget():
    x = np.array([[-1.0, 0.0], [-0.4, 0.3], [0.7, 0.8], [1.0, -0.5]])
    y = np.array([0, 0, 1, 1])
    _, ledger = fit_private(
        x, y, 2, epsilon=0.8, delta=5e-7,
        mechanism="gaussian_zcdp", allocation="equal", seed=44,
    )
    assert ledger.epsilon_used == pytest.approx(0.8)
    assert ledger.delta_used == pytest.approx(5e-7)
    assert ledger.rho_used == pytest.approx(zcdp_rho_for_epsilon(0.8, 5e-7))


def test_joint_dp_diagnostic_sensitivity_covers_small_replace_one_neighbors():
    p, n_classes = 2, 2
    vectors = []
    for record in itertools.product((-1.0, 0.0, 1.0), repeat=p):
        for klass in range(n_classes):
            counts = np.zeros(n_classes)
            sums = np.zeros((n_classes, p))
            seconds = np.zeros((n_classes, p))
            counts[klass] = 1.0
            sums[klass] = record
            seconds[klass] = np.square(record)
            vectors.append(np.r_[counts, sums.ravel(), seconds.ravel()])
    bound = diagnostic_l2_sensitivity(p)
    for left, right in itertools.combinations(vectors, 2):
        assert np.linalg.norm(left - right) <= bound + 1e-12


def test_tail_noninferiority_uses_paired_family_bootstrap():
    baseline = np.linspace(0.0, 1.0, 50)
    selector_better = baseline - 0.1
    selector_worse = baseline + 0.1
    better = _tail_noninferiority_lcb(selector_better, baseline, seed=7, draws=1000)
    worse = _tail_noninferiority_lcb(selector_worse, baseline, seed=7, draws=1000)
    assert all(value > 0 for value in better["lower_95_one_sided"].values())
    assert all(value < 0 for value in worse["lower_95_one_sided"].values())


def test_analysis_resume_requires_every_dataset_checkpoint_complete(tmp_path):
    path = tmp_path / "conditions.csv"
    pd.DataFrame({"dataset_id": [1, 1, 2, 2]}).to_csv(path, index=False)
    assert _checkpoint_is_complete(path, {1, 2}, expected_rows_per_dataset=2)
    assert not _checkpoint_is_complete(path, {1, 2, 3}, expected_rows_per_dataset=2)
    assert not _checkpoint_is_complete(path, {1, 2}, expected_rows_per_dataset=3)


def test_private_fit_composes_all_mechanisms_within_budget():
    x = np.array([[-1.0, 0.0], [0.2, 1.0], [0.7, -0.3], [1.0, 0.5]])
    y = np.array([0, 0, 1, 1])
    for mechanism in ("laplace", "staircase", "analytic_gaussian", "gaussian_zcdp"):
        model, ledger = fit_private(
            x, y, 2, epsilon=2.0, mechanism=mechanism,
            allocation="count_20_sum_40_second_40", seed=44,
        )
        assert len(ledger.releases) == 3
        assert ledger.epsilon_used == pytest.approx(2.0)
        expected_delta = 1e-6 if mechanism in {"analytic_gaussian", "gaussian_zcdp"} else 0
        assert ledger.delta_used == pytest.approx(expected_delta)
        assert np.isfinite(model.means).all()
        assert np.all(model.variances >= 1e-6)
        assert np.isclose(model.priors.sum(), 1.0)
        if mechanism == "gaussian_zcdp":
            assert ledger.accounting == "zcdp"
            assert ledger.rho_used == pytest.approx(zcdp_rho_for_epsilon(2.0, 1e-6))
        else:
            assert ledger.accounting == "basic"


def test_private_statistics_helper_matches_private_model_release():
    x = np.array([[-1.0, 0.0], [0.2, 1.0], [0.7, -0.3], [1.0, 0.5]])
    y = np.array([0, 0, 1, 1])
    clean_stats = sufficient_statistics(x, y, 2)
    released_stats, ledger = privatize_sufficient_statistics(
        clean_stats, epsilon=1.3, mechanism="laplace", allocation="equal", seed=92,
    )
    private_model, model_ledger = fit_private(
        x, y, 2, epsilon=1.3, mechanism="laplace", allocation="equal", seed=92,
    )
    np.testing.assert_allclose(
        predict(x, model_from_statistics(released_stats)), predict(x, private_model),
    )
    assert ledger.as_dict() == model_ledger.as_dict()


def test_sufficient_statistic_l2_distortion_is_public_sensitivity_scaled():
    clean = SufficientStatistics(np.zeros(2), np.zeros((2, 2)), np.zeros((2, 2)))
    released = SufficientStatistics(
        np.array([np.sqrt(2), 0]), np.array([[2.0, 0], [0, 0]]),
        np.array([[2.0, 0], [0, 0]]),
    )
    expected = np.sqrt((1.0**2 + (1 / np.sqrt(2))**2 + 1.0**2) / 3)
    assert sufficient_statistic_l2_distortion(clean, released) == pytest.approx(expected)


def test_relative_l2_regret_uses_best_action_and_rejects_undefined_cases():
    np.testing.assert_allclose(relative_l2_regrets(np.array([2.0, 1.0, 4.0])), [2.0, 1.0, 4.0])
    np.testing.assert_array_equal(relative_l2_regrets(np.zeros(3)), np.ones(3))
    with pytest.raises(ValueError, match="indefinido"):
        relative_l2_regrets(np.array([0.0, 1.0]))


def test_schema_uses_only_declared_domains_and_public_descriptors():
    schema = PublicSchema(
        ("numeric", "color"), ("numeric", "nominal"), ((), ("blue", "red")),
        ("no", "yes"),
    )
    frame = pd.DataFrame({"numeric": [-2.0, np.nan], "color": ["red", None]})
    encoded = schema.encode(frame)
    assert encoded.shape == (2, 4)
    assert np.isfinite(encoded).all()
    features = public_features(100, schema, 1.0)
    assert features.shape == (17,)
    with pytest.raises(ValueError, match="fora do schema"):
        schema.encode(pd.DataFrame({"numeric": [0.0], "color": ["green"]}))


def test_asinh10_numeric_encoding_is_fixed_monotone_and_bounded():
    schema = PublicSchema(("x",), ("numeric",), ((),), ("no", "yes"))
    values = np.array([-1e6, -10.0, -2.0, -1.0, 0.0, 1.0, 2.0, 10.0, 1e6, np.nan])
    encoded = schema.encode(pd.DataFrame({"x": values}), numeric_transform="asinh10")[:, 0]

    assert np.isfinite(encoded).all()
    assert np.all((-1.0 <= encoded) & (encoded <= 1.0))
    assert np.all(np.diff(encoded[:-1]) >= 0)
    assert encoded[4] == encoded[-1] == 0.0
    assert encoded[6] < 0.2  # tanh(2) would already be close to saturation.
    with pytest.raises(ValueError, match="numeric_transform"):
        schema.encode(pd.DataFrame({"x": [0.0]}), numeric_transform="minmax")


def test_public_bounds_encoding_requires_declared_bounds_and_clips_only_outliers():
    frame = pd.DataFrame({"value": [-2.0, 0.0, 2.0, 4.0, np.nan],
                          "kind": ["a", "b", None, "a", "b"]})
    schema = PublicSchema(
        ("value", "kind"), ("numeric", "nominal"), ((), ("a", "b")),
        ("no", "yes"), ((-2.0, 2.0), None),
    )
    encoded = schema.encode(frame, numeric_transform="public_bounds")
    np.testing.assert_allclose(encoded[:, 0], [-1.0, 0.0, 1.0, 1.0, 0.0])
    assert np.isfinite(encoded).all()
    assert np.all((-1.0 <= encoded) & (encoded <= 1.0))
    features = public_features(5, schema, 1.0)
    assert features[-2] == 1.0
    assert features[-1] == pytest.approx(np.log1p(4.0))

    without_bounds = PublicSchema(("x",), ("numeric",), ((),), ("no", "yes"))
    with pytest.raises(ValueError, match="limites explícitos"):
        without_bounds.encode(pd.DataFrame({"x": [0.0]}), numeric_transform="public_bounds")


def test_full_batch_dp_sgd_softmax_composes_zcdp_steps_and_is_reproducible():
    x = np.array([[-1.0, 0.0], [-0.5, 0.5], [0.5, -0.5], [1.0, 0.0]])
    y = np.array([0, 0, 1, 1])
    model, ledger = fit_private_sgd_softmax(
        x, y, 2, epsilon=2.0, delta=1e-6, seed=314, n_steps=5,
    )
    repeated, repeated_ledger = fit_private_sgd_softmax(
        x, y, 2, epsilon=2.0, delta=1e-6, seed=314, n_steps=5,
    )
    assert np.isfinite(model.weights).all()
    assert model.predict(x).shape == y.shape
    np.testing.assert_allclose(model.weights, repeated.weights)
    assert ledger.epsilon_used == pytest.approx(2.0)
    assert ledger.delta_used == pytest.approx(1e-6)
    assert repeated_ledger.as_dict() == ledger.as_dict()
    with pytest.raises(ValueError, match="domínio público"):
        fit_private_sgd_softmax(x * 2.0, y, 2, epsilon=2.0, delta=1e-6,
                                seed=314, n_steps=5)


def test_openml_boolean_targets_match_declared_nominal_schema_case_insensitively():
    schema = PublicSchema(("x",), ("numeric",), ((),), ("FALSE", "TRUE"))
    np.testing.assert_array_equal(schema.class_indices([False, True]), [0, 1])


def test_case_sensitive_nominal_values_are_kept_distinct():
    schema = PublicSchema(("letter",), ("nominal",), (("A", "a"),), ("no", "yes"))
    encoded = schema.encode(pd.DataFrame({"letter": ["A", "a"]}))
    np.testing.assert_array_equal(encoded[0], [1, 0, 0])
    np.testing.assert_array_equal(encoded[1], [0, 1, 0])


def test_singleton_class_stays_in_training_split():
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2])
    train, test = stratified_holdout_indices(labels, seed=123)
    assert 8 in train
    assert 8 not in test
    assert set(train).isdisjoint(test)
    np.testing.assert_array_equal(np.sort(np.r_[train, test]), np.arange(len(labels)))


def test_clean_and_private_use_same_model_representation():
    x = np.array([[-0.8, 0.0], [-0.6, 0.1], [0.7, 0.9], [0.9, 0.8]])
    y = np.array([0, 0, 1, 1])
    clean = fit_clean(x, y, 2)
    private, ledger = fit_private(
        x, y, 2, epsilon=8.0, mechanism="laplace", allocation="equal", seed=2,
    )
    assert balanced_accuracy(y, predict(x, clean)) == 1.0
    assert private.means.shape == clean.means.shape
    assert ledger.epsilon_used == pytest.approx(8.0)


def test_softmax_output_perturbation_uses_conservative_neighbor_sensitivity():
    rows = np.array([[-1.0], [0.0], [1.0]])
    labels = np.array([0, 1, 1])
    fit = fit_clean_softmax(rows, labels, 2, regularization=0.5,
                            optimizer_tolerance=1e-5)
    # Replace one row and/or label; every neighboring fit must be within the
    # declared sensitivity bound plus numerical slack.
    neighbors = []
    for changed_row in range(len(rows)):
        for replacement_x in (-1.0, 0.0, 1.0):
            for replacement_y in (0, 1):
                other_x, other_y = rows.copy(), labels.copy()
                other_x[changed_row, 0] = replacement_x
                other_y[changed_row] = replacement_y
                neighbors.append(fit_clean_softmax(
                    other_x, other_y, 2, regularization=0.5,
                    optimizer_tolerance=1e-5,
                ))
    for other in neighbors:
        difference = np.linalg.norm(fit.model.weights - other.model.weights)
        assert difference <= fit.sensitivity_l2 + 1e-8
    private, ledger = privatize_softmax(fit, epsilon=1.5, seed=3)
    assert np.isfinite(private.predict(rows)).all()
    assert ledger.epsilon_used == pytest.approx(1.5)
    assert ledger.delta_used == pytest.approx(1e-6)
    assert len(ledger.releases) == 1


def test_random_forest_uses_data_independent_disjoint_shards_and_parallel_budget():
    schema = PublicSchema(
        ("x", "color"), ("numeric", "nominal"), ((), ("blue", "red")),
        ("no", "yes"),
    )
    numeric_mask = encoded_numeric_mask(schema)
    np.testing.assert_array_equal(numeric_mask, [True, False, False, False])
    x = np.array([
        [-1.0, 1, 0, 0], [-0.5, 0, 1, 0], [0.2, 0, 0, 1],
        [0.8, 1, 0, 0], [0.5, 0, 1, 0], [-0.7, 0, 0, 1],
    ])
    y = np.array([0, 0, 1, 1, 0, 1])
    plan = make_random_forest_plan(len(y), numeric_mask, seed=17, n_trees=3, depth=2)
    all_rows = np.concatenate(plan.shards)
    np.testing.assert_array_equal(np.sort(all_rows), np.arange(len(y)))
    model, ledger = fit_private_random_forest(
        x, y, 2, epsilon=2.0, seed=19, plan=plan,
    )
    assert model.predict_proba(x).shape == (len(y), 2)
    assert np.isfinite(model.predict_proba(x)).all()
    assert ledger["accounting"] == "parallel_disjoint_row_shards"
    assert ledger["epsilon_used"] == pytest.approx(2.0)
    assert ledger["delta_used"] == 0.0
    assert ledger["noise_scale"] == pytest.approx(1.0)
    clean = fit_clean_random_forest(x, y, 2, plan)
    assert clean.predict_proba(x).shape == (len(y), 2)


def test_system_random_sampler_returns_finite_noise_in_expected_shapes():
    rng = SystemRandomGenerator()
    assert rng.normal(size=(20, 3)).shape == (20, 3)
    assert np.isfinite(rng.normal(size=100)).all()
    assert np.isfinite(rng.laplace(scale=2.0, size=100)).all()
    assert (rng.geometric(0.2, size=100) >= 1).all()
    # None is the non-reproducible OS-entropy path, unlike a benchmark seed.
    x = np.array([[-1.0], [0.3], [1.0], [-0.4]])
    y = np.array([0, 0, 1, 1])
    model, ledger = fit_private(
        x, y, 2, epsilon=1.0, mechanism="laplace", allocation="equal",
    )
    assert predict(x, model).shape == (len(y),)
    assert ledger.epsilon_used == pytest.approx(1.0)


def test_confirmatory_family_aliases_exclude_known_parent_variants():
    pairs = (
        ("Australian_seed_2_nrows_2000", "credit-approval"),
        ("credit_seed_0_nrows_2000", "credit-g"),
        ("credit-g_copy", "credit-g"),
        ("cars1", "car"),
        ("vehicle_reproduced", "vehicle"),
        ("glass_clean", "glass"),
        ("fri_c2_250_25", "fri_c4_500_25"),
        ("Fetal_cardiotocography_dataset", "cardiotocography"),
        ("camel-1.0", "camel-1.6"),
        ("xerces-1.2", "xerces-1.3"),
        ("ex1-features-holiday-person-es-201610",
         "ex2-features-holiday-person-event-en-201610"),
        ("data-v3.en-es-lit.clean.anno_without_14top.uniform",
         "data-v3.en-es-lit.clean.anno_without_14top.uniform.arff"),
    )
    for left, right in pairs:
        assert confirmation_family(left) == confirmation_family(right)


def test_dp_specific_calibration_features_are_finite_action_dependent_and_public():
    schema = PublicSchema(
        ("numeric", "color"), ("numeric", "nominal"), ((), ("blue", "red")),
        ("no", "yes"),
    )
    laplace = dp_calibration_features(500, schema, 1.0, "laplace", "equal")
    gaussian = dp_calibration_features(500, schema, 1.0, "gaussian_zcdp", "equal")
    staircase = dp_calibration_features(500, schema, 1.0, "staircase", "equal")
    assert len(laplace) == len(DP_CALIBRATION_FEATURE_NAMES)
    assert np.isfinite(laplace).all() and np.isfinite(gaussian).all()
    assert np.isfinite(staircase).all()
    assert not np.array_equal(laplace, gaussian)
    assert not np.array_equal(laplace, staircase)


def test_privileged_train_features_are_explicit_and_finite():
    x = np.array([[-0.8, 0.0], [-0.6, 0.1], [0.7, 0.9], [0.9, 0.8]])
    y = np.array([0, 0, 1, 1])
    values = private_training_meta_features(x, y, 3)
    assert len(values) == len(PRIVATE_META_COLUMNS)
    assert np.isfinite(values).all()
    assert values[4] == pytest.approx(1 / 3)


def test_research_rich_features_add_pipeline_interactions():
    schema = PublicSchema(("x",), ("numeric",), ((),), ("no", "yes"))
    row = pd.Series({
        "dataset_id": 7,
        "n_instances": 100,
        "public_schema_json": schema_json(schema),
    })
    private = np.linspace(0.0, 1.0, len(PRIVATE_META_COLUMNS))
    private_by_dataset = {7: private}
    base = _candidate_features(
        row, 2.0, PIPELINES[0], "research_privileged", private_by_dataset,
    )
    rich = _candidate_features(
        row, 2.0, PIPELINES[0], "research_rich", private_by_dataset,
    )
    assert len(rich) == len(base) + len(private) * (len(PIPELINES) + 1)
    assert np.isfinite(rich).all()


def test_candidate_feature_names_align_with_columns_and_calibration_groups():
    schema = PublicSchema(("x", "c"), ("numeric", "nominal"), ((), ("a", "b")), ("no", "yes"))
    row = pd.Series({"dataset_id": 7, "n_instances": 100,
                     "public_schema_json": schema_json(schema)})
    private = {7: np.ones(len(PRIVATE_META_COLUMNS))}
    dp = {7: np.ones(len(portfolio.DP_META_COLUMNS))}
    expected_group = {
        "gnb": "calib_gnb", "softmax__analytic_gaussian": "calib_softmax_output",
        "softmax__dp_sgd": "calib_dp_sgd", "random_forest": "calib_forest",
        "categorical_naive_bayes": "calib_categorical_nb",
    }
    for mode in portfolio.FEATURE_MODES:
        names = portfolio.candidate_feature_names(mode)
        groups = np.asarray([group for _, group in names])
        for pipeline in PIPELINES:
            values = _candidate_features(row, 1.0, pipeline, mode, private, dp)
            assert len(values) == len(names)
            calibration = np.char.startswith(groups.astype(str), "calib_")
            used = set(groups[calibration & (values != 0)])
            if mode == "public_schema_only":
                assert not calibration.any()
            else:
                prefix = next(key for key in expected_group if pipeline.startswith(key))
                assert used == {expected_group[prefix]}


def test_portfolio_has_expanded_pipelines_and_requested_epsilon_grid():
    assert len(PIPELINES) == 12
    assert DEFAULT_EPSILONS == (0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
    assert len(set(PIPELINES)) == len(PIPELINES)
    assert "random_forest__laplace_parallel_shards_25" in PIPELINES
    assert "random_forest__laplace_parallel_shards_depth2" not in PIPELINES


def test_condition_grid_rejects_partial_or_duplicated_dataset():
    rows = [
        {"dataset_id": dataset_id, "epsilon": epsilon, "pipeline": pipeline,
         "repetition": repetition, "private_ba": 0.5, "clean_ba": 0.6,
         "best_clean_portfolio_ba": 0.6, "portfolio_regret": 0.1,
         "pipeline_privacy_penalty": 0.1}
        for dataset_id in (1, 2)
        for epsilon in (0.5, 1.0)
        for pipeline in PIPELINES[:2]
        for repetition in (0, 1)
    ]
    conditions = pd.DataFrame(rows)
    manifest = pd.DataFrame({"dataset_id": [1, 2]})
    assert complete_dataset_ids(conditions, epsilons=(0.5, 1.0), repeats=2,
                                pipelines=PIPELINES[:2]) == {1, 2}
    validate_condition_grid(conditions, manifest, epsilons=(0.5, 1.0),
                            repeats=2, pipelines=PIPELINES[:2])
    partial = pd.concat([conditions.iloc[:-1], conditions.iloc[:1]], ignore_index=True)
    with pytest.raises(ValueError, match="grade incompleta"):
        validate_condition_grid(partial, manifest, epsilons=(0.5, 1.0),
                                repeats=2, pipelines=PIPELINES[:2])


def test_critical_difference_uses_only_executable_methods():
    methods = ["public_plus_dp", "best_fixed_regret", "h27_fixed_regret",
               "oracle_regret", "random_pipeline_regret", "worst_pipeline_regret"]
    rows = pd.DataFrame({"method": methods, "regret": range(len(methods))})
    retained = _inferential_methods(rows, ["public_plus_dp"])
    assert set(retained.method) == {
        "public_plus_dp", "best_fixed_regret", "h27_fixed_regret",
    }


def test_new_portfolio_options_report_their_full_privacy_budget():
    schema = PublicSchema(("x", "z"), ("numeric", "numeric"),
                          ((), ()), ("no", "yes"))
    x = np.array([[-1., -1.], [-.8, -.5], [-.5, .2], [-.2, .6],
                  [.2, -.6], [.5, -.2], [.8, .5], [1., 1.]])
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    plan = make_random_forest_plan(len(y), encoded_numeric_mask(schema),
                                   seed=17, n_trees=3, depth=2)
    gaussian, gaussian_ledger = fit_private_random_forest_gaussian(
        x, y, 2, epsilon=1.0, delta=1e-6, seed=18, plan=plan,
    )
    exponential, exponential_ledger = fit_private_random_forest_exponential(
        x, y, 2, epsilon=1.0, seed=19, plan=plan,
    )
    categorical, categories, categorical_ledger = fit_private_categorical_nb(
        x, y, 2, schema, epsilon=1.0, seed=20,
    )
    assert gaussian.predict(x).shape == exponential.predict(x).shape == (len(y),)
    assert categorical.predict(categories).shape == (len(y),)
    assert gaussian_ledger["epsilon_used"] == exponential_ledger["epsilon_used"] == 1.0
    assert gaussian_ledger["delta_used"] == 1e-6
    assert exponential_ledger["delta_used"] == 0.0
    assert categorical_ledger.epsilon_used == pytest.approx(1.0)
    assert categorical_ledger.delta_used == 0.0


def test_new_histogram_options_respect_replace_one_sensitivity():
    schema = PublicSchema(("x", "z"), ("numeric", "numeric"),
                          ((), ()), ("no", "yes"))
    x = np.array([[-1., -1.], [-.5, .2], [.2, -.5], [1., 1.]])
    y = np.array([0, 0, 1, 1])
    changed_x, changed_y = x.copy(), y.copy()
    changed_x[0] = [.9, -.9]
    changed_y[0] = 1
    plan = make_random_forest_plan(len(y), encoded_numeric_mask(schema),
                                   seed=23, n_trees=3, depth=2)
    before = random_forest_leaf_counts(x, y, 2, plan)
    after = random_forest_leaf_counts(changed_x, changed_y, 2, plan)
    differences = [right - left for left, right in zip(before, after)]
    assert sum(np.any(difference) for difference in differences) == 1
    assert max(np.linalg.norm(difference) for difference in differences) <= np.sqrt(2)
    before_nb = categorical_nb_statistics(x, y, 2, schema)
    after_nb = categorical_nb_statistics(changed_x, changed_y, 2, schema)
    assert np.abs(after_nb.class_counts - before_nb.class_counts).sum() <= 2
    for left, right in zip(before_nb.feature_counts, after_nb.feature_counts):
        assert np.abs(right - left).sum() <= 2


def test_shared_portfolio_selector_fits_one_model_per_fold(monkeypatch):
    schema = PublicSchema(("x",), ("numeric",), ((),), ("no", "yes"))
    manifest = pd.DataFrame([
        {"dataset_id": dataset_id, "family_id": f"family_{dataset_id}",
         "n_instances": 100, "public_schema_json": schema_json(schema)}
        for dataset_id in range(5)
    ])
    results = pd.DataFrame([
        {"dataset_id": dataset_id, "epsilon": epsilon, "pipeline": pipeline,
         "private_ba": 0.5 + 0.01 * dataset_id + 0.02 * action,
         "clean_ba": 0.7, "best_clean_portfolio_ba": 0.7,
         "portfolio_regret": 0.2 - 0.01 * dataset_id - 0.02 * action,
         "pipeline_privacy_penalty": 0.2 - 0.01 * dataset_id - 0.02 * action}
        for dataset_id in range(5)
        for epsilon in (0.5, 1.0)
        for action, pipeline in enumerate((PIPELINES[0], PIPELINES[8]))
    ])
    monkeypatch.setattr(portfolio, "SELECTOR_TREE_COUNT", 5)
    fits = []
    original = portfolio._fit_selector

    def counting_fit(*args, **kwargs):
        fits.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(portfolio, "_fit_selector", counting_fit)
    summary, predictions = portfolio.evaluate_grouped(
        results, manifest, n_splits=5, selector_strategy="absolute_shared",
        feature_modes_override=("public_plus_dp",),
    )
    assert len(fits) == 5
    assert len(predictions) == 10
    assert summary["selector_strategy"] == "absolute_shared"


def test_public_only_portfolio_mode_needs_no_train_metadata_columns():
    conditions = pd.DataFrame({
        "dataset_id": [1],
        "epsilon": [1.0],
        "pipeline": [PIPELINES[0]],
        "private_ba": [0.5],
        "clean_ba": [0.6],
        "best_clean_portfolio_ba": [0.6],
        "portfolio_regret": [0.1],
        "pipeline_privacy_penalty": [0.1],
    })
    assert PUBLIC_ONLY_FEATURE_MODES == ("public_schema_only",)
    assert _available_feature_modes(conditions) == (
        "public_schema_only", "public_plus_dp",
    )


def test_public_schema_only_candidate_ignores_train_metadata():
    schema = PublicSchema(("x",), ("numeric",), ((),), ("no", "yes"))
    row = pd.Series({
        "dataset_id": 7,
        "n_instances": 100,
        "public_schema_json": schema_json(schema),
    })
    first = _candidate_features(
        row, 2.0, PIPELINES[0], "public_schema_only", {7: np.zeros(18)},
    )
    second = _candidate_features(
        row, 2.0, PIPELINES[0], "public_schema_only", {7: np.full(18, 99.0)},
    )
    np.testing.assert_array_equal(first, second)


def test_figure_baselines_replace_epsilon_fixed_with_random_and_worst():
    predictions = pd.DataFrame({
        "dataset_id": [1, 2],
        "epsilon": [1.0, 1.0],
        "feature_mode": ["public_schema_only", "public_schema_only"],
        "selector_regret": [0.5, 0.7],
        "oracle_regret": [0.0, 0.0],
        "best_fixed_regret": [1.0, 2.0],
        "gnb_fixed_regret": [3.0, 4.0],
    })
    conditions = pd.DataFrame({
        "dataset_id": [1, 1, 2, 2],
        "epsilon": [1.0, 1.0, 1.0, 1.0],
        "pipeline": ["a", "b", "a", "b"],
        "portfolio_regret": [1.0, 3.0, 2.0, 4.0],
    })
    long, _, _ = _regret_methods(predictions, conditions)
    methods = set(long.method)
    assert "epsilon_fixed_regret" not in BASELINES
    assert {"random_pipeline_regret", "worst_pipeline_regret"} <= methods
    random_values = long[long.method == "random_pipeline_regret"].sort_values("dataset_id")
    worst_values = long[long.method == "worst_pipeline_regret"].sort_values("dataset_id")
    np.testing.assert_allclose(random_values.regret, [2.0, 3.0])
    np.testing.assert_allclose(worst_values.regret, [3.0, 4.0])


def test_dp_sgd_public_calibration_matches_the_fitted_hyperparameters():
    """The public action features must describe the configuration actually fitted."""
    schema = PublicSchema(("x",), ("numeric",), ((),), ("no", "yes"))
    candidate = _public_dp_features(100, schema, 1.0, PIPELINES[5])
    assert len(candidate) == DP_FEATURE_WIDTH
    assert np.isfinite(candidate).all()
    start = GNB_FEATURE_WIDTH + SOFTMAX_FEATURE_WIDTH
    n_steps, gradient_clip, regularization = candidate[start + 3:start + 6]
    assert n_steps == pytest.approx(np.log1p(DP_SGD_STEPS))
    assert gradient_clip == pytest.approx(np.log1p(DP_SGD_GRADIENT_CLIP))
    assert regularization == pytest.approx(np.log1p(DP_SGD_REGULARIZATION))
    softmax = _public_dp_features(100, schema, 1.0, PIPELINES[4])
    assert not np.array_equal(softmax, candidate)
    # A variante de 100 passos compartilha o bloco, mas com outra contagem.
    alternate = _public_dp_features(100, schema, 1.0, PIPELINES[8])
    assert alternate[start + 3] == pytest.approx(np.log1p(DP_SGD_STEPS_ALTERNATE))
    assert not np.array_equal(alternate, candidate)


def test_added_statistical_tests_are_directional_and_paired():
    gains = np.array([0.4, 0.2, 0.3, 0.1, 0.5])
    assert _safe_wilcoxon(gains) < 0.1
    assert _rank_biserial(gains) == pytest.approx(1.0)
    result = _mc_nemar(
        np.array([True, True, False, True]),
        np.array([False, True, True, False]),
    )
    assert result["selector_correct_fixed_wrong"] == 2
    assert result["selector_wrong_fixed_correct"] == 1
    assert 0.0 <= result["exact_binomial_p_two_sided"] <= 1.0


def test_dp_training_meta_features_are_bounded_and_composed():
    x = np.array([[-0.8, 0.0], [-0.6, 0.1], [0.7, 0.9], [0.9, 0.8]])
    y = np.array([0, 0, 1, 1])
    values, ledger = dp_training_meta_features(x, y, 2, epsilon=1.3, seed=9)
    assert len(values) == len(DP_META_COLUMNS)
    assert np.isfinite(values).all()
    assert ledger.epsilon_used == pytest.approx(1.3)
    assert ledger.delta_used == 0.0
    assert 0.0 <= values[0] <= 1.0
    assert 0.0 <= values[1] <= 1.0
    with pytest.raises(ValueError):
        dp_training_meta_features(x * 2.0, y, 2, epsilon=1.3, seed=9)


def test_dp_research_rich_features_use_dp_release_and_interactions():
    schema = PublicSchema(("x",), ("numeric",), ((),), ("no", "yes"))
    row = pd.Series({
        "dataset_id": 7,
        "n_instances": 100,
        "public_schema_json": schema_json(schema),
    })
    private = np.linspace(0.0, 1.0, len(PRIVATE_META_COLUMNS))
    dp_private = np.linspace(0.0, 1.0, len(DP_META_COLUMNS))
    rich = _candidate_features(
        row, 2.0, PIPELINES[0], "dp_research_rich", {7: private}, {7: dp_private},
    )
    assert len(rich) > len(dp_private)
    assert np.isfinite(rich).all()


def test_dp_bounds_are_bounded_reproducible_and_composed():
    frame = pd.DataFrame({
        "x": np.linspace(-5.0, 5.0, 100),
        "y": np.linspace(0.0, 1.0, 100),
    })
    first = estimate_dp_bounds_frame(
        frame, {"x": (-10.0, 10.0), "y": (0.0, 1.0)},
        epsilon=1.2, bins=16, seed=17,
    )
    second = estimate_dp_bounds_frame(
        frame, {"x": (-10.0, 10.0), "y": (0.0, 1.0)},
        epsilon=1.2, bins=16, seed=17,
    )
    assert first.bounds == second.bounds
    assert first.ledger.epsilon_used == pytest.approx(1.2)
    assert first.ledger.delta_used == pytest.approx(0.0)
    assert all(-10.0 <= value <= 10.0 for value in first.bounds["x"])
    assert all(0.0 <= value <= 1.0 for value in first.bounds["y"])

    one, scale = estimate_dp_numeric_bounds(
        frame["x"], coarse_lower=-10.0, coarse_upper=10.0,
        epsilon=0.6, bins=8, rng=np.random.default_rng(4),
    )
    assert one[0] < one[1]
    assert scale == pytest.approx(2.0 / 0.6)


def test_outer_evaluation_keeps_related_datasets_in_one_fold():
    schema = PublicSchema(("x",), ("numeric",), ((),), ("a", "b"))
    manifest_rows, result_rows = [], []
    for dataset_id in range(10):
        family = "related_pair" if dataset_id in (0, 1) else f"family_{dataset_id}"
        manifest_rows.append({
            "dataset_id": dataset_id, "family_id": family, "n_instances": 100 + dataset_id,
            "public_schema_json": schema_json(schema),
        })
        for epsilon in (1.0, 4.0):
            for repetition in (0, 1):
                for action_index, (mechanism, allocation) in enumerate(ACTIONS):
                    regret = 0.12 - 0.025 * action_index + dataset_id * 0.0001 + repetition * 0.0002
                    result_rows.append({
                        "dataset_id": dataset_id, "family_id": family, "epsilon": epsilon,
                        "mechanism": mechanism, "allocation": allocation,
                        "repetition": repetition, "clean_ba": 0.8,
                        "private_ba": 0.8 - regret, "regret": regret,
                        "l2_distortion": 1.0 + action_index * 0.1 + dataset_id * 0.0001,
                    })
    summary, predictions = evaluate_grouped(
        pd.DataFrame(result_rows), pd.DataFrame(manifest_rows), n_splits=5, seed=7,
    )
    folds_per_family = predictions.groupby("family_id").fold.nunique()
    assert (folds_per_family == 1).all()
    pair_fold = predictions[predictions.dataset_id.isin([0, 1])].groupby("dataset_id").fold.first()
    assert pair_fold.loc[0] == pair_fold.loc[1]
    assert summary["n_families"] == 9
    assert summary["top1_auxiliary"] == 1.0
    l2_summary, l2_predictions = evaluate_grouped(
        pd.DataFrame(result_rows), pd.DataFrame(manifest_rows),
        n_splits=5, seed=7, target_column="l2_regret",
    )
    assert l2_summary["target_column"] == "l2_regret"
    assert l2_summary["top1_auxiliary"] == 1.0
    assert l2_predictions.selector_regret.min() >= 1.0
    assert "classifier_utility_of_l2_selected_action" in l2_summary


def schema_json(schema):
    import json
    return json.dumps({
        "feature_names": schema.feature_names,
        "feature_kinds": schema.feature_kinds,
        "nominal_domains": schema.nominal_domains,
        "classes": schema.classes,
    })


def test_sparse_arff_omitted_nominal_target_maps_to_first_declared_value():
    from dpmeta.openml_benchmark import sparse_nominal_default
    y = pd.Series([0, "1", "-1", 0.0], dtype=object)
    assert sparse_nominal_default(y, ["1", "-1"]).tolist() == ["1", "1", "-1", "1"]
    kept = sparse_nominal_default(pd.Series(["0", "1"], dtype=object), ["0", "1"])
    assert kept.tolist() == ["0", "1"]

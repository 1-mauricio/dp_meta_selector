"""Additional end-to-end DP classifiers for portfolio feasibility experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import softmax

from .mechanisms import analytic_gaussian_sigma, privacy_rng, zcdp_rho_for_epsilon
from .model import DELTA_TOTAL
from .privacy import PrivacyLedger, Release, Sensitivity
from .schema import PublicSchema


@dataclass(frozen=True)
class SoftmaxModel:
    weights: np.ndarray

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        augmented = _augment(x)
        return softmax(augmented @ self.weights, axis=1)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.predict_proba(x).argmax(axis=1)


@dataclass(frozen=True)
class SoftmaxFit:
    model: SoftmaxModel
    sensitivity_l2: float
    optimizer_tolerance: float
    regularization: float


def _augment(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or not np.isfinite(x).all() or (x.size and np.max(np.abs(x)) > 1.0 + 1e-12):
        raise ValueError("softmax requer features finitas no domínio público [-1,1]")
    return np.column_stack((x, np.ones(len(x), dtype=float)))


def fit_clean_softmax(x: np.ndarray, y: np.ndarray, n_classes: int, *,
                      regularization: float = 0.1,
                      optimizer_tolerance: float = 1e-5,
                      record_norm_bound: float | None = None) -> SoftmaxFit:
    """Fit strongly-convex multinomial logistic regression.

    Objective: mean multiclass log loss + regularization * ||W||_F^2 / 2.
    The solver uses a fixed, public number of accelerated-gradient iterations,
    avoiding a data-dependent convergence/abort signal. `optimizer_tolerance`
    is a uniform parameter-distance error bound from the exact minimizer.
    """
    augmented = _augment(x)
    y = np.asarray(y, dtype=int)
    n, dimension = augmented.shape
    if y.shape != (n,) or n == 0 or n_classes < 2:
        raise ValueError("shapes/classes inválidos para softmax")
    if np.any(y < 0) or np.any(y >= n_classes):
        raise ValueError("rótulos fora do schema de classes")
    if regularization <= 0 or optimizer_tolerance <= 0:
        raise ValueError("regularização e tolerância devem ser positivas")

    one_hot = np.zeros((n, n_classes), dtype=float)
    one_hot[np.arange(n), y] = 1.0

    def gradient(weights: np.ndarray) -> np.ndarray:
        logits = augmented @ weights
        probabilities = softmax(logits, axis=1)
        return augmented.T @ (probabilities - one_hot) / n + regularization * weights

    # The logistic Hessian is bounded by 0.5 ||z||² + lambda. The strongly
    # convex accelerated-gradient distance bound includes sqrt(kappa + 1),
    # where kappa = L / mu; the iteration count depends only on public inputs.
    feature_norm_bound = (np.sqrt(dimension) if record_norm_bound is None
                          else float(record_norm_bound))
    if not np.isfinite(feature_norm_bound) or feature_norm_bound <= 0:
        raise ValueError("record_norm_bound deve ser finito e positivo")
    smoothness = regularization + 0.5 * feature_norm_bound**2
    contraction = 1.0 - np.sqrt(regularization / smoothness)
    initial_distance_bound = (
        np.sqrt(smoothness / regularization + 1.0)
        * 2.0 * np.sqrt(2.0 * np.log(n_classes) / regularization)
    )
    if optimizer_tolerance >= initial_distance_bound:
        iterations = 1
    else:
        log_contraction = np.log1p(-np.sqrt(regularization / smoothness))
        iterations = int(np.ceil(
            2.0 * np.log(initial_distance_bound / optimizer_tolerance) / -log_contraction
        ))
    weights = np.zeros((dimension, n_classes), dtype=float)
    previous = weights.copy()
    momentum = ((np.sqrt(smoothness) - np.sqrt(regularization))
                / (np.sqrt(smoothness) + np.sqrt(regularization)))
    for _ in range(iterations):
        extrapolated = weights + momentum * (weights - previous)
        updated = extrapolated - gradient(extrapolated) / smoothness
        previous, weights = weights, updated
    if not np.isfinite(weights).all():
        raise FloatingPointError("softmax resultou em coeficientes não finitos")

    # Replace-one sensitivity for the exact minimizer follows from strong
    # convexity. Inflate by the uniform optimizer residual bound on both sides.
    # A public schema can give a sharper bound than encoded dimension: each
    # original numeric or one-hot feature contributes at most one to ||x||².
    per_record_gradient_bound = np.sqrt(2.0) * feature_norm_bound
    exact_sensitivity = 2.0 * per_record_gradient_bound / (regularization * n)
    approximate_sensitivity = exact_sensitivity + 2.0 * optimizer_tolerance
    return SoftmaxFit(
        SoftmaxModel(weights), float(approximate_sensitivity),
        float(optimizer_tolerance), float(regularization),
    )


def privatize_softmax(fit: SoftmaxFit, *, epsilon: float, seed: int | None = None,
                      delta: float = DELTA_TOTAL) -> tuple[SoftmaxModel, PrivacyLedger]:
    """Analytic-Gaussian output perturbation of the bounded-sensitivity optimum."""
    if epsilon <= 0 or not 0 < delta < 1:
        raise ValueError("epsilon/delta inválidos")
    weights = fit.model.weights
    sensitivity = fit.sensitivity_l2
    sigma = analytic_gaussian_sigma(epsilon, delta, sensitivity)
    rng = privacy_rng(seed)
    noisy = weights + rng.normal(0.0, sigma, size=weights.shape)
    dimension = weights.size
    sensitivity_record = Sensitivity(
        l1=float(np.sqrt(dimension) * sensitivity),
        l2=sensitivity,
        linf=sensitivity,
    )
    ledger = PrivacyLedger(epsilon, delta)
    ledger.record(Release(
        name="softmax_coefficients",
        mechanism="analytic_gaussian_output_perturbation",
        epsilon=float(epsilon),
        delta=float(delta),
        sensitivity=sensitivity_record,
        noise_scale=float(sigma),
    ))
    return SoftmaxModel(noisy), ledger.verify()


def fit_private_sgd_softmax(x: np.ndarray, y: np.ndarray, n_classes: int, *,
                            epsilon: float, delta: float, seed: int | None = None,
                            n_steps: int = 100, gradient_clip: float = 1.0,
                            regularization: float = 0.1
                            ) -> tuple[SoftmaxModel, PrivacyLedger]:
    """Full-batch DP-SGD softmax with per-record clipping and zCDP composition.

    Each fixed-step gradient is a Gaussian mechanism on the mean of clipped
    per-record cross-entropy gradients. The updates are adaptive post-processing;
    their rho values compose, then convert once to (epsilon, delta)-DP.
    This reference implementation does not use subsampling amplification.
    """
    augmented = _augment(x)
    y = np.asarray(y, dtype=int)
    n, dimension = augmented.shape
    if (y.shape != (n,) or n == 0 or n_classes < 2 or np.any(y < 0)
            or np.any(y >= n_classes)):
        raise ValueError("shapes/classes inválidos para DP-SGD softmax")
    if epsilon <= 0 or not 0 < delta < 1:
        raise ValueError("epsilon/delta inválidos para DP-SGD softmax")
    if n_steps < 1 or gradient_clip <= 0 or regularization <= 0:
        raise ValueError("steps, gradient_clip e regularization precisam ser positivos")

    # Public worst-case norm for one encoded record plus the intercept.
    record_norm_bound = np.sqrt(dimension)
    smoothness = regularization + 0.5 * record_norm_bound**2
    momentum = ((np.sqrt(smoothness) - np.sqrt(regularization))
                / (np.sqrt(smoothness) + np.sqrt(regularization)))
    rho_total = zcdp_rho_for_epsilon(epsilon, delta)
    rho_step = rho_total / n_steps
    sensitivity = 2.0 * gradient_clip / n
    sigma = sensitivity / np.sqrt(2.0 * rho_step)
    rng = privacy_rng(seed)
    ledger = PrivacyLedger(epsilon, delta, accounting="zcdp")
    sensitivity_record = Sensitivity(
        l1=float(sensitivity * np.sqrt(dimension * n_classes)),
        l2=float(sensitivity),
        linf=float(sensitivity),
    )
    weights = np.zeros((dimension, n_classes), dtype=float)
    previous = weights.copy()
    chunk_size = 128

    for step in range(n_steps):
        extrapolated = weights + momentum * (weights - previous)
        gradient_sum = np.zeros_like(weights)
        for start in range(0, n, chunk_size):
            stop = min(start + chunk_size, n)
            block = augmented[start:stop]
            logits = block @ extrapolated
            residual = softmax(logits, axis=1)
            residual[np.arange(stop - start), y[start:stop]] -= 1.0
            outer_norm = (np.linalg.norm(block, axis=1)
                          * np.linalg.norm(residual, axis=1))
            clipping = np.minimum(1.0, gradient_clip / np.maximum(outer_norm, 1e-300))
            # This is the same clipped per-record gradient sum expressed as
            # a matrix product. It is substantially faster for the repeated
            # full-batch DP-SGD updates; only floating-point reduction order
            # differs from the reference einsum.
            gradient_sum += block.T @ (clipping[:, None] * residual)
        noisy_gradient = gradient_sum / n + regularization * extrapolated
        noisy_gradient += rng.normal(0.0, sigma, size=noisy_gradient.shape)
        updated = extrapolated - noisy_gradient / smoothness
        previous, weights = weights, updated
        ledger.record(Release(
            name=f"dp_sgd_gradient_{step + 1}",
            mechanism="gaussian_zcdp_full_batch_gradient",
            epsilon=0.0,
            delta=0.0,
            sensitivity=sensitivity_record,
            noise_scale=float(sigma),
            rho=float(rho_step),
        ))

    if not np.isfinite(weights).all():
        raise FloatingPointError("DP-SGD softmax resultou em coeficientes não finitos")
    return SoftmaxModel(weights), ledger.verify()


@dataclass(frozen=True)
class RandomTreeStructure:
    features: np.ndarray
    thresholds: np.ndarray
    depth: int

    @property
    def leaf_count(self) -> int:
        return 2**self.depth

    @property
    def first_leaf_node(self) -> int:
        return 2**self.depth - 1

    def leaf_ids(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        nodes = np.zeros(len(x), dtype=int)
        for _ in range(self.depth):
            feature = self.features[nodes]
            threshold = self.thresholds[nodes]
            go_right = x[np.arange(len(x)), feature] > threshold
            nodes = 2 * nodes + 1 + go_right.astype(int)
        return nodes - self.first_leaf_node


@dataclass(frozen=True)
class RandomForestPlan:
    structures: tuple[RandomTreeStructure, ...]
    shards: tuple[np.ndarray, ...]

    @property
    def shard_sizes(self) -> tuple[int, ...]:
        return tuple(len(shard) for shard in self.shards)


@dataclass(frozen=True)
class RandomForestModel:
    structures: tuple[RandomTreeStructure, ...]
    leaf_probabilities: tuple[np.ndarray, ...]

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if not len(self.structures):
            raise ValueError("floresta vazia")
        n_classes = self.leaf_probabilities[0].shape[1]
        combined = np.zeros((len(x), n_classes), dtype=float)
        for structure, probabilities in zip(self.structures, self.leaf_probabilities):
            combined += probabilities[structure.leaf_ids(x)]
        return combined / len(self.structures)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.predict_proba(x).argmax(axis=1)


@dataclass(frozen=True)
class CategoricalNBStatistics:
    categories: np.ndarray
    cardinalities: tuple[int, ...]
    class_counts: np.ndarray
    feature_counts: tuple[np.ndarray, ...]


@dataclass(frozen=True)
class CategoricalNaiveBayesModel:
    class_counts: np.ndarray
    feature_counts: tuple[np.ndarray, ...]
    feature_cardinalities: tuple[int, ...]

    def predict(self, categories: np.ndarray) -> np.ndarray:
        categories = np.asarray(categories, dtype=int)
        if categories.ndim != 2 or categories.shape[1] != len(self.feature_counts):
            raise ValueError("categorias incompatíveis com Categorical Naive Bayes")
        class_counts = np.maximum(np.asarray(self.class_counts, dtype=float), 0.0)
        class_probs = (class_counts + 1.0) / (class_counts.sum() + len(class_counts))
        scores = np.tile(np.log(class_probs), (len(categories), 1))
        for feature_index, (counts, cardinality) in enumerate(
                zip(self.feature_counts, self.feature_cardinalities)):
            counts = np.maximum(np.asarray(counts, dtype=float), 0.0)
            totals = counts.sum(axis=1) + cardinality
            selected = counts[:, categories[:, feature_index]]
            scores += np.log((selected.T + 1.0) / totals[None, :])
        return scores.argmax(axis=1)


def make_random_forest_plan(n_rows: int, numeric_mask: np.ndarray, *, seed: int,
                             n_trees: int = 5, depth: int = 3) -> RandomForestPlan:
    """Create data-independent tree structures and disjoint, random row shards."""
    numeric_mask = np.asarray(numeric_mask, dtype=bool)
    if n_rows < 1 or numeric_mask.ndim != 1 or len(numeric_mask) < 1:
        raise ValueError("shape inválido para plano de floresta")
    if n_trees < 1 or depth < 1:
        raise ValueError("n_trees e depth devem ser positivos")
    n_trees = min(n_trees, n_rows)
    rng = np.random.default_rng(seed)
    shards = tuple(np.asarray(shard, dtype=int) for shard in
                   np.array_split(rng.permutation(n_rows), n_trees))
    structures = []
    n_internal = 2**depth - 1
    n_nodes = 2**(depth + 1) - 1
    for _ in range(n_trees):
        features = np.full(n_nodes, -1, dtype=int)
        thresholds = np.zeros(n_nodes, dtype=float)
        for node in range(n_internal):
            feature = int(rng.integers(0, len(numeric_mask)))
            features[node] = feature
            thresholds[node] = (float(rng.uniform(-0.8, 0.8))
                                if numeric_mask[feature] else 0.5)
        structures.append(RandomTreeStructure(features, thresholds, depth))
    return RandomForestPlan(tuple(structures), shards)


def encoded_numeric_mask(schema) -> np.ndarray:
    """Public map from encoded coordinates to numerical vs one-hot features."""
    mask: list[bool] = []
    for kind, domain in zip(schema.feature_kinds, schema.nominal_domains):
        mask.extend([True] if kind == "numeric" else [False] * (len(domain) + 1))
    return np.asarray(mask, dtype=bool)


def _leaf_counts(x: np.ndarray, y: np.ndarray, n_classes: int,
                 structure: RandomTreeStructure) -> np.ndarray:
    leaves = structure.leaf_ids(x)
    counts = np.zeros((structure.leaf_count, n_classes), dtype=float)
    np.add.at(counts, (leaves, y), 1.0)
    return counts


def random_forest_leaf_counts(
    x: np.ndarray, y: np.ndarray, n_classes: int, plan: RandomForestPlan,
) -> tuple[np.ndarray, ...]:
    """Compute the deterministic leaf histograms once for a split and plan.

    Callers can reuse these exact private sufficient statistics across epsilon
    values and mechanisms; each model fit still draws its own mechanism noise.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=int)
    if (x.ndim != 2 or y.shape != (len(x),) or len(y) == 0
            or not np.isfinite(x).all() or np.any(y < 0) or np.any(y >= n_classes)
            or n_classes < 2):
        raise ValueError("dados/classes inválidos para contagens da floresta")
    if len(plan.shards) != len(plan.structures):
        raise ValueError("plano de floresta inconsistente")
    all_indices = np.concatenate(plan.shards)
    if len(all_indices) != len(x) or not np.array_equal(
            np.sort(all_indices), np.arange(len(x))):
        raise ValueError("shards precisam particionar cada linha exatamente uma vez")
    return tuple(
        _leaf_counts(x[shard], y[shard], n_classes, structure)
        for structure, shard in zip(plan.structures, plan.shards)
    )


def _resolve_leaf_counts(
    x: np.ndarray, y: np.ndarray, n_classes: int, plan: RandomForestPlan,
    leaf_counts: tuple[np.ndarray, ...] | None,
) -> tuple[np.ndarray, ...]:
    if leaf_counts is None:
        return random_forest_leaf_counts(x, y, n_classes, plan)
    if (len(plan.shards) != len(plan.structures)
            or len(leaf_counts) != len(plan.structures)):
        raise ValueError("contagens pré-calculadas incompatíveis com o plano")
    resolved = tuple(np.asarray(counts, dtype=float) for counts in leaf_counts)
    if any(
        counts.shape != (structure.leaf_count, n_classes)
        or not np.isfinite(counts).all()
        or np.any(counts < 0)
        or not np.isclose(counts.sum(), len(shard))
        for structure, shard, counts in zip(plan.structures, plan.shards, resolved)
    ):
        raise ValueError("formas das contagens pré-calculadas incompatíveis")
    return resolved


def _counts_to_probabilities(counts: np.ndarray) -> np.ndarray:
    # Fixed Laplace smoothing is post-processing and applied identically to all.
    smoothed = np.maximum(np.asarray(counts, dtype=float), 0.0) + 1.0
    return smoothed / smoothed.sum(axis=1, keepdims=True)


def fit_clean_random_forest(x: np.ndarray, y: np.ndarray, n_classes: int,
                            plan: RandomForestPlan, *,
                            leaf_counts: tuple[np.ndarray, ...] | None = None
                            ) -> RandomForestModel:
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=int)
    if y.shape != (len(x),) or len(plan.shards) != len(plan.structures):
        raise ValueError("dados/plano incompatíveis")
    counts_by_tree = _resolve_leaf_counts(x, y, n_classes, plan, leaf_counts)
    probabilities = [_counts_to_probabilities(counts) for counts in counts_by_tree]
    return RandomForestModel(plan.structures, tuple(probabilities))


def fit_private_random_forest(x: np.ndarray, y: np.ndarray, n_classes: int, *,
                              epsilon: float, seed: int | None = None,
                              plan: RandomForestPlan,
                              leaf_counts: tuple[np.ndarray, ...] | None = None
                              ) -> tuple[RandomForestModel, dict]:
    """Pure-DP leaf histograms with parallel composition over disjoint shards.

    Tree structures and shard assignments are independent of feature/label
    values. Each row affects exactly one tree; that tree's joint leaf-class
    count vector has replace-one L1 sensitivity 2 and gets epsilon-DP noise.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=int)
    if (epsilon <= 0 or x.ndim != 2 or y.shape != (len(x),) or len(y) == 0
            or not np.isfinite(x).all() or np.any(y < 0) or np.any(y >= n_classes)
            or n_classes < 2):
        raise ValueError("epsilon/dados inválidos")
    counts_by_tree = _resolve_leaf_counts(x, y, n_classes, plan, leaf_counts)

    rng = privacy_rng(seed)
    scale = 2.0 / epsilon
    probabilities = []
    for counts in counts_by_tree:
        noisy = counts + rng.laplace(0.0, scale, size=counts.shape)
        probabilities.append(_counts_to_probabilities(noisy))
    ledger = {
        "accounting": "parallel_disjoint_row_shards",
        "epsilon_limit": float(epsilon),
        "epsilon_used": float(epsilon),
        "delta_limit": 0.0,
        "delta_used": 0.0,
        "per_shard_epsilon": float(epsilon),
        "n_shards": len(plan.shards),
        "shard_sizes": list(plan.shard_sizes),
        "replace_one_l1_sensitivity_per_shard": 2.0,
        "noise_scale": float(scale),
    }
    return RandomForestModel(plan.structures, tuple(probabilities)), ledger


def fit_private_random_forest_gaussian(
    x: np.ndarray, y: np.ndarray, n_classes: int, *, epsilon: float,
    delta: float = DELTA_TOTAL, seed: int | None = None,
    plan: RandomForestPlan, leaf_counts: tuple[np.ndarray, ...] | None = None,
) -> tuple[RandomForestModel, dict]:
    """Analytic-Gaussian leaf histograms with parallel composition by shard."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=int)
    if (epsilon <= 0 or not 0 < delta < 1 or x.ndim != 2
            or y.shape != (len(x),) or len(y) == 0 or not np.isfinite(x).all()
            or np.any(y < 0) or np.any(y >= n_classes) or n_classes < 2):
        raise ValueError("epsilon/delta/dados inválidos para floresta Gaussiana")
    counts_by_tree = _resolve_leaf_counts(x, y, n_classes, plan, leaf_counts)

    sensitivity_l2 = np.sqrt(2.0)
    sigma = analytic_gaussian_sigma(epsilon, delta, sensitivity_l2)
    rng = privacy_rng(seed)
    probabilities = []
    for counts in counts_by_tree:
        noisy = counts + rng.normal(0.0, sigma, size=counts.shape)
        probabilities.append(_counts_to_probabilities(noisy))
    ledger = {
        "accounting": "parallel_disjoint_row_shards",
        "epsilon_limit": float(epsilon),
        "epsilon_used": float(epsilon),
        "delta_limit": float(delta),
        "delta_used": float(delta),
        "per_shard_epsilon": float(epsilon),
        "per_shard_delta": float(delta),
        "n_shards": len(plan.shards),
        "shard_sizes": list(plan.shard_sizes),
        "replace_one_l2_sensitivity_per_shard": float(sensitivity_l2),
        "noise_scale": float(sigma),
        "mechanism": "analytic_gaussian",
    }
    return RandomForestModel(plan.structures, tuple(probabilities)), ledger


def fit_private_random_forest_exponential(
    x: np.ndarray, y: np.ndarray, n_classes: int, *, epsilon: float,
    seed: int | None = None, plan: RandomForestPlan,
    leaf_counts: tuple[np.ndarray, ...] | None = None,
) -> tuple[RandomForestModel, dict]:
    """Private majority-label selection per leaf; output is a one-hot forest."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=int)
    if (epsilon <= 0 or x.ndim != 2 or y.shape != (len(x),) or len(y) == 0
            or not np.isfinite(x).all() or np.any(y < 0) or np.any(y >= n_classes)
            or n_classes < 2):
        raise ValueError("epsilon/dados inválidos para floresta exponencial")
    counts_by_tree = _resolve_leaf_counts(x, y, n_classes, plan, leaf_counts)

    rng = privacy_rng(seed)
    # Replacing one row can affect the majority query in at most two leaves.
    # Each leaf gets epsilon/2, and disjoint row shards compose in parallel.
    epsilon_per_leaf = epsilon / 2.0
    probabilities = []
    for counts in counts_by_tree:
        logits = epsilon_per_leaf * counts / 2.0
        logits -= logits.max(axis=1, keepdims=True)
        weights = np.exp(logits)
        choices = np.empty(len(counts), dtype=int)
        for leaf_index, row_weights in enumerate(weights):
            probabilities_for_leaf = row_weights / row_weights.sum()
            choices[leaf_index] = rng.choice(n_classes, p=probabilities_for_leaf)
        probabilities.append(np.eye(n_classes, dtype=float)[choices])
    ledger = {
        "accounting": "parallel_shards_and_two_leaf_composition",
        "epsilon_limit": float(epsilon),
        "epsilon_used": float(epsilon),
        "delta_limit": 0.0,
        "delta_used": 0.0,
        "epsilon_per_leaf": float(epsilon_per_leaf),
        "n_shards": len(plan.shards),
        "shard_sizes": list(plan.shard_sizes),
        "max_affected_leaves_per_record": 2,
        "mechanism": "exponential_majority_label",
    }
    return RandomForestModel(plan.structures, tuple(probabilities)), ledger


CATEGORICAL_NB_NUMERIC_BINS = 16


def categorical_nb_features(x: np.ndarray, schema: PublicSchema,
                             numeric_bins: int = CATEGORICAL_NB_NUMERIC_BINS
                             ) -> tuple[np.ndarray, tuple[int, ...]]:
    """Map public one-hot/numeric encodings to bounded categorical values."""
    x = np.asarray(x, dtype=float)
    if (x.ndim != 2 or x.shape[1] != schema.encoded_dim or numeric_bins < 2
            or not np.isfinite(x).all() or (x.size and np.max(np.abs(x)) > 1.0 + 1e-12)):
        raise ValueError("features inválidas para Categorical Naive Bayes")
    values = np.empty((len(x), len(schema.feature_names)), dtype=int)
    cardinalities = []
    column = 0
    for feature_index, (kind, domain) in enumerate(
            zip(schema.feature_kinds, schema.nominal_domains)):
        if kind == "numeric":
            bins = np.floor((np.clip(x[:, column], -1.0, 1.0) + 1.0)
                            * (numeric_bins / 2.0)).astype(int)
            values[:, feature_index] = np.clip(bins, 0, numeric_bins - 1)
            cardinalities.append(numeric_bins)
            column += 1
        else:
            width = len(domain) + 1
            values[:, feature_index] = np.argmax(x[:, column:column + width], axis=1)
            cardinalities.append(width)
            column += width
    return values, tuple(cardinalities)


def _categorical_nb_counts(categories: np.ndarray, y: np.ndarray, n_classes: int,
                           cardinalities: tuple[int, ...]
                           ) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    class_counts = np.bincount(y, minlength=n_classes).astype(float)
    feature_counts = []
    for feature_index, cardinality in enumerate(cardinalities):
        counts = np.zeros((n_classes, cardinality), dtype=float)
        np.add.at(counts, (y, categories[:, feature_index]), 1.0)
        feature_counts.append(counts)
    return class_counts, tuple(feature_counts)


def categorical_nb_statistics(
    x: np.ndarray, y: np.ndarray, n_classes: int, schema: PublicSchema,
    *, numeric_bins: int = CATEGORICAL_NB_NUMERIC_BINS,
) -> CategoricalNBStatistics:
    """Compute deterministic categorical histograms once for one train split."""
    y = np.asarray(y, dtype=int)
    categories, cardinalities = categorical_nb_features(x, schema, numeric_bins=numeric_bins)
    if y.shape != (len(categories),) or np.any(y < 0) or np.any(y >= n_classes):
        raise ValueError("rótulos inválidos para Categorical Naive Bayes")
    class_counts, feature_counts = _categorical_nb_counts(
        categories, y, n_classes, cardinalities,
    )
    return CategoricalNBStatistics(categories, cardinalities, class_counts, feature_counts)


def fit_clean_categorical_nb(x: np.ndarray, y: np.ndarray, n_classes: int,
                             schema: PublicSchema, *,
                             statistics: CategoricalNBStatistics | None = None,
                             numeric_bins: int = CATEGORICAL_NB_NUMERIC_BINS,
                             uniform_prior: bool = False,
                             ) -> tuple[CategoricalNaiveBayesModel, np.ndarray]:
    y = np.asarray(y, dtype=int)
    if statistics is None:
        statistics = categorical_nb_statistics(x, y, n_classes, schema, numeric_bins=numeric_bins)
    categories, cardinalities = statistics.categories, statistics.cardinalities
    if y.shape != (len(categories),) or np.any(y < 0) or np.any(y >= n_classes):
        raise ValueError("rótulos inválidos para Categorical Naive Bayes")
    model = CategoricalNaiveBayesModel(
        (np.ones_like(statistics.class_counts) if uniform_prior else statistics.class_counts),
        statistics.feature_counts, cardinalities,
    )
    return model, categories


def fit_private_categorical_nb(
    x: np.ndarray, y: np.ndarray, n_classes: int, schema: PublicSchema, *,
    epsilon: float, seed: int | None = None,
    statistics: CategoricalNBStatistics | None = None,
    numeric_bins: int = CATEGORICAL_NB_NUMERIC_BINS,
    uniform_prior: bool = False,
) -> tuple[CategoricalNaiveBayesModel, np.ndarray, PrivacyLedger]:
    """Laplace histograms with basic composition over class/feature count vectors."""
    if epsilon <= 0:
        raise ValueError("epsilon precisa ser positivo")
    y = np.asarray(y, dtype=int)
    if statistics is None:
        statistics = categorical_nb_statistics(x, y, n_classes, schema, numeric_bins=numeric_bins)
    categories, cardinalities = statistics.categories, statistics.cardinalities
    if y.shape != (len(categories),) or len(y) == 0 or np.any(y < 0) or np.any(y >= n_classes):
        raise ValueError("rótulos inválidos para Categorical Naive Bayes")
    class_counts, feature_counts = statistics.class_counts, statistics.feature_counts
    query_count = len(feature_counts) + 1
    epsilon_per_query = epsilon / query_count
    scale = 2.0 / epsilon_per_query
    sensitivity = Sensitivity(2.0, np.sqrt(2.0), 1.0)
    rng = privacy_rng(seed)
    ledger = PrivacyLedger(epsilon, 0.0, accounting="basic")
    noisy_class_counts = class_counts + rng.laplace(0.0, scale, size=class_counts.shape)
    ledger.record(Release(
        name="class_prior_histogram", mechanism="laplace", epsilon=epsilon_per_query,
        delta=0.0, sensitivity=sensitivity, noise_scale=float(scale),
    ))
    noisy_features = []
    for feature_index, counts in enumerate(feature_counts):
        noisy = counts + rng.laplace(0.0, scale, size=counts.shape)
        noisy_features.append(noisy)
        ledger.record(Release(
            name=f"feature_{feature_index}_histogram", mechanism="laplace",
            epsilon=epsilon_per_query, delta=0.0,
            sensitivity=sensitivity, noise_scale=float(scale),
        ))
    ledger.verify()
    model = CategoricalNaiveBayesModel(
        # Keep the prior query and RNG draw above even when its result is unused.
        # Equal counts reproduce the E4 decision rule without changing the ledger.
        (np.ones_like(noisy_class_counts) if uniform_prior else noisy_class_counts),
        tuple(noisy_features), cardinalities,
    )
    return model, categories, ledger

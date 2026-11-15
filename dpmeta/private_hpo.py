"""Private softmax hyperparameter tuning with subsample selection and refit.

The tuning rows are sampled with public, value-independent Bernoulli coins.
Each candidate is released through Gaussian output perturbation, the private
validation choice uses the exponential mechanism on bounded accuracy, and the
chosen regularization is refit on all outer-training rows.  The implementation
uses sequential composition; it does not claim privacy amplification from the
Poisson sample.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from math import sqrt

import numpy as np

from .accounting import (
    compose_gaussian_privacy_profile,
    gaussian_mu_for_delta,
    gaussian_privacy_profile_delta,
)
from .classifiers import SoftmaxModel, fit_clean_softmax, privatize_softmax
from .mechanisms import privacy_rng
from .model import DELTA_TOTAL, balanced_accuracy
from .privacy import PrivacyLedger, Release, Sensitivity


DEFAULT_REGULARIZATIONS = (0.03, 0.1, 0.3)
TUNING_FRACTION = 0.20
TUNING_EPSILON_FRACTION = 0.10
SELECTION_EPSILON_FRACTION = 0.10
TUNING_DELTA_FRACTION = 0.10


@dataclass(frozen=True)
class PrivateHPOResult:
    """Selected refit model and auditable intermediate quantities."""

    model: SoftmaxModel
    selected_regularization: float
    candidate_scores: tuple[float, ...]
    selected_index: int
    tuning_indices: np.ndarray
    validation_indices: np.ndarray
    epsilon_tuning_models: float
    epsilon_selection: float
    epsilon_final_model: float
    delta_tuning_models: float
    delta_final_model: float
    ledger: dict


def poisson_split(n: int, sampling_rate: float, seed: int | None) -> tuple[np.ndarray, np.ndarray]:
    """Return a value-independent Bernoulli sample and its complement.

    The complement is used only for the private utility query.  A fallback
    keeps both sets non-empty for tiny smoke-test datasets; the fallback is
    still independent of record values.
    """
    if n < 2 or not 0.0 < sampling_rate < 1.0:
        raise ValueError("n precisa ser >= 2 e sampling_rate deve estar em (0,1)")
    rng = np.random.default_rng(seed)
    tuning = np.flatnonzero(rng.random(n) < sampling_rate)
    if len(tuning) == 0:
        tuning = np.asarray([int(rng.integers(0, n))])
    if len(tuning) == n:
        tuning = tuning[:-1]
    tuning_mask = np.zeros(n, dtype=bool)
    tuning_mask[tuning] = True
    validation = np.flatnonzero(~tuning_mask)
    if len(validation) == 0:
        raise ValueError("a amostra Poisson deixou a validação vazia")
    return tuning.astype(int), validation.astype(int)


def exponential_select(scores: np.ndarray, epsilon: float, n_validation: int,
                       seed: int | None) -> int:
    """Select a candidate with the exponential mechanism on mean accuracy."""
    scores = np.asarray(scores, dtype=float)
    if (scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all()
            or np.any(scores < 0.0) or np.any(scores > 1.0)):
        raise ValueError("scores devem ser uma lista finita em [0,1]")
    if epsilon <= 0 or n_validation < 1:
        raise ValueError("epsilon e n_validation devem ser positivos")
    sensitivity = 1.0 / n_validation
    logits = epsilon * scores / (2.0 * sensitivity)
    probabilities = np.exp(logits - np.max(logits))
    probabilities /= probabilities.sum()
    draw = float(privacy_rng(seed).random())
    return min(int(np.searchsorted(np.cumsum(probabilities), draw, side="right")),
               len(scores) - 1)


def _compose_ledger(*, epsilon_total: float, delta_total: float,
                    candidate_ledgers: list[dict], epsilon_selection: float,
                    n_validation: int, final_ledger: dict) -> dict:
    ledger = PrivacyLedger(epsilon_total, delta_total, accounting="basic")
    for index, item in enumerate(candidate_ledgers):
        ledger.record(Release(
            name=f"private_hpo_candidate_{index}",
            mechanism="analytic_gaussian_output_perturbation",
            epsilon=float(item["epsilon_used"]),
            delta=float(item["delta_used"]),
            sensitivity=Sensitivity(0.0, 0.0, 0.0),
            noise_scale=0.0,
        ))
    sensitivity = 1.0 / n_validation
    ledger.record(Release(
        name="private_hpo_validation_choice",
        mechanism="exponential_mechanism_mean_accuracy",
        epsilon=float(epsilon_selection),
        delta=0.0,
        sensitivity=Sensitivity(sensitivity, sensitivity, sensitivity),
        noise_scale=0.0,
    ))
    ledger.record(Release(
        name="private_hpo_final_refit",
        mechanism="analytic_gaussian_output_perturbation",
        epsilon=float(final_ledger["epsilon_used"]),
        delta=float(final_ledger["delta_used"]),
        sensitivity=Sensitivity(0.0, 0.0, 0.0),
        noise_scale=0.0,
    ))
    return ledger.verify().as_dict()


def fit_private_softmax_hpo(
    x: np.ndarray,
    y: np.ndarray,
    n_classes: int,
    *,
    epsilon: float,
    seed: int | None = None,
    delta: float = DELTA_TOTAL,
    regularizations: tuple[float, ...] = DEFAULT_REGULARIZATIONS,
    sampling_rate: float = TUNING_FRACTION,
    tuning_epsilon_fraction: float = TUNING_EPSILON_FRACTION,
    selection_epsilon_fraction: float = SELECTION_EPSILON_FRACTION,
    tuning_delta_fraction: float = TUNING_DELTA_FRACTION,
    optimizer_tolerance: float = 1e-5,
    record_norm_bound: float | None = None,
) -> PrivateHPOResult:
    """Tune regularization privately and refit the selected model on all rows.

    Candidate fit releases share ``tuning_epsilon_fraction`` and
    ``tuning_delta_fraction`` equally.  The private validation choice receives
    ``selection_epsilon_fraction``.  The final refit receives the remainder of
    epsilon and delta.  All candidate releases, the choice, and the refit are
    recorded in one basic-composition ledger.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=int)
    if x.ndim != 2 or y.shape != (len(x),) or len(x) < 2:
        raise ValueError("shapes inválidos para private HPO")
    if epsilon <= 0 or not 0 < delta < 1:
        raise ValueError("epsilon/delta inválidos para private HPO")
    if not regularizations or any(value <= 0 for value in regularizations):
        raise ValueError("regularizações devem ser positivas")
    if (tuning_epsilon_fraction <= 0 or selection_epsilon_fraction <= 0
            or tuning_epsilon_fraction + selection_epsilon_fraction >= 1):
        raise ValueError("frações de epsilon inválidas")
    if not 0 < tuning_delta_fraction < 1:
        raise ValueError("fração de delta inválida")

    tuning_indices, validation_indices = poisson_split(
        len(x), sampling_rate, None if seed is None else seed + 1,
    )
    x_tune, y_tune = x[tuning_indices], y[tuning_indices]
    x_validation, y_validation = x[validation_indices], y[validation_indices]
    n_candidates = len(regularizations)
    epsilon_tuning_total = epsilon * tuning_epsilon_fraction
    epsilon_selection = epsilon * selection_epsilon_fraction
    epsilon_final = epsilon - epsilon_tuning_total - epsilon_selection
    epsilon_candidate = epsilon_tuning_total / n_candidates
    delta_tuning_total = delta * tuning_delta_fraction
    delta_candidate = delta_tuning_total / n_candidates
    delta_final = delta - delta_tuning_total
    if epsilon_final <= 0 or delta_final <= 0:
        raise ValueError("orçamento restante para o refit é inválido")

    norm_bound = record_norm_bound
    if norm_bound is None:
        norm_bound = sqrt(x.shape[1] + 1)
    models: list[SoftmaxModel] = []
    candidate_ledgers: list[dict] = []
    scores = []
    for index, regularization in enumerate(regularizations):
        clean = fit_clean_softmax(
            x_tune, y_tune, n_classes,
            regularization=float(regularization),
            optimizer_tolerance=optimizer_tolerance,
            record_norm_bound=norm_bound,
        )
        model, model_ledger = privatize_softmax(
            clean,
            epsilon=epsilon_candidate,
            delta=delta_candidate,
            seed=None if seed is None else seed + 100 + index,
        )
        models.append(model)
        candidate_ledgers.append(model_ledger.as_dict())
        scores.append(float(np.mean(model.predict(x_validation) == y_validation)))
    candidate_scores = np.asarray(scores, dtype=float)
    selected_index = exponential_select(
        candidate_scores, epsilon_selection, len(y_validation),
        None if seed is None else seed + 200,
    )
    selected_regularization = float(regularizations[selected_index])
    clean_final = fit_clean_softmax(
        x, y, n_classes,
        regularization=selected_regularization,
        optimizer_tolerance=optimizer_tolerance,
        record_norm_bound=norm_bound,
    )
    final_model, final_ledger = privatize_softmax(
        clean_final,
        epsilon=epsilon_final,
        delta=delta_final,
        seed=None if seed is None else seed + 300,
    )
    composed = _compose_ledger(
        epsilon_total=epsilon,
        delta_total=delta,
        candidate_ledgers=candidate_ledgers,
        epsilon_selection=epsilon_selection,
        n_validation=len(y_validation),
        final_ledger=final_ledger.as_dict(),
    )
    return PrivateHPOResult(
        model=final_model,
        selected_regularization=selected_regularization,
        candidate_scores=tuple(float(value) for value in candidate_scores),
        selected_index=int(selected_index),
        tuning_indices=tuning_indices,
        validation_indices=validation_indices,
        epsilon_tuning_models=float(epsilon_tuning_total),
        epsilon_selection=float(epsilon_selection),
        epsilon_final_model=float(epsilon_final),
        delta_tuning_models=float(delta_tuning_total),
        delta_final_model=float(delta_final),
        ledger=composed,
    )


def fit_private_softmax_hpo_profile(
    x: np.ndarray,
    y: np.ndarray,
    n_classes: int,
    *,
    epsilon: float,
    seed: int | None = None,
    delta: float = DELTA_TOTAL,
    regularizations: tuple[float, ...] = DEFAULT_REGULARIZATIONS,
    sampling_rate: float = TUNING_FRACTION,
    tuning_epsilon_fraction: float = TUNING_EPSILON_FRACTION,
    selection_epsilon_fraction: float = SELECTION_EPSILON_FRACTION,
    tuning_delta_fraction: float = TUNING_DELTA_FRACTION,
    optimizer_tolerance: float = 1e-5,
    record_norm_bound: float | None = None,
) -> PrivateHPOResult:
    """Private HPO using exact composition of Gaussian privacy profiles.

    Candidate models keep the same analytic-Gaussian releases as the basic
    runner. After the private choice, the final refit noise is calibrated from
    the exact composed Gaussian profile, rather than adding per-release
    ``(epsilon, delta)`` bounds. The pure-DP exponential choice is composed
    additively. No amplification from the public Bernoulli sample is used.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=int)
    if x.ndim != 2 or y.shape != (len(x),) or len(x) < 2:
        raise ValueError("shapes inválidos para private HPO")
    if epsilon <= 0 or not 0 < delta < 1:
        raise ValueError("epsilon/delta inválidos para private HPO")
    if not regularizations or any(value <= 0 for value in regularizations):
        raise ValueError("regularizações devem ser positivas")
    if (tuning_epsilon_fraction <= 0 or selection_epsilon_fraction <= 0
            or tuning_epsilon_fraction + selection_epsilon_fraction >= 1):
        raise ValueError("frações de epsilon inválidas")
    if not 0 < tuning_delta_fraction < 1:
        raise ValueError("fração de delta inválida")

    tuning_indices, validation_indices = poisson_split(
        len(x), sampling_rate, None if seed is None else seed + 1,
    )
    x_tune, y_tune = x[tuning_indices], y[tuning_indices]
    x_validation, y_validation = x[validation_indices], y[validation_indices]
    n_candidates = len(regularizations)
    epsilon_tuning_total = epsilon * tuning_epsilon_fraction
    epsilon_selection = epsilon * selection_epsilon_fraction
    epsilon_gaussian = epsilon - epsilon_selection
    epsilon_candidate = epsilon_tuning_total / n_candidates
    delta_tuning_total = delta * tuning_delta_fraction
    delta_candidate = delta_tuning_total / n_candidates
    norm_bound = (record_norm_bound if record_norm_bound is not None
                  else sqrt(x.shape[1] + 1))

    models: list[SoftmaxModel] = []
    fits = []
    candidate_ledgers: list[dict] = []
    scores = []
    for index, regularization in enumerate(regularizations):
        clean = fit_clean_softmax(
            x_tune, y_tune, n_classes,
            regularization=float(regularization),
            optimizer_tolerance=optimizer_tolerance,
            record_norm_bound=norm_bound,
        )
        model, model_ledger = privatize_softmax(
            clean,
            epsilon=epsilon_candidate,
            delta=delta_candidate,
            seed=None if seed is None else seed + 100 + index,
        )
        models.append(model)
        fits.append(clean)
        candidate_ledgers.append(model_ledger.as_dict())
        scores.append(float(np.mean(model.predict(x_validation) == y_validation)))

    candidate_scores = np.asarray(scores, dtype=float)
    selected_index = exponential_select(
        candidate_scores, epsilon_selection, len(y_validation),
        None if seed is None else seed + 200,
    )
    selected_regularization = float(regularizations[selected_index])
    clean_final = fit_clean_softmax(
        x, y, n_classes,
        regularization=selected_regularization,
        optimizer_tolerance=optimizer_tolerance,
        record_norm_bound=norm_bound,
    )
    candidate_releases = [
        (fit.sensitivity_l2, float(ledger["releases"][0]["noise_scale"]))
        for fit, ledger in zip(fits, candidate_ledgers)
    ]
    target_mu = gaussian_mu_for_delta(epsilon_gaussian, delta)
    candidate_mu_squared = sum((sensitivity / scale) ** 2
                               for sensitivity, scale in candidate_releases)
    final_mu_squared = target_mu**2 - candidate_mu_squared
    if final_mu_squared <= 0:
        raise ValueError("candidatos consumiram todo o perfil Gaussiano")
    final_mu = sqrt(final_mu_squared)
    final_noise_scale = clean_final.sensitivity_l2 / final_mu
    final_model = SoftmaxModel(
        clean_final.model.weights + privacy_rng(
            None if seed is None else seed + 300,
        ).normal(0.0, final_noise_scale, size=clean_final.model.weights.shape)
    )
    profile = compose_gaussian_privacy_profile(
        candidate_releases + [(clean_final.sensitivity_l2, final_noise_scale)],
        epsilon_gaussian=epsilon_gaussian,
        delta=delta,
        pure_dp_epsilon=epsilon_selection,
    )
    if not profile["within_delta"] or profile["epsilon_used"] > epsilon + 1e-10:
        raise AssertionError("privacy profile excedeu o contrato")
    profile.update({
        "epsilon_limit": float(epsilon),
        "delta_limit": float(delta),
        "candidate_mu_squared": float(candidate_mu_squared),
        "target_mu": float(target_mu),
        "final_mu": float(final_mu),
        "final_noise_scale": float(final_noise_scale),
        "selection_epsilon": float(epsilon_selection),
        "candidate_ledgers": candidate_ledgers,
    })
    return PrivateHPOResult(
        model=final_model,
        selected_regularization=selected_regularization,
        candidate_scores=tuple(float(value) for value in candidate_scores),
        selected_index=int(selected_index),
        tuning_indices=tuning_indices,
        validation_indices=validation_indices,
        epsilon_tuning_models=float(epsilon_tuning_total),
        epsilon_selection=float(epsilon_selection),
        epsilon_final_model=float(epsilon_gaussian),
        delta_tuning_models=float(delta_tuning_total),
        delta_final_model=float(delta),
        ledger=profile,
    )


def hpo_result_json(result: PrivateHPOResult) -> str:
    """Serialize audit-relevant HPO details without model coefficients."""
    return json.dumps({
        "selected_regularization": result.selected_regularization,
        "candidate_scores": list(result.candidate_scores),
        "selected_index": result.selected_index,
        "n_tuning": int(len(result.tuning_indices)),
        "n_validation": int(len(result.validation_indices)),
        "epsilon_tuning_models": result.epsilon_tuning_models,
        "epsilon_selection": result.epsilon_selection,
        "epsilon_final_model": result.epsilon_final_model,
        "delta_tuning_models": result.delta_tuning_models,
        "delta_final_model": result.delta_final_model,
        "ledger": result.ledger,
    }, sort_keys=True)


__all__ = [
    "DEFAULT_REGULARIZATIONS",
    "PrivateHPOResult",
    "fit_private_softmax_hpo",
    "fit_private_softmax_hpo_profile",
    "hpo_result_json",
    "poisson_split",
]

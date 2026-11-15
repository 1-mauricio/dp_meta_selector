"""Baselines que pagam privacidade pela escolha da opção (emenda secundária).

Todos gastam o mesmo ε total que o seletor gratuito:

* ``search_shards``: 80% do treino em 12 partições disjuntas, uma por opção,
  cada uma treinada com ε; a escolha usa o Mecanismo Exponencial com ε sobre os
  acertos nos 20% de validação (sensibilidade 1). Composição paralela: ε.
* ``search_retrain``: a mesma busca com ε/2 e δ/2, e o retreino da opção
  escolhida em todo o treino com ε/2 e δ/2. Composição sequencial: ε.
* ``pythia_like``: meta-features DP liberadas com 0,1ε, escolha pelo seletor
  ``dp_research_rich`` do desenvolvimento e treino da opção com 0,9ε.
"""

from __future__ import annotations

from math import sqrt

import numpy as np

from .classifiers import (
    categorical_nb_features, encoded_numeric_mask, fit_clean_softmax,
    fit_private_categorical_nb, fit_private_random_forest,
    fit_private_random_forest_exponential, fit_private_random_forest_gaussian,
    fit_private_sgd_softmax, make_random_forest_plan, privatize_softmax,
)
from .meta_features import dp_training_meta_features
from .model import DELTA_TOTAL, fit_private, predict
from .openml_benchmark import _seed_for
from . import portfolio_benchmark as pb

GNB_MECHANISMS = {pb.PIPELINES[0]: "gaussian_zcdp", pb.PIPELINES[1]: "laplace",
                  pb.PIPELINES[2]: "staircase", pb.PIPELINES[3]: "analytic_gaussian"}
VALIDATION_FRACTION = 0.2
PYTHIA_RHO = 0.1


def fit_option(pipeline: str, x, y, schema, *, epsilon: float, delta: float, seed: int):
    """Treina uma opção do portfólio com (ε, δ) e devolve ``predict(x) -> rótulos``.

    Reproduz as configurações de ``portfolio_benchmark.run_dataset``.
    """
    k = len(schema.classes)
    if pipeline in GNB_MECHANISMS:
        mechanism = GNB_MECHANISMS[pipeline]
        model, _ = fit_private(x, y, k, epsilon=epsilon, mechanism=mechanism,
                               allocation="equal", seed=seed,
                               delta=delta if "gaussian" in mechanism else None)
        return lambda z: predict(z, model)
    if pipeline == pb.PIPELINES[4]:
        clean = fit_clean_softmax(x, y, k, regularization=pb.SOFTMAX_REGULARIZATION,
                                  optimizer_tolerance=pb.SOFTMAX_OPTIMIZER_TOLERANCE,
                                  record_norm_bound=sqrt(len(schema.feature_names) + 1))
        model, _ = privatize_softmax(clean, epsilon=epsilon, seed=seed, delta=delta)
        return model.predict
    if pipeline in (pb.PIPELINES[5], pb.PIPELINES[7], pb.PIPELINES[8]):
        steps = pb.DP_SGD_STEPS if pipeline == pb.PIPELINES[5] else pb.DP_SGD_STEPS_ALTERNATE
        clip2 = pipeline == pb.PIPELINES[8]
        model, _ = fit_private_sgd_softmax(
            x, y, k, epsilon=epsilon, delta=delta, seed=seed, n_steps=steps,
            gradient_clip=pb.DP_SGD_CLIP2_GRADIENT_CLIP if clip2 else pb.DP_SGD_GRADIENT_CLIP,
            regularization=pb.DP_SGD_CLIP2_REGULARIZATION if clip2 else pb.DP_SGD_REGULARIZATION)
        return model.predict
    if pipeline in (pb.PIPELINES[6], pb.PIPELINES[9], pb.PIPELINES[11]):
        plan = make_random_forest_plan(len(y), encoded_numeric_mask(schema), seed=seed + 1,
                                       n_trees=pb.FOREST_SMALL_TREE_COUNT, depth=pb.TREE_DEPTH)
        if pipeline == pb.PIPELINES[6]:
            model, _ = fit_private_random_forest(x, y, k, epsilon=epsilon, seed=seed, plan=plan)
        elif pipeline == pb.PIPELINES[9]:
            model, _ = fit_private_random_forest_gaussian(x, y, k, epsilon=epsilon, delta=delta,
                                                          seed=seed, plan=plan)
        else:
            model, _ = fit_private_random_forest_exponential(x, y, k, epsilon=epsilon,
                                                             seed=seed, plan=plan)
        return model.predict
    if pipeline == pb.PIPELINES[10]:
        model, _, _ = fit_private_categorical_nb(x, y, k, schema, epsilon=epsilon, seed=seed)
        return lambda z: model.predict(categorical_nb_features(z, schema)[0])
    raise ValueError(f"opção desconhecida: {pipeline}")


def exponential_choice(scores: np.ndarray, epsilon: float, rng: np.random.Generator) -> int:
    """Mecanismo Exponencial sobre contagens de acerto (sensibilidade 1)."""
    logits = epsilon * (scores - scores.max()) / 2.0
    weights = np.exp(logits)
    return int(rng.choice(len(scores), p=weights / weights.sum()))


def private_search(x_train, y_train, schema, *, epsilon: float, delta: float, seed: int,
                   rng_seed: int):
    """Busca por partições com (ε, δ); devolve (opção escolhida, preditor)."""
    order = np.random.default_rng(rng_seed).permutation(len(y_train))  # pública
    n_val = max(1, int(round(VALIDATION_FRACTION * len(order))))
    val, fit_rows = order[:n_val], order[n_val:]
    shards = np.array_split(fit_rows, len(pb.PIPELINES))
    scores, predictors = np.zeros(len(pb.PIPELINES)), []
    for index, (pipeline, shard) in enumerate(zip(pb.PIPELINES, shards)):
        try:
            predictor = fit_option(pipeline, x_train[shard], y_train[shard], schema,
                                   epsilon=epsilon, delta=delta, seed=seed + index)
            scores[index] = float((predictor(x_train[val]) == y_train[val]).sum())
        except Exception:  # noqa: BLE001 - falha depende só do tamanho público da partição
            predictor = None  # partição pequena demais para a opção; nunca vence
            scores[index] = -np.inf
        predictors.append(predictor)
    ok = np.isfinite(scores)
    if not ok.any():
        raise ValueError("nenhuma opção treinou nas partições")
    finite = np.where(ok, scores, scores[ok].min() - 1e6)
    chosen = exponential_choice(finite, epsilon, np.random.default_rng(rng_seed + 1))
    return pb.PIPELINES[chosen], predictors[chosen]


def run_baselines(x_train, y_train, x_test, y_test, schema, *, epsilon: float, seed: int,
                  dataset_id: int, repetition: int, pythia_selector=None, meta_row=None):
    """Executa os três baselines em uma condição e devolve BA e opção de cada um."""
    from .model import balanced_accuracy

    out = {}
    key = (seed, dataset_id, repetition, epsilon)
    chosen, predictor = private_search(
        x_train, y_train, schema, epsilon=epsilon, delta=DELTA_TOTAL,
        seed=_seed_for(*key, "search_shards"), rng_seed=_seed_for(*key, "search_order"))
    out["search_shards"] = (chosen, balanced_accuracy(y_test, predictor(x_test)))

    chosen, _ = private_search(
        x_train, y_train, schema, epsilon=epsilon / 2, delta=DELTA_TOTAL / 2,
        seed=_seed_for(*key, "search_half"), rng_seed=_seed_for(*key, "search_half_order"))
    predictor = fit_option(chosen, x_train, y_train, schema, epsilon=epsilon / 2,
                           delta=DELTA_TOTAL / 2, seed=_seed_for(*key, "search_retrain"))
    out["search_retrain"] = (chosen, balanced_accuracy(y_test, predictor(x_test)))

    if pythia_selector is not None:
        features, _ = dp_training_meta_features(
            x_train, y_train, len(schema.classes), epsilon=PYTHIA_RHO * epsilon,
            seed=_seed_for(*key, "pythia_meta"))
        model_epsilon = (1 - PYTHIA_RHO) * epsilon
        candidates = np.vstack([
            pb._candidate_features(meta_row, model_epsilon, pipeline, "dp_research_rich",
                                   {}, {int(dataset_id): features})
            for pipeline in pb.PIPELINES])
        chosen = pb.PIPELINES[int(np.argmax(pythia_selector.predict(candidates)))]
        predictor = fit_option(chosen, x_train, y_train, schema, epsilon=model_epsilon,
                               delta=DELTA_TOTAL, seed=_seed_for(*key, "pythia_train"))
        out["pythia_like"] = (chosen, balanced_accuracy(y_test, predictor(x_test)))
    return out

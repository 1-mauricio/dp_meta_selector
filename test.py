#!/usr/bin/env python3
"""Smoke test for loading a saved model and using the recommender."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def _import_project() -> tuple:
    """Import package symbols even when executed from inside the package folder."""
    here = Path(__file__).resolve().parent
    parent = here.parent
    if str(parent) not in sys.path:
        sys.path.insert(0, str(parent))

    from dp_meta_selector.mechanisms import MECHANISM_NAMES
    from dp_meta_selector.selector import DPMechanismSelector

    return DPMechanismSelector, MECHANISM_NAMES


def _build_demo_dataset(seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    X = rng.normal(loc=0.0, scale=1.0, size=(120, 6)).astype(float)
    score = 0.7 * X[:, 0] - 0.4 * X[:, 1] + 0.2 * X[:, 2] + rng.normal(0, 0.25, 120)
    q1, q2 = np.quantile(score, [0.33, 0.66])
    y = np.digitize(score, bins=[q1, q2]).astype(int)
    return X, y


def main() -> int:
    DPMechanismSelector, mechanism_names = _import_project()

    model_path = Path(__file__).resolve().parent / "dp_meta_selector.joblib"
    if not model_path.is_file():
        print(f"[ERRO] Modelo nao encontrado: {model_path}")
        return 1

    try:
        selector = DPMechanismSelector.load_from(str(model_path))
    except Exception as exc:
        print(f"[ERRO] Falha ao carregar modelo: {exc}")
        return 2

    X, y = _build_demo_dataset(seed=42)

    try:
        rec = selector.recommend(X, y, verbose=False)
    except Exception as exc:
        print(f"[ERRO] Falha no recommend(): {exc}")
        return 3

    mech = rec.get("recommended_mechanism")
    conf = float(rec.get("confidence", -1.0))
    model_used = rec.get("meta_model_used", "?")

    if mech not in mechanism_names:
        print(f"[ERRO] Mecanismo recomendado invalido: {mech}")
        return 4
    if not (0.0 <= conf <= 1.0):
        print(f"[ERRO] Confianca fora de faixa [0,1]: {conf}")
        return 5

    try:
        X_dp = selector.apply(X, mech, verbose=False)
    except Exception as exc:
        print(f"[ERRO] Falha no apply(): {exc}")
        return 6

    if X_dp.shape != X.shape:
        print(f"[ERRO] Shape invalido apos apply(): {X_dp.shape} vs {X.shape}")
        return 7
    if not np.isfinite(X_dp).all():
        print("[ERRO] Resultado DP contem NaN/Inf.")
        return 8

    print("[OK] Modelo carregado e recomendador testado com sucesso.")
    print(f"[OK] Mecanismo recomendado: {mech}")
    print(f"[OK] Confianca: {conf:.4f}")
    print(f"[OK] Meta-modelo usado: {model_used}")
    print(f"[OK] Shape entrada/saida: {X.shape} -> {X_dp.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

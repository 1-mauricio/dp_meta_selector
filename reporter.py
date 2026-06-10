"""Gerador de relatório estruturado de uma run completa do dp_meta_selector."""

import json
import logging
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.pipeline import Pipeline

from .calibration import FAMILY_EPSILON
from .config import (
    DELTA_DEFAULT,
    FINGERPRINT_SAMPLE_SIZE,
    FRAMEWORK_VERSION,
    MAX_ROWS_PER_DATASET,
    OPENML_TRAINING_TARGET,
    TARGET_NOISE_RATIO,
)
from .mechanisms import DP_MECHANISMS, FAMILY_OF, MECHANISM_NAMES

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe(v) -> Any:
    """Converte tipos numpy/pandas para tipos nativos Python (serializáveis)."""
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return None if np.isnan(v) else float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, np.ndarray):
        return [_safe(x) for x in v.tolist()]
    if isinstance(v, pd.Series):
        return {str(k): _safe(vv) for k, vv in v.items()}
    if isinstance(v, dict):
        return {str(k): _safe(vv) for k, vv in v.items()}
    if isinstance(v, (list, tuple)):
        return [_safe(x) for x in v]
    return v


def _ci95(series: pd.Series) -> Dict[str, float]:
    if len(series) < 2:
        return {"mean": _safe(series.mean()), "ci95_low": None, "ci95_high": None, "std": None}
    lo, hi = stats.t.interval(
        0.95, len(series) - 1,
        loc=series.mean(),
        scale=stats.sem(series),
    )
    return {
        "mean": _safe(series.mean()),
        "ci95_low": _safe(lo),
        "ci95_high": _safe(hi),
        "std": _safe(series.std()),
        "min": _safe(series.min()),
        "max": _safe(series.max()),
        "median": _safe(series.median()),
    }


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------

def _section_metadata(start_time: float, run_label: str, log_file: Optional[Path]) -> Dict:
    return {
        "run_label": run_label,
        "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(start_time)),
        "run_duration_seconds": _safe(time.time() - start_time),
        "framework_version": FRAMEWORK_VERSION,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "log_file": str(log_file.resolve()) if log_file else None,
    }


def _section_config() -> Dict:
    return {
        "delta": DELTA_DEFAULT,
        "target_noise_ratio": TARGET_NOISE_RATIO,
        "openml_training_target": OPENML_TRAINING_TARGET,
        "max_rows_per_dataset": MAX_ROWS_PER_DATASET,
        "fingerprint_sample_size": FINGERPRINT_SAMPLE_SIZE,
        "family_epsilon": _safe(FAMILY_EPSILON),
    }


def _section_mechanisms() -> list:
    return [
        {
            "name": m.name,
            "family": m.family,
            "description": m.description,
            "epsilon": _safe(FAMILY_EPSILON.get(m.family)),
        }
        for m in DP_MECHANISMS
    ]


def _section_data_splits(train_ds, test_ds, all_ds) -> Dict:
    def _ds_summary(item):
        X, y, name = item
        unique_y = np.unique(y)
        return {
            "name": name,
            "n_samples": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "n_classes": int(len(unique_y)),
        }

    return {
        "total_loaded": len(all_ds),
        "train_count": len(train_ds),
        "test_count": len(test_ds),
        "train_datasets": [_ds_summary(d) for d in train_ds],
        "test_datasets": [_ds_summary(d) for d in test_ds],
    }


def _section_meta_dataset(selector) -> Dict:
    meta_df = selector.meta_df
    if meta_df is None:
        return {}

    excl = (
        {"dataset_name", "best_mechanism", "best_relative_acc", "baseline_acc"}
        | {f"acc_{m}" for m in MECHANISM_NAMES}
    )
    feature_cols = [c for c in meta_df.columns if c not in excl]

    class_dist = meta_df["best_mechanism"].value_counts().to_dict()
    n = len(meta_df)

    # Stats por meta-feature
    feat_stats = {}
    for col in feature_cols:
        s = meta_df[col].dropna()
        if len(s) > 0:
            feat_stats[col] = {
                "mean": _safe(s.mean()),
                "std": _safe(s.std()),
                "min": _safe(s.min()),
                "max": _safe(s.max()),
                "median": _safe(s.median()),
                "pct_nonzero": _safe((s != 0).mean()),
            }

    # Registros de treino com todas as acurácias por mecanismo
    training_records = []
    for _, row in meta_df.iterrows():
        acc_per_mech = {m: _safe(row.get(f"acc_{m}", None)) for m in MECHANISM_NAMES}
        training_records.append({
            "name": row["dataset_name"],
            "best_mechanism": row["best_mechanism"],
            "best_relative_acc": _safe(row.get("best_relative_acc")),
            "baseline_acc": _safe(row.get("baseline_acc")),
            "n_samples": _safe(row.get("n_samples")),
            "n_features": _safe(row.get("n_features")),
            "n_classes": _safe(row.get("n_classes")),
            "class_imbalance": _safe(row.get("class_imbalance")),
            "mean_mi": _safe(row.get("mean_mi")),
            "mean_corr": _safe(row.get("mean_corr")),
            "ratio_discrete": _safe(row.get("ratio_discrete")),
            "sparsity": _safe(row.get("sparsity")),
            "pca_intrinsic_dim_ratio": _safe(row.get("pca_intrinsic_dim_ratio")),
            "acc_per_mechanism": acc_per_mech,
        })

    return {
        "n_samples": n,
        "n_features": len(feature_cols),
        "feature_names": feature_cols,
        "class_distribution": {k: int(v) for k, v in class_dist.items()},
        "class_percentages": {k: round(v / n * 100, 2) for k, v in class_dist.items()},
        "feature_statistics": feat_stats,
        "training_datasets": training_records,
    }


def _section_meta_learner(selector) -> Dict:
    learner = selector._learner
    cv_scores = _safe(selector.cv_scores or {})
    model_names = list(learner.models.keys())
    classes = list(learner.label_encoder.classes_) if learner.label_encoder else []
    feature_cols = learner.META_FEATURE_COLS or []

    # Feature importances para modelos baseados em árvore
    feature_importances: Dict[str, Any] = {}
    for name, model in learner.models.items():
        try:
            # Desembrulha calibrador se necessário
            inner = model
            while hasattr(inner, "estimator"):
                inner = inner.estimator
            # Desembrulha Pipeline
            if isinstance(inner, Pipeline):
                inner = inner.steps[-1][1]
            if hasattr(inner, "feature_importances_"):
                fi = inner.feature_importances_
                feature_importances[name] = {
                    col: _safe(imp)
                    for col, imp in sorted(
                        zip(feature_cols, fi), key=lambda x: -x[1]
                    )[:30]  # top-30
                }
            elif hasattr(inner, "coef_"):
                coef = np.abs(inner.coef_)
                if coef.ndim > 1:
                    coef = coef.mean(axis=0)
                feature_importances[name] = {
                    col: _safe(imp)
                    for col, imp in sorted(
                        zip(feature_cols, coef), key=lambda x: -x[1]
                    )[:30]
                }
        except Exception:
            pass

    return {
        "fast_mode": selector._learner._fast_landmarks,
        "models": model_names,
        "best_model": learner.best_model_name,
        "cv_scores_f1_macro": cv_scores,
        "meta_feature_cols": feature_cols,
        "n_meta_features": len(feature_cols),
        "label_classes": classes,
        "calibration": "isotonic",
        "feature_importances": feature_importances,
    }


def _section_evaluation(results_df: pd.DataFrame) -> Dict:
    df = results_df.copy()
    n = len(df)
    if n == 0:
        return {}

    # Ganho normalizado
    df["delta_laplace"] = df["rec_acc"] - df["laplace_acc"]
    df["spread"] = df["best_acc"] - df["random_acc"] + 1e-9
    df["norm_ganho"] = df["delta_laplace"] / df["spread"]

    n_melhor = int((df["delta_laplace"] > 1e-6).sum())
    n_pior   = int((df["delta_laplace"] < -1e-6).sum())
    n_igual  = n - n_melhor - n_pior

    # Breakdown por família
    by_family: Dict[str, Any] = {}
    families = sorted(set(FAMILY_OF.values()))
    for fam in families:
        mask = df["best_mech"].apply(lambda m: FAMILY_OF.get(m, "") == fam)
        sub = df[mask]
        if not sub.empty:
            by_family[fam] = {
                "n": int(len(sub)),
                "hit_rate": _ci95(sub["hit"].astype(float)),
                "regret": _ci95(sub["regret"]),
                "relative_performance": _ci95(sub["relative_performance"]),
                "rec_acc": _ci95(sub["rec_acc"]),
            }

    # Acurácias por mecanismo
    acc_cols = [c for c in df.columns if c.startswith("acc_")]
    acc_by_mech: Dict[str, Any] = {}
    for col in acc_cols:
        mech = col[4:]
        acc_by_mech[mech] = _ci95(df[col]) if col in df.columns else {}

    # Registros individuais de teste
    test_records = []
    for _, row in df.iterrows():
        record: Dict[str, Any] = {
            "dataset": row["dataset"],
            "best_mechanism": row["best_mech"],
            "recommended_mechanism": row["rec_mech"],
            "hit": bool(row["hit"]),
            "best_acc": _safe(row["best_acc"]),
            "rec_acc": _safe(row["rec_acc"]),
            "base_acc": _safe(row["base_acc"]),
            "regret": _safe(row["regret"]),
            "relative_performance": _safe(row["relative_performance"]),
            "random_acc": _safe(row["random_acc"]),
            "laplace_acc": _safe(row["laplace_acc"]),
            "delta_laplace": _safe(row["delta_laplace"]),
            "norm_gain": _safe(row["norm_ganho"]),
            "oracle_family": FAMILY_OF.get(str(row["best_mech"]), "unknown"),
        }
        test_records.append(record)

    return {
        "n_test_datasets": n,
        "hit_rate": _ci95(df["hit"].astype(float)),
        "regret": _ci95(df["regret"]),
        "relative_performance": _ci95(df["relative_performance"]),
        "baselines": {
            "model_rec_acc": _safe(df["rec_acc"].mean()),
            "random_all_mechs": _safe(df["random_acc"].mean()),
            "laplace_fixed": _safe(df["laplace_acc"].mean()),
            "oracle_best": _safe(df["best_acc"].mean()),
            "base_no_dp": _safe(df["base_acc"].mean()),
        },
        "model_vs_laplace": {
            "better_count": n_melhor,
            "equal_count": n_igual,
            "worse_count": n_pior,
            "better_pct": round(n_melhor / n * 100, 2),
            "worse_pct": round(n_pior / n * 100, 2),
            "normalized_gain_mean": _safe(df["norm_ganho"].mean()),
        },
        "by_family": by_family,
        "test_datasets": test_records,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_report(
    selector,
    results_df: pd.DataFrame,
    train_ds: list,
    test_ds: list,
    all_ds: list,
    start_time: float,
    output_dir: Path = Path("reports"),
    run_label: Optional[str] = None,
    log_file: Optional[Path] = None,
) -> Path:
    """Gera e salva um relatório JSON completo da run.

    Parameters
    ----------
    selector:
        Instância treinada de ``DPMechanismSelector``.
    results_df:
        DataFrame retornado por ``FrameworkEvaluator.evaluate()``.
    train_ds, test_ds, all_ds:
        Listas de datasets (X, y, name).
    start_time:
        ``time.time()`` capturado no início da run.
    output_dir:
        Diretório onde o relatório será salvo.
    run_label:
        Identificador legível (padrão: timestamp).
    log_file:
        Caminho do arquivo de log da run (opcional).

    Returns
    -------
    Path
        Caminho completo do arquivo JSON gerado.
    """
    if run_label is None:
        run_label = time.strftime("run_%Y%m%d_%H%M%S", time.localtime(start_time))

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{run_label}_report.json"

    _log.info("[Reporter] Gerando relatório completo...")

    report: Dict[str, Any] = {
        "metadata": _section_metadata(start_time, run_label, log_file),
        "config": _section_config(),
        "mechanisms": _section_mechanisms(),
        "data_splits": _section_data_splits(train_ds, test_ds, all_ds),
        "meta_dataset": _section_meta_dataset(selector),
        "meta_learner": _section_meta_learner(selector),
        "evaluation": _section_evaluation(results_df),
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)

    size_kb = out_path.stat().st_size / 1024
    _log.info("[Reporter] Relatório salvo em: %s  (%.1f KB)", out_path.resolve(), size_kb)
    return out_path

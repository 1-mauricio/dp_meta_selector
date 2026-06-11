"""Diagnósticos avançados para o meta-modelo DP.

Implementa métricas adicionais para análise detalhada do desempenho:
- F1-macro por família
- Confusion matrix do meta-modelo
- Calibration plot (confiança vs acurácia)
- K-fold cross-validation no nível de datasets
- Ablation study de meta-features
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import KFold

from .mechanisms import FAMILY_OF, MECHANISM_NAMES

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 4.1 Métricas de Avaliação por Família
# ---------------------------------------------------------------------------

def compute_family_f1_scores(
    results_df: pd.DataFrame,
) -> Dict[str, Dict[str, float]]:
    """Calcula F1-macro, precision e recall por família do mecanismo oracle.

    Parameters
    ----------
    results_df : pd.DataFrame
        DataFrame retornado por FrameworkEvaluator.evaluate().

    Returns
    -------
    Dict[str, Dict[str, float]]
        Dicionário com métricas por família.
    """
    df = results_df.copy()
    
    # Adiciona coluna de família
    df["oracle_family"] = df["best_mech"].apply(lambda m: FAMILY_OF.get(m, "unknown"))
    df["rec_family"] = df["rec_mech"].apply(lambda m: FAMILY_OF.get(m, "unknown"))
    
    families = sorted(df["oracle_family"].unique())
    
    metrics: Dict[str, Dict[str, float]] = {}
    
    for fam in families:
        mask = df["oracle_family"] == fam
        sub = df[mask]
        
        if len(sub) == 0:
            continue
        
        # Hit rate = recall da família
        hit_rate = sub["hit"].mean()
        
        # Precision: dos datasets onde recomendamos algo dessa família,
        # quantos realmente tinham oracle dessa família?
        rec_fam_mask = df["rec_family"] == fam
        if rec_fam_mask.sum() > 0:
            precision = (df[rec_fam_mask]["oracle_family"] == fam).mean()
        else:
            precision = 0.0
        
        # F1 da família
        if precision + hit_rate > 0:
            f1 = 2 * precision * hit_rate / (precision + hit_rate)
        else:
            f1 = 0.0
        
        metrics[fam] = {
            "n_samples": int(len(sub)),
            "precision": float(precision),
            "recall": float(hit_rate),
            "f1": float(f1),
            "hit_rate": float(hit_rate),
            "regret_mean": float(sub["regret"].mean()),
            "regret_std": float(sub["regret"].std()),
        }
    
    # F1-macro global (média das famílias)
    if metrics:
        f1_values = [m["f1"] for m in metrics.values()]
        metrics["_macro_avg"] = {
            "f1_macro": float(np.mean(f1_values)),
            "f1_weighted": float(
                np.average(
                    f1_values,
                    weights=[m["n_samples"] for m in metrics.values()]
                )
            ),
        }
    
    return metrics


def print_family_f1_report(results_df: pd.DataFrame) -> None:
    """Imprime relatório de F1 por família."""
    metrics = compute_family_f1_scores(results_df)
    
    print("\n" + "=" * 70)
    print("F1-SCORE POR FAMÍLIA")
    print("=" * 70)
    print(f"{'Família':<15} {'N':>5} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Regret':>10}")
    print("-" * 70)
    
    for fam, m in metrics.items():
        if fam == "_macro_avg":
            continue
        print(
            f"{fam:<15} {m['n_samples']:>5} "
            f"{m['precision']:>10.3f} {m['recall']:>10.3f} "
            f"{m['f1']:>10.3f} {m['regret_mean']:>10.4f}"
        )
    
    if "_macro_avg" in metrics:
        print("-" * 70)
        print(f"{'F1-Macro':>47}: {metrics['_macro_avg']['f1_macro']:.3f}")
        print(f"{'F1-Weighted':>47}: {metrics['_macro_avg']['f1_weighted']:.3f}")


# ---------------------------------------------------------------------------
# 4.1 Confusion Matrix do Meta-Modelo
# ---------------------------------------------------------------------------

def compute_confusion_matrix(
    results_df: pd.DataFrame,
    normalize: str = "true",
) -> Tuple[np.ndarray, List[str]]:
    """Calcula confusion matrix do meta-modelo.

    Parameters
    ----------
    results_df : pd.DataFrame
        DataFrame retornado por FrameworkEvaluator.evaluate().
    normalize : str
        'true' para normalizar por linha (recall), 'pred' por coluna (precision),
        'all' para normalizar pelo total, None para counts absolutos.

    Returns
    -------
    Tuple[np.ndarray, List[str]]
        Matriz de confusão e lista de labels.
    """
    y_true = results_df["best_mech"].values
    y_pred = results_df["rec_mech"].values
    
    # Labels únicos (sorted)
    labels = sorted(set(y_true) | set(y_pred))
    
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize=normalize)
    
    return cm, labels


def print_confusion_matrix(
    results_df: pd.DataFrame,
    normalize: str = "true",
) -> None:
    """Imprime confusion matrix formatada."""
    cm, labels = compute_confusion_matrix(results_df, normalize=normalize)
    
    print("\n" + "=" * 70)
    print(f"CONFUSION MATRIX (normalize='{normalize}')")
    print("=" * 70)
    print("Linhas = Oracle (best_mech), Colunas = Recomendado (rec_mech)\n")
    
    # Cabeçalho
    header = "              " + "".join(f"{lab[:8]:>10}" for lab in labels)
    print(header)
    print("-" * len(header))
    
    # Linhas
    for i, label in enumerate(labels):
        row_str = f"{label[:12]:<12} |"
        for j in range(len(labels)):
            val = cm[i, j]
            if normalize:
                row_str += f"{val:>10.2f}"
            else:
                row_str += f"{int(val):>10}"
        print(row_str)
    
    print()


def compute_classification_report(results_df: pd.DataFrame) -> Dict[str, Any]:
    """Gera classification report completo do meta-modelo."""
    y_true = results_df["best_mech"].values
    y_pred = results_df["rec_mech"].values
    
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    return report


# ---------------------------------------------------------------------------
# 4.1 Calibration Plot (Confiança vs Acurácia)
# ---------------------------------------------------------------------------

def compute_calibration_data(
    selector,
    test_datasets: List[Tuple[np.ndarray, np.ndarray, str]],
    n_bins: int = 10,
) -> Dict[str, Any]:
    """Calcula dados para calibration plot.

    Parameters
    ----------
    selector : DPMechanismSelector
        Selector treinado.
    test_datasets : List[Tuple]
        Lista de (X, y, name) para teste.
    n_bins : int
        Número de bins para calibration curve.

    Returns
    -------
    Dict[str, Any]
        Dados de calibração: fraction_of_positives, mean_predicted_value, etc.
    """
    from sklearn.preprocessing import LabelEncoder
    
    confidences = []
    hits = []
    
    for X, y, name in test_datasets:
        y_enc = LabelEncoder().fit_transform(y)
        
        # Pega predição com confiança
        rec = selector.recommend(X, y, verbose=False)
        rec_mech = rec["recommended_mechanism"]
        confidence = rec.get("confidence", 0.5)
        
        # Avalia se acertou
        ev = selector._evaluator
        dp_all = ev.evaluate_all(X, y_enc)
        best_mech = max(dp_all, key=dp_all.get)
        
        hit = int(rec_mech == best_mech)
        
        confidences.append(confidence)
        hits.append(hit)
    
    confidences = np.array(confidences)
    hits = np.array(hits)
    
    # Calibration curve
    try:
        fraction_of_positives, mean_predicted_value = calibration_curve(
            hits, confidences, n_bins=n_bins, strategy="uniform"
        )
    except ValueError:
        fraction_of_positives = np.array([])
        mean_predicted_value = np.array([])
    
    # Expected Calibration Error (ECE)
    if len(fraction_of_positives) > 0:
        bin_counts = np.histogram(confidences, bins=n_bins, range=(0, 1))[0]
        bin_weights = bin_counts / len(confidences)
        ece = np.sum(
            bin_weights[:len(fraction_of_positives)] * 
            np.abs(fraction_of_positives - mean_predicted_value)
        )
    else:
        ece = float("nan")
    
    return {
        "n_samples": len(confidences),
        "mean_confidence": float(confidences.mean()),
        "mean_accuracy": float(hits.mean()),
        "ece": float(ece),
        "fraction_of_positives": fraction_of_positives.tolist(),
        "mean_predicted_value": mean_predicted_value.tolist(),
        "confidence_histogram": np.histogram(confidences, bins=n_bins, range=(0, 1))[0].tolist(),
    }


def print_calibration_report(
    selector,
    test_datasets: List[Tuple[np.ndarray, np.ndarray, str]],
) -> None:
    """Imprime relatório de calibração."""
    data = compute_calibration_data(selector, test_datasets)
    
    print("\n" + "=" * 70)
    print("CALIBRATION REPORT")
    print("=" * 70)
    print(f"N datasets:          {data['n_samples']}")
    print(f"Mean confidence:     {data['mean_confidence']:.3f}")
    print(f"Mean accuracy:       {data['mean_accuracy']:.3f}")
    print(f"ECE (lower=better):  {data['ece']:.4f}")
    print()
    
    # Bins de calibração
    if data["fraction_of_positives"]:
        print("Calibration bins (predicted → actual):")
        for i, (pred, actual) in enumerate(zip(
            data["mean_predicted_value"],
            data["fraction_of_positives"]
        )):
            gap = actual - pred
            bar = "▓" * int(actual * 20) + "░" * (20 - int(actual * 20))
            print(f"  [{pred:.2f}] {bar} → {actual:.2f} (gap={gap:+.2f})")


# ---------------------------------------------------------------------------
# 4.2 K-Fold Cross-Validation no Nível de Datasets
# ---------------------------------------------------------------------------

def dataset_level_kfold_cv(
    all_datasets: List[Tuple[np.ndarray, np.ndarray, str]],
    n_splits: int = 5,
    seed: int = 42,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Executa k-fold CV no nível de datasets (não no meta-modelo).

    Em vez de dividir os dados dentro de cada dataset, divide os datasets
    entre treino e teste k vezes.

    Parameters
    ----------
    all_datasets : List[Tuple]
        Lista completa de (X, y, name).
    n_splits : int
        Número de folds.
    seed : int
        Random seed.
    verbose : bool
        Se True, imprime progresso.

    Returns
    -------
    Dict[str, Any]
        Métricas agregadas por fold e estatísticas gerais.
    """
    from .selector import DPMechanismSelector
    from .utility import DPUtilityEvaluator
    from sklearn.preprocessing import LabelEncoder
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    
    fold_results = []
    indices = np.arange(len(all_datasets))
    
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(indices)):
        if verbose:
            print(f"\n[K-Fold] Fold {fold_idx + 1}/{n_splits} "
                  f"(train={len(train_idx)}, test={len(test_idx)})")
        
        train_ds = [all_datasets[i] for i in train_idx]
        test_ds = [all_datasets[i] for i in test_idx]
        
        # Treina selector neste fold
        selector = DPMechanismSelector(fast_mode=True)
        selector.fit(train_ds)
        
        # Avalia
        hits = []
        regrets = []
        
        for X, y, name in test_ds:
            y_enc = LabelEncoder().fit_transform(y)
            
            rec = selector.recommend(X, y, verbose=False)
            rec_mech = rec["recommended_mechanism"]
            
            ev = selector._evaluator
            dp_all = ev.evaluate_all(X, y_enc)
            best_mech = max(dp_all, key=dp_all.get)
            best_acc = dp_all[best_mech]
            rec_acc = dp_all[rec_mech]
            
            hits.append(int(rec_mech == best_mech))
            regrets.append(best_acc - rec_acc)
        
        fold_results.append({
            "fold": fold_idx + 1,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "hit_rate": float(np.mean(hits)),
            "regret_mean": float(np.mean(regrets)),
            "regret_std": float(np.std(regrets)),
        })
        
        if verbose:
            print(f"    Hit rate: {np.mean(hits):.3f}  Regret: {np.mean(regrets):.4f}")
    
    # Estatísticas agregadas
    hit_rates = [r["hit_rate"] for r in fold_results]
    regrets = [r["regret_mean"] for r in fold_results]
    
    summary = {
        "n_folds": n_splits,
        "n_datasets": len(all_datasets),
        "hit_rate_mean": float(np.mean(hit_rates)),
        "hit_rate_std": float(np.std(hit_rates)),
        "hit_rate_min": float(np.min(hit_rates)),
        "hit_rate_max": float(np.max(hit_rates)),
        "regret_mean": float(np.mean(regrets)),
        "regret_std": float(np.std(regrets)),
        "fold_results": fold_results,
    }
    
    if verbose:
        print("\n" + "=" * 70)
        print(f"K-FOLD CROSS-VALIDATION (k={n_splits})")
        print("=" * 70)
        print(f"Hit rate: {summary['hit_rate_mean']:.3f} ± {summary['hit_rate_std']:.3f} "
              f"(range: [{summary['hit_rate_min']:.3f}, {summary['hit_rate_max']:.3f}])")
        print(f"Regret:   {summary['regret_mean']:.4f} ± {summary['regret_std']:.4f}")
    
    return summary


# ---------------------------------------------------------------------------
# 4.3 Ablation Study de Meta-Features
# ---------------------------------------------------------------------------

def ablation_study(
    train_datasets: List[Tuple[np.ndarray, np.ndarray, str]],
    test_datasets: List[Tuple[np.ndarray, np.ndarray, str]],
    feature_groups: Optional[Dict[str, List[str]]] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Executa ablation study removendo grupos de meta-features.

    Parameters
    ----------
    train_datasets : List[Tuple]
        Datasets de treino.
    test_datasets : List[Tuple]
        Datasets de teste.
    feature_groups : Dict[str, List[str]], optional
        Mapeamento de nome do grupo para lista de prefixos de features.
        Se None, usa grupos padrão.
    verbose : bool
        Se True, imprime progresso.

    Returns
    -------
    Dict[str, Any]
        Impacto de cada grupo de features.
    """
    from .selector import DPMechanismSelector
    from sklearn.preprocessing import LabelEncoder
    
    if feature_groups is None:
        feature_groups = {
            "statistical": ["n_samples", "n_features", "n_classes", "mean_", "std_", "min_", "max_"],
            "categorical": ["cat_", "ratio_low_cardinality", "entropy_"],
            "discrete": ["disc_", "ratio_integer"],
            "family_discriminators": ["fam_"],
            "landmarks": ["landmark_", "lm_"],
            "correlation": ["mean_corr", "corr_"],
            "pca": ["pca_"],
            "sensitivity": ["sensitivity", "mean_sensitivity", "max_sensitivity"],
            "information": ["mean_mi", "mi_"],
        }
    
    def _evaluate_selector(selector, test_ds):
        hits = []
        for X, y, name in test_ds:
            y_enc = LabelEncoder().fit_transform(y)
            rec = selector.recommend(X, y, verbose=False)
            rec_mech = rec["recommended_mechanism"]
            ev = selector._evaluator
            dp_all = ev.evaluate_all(X, y_enc)
            best_mech = max(dp_all, key=dp_all.get)
            hits.append(int(rec_mech == best_mech))
        return float(np.mean(hits))
    
    # Baseline: todas as features
    if verbose:
        print("\n" + "=" * 70)
        print("ABLATION STUDY")
        print("=" * 70)
        print("Treinando baseline (todas as features)...")
    
    selector_baseline = DPMechanismSelector(fast_mode=True)
    selector_baseline.fit(train_datasets)
    baseline_hit = _evaluate_selector(selector_baseline, test_datasets)
    
    if verbose:
        print(f"Baseline hit rate: {baseline_hit:.3f}")
        print("\nRemovendo grupos de features...")
    
    results = {"baseline": baseline_hit, "ablations": {}}
    
    for group_name, prefixes in feature_groups.items():
        if verbose:
            print(f"  - {group_name}...", end=" ", flush=True)
        
        # Treina sem esse grupo
        selector_ablated = DPMechanismSelector(fast_mode=True)
        selector_ablated.fit(train_datasets)
        
        # Remove features do grupo
        if selector_ablated.meta_df is not None and selector_ablated._learner.META_FEATURE_COLS:
            cols_to_remove = []
            for col in selector_ablated._learner.META_FEATURE_COLS:
                for prefix in prefixes:
                    if col.startswith(prefix) or prefix in col:
                        cols_to_remove.append(col)
                        break
            
            # Zera as colunas removidas (simula remoção)
            if cols_to_remove:
                for col in cols_to_remove:
                    if col in selector_ablated.meta_df.columns:
                        selector_ablated.meta_df[col] = 0.0
        
        ablated_hit = _evaluate_selector(selector_ablated, test_datasets)
        delta = ablated_hit - baseline_hit
        
        results["ablations"][group_name] = {
            "hit_rate": ablated_hit,
            "delta": delta,
            "impact": "negative" if delta < -0.01 else ("positive" if delta > 0.01 else "neutral"),
        }
        
        if verbose:
            sign = "+" if delta >= 0 else ""
            print(f"{ablated_hit:.3f} ({sign}{delta:.3f})")
    
    # Ordena por impacto
    sorted_groups = sorted(
        results["ablations"].items(),
        key=lambda x: x[1]["delta"],
    )
    
    if verbose:
        print("\n" + "-" * 70)
        print("RANKING DE IMPORTÂNCIA (grupos mais importantes primeiro):")
        for i, (name, data) in enumerate(sorted_groups, 1):
            impact_str = "⬆️" if data["delta"] < -0.01 else ("⬇️" if data["delta"] > 0.01 else "➡️")
            print(f"  {i}. {name:<25} {impact_str} {data['delta']:+.3f}")
    
    return results


# ---------------------------------------------------------------------------
# Relatório Consolidado
# ---------------------------------------------------------------------------

def run_full_diagnostics(
    selector,
    results_df: pd.DataFrame,
    train_datasets: List[Tuple[np.ndarray, np.ndarray, str]],
    test_datasets: List[Tuple[np.ndarray, np.ndarray, str]],
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Executa todos os diagnósticos e opcionalmente salva em arquivo.

    Parameters
    ----------
    selector : DPMechanismSelector
        Selector treinado.
    results_df : pd.DataFrame
        Resultados da avaliação.
    train_datasets, test_datasets : List[Tuple]
        Datasets usados.
    output_dir : Path, optional
        Se fornecido, salva relatório JSON.

    Returns
    -------
    Dict[str, Any]
        Todos os diagnósticos.
    """
    import json
    import time
    
    print("\n" + "=" * 70)
    print("EXECUTANDO DIAGNÓSTICOS COMPLETOS")
    print("=" * 70)
    
    diagnostics: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    
    # 4.1 F1 por família
    print("\n1/4 Calculando F1 por família...")
    diagnostics["family_f1"] = compute_family_f1_scores(results_df)
    print_family_f1_report(results_df)
    
    # 4.1 Confusion matrix
    print("\n2/4 Gerando confusion matrix...")
    cm, labels = compute_confusion_matrix(results_df, normalize="true")
    diagnostics["confusion_matrix"] = {
        "matrix": cm.tolist(),
        "labels": labels,
    }
    print_confusion_matrix(results_df)
    
    # 4.1 Calibration
    print("\n3/4 Calculando métricas de calibração...")
    diagnostics["calibration"] = compute_calibration_data(selector, test_datasets)
    print_calibration_report(selector, test_datasets)
    
    # 4.1 Classification report
    diagnostics["classification_report"] = compute_classification_report(results_df)
    
    # Não roda k-fold e ablation por padrão (são caros)
    print("\n4/4 Diagnósticos completos (k-fold e ablation não executados por padrão)")
    print("    Use dataset_level_kfold_cv() e ablation_study() separadamente se necessário.")
    
    # Salva se output_dir fornecido
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / "diagnostics.json"
        
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(diagnostics, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"\nDiagnósticos salvos em: {out_path}")
    
    return diagnostics

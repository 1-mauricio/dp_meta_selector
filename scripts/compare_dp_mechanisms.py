#!/usr/bin/env python3
"""
Comparação direta de mecanismos DP em datasets reais.

Este script avalia todos os mecanismos DP em cada dataset e mostra:
1. Qual mecanismo é melhor para cada dataset
2. Qual a diferença de performance entre mecanismos
3. Se existe um "melhor mecanismo universal" ou se depende do dataset

Uso:
    python scripts/compare_dp_mechanisms.py [--n-datasets 50] [--output results.csv]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

# Adiciona o diretório pai ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dp_meta_selector.datasets import load_openml_training_datasets
from dp_meta_selector.mechanisms import FAMILY_OF
from dp_meta_selector.calibration import FAMILY_EPSILON
from dp_meta_selector.applicator import DPApplicator


def evaluate_mechanism(X: np.ndarray, y: np.ndarray, mech_name: str, applicator: DPApplicator, n_runs: int = 3) -> float:
    """Avalia um mecanismo DP em um dataset usando cross-validation."""
    accs = []
    for _ in range(n_runs):
        try:
            X_noisy = applicator.apply(mech_name, X.copy())
            clf = LogisticRegression(max_iter=500, random_state=42)
            scores = cross_val_score(clf, X_noisy, y, cv=3, scoring="accuracy")
            accs.append(scores.mean())
        except Exception:
            accs.append(0.0)
    return float(np.mean(accs))


def evaluate_baseline(X: np.ndarray, y: np.ndarray) -> float:
    """Avalia acurácia sem DP (baseline)."""
    try:
        clf = LogisticRegression(max_iter=500, random_state=42)
        scores = cross_val_score(clf, X, y, cv=3, scoring="accuracy")
        return float(scores.mean())
    except Exception:
        return 0.0


def main():
    parser = argparse.ArgumentParser(description="Compara mecanismos DP em datasets reais")
    parser.add_argument("--n-datasets", type=int, default=500, help="Número de datasets a avaliar")
    parser.add_argument("--output", type=str, default="dp_comparison.csv", help="Arquivo de saída")
    parser.add_argument("--n-runs", type=int, default=3, help="Número de runs por mecanismo")
    args = parser.parse_args()

    print("=" * 80)
    print("  COMPARAÇÃO DIRETA DE MECANISMOS DP")
    print("=" * 80)
    print(f"\nCarregando datasets...")
    
    datasets = load_openml_training_datasets()
    datasets = datasets[:args.n_datasets]
    print(f"Avaliando {len(datasets)} datasets\n")
    
    # Mecanismos principais (representantes de cada família)
    mech_names = ["Laplace", "GaussianAnalytic", "Exponential"]
    applicator = DPApplicator()
    
    print(f"Mecanismos: {mech_names}")
    print(f"Epsilon por família: {FAMILY_EPSILON}\n")
    
    results = []
    
    for i, (X, y, name) in enumerate(datasets):
        y_enc = LabelEncoder().fit_transform(y)
        
        print(f"[{i+1}/{len(datasets)}] {name[:40]:<40}", end=" ", flush=True)
        
        row = {"dataset": name, "n_samples": X.shape[0], "n_features": X.shape[1]}
        
        # Baseline sem DP
        base_acc = evaluate_baseline(X, y_enc)
        row["baseline_no_dp"] = base_acc
        
        # Cada mecanismo
        best_acc = 0
        best_mech = None
        
        for mech_name in mech_names:
            acc = evaluate_mechanism(X, y_enc, mech_name, applicator, n_runs=args.n_runs)
            row[f"acc_{mech_name}"] = acc
            
            if acc > best_acc:
                best_acc = acc
                best_mech = mech_name
        
        row["best_mechanism"] = best_mech
        row["best_acc"] = best_acc
        row["laplace_acc"] = row["acc_Laplace"]
        
        # Métricas derivadas
        row["gap_vs_laplace"] = best_acc - row["laplace_acc"]
        row["gap_vs_baseline"] = base_acc - best_acc  # Quanto DP perde
        row["relative_to_baseline"] = best_acc / (base_acc + 1e-9)
        
        results.append(row)
        
        # Print inline
        gap = row["gap_vs_laplace"]
        symbol = "✓" if gap > 0.001 else "=" if abs(gap) < 0.001 else "✗"
        print(f"best={best_mech:<18} gap={gap:+.3f} {symbol}")
    
    # Cria DataFrame
    df = pd.DataFrame(results)
    
    # Salva CSV
    df.to_csv(args.output, index=False)
    print(f"\nResultados salvos em: {args.output}")
    
    # ========== ANÁLISE ==========
    print("\n" + "=" * 80)
    print("  ANÁLISE DOS RESULTADOS")
    print("=" * 80)
    
    # 1. Distribuição do melhor mecanismo
    print("\n1. QUAL MECANISMO É MELHOR COM MAIS FREQUÊNCIA?")
    print("-" * 50)
    best_counts = df["best_mechanism"].value_counts()
    total = len(df)
    for mech, count in best_counts.items():
        pct = count / total * 100
        bar = "█" * int(pct / 2)
        print(f"   {mech:<20} {count:>3} ({pct:>5.1f}%) {bar}")
    
    # 2. Gap médio vs Laplace
    print("\n2. QUANTO O MELHOR MECANISMO SUPERA LAPLACE?")
    print("-" * 50)
    gaps = df["gap_vs_laplace"]
    print(f"   Gap médio:     {gaps.mean():+.4f} ({gaps.mean()*100:+.2f}pp)")
    print(f"   Gap mediano:   {gaps.median():+.4f}")
    print(f"   Gap máximo:    {gaps.max():+.4f}")
    print(f"   Gap mínimo:    {gaps.min():+.4f}")
    print(f"   Desvio padrão: {gaps.std():.4f}")
    
    # 3. Em quantos datasets cada mecanismo supera Laplace
    print("\n3. EM QUANTOS DATASETS CADA MECANISMO SUPERA LAPLACE?")
    print("-" * 50)
    laplace_acc = df["acc_Laplace"]
    for mech in mech_names:
        if mech == "Laplace":
            continue
        mech_acc = df[f"acc_{mech}"]
        n_better = (mech_acc > laplace_acc + 0.001).sum()
        n_equal = ((mech_acc >= laplace_acc - 0.001) & (mech_acc <= laplace_acc + 0.001)).sum()
        n_worse = (mech_acc < laplace_acc - 0.001).sum()
        print(f"   {mech:<20} melhor={n_better:>3}  igual={n_equal:>3}  pior={n_worse:>3}")
    
    # 4. Performance por tipo de dataset (baseado em features)
    print("\n4. EXISTE PADRÃO POR TIPO DE DATASET?")
    print("-" * 50)
    
    # Classifica datasets por número de features
    df["size_category"] = pd.cut(df["n_features"], bins=[0, 10, 50, 1000], labels=["small", "medium", "large"])
    for cat in ["small", "medium", "large"]:
        sub = df[df["size_category"] == cat]
        if len(sub) > 0:
            best_dist = sub["best_mechanism"].value_counts(normalize=True)
            print(f"\n   {cat.upper()} datasets (n_features {'<10' if cat=='small' else '10-50' if cat=='medium' else '>50'})  n={len(sub)}")
            for mech, pct in best_dist.items():
                print(f"      {mech:<20} {pct*100:>5.1f}%")
    
    # 5. Impacto do DP na acurácia
    print("\n5. QUANTO O DP REDUZ A ACURÁCIA?")
    print("-" * 50)
    drop = df["gap_vs_baseline"]
    print(f"   Queda média com DP:   {drop.mean():.4f} ({drop.mean()*100:.2f}pp)")
    print(f"   Queda mediana:        {drop.median():.4f}")
    print(f"   Melhor caso (menor queda): {drop.min():.4f}")
    print(f"   Pior caso (maior queda):   {drop.max():.4f}")
    
    # 6. Conclusão
    print("\n" + "=" * 80)
    print("  CONCLUSÃO")
    print("=" * 80)
    
    # Calcula se vale a pena um seletor
    n_laplace_best = (df["best_mechanism"] == "Laplace").sum()
    n_other_best = total - n_laplace_best
    if n_other_best > 0:
        avg_gap_when_other_wins = df[df["best_mechanism"] != "Laplace"]["gap_vs_laplace"].mean()
    else:
        avg_gap_when_other_wins = 0.0
    
    print(f"""
   • Laplace é o melhor em {n_laplace_best}/{total} ({n_laplace_best/total*100:.1f}%) dos datasets
   • Outros mecanismos são melhores em {n_other_best}/{total} ({n_other_best/total*100:.1f}%) dos datasets
   • Quando outro mecanismo vence, o ganho médio é {avg_gap_when_other_wins:+.4f} ({avg_gap_when_other_wins*100:+.2f}pp)
   
   JUSTIFICATIVA DO FRAMEWORK:
   ─────────────────────────────
   Se sempre usássemos Laplace, perderíamos {avg_gap_when_other_wins*n_other_best/total*100:.2f}pp de acurácia
   em média por não escolher o mecanismo correto.
   
   Um seletor perfeito (oracle) daria ganho de {gaps.mean()*100:.2f}pp sobre Laplace fixo.
    """)
    
    # 7. Tabela de exemplos
    print("\n6. EXEMPLOS DE DATASETS ONDE LAPLACE NÃO É IDEAL")
    print("-" * 80)
    non_laplace = df[df["best_mechanism"] != "Laplace"]
    if len(non_laplace) > 0:
        top_gaps = non_laplace.nlargest(min(10, len(non_laplace)), "gap_vs_laplace")[
            ["dataset", "best_mechanism", "gap_vs_laplace", "acc_Laplace", "best_acc"]
        ]
        print(top_gaps.to_string(index=False))
    else:
        print("   (Laplace foi o melhor em todos os datasets)")
    
    return df


if __name__ == "__main__":
    main()

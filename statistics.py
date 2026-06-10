"""
Análise estatística: impacto dos mecanismos DP na acurácia por dataset.

Execução:
    python statistics.py
    python statistics.py --repeats 30 --datasets iris breast_cancer wine digits synthetic

Testes realizados:
  1. Estatísticas descritivas (média ± dp por mecanismo × dataset)
  2. Teste de Friedman por dataset
     H₀: todos os mecanismos têm a mesma distribuição de acurácia
  3. Post-hoc Nemenyi (pares com diferença significativa, α=0.05)
  4. Two-way ANOVA (mecanismo × dataset) — efeito de interação
  5. Ranking dos mecanismos por dataset + Kendall's W
  6. Delta de acurácia vs baseline (média e desvio entre datasets)

Dependências extras (além das do projeto):
    pip install scikit-posthocs pingouin
"""

import argparse
import logging
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.datasets import (
    load_breast_cancer,
    load_digits,
    load_iris,
    load_wine,
    make_classification,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from dp_meta_selector.applicator import DPApplicator
from dp_meta_selector.calibration import DELTA_DEFAULT

warnings.filterwarnings("ignore")
_log = logging.getLogger(__name__)

# ── Constantes ────────────────────────────────────────────────────────────────
DEFAULT_MECHS: List[str] = [
    "Laplace",
    "LaplaceTruncated",
    "GaussianAnalytic",
    "Geometric",
    "Exponential",
]

ALPHA = 0.05  # nível de significância


# ── Datasets disponíveis ──────────────────────────────────────────────────────
def _load_datasets(names: List[str]) -> List[Tuple[str, np.ndarray, np.ndarray]]:
    available = {
        "iris": lambda: (
            "Iris",
            *[v for v in [load_iris().data, load_iris().target]],
        ),
        "breast_cancer": lambda: (
            "BreastCancer",
            load_breast_cancer().data,
            load_breast_cancer().target,
        ),
        "wine": lambda: ("Wine", load_wine().data, load_wine().target),
        "digits": lambda: _digits_subset(),
        "synthetic": lambda: _synthetic(),
    }
    result = []
    for name in names:
        key = name.lower()
        if key not in available:
            _log.warning("Dataset '%s' desconhecido — ignorado.", name)
            continue
        result.append(available[key]())
    return result


def _digits_subset(n: int = 600, seed: int = 42):
    raw = load_digits()
    idx = np.random.RandomState(seed).choice(len(raw.target), n, replace=False)
    return f"Digits{n}", raw.data[idx], raw.target[idx]


def _synthetic(n: int = 800, seed: int = 42):
    X, y = make_classification(
        n_samples=n,
        n_features=20,
        n_informative=10,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=seed,
    )
    return f"Synthetic{n}", X, y


# ── Coleta de scores ──────────────────────────────────────────────────────────
def collect_scores(
    datasets: List[Tuple],
    mechs: List[str],
    n_repeats: int = 30,
    n_splits: int = 5,
    n_estimators: int = 50,
    delta: float = DELTA_DEFAULT,
) -> pd.DataFrame:
    """Avalia baseline + cada mecanismo DP em cada dataset com n_repeats repetições."""
    applicator = DPApplicator(delta=delta)
    records = []

    for ds_name, X_raw, y in datasets:
        X_s = StandardScaler().fit_transform(X_raw.astype(float))
        for mech in ["baseline"] + mechs:
            for rep in range(n_repeats):
                if mech == "baseline":
                    X_use = X_s
                else:
                    X_use = applicator.apply(mech, X_s.copy())

                skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=rep)
                fold_scores = [
                    RandomForestClassifier(n_estimators=n_estimators, random_state=rep)
                    .fit(X_use[tr], y[tr])
                    .score(X_use[te], y[te])
                    for tr, te in skf.split(X_use, y)
                ]
                records.append(
                    {
                        "dataset": ds_name,
                        "mecanismo": mech,
                        "rep": rep,
                        "acc": float(np.mean(fold_scores)),
                    }
                )
            _log.info("  ✓ %-15s %s", ds_name, mech)

    return pd.DataFrame(records)


# ── Testes estatísticos ───────────────────────────────────────────────────────
def section(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def test_descriptive(df: pd.DataFrame, mechs: List[str]) -> None:
    section("1. ESTATÍSTICAS DESCRITIVAS (média ± dp)")
    order = ["baseline"] + mechs
    pivot_m = df.pivot_table(index="mecanismo", columns="dataset", values="acc", aggfunc="mean")
    pivot_s = df.pivot_table(index="mecanismo", columns="dataset", values="acc", aggfunc="std")
    pivot_m = pivot_m.reindex([m for m in order if m in pivot_m.index])
    pivot_s = pivot_s.reindex([m for m in order if m in pivot_s.index])
    combined = pivot_m.round(4).astype(str) + " ±" + pivot_s.round(4).astype(str)
    print(combined.to_string())


def test_friedman(df: pd.DataFrame, mechs: List[str]) -> Dict[str, float]:
    """Teste de Friedman por dataset. Retorna dict {dataset: p-value}."""
    section(
        "2. TESTE DE FRIEDMAN POR DATASET\n"
        "   H₀: todos os mecanismos têm a mesma distribuição de acurácia"
    )
    pvalues = {}
    for ds in df["dataset"].unique():
        sub = df[(df["dataset"] == ds) & (df["mecanismo"] != "baseline")]
        groups = [sub[sub["mecanismo"] == m]["acc"].values for m in mechs]
        stat, p = stats.friedmanchisquare(*groups)
        sig = "✓ SIGNIFICATIVO" if p < ALPHA else "✗ não significativo"
        pvalues[ds] = p
        print(f"  {ds:<15}  χ²={stat:7.2f}  p={p:.4e}  {sig}")
    return pvalues


def test_nemenyi(df: pd.DataFrame, mechs: List[str], friedman_pvalues: Dict[str, float]) -> None:
    """Post-hoc Nemenyi nos datasets com Friedman significativo."""
    try:
        import scikit_posthocs as sp
    except ImportError:
        print("\n[AVISO] scikit-posthocs não instalado — post-hoc Nemenyi pulado.")
        print("         pip install scikit-posthocs")
        return

    section(
        "3. POST-HOC NEMENYI (pares com diferença significativa, α=0.05)\n"
        "   Aplicado apenas nos datasets com Friedman significativo"
    )
    for ds, p in friedman_pvalues.items():
        if p >= ALPHA:
            print(f"\n  {ds}: Friedman não significativo — post-hoc não aplicável.")
            continue
        sub = df[(df["dataset"] == ds) & (df["mecanismo"] != "baseline")]
        matrix = sub.pivot_table(index="rep", columns="mecanismo", values="acc")[mechs]
        nemenyi = sp.posthoc_nemenyi_friedman(matrix.values)
        nemenyi.index = mechs
        nemenyi.columns = mechs
        print(f"\n  {ds}:")
        found = False
        for i, m1 in enumerate(mechs):
            for j, m2 in enumerate(mechs):
                if j > i:
                    pv = nemenyi.loc[m1, m2]
                    if pv < ALPHA:
                        print(f"    {m1:<20} vs {m2:<20}  p={pv:.4f}  ← diferença significativa")
                        found = True
        if not found:
            print("    Nenhum par significativo.")


def test_anova(df: pd.DataFrame) -> None:
    """Two-way ANOVA mecanismo × dataset."""
    try:
        import pingouin as pg
    except ImportError:
        print("\n[AVISO] pingouin não instalado — ANOVA pulada.")
        print("         pip install pingouin")
        return

    section(
        "4. TWO-WAY ANOVA  (mecanismo × dataset)\n"
        "   Testa se o efeito do mecanismo DEPENDE do dataset (interação)"
    )
    sub = df[df["mecanismo"] != "baseline"].copy()
    aov = pg.anova(data=sub, dv="acc", between=["mecanismo", "dataset"], effsize="np2")
    print(aov[["Source", "F", "p_unc", "np2"]].to_string(index=False))

    inter = aov[aov["Source"] == "mecanismo * dataset"]
    if inter.empty:
        return
    inter_p   = float(inter["p_unc"].values[0])
    inter_f   = float(inter["F"].values[0])
    inter_np2 = float(inter["np2"].values[0])

    print(f"\n  Interação:  F={inter_f:.2f}  p={inter_p:.4e}  η²p={inter_np2:.3f}")
    if inter_p < ALPHA:
        print("  → INTERAÇÃO SIGNIFICATIVA: o impacto do mecanismo varia entre datasets.")
        label = (
            "GRANDE" if inter_np2 > 0.14 else
            "MÉDIO"  if inter_np2 > 0.06 else
            "PEQUENO"
        )
        print(f"  → Tamanho do efeito {label} (η²p={inter_np2:.3f})")
    else:
        print("  → Interação não significativa: efeito uniforme entre datasets.")


def test_ranking(df: pd.DataFrame, mechs: List[str]) -> None:
    """Ranking por dataset + Kendall's W."""
    section("5. RANKING DOS MECANISMOS POR DATASET  (1 = melhor)")
    ds_names = df["dataset"].unique()
    ranking: Dict[str, List[str]] = {}

    for ds in ds_names:
        sub = df[(df["dataset"] == ds) & (df["mecanismo"] != "baseline")]
        means = sub.groupby("mecanismo")["acc"].mean().sort_values(ascending=False)
        ranking[ds] = list(means.index)
        base_acc = df[(df["dataset"] == ds) & (df["mecanismo"] == "baseline")]["acc"].mean()
        print(f"\n  {ds}:")
        for i, m in enumerate(means.index, 1):
            ret = means[m] / base_acc
            bar = "█" * int(ret * 25)
            print(f"    {i}. {m:<20} {means[m]:.4f}  ret={ret:.2%}  {bar}")

    # Kendall's W
    rank_matrix = pd.DataFrame(
        {ds: {m: ranking[ds].index(m) + 1 for m in mechs if m in ranking[ds]}
         for ds in ds_names}
    ).T.values.astype(float)
    n_r, n_c = rank_matrix.shape
    S = np.sum((rank_matrix - rank_matrix.mean(axis=0)) ** 2)
    W = 12 * S / (n_r**2 * (n_c**3 - n_c))
    label = (
        "ranking MUITO CONSISTENTE entre datasets"        if W > 0.7 else
        "ranking MODERADAMENTE consistente"               if W > 0.4 else
        "ranking INCONSISTENTE — varia significativamente por dataset"
    )
    print(f"\n  Kendall's W = {W:.3f}  →  {label}")


def test_delta(df: pd.DataFrame, mechs: List[str]) -> None:
    """Delta de acurácia vs baseline, média e desvio entre datasets."""
    section(
        "6. DELTA DE ACURÁCIA vs BASELINE (por mecanismo)\n"
        "   DP alto → mecanismo afeta datasets de forma muito diferente"
    )
    baselines = df[df["mecanismo"] == "baseline"].groupby("dataset")["acc"].mean()
    rows = []
    for mech in mechs:
        deltas = []
        for ds in baselines.index:
            m_acc = df[(df["dataset"] == ds) & (df["mecanismo"] == mech)]["acc"].mean()
            deltas.append(m_acc - baselines[ds])
        rows.append(
            {
                "mecanismo": mech,
                "delta_médio": np.mean(deltas),
                "dp_entre_datasets": np.std(deltas),
                "min": np.min(deltas),
                "max": np.max(deltas),
            }
        )
    delta_df = pd.DataFrame(rows).sort_values("delta_médio", ascending=False)
    print(delta_df.round(4).to_string(index=False))
    print("\n  DP alto = mecanismo comporta-se de forma heterogênea entre datasets.")


# ── Ponto de entrada ──────────────────────────────────────────────────────────
def run(
    dataset_names: List[str] = None,
    mechs: List[str] = None,
    n_repeats: int = 30,
) -> pd.DataFrame:
    """Executa a análise completa e retorna o DataFrame de scores."""
    dataset_names = dataset_names or ["iris", "breast_cancer", "wine", "digits", "synthetic"]
    mechs = mechs or DEFAULT_MECHS

    datasets = _load_datasets(dataset_names)
    if not datasets:
        raise ValueError("Nenhum dataset válido encontrado.")

    print(f"Coletando scores ({n_repeats} repetições × {len(datasets)} datasets × {len(mechs)+1} condições)...")
    df = collect_scores(datasets, mechs, n_repeats=n_repeats)
    print(f"Total de observações: {len(df)}\n")

    test_descriptive(df, mechs)
    friedman_p = test_friedman(df, mechs)
    test_nemenyi(df, mechs, friedman_p)
    test_anova(df)
    test_ranking(df, mechs)
    test_delta(df, mechs)

    print("\n" + "=" * 72)
    print("FIM — todos os testes concluídos")
    print("=" * 72)

    return df


def _parse_args():
    p = argparse.ArgumentParser(description="Análise estatística dos mecanismos DP.")
    p.add_argument("--repeats", type=int, default=30,
                   help="Número de repetições por condição (padrão: 30)")
    p.add_argument("--datasets", nargs="+",
                   default=["iris", "breast_cancer", "wine", "digits", "synthetic"],
                   help="Datasets a usar")
    p.add_argument("--mechs", nargs="+", default=None,
                   help="Mecanismos a testar (padrão: todos os principais)")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[%(levelname)s] %(message)s",
    )
    run(dataset_names=args.datasets, mechs=args.mechs, n_repeats=args.repeats)

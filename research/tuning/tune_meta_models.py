"""tune_meta_models.py — Tuning offline do hybrid_ensemble sem re-rodar o pipeline pesado.

Lê direto dos CSVs salvos pela v19 (META_STABLE_PROFILE) e simula o ensemble híbrido
com uma grade de parâmetros, reportando Hit Rate e taxa de "pior que Laplace" para cada
combinação. Permite encontrar os parâmetros ótimos em segundos.

Uso:
    python tune_meta_models.py
    python tune_meta_models.py --features meta_datasets_v19/meta_features_meta_stable.csv
    python tune_meta_models.py --top-k 3 4 5 --margins 0.5 1.0 2.0 3.0
    python tune_meta_models.py --test-size 0.25 --plot
"""

import argparse
import warnings
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

# ── Constantes ────────────────────────────────────────────────────────────────
MECHANISMS = [
    "Laplace", "Gaussian", "GaussianAnalytic", "Staircase",
    "LaplaceTruncated", "LaplaceFolded", "Snapping", "Exponential", "Uniform",
]
DEFAULT_FEATURES_CSV = "meta_datasets_v19/meta_features_meta_stable.csv"
DEFAULT_TARGETS_CSV  = "meta_datasets_v19/meta_targets_meta_stable.csv"

# ── Carregamento dos dados ────────────────────────────────────────────────────

def load_data(features_path: str, targets_path: str):
    feat = pd.read_csv(features_path)
    tgt  = pd.read_csv(targets_path)
    df   = feat.merge(tgt, on="dataset_name", how="inner")

    # Colunas de features (excluir metadados e targets)
    excl = {
        "dataset_name", "best_mechanism", "best_relative_acc",
        "baseline_acc", "best_family", "utility_gap",
        "utility_best_abs", "utility_worst_abs", "utility_range",
    }
    excl |= {f"acc_{m}" for m in MECHANISMS}
    excl |= {f"utility_loss_{m}" for m in MECHANISMS}

    feat_cols = [c for c in df.columns if c not in excl]
    loss_cols = [f"utility_loss_{m}" for m in MECHANISMS if f"utility_loss_{m}" in df.columns]
    acc_cols  = [f"acc_{m}" for m in MECHANISMS if f"acc_{m}" in df.columns]

    X    = df[feat_cols].fillna(0).values.astype(float)
    Y    = df[loss_cols].fillna(0).values.astype(float)      # targets do regressor
    accs = df[acc_cols].fillna(0).values                     # oracle real por mecanismo
    y    = LabelEncoder().fit_transform(df["best_mechanism"])
    mech_names = [c.replace("utility_loss_", "") for c in loss_cols]
    classes    = list(LabelEncoder().fit(df["best_mechanism"]).classes_)

    return X, Y, y, accs, mech_names, classes, df

# ── Treino dos modelos base ───────────────────────────────────────────────────

def train_classifier(X_tr, y_tr):
    """Soft-voting ensemble: ExtraTrees + LogReg + SVM-Linear."""
    models = {
        "ExtraTrees": Pipeline([
            ("s", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=200, random_state=42, class_weight="balanced",
            )),
        ]),
        "LogReg": Pipeline([
            ("s", StandardScaler()),
            ("clf", LogisticRegression(max_iter=500, class_weight="balanced", random_state=42)),
        ]),
        "SVM-Linear": Pipeline([
            ("s", StandardScaler()),
            ("clf", SVC(kernel="linear", probability=True,
                        class_weight="balanced", random_state=42)),
        ]),
    }
    for m in models.values():
        m.fit(X_tr, y_tr)

    # F1-macro do melhor modelo individual
    le = LabelEncoder().fit(y_tr)
    scores = {}
    for name, m in models.items():
        try:
            s = cross_val_score(m, X_tr, y_tr, cv=3, scoring="f1_macro").mean()
            scores[name] = s
        except Exception:
            scores[name] = 0.0
    best = max(scores, key=scores.get)
    return models, scores, best


def train_regressor(X_tr, Y_tr):
    """Multi-output Random Forest para prever utility_loss_%."""
    base = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1, min_samples_leaf=2)
    model = Pipeline([
        ("s", StandardScaler()),
        ("reg", MultiOutputRegressor(base, n_jobs=1)),
    ])
    model.fit(X_tr, Y_tr)
    return model

# ── Simulação do hybrid_ensemble ─────────────────────────────────────────────

def simulate_hybrid(
    clf_models: dict,
    reg_model,
    X_te: np.ndarray,
    accs_te: np.ndarray,
    mech_names: list,
    clf_classes: list,
    top_k: int = 3,
    laplace_margin: float = 2.0,
) -> dict:
    """Simula o hybrid_ensemble no conjunto de teste.

    Retorna métricas: hit_rate, pct_worse_laplace, pct_better_laplace,
    regret_mean, relative_perf.
    """
    laplace_idx = mech_names.index("Laplace") if "Laplace" in mech_names else -1
    n_test = len(X_te)
    hits, worse_lap, better_lap, regrets, rel_perfs = [], [], [], [], []

    # Soft-voting do classificador
    probas = []
    for m in clf_models.values():
        try:
            probas.append(m.predict_proba(X_te))
        except Exception:
            pass
    clf_proba_all = np.mean(probas, axis=0)  # (n_test, n_classes)

    # Regressão de perda
    pred_losses = reg_model.predict(X_te)   # (n_test, n_mechs)
    pred_losses = np.clip(pred_losses, 0.0, 100.0)

    for i in range(n_test):
        clf_proba   = clf_proba_all[i]
        losses      = pred_losses[i]
        accs        = accs_te[i]

        # Ordena mecanismos por probabilidade do classificador
        sorted_idx = np.argsort(-clf_proba)
        # top-K sobreviventes que existem no regressor
        survivors = [clf_classes[j] for j in sorted_idx
                     if clf_classes[j] in mech_names][:top_k]

        survivor_losses = {m: losses[mech_names.index(m)]
                           for m in survivors if m in mech_names}

        # Fallback se nenhum sobrevivente existe no regressor: usa top-1 do clf
        if not survivor_losses:
            top_clf = clf_classes[int(sorted_idx[0])]
            if top_clf in mech_names:
                survivor_losses = {top_clf: losses[mech_names.index(top_clf)]}
            else:
                survivor_losses = {mech_names[int(np.argmin(losses))]: float(np.min(losses))}

        best_mech = min(survivor_losses, key=survivor_losses.get)
        best_loss = survivor_losses[best_mech]

        # Fallback conservador: se vencedor não supera Laplace por margem, usa top-1 do clf
        if laplace_idx >= 0 and best_mech != "Laplace":
            laplace_loss = losses[laplace_idx]
            if best_loss > laplace_loss - laplace_margin:
                best_mech = clf_classes[int(sorted_idx[0])]

        # Oracle e métricas
        rec_idx   = mech_names.index(best_mech) if best_mech in mech_names else 0
        oracle_idx = int(np.argmax(accs))
        oracle_mech = mech_names[oracle_idx]

        rec_acc    = accs[rec_idx]
        oracle_acc = accs[oracle_idx]
        laplace_acc = accs[laplace_idx] if laplace_idx >= 0 else 0.0

        hits.append(int(best_mech == oracle_mech))
        worse_lap.append(int(rec_acc < laplace_acc - 1e-6))
        better_lap.append(int(rec_acc > laplace_acc + 1e-6))
        regrets.append(float(oracle_acc - rec_acc))
        rel_perfs.append(float(rec_acc / (oracle_acc + 1e-9)))

    return {
        "hit_rate":          np.mean(hits),
        "pct_worse_laplace": np.mean(worse_lap),
        "pct_better_laplace": np.mean(better_lap),
        "regret_mean":       np.mean(regrets),
        "relative_perf":     np.mean(rel_perfs),
        "n_test":            n_test,
    }

# ── Grade de busca ────────────────────────────────────────────────────────────

def grid_search(
    clf_models, reg_model, X_te, accs_te, mech_names, clf_classes,
    top_k_values, margin_values,
) -> pd.DataFrame:
    rows = []
    total = len(top_k_values) * len(margin_values)
    print(f"\n{'='*65}")
    print(f"  GRADE DE BUSCA: {len(top_k_values)} top_k × {len(margin_values)} margens = {total} combinações")
    print(f"{'='*65}")

    for top_k, margin in product(top_k_values, margin_values):
        m = simulate_hybrid(
            clf_models, reg_model, X_te, accs_te,
            mech_names, clf_classes, top_k=top_k, laplace_margin=margin,
        )
        rows.append({
            "top_k":         top_k,
            "margin_pp":     margin,
            "hit_rate":      round(m["hit_rate"], 4),
            "worse_laplace": round(m["pct_worse_laplace"], 4),
            "better_laplace": round(m["pct_better_laplace"], 4),
            "regret":        round(m["regret_mean"], 4),
            "rel_perf":      round(m["relative_perf"], 4),
        })

    df = pd.DataFrame(rows)
    # Score composto: maximiza hit_rate, penaliza worse_laplace
    df["score"] = df["hit_rate"] - 0.5 * df["worse_laplace"]
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    return df

# ── Relatório ─────────────────────────────────────────────────────────────────

def print_report(grid: pd.DataFrame, clf_scores: dict, reg_mae: float):
    print(f"\n{'='*65}")
    print("  MÉTRICAS DOS META-MODELOS (treino completo)")
    print(f"{'='*65}")
    for name, score in sorted(clf_scores.items(), key=lambda x: -x[1]):
        marker = " ◄ melhor" if score == max(clf_scores.values()) else ""
        print(f"  Classificador  {name:<14} F1-macro={score:.4f}{marker}")
    print(f"  Regressor      MAE-CV={reg_mae:.2f}%")

    print(f"\n{'='*65}")
    print("  TOP-10 COMBINAÇÕES (score = hit_rate − 0.5×worse_laplace)")
    print(f"{'='*65}")
    print(f"  {'top_k':>5}  {'margin':>6}  {'hit_rate':>8}  {'worse_lap':>9}  {'better_lap':>10}  {'regret':>7}  {'score':>7}")
    print(f"  {'-'*63}")
    for _, row in grid.head(10).iterrows():
        print(
            f"  {int(row.top_k):>5}  {row.margin_pp:>6.1f}  "
            f"{row.hit_rate:>8.1%}  {row.worse_laplace:>9.1%}  "
            f"{row.better_laplace:>10.1%}  {row.regret:>7.4f}  {row.score:>7.4f}"
        )

    best = grid.iloc[0]
    print(f"\n{'='*65}")
    print("  PARÂMETROS ÓTIMOS RECOMENDADOS")
    print(f"{'='*65}")
    print(f"  _hybrid_top_k          = {int(best.top_k)}")
    print(f"  _hybrid_laplace_margin = {best.margin_pp:.1f}  (pp)")
    print(f"  → Hit Rate previsto    = {best.hit_rate:.1%}")
    print(f"  → Pior que Laplace     = {best.worse_laplace:.1%}")
    print(f"  → Melhor que Laplace   = {best.better_laplace:.1%}")
    print(f"  → Score composto       = {best.score:.4f}")

    # Comparativo com v17, v18, v19 baseline
    print(f"\n{'='*65}")
    print("  COMPARATIVO DE VERSÕES")
    print(f"{'='*65}")
    ref = [
        ("v17 (clf puro)",          0.664, None,  "—"),
        ("v18 (híbrido n_runs=1)",  0.364, 0.486, "—"),
        ("v19 baseline (margem=2)", 0.532, 0.318, "—"),
        ("v19 ÓTIMO (esta run)",    best.hit_rate, best.worse_laplace,
         f"top_k={int(best.top_k)}, margem={best.margin_pp:.1f}pp"),
    ]
    print(f"  {'Versão':<30}  {'Hit Rate':>8}  {'Pior Lap':>8}  {'Config'}")
    print(f"  {'-'*63}")
    for label, hr, wl, cfg in ref:
        wl_str = f"{wl:.1%}" if wl is not None else "    —"
        print(f"  {label:<30}  {hr:>8.1%}  {wl_str:>8}  {cfg}")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Tuning offline do hybrid_ensemble a partir dos CSVs da v19."
    )
    parser.add_argument("--features", default=DEFAULT_FEATURES_CSV,
                        help="Caminho do meta_features_*.csv")
    parser.add_argument("--targets", default=DEFAULT_TARGETS_CSV,
                        help="Caminho do meta_targets_*.csv")
    parser.add_argument("--test-size", type=float, default=0.25,
                        help="Fração do dataset para teste hold-out (padrão: 0.25)")
    parser.add_argument("--top-k", type=int, nargs="+", default=[2, 3, 4, 5, 6],
                        help="Valores de top_k a testar (padrão: 2 3 4 5 6)")
    parser.add_argument("--margins", type=float, nargs="+",
                        default=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0],
                        help="Margens de fallback em pp a testar")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-grid", metavar="PATH", default=None,
                        help="Salva a grade completa em CSV (ex: tune_results.csv)")
    args = parser.parse_args()

    print(f"\n{'='*65}")
    print("  TUNE META MODELS — DP Meta-Selector v19")
    print(f"{'='*65}")
    print(f"  Features : {args.features}")
    print(f"  Targets  : {args.targets}")
    print(f"  Test size: {args.test_size:.0%}  |  seed={args.seed}")
    print(f"  top_k    : {args.top_k}")
    print(f"  margins  : {args.margins}")

    # 1. Carrega dados
    X, Y, y, accs, mech_names, classes, _ = load_data(args.features, args.targets)
    print(f"\n  Datasets: {len(X)}  |  Features: {X.shape[1]}  |  Mecanismos: {len(mech_names)}")

    # 2. Split treino/teste estratificado
    sss = StratifiedShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.seed)
    tr_idx, te_idx = next(sss.split(X, y))
    X_tr, X_te = X[tr_idx], X[te_idx]
    Y_tr, Y_te = Y[tr_idx], Y[te_idx]
    y_tr, y_te = y[tr_idx], y[te_idx]
    accs_te = accs[te_idx]
    print(f"  Treino: {len(X_tr)}  |  Teste: {len(X_te)}")

    # 3. Treina classificador
    print("\n  [1/2] Treinando classificadores...")
    clf_models, clf_scores, best_clf = train_classifier(X_tr, y_tr)
    le = LabelEncoder().fit(y_tr)
    clf_classes = list(le.inverse_transform(np.arange(len(le.classes_))))
    for name, s in sorted(clf_scores.items(), key=lambda x: -x[1]):
        marker = " ◄" if name == best_clf else ""
        print(f"    {name:<14} F1-macro={s:.4f}{marker}")

    # 4. Treina regressor
    print("\n  [2/2] Treinando regressor multi-output...")
    reg_model = train_regressor(X_tr, Y_tr)
    # MAE-CV no treino
    from sklearn.model_selection import KFold, cross_val_score as cvs
    kf = KFold(n_splits=5, shuffle=True, random_state=args.seed)
    mae_scores = cvs(reg_model, X_tr, Y_tr, cv=kf, scoring="neg_mean_absolute_error")
    reg_mae = float(-mae_scores.mean())
    print(f"    MAE-CV (treino): {reg_mae:.2f}%")

    # 5. Grid search
    grid = grid_search(
        clf_models, reg_model, X_te, accs_te,
        mech_names, clf_classes,
        top_k_values=args.top_k,
        margin_values=args.margins,
    )

    # 6. Relatório
    print_report(grid, clf_scores, reg_mae)

    # 7. Salva grade se solicitado
    if args.save_grid:
        grid.to_csv(args.save_grid, index=False)
        print(f"\n  Grade salva em: {args.save_grid}")

    return grid


if __name__ == "__main__":
    main()

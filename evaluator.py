"""Avaliação do framework em datasets de teste."""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import LabelEncoder

from .mechanisms import FAMILY_OF, MECHANISM_NAMES
from .selector import DPMechanismSelector
from .utility import DPUtilityEvaluator, EVAL_FULL_PROFILE


class FrameworkEvaluator:
    def __init__(
        self,
        selector: DPMechanismSelector,
        use_full_oracle: bool = False,
        seed: int = 42,  # B4: seed reproduzível
    ):
        self.selector = selector
        self._rng = np.random.RandomState(seed)  # B4
        self._oracle_evaluator = selector._evaluator
        if use_full_oracle:
            self._oracle_evaluator = DPUtilityEvaluator(
                delta=selector.delta,
                profile=EVAL_FULL_PROFILE,
                cache=selector._cache,
            )

    def evaluate(self, test_datasets):
        results = []

        for X, y, name in test_datasets:
            print(f"\n[TEST] Dataset: {name}")

            y_enc = LabelEncoder().fit_transform(y)

            ev = self._oracle_evaluator
            base = ev.baseline(X, y_enc)
            dp_all = ev.evaluate_all(X, y_enc)

            best_acc = max(dp_all.values())
            best_mech = max(dp_all, key=dp_all.get)

            rec = self.selector.recommend(X, y, verbose=False)
            rec_mech = rec["recommended_mechanism"]
            rec_acc = dp_all[rec_mech]

            # E3: random baseline = média sobre todos os mecanismos (expected value)
            random_acc = float(np.mean(list(dp_all.values())))
            laplace_acc = dp_all.get("Laplace", 0.0)

            regret = best_acc - rec_acc
            rel = rec_acc / (best_acc + 1e-9)

            results.append({
                "dataset": name,
                "best_mech": best_mech,
                "rec_mech": rec_mech,
                "best_acc": best_acc,
                "rec_acc": rec_acc,
                "base_acc": base,
                "regret": regret,
                "relative_performance": rel,
                "random_acc": random_acc,
                "laplace_acc": laplace_acc,
                "hit": int(rec_mech == best_mech),
            })

        df = pd.DataFrame(results)

        print("\n" + "=" * 70)
        print("RESULTADOS GERAIS")
        print("=" * 70)

        n = len(df)

        # E1: Intervalo de confiança 95%
        def _ci95(series):
            if len(series) < 2:
                return float(series.mean()), float(series.mean())
            lo, hi = stats.t.interval(
                0.95, len(series) - 1,
                loc=series.mean(),
                scale=stats.sem(series),
            )
            return float(lo), float(hi)

        hit_lo, hit_hi = _ci95(df["hit"])
        reg_lo, reg_hi = _ci95(df["regret"])
        rel_lo, rel_hi = _ci95(df["relative_performance"])

        print(
            f"Hit rate (acertou melhor mec.): {df['hit'].mean():.4f}  "
            f"IC95%=[{hit_lo:.4f}, {hit_hi:.4f}]  (n={n})"
        )
        print(
            f"Regret médio                  : {df['regret'].mean():.4f}  "
            f"IC95%=[{reg_lo:.4f}, {reg_hi:.4f}]"
        )
        print(
            f"Performance relativa média    : {df['relative_performance'].mean():.4f}  "
            f"IC95%=[{rel_lo:.4f}, {rel_hi:.4f}]"
        )

        print("\nComparação com baselines:")
        print(f"  Modelo (recomendado): {df['rec_acc'].mean():.4f}")
        print(f"  Random (média todos): {df['random_acc'].mean():.4f}")
        print(f"  Laplace fixo        : {df['laplace_acc'].mean():.4f}")
        print(f"  Oracle (melhor real): {df['best_acc'].mean():.4f}")

        # Métrica mais informativa: ganho normalizado sobre Laplace fixo
        df["delta_laplace"] = df["rec_acc"] - df["laplace_acc"]
        df["spread"] = df["best_acc"] - df["random_acc"] + 1e-9
        df["norm_ganho"] = df["delta_laplace"] / df["spread"]
        n_melhor = (df["delta_laplace"] > 1e-6).sum()
        n_pior   = (df["delta_laplace"] < -1e-6).sum()
        n_igual  = n - n_melhor - n_pior
        print(
            f"\n  Modelo vs Laplace fixo (por dataset):"
            f"\n    melhor em {n_melhor}/{n} ({n_melhor/n:.1%})  "
            f"| igual em {n_igual}/{n} ({n_igual/n:.1%})  "
            f"| pior em {n_pior}/{n} ({n_pior/n:.1%})"
            f"\n    Ganho médio normalizado: {df['norm_ganho'].mean():+.4f}"
            f"  (>0 = modelo agrega valor)"
        )

        # E2: Breakdown por família do mecanismo oracle
        print("\nBreakdown por família (oracle):")
        for fam in sorted(set(FAMILY_OF.values())):
            mask = df["best_mech"].apply(lambda m: FAMILY_OF.get(m, "") == fam)
            sub = df[mask]
            if not sub.empty:
                print(
                    f"  {fam:<12}: n={len(sub):>2}  hit={sub['hit'].mean():.3f}  "
                    f"regret={sub['regret'].mean():.4f}  "
                    f"rel={sub['relative_performance'].mean():.3f}"
                )

        # v17: MÉTRICAS FOCADAS EM OPORTUNIDADE
        print("\n" + "=" * 70)
        print("MÉTRICAS DE OPORTUNIDADE (v17)")
        print("=" * 70)
        
        # 1. Casos onde Laplace NÃO é ótimo
        df["laplace_suboptimal"] = df["best_mech"] != "Laplace"
        n_subopt = df["laplace_suboptimal"].sum()
        pct_subopt = n_subopt / n
        print(f"\n1. OPORTUNIDADE DE MELHORIA:")
        print(f"   Datasets onde Laplace NÃO é ótimo: {n_subopt}/{n} ({pct_subopt:.1%})")
        
        # 2. Ganho capturado nesses casos
        subopt_df = df[df["laplace_suboptimal"]]
        if len(subopt_df) > 0:
            # Gap disponível = best_acc - laplace_acc
            subopt_df = subopt_df.copy()
            subopt_df["gap_available"] = subopt_df["best_acc"] - subopt_df["laplace_acc"]
            subopt_df["gap_captured"] = subopt_df["rec_acc"] - subopt_df["laplace_acc"]
            subopt_df["capture_ratio"] = subopt_df["gap_captured"] / (subopt_df["gap_available"] + 1e-9)
            
            avg_gap_available = subopt_df["gap_available"].mean() * 100
            avg_gap_captured = subopt_df["gap_captured"].mean() * 100
            avg_capture_ratio = subopt_df[subopt_df["gap_available"] > 0.001]["capture_ratio"].mean()
            
            print(f"   Gap médio disponível: +{avg_gap_available:.2f}pp vs Laplace")
            print(f"   Gap médio capturado:  +{avg_gap_captured:.2f}pp")
            print(f"   Taxa de captura: {avg_capture_ratio:.1%} do ganho possível")
            
            # Hit rate específico nesses casos
            subopt_hit = subopt_df["hit"].mean()
            print(f"   Hit rate em casos subótimos: {subopt_hit:.1%}")
        
        # 3. Worst-case: quantas vezes modelo é PIOR que Laplace
        df["worse_than_laplace"] = df["rec_acc"] < df["laplace_acc"] - 1e-6
        n_worse = df["worse_than_laplace"].sum()
        pct_worse = n_worse / n
        print(f"\n2. SEGURANÇA (worst-case):")
        print(f"   Modelo pior que Laplace: {n_worse}/{n} ({pct_worse:.1%})")
        
        if n_worse > 0:
            worse_df = df[df["worse_than_laplace"]]
            avg_loss = (worse_df["laplace_acc"] - worse_df["rec_acc"]).mean() * 100
            max_loss = (worse_df["laplace_acc"] - worse_df["rec_acc"]).max() * 100
            print(f"   Perda média quando pior: -{avg_loss:.2f}pp")
            print(f"   Perda máxima: -{max_loss:.2f}pp")
        else:
            print(f"   ✓ Framework NUNCA é pior que Laplace!")
        
        # 4. Top-K datasets com maior ganho
        print(f"\n3. TOP-10 GANHOS vs Laplace:")
        df_sorted = df.sort_values("delta_laplace", ascending=False)
        for i, row in df_sorted.head(10).iterrows():
            delta = row["delta_laplace"] * 100
            if delta > 0:
                print(f"   {row['dataset'][:40]:<40} +{delta:.2f}pp ({row['rec_mech']} vs Laplace)")

        print(f"\n  {self.selector._cache.summary()}")

        return df

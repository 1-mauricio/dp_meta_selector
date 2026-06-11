# Documentação do DP Meta-Selector

> Framework de meta-aprendizagem para seleção automática de mecanismos de Privacidade Diferencial.

**Atualizado em:** 2026-06-11

---

## Visão Geral

O `dp_meta_selector` é um framework que seleciona automaticamente o melhor mecanismo de Privacidade Diferencial (DP) para um dataset tabular, sem necessidade de intervenção manual.

**Fluxo geral (v16, atual):**
1. Extrair meta-features do dataset (39 features)
2. CAT1: pré-filtro binário Exponential (threshold ≥ 0.75) com dual-gate família (p_cat ≥ 0.15)
3. GAUSS: pré-filtro para GaussianAnalytic (threshold ≥ 0.80)
4. Ensemble ExtraTrees + portão hierárquico HIER (gate 0.55)
5. Aplicar o mecanismo escolhido com ε calibrado por família

---

## Estrutura da Documentação

| Arquivo | Descrição |
|---------|-----------|
| [01_baseline_analysis.md](01_baseline_analysis.md) | Análise do baseline e diagnóstico inicial |
| [02_categorical_prefilter.md](02_categorical_prefilter.md) | Pré-filtro categórico (CAT1) e dual-gate |
| [03_family_hierarchy.md](03_family_hierarchy.md) | Portão hierárquico (HIER) e decisões sobre Geometric |
| [04_gaussian_optimization.md](04_gaussian_optimization.md) | Otimização do GaussianAnalytic e thresholds |
| [05_improvements_v14_v16.md](05_improvements_v14_v16.md) | Melhorias das versões v14 a v16 |
| [06_mechanism_comparison.md](06_mechanism_comparison.md) | Estudo comparativo de mecanismos DP (489 datasets) |
| [07_lessons_learned.md](07_lessons_learned.md) | Lições aprendidas consolidadas |
| [08_datasets.md](08_datasets.md) | Detalhes dos datasets utilizados |
| [09_results_summary.md](09_results_summary.md) | Resumo de resultados por versão |

---

## Métricas Principais

| Métrica | Descrição |
|---------|-----------|
| **hit_rate** | Proporção de acertos do mecanismo oracle |
| **regret** | Perda de acurácia em relação ao oracle |
| **model_acc** | Acurácia média do classificador com DP |
| **cat_hit** | Hit rate em datasets categóricos |
| **cont_hit** | Hit rate em datasets contínuos |
| **F1-macro** | F1-score macro do meta-modelo |

---

## Evolução do Framework

| Versão | Hit Rate | Principais Mudanças |
|--------|----------|---------------------|
| v0 (baseline) | 53.1% | Meta-modelo básico |
| v1 | 53.7% | +CAT1 prefilter |
| v2b | 68.0% | +HIER gate (corrigido) |
| v8b | 64.6% | +Dual-gate T2=0.20 |
| v13 | 67.4% | T1=0.90 (otimizado) |
| v14 | 67.6% | +Sintéticos, +family discriminators |
| **v16** | **61.9%** | Thresholds ajustados para F1-macro (0.70) |

---

## Arquivos do Repositório

```
dp_meta_selector/
├── __init__.py
├── main.py              # CLI principal
├── selector.py          # DPSelector (API)
├── meta_learner.py      # MetaLearner (modelo)
├── meta_features.py     # Extração de features
├── meta_dataset.py      # Construção do meta-dataset
├── mechanisms.py        # Mecanismos DP disponíveis
├── calibration.py       # Calibração de epsilon
├── applicator.py        # Aplicação de DP
├── diagnostics.py       # Métricas avançadas
├── synthetic_datasets.py # Geradores de sintéticos
└── scripts/
    └── compare_dp_mechanisms.py  # Comparação de mecanismos
```

---

## Referências Rápidas

- **Mecanismos ativos:** Laplace, Gaussian, GaussianAnalytic, Staircase, LaplaceTruncated, LaplaceFolded, Snapping, Exponential, Uniform
- **Mecanismos de screening:** Laplace, GaussianAnalytic, Exponential
- **Epsilon por família:** continuous=5.0, categorical=2.0
- **Meta-features:** 39 features (estatísticas + família + discriminadores)

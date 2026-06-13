# Documentação do DP Meta-Selector

> Framework de meta-aprendizagem para seleção automática de mecanismos de Privacidade Diferencial.

**Atualizado em:** 2026-06-13

---

## Visão Geral

O `dp_meta_selector` é um framework que seleciona automaticamente o melhor mecanismo de Privacidade Diferencial (DP) para um dataset tabular, sem necessidade de intervenção manual.

**Fluxo geral (v17, atual):**
1. Extrair meta-features do dataset (**116 features** — inclui features DP-específicas e contexto do usuário)
2. CAT1: pré-filtro binário Exponential (threshold ≥ 0.75) com dual-gate família (p_cat ≥ 0.15)
3. DISC: pré-filtro para mecanismos discretos/Geometric (threshold ≥ 0.70)
4. GAUSS: pré-filtro para GaussianAnalytic (threshold ≥ 0.80)
5. **Regressão multi-output** de perda de utilidade — recomenda o mecanismo com **menor perda prevista** (padrão)
6. Ensemble ExtraTrees + portão hierárquico HIER (gate 0.55) — fallback
7. Aplicar o mecanismo escolhido com ε calibrado por família

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
| [10_technical_architecture.md](10_technical_architecture.md) | **Arquitetura técnica completa** (pipeline, algoritmos, fórmulas) |

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
| **MAE-CV** | ⚡ NOVO: Erro médio da regressão de perda de utilidade |

---

## Evolução do Framework

| Versão | Hit Rate | F1-macro | Principais Mudanças |
|--------|----------|----------|---------------------|
| v0 (baseline) | 53.1% | — | Meta-modelo básico |
| v1 | 53.7% | — | +CAT1 prefilter |
| v2b | 68.0% | — | +HIER gate (corrigido) |
| v8b | 64.6% | — | +Dual-gate T2=0.20 |
| v13 | 67.4% | 0.55 | T1=0.90 (otimizado) |
| v14 | 67.6% | 0.55 | +Sintéticos, +family discriminators |
| v16 | 61.9% | **0.70** | Thresholds ajustados para recall balanceado |
| **v17** | **66.4%** | **0.87** | ⚡ +40 meta-features DP, regressão de utilidade, variáveis de contexto |

> **v17** é a versão com a refatoração DP-aware. Ver [09_results_summary.md](09_results_summary.md) para análise detalhada.

---

## Arquivos do Repositório

```
dp_meta_selector/
├── __init__.py
├── main.py              # CLI principal
├── selector.py          # DPSelector (API) — aceita epsilon e task_type
├── meta_learner.py      # MetaLearner — classificação + regressão de perda
├── meta_features.py     # Extração de features (116 features, inclui DP e contexto)
├── meta_dataset.py      # Construção do meta-dataset (inclui utility_loss_*)
├── mechanisms.py        # Mecanismos DP disponíveis
├── calibration.py       # Calibração de epsilon
├── applicator.py        # Aplicação de DP
├── utility.py           # Perfis de avaliação (inclui META_STABLE_PROFILE n_runs=5)
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
- **Meta-features:** 116 features (estatísticas + família + discriminadores + **DP-específicas** + **contexto**)
- **Variáveis de contexto obrigatórias:** `epsilon` (orçamento), `task_type` (classificação/regressão/queries)
- **Target do regressor:** `utility_loss_{mechanism}` = perda relativa % por mecanismo

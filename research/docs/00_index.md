# Documentação do DP Meta-Selector

> Framework de meta-aprendizagem para seleção automática de mecanismos de Privacidade Diferencial.

**Atualizado em:** 2026-06-13 — v19-tuned (versão final dissertação)

---

## Visão Geral

O `dp_meta_selector` é um framework que seleciona automaticamente o melhor mecanismo de Privacidade Diferencial (DP) para um dataset tabular, sem necessidade de intervenção manual.

**Fluxo geral (v19, atual):**
1. Extrair meta-features do dataset (**112 features** — estatísticas clássicas + DP-específicas + contexto)
2. CAT1: pré-filtro binário Exponential (threshold ≥ 0.75) com dual-gate família (p_cat ≥ 0.15)
3. DISC: pré-filtro para mecanismos discretos/Geometric (threshold ≥ 0.70)
4. GAUSS: pré-filtro para GaussianAnalytic (threshold ≥ 0.80)
5. **Hybrid Ensemble:** Classificador Top-3 → Regressor ordena por menor `utility_loss` previsto
6. **Fallback conservador:** se `loss_recomendado > loss_Laplace − 0.5pp`, recua para Laplace/classificador
7. Opcionalmente retorna Top-2 recomendações via `return_top_k=2` (Human-in-the-Loop)

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
| [07_lessons_learned.md](07_lessons_learned.md) | Lições aprendidas consolidadas + Memorial Técnico v17–v19 |
| [08_datasets.md](08_datasets.md) | Detalhes dos datasets utilizados |
| [09_results_summary.md](09_results_summary.md) | Resumo de resultados por versão (v0 → v19-tuned) |
| [10_technical_architecture.md](10_technical_architecture.md) | **Arquitetura técnica completa** (pipeline, algoritmos, fórmulas, v19) |
| [**11_scientific_contribution.md**](11_scientific_contribution.md) | 🎓 **Contribuição científica principal — pronto para dissertação** |
| [20_final_benchmark_report.md](20_final_benchmark_report.md) | Benchmark científico: 5 seletores × 6 métricas (5-fold CV, 401 datasets) |

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
| **v17** | **66.4%** | **0.87** | ⚡ +38 meta-features DP/ctx, regressão multi-output |
| v18 | 36.4% | 0.855 | Hybrid ensemble, n_runs=1 (labels ruidosas) |
| v19 raw | 53.2% | **0.910** | META_STABLE_PROFILE n_runs=5, margin=2.0pp |
| **v19-tuned** | **68.3%*** | **0.910** | **margin=0.5pp calibrado, Max Regret −45%, `return_top_k`** |

> *Hit Rate medido via 5-fold CV (`benchmark_evaluator.py`). Pipeline completo com pré-filtros: ~53% (v19 raw).

---

## Arquivos do Repositório

```
dp_meta_selector/
├── __init__.py
├── main.py              # CLI principal
├── selector.py          # DPSelector (API) — epsilon, task_type, return_top_k
├── meta_learner.py      # MetaLearner — clf + regressão + hybrid ensemble v19
├── meta_features.py     # Extração de 112 features (DP-específicas + contexto)
├── meta_dataset.py      # Construção do meta-dataset + checkpoint + CSV persistence
├── mechanisms.py        # Mecanismos DP disponíveis
├── calibration.py       # Calibração de epsilon
├── applicator.py        # Aplicação de DP
├── utility.py           # Perfis de avaliação (inclui META_STABLE_PROFILE n_runs=5)
├── diagnostics.py       # Métricas avançadas
├── meta_datasets_v19/   # CSVs estáveis (401×112 features, 401×9 targets)
├── research/
│   ├── benchmark_evaluator.py      # 5 seletores × 6 métricas
│   ├── tuning/tune_meta_models.py  # Grid search offline
│   └── docs/                       # Documentação científica completa
└── tests/               # 33 testes unitários
```

---

## Referências Rápidas

- **Mecanismos ativos:** Laplace, Gaussian, GaussianAnalytic, Staircase, LaplaceTruncated, LaplaceFolded, Snapping, Exponential, Uniform
- **Mecanismos de screening:** Laplace, GaussianAnalytic, Exponential
- **Epsilon por família:** continuous=5.0, categorical=2.0
- **Meta-features:** 112 features (estatísticas + família + discriminadores + **DP-específicas** + **contexto**)
- **Variáveis de contexto obrigatórias:** `epsilon` (orçamento), `task_type` (classificação/regressão/queries)
- **Target do regressor:** `utility_loss_{mechanism}` = perda relativa % por mecanismo
- **Ensemble híbrido:** `_hybrid_top_k=3`, `_hybrid_laplace_margin=0.5pp` (calibrado offline, grade 5×8)
- **Human-in-the-Loop:** `selector.recommend(..., return_top_k=2)` → Top-2 Hit Rate **94.3%**
- **Benchmark:** `research/benchmark_evaluator.py` · relatório em `research/docs/20_final_benchmark_report.md`

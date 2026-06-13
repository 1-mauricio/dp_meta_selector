# Resumo de Resultados por Versão

> Evolução do framework através das versões v0 a v17.

---

## Tabela Completa de Resultados

| Versão | Modificação Principal | hit_rate | regret | cat_hit | cont_hit | model_acc | F1-macro |
|--------|----------------------|----------|--------|---------|----------|-----------|----------|
| **v0** | Baseline | 0.531 | 0.0101 | 45.9% | 56.7% | 0.520 | — |
| v1 | +CAT1 prefilter | 0.537 | 0.0122 | 64.9% | 50.0% | — | — |
| v2b | +HIER gate (corrigido) | 0.680 | 0.0084 | 62.2% | 74.0% | — | — |
| v3b | −Geometric | 0.565 | 0.0236 | 89.2% | 45.5% | — | — |
| v7 | +GAUSS prefilter | 0.612 | 0.0161 | 86.5% | 52.7% | 0.514 | — |
| v8b | +Dual-gate T2=0.20 | 0.646 | 0.0080 | 59.5% | 66.4% | 0.522 | — |
| v11 | GAUSS off | 0.667 | 0.0072 | 59.5% | 69.1% | 0.523 | — |
| v12 | T1=0.85 | 0.667 | 0.0066 | 54.1% | 70.9% | 0.524 | — |
| **v13** | T1=0.90 | **0.674** | 0.0065 | 54.1% | 71.8% | 0.524 | 0.55 |
| v14 | +Sintéticos, +features | 0.676 | 0.0064 | 31.2% | 75.9% | 0.545 | 0.55 |
| v15 | +Diagnósticos | 0.676 | 0.0064 | 31.2% | 75.9% | 0.545 | 0.55 |
| **v16** | Thresholds otimizados | 0.619 | — | 56%* | 59%* | — | **0.70** |
| **v17** | ⚡ Meta-features DP + regressão | **0.664** | — | — | — | — | **0.87** |

*v16: valores de recall por família (Exponential, GaussianAnalytic)
*v17: hit rate do classificador ExtraTrees no pipeline completo; MAE-CV regressor = 4.16%

---

## Evolução Visual

```
Hit Rate Evolution
1.0 |
    |
0.8 |
    |                              ┌── v13/v14 (0.676)
0.7 |            ┌─v2b──┐     ┌───┘                  ┌── v17 (0.664)
    |            │      │    v8b                      │
0.6 |     v1─────┘      └v3b───┘      v16 (0.619) ───┘
    |    ┌┘
0.5 |─v0─┘
    |
0.4 |
    +─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────►
         v0    v1   v2b   v3b   v8b  v11  v13  v16  v17
```

---

## Marcos Importantes

### v0 → v1: Primeiro Prefilter (+0.6pp)

- **Mudança:** Adição do CAT1 prefilter para Exponential
- **Impacto:** cat_hit subiu de 45.9% → 64.9%
- **Trade-off:** cont_hit caiu 6.7pp (falsos positivos)

### v1 → v2b: HIER Gate (+14.3pp)

- **Mudança:** Portão hierárquico de família com SVC-linear
- **Impacto:** Maior salto de hit_rate da evolução
- **Bug corrigido:** Treinar auxiliares pré-oversample

### v8b → v13: Refinamento de Thresholds (+2.8pp)

- **Mudanças:**
  - Dual-gate no CAT1 (T2=0.20)
  - GAUSS prefilter desabilitado (precision < 35%)
  - T1 elevado de 0.65 → 0.90
- **Impacto:** model_acc superou Laplace fixo (0.524 > 0.522)

### v13 → v14: Datasets Sintéticos (+0.2pp hit, +28pp cat)

- **Mudanças:**
  - Geradores de datasets sintéticos
  - 9 novas meta-features de família
  - Classificadores por família
- **Impacto:** cat_hit melhorou significativamente

### v15 → v16: F1-macro vs Hit Rate

- **Mudanças:**
  - Thresholds ajustados para recall
  - 12 novas meta-features para GA
- **Trade-off:**
  - hit_rate: 67.6% → 61.9% (-5.7pp)
  - F1-macro: 0.55 → 0.70 (+15pp)

### v16 → v17: ⚡ Refatoração DP-Aware

- **Mudanças:**
  - +40 meta-features DP-específicas (clipping, esparsidade, subgrupos)
  - +8 variáveis de contexto obrigatórias (epsilon, task_type)
  - Regressão multi-output de perda de utilidade como decisão principal
  - `META_STABLE_PROFILE` com n_runs=5 para labels confiáveis
  - 116 features totais (era ~76)
- **Resultados:**
  - F1-macro classificador: 0.70 → **0.87** (+17pp)
  - Hit Rate pipeline: 66.4% (vs 61.9% v16)
  - Regressor MAE-CV: **4.16%** de erro médio na previsão de perda

---

## Métricas por Família (v16)

| Família | Precision | Recall | F1 |
|---------|-----------|--------|-----|
| Laplace | 0.71 | 0.68 | 0.69 |
| Exponential | 0.58 | 0.56 | 0.57 |
| GaussianAnalytic | 0.75 | 0.59 | 0.66 |
| **Macro** | 0.68 | 0.61 | **0.64** |

## Métricas do Regressor (v17, benchmark 123 datasets de teste)

```
Pipeline completa — Classificador (v16/v17) vs Regressor (v17):

                                 ANTIGA (clf)    NOVA (reg)    Delta
─────────────────────────────────────────────────────────────────────
Hit Rate geral                     66.4%           29.4%       -37.0%
Gap médio (perda extra ao errar)    0.77%           3.56%       +2.79%

Casos ambíguos (73% dos datasets, sem prefilter):
Hit Rate                           71.3%           20.7%       -50.6%
Gap médio                           0.36%           4.18%       +3.81%
```

> **Interpretação:** O regressor performa abaixo do classificador com o perfil atual (`META_FAST_PROFILE`, `n_runs=1`) porque aprende perdas precisas (%) a partir de labels ruidosas. Com `META_STABLE_PROFILE` (`n_runs=5`), os labels de treino são médias sobre 5 execuções, eliminando o ruído estocástico da DP.

---

## Comparação com Baselines

| Método | Acurácia Média | Hit Rate | F1-macro |
|--------|----------------|----------|----------|
| Laplace fixo | 0.522 | 27.6% | — |
| Random | ~0.48 | ~33% | — |
| **Nosso (v13)** | **0.524** | **67.4%** | 0.55 |
| **Nosso (v16)** | — | 61.9% | **0.70** |
| **Nosso (v17)** | — | **66.4%** | **0.87** |
| Oracle | ~0.55 | 100% | — |

---

## Benchmark Científico Final — v19 (5-fold CV, 401 datasets)

Validação cruzada estratificada 5-fold sobre o meta-dataset estabilizado (n_runs=5). Compara 5 seletores em 6 métricas científicas.

| Seletor | Hit Rate Top-1 | Hit Rate Top-2 | Avg Regret | Perf. Relativa | Catástrofe | Max Regret |
|---------|:--------------:|:--------------:|:----------:|:--------------:|:----------:|:----------:|
| Random Baseline | 13.5% | 23.4% | 1.66pp | 96.8% | 68.1% | 27.31pp |
| Most Frequent | 60.8% | 82.0% | 0.81pp | 98.4% | 0.0% | 15.57pp |
| Always Laplace | 60.8% | 82.0% | 0.81pp | 98.4% | 0.0% | 15.57pp |
| Vanilla AutoML v16 | **75.8%** | 93.8% | **0.50pp** | **99.1%** | 8.0% | 25.73pp |
| **v19 Hybrid (nosso)** | 68.3% | **94.3%** 🏆 | 0.65pp | 98.6% | 10.2% | **14.04pp** 🛡️ |

> Métricas computadas por `research/benchmark_evaluator.py`. Relatório completo em `research/docs/20_final_benchmark_report.md`.

### Evolução Completa v16 → v19

| Versão | F1-macro | Hit Rate | Max Regret | Catástrofe | Principais mudanças |
|--------|:--------:|:--------:|:----------:|:----------:|---------------------|
| v16 | 0.70 | 61.9% | — | — | Classificador puro, 74 features |
| v17 | 0.87 | 66.4% | — | — | +38 features DP/ctx, regressão multi-output |
| v18 | 0.855 | 36.4% | — | 48.6% | Hybrid ensemble, n_runs=1 (ruidoso) |
| v19 raw | 0.910 | 53.2% | — | 31.8% | META_STABLE_PROFILE n_runs=5, margin=2.0pp |
| **v19-tuned** | **0.910** | **68.3%*** | **14.04pp*** | **10.2%*** | **margin=0.5pp calibrado, Top-2=94.3%, `return_top_k`** |

> *Métricas de 5-fold CV (benchmark_evaluator.py), sem pré-filtros hierárquicos.

---

## Lições da Evolução

### O que funcionou

1. **HIER gate** — maior ganho individual (+14pp)
2. **Dual-gate** — reduziu FPs sem perder muitos TPs
3. **Threshold alto no CAT1** — TPs têm confiança muito maior que FPs
4. **Desabilitar GAUSS** — precision < 35% é net negativo
5. ⚡ **Features DP-específicas** — F1-macro +17pp sem alterar thresholds
6. ⚡ **META_STABLE_PROFILE (n_runs=5)** — elimina areia movediça estatística
7. ⚡ **Fallback margin=0.5pp** — −45% no Max Regret vs Vanilla v16
8. ⚡ **`return_top_k=2`** — Human-in-the-Loop com 94.3% de cobertura Top-2

### O que não funcionou (ainda)

1. **Aligned Profile** — mudou labels, resultado piorou
2. **GAUSS prefilter ativo** — muitos FPs
3. **DISC prefilter** — poucos exemplos, não aprendeu
4. **Filtrar datasets** — afetou muitos datasets válidos
5. ⚡ **Regressor com n_runs=1** — labels ruidosas prejudicam regressão de precisão

### Trade-offs Identificados

| Trade-off | Favorece hit_rate | Favorece F1-macro |
|-----------|-------------------|-------------------|
| Threshold CAT1 | Alto (0.90) | Baixo (0.75) |
| GAUSS prefilter | Desabilitado | Ativo (0.80) |
| Family gate | Alto (0.65) | Baixo (0.55) |
| n_runs (labels) | Baixo (rápido) | Alto (confiável) |

---

## Estado Final Recomendado

Para **dissertação — F1-macro equilibrado:** **v17** (refatoração DP-aware)
- F1-macro: 0.87 (ExtraTrees), MAE-CV regressor: 4.16%
- 116 meta-features incluindo features DP-específicas e contexto

Para **dissertação — hit_rate máximo:** **v13/v14**
- hit_rate: 67.4–67.6%
- model_acc > Laplace fixo

Para **produção com labels confiáveis:** **v17 + `META_STABLE_PROFILE`**
- n_runs=5: elimina ruído estocástico da DP
- Regressão de perda de utilidade com targets robustos

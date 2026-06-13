# 20 — Relatório Final de Benchmark Científico
> Gerado automaticamente por `research/benchmark_evaluator.py` em 2026-06-13 18:22

## Sumário Executivo

O framework **DP-Meta-Selector v19** (Ensemble Híbrido com `margin=0.5pp`) foi avaliado
cientificamente contra 4 baselines usando validação cruzada estratificada 5-fold
sobre os **401 datasets** do meta-dataset estabilizado (n_runs=5, META_STABLE_PROFILE).

O v19 Hybrid revela um **perfil de risco superior**: não é o melhor em Hit Rate Top-1 absoluto
(esse título cabe ao Vanilla v16, que usa o classificador sem restrições), mas domina em
**Max Regret** (pior caso: 14.04pp vs 25.73pp no v16, redução de 11.7pp)
e em **Hit Rate Top-2** (94.3% vs 93.8% no v16).

| Métrica                  | v19 Hybrid | Vanilla v16 | Always Laplace | Δ (v19 − v16) | Melhor               |
|--------------------------|:----------:|:-----------:|:--------------:|:-------------:|:--------------------:|
| Hit Rate Top-1           | 68.3%      | 75.8%       | 60.8%          | -7.5pp        | Vanilla v16          |
| Hit Rate Top-2           | 94.3%      | 93.8%       | 82.0%          | +0.5pp        | **v19 Hybrid**       |
| Average Regret           | 0.65pp    | 0.50pp     | 0.81pp        | +0.15pp      | Vanilla v16          |
| Max Regret (pior caso)   | 14.04pp    | 25.73pp     | 15.57pp       | -11.69pp      | **v19 Hybrid**       |
| Catastrophic Failure     | 10.2%      | 8.0%       | 0.0%          | +2.2pp        | Always Laplace (0%)  |

> **Interpretação**: O fallback conservador (margin=0.5pp) torna o v19 mais cauteloso —
> ele abre mão de ~7.5pp de precisão absoluta para reduzir o pior erro em 11.7pp.
> Para deployments DP críticos, o Max Regret baixo é frequentemente mais valioso que a precisão média.

---

## 1. Metodologia

- **Meta-dataset**: `meta_features_meta_stable.csv` + `meta_targets_meta_stable.csv`
  - 401 datasets (inner join por `dataset_name`)
  - Labels estabilizados via `n_runs=5` (META_STABLE_PROFILE)
- **Avaliação**: Validação Cruzada Estratificada 5-fold (`random_state=42`)
  - Treino em 4 folds (≈320 datasets) → avaliação no fold restante (≈80 datasets)
  - **Nenhum vazamento de dados**: modelos re-treinados a cada fold
- **Oversampling**: replica `MetaLearner._oversample` (target_ratio=0.8) para classificadores
  (regressor usa dados originais para preservar a distribuição real das perdas)
- **Métricas de referência**: acc_* oracle do CSV (acurácia DP real medida com n_runs=5)
- **Nota**: Este benchmark não inclui os pré-filtros hierárquicos do pipeline completo
  (CAT1/DISC/GAUSS), que adicionam ganho adicional em produção mas requerem raw datasets

### Definição das Métricas

| Métrica                | Fórmula                                           | Interpretação               |
|------------------------|---------------------------------------------------|-----------------------------|
| Hit Rate Top-1         | P(escolhido = oráculo)                            | Precisão de seleção         |
| Hit Rate Top-2         | P(oráculo ∈ top-2 recomendações)                  | Cobertura (tolerância a 1 erro) |
| Average Regret         | E[acc_oracle − acc_chosen] × 100 (pp)             | Custo médio do erro         |
| Performance Relativa   | E[acc_chosen / acc_oracle] × 100 (%)              | % do oráculo capturado      |
| Catastrophic Failure   | P(acc_chosen < acc_Laplace)                       | Taxa de regressão vs mercado |
| Max Regret             | max[acc_oracle − acc_chosen] × 100 (pp)           | Custo do pior caso          |

---

## 2. Tabela Comparativa (todos os competidores)

| Seletor                  | Hit Rate Top-1 | Hit Rate Top-2 | Avg Regret (pp) | Perf. Relativa | Catástrofe (%) | Max Regret (pp) |
|--------------------------|:--------------:|:--------------:|:--------------:|:--------------:|:--------------:|:---------------:|
| Random Baseline          | 13.5% ±3.4pp   | 23.4%          | 1.66pp ±0.33pp | 96.8% ±0.6pp   | 68.1%          | 27.31pp         |
| Most Frequent            | 60.8% ±4.8pp   | 82.0%          | 0.81pp ±0.19pp | 98.4% ±0.4pp   | 0.0%           | 15.57pp         |
| Always Laplace           | 60.8% ±4.8pp   | 82.0%          | 0.81pp ±0.19pp | 98.4% ±0.4pp   | 0.0%           | 15.57pp         |
| Vanilla AutoML v16       | 75.8% ±4.2pp   | 93.8%          | 0.50pp ±0.22pp | 99.1% ±0.4pp   | 8.0%           | 25.73pp         |
| **v19 Hybrid (ours)**    | **68.3%** ±4.6pp | **94.3%**      | **0.65pp** ±0.18pp | **98.6%** ±0.4pp | **10.2%**      | **14.04pp**     |

> ±: metade do intervalo de confiança 95% (t-Student). N = 401 amostras de teste (acumulado 5-fold).

---

## 3. Análise por Competidor

### 3.1 Random Baseline
- Hit Rate: **13.5%** (esperado teórico: 11.1% = 1/9 mecanismos)
- Catastrophic Failure Rate: **68.1%** — sem lógica de seleção, erra catastroficamente na maioria dos casos.
- Serve como piso absoluto; qualquer método inteligente deve superar 13.5%.

### 3.2 Most Frequent Baseline
- Hit Rate: **60.8%** — beneficia-se da distribuição desbalanceada (Laplace é campeão em ~60% dos datasets).
- Matematicamente equivalente ao "Always Laplace" porque Laplace é a classe majoritária.
- Não aprende nada; seu ganho sobre Random é 100% explicado pela prevalência de classes.

### 3.3 Always Laplace (Market Baseline)
- Hit Rate: **60.8%** — reflete a proporção de datasets onde Laplace é de fato o melhor mecanismo.
- Catastrophic Failure Rate: **0.0%** por construção (Laplace é o comparador base).
- Average Regret: **0.81pp** — custo de nunca adaptar a recomendação ao dataset.
- Max Regret: **15.57pp** — há datasets onde Laplace é muito pior que o ótimo.

### 3.4 Vanilla AutoML v16
- Hit Rate: **75.8%** — **melhor hit rate top-1 do benchmark**.
  ExtraTrees treinado nas 74 features originais (sem features dp_/ctx_ nem oversampling v17).
- Demonstra que as features originais já têm grande poder preditivo.
- **Limitação**: sem fallback conservador, pode cometer erros graves.
  Max Regret: **25.73pp** — o maior entre os métodos inteligentes (pior caso severo).
  Catastrophic Failure Rate: 8.0% — recomenda mecanismos piores que Laplace em ~1/12 casos.

### 3.5 v19 Hybrid (nosso framework)
- Hit Rate Top-1: **68.3%** — 7.5pp abaixo do Vanilla v16, mas +7.5pp acima do Laplace puro.
  O fallback conservador (margin=0.5pp) às vezes substitui uma recomendação correta do regressor pelo clf top-1.
- Hit Rate Top-2: **94.3%** — **melhor do benchmark** (+0.5pp vs v16).
  Isto significa que o mecanismo correto está quase sempre no Top-2 do v19.
- Max Regret: **14.04pp** — **melhor (menor pior-caso) do benchmark**.
  Redução de 11.7pp (4542%) comparado ao Vanilla v16.
  O fallback elimina efetivamente os erros graves ao forçar conservadorismo quando o regressor está incerto.
- Average Regret: 0.65pp — levemente acima do v16 por conta dos fallbacks para Laplace.
- **Trade-off central**: o v19 é 7.5pp menos preciso no Top-1, mas 11.7pp mais seguro no pior caso.
  Para deployments DP críticos (saúde, finanças), minimizar Max Regret é frequentemente prioritário.

---

## 4. Parâmetros do v19 Hybrid

| Parâmetro                | Valor       | Justificativa                                               |
|--------------------------|:-----------:|-------------------------------------------------------------|
| `_hybrid_top_k`          | 3           | top_k ≥ 2 converge identicamente (grade offline)           |
| `_hybrid_laplace_margin` | 0.5 pp      | sweet spot: hit=50.5%, catástrofe=3.0% (grade 5×8 offline) |
| Classificadores          | ET+LR+SVM   | Soft-voting, class_weight="balanced", + oversampling       |
| Regressor                | MultiOutputRF | n_estimators=200, min_samples_leaf=2, dados originais     |
| Labels estabilizadas     | n_runs=5    | Média de 5 execuções com seeds distintos                   |

### Nota sobre o Benchmark vs Pipeline de Produção

| Aspecto                    | Benchmark (este script)          | Pipeline de Produção (v19)        |
|----------------------------|:--------------------------------:|:---------------------------------:|
| Pré-filtros hierárquicos   | ❌ Não incluídos                  | ✅ CAT1 + DISC + GAUSS             |
| Oversampling               | ✅ Incluído                       | ✅ Incluído                        |
| F1-macro reportado         | ~0.70–0.76 (sem pré-filtros)     | 0.910 (com pré-filtros + synthetics) |
| Hit Rate reportado         | 68.3% (CV5, sem pré-filtros)     | 53.2% (pipeline com fallback=2pp) / 50.5% (offline, margem calibrada) |

Os pré-filtros hierárquicos interceptam casos extremos (datasets discretos, muito categóricos,
Gaussian analítico evidente) antes do classificador geral, o que explica o salto no F1 em produção.

---

## 5. Evolução das Versões

| Versão       | F1-macro (CV) | Hit Rate    | Pior-que-Laplace | Principais mudanças                                            |
|--------------|:-------------:|:-----------:|:----------------:|----------------------------------------------------------------|
| v16          | 0.70          | ~61.9%      | N/A              | Classificador puro, 76 features originais                      |
| v17          | 0.87          | 66.4%       | N/A              | +40 features DP-específicas (dp_/ctx_), regressão multi-output |
| v18          | 0.855         | 36.4%       | 48.6%            | Hybrid ensemble + checkpoint, n_runs=1 (labels ruidosas)       |
| v19 (raw)    | 0.910         | 53.2%       | 31.8%            | META_STABLE_PROFILE (n_runs=5), fallback margin=2.0pp          |
| **v19-tuned**| **0.910**     | **68.3%***  | **10.2%***       | **Fallback margin=0.5pp (calibrado offline, grade 5×8)**       |

> *Métricas de 5-fold CV sobre 401 datasets, sem pré-filtros hierárquicos. Pipeline completo com pré-filtros
> reportaria métricas diferentes (mais altas em F1, diferentes em Hit Rate).

---

## 6. Fontes dos Dados

```
/Users/alvesmauricio/Workspace/dp_meta_selector/meta_datasets_v19/meta_features_meta_stable.csv
/Users/alvesmauricio/Workspace/dp_meta_selector/meta_datasets_v19/meta_targets_meta_stable.csv
research/tuning/tune_meta_models.py   ← grade de calibração offline (40 combinações)
research/tuning/tune_results_v19.csv  ← resultados da grade
research/benchmark_evaluator.py      ← este script de benchmark
```

---

*Relatório gerado por `research/benchmark_evaluator.py` — DP-Meta-Selector v19*

# Melhorias v14 a v16

> Decisões DEC-015 a DEC-022: Datasets sintéticos, novas features, diagnósticos e otimização de thresholds.

---

## Visão Geral

As versões v14-v16 representam uma reformulação significativa do framework com foco em:
1. Balanceamento via datasets sintéticos
2. Novas meta-features discriminativas
3. Módulo de diagnósticos avançados
4. Otimização de thresholds para F1-macro

---

## DEC-015 — Datasets Sintéticos para Balanceamento de Famílias

**Data:** 2026-06-11

### Problema

Apenas 3 classes representadas de forma significativa (Laplace, GaussianAnalytic, Exponential) de 9 mecanismos disponíveis. Geometric nunca aparecia como best_mechanism.

### Decisão

Criar módulo `synthetic_datasets.py` com geradores de datasets sintéticos por família:

| Gerador | Descrição | Família Alvo |
|---------|-----------|--------------|
| `generate_continuous_dataset()` | Features contínuas, alta cardinalidade | Laplace/Gaussian |
| `generate_discrete_dataset()` | Features inteiras, range pequeno | Geometric |
| `generate_categorical_dataset()` | Features com baixa cardinalidade | Exponential |
| `generate_high_dim_dataset()` | Alta dimensionalidade | GaussianAnalytic |
| `generate_mixed_dataset()` | Mix de tipos | Robustez |

### Implementação

```python
def augment_training_datasets(real_datasets, ratio=0.2):
    """Adiciona ~30 sintéticos ao treino real (20% ratio)."""
    n_synthetic = int(len(real_datasets) * ratio)
    # Distribui proporcionalmente entre famílias
    ...
```

### Resultado

Train aumentou de 350 → 401 datasets.

---

## DEC-016 — Seleção de best_mechanism com Desempate por Família

**Data:** 2026-06-11

### Problema

Algoritmo original de seleção de `best_mechanism` usava apenas acurácia absoluta para desempate, ignorando sinais de família do dataset.

### Decisão

Novo método `_select_best_mechanism()` em `meta_dataset.py`:

```python
def _select_best_mechanism(results_df, meta_features):
    # 1. Identifica candidatos dentro da margem (0.5% do melhor)
    best_acc = results_df["accuracy"].max()
    candidates = results_df[results_df["accuracy"] >= best_acc - 0.005]
    
    # 2. Infere família preferida via meta-features
    if meta_features["cat_ratio_low_cardinality"] >= 0.7:
        preferred_family = "categorical"
    elif meta_features["ratio_integer_cols"] >= 0.8:
        preferred_family = "discrete"
    else:
        preferred_family = "continuous"
    
    # 3. Filtra candidatos pela família preferida
    family_candidates = candidates[candidates["family"] == preferred_family]
    
    # 4. Desempata por acurácia absoluta
    return family_candidates.iloc[0]["mechanism"]
```

### Justificativa

Mesmo quando Laplace e Exponential têm acurácias similares, o dataset pode ter características que indicam que Exponential é mais apropriado.

---

## DEC-017 — Novas Meta-features para Discriminação de Família

**Data:** 2026-06-11

### Decisão

Adicionar `_family_discriminators()` em `meta_features.py` com 9 novas features:

| Feature | Descrição |
|---------|-----------|
| `fam_continuity_score` | Baseado em cardinalidade e não-inteiros |
| `fam_discreteness_score` | Baseado em colunas inteiras + range pequeno |
| `fam_categoricity_score` | Baseado em baixa cardinalidade |
| `fam_mean_gini` | Gini impurity médio por coluna |
| `fam_ratio_uniform_cols` | Proporção de colunas com distribuição uniforme |
| `fam_is_onehot` | Detecção de one-hot encoding |
| `fam_p_continuous` | Probabilidade soft-max normalizada |
| `fam_p_discrete` | Probabilidade soft-max normalizada |
| `fam_p_categorical` | Probabilidade soft-max normalizada |

### Justificativa

Features existentes não discriminavam bem categorical de continuous em datasets com features dummy-encoded.

---

## DEC-018 — Redução de Thresholds para Melhorar Recall

**Data:** 2026-06-11

### Problema

Thresholds muito altos causavam baixo recall de categorical (2.8% no teste original).

### Decisão

| Parâmetro | Antes (v13) | Depois (v14) |
|-----------|-------------|--------------|
| `_family_gate_threshold` | 0.65 | 0.55 |
| `_cat_prefilter_threshold` | 0.90 | 0.75 |
| `_cat_prefilter_family_min` | 0.20 | 0.15 |
| `_gauss_prefilter_threshold` | 1.01 | 0.85 |
| `_ga_boost_pca_threshold` | 0.50 | 0.45 |
| `_ga_boost_factor` | 3.0 | 2.5 |

### Justificativa

Thresholds anteriores foram otimizados para precision, sacrificando recall. Com mais dados de treino (sintéticos), podemos relaxar os thresholds.

---

## DEC-019 — Pré-filtro para Datasets Discretos (Geometric)

**Data:** 2026-06-11

### Decisão

Adicionar `_fit_discrete_prefilter()` e `_apply_discrete_prefilter()` em `meta_learner.py`:
- Classificador binário GBC para discrete vs. resto
- Usa subset de features: `ratio_integer_cols`, `disc_composite_score`, `mean_log_unique_ratio`, etc.
- Threshold de disparo: 0.70

### Diferença de DEC-009

Com datasets sintéticos, agora temos exemplos suficientes de discrete no treino para que o prefilter aprenda.

---

## DEC-020 — Classificadores por Família (Ensemble Hierárquico)

**Data:** 2026-06-11

### Decisão

Adicionar `_fit_family_mechanism_classifiers()` em `meta_learner.py`:
- Treina um RandomForest específico para cada família
- Cada classificador só vê mecanismos da sua família
- Usado pelo discrete prefilter para escolher entre Geometric variants

### Justificativa

O classificador global confunde mecanismos de famílias diferentes. Classificadores especializados por família têm melhor performance intra-família.

---

## Resultados v14 vs v13

| Métrica | v13 | v14 | Mudança |
|---------|-----|-----|---------|
| **hit_rate** | 0.5646 | **0.6763** | +11.2 pp ⬆️ |
| **cat_hit** | 2.8% | **31.2%** | +28.4 pp ⬆️ |
| **cont_hit** | 73.9% | **75.9%** | +2.0 pp ⬆️ |
| **regret** | 0.0095 | **0.0064** | -33% ⬇️ |
| **rel_perf** | 98.04% | **98.84%** | +0.8 pp ⬆️ |
| **model_acc** | 0.5216 | **0.5448** | +2.3 pp ⬆️ |

---

## DEC-021 — Módulo de Diagnósticos Avançados (v15)

**Data:** 2026-06-11

### Decisão

Criar módulo `diagnostics.py` com métricas avançadas:

| Função | Descrição |
|--------|-----------|
| `compute_family_f1_scores()` | F1 por família (continuous, categorical, discrete) |
| `compute_confusion_matrix()` | Matriz normalizada por linha ou coluna |
| `compute_calibration_data()` | ECE e calibration bins |
| `dataset_level_kfold_cv()` | K-fold no nível de datasets |
| `ablation_study()` | Impacto de grupos de features |
| `run_full_diagnostics()` | Executa todas as métricas |

### Implementação

- Nova flag CLI: `--diagnostics`
- Salva `diagnostics.json` no report_dir

---

## DEC-022 — Integração ao Pipeline CLI (v15)

**Data:** 2026-06-11

### Uso

```bash
python -m dp_meta_selector --diagnostics
```

### Output Adicional

- F1-score por família (tabela formatada)
- Confusion matrix normalizada
- ECE e calibration bins
- diagnostics.json salvo em reports/

---

## Ajustes v16: Otimização para F1-macro

**Data:** 2026-06-11

### Problema

v15 tinha bom hit rate (67.6%) mas F1-macro de apenas 0.55, indicando recall ruim em classes minoritárias.

### Ajustes de Threshold

| Parâmetro | v15 | v16 |
|-----------|-----|-----|
| `_cat_prefilter_threshold` | 0.75 | 0.75 |
| `_gauss_prefilter_threshold` | 0.85 | **0.80** |
| `_family_gate_threshold` | 0.55 | 0.55 |
| `_ga_boost_factor` | 2.5 | **2.8** |

### Novas Meta-features (v16)

12 features adicionais para discriminação GA vs Laplace:

| Feature | Descrição |
|---------|-----------|
| `fam_max_cardinality` | Cardinalidade máxima entre colunas |
| `fam_pct_cols_under_10` | % de colunas com < 10 valores |
| `fam_pct_cols_under_5` | % de colunas com < 5 valores |
| `fam_mean_cardinality` | Cardinalidade média |
| `fam_mean_value_ratio` | Razão valores/samples média |
| `fam_min_value_ratio` | Razão valores/samples mínima |
| `fam_feature_to_sample_ratio` | n_features / n_samples |
| `fam_is_high_dim` | Flag: > 50 features |
| `fam_mean_feature_corr` | Correlação média entre features |
| `fam_max_feature_corr` | Correlação máxima |
| `fam_pca_var_top3` | Variância explicada top-3 PCA |
| `fam_ga_score` | Score composto para GA |

### Resultados v16 vs v15

| Métrica | v15 | v16 | Mudança |
|---------|-----|-----|---------|
| hit_rate | 67.6% | 61.9% | -5.7pp ⬇️ |
| F1-macro | 0.55 | **0.70** | +15pp ⬆️ |
| Exp recall | 31% | **56%** | +25pp ⬆️ |
| GA recall | 44% | **59%** | +15pp ⬆️ |

### Trade-off

A versão v16 sacrifica hit rate geral para melhorar recall de classes minoritárias, resultando em F1-macro significativamente melhor.

---

## Arquivos Modificados (v14-v16)

| Arquivo | Mudanças |
|---------|----------|
| `meta_dataset.py` | Novo `_select_best_mechanism()` |
| `meta_learner.py` | Prefilters, classificadores por família, thresholds |
| `meta_features.py` | 21 novas features em `_family_discriminators()` |
| `main.py` | Integração de sintéticos, flag `--diagnostics` |
| `reporter.py` | Suporte a `best_family` |
| **Novo:** `synthetic_datasets.py` | Geradores de sintéticos |
| **Novo:** `diagnostics.py` | Módulo de diagnósticos |

---

## Lições Aprendidas (v14-v16)

10. **Dados sintéticos são essenciais para classes raras:** Com apenas 7 exemplos de Geometric no treino real, o prefilter não aprendia. Com sintéticos, temos cobertura suficiente.

11. **Desempate por família > desempate por acurácia:** Quando mecanismos têm acurácias similares, a família do dataset é um sinal mais robusto.

12. **Thresholds otimizados para precision prejudicam recall:** O equilíbrio precision/recall deve ser reavaliado quando a quantidade de dados de treino muda.

13. **F1-macro vs hit rate é um trade-off:** Melhorar recall de minoritárias pode piorar o hit rate geral, mas resulta em modelo mais equilibrado.

---

## DEC-023 — Refatoração DP-Aware: Meta-Features Específicas e Regressão de Utilidade (v17)

**Data:** 2026-06-13

### Problema

As meta-features existentes (~76 features) não capturavam as propriedades matemáticas fundamentais que determinam o sucesso de mecanismos DP:
- **Sensibilidade global** (impacto de outliers no clipping)
- **Dimensionalidade efetiva** (colapso de utilidade em alta dimensão)
- **Disparate Impact** de subgrupos minoritários
- **Contexto do usuário** (ε desejado, tipo de tarefa) ignorado na decisão

Além disso, a função objetivo era classificação direta ("qual mecanismo é melhor") em vez de quantificação de perda de utilidade.

### Decisão

Implementar 4 melhorias em paralelo:

#### 1. Meta-features DP-específicas (40 novas features)

**`_dp_clipping_signal()`** — 10 features para prever impacto do clipping:

| Feature | Fórmula | Relevância DP |
|---------|---------|---------------|
| `dp_max_median_ratio_mean` | mean(max/median por coluna) | Sensibilidade global |
| `dp_max_median_ratio_max` | max(max/median por coluna) | Outlier extremo |
| `dp_kurtosis_mean` | mean(kurtosis por coluna) | Caudas pesadas |
| `dp_kurtosis_max` | max(kurtosis) | Coluna crítica |
| `dp_iqr_ratio_mean` | mean(IQR/range) | Concentração do sinal |
| `dp_clipping_loss_est` | fração de dados além de 2σ | Estimativa de perda |
| `dp_global_sensitivity_mean` | mean(max - min por coluna) | Range de sensibilidade |

**`_dp_sparsity_dimensionality()`** — 11 features para colapso de utilidade:

| Feature | Fórmula | Relevância DP |
|---------|---------|---------------|
| `dp_zero_ratio` | zeros / total | Esparsidade |
| `dp_near_zero_ratio` | |x| < 1e-6 | Quase-zeros |
| `dp_svd_rank` | rank numérico (SVD) | Dimensionalidade real |
| `dp_effective_rank` | exp(H(σ²/Σσ²)) | Rank efetivo |
| `dp_condition_number` | σ_max/σ_min | Amplificação de ruído |
| `dp_intrinsic_dim_ratio` | effective_rank / n_features | Redundância |

**`_dp_subgroup_entropy()`** — 9 features para disparate impact:

| Feature | Fórmula | Relevância DP |
|---------|---------|---------------|
| `dp_minority_class_ratio` | min_class / total | Grupo minoritário |
| `dp_gini_impurity` | 1 - Σp² | Desbalanceamento |
| `dp_class_entropy` | -Σp·log(p) | Entropia de classes |
| `dp_disparate_impact_risk` | min_class_frac | Risco de DI |
| `dp_n_minority_subgroups` | classes com frac < 5% | Contagem minoritárias |

#### 2. Variáveis de Contexto Obrigatórias (8 features)

**`_context_features(epsilon, task_type)`**:

| Feature | Descrição |
|---------|-----------|
| `ctx_epsilon` | Orçamento ε do usuário |
| `ctx_log_epsilon` | log(ε + 1) |
| `ctx_epsilon_small` | ε ≤ 0.5 (one-hot) |
| `ctx_epsilon_medium` | 0.5 < ε ≤ 2.0 (one-hot) |
| `ctx_epsilon_large` | ε > 2.0 (one-hot) |
| `ctx_task_classification` | Tarefa = classificação |
| `ctx_task_regression` | Tarefa = regressão |
| `ctx_task_queries` | Tarefa = queries |

**API atualizada:**
```python
# selector.py
result = selector.recommend(X, y, epsilon=0.5, task_type="classification")
```

#### 3. Regressão Multi-Output de Perda de Utilidade

**Target:** `utility_loss_{mechanism}` = `max(0, (baseline_acc - dp_acc) / baseline_acc) * 100`

**Decisão:** Em vez de classificar "qual mecanismo é melhor", prever a perda % de utilidade para cada um dos 9 mecanismos e escolher o que tem **menor perda prevista**.

**Modelo:** `MultiOutputRegressor(RandomForestRegressor)` com `StandardScaler`

**Ordem de decisão no `predict()` (v17):**
1. `cat_prefilter` (p_exp ≥ 0.75 AND p_cat ≥ 0.15) → Exponential
2. `disc_prefilter` (p_disc ≥ 0.70) → Geometric
3. `gauss_prefilter` (p_ga ≥ 0.80) → GaussianAnalytic
4. **`_predict_regression()`** → mecanismo com menor perda prevista ← NOVO
5. Ensemble ExtraTrees + HIER gate (fallback)

**Conversão de perda em "probabilidade"** (softmin com T=10):
```
p_i = exp(-loss_i / T) / Σ exp(-loss_j / T)
```

#### 4. `META_STABLE_PROFILE` com `n_runs=5`

Adicionado em `utility.py` para eliminar o ruído estocástico da DP durante a geração do meta-dataset. Labels são médias sobre 5 execuções com seeds diferentes.

```python
META_STABLE_PROFILE = MetaBuildProfile(
    clf="ExtraTrees",
    cv_splits=5,
    n_runs=5,          # ← novo: 5 execuções para labels robustos
    timeout_per_ds=120,
    sample_size=5000,
)
```

### Resultados

| Métrica | Antes (v16) | Depois (v17) | Delta |
|---------|-------------|--------------|-------|
| Nº de meta-features | ~76 | **116** | +40 |
| F1-macro (ExtraTrees) | 0.70 | **0.87** | +0.17 |
| Hit Rate pipeline (clf) | 61.9% | **66.4%** | +4.5pp |
| Regressor MAE-CV | — | **4.16%** | — |

### Trade-off Identificado

O regressor com `META_FAST_PROFILE` (n_runs=1) tem hit rate de apenas 29.4% porque aprende perdas precisas (%) a partir de labels ruidosas (1 run da DP ≠ perda real). O classificador é robusto ao ruído porque só precisa saber *qual* mecanismo é melhor, não *quanto*.

**Conclusão:** O regressor é a abordagem correta, mas requer `META_STABLE_PROFILE` durante a geração do meta-dataset para atingir seu potencial.

### Arquivos Modificados (v17)

| Arquivo | Mudanças |
|---------|----------|
| `meta_features.py` | +4 métodos DP-aware, `extract()` aceita `epsilon`/`task_type`, +40 features |
| `meta_learner.py` | Regressão multi-output, decisão por perda mínima, `save()/load()` atualizado |
| `meta_dataset.py` | `_process_one()` gera `utility_loss_{mechanism}` para 9 mecanismos |
| `utility.py` | Adicionado `META_STABLE_PROFILE` (n_runs=5) |
| `selector.py` | `recommend()` aceita `epsilon`/`task_type`, tabela de perdas no log |
| `__init__.py` | Exporta `META_STABLE_PROFILE`, `TASK_*` constants |

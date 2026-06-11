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

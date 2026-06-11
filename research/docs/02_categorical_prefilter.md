# Pré-filtro Categórico (CAT1)

> Decisões DEC-004 e DEC-012: Pré-filtro para datasets categóricos e dual-gate.

---

## DEC-004 — CAT1: Pré-filtro Binário para Datasets Categóricos

**Data:** Primeira rodada de melhorias

### Problema

Datasets categóricos tinham hit_rate de apenas 45.9%, apesar de terem o maior potencial de ganho (+2.6pp sobre Laplace).

### Decisão

Treinar um classificador binário GradientBoosting (Exponential vs. resto) nos dados *originais* (antes do oversample) e disparar diretamente "Exponential" quando confiança ≥ 0.55.

### Implementação

**Arquivos modificados:**
- `meta_learner.py`: `_fit_categorical_prefilter()` / `_apply_categorical_prefilter()`
- `meta_features.py`: `_categorical_signal()` com 7 novas features

### Novas Meta-features (CAT1)

| Feature | Descrição |
|---------|-----------|
| `cat_mean_cardinality` | Cardinalidade média por coluna |
| `cat_max_cardinality` | Cardinalidade máxima |
| `cat_min_cardinality` | Cardinalidade mínima |
| `cat_ratio_low_cardinality` | Proporção de colunas com < 20 valores únicos |
| `cat_mean_entropy` | Entropia média por coluna |
| `cat_target_dominance` | Dominância da classe majoritária no target |
| `cat_target_uniformity` | Uniformidade das classes target |

### Resultados

| Métrica | Antes (v0) | Depois (v1) | Mudança |
|---------|------------|-------------|---------|
| hit_rate | 0.5306 | 0.5374 | +0.7pp |
| cat_hit | 45.9% | **64.9%** | **+19pp** ✅ |
| cont_hit | 56.7% | 50.0% | -6.7pp ⚠️ |
| F1-macro CV | 0.815 | 0.922 | +10.7pp |

### Trade-off

O ganho em categóricos (+19pp) veio com custo em contínuos (-6.7pp) devido a falsos positivos: datasets contínuos com features dummy-encoded eram classificados como categóricos.

### Threshold

**Valor escolhido:** 0.55

| Threshold | cat_hit | cont_hit | Notas |
|-----------|---------|----------|-------|
| 0.55 | 64.9% | 50.0% | Escolhido |
| 0.65 | 58.2% | 54.1% | cat_hit reduzido sem benefício |

---

## DEC-012 — CAT1 Dual-Gate: Confirmação de Família

**Data:** Sétima rodada (v8)

### Problema

Após várias iterações, CAT1 ainda produzia muitos falsos positivos:
- 65 Exponential recomendados (oracle = 37)
- 33 FPs (51% de erro)

**Diagnóstico detalhado:**
- TODOS os 65 vinham via CAT1
- Média p_exp para FPs: 0.864
- Média p_exp para TPs: 0.920
- CAT1 intercepta e retorna ANTES do portão HIER
- FPs: datasets contínuos com features dummy-encoded

### Decisão

Implementar dual-gate em `_apply_categorical_prefilter()`:
- Exige `p_exp >= T1 (CAT1)` **E** `p_cat >= T2 (family classifier)`
- T1 = 0.65 (threshold original)
- T2 = 0.20 (família mínima)

### Implementação

```python
def _apply_categorical_prefilter(self, X, family_probs):
    p_exp = self._cat_prefilter.predict_proba(X)[0, 1]
    p_cat = family_probs.get("categorical", 0.0)
    
    if p_exp >= self._cat_prefilter_threshold:
        if p_cat >= self._cat_prefilter_family_min:
            return "Exponential", p_exp
    return None, p_exp
```

### Resultados (v8b, T2=0.20)

| Métrica | Antes (v7) | Depois (v8b) | Mudança |
|---------|------------|--------------|---------|
| hit_rate | 0.6122 | **0.6463** | +3.4pp |
| cat_hit | 86.5% | 59.5% | -27pp |
| cont_hit | 52.7% | **66.4%** | +13.7pp |
| model_acc | 0.5140 | 0.5221 | +0.8pp |

### Análise do Trade-off

| T2 | cat_hit | cont_hit | hit_rate | Notas |
|----|---------|----------|----------|-------|
| 0.00 | 86.5% | 52.7% | 0.6122 | Sem gate |
| 0.20 | 59.5% | 66.4% | **0.6463** | **Sweet spot** |
| 0.25 | 51.2% | 68.9% | 0.6380 | Perde mais TPs |

**T2=0.20** bloqueia 22 FPs que se tornam hits (Laplace) enquanto perde apenas 13 TPs.

### Lição Aprendida

> O dual-gate dentro do CAT1 é essencial porque CAT1 intercepta ANTES do HIER. O portão hierárquico não é proteção suficiente para falsos positivos do CAT1.

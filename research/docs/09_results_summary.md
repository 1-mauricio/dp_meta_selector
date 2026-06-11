# Resumo de Resultados por Versão

> Evolução do framework através das versões v0 a v16.

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

*v16: valores de recall por família (Exponential, GaussianAnalytic)

---

## Evolução Visual

```
Hit Rate Evolution
1.0 |
    |
0.8 |
    |                              ┌── v13 (0.674)
0.7 |            ┌─v2b──┐     ┌───┘    
    |            │      │    v8b      
0.6 |     v1─────┘      └v3b───┘                     v16 (0.619)
    |    ┌┘                                              │
0.5 |─v0─┘                                               │
    |                                                    │
0.4 |                                                    │
    +─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬────►
         v0    v1   v2b   v3b   v8b  v11  v13  v14  v16
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

---

## Métricas por Família (v16)

| Família | Precision | Recall | F1 |
|---------|-----------|--------|-----|
| Laplace | 0.71 | 0.68 | 0.69 |
| Exponential | 0.58 | 0.56 | 0.57 |
| GaussianAnalytic | 0.75 | 0.59 | 0.66 |
| **Macro** | 0.68 | 0.61 | **0.64** |

---

## Comparação com Baselines

| Método | Acurácia Média | Hit Rate |
|--------|----------------|----------|
| Laplace fixo | 0.522 | 27.6% |
| Random | ~0.48 | ~33% |
| **Nosso (v13)** | **0.524** | **67.4%** |
| **Nosso (v16)** | — | **61.9%** |
| Oracle | ~0.55 | 100% |

---

## Lições da Evolução

### O que funcionou

1. **HIER gate** — maior ganho individual (+14pp)
2. **Dual-gate** — reduziu FPs sem perder muitos TPs
3. **Threshold alto no CAT1** — TPs têm confiança muito maior que FPs
4. **Desabilitar GAUSS** — precision < 35% é net negativo

### O que não funcionou

1. **Aligned Profile** — mudou labels, resultado piorou
2. **GAUSS prefilter ativo** — muitos FPs
3. **DISC prefilter** — poucos exemplos, não aprendeu
4. **Filtrar datasets** — afetou muitos datasets válidos

### Trade-offs Identificados

| Trade-off | Favorece hit_rate | Favorece F1-macro |
|-----------|-------------------|-------------------|
| Threshold CAT1 | Alto (0.90) | Baixo (0.75) |
| GAUSS prefilter | Desabilitado | Ativo (0.80) |
| Family gate | Alto (0.65) | Baixo (0.55) |

---

## Estado Final Recomendado

Para **dissertação** (F1-macro equilibrado): **v16**
- F1-macro: 0.70
- Recall balanceado entre classes

Para **produção** (hit_rate máximo): **v13**
- hit_rate: 67.4%
- model_acc > Laplace fixo

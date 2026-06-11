# Otimização do GaussianAnalytic

> Decisões DEC-011, DEC-013 e DEC-014: GAUSS prefilter, thresholds e otimização final.

---

## DEC-011 — GAUSS: Pré-filtro Binário para GaussianAnalytic (v4-v7)

**Data:** Sexta rodada

### Decisão

Treinar GBC binário (GaussianAnalytic vs Laplace) nos dados contínuos de treino, usando apenas 12 meta-features selecionadas (`_gauss_feature_idx`) para não poluir o espaço global de features.

### Resultado (v7)

| Métrica | Valor |
|---------|-------|
| GA recomendado | 14x |
| GA oracle | 24x |
| True Positives | 6 |
| False Positives | 8 |
| Precision | 43% |

### Problema Crítico

Adicionar `_gaussian_signal()` às features globais foi catastrófico:
- CAT1 absorveu as novas features
- Disparou 141x Exponential (vs 37 oracle)
- Hit rate colapsou

### Lição Aprendida

> Usar apenas feature subset isolado para o GAUSS prefilter. Adicionar features globais é arriscado.

---

## DEC-013 — Desabilitação do GAUSS Prefilter (v11)

**Data:** Sétima rodada (continuação)

### Análise

Com dual-gate ativo:
- GaussianAnalytic = 19 recomendados
- 5 True Positives
- 14 False Positives
- Precision = 26%

### Cálculo do Impacto Líquido

```
GAUSS prefilter intercepta 19 datasets ANTES do ensemble

14 FPs: oracle=Laplace mas GAUSS diz GA
  → Esses 14 se tornariam hits se Laplace fosse recomendado

5 TPs: corretos
  → Se GAUSS desabilitado, ensemble poderia ainda acertar alguns
```

### Resultado (v11)

| Métrica | Com GAUSS | Sem GAUSS | Mudança |
|---------|-----------|-----------|---------|
| hit_rate | 0.6463 | **0.6667** | **+2.0pp** |
| model_acc | 0.5221 | 0.5229 | +0.08pp |
| regret | 0.0080 | 0.0072 | -10% |
| GA rec | 19 | 1 | -94% |
| GA precision | 26% | 100% | +74pp |

### Decisão Final

GAUSS prefilter removido da produção (`_gauss_prefilter_threshold=1.01`).

### Lição Aprendida

> Um prefilter com precision < 35% é net negativo — os FPs custam mais hits do que os TPs contribuem.

---

## DEC-014 — Elevação do Threshold CAT1: 0.65 → 0.90 (v12/v13)

**Data:** Oitava rodada

### Problema

11 FPs Exponential ainda restantes após dual-gate (T2=0.20).

### Distribuição de Probabilidades

| Grupo | Média p_exp |
|-------|-------------|
| False Positives | 0.810 |
| True Positives | 0.920 |

3 FPs tinham p_exp < 0.85, incluindo `first-order-theorem-proving` (regret=0.137).

### Testes Realizados

| T1 | hit_rate | FP Exp | model_acc | regret |
|----|----------|--------|-----------|--------|
| 0.65 | 0.6667 (98) | 11 | 0.5229 | 0.0072 |
| 0.85 | 0.6667 (98) | 9 | 0.5235 | 0.0066 |
| **0.90** | **0.6735 (99)** | 8 | **0.5236** | **0.0065** |

### Resultado (v13)

| Métrica | Valor |
|---------|-------|
| hit_rate | 0.6735 |
| model_acc | 0.5236 |
| regret | 0.0065 |
| FPs Exponential | 8 |

**Destaque:** model_acc (0.5236) > Laplace fixo (0.5222) ✅

### Decisão Final

`_cat_prefilter_threshold=0.90`

### Insight

> Elevar T1 de 0.65 → 0.90 bloqueia FPs com p_exp moderado enquanto mantém TPs (que têm p_exp muito alto, mediana ~0.97). Net positivo.

---

## Estado dos Thresholds (v13)

| Parâmetro | Valor | Descrição |
|-----------|-------|-----------|
| `_cat_prefilter_threshold` | 0.90 | T1: confiança mínima CAT1 |
| `_cat_prefilter_family_min` | 0.20 | T2: família mínima (dual-gate) |
| `_gauss_prefilter_threshold` | 1.01 | Efetivamente desabilitado |
| `_family_gate_threshold` | 0.65 | HIER hard gate |
| `_oversample_target_ratio` | 0.80 | Crescimento de minoritárias |

---

## Problema Remanescente: GA Sub-recomendado

### Situação (v13)

| Métrica | Valor |
|---------|-------|
| GA oracle | 24 datasets |
| GA recomendado (ensemble) | 1 dataset |
| GA recall | 4.2% |

### Potencial de Ganho

Se GA fosse acertado em todos os 24 casos:
- +23 hits adicionais
- hit_rate: 0.6735 → ~0.83 (+16pp teórico)

### Abordagens Futuras

1. **GBC específico para GA** com features mais discriminativas:
   - Dimensionalidade
   - Estrutura de correlação
   - PCA variance ratio

2. **Subset isolado de features** para não poluir CAT1

3. **Threshold adaptativo** baseado em características do dataset

---

## Curva de Trade-off: Precision vs Recall

```
                  GAUSS Prefilter Analysis
                  
Precision  |
    100% - |                              ● v11 (disabled)
     80% - |                    
     60% - |          ● v7 (43%)
     40% - |                  ● v8 (26%)
     20% - |
      0% - +-----------------------------→
           0        5       10       15       20
                    GA Recommendations
                    
Conclusão: Para precision < 35%, desabilitar prefilter
           é melhor que manter ativo.
```

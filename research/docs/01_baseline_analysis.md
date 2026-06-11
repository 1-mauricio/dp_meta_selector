# Análise do Baseline

> Decisões DEC-001 a DEC-003: Métricas, modelo inicial e diagnóstico.

---

## DEC-001 — Métricas de Avaliação Primárias

**Data:** Sessão inicial  
**Decisão:** Adotar `hit_rate` como métrica principal (acertou o mecanismo oracle?) e `regret` como métrica de qualidade (quanto de acurácia foi perdida em relação ao oracle?).

**Alternativas consideradas:**
- Balanced accuracy
- F1-macro somente

**Justificativa:** 
- `hit_rate` é intuitivo e alinhado com o objetivo do framework
- `regret` captura o impacto real mesmo quando o modelo erra

---

## DEC-002 — Meta-modelo: ExtraTrees como Melhor Modelo

**Data:** Sessão inicial  
**Decisão:** Usar ExtraTrees (RandomForest com 200 árvores, `class_weight="balanced"`) como meta-modelo principal, selecionado via F1-macro em CV.

**Resultado baseline:** F1-macro CV = 0.815

**Modelos avaliados:**

| Modelo | F1-macro CV |
|--------|-------------|
| LogReg | 0.821 |
| SVM-Linear | 0.837 |
| **ExtraTrees** | **Selecionado após calibração** |

**Justificativa:** ExtraTrees oferece melhor calibração de probabilidades e robustez a desbalanceamento.

---

## DEC-003 — Diagnóstico do Baseline (Resultado Ruim)

**Data:** Sessão de diagnóstico

### Observações

O modelo original apresentou resultados abaixo do esperado:

| Métrica | Valor |
|---------|-------|
| hit_rate | 0.5306 |
| model_acc | 0.5202 |
| Laplace fixo | 0.5222 |

**Problema crítico:** O modelo performou *pior* que simplesmente usar Laplace fixo.

### Causas Identificadas

#### 1. Severo Desbalanceamento de Classes

| Mecanismo | Exemplos no Treino |
|-----------|-------------------|
| Laplace | 199 |
| Exponential | 83 |
| GaussianAnalytic | 53 |
| Geometric | 7 |

#### 2. GaussianAnalytic Nunca Recomendado

- Oracle indicava GA em 22 casos de teste
- Modelo recomendou GA: **0 vezes**
- Recall de GA: 0%

#### 3. Geometric Super-recomendado

- Oracle indicava Geometric: 6 vezes
- Modelo recomendou Geometric: **39 vezes**
- Precision de Geometric: ~15%

#### 4. Datasets Categóricos com Maior Potencial

| Tipo | Potencial de Ganho | Hit Rate |
|------|-------------------|----------|
| Categóricos | +2.6pp médio | 45.9% |
| Contínuos | +0.3pp médio | 56.7% |

**Conclusão:** O modelo falhava exatamente onde havia maior potencial de ganho.

---

## Implicações para o Design

O diagnóstico revelou três problemas estruturais:

1. **Desbalanceamento:** Classes minoritárias (Geometric, GA) não aprendidas
2. **Bias para Laplace:** Classe majoritária dominava predições
3. **Categóricos sub-atendidos:** Maior ganho potencial, pior performance

Estes problemas guiaram as decisões subsequentes:
- DEC-004: Pré-filtro categórico (CAT1)
- DEC-008: Portão hierárquico (HIER)
- DEC-010: Remoção do Geometric

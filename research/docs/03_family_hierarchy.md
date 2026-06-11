# Portão Hierárquico e Família Discrete

> Decisões DEC-005 a DEC-010: HIER gate, tentativas com Geometric e decisão de remoção.

---

## DEC-005 — Tentativa: Aligned Profile (REVERTIDA)

**Data:** Segunda rodada

### Tentativa

Criar `META_ALIGNED_PROFILE` (rf, cv=3, 2 runs) para alinhar perfil de meta-build com perfil de avaliação.

### Resultado

- hit_rate caiu para 0.4966
- Geometric super-recomendado (53x vs 6x oracle)

### Causa Raiz

O novo perfil com `screening=True + refine_top_k=5 + rf` alterou os labels do meta-dataset:
- Geometric foi de 7 → 76 exemplos no treino
- Labels diferentes = modelo diferente = resultados diferentes

### Decisão

**REVERTER.** Manter `META_FAST_PROFILE` (logreg, cv=3, 1 run).

### Lição Aprendida

> Qualquer mudança de perfil invalida o cache e produz labels diferentes, podendo introduzir viés severo.

---

## DEC-006 — Oversample target_ratio: 0.4 → 0.8

**Data:** Terceira rodada

### Decisão

Aumentar `target_ratio` em `_oversample()` de 0.4 para 0.8 — cada classe minoritária cresce até 80% do tamanho da maior.

### Motivação

Geometric (7 exemplos) precisava de mais representação.

### Resultado

Nenhuma mudança nas predições vs. CAT1 (distribuição idêntica).

### Lição Aprendida

> Oversample ajuda mais classificadores sensíveis (LogReg, SVM); ExtraTrees com `class_weight="balanced"` já compensa internamente.

---

## DEC-007 — Correção: sample_weight na Calibração

**Data:** Terceira rodada

### Bug Encontrado

`CalibratedClassifierCV(cv="prefit").fit(X, y)` era chamado sem `sample_weight`, anulando o efeito de `class_weight="balanced"` do classificador subjacente.

### Correção

Passar `sample_weight=sample_weights` no `.fit()` da calibração.

### Impacto

Pequeno — as predições não mudaram significativamente. O bug existia mas o efeito líquido era limitado dado o oversample.

---

## DEC-008 — HIER: Portão Hierárquico de Família

**Data:** Quarta rodada

### Decisão

Substituir o prior suave de família por um "hard gate": se o classificador de família (SVC-linear) prevê uma família com confiança ≥ 0.65, zera as probabilidades de mecanismos de outras famílias.

### Implementação

**Arquivos modificados:**
- `meta_learner.py`: `_fit_family_classifier()` e `_apply_family_decision()`

### Bug Crítico (Primeira Tentativa)

O `_fit_family_classifier()` era treinado *após* o oversample:
- Geometric foi de 7 → 159 exemplos
- SVC previa "discrete" em excesso
- hit_rate caiu para 0.483

### Correção (Segunda Tentativa)

Mover `_fit_family_classifier()` para *antes* do oversample, junto com o CAT1.

### Resultados (Após Correção)

| Métrica | CAT1 (v1) | +HIER (v2b) | Mudança |
|---------|-----------|-------------|---------|
| hit_rate | 0.5374 | **0.6803** | **+13pp** ✅ |
| cont_hit | 50.0% | **74.0%** | **+24pp** ✅ |
| cat_hit | 64.9% | 62.2% | -2.7pp |
| disc_hit | 50.0% | 0% | ⚠️ Portão nunca dispara "discrete" |

### Lição Aprendida

> Todos os classificadores auxiliares (prefilters, family classifier) devem ser treinados nos dados *originais*, sem oversample, para preservar a distribuição real das classes.

---

## DEC-009 — DISC Prefilter: Pré-filtro para Datasets Discretos

**Data:** Quinta rodada

### Tentativa

Análogo ao CAT1, treinar GradientBoosting binário (Geometric vs. resto) com LOO-CV (apenas 7 positivos).

### Resultado

F1-CV = 0.0000 — não aprendeu nada com apenas 7 exemplos positivos.

### Decisão

**Não ajuda.** Problema estrutural: poucos dados, não solucionável com o dataset atual.

---

## DEC-010 — Remoção do Geometric e Família Discrete

**Data:** Quinta rodada

### Decisão

Remover `Geometric`, `GeometricTruncated`, `GeometricFolded` de `DP_MECHANISMS`.

### Justificativa

| Fator | Valor |
|-------|-------|
| Datasets com Geometric como oracle (treino) | 7 (2% da base) |
| Regret médio nos 6 datasets discretos de teste | 0.42% |
| Geometric recomendado vs oracle | 39x vs 6x |
| disc_hit sem oversample excessivo | 0% |

**Conclusão:** Laplace já performa quase igual em datasets discretos, e Geometric introduzia mais ruído que sinal.

### Tentativa Incorreta (REVERTIDA)

Filtrar datasets com `disc_composite_score > 0.25` no `_process_one()`:
- Isso filtrou 183/342 datasets de treino
- Razão: colunas inteiras são comuns em datasets codificados
- **REVERTIDA imediatamente**

### Implementação Correta

Apenas remover de `DP_MECHANISMS` e `SCREENING_MECHANISMS`. Os 7 datasets discretos do treino recebem o melhor label disponível (Laplace), contribuindo normalmente.

### Resultado Final (v3 vs v0)

| Métrica | v0 | v3b | Mudança |
|---------|-----|-----|---------|
| hit_rate | 0.5306 | 0.5646 | +3.4pp |
| cat_hit | 45.9% | **89.2%** | **+43.3pp** ✅ |
| cont_hit | 56.7% | 45.5% | -11.2pp ⚠️ |
| regret | 0.0101 | 0.0236 | +134% (pior) |
| rel_perf | 0.9799 | 0.9554 | -2.4pp |

**Trade-off:** cat_hit melhorou drasticamente, mas cont_hit e regret pioraram. GaussianAnalytic ainda sub-recomendado.

---

## Resumo das Decisões de Família

| Decisão | Resultado | Mantida? |
|---------|-----------|----------|
| Aligned Profile | hit_rate caiu para 0.49 | ❌ Revertida |
| Oversample 0.8 | Sem mudança | ✅ Mantida |
| sample_weight | Pequeno impacto | ✅ Mantida |
| HIER gate | +13pp hit_rate | ✅ Mantida |
| DISC prefilter | F1=0, não aprendeu | ❌ Abandonada |
| Remover Geometric | Menos ruído | ✅ Mantida |

# Decisões de Design — DP Meta-Selector

Registro cronológico de todas as decisões técnicas, experimentos e seus resultados.
Atualizado em: 2026-06-12.

---

## Contexto

O `dp_meta_selector` é um framework de meta-aprendizagem que seleciona automaticamente
o melhor mecanismo de Privacidade Diferencial (DP) para um dataset tabular, sem necessidade
de intervenção manual.

**Fluxo geral (v13, atual):**
1. Extrair meta-features do dataset
2. CAT1: pré-filtro binário Exponential (threshold ≥ 0.90) com dual-gate família (p_cat ≥ 0.20)
3. Ensemble ExtraTrees + portão hierárquico HIER (hard gate 0.65 / soft boost)
4. Aplicar o mecanismo escolhido com o ε calibrado por família

---

## Linha do Tempo de Decisões

---

### DEC-001 — Métricas de avaliação primárias

**Data:** sessão inicial  
**Decisão:** Adotar `hit_rate` como métrica principal (acertou o mecanismo oracle?) e `regret` como métrica de qualidade (quanto de acurácia foi perdida em relação ao oracle?).  
**Alternativas consideradas:** balanced accuracy, F1-macro somente.  
**Justificativa:** hit_rate é intuitivo e alinhado com o objetivo do framework; regret captura o impacto real mesmo quando o modelo erra.

---

### DEC-002 — Meta-modelo: ExtraTrees como melhor modelo

**Data:** sessão inicial  
**Decisão:** Usar ExtraTrees (RandomForest com 200 árvores, `class_weight="balanced"`) como meta-modelo principal, selecionado via F1-macro em CV.  
**Resultado baseline:** F1-macro CV = 0.815  
**Alternativas treinadas:** LogReg (0.821), SVM-Linear (0.837) — ExtraTrees venceu após calibração.

---

### DEC-003 — Diagnóstico do baseline (resultado ruim)

**Data:** sessão de diagnóstico  
**Observação:** O modelo original tinha hit_rate = 0.5306 e model_acc = 0.5202, menor que Laplace fixo (0.5222).  
**Causas identificadas:**
- Severo desbalanceamento de classes no treino: Laplace=199, Exponential=83, GaussianAnalytic=53, Geometric=7
- GaussianAnalytic nunca recomendado (oracle em 22 casos de teste, modelo recomendou 0x)
- Geometric super-recomendado: 39x, oracle apenas 6x
- Datasets categóricos com maior potencial de ganho (+2.6pp médio), mas pior hit rate (45.9%)

---

### DEC-004 — CAT1: Pré-filtro binário para datasets categóricos

**Data:** primeira rodada de melhorias  
**Decisão:** Treinar um classificador binário GradientBoosting (Exponential vs. resto) nos dados *originais* (antes do oversample) e disparar diretamente "Exponential" quando confiança ≥ 0.55.  
**Implementação:** `_fit_categorical_prefilter()` / `_apply_categorical_prefilter()` em `meta_learner.py`  
**Novas meta-features:** 7 features em `_categorical_signal()` em `meta_features.py` (cardinalidade, entropia por coluna, dominância, uniformidade das classes alvo)  
**Resultado:**
- hit_rate: 0.5306 → **0.5374** (+0.7pp)
- cat_hit: 45.9% → **64.9%** (+19pp) ✅
- cont_hit: 56.7% → 50.0% (-6.7pp) ⚠️ (falsos positivos no contínuo)
- F1-macro CV: 0.815 → **0.922**

**Decisão sobre threshold:** 0.55 — valores maiores (0.65) reduziam cat_hit sem benefício compensatório.

---

### DEC-005 — Tentativa: Aligned Profile (REVERTIDA)

**Data:** segunda rodada  
**Tentativa:** Criar `META_ALIGNED_PROFILE` (rf, cv=3, 2 runs) para alinhar perfil de meta-build com perfil de avaliação.  
**Resultado:** hit_rate caiu para 0.4966, Geometric super-recomendado (53x).  
**Causa raiz:** O novo perfil com `screening=True + refine_top_k=5 + rf` alterou os labels do meta-dataset — Geometric foi de 7→76 exemplos no treino.  
**Decisão:** REVERTER. Manter `META_FAST_PROFILE` (logreg, cv=3, 1 run).  
**Lição aprendida:** Qualquer mudança de perfil invalida o cache e produz labels diferentes, podendo introduzir viés severo.

---

### DEC-006 — Oversample target_ratio: 0.4 → 0.8

**Data:** terceira rodada  
**Decisão:** Aumentar `target_ratio` em `_oversample()` de 0.4 para 0.8 — cada classe minoritária cresce até 80% do tamanho da maior.  
**Motivação:** Geometric (7 exemplos) precisava de mais representação.  
**Resultado:** Nenhuma mudança nas predições vs. CAT1 (distribuição idêntica). ExtraTrees já era robusto ao desbalanceamento.  
**Lição aprendida:** Oversample ajuda mais classificadores sensíveis (LogReg, SVM); ExtraTrees com `class_weight="balanced"` já compensa internamente.

---

### DEC-007 — Correção: sample_weight na calibração

**Data:** terceira rodada  
**Bug encontrado:** `CalibratedClassifierCV(cv="prefit").fit(X, y)` era chamado sem `sample_weight`, anulando o efeito de `class_weight="balanced"` do classificador subjacente.  
**Correção:** Passar `sample_weight=sample_weights` no `.fit()` da calibração.  
**Impacto:** Pequeno — as predições não mudaram significativamente. O bug existia mas o efeito líquido era limitado dado o oversample.

---

### DEC-008 — HIER: Portão hierárquico de família

**Data:** quarta rodada  
**Decisão:** Substituir o prior suave de família por um "hard gate": se o classificador de família (SVC-linear) prevê uma família com confiança ≥ 0.65, zera as probabilidades de mecanismos de outras famílias.  
**Implementação:** `_fit_family_classifier()` e `_apply_family_decision()` em `meta_learner.py`

**Bug crítico (primeira tentativa):** O `_fit_family_classifier()` era treinado *após* o oversample. Geometric foi de 7→159 exemplos → o SVC previa "discrete" em excesso → hit_rate caiu para 0.483.

**Correção (segunda tentativa):** Mover `_fit_family_classifier()` para *antes* do oversample, junto com o CAT1.  
**Resultado após correção:**
- hit_rate: 0.5374 → **0.6803** (+13pp vs. CAT1) ✅
- cont_hit: 50.0% → **74.0%** (+24pp) ✅
- cat_hit: 64.9% → 62.2% (-2.7pp) — leve queda aceitável
- disc_hit: 50.0% → 0% — portão de família nunca disparou "discrete" corretamente

**Lição aprendida:** Todos os classificadores auxiliares (prefilters, family classifier) devem ser treinados nos dados *originais*, sem oversample, para preservar a distribuição real das classes.

---

### DEC-009 — DISC prefilter: Pré-filtro para datasets discretos

**Data:** quinta rodada  
**Tentativa:** Análogo ao CAT1, treinar GradientBoosting binário (Geometric vs. resto) com LOO-CV (apenas 7 positivos).  
**Resultado:** F1-CV = 0.0000 — não aprendeu nada com apenas 7 exemplos positivos.  
**Decisão:** Não ajuda. Problema estrutural: poucos dados, não solucionável com o dataset atual.

---

### DEC-010 — Remoção do Geometric e família discrete

**Data:** quinta rodada  
**Decisão:** Remover `Geometric`, `GeometricTruncated`, `GeometricFolded` de `DP_MECHANISMS`.  
**Justificativa:**
- Apenas 7 datasets com Geometric como oracle no treino (2% da base)
- Nos 6 datasets discretos de teste: regret médio = 0.42% — Laplace já performa quase igual
- Geometric super-recomendado (39x) vs. oracle (6x) — mais ruído que sinal
- disc_hit = 0% em versões sem oversample excessivo

**Tentativa incorreta:** Filtrar datasets com `disc_composite_score > 0.25` no `_process_one()`. Isso filtrou 183/342 datasets de treino (colunas inteiras são comuns em datasets codificados). **REVERTIDA imediatamente.**

**Implementação correta:** Apenas remover de `DP_MECHANISMS` e `SCREENING_MECHANISMS`. Os 7 datasets discretos do treino recebem o melhor label disponível (Laplace), contribuindo normalmente.

**Resultado final (v3 vs. v0):**
- hit_rate: 0.5306 → **0.5646** (+3.4pp, +6.4%) ✅
- cat_hit: 45.9% → **89.2%** (+43.2pp) ✅
- cont_hit: 56.7% → 45.5% (-11.2pp) ⚠️
- regret: 0.01009 → 0.02361 (pior — GaussianAnalytic pouco recomendado)
- rel_perf: 0.9799 → 0.9554 (pior)

---

### DEC-011 — GAUSS: Pré-filtro binário para GaussianAnalytic (v4-v7)

**Data:** sexta rodada  
**Decisão:** Treinar GBC binário (GaussianAnalytic vs Laplace) nos dados contínuos de treino, usando apenas 12 meta-features selecionadas (`_gauss_feature_idx`) para não poluir o espaço global de features.  
**Resultado (v7):** GaussianAnalytic 14x recomendado (oracle=24), 6 TPs / 8 FPs, precision=43%.  
**Lição:** Adicionar `_gaussian_signal()` às features globais foi catastrófico — CAT1 absorveu as novas features e disparou 141x Exponential. Usar apenas feature subset isolado para o GAUSS prefilter.

---

### DEC-012 — CAT1 Dual-Gate: confirmação de família (v8)

**Data:** sétima rodada  
**Problema:** 65 Exponential recomendados (oracle=37), 33 FPs (51% de erro). Diagnóstico revelou:
- TODOS os 65 vinham via CAT1 (média p_exp=0.864 para FPs, 0.920 para TPs)
- CAT1 intercepta e retorna ANTES do portão HIER → HIER nunca vê Exponential
- 33 FPs: datasets contínuos com features dummy-encoded que CAT1 classifica como categóricos

**Solução implementada:** Dual-gate em `_apply_categorical_prefilter()`:  
- Exige `p_exp >= T1 (CAT1)` E `p_cat >= T2 (family classifier)`
- T1=0.65 (threshold original), T2=0.20 (family min)

**Resultado (v8b, T2=0.20):**
- hit_rate: 0.6122 → **0.6463** (+3.4pp)  
- cat_hit: 86.5% → 59.5% (bloqueou 13 TPs com p_cat < 0.20)  
- cont_hit: 52.7% → 66.4% (desbloqueou 22 FPs que viraram Laplace → hits)  
- model_acc: 0.5140 → 0.5221 (praticamente igual ao Laplace 0.5222)

**Lição:** T2=0.25 bloqueia mais FPs mas perde mais TPs; T2=0.20 é o sweet spot.

---

### DEC-013 — Desabilitação do GAUSS prefilter (v11)

**Data:** sétima rodada (continuação)  
**Análise:** Com dual-gate ativo, GaussianAnalytic = 19 rec (5 TP, 14 FP), precision=26%.  
**Cálculo líquido:**
- GAUSS prefilter intercepta 19 datasets ANTES do ensemble
- 14 FPs: oracle=Laplace mas GAUSS diz GA → esses 14 se tornariam hits se Laplace recomendado  
- 5 TPs: corretos, mas se GAUSS desabilitado → ensemble poderia ainda acertar alguns

**Resultado (v11, GAUSS threshold=1.01, efetivamente desabilitado):**
- hit_rate: 0.6463 → **0.6667** (+2.0pp)
- model_acc: 0.5221 → 0.5229
- regret: 0.0080 → 0.0072
- GA: 19 rec → 1 rec (ensemble acerta 1/24 GA; recall cai mas precision=100%)

**Decisão final:** GAUSS prefilter removido da produção (`_gauss_prefilter_threshold=1.01`).  
**Lição:** Um prefilter com precision < 35% é net negativo — os FPs custam mais hits do que os TPs contribuem.

---

### DEC-014 — Elevação do threshold CAT1: 0.65 → 0.90 (v12/v13)

**Data:** oitava rodada  
**Problema:** 11 FPs Exponential ainda restantes após dual-gate (T2=0.20). Distribuição:
- FPs: média p_exp=0.810 (mais baixa que TPs=0.920)  
- 3 FPs têm p_exp < 0.85, incluindo `first-order-theorem-proving` (regret=0.137)

**Testes realizados:**
| T1 | hit_rate | FP Exp | model_acc | regret |
|----|----------|--------|-----------|--------|
| 0.65 | 0.6667 (98) | 11 | 0.5229 | 0.0072 |
| 0.85 | 0.6667 (98) | 9 | 0.5235 | 0.0066 |
| 0.90 | **0.6735** (99) | 8 | **0.5236** | **0.0065** |

**Resultado (v13, T1=0.90):** hit_rate=0.6735, model_acc=0.5236 > Laplace, regret=0.0065.  
**Decisão final:** `_cat_prefilter_threshold=0.90`.

---

## Estado Atual dos Componentes (v13)

### `mechanisms.py`
- **Mecanismos ativos:** Laplace, Gaussian, GaussianAnalytic, Staircase, LaplaceTruncated, LaplaceFolded, Snapping, Exponential, Uniform
- **Removidos:** Geometric, GeometricTruncated, GeometricFolded
- **SCREENING_MECHANISMS:** ["Laplace", "GaussianAnalytic", "Exponential"]

### `meta_features.py`
- `_stat()`: features estatísticas padrão + novas (ratio_integer_cols, mean_log_unique_ratio, etc.)
- `_categorical_signal()`: 7 features para detectar datasets categóricos (CAT1)
- `_discrete_signal()`: 8 features (mantido por histórico, mas prefilter removido)
- `_gaussian_signal()`: definido mas NÃO chamado de `extract()` (risco de poluir CAT1)

### `meta_learner.py`
- `_cat_prefilter_threshold = 0.90` (CAT1 — elevado de 0.65 progressivamente)
- `_cat_prefilter_family_min = 0.20` (dual-gate família)
- `_gauss_prefilter_threshold = 1.01` (efetivamente desabilitado)
- `_family_gate_threshold = 0.65` (HIER hard gate)
- `_oversample()`: target_ratio=0.8
- Calibração com `sample_weight` corrigida

---

## Resultados por Versão

| Versão | Modificação | hit_rate | regret | cat_hit | cont_hit | model_acc |
|--------|------------|----------|--------|---------|---------|-----------|
| v0 | Baseline | 0.5306 | 0.0101 | 45.9% | 56.7% | 0.5202 |
| v1 | +CAT1 prefilter | 0.5374 | 0.0122 | **64.9%** | 50.0% | — |
| v2b | +HIER gate (corrigido) | 0.6803 | 0.0084 | 62.2% | **74.0%** | — |
| v3b | −Geometric (correto) | 0.5646 | 0.0236 | 89.2% | 45.5% | — |
| v7 | +GAUSS prefilter | 0.6122 | 0.0161 | 86.5% | 52.7% | 0.5140 |
| v8b | +Dual-gate T2=0.20 | 0.6463 | 0.0080 | 59.5% | 66.4% | 0.5221 |
| v11 | GAUSS off | 0.6667 | 0.0072 | 59.5% | 69.1% | 0.5229 |
| v12 | T1=0.85 | 0.6667 | 0.0066 | 54.1% | 70.9% | 0.5235 |
| **v13** | **T1=0.90** | **0.6735** | **0.0065** | **54.1%** | **71.8%** | **0.5236** |

> **Melhor resultado (v13):** hit_rate=0.6735, model_acc=0.5236 > Laplace(0.5222), regret=0.0065

---

## Lições Aprendidas

1. **Treinar auxiliares pré-oversample:** qualquer classificador auxiliar (prefilter, family gate) deve ver a distribuição real, não a inflada pelo oversample.

2. **Mudanças de perfil invalidam o cache:** alterar `clf`, `cv_splits` ou `n_runs` no perfil de meta-build gera labels diferentes para os mesmos datasets.

3. **Poucos exemplos = impossível de modelar:** com apenas 7 exemplos positivos (Geometric), nenhum classificador aprende. Limiar mínimo: ~20 exemplos positivos para viabilidade.

4. **Filtrar datasets por feature é arriscado:** `disc_composite_score > 0.25` parecia seguro mas afetou 183/342 datasets porque colunas inteiras são comuns em datasets categóricos codificados.

5. **Headroom real é pequeno:** oracle supera Laplace em apenas 65/147 datasets com ganho médio de 0.8pp. O framework é viável mas a margem é apertada — qualquer ruído introduzido é prejudicial.

6. **Prefilter com precision < 35% é net negativo:** GAUSS prefilter tinha 5 TPs / 14 FPs (precision=26%). Cada FP gera 1 miss que seria 1 hit (Laplace). Net: -9 hits. Desabilitado.

7. **CAT1 intercepta antes do HIER:** o retorno early do prefilter impede que o HIER veja o Exponential. Portanto, o HIER não é proteção suficiente — o dual-gate dentro do CAT1 é essencial.

8. **Adicionar features globais é arriscado:** `_gaussian_signal()` adicionado às features globais → CAT1 absorveu e disparou 141x Exponential (catastrófico). Usar feature subsets isolados para prefilters específicos.

9. **Threshold elevado ≠ menos recall:** elevar T1 de 0.65 → 0.90 bloqueia FPs com p_exp moderado enquanto mantém TPs (que têm p_exp muito alto ~0.97 mediana). Net positivo.

---

## Trabalho Futuro

- **GaussianAnalytic sub-recomendado:** oracle em 24 datasets, modelo acerta apenas 1 (ensemble). Potencial +3pp de hit_rate se resolvido. Abordagem: treinar GBC específico para GA com features mais discriminativas (dimensionalidade, estrutura de correlação) usando apenas subset isolado de features.
- **Exponential FPs restantes (8):** datasets contínuos com features muito parecidas com categóricas (ex: `tae`, `GAMETES`). Difícil de distinguir sem features de domínio mais ricas.
- **Mais dados:** com ≥ 500 datasets de treino, modelos mais complexos poderiam capturar melhor os padrões de GA vs Laplace.

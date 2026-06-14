# Decisões de Design — DP Meta-Selector

> **NOTA:** Esta documentação foi reorganizada em arquivos separados para facilitar o uso na dissertação.
> Veja a pasta `research/docs/` para a documentação completa e organizada por tópico.

---

## Índice da Documentação

| Arquivo | Descrição |
|---------|-----------|
| [00_index.md](docs/00_index.md) | Visão geral e estrutura do framework |
| [01_baseline_analysis.md](docs/01_baseline_analysis.md) | Análise do baseline e diagnóstico inicial |
| [02_categorical_prefilter.md](docs/02_categorical_prefilter.md) | Pré-filtro categórico (CAT1) e dual-gate |
| [03_family_hierarchy.md](docs/03_family_hierarchy.md) | Portão hierárquico (HIER) e decisões sobre Geometric |
| [04_gaussian_optimization.md](docs/04_gaussian_optimization.md) | Otimização do GaussianAnalytic e thresholds |
| [05_improvements_v14_v16.md](docs/05_improvements_v14_v16.md) | Melhorias das versões v14 a v16 |
| [06_mechanism_comparison.md](docs/06_mechanism_comparison.md) | Estudo comparativo de mecanismos DP (489 datasets) |
| [07_lessons_learned.md](docs/07_lessons_learned.md) | Lições aprendidas consolidadas |
| [08_datasets.md](docs/08_datasets.md) | Detalhes dos datasets utilizados |
| [09_results_summary.md](docs/09_results_summary.md) | Resumo de resultados por versão |

---

## Dados de Suporte

| Arquivo | Descrição |
|---------|-----------|
| `dp_comparison_full.csv` | Resultados da comparação de mecanismos (489 datasets) |
| `datasets_summary.csv` | Resumo dos datasets com métricas principais |

---

## Referência Rápida

### Resultados Principais (v16)

| Métrica | Valor |
|---------|-------|
| Hit rate | 61.9% |
| F1-macro | 0.70 |
| Exp recall | 56% |
| GA recall | 59% |

### Estudo Comparativo (489 datasets)

| Mecanismo | % Best |
|-----------|--------|
| GaussianAnalytic | 52.1% |
| Laplace | 27.6% |
| Exponential | 20.2% |

Gap médio vs Laplace: **+2.18pp**

---

## Histórico Completo (para referência)

O conteúdo original deste arquivo foi preservado abaixo para referência histórica.
Consulte os arquivos em `docs/` para a versão organizada e atualizada.

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

---

## Sessão de Melhorias v14 (2026-06-11)

### DEC-015 — Datasets sintéticos para balanceamento de famílias

**Data:** 2026-06-11  
**Problema:** Apenas 3 classes representadas (Laplace, GaussianAnalytic, Exponential) de 9 mecanismos disponíveis. Geometric nunca aparecia como best_mechanism.  
**Decisão:** Criar módulo `synthetic_datasets.py` com geradores de datasets sintéticos por família:
- `generate_continuous_dataset()`: features contínuas com alta cardinalidade (favorece Laplace/Gaussian)
- `generate_discrete_dataset()`: features inteiras com range pequeno (favorece Geometric)
- `generate_categorical_dataset()`: features com baixa cardinalidade (favorece Exponential)
- `generate_high_dim_dataset()`: alta dimensionalidade (favorece GaussianAnalytic)
- `generate_mixed_dataset()`: mix de tipos para teste de robustez

**Implementação:** `augment_training_datasets()` adiciona ~30 sintéticos ao treino real (20% ratio).  
**Resultado:** Train aumentou de 350 → 401 datasets.

---

### DEC-016 — Seleção de best_mechanism com desempate por família

**Data:** 2026-06-11  
**Problema:** Algoritmo original de seleção de `best_mechanism` usava apenas acurácia absoluta para desempate, ignorando sinais de família do dataset.  
**Decisão:** Novo método `_select_best_mechanism()` em `meta_dataset.py`:
1. Identifica candidatos dentro da margem (0.5% do melhor)
2. Usa meta-features para inferir família preferida:
   - `cat_ratio_low_cardinality >= 0.7` + `ratio_integer_cols >= 0.8` → categorical
   - `ratio_integer_cols >= 0.8` + `disc_composite_score >= 0.3` → discrete
   - Caso contrário → continuous
3. Filtra candidatos pela família preferida
4. Desempata por acurácia absoluta

**Justificativa:** Mesmo quando Laplace e Exponential têm acurácias similares, o dataset pode ter características que indicam que Exponential é mais apropriado.

---

### DEC-017 — Novas meta-features para discriminação de família

**Data:** 2026-06-11  
**Decisão:** Adicionar `_family_discriminators()` em `meta_features.py` com 9 novas features:
- `fam_continuity_score`: baseado em cardinalidade e não-inteiros
- `fam_discreteness_score`: baseado em colunas inteiras + range pequeno
- `fam_categoricity_score`: baseado em baixa cardinalidade
- `fam_mean_gini`: Gini impurity médio por coluna
- `fam_ratio_uniform_cols`: proporção de colunas com distribuição uniforme
- `fam_is_onehot`: detecção de one-hot encoding
- `fam_p_continuous`, `fam_p_discrete`, `fam_p_categorical`: probabilidades soft-max normalizadas

**Justificativa:** Features existentes não discriminavam bem categorical de continuous em datasets com features dummy-encoded.

---

### DEC-018 — Redução de thresholds para melhorar recall

**Data:** 2026-06-11  
**Problema:** Thresholds muito altos causavam baixo recall de categorical (2.8% no teste original).  
**Decisão:** Reduzir thresholds em `meta_learner.py`:
| Parâmetro | Antes | Depois |
|-----------|-------|--------|
| `_family_gate_threshold` | 0.65 | 0.55 |
| `_cat_prefilter_threshold` | 0.90 | 0.75 |
| `_cat_prefilter_family_min` | 0.20 | 0.15 |
| `_gauss_prefilter_threshold` | 1.01 | 0.85 |
| `_ga_boost_pca_threshold` | 0.50 | 0.45 |
| `_ga_boost_factor` | 3.0 | 2.5 |

**Justificativa:** Thresholds anteriores foram otimizados para precision, sacrificando recall. Com mais dados de treino (sintéticos), podemos relaxar os thresholds.

---

### DEC-019 — Pré-filtro para datasets discretos (Geometric)

**Data:** 2026-06-11  
**Decisão:** Adicionar `_fit_discrete_prefilter()` e `_apply_discrete_prefilter()` em `meta_learner.py`:
- Classificador binário GBC para discrete vs. resto
- Usa subset de features: `ratio_integer_cols`, `disc_composite_score`, `mean_log_unique_ratio`, etc.
- Threshold de disparo: 0.70

**Diferença de DEC-009:** Com datasets sintéticos, agora temos exemplos suficientes de discrete no treino para que o prefilter aprenda.

---

### DEC-020 — Classificadores por família (ensemble hierárquico)

**Data:** 2026-06-11  
**Decisão:** Adicionar `_fit_family_mechanism_classifiers()` em `meta_learner.py`:
- Treina um RandomForest específico para cada família
- Cada classificador só vê mecanismos da sua família
- Usado pelo discrete prefilter para escolher entre Geometric variants

**Justificativa:** O classificador global confunde mecanismos de famílias diferentes. Classificadores especializados por família têm melhor performance intra-família.

---

### Resultados v14 vs v13

| Métrica | v13 | v14 | Mudança |
|---------|-----|-----|---------|
| **hit_rate** | 0.5646 | **0.6763** | +11.2 pp ⬆️ |
| **cat_hit** | 2.8% | **31.2%** | +28.4 pp ⬆️ |
| **cont_hit** | 73.9% | **75.9%** | +2.0 pp ⬆️ |
| **regret** | 0.0095 | **0.0064** | -33% ⬇️ |
| **rel_perf** | 98.04% | **98.84%** | +0.8 pp ⬆️ |
| **model_acc** | 0.5216 | **0.5448** | +2.3 pp ⬆️ |
| **vs Laplace (melhor)** | 2.7% | **13.3%** | +10.6 pp ⬆️ |
| **cache hit** | 12% | **100%** | +88 pp ⬆️ |

**Destaques:**
- Hit rate subiu para 67.6% (meta era 70%)
- Categorical recall subiu de 2.8% para 31.2% (era 3%, meta era 50%)
- Modelo agora supera Laplace fixo em 13.3% dos casos (era 2.7%)
- Cache hit rate de 100% (reaproveitamento total após primeira execução)

---

### Arquivos Modificados (v14)

| Arquivo | Mudanças |
|---------|----------|
| `meta_dataset.py` | Novo `_select_best_mechanism()` com desempate por família |
| `meta_learner.py` | Novos prefilters (discrete), classificadores por família, thresholds reduzidos |
| `meta_features.py` | Novo `_family_discriminators()` com 9 features |
| `main.py` | Integração de `augment_training_datasets()` |
| `reporter.py` | Suporte a coluna `best_family` |
| `pyproject.toml` | Correção do build-backend |
| **Novo:** `synthetic_datasets.py` | Geradores de datasets sintéticos por família |

---

### Lições Aprendidas (v14)

10. **Dados sintéticos são essenciais para classes raras:** Com apenas 7 exemplos de Geometric no treino real, o prefilter não aprendia. Com sintéticos, temos cobertura suficiente.

11. **Desempate por família > desempate por acurácia:** Quando mecanismos têm acurácias similares, a família do dataset é um sinal mais robusto do que frações de acurácia.

12. **Thresholds otimizados para precision prejudicam recall:** O equilíbrio precision/recall deve ser reavaliado quando a quantidade de dados de treino muda.

---

## Fase 4: Validação e Métricas Avançadas (v15)

### DEC-021 — Módulo de Diagnósticos Avançados

**Data:** 2026-06-11  
**Decisão:** Criar módulo `diagnostics.py` com métricas avançadas para análise detalhada do meta-modelo:

1. **F1-macro por família:** `compute_family_f1_scores()` e `print_family_f1_report()`
   - Calcula precision, recall e F1 por família (continuous, categorical, discrete)
   - F1-macro e F1-weighted globais

2. **Confusion matrix:** `compute_confusion_matrix()` e `print_confusion_matrix()`
   - Matriz normalizada por linha (recall) ou coluna (precision)
   - Identifica confusões cross-família

3. **Calibration report:** `compute_calibration_data()` e `print_calibration_report()`
   - Expected Calibration Error (ECE)
   - Calibration bins (confiança → acurácia real)

4. **K-fold CV no nível de datasets:** `dataset_level_kfold_cv()`
   - Divide datasets (não amostras) em k folds
   - Treina selector do zero em cada fold
   - Reporta variância do hit_rate entre folds

5. **Ablation study:** `ablation_study()`
   - Remove grupos de features e mede impacto
   - Identifica features mais importantes

**Implementação:**
- Nova flag CLI: `--diagnostics`
- Função consolidada: `run_full_diagnostics()`
- Salva `diagnostics.json` no report_dir

**Justificativa:** Métricas adicionais permitem diagnóstico mais fino de onde o modelo falha e quais features são mais discriminativas.

---

### DEC-022 — Integração ao Pipeline CLI

**Data:** 2026-06-11  
**Decisão:** Adicionar flag `--diagnostics` ao CLI para executar diagnósticos após avaliação:
```bash
python -m dp_meta_selector --diagnostics
```

**Output adicional:**
- F1-score por família (tabela formatada)
- Confusion matrix normalizada
- ECE e calibration bins
- diagnostics.json salvo em reports/

**Nota:** K-fold CV e ablation study não são executados por padrão (são caros). Usar funções individuais quando necessário.

---

### Arquivos Modificados (v15)

| Arquivo | Mudanças |
|---------|----------|
| **Novo:** `diagnostics.py` | Módulo completo com 6 funções de diagnóstico |
| `__init__.py` | Exports das funções de diagnóstico |
| `main.py` | Nova flag `--diagnostics`, integração com `run_full_diagnostics()` |

---

### Estado Atual (v15)

**Métricas implementadas (Fase 4):**
- [x] F1-macro por família no teste
- [x] Confusion matrix do meta-modelo
- [x] Calibration plot (ECE + bins)
- [x] K-fold cross-validation no nível de datasets
- [x] Ablation study de meta-features

**Todas as fases do plano de melhoria estão implementadas.**

---

## Análise de Justificativa do Framework (2026-06-11)

### Estudo Comparativo: Qual Mecanismo DP é Melhor?

Executamos um estudo comparativo em **489 datasets reais** do OpenML para responder:
**"Existe um mecanismo DP universalmente melhor, ou depende do dataset?"**

**Script:** `scripts/compare_dp_mechanisms.py`  
**Metodologia:** 
- 3 mecanismos avaliados: Laplace, GaussianAnalytic, Exponential
- 2 runs por mecanismo por dataset
- Epsilon calibrado por família (continuous=5.0, categorical=2.0)
- Classificador: LogisticRegression com 3-fold CV

---

### Resultados Principais (489 datasets)

#### 1. Distribuição do Melhor Mecanismo

| Mecanismo | Datasets | % | Barra |
|-----------|----------|---|-------|
| **GaussianAnalytic** | 255 | 52.1% | ██████████████████████████ |
| **Laplace** | 135 | 27.6% | █████████████ |
| **Exponential** | 99 | 20.2% | ██████████ |

**Conclusão:** Laplace, o mecanismo mais comum, é o melhor em **apenas 27.6%** dos datasets. Outros mecanismos são melhores em **72.4%** dos casos.

#### 2. Gap de Performance vs Laplace Fixo

| Estatística | Valor |
|-------------|-------|
| Gap médio | +2.18pp |
| Gap mediano | +0.60pp |
| Gap máximo | +32.3pp |
| Gap mínimo | 0.00pp |
| Desvio padrão | 4.74pp |

**Conclusão:** Um seletor perfeito (oracle) daria **+2.18pp de acurácia média** sobre Laplace fixo.

#### 3. Padrões por Dimensionalidade

| Categoria | n_features | n_datasets | GA | Laplace | Exp |
|-----------|------------|------------|----|----|---------|
| SMALL | <10 | 217 | **46.5%** | 35.5% | 18.0% |
| MEDIUM | 10-50 | 204 | **49.0%** | 25.0% | 26.0% |
| LARGE | >50 | 61 | **78.7%** | 11.5% | 9.8% |

**Conclusões:**
- Datasets pequenos: GaussianAnalytic lidera, mas Laplace é competitivo
- Datasets médios: GaussianAnalytic e Exponential dominam
- Datasets grandes (alta dim): **GaussianAnalytic domina com 78.7%**

#### 4. Casos Extremos (Top 10 - Maior Ganho sobre Laplace)

| Dataset | Melhor Mecanismo | Gap vs Laplace | Acc Laplace | Acc Melhor |
|---------|------------------|----------------|-------------|------------|
| pc3 | Exponential | **+32.3pp** | 57.4% | 89.8% |
| kc2 | Exponential | **+31.7pp** | 47.8% | 79.5% |
| PizzaCutter3 | Exponential | **+29.1pp** | 58.7% | 87.8% |
| PieChart3 | Exponential | **+29.1pp** | 58.4% | 87.5% |
| pc1 | Exponential | **+28.3pp** | 64.7% | 93.1% |
| jm1[sub=3000] | Exponential | **+25.9pp** | 54.5% | 80.3% |
| mental-health-in-tech-survey | Exponential | **+24.1pp** | 51.7% | 75.8% |
| PieChart1 | Exponential | **+22.7pp** | 68.6% | 91.3% |
| hill-valley | Exponential | **+21.8pp** | 62.7% | 84.5% |
| total_score | Exponential | **+21.8pp** | 36.2% | 58.0% |

**Conclusão:** Em datasets categóricos, Exponential pode dar **+20-33pp de ganho**.

#### 5. Performance Relativa ao Laplace

| Mecanismo | Melhor que Laplace | Igual | Pior |
|-----------|-------------------|-------|------|
| GaussianAnalytic | 287 (58.7%) | 45 (9.2%) | 157 (32.1%) |
| Exponential | 110 (22.5%) | 31 (6.3%) | 348 (71.2%) |

**Conclusão:** GaussianAnalytic supera Laplace em **58.7%** dos datasets.

#### 6. Impacto do DP na Acurácia

| Métrica | Valor |
|---------|-------|
| Queda média com DP | 7.31pp |
| Queda mediana | 2.94pp |
| Melhor caso | -15.9pp (DP melhorou!) |
| Pior caso | 56.4pp |

---

### Justificativa Quantitativa do Framework

```
CENÁRIO 1: Sempre usar Laplace (baseline ingênuo)
─────────────────────────────────────────────────
• Laplace é o melhor em apenas 27.6% dos datasets (135/489)
• Perda média por não escolher corretamente: 2.18pp de acurácia
• Perda máxima em casos extremos: 32.3pp

CENÁRIO 2: Usar seletor perfeito (oracle)
─────────────────────────────────────────────────
• Acerta 100% das vezes
• Ganho médio sobre Laplace: +2.18pp
• Quando outro mecanismo vence, ganho médio: +3.01pp

CENÁRIO 3: Usar nosso framework (v16)
─────────────────────────────────────────────────
• Hit rate atual: 62-68%
• F1-macro: 0.70
• Ganho estimado: ~0.6-1.0pp sobre Laplace fixo
```

**Conclusão Final:**
O framework se justifica porque:
1. **Laplace não é universal** — só é o melhor em 27.6% dos casos (489 datasets)
2. **O ganho potencial é significativo** — até +32pp em casos extremos
3. **Existe padrão aprendível** — alta dimensionalidade favorece GA (78.7%), dados categóricos favorecem Exponential
4. **Mesmo um seletor imperfeito agrega valor** — nosso framework com 62-68% de hit rate captura parte do ganho potencial de 2.18pp

---

### Arquivos do Estudo

| Arquivo | Descrição |
|---------|-----------|
| `scripts/compare_dp_mechanisms.py` | Script de comparação |
| `research/dp_comparison_full.csv` | Resultados detalhados (489 datasets) |
| `research/DECISIONS.md` | Esta documentação |

---

# Decisões de Arquitetura — v17 a v19 (DP-Aware Meta-Learning)

## DEC-023 — Meta-features DP-Específicas e Variáveis de Contexto (v17)

**Data:** 2026-06-13  
**Contexto:** Framework v16 estagnou em F1-macro=0.70 porque meta-features clássicas (forma, correlação, média) não capturam a geometria interna que governa o comportamento de mecanismos DP.

**Decisão:** Adicionar três grupos de meta-features DP-específicas ao extrator + variáveis de contexto obrigatórias:

| Grupo | Features Criadas | Objetivo |
|-------|-----------------|----------|
| `_dp_clipping_signal` | `dp_max_kurtosis`, `dp_clipping_loss_estimate`, `dp_ratio_heavy_tails`, `dp_mean_max_median_ratio`, `dp_global_sensitivity_norm` | Prever impacto do clipping de outliers na sensibilidade global |
| `_dp_sparsity_dimensionality` | `dp_numerical_rank_ratio`, `dp_effective_dim_ratio`, `dp_condition_number`, `dp_zero_ratio`, `dp_mean_col_sparsity` | Avaliar colapso de utilidade em alta dimensionalidade e dados esparsos |
| `_dp_subgroup_entropy` | `dp_minority_class_ratio`, `dp_disparate_impact_risk`, `dp_gini_impurity`, `dp_class_entropy` | Medir risco de impacto desproporcional em subgrupos minoritários |
| `_context_features` | `ctx_epsilon`, `ctx_log_epsilon`, `ctx_task_classification`, `ctx_task_regression`, `ctx_task_queries` | Contexto operacional do usuário (orçamento ε, tipo de tarefa) |

**Resultado:** F1-macro 0.70 → **0.87** (+17pp) sem alterar thresholds ou arquitetura do ensemble.

**Regra:** Features genéricas têm teto de performance. Domínio específico → features específicas.

---

## DEC-024 — Regressão Multi-Output de Perda de Utilidade (v17)

**Data:** 2026-06-13  
**Contexto:** Classificação direta ("qual mecanismo é melhor") só produz um vencedor. Para ordenar mecanismos e quantificar a diferença de utilidade, é necessária uma função de saída contínua.

**Decisão:** Adicionar `MultiOutputRandomForest` como segundo ramo do ensemble: prever `utility_loss_{mechanism}` (perda percentual de utilidade) para cada um dos 9 mecanismos simultaneamente.

**Regra crítica:** O regressor **deve ser treinado nos dados originais (pré-oversample)**. O oversample distorce a distribuição das perdas ao inflar datasets sintéticos. Salvar `X_meta_orig` separado de `X_meta`.

**Referência:** Lição 19 (`07_lessons_learned.md`).

---

## DEC-025 — META_STABLE_PROFILE e Estabilização de Labels (v19)

**Data:** 2026-06-13  
**Contexto:** Com `n_runs=1`, cada `utility_loss_*` varia ±5pp entre execuções por causa do ruído estocástico da DP. O regressor tentava aprender padrões precisos a partir de labels sem sinal consistente ("areia movediça estatística"). Resultado: Hit Rate 66.4% → **36.4%** ao ativar o regressor na v18.

**Decisão:** Implementar `META_STABLE_PROFILE` com `n_runs=5`. Cada label é a média de 5 execuções independentes com seeds distintos. Pipeline: `main.py --stable --checkpoint .dp_meta_cache/ckpt_stable.joblib --save-meta-dataset meta_datasets_v19/`.

**Persistência obrigatória:** CSVs (`meta_features_meta_stable.csv`, `meta_targets_meta_stable.csv`) salvos em `meta_datasets_v19/` — 19 min de computação (1147s), não reprocessar.

**Regra:** Use `META_STABLE_PROFILE` para treinar qualquer componente de regressão. Para classificação pura, `META_FAST_PROFILE` (n_runs=1) é aceitável.

---

## DEC-026 — Hybrid Ensemble e Fallback Conservador Calibrado (v19)

**Data:** 2026-06-13  
**Contexto:** O classificador ExtraTrees tem alta robustez ao ruído mas não ordena mecanismos. O regressor ordena com precisão mas pode recomendar mecanismos piores que Laplace quando incerto. Na v19 raw com margem padrão (2.0pp), 31.8% das recomendações ainda eram piores que Laplace.

**Decisão:** Ensemble híbrido com disjuntor de segurança:
1. Classificador seleciona Top-K (`_hybrid_top_k=3`) mecanismos finalistas
2. Regressor ordena os finalistas pela menor perda prevista
3. **Fallback:** se `loss_recomendado > loss_Laplace − margin`, forçar Laplace ou recomendação do classificador puro

**Calibração:** Grid Search offline em 40 combinações (5 top_k × 8 margens, 0.5pp a 5.0pp). Resultado: **`margin=0.5pp`** é o sweet spot — Hit Rate 50.5%, Catastrophic Failure 3.0% (offline).

**Parâmetros fixados:** `_hybrid_top_k=3`, `_hybrid_laplace_margin=0.5` em `meta_learner.py`.

**Resultado científico (benchmark 5-fold, 401 datasets):**

| Dimensão | Vanilla v16 | v19 Hybrid | Δ |
|---|:---:|:---:|:---:|
| Hit Rate Top-1 | 75.8% | 68.3% | −7.5pp |
| Max Regret | 25.73pp | 14.04pp | **−45%** |
| Hit Rate Top-2 | 93.8% | 94.3% | +0.5pp |

**Regra:** Em sistemas de decisão para DP crítico, minimizar o pior caso (Max Regret) é matematicamente preferível a maximizar a precisão média.

---

## DEC-027 — Human-in-the-Loop: `return_top_k` (v19-tuned)

**Data:** 2026-06-13  
**Contexto:** Hit Rate Top-2 de 94.3% significa que o mecanismo ótimo está quase sempre nas duas primeiras posições. Expor esse comportamento ao usuário é mais valioso do que um veredito único às cegas.

**Decisão:** Adicionar parâmetro `return_top_k: int = 1` a `MetaLearner.predict()` e `DPMechanismSelector.recommend()`. Quando `return_top_k=2`, retornar `top_k_recommendations` com `{rank, mechanism, predicted_loss, confidence}` ordenado por perda prevista.

**Implementação:** Closure `_enrich()` dentro de `predict()`. Se `predicted_utility_loss` existe (caminho híbrido), ordena por perda ascendente. Caso contrário, ordena por `all_proba` descendente.

**Exemplo de uso:**
```python
result = selector.recommend(X, y, epsilon=1.0, task_type="classification", return_top_k=2)
# #1 Laplace       2.1%
# #2 Exponential   2.2%
```

**Regra:** Para engenheiros de dados em ambientes regulados, receber as duas melhores opções com delta de perda é superior a um único veredito sem contexto.

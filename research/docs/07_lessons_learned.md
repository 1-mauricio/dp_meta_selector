# Lições Aprendidas

> Compilação de todas as lições aprendidas durante o desenvolvimento do DP Meta-Selector.

---

## Arquitetura do Meta-modelo

### 1. Treinar auxiliares pré-oversample

**Contexto:** DEC-008 (HIER gate)

Todos os classificadores auxiliares (prefilters, family classifier) devem ser treinados nos dados *originais*, não nos dados após oversample.

**Por quê:** O oversample altera a distribuição de classes. Um classificador treinado após oversample verá Geometric com 159 exemplos (vs 7 reais) e vai predizer "discrete" em excesso.

**Regra:** Qualquer classificador que tome decisões de roteamento deve ver a distribuição real.

---

### 2. Mudanças de perfil invalidam o cache

**Contexto:** DEC-005 (Aligned Profile)

Alterar `clf`, `cv_splits` ou `n_runs` no perfil de meta-build gera labels diferentes para os mesmos datasets.

**Consequência:** Um modelo treinado com um perfil não é comparável a um modelo treinado com outro perfil. O cache deve ser invalidado.

**Regra:** Documente o perfil exato usado para cada experimento.

---

### 3. Poucos exemplos = impossível de modelar

**Contexto:** DEC-009 (DISC prefilter)

Com apenas 7 exemplos positivos (Geometric), nenhum classificador binário consegue aprender o padrão.

**Limiar prático:** ~20 exemplos positivos mínimo para viabilidade.

**Solução:** Datasets sintéticos (DEC-015) ou remoção da classe (DEC-010).

---

## Engenharia de Features

### 4. Adicionar features globais é arriscado

**Contexto:** DEC-011 (GAUSS prefilter)

`_gaussian_signal()` adicionado às features globais causou colapso:
- CAT1 absorveu as novas features
- Disparou 141x Exponential (vs 37 oracle)

**Regra:** Features específicas de um prefilter devem ficar isoladas nesse prefilter, não no espaço global.

---

### 5. Filtrar datasets por feature é arriscado

**Contexto:** DEC-010 (tentativa revertida)

Filtrar datasets com `disc_composite_score > 0.25` parecia seguro, mas afetou 183/342 datasets porque colunas inteiras são comuns em datasets categóricos codificados.

**Regra:** Sempre verificar o impacto de um filtro em todo o dataset antes de aplicar.

---

### 6. Dados sintéticos são essenciais para classes raras

**Contexto:** DEC-015

Com apenas 7 exemplos de Geometric no treino real, o prefilter não aprendia. Com sintéticos, temos cobertura suficiente.

**Regra:** Para classes com < 20 exemplos, gerar sintéticos ou aceitar que o modelo não vai aprender.

---

## Thresholds e Trade-offs

### 7. Prefilter com precision < 35% é net negativo

**Contexto:** DEC-013 (GAUSS desabilitado)

GAUSS prefilter tinha 5 TPs / 14 FPs (precision=26%). Cada FP gera 1 miss que seria 1 hit se Laplace.

**Cálculo:** Net impact = TPs - FPs = 5 - 14 = -9 hits.

**Regra:** Se precision < 35%, desabilitar o prefilter.

---

### 8. Threshold elevado ≠ menos recall

**Contexto:** DEC-014

Elevar T1 de 0.65 → 0.90 bloqueou FPs com p_exp moderado enquanto manteve TPs (que têm p_exp muito alto, mediana ~0.97).

**Insight:** Se TPs têm confiança muito maior que FPs, elevar o threshold melhora net.

---

### 9. CAT1 intercepta antes do HIER

**Contexto:** DEC-012 (dual-gate)

O retorno early do prefilter impede que o HIER veja o Exponential. Portanto, o HIER não é proteção suficiente para FPs do CAT1.

**Solução:** Dual-gate dentro do CAT1 (verificar família antes de retornar).

---

### 10. Desempate por família > desempate por acurácia

**Contexto:** DEC-016

Quando mecanismos têm acurácias similares (diferença < 0.5%), a família do dataset é um sinal mais robusto.

**Regra:** Usar características do dataset para desempate, não frações decimais de acurácia.

---

### 11. Thresholds otimizados para precision prejudicam recall

**Contexto:** DEC-018

Thresholds altos (v13: T1=0.90) maximizam precision mas causam recall de 2.8% em categóricos.

**Trade-off:** Com mais dados de treino, podemos relaxar thresholds para melhor recall.

---

### 12. F1-macro vs hit rate é um trade-off

**Contexto:** v16

Melhorar recall de classes minoritárias (Exponential, GA) piora o hit rate geral porque Laplace é a maioria.

| Métrica | v15 | v16 |
|---------|-----|-----|
| hit_rate | 67.6% | 61.9% |
| F1-macro | 0.55 | 0.70 |

**Decisão:** Depende do objetivo. Para dissertação, F1-macro balanceado é mais defensável.

---

## Meta-aprendizagem

### 13. Headroom real é pequeno

**Contexto:** DEC-003 (diagnóstico)

Oracle supera Laplace em apenas 65/147 datasets com ganho médio de 0.8pp. O framework é viável mas a margem é apertada.

**Implicação:** Qualquer ruído introduzido é prejudicial. Cada decisão de design importa.

---

### 14. Laplace não é universal

**Contexto:** Estudo comparativo (489 datasets)

Laplace é o melhor em apenas 27.6% dos datasets. GaussianAnalytic e Exponential são melhores em 72.4% dos casos.

**Implicação:** O problema de seleção de mecanismo é real e significativo.

---

### 15. Padrões são aprendíveis

**Contexto:** Estudo comparativo

- Alta dimensionalidade → GaussianAnalytic (78.7%)
- Dados categóricos → Exponential (+20-33pp de ganho)
- Baixa dimensionalidade → Laplace competitivo

**Implicação:** Meta-features capturam informação suficiente para predição.

---

## Resumo Visual

```
DECISÕES DE ARQUITETURA
├── Treinar auxiliares pré-oversample
├── Manter features isoladas por prefilter
└── Dual-gate para early returns

THRESHOLDS
├── Precision < 35% → desabilitar
├── TPs têm alta confiança → elevar threshold
└── Mais dados → relaxar thresholds

TRADE-OFFS
├── Precision vs Recall
├── Hit rate vs F1-macro
└── Simplicidade vs Performance

META-LEARNING
├── Headroom é pequeno (~2pp)
├── Laplace não é universal (27.6%)
└── Padrões são aprendíveis (dimensionalidade, categoricidade)
```
